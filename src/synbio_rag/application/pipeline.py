from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from collections import Counter

from pymilvus import MilvusClient

from ..domain.confidence import ConfidenceScorer
from ..domain.config import Settings
from ..domain.router import QueryRouter
from ..domain.schemas import ConversationTurn, QueryAnalysis, QueryFilters, QueryIntent, RAGResponse, RetrievedChunk
from ..infrastructure.embedding.bge import BGEM3Embedder
from ..infrastructure.clients.openai_compatible import OpenAICompatibleClient
from ..infrastructure.external_tools.literature_search import ExternalToolManager
from ..infrastructure.index.parent_store import ParentStore
from ..infrastructure.vectorstores.bm25 import BM25Retriever
from ..infrastructure.vectorstores.hybrid import HybridRetriever
from ..infrastructure.vectorstores.milvus import MilvusRetriever
from .context_builder import ContextBuilder
from .generation_v2 import GenerationV2Service
from .generation_v2.evidence_lifecycle_debug import rerank_output_debug, stage_debug_from_chunks
from .generation_v2.neighbor_audit import NeighborAuditEngine
from .generation_service import QwenChatGenerator
from ..rewrite.query_rewrite_service import QueryRewriteService, QueryRewriteMode
from .neighbor_expansion import ChunkNeighborExpander
from .parent_expansion import ParentContextExpander
from .rerank_service import QwenReranker
from .table_preview import TablePreviewCandidateProvider, apply_table_preview


class SynBioRAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = BGEM3Embedder(
            model_path=settings.kb.embedding_model_path,
            dim=settings.kb.embedding_dim,
            max_length=settings.kb.embedding_max_length,
        )
        self.router = QueryRouter(settings.retrieval)
        self.dense_retriever = MilvusRetriever(settings.retrieval, self.embedder)
        self.bm25_retriever = BM25Retriever(
            retrieval_config=settings.retrieval,
            kb_config=settings.kb,
            milvus_client=self.dense_retriever.client,
        )
        self.retriever = HybridRetriever(
            config=settings.retrieval,
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker = QwenReranker(
            api_base=settings.reranker.api_base,
            api_key=settings.reranker.api_key,
            model_name=settings.reranker.model_name,
            model_path=settings.reranker.model_path,
            service_url=settings.reranker.service_url,
            batch_size=settings.reranker.batch_size,
            use_fp16=settings.reranker.use_fp16,
            retrieval_config=settings.retrieval,
        )
        self.context_builder = ContextBuilder()
        self.neighbor_expander = ChunkNeighborExpander(
            kb_config=settings.kb,
            retrieval_config=settings.retrieval,
        )
        self.generator = QwenChatGenerator(
            api_base=settings.llm.api_base,
            api_key=settings.llm.api_key,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
            round8_config=settings.round8,
        )
        # Build neighbor audit engine from the same corpus index as neighbor_expander.
        # _ensure_loaded is lazy; we call it once here so the index is ready.
        self.neighbor_expander._ensure_loaded()
        _audit_engine: NeighborAuditEngine | None = None
        if self.neighbor_expander._by_id:
            _audit_engine = NeighborAuditEngine(
                chunk_index=dict(self.neighbor_expander._by_id),
                position_index=dict(self.neighbor_expander._positions),
                doc_chunks=dict(self.neighbor_expander._doc_chunks),
            )
        self.generator_v2 = GenerationV2Service(settings.llm, neighbor_audit_engine=_audit_engine)
        parent_store: ParentStore | None = None
        parent_index_path = Path(settings.retrieval.parent_index_path)
        if parent_index_path.exists():
            try:
                parent_store = ParentStore.from_jsonl(
                    parent_index_path,
                    chunk_jsonl_path=settings.kb.chunk_jsonl,
                )
            except Exception:
                parent_store = None
        self.parent_store = parent_store
        self.parent_expander = ParentContextExpander(parent_store=parent_store, config=settings.retrieval)
        self.table_preview_provider = (
            TablePreviewCandidateProvider(settings.retrieval.table_preview_units_path)
            if settings.retrieval.table_preview_enabled
            else None
        )
        self.confidence_scorer = ConfidenceScorer(settings.confidence)
        self.external_tools = ExternalToolManager(settings.tools)
        # Phase 19: query rewrite service (default off)
        qrc = settings.query_rewrite
        rewrite_llm_client, rewrite_llm_error = _build_query_rewrite_llm_client(settings)
        self._rewrite_svc = QueryRewriteService(
            mode=QueryRewriteMode(qrc.mode),
            model=qrc.model, temperature=qrc.temperature,
            cache_enabled=qrc.cache_enabled, timeout_ms=qrc.timeout_ms,
            fallback_on_error=qrc.fallback_on_error,
            guard_implicit=qrc.guard_implicit_reference,
            guard_negative=qrc.guard_negative_intent,
            cache_version=qrc.cache_key_version,
            llm_client=rewrite_llm_client,
            llm_client_error=rewrite_llm_error,
            eval_cache_path=qrc.eval_rewrite_cache_path,
            eval_require_cache=qrc.eval_rewrite_require_cache,
            eval_fail_fast_on_missing=qrc.eval_rewrite_fail_fast_on_missing,
        )

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        history: list[ConversationTurn] | None = None,
        filters: QueryFilters | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()
        analysis = self.router.analyze(question)
        # Phase 19: query rewrite — prepare retrieval query
        retrieval_question, rewrite_trace = self._rewrite_svc.rewrite(question, is_negative=False)
        retrieved, retrieval_debug = self._search_with_filter_fallback(
            question=retrieval_question,
            analysis=analysis,
            filters=filters,
            original_question=question,
        )
        # Phase 20L: original CN fallback floor
        cn_fallback_debug = _run_original_cn_fallback(
            question=question,
            retrieval_question=retrieval_question,
            rewrite_trace=rewrite_trace,
            retrieved=retrieved,
            analysis=analysis,
            filters=filters,
            config=self.settings.retrieval,
            pipeline=self,
        )
        if cn_fallback_debug.get("triggered"):
            retrieved = cn_fallback_debug["merged_candidates"]
        retrieved, table_preview_debug = _run_table_preview(
            question=question,
            retrieved=retrieved,
            config=self.settings.retrieval,
            generation_version=self.settings.generation.version,
            provider=getattr(self, "table_preview_provider", None),
        )
        reranked = self.reranker.rerank(
            question,
            retrieved,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
        )
        seed_chunks = reranked[: self.settings.retrieval.final_top_k]

        # Phase 15A/15C: annotate rerank_rank on each seed chunk for protection downstream
        for rank_idx, chunk in enumerate(seed_chunks, start=1):
            if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                chunk.metadata["rerank_rank"] = rank_idx
            else:
                chunk.metadata = {"rerank_rank": rank_idx}

        # Phase 7C: summary section supplement — boost Abstract/Conclusion from top docs
        summary_supplement_debug = _build_empty_supplement_debug()
        if (self.settings.generation.version == "v2"
                and analysis.intent.value == "summary"
                and seed_chunks):
            # Get Milvus client — works for both MilvusRetriever and HybridRetriever
            milvus_retriever = getattr(self.retriever, "dense_retriever", self.retriever)
            milvus_client = getattr(milvus_retriever, "client", None)
            seed_chunks, summary_supplement_debug = _supplement_summary_sections(
                question=question,
                seed_chunks=seed_chunks,
                milvus_client=milvus_client,
                collection_name=self.settings.retrieval.collection_name,
                max_docs=3,
                max_per_doc=2,
                max_total=5,
            )

        if self.settings.generation.version == "v2":
            parent_expander = getattr(self, "parent_expander", ParentContextExpander(parent_store=None, config=self.settings.retrieval))
            final_chunks, parent_expansion_debug = parent_expander.expand(
                question=question,
                seed_chunks=seed_chunks,
                analysis=analysis,
            )
            # Phase 15A: protect top-N rerank seeds in final_chunks
            protect_enabled = self.settings.retrieval.protect_rerank_seeds_enabled
            protect_k = min(self.settings.retrieval.protect_rerank_seeds_top_k, len(seed_chunks))
            protected_ids = set()
            if protect_enabled and protect_k > 0:
                protected = seed_chunks[:protect_k]
                protected_ids = {s.chunk_id for s in protected}
                final_ids = {c.chunk_id for c in final_chunks}
                for s in protected:
                    if s.chunk_id not in final_ids:
                        final_chunks.append(s)
            parent_expansion_debug["protected_seed_count"] = len(protected_ids)
            parent_expansion_debug["protected_seed_chunk_ids"] = list(protected_ids)[:10]
            parent_expansion_debug["final_contains_rerank_top3"] = all(
                seed_chunks[i].chunk_id in {c.chunk_id for c in final_chunks}
                for i in range(min(3, len(seed_chunks)))
            )
            evidence_lifecycle_debug = {
                "rerank_output": rerank_output_debug(
                    reranked,
                    protected_ids=protected_ids,
                ),
                "seed_chunks": {
                    "input_count": len(reranked),
                    "output_count": len(seed_chunks),
                    "chunk_ids": [chunk.chunk_id for chunk in seed_chunks],
                    "doc_ids": [chunk.doc_id for chunk in seed_chunks],
                    "drop_reasons": {
                        chunk.chunk_id: "topk_cutoff"
                        for chunk in reranked
                        if chunk.chunk_id not in {seed.chunk_id for seed in seed_chunks}
                    },
                },
                "final_chunks": stage_debug_from_chunks(
                    input_chunks=seed_chunks,
                    output_chunks=final_chunks,
                    protected_ids=protected_ids,
                    default_drop_reason="context_budget",
                ),
            }

            seed_confidence = self.confidence_scorer.score(seed_chunks)
            confidence = self.confidence_scorer.score(final_chunks)
            generation_config = self.settings.generation
            table_preview_answer_without_formal_citation = bool(
                table_preview_debug.get("merged_count", 0)
            )
            if table_preview_answer_without_formal_citation:
                generation_config = replace(generation_config, v2_require_citation=False)
            gen_result = self.generator_v2.run(
                question=question,
                analysis=analysis,
                seed_chunks=final_chunks,
                config=generation_config,
                history=history if self.settings.generation.v2_use_history else None,
            )
            # Merge supplement debug into generation debug (always, for diagnostics)
            gv2_debug = gen_result.debug
            gv2_debug["table_preview_answer_without_formal_citation"] = (
                table_preview_answer_without_formal_citation
            )
            gv2_debug["summary_section_supplement"] = summary_supplement_debug
            gv2_lifecycle_debug = dict(gv2_debug.get("evidence_lifecycle_debug", {}))
            evidence_lifecycle_debug.update(gv2_lifecycle_debug)
            gv2_debug["evidence_lifecycle_debug"] = evidence_lifecycle_debug
            # Phase 21A-9I: negative/no-answer guard — suppress citations in v2
            v2_citations = gen_result.citations
            if analysis.intent == QueryIntent.NEGATIVE:
                v2_citations = []
            return RAGResponse(
                answer=gen_result.answer,
                confidence=confidence,
                route=analysis.intent,
                citations=v2_citations,
                used_external_tool=False,
                tool_name=None,
                tool_result=None,
                session_id=session_id,
                external_references=[],
                debug={
                    "analysis_notes": analysis.notes,
                    "retrieved_count": len(retrieved),
                    "reranked_count": len(reranked),
                    "seed_context_count": len(seed_chunks),
                    "final_context_count": len(final_chunks),
                    "context_chars": 0,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "seed_confidence": seed_confidence,
                    "final_confidence": confidence,
                    "tenant_id": filters.tenant_id if filters else "default",
                    "hybrid_enabled": self.settings.retrieval.hybrid_enabled,
                    "bm25_enabled": self.settings.retrieval.bm25_enabled,
                    "retrieval_hits": getattr(self.retriever, "last_debug", {}),
                    "rerank_hits": getattr(self.reranker, "last_debug", {}),
                    "neighbor_expansion": {
                        "enabled": False,
                        "reason": "generation_v2_seed_only_or_replaced_by_parent_expansion",
                        "input_count": len(seed_chunks),
                        "output_count": len(seed_chunks),
                    },
                    "original_cn_fallback": _sanitize_original_cn_fallback_debug(cn_fallback_debug),
                    "table_preview": _sanitize_table_preview_debug(table_preview_debug),
                    "parent_expansion": parent_expansion_debug,
                    "filter_strategy": retrieval_debug,
                    "generation_v2": gen_result.debug,
                    "evidence_lifecycle_debug": evidence_lifecycle_debug,
                    "query_rewrite": rewrite_trace.to_dict(),
                },
            )
        final_chunks = self.neighbor_expander.expand(seed_chunks)
        context = self.context_builder.build(question, final_chunks, history=history, intent=analysis.intent)
        evidence_quality = self.generator.assess_evidence(question, final_chunks, analysis=analysis)
        answer = self.generator.generate(
            question,
            context,
            final_chunks,
            analysis=analysis,
            history=history,
            assessment=evidence_quality,
        )
        confidence = self.confidence_scorer.score(final_chunks)
        tool_execution = self.external_tools.run_if_needed(
            question=question,
            analysis=analysis,
            low_confidence=self.confidence_scorer.needs_external_tool(confidence),
        )
        citations = self.generator.build_citations(final_chunks, evidence_quality)
        # Phase 21A-9I: negative/no-answer guard — suppress citations
        if analysis.intent == QueryIntent.NEGATIVE:
            citations = []
        answer = self.generator.validate_generated_answer(answer, citations, evidence_quality)

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            route=analysis.intent,
            citations=citations,
            used_external_tool=tool_execution.invoked,
            tool_name=tool_execution.tool_name,
            tool_result=tool_execution.result,
            session_id=session_id,
            external_references=tool_execution.references,
            debug={
                "query_rewrite": rewrite_trace.to_dict(),
                "analysis_notes": analysis.notes,
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "seed_context_count": len(seed_chunks),
                "final_context_count": len(final_chunks),
                "context_chars": len(context),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "tenant_id": filters.tenant_id if filters else "default",
                "hybrid_enabled": self.settings.retrieval.hybrid_enabled,
                "bm25_enabled": self.settings.retrieval.bm25_enabled,
                "retrieval_hits": getattr(self.retriever, "last_debug", {}),
                "rerank_hits": getattr(self.reranker, "last_debug", {}),
                "neighbor_expansion": getattr(self.neighbor_expander, "last_debug", {}),
                "original_cn_fallback": _sanitize_original_cn_fallback_debug(cn_fallback_debug),
                "table_preview": _sanitize_table_preview_debug(table_preview_debug),
                "filter_strategy": retrieval_debug,
                "evidence_quality": evidence_quality.__dict__,
            },
        )

    def _search_with_filter_fallback(
        self,
        question: str,
        analysis: QueryAnalysis,
        filters: QueryFilters | None,
        original_question: str | None = None,
    ) -> tuple[list, dict[str, object]]:
        attempts: list[dict[str, object]] = []
        filter_plan = _build_filter_plan(filters)
        for name, candidate_filters in filter_plan:
            retrieved = self.retriever.search(
                question,
                limit=analysis.search_limit,
                filters=candidate_filters,
                analysis=analysis,
                original_question=original_question,
            )
            attempts.append(
                {
                    "name": name,
                    "filters": candidate_filters.__dict__ if candidate_filters else None,
                    "retrieved_count": len(retrieved),
                }
            )
            if retrieved:
                return retrieved, {"selected": name, "attempts": attempts}
        return [], {"selected": "empty", "attempts": attempts}


def _build_filter_plan(filters: QueryFilters | None) -> list[tuple[str, QueryFilters | None]]:
    if not filters:
        return [("original", None)]
    plan: list[tuple[str, QueryFilters | None]] = []
    if filters.sections and not filters.doc_ids and not filters.source_files:
        plan.append(("drop_sections", replace(filters, sections=[])))
        plan.append(("original", filters))
    else:
        plan.append(("original", filters))
    if filters.sections:
        plan.append(("drop_sections", replace(filters, sections=[])))
    if filters.sections and filters.source_files:
        plan.append(("doc_ids_only", replace(filters, sections=[], source_files=[])))
    elif filters.source_files:
        plan.append(("drop_source_files", replace(filters, source_files=[])))
    deduped: list[tuple[str, QueryFilters | None]] = []
    seen: set[tuple] = set()
    for name, candidate in plan:
        key = (
            tuple(candidate.doc_ids) if candidate else (),
            tuple(candidate.sections) if candidate else (),
            tuple(candidate.source_files) if candidate else (),
            candidate.min_score if candidate else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, candidate))
    return deduped


def _build_query_rewrite_llm_client(settings: Settings):
    qrc = settings.query_rewrite
    mode = QueryRewriteMode(qrc.mode)
    if mode == QueryRewriteMode.OFF:
        return None, ""

    api_base = settings.llm.api_base
    api_key = settings.llm.api_key
    if not api_base or not api_key:
        message = "query_rewrite_llm_client_unavailable: missing QWEN_CHAT_API_BASE or QWEN_CHAT_API_KEY"
        if mode == QueryRewriteMode.ENABLED and qrc.require_llm_for_eval:
            raise RuntimeError(message)
        return None, message

    timeout_seconds = qrc.timeout_ms / 1000.0 if qrc.timeout_ms else settings.llm.timeout_seconds
    try:
        client = OpenAICompatibleClient(
            api_base=api_base,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        message = f"query_rewrite_llm_client_init_failed: {type(exc).__name__}: {exc}"
        if mode == QueryRewriteMode.ENABLED and qrc.require_llm_for_eval:
            raise RuntimeError(message) from exc
        return None, message
    return client, ""


def _sanitize_original_cn_fallback_debug(debug: dict) -> dict:
    return {
        key: value
        for key, value in debug.items()
        if key != "merged_candidates"
    }


def _run_table_preview(
    *,
    question: str,
    retrieved: list[RetrievedChunk],
    config,
    generation_version: str,
    provider: TablePreviewCandidateProvider | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    if generation_version != "v2":
        return list(retrieved), {
            "enabled": False,
            "reason": "generation_v2_required",
            "input_count": len(retrieved),
            "output_count": len(retrieved),
            "candidate_count": 0,
            "merged_count": 0,
            "table_branch_executed": False,
            "table_candidates_in_rerank_input": False,
            "formal_citation_allowed": False,
        }
    return apply_table_preview(
        question=question,
        retrieved=retrieved,
        config=config,
        provider=provider,
    )


def _sanitize_table_preview_debug(debug: dict) -> dict:
    return {
        key: value
        for key, value in debug.items()
        if key != "merged_candidates"
    }


# ── Phase 7C: Summary section supplement ─────────────────────────

_SUMMARY_SECTIONS = {"abstract", "conclusion", "conclusions"}
_SUMMARY_LIKE_TITLE_PATTERNS = {"summary", "conclusion", "outlook", "perspective", "overview"}


def _build_empty_supplement_debug() -> dict:
    return {
        "enabled": False,
        "used": False,
        "reason": "",
        "doc_ids": [],
        "chunk_ids": [],
        "sections": [],
        "count": 0,
        "source": "",
        "abstract_or_conclusion_available_count": 0,
        "abstract_or_conclusion_added_count": 0,
    }


def _supplement_summary_sections(
    *,
    question: str,
    seed_chunks: list[RetrievedChunk],
    milvus_client,
    collection_name: str,
    max_docs: int = 3,
    max_per_doc: int = 2,
    max_total: int = 5,
) -> tuple[list[RetrievedChunk], dict]:
    """Supplement seed_chunks with Abstract/Conclusion chunks from top documents.

    Only affects summary route. Identifies top docs in seed_chunks,
    queries Milvus for Abstract/Conclusion chunks from those docs,
    and appends them to the seed_chunk list.
    """
    if milvus_client is None:
        return seed_chunks, _build_empty_supplement_debug()

    # Identify top documents by chunk count
    doc_counts: Counter[str] = Counter()
    for chunk in seed_chunks:
        if chunk.doc_id:
            doc_counts[chunk.doc_id] += 1
    top_docs = [doc for doc, _ in doc_counts.most_common(max_docs)]

    # Check which top docs already have Abstract/Conclusion in seed_chunks
    existing_abs_conc = set()
    for chunk in seed_chunks:
        section_lower = (chunk.section or "").lower()
        if section_lower in _SUMMARY_SECTIONS and chunk.doc_id in top_docs:
            existing_abs_conc.add(chunk.doc_id)

    # Docs that need supplement
    missing_docs = [d for d in top_docs if d not in existing_abs_conc]
    if not missing_docs:
        return seed_chunks, _build_empty_supplement_debug()

    supplemental_chunks: list[RetrievedChunk] = []
    added_doc_ids: list[str] = []
    added_chunk_ids: list[str] = []
    added_sections: list[str] = []
    abstract_conc_available = 0

    for doc_id in missing_docs[:max_docs]:
        if len(supplemental_chunks) >= max_total:
            break
        doc_supplement_count = 0
        for section in ("Abstract", "Conclusion", "Conclusions"):
            if doc_supplement_count >= max_per_doc or len(supplemental_chunks) >= max_total:
                break
            filter_expr = f'doc_id == "{doc_id}" and section == "{section}"'
            try:
                results = milvus_client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=[
                        "chunk_id", "doc_id", "source_file", "title",
                        "section", "page_start", "page_end", "chunk_index", "text",
                    ],
                    limit=2,
                )
            except Exception:
                continue

            for hit in (results or []):
                text = hit.get("text") or ""
                if len(text) < 20:
                    continue
                # Skip bibliography-like chunks
                if _is_bibliography_like(text):
                    continue
                abstract_conc_available += 1

                # Check if already in seed_chunks
                chunk_id = hit.get("chunk_id", "")
                if any(c.chunk_id == chunk_id for c in seed_chunks):
                    continue

                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=hit.get("doc_id", ""),
                    source_file=hit.get("source_file", ""),
                    title=hit.get("title", ""),
                    section=hit.get("section", ""),
                    text=text,
                    page_start=hit.get("page_start"),
                    page_end=hit.get("page_end"),
                    vector_score=0.0,
                    bm25_score=0.0,
                    rerank_score=None,
                    fusion_score=None,
                    metadata={"chunk_index": hit.get("chunk_index")},
                )
                supplemental_chunks.append(chunk)
                added_doc_ids.append(doc_id)
                added_chunk_ids.append(chunk_id)
                added_sections.append(hit.get("section", ""))
                doc_supplement_count += 1

    if not supplemental_chunks:
        debug = {
            "enabled": True,
            "used": False,
            "reason": f"no_abstract_conclusion_found_for_missing_docs:{','.join(missing_docs[:3])}",
            "doc_ids": missing_docs[:3],
            "chunk_ids": [],
            "sections": [],
            "count": 0,
            "source": "retrieved_doc",
            "abstract_or_conclusion_available_count": abstract_conc_available,
            "abstract_or_conclusion_added_count": 0,
        }
        return seed_chunks, debug

    all_chunks = list(seed_chunks) + supplemental_chunks
    debug = {
        "enabled": True,
        "used": True,
        "reason": f"supplemented_abstract_conclusion_from_{len(missing_docs)}_docs",
        "doc_ids": added_doc_ids,
        "chunk_ids": added_chunk_ids,
        "sections": added_sections,
        "count": len(supplemental_chunks),
        "source": "retrieved_doc",
        "abstract_or_conclusion_available_count": abstract_conc_available,
        "abstract_or_conclusion_added_count": len(supplemental_chunks),
    }
    return all_chunks, debug


def _is_bibliography_like(text: str) -> bool:
    """Detect bibliography/reference-list chunks (avoid supplementing with these)."""
    lowered = text.lower()
    doi_count = lowered.count("https://doi.org")
    if doi_count >= 2:
        return True
    et_al_count = lowered.count("et al.")
    if et_al_count >= 3:
        return True
    return False


# ── Phase 20L: Original CN Fallback Floor ──────────────────────────────

def _contains_cjk(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def _run_original_cn_fallback(
    *,
    question: str,
    retrieval_question: str,
    rewrite_trace,
    retrieved: list,
    analysis,
    filters,
    config,
    pipeline,
) -> dict:
    """Run a small retrieval pass with the original CN query as fallback.

    Only triggers when:
    - original_cn_fallback_enabled is True
    - query rewrite is enabled (retrieval query differs from original)
    - original query contains CJK characters
    - rewritten query differs from original query
    """
    debug = {
        "triggered": False,
        "reason": "",
        "fallback_added_count": 0,
        "fallback_added_chunk_ids": [],
        "fallback_added_doc_ids": [],
        "merged_candidates": list(retrieved),
    }

    if not config.original_cn_fallback_enabled:
        debug["reason"] = "disabled"
        return debug

    if config.original_cn_fallback_require_rewrite_enabled:
        rewrite_mode = getattr(
            rewrite_trace,
            'query_rewrite_mode',
            getattr(rewrite_trace, 'mode', None),
        )
        is_enabled = str(rewrite_mode).lower() in ("enabled", "shadow")
        if not is_enabled:
            debug["reason"] = "rewrite_not_enabled"
            return debug

    if config.original_cn_fallback_require_cjk and not _contains_cjk(question):
        debug["reason"] = "no_cjk_in_original_query"
        return debug

    if config.original_cn_fallback_min_query_diff:
        if question.strip() == retrieval_question.strip():
            debug["reason"] = "query_unchanged_by_rewrite"
            return debug

    try:
        cn_retrieved, _ = pipeline._search_with_filter_fallback(
            question=question,
            analysis=analysis,
            filters=filters,
        )
    except Exception:
        debug["reason"] = "fallback_search_error"
        return debug

    if not cn_retrieved:
        debug["reason"] = "fallback_no_results"
        return debug

    existing_ids = {chunk.chunk_id for chunk in retrieved}
    added = []
    for chunk in cn_retrieved:
        if chunk.chunk_id in existing_ids:
            if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                chunk.metadata["additional_query_branch"] = "original_cn_fallback"
            continue
        if len(added) >= config.original_cn_fallback_max_total:
            break
        if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
            chunk.metadata["query_branch"] = "original_cn_fallback"
            chunk.metadata["fallback_reason"] = "rewrite_enabled_cjk_query"
        added.append(chunk)

    merged = list(retrieved) + added
    debug["triggered"] = True
    debug["reason"] = "fallback_applied"
    debug["fallback_added_count"] = len(added)
    debug["fallback_added_chunk_ids"] = [c.chunk_id for c in added]
    debug["fallback_added_doc_ids"] = list(dict.fromkeys(c.doc_id for c in added))
    debug["merged_candidates"] = merged

    return debug
