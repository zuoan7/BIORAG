from __future__ import annotations

import time
from dataclasses import replace

from collections import Counter

from pymilvus import MilvusClient

from ..domain.confidence import ConfidenceScorer
from ..domain.config import Settings
from ..domain.router import QueryRouter
from ..domain.schemas import ConversationTurn, QueryAnalysis, QueryFilters, RAGResponse, RetrievedChunk
from ..infrastructure.embedding.bge import BGEM3Embedder
from ..infrastructure.external_tools.literature_search import ExternalToolManager
from ..infrastructure.vectorstores.bm25 import BM25Retriever
from ..infrastructure.vectorstores.hybrid import HybridRetriever
from ..infrastructure.vectorstores.milvus import MilvusRetriever
from .context_builder import ContextBuilder
from .generation_v2 import GenerationV2Service
from .generation_v2.neighbor_audit import NeighborAuditEngine
from .generation_service import QwenChatGenerator
from .neighbor_expansion import ChunkNeighborExpander
from .rerank_service import QwenReranker


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
        self.confidence_scorer = ConfidenceScorer(settings.confidence)
        self.external_tools = ExternalToolManager(settings.tools)

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        history: list[ConversationTurn] | None = None,
        filters: QueryFilters | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()
        analysis = self.router.analyze(question)
        retrieved, retrieval_debug = self._search_with_filter_fallback(
            question=question,
            analysis=analysis,
            filters=filters,
        )
        reranked = self.reranker.rerank(
            question,
            retrieved,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
        )
        seed_chunks = reranked[: self.settings.retrieval.final_top_k]

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
            final_chunks = seed_chunks
            confidence = self.confidence_scorer.score(seed_chunks)
            gen_result = self.generator_v2.run(
                question=question,
                analysis=analysis,
                seed_chunks=seed_chunks,
                config=self.settings.generation,
                history=history if self.settings.generation.v2_use_history else None,
            )
            # Merge supplement debug into generation debug (always, for diagnostics)
            gv2_debug = gen_result.debug
            gv2_debug["summary_section_supplement"] = summary_supplement_debug
            return RAGResponse(
                answer=gen_result.answer,
                confidence=confidence,
                route=analysis.intent,
                citations=gen_result.citations,
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
                    "tenant_id": filters.tenant_id if filters else "default",
                    "hybrid_enabled": self.settings.retrieval.hybrid_enabled,
                    "bm25_enabled": self.settings.retrieval.bm25_enabled,
                    "retrieval_hits": getattr(self.retriever, "last_debug", {}),
                    "rerank_hits": getattr(self.reranker, "last_debug", {}),
                    "neighbor_expansion": {
                        "enabled": False,
                        "reason": "generation_v2_seed_only",
                        "input_count": len(seed_chunks),
                        "output_count": len(seed_chunks),
                    },
                    "filter_strategy": retrieval_debug,
                    "generation_v2": gen_result.debug,
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
                "filter_strategy": retrieval_debug,
                "evidence_quality": evidence_quality.__dict__,
            },
        )

    def _search_with_filter_fallback(
        self,
        question: str,
        analysis: QueryAnalysis,
        filters: QueryFilters | None,
    ) -> tuple[list, dict[str, object]]:
        attempts: list[dict[str, object]] = []
        filter_plan = _build_filter_plan(filters)
        for name, candidate_filters in filter_plan:
            retrieved = self.retriever.search(
                question,
                limit=analysis.search_limit,
                filters=candidate_filters,
                analysis=analysis,
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
