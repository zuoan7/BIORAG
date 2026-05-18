from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from ..domain.confidence import ConfidenceScorer
from ..domain.config import Settings
from ..domain.router import QueryRouter
from ..domain.schemas import ConversationTurn, QueryAnalysis, QueryFilters, RAGResponse
from ..infrastructure.embedding.bge import BGEM3Embedder
from ..infrastructure.index.parent_store import ParentStore
from ..infrastructure.vectorstores.bm25 import BM25Retriever
from ..infrastructure.vectorstores.hybrid import HybridRetriever
from ..infrastructure.vectorstores.milvus import MilvusRetriever
from .evidence_selection_stage import select_generation_v2_evidence
from .generation_v2 import GenerationV2Service
from .generation_v2.neighbor_audit import NeighborAuditEngine
from .generation_v2_response import build_generation_v2_response
from .neighbor_expansion import ChunkNeighborExpander
from .original_cn_fallback import (
    contains_cjk as _contains_cjk,
    run_original_cn_fallback as _run_original_cn_fallback,
)
from .parent_expansion import ParentContextExpander
from .query_rewrite_adapter import (
    _build_query_rewrite_llm_client,
    build_query_rewrite_service,
)
from .rerank_service import LocalBGERerankerService
from .summary_supplement import (
    build_empty_supplement_debug as _build_empty_supplement_debug,
    supplement_summary_sections as _supplement_summary_sections,
)
from .table_preview import TablePreviewCandidateProvider
from .table_preview_pipeline import (
    run_table_preview as _run_table_preview,
)


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
        self.reranker = LocalBGERerankerService(
            model_path=settings.reranker.model_path,
            batch_size=settings.reranker.batch_size,
            use_fp16=settings.reranker.use_fp16,
            retrieval_config=settings.retrieval,
        )
        neighbor_index_source = ChunkNeighborExpander(
            kb_config=settings.kb,
            retrieval_config=settings.retrieval,
        )
        # Build neighbor audit engine from the corpus index used by generation v2.
        neighbor_index_source._ensure_loaded()
        _audit_engine: NeighborAuditEngine | None = None
        if neighbor_index_source._by_id:
            _audit_engine = NeighborAuditEngine(
                chunk_index=dict(neighbor_index_source._by_id),
                position_index=dict(neighbor_index_source._positions),
                doc_chunks=dict(neighbor_index_source._doc_chunks),
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
        # Phase 19: query rewrite service (default off)
        self._rewrite_svc = build_query_rewrite_service(settings)

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
        if analysis.intent.value == "summary" and seed_chunks:
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

        parent_expander = getattr(self, "parent_expander", ParentContextExpander(parent_store=None, config=self.settings.retrieval))
        evidence_selection = select_generation_v2_evidence(
            question=question,
            seed_chunks=seed_chunks,
            reranked=reranked,
            analysis=analysis,
            settings=self.settings,
            parent_expander=parent_expander,
        )
        final_chunks = evidence_selection.final_chunks
        parent_expansion_debug = evidence_selection.parent_expansion_debug
        evidence_lifecycle_debug = evidence_selection.evidence_lifecycle_debug

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
        return build_generation_v2_response(
            gen_result=gen_result,
            analysis=analysis,
            session_id=session_id,
            filters=filters,
            settings=self.settings,
            retrieved=retrieved,
            reranked=reranked,
            seed_chunks=seed_chunks,
            final_chunks=final_chunks,
            seed_confidence=seed_confidence,
            confidence=confidence,
            start_time=start,
            retrieval_debug=retrieval_debug,
            retriever_debug=getattr(self.retriever, "last_debug", {}),
            reranker_debug=getattr(self.reranker, "last_debug", {}),
            cn_fallback_debug=cn_fallback_debug,
            table_preview_debug=table_preview_debug,
            parent_expansion_debug=parent_expansion_debug,
            summary_supplement_debug=summary_supplement_debug,
            evidence_lifecycle_debug=evidence_lifecycle_debug,
            rewrite_trace=rewrite_trace,
            table_preview_answer_without_formal_citation=(
                table_preview_answer_without_formal_citation
            ),
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
