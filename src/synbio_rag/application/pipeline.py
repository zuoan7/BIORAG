from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..domain.confidence import ConfidenceScorer
from ..domain.config import Settings
from ..domain.router import QueryRouter
from ..domain.schemas import (
    ConversationTurn,
    QueryAnalysis,
    QueryFilters,
    RAGPipelineResponse,
    RetrievedChunk,
)
from ..infrastructure.embedding.bge import BGEM3Embedder
from ..infrastructure.index.parent_store import ParentStore
from ..infrastructure.vectorstores.bm25 import BM25Retriever
from ..infrastructure.vectorstores.hybrid import HybridRetriever
from ..infrastructure.vectorstores.milvus import MilvusRetriever
from .alias_expansion_policy import AliasExpansionPolicy
from .generation_v2 import GenerationV2Service
from .generation_v2.neighbor_audit import NeighborAuditEngine
from .neighbor_index import ChunkNeighborExpander
from .original_cn_fallback import (
    contains_cjk as _contains_cjk,
)
from .original_cn_fallback import (
    run_original_cn_fallback as _run_original_cn_fallback,
)
from .parent_expansion import ParentContextExpander
from .pipeline_stages import (
    ContextStage,
    GenerationStage,
    RerankStage,
    ResponseStage,
    RetrievalStage,
)
from .query_rewrite_adapter import (
    _build_query_rewrite_llm_client,
    build_query_rewrite_service,
)
from .rerank_service import LocalBGERerankerService
from .retrieval_postprocessor import RetrievalPostProcessor
from .retrieval_query_planner import RetrievalQueryPlanner
from .summary_supplement import supplement_summary_sections as _supplement_summary_sections
from .table_preview import (
    TablePreviewCandidateProvider,
)
from .table_preview import (
    run_table_preview as _run_table_preview,
)

__all__ = [
    "SynBioRAGPipeline",
    "_build_filter_plan",
    "_build_query_rewrite_llm_client",
    "_contains_cjk",
    "_run_original_cn_fallback",
    "_run_table_preview",
    "_supplement_summary_sections",
]


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
            query_planner=RetrievalQueryPlanner(),
            alias_policy=AliasExpansionPolicy(),
            postprocessor=RetrievalPostProcessor(),
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
                child_chunk_jsonl = (
                    settings.kb.child_chunk_jsonl
                    if Path(settings.kb.child_chunk_jsonl).exists()
                    else settings.kb.chunk_jsonl
                )
                parent_chunk_jsonl = (
                    settings.kb.parent_chunk_jsonl
                    if Path(settings.kb.parent_chunk_jsonl).exists()
                    else settings.kb.chunk_jsonl
                )
                parent_store = ParentStore.from_jsonl(
                    parent_index_path,
                    chunk_jsonl_path=child_chunk_jsonl,
                    parent_chunk_jsonl_path=parent_chunk_jsonl,
                )
            except Exception:
                parent_store = None
        self.parent_store = parent_store
        self.parent_expander = ParentContextExpander(
            parent_store=parent_store,
            config=settings.retrieval,
        )
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
    ) -> RAGPipelineResponse:
        start = time.perf_counter()
        retrieval = RetrievalStage(
            settings=self.settings,
            router=self.router,
            rewrite_service=self._rewrite_svc,
            search_with_filter_fallback=self._search_with_filter_fallback,
            table_preview_provider=getattr(self, "table_preview_provider", None),
            fallback_pipeline=self,
        ).run(
            question=question,
            filters=filters,
        )

        rerank = RerankStage(settings=self.settings, reranker=self.reranker).run(
            question=retrieval.retrieval_question,
            retrieved=retrieval.retrieved,
            analysis=retrieval.analysis,
            original_question=question,
            rewritten_question=str(getattr(retrieval.rewrite_trace, "rewritten_query", "") or ""),
        )

        context = ContextStage(
            settings=self.settings,
            retriever=self.retriever,
            parent_expander=getattr(self, "parent_expander", None),
        ).run(
            question=question,
            analysis=retrieval.analysis,
            seed_chunks=rerank.seed_chunks,
            reranked=rerank.reranked,
        )

        generation = GenerationStage(
            settings=self.settings,
            confidence_scorer=self.confidence_scorer,
            generator_v2=self.generator_v2,
        ).run(
            question=question,
            analysis=retrieval.analysis,
            seed_chunks=context.seed_chunks,
            final_chunks=context.final_chunks,
            table_preview_debug=retrieval.table_preview_debug,
            history=history,
        )

        return ResponseStage(
            settings=self.settings,
            retriever=self.retriever,
            reranker=self.reranker,
        ).run(
            retrieval=retrieval,
            rerank=rerank,
            context=context,
            generation=generation,
            session_id=session_id,
            filters=filters,
            start_time=start,
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
            raw_child_debug = self._build_raw_child_debug(retrieved)
            raw_retrieved_count = len(retrieved)
            retrieved = self._materialize_parent_child_hits(retrieved)
            aggregation_debug = raw_child_debug["child_to_parent_aggregation"]
            aggregation_debug["materialized_parent_count"] = len(retrieved)
            attempts.append(
                {
                    "name": name,
                    "filters": candidate_filters.__dict__ if candidate_filters else None,
                    "retrieved_count": len(retrieved),
                    "raw_retrieved_count": raw_retrieved_count,
                    "raw_child_count": aggregation_debug["raw_child_count"],
                    "raw_parent_count": aggregation_debug["raw_parent_count"],
                }
            )
            if retrieved:
                return retrieved, {"selected": name, "attempts": attempts, **raw_child_debug}
        return [], {"selected": "empty", "attempts": attempts}

    def _materialize_parent_child_hits(self, retrieved: list) -> list:
        parent_store = getattr(self, "parent_store", None)
        if parent_store is None or not retrieved:
            return retrieved
        return parent_store.materialize_parent_hits(retrieved)

    def _build_raw_child_debug(self, retrieved: list[RetrievedChunk]) -> dict[str, Any]:
        parent_store = getattr(self, "parent_store", None)
        trace: list[dict[str, Any]] = []
        child_ids_by_parent_id: dict[str, list[str]] = {}
        for rank, chunk in enumerate(retrieved, start=1):
            chunk_id = str(chunk.chunk_id or "")
            parent_chunk_id = _debug_parent_chunk_id(chunk, parent_store)
            child_ids_by_parent_id.setdefault(parent_chunk_id, []).append(chunk_id)
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            trace.append(
                {
                    "rank": rank,
                    "child_chunk_id": chunk_id,
                    "parent_chunk_id": parent_chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_file": chunk.source_file,
                    "section": chunk.section,
                    "source": _debug_retrieval_source(chunk),
                    "vector_score": _debug_round_float(chunk.vector_score),
                    "bm25_score": _debug_round_float(chunk.bm25_score),
                    "fusion_score": _debug_round_float(chunk.fusion_score),
                    "chunk_index": metadata.get("chunk_index"),
                    "child_index": metadata.get("child_index"),
                }
            )
        parent_ids = list(child_ids_by_parent_id.keys())
        return {
            "raw_child_trace": trace,
            "child_to_parent_aggregation": {
                "raw_child_count": len(retrieved),
                "raw_parent_count": len(parent_ids),
                "parent_ids_from_children": parent_ids,
                "child_ids_by_parent_id": child_ids_by_parent_id,
            },
        }


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


def _debug_parent_chunk_id(chunk: RetrievedChunk, parent_store: Any | None) -> str:
    chunk_id = str(chunk.chunk_id or "")
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    parent_chunk_id = str(metadata.get("parent_chunk_id") or "")
    if parent_chunk_id:
        return parent_chunk_id
    if parent_store is not None and hasattr(parent_store, "get_parents_for_chunk"):
        for parent in parent_store.get_parents_for_chunk(chunk_id):
            if (
                getattr(parent, "parent_type", "") == "retrieval_parent"
                and getattr(parent, "parent_chunk_id", "")
            ):
                return str(parent.parent_chunk_id)
    return chunk_id.split("::child", 1)[0]


def _debug_retrieval_source(chunk: RetrievedChunk) -> str:
    sources: list[str] = []
    if chunk.vector_score:
        sources.append("vector")
    if chunk.bm25_score:
        sources.append("bm25")
    if chunk.fusion_score:
        sources.append("rrf")
    return "+".join(sources) if sources else "unknown"


def _debug_round_float(value: object) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0
