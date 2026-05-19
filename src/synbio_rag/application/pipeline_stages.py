from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from ..domain.confidence import ConfidenceScorer
from ..domain.config import Settings
from ..domain.schemas import (
    ConversationTurn,
    QueryAnalysis,
    QueryFilters,
    RAGPipelineResponse,
    RetrievedChunk,
)
from .evidence_selection_stage import select_generation_v2_evidence
from .generation_v2_response import build_generation_v2_response
from .original_cn_fallback import run_original_cn_fallback as _run_original_cn_fallback
from .parent_expansion import ParentContextExpander
from .summary_supplement import (
    build_empty_supplement_debug as _build_empty_supplement_debug,
    supplement_summary_sections as _supplement_summary_sections,
)
from .table_preview import run_table_preview as _run_table_preview


@dataclass
class RetrievalStageResult:
    analysis: QueryAnalysis
    retrieval_question: str
    rewrite_trace: Any
    retrieved: list[RetrievedChunk]
    retrieval_debug: dict[str, object]
    cn_fallback_debug: dict
    table_preview_debug: dict


@dataclass
class RerankStageResult:
    reranked: list[RetrievedChunk]
    seed_chunks: list[RetrievedChunk]


@dataclass
class ContextStageResult:
    seed_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    summary_supplement_debug: dict
    parent_expansion_debug: dict
    evidence_lifecycle_debug: dict


@dataclass
class GenerationStageResult:
    gen_result: Any
    seed_confidence: float
    confidence: float
    table_preview_answer_without_formal_citation: bool


class RetrievalStage:
    def __init__(
        self,
        *,
        settings: Settings,
        router: Any,
        rewrite_service: Any,
        search_with_filter_fallback: Callable[..., tuple[list[RetrievedChunk], dict[str, object]]],
        table_preview_provider: Any,
        fallback_pipeline: Any,
    ) -> None:
        self.settings = settings
        self.router = router
        self.rewrite_service = rewrite_service
        self.search_with_filter_fallback = search_with_filter_fallback
        self.table_preview_provider = table_preview_provider
        self.fallback_pipeline = fallback_pipeline

    def run(self, *, question: str, filters: QueryFilters | None) -> RetrievalStageResult:
        analysis = self.router.analyze(question)
        retrieval_question, rewrite_trace = self.rewrite_service.rewrite(
            question,
            is_negative=False,
        )
        retrieved, retrieval_debug = self.search_with_filter_fallback(
            question=retrieval_question,
            analysis=analysis,
            filters=filters,
            original_question=question,
        )
        cn_fallback_debug = _run_original_cn_fallback(
            question=question,
            retrieval_question=retrieval_question,
            rewrite_trace=rewrite_trace,
            retrieved=retrieved,
            analysis=analysis,
            filters=filters,
            config=self.settings.retrieval,
            pipeline=self.fallback_pipeline,
        )
        if cn_fallback_debug.get("triggered"):
            retrieved = cn_fallback_debug["merged_candidates"]
        retrieved, table_preview_debug = _run_table_preview(
            question=question,
            retrieved=retrieved,
            config=self.settings.retrieval,
            provider=self.table_preview_provider,
        )
        return RetrievalStageResult(
            analysis=analysis,
            retrieval_question=retrieval_question,
            rewrite_trace=rewrite_trace,
            retrieved=retrieved,
            retrieval_debug=retrieval_debug,
            cn_fallback_debug=cn_fallback_debug,
            table_preview_debug=table_preview_debug,
        )


class RerankStage:
    def __init__(self, *, settings: Settings, reranker: Any) -> None:
        self.settings = settings
        self.reranker = reranker

    def run(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        analysis: QueryAnalysis,
    ) -> RerankStageResult:
        reranked = self.reranker.rerank(
            question,
            retrieved,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
        )
        seed_chunks = reranked[: self.settings.retrieval.final_top_k]
        for rank_idx, chunk in enumerate(seed_chunks, start=1):
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                chunk.metadata["rerank_rank"] = rank_idx
            else:
                chunk.metadata = {"rerank_rank": rank_idx}
        return RerankStageResult(reranked=reranked, seed_chunks=seed_chunks)


class ContextStage:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: Any,
        parent_expander: Any | None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.parent_expander = parent_expander

    def run(
        self,
        *,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
        reranked: list[RetrievedChunk],
    ) -> ContextStageResult:
        summary_supplement_debug = _build_empty_supplement_debug()
        if analysis.intent.value == "summary" and seed_chunks:
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

        parent_expander = self.parent_expander or ParentContextExpander(
            parent_store=None,
            config=self.settings.retrieval,
        )
        evidence_selection = select_generation_v2_evidence(
            question=question,
            seed_chunks=seed_chunks,
            reranked=reranked,
            analysis=analysis,
            settings=self.settings,
            parent_expander=parent_expander,
        )
        return ContextStageResult(
            seed_chunks=seed_chunks,
            final_chunks=evidence_selection.final_chunks,
            summary_supplement_debug=summary_supplement_debug,
            parent_expansion_debug=evidence_selection.parent_expansion_debug,
            evidence_lifecycle_debug=evidence_selection.evidence_lifecycle_debug,
        )


class GenerationStage:
    def __init__(
        self,
        *,
        settings: Settings,
        confidence_scorer: ConfidenceScorer,
        generator_v2: Any,
    ) -> None:
        self.settings = settings
        self.confidence_scorer = confidence_scorer
        self.generator_v2 = generator_v2

    def run(
        self,
        *,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
        final_chunks: list[RetrievedChunk],
        table_preview_debug: dict,
        history: list[ConversationTurn] | None,
    ) -> GenerationStageResult:
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
        return GenerationStageResult(
            gen_result=gen_result,
            seed_confidence=seed_confidence,
            confidence=confidence,
            table_preview_answer_without_formal_citation=(
                table_preview_answer_without_formal_citation
            ),
        )


class ResponseStage:
    def __init__(self, *, settings: Settings, retriever: Any, reranker: Any) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker

    def run(
        self,
        *,
        retrieval: RetrievalStageResult,
        rerank: RerankStageResult,
        context: ContextStageResult,
        generation: GenerationStageResult,
        session_id: str | None,
        filters: QueryFilters | None,
        start_time: float,
    ) -> RAGPipelineResponse:
        return build_generation_v2_response(
            gen_result=generation.gen_result,
            analysis=retrieval.analysis,
            session_id=session_id,
            filters=filters,
            settings=self.settings,
            retrieved=retrieval.retrieved,
            reranked=rerank.reranked,
            seed_chunks=context.seed_chunks,
            final_chunks=context.final_chunks,
            seed_confidence=generation.seed_confidence,
            confidence=generation.confidence,
            start_time=start_time,
            retrieval_debug=retrieval.retrieval_debug,
            retriever_debug=getattr(self.retriever, "last_debug", {}),
            reranker_debug=getattr(self.reranker, "last_debug", {}),
            cn_fallback_debug=retrieval.cn_fallback_debug,
            table_preview_debug=retrieval.table_preview_debug,
            parent_expansion_debug=context.parent_expansion_debug,
            summary_supplement_debug=context.summary_supplement_debug,
            evidence_lifecycle_debug=context.evidence_lifecycle_debug,
            rewrite_trace=retrieval.rewrite_trace,
            table_preview_answer_without_formal_citation=(
                generation.table_preview_answer_without_formal_citation
            ),
        )
