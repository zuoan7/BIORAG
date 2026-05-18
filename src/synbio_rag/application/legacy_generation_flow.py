from __future__ import annotations

import time
from typing import Any

from ..domain.config import Settings
from ..domain.schemas import (
    ConversationTurn,
    QueryAnalysis,
    QueryFilters,
    QueryIntent,
    RAGResponse,
    RetrievedChunk,
)
from .original_cn_fallback import sanitize_original_cn_fallback_debug
from .table_preview_pipeline import sanitize_table_preview_debug


def run_legacy_generation_flow(
    *,
    question: str,
    session_id: str | None,
    history: list[ConversationTurn] | None,
    filters: QueryFilters | None,
    analysis: QueryAnalysis,
    settings: Settings,
    retrieved: list[RetrievedChunk],
    reranked: list[RetrievedChunk],
    seed_chunks: list[RetrievedChunk],
    start_time: float,
    retrieval_debug: dict[str, object],
    retriever_debug: dict[str, object],
    reranker_debug: dict[str, object],
    cn_fallback_debug: dict,
    table_preview_debug: dict,
    rewrite_trace: Any,
    neighbor_expander: Any,
    context_builder: Any,
    generator: Any,
    confidence_scorer: Any,
    external_tools: Any,
) -> RAGResponse:
    final_chunks = neighbor_expander.expand(seed_chunks)
    context = context_builder.build(question, final_chunks, history=history, intent=analysis.intent)
    evidence_quality = generator.assess_evidence(question, final_chunks, analysis=analysis)
    answer = generator.generate(
        question,
        context,
        final_chunks,
        analysis=analysis,
        history=history,
        assessment=evidence_quality,
    )
    confidence = confidence_scorer.score(final_chunks)
    tool_execution = external_tools.run_if_needed(
        question=question,
        analysis=analysis,
        low_confidence=confidence_scorer.needs_external_tool(confidence),
    )
    citations = generator.build_citations(final_chunks, evidence_quality)
    if analysis.intent == QueryIntent.NEGATIVE:
        citations = []
    answer = generator.validate_generated_answer(answer, citations, evidence_quality)

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
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "tenant_id": filters.tenant_id if filters else "default",
            "hybrid_enabled": settings.retrieval.hybrid_enabled,
            "bm25_enabled": settings.retrieval.bm25_enabled,
            "retrieval_hits": retriever_debug,
            "rerank_hits": reranker_debug,
            "neighbor_expansion": getattr(neighbor_expander, "last_debug", {}),
            "original_cn_fallback": sanitize_original_cn_fallback_debug(cn_fallback_debug),
            "table_preview": sanitize_table_preview_debug(table_preview_debug),
            "filter_strategy": retrieval_debug,
            "evidence_quality": evidence_quality.__dict__,
        },
    )
