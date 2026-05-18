from __future__ import annotations

import time
from typing import Any

from ..domain.config import Settings
from ..domain.schemas import QueryFilters, QueryIntent, RAGResponse, RetrievedChunk
from .original_cn_fallback import sanitize_original_cn_fallback_debug
from .table_preview_pipeline import sanitize_table_preview_debug


def build_generation_v2_response(
    *,
    gen_result: Any,
    analysis: Any,
    session_id: str | None,
    filters: QueryFilters | None,
    settings: Settings,
    retrieved: list[RetrievedChunk],
    reranked: list[RetrievedChunk],
    seed_chunks: list[RetrievedChunk],
    final_chunks: list[RetrievedChunk],
    seed_confidence: float,
    confidence: float,
    start_time: float,
    retrieval_debug: dict[str, object],
    retriever_debug: dict[str, object],
    reranker_debug: dict[str, object],
    cn_fallback_debug: dict,
    table_preview_debug: dict,
    parent_expansion_debug: dict,
    summary_supplement_debug: dict,
    evidence_lifecycle_debug: dict,
    rewrite_trace: Any,
    table_preview_answer_without_formal_citation: bool,
) -> RAGResponse:
    gv2_debug = gen_result.debug
    gv2_debug["table_preview_answer_without_formal_citation"] = (
        table_preview_answer_without_formal_citation
    )
    gv2_debug["summary_section_supplement"] = summary_supplement_debug
    gv2_lifecycle_debug = dict(gv2_debug.get("evidence_lifecycle_debug", {}))
    evidence_lifecycle_debug.update(gv2_lifecycle_debug)
    gv2_debug["evidence_lifecycle_debug"] = evidence_lifecycle_debug

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
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "seed_confidence": seed_confidence,
            "final_confidence": confidence,
            "tenant_id": filters.tenant_id if filters else "default",
            "hybrid_enabled": settings.retrieval.hybrid_enabled,
            "bm25_enabled": settings.retrieval.bm25_enabled,
            "retrieval_hits": retriever_debug,
            "rerank_hits": reranker_debug,
            "neighbor_expansion": {
                "enabled": False,
                "reason": "generation_v2_seed_only_or_replaced_by_parent_expansion",
                "input_count": len(seed_chunks),
                "output_count": len(seed_chunks),
            },
            "original_cn_fallback": sanitize_original_cn_fallback_debug(cn_fallback_debug),
            "table_preview": sanitize_table_preview_debug(table_preview_debug),
            "parent_expansion": parent_expansion_debug,
            "filter_strategy": retrieval_debug,
            "generation_v2": gen_result.debug,
            "evidence_lifecycle_debug": evidence_lifecycle_debug,
            "query_rewrite": rewrite_trace.to_dict(),
        },
    )
