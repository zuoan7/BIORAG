from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain.config import Settings
from ..domain.schemas import QueryAnalysis, RetrievedChunk
from .generation_v2.evidence_lifecycle_debug import rerank_output_debug, stage_debug_from_chunks


@dataclass
class EvidenceSelectionResult:
    final_chunks: list[RetrievedChunk]
    parent_expansion_debug: dict
    evidence_lifecycle_debug: dict
    protected_ids: set[str]


def select_generation_v2_evidence(
    *,
    question: str,
    seed_chunks: list[RetrievedChunk],
    reranked: list[RetrievedChunk],
    analysis: QueryAnalysis,
    settings: Settings,
    parent_expander: Any,
) -> EvidenceSelectionResult:
    final_chunks, parent_expansion_debug = parent_expander.expand(
        question=question,
        seed_chunks=seed_chunks,
        analysis=analysis,
    )

    protect_enabled = settings.retrieval.protect_rerank_seeds_enabled
    protect_k = min(settings.retrieval.protect_rerank_seeds_top_k, len(seed_chunks))
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
    return EvidenceSelectionResult(
        final_chunks=final_chunks,
        parent_expansion_debug=parent_expansion_debug,
        evidence_lifecycle_debug=evidence_lifecycle_debug,
        protected_ids=protected_ids,
    )
