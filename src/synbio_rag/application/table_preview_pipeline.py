from __future__ import annotations

from ..domain.schemas import RetrievedChunk
from .table_preview import TablePreviewCandidateProvider, apply_table_preview


def run_table_preview(
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


def sanitize_table_preview_debug(debug: dict) -> dict:
    return {
        key: value
        for key, value in debug.items()
        if key != "merged_candidates"
    }
