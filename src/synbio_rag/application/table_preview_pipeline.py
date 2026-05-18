from __future__ import annotations

from ..domain.schemas import RetrievedChunk
from .table_preview import TablePreviewCandidateProvider, apply_table_preview


def run_table_preview(
    *,
    question: str,
    retrieved: list[RetrievedChunk],
    config,
    provider: TablePreviewCandidateProvider | None = None,
) -> tuple[list[RetrievedChunk], dict]:
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
