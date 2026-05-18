from __future__ import annotations

from ..domain.schemas import RetrievedChunk
from .table_preview import (
    TablePreviewCandidateProvider,
    run_table_preview as _run_table_preview,
    sanitize_table_preview_debug,
)


def run_table_preview(
    *,
    question: str,
    retrieved: list[RetrievedChunk],
    config,
    provider: TablePreviewCandidateProvider | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    return _run_table_preview(
        question=question,
        retrieved=retrieved,
        config=config,
        provider=provider,
    )
