from __future__ import annotations

from typing import Any

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, RetrievedChunk
from .retrieval_postprocess import (
    _apply_comparison_diversity,
    _apply_same_doc_body_expansion,
    _apply_source_floor,
    _apply_structure_marker_boost,
    _apply_title_keyword_boost,
)


class RetrievalPostProcessor:
    def apply_comparison_diversity(
        self,
        chunks: list[RetrievedChunk],
        limit: int,
        analysis: QueryAnalysis | None,
        config: RetrievalConfig,
    ) -> list[RetrievedChunk]:
        return _apply_comparison_diversity(chunks, limit, analysis, config)

    def apply_title_keyword_boost(
        self,
        chunks: list[RetrievedChunk],
        question: str,
        config: RetrievalConfig,
    ) -> list[RetrievedChunk]:
        return _apply_title_keyword_boost(chunks, question, config)

    def apply_structure_marker_boost(
        self,
        chunks: list[RetrievedChunk],
        question: str,
        config: RetrievalConfig,
    ) -> list[RetrievedChunk]:
        return _apply_structure_marker_boost(chunks, question, config)

    def apply_same_doc_body_expansion(
        self,
        *,
        diversified: list[RetrievedChunk],
        dense_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        config: Any,
        question: str = "",
        bm25_retriever: Any = None,
    ) -> list[RetrievedChunk]:
        return _apply_same_doc_body_expansion(
            diversified=diversified,
            dense_results=dense_results,
            bm25_results=bm25_results,
            config=config,
            question=question,
            bm25_retriever=bm25_retriever,
        )

    def apply_source_floor(
        self,
        *,
        expanded: list[RetrievedChunk],
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        config: Any,
    ) -> list[RetrievedChunk]:
        return _apply_source_floor(
            expanded=expanded,
            dense_results=dense_results,
            sparse_results=sparse_results,
            config=config,
        )
