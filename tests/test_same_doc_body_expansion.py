"""Test same-doc body candidate expansion in hybrid.py."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.vectorstores.hybrid import (
    _apply_same_doc_body_expansion,
    _BODY_EXPAND_SECTIONS,
)


def _make_chunk(chunk_id: str, doc_id: str, section: str,
                vector_score: float = 0.5, bm25_score: float = 10.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title="Test Paper",
        section=section,
        text=f"Content of {chunk_id}",
        vector_score=vector_score,
        bm25_score=bm25_score,
        metadata={},
    )


def _config(**overrides) -> RetrievalConfig:
    defaults = {
        "same_doc_body_expand_enabled": True,
        "same_doc_body_expand_top_docs": 5,
        "same_doc_body_expand_per_doc": 2,
        "same_doc_body_expand_max_total": 8,
        "same_doc_body_expand_min_doc_rank": 20,
        "same_doc_body_expand_require_missing_body": True,
    }
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


class TestExpansionDisabled:
    def test_disabled_returns_original(self):
        """expansion disabled 时返回原始列表。"""
        config = _config(same_doc_body_expand_enabled=False)
        diversified = [_make_chunk("c1", "doc_A", "Abstract")]
        result = _apply_same_doc_body_expansion(diversified, [], [], config)
        assert result is diversified


class TestExpansionAddsMissingBody:
    def test_adds_body_when_doc_missing_body(self):
        """当 doc 缺少 body section 时补入 body chunk。"""
        config = _config()
        diversified = [_make_chunk("c1", "doc_A", "Abstract", 0.5, 10.0)]
        dense_results = [_make_chunk("c2", "doc_A", "Results", 0.3, 5.0)]
        bm25_results = []

        result = _apply_same_doc_body_expansion(
            diversified, dense_results, bm25_results, config
        )
        assert len(result) > len(diversified)
        expanded_ids = {c.chunk_id for c in result}
        assert "c2" in expanded_ids

    def test_adds_body_with_metadata(self):
        """扩充的 chunk 应有正确的 metadata。"""
        config = _config()
        diversified = [_make_chunk("c1", "doc_A", "Abstract")]
        dense = [_make_chunk("c2", "doc_A", "Discussion", 0.4, 8.0)]

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        added = [c for c in result if c.chunk_id not in {d.chunk_id for d in diversified}]
        assert len(added) == 1
        assert added[0].metadata.get("added_by_same_doc_body_expand") is True
        assert "body_section" in added[0].metadata


class TestExpansionNoFalsePositive:
    def test_no_expand_when_body_present(self):
        """doc 已有 body section 时不扩充。"""
        config = _config()
        diversified = [
            _make_chunk("c1", "doc_A", "Abstract"),
            _make_chunk("c2", "doc_A", "Results"),
        ]
        result = _apply_same_doc_body_expansion(diversified, [], [], config)
        assert len(result) == len(diversified)

    def test_no_duplicate_chunks(self):
        """不重复加入已有 chunk。"""
        config = _config()
        diversified = [_make_chunk("c1", "doc_A", "Abstract")]
        dense = [_make_chunk("c1", "doc_A", "Discussion")]  # same chunk_id

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        assert len(result) == len(diversified)

    def test_does_not_add_title_or_abstract(self):
        """不扩充 Title/Abstract/References/Funding。"""
        config = _config()
        diversified = [_make_chunk("c1", "doc_A", "Abstract")]
        dense = [
            _make_chunk("t1", "doc_A", "Title"),
            _make_chunk("r1", "doc_A", "References"),
            _make_chunk("f1", "doc_A", "Funding"),
        ]

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        assert len(result) == len(diversified)


class TestExpansionLimits:
    def test_per_doc_limit(self):
        """per_doc 限制应生效。"""
        config = _config(same_doc_body_expand_per_doc=1)
        diversified = [_make_chunk("c1", "doc_A", "Abstract")]
        dense = [
            _make_chunk("r1", "doc_A", "Results", 0.5),
            _make_chunk("r2", "doc_A", "Discussion", 0.4),
        ]

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        added = [c for c in result if c.chunk_id not in {"c1"}]
        assert len(added) == 1

    def test_max_total_limit(self):
        """max_total 限制应生效。"""
        config = _config(same_doc_body_expand_max_total=1, same_doc_body_expand_per_doc=2)
        diversified = [
            _make_chunk("a1", "doc_A", "Abstract"),
            _make_chunk("b1", "doc_B", "Abstract"),
        ]
        dense = [
            _make_chunk("ar1", "doc_A", "Results", 0.5),
            _make_chunk("ar2", "doc_A", "Discussion", 0.4),
            _make_chunk("br1", "doc_B", "Results", 0.5),
        ]

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        added = [c for c in result if c.chunk_id not in {"a1", "b1"}]
        assert len(added) <= 1

    def test_min_doc_rank_filter(self):
        """min_doc_rank 之外的 doc 不应扩充。"""
        config = _config(same_doc_body_expand_min_doc_rank=5)
        # doc_A at rank 0, doc_B at rank 10
        diversified = []
        for i in range(6):
            diversified.append(_make_chunk(f"x{i}", f"doc_X{i}", "Title"))
        diversified.append(_make_chunk("a1", "doc_A", "Abstract", 0.5, 10.0))
        # doc_B at rank 10, beyond min_doc_rank=5
        for i in range(5):
            diversified.append(_make_chunk(f"y{i}", f"doc_Y{i}", "Title"))
        diversified.append(_make_chunk("b1", "doc_B", "Abstract", 0.3, 5.0))

        dense = [
            _make_chunk("ar1", "doc_A", "Results", 0.3),
            _make_chunk("br1", "doc_B", "Results", 0.3),
        ]

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        added = [c for c in result if c.chunk_id not in {d.chunk_id for d in diversified}]
        # doc_A at rank 6 (>=5) should not be expanded
        doc_a_added = [c for c in added if c.doc_id == "doc_A"]
        assert len(doc_a_added) == 0


class TestExpansionPreservesRerankPipeline:
    def test_expansion_is_rerank_input_not_final(self):
        """expansion 扩充 rerank 输入，不直接进入 final output。"""
        config = _config()
        diversified = [_make_chunk("c1", "doc_A", "Abstract")]
        dense = [_make_chunk("c2", "doc_A", "Results", 0.3)]

        result = _apply_same_doc_body_expansion(diversified, dense, [], config)
        # expansion 返回的列表直接交给 reranker
        assert len(result) > len(diversified)
        # 原始 hybrid chunks 保持不变
        for orig in diversified:
            assert orig in result
        # 新 chunk 在 result 中（会经过 rerank）
        assert any(c.metadata.get("added_by_same_doc_body_expand") for c in result)
