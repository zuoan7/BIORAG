"""Test post-rerank same-doc body section coverage."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk, QueryAnalysis, QueryIntent
from src.synbio_rag.application.rerank_service import (
    _apply_same_doc_body_coverage,
    _BODY_SECTION_GROUPS,
)


def _chunk(cid: str, doc: str, section: str, rerank: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc, source_file=f"{doc}.pdf",
        title="Test", section=section, text=f"Content {cid}",
        rerank_score=rerank, metadata={},
    )


def _config(**overrides):
    defaults = {
        "same_doc_body_coverage_enabled": True,
        "same_doc_body_coverage_intents": ["factoid"],
        "same_doc_body_coverage_margin": 5,
        "same_doc_body_coverage_max_total": 1,
    }
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def _factoid_analysis():
    return QueryAnalysis(intent=QueryIntent.FACTOID, search_limit=40, rerank_top_k=10, requires_external_tools=False)


class TestDisabled:
    def test_disabled_no_change(self):
        config = _config(same_doc_body_coverage_enabled=False)
        selected = [_chunk("c1", "doc_A", "Abstract", 2.0)]
        pre_floor = [_chunk("c1", "doc_A", "Abstract", 2.0),
                     _chunk("c2", "doc_A", "Results", 0.3)]
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        assert result == selected


class TestCoverageReplaces:
    def test_replaces_abstract_with_results(self):
        config = _config()
        selected = [_chunk("c1", "doc_A", "Abstract", 2.0)]
        pre_floor = [
            _chunk("c1", "doc_A", "Abstract", 2.0),
            _chunk("c2", "doc_A", "Results", 0.3),
            _chunk("c3", "doc_A", "Discussion", 0.2),
        ]
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        assert len(result) == len(selected)
        assert result[0].section == "Results"
        assert result[0].metadata.get("added_by_body_coverage") is True

    def test_prefers_title_over_abstract_as_victim(self):
        config = _config()
        selected = [
            _chunk("t1", "doc_A", "Title", 3.0),
            _chunk("a1", "doc_A", "Abstract", 2.0),
        ]
        pre_floor = selected + [_chunk("r1", "doc_A", "Results", 0.5)]
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        # Title (idx 0) should be replaced, not Abstract
        assert result[0].section == "Results"
        assert result[1].section == "Abstract"

    def test_keeps_length_unchanged(self):
        config = _config()
        selected = [_chunk("a1", "doc_A", "Abstract", 2.0),
                    _chunk("x1", "doc_B", "Results", 1.5)]
        pre_floor = selected + [_chunk("r1", "doc_A", "Discussion", 0.5)]
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        assert len(result) == 2


class TestNoFalsePositive:
    def test_no_trigger_when_body_present(self):
        config = _config()
        selected = [_chunk("r1", "doc_A", "Results", 2.0)]
        pre_floor = selected
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        assert result == selected

    def test_no_trigger_when_body_beyond_margin(self):
        config = _config(same_doc_body_coverage_margin=2)
        selected = [_chunk("a1", "doc_A", "Abstract", 2.0)]
        # Results at rank 8, top_k=1, margin=2 → 1+2=3 < 8 → no trigger
        pre_floor = (
            [_chunk("a1", "doc_A", "Abstract", 2.0)] +
            [_chunk(f"x{i}", f"doc_X{i}", "Title", 1.0) for i in range(6)] +
            [_chunk("r1", "doc_A", "Results", 0.3)]
        )
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        assert result[0].section == "Abstract"

    def test_comparison_intent_no_trigger(self):
        config = _config()
        selected = [_chunk("a1", "doc_A", "Abstract", 2.0)]
        pre_floor = selected + [_chunk("r1", "doc_A", "Results", 0.5)]
        analysis = QueryAnalysis(intent=QueryIntent.COMPARISON, search_limit=40, rerank_top_k=10, requires_external_tools=False)
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), analysis, config
        )
        assert result[0].section == "Abstract"

    def test_no_replace_other_doc(self):
        config = _config()
        selected = [_chunk("a1", "doc_A", "Abstract", 2.0),
                    _chunk("t1", "doc_B", "Title", 1.5)]
        pre_floor = selected + [_chunk("r1", "doc_A", "Results", 0.5)]
        result = _apply_same_doc_body_coverage(
            selected, pre_floor, len(selected), _factoid_analysis(), config
        )
        # doc_A's Abstract replaced with Results; doc_B unchanged
        assert result[0].section == "Results"
        assert result[1].doc_id == "doc_B"

    def test_abstract_title_not_body(self):
        assert "Abstract" not in _BODY_SECTION_GROUPS
        assert "Title" not in _BODY_SECTION_GROUPS

    def test_results_discussion_methods_are_body(self):
        for s in ["Results", "Discussion", "Materials and Methods",
                   "Experimental Procedures", "Introduction", "Methods",
                   "Full Text", "Conclusion"]:
            assert s in _BODY_SECTION_GROUPS, f"{s} should be in BODY_SECTION_GROUPS"
