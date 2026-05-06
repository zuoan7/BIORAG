"""Tests for v5 Phase3b chunk retrieval smoke diagnostic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Import from the diagnostic module (namespace package workaround)
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "chunk_retrieval_smoke_v5",
    Path(__file__).parent.parent / "scripts" / "diagnostics" / "chunk_retrieval_smoke_v5.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["chunk_retrieval_smoke_v5"] = _mod

BM25DiagnosticRetriever = _mod.BM25DiagnosticRetriever
load_chunks = _mod.load_chunks
load_cases = _mod.load_cases
reciprocal_rank_fusion = _mod.reciprocal_rank_fusion
evaluate_case = _mod.evaluate_case
check_terms = _mod.check_terms
compute_mode_metrics = _mod.compute_mode_metrics


def _make_chunk(chunk_id: str, doc_id: str, text: str, evidence_types=None, page_start=1, page_end=1):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "retrieval_text": text,
        "evidence_types": evidence_types or ["paragraph"],
        "block_types": ["paragraph"],
        "source_block_ids": [f"{doc_id}_b0001"],
        "page_start": page_start,
        "page_end": page_end,
    }


class TestCaseSchema:
    def test_valid_cases_load(self):
        cases = load_cases("data/evaluation/v5_phase3b_chunk_retrieval_smoke_cases.jsonl")
        assert len(cases) >= 20
        for c in cases:
            assert "case_id" in c
            assert "category" in c
            assert "query" in c
            assert c["category"] in ("table", "figure", "paragraph", "negative")
            if c.get("negative_only"):
                assert "forbidden_terms" in c

    def test_case_categories_coverage(self):
        cases = load_cases("data/evaluation/v5_phase3b_chunk_retrieval_smoke_cases.jsonl")
        cats = {c["category"] for c in cases}
        assert cats >= {"table", "figure", "paragraph", "negative"}

    def test_gate_cases_present(self):
        cases = load_cases("data/evaluation/v5_phase3b_chunk_retrieval_smoke_cases.jsonl")
        ids = {c["case_id"] for c in cases}
        assert "tbl_doc0005_gm_orf2729" in ids
        assert "fig_doc0005_pathway" in ids


class TestBM25Retriever:
    def test_index_and_search(self):
        chunks = [
            _make_chunk("c1", "doc_a", "lignin degradation by Bacillus ligniniphilus strain L1"),
            _make_chunk("c2", "doc_a", "cytochrome c551 protein encoding gene gm_orf2729 length 369 bp"),
            _make_chunk("c3", "doc_b", "Pichia pastoris secretion signal Ost1"),
        ]
        bm25 = BM25DiagnosticRetriever()
        bm25.index(chunks)

        results = bm25.search("gm_orf2729 cytochrome c551", limit=5)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "c2"
        assert results[0]["doc_id"] == "doc_a"

    def test_empty_query(self):
        chunks = [_make_chunk("c1", "doc_a", "some text")]
        bm25 = BM25DiagnosticRetriever()
        bm25.index(chunks)
        results = bm25.search("", limit=5)
        assert results == []

    def test_no_match_query(self):
        chunks = [_make_chunk("c1", "doc_a", "some text")]
        bm25 = BM25DiagnosticRetriever()
        bm25.index(chunks)
        results = bm25.search("xyzabc12345", limit=5)
        assert len(results) == 0

    def test_search_returns_required_fields(self):
        chunks = [_make_chunk("c1", "doc_a", "lignin degradation pathway")]
        bm25 = BM25DiagnosticRetriever()
        bm25.index(chunks)
        results = bm25.search("lignin", limit=5)
        assert len(results) == 1
        for key in ["rank", "score", "doc_id", "chunk_id", "evidence_types"]:
            assert key in results[0]


class TestTermMatching:
    def test_check_terms_finds_exact(self):
        assert check_terms("gm_orf2729 369 Cytochrome c551", ["gm_orf2729", "369"]) == ["gm_orf2729", "369"]

    def test_check_terms_case_insensitive(self):
        assert "cytochrome c551" in check_terms("Cytochrome c551", ["cytochrome c551"])

    def test_check_terms_empty(self):
        assert check_terms("some text", []) == []

    def test_check_terms_no_match(self):
        assert check_terms("some text", ["xyz123"]) == []


class TestNegativeDetection:
    def test_forbidden_terms_detected(self):
        hits = check_terms("Page 11 of 14 text here", ["Page 11 of 14", "Zhu et al."])
        assert "Page 11 of 14" in hits

    def test_forbidden_terms_absent(self):
        hits = check_terms("normal paragraph text", ["Journal Pre-proof"])
        assert hits == []


class TestEvaluateCase:
    def test_positive_case_pass(self):
        case = {
            "case_id": "t1", "category": "table",
            "query": "what is gm_orf2729?",
            "expected_doc_ids": ["doc_0005"],
            "expected_terms_any": ["gm_orf2729", "369"],
            "expected_evidence_types_any": ["table_text"],
            "negative_terms_absent": [],
        }
        results = [
            _make_chunk("c1", "doc_0005", "gm_orf2729 369 Cytochrome c551",
                       evidence_types=["table_text"]),
        ]
        ev = evaluate_case(case, results, top_k=5)
        assert ev["pass"] is True
        assert ev["doc_hit_at_1"] is True
        assert ev["evidence_type_hit"] is True
        assert len(ev["expected_terms_hit"]) >= 2

    def test_positive_case_wrong_doc(self):
        case = {
            "case_id": "t2", "category": "table",
            "query": "find doc_0005",
            "expected_doc_ids": ["doc_0005"],
            "expected_terms_any": ["target"],
            "expected_evidence_types_any": [],
            "negative_terms_absent": [],
        }
        results = [
            _make_chunk("c1", "doc_other", "target info here", evidence_types=["paragraph"]),
        ]
        ev = evaluate_case(case, results, top_k=5)
        assert ev["doc_hit_at_5"] is False

    def test_negative_case_pass(self):
        case = {
            "case_id": "n1", "category": "negative", "query": "test",
            "expected_doc_ids": [], "expected_terms_any": [],
            "expected_evidence_types_any": [],
            "negative_only": True,
            "forbidden_terms": ["Journal Pre-proof"],
            "max_forbidden_topk_hits": 0,
        }
        results = [
            _make_chunk("c1", "doc_a", "normal paragraph about science", evidence_types=["paragraph"]),
        ]
        ev = evaluate_case(case, results, top_k=5)
        assert ev["pass"] is True

    def test_negative_case_fail(self):
        case = {
            "case_id": "n2", "category": "negative", "query": "test",
            "expected_doc_ids": [], "expected_terms_any": [],
            "expected_evidence_types_any": [],
            "negative_only": True,
            "forbidden_terms": ["Journal Pre-proof"],
            "max_forbidden_topk_hits": 0,
        }
        results = [
            _make_chunk("c1", "doc_a", "This is a Journal Pre-proof article", evidence_types=["paragraph"]),
        ]
        ev = evaluate_case(case, results, top_k=5)
        assert ev["pass"] is False
        assert len(ev["forbidden_terms_top5"]) > 0


class TestReciprocalRankFusion:
    def test_rrf_combines(self):
        bm25 = [
            {"chunk_id": "c1", "rank": 1, "score": 5.0, "doc_id": "d1", "text": "a"},
            {"chunk_id": "c2", "rank": 2, "score": 3.0, "doc_id": "d2", "text": "b"},
        ]
        vec = [
            {"chunk_id": "c2", "rank": 1, "score": 0.9, "doc_id": "d2", "text": "b"},
            {"chunk_id": "c1", "rank": 2, "score": 0.8, "doc_id": "d1", "text": "a"},
        ]
        fused = reciprocal_rank_fusion(bm25, vec, k=60)
        assert len(fused) >= 2
        assert "rrf_score" in fused[0]


class TestMetrics:
    def test_compute_metrics(self):
        case_results = [
            {"case_id": "t1", "category": "table", "pass": True, "verdict": "PASS",
             "doc_hit_at_1": True, "doc_hit_at_3": True, "doc_hit_at_5": True,
             "expected_terms_hit": ["a"], "expected_terms_missed": [],
             "evidence_type_hit": True, "mrr_doc": 1.0,
             "forbidden_terms_top1": [], "forbidden_terms_top5": [],
             "top_k_results": [{"evidence_types": ["table_text"]}]},
            {"case_id": "n1", "category": "negative", "pass": True, "verdict": "PASS",
             "doc_hit_at_1": False, "doc_hit_at_3": False, "doc_hit_at_5": False,
             "expected_terms_hit": [], "expected_terms_missed": [],
             "evidence_type_hit": True, "mrr_doc": 0.0,
             "forbidden_terms_top1": [], "forbidden_terms_top5": [],
             "top_k_results": [{"evidence_types": ["paragraph"]}]},
        ]
        metrics = compute_mode_metrics("bm25", "RUN", case_results)
        assert metrics["mode"] == "bm25"
        assert metrics["status"] == "RUN"
        assert metrics["positive_case_count"] == 1
        assert metrics["negative_case_count"] == 1
        assert metrics["doc_hit_at_5"] == 1.0
        assert metrics["negative_case_pass_rate"] == 1.0
        assert metrics["forbidden_term_top5_count"] == 0

    def test_json_report_contains_metrics(self):
        """Verify the output JSON has required per-mode metrics structure."""
        # This test doesn't require running the full smoke
        metrics = compute_mode_metrics("bm25", "RUN", [])
        for key in ["doc_hit_at_1", "doc_hit_at_5", "expected_terms_hit_at_5",
                     "evidence_type_hit_at_5", "mrr_doc", "forbidden_term_top5_count"]:
            assert key in metrics, f"Missing key: {key}"
