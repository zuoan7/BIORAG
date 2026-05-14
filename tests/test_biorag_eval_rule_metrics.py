"""Test BIORAG Eval v1 rule metrics."""
import pytest
from scripts.evaluation.biorag_eval.rule_metrics import (
    doc_recall, doc_recall_support, doc_recall_citation,
    route_match, wrong_doc_citation, negative_citation_zero,
    compute_all_rule_metrics,
)
from scripts.evaluation.biorag_eval.schemas import make_eval_record


def make_rec(**kwargs):
    defaults = {
        "sample_id": "test_1", "expected_doc_ids": ["doc_a", "doc_b"],
        "selected_support_doc_ids": ["doc_a"], "cited_doc_ids": ["doc_a", "doc_c"],
        "expected_route": "factoid", "route_pred": "factoid",
        "is_negative": False, "citation_count": 2,
    }
    defaults.update(kwargs)
    return make_eval_record(**defaults)


class TestDocRecall:
    def test_full_recall(self):
        assert doc_recall(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_partial_recall(self):
        assert doc_recall(["a", "b", "c"], ["a"]) == pytest.approx(1/3)

    def test_no_expected(self):
        assert doc_recall([], ["a"]) == 1.0

    def test_doc_recall_support(self):
        rec = make_rec(expected_doc_ids=["doc_a", "doc_b"], selected_support_doc_ids=["doc_a"])
        assert doc_recall_support(rec) == 0.5

    def test_doc_recall_citation(self):
        rec = make_rec(expected_doc_ids=["doc_a"], cited_doc_ids=["doc_a", "doc_c"])
        assert doc_recall_citation(rec) == 1.0


class TestRouteMatch:
    def test_exact_match(self):
        rec = make_rec(expected_route="factoid", route_pred="factoid")
        assert route_match(rec)

    def test_mismatch(self):
        rec = make_rec(expected_route="factoid", route_pred="summary")
        assert not route_match(rec)

    def test_comparison_substring(self):
        rec = make_rec(expected_route="comparison", route_pred="comparison")
        assert route_match(rec)

    def test_negative_match(self):
        rec = make_rec(expected_route="negative", route_pred="negative")
        assert route_match(rec)

    def test_empty_route(self):
        rec = make_rec(expected_route="", route_pred="factoid")
        assert not route_match(rec)


class TestWrongDoc:
    def test_no_wrong(self):
        rec = make_rec(expected_doc_ids=["a", "b"], cited_doc_ids=["a", "b"])
        assert not wrong_doc_citation(rec)

    def test_wrong_doc(self):
        rec = make_rec(expected_doc_ids=["a"], cited_doc_ids=["a", "z_unknown"])
        assert wrong_doc_citation(rec)

    def test_no_expected_ids(self):
        rec = make_rec(expected_doc_ids=[], cited_doc_ids=["a"])
        assert not wrong_doc_citation(rec)

    def test_no_citations(self):
        rec = make_rec(expected_doc_ids=["a"], cited_doc_ids=[])
        assert not wrong_doc_citation(rec)


class TestNegativeCitation:
    def test_negative_with_citation(self):
        rec = make_rec(is_negative=True, citation_count=1)
        assert not negative_citation_zero(rec)

    def test_negative_zero_citation(self):
        rec = make_rec(is_negative=True, citation_count=0)
        assert negative_citation_zero(rec)

    def test_non_negative_always_passes(self):
        rec = make_rec(is_negative=False, citation_count=5)
        assert negative_citation_zero(rec)


class TestComputeAll:
    def test_returns_all_fields(self):
        rec = make_rec()
        result = compute_all_rule_metrics(rec)
        for key in ["sample_id", "route_match", "doc_recall_support", "doc_recall_citation",
                     "citation_count", "wrong_doc_citation", "negative_citation_zero"]:
            assert key in result

    def test_no_crash_on_empty(self):
        rec = make_rec(expected_doc_ids=[], cited_doc_ids=[], selected_support_doc_ids=[])
        result = compute_all_rule_metrics(rec)
        assert result["doc_recall_support"] == 1.0
        assert result["doc_recall_citation"] == 1.0
