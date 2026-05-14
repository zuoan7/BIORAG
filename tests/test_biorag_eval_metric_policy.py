"""Test BIORAG Eval v2 — route-aware metric applicability."""
from scripts.evaluation.biorag_eval.schemas import (
    applicable_metrics, metric_applicable, make_eval_record,
    score_bucket, PASS_THRESHOLD, PARTIAL_THRESHOLD,
)


def _rec(**kw):
    d = {"sample_id": "t1", "question": "Q?", "answer": "A.", "category": "factoid",
         "reference": "Ref.", "answer_key": "", "expected_behavior": "factual",
         "is_negative": False, "expected_route": "factoid"}
    d.update(kw)
    return make_eval_record(**d)


class TestRouteAwareDispatch:
    def test_factoid_metrics(self):
        m = applicable_metrics(_rec(category="factoid"))
        assert "faithfulness" in m
        assert "answer_relevance" in m
        assert "abstention_correctness" not in m

    def test_summary_has_completeness(self):
        m = applicable_metrics(_rec(category="summary"))
        assert "answer_completeness" in m

    def test_comparison_has_both_branches(self):
        m = applicable_metrics(_rec(category="comparison"))
        assert "both_branches_covered" in m
        assert "comparison_axis_covered" in m

    def test_numeric_has_unit(self):
        m = applicable_metrics(_rec(category="table_figure_caption"))
        assert "numeric_accuracy" in m
        assert "unit_correct" in m

    def test_negative_only_abstention(self):
        m = applicable_metrics(_rec(is_negative=True))
        assert m == ["abstention_correctness"]

    def test_no_accuracy_without_ref(self):
        m = applicable_metrics(_rec(category="factoid", reference="", answer_key="", expected_behavior="factual"))
        assert "answer_accuracy" not in m

    def test_has_accuracy_with_ref(self):
        m = applicable_metrics(_rec(category="factoid", reference="Some ref"))
        assert "answer_accuracy" in m


class TestMetricApplicableV2:
    def test_faithfulness_applicable(self):
        ok, _ = metric_applicable(_rec(), "faithfulness")
        assert ok

    def test_abstention_not_applicable_non_negative(self):
        ok, reason = metric_applicable(_rec(is_negative=False), "abstention_correctness")
        assert not ok
        assert "not_applicable" in reason

    def test_abstention_applicable_negative(self):
        ok, _ = metric_applicable(_rec(is_negative=True), "abstention_correctness")
        assert ok

    def test_comparison_not_applicable_factoid(self):
        ok, _ = metric_applicable(_rec(category="factoid"), "both_branches_covered")
        assert not ok


class TestScoreBucket:
    def test_pass(self):
        assert score_bucket(1.0) == "pass"
        assert score_bucket(0.75) == "pass"
        assert score_bucket(0.8) == "pass"

    def test_partial(self):
        assert score_bucket(0.5) == "partial"
        assert score_bucket(0.6) == "partial"

    def test_fail(self):
        assert score_bucket(0.0) == "fail"
        assert score_bucket(0.4) == "fail"

    def test_null(self):
        assert score_bucket(None) == "not_applicable"
