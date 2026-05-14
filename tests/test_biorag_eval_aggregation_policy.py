"""Test BIORAG Eval v2 — aggregation policy (null handling, pass/partial/fail)."""
from scripts.evaluation.biorag_eval.aggregate_scores import (
    aggregate_judge_scores, aggregate_rule_scores,
)
from scripts.evaluation.biorag_eval.schemas import score_bucket


class TestJudgeAggregation:
    def _make_row(self, sid="t1", metric="faithfulness", score=None, valid=True, error=""):
        return {
            "sample_id": sid, "metric_name": metric,
            "score": score, "score_valid": valid,
            "judge_error_type": error, "cache_hit": False,
        }

    def test_basic_mean(self):
        rows = [
            self._make_row(metric="faithfulness", score=1.0),
            self._make_row(metric="faithfulness", score=0.5),
            self._make_row(metric="faithfulness", score=0.0),
        ]
        agg = aggregate_judge_scores(rows)
        f = agg["by_metric"]["faithfulness"]
        assert f["mean"] == 0.5
        assert f["count"] == 3

    def test_null_excluded_from_mean(self):
        rows = [
            self._make_row(metric="faithfulness", score=1.0),
            self._make_row(metric="faithfulness", score=None, valid=False),
            self._make_row(metric="faithfulness", score=0.0),
        ]
        agg = aggregate_judge_scores(rows)
        f = agg["by_metric"]["faithfulness"]
        assert f["count"] == 2
        assert f["mean"] == 0.5

    def test_pass_partial_fail_buckets(self):
        rows = [
            self._make_row(metric="faithfulness", score=1.0),
            self._make_row(metric="faithfulness", score=0.75),
            self._make_row(metric="faithfulness", score=0.5),
            self._make_row(metric="faithfulness", score=0.0),
        ]
        agg = aggregate_judge_scores(rows)
        f = agg["by_metric"]["faithfulness"]
        assert f["pass_count"] == 2
        assert f["pass_rate"] == 0.5
        assert f["partial_count"] == 1
        assert f["partial_rate"] == 0.25
        assert f["fail_count"] == 1
        assert f["fail_rate"] == 0.25

    def test_error_counted(self):
        rows = [
            self._make_row(metric="faithfulness", score=None, valid=False, error="json_parse_error"),
        ]
        agg = aggregate_judge_scores(rows)
        assert agg["judge_error_count"] == 1

    def test_multiple_metrics(self):
        rows = [
            self._make_row(metric="faithfulness", score=1.0),
            self._make_row(metric="answer_relevance", score=0.75),
            self._make_row(metric="abstention_correctness", score=0.0),
        ]
        agg = aggregate_judge_scores(rows)
        assert len(agg["by_metric"]) == 3
        assert agg["by_metric"]["faithfulness"]["count"] == 1
        assert agg["by_metric"]["abstention_correctness"]["count"] == 1


class TestScoreBucket:
    def test_pass(self):
        assert score_bucket(1.0) == "pass"
        assert score_bucket(0.75) == "pass"

    def test_partial(self):
        assert score_bucket(0.5) == "partial"
        assert score_bucket(0.6) == "partial"

    def test_fail(self):
        assert score_bucket(0.0) == "fail"
        assert score_bucket(0.25) == "fail"

    def test_not_applicable(self):
        assert score_bucket(None) == "not_applicable"
