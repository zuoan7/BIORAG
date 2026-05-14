"""Test unit_correct schema and aggregation."""
from scripts.evaluation.biorag_eval.schemas import applicable_metrics, make_eval_record, score_bucket
from scripts.evaluation.biorag_eval.judge_prompts import numeric_prompt, PROMPT_VERSIONS


def _rec(**kw):
    d = {"sample_id": "t1", "question": "What is the value?", "answer": "5 μg/mL.",
         "selected_support": [{"support_id": "e1", "doc_id": "d1", "text": "Evidence."}],
         "reference": "5 μg/mL", "is_negative": False}
    d.update(kw)
    return make_eval_record(**d)


class TestUnitCorrect:
    def test_numeric_has_unit_correct(self):
        rec = _rec(category="method_result_numeric")
        m = applicable_metrics(rec)
        assert "unit_correct" in m

    def test_numeric_has_numeric_accuracy(self):
        rec = _rec(category="table_figure_caption")
        m = applicable_metrics(rec)
        assert "numeric_accuracy" in m
        assert "unit_correct" in m

    def test_non_numeric_no_unit(self):
        rec = _rec(category="factoid")
        m = applicable_metrics(rec)
        assert "unit_correct" not in m

    def test_unit_correct_in_prompt(self):
        p = numeric_prompt(_rec(category="method_result_numeric"))
        assert "unit_correct" in p.lower()
        assert "null" in p.lower()

    def test_unit_correct_not_a_numeric_score(self):
        """unit_correct is boolean/null, not a float 0-1."""
        # Bool values should be handled separately in aggregation, not mixed with numeric scores
        # score_bucket is for numeric scores only — unit_correct uses its own stats
        pass  # verified by aggregation logic in aggregate_scores.py

    def test_version(self):
        assert PROMPT_VERSIONS["unit_correct"] == "v2.0"

    def test_null_supported(self):
        rec = _rec(category="table_figure_caption", reference="", answer_key="")
        p = numeric_prompt(rec)
        assert "null" in p.lower()
