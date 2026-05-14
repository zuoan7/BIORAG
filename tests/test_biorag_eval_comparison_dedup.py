"""Test comparison dedup: composite prompt must not add extra faithfulness rows."""
from scripts.evaluation.biorag_eval.schemas import applicable_metrics, make_eval_record
from scripts.evaluation.biorag_eval.judge_prompts import comparison_prompt, PROMPT_BUILDERS, PROMPT_VERSIONS


def _rec(**kw):
    d = {"sample_id": "t1", "question": "Compare A and B", "answer": "A is better.",
         "selected_support": [{"support_id": "e1", "doc_id": "d1", "text": "Evidence."}],
         "reference": "Ref.", "is_negative": False, "category": "comparison"}
    d.update(kw)
    return make_eval_record(**d)


class TestComparisonDedup:
    def test_comparison_faithfulness_not_in_global_metrics(self):
        rec = _rec()
        metrics = applicable_metrics(rec)
        # Comparison samples should have global faithfulness AND comparison_faithfulness
        assert "faithfulness" in metrics  # global faithfulness
        assert "comparison_faithfulness" in metrics  # comparison-specific
        assert "both_branches_covered" in metrics

    def test_global_faithfulness_not_from_comparison_composite(self):
        rec = _rec()
        p = comparison_prompt(rec)
        # The comparison prompt must use "comparison_faithfulness" not "faithfulness"
        assert '"comparison_faithfulness"' in p
        # "faithfulness" alone (outside of comparison_faithfulness) should NOT appear as a standalone JSON key
        assert '"faithfulness":' not in p, "comparison prompt should not output global 'faithfulness' key"

    def test_comparison_faithfulness_has_builder(self):
        assert "comparison_faithfulness" in PROMPT_BUILDERS

    def test_comparison_faithfulness_version(self):
        assert PROMPT_VERSIONS["comparison_faithfulness"] == "v3.1"

    def test_both_branches_still_in_metrics(self):
        rec = _rec()
        metrics = applicable_metrics(rec)
        assert "both_branches_covered" in metrics
        assert "comparison_axis_covered" in metrics

    def test_negative_excluded_from_comparison(self):
        rec = _rec(is_negative=True)
        metrics = applicable_metrics(rec)
        assert "both_branches_covered" not in metrics
        assert "comparison_faithfulness" not in metrics
        assert "abstention_correctness" in metrics
