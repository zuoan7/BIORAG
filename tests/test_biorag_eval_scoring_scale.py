"""Test BIORAG Eval v2 — scoring scale and prompt schema."""
from scripts.evaluation.biorag_eval.schemas import SCORE_SCALE, PASS_THRESHOLD, PARTIAL_THRESHOLD
from scripts.evaluation.biorag_eval.judge_prompts import (
    faithfulness_prompt, answer_relevance_prompt,
    evidence_recall_prompt, answer_accuracy_prompt,
    answer_completeness_prompt, negative_abstention_prompt,
    comparison_prompt, numeric_prompt,
    PROMPT_BUILDERS, PROMPT_VERSIONS,
)
from scripts.evaluation.biorag_eval.schemas import make_eval_record


def _rec(**kw):
    d = {"sample_id": "t1", "question": "Q?", "answer": "A.",
         "selected_support": [{"support_id": "e1", "doc_id": "d1", "text": "Evidence."}],
         "reference": "Ref.", "answer_key": "Key.", "expected_behavior": "factual",
         "is_negative": False, "citation_count": 1}
    d.update(kw)
    return make_eval_record(**d)


class TestScoringScale:
    def test_scale_values(self):
        assert 1.0 in SCORE_SCALE
        assert 0.75 in SCORE_SCALE
        assert 0.5 in SCORE_SCALE
        assert 0.0 in SCORE_SCALE
        assert None not in SCORE_SCALE  # null is separate

    def test_thresholds(self):
        assert PASS_THRESHOLD == 0.75
        assert PARTIAL_THRESHOLD == 0.5


class TestV2PromptsReference4pt:
    """All v2 prompts must reference the 4-point scale."""
    def test_faithfulness_mentions_0_75(self):
        p = faithfulness_prompt(_rec())
        assert "0.75" in p

    def test_relevance_mentions_0_75(self):
        p = answer_relevance_prompt(_rec())
        assert "0.75" in p

    def test_evidence_recall_null_when_no_ref(self):
        rec = _rec(reference="", answer_key="", expected_behavior="")
        p = evidence_recall_prompt(rec)
        assert "null" in p.lower()

    def test_completeness_prompt_exists(self):
        p = answer_completeness_prompt(_rec())
        assert "completeness" in p.lower() or "0.75" in p

    def test_negative_mentions_0_75(self):
        p = negative_abstention_prompt(_rec(is_negative=True))
        assert "0.75" in p

    def test_comparison_three_outputs(self):
        p = comparison_prompt(_rec(category="comparison"))
        assert "both_branches_covered" in p
        assert "comparison_axis_covered" in p
        assert "faithfulness" in p

    def test_numeric_has_unit_correct(self):
        p = numeric_prompt(_rec(category="table_figure_caption"))
        assert "unit_correct" in p


class TestAllPromptVersions:
    def test_all_v3_or_v2(self):
        for name, version in PROMPT_VERSIONS.items():
            assert version in ("v2.0", "v3.0", "v3.1"), f"{name} bad version: {version}"

    def test_all_providers_have_builders(self):
        for name in PROMPT_VERSIONS:
            assert name in PROMPT_BUILDERS, f"{name} missing builder"
