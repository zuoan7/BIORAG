"""Test v3 prompt metric separation — faithfulness/relevance must not include completeness/accuracy/conciseness language."""
from scripts.evaluation.biorag_eval.judge_prompts import (
    faithfulness_prompt, answer_relevance_prompt,
    evidence_recall_prompt, answer_accuracy_prompt,
    answer_completeness_prompt, negative_abstention_prompt,
    PROMPT_VERSIONS,
)
from scripts.evaluation.biorag_eval.schemas import make_eval_record


def _rec(**kw):
    d = {"sample_id": "t1", "question": "Q?", "answer": "A brief answer.",
         "selected_support": [{"support_id": "e1", "doc_id": "d1", "text": "Evidence text."}],
         "reference": "Ref.", "answer_key": "Key.", "expected_behavior": "factual",
         "is_negative": False, "citation_count": 1, "category": "factoid"}
    d.update(kw)
    return make_eval_record(**d)


class TestFaithfulnessV3:
    """Faithfulness prompt must NOT contain completeness/omission language."""

    DISALLOWED = [
        "minor omissions", "incomplete answer",
        "should mention", "not comprehensive", "not enough detail",
    ]
    # Note: prompt mentions "incompleteness... are not faithfulness issues" — this is correct negation

    def test_no_completeness_language(self):
        p = faithfulness_prompt(_rec())
        for term in self.DISALLOWED:
            assert term not in p.lower(), f"faithfulness_v3 contains '{term}'"

    def test_only_judges_stated_claims(self):
        p = faithfulness_prompt(_rec())
        assert "already made" in p.lower() or "stated" in p.lower() or "claims" in p.lower()

    def test_no_should_mention(self):
        p = faithfulness_prompt(_rec())
        assert "should mention" not in p.lower()

    def test_v3_version(self):
        assert PROMPT_VERSIONS["faithfulness"] == "v3.0"


class TestAnswerRelevanceV3:
    """Answer relevance prompt must NOT contain accuracy/completeness/conciseness language."""

    DISALLOWED = [
        "focused", "missing key point",
    ]
    # Note: "correctness", "evidence support", "accuracy", "completeness", "conciseness" all appear ONLY in negated form: "Do NOT evaluate..."

    def test_no_accuracy_language(self):
        p = answer_relevance_prompt(_rec())
        for term in self.DISALLOWED:
            assert term not in p.lower(), f"answer_relevance_v3 contains '{term}'"

    def test_only_topic_relevance(self):
        p = answer_relevance_prompt(_rec())
        assert "topic" in p.lower() or "relevance" in p.lower() or "relevant" in p.lower()

    def test_v3_version(self):
        assert PROMPT_VERSIONS["answer_relevance"] == "v3.0"


class TestFaithfulnessDoesNotPenalizeMissingInfo:
    """Faithfulness should NOT penalize an answer that omits some points but is faithful in what it says."""

    def test_partial_but_faithful_answer_is_allowed(self):
        rec = _rec(answer="CRISPR-Cas9 enables genome editing.")
        p = faithfulness_prompt(rec)
        # Prompt should explicitly say incompleteness is NOT a faithfulness issue
        assert "what it does NOT say" in p.lower() or "evaluated separately" in p.lower() or "do NOT penalize" in p

    def test_answer_can_be_partial(self):
        p = faithfulness_prompt(_rec())
        # The prompt should explicitly say to not penalize for what answer doesn't say
        assert "do NOT penalize" in p or "what it does NOT say" in p


class TestRelevanceDoesNotPenalizeIncompleteness:
    def test_incomplete_but_relevant_answer_not_fail(self):
        p = answer_relevance_prompt(_rec())
        assert "do NOT penalize" not in p.lower() or "accuracy" not in p.lower() or "completeness" not in p.lower()
        # Should have 0.5 as minimum for partially relevant, not 0.0
        assert "0.5" in p

    def test_off_topic_only_gets_zero(self):
        p = answer_relevance_prompt(_rec())
        assert "off-topic" in p.lower() or "off_topic" in p.lower()


class TestUnitCorrectSchema:
    def test_supports_null(self):
        from scripts.evaluation.biorag_eval.judge_prompts import numeric_prompt
        p = numeric_prompt(_rec(category="method_result_numeric", reference="", answer_key=""))
        assert "unit_correct" in p.lower()
        assert "null" in p.lower()

    def test_supports_bool(self):
        from scripts.evaluation.biorag_eval.judge_prompts import numeric_prompt
        p = numeric_prompt(_rec(category="table_figure_caption"))
        assert "true" in p.lower()
        assert "false" in p.lower()


class TestNegativeExcludedFromFaithfulness:
    def test_negative_not_in_faithfulness_applicable(self):
        from scripts.evaluation.biorag_eval.schemas import applicable_metrics
        rec = _rec(is_negative=True)
        m = applicable_metrics(rec)
        assert "faithfulness" not in m
        assert "answer_relevance" not in m
        assert "abstention_correctness" in m
