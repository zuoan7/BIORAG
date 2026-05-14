"""Test BIORAG Eval v1 prompt builder constraints."""
import json
import pytest
from scripts.evaluation.biorag_eval.judge_prompts import (
    faithfulness_prompt, evidence_recall_prompt, answer_accuracy_prompt,
    negative_abstention_prompt, comparison_prompt, numeric_prompt,
    answer_relevance_prompt, PROMPT_BUILDERS, _format_support, _format_answer_key,
    MAX_SUPPORT_CHARS,
)
from scripts.evaluation.biorag_eval.schemas import make_eval_record


def _make_rec(**kw):
    d = {
        "sample_id": "t1", "question": "Q?", "answer": "A.",
        "selected_support": [{"support_id": "e1", "doc_id": "d1", "text": "Evidence text."}],
        "expected_behavior": "factual answer", "answer_key": "Key answer.",
        "reference": "Reference answer.", "is_negative": False,
        "citation_count": 1, "category": "factoid", "expected_route": "factoid",
    }
    d.update(kw)
    return make_eval_record(**d)


class TestSupportFormat:
    def test_string_output(self):
        s = _format_support([{"doc_id": "d1", "text": "Evidence."}])
        assert isinstance(s, str)

    def test_truncation(self):
        long_text = "X" * 1000
        s = _format_support([{"doc_id": "d1", "text": long_text}])
        assert len(s) < len(long_text) + 200

    def test_empty_support(self):
        s = _format_support([])
        assert "no support" in s.lower() or s == ""


class TestAnswerKeyFormat:
    def test_includes_key(self):
        rec = _make_rec(answer_key="Specific answer key text")
        formatted = _format_answer_key(rec)
        assert "Specific answer" in formatted

    def test_includes_behavior(self):
        rec = _make_rec(answer_key="", expected_behavior="Should cite doc_0001")
        formatted = _format_answer_key(rec)
        assert "cite" in formatted.lower() or "doc_0001" in formatted

    def test_fallback(self):
        rec = _make_rec(answer_key="", reference="", expected_behavior="")
        formatted = _format_answer_key(rec)
        assert "no reference" in formatted.lower()


class TestPromptLength:
    def test_faithfulness_under_limit(self):
        """Prompt should be reasonably sized (not excessively long)."""
        rec = _make_rec(answer="A" * 1200, selected_support=[
            {"doc_id": "d1", "text": "X" * 600} for _ in range(3)
        ])
        p = faithfulness_prompt(rec)
        assert len(p) < 8000  # well under typical token limits

    def test_all_prompts_under_limit(self):
        rec = _make_rec(answer="A" * 800)
        for name, builder in PROMPT_BUILDERS.items():
            p = builder(rec)
            assert len(p) < 10000, f"{name} prompt too long: {len(p)} chars"


class TestAnswerAccuracyNull:
    def test_no_reference_returns_null(self):
        rec = _make_rec(reference="", answer_key="")
        p = answer_accuracy_prompt(rec)
        data = json.loads(p)
        assert data["answer_accuracy"] is None

    def test_has_reference_has_accuracy_key(self):
        rec = _make_rec(reference="Some reference text.")
        p = answer_accuracy_prompt(rec)
        assert "answer_accuracy" in p.lower() or "1.0" in p


class TestNegativePrompt:
    def test_contains_abstention(self):
        rec = _make_rec(is_negative=True, answer="Unable to answer due to insufficient evidence.")
        p = negative_abstention_prompt(rec)
        assert "abstention" in p.lower() or "refuses" in p.lower() or "insufficient" in p.lower()


class TestComparisonPrompt:
    def test_contains_branches(self):
        rec = _make_rec(category="comparison", question="Compare A and B.")
        p = comparison_prompt(rec)
        assert "both_branches" in p or "comparison" in p.lower()


class TestNumericPrompt:
    def test_contains_unit_check(self):
        rec = _make_rec(category="table_figure_caption", question="What is the value?")
        p = numeric_prompt(rec)
        assert "unit_correct" in p or "numeric" in p.lower()
