from __future__ import annotations

from scripts.evaluation.run_v3_baseline_b0_b1 import (
    apply_run_level_overrides,
    build_concurrency_schedule,
    compute_rule_metrics,
    judge_row_with_retry,
    resolve_answer_model_config,
    run_judges,
    summarize_variant,
)
from src.synbio_rag.domain.config import Settings


def test_matched_child_metrics_do_not_depend_on_exact_child_in_top10() -> None:
    child_id = "doc_a_sec01_chunk02::child003"
    parent_id = "doc_a_sec01_chunk02"
    metrics = compute_rule_metrics(
        sample={
            "expected_doc_ids": ["doc_a"],
            "expected_route": "factoid",
            "answer_rubric": {"source_trace": {"chunk_ids": [child_id]}},
        },
        retrieved_doc_ids=["doc_a"],
        retrieved_chunk_ids=[parent_id],
        retrieval_matched_child_chunk_ids=[child_id],
        final_matched_child_chunk_ids=[child_id],
        support_chunk_ids=[parent_id],
        support_matched_child_chunk_ids=[child_id],
        citations=[{"doc_id": "doc_a", "chunk_id": parent_id}],
        citation_matched_child_chunk_ids=[child_id],
        inferred_matched_child_chunk_ids=[child_id],
        actual_route="factoid",
    )

    assert metrics["exact_gold_chunk_hit_at10"] is False
    assert metrics["parent_chunk_hit_at10"] is True
    assert metrics["child_gold_sample"] is True
    assert metrics["parent_hit_child_gold_sample"] is True
    assert metrics["matched_child_hit_at_retrieval"] is True
    assert metrics["matched_child_hit_at_final"] is True
    assert metrics["matched_child_hit_at_support"] is True
    assert metrics["matched_child_hit_at_citation"] is True
    assert metrics["parent_hit_child_gold_matched_child_hit_at_retrieval"] is True


def test_parent_gold_samples_do_not_enter_matched_child_denominators() -> None:
    parent_id = "doc_a_sec01_chunk02"
    metrics = compute_rule_metrics(
        sample={
            "expected_doc_ids": ["doc_a"],
            "expected_route": "factoid",
            "answer_rubric": {"source_trace": {"chunk_ids": [parent_id]}},
        },
        retrieved_doc_ids=["doc_a"],
        retrieved_chunk_ids=[parent_id],
        retrieval_matched_child_chunk_ids=[],
        final_matched_child_chunk_ids=[],
        support_chunk_ids=[parent_id],
        support_matched_child_chunk_ids=[],
        citations=[{"doc_id": "doc_a", "chunk_id": parent_id}],
        citation_matched_child_chunk_ids=[],
        inferred_matched_child_chunk_ids=[],
        actual_route="factoid",
    )

    assert metrics["exact_gold_chunk_hit_at10"] is True
    assert metrics["child_gold_sample"] is False
    assert metrics["parent_gold_sample"] is True
    assert metrics["matched_child_hit_at_retrieval"] is None
    assert metrics["matched_child_hit_at_support"] is None


def test_deepseek_answer_model_overrides_generation_and_rewrite(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_URL", "https://deepseek.example/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-test")

    answer_model = resolve_answer_model_config("deepseek")
    settings = Settings()
    apply_run_level_overrides(
        settings,
        answer_model=answer_model,
        rewrite_mode="enabled",
    )

    assert settings.generation.v2_use_qwen_synthesis is True
    assert settings.llm.provider == "deepseek"
    assert settings.llm.api_base == "https://deepseek.example/v1"
    assert settings.llm.model_name == "deepseek-test"
    assert settings.query_rewrite.mode == "enabled"
    assert settings.query_rewrite.model == "deepseek-test"


def test_judge_correctness_uses_answer_for_judge() -> None:
    judge = _CapturingJudge()
    row = {
        "sample_id": "sample_1",
        "question": "question",
        "expected_answer": "expected",
        "answer_rubric": {},
        "system_answer": "structured fallback answer",
        "answer_for_judge": "model generated answer",
        "answer_for_judge_source": "model_generated",
        "answer_generation": {"used_model": True},
        "citations": [],
        "expected_route": "factoid",
        "category": "factoid",
    }

    result = judge_row_with_retry(judge, row)

    assert result["answer_correctness"]["correctness_pass"] is True
    assert judge.payload["system_answer"] == "model generated answer"
    assert judge.payload["answer_for_judge_source"] == "model_generated"


def test_summarize_variant_reports_model_answer_usage() -> None:
    row = {
        "sample_id": "sample_1",
        "category": "factoid",
        "difficulty": "easy",
        "expected_route": "factoid",
        "latency_seconds": 1.0,
        "rule_metrics": {
            "doc_hit_at10": True,
            "doc_recall_at10": 1.0,
            "parent_chunk_hit_at10": True,
            "support_parent_chunk_hit": True,
            "citation_parent_chunk_hit": True,
        },
        "answer_generation": {
            "enabled": True,
            "attempted": True,
            "used_model": True,
            "fallback_used": False,
            "answer_source": "model_generated",
            "fallback_reason": "",
        },
        "judge": {
            "answer_correctness": {
                "correctness_score": 1.0,
                "correctness_pass": True,
                "critical_error": False,
            },
            "faithfulness": {"faithfulness_score": 1.0, "faithfulness_pass": True},
            "citation_accuracy": {
                "citation_support_score": 1.0,
                "citation_pass": True,
            },
        },
    }

    summary = summarize_variant([row])

    assert summary["answer_generation_metrics"]["enabled_rate"] == 1.0
    assert summary["answer_generation_metrics"]["used_model_rate"] == 1.0
    assert summary["answer_generation_metrics"]["answer_source_counts"] == {
        "model_generated": 1
    }


def test_judge_concurrency_schedule_drops_from_10_to_5_to_2() -> None:
    assert build_concurrency_schedule(10, "5,2") == [10, 5, 2]
    assert build_concurrency_schedule(5, "5,2") == [5, 2]
    assert build_concurrency_schedule(2, "5,2") == [2]


def test_run_judges_retries_failed_rows_at_lower_concurrency(tmp_path) -> None:
    row = {
        "sample_id": "sample_1",
        "question": "question",
        "expected_answer": "expected",
        "answer_rubric": {},
        "system_answer": "answer",
        "answer_for_judge": "answer",
        "answer_for_judge_source": "model_generated",
        "answer_generation": {"used_model": True},
        "citations": [],
        "expected_route": "factoid",
        "category": "factoid",
    }
    judge = _FailsFirstPassJudge()

    rows = run_judges(
        rows=[row],
        judge=judge,
        concurrency_schedule=[10, 5, 2],
        output_path=tmp_path / "partial.jsonl",
    )

    assert "judge_error" not in rows[0]["judge"]
    assert rows[0]["judge"]["_judge_concurrency"] == 5
    assert judge.calls == 4


class _CapturingJudge:
    api_key = "test-key"

    def complete_json(self, payload, *, purpose):
        self.payload = payload
        return {
            "answer_correctness": {
                "correctness_score": 1.0,
                "correctness_pass": True,
                "critical_error": False,
                "must_include_judgments": [],
                "reject_if_triggered": [],
            },
            "faithfulness": {
                "faithfulness_score": 1.0,
                "faithfulness_pass": True,
                "claims": [],
            },
            "citation_accuracy": {
                "citation_support_score": 1.0,
                "citation_pass": True,
                "claim_citation_judgments": [],
            },
        }


class _FailsFirstPassJudge(_CapturingJudge):
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, payload, *, purpose):
        self.calls += 1
        if self.calls <= 3:
            raise RuntimeError("temporary rate limit")
        return super().complete_json(payload, purpose=purpose)
