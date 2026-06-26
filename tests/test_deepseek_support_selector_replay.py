from __future__ import annotations

import json

import pytest

from scripts.evaluation.replay_generation_deepseek_support_selector import (
    SelectionValidationError,
    build_model_candidates,
    build_prompt_payload,
    build_selector_messages,
    call_order_key,
    classify_append_outcome,
    classify_outcome,
    classify_sample,
    compute_append_metrics,
    compute_model_metrics,
    prepare_samples,
    validate_model_selection,
)


def test_classify_sample_splits_target_visible_target_invisible_and_control() -> None:
    target_visible = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="p1",
        candidates=[_candidate("E1", "p1")],
    )
    target_invisible = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="p1",
        candidates=[_candidate("E1", "p2")],
    )
    control = _row(
        parent_hit=True,
        support_hit=True,
        gold_parent="p1",
        candidates=[_candidate("E1", "p1")],
    )

    assert classify_sample(target_visible) == "target_visible"
    assert classify_sample(target_invisible) == "target_invisible"
    assert classify_sample(control) == "control"


def test_call_order_prioritizes_target_visible_before_controls() -> None:
    samples = [
        {"group": "control", "sample_id": "b"},
        {"group": "target_visible", "sample_id": "c"},
        {"group": "target_visible", "sample_id": "a"},
    ]

    assert [item["sample_id"] for item in sorted(samples, key=call_order_key)] == [
        "a",
        "c",
        "b",
    ]


def test_append_mode_excludes_baseline_support_from_selectable_candidates() -> None:
    row = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="p2",
        candidates=[_candidate("E1", "p1"), _candidate("E2", "p2")],
    )

    samples = prepare_samples(
        rows=[row],
        parent_index={},
        child_index={},
        include_targets=True,
        include_controls=False,
        sample_ids=set(),
        selection_mode="append",
        prompt_version="id_only_v2",
        max_extra_support=1,
    )

    assert [item["evidence_id"] for item in samples[0]["baseline_support"]] == ["E1"]
    assert [item["evidence_id"] for item in samples[0]["selectable_candidates"]] == ["E2"]


def test_prompt_payload_does_not_include_gold_expected_or_baseline_citation() -> None:
    row = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="secret_gold_parent",
        candidates=[_candidate("E1", "p1")],
    )
    row["expected_answer"] = "secret expected answer"
    row["expected_doc_ids"] = ["secret_doc"]
    row["citation_chunk_ids"] = ["secret_baseline_citation"]

    candidates = build_model_candidates(
        row=row,
        parent_index={"p1": {"text": "candidate parent text"}},
        child_index={},
    )
    payload = build_prompt_payload(
        row,
        candidates,
        baseline_support=[],
        max_select=1,
        selection_mode="append",
        prompt_version="id_only_v2",
    )
    messages = build_selector_messages(payload)
    prompt_text = json.dumps(messages, ensure_ascii=False)

    assert "secret_gold_parent" not in prompt_text
    assert "secret expected answer" not in prompt_text
    assert "secret_doc" not in prompt_text
    assert "secret_baseline_citation" not in prompt_text
    assert "candidate parent text" in prompt_text


def test_validate_model_selection_rejects_unknown_empty_and_oversized_ids() -> None:
    assert validate_model_selection(
        {"selected_evidence_ids": ["E1", "E1"], "reason": "ok", "confidence": "0.8"},
        allowed_ids={"E1", "E2"},
        max_select=1,
    ) == {"selected_evidence_ids": ["E1"], "reason": "ok", "confidence": 0.8}

    with pytest.raises(SelectionValidationError):
        validate_model_selection(
            {"selected_evidence_ids": ["E3"]},
            allowed_ids={"E1"},
            max_select=1,
        )
    with pytest.raises(SelectionValidationError):
        validate_model_selection({"selected_evidence_ids": []}, allowed_ids={"E1"}, max_select=1)
    assert validate_model_selection(
        {"selected_evidence_ids": []},
        allowed_ids={"E1"},
        max_select=1,
        allow_empty=True,
    )["selected_evidence_ids"] == []
    with pytest.raises(SelectionValidationError):
        validate_model_selection(
            {"selected_evidence_ids": ["E1", "E2"]},
            allowed_ids={"E1", "E2"},
            max_select=1,
        )


def test_compute_model_metrics_and_outcomes_for_rescue_and_regression() -> None:
    target = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="p1",
        gold_child="p1::child001",
        candidates=[
            _candidate("E1", "p1", matched_children=["p1::child001"]),
            _candidate("E2", "p2"),
        ],
    )
    rescued_metrics = compute_model_metrics(
        row=target,
        candidates=build_model_candidates(row=target, parent_index={}, child_index={}),
        selected_evidence_ids=["E1"],
    )
    still_miss_metrics = compute_model_metrics(
        row=target,
        candidates=build_model_candidates(row=target, parent_index={}, child_index={}),
        selected_evidence_ids=["E2"],
    )

    assert rescued_metrics["model_support_parent_hit"] is True
    assert rescued_metrics["model_support_child_evidence_hit"] is True
    assert classify_outcome("target_visible", rescued_metrics) == "rescued"
    assert classify_outcome("target_visible", still_miss_metrics) == "still_miss"
    assert classify_outcome("control", still_miss_metrics) == "regressed"


def test_compute_append_metrics_uses_baseline_union_extra_support() -> None:
    row = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="p2",
        gold_child="p2::child001",
        candidates=[
            _candidate("E1", "p1", matched_children=["p1::child001"]),
            _candidate("E2", "p2", matched_children=["p2::child001"]),
        ],
    )
    candidates = build_model_candidates(row=row, parent_index={}, child_index={})

    metrics = compute_append_metrics(
        row=row,
        candidates=candidates,
        baseline_support=[candidates[0]],
        selected_extra_evidence_ids=["E2"],
    )

    assert metrics["baseline_support_parent_hit_recomputed"] is False
    assert metrics["hybrid_support_parent_hit"] is True
    assert metrics["hybrid_support_child_evidence_hit"] is True
    assert metrics["rescued_by_append"] is True
    assert metrics["extra_support_count"] == 1
    assert classify_append_outcome("target_visible", metrics) == "rescued_by_append"


def test_max_extra_support_caps_selected_ids() -> None:
    assert validate_model_selection(
        {"selected_evidence_ids": ["E1"]},
        allowed_ids={"E1", "E2"},
        max_select=1,
        allow_empty=True,
    )["selected_evidence_ids"] == ["E1"]

    with pytest.raises(SelectionValidationError):
        validate_model_selection(
            {"selected_evidence_ids": ["E1", "E2"]},
            allowed_ids={"E1", "E2"},
            max_select=1,
            allow_empty=True,
        )

    assert validate_model_selection(
        {"selected_evidence_ids": ["E1", "E2"]},
        allowed_ids={"E1", "E2"},
        max_select=2,
        allow_empty=True,
    )["selected_evidence_ids"] == ["E1", "E2"]


def test_build_model_candidates_prefers_child_text_then_parent_excerpt() -> None:
    row = _row(
        parent_hit=True,
        support_hit=False,
        gold_parent="p1",
        candidates=[_candidate("E1", "p1", matched_children=["p1::child001"])],
    )

    candidates = build_model_candidates(
        row=row,
        parent_index={"p1": {"text": "parent evidence text"}},
        child_index={"p1::child001": {"text": "child evidence text"}},
    )

    assert candidates[0]["evidence_text"].startswith("matched child p1::child001")
    assert "child evidence text" in candidates[0]["evidence_text"]
    assert "parent evidence text" in candidates[0]["evidence_text"]


def _row(
    *,
    parent_hit: bool,
    support_hit: bool,
    gold_parent: str,
    candidates: list[dict],
    gold_child: str | None = None,
) -> dict:
    gold_chunks = [gold_child or gold_parent]
    return {
        "sample_id": "s1",
        "question": "question",
        "category": "table_content",
        "actual_route": "QueryIntent.FACTOID",
        "expected_route": "factoid",
        "gold_chunk_ids": gold_chunks,
        "gold_parent_chunk_ids": [gold_parent],
        "rule_metrics": {
            "parent_chunk_hit_at10": parent_hit,
            "support_parent_chunk_hit": support_hit,
            "route_match": True,
        },
        "debug_digest": {
            "generation_v2": {
                "candidates": candidates,
                "support_pack": candidates[:1],
            }
        },
    }


def _candidate(
    evidence_id: str,
    parent_id: str,
    *,
    matched_children: list[str] | None = None,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "chunk_id": parent_id,
        "parent_chunk_id": parent_id,
        "doc_id": parent_id.split("_sec", 1)[0],
        "section": "Results",
        "rerank_rank": 1,
        "scores": {"rerank_score": 1.0},
        "features": {"has_numeric": True},
        "matched_child_chunk_ids": matched_children or [],
        "generation_evidence_role": "parent_text",
        "text_preview": f"preview for {parent_id}",
    }
