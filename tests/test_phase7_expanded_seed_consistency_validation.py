import copy
from pathlib import Path

from scripts.evaluation import validate_expanded_table_seed_consistency as validator


def _make_inputs(tmp_path: Path, count: int = 15):
    review_pack = tmp_path / "review_pack"
    markdown_dir = review_pack / "markdown_cards"
    csv_dir = review_pack / "csv_tables"
    crop_dir = review_pack / "pdf_crops"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    seeds = []
    draft_rows = []
    frozen_rows = []
    index_rows = []
    pool_rows = []
    summary_rows = []
    for index in range(1, count + 1):
        candidate_id = f"phase7g_candidate_{index:03d}__doc_test__table_{index}"
        table_object_id = f"doc_test__table_{index}__phase7g_review_{index:03d}"
        doc_id = f"doc_{index:04d}"
        table_id = f"Table {index}"
        markdown_path = markdown_dir / f"{candidate_id}.md"
        csv_path = csv_dir / f"{candidate_id}.csv"
        crop_path = crop_dir / f"{candidate_id}.png"
        markdown_path.write_text(f"# {candidate_id}\nvalue (g/L)\n", encoding="utf-8")
        csv_path.write_text("row,col_1\nstrain,1.0\n", encoding="utf-8")
        crop_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        seed = {
            "seed_id": f"phase7g2_expanded_seed_{index:03d}__{candidate_id}",
            "candidate_id": candidate_id,
            "table_object_id": table_object_id,
            "doc_id": doc_id,
            "table_id": table_id,
            "caption": f"Table {index}. Fixture.",
            "page": str(index),
            "source_phase": "phase7g_2_bulk_binding_from_human_review",
            "seed_status": "confirmed_seed_with_warnings",
            "boundary_ok": "yes",
            "grid_ok": "yes",
            "key_values_ok": "yes",
            "unit_or_note_ok": "warning",
            "reference_ok": "not_applicable",
            "binding_review_mode": "global_bulk_no_obvious_issue",
            "binding_review_limitation": "no_per_cell_binding_review",
            "binding_notes": "fixture",
            "required_values_source": "phase7g_1_confirmed_seed_draft_human_core_labels",
            "markdown_path": str(markdown_path),
            "csv_path": str(csv_path),
            "pdf_crop_path": str(crop_path),
            "crop_status": "ok",
            "risk_tags": "unit_or_footnote_table",
            "source_span_granularity": "cell_level",
            "value_bboxes_available": "false",
            "cell_bboxes_available": "true",
            "auto_binding_fill": "true",
            "unit_or_note_signal_detected": "true",
            "reference_signal_detected": "false",
            "unit_or_note_signal_hits": "g_per_l",
            "reference_signal_hits": "none",
            "seed_warnings": "bulk_binding_no_per_cell_review;formal_benchmark_guardrail;unit_or_note_binding_warning;value_bboxes_not_available",
            "seed_notes": "fixture",
        }
        seeds.append(seed)

        common = {
            "candidate_id": candidate_id,
            "table_object_id": table_object_id,
            "doc_id": doc_id,
            "table_id": table_id,
            "caption": seed["caption"],
            "page": str(index),
            "markdown_path": str(markdown_path),
            "csv_path": str(csv_path),
            "pdf_crop_path": str(crop_path),
            "crop_status": "ok",
            "risk_tags": "unit_or_footnote_table",
        }
        draft_rows.append(
            {
                **common,
                "review_priority": "P0_quick_review",
                "suggested_decision": "likely_confirmed_candidate",
                "boundary_ok": "yes",
                "grid_ok": "yes",
                "key_values_ok": "yes",
                "unit_or_note_ok": "unchecked",
                "reference_ok": "unchecked",
                "seed_draft_status": "confirmed_seed_draft",
            }
        )
        frozen_rows.append(
            {
                **common,
                "review_decision": "accept_confirmed_seed_candidate",
                "boundary_ok": "yes",
                "grid_ok": "yes",
                "key_values_ok": "yes",
                "unit_or_note_ok": "unchecked",
                "reference_ok": "unchecked",
            }
        )
        index_rows.append(
            {
                **common,
                "source_span_granularity": "cell_level",
                "value_bboxes_available": "false",
                "cell_bboxes_available": "true",
            }
        )
        pool_rows.append(dict(common))
        summary_rows.append(
            {
                "seed_id": seed["seed_id"],
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "table_id": table_id,
                "seed_status": "confirmed_seed_with_warnings",
            }
        )

    return {
        "base_dir": tmp_path,
        "bulk_binding_policy": {
            "binding_review_mode": "global_bulk_no_obvious_issue",
            "binding_review_limitation": "no_per_cell_binding_review",
        },
        "expanded_seeds": copy.deepcopy(seeds),
        "expanded_seed_summary": summary_rows,
        "confirmed_seeds": seeds,
        "partial_carried_forward": [{"candidate_id": f"partial_{i}"} for i in range(10)],
        "reject_carried_forward": [{"candidate_id": f"reject_{i}"} for i in range(4)],
        "unreviewed_carried_forward": [{"candidate_id": f"unreviewed_{i}"} for i in range(11)],
        "review_pack_index": index_rows,
        "candidate_pool_scored": pool_rows,
        "frozen_review_labels": frozen_rows,
        "confirmed_seed_draft_rows": draft_rows,
    }


def _payload(tmp_path: Path, mutate=None):
    inputs = _make_inputs(tmp_path)
    if mutate:
        mutate(inputs)
    return validator.build_validation_results(inputs)


def _first_result(tmp_path: Path, mutate=None):
    return _payload(tmp_path, mutate)["results"][0]


def test_15_confirmed_seed_with_warnings_enter_formal_validation(tmp_path):
    payload = _payload(tmp_path)

    assert payload["formal_overall"]["formal_seed_count"] == 15
    assert len(payload["results"]) == 15
    assert {row["validation_subset"] for row in payload["results"]} == {"formal_confirmed_seed_with_warnings"}


def test_partial_carried_forward_does_not_enter_formal(tmp_path):
    payload = _payload(tmp_path)

    formal_ids = {row["candidate_id"] for row in payload["results"]}
    assert payload["appendix_counts"]["partial_carried_forward"] == 10
    assert not any(f"partial_{index}" in formal_ids for index in range(10))


def test_reject_carried_forward_does_not_enter_formal(tmp_path):
    payload = _payload(tmp_path)

    formal_ids = {row["candidate_id"] for row in payload["results"]}
    assert payload["appendix_counts"]["reject_carried_forward"] == 4
    assert not any(f"reject_{index}" in formal_ids for index in range(4))


def test_unreviewed_carried_forward_does_not_enter_formal(tmp_path):
    payload = _payload(tmp_path)

    formal_ids = {row["candidate_id"] for row in payload["results"]}
    assert payload["appendix_counts"]["unreviewed_carried_forward"] == 11
    assert not any(f"unreviewed_{index}" in formal_ids for index in range(11))


def test_seed_id_must_be_unique(tmp_path):
    def mutate(inputs):
        inputs["confirmed_seeds"][1]["seed_id"] = inputs["confirmed_seeds"][0]["seed_id"]

    result = _first_result(tmp_path, mutate)

    assert result["seed_identity_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_boundary_grid_key_values_must_be_yes(tmp_path):
    for field in ["boundary_ok", "grid_ok", "key_values_ok"]:
        def mutate(inputs, field=field):
            inputs["confirmed_seeds"][0][field] = "no"

        result = _first_result(tmp_path, mutate)

        assert result["human_label_consistency_check"] == "fail"
        assert result["overall_validation_status"] == "fail"


def test_unit_or_note_ok_must_not_be_unchecked(tmp_path):
    def mutate(inputs):
        inputs["confirmed_seeds"][0]["unit_or_note_ok"] = "unchecked"

    result = _first_result(tmp_path, mutate)

    assert result["binding_policy_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_reference_ok_must_not_be_unchecked(tmp_path):
    def mutate(inputs):
        inputs["confirmed_seeds"][0]["reference_ok"] = "unchecked"

    result = _first_result(tmp_path, mutate)

    assert result["binding_policy_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_binding_review_mode_must_match_bulk_policy(tmp_path):
    def mutate(inputs):
        inputs["confirmed_seeds"][0]["binding_review_mode"] = "per_cell_confirmed"

    result = _first_result(tmp_path, mutate)

    assert result["binding_policy_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_binding_review_limitation_must_match_bulk_policy(tmp_path):
    def mutate(inputs):
        inputs["confirmed_seeds"][0]["binding_review_limitation"] = "per_cell_binding_review"

    result = _first_result(tmp_path, mutate)

    assert result["binding_policy_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_csv_missing_cannot_pass(tmp_path):
    def mutate(inputs):
        Path(inputs["confirmed_seeds"][0]["csv_path"]).unlink()

    result = _first_result(tmp_path, mutate)

    assert result["csv_structure_sanity_check"] == "fail"
    assert result["artifact_availability_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_markdown_missing_is_warning_not_fail(tmp_path):
    def mutate(inputs):
        Path(inputs["confirmed_seeds"][0]["markdown_path"]).unlink()

    result = _first_result(tmp_path, mutate)

    assert result["markdown_card_check"] == "pass_with_warnings"
    assert result["artifact_availability_check"] == "pass_with_warnings"
    assert result["overall_validation_status"] == "pass_with_warnings"


def test_crop_missing_is_non_blocking_warning(tmp_path):
    def mutate(inputs):
        Path(inputs["confirmed_seeds"][0]["pdf_crop_path"]).unlink()

    result = _first_result(tmp_path, mutate)

    assert result["pdf_crop_check"] == "pass_with_warnings"
    assert result["artifact_availability_check"] == "pass_with_warnings"
    assert result["overall_validation_status"] == "pass_with_warnings"


def test_value_bboxes_available_false_is_not_failure(tmp_path):
    result = _first_result(tmp_path)

    assert result["value_bboxes_available"] == "false"
    assert result["bbox_provenance_check"] == "pass_with_warnings"
    assert result["overall_validation_status"] == "pass_with_warnings"


def test_source_span_granularity_must_not_be_value_level(tmp_path):
    def mutate(inputs):
        inputs["confirmed_seeds"][0]["source_span_granularity"] = "value_level"

    result = _first_result(tmp_path, mutate)

    assert result["source_span_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_production_ready_marker_is_forbidden(tmp_path):
    marker = "production" + "_ready"

    def mutate(inputs):
        inputs["confirmed_seeds"][0][marker] = True

    result = _first_result(tmp_path, mutate)

    assert result["formal_membership_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_official_benchmark_seed_marker_is_forbidden(tmp_path):
    marker = "official" + "_benchmark_seed"

    def mutate(inputs):
        inputs["confirmed_seeds"][0][marker] = True

    result = _first_result(tmp_path, mutate)

    assert result["formal_membership_check"] == "fail"
    assert result["overall_validation_status"] == "fail"


def test_validation_does_not_access_bm25_or_milvus(tmp_path, monkeypatch):
    inputs = _make_inputs(tmp_path)
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        lowered = str(self).lower()
        if "bm25" in lowered or "milvus" in lowered:
            raise AssertionError(f"forbidden path access: {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    payload = validator.build_validation_results(inputs)

    assert payload["formal_overall"]["formal_seed_count"] == 15
