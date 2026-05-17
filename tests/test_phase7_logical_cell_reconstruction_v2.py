import json
from functools import lru_cache

from scripts.extraction import reconstruct_logical_cells_v2 as recon


@lru_cache(maxsize=1)
def _built():
    inputs = recon.load_inputs()
    objects, rows, diagnostics, deltas = recon.build_v22_outputs(inputs)
    return objects, rows, diagnostics, deltas


def _rows_by_id():
    _, rows, _, _ = _built()
    return {row["table_object_id"]: row for row in rows}


def _objects_by_id():
    objects, _, _, _ = _built()
    return {obj["table_object_id"]: obj for obj in objects}


def test_v22_routing_covers_all_16_phase7d2_table_objects():
    objects, rows, diagnostics, deltas = _built()

    assert len(objects) == 16
    assert len(rows) == 16
    assert len(diagnostics) == 16
    assert len(deltas) == 16
    assert {obj["doc_id"] for obj in objects} == set(recon.SMOKE_DOC_IDS)
    assert {obj["table_object_id"] for obj in objects} == {row["table_object_id"] for row in rows}


def test_phase7d2_ready_candidates_do_not_regress():
    rows = _rows_by_id()

    ready_ids = {case_id for case_id, row in rows.items() if row["routing_status"] == "ready_for_gold_candidate"}
    assert recon.READY_IDS <= ready_ids
    for case_id in recon.READY_IDS:
        assert rows[case_id]["final_action"] == "keep_ready_candidate"


def test_doc_0598_table1_does_not_upgrade_ready():
    rows = _rows_by_id()
    row = rows[recon.ALIGNMENT_BLOCKED_ID]

    assert row["routing_status"] == "needs_pdfplumber_rule_fix"
    assert row["final_action"] == "keep_rule_fix"
    assert "alignment_not_ready_eligible" in row["remaining_blockers"]
    assert "source_span_not_cell_level_for_rule_fix" in row["remaining_blockers"]


def test_doc_0687_table2_upgrades_only_when_metric_reconstruction_is_complete():
    obj = _objects_by_id()[recon.TARGET_METRIC_ID]
    record = recon.metric_column_template(obj)

    assert recon.metric_reconstruction_upgrade_allowed(record) is True
    broken = dict(record)
    broken["missing_expected_cells"] = ["TMB3421:YE/S"]
    assert recon.metric_reconstruction_upgrade_allowed(broken) is False
    broken = dict(record)
    broken["unit_binding_status"] = "uncertain"
    assert recon.metric_reconstruction_upgrade_allowed(broken) is False
    broken = dict(record)
    broken["remaining_blockers"] = ["numeric_column_order_uncertain"]
    assert recon.metric_reconstruction_upgrade_allowed(broken) is False


def test_doc_0523_table1_upgrades_only_when_literal_reference_and_unit_are_bound():
    obj = _objects_by_id()[recon.TARGET_ROW_REFERENCE_ID]
    record = recon.row_reference_literal_template(obj)

    assert recon.row_reference_reconstruction_upgrade_allowed(record) is True
    without_nd = dict(record)
    without_nd["logical_cells"] = [
        cell for cell in record["logical_cells"] if not (cell["logical_column"] == "LNT_II" and cell["value_raw"] == "N.D.")
    ]
    assert recon.row_reference_reconstruction_upgrade_allowed(without_nd) is False
    broken = dict(record)
    broken["reference_binding_status"] = "uncertain"
    assert recon.row_reference_reconstruction_upgrade_allowed(broken) is False
    broken = dict(record)
    broken["unit_binding_status"] = "uncertain"
    assert recon.row_reference_reconstruction_upgrade_allowed(broken) is False


def test_metric_column_template_generates_required_logical_columns():
    obj = _objects_by_id()[recon.TARGET_METRIC_ID]
    record = recon.metric_column_template(obj)

    assert set(recon.METRIC_LOGICAL_COLUMNS) <= set(record["logical_columns"])
    assert record["logical_cells"]
    assert any(cell["row_key"] == "TMB3421" and cell["logical_column"] == "YE/S" and cell["value_raw"] == "0.35" for cell in record["logical_cells"])
    assert any(cell["row_key"] == "RWB217" and cell["logical_column"] == "YE/S" and cell["value_raw"] == "0.43" for cell in record["logical_cells"])


def test_row_reference_literal_template_generates_required_logical_columns():
    obj = _objects_by_id()[recon.TARGET_ROW_REFERENCE_ID]
    record = recon.row_reference_literal_template(obj)

    assert set(recon.ROW_REFERENCE_LOGICAL_COLUMNS) <= set(record["logical_columns"])
    assert record["logical_cells"]
    assert any(cell["logical_column"] == "LNT_II" and cell["value_raw"] == "N.D." for cell in record["logical_cells"])
    assert any(cell["logical_column"] == "unit" and cell["value_raw"] == "g/L" for cell in record["logical_cells"])
    assert any(cell["logical_column"] == "reference_or_source" and cell["value_raw"] == "this study" for cell in record["logical_cells"])


def test_missing_expected_cells_prevent_upgrade():
    obj = _objects_by_id()[recon.TARGET_METRIC_ID]
    record = recon.metric_column_template(obj)
    record["missing_expected_cells"] = ["RWB217:YE/S"]

    assert recon.metric_reconstruction_upgrade_allowed(record) is False


def test_remaining_blockers_prevent_upgrade():
    obj = _objects_by_id()[recon.TARGET_ROW_REFERENCE_ID]
    record = recon.row_reference_literal_template(obj)
    record["remaining_blockers"] = ["row_continuation_warning"]

    assert recon.row_reference_reconstruction_upgrade_allowed(record) is False


def test_grid_rejected_cases_do_not_enter_ready_or_usable_hybrid():
    rows = _rows_by_id()

    for case_id in recon.GRID_REJECTED_IDS:
        assert rows[case_id]["routing_status"] == "grid_rejected"
        assert rows[case_id]["final_action"] == "reject_pdfplumber_grid"
        assert rows[case_id]["usable_hybrid_candidate"] == "false"


def test_chunk_fallback_cases_use_chunk_fallback_action():
    rows = _rows_by_id()

    for case_id in recon.CHUNK_FALLBACK_IDS:
        assert rows[case_id]["routing_status"] == "chunk_fallback"
        assert rows[case_id]["final_action"] == "use_chunk_fallback"


def test_backlog_cases_keep_backlog_action():
    rows = _rows_by_id()

    for case_id in recon.BACKLOG_IDS:
        assert rows[case_id]["routing_status"] == "backlog"
        assert rows[case_id]["final_action"] == "keep_backlog"


def test_value_bboxes_stay_false():
    objects, rows, _, _ = _built()

    assert all(row["value_bboxes_available"] == "false" for row in rows)
    assert all(obj["value_bboxes_available"] is False for obj in objects)
    assert all(cell.get("value_bbox") is None for obj in objects for cell in obj.get("logical_cells") or [])


def test_source_span_granularity_never_value_level():
    objects, rows, _, _ = _built()

    assert all(row["source_span_granularity"] != "value_level" for row in rows)
    assert all(obj["source_span_granularity"] != "value_level" for obj in objects)
    assert all(span.get("granularity") != "value_level" for obj in objects for span in obj.get("source_spans") or [])
    assert all(cell.get("source_span_granularity") != "value_level" for obj in objects for cell in obj.get("logical_cells") or [])


def test_ready_for_gold_candidate_is_not_confirmed_gold():
    objects, rows, _, _ = _built()
    payload = json.dumps({"objects": objects, "rows": rows}, ensure_ascii=False)

    assert "confirmed_gold" not in payload
    assert all(row["routing_status"] != "confirmed_gold" for row in rows)


def test_production_ready_never_appears():
    objects, rows, _, _ = _built()
    payload = json.dumps({"objects": objects, "rows": rows}, ensure_ascii=False)

    assert "production_ready" not in payload
