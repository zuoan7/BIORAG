import json
from functools import lru_cache

from scripts.extraction import align_chunk_pdfplumber_tables as align
from scripts.extraction import apply_hybrid_rule_fixes_v2 as rulefix
from scripts.extraction import run_hybrid_table_extractor_v2 as v2


@lru_cache(maxsize=1)
def _built():
    inputs = rulefix.load_inputs()
    objects, rows, diagnostics, deltas = rulefix.build_v21_outputs(inputs)
    return objects, rows, diagnostics, deltas


def _rows_by_id():
    _, rows, _, _ = _built()
    return {row["table_object_id"]: row for row in rows}


def _objects_by_id():
    objects, _, _, _ = _built()
    return {obj["table_object_id"]: obj for obj in objects}


def _raw_table(rows):
    return {
        "pdfplumber_table_id": "pdf_fixture",
        "row_count": len(rows),
        "column_count": max(len(row) for row in rows),
        "empty_cell_ratio": 0.0,
        "rows": rows,
    }


def test_v21_routing_covers_all_16_phase7d_table_objects():
    objects, rows, _, deltas = _built()

    assert len(objects) == 16
    assert len(rows) == 16
    assert len(deltas) == 16
    assert {obj["doc_id"] for obj in objects} == set(rulefix.SMOKE_DOC_IDS)
    assert {obj["table_object_id"] for obj in objects} == {row["table_object_id"] for row in rows}


def test_phase7d_ready_candidates_do_not_regress():
    rows = _rows_by_id()

    assert {case_id for case_id, row in rows.items() if row["routing_status"] == "ready_for_gold_candidate"} == rulefix.READY_IDS
    for case_id in rulefix.READY_IDS:
        assert rows[case_id]["final_action"] == "keep_ready_candidate"


def test_phase7d_rule_fix_cases_are_not_marked_ready():
    rows = _rows_by_id()

    for case_id in rulefix.RULE_FIX_IDS:
        assert rows[case_id]["routing_status"] == "needs_pdfplumber_rule_fix"
        assert rows[case_id]["final_action"] == "keep_rule_fix"
        assert rows[case_id]["candidate_upgrade_eligible"] == "false"


def test_rule_fix_case_can_upgrade_only_when_all_blockers_are_solved():
    row = {
        "routing_status": "needs_pdfplumber_rule_fix",
        "alignment_status": "matched",
        "alignment_confidence": "high",
    }
    statuses = {
        "unit_binding_status": "pass_with_warnings",
        "footnote_binding_status": "not_applicable",
        "reference_binding_status": "pass_with_warnings",
        "literal_preservation_status": "pass",
    }

    assert rulefix.candidate_upgrade_eligible(row, [], statuses) is True
    assert rulefix.candidate_upgrade_eligible(row, ["split_cell_warning"], statuses) is False


def test_split_cell_warning_can_be_triggered():
    raw = _raw_table([["strain", "Y", "E", "/S"], ["TMB3421", "0", ".35", "0.20"]])

    detected, evidence = rulefix.detect_split_cells(raw)

    assert detected is True
    assert evidence


def test_merged_cell_warning_can_be_triggered():
    raw = _raw_table(
        [
            ["Primername Primersequence(5=to3=)", "Location"],
            ["mvaEF CGGTAAGGATCCAGGAGAAATTAACTATGAAATTTTACGAG", "Forward primer"],
        ]
    )

    detected, evidence = rulefix.detect_merged_cells(raw)

    assert detected is True
    assert evidence


def test_row_continuation_warning_can_be_triggered():
    raw = _raw_table([["TMB3421", "0.35", "0.20"], ["", "", "Runquist"], ["", "", "Hahn-Hagerdal"]])

    detected, evidence = rulefix.detect_row_continuation(raw)

    assert detected is True
    assert evidence


def test_metric_level_cell_gap_can_be_triggered():
    raw = _raw_table(
        [
            ["strain", "Y", "E", "/S", "qethanol", "q", "xylose", "qarabinose"],
            ["TMB3421", "0", ".35", "0.20", "", "0.57", "", ""],
        ]
    )

    checks = rulefix.detect_metric_checks(raw, split_detected=True, merged_detected=False)

    assert checks["metric_level_cell_gap"] is True


def test_missing_metric_cell_warning_can_be_triggered():
    raw = _raw_table(
        [
            ["strain", "Y", "E", "/S", "qethanol", "q", "xylose", "qarabinose"],
            ["RWB217", "0", ".43", "0.46", "", "1.06", "", ""],
        ]
    )

    checks = rulefix.detect_metric_checks(raw, split_detected=True, merged_detected=False)

    assert checks["missing_metric_cell_warning"] is True


def test_grid_rejected_cases_do_not_enter_ready_or_usable_hybrid():
    rows = _rows_by_id()

    for case_id in rulefix.GRID_REJECTED_IDS:
        assert rows[case_id]["routing_status"] == "grid_rejected"
        assert rows[case_id]["final_action"] == "reject_pdfplumber_grid"
        assert rows[case_id]["usable_hybrid_candidate"] == "false"


def test_chunk_fallback_cases_use_chunk_fallback_action():
    rows = _rows_by_id()

    for case_id in rulefix.CHUNK_FALLBACK_IDS:
        assert rows[case_id]["routing_status"] == "chunk_fallback"
        assert rows[case_id]["final_action"] == "use_chunk_fallback"


def test_backlog_cases_keep_backlog_action():
    rows = _rows_by_id()

    for case_id in rulefix.BACKLOG_IDS:
        assert rows[case_id]["routing_status"] == "backlog"
        assert rows[case_id]["final_action"] == "keep_backlog"


def test_page_only_match_does_not_allow_ready_confidence():
    assert align.alignment_allows_ready_candidate("page_only_match", "high") is False
    route = v2.compute_route(
        {"source_review_decision": v2.SOURCE_DECISION_KEEP},
        {"final_binding_action": v2.READY_BINDING_ACTION},
        {
            "alignment_status": "page_only_match",
            "alignment_confidence": "high",
            "layout_quality_status": "usable",
        },
    )
    assert route["routing_status"] == "manual_review_required"


def test_conflict_and_multiple_pdf_tables_do_not_route_ready():
    for alignment_status in ["conflict", "multiple_pdf_tables"]:
        route = v2.compute_route(
            {"source_review_decision": v2.SOURCE_DECISION_KEEP},
            {"final_binding_action": v2.READY_BINDING_ACTION},
            {
                "alignment_status": alignment_status,
                "alignment_confidence": "high",
                "layout_quality_status": "usable",
            },
        )
        assert route["routing_status"] != "ready_for_gold_candidate"


def test_value_bboxes_stay_false():
    objects, rows, _, _ = _built()

    assert all(row["value_bboxes_available"] == "false" for row in rows)
    assert all(obj["value_bboxes_available"] is False for obj in objects)


def test_source_span_granularity_never_value_level():
    objects, rows, _, _ = _built()

    assert all(row["source_span_granularity"] != "value_level" for row in rows)
    assert all(obj["source_span_granularity"] != "value_level" for obj in objects)
    assert all(
        span.get("granularity") != "value_level"
        for obj in objects
        for span in obj.get("source_spans") or []
    )


def test_ready_for_gold_candidate_is_not_confirmed_gold():
    objects = _objects_by_id()

    for case_id in rulefix.READY_IDS:
        obj = objects[case_id]
        assert obj["routing_status"] == "ready_for_gold_candidate"
        assert "confirmed_gold" not in json.dumps(obj, ensure_ascii=False)


def test_production_ready_never_appears():
    objects, rows, _, _ = _built()
    payload = json.dumps({"objects": objects, "rows": rows}, ensure_ascii=False)

    assert "production_ready" not in payload
