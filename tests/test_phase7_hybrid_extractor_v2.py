import json
from functools import lru_cache

from scripts.extraction import run_hybrid_table_extractor_v2 as v2


@lru_cache(maxsize=1)
def _built():
    inputs = v2.load_phase_inputs()
    objects, rows = v2.build_table_objects_v2(inputs)
    return objects, rows


def _rows_by_id():
    _, rows = _built()
    return {row["table_object_id"]: row for row in rows}


def _objects_by_id():
    objects, _ = _built()
    return {obj["table_object_id"]: obj for obj in objects}


def test_v2_routing_covers_all_16_phase7c2_hybrid_cases():
    objects, rows = _built()

    assert len(objects) == 16
    assert len(rows) == 16
    assert {obj["doc_id"] for obj in objects} == set(v2.SMOKE_DOC_IDS)
    assert {obj["table_object_id"] for obj in objects} == {row["table_object_id"] for row in rows}


def test_phase7c4_ready_candidates_stay_in_ready_candidate_pool():
    rows = _rows_by_id()

    assert {case_id for case_id, row in rows.items() if row["routing_status"] == "ready_for_gold_candidate"} == v2.READY_IDS
    for case_id in v2.READY_IDS:
        assert rows[case_id]["final_action"] == "keep_ready_candidate"


def test_phase7c4_rule_fix_cases_are_not_marked_ready():
    rows = _rows_by_id()

    for case_id in v2.RULE_FIX_IDS:
        assert rows[case_id]["routing_status"] == "needs_pdfplumber_rule_fix"
        assert rows[case_id]["final_action"] == "keep_rule_fix"
        assert rows[case_id]["routing_status"] != "ready_for_gold_candidate"


def test_grid_rejected_cases_are_not_ready_or_usable_hybrid():
    rows = _rows_by_id()
    grid_ids = {
        "doc_0322__table_1__phase7c2_hybrid_01",
        "doc_0158__table_2__phase7c2_hybrid_01",
        "doc_0598__table_2__phase7c2_hybrid_02",
        "doc_0452__table_1__phase7c2_hybrid_01",
        "doc_0687__table_1__phase7c2_hybrid_01",
    }

    for case_id in grid_ids:
        assert rows[case_id]["routing_status"] == "grid_rejected"
        assert rows[case_id]["final_action"] == "reject_pdfplumber_grid"
        assert rows[case_id]["usable_hybrid_candidate"] == "false"


def test_chunk_fallback_cases_use_chunk_fallback_action():
    rows = _rows_by_id()
    fallback_ids = {
        "doc_0158__table_3__phase7c2_hybrid_02",
        "doc_0468__table_3__phase7c2_hybrid_02",
        "doc_0522__table_1__phase7c2_hybrid_01",
    }

    assert {case_id for case_id, row in rows.items() if row["routing_status"] == "chunk_fallback"} == fallback_ids
    assert all(rows[case_id]["final_action"] == "use_chunk_fallback" for case_id in fallback_ids)


def test_backlog_cases_keep_backlog_action():
    rows = _rows_by_id()
    backlog_ids = {
        "doc_0458__table_1__phase7c2_hybrid_01",
        "doc_0458__table_2__phase7c2_hybrid_02",
        "doc_0458__table_3__phase7c2_hybrid_03",
    }

    assert {case_id for case_id, row in rows.items() if row["routing_status"] == "backlog"} == backlog_ids
    assert all(rows[case_id]["final_action"] == "keep_backlog" for case_id in backlog_ids)


def test_page_only_match_cannot_route_ready_even_if_binding_fixture_is_ready():
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
    assert "page_only_match_not_high_confidence" in route["routing_blockers"]


def test_conflict_and_multiple_pdf_tables_cannot_route_ready():
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
        assert route["routing_status"] == "manual_review_required"
        assert route["final_action"] == "manual_review_required"


def test_value_bboxes_stay_false_and_source_span_never_value_level():
    objects, rows = _built()

    assert all(row["value_bboxes_available"] == "false" for row in rows)
    assert all(obj["value_bboxes_available"] is False for obj in objects)
    assert all(row["source_span_granularity"] != "value_level" for row in rows)
    assert all(obj["source_span_granularity"] != "value_level" for obj in objects)
    assert all(
        span.get("granularity") != "value_level"
        for obj in objects
        for span in obj.get("source_spans") or []
    )


def test_ready_for_gold_candidate_is_not_confirmed_gold():
    objects = _objects_by_id()

    for case_id in v2.READY_IDS:
        obj = objects[case_id]
        assert obj["routing_status"] == "ready_for_gold_candidate"
        assert "confirmed_gold" not in json.dumps(obj, ensure_ascii=False)


def test_production_ready_never_appears_in_v2_outputs():
    objects, rows = _built()
    payload = json.dumps({"objects": objects, "rows": rows}, ensure_ascii=False)

    assert "production_ready" not in payload
