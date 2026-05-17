from scripts.extraction import apply_hybrid_source_review_decisions as gate


def _source_rows():
    return gate.load_csv(gate.DEFAULT_SOURCE_REVIEW_CSV)


def _fake_objects(rows):
    objects = []
    for row in rows:
        objects.append(
            {
                "table_object_id": row["hybrid_table_object_id"],
                "doc_id": row["doc_id"],
                "table_id": row["table_id"],
                "hybrid_metadata": {
                    "original_chunk_table_object_id": row["hybrid_table_object_id"].replace(
                        "__phase7c2_hybrid", "__phase7b2"
                    ),
                    "pdfplumber_table_id": "pdf_fixture",
                    "cell_bboxes_available": True,
                    "value_bboxes_available": False,
                    "source_span_granularity": "cell_level",
                },
                "rows": [{"row_id": "r1", "row_text": "fixture"}],
                "columns": [{"column_id": "c1", "header": "fixture"}],
                "cells": [{"cell_id": "cell1", "row_id": "r1", "column_id": "c1", "value_raw": "fixture"}],
                "warnings": [],
                "notes": [],
                "value_bboxes_available": False,
            }
        )
    return objects


def _fake_validation_rows(rows):
    validation_rows = []
    for row in rows:
        validation_rows.append(
            {
                "hybrid_table_object_id": row["hybrid_table_object_id"],
                "original_chunk_table_object_id": "chunk_fixture",
                "pdfplumber_table_id": "pdf_fixture",
                "doc_id": row["doc_id"],
                "table_id": row["table_id"],
                "alignment_status": "matched",
                "alignment_confidence": "high",
                "layout_quality_status": "usable",
                "extraction_method": "hybrid_pdfplumber_chunk",
                "cell_bboxes_available": "true",
                "value_bboxes_available": "false",
                "source_span_granularity": "cell_level",
                "hybrid_validation_status": "partial",
                "primary_failure_stage": "binding",
                "manual_review_reason": "fixture",
                "recommended_next_action": "manual_review_binding",
                "blocking_warnings": "none",
                "nonblocking_warnings": "value_bbox_absent_limitation",
                "notes": "fixture",
            }
        )
    return validation_rows


def _decisions_and_gated():
    source_rows = _source_rows()
    objects = _fake_objects(source_rows)
    validation_rows = _fake_validation_rows(source_rows)
    decisions = gate.build_case_decisions(source_rows, objects, validation_rows)
    gated_objects, gated_rows = gate.build_gated_outputs(decisions, objects, validation_rows)
    return decisions, gated_objects, gated_rows


def test_source_review_cases_cover_all_16():
    decisions, _, _ = _decisions_and_gated()

    assert len(decisions) == 16
    assert {row["doc_id"] for row in decisions} <= set(gate.SMOKE_DOC_IDS)


def test_keep_candidates_go_to_binding_review_queue():
    decisions, _, gated_rows = _decisions_and_gated()
    keep = [row for row in decisions if row["source_review_category"] == "keep_hybrid_candidate_needs_binding_review"]

    assert len(keep) == 5
    assert {row["hybrid_table_object_id"] for row in keep} == gate.EXPECTED_BINDING_REVIEW_IDS
    assert all(row["final_case_action"] == "manual_review_binding" for row in keep)
    assert sum(1 for row in gated_rows if row["final_case_action"] == "manual_review_binding") == 5


def test_alignment_confirmed_grid_rejected_sets_grid_rejected():
    decisions, _, gated_rows = _decisions_and_gated()
    grid = [row for row in decisions if row["source_review_category"] == "alignment_confirmed_grid_rejected"]

    assert len(grid) == 5
    assert all(row["grid_quality_status"] == "grid_rejected" for row in grid)
    assert sum(1 for row in gated_rows if row["grid_quality_status"] == "grid_rejected") == 5


def test_chunk_fallback_and_backlog_actions_are_preserved():
    decisions, _, gated_rows = _decisions_and_gated()
    chunk_fallback = [row for row in decisions if row["source_review_decision"] == gate.CHUNK_FALLBACK_DECISION]
    backlog = [row for row in decisions if row["source_review_decision"] == gate.BACKLOG_DECISION]

    assert len(chunk_fallback) == 3
    assert all(row["final_case_action"] == "chunk_fallback" for row in chunk_fallback)
    assert len(backlog) == 3
    assert all(row["final_case_action"] == "backlog" for row in backlog)
    assert sum(row["final_case_action"] in {"chunk_fallback", "backlog"} for row in gated_rows) == 6


def test_no_case_upgrades_to_pass_or_production_ready_and_value_bbox_stays_false():
    _, gated_objects, gated_rows = _decisions_and_gated()

    assert all(row["hybrid_validation_status"] != "pass_with_warnings" for row in gated_rows)
    assert all(not obj.get("production_ready") for obj in gated_objects)
    assert all(row["value_bboxes_available"] == "false" for row in gated_rows)


def test_required_bucket_counts():
    _, _, gated_rows = _decisions_and_gated()

    assert sum(1 for row in gated_rows if row["final_case_action"] == "manual_review_binding") == 5
    assert sum(1 for row in gated_rows if row["grid_quality_status"] == "grid_rejected") == 5
    assert (
        sum(
            1
            for row in gated_rows
            if row["final_case_action"] in {"chunk_fallback", "backlog", "exclude_current_pdfplumber_candidate"}
        )
        == 6
    )
