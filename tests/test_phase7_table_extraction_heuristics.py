from scripts.extraction import extract_table_objects_v1 as extract
from scripts.extraction import validate_table_objects_v1 as validate


def chunk(doc_id, index, text, *, contains_table_caption=False, block_type="table_caption"):
    block_id = f"p1_b{index:04d}"
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_chunk_{index}",
        "chunk_index": index,
        "text": text,
        "contains_table_caption": contains_table_caption,
        "contains_table_text": False,
        "contains_figure_caption": False,
        "source_block_ids": [block_id],
        "source_block_metadata": [
            {
                "block_id": block_id,
                "type": block_type,
                "page": 1,
                "text_preview": text,
            }
        ],
        "block_types": [block_type],
        "page_start": 1,
        "section_path": ["Test"],
        "source_file": f"{doc_id}.pdf",
    }


def candidate(doc_id, table_id, chunk_id):
    return {
        "candidate_id": f"phase7b2_{doc_id}_{extract.compact_id(table_id)}",
        "doc_id": doc_id,
        "table_id": table_id,
        "chunk_ids": [chunk_id],
        "warnings": [],
        "candidate_status": "active",
        "candidate_status_reason": "candidate_detected",
        "candidate_decision_warnings": [],
    }


def complete_object(warnings=None):
    return {
        "table_object_id": "doc_x__table_1__phase7b2_01",
        "doc_id": "doc_x",
        "table_id": "Table 1",
        "caption": "Table 1. Test",
        "source_block_ids": ["b1"],
        "chunk_ids": ["c1"],
        "caption_block_ids": ["b1"],
        "header_block_ids": ["b1"],
        "body_block_ids": ["b1"],
        "rows": [{"row_id": "r1"}],
        "columns": [{"column_id": "c1"}],
        "cells": [{"cell_id": "cell1", "value_raw": "1"}],
        "source_spans": [{"source_span_id": "s1", "granularity": "table_row_level", "bbox": None}],
        "source_span_granularity": "table_row_level",
        "source_span_limitation": "no value-level bbox",
        "warnings": warnings or ["no_value_level_bbox"],
    }


def test_duplicate_candidate_detection_marks_shadow_candidate():
    doc_id = "doc_dup"
    rich = chunk(
        doc_id,
        1,
        "[TABLE CAPTION] Table 1. Strain or plasmid Reference or source Strains BW25113 CMEV-1 pKD13",
        contains_table_caption=True,
    )
    shadow = chunk(
        doc_id,
        2,
        "[TABLE CAPTION] Table 1. The primers used for vector construction are listed in Table S1.",
        contains_table_caption=True,
    )
    chunks_by_doc = {doc_id: [rich, shadow]}
    candidates = [
        candidate(doc_id, "Table 1", rich["chunk_id"]),
        candidate(doc_id, "Table 1", shadow["chunk_id"]),
    ]

    extract.annotate_candidate_decisions(candidates, chunks_by_doc)

    assert candidates[1]["candidate_status"] in {"filtered", "deduped"}
    assert "duplicate_table_candidate" in candidates[1]["warnings"]
    assert "candidate_deduped" in candidates[1]["warnings"]


def test_continued_table_candidate_is_marked_for_merge():
    doc_id = "doc_cont"
    main = chunk(
        doc_id,
        1,
        "[TABLE CAPTION] Table 2. Bacterial Strains source type strain medium T atmosphere DSM 20083",
        contains_table_caption=True,
    )
    continued = chunk(
        doc_id,
        2,
        "[TABLE CAPTION] Table 2 continued bacterial species designation source type strain medium T atmosphere",
        contains_table_caption=True,
    )
    chunks_by_doc = {doc_id: [main, continued]}
    candidates = [
        candidate(doc_id, "Table 2", main["chunk_id"]),
        candidate(doc_id, "Table 2 continued", continued["chunk_id"]),
    ]

    extract.annotate_candidate_decisions(candidates, chunks_by_doc)

    assert candidates[1]["candidate_status"] == "merged_into_primary"
    assert candidates[1]["merge_target_candidate_id"] == candidates[0]["candidate_id"]
    assert "continued_table_merged" in candidates[0]["warnings"]


def test_body_blocks_missing_cannot_pass_with_warnings():
    obj = complete_object()
    obj["body_block_ids"] = []
    obj["warnings"] = ["body_blocks_missing"]

    row = validate.validate_object(obj)

    assert row["validation_status"] == "partial"
    assert "body_blocks_missing" in row["blocking_warnings"]


def test_mixed_table_block_risk_blocks_pass_with_warnings():
    row = validate.validate_object(complete_object(["mixed_table_block_risk"]))

    assert row["validation_status"] == "partial"
    assert "mixed_table_block_risk" in row["blocking_warnings"]


def test_false_positive_candidate_cannot_pass_with_warnings():
    row = validate.validate_object(complete_object(["false_positive_candidate"]))

    assert row["validation_status"] == "fail"
    assert "false_positive_candidate" in row["blocking_warnings"]


def test_table_row_level_source_span_does_not_create_value_level_bbox():
    source_spans, _ = extract.make_source_spans(
        "doc_x__table_1__phase7b2_01",
        "doc_x",
        [chunk("doc_x", 1, "Table 1 A B 1 2", contains_table_caption=True)],
        ["p1_b0001"],
    )

    assert source_spans[0]["granularity"] == "table_row_level"
    assert source_spans[0]["bbox"] is None


def test_tightened_validation_allows_only_nonblocking_warnings_to_pass():
    nonblocking = validate.validate_object(complete_object(["no_value_level_bbox"]))
    blocking_obj = complete_object(["target_mapping_risk"])

    blocking = validate.validate_object(blocking_obj)

    assert nonblocking["validation_status"] == "pass_with_warnings"
    assert blocking["validation_status"] == "partial"
