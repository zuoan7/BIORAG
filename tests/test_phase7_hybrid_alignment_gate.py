import csv

from scripts.extraction import align_chunk_pdfplumber_tables as align
from scripts.extraction import build_hybrid_table_objects_v1 as build
from scripts.extraction import validate_hybrid_table_objects_v1 as validate


def chunk_obj(**overrides):
    obj = {
        "table_object_id": "doc_x__table_1__phase7b2_01",
        "doc_id": "doc_x",
        "table_id": "Table 1",
        "page": 3,
        "caption": "Table 1. Energy source Gb3 titer with different energy sources",
        "validation_status": "partial",
        "warnings": ["no_value_level_bbox"],
        "chunk_ids": ["doc_x_chunk_1"],
        "caption_block_ids": ["b1"],
        "body_block_ids": ["b1"],
        "header_block_ids": ["b1"],
        "source_block_ids": ["b1"],
        "columns": [{"column_id": "old_col_1", "header": "row_text"}],
        "rows": [{"row_id": "old_row_1", "row_index": 1, "row_text": "Galactose 0.67"}],
        "cells": [{"cell_id": "old_cell_1", "row_id": "old_row_1", "column_id": "old_col_1", "value_raw": "Galactose 0.67"}],
        "source_spans": [{"source_span_id": "s1", "granularity": "table_row_level", "bbox": None}],
        "source_span_granularity": "table_row_level",
        "source_span_limitation": "chunk only",
        "source_file": "doc_x.pdf",
        "section_path": ["Test"],
        "footnotes": [],
        "references": [],
    }
    obj.update(overrides)
    return obj


def pdf_table(**overrides):
    obj = {
        "pdfplumber_table_id": "pdfplumber_doc_x_p003_lines_001",
        "doc_id": "doc_x",
        "page_number": 3,
        "strategy": "lines",
        "bbox": [10, 20, 200, 120],
        "row_count": 3,
        "column_count": 3,
        "cell_count": 9,
        "non_empty_cell_count": 9,
        "empty_cell_ratio": 0.0,
        "rows": [
            ["Energy source", "JAT/pGb3", "JAET/pGb3"],
            ["Galactose", "0.67", "0.12"],
            ["Glucose", "0.18", "0.17"],
        ],
        "cells": [
            {"row_index": r, "column_index": c, "text": f"r{r}c{c}", "bbox": [c, r, c + 1, r + 1]}
            for r in range(1, 4)
            for c in range(1, 4)
        ],
        "cell_bboxes": [{"cell_id": "x", "bbox": [1, 1, 2, 2]}],
        "cell_bboxes_available": True,
        "cell_bbox_coverage": 1.0,
        "table_text": "Table 1 Energy source Gb3 titer Galactose 0.67 0.12 Glucose 0.18 0.17",
        "text_preview": "Table 1 Energy source Gb3 titer Galactose 0.67 0.12",
        "extraction_confidence": "high",
        "extraction_warnings": [],
        "layout_quality_status": "usable",
        "layout_quality_score": 0.95,
        "likely_false_positive_layout": False,
    }
    obj.update(overrides)
    return obj


def alignment_row(**overrides):
    row = {
        "chunk_table_object_id": "doc_x__table_1__phase7b2_01",
        "doc_id": "doc_x",
        "table_id": "Table 1",
        "chunk_page": "3",
        "chunk_validation_status": "partial",
        "pdfplumber_table_id": "pdfplumber_doc_x_p003_lines_001",
        "pdf_page": "3",
        "pdf_strategy": "lines",
        "pdf_table_bbox": "[10, 20, 200, 120]",
        "layout_quality_status": "usable",
        "alignment_status": "matched",
        "alignment_confidence": "high",
        "alignment_score": "0.9",
        "alignment_basis": "same_doc;same_page;same_table_id;layout_quality_usable",
        "alignment_blockers": "none",
        "needs_manual_alignment_review": "false",
        "notes": "test",
    }
    row.update(overrides)
    return row


def test_page_only_match_is_low_confidence_and_manual_review():
    obj = chunk_obj(caption="Table 1. Completely unrelated caption words")
    pdf = pdf_table(table_text="layout grid values only", text_preview="layout grid only")

    row = align.choose_alignment(obj, [pdf], chunks_by_id={})

    assert row["alignment_status"] == "page_only_match"
    assert row["alignment_confidence"] == "low"
    assert row["needs_manual_alignment_review"] == "true"
    assert "page_only_match" in row["alignment_blockers"]


def test_likely_false_positive_layout_cannot_be_high_confidence():
    row = align.choose_alignment(
        chunk_obj(),
        [
            pdf_table(
                layout_quality_status="likely_false_positive",
                layout_quality_score=0.1,
                likely_false_positive_layout=True,
            )
        ],
        chunks_by_id={},
    )

    assert row["alignment_confidence"] != "high"
    assert row["needs_manual_alignment_review"] == "true"


def test_high_confidence_requires_same_page_table_id_and_usable_layout():
    row = align.choose_alignment(chunk_obj(), [pdf_table()], chunks_by_id={})

    assert row["alignment_status"] == "matched"
    assert row["alignment_confidence"] == "high"
    assert "same_doc" in row["alignment_basis"]
    assert "same_page" in row["alignment_basis"]
    assert "same_table_id" in row["alignment_basis"]
    assert "layout_quality_usable" in row["alignment_basis"]


def test_no_pdf_table_found_builds_chunk_fallback_not_matched():
    row = align.choose_alignment(chunk_obj(), [], chunks_by_id={})
    hybrid = build.build_hybrid_object(chunk_obj(), None, row)

    assert row["alignment_status"] == "no_pdf_table_found"
    assert hybrid["hybrid_metadata"]["alignment_status"] == "no_pdf_table_found"
    assert hybrid["hybrid_metadata"]["extraction_method"] == "chunk_fallback"
    assert hybrid["hybrid_metadata"]["pdfplumber_table_id"] is None


def test_cell_bboxes_available_does_not_set_value_bboxes_available():
    hybrid = build.build_hybrid_object(chunk_obj(), pdf_table(), alignment_row())

    assert hybrid["hybrid_metadata"]["cell_bboxes_available"] is True
    assert hybrid["hybrid_metadata"]["value_bboxes_available"] is False
    assert hybrid["source_span_granularity"] == "cell_level"


def test_source_span_granularity_never_uses_value_level():
    hybrid = build.build_hybrid_object(
        chunk_obj(source_span_granularity="value_level"),
        pdf_table(),
        alignment_row(alignment_confidence="low"),
    )

    assert hybrid["source_span_granularity"] != "value_level"
    assert hybrid["hybrid_metadata"]["source_span_granularity"] != "value_level"


def test_primary_failure_stage_is_written_to_validation_summary(tmp_path):
    row = alignment_row(
        alignment_status="page_only_match",
        alignment_confidence="low",
        alignment_blockers="page_only_match",
        needs_manual_alignment_review="true",
    )
    hybrid = build.build_hybrid_object(chunk_obj(), pdf_table(), row)
    summary_row = validate.validate_hybrid_object(hybrid, row, pdf_table())
    output = tmp_path / "summary.csv"

    validate.write_summary_csv([summary_row], output)

    with output.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["primary_failure_stage"] == "alignment"
    assert written[0]["manual_review_reason"] == "page_only_match_requires_manual_alignment_review"
