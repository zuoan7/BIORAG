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
        "warnings": ["cell_alignment_error", "no_value_level_bbox"],
        "chunk_ids": ["doc_x_chunk_1"],
        "caption_block_ids": ["b1"],
        "body_block_ids": ["b1"],
        "header_block_ids": ["b1"],
        "source_block_ids": ["b1"],
        "columns": [{"column_id": "old_col_1", "header": "row_text"}],
        "rows": [{"row_id": "old_row_1", "row_text": "Galactose 0.67"}],
        "cells": [{"cell_id": "old_cell_1", "value_raw": "Galactose 0.67"}],
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
    }
    obj.update(overrides)
    return obj


def alignment_row(**overrides):
    row = {
        "chunk_table_object_id": "doc_x__table_1__phase7b2_01",
        "doc_id": "doc_x",
        "table_id": "Table 1",
        "alignment_status": "matched",
        "alignment_confidence": "medium",
        "alignment_reason": "test",
        "pdfplumber_table_id": "pdfplumber_doc_x_p003_lines_001",
    }
    row.update(overrides)
    return row


def test_normalized_table_id_matching():
    assert align.normalize_table_id("Supplementary Table S1 continued") == "table_s1"
    assert align.normalize_table_id("TABLE 2.") == "table_2"


def test_page_based_alignment_can_return_page_only_match():
    obj = chunk_obj(caption="Table 1. Completely unrelated caption words")
    pdf = pdf_table(table_text="layout grid no overlapping caption", text_preview="layout grid")

    row = align.choose_alignment(obj, [pdf], chunks_by_id={})

    assert row["alignment_status"] == "page_only_match"
    assert row["alignment_confidence"] in {"low", "medium"}


def test_caption_text_overlap_scoring():
    score = align.caption_text_overlap_score(
        "Energy source Gb3 titer Galactose Glucose",
        "Table 1 Energy source Gb3 titer Galactose values",
    )

    assert score > 0.45


def test_multiple_pdf_tables_conflict_needs_review():
    obj = chunk_obj(caption="Table 1. no overlap tokens")
    pdfs = [
        pdf_table(
            pdfplumber_table_id=f"pdf_{i}",
            table_text=f"layout candidate {i}",
            text_preview=f"layout candidate {i}",
            extraction_confidence="medium",
        )
        for i in range(3)
    ]

    row = align.choose_alignment(obj, pdfs, chunks_by_id={})

    assert row["alignment_status"] == "multiple_pdf_tables"
    assert row["needs_manual_alignment_review"] == "true"


def test_no_pdf_table_found():
    row = align.choose_alignment(chunk_obj(), [], chunks_by_id={})

    assert row["alignment_status"] == "no_pdf_table_found"
    assert row["alignment_confidence"] == "none"


def test_cell_bbox_available_sets_cell_level_source_span():
    hybrid = build.build_hybrid_object(chunk_obj(), pdf_table(), alignment_row())

    assert hybrid["cell_bboxes_available"] is True
    assert hybrid["source_span_granularity"] == "cell_level"


def test_no_token_bbox_never_sets_value_level():
    hybrid = build.build_hybrid_object(chunk_obj(), pdf_table(), alignment_row())

    assert hybrid["source_span_granularity"] != "value_level"
    assert hybrid["no_value_level_bbox"] is True
    assert all(span["granularity"] != "value_level" for span in hybrid["source_spans"])


def test_low_confidence_alignment_cannot_pass():
    hybrid = build.build_hybrid_object(
        chunk_obj(warnings=["no_value_level_bbox"]),
        pdf_table(),
        alignment_row(alignment_confidence="low"),
    )

    row = validate.validate_hybrid_object(hybrid)

    assert row["hybrid_validation_status"] == "manual_review"
    assert "低置信对齐不能 pass" in row["notes"]
