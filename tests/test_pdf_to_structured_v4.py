#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import base64

import fitz

from scripts.ingestion.pdf_to_structured import (
    TextLine,
    analyze_document_two_column_prior,
    analyze_page_layout,
    build_page_blocks,
    build_structured_text,
    collect_repeated_journal_preproof_keys,
    extract_layout_from_page,
    extract_layout_from_blocks,
    guess_column,
    is_figure_caption_candidate,
    is_likely_two_column_page,
    is_table_caption_candidate,
    should_strip_journal_preproof_noise,
    sort_two_column_region_reading_order,
    sort_lines_reading_order,
    strip_journal_preproof_noise,
)
from scripts.diagnostics.validate_parsed_raw_v4 import analyze_doc, render_report, summarize


def make_line(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    page_width: float = 600.0,
    page_num: int = 1,
    size: float = 9.0,
) -> TextLine:
    return TextLine(
        text=text,
        size=size,
        is_bold=False,
        is_italic=False,
        page_num=page_num,
        y_pos=y0,
        bbox=(x0, y0, x1, y1),
        block_no=0,
        line_no=0,
        column=guess_column(x0, x1, page_width),
    )


def make_weak_two_column_page(page_num: int, *, rows: int = 8, page_width: float = 600.0) -> list[TextLine]:
    left = [
        make_line(
            f"L{i} weak gap left column sentence",
            60,
            100 + i * 14,
            315,
            110 + i * 14,
            page_width=page_width,
            page_num=page_num,
        )
        for i in range(rows)
    ]
    right = [
        make_line(
            f"R{i} weak gap right column sentence",
            320,
            100 + i * 14,
            560,
            110 + i * 14,
            page_width=page_width,
            page_num=page_num,
        )
        for i in range(rows)
    ]
    return [item for pair in zip(left, right) for item in pair]


def test_guess_column_returns_l_r_span_unk() -> None:
    assert guess_column(40, 250, 600) == "L"
    assert guess_column(350, 560, 600) == "R"
    assert guess_column(40, 560, 600) == "SPAN"
    assert guess_column(292, 308, 600) == "UNK"


def test_sort_lines_reading_order_two_column_and_single_column() -> None:
    left = [
        make_line(f"L{i} left column sentence", 50, 100 + i * 12, 260, 110 + i * 12)
        for i in range(1, 9)
    ]
    right = [
        make_line(f"R{i} right column sentence", 340, 100 + i * 12, 550, 110 + i * 12)
        for i in range(1, 9)
    ]
    interleaved = [item for pair in zip(left, right) for item in pair]

    assert is_likely_two_column_page(interleaved, 600, 800)
    sorted_lines = sort_lines_reading_order(interleaved, 600, 800)
    assert [line.text.split()[0] for line in sorted_lines[:4]] == ["L1", "L2", "L3", "L4"]
    assert [line.text.split()[0] for line in sorted_lines[8:12]] == ["R1", "R2", "R3", "R4"]

    single_column = [
        make_line(f"S{i} single column sentence", 60, 100 + i * 12, 540, 110 + i * 12)
        for i in range(1, 10)
    ]
    assert not is_likely_two_column_page(single_column, 600, 800)
    assert sort_lines_reading_order(list(reversed(single_column)), 600, 800) == single_column


def test_two_column_region_sort_keeps_middle_caption_between_regions() -> None:
    top = make_line("Results", 60, 80, 540, 94, size=10.5)
    upper_left = [
        make_line(f"L{i} upper left column sentence", 60, 110 + i * 14, 260, 120 + i * 14)
        for i in range(1, 4)
    ]
    upper_right = [
        make_line(f"R{i} upper right column sentence", 340, 110 + i * 14, 540, 120 + i * 14)
        for i in range(1, 4)
    ]
    caption = make_line("Fig. 1. Overview of the workflow", 60, 190, 540, 204)
    lower_left = [
        make_line(f"L{i} lower left column sentence", 60, 230 + i * 14, 260, 240 + i * 14)
        for i in range(4, 7)
    ]
    lower_right = [
        make_line(f"R{i} lower right column sentence", 340, 230 + i * 14, 540, 240 + i * 14)
        for i in range(4, 7)
    ]
    mixed = [top, *sum(zip(upper_left, upper_right), ()), caption, *sum(zip(lower_left, lower_right), ())]

    sorted_lines = sort_two_column_region_reading_order(mixed, 600, 800)
    tokens = [line.text.split()[0] for line in sorted_lines]

    assert tokens == [
        "Results",
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
        "Fig.",
        "L4",
        "L5",
        "L6",
        "R4",
        "R5",
        "R6",
    ]


def test_journal_preproof_noise_stops_at_abstract() -> None:
    lines = [
        make_line("Journal Pre-proof", 60, 80, 300, 90),
        make_line("PII: S1234-5678(26)00001-2", 60, 95, 340, 105),
        make_line("DOI: 10.1016/example", 60, 110, 340, 120),
        make_line("This is a PDF file of an article that has undergone enhancements", 60, 125, 540, 135),
        make_line("Abstract", 60, 150, 160, 160),
        make_line("This is a PDF file of an article inside the body and should remain", 60, 165, 540, 175),
    ]
    filtered, diag = strip_journal_preproof_noise(lines, page_num=1)
    page_text = build_structured_text(filtered, {})

    assert diag["stripped_count"] == 4
    assert "Journal Pre-proof" not in page_text
    assert "PII:" not in page_text
    assert "DOI:" not in page_text
    assert "## Abstract" in page_text
    assert "inside the body and should remain" in page_text


def test_unicode_caption_candidates() -> None:
    assert is_figure_caption_candidate("Fig.\u00a01. Overview")
    assert is_figure_caption_candidate("Figure\u202fS1: Workflow")
    assert is_table_caption_candidate("Table\u00a01 – Strains used")
    assert is_table_caption_candidate("Table S2. Primers")


def test_image_block_is_preserved_in_blocks() -> None:
    blocks = [
        {
            "type": 0,
            "number": 1,
            "bbox": [60, 80, 540, 100],
            "lines": [
                {
                    "bbox": [60, 80, 300, 90],
                    "spans": [{"text": "A visible text line", "size": 9.0, "flags": 0, "bbox": [60, 80, 300, 90]}],
                }
            ],
        },
        {"type": 1, "number": 2, "bbox": [100, 120, 500, 360]},
    ]

    lines, image_blocks = extract_layout_from_blocks(blocks, page_num=1, page_width=600)
    page_blocks = build_page_blocks(lines, image_blocks, page_width=600, page_height=800)

    assert len(lines) == 1
    assert len(image_blocks) == 1
    image_output = [block for block in page_blocks if block.type == "image"]
    assert len(image_output) == 1
    assert image_output[0].bbox == [100.0, 120.0, 500.0, 360.0]


def test_pymupdf_image_rect_metadata_is_preserved(tmp_path) -> None:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    pdf_path = tmp_path / "image_sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((40, 50), "A visible line")
    page.insert_image(fitz.Rect(80, 100, 160, 180), stream=png_bytes)
    doc.save(pdf_path)
    doc.close()

    parsed_doc = fitz.open(pdf_path)
    try:
        lines, image_blocks = extract_layout_from_page(parsed_doc[0], page_num=1)
    finally:
        parsed_doc.close()

    assert lines
    assert len(image_blocks) == 1
    image = image_blocks[0]
    assert image.type == "image"
    assert image.text == ""
    assert image.bbox == [80.0, 100.0, 160.0, 180.0]
    assert image.metadata
    assert image.metadata.get("xref")
    assert image.metadata.get("image_width") == 1
    assert image.metadata.get("image_height") == 1


def test_validation_outputs_residual_examples_without_counting_diagnostics() -> None:
    new_data = {
        "doc_id": "doc_test",
        "total_pages": 1,
        "parser_stage": "parsed_raw_v4",
        "pages": [
            {
                "page": 1,
                "text": "# Journal Pre-proof",
                "blocks": [
                    {
                        "block_id": "p1_b0001",
                        "type": "text",
                        "text": "Journal Pre-proof",
                        "bbox": [100, 200, 400, 500],
                        "column": "SPAN",
                        "reading_order": 1,
                        "page": 1,
                        "size": 43.9,
                    }
                ],
            }
        ],
        "diagnostics": {
            "stripped_noise_examples": ["Journal Pre-proof"],
            "stripped_noise_line_count": 1,
        },
    }

    result = analyze_doc("doc_test", new_data, None)
    summary = summarize([result], has_old_compare=False, old_dir_missing=False)
    report = render_report(summary, [result], None, __import__("pathlib").Path("new"))

    assert result["journal_preproof_noise_in_page_text_count"] == 1
    assert result["journal_preproof_noise_in_block_text_count"] == 1
    assert result["journal_preproof_noise_in_diagnostics_count"] == 1
    assert result["residual_examples"][0]["is_in_diagnostics_only"] is False
    assert "raw_text_preview" in report
    assert "journal pre-proof" in report


def test_exact_journal_preproof_variants_are_stripped() -> None:
    lines = [
        make_line("Journal Pre-proof", 100, 200, 400, 500, size=43.9),
        make_line("JOURNAL PRE-PROOF", 100, 220, 400, 520, size=43.9),
        make_line("Journal pre-proof", 100, 240, 400, 540, size=43.9),
    ]
    filtered, diag = strip_journal_preproof_noise(lines, page_num=3, total_pages=10)

    assert filtered == []
    assert diag["stripped_count"] == 3
    assert diag["reason_counts"]["journal_preproof_exact_line"] == 3


def test_repeated_journal_preproof_header_key_is_collected_and_stripped() -> None:
    lines = [
        make_line("Journal Pre-proof watermark", 120, 220, 420, 500, page_num=page, size=24)
        for page in range(1, 4)
    ]
    repeated_keys = collect_repeated_journal_preproof_keys(lines)

    assert "journal pre-proof watermark" in repeated_keys
    should_strip, reason = should_strip_journal_preproof_noise(
        lines[2],
        page_num=3,
        total_pages=10,
        front_matter_active=False,
        repeated_journal_preproof_keys=repeated_keys,
    )
    assert should_strip
    assert reason == "journal_preproof_repeated_header"


def test_front_matter_metadata_is_not_global() -> None:
    front_lines = [
        make_line("PII: S1234", 60, 80, 260, 90, page_num=1),
        make_line("DOI: 10.1016/example", 60, 95, 300, 105, page_num=1),
        make_line("Reference: BITE 12345", 60, 110, 300, 120, page_num=1),
        make_line("To appear in: Bioresource Technology", 60, 125, 360, 135, page_num=1),
        make_line("Please cite this article as: Example", 60, 140, 440, 150, page_num=1),
    ]
    filtered, diag = strip_journal_preproof_noise(front_lines, page_num=1, total_pages=10)
    assert filtered == []
    assert diag["stripped_count"] == 5

    body_doi = make_line("DOI: 10.1016/body should remain in a later page", 60, 200, 500, 210, page_num=10)
    body_reference = make_line("Reference: this is ordinary body text on page ten", 60, 215, 500, 225, page_num=10)
    filtered_body, body_diag = strip_journal_preproof_noise([body_doi, body_reference], page_num=10, total_pages=10)
    assert filtered_body == [body_doi, body_reference]
    assert body_diag["stripped_count"] == 0


def test_stripped_lines_do_not_enter_page_text_or_blocks() -> None:
    lines = [
        make_line("Journal Pre-proof", 100, 200, 400, 500, page_num=3, size=43.9),
        make_line("Real body text should remain.", 60, 120, 320, 132, page_num=3),
    ]
    filtered, diag = strip_journal_preproof_noise(lines, page_num=3, total_pages=5)
    page_text = build_structured_text(filtered, {})
    page_blocks = build_page_blocks(filtered, [], page_width=600, page_height=800)

    assert "Journal Pre-proof" not in page_text
    assert all("Journal Pre-proof" not in block.text for block in page_blocks)
    assert "Journal Pre-proof" in diag["examples"]


def test_long_body_sentence_with_journal_preproof_is_not_stripped() -> None:
    line = make_line(
        "This long body sentence discusses why the phrase journal pre-proof appears in a citation context and should remain in the parsed body.",
        60,
        220,
        560,
        232,
        page_num=5,
    )
    should_strip, reason = should_strip_journal_preproof_noise(
        line,
        page_num=5,
        total_pages=10,
        front_matter_active=False,
        repeated_journal_preproof_keys=set(),
    )
    assert not should_strip
    assert reason == ""


def test_document_level_two_column_prior_from_weak_pages() -> None:
    diagnostics = [
        analyze_page_layout(make_weak_two_column_page(page_num), 600, 800)
        for page_num in range(2, 6)
    ]
    prior = analyze_document_two_column_prior(diagnostics)

    assert prior["document_two_column_prior"] is True
    assert prior["document_two_column_confidence"] >= 0.75
    assert prior["relaxed_two_column_signal_pages"] == [2, 3, 4, 5]


def test_document_prior_relaxes_weak_column_gap_page() -> None:
    lines = make_weak_two_column_page(3)

    strict = analyze_page_layout(lines, 600, 800)
    relaxed = analyze_page_layout(lines, 600, 800, document_two_column_prior=True)

    assert strict["is_two_column"] is False
    assert strict["reason"] in {"weak_column_gap", "likely_two_column_but_fallback"}
    assert relaxed["is_two_column"] is True
    assert relaxed["reason"] == "document_prior_relaxed_two_column"


def test_document_prior_exempts_front_matter_page() -> None:
    lines = make_weak_two_column_page(1)

    diagnostic = analyze_page_layout(lines, 600, 800, document_two_column_prior=True)

    assert diagnostic["is_two_column"] is False
    assert diagnostic["reason"] == "document_prior_but_exempt_front_matter"


def test_document_prior_exempts_references_page() -> None:
    lines = [
        make_line("References", 60, 80, 180, 92, page_num=9, size=10.0, page_width=600),
        *make_weak_two_column_page(9),
    ]

    diagnostic = analyze_page_layout(lines, 600, 800, document_two_column_prior=True)

    assert diagnostic["is_two_column"] is False
    assert diagnostic["reason"] == "document_prior_but_exempt_references"


def test_document_prior_exempts_figure_table_heavy_page() -> None:
    lines = [
        make_line("Fig. 1. Overview", 60, 80, 540, 92, page_num=4, page_width=600),
        make_line("Table 1. Strains used", 60, 100, 540, 112, page_num=4, page_width=600),
        make_line("Figure S1: Workflow", 60, 120, 540, 132, page_num=4, page_width=600),
        *make_weak_two_column_page(4),
    ]

    diagnostic = analyze_page_layout(lines, 600, 800, document_two_column_prior=True)

    assert diagnostic["is_two_column"] is True
    assert diagnostic["reason"] == "document_prior_region_mixed_figure_table"
    assert diagnostic["selected_order_strategy"] == "two_column_region_mixed"


def test_document_prior_exempts_sparse_figure_table_heavy_page() -> None:
    lines = [
        make_line("Fig. 1. Overview", 60, 80, 540, 92, page_num=4, page_width=600),
        make_line("Table 1. Strains used", 60, 100, 540, 112, page_num=4, page_width=600),
        make_line("Figure S1: Workflow", 60, 120, 540, 132, page_num=4, page_width=600),
        make_line("Only a short left note", 60, 180, 250, 192, page_num=4, page_width=600),
        make_line("Only a short right note", 340, 180, 540, 192, page_num=4, page_width=600),
    ]

    diagnostic = analyze_page_layout(lines, 600, 800, document_two_column_prior=True)

    assert diagnostic["is_two_column"] is False
    assert diagnostic["reason"] == "document_prior_but_exempt_figure_table"


def test_single_column_document_does_not_get_two_column_prior() -> None:
    diagnostics = []
    for page_num in range(2, 6):
        lines = [
            make_line(
                f"Single column body sentence {i}",
                60,
                100 + i * 14,
                540,
                110 + i * 14,
                page_num=page_num,
                page_width=600,
            )
            for i in range(16)
        ]
        diagnostics.append(analyze_page_layout(lines, 600, 800))

    prior = analyze_document_two_column_prior(diagnostics)

    assert prior["document_two_column_prior"] is False
    assert prior["relaxed_two_column_signal_pages"] == []


def test_journal_preproof_residual_filter_still_strips_before_layout() -> None:
    lines = [
        make_line("Journal Pre-proof", 100, 200, 400, 500, page_num=3, size=43.9),
        *make_weak_two_column_page(3),
    ]
    filtered, diag = strip_journal_preproof_noise(lines, page_num=3, total_pages=8)
    page_text = build_structured_text(filtered, {})
    page_blocks = build_page_blocks(
        filtered,
        [],
        page_width=600,
        page_height=800,
        layout_diagnostic=analyze_page_layout(filtered, 600, 800, document_two_column_prior=True),
    )

    assert diag["stripped_count"] == 1
    assert "Journal Pre-proof" not in page_text
    assert all("Journal Pre-proof" not in block.text for block in page_blocks)
