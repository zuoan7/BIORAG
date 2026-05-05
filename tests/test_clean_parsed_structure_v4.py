#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from scripts.ingestion.clean_parsed_structure import ProcessingCounters, process_document


def raw_block(
    text: str,
    order: int,
    *,
    page: int = 1,
    block_id: str | None = None,
    block_type: str = "text",
    bbox: list[float] | None = None,
) -> dict:
    return {
        "block_id": block_id or f"p{page}_b{order:04d}",
        "type": block_type,
        "text": text,
        "bbox": bbox or [72.0, 100.0 + order * 12, 240.0, 112.0 + order * 12],
        "column": "L",
        "reading_order": order,
        "page": page,
        "size": 8.9,
        "is_bold": False,
        "is_italic": False,
        "block_no": order,
        "line_no": 0,
    }


def run_clean(tmp_path: Path, data: dict) -> dict:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "clean"
    preview_dir = tmp_path / "preview"
    input_dir.mkdir()
    output_dir.mkdir()
    preview_dir.mkdir()
    path = input_dir / f"{data.get('doc_id', 'doc_test')}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    process_document(path, output_dir, preview_dir, ProcessingCounters())
    return json.loads((output_dir / path.name).read_text(encoding="utf-8"))


def make_v4_doc(pages: list[dict], total_pages: int | None = None) -> dict:
    return {
        "doc_id": "doc_test",
        "source_file": "doc_test.pdf",
        "total_pages": total_pages or len(pages),
        "parser_stage": "parsed_raw_v4",
        "pages": pages,
    }


def all_blocks(clean: dict) -> list[dict]:
    return [block for page in clean["pages"] for block in page["blocks"]]


def test_raw_type_text_does_not_pass_through(tmp_path: Path) -> None:
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": "Normal body sentence.", "blocks": [raw_block("Normal body sentence.", 1)]}
    ]))

    assert {block["type"] for block in all_blocks(clean)} == {"paragraph"}
    assert all(block["type"] != "text" for block in all_blocks(clean))


def test_references_heading_from_raw_text_enters_references(tmp_path: Path) -> None:
    clean = run_clean(tmp_path, make_v4_doc([
        {
            "page": 4,
            "text": "References\n1. Smith et al. Journal 10, 1-9 (2024). doi:10/example",
            "blocks": [
                raw_block("References", 1, page=4),
                raw_block("1. Smith et al. Journal 10, 1-9 (2024). doi:10/example", 2, page=4),
            ],
        }
    ], total_pages=4))

    types = [block["type"] for block in all_blocks(clean)]
    assert types == ["section_heading", "references"]


def test_numbered_references_in_late_document_are_not_paragraph(tmp_path: Path) -> None:
    pages = [
        {"page": 1, "text": "Body.", "blocks": [raw_block("Body sentence remains.", 1, page=1)]},
        {"page": 2, "text": "Body.", "blocks": [raw_block("More body sentence remains.", 1, page=2)]},
        {
            "page": 3,
            "text": "1. Smith et al. Journal 10, 1-9 (2024). doi:10/example",
            "blocks": [
                raw_block("1. Smith et al. Journal 10, 1-9 (2024). doi:10/example", 1, page=3),
                raw_block("2. Jones et al. Yeast 11, 2-8 (2023). doi:10/example", 2, page=3),
            ],
        },
    ]
    clean = run_clean(tmp_path, make_v4_doc(pages, total_pages=3))

    late_types = [block["type"] for block in clean["pages"][2]["blocks"]]
    assert late_types == ["references", "references"]


def test_front_matter_metadata_not_paragraph(tmp_path: Path) -> None:
    lines = [
        "PII: S1234",
        "DOI: 10.1016/example",
        "To appear in: Bioresource Technology",
        "Received Date: 1 January 2024",
        "Accepted Date: 2 January 2024",
        "Journal homepage: www.example.com",
    ]
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": "\n".join(lines), "blocks": [raw_block(line, i + 1) for i, line in enumerate(lines)]}
    ]))

    assert {block["type"] for block in all_blocks(clean)} == {"metadata"}
    assert clean["pages"][0]["text"] == ""


def test_body_doi_is_not_global_metadata(tmp_path: Path) -> None:
    text = "The enzyme was compared against DOI: 10.1000/body in the main discussion."
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 5, "text": text, "blocks": [raw_block(text, 1, page=5)]}
    ], total_pages=5))

    block = all_blocks(clean)[0]
    assert block["type"] == "paragraph"
    assert "DOI: 10.1000/body" in clean["pages"][0]["text"]


def test_correspondence_metadata_not_in_page_text(tmp_path: Path) -> None:
    text = "*Correspondence: author@example.org"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": text, "blocks": [raw_block(text, 1, page=1)]}
    ]))

    block = all_blocks(clean)[0]
    assert block["type"] == "metadata"
    assert "Correspondence" not in clean["pages"][0]["text"]


def test_to_whom_correspondence_metadata(tmp_path: Path) -> None:
    text = "*To whom correspondence should be addressed. Tel: +81-538-32-7337"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": text, "blocks": [raw_block(text, 1, page=1)]}
    ]))

    assert all_blocks(clean)[0]["type"] == "metadata"
    assert clean["pages"][0]["text"] == ""


def test_correspondence_may_also_be_addressed_not_table_text(tmp_path: Path) -> None:
    text = "yCorrespondence may also be addressed: Tel: +81-538-32-7337,"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": text, "blocks": [raw_block(text, 1, page=1)]}
    ]))

    assert all_blocks(clean)[0]["type"] == "metadata"
    assert clean["pages"][0]["text"] == ""


def test_numbered_affiliation_metadata(tmp_path: Path) -> None:
    text = "1 Department of Food Engineering, Akdeniz University, Antalya, Turkey"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": text, "blocks": [raw_block(text, 1, page=1)]}
    ]))

    assert all_blocks(clean)[0]["type"] == "metadata"
    assert clean["pages"][0]["text"] == ""


def test_body_university_reference_is_not_removed(tmp_path: Path) -> None:
    text = "The University of Delaware Research Foundation grant supported this fermentation work."
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 6, "text": text, "blocks": [raw_block(text, 1, page=6)]}
    ], total_pages=8))

    block = all_blocks(clean)[0]
    assert block["type"] == "paragraph"
    assert "University of Delaware" in clean["pages"][0]["text"]


def test_marginal_access_banner_is_metadata(tmp_path: Path) -> None:
    text = "at University of Hawaii at Manoa Library on June 16, 2015"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": text, "blocks": [raw_block(text, 1, page=1, bbox=[593.75, 374.4, 604.42, 566.37])]}
    ]))

    assert all_blocks(clean)[0]["type"] == "metadata"
    assert clean["pages"][0]["text"] == ""


def test_marginal_banner_priority_over_references(tmp_path: Path) -> None:
    banner = "at University of Hawaii at Manoa Library on June 16, 2015"
    clean = run_clean(tmp_path, make_v4_doc([
        {
            "page": 4,
            "text": "References\n" + banner,
            "blocks": [
                raw_block("References", 1, page=4),
                raw_block(banner, 2, page=4, bbox=[593.75, 374.4, 604.42, 566.37]),
            ],
        }
    ], total_pages=4))

    assert [block["type"] for block in all_blocks(clean)] == ["section_heading", "metadata"]


def test_false_table_text_body_sentence_is_paragraph(tmp_path: Path) -> None:
    text = "at a compound annual growth rate of 6.3%, reaching US$4.9 billion by 2033 [7]. Caseins are"
    clean = run_clean(tmp_path, make_v4_doc([
        {
            "page": 2,
            "text": "Table 1: Data\n" + text,
            "blocks": [
                raw_block("Table 1: Data", 1, page=2),
                raw_block(text, 2, page=2, bbox=[55.8, 621.53, 415.58, 630.5]),
            ],
        }
    ], total_pages=3))

    assert [block["type"] for block in all_blocks(clean)] == ["table_caption", "paragraph"]


def test_real_table_text_still_works(tmp_path: Path) -> None:
    text = "Primer name Sequence OD600 1.0 2.0 3.0"
    clean = run_clean(tmp_path, make_v4_doc([
        {
            "page": 2,
            "text": "Table 1: Strains\n" + text,
            "blocks": [
                raw_block("Table 1: Strains", 1, page=2),
                raw_block(text, 2, page=2),
            ],
        }
    ], total_pages=3))

    assert [block["type"] for block in all_blocks(clean)] == ["table_caption", "table_text"]


def test_journal_preproof_noise_still_removed_from_page_text(tmp_path: Path) -> None:
    text = "Journal Pre-proof"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": text, "blocks": [raw_block(text, 1, page=1)]}
    ]))

    assert all_blocks(clean)[0]["type"] == "metadata"
    assert clean["pages"][0]["text"] == ""


def test_figure_and_table_caption_conversion(tmp_path: Path) -> None:
    clean = run_clean(tmp_path, make_v4_doc([
        {
            "page": 2,
            "text": "Fig. 1. Overview\nTable S1: Primers",
            "blocks": [
                raw_block("Fig.\u00a01. Overview", 1, page=2),
                raw_block("Table S1: Primers", 2, page=2),
            ],
        }
    ], total_pages=2))

    assert [block["type"] for block in all_blocks(clean)] == ["figure_caption", "table_caption"]


def test_table_caption_tail_becomes_table_text(tmp_path: Path) -> None:
    text = "Table 1: Strains used content strain name S1 OD600 1.0 2.0 3.0 4.0 5.0"
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 2, "text": text, "blocks": [raw_block(text, 1, page=2)]}
    ], total_pages=2))

    assert [block["type"] for block in all_blocks(clean)] == ["table_caption", "table_text"]


def test_image_block_preserved_and_not_in_page_text(tmp_path: Path) -> None:
    image = raw_block("", 1, block_type="image", bbox=[100, 100, 400, 300])
    image["metadata"] = {
        "xref": 7,
        "image_width": 640,
        "image_height": 480,
        "image_source": "page_get_images",
    }
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": "", "blocks": [image]}
    ]))

    block = all_blocks(clean)[0]
    assert block["type"] == "image"
    assert block["text"] == ""
    assert clean["pages"][0]["text"] == ""
    assert block["metadata"]["source_block_type"] == "image"
    assert block["metadata"]["xref"] == 7
    assert block["metadata"]["image_width"] == 640
    assert block["metadata"]["source_metadata"]["image_source"] == "page_get_images"


def test_layout_metadata_retention(tmp_path: Path) -> None:
    clean = run_clean(tmp_path, make_v4_doc([
        {"page": 1, "text": "Body.", "blocks": [raw_block("Body sentence remains.", 3, block_id="p1_b0003")]}
    ]))

    metadata = all_blocks(clean)[0]["metadata"]
    assert metadata["source_block_id"] == "p1_b0003"
    assert metadata["bbox"] == [72.0, 136.0, 240.0, 148.0]
    assert metadata["column"] == "L"
    assert metadata["reading_order"] == 3


def test_old_pages_text_fallback_still_works(tmp_path: Path) -> None:
    data = {
        "doc_id": "doc_old",
        "source_file": "doc_old.pdf",
        "total_pages": 1,
        "pages": [{"page": 1, "text": "## Introduction\n\nBody sentence remains."}],
    }
    clean = run_clean(tmp_path, data)

    assert [block["type"] for block in all_blocks(clean)] == ["section_heading", "paragraph"]


def test_reading_order_is_consumed_stably(tmp_path: Path) -> None:
    clean = run_clean(tmp_path, make_v4_doc([
        {
            "page": 1,
            "text": "A\nB\nC",
            "blocks": [
                raw_block("third sentence.", 3),
                raw_block("first sentence.", 1),
                raw_block("second sentence.", 2),
            ],
        }
    ]))

    assert [block["text"] for block in all_blocks(clean)] == [
        "first sentence.",
        "second sentence.",
        "third sentence.",
    ]
