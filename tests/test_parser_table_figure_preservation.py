from __future__ import annotations

import json
from pathlib import Path

from scripts.ingestion.phase4_shadow_table_figure_parse import (
    append_shadow_block,
    has_numeric,
    protected_content_type,
    rebuild_page_text_from_dicts,
    source_label,
)


def make_shadow_doc() -> dict:
    return {
        "doc_id": "doc_test",
        "source_file": "doc_test.pdf",
        "total_pages": 1,
        "parser_stage": "parsed_clean_v4",
        "cleaning_stage": "document_structure_clean_v5_adapter",
        "schema_version": "parsed_clean_v5_compatible",
        "block_contract": {
            "allowed_types": [
                "title",
                "section_heading",
                "subsection_heading",
                "paragraph",
                "figure_caption",
                "table_caption",
                "table_text",
                "references",
                "metadata",
                "noise",
                "image",
            ],
        },
        "pages": [
            {
                "page": 1,
                "text": "",
                "blocks": [
                    {
                        "block_id": "p1_b0001",
                        "type": "paragraph",
                        "text": "Materials and methods body text.",
                        "section_path": [],
                        "page": 1,
                        "metadata": {},
                    }
                ],
            }
        ],
    }


def test_markdown_table_block_is_preserved_in_shadow_page_text() -> None:
    doc = make_shadow_doc()
    table = "| Step | Yield (%) |\n| --- | --- |\n| Superdex 200 | 11.4 |"
    assert append_shadow_block(doc, 1, "table_text", table, "table", {"source_label": "Table 1"}, 1)
    doc["pages"][0]["text"] = rebuild_page_text_from_dicts(doc["pages"][0]["blocks"])

    assert "[TABLE]" in doc["pages"][0]["text"]
    assert "Superdex 200" in doc["pages"][0]["text"]
    assert "11.4" in doc["pages"][0]["text"]


def test_table_caption_metadata_is_backwards_compatible() -> None:
    doc = make_shadow_doc()
    assert append_shadow_block(
        doc,
        1,
        "table_caption",
        "Table 2. Primers used in this study.",
        "table",
        {"source_label": "Table 2"},
        1,
    )
    block = doc["pages"][0]["blocks"][-1]

    assert block["type"] == "table_caption"
    assert block["metadata"]["content_type"] == "table"
    assert block["metadata"]["source_label"] == "Table 2"


def test_figure_caption_is_preserved_with_content_type() -> None:
    doc = make_shadow_doc()
    assert append_shadow_block(
        doc,
        1,
        "figure_caption",
        "Figure 5. MoSlp1 accumulated in the mutant and secretion was reduced.",
        "figure_caption",
        {"source_label": "Figure 5"},
        1,
    )
    block = doc["pages"][0]["blocks"][-1]

    assert block["type"] == "figure_caption"
    assert block["metadata"]["content_type"] == "figure_caption"
    assert block["metadata"]["source_label"] == "Figure 5"


def test_numeric_unit_line_is_marked_as_numeric_text() -> None:
    text = "The NFS-60 cell line was grown with 8% horse serum and 2% fetal bovine serum."

    assert has_numeric(text)
    assert protected_content_type("paragraph", text) == "numeric_text"


def test_primer_sequence_like_line_is_marked_as_primer_sequence() -> None:
    text = "ADH900 forward primer ATGTCTGTGATGAAAGCCCTC was used for ADH900 gene."

    assert protected_content_type("table_text", text) == "table"
    assert protected_content_type("paragraph", text) == "primer_sequence"


def test_table_figure_markdown_heading_is_not_required_for_source_label() -> None:
    assert source_label("### Table S2. Primers used for ADH genes") == "Table S2"
    assert source_label("## Figure 6. MoSlp1 secretion assay") == "Figure 6"


def test_metadata_content_type_and_has_numeric_are_correct() -> None:
    doc = make_shadow_doc()
    assert append_shadow_block(
        doc,
        1,
        "paragraph",
        "The enzyme was purified 19.3-fold with a yield of 11.4%.",
        "numeric_text",
        {},
        1,
    )
    block = doc["pages"][0]["blocks"][-1]

    assert block["metadata"]["content_type"] == "numeric_text"
    assert block["metadata"]["has_numeric"] is True


def test_shadow_output_does_not_change_main_schema(tmp_path: Path) -> None:
    doc = make_shadow_doc()
    before_top_keys = set(doc.keys())
    before_block_keys = set(doc["pages"][0]["blocks"][0].keys())
    assert append_shadow_block(
        doc,
        1,
        "table_text",
        "ADH6 ATGGCGTACCCAGACACC",
        "primer_sequence",
        {"source_label": "Table 2"},
        1,
    )
    path = tmp_path / "doc_test.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert set(loaded.keys()) == before_top_keys
    assert set(loaded["pages"][0]["blocks"][-1].keys()) == before_block_keys
