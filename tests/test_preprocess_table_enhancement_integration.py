from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from scripts.ingestion.preprocess_and_chunk import Chunk, process_document, read_json_file
from src.synbio_rag.ingestion.table_enhancement import (
    TableEnhancementRunConfig,
    run_table_enhancement,
)


def _write_doc(input_dir: Path) -> None:
    blocks = [
        {
            "block_id": "b1",
            "type": "table_caption",
            "text": "Table 1. Production strains.",
            "page": 1,
            "section_path": ["Results"],
            "metadata": {"source_block_id": "raw_b1"},
        },
        {
            "block_id": "b2",
            "type": "paragraph",
            "text": "Strain  Plasmid  Product  Titer  10  20  30 mg/L",
            "page": 1,
            "section_path": ["Results"],
            "metadata": {"source_block_id": "raw_b2"},
        },
        {
            "block_id": "b3",
            "type": "paragraph",
            "text": "The following paragraph interprets the table in normal prose.",
            "page": 1,
            "section_path": ["Results"],
            "metadata": {"source_block_id": "raw_b3"},
        },
    ]
    data = {
        "doc_id": "doc1",
        "source_file": "doc1.pdf",
        "parser_stage": "parsed_clean_v4",
        "pages": [{"page": 1, "text": "\n".join(b["text"] for b in blocks), "blocks": blocks}],
    }
    (input_dir / "doc1.json").write_text(json.dumps(data), encoding="utf-8")


def test_enhanced_parsed_clean_table_related_reaches_table_focused_chunk(tmp_path: Path) -> None:
    input_dir = tmp_path / "parsed_clean"
    output_dir = tmp_path / "parsed_clean_table_enhanced"
    audit_dir = tmp_path / "audit"
    input_dir.mkdir()
    _write_doc(input_dir)

    run_table_enhancement(
        input_dir=input_dir,
        output_dir=output_dir,
        audit_dir=audit_dir,
        config=TableEnhancementRunConfig(),
    )
    doc = read_json_file(output_dir / "doc1.json")
    chunks, low_quality = process_document(
        doc,
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_chars=1,
        min_chunk_words=1,
        quality_threshold=0.0,
    )

    assert not low_quality
    table_chunk = next(chunk for chunk in chunks if chunk.contains_table_caption)
    assert "[TABLE CAPTION] Table 1. Production strains." in table_chunk.text
    assert "Strain  Plasmid  Product" in table_chunk.text
    assert any(meta.get("table_related") is True for meta in table_chunk.source_block_metadata)
    assert set(asdict(table_chunk).keys()) == {field.name for field in fields(Chunk)}


def test_chunk_dataclass_fields_are_unchanged() -> None:
    assert "table_related" not in {field.name for field in fields(Chunk)}
    assert "table_object" not in {field.name for field in fields(Chunk)}
    assert "figure_object" not in {field.name for field in fields(Chunk)}
