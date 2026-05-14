from __future__ import annotations

import json
from pathlib import Path

from src.synbio_rag.ingestion.table_enhancement import (
    TableEnhancementRunConfig,
    run_table_enhancement,
)


def _doc(doc_id: str) -> dict:
    blocks = [
        {
            "block_id": "caption",
            "type": "table_caption",
            "text": "Table 1. Strains.",
            "page": 1,
            "section_path": ["Results"],
            "metadata": {"source_block_id": "raw_caption"},
        },
        {
            "block_id": "row",
            "type": "paragraph",
            "text": "Strain  Plasmid  Product  1.0  2.0  3.0 mg/L",
            "page": 1,
            "section_path": ["Results"],
            "metadata": {"source_block_id": "raw_row"},
        },
    ]
    return {
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "parser_stage": "parsed_clean_v4",
        "pages": [{"page": 1, "text": "\n".join(b["text"] for b in blocks), "blocks": blocks}],
    }


def test_audit_outputs_are_written(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    audit_dir = tmp_path / "audit"
    input_dir.mkdir()
    (input_dir / "doc1.json").write_text(json.dumps(_doc("doc1")), encoding="utf-8")

    result = run_table_enhancement(
        input_dir=input_dir,
        output_dir=output_dir,
        audit_dir=audit_dir,
        config=TableEnhancementRunConfig(),
    )

    assert result.association_count == 1
    assert (audit_dir / "association_audit.csv").exists()
    assert (audit_dir / "doc_level_stats.csv").exists()
    assert (audit_dir / "summary.md").exists()
    assert (audit_dir / "false_positive_review.md").exists()
    assert "safety_gate_passed" in (audit_dir / "summary.md").read_text(encoding="utf-8")


def test_suspicious_cases_can_be_recorded(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    audit_dir = tmp_path / "audit"
    input_dir.mkdir()
    (input_dir / "doc1.json").write_text(json.dumps(_doc("doc1")), encoding="utf-8")

    result = run_table_enhancement(
        input_dir=input_dir,
        output_dir=output_dir,
        audit_dir=audit_dir,
        config=TableEnhancementRunConfig(),
    )

    review = (audit_dir / "false_positive_review.md").read_text(encoding="utf-8")
    assert result.suspicious_docs == ["doc1"]
    assert "Suspicious Docs" in review
    assert "doc1" in review
