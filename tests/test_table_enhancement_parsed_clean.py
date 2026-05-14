from __future__ import annotations

import json
from pathlib import Path

from src.synbio_rag.ingestion.table_enhancement import (
    TableEnhancementRunConfig,
    run_table_enhancement,
)


def block(text: str, block_type: str, order: int, *, section_path: list[str] | None = None) -> dict:
    return {
        "block_id": f"b{order}",
        "type": block_type,
        "text": text,
        "page": 1,
        "section_path": section_path or ["Results"],
        "metadata": {"source_block_id": f"raw_b{order}", "reading_order": order},
    }


def write_doc(input_dir: Path, blocks: list[dict], doc_id: str = "doc1") -> Path:
    text = "\n\n".join(item.get("text", "") for item in blocks)
    data = {
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "parser_stage": "parsed_clean_v4",
        "pages": [{"page": 1, "text": text, "blocks": blocks}],
    }
    path = input_dir / f"{doc_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_caption_nearby_table_like_paragraph_marked(tmp_path: Path) -> None:
    input_dir = tmp_path / "parsed_clean"
    output_dir = tmp_path / "parsed_clean_table_enhanced"
    audit_dir = tmp_path / "audit"
    input_dir.mkdir()
    write_doc(input_dir, [
        block("Table 1. Strain performance.", "table_caption", 1),
        block("Strain  Plasmid  Titer  1.0  2.0  3.0 mg/L", "paragraph", 2),
        block("This paragraph showed a clear trend in the engineered strain and explains the result.", "paragraph", 3),
    ])

    result = run_table_enhancement(
        input_dir=input_dir,
        output_dir=output_dir,
        audit_dir=audit_dir,
        config=TableEnhancementRunConfig(),
    )

    enhanced = json.loads((output_dir / "doc1.json").read_text(encoding="utf-8"))
    caption = enhanced["pages"][0]["blocks"][0]
    table_like = enhanced["pages"][0]["blocks"][1]
    long_prose = enhanced["pages"][0]["blocks"][2]

    assert result.association_count == 1
    assert table_like["type"] == "paragraph"
    assert table_like["metadata"]["table_related"] is True
    assert table_like["metadata"]["table_related_type"] == "table_like_paragraph"
    assert table_like["metadata"]["table_association_rule"] == "caption_nearby_table_like"
    assert table_like["metadata"]["associated_table_caption_block_id"] == "b1"
    assert table_like["metadata"]["table_enhancement_enabled"] is True
    assert caption["metadata"]["associated_table_like_block_ids"] == ["b2"]
    assert caption["metadata"]["table_enhancement_associated_block_count"] == 1
    assert "table_related" not in long_prose["metadata"]


def test_disallowed_blocks_not_absorbed(tmp_path: Path) -> None:
    input_dir = tmp_path / "parsed_clean"
    output_dir = tmp_path / "parsed_clean_table_enhanced"
    audit_dir = tmp_path / "audit"
    input_dir.mkdir()
    write_doc(input_dir, [
        block("Table 2. Primers.", "table_caption", 1),
        block("Fig. 1. Pathway overview.", "figure_caption", 2),
        block("1. Smith, J. et al. Journal 10, 1-9.", "references", 3, section_path=["References"]),
        block("Correspondence: author@example.org", "metadata", 4),
        block("Primer  Sequence  Length  10  20  30 bp", "paragraph", 5),
    ])

    run_table_enhancement(
        input_dir=input_dir,
        output_dir=output_dir,
        audit_dir=audit_dir,
        config=TableEnhancementRunConfig(window_after_caption=5),
    )
    enhanced = json.loads((output_dir / "doc1.json").read_text(encoding="utf-8"))
    blocks = enhanced["pages"][0]["blocks"]

    assert "table_related" not in blocks[1]["metadata"]
    assert "table_related" not in blocks[2]["metadata"]
    assert "table_related" not in blocks[3]["metadata"]
    assert blocks[4]["metadata"]["table_related"] is True
