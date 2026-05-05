#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate enriched chunks produced from parsed_clean_v4."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTAMINATION_PATTERNS = {
    "journal_preproof": re.compile(r"journal\s+pre-proof|this\s+is\s+a\s+pdf\s+file\s+of\s+an\s+article", re.I),
    "correspondence": re.compile(r"\b(?:\*?\s*correspondence\s*:|to\s+whom\s+correspondence|correspondence\s+may\s+also)", re.I),
    "marginal_banner": re.compile(r"at\s+University\s+of\s+Hawaii\s+at\s+Manoa\s+Library\s+on\s+June\s+16,\s+2015", re.I),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def iter_clean_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for page in data.get("pages", []) or []
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def _preview(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _chunk_example(chunk: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "doc_id": chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "block_types": chunk.get("block_types", []),
        "text_preview": _preview(chunk.get("text", "")),
        "matched_reason": reason,
    }


def _clean_source_id(block: dict[str, Any]) -> str | None:
    metadata = block.get("metadata", {}) or {}
    return metadata.get("source_block_id") or block.get("block_id")


def analyze(clean_dir: Path, chunks_jsonl: Path) -> dict[str, Any]:
    chunks = load_chunks(chunks_jsonl)
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.get("doc_id", "")].append(chunk)

    clean_docs = [load_json(path) for path in sorted(clean_dir.glob("*.json"))]
    clean_by_doc = {doc.get("doc_id", Path(doc.get("source_file", "doc")).stem): doc for doc in clean_docs}

    clean_counts = Counter()
    clean_source_by_type: dict[str, set[str]] = defaultdict(set)
    for doc in clean_docs:
        doc_id = doc.get("doc_id", "")
        for block in iter_clean_blocks(doc):
            btype = block.get("type", "")
            clean_counts[btype] += 1
            source_id = _clean_source_id(block)
            if source_id:
                clean_source_by_type[btype].add(f"{doc_id}:{source_id}")

    chunk_type_counts = Counter()
    chunk_source_ids: set[str] = set()
    chunk_blocks_with_source = 0
    chunk_blocks_total = 0
    layout_with_metadata = 0
    layout_denominator = 0
    chunks_with_source_block_ids = 0
    chunks_missing_section_count = 0
    chunks_missing_page_count = 0
    total_text_chars = 0

    examples = {
        "metadata_in_chunk_text_examples": [],
        "references_in_chunk_text_examples": [],
        "caption_loss_examples": [],
        "table_text_loss_examples": [],
        "missing_source_block_id_examples": [],
        "suspicious_chunk_examples": [],
    }

    contamination_counts = Counter()
    false_table_text_body_sentence_count = 0
    body_sentence_table_pattern = re.compile(
        r"\[TABLE TEXT\]\s+at\s+a\s+compound\s+annual\s+growth\s+rate",
        re.I,
    )

    for chunk in chunks:
        text = chunk.get("text", "") or ""
        retrieval_text = chunk.get("retrieval_text", "") or ""
        total_text_chars += len(text)
        if chunk.get("source_block_ids"):
            chunks_with_source_block_ids += 1
        if not chunk.get("section"):
            chunks_missing_section_count += 1
        if chunk.get("page_start") is None or chunk.get("page_end") is None:
            chunks_missing_page_count += 1

        block_types = set(chunk.get("block_types", []) or [])
        chunk_type_counts.update(block_types)
        if chunk.get("contains_figure_caption"):
            chunk_type_counts["chunk_contains_figure_caption"] += 1
        if chunk.get("contains_table_caption"):
            chunk_type_counts["chunk_contains_table_caption"] += 1
        if chunk.get("contains_table_text"):
            chunk_type_counts["chunk_contains_table_text"] += 1

        if "metadata" in block_types or chunk.get("contains_metadata"):
            contamination_counts["metadata_in_chunk_text"] += 1
            if len(examples["metadata_in_chunk_text_examples"]) < 12:
                examples["metadata_in_chunk_text_examples"].append(_chunk_example(chunk, "metadata_block_type_in_chunk"))
        if "noise" in block_types or chunk.get("contains_noise"):
            contamination_counts["noise_in_chunk_text"] += 1
        if "image" in block_types or chunk.get("contains_image"):
            contamination_counts["image_in_chunk_text"] += 1
        if "references" in block_types or chunk.get("contains_references"):
            contamination_counts["references_in_chunk_text"] += 1
            if len(examples["references_in_chunk_text_examples"]) < 12:
                examples["references_in_chunk_text_examples"].append(_chunk_example(chunk, "references_block_type_in_chunk"))

        for label, pattern in CONTAMINATION_PATTERNS.items():
            if pattern.search(text) or pattern.search(retrieval_text):
                contamination_counts[f"{label}_in_chunk_text"] += 1
                if len(examples["suspicious_chunk_examples"]) < 12:
                    examples["suspicious_chunk_examples"].append(_chunk_example(chunk, label))

        if body_sentence_table_pattern.search(text):
            false_table_text_body_sentence_count += 1
            if len(examples["suspicious_chunk_examples"]) < 12:
                examples["suspicious_chunk_examples"].append(_chunk_example(chunk, "false_table_text_body_sentence"))

        source_ids = chunk.get("source_block_ids", []) or []
        chunk_source_ids.update(f"{chunk.get('doc_id', '')}:{source_id}" for source_id in source_ids)
        for meta in chunk.get("source_block_metadata", []) or []:
            chunk_blocks_total += 1
            if meta.get("source_block_id"):
                chunk_blocks_with_source += 1
            elif len(examples["missing_source_block_id_examples"]) < 12:
                examples["missing_source_block_id_examples"].append(_chunk_example(chunk, "missing_source_block_id"))
            if meta.get("type") not in {"metadata", "noise", "image"}:
                layout_denominator += 1
                if meta.get("bbox") is not None and meta.get("column") is not None and meta.get("reading_order") is not None:
                    layout_with_metadata += 1

    retained_by_type = {
        btype: len(clean_source_by_type[btype] & chunk_source_ids)
        for btype in ("figure_caption", "table_caption", "table_text")
    }
    caption_loss = (
        clean_counts["figure_caption"] - retained_by_type["figure_caption"]
        + clean_counts["table_caption"] - retained_by_type["table_caption"]
    )
    table_text_loss = clean_counts["table_text"] - retained_by_type["table_text"]

    for btype in ("figure_caption", "table_caption"):
        missing = sorted(clean_source_by_type[btype] - chunk_source_ids)
        for source_id in missing[:8]:
            examples["caption_loss_examples"].append({"source_block_id": source_id, "type": btype})
    for source_id in sorted(clean_source_by_type["table_text"] - chunk_source_ids)[:8]:
        examples["table_text_loss_examples"].append({"source_block_id": source_id, "type": "table_text"})

    source_retention_den = sum(
        len(clean_source_by_type[btype])
        for btype in ("title", "section_heading", "subsection_heading", "paragraph", "figure_caption", "table_caption", "table_text")
    )
    source_retention = len(chunk_source_ids) / source_retention_den if source_retention_den else 1.0
    block_type_coverage = (
        sum(retained_by_type.values())
        / max(clean_counts["figure_caption"] + clean_counts["table_caption"] + clean_counts["table_text"], 1)
    )
    layout_retention = layout_with_metadata / layout_denominator if layout_denominator else 1.0

    risks = []
    if source_retention < 0.95:
        risks.append("source_block_id_retention_rate_lt_0.95")
    if layout_retention < 0.95:
        risks.append("layout_metadata_retention_rate_lt_0.95")
    for key in (
        "metadata_in_chunk_text",
        "noise_in_chunk_text",
        "image_in_chunk_text",
        "references_in_chunk_text",
        "journal_preproof_in_chunk_text",
        "correspondence_in_chunk_text",
        "marginal_banner_in_chunk_text",
    ):
        if contamination_counts[key] > 0:
            risks.append(key)
    if caption_loss > 0:
        risks.append("caption_loss")
    if table_text_loss > max(2, int(clean_counts["table_text"] * 0.05)):
        risks.append("table_text_loss")
    if false_table_text_body_sentence_count > 0:
        risks.append("false_table_text_body_sentence")

    summary = {
        "doc_count": len(clean_docs),
        "chunk_count": len(chunks),
        "total_chunk_text_chars": total_text_chars,
        "chunk_with_source_block_ids_count": chunks_with_source_block_ids,
        "source_block_id_retention_rate": round(source_retention, 4),
        "block_type_coverage_rate": round(block_type_coverage, 4),
        "layout_metadata_retention_rate": round(layout_retention, 4),
        "chunks_with_figure_caption": chunk_type_counts["chunk_contains_figure_caption"],
        "chunks_with_table_caption": chunk_type_counts["chunk_contains_table_caption"],
        "chunks_with_table_text": chunk_type_counts["chunk_contains_table_text"],
        "raw_clean_figure_caption_count": clean_counts["figure_caption"],
        "raw_clean_table_caption_count": clean_counts["table_caption"],
        "raw_clean_table_text_count": clean_counts["table_text"],
        "chunk_figure_caption_count": retained_by_type["figure_caption"],
        "chunk_table_caption_count": retained_by_type["table_caption"],
        "chunk_table_text_count": retained_by_type["table_text"],
        "caption_loss_count": caption_loss,
        "table_text_loss_count": table_text_loss,
        "metadata_in_chunk_text_count": contamination_counts["metadata_in_chunk_text"],
        "noise_in_chunk_text_count": contamination_counts["noise_in_chunk_text"],
        "image_in_chunk_text_count": contamination_counts["image_in_chunk_text"],
        "references_in_chunk_text_count": contamination_counts["references_in_chunk_text"],
        "journal_preproof_in_chunk_text_count": contamination_counts["journal_preproof_in_chunk_text"],
        "correspondence_in_chunk_text_count": contamination_counts["correspondence_in_chunk_text"],
        "marginal_banner_in_chunk_text_count": contamination_counts["marginal_banner_in_chunk_text"],
        "false_table_text_body_sentence_count": false_table_text_body_sentence_count,
        "chunks_missing_section_count": chunks_missing_section_count,
        "chunks_missing_page_count": chunks_missing_page_count,
        "potential_regression_doc_count": len({c.get("doc_id") for c in chunks if risks}),
        "potential_regressions": risks,
        "conclusion": "pass" if not risks else "fail",
    }
    return {
        "summary": summary,
        "examples": examples,
        "clean_dir": str(clean_dir),
        "chunks_jsonl": str(chunks_jsonl),
        "doc_ids": sorted(clean_by_doc),
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = ["# v4 Phase3a Chunk Validation", ""]
    lines.append(f"- clean_dir: `{result['clean_dir']}`")
    lines.append(f"- chunks_jsonl: `{result['chunks_jsonl']}`")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    for label, examples in result["examples"].items():
        lines.append(f"### {label}")
        if not examples:
            lines.append("- none")
        else:
            for ex in examples[:20]:
                lines.append(f"- `{ex.get('doc_id', '')}` `{ex.get('chunk_id') or ex.get('source_block_id')}`: {ex.get('text_preview', ex.get('type', ''))}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v4 enriched chunks")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    result = analyze(Path(args.clean_dir), Path(args.chunks_jsonl))

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")

    print(f"doc_count={result['summary']['doc_count']}")
    print(f"chunk_count={result['summary']['chunk_count']}")
    print(f"conclusion={result['summary']['conclusion']}")
    print(f"report={report_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
