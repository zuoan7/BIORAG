#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check structural field contract for chunks.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STRUCTURE_FIELDS = [
    "block_types",
    "source_block_ids",
    "block_ids",
    "evidence_types",
    "page_numbers",
    "layout_columns",
    "reading_order_span",
    "bbox_span",
    "source_block_metadata",
    "contains_table_text",
    "contains_table_caption",
    "contains_figure_caption",
    "contains_image",
    "contains_references",
    "contains_metadata",
    "contains_noise",
    "parser_stage",
]

BOOL_FIELDS = [
    "contains_table_text",
    "contains_table_caption",
    "contains_figure_caption",
    "contains_image",
    "contains_references",
    "contains_metadata",
    "contains_noise",
]


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if value is False:
        return False
    if value == "":
        return False
    if value == [] or value == {}:
        return False
    if isinstance(value, dict):
        return any(_is_nonempty(item) for item in value.values())
    if isinstance(value, list):
        return any(_is_nonempty(item) for item in value)
    return True


def _preview(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def summarize(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    present = Counter()
    nonempty = Counter()
    bool_counts = {field: Counter() for field in BOOL_FIELDS}
    block_type_counts = Counter()

    for chunk in chunks:
        for field in STRUCTURE_FIELDS:
            if field in chunk:
                present[field] += 1
            if _is_nonempty(chunk.get(field)):
                nonempty[field] += 1

        for field in BOOL_FIELDS:
            if field not in chunk:
                bool_counts[field]["MISSING"] += 1
            elif chunk.get(field) is True:
                bool_counts[field]["True"] += 1
            elif chunk.get(field) is False:
                bool_counts[field]["False"] += 1
            else:
                bool_counts[field]["OTHER"] += 1

        for block_type in chunk.get("block_types") or []:
            block_type_counts[block_type] += 1

    table_examples = [
        chunk for chunk in chunks
        if chunk.get("contains_table_caption") is True or "table_caption" in (chunk.get("block_types") or [])
    ][:5]
    figure_examples = [
        chunk for chunk in chunks
        if chunk.get("contains_figure_caption") is True or "figure_caption" in (chunk.get("block_types") or [])
    ][:5]

    return {
        "total_chunks": len(chunks),
        "doc_count": len({chunk.get("doc_id") for chunk in chunks}),
        "present": present,
        "nonempty": nonempty,
        "bool_counts": bool_counts,
        "block_type_counts": block_type_counts,
        "table_examples": table_examples,
        "figure_examples": figure_examples,
    }


def print_examples(title: str, examples: list[dict[str, Any]]) -> None:
    print(f"\n{title}:")
    if not examples:
        print("  (none)")
        return
    for chunk in examples:
        payload = {
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "section": chunk.get("section"),
            "block_types": chunk.get("block_types"),
            "source_block_ids": chunk.get("source_block_ids"),
            "block_ids": chunk.get("block_ids"),
            "page_numbers": chunk.get("page_numbers"),
            "source_block_metadata": (chunk.get("source_block_metadata") or [])[:2],
            "contains_table_caption": chunk.get("contains_table_caption"),
            "contains_figure_caption": chunk.get("contains_figure_caption"),
            "text_preview": _preview(chunk.get("text", "")),
        }
        print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check structural contract for chunks.jsonl")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/paper_round1/chunks/chunks.jsonl"),
        help="Path to chunks.jsonl",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    summary = summarize(chunks)
    total = summary["total_chunks"] or 1

    print(f"total_chunks: {summary['total_chunks']}")
    print(f"doc_count: {summary['doc_count']}")

    print("\nfield_coverage:")
    for field in STRUCTURE_FIELDS:
        present = summary["present"][field]
        nonempty = summary["nonempty"][field]
        print(
            f"  {field}: present={present}/{summary['total_chunks']} ({present / total:.2%}), "
            f"non_empty={nonempty}/{summary['total_chunks']} ({nonempty / total:.2%})"
        )

    print("\ncontains_distribution:")
    for field in BOOL_FIELDS:
        counts = summary["bool_counts"][field]
        print(
            f"  {field}: True={counts['True']} False={counts['False']} "
            f"MISSING={counts['MISSING']}"
        )

    print("\nblock_types_top30:")
    for block_type, count in summary["block_type_counts"].most_common(30):
        print(f"  {block_type}: {count}")

    print_examples("table_caption_examples", summary["table_examples"])
    print_examples("figure_caption_examples", summary["figure_examples"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
