#!/usr/bin/env python3
"""Build the approved Phase 4E-3 retrieval-only eval set.

This script only copies approved records from the candidate JSONL according to
sample IDs listed in the manual review markdown. It does not run retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SAMPLE_ID_RE = re.compile(r"^\s*-\s+`(p4e3_(?:table|figure|normal(?:_supplement)?)_\d+)`", re.M)
DETAIL_RE = re.compile(
    r"^\s*-\s+`(?P<sample_id>p4e3_(?:table|figure|normal(?:_supplement)?)_\d+)`\s*\n(?P<body>.*?)(?=^\s*-\s+`p4e3_|\Z)",
    re.M | re.S,
)
FIELD_RE = re.compile(r"^\s+-\s+(?P<key>[a-z_ ]+):\s*(?P<value>.*)$", re.M)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_sample_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    sample_ids = SAMPLE_ID_RE.findall(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ordered.append(sample_id)
    return ordered


def parse_markdown_details(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    details: dict[str, dict[str, str]] = {}
    for match in DETAIL_RE.finditer(text):
        sample_id = match.group("sample_id")
        fields: dict[str, str] = {}
        for field_match in FIELD_RE.finditer(match.group("body")):
            key = field_match.group("key")
            value = field_match.group("value").strip()
            if value.startswith("`") and value.endswith("`"):
                value = value[1:-1]
            fields[key] = value
        details[sample_id] = fields
    return details


def approved_row(row: dict[str, Any], eval_group: str, include: bool) -> dict[str, Any]:
    item = dict(row)
    item.update(
        {
            "approved": True,
            "approval_source": "phase4e3_review_pack",
            "eval_group": eval_group,
            "include_in_main_denominator": include,
        }
    )
    return item


def duplicate_items(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def load_chunks_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict) and item.get("chunk_id"):
                chunks[str(item["chunk_id"])] = item
    return chunks


def page_numbers(chunk: dict[str, Any]) -> list[int]:
    start = chunk.get("page_start")
    end = chunk.get("page_end")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return list(range(start, end + 1))
    if isinstance(start, int):
        return [start]
    return []


def infer_evidence_types(chunk: dict[str, Any]) -> list[str]:
    evidence_types = chunk.get("evidence_types")
    if isinstance(evidence_types, list):
        return [str(item) for item in evidence_types]
    inferred: list[str] = []
    if chunk.get("contains_table_caption"):
        inferred.append("table_caption")
    if chunk.get("contains_table_text"):
        inferred.append("table_text")
    if chunk.get("contains_figure_caption"):
        inferred.append("figure_caption")
    return inferred or ["paragraph"]


def infer_block_types(chunk: dict[str, Any]) -> list[str]:
    block_types = chunk.get("block_types")
    if isinstance(block_types, list):
        return [str(item) for item in block_types]
    return infer_evidence_types(chunk)


def build_missing_supplement_row(
    sample_id: str,
    fields: dict[str, str],
    chunk_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_chunk_id = fields.get("target_chunk_id", "")
    chunk = chunk_by_id.get(target_chunk_id)
    if not chunk:
        raise ValueError(f"Cannot build missing supplement row without chunk: {sample_id} {target_chunk_id}")
    return {
        "sample_id": sample_id,
        "sample_type": "normal",
        "query": fields.get("query", ""),
        "target_doc_id": fields.get("target_doc_id") or chunk.get("doc_id", ""),
        "target_chunk_id": target_chunk_id,
        "target_source_file": chunk.get("source_file", ""),
        "target_page_numbers": page_numbers(chunk),
        "target_section": chunk.get("section", ""),
        "target_evidence_types": infer_evidence_types(chunk),
        "target_block_types": infer_block_types(chunk),
        "target_text_preview": fields.get("target_preview") or str(chunk.get("text", ""))[:500],
        "target_caption": "",
        "anchor_terms": [],
        "query_style": "normal_supplement_manual",
        "caption_token_overlap_ratio": 0.0,
        "longest_common_token_span": 0,
        "caption_copy_risk": "not_applicable",
        "eligibility_reason": fields.get("approve reason", "approved normal supplement"),
        "review_status": "approved",
        "review_notes": "Phase 4E-3N normal supplement; reconstructed from approved markdown and existing chunks because it was absent from candidate_eval_set.jsonl.",
    }


def write_summary(
    path: Path,
    eval_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    expected_counts: dict[str, int],
) -> None:
    group_counts = Counter(str(row.get("eval_group", "")) for row in eval_rows)
    sample_ids = [str(row.get("sample_id", "")) for row in eval_rows]
    target_chunk_ids = [str(row.get("target_chunk_id", "")) for row in eval_rows]
    doc_counts = Counter(str(row.get("target_doc_id", "")) for row in eval_rows)
    duplicates_sample = duplicate_items(sample_ids)
    duplicates_chunk = duplicate_items(target_chunk_ids)
    balanced = all(group_counts.get(group, 0) == expected for group, expected in expected_counts.items())

    lines = [
        "# Phase 4E-3 Approved Eval Set Summary",
        "",
        f"- table count: {group_counts.get('table', 0)}",
        f"- figure count: {group_counts.get('figure', 0)}",
        f"- normal count: {group_counts.get('normal', 0)}",
        f"- sanity anchor count: {len(sanity_rows)}",
        f"- duplicate sample_id: {bool(duplicates_sample)}",
        f"- duplicate sample_id values: `{duplicates_sample}`",
        f"- duplicate target_chunk_id: {bool(duplicates_chunk)}",
        f"- duplicate target_chunk_id values: `{duplicates_chunk}`",
        f"- satisfies table=30 figure=30 normal=30: {balanced}",
        "",
        "## Per-doc Sample Distribution",
        "",
    ]
    for doc_id, count in sorted(doc_counts.items()):
        lines.append(f"- `{doc_id}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved_table_md", required=True, type=Path)
    parser.add_argument("--approved_figure_md", required=True, type=Path)
    parser.add_argument("--approved_normal_md", required=True, type=Path)
    parser.add_argument("--candidate_jsonl", required=True, type=Path)
    parser.add_argument("--chunks_jsonl", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    rows_by_id = {row.get("sample_id"): row for row in load_jsonl(args.candidate_jsonl)}
    chunk_by_id = load_chunks_by_id(args.chunks_jsonl)
    groups = {
        "table": extract_sample_ids(args.approved_table_md),
        "figure": extract_sample_ids(args.approved_figure_md),
        "normal": extract_sample_ids(args.approved_normal_md),
    }
    markdown_details = {}
    for path in (args.approved_table_md, args.approved_figure_md, args.approved_normal_md):
        markdown_details.update(parse_markdown_details(path))
    missing = {
        group: [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
        for group, sample_ids in groups.items()
    }
    unsupported_missing = {
        group: [
            sample_id
            for sample_id in sample_ids
            if not (group == "normal" and sample_id.startswith("p4e3_normal_supplement_"))
        ]
        for group, sample_ids in missing.items()
    }
    unsupported_missing = {group: sample_ids for group, sample_ids in unsupported_missing.items() if sample_ids}
    if unsupported_missing:
        raise ValueError(f"Approved sample IDs missing from candidate JSONL: {unsupported_missing}")
    if missing.get("normal") and not chunk_by_id:
        raise ValueError("--chunks_jsonl is required to reconstruct normal supplement rows missing from candidate JSONL")

    eval_rows: list[dict[str, Any]] = []
    for group in ("table", "figure", "normal"):
        for sample_id in groups[group]:
            row = rows_by_id.get(sample_id)
            if row is None:
                row = build_missing_supplement_row(sample_id, markdown_details.get(sample_id, {}), chunk_by_id)
            eval_rows.append(approved_row(row, group, True))

    sanity_rows = [
        approved_row(row, "sanity_anchor", False)
        for row in rows_by_id.values()
        if row.get("sample_type") == "sanity_anchor" and row.get("target_doc_id") == "doc_0367"
    ]
    sanity_rows.sort(key=lambda row: str(row.get("sample_id", "")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "eval_set.jsonl", eval_rows)
    write_jsonl(args.output_dir / "sanity_anchor.jsonl", sanity_rows)
    write_summary(
        args.output_dir / "eval_set_summary.md",
        eval_rows,
        sanity_rows,
        {"table": 30, "figure": 30, "normal": 30},
    )

    counts = Counter(str(row.get("eval_group", "")) for row in eval_rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "counts": dict(sorted(counts.items())),
                "sanity_anchor_count": len(sanity_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
