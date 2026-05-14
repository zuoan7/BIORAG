#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Phase 5C-1 table-like paragraph preservation pilot outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MAIN_CHUNK_FIELDS = [
    "chunk_id",
    "doc_id",
    "source_file",
    "title",
    "section",
    "page_start",
    "page_end",
    "chunk_index",
    "token_count",
    "text",
    "retrieval_text",
    "quality_score",
    "section_path",
    "block_types",
    "source_block_ids",
    "block_ids",
    "evidence_types",
    "page_numbers",
    "layout_columns",
    "reading_order_span",
    "bbox_span",
    "source_block_metadata",
    "excluded_block_counts",
    "contains_figure_caption",
    "contains_table_caption",
    "contains_table_text",
    "contains_references",
    "contains_metadata",
    "contains_noise",
    "contains_image",
    "parser_stage",
]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def preview(text: Any, limit: int = 220) -> str:
    return normalize_text(text)[:limit]


def read_selected_docs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["doc_id"].strip() for row in rows if row.get("doc_id", "").strip()]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                chunks.append(item)
    return chunks


def iter_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for page in data.get("pages", []) or []
        if isinstance(page, dict)
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def block_id(block: dict[str, Any]) -> str:
    metadata = block.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    value = block.get("block_id") or block.get("id") or metadata.get("source_block_id")
    return str(value) if value is not None else ""


def page_value(block: dict[str, Any]) -> int | None:
    value = block.get("page") or block.get("page_number")
    if value is None:
        metadata = block.get("metadata", {}) or {}
        if isinstance(metadata, dict):
            value = metadata.get("page") or metadata.get("page_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def source_meta_table_related(chunk: dict[str, Any]) -> bool:
    return any(
        isinstance(meta, dict) and meta.get("table_related") is True
        for meta in chunk.get("source_block_metadata", []) or []
    )


def chunk_schema(chunks: list[dict[str, Any]]) -> set[str]:
    schema: set[str] = set()
    for chunk in chunks:
        schema.update(str(key) for key in chunk.keys())
    return schema


def is_table_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption") or chunk.get("contains_table_text"))


def is_caption_only_table(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption") and not chunk.get("contains_table_text"))


def doc_chunk_stats(chunks: list[dict[str, Any]], selected_doc_ids: set[str]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        if doc_id in selected_doc_ids:
            grouped[doc_id].append(chunk)
    for doc_id in selected_doc_ids:
        doc_chunks = grouped.get(doc_id, [])
        stats[doc_id] = {
            "chunk_count": len(doc_chunks),
            "table_focused_count": sum(1 for chunk in doc_chunks if is_table_focused(chunk)),
            "caption_only_table_count": sum(1 for chunk in doc_chunks if is_caption_only_table(chunk)),
            "table_related_chunk_count": sum(1 for chunk in doc_chunks if source_meta_table_related(chunk)),
            "table_focused_related_chunk_count": sum(
                1 for chunk in doc_chunks if is_table_focused(chunk) and source_meta_table_related(chunk)
            ),
            "paragraph_chunk_count": sum(1 for chunk in doc_chunks if "paragraph" in (chunk.get("block_types") or [])),
            "token_count": sum(int(chunk.get("token_count") or 0) for chunk in doc_chunks),
        }
    return stats


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def association_rows(enhanced_parsed_clean: Path, selected_docs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    processed_docs = 0
    failed_docs: list[str] = []
    confidence_counts: Counter[str] = Counter()

    for doc_id in selected_docs:
        path = enhanced_parsed_clean / f"{doc_id}.json"
        if not path.exists():
            failed_docs.append(doc_id)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - audit should continue and report.
            failed_docs.append(doc_id)
            continue
        processed_docs += 1
        blocks = iter_blocks(data)
        by_id = {block_id(block): (index, block) for index, block in enumerate(blocks) if block_id(block)}
        captions = [(index, block) for index, block in enumerate(blocks) if block.get("type") == "table_caption"]

        for caption_index, caption in captions:
            caption_meta = caption.get("metadata", {}) or {}
            if not isinstance(caption_meta, dict):
                caption_meta = {}
            accepted_ids = [
                str(value)
                for value in caption_meta.get("associated_table_like_block_ids", []) or []
                if value is not None
            ]
            nearby_indexes = list(range(max(0, caption_index - 1), caption_index)) + list(
                range(caption_index + 1, min(len(blocks), caption_index + 6))
            )
            seen: set[str] = set()
            for associated_id in accepted_ids:
                if associated_id not in by_id:
                    continue
                block_index, block = by_id[associated_id]
                seen.add(associated_id)
                meta = block.get("metadata", {}) or {}
                if not isinstance(meta, dict):
                    meta = {}
                confidence = str(meta.get("association_confidence") or "")
                if confidence:
                    confidence_counts[confidence] += 1
                rows.append(make_association_row(
                    doc_id,
                    caption,
                    block,
                    caption_index,
                    block_index,
                    "accepted",
                    "",
                    confidence,
                    meta.get("phase5c1_rule_hits", []) or [],
                ))

            for block_index in nearby_indexes:
                block = blocks[block_index]
                candidate_id = block_id(block)
                if candidate_id in seen:
                    continue
                status = "rejected"
                reason = reject_reason(block)
                rows.append(make_association_row(
                    doc_id,
                    caption,
                    block,
                    caption_index,
                    block_index,
                    status,
                    reason,
                    "",
                    [],
                ))

    meta = {
        "processed_docs": processed_docs,
        "failed_docs": failed_docs,
        "confidence_counts": dict(confidence_counts),
    }
    return rows, meta


def reject_reason(block: dict[str, Any]) -> str:
    btype = str(block.get("type") or "unknown")
    if btype in {"figure_caption", "table_caption", "references", "metadata", "noise", "image", "section_heading", "title"}:
        return f"excluded_type:{btype}"
    text = normalize_text(block.get("text", ""))
    if not text:
        return "empty_text"
    if len(text.split()) >= 45 and re.search(r"[.!?]\s+[A-Z]", text):
        return "normal_prose_shape"
    return "no_accepted_table_related_metadata"


def make_association_row(
    doc_id: str,
    caption: dict[str, Any],
    block: dict[str, Any],
    caption_index: int,
    block_index: int,
    status: str,
    reject: str,
    confidence: str,
    rule_hits: Any,
) -> dict[str, Any]:
    caption_page = page_value(caption)
    block_page = page_value(block)
    if isinstance(rule_hits, list):
        rule_hits_text = ";".join(str(item) for item in rule_hits)
    else:
        rule_hits_text = str(rule_hits or "")
    return {
        "doc_id": doc_id,
        "table_caption_block_id": block_id(caption),
        "associated_block_id": block_id(block),
        "associated_block_type": str(block.get("type") or "unknown"),
        "association_confidence": confidence,
        "rule_hits": rule_hits_text,
        "associated_text_preview": preview(block.get("text", "")),
        "caption_text_preview": preview(caption.get("text", "")),
        "page_distance": abs(block_page - caption_page) if block_page is not None and caption_page is not None else "",
        "block_distance": abs(block_index - caption_index),
        "accepted_or_rejected": status,
        "reject_reason": reject,
    }


def write_false_positive_review(path: Path, rows: list[dict[str, Any]]) -> None:
    accepted = [row for row in rows if row["accepted_or_rejected"] == "accepted"]
    rejected = [row for row in rows if row["accepted_or_rejected"] == "rejected"]
    uncertain = [
        row for row in rows
        if row.get("association_confidence") == "low"
        or row.get("reject_reason") in {"no_accepted_table_related_metadata", "normal_prose_shape"}
    ]

    lines = ["# Phase 5C-1 False Positive Review", ""]
    sections = [
        ("Accepted Associations", accepted[:20]),
        ("Rejected Nearby Blocks", rejected[:20]),
        ("Uncertain Cases", uncertain[:10]),
    ]
    for title, items in sections:
        lines.extend([f"## {title}", ""])
        if not items:
            lines.extend(["- none", ""])
            continue
        for row in items:
            reason = row.get("reject_reason") or row.get("association_confidence") or "accepted"
            lines.extend([
                f"- `{row['doc_id']}` `{row['table_caption_block_id']}` -> `{row['associated_block_id']}` ({row['associated_block_type']})",
                f"  - decision: {row['accepted_or_rejected']} / {reason}",
                f"  - hits: {row.get('rule_hits', '')}",
                f"  - caption: {row.get('caption_text_preview', '')}",
                f"  - block: {row.get('associated_text_preview', '')}",
            ])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 5C-1 table preservation pilot.")
    parser.add_argument("--baseline_chunks", required=True)
    parser.add_argument("--enhanced_chunks", required=True)
    parser.add_argument("--enhanced_parsed_clean", required=True)
    parser.add_argument("--selected_docs", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_docs = read_selected_docs(Path(args.selected_docs))
    selected_set = set(selected_docs)
    baseline_chunks = load_chunks(Path(args.baseline_chunks))
    enhanced_chunks = load_chunks(Path(args.enhanced_chunks))
    baseline_stats = doc_chunk_stats(baseline_chunks, selected_set)
    enhanced_stats = doc_chunk_stats(enhanced_chunks, selected_set)

    base_schema = chunk_schema([chunk for chunk in baseline_chunks if chunk.get("doc_id") in selected_set])
    enhanced_schema = chunk_schema(enhanced_chunks)
    expected_schema = set(MAIN_CHUNK_FIELDS)
    schema_same = base_schema == enhanced_schema == expected_schema

    association_audit, association_meta = association_rows(Path(args.enhanced_parsed_clean), selected_docs)
    write_csv(
        output_dir / "association_audit.csv",
        association_audit,
        [
            "doc_id",
            "table_caption_block_id",
            "associated_block_id",
            "associated_block_type",
            "association_confidence",
            "rule_hits",
            "associated_text_preview",
            "caption_text_preview",
            "page_distance",
            "block_distance",
            "accepted_or_rejected",
            "reject_reason",
        ],
    )

    rows: list[dict[str, Any]] = []
    for doc_id in selected_docs:
        base = baseline_stats[doc_id]
        enh = enhanced_stats[doc_id]
        rows.append({
            "doc_id": doc_id,
            "baseline_chunk_count": base["chunk_count"],
            "enhanced_chunk_count": enh["chunk_count"],
            "baseline_table_focused_count": base["table_focused_count"],
            "enhanced_table_focused_count": enh["table_focused_count"],
            "baseline_caption_only_table_count": base["caption_only_table_count"],
            "enhanced_caption_only_table_count": enh["caption_only_table_count"],
            "enhanced_table_related_chunk_count": enh["table_related_chunk_count"],
            "paragraph_chunk_count_delta": enh["paragraph_chunk_count"] - base["paragraph_chunk_count"],
            "token_count_delta": enh["token_count"] - base["token_count"],
            "chunks_schema_fields_same": str(schema_same).lower(),
        })
    write_csv(
        output_dir / "enhanced_vs_baseline_chunks.csv",
        rows,
        [
            "doc_id",
            "baseline_chunk_count",
            "enhanced_chunk_count",
            "baseline_table_focused_count",
            "enhanced_table_focused_count",
            "baseline_caption_only_table_count",
            "enhanced_caption_only_table_count",
            "enhanced_table_related_chunk_count",
            "paragraph_chunk_count_delta",
            "token_count_delta",
            "chunks_schema_fields_same",
        ],
    )
    write_false_positive_review(output_dir / "false_positive_review.md", association_audit)

    accepted = [row for row in association_audit if row["accepted_or_rejected"] == "accepted"]
    uncertain_count = sum(1 for row in association_audit if row["accepted_or_rejected"] == "uncertain")
    if uncertain_count == 0:
        pilot_audit = output_dir.parent / "pilot_association_audit.csv"
        if pilot_audit.exists():
            with pilot_audit.open("r", encoding="utf-8", newline="") as handle:
                uncertain_count = sum(
                    1 for row in csv.DictReader(handle)
                    if row.get("accepted_or_rejected") == "uncertain"
                )
    rejected_count = sum(1 for row in association_audit if row["accepted_or_rejected"] == "rejected")
    confidence_counts = Counter(row["association_confidence"] for row in accepted if row.get("association_confidence"))
    total_base_chunks = sum(row["baseline_chunk_count"] for row in rows)
    total_enh_chunks = sum(row["enhanced_chunk_count"] for row in rows)
    total_base_tokens = sum(baseline_stats[doc_id]["token_count"] for doc_id in selected_docs)
    total_enh_tokens = sum(enhanced_stats[doc_id]["token_count"] for doc_id in selected_docs)
    chunk_delta = total_enh_chunks - total_base_chunks
    token_delta = total_enh_tokens - total_base_tokens
    token_growth_pct = (token_delta / total_base_tokens * 100.0) if total_base_tokens else 0.0
    chunk_growth_pct = (chunk_delta / total_base_chunks * 100.0) if total_base_chunks else 0.0
    table_related_chunk_count = sum(enhanced_stats[doc_id]["table_related_chunk_count"] for doc_id in selected_docs)
    table_focused_related_chunk_count = sum(
        enhanced_stats[doc_id]["table_focused_related_chunk_count"] for doc_id in selected_docs
    )
    normal_prose_rejects = sum(1 for row in association_audit if row.get("reject_reason") == "normal_prose_shape")
    accepted_long_prose = sum(
        1
        for row in accepted
        if len(str(row.get("associated_text_preview", "")).split()) >= 45
        and re.search(r"[.!?]\s+[A-Z]", str(row.get("associated_text_preview", "")))
    )
    controllable = abs(chunk_growth_pct) <= 10.0 and token_growth_pct <= 15.0
    obvious_false_positive = accepted_long_prose > max(3, len(accepted) * 0.05)
    false_positive_risk = "elevated" if obvious_false_positive else "low"
    recommend = bool(
        association_meta["processed_docs"] == len(selected_docs)
        and schema_same
        and accepted
        and controllable
        and not obvious_false_positive
        and Path(args.enhanced_chunks).exists()
    )

    summary_lines = [
        "# Phase 5C-1 Table Preservation Summary",
        "",
        f"- selected docs: {len(selected_docs)}",
        f"- successful processed docs: {association_meta['processed_docs']}",
        f"- failed docs: {len(association_meta['failed_docs'])}",
        f"- table_related associations: {len(accepted)}",
        f"- average associations per doc: {len(accepted) / len(selected_docs):.2f}",
        f"- confidence distribution: {dict(confidence_counts)}",
        f"- accepted_long_prose count: {accepted_long_prose}",
        f"- uncertain cases count: {uncertain_count}",
        f"- rejected nearby blocks count: {rejected_count}",
        f"- enhanced chunks generated: {Path(args.enhanced_chunks).exists()}",
        f"- chunk schema unchanged: {schema_same}",
        f"- schema drift exists: {str(not schema_same).lower()}",
        f"- chunk_count delta: {chunk_delta} ({chunk_growth_pct:.2f}%)",
        f"- token_count delta: {token_delta} ({token_growth_pct:.2f}%)",
        f"- chunk/token growth controllable: {controllable}",
        f"- false positive risk: {false_positive_risk}",
        f"- obvious normal-prose false positives: {obvious_false_positive} (accepted_long_prose={accepted_long_prose}, normal_prose_rejects={normal_prose_rejects})",
        f"- enhanced table_related chunks: {table_related_chunk_count}",
        f"- table-related text in table-focused chunks: {table_focused_related_chunk_count > 0} (chunks={table_focused_related_chunk_count})",
        f"- enhanced remains safe for retrieval eval: {recommend}",
        f"- recommend Phase 5C-2 retrieval-only small-sample eval: {recommend}",
    ]
    if not recommend:
        blockers = []
        if association_meta["processed_docs"] != len(selected_docs):
            blockers.append("not all selected docs processed")
        if not schema_same:
            blockers.append("chunk schema drift")
        if not accepted:
            blockers.append("no accepted associations")
        if not controllable:
            blockers.append("chunk/token growth not controllable")
        if obvious_false_positive:
            blockers.append("possible normal prose false positives")
        summary_lines.append(f"- blocker: {', '.join(blockers) if blockers else 'none identified'}")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
