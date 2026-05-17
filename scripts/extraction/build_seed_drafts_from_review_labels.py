#!/usr/bin/env python3
"""Route frozen Phase7G human labels into seed drafts and feedback queues."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction.freeze_human_table_review_labels import load_csv, rel, resolve, write_csv, write_jsonl


DEFAULT_REVIEW_PACK_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_FREEZE_DIR = ROOT / "data/experiments/v7_phase7_human_review_label_freeze"

CONFIRMED_DRAFT_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "unit_binding_status",
    "reference_binding_status",
    "requires_light_binding_review",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
    "risk_tags",
    "seed_draft_status",
    "seed_draft_notes",
]

FOLLOWUP_FIELDS = [
    "candidate_id",
    "unit_or_note_ok",
    "reference_ok",
    "binding_notes",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "risk_tags",
]

PARTIAL_ROUTING_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "review_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "routing_status",
    "partial_evidence_usable",
    "recommended_next_action",
    "reason",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "risk_tags",
    "review_notes",
]

REJECT_BOUNDARY_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "suggested_decision",
    "review_priority",
    "auto_score",
    "risk_tags",
    "reason",
    "recommended_scoring_feedback",
]


def by_candidate(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("candidate_id", ""): row for row in rows if row.get("candidate_id")}


def merge_metadata(row: dict[str, str], *lookups: dict[str, dict[str, str]]) -> dict[str, str]:
    merged = dict(row)
    candidate_id = row.get("candidate_id", "")
    for lookup in lookups:
        source = lookup.get(candidate_id, {})
        for key, value in source.items():
            if not merged.get(key):
                merged[key] = value
    return merged


def binding_status(label_value: str) -> str:
    if label_value == "unchecked":
        return "unchecked"
    if label_value == "yes":
        return "reviewed_yes"
    if label_value in {"warning", "no", "not_applicable"}:
        return label_value
    return "unchecked"


def is_confirmed_seed_draft(row: dict[str, str]) -> bool:
    return (
        row.get("label_status") != "invalid_label"
        and row.get("review_decision") == "accept_confirmed_seed_candidate"
        and row.get("boundary_ok") == "yes"
        and row.get("grid_ok") == "yes"
        and row.get("key_values_ok") == "yes"
    )


def confirmed_draft_row(row: dict[str, str]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in CONFIRMED_DRAFT_FIELDS}
    out["unit_binding_status"] = binding_status(row.get("unit_or_note_ok", "unchecked"))
    out["reference_binding_status"] = binding_status(row.get("reference_ok", "unchecked"))
    out["requires_light_binding_review"] = "true"
    out["seed_draft_status"] = "confirmed_seed_draft"
    out["seed_draft_notes"] = (
        "仅由人工核心标签冻结生成；unit_or_note_ok/reference_ok 未作为正式 binding 确认。"
    )
    return out


def followup_row(row: dict[str, str]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in FOLLOWUP_FIELDS}
    out["binding_notes"] = ""
    return out


def route_partial_candidate(row: dict[str, str]) -> tuple[str, bool, str, str]:
    boundary = row.get("boundary_ok", "")
    grid = row.get("grid_ok", "")
    key_values = row.get("key_values_ok", "")
    risk_tags = row.get("risk_tags", "")
    notes = row.get("review_notes", "")

    if boundary == "unclear" and grid == "unclear" and key_values == "unclear":
        return (
            "backlog",
            False,
            "保留在 backlog；需要更清晰的边界、网格和关键值证据。",
            "boundary/grid/key_values 全部为 unclear。",
        )
    if key_values == "no":
        return (
            "not_seed_candidate",
            False,
            "不进入 seed draft；后续只作为负例或规则反馈参考。",
            "人工标注 key_values_ok=no，关键值证据不可用。",
        )
    if grid == "no":
        return (
            "needs_grid_fix",
            False,
            "进入网格修复候选；修复前不得进入 confirmed draft。",
            "人工标注 grid_ok=no。",
        )
    if boundary == "no" and key_values == "yes":
        return (
            "needs_boundary_fix",
            True,
            "作为边界修复候选；保留关键值可用证据。",
            "boundary_ok=no 但 key_values_ok=yes。",
        )
    if key_values == "yes" and (boundary == "unclear" or grid == "unclear"):
        risky = any(
            tag in risk_tags
            for tag in ["grid_sparse_or_unreadable", "split_cell_warning", "merged_cell_warning", "row_continuation_warning"]
        )
        rule_hint = "rule" in notes.lower() or "fix" in notes.lower()
        if risky or rule_hint:
            return (
                "needs_rule_fix",
                True,
                "进入轻量规则复核；保持 partial 与 confirmed draft 分离。",
                "存在 unclear 核心字段，且 risk_tags/review_notes 指向规则风险。",
            )
        return (
            "partial_seed_candidate",
            True,
            "保留为 partial seed candidate；下一轮可做最小证据补充。",
            "关键值可用，但 boundary 或 grid 仍不完全明确。",
        )
    return (
        "backlog",
        False,
        "暂不进入 seed draft；等待后续人工或规则复核。",
        "未满足 partial seed candidate 的最小可用证据。",
    )


def partial_routing_row(row: dict[str, str]) -> dict[str, Any]:
    routing_status, usable, action, reason = route_partial_candidate(row)
    out = {field: row.get(field, "") for field in PARTIAL_ROUTING_FIELDS}
    out["routing_status"] = routing_status
    out["partial_evidence_usable"] = str(usable).lower()
    out["recommended_next_action"] = action
    out["reason"] = reason
    return out


def reject_boundary_row(row: dict[str, str]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in REJECT_BOUNDARY_FIELDS}
    out["reason"] = row.get("review_notes") or (
        f"人工标注 reject_boundary；boundary_ok={row.get('boundary_ok')}, "
        f"grid_ok={row.get('grid_ok')}, key_values_ok={row.get('key_values_ok')}。"
    )
    out["recommended_scoring_feedback"] = (
        "后续 candidate scoring 应降低类似 boundary/risk_tags 组合的优先级；"
        "该候选只进入 reject feedback，不进入 seed draft。"
    )
    return out


def write_partial_report(partial_rows: list[dict[str, Any]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["routing_status"] for row in partial_rows)
    lines = [
        "# Partial Candidate 二次分流报告",
        "",
        "本报告只处理 `review_decision=accept_partial_seed_candidate` 的候选。",
        "partial candidate 不直接进入 `confirmed_seed_draft`，也不构造正式 partial seed。",
        "",
        "## routing_status 统计",
    ]
    if counts:
        lines.extend(f"- `{key}`：{value}" for key, value in counts.most_common())
    else:
        lines.append("- 无 partial candidate。")
    lines.extend(["", "## 分流清单"])
    for row in partial_rows:
        lines.append(
            f"- `{row['candidate_id']}`：`{row['routing_status']}`；"
            f"partial_evidence_usable=`{row['partial_evidence_usable']}`；{row['reason']}"
        )
    (report_dir / "partial_candidate_routing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_seed_drafts(
    frozen_review_labels_path: Path,
    review_pack_index_path: Path,
    candidate_pool_path: Path,
    output_dir: Path,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    frozen_review_labels_path = resolve(frozen_review_labels_path)
    review_pack_index_path = resolve(review_pack_index_path)
    candidate_pool_path = resolve(candidate_pool_path)
    output_dir = resolve(output_dir)
    report_dir = resolve(report_dir) if report_dir else ROOT / "reports/v7_phase7_human_review_label_freeze"

    frozen_rows = load_csv(frozen_review_labels_path)
    index_lookup = by_candidate(load_csv(review_pack_index_path))
    pool_lookup = by_candidate(load_csv(candidate_pool_path))
    rows = [merge_metadata(row, index_lookup, pool_lookup) for row in frozen_rows]

    confirmed_rows = [confirmed_draft_row(row) for row in rows if is_confirmed_seed_draft(row)]
    followup_rows = [followup_row(row) for row in confirmed_rows]
    partial_rows = [
        partial_routing_row(row)
        for row in rows
        if row.get("label_status") != "invalid_label"
        and row.get("review_decision") == "accept_partial_seed_candidate"
    ]
    reject_rows = [
        reject_boundary_row(row)
        for row in rows
        if row.get("label_status") != "invalid_label" and row.get("review_decision") == "reject_boundary"
    ]

    write_csv(confirmed_rows, output_dir / "confirmed_seed_draft_candidates.csv", CONFIRMED_DRAFT_FIELDS)
    write_jsonl(confirmed_rows, output_dir / "confirmed_seed_draft_candidates.jsonl")
    write_csv(followup_rows, output_dir / "unit_reference_followup_template.csv", FOLLOWUP_FIELDS)
    write_csv(partial_rows, output_dir / "partial_candidate_routing.csv", PARTIAL_ROUTING_FIELDS)
    write_csv(reject_rows, output_dir / "reject_boundary_feedback.csv", REJECT_BOUNDARY_FIELDS)
    write_partial_report(partial_rows, report_dir)

    return {
        "confirmed_seed_draft_count": len(confirmed_rows),
        "confirmed_seed_draft_ids": [row["candidate_id"] for row in confirmed_rows],
        "partial_routing_counts": Counter(row["routing_status"] for row in partial_rows),
        "reject_boundary_count": len(reject_rows),
        "reject_boundary_ids": [row["candidate_id"] for row in reject_rows],
        "output_dir": rel(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-review-labels", type=Path, default=DEFAULT_FREEZE_DIR / "frozen_review_labels.csv")
    parser.add_argument("--review-pack-index", type=Path, default=DEFAULT_REVIEW_PACK_DIR / "review_pack_index.csv")
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_REVIEW_PACK_DIR / "candidate_pool_scored.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/v7_phase7_human_review_label_freeze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_seed_drafts(
        frozen_review_labels_path=args.frozen_review_labels,
        review_pack_index_path=args.review_pack_index,
        candidate_pool_path=args.candidate_pool,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    print(
        "seed_draft_routing: "
        f"confirmed_seed_draft={result['confirmed_seed_draft_count']} "
        f"reject_boundary={result['reject_boundary_count']}"
    )


if __name__ == "__main__":
    main()
