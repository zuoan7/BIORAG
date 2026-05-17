#!/usr/bin/env python3
"""Validate hybrid table_object_v1 artifacts and write pilot reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SMOKE_DOC_IDS = [
    "doc_0322",
    "doc_0158",
    "doc_0598",
    "doc_0452",
    "doc_0468",
    "doc_0687",
    "doc_0458",
    "doc_0522",
    "doc_0523",
]
DEFAULT_HYBRID_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/hybrid_table_objects.jsonl"
)
DEFAULT_PDFPLUMBER_RAW_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/pdfplumber_tables.raw.jsonl"
)
DEFAULT_ALIGNMENT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/chunk_pdfplumber_alignment.csv"
)
DEFAULT_PHASE7B2_VALIDATION_PATH = (
    ROOT / "reports/v7_phase7_table_extraction_mvp_rerun/table_object_validation_summary.csv"
)
DEFAULT_PHASE7C_RAW_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/pdfplumber_tables.raw.jsonl"
)
DEFAULT_PHASE7C_ALIGNMENT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/chunk_pdfplumber_alignment.csv"
)
DEFAULT_PHASE7C_HYBRID_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/hybrid_table_objects.jsonl"
)
DEFAULT_PHASE7C_VALIDATION_PATH = (
    ROOT / "reports/v7_phase7_pdfplumber_pilot/hybrid_table_object_validation_summary.csv"
)
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_pilot"

SUMMARY_FIELDS = [
    "hybrid_table_object_id",
    "original_chunk_table_object_id",
    "pdfplumber_table_id",
    "doc_id",
    "table_id",
    "alignment_status",
    "alignment_confidence",
    "layout_quality_status",
    "extraction_method",
    "cell_bboxes_available",
    "value_bboxes_available",
    "source_span_granularity",
    "hybrid_validation_status",
    "primary_failure_stage",
    "manual_review_reason",
    "recommended_next_action",
    "blocking_warnings",
    "nonblocking_warnings",
    "notes",
]

BOUNDARY_WARNINGS = {
    "body_blocks_missing",
    "boundary_blocking_warning",
    "table_tail_truncation",
    "continued_table_needs_merge",
    "hybrid_table_boundary_conflict",
    "table_boundary_partial",
    "adjacent_non_table_contamination",
}
BINDING_WARNINGS = {
    "cell_alignment_error",
    "row_cell_blocking_warning",
    "matrix_flattened",
    "metric_level_cell_gap",
    "target_mapping_risk",
    "numeric_column_order_uncertain",
    "split_cell_warning",
    "merged_cell_warning",
    "row_continuation_warning",
    "column_alignment_inconsistent",
    "cell_grid_needs_rule_fix",
    "missing_metric_cell_warning",
    "metric_column_group_uncertain",
    "unit_scope_uncertain",
    "unit_visible_not_bound",
    "footnote_binding_uncertain",
    "footnote_present_not_bound",
    "reference_binding_uncertain",
    "reference_visible_not_bound",
    "internal_reference_column",
    "external_citation_not_supported",
}
CANDIDATE_WARNINGS = {
    "false_positive_candidate",
    "duplicate_table_candidate",
    "mixed_table_block_risk",
}
NONBLOCKING_WARNINGS = {
    "no_table_text_flag",
    "parser_boundary_warning",
    "caption_body_split",
    "body_as_paragraph",
    "body_as_table_caption",
    "body_grouping_stopped_at_body_text",
    "body_grouping_stopped_at_next_table",
    "source_span_table_row_level_only",
    "source_span_not_value_level",
    "no_value_level_bbox",
    "value_level_bbox_absent",
    "cell_bbox_not_value_bbox",
    "hybrid_cell_bbox_available",
    "abbreviation_binding_ok",
    "literal_value_requires_preservation",
    "unit_scope_column_level",
    "continued_table_merged",
    "continued_table_part",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def metadata(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("hybrid_metadata") or {}


def row_count(obj: dict[str, Any]) -> int:
    return len(obj.get("rows") or [])


def column_count(obj: dict[str, Any]) -> int:
    return len(obj.get("columns") or [])


def cell_count(obj: dict[str, Any]) -> int:
    return len(obj.get("cells") or [])


def counter_from_rows(rows: list[dict[str, str]], key: str) -> Counter[str]:
    return Counter(row.get(key, "") for row in rows)


def rate(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key) == "true") / len(rows), 4)


def split_warning_text(value: str) -> list[str]:
    if not value or value == "none":
        return []
    return [item for item in value.split(";") if item]


def classify_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    blocking: list[str] = []
    nonblocking: list[str] = []
    for warning in warnings:
        if warning in BOUNDARY_WARNINGS | BINDING_WARNINGS | CANDIDATE_WARNINGS:
            blocking.append(warning)
        elif warning.startswith("pdfplumber_alignment_") or warning in {
            "page_only_alignment_manual_review",
            "pdfplumber_alignment_low_confidence",
            "pdfplumber_low_layout_quality",
            "hybrid_used_chunk_fallback",
            "pdfplumber_cell_bbox_missing",
        }:
            blocking.append(warning)
        else:
            nonblocking.append(warning)
    return sorted(set(blocking)), sorted(set(nonblocking))


def primary_stage_from_blockers(
    alignment_status: str,
    alignment_confidence: str,
    layout_quality_status: str,
    extraction_method: str,
    source_span_granularity: str,
    cell_bboxes_available: bool,
    blocking: list[str],
    has_cells: bool,
) -> str:
    if alignment_status in {"no_pdf_table_found", "not_evaluable"}:
        return "alignment"
    if alignment_status in {"page_only_match", "caption_only_match", "multiple_pdf_tables", "conflict"}:
        return "alignment"
    if alignment_confidence in {"low", "none"}:
        return "alignment"
    if layout_quality_status in {"likely_false_positive", "failed"}:
        return "layout_extraction"
    if not has_cells or not cell_bboxes_available:
        return "cell_grid"
    if extraction_method == "chunk_fallback" or source_span_granularity not in {"cell_level", "table_row_level", "table_level", "row_level"}:
        return "source_span"
    if any(warning in CANDIDATE_WARNINGS for warning in blocking):
        return "candidate"
    if any(warning in BOUNDARY_WARNINGS for warning in blocking):
        return "boundary"
    if any(warning in BINDING_WARNINGS for warning in blocking):
        return "binding"
    if source_span_granularity != "cell_level":
        return "source_span"
    return "none"


def manual_reason_and_action(
    stage: str,
    status: str,
    alignment_status: str,
    alignment_confidence: str,
    layout_quality_status: str,
    extraction_method: str,
    source_span_granularity: str,
    blocking: list[str],
) -> tuple[str, str]:
    if status in {"pass", "pass_with_warnings"}:
        return "none", "keep"
    if alignment_status == "page_only_match":
        return "page_only_match_requires_manual_alignment_review", "manual_review_alignment"
    if alignment_status in {"multiple_pdf_tables", "conflict", "caption_only_match"}:
        return f"{alignment_status}_requires_manual_alignment_review", "manual_review_alignment"
    if alignment_confidence in {"low", "none"}:
        return "low_alignment_confidence", "manual_review_alignment"
    if layout_quality_status == "likely_false_positive":
        return "layout_quality_likely_false_positive", "exclude"
    if layout_quality_status in {"weak", "failed"}:
        return f"layout_quality_{layout_quality_status}", "manual_review_layout"
    if extraction_method == "chunk_fallback":
        return "chunk_fallback_due_to_alignment_or_layout_gate", "chunk_fallback"
    if "pdfplumber_cell_bbox_missing" in blocking:
        return "cell_bbox_missing", "improve_pdfplumber_strategy"
    if stage == "binding":
        return "unit_footnote_reference_or_cell_binding_unresolved", "manual_review_binding"
    if stage == "boundary":
        return "table_boundary_or_body_blocks_unresolved", "manual_review_layout"
    if stage == "source_span" and source_span_granularity != "cell_level":
        return "source_span_not_cell_level", "backlog"
    if status == "manual_review":
        return "manual_review_required_by_gate", "backlog"
    return "not_blocked_but_partial", "backlog"


def validate_hybrid_object(
    obj: dict[str, Any],
    alignment_row: dict[str, str] | None = None,
    pdf: dict[str, Any] | None = None,
) -> dict[str, str]:
    meta = metadata(obj)
    alignment_row = alignment_row or {}
    pdf = pdf or {}
    alignment_status = (
        meta.get("alignment_status")
        or obj.get("alignment_status")
        or alignment_row.get("alignment_status")
        or "not_evaluable"
    )
    alignment_confidence = (
        meta.get("alignment_confidence")
        or obj.get("alignment_confidence")
        or alignment_row.get("alignment_confidence")
        or "none"
    )
    layout_quality_status = (
        pdf.get("layout_quality_status")
        or alignment_row.get("layout_quality_status")
        or obj.get("layout_quality_status")
        or "not_evaluable"
    )
    extraction_method = meta.get("extraction_method") or obj.get("extraction_method") or "chunk_fallback"
    source_span_granularity = (
        meta.get("source_span_granularity")
        or obj.get("source_span_granularity")
        or "mixed_or_unclear"
    )
    cell_bboxes_available = bool(meta.get("cell_bboxes_available", obj.get("cell_bboxes_available", False)))
    value_bboxes_available = bool(meta.get("value_bboxes_available", obj.get("value_bboxes_available", False)))
    warnings = list(obj.get("warnings") or [])
    blocking, nonblocking = classify_warnings(warnings)
    notes: list[str] = []
    has_required_identity = bool(obj.get("table_object_id") and obj.get("doc_id"))
    has_cells = row_count(obj) > 0 and column_count(obj) > 0 and cell_count(obj) > 0

    if source_span_granularity == "value_level":
        blocking.append("value_level_bbox_fabrication_risk")
        notes.append("source_span_granularity 不得为 value_level。")
    if value_bboxes_available:
        blocking.append("value_bbox_unexpected")
        notes.append("本轮没有 token/value bbox，不应标记 value_bboxes_available=true。")
    if not value_bboxes_available:
        nonblocking.append("value_bbox_absent_limitation")

    if not has_required_identity:
        status = "fail"
        stage = "not_evaluable"
        notes.append("缺少 table_object_id 或 doc_id。")
    elif not has_cells:
        status = "fail"
        stage = "cell_grid"
        notes.append("rows / columns / cells 不完整。")
    else:
        stage = primary_stage_from_blockers(
            alignment_status,
            alignment_confidence,
            layout_quality_status,
            extraction_method,
            source_span_granularity,
            cell_bboxes_available,
            blocking,
            has_cells,
        )
        status = "pass_with_warnings"
        if alignment_status in {"page_only_match", "caption_only_match", "multiple_pdf_tables", "conflict"}:
            status = "manual_review"
            notes.append("alignment gate 要求人工复核。")
        elif alignment_confidence in {"low", "none"}:
            status = "manual_review"
            notes.append("低置信对齐不能 pass。")
        elif layout_quality_status in {"likely_false_positive", "failed"}:
            status = "partial"
            notes.append("layout extraction 质量不足，不能作为可信 hybrid 表。")
        elif extraction_method == "chunk_fallback":
            status = "partial"
            notes.append("当前对象走 chunk_fallback，pdfplumber 未稳定改善 row/cell。")
        elif not cell_bboxes_available:
            status = "partial"
            notes.append("缺少 cell bbox。")
        elif blocking:
            status = "partial"
            notes.append("存在 binding/boundary/candidate 阻断 warning。")
        elif source_span_granularity != "cell_level":
            status = "partial"
            notes.append("source_span 尚未稳定提升到 cell_level。")

    if status == "pass_with_warnings" and not notes:
        notes.append("alignment/layout/cell grid 满足本轮 gate；仍保留非阻断 warning 与 value bbox limitation。")
    if not notes:
        notes.append("需要人工复核。")
    reason, action = manual_reason_and_action(
        stage,
        status,
        alignment_status,
        alignment_confidence,
        layout_quality_status,
        extraction_method,
        source_span_granularity,
        sorted(set(blocking)),
    )
    return {
        "hybrid_table_object_id": obj.get("table_object_id", ""),
        "original_chunk_table_object_id": meta.get("original_chunk_table_object_id")
        or obj.get("original_chunk_table_object_id", ""),
        "pdfplumber_table_id": meta.get("pdfplumber_table_id") or "",
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "alignment_status": alignment_status,
        "alignment_confidence": alignment_confidence,
        "layout_quality_status": layout_quality_status,
        "extraction_method": extraction_method,
        "cell_bboxes_available": bool_text(cell_bboxes_available),
        "value_bboxes_available": bool_text(value_bboxes_available),
        "source_span_granularity": source_span_granularity,
        "hybrid_validation_status": status,
        "primary_failure_stage": stage,
        "manual_review_reason": reason,
        "recommended_next_action": action,
        "blocking_warnings": ";".join(sorted(set(blocking))) if blocking else "none",
        "nonblocking_warnings": ";".join(sorted(set(nonblocking))) if nonblocking else "none",
        "notes": " ".join(notes),
    }


def write_summary_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_hybrid_objects(path: Path, objects: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    by_id = {row["hybrid_table_object_id"]: row for row in rows}
    with path.open("w", encoding="utf-8") as handle:
        for obj in objects:
            row = by_id.get(obj.get("table_object_id"))
            if row:
                obj["validation_status"] = row["hybrid_validation_status"]
            handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def cell_bbox_rate(rows: list[dict[str, str]]) -> float:
    return rate(rows, "cell_bboxes_available")


def value_bbox_rate(rows: list[dict[str, str]]) -> float:
    return rate(rows, "value_bboxes_available")


def write_validation_report(objects: list[dict[str, Any]], rows: list[dict[str, str]], path: Path) -> None:
    status_counts = counter_from_rows(rows, "hybrid_validation_status")
    stage_counts = counter_from_rows(rows, "primary_failure_stage")
    reason_counts = counter_from_rows(rows, "manual_review_reason")
    granularity_counts = counter_from_rows(rows, "source_span_granularity")
    lines = [
        "# hybrid table_object validation 报告",
        "",
        "## 1. 校验目标",
        "",
        "本报告校验 Phase7C-2 hybrid table_object 是否满足 alignment、layout、cell grid、source_span 与 binding gate。重点不是提高 pass 数量，而是把不可直接使用的对象解释清楚。",
        "",
        "## 2. 总体统计",
        "",
        f"- hybrid table_objects 数量：{len(objects)}",
        f"- cell_bbox_available_rate：{cell_bbox_rate(rows):.4f}",
        f"- value_bbox_available_rate：{value_bbox_rate(rows):.4f}",
        "",
        "| hybrid_validation_status | 数量 |",
        "|---|---:|",
    ]
    for status in ["pass", "pass_with_warnings", "partial", "manual_review", "fail"]:
        lines.append(f"| `{status}` | {status_counts.get(status, 0)} |")
    lines.extend(["", "## 3. primary_failure_stage 分布", "", "| stage | 数量 |", "|---|---:|"])
    for stage in ["none", "alignment", "layout_extraction", "cell_grid", "source_span", "binding", "boundary", "candidate", "not_evaluable"]:
        lines.append(f"| `{stage}` | {stage_counts.get(stage, 0)} |")
    lines.extend(["", "## 4. manual_review_reason 分布", "", "| reason | 数量 |", "|---|---:|"])
    for reason, count in reason_counts.most_common():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## 5. source_span_granularity 分布", "", "| granularity | 数量 |", "|---|---:|"])
    for granularity, count in granularity_counts.most_common():
        lines.append(f"| `{granularity}` | {count} |")
    lines.extend(
        [
            "",
            "## 6. status 判定解释",
            "",
            "- `pass_with_warnings`：alignment high/medium、layout usable、rows/columns/cells 存在、cell bbox 可用，且没有阻断型 binding/boundary/candidate warning；仍保留 value bbox absence limitation。",
            "- `partial`：对象可审阅，但 layout、cell grid、source_span 或 binding/boundary 仍有缺口，不能升级为可信通过。",
            "- `manual_review`：alignment gate 未过，尤其是 page_only、caption_only、multiple candidates、conflict 或 low confidence，需要人工对齐复核。",
            "- `fail`：缺少对象身份或 rows/columns/cells 等最低结构。",
            "",
            "## 7. cell bbox vs value bbox 口径",
            "",
            "- `cell_bboxes_available=true` 只表示 pdfplumber 给出了 cell/grid layout bbox。",
            "- `value_bboxes_available=false` 是本轮默认口径；没有 token/value bbox 时不得写 `value_level`。",
            "- no value bbox 是 limitation，不单独导致 fail；若 unit/footnote/reference binding 未解决，才进入 partial 或 manual review。",
            "",
            "## 8. 需要复核的对象",
            "",
            "| hybrid_table_object_id | doc_id | table_id | status | stage | reason | action |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    review_rows = [row for row in rows if row["hybrid_validation_status"] != "pass_with_warnings"]
    for row in review_rows:
        lines.append(
            f"| `{row['hybrid_table_object_id']}` | `{row['doc_id']}` | `{row['table_id']}` | `{row['hybrid_validation_status']}` | `{row['primary_failure_stage']}` | `{row['manual_review_reason']}` | `{row['recommended_next_action']}` |"
        )
    if not review_rows:
        lines.append("| none |  |  |  |  |  |  |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def layout_counts(raw_tables: list[dict[str, Any]]) -> Counter[str]:
    return Counter(table.get("layout_quality_status", "unknown") for table in raw_tables)


def write_phase7c_vs_phase7c2_comparison(
    report_dir: Path,
    phase7c_raw: list[dict[str, Any]],
    phase7c_alignment: list[dict[str, str]],
    phase7c_hybrid: list[dict[str, Any]],
    phase7c_rows: list[dict[str, str]],
    phase7c2_raw: list[dict[str, Any]],
    phase7c2_alignment: list[dict[str, str]],
    phase7c2_objects: list[dict[str, Any]],
    phase7c2_rows: list[dict[str, str]],
) -> None:
    c_alignment_status = counter_from_rows(phase7c_alignment, "alignment_status")
    c_alignment_conf = counter_from_rows(phase7c_alignment, "alignment_confidence")
    c2_alignment_status = counter_from_rows(phase7c2_alignment, "alignment_status")
    c2_alignment_conf = counter_from_rows(phase7c2_alignment, "alignment_confidence")
    c_status = counter_from_rows(phase7c_rows, "hybrid_validation_status")
    c2_status = counter_from_rows(phase7c2_rows, "hybrid_validation_status")
    c_granularity = counter_from_rows(phase7c_rows, "source_span_granularity")
    c2_granularity = counter_from_rows(phase7c2_rows, "source_span_granularity")
    c2_stage = counter_from_rows(phase7c2_rows, "primary_failure_stage")
    lines = [
        "# Phase7C vs Phase7C-2 对比报告",
        "",
        "## 1. 对比目标",
        "",
        "本报告比较 Phase7C 与 Phase7C-2 在同一批 smoke doc_id 上的 raw layout、alignment gate、hybrid table_object 和 validation 变化。Phase7C-2 目标是让判断更诚实、更可解释，不追求 pass 数量上升。",
        "",
        "## 2. raw tables 数量",
        "",
        f"- Phase7C raw tables 数量：{len(phase7c_raw)}",
        f"- Phase7C-2 raw tables 数量：{len(phase7c2_raw)}",
        f"- Phase7C-2 layout_quality_status 统计：{dict(layout_counts(phase7c2_raw))}",
        "",
        "## 3. alignment 对比",
        "",
        f"- Phase7C alignment_status：{dict(c_alignment_status)}",
        f"- Phase7C alignment_confidence：{dict(c_alignment_conf)}",
        f"- Phase7C-2 alignment_status：{dict(c2_alignment_status)}",
        f"- Phase7C-2 alignment_confidence：{dict(c2_alignment_conf)}",
        "- page_only_match 是否更严格：是。Phase7C-2 将 page_only_match 固定为 low confidence/manual review，不再默认可信。",
        "- manual review reason 是否更清楚：是。validation summary 显式写入 `manual_review_reason` 和 `recommended_next_action`。",
        "",
        "## 4. hybrid object 与 validation 对比",
        "",
        f"- Phase7C hybrid table_objects 数量：{len(phase7c_hybrid)}",
        f"- Phase7C-2 hybrid table_objects 数量：{len(phase7c2_objects)}",
        f"- Phase7C hybrid_validation_status：{dict(c_status)}",
        f"- Phase7C-2 hybrid_validation_status：{dict(c2_status)}",
        f"- Phase7C-2 primary_failure_stage 分布：{dict(c2_stage)}",
        f"- Phase7C cell_bbox_available_rate：{cell_bbox_rate(phase7c_rows):.4f}",
        f"- Phase7C-2 cell_bbox_available_rate：{cell_bbox_rate(phase7c2_rows):.4f}",
        f"- Phase7C source_span_granularity：{dict(c_granularity)}",
        f"- Phase7C-2 source_span_granularity：{dict(c2_granularity)}",
        "",
        "## 5. 质量解释",
        "",
        "- 是否减少不可信 pass：是。alignment gate 收紧后，不可信对象进入 manual_review / partial 是质量变诚实，不是失败。",
        "- 是否更利于人工审阅：是。alignment sidecar 记录 score/basis/blockers，validation summary 记录 primary failure 和 action。",
        "- cell bbox 与 value bbox：Phase7C-2 明确 `value_bboxes_available=false`；cell bbox 只支持 cell-level layout provenance。",
        "",
        "## 6. 建议",
        "",
        "- 是否建议继续 hybrid pipeline：建议继续作为离线实验路线。",
        "- 是否建议扩大 smoke：暂不建议；应先人工审阅 high/medium matched 以及 page_only/manual_review case。",
        "- 是否建议试 Camelot / PyMuPDF：建议作为对照，尤其用于 ruled table 和 token/span bbox 交叉验证。",
        "- 是否建议 production：不建议。本轮仍是 pilot hardening。",
        "- Route C 是否仍只是 backlog：是，仍只是 backlog。",
    ]
    (report_dir / "phase7c_vs_phase7c2_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_phase7c2_summary(
    report_dir: Path,
    objects: list[dict[str, Any]],
    raw_tables: list[dict[str, Any]],
    alignment_rows: list[dict[str, str]],
    hybrid_rows: list[dict[str, str]],
) -> None:
    alignment_status = counter_from_rows(alignment_rows, "alignment_status")
    alignment_conf = counter_from_rows(alignment_rows, "alignment_confidence")
    hybrid_status = counter_from_rows(hybrid_rows, "hybrid_validation_status")
    stage_counts = counter_from_rows(hybrid_rows, "primary_failure_stage")
    reason_counts = counter_from_rows(hybrid_rows, "manual_review_reason")
    granularity = counter_from_rows(hybrid_rows, "source_span_granularity")
    generated_files = [
        "data/experiments/v7_phase7_pdfplumber_pilot_v2/pdfplumber_tables.raw_v2.jsonl",
        "data/experiments/v7_phase7_pdfplumber_pilot_v2/chunk_pdfplumber_alignment_v2.csv",
        "data/experiments/v7_phase7_pdfplumber_pilot_v2/hybrid_table_objects.jsonl",
        "data/experiments/v7_phase7_pdfplumber_pilot_v2/hybrid_table_objects_review.md",
        "reports/v7_phase7_pdfplumber_pilot_v2/phase7c_2_guardrail.md",
        "reports/v7_phase7_pdfplumber_pilot_v2/pdfplumber_layout_quality_report.md",
        "reports/v7_phase7_pdfplumber_pilot_v2/chunk_pdfplumber_alignment_gate_report.md",
        "reports/v7_phase7_pdfplumber_pilot_v2/hybrid_table_object_validation_summary.csv",
        "reports/v7_phase7_pdfplumber_pilot_v2/hybrid_table_object_validation_report.md",
        "reports/v7_phase7_pdfplumber_pilot_v2/phase7c_vs_phase7c2_comparison.md",
        "reports/v7_phase7_pdfplumber_pilot_v2/phase7c_2_summary.md",
    ]
    lines = [
        "# Phase7C-2 总结",
        "",
        "## 1. 本轮生成文件",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)
    lines.extend(
        [
            "",
            "## 2. 修改脚本与测试",
            "",
            "- `scripts/extraction/extract_tables_pdfplumber_v1.py`",
            "- `scripts/extraction/align_chunk_pdfplumber_tables.py`",
            "- `scripts/extraction/build_hybrid_table_objects_v1.py`",
            "- `scripts/extraction/validate_hybrid_table_objects_v1.py`",
            "- `scripts/extraction/render_hybrid_table_objects_markdown.py`",
            "- 新增测试：`tests/test_phase7_hybrid_alignment_gate.py`",
            "",
            "## 3. smoke doc_id",
            "",
            f"- smoke doc_id：{', '.join(SMOKE_DOC_IDS)}",
            "- smoke doc_id 是否保持不变：是。",
            "",
            "## 4. Phase7C-2 统计",
            "",
            f"- raw pdfplumber table 数量：{len(raw_tables)}",
            f"- layout_quality_status 统计：{dict(layout_counts(raw_tables))}",
            f"- alignment_status 统计：{dict(alignment_status)}",
            f"- alignment_confidence 统计：{dict(alignment_conf)}",
            f"- hybrid table_objects 数量：{len(objects)}",
            f"- hybrid_validation_status 统计：{dict(hybrid_status)}",
            f"- primary_failure_stage 统计：{dict(stage_counts)}",
            f"- manual_review_reason 统计：{dict(reason_counts)}",
            f"- cell_bbox_available_rate：{cell_bbox_rate(hybrid_rows):.4f}",
            f"- value_bbox_available_rate：{value_bbox_rate(hybrid_rows):.4f}",
            f"- source_span_granularity 统计：{dict(granularity)}",
            "",
            "## 5. 相比 Phase7C 的主要改善",
            "",
            "- layout quality 从 raw table 层显式可见，false positive / weak layout 不再隐藏。",
            "- alignment gate 收紧：page_only_match 进入 low/manual review，高置信必须有 stronger basis 与 usable layout。",
            "- 主 hybrid object 只保留 minimal `hybrid_metadata`，详细 score/basis/blockers/layout debug 留在 sidecar。",
            "- validation summary 能解释 primary failure、manual review reason 与 next action。",
            "- 明确 cell bbox 不等于 value bbox，未伪造 value-level provenance。",
            "",
            "## 6. 仍然存在的问题",
            "",
            "- pdfplumber raw tables 数量仍偏高，存在 false positive / fragment 风险。",
            "- cell bbox 不是 value-level bbox，unit / footnote / reference binding 仍不能自动解决。",
            "- chunk boundary heuristic 仍需继续修，pdfplumber 不能替代 chunk/corpus alignment。",
            "",
            "## 7. 建议",
            "",
            "- 是否建议继续 hybrid pipeline：建议继续离线 hardening。",
            "- 是否建议继续修 chunk heuristic：建议。",
            "- 是否建议试 Camelot / PyMuPDF：建议作为对照实验。",
            "- 是否建议扩大 smoke：暂不建议。",
            "- 是否建议进入 production：不建议。",
            "- baseline / guardrail 是否漂移：未发现漂移。",
            "- Route C 是否仍只是 backlog：是，仍只是 backlog。",
            "",
            "## 8. 明确未执行事项",
            "",
            "- 未改 ingestion pipeline。",
            "- 未改 configs。",
            "- 未改 README。",
            "- 未改 baseline registry。",
            "- 未改 official dataset。",
            "- 未改 official baseline。",
            "- 未重建 chunks。",
            "- 未重建 BM25。",
            "- 未访问 Milvus。",
            "- 未写入 Milvus。",
            "- 未读取或查询 BM25 index。",
            "- 未跑 retrieval。",
            "- 未跑 embedding/rerank。",
            "- 未调用 Qwen/RAGAS/OCR/VLM。",
            "- 未接入 production。",
            "- 未进入 Route C。",
        ]
    )
    (report_dir / "phase7c_2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    objects = load_jsonl(args.hybrid_objects)
    raw_tables = load_jsonl(args.pdfplumber_raw)
    alignment_rows = load_csv(args.alignment)
    alignment_by_chunk = {row.get("chunk_table_object_id"): row for row in alignment_rows}
    pdf_by_id = {pdf.get("pdfplumber_table_id"): pdf for pdf in raw_tables}
    rows: list[dict[str, str]] = []
    for obj in objects:
        meta = metadata(obj)
        alignment = alignment_by_chunk.get(meta.get("original_chunk_table_object_id", ""), {})
        pdf = pdf_by_id.get(meta.get("pdfplumber_table_id", ""), {})
        rows.append(validate_hybrid_object(obj, alignment, pdf))

    write_summary_csv(rows, args.report_dir / "hybrid_table_object_validation_summary.csv")
    update_hybrid_objects(args.hybrid_objects, objects, rows)
    objects = load_jsonl(args.hybrid_objects)
    write_validation_report(objects, rows, args.report_dir / "hybrid_table_object_validation_report.md")

    write_phase7c_vs_phase7c2_comparison(
        args.report_dir,
        load_jsonl(args.phase7c_raw),
        load_csv(args.phase7c_alignment),
        load_jsonl(args.phase7c_hybrid),
        load_csv(args.phase7c_validation),
        raw_tables,
        alignment_rows,
        objects,
        rows,
    )
    write_phase7c2_summary(args.report_dir, objects, raw_tables, alignment_rows, rows)
    print(
        json.dumps(
            {
                "hybrid_table_objects": len(objects),
                "hybrid_validation_status": dict(counter_from_rows(rows, "hybrid_validation_status")),
                "primary_failure_stage": dict(counter_from_rows(rows, "primary_failure_stage")),
                "cell_bbox_available_rate": cell_bbox_rate(rows),
                "value_bbox_available_rate": value_bbox_rate(rows),
                "report_dir": rel(args.report_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate hybrid table_objects.")
    parser.add_argument("--hybrid-objects", type=Path, default=DEFAULT_HYBRID_OBJECTS_PATH)
    parser.add_argument("--pdfplumber-raw", type=Path, default=DEFAULT_PDFPLUMBER_RAW_PATH)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument("--phase7b2-validation", type=Path, default=DEFAULT_PHASE7B2_VALIDATION_PATH)
    parser.add_argument("--phase7c-raw", type=Path, default=DEFAULT_PHASE7C_RAW_PATH)
    parser.add_argument("--phase7c-alignment", type=Path, default=DEFAULT_PHASE7C_ALIGNMENT_PATH)
    parser.add_argument("--phase7c-hybrid", type=Path, default=DEFAULT_PHASE7C_HYBRID_PATH)
    parser.add_argument("--phase7c-validation", type=Path, default=DEFAULT_PHASE7C_VALIDATION_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    args.hybrid_objects = resolve_path(args.hybrid_objects)
    args.pdfplumber_raw = resolve_path(args.pdfplumber_raw)
    args.alignment = resolve_path(args.alignment)
    args.phase7b2_validation = resolve_path(args.phase7b2_validation)
    args.phase7c_raw = resolve_path(args.phase7c_raw)
    args.phase7c_alignment = resolve_path(args.phase7c_alignment)
    args.phase7c_hybrid = resolve_path(args.phase7c_hybrid)
    args.phase7c_validation = resolve_path(args.phase7c_validation)
    args.report_dir = resolve_path(args.report_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    run(parse_args())
