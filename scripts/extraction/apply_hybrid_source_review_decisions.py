#!/usr/bin/env python3
"""Apply Phase7C-3 source-review decisions to hybrid table objects.

This is an offline gate-hardening script. It reads Phase7C-2 artifacts and
manual source-review labels, writes Phase7C-3 gated artifacts, and does not
access BM25, Milvus, retrieval, model calls, OCR, VLM, or production pipelines.
"""

from __future__ import annotations

import argparse
import copy
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

PHASE7C2_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_pilot_v2"
PHASE7C2_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_pilot_v2"
PHASE6D_REPORT_DIR = ROOT / "reports/v7_phase6d_table_contract_refinement"

DEFAULT_SOURCE_REVIEW_CSV = PHASE7C2_REPORT_DIR / "hybrid_case_source_review.csv"
DEFAULT_SOURCE_REVIEW_MD = PHASE7C2_REPORT_DIR / "hybrid_case_source_review.md"
DEFAULT_HYBRID_OBJECTS = PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl"
DEFAULT_VALIDATION_SUMMARY = PHASE7C2_REPORT_DIR / "hybrid_table_object_validation_summary.csv"
DEFAULT_ALIGNMENT = PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv"
DEFAULT_RAW_TABLES = PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_gate_hardening"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_gate_hardening"

REQUIRED_PHASE7C2_INPUTS = [
    PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl",
    PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv",
    PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl",
    PHASE7C2_DATA_DIR / "hybrid_table_objects_review.md",
    PHASE7C2_REPORT_DIR / "phase7c_2_guardrail.md",
    PHASE7C2_REPORT_DIR / "pdfplumber_layout_quality_report.md",
    PHASE7C2_REPORT_DIR / "chunk_pdfplumber_alignment_gate_report.md",
    PHASE7C2_REPORT_DIR / "hybrid_table_object_validation_summary.csv",
    PHASE7C2_REPORT_DIR / "hybrid_table_object_validation_report.md",
    PHASE7C2_REPORT_DIR / "phase7c_vs_phase7c2_comparison.md",
    PHASE7C2_REPORT_DIR / "phase7c_2_summary.md",
    PHASE7C2_REPORT_DIR / "hybrid_case_source_review.csv",
    PHASE7C2_REPORT_DIR / "hybrid_case_source_review.md",
]

REQUIRED_PHASE6D_INPUTS = [
    PHASE6D_REPORT_DIR / "phase6d_refine_round1_summary.md",
    PHASE6D_REPORT_DIR / "numeric_unit_footnote_contract.md",
    PHASE6D_REPORT_DIR / "numeric_unit_footnote_rules.csv",
    PHASE6D_REPORT_DIR / "matrix_superscript_literal_contract.md",
    PHASE6D_REPORT_DIR / "matrix_superscript_literal_rules.csv",
    PHASE6D_REPORT_DIR / "source_span_granularity_contract.md",
    PHASE6D_REPORT_DIR / "source_span_granularity_rules.csv",
    PHASE6D_REPORT_DIR / "partial_to_confirmed_decision_guide.md",
    PHASE6D_REPORT_DIR / "partial_to_confirmed_rules.csv",
]

KEEP_DECISION = "keep_as_hybrid_candidate_requires_binding_review"
GRID_REJECT_DECISION = "alignment_confirmed_reject_pdfplumber_cell_grid"
REJECT_ALIGNMENT_DECISION = "reject_current_pdfplumber_alignment"
CHUNK_FALLBACK_DECISION = "reject_selected_candidate_use_chunk_fallback"
BACKLOG_DECISION = "backlog_pdf_text_layer_unresolved"

SOURCE_REVIEW_DECISIONS = {
    KEEP_DECISION,
    GRID_REJECT_DECISION,
    REJECT_ALIGNMENT_DECISION,
    CHUNK_FALLBACK_DECISION,
    BACKLOG_DECISION,
}

SOURCE_REVIEW_CATEGORIES = {
    "keep_hybrid_candidate_needs_binding_review",
    "alignment_confirmed_grid_rejected",
    "reject_current_pdfplumber_candidate",
}

GRID_QUALITY_STATUSES = {"grid_needs_binding_review", "grid_rejected", "not_evaluable"}
FINAL_CASE_ACTIONS = {
    "manual_review_binding",
    "manual_review_layout",
    "chunk_fallback",
    "backlog",
    "exclude_current_pdfplumber_candidate",
}

KEEP_SOURCE_LABELS = {
    "alignment_confirmed_keep_only_after_cell_review",
    "keep_as_hybrid_candidate_requires_cell_grid_review",
    "keep_as_hybrid_candidate_requires_metric_binding_review",
    "keep_as_hybrid_candidate_requires_binding_review",
    "keep_as_hybrid_candidate_requires_cell_binding_review",
}

EXPECTED_BINDING_REVIEW_IDS = {
    "doc_0598__table_1__phase7c2_hybrid_01",
    "doc_0468__table_2__phase7c2_hybrid_01",
    "doc_0687__table_2__phase7c2_hybrid_02",
    "doc_0687__table_3__phase7c2_hybrid_03",
    "doc_0523__table_1__phase7c2_hybrid_01",
}

DECISION_FIELDS = [
    "hybrid_table_object_id",
    "doc_id",
    "table_id",
    "source_review_category",
    "source_review_decision",
    "grid_quality_status",
    "final_case_action",
    "review_evidence_summary",
    "review_notes",
    "original_source_review_decision",
    "alignment_review",
    "layout_review",
    "binding_review",
    "certainty",
]

GATED_SUMMARY_FIELDS = [
    "hybrid_table_object_id",
    "original_chunk_table_object_id",
    "pdfplumber_table_id",
    "doc_id",
    "table_id",
    "source_review_category",
    "source_review_decision",
    "grid_quality_status",
    "final_case_action",
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
    "review_evidence_summary",
    "original_hybrid_validation_status",
    "original_manual_review_reason",
    "original_recommended_next_action",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def read_input_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(rel(path))
            continue
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            record_count = sum(1 for line in text.splitlines() if line.strip())
        elif suffix == ".csv":
            record_count = max(0, len(text.splitlines()) - 1)
        else:
            record_count = 0
        inventory.append(
            {
                "path": rel(path),
                "line_count": len(text.splitlines()),
                "record_count": record_count,
                "bytes": len(text.encode("utf-8")),
            }
        )
    if missing:
        raise FileNotFoundError("缺少必读输入：" + "; ".join(missing))
    return inventory


def normalize_source_review_decision(row: dict[str, str]) -> str:
    original = row.get("source_review_decision", "")
    next_action = row.get("recommended_next_action", "")
    if original in KEEP_SOURCE_LABELS:
        return KEEP_DECISION
    if original == GRID_REJECT_DECISION:
        return GRID_REJECT_DECISION
    if original == "reject_selected_candidate_use_source_page2_only_as_text_signal":
        return CHUNK_FALLBACK_DECISION
    if original == REJECT_ALIGNMENT_DECISION:
        if next_action == "chunk_fallback":
            return CHUNK_FALLBACK_DECISION
        if next_action == "backlog":
            return BACKLOG_DECISION
        return REJECT_ALIGNMENT_DECISION
    raise ValueError(
        f"无法归一化 source_review_decision: {original} "
        f"({row.get('hybrid_table_object_id', '')})"
    )


def category_for_decision(decision: str) -> str:
    if decision == KEEP_DECISION:
        return "keep_hybrid_candidate_needs_binding_review"
    if decision == GRID_REJECT_DECISION:
        return "alignment_confirmed_grid_rejected"
    if decision in {REJECT_ALIGNMENT_DECISION, CHUNK_FALLBACK_DECISION, BACKLOG_DECISION}:
        return "reject_current_pdfplumber_candidate"
    raise ValueError(f"未知 source_review_decision: {decision}")


def grid_status_for_decision(decision: str) -> str:
    if decision == KEEP_DECISION:
        return "grid_needs_binding_review"
    if decision == GRID_REJECT_DECISION:
        return "grid_rejected"
    return "not_evaluable"


def action_for_decision(decision: str) -> str:
    if decision == KEEP_DECISION:
        return "manual_review_binding"
    if decision == GRID_REJECT_DECISION:
        return "manual_review_layout"
    if decision == CHUNK_FALLBACK_DECISION:
        return "chunk_fallback"
    if decision == BACKLOG_DECISION:
        return "backlog"
    if decision == REJECT_ALIGNMENT_DECISION:
        return "exclude_current_pdfplumber_candidate"
    raise ValueError(f"未知 source_review_decision: {decision}")


def validation_policy_for_decision(decision: str) -> tuple[str, str, str]:
    if decision == KEEP_DECISION:
        return "partial", "binding", "source_review_binding_review_required"
    if decision == GRID_REJECT_DECISION:
        return "manual_review", "layout_extraction", "source_review_rejected_pdfplumber_cell_grid"
    if decision == CHUNK_FALLBACK_DECISION:
        return "partial", "alignment", "source_review_selected_candidate_rejected_chunk_fallback"
    if decision == BACKLOG_DECISION:
        return "manual_review", "alignment", "source_review_pdf_text_layer_unresolved"
    if decision == REJECT_ALIGNMENT_DECISION:
        return "fail", "alignment", "source_review_current_pdfplumber_alignment_rejected"
    raise ValueError(f"未知 source_review_decision: {decision}")


def build_case_decisions(
    source_rows: list[dict[str, str]],
    hybrid_objects: list[dict[str, Any]],
    validation_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    object_ids = {obj.get("table_object_id", "") for obj in hybrid_objects}
    validation_ids = {row.get("hybrid_table_object_id", "") for row in validation_rows}
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        hybrid_id = row.get("hybrid_table_object_id", "")
        if hybrid_id in seen:
            raise ValueError(f"source review case 重复：{hybrid_id}")
        seen.add(hybrid_id)
        decision = normalize_source_review_decision(row)
        category = category_for_decision(decision)
        grid_status = grid_status_for_decision(decision)
        action = action_for_decision(decision)
        decisions.append(
            {
                "hybrid_table_object_id": hybrid_id,
                "doc_id": row.get("doc_id", ""),
                "table_id": row.get("table_id", ""),
                "source_review_category": category,
                "source_review_decision": decision,
                "grid_quality_status": grid_status,
                "final_case_action": action,
                "review_evidence_summary": row.get("evidence_note_cn", ""),
                "review_notes": "; ".join(
                    value
                    for value in [
                        row.get("alignment_review", ""),
                        row.get("layout_review", ""),
                        row.get("binding_review", ""),
                    ]
                    if value
                ),
                "original_source_review_decision": row.get("source_review_decision", ""),
                "alignment_review": row.get("alignment_review", ""),
                "layout_review": row.get("layout_review", ""),
                "binding_review": row.get("binding_review", ""),
                "certainty": row.get("certainty", ""),
            }
        )

    if len(decisions) != 16:
        raise ValueError(f"source review case 数量应为 16，实际为 {len(decisions)}")
    missing_objects = sorted(set(seen) - object_ids)
    missing_reviews = sorted(object_ids - set(seen))
    missing_validation = sorted(set(seen) - validation_ids)
    if missing_objects or missing_reviews or missing_validation:
        raise ValueError(
            "source review 覆盖不完整："
            f"source_not_in_hybrid={missing_objects}; "
            f"hybrid_not_in_source={missing_reviews}; "
            f"source_not_in_validation={missing_validation}"
        )

    validate_decision_distribution(decisions)
    return decisions


def validate_decision_distribution(decisions: list[dict[str, Any]]) -> None:
    decision_counts = Counter(row["source_review_decision"] for row in decisions)
    category_counts = Counter(row["source_review_category"] for row in decisions)
    action_counts = Counter(row["final_case_action"] for row in decisions)
    binding_ids = {row["hybrid_table_object_id"] for row in decisions if row["source_review_decision"] == KEEP_DECISION}
    reject_bucket = sum(
        decision_counts.get(decision, 0)
        for decision in [REJECT_ALIGNMENT_DECISION, CHUNK_FALLBACK_DECISION, BACKLOG_DECISION]
    )
    errors: list[str] = []
    if decision_counts.get(KEEP_DECISION, 0) != 5:
        errors.append(f"{KEEP_DECISION} 应为 5，实际 {decision_counts.get(KEEP_DECISION, 0)}")
    if decision_counts.get(GRID_REJECT_DECISION, 0) != 5:
        errors.append(f"{GRID_REJECT_DECISION} 应为 5，实际 {decision_counts.get(GRID_REJECT_DECISION, 0)}")
    if reject_bucket != 6:
        errors.append(f"reject/fallback/backlog 合计应为 6，实际 {reject_bucket}")
    if category_counts.get("keep_hybrid_candidate_needs_binding_review", 0) != 5:
        errors.append("binding review category 应为 5")
    if category_counts.get("alignment_confirmed_grid_rejected", 0) != 5:
        errors.append("grid rejected category 应为 5")
    if category_counts.get("reject_current_pdfplumber_candidate", 0) != 6:
        errors.append("reject category 应为 6")
    if action_counts.get("manual_review_binding", 0) != 5:
        errors.append("manual_review_binding action 应为 5")
    if binding_ids != EXPECTED_BINDING_REVIEW_IDS:
        errors.append(
            "binding review queue case 集合不一致："
            f"expected={sorted(EXPECTED_BINDING_REVIEW_IDS)} actual={sorted(binding_ids)}"
        )
    if any(row["source_review_decision"] not in SOURCE_REVIEW_DECISIONS for row in decisions):
        errors.append("source_review_decision 存在非法枚举")
    if any(row["source_review_category"] not in SOURCE_REVIEW_CATEGORIES for row in decisions):
        errors.append("source_review_category 存在非法枚举")
    if any(row["grid_quality_status"] not in GRID_QUALITY_STATUSES for row in decisions):
        errors.append("grid_quality_status 存在非法枚举")
    if any(row["final_case_action"] not in FINAL_CASE_ACTIONS for row in decisions):
        errors.append("final_case_action 存在非法枚举")
    if errors:
        raise ValueError("source review 分层校验失败：" + "; ".join(errors))


def split_warning_text(value: str) -> list[str]:
    if not value or value == "none":
        return []
    return [item for item in value.split(";") if item]


def add_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for item in additions:
        if item and item not in result:
            result.append(item)
    return result


def build_gated_outputs(
    decisions: list[dict[str, Any]],
    hybrid_objects: list[dict[str, Any]],
    validation_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    decision_by_id = {row["hybrid_table_object_id"]: row for row in decisions}
    validation_by_id = {row["hybrid_table_object_id"]: row for row in validation_rows}
    gated_objects: list[dict[str, Any]] = []
    gated_rows: list[dict[str, str]] = []
    for source_obj in hybrid_objects:
        obj = copy.deepcopy(source_obj)
        hybrid_id = obj.get("table_object_id", "")
        decision = decision_by_id[hybrid_id]
        original = validation_by_id[hybrid_id]
        status, stage, reason = validation_policy_for_decision(decision["source_review_decision"])
        action = decision["final_case_action"]
        meta = obj.setdefault("hybrid_metadata", {})
        meta.update(
            {
                "phase7c3_source_review_applied": True,
                "source_review_decision": decision["source_review_decision"],
                "source_review_category": decision["source_review_category"],
                "grid_quality_status": decision["grid_quality_status"],
                "final_case_action": action,
                "manual_review_reason": reason,
                "recommended_next_action": action,
                "pdfplumber_grid_reliable": False,
                "value_bboxes_available": False,
            }
        )
        obj.update(
            {
                "phase": "v7_phase7C_3_source_review_gate_hardening",
                "validation_status": status,
                "source_review_decision": decision["source_review_decision"],
                "source_review_category": decision["source_review_category"],
                "grid_quality_status": decision["grid_quality_status"],
                "final_case_action": action,
                "primary_failure_stage": stage,
                "manual_review_reason": reason,
                "recommended_next_action": action,
                "review_evidence_summary": decision["review_evidence_summary"],
                "review_notes": decision["review_notes"],
                "value_bboxes_available": False,
                "no_value_level_bbox": True,
            }
        )
        warnings = list(obj.get("warnings") or [])
        if decision["source_review_decision"] == KEEP_DECISION:
            warnings = add_unique(warnings, ["source_review_binding_review_required"])
        elif decision["source_review_decision"] == GRID_REJECT_DECISION:
            warnings = add_unique(warnings, ["source_review_grid_rejected", "pdfplumber_grid_not_reliable"])
        elif decision["source_review_decision"] == CHUNK_FALLBACK_DECISION:
            warnings = add_unique(warnings, ["source_review_selected_pdfplumber_candidate_rejected"])
        elif decision["source_review_decision"] == BACKLOG_DECISION:
            warnings = add_unique(warnings, ["source_review_pdf_text_layer_unresolved"])
        else:
            warnings = add_unique(warnings, ["source_review_pdfplumber_alignment_rejected"])
        warnings = add_unique(warnings, ["value_level_bbox_absent", "cell_bbox_not_value_bbox"])
        obj["warnings"] = warnings
        notes = list(obj.get("notes") or [])
        notes = add_unique(
            notes,
            [
                "Phase7C-3 source review gate: no case is upgraded to pass_with_warnings or production-ready.",
                f"Source review action: {action}.",
            ],
        )
        obj["notes"] = notes
        gated_objects.append(obj)

        blocking = split_warning_text(original.get("blocking_warnings", ""))
        nonblocking = split_warning_text(original.get("nonblocking_warnings", ""))
        if decision["source_review_decision"] == GRID_REJECT_DECISION:
            blocking = add_unique(blocking, ["source_review_grid_rejected"])
        elif decision["source_review_decision"] == KEEP_DECISION:
            blocking = add_unique(blocking, ["source_review_binding_review_required"])
        elif decision["source_review_decision"] == CHUNK_FALLBACK_DECISION:
            blocking = add_unique(blocking, ["source_review_selected_pdfplumber_candidate_rejected"])
        elif decision["source_review_decision"] == BACKLOG_DECISION:
            blocking = add_unique(blocking, ["source_review_pdf_text_layer_unresolved"])
        nonblocking = add_unique(nonblocking, ["value_bbox_absent_limitation", "cell_bbox_not_value_bbox"])
        gated_rows.append(
            {
                "hybrid_table_object_id": hybrid_id,
                "original_chunk_table_object_id": original.get("original_chunk_table_object_id", ""),
                "pdfplumber_table_id": original.get("pdfplumber_table_id", ""),
                "doc_id": obj.get("doc_id", ""),
                "table_id": obj.get("table_id", ""),
                "source_review_category": decision["source_review_category"],
                "source_review_decision": decision["source_review_decision"],
                "grid_quality_status": decision["grid_quality_status"],
                "final_case_action": action,
                "alignment_status": original.get("alignment_status", ""),
                "alignment_confidence": original.get("alignment_confidence", ""),
                "layout_quality_status": original.get("layout_quality_status", ""),
                "extraction_method": original.get("extraction_method", ""),
                "cell_bboxes_available": original.get("cell_bboxes_available", bool_text(False)),
                "value_bboxes_available": "false",
                "source_span_granularity": original.get("source_span_granularity", ""),
                "hybrid_validation_status": status,
                "primary_failure_stage": stage,
                "manual_review_reason": reason,
                "recommended_next_action": action,
                "blocking_warnings": ";".join(blocking) if blocking else "none",
                "nonblocking_warnings": ";".join(nonblocking) if nonblocking else "none",
                "notes": (
                    "Phase7C-3 按 source review 覆盖 Phase7C-2 gate；"
                    "不升级 pass_with_warnings，不写 production-ready，不伪造 value bbox。"
                ),
                "review_evidence_summary": decision["review_evidence_summary"],
                "original_hybrid_validation_status": original.get("hybrid_validation_status", ""),
                "original_manual_review_reason": original.get("manual_review_reason", ""),
                "original_recommended_next_action": original.get("recommended_next_action", ""),
            }
        )

    validate_gated_outputs(gated_rows, gated_objects)
    return gated_objects, gated_rows


def validate_gated_outputs(gated_rows: list[dict[str, str]], gated_objects: list[dict[str, Any]]) -> None:
    action_counts = Counter(row["final_case_action"] for row in gated_rows)
    grid_counts = Counter(row["grid_quality_status"] for row in gated_rows)
    reject_fallback_backlog = sum(
        1
        for row in gated_rows
        if row["final_case_action"] in {"chunk_fallback", "backlog", "exclude_current_pdfplumber_candidate"}
    )
    errors: list[str] = []
    if len(gated_rows) != 16 or len(gated_objects) != 16:
        errors.append(f"gated case 数量应为 16，实际 rows={len(gated_rows)} objects={len(gated_objects)}")
    if action_counts.get("manual_review_binding", 0) != 5:
        errors.append("binding review queue 数量应为 5")
    if grid_counts.get("grid_rejected", 0) != 5:
        errors.append("grid rejected 数量应为 5")
    if reject_fallback_backlog != 6:
        errors.append(f"rejected/fallback/backlog 合计应为 6，实际 {reject_fallback_backlog}")
    if any(row["hybrid_validation_status"] == "pass_with_warnings" for row in gated_rows):
        errors.append("不得直接升级 pass_with_warnings")
    if any(row.get("value_bboxes_available") != "false" for row in gated_rows):
        errors.append("value_bboxes_available 必须保持 false")
    if any(obj.get("production_ready") for obj in gated_objects):
        errors.append("不得写 production_ready=true")
    if errors:
        raise ValueError("gated validation 校验失败：" + "; ".join(errors))


def count_rows(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "")) for row in rows)


def markdown_counter(counter: Counter[str], order: list[str] | None = None) -> list[str]:
    lines = ["| 值 | 数量 |", "|---|---:|"]
    emitted: set[str] = set()
    if order:
        for key in order:
            lines.append(f"| `{key}` | {counter.get(key, 0)} |")
            emitted.add(key)
    for key, count in counter.most_common():
        if key not in emitted:
            lines.append(f"| `{key}` | {count} |")
    return lines


def write_phase7c3_guardrail(report_dir: Path, inventory: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7C-3 护栏",
        "",
        "## 1. 本轮定位",
        "",
        "本轮是 `v7-phase7C-3 Source-review-driven Gate Hardening`。目标是把 Phase7C-2 的 16 个 hybrid case 源文档逐案审阅结果固化为机器可读 decision layer，并据此收紧 layout、alignment 和 hybrid validation gate。",
        "",
        "本轮只做 source-review-driven gate hardening，不继续多抽表，不扩大 smoke，不接 production，不构造 gold，不运行 coverage，不进入 Route C implementation。",
        "",
        "## 2. Smoke 范围",
        "",
        "本轮固定同一批 smoke doc_id，未扩大 smoke：",
        "",
    ]
    lines.extend(f"- `{doc_id}`" for doc_id in SMOKE_DOC_IDS)
    lines.extend(
        [
            "",
            "## 3. 禁止事项",
            "",
            "- 不引入 Camelot。",
            "- 不引入 PyMuPDF。",
            "- 不引入 OCR / VLM。",
            "- 不调用 Qwen / RAGAS。",
            "- 不接入 RAG、retrieval、embedding 或 rerank。",
            "- 不访问 Milvus，不写入 Milvus。",
            "- 不读取或查询 BM25 index，不重建 BM25。",
            "- 不重建 chunks。",
            "- 不修改 ingestion 主链路或 production pipeline。",
            "- 不修改 official dataset、official baseline、baseline registry、configs 或 README。",
            "- 不构造 confirmed gold、row/cell gold。",
            "- 不运行 coverage evaluation。",
            "- 不伪造 value-level bbox，不把 cell bbox 写成 value-level bbox。",
            "",
            "## 4. Bbox 口径",
            "",
            "`pdfplumber` cell bbox 只表示 layout cell/grid provenance。Phase7C-3 不产生 token/value bbox，`value_bboxes_available` 预期保持 `false`。",
            "",
            "## 5. Route C",
            "",
            "Route C 仍只是 backlog。本轮不实施 Camelot/PyMuPDF/OCR/VLM 对照，不把 Route C 写成下一阶段立即实施项。",
            "",
            "## 6. 已读取输入清单",
            "",
            "| 输入 | 行数 | 记录数 | bytes |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in inventory:
        lines.append(f"| `{item['path']}` | {item['line_count']} | {item['record_count']} | {item['bytes']} |")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase7c_3_guardrail.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_review_decision_report(decisions: list[dict[str, Any]], report_dir: Path) -> None:
    decision_counts = count_rows(decisions, "source_review_decision")
    category_counts = count_rows(decisions, "source_review_category")
    grid_counts = count_rows(decisions, "grid_quality_status")
    action_counts = count_rows(decisions, "final_case_action")
    lines = [
        "# 源文档审阅决策报告",
        "",
        "## 1. 目标",
        "",
        "本报告把 Phase7C-2 的 16 个 hybrid case 源文档逐案审阅结论冻结为机器可读 decision layer。该 layer 只用于离线 gate hardening，不升级 confirmed，不写 production-ready。",
        "",
        "## 2. 覆盖与硬校验",
        "",
        f"- source review case 覆盖数：{len(decisions)} / 16",
        f"- binding review queue：{action_counts.get('manual_review_binding', 0)} / 5",
        f"- grid rejected：{grid_counts.get('grid_rejected', 0)} / 5",
        f"- rejected / fallback / backlog：{sum(action_counts.get(key, 0) for key in ['chunk_fallback', 'backlog', 'exclude_current_pdfplumber_candidate'])} / 6",
        "- direct upgrade to pass_with_warnings：0",
        "- production-ready：0",
        "",
        "## 3. source_review_decision 分布",
        "",
    ]
    lines.extend(markdown_counter(decision_counts))
    lines.extend(["", "## 4. source_review_category 分布", ""])
    lines.extend(markdown_counter(category_counts))
    lines.extend(["", "## 5. grid_quality_status 分布", ""])
    lines.extend(markdown_counter(grid_counts))
    lines.extend(["", "## 6. final_case_action 分布", ""])
    lines.extend(markdown_counter(action_counts))
    lines.extend(
        [
            "",
            "## 7. 逐案决策",
            "",
            "| case | doc_id | table_id | source_review_decision | grid_quality_status | final_case_action | evidence |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in decisions:
        lines.append(
            f"| `{row['hybrid_table_object_id']}` | `{row['doc_id']}` | `{row['table_id']}` | "
            f"`{row['source_review_decision']}` | `{row['grid_quality_status']}` | "
            f"`{row['final_case_action']}` | {row['review_evidence_summary']} |"
        )
    lines.extend(
        [
            "",
            "## 8. 口径说明",
            "",
            "- `keep_as_hybrid_candidate_requires_binding_review` 只表示可进入 binding review queue，不表示 confirmed。",
            "- `alignment_confirmed_reject_pdfplumber_cell_grid` 保留 alignment evidence，但拒绝当前 pdfplumber cell grid 作为可靠 cells。",
            "- `chunk_fallback` 只表示拒绝当前 pdfplumber candidate 后回退到 chunk object 审阅，不构造 row/cell gold。",
            "- `backlog_pdf_text_layer_unresolved` 表示当前 PDF 文本层未能提供可审阅目标表格，Route C 仍只是 backlog。",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "source_review_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_layout_quality_gate_report(
    raw_tables: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    raw_counts = Counter(table.get("layout_quality_status", "unknown") for table in raw_tables)
    grid_rejected = [row for row in decisions if row["grid_quality_status"] == "grid_rejected"]
    rejected = [row for row in decisions if row["source_review_category"] == "reject_current_pdfplumber_candidate"]
    lines = [
        "# layout_quality gate 更新报告",
        "",
        "## 1. 目标",
        "",
        "本报告记录 Phase7C-3 对 pdfplumber layout_quality gate 的收紧。收紧依据来自 16 个 source review case，而不是重新抽表或扩大 smoke。",
        "",
        "## 2. Phase7C-2 暴露的问题",
        "",
        f"- Phase7C-2 raw pdfplumber table 数量：{len(raw_tables)}",
        f"- Phase7C-2 layout_quality_status 分布：{dict(raw_counts)}",
        "- 多个 `usable` raw candidate 在源文档审阅后被确认是页面级正文混排、Figure/正文污染、单列线性化或错列表格。",
        "- `cell_bboxes_available=true` 只能说明 pdfplumber 给出了 layout cell bbox，不能证明该 grid 是目标表格的可靠 cells。",
        "",
        "## 3. 暴露 layout_quality 过宽的 case",
        "",
        "| case | decision | 主要证据 |",
        "|---|---|---|",
    ]
    for row in grid_rejected + rejected:
        lines.append(
            f"| `{row['hybrid_table_object_id']}` | `{row['source_review_decision']}` | {row['review_evidence_summary']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 固化的降级规则",
            "",
            "| rule_id | 可执行判断 | 降级效果 |",
            "|---|---|---|",
            "| `LQH001` | 页面级正文混排过多，尤其 candidate 覆盖大面积正文、多栏页面或正文段落 | `layout_quality_status` 不得为 `usable` |",
            "| `LQH002` | candidate 包含 `Figure` / `Fig.` / 图注 / journal header / page footer 等明显非表格区域 | 降级为 `weak` 或 `likely_false_positive` |",
            "| `LQH003` | 单列线性化表格或只有一列长文本 | 不能标记为 usable grid |",
            "| `LQH004` | `row_count` 很高但稳定 header 不在前几行 | 降级，避免整页正文被切成行 |",
            "| `LQH005` | `empty_cell_ratio` 高且伴随断词、断字、错列 | 降级为 weak / likely_false_positive |",
            "| `LQH006` | bbox 明显覆盖正文区、图区或整页多栏文本 | 降级，不能作为可靠 hybrid grid |",
            "| `LQH007` | source review 确认 alignment 但 grid 不可信 | `grid_quality_status=grid_rejected`，不得作为 usable hybrid cells |",
            "",
            "## 5. 规则对应的污染类型",
            "",
            "- 页面级正文混排：`doc_0322 Table 1`、`doc_0158 Table 2`、`doc_0452 Table 1` 暴露了正文/多栏文本被切成 grid 的风险。",
            "- Figure / 正文污染：`doc_0322 Table 1`、`doc_0598 Table 2`、`doc_0158 Table 3` 暴露了 Figure caption 或正文引用被误当作表格的风险。",
            "- 单列线性化：`doc_0687 Table 1`、`doc_0522 Table 1` 说明 alignment/text signal 可确认时，仍不能把单列文本当作 usable grid。",
            "- 断词 / 错列：`doc_0598 Table 1`、`doc_0468 Table 2`、`doc_0687 Table 2/3` 只能进入 binding review，不能直接 pass。",
            "",
            "## 6. 边界声明",
            "",
            "这些规则只是 Phase7C-3 离线 gate hardening，不等于 production parser。本轮不引入 Camelot，不引入 PyMuPDF，不重新抽表，不扩大 smoke。",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "layout_quality_gate_update_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_alignment_gate_report(
    alignment_rows: list[dict[str, str]],
    decisions: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    alignment_counts = Counter(row.get("alignment_status", "") for row in alignment_rows)
    confidence_counts = Counter(row.get("alignment_confidence", "") for row in alignment_rows)
    corrected = [row for row in decisions if row["source_review_decision"] != KEEP_DECISION]
    lines = [
        "# alignment gate 更新报告",
        "",
        "## 1. 目标",
        "",
        "本报告记录 Phase7C-3 对 chunk/pdf alignment gate 的收紧。alignment gate 只判断候选是否可能对应目标表，不等于 cell grid 或 binding 正确。",
        "",
        "## 2. Phase7C-2 alignment 统计",
        "",
        f"- alignment_status 分布：{dict(alignment_counts)}",
        f"- alignment_confidence 分布：{dict(confidence_counts)}",
        "",
        "## 3. source review 纠正的 case",
        "",
        "| case | source_review_decision | final_case_action | 纠正点 |",
        "|---|---|---|---|",
    ]
    for row in corrected:
        lines.append(
            f"| `{row['hybrid_table_object_id']}` | `{row['source_review_decision']}` | "
            f"`{row['final_case_action']}` | {row['review_evidence_summary']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 固化的 alignment gate 规则",
            "",
            "| rule_id | 规则 | 处理 |",
            "|---|---|---|",
            "| `AGH001` | `page_only_match` 永远不能 high confidence | 固定 low / manual review |",
            "| `AGH002` | `same_table_id` 不能只来自整页任意正文引用 | 需要 caption/table title 附近或目标 grid 附近证据 |",
            "| `AGH003` | 同页多个 pdfplumber candidates 时不能只靠 table_id | 同时考虑 layout_quality、bbox、caption/body proximity、text overlap |",
            "| `AGH004` | `conflict` / `multiple_pdf_tables` 不得直接 usable | 进入 manual_review 或 reject/fallback |",
            "| `AGH005` | source review 标记 reject 的 case 不得继续作为 usable hybrid | 写入 `final_case_action=chunk_fallback/backlog/exclude_current_pdfplumber_candidate` |",
            "| `AGH006` | `alignment_confirmed_grid_rejected` 保留 alignment evidence | 拒绝 pdfplumber cell grid，`grid_quality_status=grid_rejected` |",
            "| `AGH007` | `keep_hybrid_candidate_needs_binding_review` 只能进入 binding review queue | 不直接 pass，不 production-ready |",
            "",
            "## 5. high / medium matched 仍可能 grid 不可信",
            "",
            "Phase7C-2 中若干 high/medium matched case 经 source review 后仍被拒绝 grid，例如 `doc_0322 Table 1`、`doc_0158 Table 2`、`doc_0598 Table 2`、`doc_0452 Table 1`。原因是 alignment signal 能确认目标页或标题，但 pdfplumber grid 可能从正文、图注或页面多栏文本开始，无法代表目标表 cell structure。",
            "",
            "## 6. page_only / conflict / multiple candidates 处理",
            "",
            "- `page_only_match` 只说明页码相近或同页，缺少 caption/grid proximity 时保持 low/manual review。",
            "- `conflict` 表示最佳候选与页码、caption 或 body 信号不一致，不能直接 usable。",
            "- `multiple_pdf_tables` 表示同页候选无法由 table_id 单独消歧，必须人工审阅或拒绝当前候选。",
            "",
            "## 7. 边界声明",
            "",
            "alignment gate 不等于 binding correctness。即使 alignment 被 source review 确认，unit、footnote、reference、metric-level cell binding 未确认时仍只能进入 manual review / partial。",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "alignment_gate_update_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hybrid_validation_report(gated_rows: list[dict[str, str]], report_dir: Path) -> None:
    decision_counts = count_rows(gated_rows, "source_review_decision")
    grid_counts = count_rows(gated_rows, "grid_quality_status")
    action_counts = count_rows(gated_rows, "final_case_action")
    status_counts = count_rows(gated_rows, "hybrid_validation_status")
    stage_counts = count_rows(gated_rows, "primary_failure_stage")
    reason_counts = count_rows(gated_rows, "manual_review_reason")
    binding = [row for row in gated_rows if row["final_case_action"] == "manual_review_binding"]
    grid = [row for row in gated_rows if row["grid_quality_status"] == "grid_rejected"]
    rejected = [
        row
        for row in gated_rows
        if row["final_case_action"] in {"chunk_fallback", "backlog", "exclude_current_pdfplumber_candidate"}
    ]
    lines = [
        "# gated hybrid validation 报告",
        "",
        "## 1. 目标",
        "",
        "本报告按 Phase7C-3 source review decision layer 更新 hybrid validation。目标不是提高 pass 数量，而是让每个 case 的 final action 可复现。",
        "",
        "## 2. 总体统计",
        "",
        f"- gated case 数量：{len(gated_rows)}",
        "- direct pass / pass_with_warnings：0",
        "- production-ready：0",
        "- value_bboxes_available：全部 `false`",
        "",
        "## 3. source_review_decision 分布",
        "",
    ]
    lines.extend(markdown_counter(decision_counts))
    lines.extend(["", "## 4. grid_quality_status 分布", ""])
    lines.extend(markdown_counter(grid_counts))
    lines.extend(["", "## 5. final_case_action 分布", ""])
    lines.extend(markdown_counter(action_counts))
    lines.extend(["", "## 6. hybrid_validation_status 分布", ""])
    lines.extend(markdown_counter(status_counts, ["partial", "manual_review", "fail", "pass_with_warnings"]))
    lines.extend(["", "## 7. primary_failure_stage 分布", ""])
    lines.extend(markdown_counter(stage_counts))
    lines.extend(["", "## 8. manual_review_reason 分布", ""])
    lines.extend(markdown_counter(reason_counts))
    lines.extend(
        [
            "",
            "## 9. binding review 候选",
            "",
        ]
    )
    lines.extend(f"- `{row['hybrid_table_object_id']}`" for row in binding)
    lines.extend(["", "## 10. grid rejected case", ""])
    lines.extend(f"- `{row['hybrid_table_object_id']}`" for row in grid)
    lines.extend(["", "## 11. rejected / fallback / backlog case", ""])
    lines.extend(
        f"- `{row['hybrid_table_object_id']}` -> `{row['final_case_action']}`" for row in rejected
    )
    lines.extend(
        [
            "",
            "## 12. 判定规则",
            "",
            "- keep 候选只能进入 `manual_review_binding`，`hybrid_validation_status=partial`。",
            "- alignment confirmed 但 grid rejected 的 case 进入 `manual_review_layout`，不得使用 pdfplumber grid 作为 reliable cells。",
            "- chunk fallback case 进入 `partial`，当前 pdfplumber candidate 被拒绝。",
            "- backlog case 进入 `manual_review`，等待后续单独授权的文本层/Route C 处理。",
            "- 任何 case 都不得升级 confirmed、`pass_with_warnings` 或 production-ready。",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "hybrid_validation_gated_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def truncate(value: Any, limit: int = 72) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def table_preview(obj: dict[str, Any], max_rows: int = 5, max_cols: int = 6) -> list[str]:
    columns = obj.get("columns") or []
    cells = obj.get("cells") or []
    if not columns or not cells:
        rows = obj.get("rows") or []
        lines = ["| row | text |", "|---:|---|"]
        for index, row in enumerate(rows[:max_rows], start=1):
            lines.append(f"| {index} | {truncate(row.get('row_text') or row)} |")
        if len(lines) == 2:
            lines.append("|  | 无可显示 preview |")
        return lines
    col_ids = [col.get("column_id", "") for col in columns[:max_cols]]
    headers = [truncate(col.get("header") or col.get("header_path") or col.get("column_id"), 32) for col in columns[:max_cols]]
    by_row: dict[str, dict[str, str]] = {}
    row_order: list[str] = []
    for cell in cells:
        row_id = cell.get("row_id") or str(cell.get("row_index", ""))
        col_id = cell.get("column_id") or str(cell.get("column_index", ""))
        if row_id not in by_row:
            by_row[row_id] = {}
            row_order.append(row_id)
        by_row[row_id][col_id] = truncate(cell.get("value_raw") or cell.get("text"), 48)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row_id in row_order[:max_rows]:
        lines.append("| " + " | ".join(by_row[row_id].get(col_id, "") for col_id in col_ids) + " |")
    if len(lines) == 2:
        lines.append("| " + " | ".join("" for _ in headers) + " |")
    return lines


def write_gated_review(
    gated_objects: list[dict[str, Any]],
    gated_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    row_by_id = {row["hybrid_table_object_id"]: row for row in gated_rows}
    lines = [
        "# Phase7C-3 gated hybrid table_objects 审阅",
        "",
        "本 Markdown 只用于离线人工审阅。所有 case 均未升级 confirmed、pass_with_warnings 或 production-ready。",
        "",
    ]
    for obj in gated_objects:
        row = row_by_id[obj.get("table_object_id", "")]
        lines.extend(
            [
                f"## {row['hybrid_table_object_id']}",
                "",
                f"- hybrid_table_object_id：`{row['hybrid_table_object_id']}`",
                f"- doc_id：`{row['doc_id']}`",
                f"- table_id：`{row['table_id']}`",
                f"- source_review_decision：`{row['source_review_decision']}`",
                f"- grid_quality_status：`{row['grid_quality_status']}`",
                f"- final_case_action：`{row['final_case_action']}`",
                f"- hybrid_validation_status：`{row['hybrid_validation_status']}`",
                f"- primary_failure_stage：`{row['primary_failure_stage']}`",
                f"- manual_review_reason：`{row['manual_review_reason']}`",
                f"- recommended_next_action：`{row['recommended_next_action']}`",
                f"- cell_bboxes_available：`{row['cell_bboxes_available']}`",
                f"- value_bboxes_available：`{row['value_bboxes_available']}`",
                f"- source_span_granularity：`{row['source_span_granularity']}`",
                f"- review evidence summary：{row['review_evidence_summary']}",
                f"- warnings：`{truncate(';'.join(obj.get('warnings') or []), 320)}`",
                "",
                "### table preview",
                "",
            ]
        )
        lines.extend(table_preview(obj))
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hybrid_table_objects_gated_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_lists(decisions: list[dict[str, Any]], output_dir: Path) -> None:
    binding = [row for row in decisions if row["final_case_action"] == "manual_review_binding"]
    grid = [row for row in decisions if row["grid_quality_status"] == "grid_rejected"]
    rejected = [
        row
        for row in decisions
        if row["final_case_action"] in {"chunk_fallback", "backlog", "exclude_current_pdfplumber_candidate"}
    ]
    backlog = [row for row in decisions if row["final_case_action"] == "backlog"]
    write_jsonl(binding, output_dir / "hybrid_candidates_for_binding_review.jsonl")
    write_csv(grid, output_dir / "hybrid_grid_rejected_cases.csv", DECISION_FIELDS)
    write_csv(rejected, output_dir / "hybrid_rejected_or_fallback_cases.csv", DECISION_FIELDS)
    write_csv(backlog, output_dir / "hybrid_backlog_cases.csv", DECISION_FIELDS)


def list_ids(rows: list[dict[str, Any]], key: str = "hybrid_table_object_id") -> list[str]:
    return [str(row.get(key, "")) for row in rows]


def write_comparison_report(
    phase7c2_rows: list[dict[str, str]],
    gated_rows: list[dict[str, str]],
    report_dir: Path,
) -> None:
    binding = [row for row in gated_rows if row["final_case_action"] == "manual_review_binding"]
    grid = [row for row in gated_rows if row["grid_quality_status"] == "grid_rejected"]
    rejected = [
        row
        for row in gated_rows
        if row["final_case_action"] in {"chunk_fallback", "backlog", "exclude_current_pdfplumber_candidate"}
    ]
    lines = [
        "# Phase7C-2 vs Phase7C-3 对比报告",
        "",
        "## 1. 对比目标",
        "",
        "本报告比较 Phase7C-2 与 Phase7C-3 在同一批 16 个 hybrid case 上的 gate 变化。Phase7C-3 不是为了提高 pass 数量，而是把 source review 决策机器化。",
        "",
        "## 2. 数量对比",
        "",
        f"- Phase7C-2 hybrid table_objects 数量：{len(phase7c2_rows)}",
        f"- Phase7C-3 gated case 数量：{len(gated_rows)}",
        "",
        "## 3. source_review_decision 分布",
        "",
    ]
    lines.extend(markdown_counter(count_rows(gated_rows, "source_review_decision")))
    lines.extend(["", "## 4. grid_quality_status 分布", ""])
    lines.extend(markdown_counter(count_rows(gated_rows, "grid_quality_status")))
    lines.extend(["", "## 5. final_case_action 分布", ""])
    lines.extend(markdown_counter(count_rows(gated_rows, "final_case_action")))
    lines.extend(["", "## 6. hybrid_validation_status 变化", ""])
    lines.append(f"- Phase7C-2：{dict(count_rows(phase7c2_rows, 'hybrid_validation_status'))}")
    lines.append(f"- Phase7C-3：{dict(count_rows(gated_rows, 'hybrid_validation_status'))}")
    lines.extend(
        [
            "",
            "## 7. manual_review_reason",
            "",
            f"- Phase7C-3 manual_review_reason 分布：{dict(count_rows(gated_rows, 'manual_review_reason'))}",
            "- 相比 Phase7C-2，原因从 alignment/boundary/binding 的自动 gate 解释，进一步细化为 source review 结论：binding review、grid rejected、chunk fallback、PDF 文本层 unresolved。",
            "",
            "## 8. 三类清单",
            "",
            "### 5 个 binding review 候选",
            "",
        ]
    )
    lines.extend(f"- `{row['hybrid_table_object_id']}`" for row in binding)
    lines.extend(["", "### 5 个 grid rejected case", ""])
    lines.extend(f"- `{row['hybrid_table_object_id']}`" for row in grid)
    lines.extend(["", "### 6 个 rejected / fallback / backlog case", ""])
    lines.extend(f"- `{row['hybrid_table_object_id']}` -> `{row['final_case_action']}`" for row in rejected)
    lines.extend(
        [
            "",
            "## 9. 关键结论",
            "",
            "- 是否仍有 direct pass：没有。",
            "- 是否仍有 production-ready：没有。",
            "- 是否更可复现 source review 结论：是，16 个 case 均写入机器可读 decision layer。",
            "- 是否建议继续 pdfplumber 主线：建议继续作为离线 hardening 主线，但必须受 source review gate 约束。",
            "- 是否建议扩大 smoke：不建议。本轮先固定同一批 smoke。",
            "- 是否建议引入 Camelot / PyMuPDF：不建议在下一阶段立即实施，保持 backlog。",
            "- 是否建议 production：不建议。",
            "- Route C 是否仍只是 backlog：是。",
            "",
            "Phase7C-3 的核心改善是 gate 更诚实。5 个可保留候选只是进入 binding review queue，不是 confirmed；Camelot / PyMuPDF 保持 backlog，不进入下一阶段实施。",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase7c2_vs_phase7c3_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(
    output_dir: Path,
    report_dir: Path,
    gated_rows: list[dict[str, str]],
) -> None:
    binding = [row for row in gated_rows if row["final_case_action"] == "manual_review_binding"]
    grid = [row for row in gated_rows if row["grid_quality_status"] == "grid_rejected"]
    rejected = [
        row
        for row in gated_rows
        if row["final_case_action"] in {"chunk_fallback", "backlog", "exclude_current_pdfplumber_candidate"}
    ]
    generated_data = [
        "hybrid_case_decisions.jsonl",
        "hybrid_case_decision_summary.csv",
        "hybrid_table_objects_gated.jsonl",
        "hybrid_candidates_for_binding_review.jsonl",
        "hybrid_grid_rejected_cases.csv",
        "hybrid_rejected_or_fallback_cases.csv",
        "hybrid_backlog_cases.csv",
        "hybrid_table_objects_gated_review.md",
    ]
    generated_reports = [
        "phase7c_3_guardrail.md",
        "source_review_decision_report.md",
        "layout_quality_gate_update_report.md",
        "alignment_gate_update_report.md",
        "hybrid_validation_gated_summary.csv",
        "hybrid_validation_gated_report.md",
        "phase7c2_vs_phase7c3_comparison.md",
        "phase7c_3_summary.md",
    ]
    lines = [
        "# Phase7C-3 总结",
        "",
        "## 1. 本轮生成文件",
        "",
    ]
    lines.extend(f"- `{rel(output_dir / name)}`" for name in generated_data)
    lines.extend(f"- `{rel(report_dir / name)}`" for name in generated_reports)
    lines.extend(
        [
            "",
            "## 2. 修改脚本与测试",
            "",
            "- 新增脚本：`scripts/extraction/apply_hybrid_source_review_decisions.py`",
            "- 修改脚本：`scripts/extraction/extract_tables_pdfplumber_v1.py`",
            "- 修改脚本：`scripts/extraction/align_chunk_pdfplumber_tables.py`",
            "- 新增测试：`tests/test_phase7_source_review_gate.py`",
            "",
            "## 3. Smoke 与覆盖",
            "",
            f"- smoke doc_id 是否保持不变：是，仍为 {', '.join(SMOKE_DOC_IDS)}。",
            f"- source review 覆盖 case 数：{len(gated_rows)} / 16。",
            "",
            "## 4. 统计",
            "",
            f"- source_review_decision：{dict(count_rows(gated_rows, 'source_review_decision'))}",
            f"- grid_quality_status：{dict(count_rows(gated_rows, 'grid_quality_status'))}",
            f"- final_case_action：{dict(count_rows(gated_rows, 'final_case_action'))}",
            f"- hybrid_validation_status：{dict(count_rows(gated_rows, 'hybrid_validation_status'))}",
            "",
            "## 5. 三类清单",
            "",
            f"- binding review 候选数量：{len(binding)}",
        ]
    )
    lines.extend(f"  - `{row['hybrid_table_object_id']}`" for row in binding)
    lines.append(f"- grid rejected 数量：{len(grid)}")
    lines.extend(f"  - `{row['hybrid_table_object_id']}`" for row in grid)
    lines.append(f"- rejected / fallback / backlog 数量：{len(rejected)}")
    lines.extend(f"  - `{row['hybrid_table_object_id']}` -> `{row['final_case_action']}`" for row in rejected)
    lines.extend(
        [
            "",
            "## 6. 结论",
            "",
            "- direct pass / production-ready 数量：0 / 0。",
            "- 是否复现 source review 结论：是，5/5/6 分层已机器化并通过硬校验。",
            "- 相比 Phase7C-2 的主要改善：source review 决策进入 JSONL/CSV/report，layout/grid/alignment 风险不再隐藏在 usable hybrid object 中。",
            "- 仍然存在的问题：binding review 尚未执行，grid rejected case 需要更可靠 parser 或人工 layout 审阅，doc_0458 文本层 unresolved。",
            "- 是否建议继续 pdfplumber 主线：建议继续离线 hardening，但不作为 production parser。",
            "- 是否建议进行 binding review：建议，只针对 5 个 queue case。",
            "- 是否建议扩大 smoke：不建议。",
            "- 是否建议引入 Camelot / PyMuPDF：不建议立即实施，保持 backlog。",
            "- 是否建议进入 production：不建议。",
            "- baseline / guardrail 是否漂移：未发现漂移。",
            "- Route C 是否仍只是 backlog：是。",
            "",
            "## 7. 明确未执行事项",
            "",
            "- 未扩大 smoke。",
            "- 未引入 Camelot。",
            "- 未引入 PyMuPDF。",
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
            "- 未构造 confirmed gold。",
            "- 未运行 coverage。",
            "- 未接入 production。",
            "- 未进入 Route C。",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase7c_3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review-csv", type=Path, default=DEFAULT_SOURCE_REVIEW_CSV)
    parser.add_argument("--source-review-md", type=Path, default=DEFAULT_SOURCE_REVIEW_MD)
    parser.add_argument("--hybrid-objects", type=Path, default=DEFAULT_HYBRID_OBJECTS)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--raw-tables", type=Path, default=DEFAULT_RAW_TABLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_review_csv = resolve_path(args.source_review_csv)
    args.source_review_md = resolve_path(args.source_review_md)
    args.hybrid_objects = resolve_path(args.hybrid_objects)
    args.validation_summary = resolve_path(args.validation_summary)
    args.alignment = resolve_path(args.alignment)
    args.raw_tables = resolve_path(args.raw_tables)
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)

    required_inputs = list(dict.fromkeys(REQUIRED_PHASE7C2_INPUTS + REQUIRED_PHASE6D_INPUTS))
    inventory = read_input_inventory(required_inputs)
    source_rows = load_csv(args.source_review_csv)
    hybrid_objects = load_jsonl(args.hybrid_objects)
    validation_rows = load_csv(args.validation_summary)
    alignment_rows = load_csv(args.alignment)
    raw_tables = load_jsonl(args.raw_tables)

    decisions = build_case_decisions(source_rows, hybrid_objects, validation_rows)
    gated_objects, gated_rows = build_gated_outputs(decisions, hybrid_objects, validation_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(decisions, args.output_dir / "hybrid_case_decisions.jsonl")
    write_csv(decisions, args.output_dir / "hybrid_case_decision_summary.csv", DECISION_FIELDS)
    write_jsonl(gated_objects, args.output_dir / "hybrid_table_objects_gated.jsonl")
    write_case_lists(decisions, args.output_dir)
    write_csv(gated_rows, args.report_dir / "hybrid_validation_gated_summary.csv", GATED_SUMMARY_FIELDS)
    write_gated_review(gated_objects, gated_rows, args.output_dir)

    write_phase7c3_guardrail(args.report_dir, inventory)
    write_source_review_decision_report(decisions, args.report_dir)
    write_layout_quality_gate_report(raw_tables, decisions, args.report_dir)
    write_alignment_gate_report(alignment_rows, decisions, args.report_dir)
    write_hybrid_validation_report(gated_rows, args.report_dir)
    write_comparison_report(validation_rows, gated_rows, args.report_dir)
    write_summary_report(args.output_dir, args.report_dir, gated_rows)

    print("Phase7C-3 source-review gate hardening outputs written.")
    print(f"decisions={len(decisions)} gated_rows={len(gated_rows)}")
    print(f"source_review_decision={dict(count_rows(gated_rows, 'source_review_decision'))}")
    print(f"grid_quality_status={dict(count_rows(gated_rows, 'grid_quality_status'))}")
    print(f"final_case_action={dict(count_rows(gated_rows, 'final_case_action'))}")
    print(f"hybrid_validation_status={dict(count_rows(gated_rows, 'hybrid_validation_status'))}")


if __name__ == "__main__":
    main()
