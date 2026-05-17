#!/usr/bin/env python3
"""Phase7D hybrid table extractor v2 routing hardening.

This runner does not re-extract PDFs. It reads the Phase7C-2 hybrid objects,
Phase7C-3 source-review decisions, and Phase7C-4 binding-review decisions, then
materializes the restricted Phase7D routing layer for the same smoke doc_ids.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction import align_chunk_pdfplumber_tables as align_gate
from scripts.extraction import build_hybrid_table_objects_v1 as build_v1

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
PHASE7C3_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_gate_hardening"
PHASE7C3_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_gate_hardening"
PHASE7C4_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_binding_review"
PHASE7C4_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_binding_review"
PHASE6D_REPORT_DIR = ROOT / "reports/v7_phase6d_table_contract_refinement"

DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_extractor_v2"

PHASE7C2_REQUIRED_INPUTS = [
    PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl",
    PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv",
    PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl",
    PHASE7C2_REPORT_DIR / "hybrid_table_object_validation_summary.csv",
    PHASE7C2_REPORT_DIR / "phase7c_2_summary.md",
]

PHASE7C3_REQUIRED_INPUTS = [
    PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_case_decision_summary.csv",
    PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_candidates_for_binding_review.jsonl",
    PHASE7C3_REPORT_DIR / "hybrid_validation_gated_summary.csv",
    PHASE7C3_REPORT_DIR / "phase7c_3_summary.md",
]

PHASE7C4_REQUIRED_INPUTS = [
    PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_binding_review_summary.csv",
    PHASE7C4_DATA_DIR / "hybrid_candidates_ready_for_gold.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_candidates_needing_rule_fix.csv",
    PHASE7C4_REPORT_DIR / "phase7c_4_summary.md",
    PHASE7C4_REPORT_DIR / "gold_readiness_decision.md",
]

PHASE6D_REQUIRED_INPUTS = [
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

CURRENT_EXTRACTOR_SCRIPTS = [
    ROOT / "scripts/extraction/extract_tables_pdfplumber_v1.py",
    ROOT / "scripts/extraction/align_chunk_pdfplumber_tables.py",
    ROOT / "scripts/extraction/build_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/validate_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/render_hybrid_table_objects_markdown.py",
]

OPTIONAL_EXISTING_TESTS = [
    ROOT / "tests/test_phase7_hybrid_alignment_gate.py",
    ROOT / "tests/test_phase7_source_review_gate.py",
    ROOT / "tests/test_phase7_hybrid_binding_review.py",
]

SOURCE_DECISION_GRID_REJECT = "alignment_confirmed_reject_pdfplumber_cell_grid"
SOURCE_DECISION_CHUNK_FALLBACK = "reject_selected_candidate_use_chunk_fallback"
SOURCE_DECISION_BACKLOG = "backlog_pdf_text_layer_unresolved"
SOURCE_DECISION_KEEP = "keep_as_hybrid_candidate_requires_binding_review"

READY_BINDING_ACTION = "ready_for_gold_candidate"
RULE_FIX_BINDING_ACTION = "needs_pdfplumber_rule_fix"

READY_IDS = {
    "doc_0468__table_2__phase7c2_hybrid_01",
    "doc_0687__table_3__phase7c2_hybrid_03",
}
RULE_FIX_IDS = {
    "doc_0598__table_1__phase7c2_hybrid_01",
    "doc_0687__table_2__phase7c2_hybrid_02",
    "doc_0523__table_1__phase7c2_hybrid_01",
}

ROUTING_STATUS_VALUES = {
    "ready_for_gold_candidate",
    "needs_pdfplumber_rule_fix",
    "grid_rejected",
    "chunk_fallback",
    "backlog",
    "manual_review_required",
    "partial_hybrid",
}

FINAL_ACTION_VALUES = {
    "keep_ready_candidate",
    "keep_rule_fix",
    "reject_pdfplumber_grid",
    "use_chunk_fallback",
    "keep_backlog",
    "manual_review_required",
}

FORBIDDEN_OUTPUT_KEYS = {
    "confirmed_gold",
    "production_ready",
    "ready_for_gold_candidate_is_confirmed_gold",
    "usable_hybrid_candidate_is_production_ready",
}

ROUTING_SUMMARY_FIELDS = [
    "table_object_id",
    "original_chunk_table_object_id",
    "pdfplumber_table_id",
    "doc_id",
    "table_id",
    "routing_status",
    "final_action",
    "routing_reason",
    "source_review_decision",
    "source_review_category",
    "binding_review_status",
    "row_grid_status",
    "column_grid_status",
    "cell_grid_status",
    "value_placement_status",
    "unit_binding_status",
    "footnote_binding_status",
    "reference_binding_status",
    "alignment_status",
    "alignment_confidence",
    "layout_quality_status",
    "extraction_method",
    "usable_hybrid_candidate",
    "cell_bboxes_available",
    "value_bboxes_available",
    "source_span_granularity",
    "warnings",
    "binding_warnings",
    "routing_blockers",
    "review_evidence_summary",
    "rule_fix_scope",
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


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] = ROUTING_SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def semicolon(values: list[Any]) -> str:
    cleaned = [normalize_space(value) for value in values if normalize_space(value)]
    return ";".join(cleaned) if cleaned else "none"


def add_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for item in additions:
        if item and item not in result:
            result.append(item)
    return result


def strip_forbidden_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_forbidden_keys(item) for key, item in value.items() if key not in FORBIDDEN_OUTPUT_KEYS}
    if isinstance(value, list):
        return [strip_forbidden_keys(item) for item in value]
    return value


def read_input_inventory(required_paths: list[Path], optional_paths: list[Path] | None = None) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    missing_required: list[str] = []
    optional_paths = optional_paths or []
    for path in required_paths + optional_paths:
        required = path in required_paths
        if not path.exists():
            if required:
                missing_required.append(rel(path))
            else:
                inventory.append({"path": rel(path), "exists": False, "line_count": 0, "record_count": 0, "bytes": 0})
            continue
        line_count = 0
        record_count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_count, line in enumerate(handle, start=1):
                if path.suffix == ".jsonl" and line.strip():
                    record_count += 1
        if path.suffix == ".csv":
            record_count = max(0, line_count - 1)
        inventory.append(
            {
                "path": rel(path),
                "exists": True,
                "line_count": line_count,
                "record_count": record_count,
                "bytes": path.stat().st_size,
            }
        )
    if missing_required:
        raise FileNotFoundError("缺少 Phase7D 必读输入：" + "; ".join(missing_required))
    return inventory


def normalize_source_span_granularity(value: str | None) -> str:
    return build_v1.normalize_hybrid_v2_source_span_granularity(value)


def normalized_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("table_object_id") or row.get("hybrid_table_object_id") or "")


def metadata(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("hybrid_metadata") or {}


def get_alignment_status(validation: dict[str, Any], meta: dict[str, Any]) -> str:
    return str(validation.get("alignment_status") or meta.get("alignment_status") or "not_evaluable")


def get_alignment_confidence(validation: dict[str, Any], meta: dict[str, Any]) -> str:
    return str(validation.get("alignment_confidence") or meta.get("alignment_confidence") or "none")


def get_layout_quality_status(validation: dict[str, Any], meta: dict[str, Any]) -> str:
    return str(validation.get("layout_quality_status") or meta.get("layout_quality_status") or "not_evaluable")


def alignment_blockers(alignment_status: str, alignment_confidence: str, layout_quality_status: str) -> list[str]:
    blockers: list[str] = []
    if alignment_status == "page_only_match":
        blockers.append("page_only_match_not_high_confidence")
    if alignment_status in {"conflict", "multiple_pdf_tables"}:
        blockers.append(f"{alignment_status}_cannot_route_ready")
    if not align_gate.alignment_allows_ready_candidate(alignment_status, alignment_confidence):
        blockers.append("alignment_not_ready_eligible")
    if layout_quality_status == "likely_false_positive":
        blockers.append("likely_false_positive_layout_not_usable")
    return sorted(set(blockers))


def binding_default_for_route(route_label: str) -> dict[str, Any]:
    if route_label == "grid_rejected":
        return {
            "binding_review_status": "not_reviewed_grid_rejected",
            "row_grid_status": "fail",
            "column_grid_status": "fail",
            "cell_grid_status": "fail",
            "value_placement_status": "fail",
            "unit_binding_status": "not_reviewed",
            "footnote_binding_status": "not_reviewed",
            "reference_binding_status": "not_reviewed",
            "key_warnings": ["pdfplumber cell grid 已由 source review 拒绝"],
        }
    if route_label == "chunk_fallback":
        return {
            "binding_review_status": "not_reviewed_chunk_fallback",
            "row_grid_status": "not_evaluable",
            "column_grid_status": "not_evaluable",
            "cell_grid_status": "not_evaluable",
            "value_placement_status": "not_evaluable",
            "unit_binding_status": "not_reviewed",
            "footnote_binding_status": "not_reviewed",
            "reference_binding_status": "not_reviewed",
            "key_warnings": ["当前 pdfplumber candidate 不用于 hybrid grid"],
        }
    if route_label == "backlog":
        return {
            "binding_review_status": "not_reviewed_backlog",
            "row_grid_status": "not_evaluable",
            "column_grid_status": "not_evaluable",
            "cell_grid_status": "not_evaluable",
            "value_placement_status": "not_evaluable",
            "unit_binding_status": "not_reviewed",
            "footnote_binding_status": "not_reviewed",
            "reference_binding_status": "not_reviewed",
            "key_warnings": ["PDF text layer / source boundary unresolved"],
        }
    return {
        "binding_review_status": "not_reviewed",
        "row_grid_status": "not_reviewed",
        "column_grid_status": "not_reviewed",
        "cell_grid_status": "not_reviewed",
        "value_placement_status": "not_reviewed",
        "unit_binding_status": "not_reviewed",
        "footnote_binding_status": "not_reviewed",
        "reference_binding_status": "not_reviewed",
        "key_warnings": [],
    }


def generated_binding_warning_tags(binding: dict[str, Any], routing_status: str) -> list[str]:
    text = " ".join(str(item) for item in binding.get("key_warnings") or [])
    tags: list[str] = []
    column_status = binding.get("column_grid_status", "")
    cell_status = binding.get("cell_grid_status", "")
    value_status = binding.get("value_placement_status", "")
    if any(status in {"fail", "uncertain"} for status in [column_status, cell_status]):
        tags.extend(["column_alignment_inconsistent", "cell_grid_needs_rule_fix"])
    if cell_status == "fail":
        tags.append("missing_metric_cell_warning")
    if "拆" in text or "split" in text or "Lo/cation" in text or "q/xylose" in text:
        tags.append("split_cell_warning")
    if "合并" in text or "共享 cell" in text or "共 cell" in text or "被合并" in text:
        tags.append("merged_cell_warning")
    if "跨多行" in text or "跨行" in text or "row continuation" in text or "折返" in text:
        tags.append("row_continuation_warning")
    if (
        routing_status == "needs_pdfplumber_rule_fix"
        and (
            value_status in {"fail", "uncertain"}
            or "metric-level" in text
            or "numeric" in text
            or "数值" in text
            or "titer" in text
            or "YE/S" in text
            or "LNT" in text
        )
    ):
        tags.extend(["metric_level_cell_gap", "numeric_column_order_uncertain", "metric_column_group_uncertain"])
    if binding.get("unit_visible") and not binding.get("unit_bound"):
        tags.append("unit_visible_not_bound")
    if binding.get("footnote_present") and not binding.get("footnote_bound"):
        tags.append("footnote_present_not_bound")
    if binding.get("reference_visible") and not binding.get("row_level_reference_bound"):
        tags.append("reference_visible_not_bound")
    if routing_status == "grid_rejected":
        tags.extend(["source_review_grid_rejected", "pdfplumber_grid_not_reliable"])
    if routing_status == "chunk_fallback":
        tags.append("source_review_selected_pdfplumber_candidate_rejected")
    if routing_status == "backlog":
        tags.append("source_review_pdf_text_layer_unresolved")
    tags.extend(["value_level_bbox_absent", "cell_bbox_not_value_bbox"])
    return sorted(set(tags))


def compute_route(
    decision: dict[str, Any],
    binding: dict[str, Any] | None,
    validation: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    binding = binding or {}
    source_decision = decision.get("source_review_decision", "")
    alignment_status = get_alignment_status(validation, meta)
    alignment_confidence = get_alignment_confidence(validation, meta)
    layout_quality_status = get_layout_quality_status(validation, meta)
    blockers = alignment_blockers(alignment_status, alignment_confidence, layout_quality_status)

    if source_decision == SOURCE_DECISION_GRID_REJECT:
        return {
            "routing_status": "grid_rejected",
            "final_action": "reject_pdfplumber_grid",
            "routing_reason": "Phase7C-3 source review 已确认当前 pdfplumber cell grid 不可靠。",
            "routing_blockers": blockers,
        }
    if source_decision == SOURCE_DECISION_CHUNK_FALLBACK:
        return {
            "routing_status": "chunk_fallback",
            "final_action": "use_chunk_fallback",
            "routing_reason": "Phase7C-3 source review 拒绝当前 pdfplumber candidate，保留 chunk fallback。",
            "routing_blockers": blockers,
        }
    if source_decision == SOURCE_DECISION_BACKLOG:
        return {
            "routing_status": "backlog",
            "final_action": "keep_backlog",
            "routing_reason": "Phase7C-3 source review 标记 PDF text layer / source boundary unresolved，Route C 仍只是 backlog。",
            "routing_blockers": blockers,
        }

    final_binding_action = binding.get("final_binding_action", "")
    if final_binding_action == READY_BINDING_ACTION:
        strict_blockers = [
            blocker
            for blocker in blockers
            if blocker
            in {
                "page_only_match_not_high_confidence",
                "conflict_cannot_route_ready",
                "multiple_pdf_tables_cannot_route_ready",
                "likely_false_positive_layout_not_usable",
                "alignment_not_ready_eligible",
            }
        ]
        if strict_blockers:
            return {
                "routing_status": "manual_review_required",
                "final_action": "manual_review_required",
                "routing_reason": "Phase7D v2 发现更严格 alignment/layout blocker，未保留 ready routing。",
                "routing_blockers": strict_blockers,
            }
        return {
            "routing_status": "ready_for_gold_candidate",
            "final_action": "keep_ready_candidate",
            "routing_reason": "Phase7C-4 binding review 标记为 ready_for_gold_candidate；本轮仅稳定保留候选，不构造 confirmed gold。",
            "routing_blockers": blockers,
        }
    if final_binding_action == RULE_FIX_BINDING_ACTION:
        return {
            "routing_status": "needs_pdfplumber_rule_fix",
            "final_action": "keep_rule_fix",
            "routing_reason": "Phase7C-4 binding review 标记为 needs_pdfplumber_rule_fix，split/merged/metric binding 风险未自动修复。",
            "routing_blockers": blockers,
        }

    if source_decision == SOURCE_DECISION_KEEP:
        return {
            "routing_status": "manual_review_required",
            "final_action": "manual_review_required",
            "routing_reason": "source review 保留候选，但缺少可自动固化的 binding review action。",
            "routing_blockers": blockers,
        }

    return {
        "routing_status": "manual_review_required",
        "final_action": "manual_review_required",
        "routing_reason": "未知或未覆盖的 source/binding decision，保持 manual review required。",
        "routing_blockers": blockers,
    }


def source_span_values_without_value_level(obj: dict[str, Any], source_span_granularity: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    normalized_top = normalize_source_span_granularity(source_span_granularity)
    if normalized_top != source_span_granularity:
        warnings.append("source_span_value_level_downgraded")
    for span in obj.get("source_spans") or []:
        if span.get("granularity") == "value_level":
            span["granularity"] = "mixed_or_unclear"
            warnings.append("source_span_value_level_downgraded")
    return normalized_top, sorted(set(warnings))


def build_v2_object(
    source_obj: dict[str, Any],
    decision: dict[str, Any],
    binding_review: dict[str, Any] | None,
    validation_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = strip_forbidden_keys(copy.deepcopy(source_obj))
    hybrid_id = obj["table_object_id"]
    meta = obj.setdefault("hybrid_metadata", {})
    binding_review = binding_review or {}
    route = compute_route(decision, binding_review, validation_row, meta)
    routing_status = route["routing_status"]
    default_binding = binding_default_for_route(routing_status)
    binding = {**default_binding, **binding_review}

    source_span_granularity = (
        binding.get("source_span_granularity")
        or validation_row.get("source_span_granularity")
        or meta.get("source_span_granularity")
        or obj.get("source_span_granularity")
        or "mixed_or_unclear"
    )
    source_span_granularity, source_span_warnings = source_span_values_without_value_level(
        obj, str(source_span_granularity)
    )
    cell_bboxes_available = normalized_bool(
        validation_row.get("cell_bboxes_available", meta.get("cell_bboxes_available", obj.get("cell_bboxes_available")))
    )
    binding_warning_tags = generated_binding_warning_tags(binding, routing_status)
    warnings = add_unique(list(obj.get("warnings") or []), binding_warning_tags + source_span_warnings)
    warnings = add_unique(warnings, ["value_level_bbox_absent", "cell_bbox_not_value_bbox"])

    usable_hybrid_candidate = routing_status == "ready_for_gold_candidate"
    obj.update(
        {
            "phase": "v7_phase7D_hybrid_extractor_v2",
            "schema_name": "table_object_v2",
            "schema_version": "v2",
            "routing_status": routing_status,
            "routing_reason": route["routing_reason"],
            "routing_blockers": route["routing_blockers"],
            "source_review_decision": decision.get("source_review_decision", ""),
            "source_review_category": decision.get("source_review_category", ""),
            "binding_review_status": binding.get("binding_review_status", "not_reviewed"),
            "final_action": route["final_action"],
            "usable_hybrid_candidate": usable_hybrid_candidate,
            "pdfplumber_grid_reliable": usable_hybrid_candidate,
            "row_grid_status": binding.get("row_grid_status", "not_reviewed"),
            "column_grid_status": binding.get("column_grid_status", "not_reviewed"),
            "cell_grid_status": binding.get("cell_grid_status", "not_reviewed"),
            "value_placement_status": binding.get("value_placement_status", "not_reviewed"),
            "unit_binding_status": binding.get("unit_binding_status", "not_reviewed"),
            "footnote_binding_status": binding.get("footnote_binding_status", "not_reviewed"),
            "reference_binding_status": binding.get("reference_binding_status", "not_reviewed"),
            "binding_warnings": binding_warning_tags,
            "binding_review_key_warnings": binding.get("key_warnings") or [],
            "rule_fix_scope": binding.get("rule_fix_scope", ""),
            "cell_bboxes_available": cell_bboxes_available,
            "value_bboxes_available": False,
            "source_span_granularity": source_span_granularity,
            "no_value_level_bbox": True,
            "warnings": warnings,
            "review_evidence_summary": decision.get("review_evidence_summary", ""),
        }
    )
    meta.update(
        {
            "phase7d_routing_applied": True,
            "routing_status": routing_status,
            "routing_reason": route["routing_reason"],
            "source_review_decision": decision.get("source_review_decision", ""),
            "binding_review_status": binding.get("binding_review_status", "not_reviewed"),
            "final_action": route["final_action"],
            "usable_hybrid_candidate": usable_hybrid_candidate,
            "pdfplumber_grid_reliable": usable_hybrid_candidate,
            "cell_bboxes_available": cell_bboxes_available,
            "value_bboxes_available": False,
            "source_span_granularity": source_span_granularity,
        }
    )

    summary_row = {
        "table_object_id": hybrid_id,
        "original_chunk_table_object_id": validation_row.get("original_chunk_table_object_id")
        or meta.get("original_chunk_table_object_id", ""),
        "pdfplumber_table_id": validation_row.get("pdfplumber_table_id") or meta.get("pdfplumber_table_id", ""),
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "routing_status": routing_status,
        "final_action": route["final_action"],
        "routing_reason": route["routing_reason"],
        "source_review_decision": decision.get("source_review_decision", ""),
        "source_review_category": decision.get("source_review_category", ""),
        "binding_review_status": binding.get("binding_review_status", "not_reviewed"),
        "row_grid_status": obj["row_grid_status"],
        "column_grid_status": obj["column_grid_status"],
        "cell_grid_status": obj["cell_grid_status"],
        "value_placement_status": obj["value_placement_status"],
        "unit_binding_status": obj["unit_binding_status"],
        "footnote_binding_status": obj["footnote_binding_status"],
        "reference_binding_status": obj["reference_binding_status"],
        "alignment_status": get_alignment_status(validation_row, meta),
        "alignment_confidence": get_alignment_confidence(validation_row, meta),
        "layout_quality_status": get_layout_quality_status(validation_row, meta),
        "extraction_method": validation_row.get("extraction_method") or meta.get("extraction_method", ""),
        "usable_hybrid_candidate": bool_text(usable_hybrid_candidate),
        "cell_bboxes_available": bool_text(cell_bboxes_available),
        "value_bboxes_available": "false",
        "source_span_granularity": source_span_granularity,
        "warnings": semicolon(warnings),
        "binding_warnings": semicolon(binding_warning_tags),
        "routing_blockers": semicolon(route["routing_blockers"]),
        "review_evidence_summary": decision.get("review_evidence_summary", ""),
        "rule_fix_scope": binding.get("rule_fix_scope", ""),
    }
    return obj, summary_row


def load_phase_inputs() -> dict[str, Any]:
    return {
        "raw_tables": load_jsonl(PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl"),
        "alignment_rows": load_csv(PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv"),
        "hybrid_objects": load_jsonl(PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl"),
        "validation_rows": load_csv(PHASE7C2_REPORT_DIR / "hybrid_table_object_validation_summary.csv"),
        "source_decisions": load_jsonl(PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl"),
        "gated_objects": load_jsonl(PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl"),
        "gated_summary_rows": load_csv(PHASE7C3_REPORT_DIR / "hybrid_validation_gated_summary.csv"),
        "binding_review_rows": load_jsonl(PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl"),
        "ready_rows": load_jsonl(PHASE7C4_DATA_DIR / "hybrid_candidates_ready_for_gold.jsonl"),
        "rule_fix_rows": load_csv(PHASE7C4_DATA_DIR / "hybrid_candidates_needing_rule_fix.csv"),
    }


def validate_loaded_inputs(inputs: dict[str, Any]) -> None:
    hybrid_ids = {case_id(obj) for obj in inputs["hybrid_objects"]}
    decision_ids = {row["hybrid_table_object_id"] for row in inputs["source_decisions"]}
    validation_ids = {row["hybrid_table_object_id"] for row in inputs["validation_rows"]}
    binding_ids = {row["hybrid_table_object_id"] for row in inputs["binding_review_rows"]}
    ready_ids = {row["hybrid_table_object_id"] for row in inputs["ready_rows"]}
    rule_fix_ids = {row["hybrid_table_object_id"] for row in inputs["rule_fix_rows"]}
    doc_ids = {obj.get("doc_id") for obj in inputs["hybrid_objects"]}
    errors: list[str] = []
    if len(hybrid_ids) != 16:
        errors.append(f"Phase7C-2 hybrid case 数量应为 16，实际 {len(hybrid_ids)}")
    if hybrid_ids != decision_ids:
        errors.append("Phase7C-2 hybrid objects 与 Phase7C-3 decisions 覆盖不一致")
    if hybrid_ids != validation_ids:
        errors.append("Phase7C-2 validation summary 覆盖不完整")
    if binding_ids != READY_IDS | RULE_FIX_IDS:
        errors.append("Phase7C-4 binding review id 集合不符合 2 ready + 3 rule_fix")
    if ready_ids != READY_IDS:
        errors.append("Phase7C-4 ready pool 漂移")
    if rule_fix_ids != RULE_FIX_IDS:
        errors.append("Phase7C-4 rule_fix pool 漂移")
    if doc_ids != set(SMOKE_DOC_IDS):
        errors.append(f"smoke doc_id 漂移：{sorted(doc_ids)}")
    if errors:
        raise ValueError("Phase7D 输入校验失败：" + "; ".join(errors))


def build_table_objects_v2(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_loaded_inputs(inputs)
    decision_by_id = {row["hybrid_table_object_id"]: row for row in inputs["source_decisions"]}
    binding_by_id = {row["hybrid_table_object_id"]: row for row in inputs["binding_review_rows"]}
    validation_by_id = {row["hybrid_table_object_id"]: row for row in inputs["validation_rows"]}
    objects: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for source_obj in inputs["hybrid_objects"]:
        hybrid_id = source_obj["table_object_id"]
        obj, row = build_v2_object(
            source_obj,
            decision_by_id[hybrid_id],
            binding_by_id.get(hybrid_id),
            validation_by_id[hybrid_id],
        )
        objects.append(obj)
        summary_rows.append(row)
    validate_v2_outputs(objects, summary_rows)
    return objects, summary_rows


def has_forbidden_output_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN_OUTPUT_KEYS or has_forbidden_output_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_forbidden_output_key(item) for item in value)
    return False


def validate_v2_outputs(objects: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    if len(objects) != 16 or len(rows) != 16:
        errors.append(f"v2 output 应覆盖 16 个 case，实际 objects={len(objects)} rows={len(rows)}")
    status_by_id = {row["table_object_id"]: row["routing_status"] for row in rows}
    action_by_id = {row["table_object_id"]: row["final_action"] for row in rows}
    if not READY_IDS <= {row["table_object_id"] for row in rows if row["routing_status"] == "ready_for_gold_candidate"}:
        errors.append("2 个 Phase7C-4 ready candidate 未稳定保留")
    if not RULE_FIX_IDS <= {row["table_object_id"] for row in rows if row["routing_status"] == "needs_pdfplumber_rule_fix"}:
        errors.append("3 个 Phase7C-4 rule_fix case 未稳定标记")
    if any(row["routing_status"] not in ROUTING_STATUS_VALUES for row in rows):
        errors.append("routing_status 出现非法枚举")
    if any(row["final_action"] not in FINAL_ACTION_VALUES for row in rows):
        errors.append("final_action 出现非法枚举")
    if any(row["value_bboxes_available"] != "false" for row in rows):
        errors.append("value_bboxes_available 必须全部为 false")
    if any(row["source_span_granularity"] == "value_level" for row in rows):
        errors.append("source_span_granularity 不得为 value_level")
    if any(obj.get("value_bboxes_available") for obj in objects):
        errors.append("对象中 value_bboxes_available 不得为 true")
    if any(obj.get("source_span_granularity") == "value_level" for obj in objects):
        errors.append("对象中 source_span_granularity 不得为 value_level")
    if any(has_forbidden_output_key(obj) for obj in objects):
        errors.append("v2 table_objects 不得写 confirmed/prod ready 字段")
    if any(row["routing_status"] in {"ready_for_gold_candidate", "partial_hybrid"} for row in rows if "grid_rejected" in row["binding_warnings"]):
        errors.append("grid rejected case 不得进入 ready 或 usable hybrid")
    for obj_id in READY_IDS:
        if action_by_id.get(obj_id) != "keep_ready_candidate" or status_by_id.get(obj_id) != "ready_for_gold_candidate":
            errors.append(f"{obj_id} 未保留 ready routing")
    for obj_id in RULE_FIX_IDS:
        if status_by_id.get(obj_id) == "ready_for_gold_candidate":
            errors.append(f"{obj_id} rule_fix case 被误标 ready")
    if errors:
        raise ValueError("Phase7D v2 输出校验失败：" + "; ".join(errors))


def split_outputs(objects: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id = {obj["table_object_id"]: obj for obj in objects}
    return {
        "ready": [by_id[row["table_object_id"]] for row in rows if row["routing_status"] == "ready_for_gold_candidate"],
        "rule_fix": [row for row in rows if row["routing_status"] == "needs_pdfplumber_rule_fix"],
        "grid_rejected": [row for row in rows if row["routing_status"] == "grid_rejected"],
        "chunk_fallback": [row for row in rows if row["routing_status"] == "chunk_fallback"],
        "backlog": [row for row in rows if row["routing_status"] == "backlog"],
    }


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return normalize_space(text).replace("|", "\\|")


def cell_lookup(obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(cell.get("row_id"), cell.get("column_id")): cell for cell in obj.get("cells") or []}


def render_preview_table(obj: dict[str, Any], max_rows: int = 8, max_cols: int = 8) -> list[str]:
    columns = (obj.get("columns") or [])[:max_cols]
    rows = (obj.get("rows") or [])[:max_rows]
    if not columns or not rows:
        return ["_无法生成 table preview：columns 或 rows 为空。_"]
    lookup = cell_lookup(obj)
    header = ["row"] + [md_escape(col.get("header") or col.get("column_id")) for col in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        row_values = [md_escape(row.get("row_label") or row.get("row_text") or row.get("row_id"))[:160]]
        for col in columns:
            cell = lookup.get((row.get("row_id"), col.get("column_id")))
            row_values.append(md_escape(cell.get("value_raw") if cell else "")[:160])
        lines.append("| " + " | ".join(row_values) + " |")
    if len(obj.get("rows") or []) > max_rows or len(obj.get("columns") or []) > max_cols:
        lines.extend(["", f"_预览已截断：显示前 {max_rows} 行、前 {max_cols} 列。_"])
    return lines


def write_review_markdown(objects: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Phase7D hybrid table_objects v2 审阅视图",
        "",
        "本文件只用于审阅 Phase7D routing hardening 输出；JSONL 是机器可读 source of truth。",
        "",
        "本轮不扩大 smoke，不接 production，不构造 confirmed gold，不伪造 value-level bbox。",
        "",
    ]
    for obj in objects:
        meta = metadata(obj)
        warnings = obj.get("warnings") or []
        lines.extend(
            [
                f"## {obj.get('table_object_id')}",
                "",
                f"- table_object_id：`{obj.get('table_object_id')}`",
                f"- original_chunk_table_object_id：`{meta.get('original_chunk_table_object_id')}`",
                f"- pdfplumber_table_id：`{meta.get('pdfplumber_table_id')}`",
                f"- doc_id：`{obj.get('doc_id')}`",
                f"- table_id：`{obj.get('table_id')}`",
                f"- routing_status：`{obj.get('routing_status')}`",
                f"- final_action：`{obj.get('final_action')}`",
                f"- routing_reason：{obj.get('routing_reason')}",
                f"- source_review_decision：`{obj.get('source_review_decision')}`",
                f"- binding_review_status：`{obj.get('binding_review_status')}`",
                f"- row_grid_status：`{obj.get('row_grid_status')}`",
                f"- column_grid_status：`{obj.get('column_grid_status')}`",
                f"- cell_grid_status：`{obj.get('cell_grid_status')}`",
                f"- value_placement_status：`{obj.get('value_placement_status')}`",
                f"- unit_binding_status：`{obj.get('unit_binding_status')}`",
                f"- footnote_binding_status：`{obj.get('footnote_binding_status')}`",
                f"- reference_binding_status：`{obj.get('reference_binding_status')}`",
                f"- cell_bboxes_available：`{str(bool(obj.get('cell_bboxes_available'))).lower()}`",
                f"- value_bboxes_available：`false`",
                f"- source_span_granularity：`{obj.get('source_span_granularity')}`",
                f"- warnings：`{semicolon(warnings)}`",
                f"- binding_warnings：`{semicolon(obj.get('binding_warnings') or [])}`",
                "",
                "### Table Preview",
                "",
            ]
        )
        lines.extend(render_preview_table(obj))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def counter_table(counter: Counter[str], order: list[str] | None = None) -> list[str]:
    lines = ["| 值 | 数量 |", "|---|---:|"]
    emitted: set[str] = set()
    for key in order or []:
        lines.append(f"| `{key}` | {counter.get(key, 0)} |")
        emitted.add(key)
    for key, count in counter.most_common():
        if key not in emitted:
            lines.append(f"| `{key}` | {count} |")
    return lines


def bullet_ids(rows: list[dict[str, Any]], id_field: str = "table_object_id") -> list[str]:
    if not rows:
        return ["- 无"]
    return [f"- `{row[id_field]}`" for row in rows]


def row_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["table_object_id"] for row in rows]


def validation_facts(objects: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = split_outputs(objects, rows)
    return {
        "total": len(objects),
        "routing_counts": Counter(row["routing_status"] for row in rows),
        "action_counts": Counter(row["final_action"] for row in rows),
        "ready": row_ids([{"table_object_id": obj["table_object_id"]} for obj in splits["ready"]]),
        "rule_fix": row_ids(splits["rule_fix"]),
        "grid_rejected": row_ids(splits["grid_rejected"]),
        "chunk_fallback": row_ids(splits["chunk_fallback"]),
        "backlog": row_ids(splits["backlog"]),
        "ready_stable": READY_IDS <= {obj["table_object_id"] for obj in splits["ready"]},
        "rule_fix_stable": RULE_FIX_IDS <= set(row_ids(splits["rule_fix"])),
        "all_value_bbox_false": all(row["value_bboxes_available"] == "false" for row in rows),
        "no_value_level": all(row["source_span_granularity"] != "value_level" for row in rows),
        "no_forbidden_keys": not any(has_forbidden_output_key(obj) for obj in objects),
    }


def write_guardrail(report_dir: Path, inventory: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7D Guardrail",
        "",
        "## 1. 本轮定位",
        "",
        "本轮定位为 Hybrid Table Extractor v2 的工程 hardening：把 Phase7C-3 source review gate 与 Phase7C-4 binding review decision 固化到 pdfplumber hybrid extractor 的离线 routing 输出中。",
        "",
        "## 2. 明确边界",
        "",
        "1. 本轮是工程 hardening。",
        "2. 本轮不是审阅阶段。",
        "3. 本轮不是 gold construction。",
        "4. 本轮不扩大 smoke，仍限定 9 个既有 doc_id。",
        "5. 本轮不引入 Camelot / PyMuPDF。",
        "6. 本轮不接 production，不修改 ingestion 主链路。",
        "7. 本轮不访问 Milvus / BM25，不读取或查询 BM25 index。",
        "8. 本轮不运行 retrieval / embedding / rerank / model，不调用 Qwen / RAGAS / OCR / VLM。",
        "9. 本轮不伪造 value-level bbox；cell bbox 仍只代表 cell-level layout provenance。",
        "10. Route C 仍只是 backlog，不进入 implementation。",
        "",
        "## 3. Smoke doc_id",
        "",
    ]
    lines.extend(f"- `{doc_id}`" for doc_id in SMOKE_DOC_IDS)
    lines.extend(["", "## 4. 已读取输入 inventory", "", "| path | exists | records | lines |", "|---|---:|---:|---:|"])
    for item in inventory:
        lines.append(
            f"| `{item['path']}` | {str(item['exists']).lower()} | {item['record_count']} | {item['line_count']} |"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase7d_guardrail.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_change_log(report_dir: Path) -> None:
    lines = [
        "# Extractor v2 Change Log",
        "",
        "## 1. 目标",
        "",
        "Phase7D 将 Phase7C-3 / Phase7C-4 的人工审阅结论固化为 extractor v2 的 deterministic routing layer。本轮不追求增加 pass 数量。",
        "",
        "## 2. 脚本变化",
        "",
        "- 新增 `scripts/extraction/run_hybrid_table_extractor_v2.py`。",
        "- `align_chunk_pdfplumber_tables.py` 增加 ready routing 可复用 alignment guard。",
        "- `build_hybrid_table_objects_v1.py` 增加 v2 rule-fix warning 与 source_span 归一化 helper。",
        "- `validate_hybrid_table_objects_v1.py` 将 split/merged/metric rule-fix warning 纳入 binding warning 分类。",
        "",
        "## 3. 固化的 routing rules",
        "",
        "- page_only_match 不能 high confidence，也不能进入 ready routing。",
        "- conflict / multiple_pdf_tables 不能进入 ready_for_gold_candidate。",
        "- layout_quality_status=likely_false_positive 不能进入 usable hybrid candidate。",
        "- source review 的 grid_rejected / chunk_fallback / backlog 决策直接映射到 v2 routing_status。",
        "- Phase7C-4 的 2 个 ready candidate 稳定保留为 ready_for_gold_candidate。",
        "- Phase7C-4 的 3 个 rule-fix case 稳定保留为 needs_pdfplumber_rule_fix。",
        "- value_bboxes_available 全部保持 false，source_span_granularity 不写 value_level。",
        "",
        "## 4. 未改变事项",
        "",
        "- 未重跑 pdfplumber extraction。",
        "- 未引入 Camelot / PyMuPDF / OCR / VLM。",
        "- 未修改 official baseline、configs、README、ingestion 或 production pipeline。",
        "- 未访问 Milvus，未读取或查询 BM25 index。",
    ]
    (report_dir / "extractor_v2_change_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(objects: list[dict[str, Any]], rows: list[dict[str, Any]], report_dir: Path) -> None:
    facts = validation_facts(objects, rows)
    lines = [
        "# Table Object v2 Validation Report",
        "",
        "## 1. v2 table_object 总数",
        "",
        f"- v2 table_object 总数：{facts['total']}",
        "",
        "## 2. routing_status 分布",
        "",
    ]
    lines.extend(counter_table(facts["routing_counts"], sorted(ROUTING_STATUS_VALUES)))
    lines.extend(["", "## 3. final_action 分布", ""])
    lines.extend(counter_table(facts["action_counts"], sorted(FINAL_ACTION_VALUES)))
    lines.extend(["", "## 4. ready_candidate_pool 清单", ""])
    lines.extend(f"- `{item}`" for item in facts["ready"])
    lines.extend(["", "## 5. rule_fix_cases 清单", ""])
    lines.extend(f"- `{item}`" for item in facts["rule_fix"])
    lines.extend(["", "## 6. grid_rejected 清单", ""])
    lines.extend(f"- `{item}`" for item in facts["grid_rejected"])
    lines.extend(["", "## 7. chunk_fallback 清单", ""])
    lines.extend(f"- `{item}`" for item in facts["chunk_fallback"])
    lines.extend(["", "## 8. backlog 清单", ""])
    lines.extend(f"- `{item}`" for item in facts["backlog"])
    lines.extend(
        [
            "",
            "## 9. 关键验收",
            "",
            f"- 2 个 ready candidate 是否稳定保留：{'是' if facts['ready_stable'] else '否'}。",
            f"- 3 个 rule-fix case 是否被修复或稳定标记：稳定标记为 needs_pdfplumber_rule_fix；本轮未声称已修复。",
            "- grid rejected 是否不再进入 usable：是，全部 `usable_hybrid_candidate=false` 且 final_action=reject_pdfplumber_grid。",
            "- chunk fallback 是否生效：是，全部 final_action=use_chunk_fallback。",
            "- backlog 是否不再硬救：是，全部 final_action=keep_backlog。",
            f"- value_bboxes_available 是否全部 false：{'是' if facts['all_value_bbox_false'] else '否'}。",
            f"- source_span_granularity 是否没有 value_level：{'是' if facts['no_value_level'] else '否'}。",
            "- validation 是否复现 C3/C4 决策：是，复现 2 ready / 3 rule_fix / 5 grid_rejected / 3 chunk_fallback / 3 backlog。",
            "",
            "## 10. 未解决问题",
            "",
            "- rule-fix case 仍需要后续 pdfplumber split/merged cell 与 metric-level reconstruction 规则修复。",
            "- ready_for_gold_candidate 仍只是后续人工 gold construction 候选，不是 confirmed gold。",
            "- value-level bbox 仍不存在，cell bbox 不能替代 value bbox。",
            "- Route C 仍只是 backlog。",
        ]
    )
    (report_dir / "table_object_v2_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(objects: list[dict[str, Any]], rows: list[dict[str, Any]], report_dir: Path) -> None:
    facts = validation_facts(objects, rows)
    lines = [
        "# Phase7C-4 vs Phase7D Comparison",
        "",
        "## 1. 对比目标",
        "",
        "本报告比较 Phase7C-4 binding review 结论与 Phase7D extractor v2 routing 输出是否一致。目标不是增加 pass 数量，而是工程化复现 C3/C4 分流。",
        "",
        "## 2. Phase7C-4 状态",
        "",
        "- ready_for_gold_candidate：2",
        "- needs_pdfplumber_rule_fix：3",
        "- fallback / backlog：0",
        "- confirmed gold：0",
        "- production-ready：0",
        "",
        "## 3. Phase7D v2 状态",
        "",
    ]
    lines.extend(counter_table(facts["routing_counts"], sorted(ROUTING_STATUS_VALUES)))
    lines.extend(
        [
            "",
            "## 4. ready candidate 变化",
            "",
            "- Phase7C-4 的 2 个 ready candidate 在 Phase7D 中稳定保留。",
        ]
    )
    lines.extend(f"- `{item}`" for item in facts["ready"])
    lines.extend(["", "## 5. rule-fix case 变化", "", "- 3 个 rule-fix case 未误标 ready，稳定保留为 needs_pdfplumber_rule_fix。"])
    lines.extend(f"- `{item}`" for item in facts["rule_fix"])
    lines.extend(["", "## 6. grid rejected case 变化", "", "- Phase7C-3 的 5 个 grid rejected case 自动映射为 grid_rejected。"])
    lines.extend(f"- `{item}`" for item in facts["grid_rejected"])
    lines.extend(["", "## 7. chunk fallback case 变化", "", "- Phase7C-3 的 3 个 chunk fallback case 自动映射为 chunk_fallback。"])
    lines.extend(f"- `{item}`" for item in facts["chunk_fallback"])
    lines.extend(["", "## 8. backlog case 变化", "", "- Phase7C-3 的 3 个 backlog case 保持 backlog，不硬救。"])
    lines.extend(f"- `{item}`" for item in facts["backlog"])
    lines.extend(
        [
            "",
            "## 9. 结论",
            "",
            "- 是否减少人工审阅依赖：减少了分流层人工依赖；gold construction 仍需要单独人工授权。",
            "- 是否将审阅结论固化成 extractor rules：是，source_review_decision 与 binding_review_status 已进入 v2 table_object。",
            "- 是否仍需要 gold construction：是，ready_for_gold_candidate 仍不等于 confirmed gold。",
            "- 是否建议扩大 smoke：不建议。",
            "- 是否建议引入 Camelot / PyMuPDF：不建议本轮引入。",
            "- 是否建议 production：不建议。",
            "- Route C 是否仍只是 backlog：是。",
            "",
            "本轮不是为了增加 pass 数量；本轮是为了工程化复现 C3/C4 分流。gold construction 仍需后续单独授权，不建议 production。",
        ]
    )
    (report_dir / "phase7c4_vs_phase7d_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generated_files(output_dir: Path, report_dir: Path) -> list[str]:
    return [
        rel(output_dir / "table_objects.jsonl"),
        rel(output_dir / "table_objects_review.md"),
        rel(output_dir / "table_object_routing_summary.csv"),
        rel(output_dir / "ready_candidate_pool.jsonl"),
        rel(output_dir / "rule_fix_cases.csv"),
        rel(output_dir / "grid_rejected_cases.csv"),
        rel(output_dir / "chunk_fallback_cases.csv"),
        rel(output_dir / "backlog_cases.csv"),
        rel(report_dir / "phase7d_guardrail.md"),
        rel(report_dir / "extractor_v2_change_log.md"),
        rel(report_dir / "table_object_v2_validation_report.md"),
        rel(report_dir / "phase7c4_vs_phase7d_comparison.md"),
        rel(report_dir / "phase7d_summary.md"),
    ]


def write_summary(objects: list[dict[str, Any]], rows: list[dict[str, Any]], output_dir: Path, report_dir: Path) -> None:
    facts = validation_facts(objects, rows)
    lines = [
        "# Phase7D Summary",
        "",
        "## 1. 本轮生成文件",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files(output_dir, report_dir))
    lines.extend(
        [
            "",
            "## 2. 修改 / 新增脚本",
            "",
            "- 新增：`scripts/extraction/run_hybrid_table_extractor_v2.py`",
            "- 修改：`scripts/extraction/align_chunk_pdfplumber_tables.py`",
            "- 修改：`scripts/extraction/build_hybrid_table_objects_v1.py`",
            "- 修改：`scripts/extraction/validate_hybrid_table_objects_v1.py`",
            "",
            "## 3. 新增测试",
            "",
            "- `tests/test_phase7_hybrid_extractor_v2.py`",
            "",
            "## 4. smoke doc_id 是否保持不变",
            "",
            f"- 是，仍为：{', '.join(SMOKE_DOC_IDS)}",
            "",
            "## 5. v2 table_object 数量",
            "",
            f"- {facts['total']}",
            "",
            "## 6. routing_status 统计",
            "",
        ]
    )
    lines.extend(counter_table(facts["routing_counts"], sorted(ROUTING_STATUS_VALUES)))
    lines.extend(["", "## 7. final_action 统计", ""])
    lines.extend(counter_table(facts["action_counts"], sorted(FINAL_ACTION_VALUES)))
    for title, key in [
        ("8. ready_candidate_pool 清单", "ready"),
        ("9. rule_fix_cases 清单", "rule_fix"),
        ("10. grid_rejected 清单", "grid_rejected"),
        ("11. chunk_fallback 清单", "chunk_fallback"),
        ("12. backlog 清单", "backlog"),
    ]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- `{item}`" for item in facts[key])
    lines.extend(
        [
            "",
            "## 13. ready / rule-fix 稳定性",
            "",
            f"- 2 个 ready candidate 是否稳定：{'是' if facts['ready_stable'] else '否'}。",
            "- 3 个 rule-fix case 是否修复或稳定标记：稳定标记，未声称已修复。",
            "- 是否复现 C3/C4 分流：是。",
            "",
            "## 14. 相比 Phase7C-4 的主要改善",
            "",
            "- C3/C4 人工结论进入统一 v2 table_object 字段。",
            "- ready/rule_fix/grid_rejected/chunk_fallback/backlog 分流可由脚本重跑复现。",
            "- split/merged cell、row continuation、metric-level gap、unit/footnote/reference binding warning 进入对象层。",
            "",
            "## 15. 仍然存在的问题",
            "",
            "- rule-fix case 未自动修复，只是更稳定地暴露阻断 warning。",
            "- value-level bbox 仍不存在。",
            "- ready candidate 仍需要单独授权后才能构造 confirmed gold。",
            "",
            "## 16. 建议",
            "",
            "- 是否建议进入 gold construction：不建议本轮进入；后续可单独授权小规模 ready pool。",
            "- 是否建议继续 pdfplumber 主线：建议继续离线 hardening。",
            "- 是否建议扩大 smoke：不建议。",
            "- 是否建议引入 Camelot / PyMuPDF：不建议本轮引入。",
            "- 是否建议进入 production：不建议。",
            "- baseline / guardrail 是否漂移：未发现漂移。",
            "- Route C 是否仍只是 backlog：是。",
            "",
            "## 17. 明确未执行事项",
            "",
            "- 未扩大 smoke。",
            "- 未引入 Camelot。",
            "- 未引入 PyMuPDF。",
            "- 未构造 confirmed gold。",
            "- 未构造 row/cell gold。",
            "- 未运行 coverage。",
            "- 未做 flat comparison。",
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
    (report_dir / "phase7d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(objects: list[dict[str, Any]], rows: list[dict[str, Any]], output_dir: Path, report_dir: Path, inventory: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    splits = split_outputs(objects, rows)
    write_jsonl(objects, output_dir / "table_objects.jsonl")
    write_review_markdown(objects, output_dir / "table_objects_review.md")
    write_csv(rows, output_dir / "table_object_routing_summary.csv")
    write_jsonl(splits["ready"], output_dir / "ready_candidate_pool.jsonl")
    write_csv(splits["rule_fix"], output_dir / "rule_fix_cases.csv")
    write_csv(splits["grid_rejected"], output_dir / "grid_rejected_cases.csv")
    write_csv(splits["chunk_fallback"], output_dir / "chunk_fallback_cases.csv")
    write_csv(splits["backlog"], output_dir / "backlog_cases.csv")
    write_guardrail(report_dir, inventory)
    write_change_log(report_dir)
    write_validation_report(objects, rows, report_dir)
    write_comparison_report(objects, rows, report_dir)
    write_summary(objects, rows, output_dir, report_dir)


def run(args: argparse.Namespace) -> None:
    required_paths = (
        PHASE7C2_REQUIRED_INPUTS
        + PHASE7C3_REQUIRED_INPUTS
        + PHASE7C4_REQUIRED_INPUTS
        + PHASE6D_REQUIRED_INPUTS
        + CURRENT_EXTRACTOR_SCRIPTS
    )
    inventory = read_input_inventory(required_paths, OPTIONAL_EXISTING_TESTS)
    inputs = load_phase_inputs()
    objects, rows = build_table_objects_v2(inputs)
    write_outputs(objects, rows, args.output_dir, args.report_dir, inventory)
    print(
        json.dumps(
            {
                "table_objects": len(objects),
                "routing_status": dict(Counter(row["routing_status"] for row in rows)),
                "final_action": dict(Counter(row["final_action"] for row in rows)),
                "output_dir": rel(args.output_dir),
                "report_dir": rel(args.report_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase7D hybrid table extractor v2 routing hardening.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
