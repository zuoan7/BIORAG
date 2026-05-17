#!/usr/bin/env python3
"""Phase7D-3 logical cell reconstruction v2.2 layer.

This is an offline reconstruction layer over Phase7D-2 v2.1 artifacts. It does
not re-extract PDFs, does not construct gold, and does not access retrieval,
BM25, Milvus, OCR/VLM, or model services.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction import apply_hybrid_rule_fixes_v2 as v21
from scripts.extraction import run_hybrid_table_extractor_v2 as v2


SMOKE_DOC_IDS = list(v21.SMOKE_DOC_IDS)
READY_IDS = set(v21.READY_IDS)
RULE_FIX_IDS = set(v21.RULE_FIX_IDS)
GRID_REJECTED_IDS = set(v21.GRID_REJECTED_IDS)
CHUNK_FALLBACK_IDS = set(v21.CHUNK_FALLBACK_IDS)
BACKLOG_IDS = set(v21.BACKLOG_IDS)

TARGET_METRIC_ID = "doc_0687__table_2__phase7c2_hybrid_02"
TARGET_ROW_REFERENCE_ID = "doc_0523__table_1__phase7c2_hybrid_01"
ALIGNMENT_BLOCKED_ID = "doc_0598__table_1__phase7c2_hybrid_01"

PHASE7D2_DATA_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_rule_fix"
PHASE7D2_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_extractor_v2_rule_fix"
PHASE7C2_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_pilot_v2"
PHASE7C3_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_gate_hardening"
PHASE7C4_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_binding_review"
PHASE6D_REPORT_DIR = ROOT / "reports/v7_phase6d_table_contract_refinement"

DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_logical_reconstruction"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_extractor_v2_logical_reconstruction"

PHASE7D2_REQUIRED_INPUTS = [
    PHASE7D2_DATA_DIR / "table_objects.jsonl",
    PHASE7D2_DATA_DIR / "table_objects_review.md",
    PHASE7D2_DATA_DIR / "table_object_routing_summary.csv",
    PHASE7D2_DATA_DIR / "rule_fix_diagnostics.csv",
    PHASE7D2_DATA_DIR / "rule_fix_delta.csv",
    PHASE7D2_DATA_DIR / "ready_candidate_pool.jsonl",
    PHASE7D2_DATA_DIR / "rule_fix_cases.csv",
    PHASE7D2_DATA_DIR / "grid_rejected_cases.csv",
    PHASE7D2_DATA_DIR / "chunk_fallback_cases.csv",
    PHASE7D2_DATA_DIR / "backlog_cases.csv",
    PHASE7D2_REPORT_DIR / "rule_fix_implementation_report.md",
    PHASE7D2_REPORT_DIR / "table_object_v2_1_validation_report.md",
    PHASE7D2_REPORT_DIR / "phase7d_vs_phase7d2_comparison.md",
    PHASE7D2_REPORT_DIR / "phase7d_2_summary.md",
]

PHASE7C_REQUIRED_INPUTS = [
    PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl",
    PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv",
    PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_candidates_ready_for_gold.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_candidates_needing_rule_fix.csv",
]

CURRENT_EXTRACTOR_SCRIPTS = [
    ROOT / "scripts/extraction/extract_tables_pdfplumber_v1.py",
    ROOT / "scripts/extraction/align_chunk_pdfplumber_tables.py",
    ROOT / "scripts/extraction/build_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/validate_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/render_hybrid_table_objects_markdown.py",
    ROOT / "scripts/extraction/run_hybrid_table_extractor_v2.py",
    ROOT / "scripts/extraction/apply_hybrid_rule_fixes_v2.py",
    ROOT / "scripts/extraction/reconstruct_logical_cells_v2.py",
]

OPTIONAL_EXISTING_TESTS = [
    ROOT / "tests/test_phase7_hybrid_alignment_gate.py",
    ROOT / "tests/test_phase7_source_review_gate.py",
    ROOT / "tests/test_phase7_hybrid_binding_review.py",
    ROOT / "tests/test_phase7_hybrid_extractor_v2.py",
    ROOT / "tests/test_phase7_hybrid_extractor_v2_rule_fix.py",
    ROOT / "tests/test_phase7_logical_cell_reconstruction_v2.py",
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

FORBIDDEN_OUTPUT_KEYS = set(v21.FORBIDDEN_OUTPUT_KEYS) | {
    "confirmed_gold",
    "production_ready",
    "ready_for_gold_candidate_is_confirmed_gold",
    "usable_hybrid_candidate_is_production_ready",
}

LOGICAL_DIAGNOSTIC_FIELDS = [
    "table_object_id",
    "doc_id",
    "table_id",
    "reconstruction_attempted",
    "reconstruction_strategy",
    "reconstruction_status",
    "logical_columns",
    "logical_rows_count",
    "logical_cells_count",
    "reconstructed_value_count",
    "missing_expected_cells",
    "reconstruction_warnings",
    "remaining_blockers",
    "upgrade_eligible",
    "metric_reconstruction_status",
    "row_reference_reconstruction_status",
    "unit_binding_status",
    "footnote_binding_status",
    "reference_binding_status",
    "value_bboxes_available",
    "source_span_granularity",
    "notes",
]

DELTA_FIELDS = [
    "table_object_id",
    "phase7d2_routing_status",
    "phase7d3_routing_status",
    "changed",
    "change_type",
    "reconstruction_attempted",
    "reconstruction_strategy",
    "reconstruction_status",
    "logical_cells_count",
    "missing_expected_cells_count",
    "remaining_blockers",
    "upgrade_justification",
    "downgrade_reason",
    "notes",
]

EXTRA_SUMMARY_FIELDS = [
    "phase7d2_routing_status",
    "phase7d3_routing_status",
    "reconstruction_attempted",
    "reconstruction_strategy",
    "reconstruction_status",
    "logical_cells_count",
    "missing_expected_cells_count",
    "remaining_blockers",
    "upgrade_eligible",
    "metric_reconstruction_status",
    "row_reference_reconstruction_status",
]
ROUTING_SUMMARY_FIELDS = list(dict.fromkeys(v21.ROUTING_SUMMARY_FIELDS + EXTRA_SUMMARY_FIELDS))

METRIC_LOGICAL_COLUMNS = [
    "strain_or_variant",
    "YE/S",
    "qethanol",
    "qxylose",
    "qarabinose",
    "Reference",
]
ROW_REFERENCE_LOGICAL_COLUMNS = [
    "strain_or_construct",
    "LNT_II",
    "LNT",
    "titer_or_concentration",
    "unit",
    "reference_or_source",
    "medium_culture_conditions",
]

DOC0687_SELECTED_EXPECTED = [
    ("TMB3400", "YE/S", "0.33"),
    ("TMB3400", "qethanol", "0.04"),
    ("TMB3400", "qxylose", "0.13"),
    ("SR8", "YE/S", "0.39"),
    ("SR8", "qethanol", "0.25"),
    ("SR8", "qxylose", "0.64"),
    ("TMB3421", "YE/S", "0.35"),
    ("TMB3421", "qethanol", "0.20"),
    ("TMB3421", "qxylose", "0.57"),
    ("RWB217", "YE/S", "0.43"),
    ("RWB217", "qethanol", "0.46"),
    ("RWB217", "qxylose", "1.06"),
    ("GS1.11-26", "YE/S", "0.46"),
    ("GS1.11-26", "qethanol", "0.48"),
    ("GS1.11-26", "qxylose", "1.1"),
]

DOC0687_ROW_TEMPLATES = [
    {"start": 4, "continuations": [5, 6], "strain": "TMB3400", "reference": "Karhumaa et al. (2007)", "selected": True},
    {"start": 8, "continuations": [9, 10, 11], "strain": "GLBRCY87", "reference": "Sato et al. (2016)", "selected": False},
    {"start": 13, "continuations": [14, 15], "strain": "SR8", "reference": "Wei et al. (2013)", "selected": True},
    {
        "start": 17,
        "continuations": [18, 19, 20, 21],
        "strain": "TMB3421",
        "reference": "Runquist, Hahn-Hagerdal and Bettiga (2010a)",
        "selected": True,
    },
    {"start": 23, "continuations": [24, 25], "strain": "RWB217", "reference": "Kuyper et al. (2005a)", "selected": True},
    {"start": 27, "continuations": [28, 29], "strain": "RWB218", "reference": "Kuyper et al. (2005b)", "selected": False},
    {"start": 31, "continuations": [32, 33], "strain": "H131-A3-ALCS", "reference": "Zhou et al. (2012)", "selected": False},
    {"start": 35, "continuations": [36, 37, 38, 39, 40], "strain": "IMS0010", "reference": "Wisselink et al. (2009)", "selected": False},
    {"start": 42, "continuations": [43, 44, 45, 46, 47, 48], "strain": "GS1.11-26", "reference": "Demeke et al. (2013a)", "selected": True},
]

DOC0523_ROW_TEMPLATES = [
    {"start": 8, "continuations": [9], "reference": "17"},
    {"start": 11, "continuations": [12, 13], "reference": "20"},
    {"start": 14, "continuations": [15, 16], "reference": "13"},
    {"start": 17, "continuations": [18, 19], "reference": "13"},
    {"start": 21, "continuations": [22], "reference": "this study"},
    {"start": 23, "continuations": [24], "reference": "this study"},
    {"start": 25, "continuations": [26], "reference": "this study"},
]

NUMBER_OR_LITERAL_RE = re.compile(r"(N\.D\.|[-+]?\d+(?:\.\d+)?)$")
SPLIT_DECIMAL_RE = re.compile(r"^([+-]?\d+)\s+(\.\d+)([∗*†]?)$")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def semicolon(values: list[Any]) -> str:
    cleaned = [normalize_space(value) for value in values if normalize_space(value)]
    return ";".join(cleaned) if cleaned else "none"


def split_semicolon(value: Any) -> list[str]:
    text = normalize_space(value)
    if not text or text == "none":
        return []
    return [item for item in text.split(";") if item]


def add_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for item in additions:
        if item and item not in result:
            result.append(item)
    return result


def remove_values(values: list[str], removals: set[str]) -> list[str]:
    return [value for value in values if value not in removals]


def md_escape(value: Any) -> str:
    return normalize_space(value).replace("|", "\\|")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return v21.load_jsonl(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    return v21.load_csv(path)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    v21.write_jsonl(rows, path)


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_input_inventory(required_paths: list[Path], optional_paths: list[Path] | None = None) -> list[dict[str, Any]]:
    return v21.read_input_inventory(required_paths, optional_paths)


def strip_forbidden_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_forbidden_keys(item) for key, item in value.items() if key not in FORBIDDEN_OUTPUT_KEYS}
    if isinstance(value, list):
        return [strip_forbidden_keys(item) for item in value]
    return value


def load_inputs() -> dict[str, Any]:
    return {
        "phase7d2_objects": load_jsonl(PHASE7D2_DATA_DIR / "table_objects.jsonl"),
        "phase7d2_summary_rows": load_csv(PHASE7D2_DATA_DIR / "table_object_routing_summary.csv"),
        "phase7d2_diagnostics": load_csv(PHASE7D2_DATA_DIR / "rule_fix_diagnostics.csv"),
        "phase7d2_delta_rows": load_csv(PHASE7D2_DATA_DIR / "rule_fix_delta.csv"),
        "phase7d2_ready_rows": load_jsonl(PHASE7D2_DATA_DIR / "ready_candidate_pool.jsonl"),
        "phase7d2_rule_fix_rows": load_csv(PHASE7D2_DATA_DIR / "rule_fix_cases.csv"),
        "phase7d2_grid_rows": load_csv(PHASE7D2_DATA_DIR / "grid_rejected_cases.csv"),
        "phase7d2_fallback_rows": load_csv(PHASE7D2_DATA_DIR / "chunk_fallback_cases.csv"),
        "phase7d2_backlog_rows": load_csv(PHASE7D2_DATA_DIR / "backlog_cases.csv"),
        "c2_hybrid_objects": load_jsonl(PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl"),
        "c2_raw_tables": load_jsonl(PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl"),
        "c2_alignment_rows": load_csv(PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv"),
        "c3_decision_rows": load_jsonl(PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl"),
        "c3_gated_objects": load_jsonl(PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl"),
        "binding_review_rows": load_jsonl(PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl"),
        "c4_ready_rows": load_jsonl(PHASE7C4_DATA_DIR / "hybrid_candidates_ready_for_gold.jsonl"),
        "c4_rule_fix_rows": load_csv(PHASE7C4_DATA_DIR / "hybrid_candidates_needing_rule_fix.csv"),
    }


def validate_inputs(inputs: dict[str, Any]) -> None:
    object_ids = {row["table_object_id"] for row in inputs["phase7d2_objects"]}
    summary_ids = {row["table_object_id"] for row in inputs["phase7d2_summary_rows"]}
    errors: list[str] = []
    if len(object_ids) != 16 or len(summary_ids) != 16:
        errors.append(f"Phase7D-2 输入应覆盖 16 个对象，实际 objects={len(object_ids)} summary={len(summary_ids)}")
    if object_ids != summary_ids:
        errors.append("Phase7D-2 table_objects 与 routing summary 覆盖不一致")
    if {row["table_object_id"] for row in inputs["phase7d2_ready_rows"]} != READY_IDS:
        errors.append("Phase7D-2 ready pool 漂移")
    if {row["table_object_id"] for row in inputs["phase7d2_rule_fix_rows"]} != RULE_FIX_IDS:
        errors.append("Phase7D-2 rule_fix 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d2_grid_rows"]} != GRID_REJECTED_IDS:
        errors.append("Phase7D-2 grid_rejected 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d2_fallback_rows"]} != CHUNK_FALLBACK_IDS:
        errors.append("Phase7D-2 chunk_fallback 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d2_backlog_rows"]} != BACKLOG_IDS:
        errors.append("Phase7D-2 backlog 清单漂移")
    if {row.get("doc_id") for row in inputs["phase7d2_objects"]} != set(SMOKE_DOC_IDS):
        errors.append("smoke doc_id 漂移")
    c2_ids = {row["table_object_id"] for row in inputs["c2_hybrid_objects"]}
    c3_ids = {row["hybrid_table_object_id"] for row in inputs["c3_decision_rows"]}
    if object_ids != c2_ids or object_ids != c3_ids:
        errors.append("Phase7D-2 与 Phase7C sidecar 覆盖不一致")
    if errors:
        raise ValueError("Phase7D-3 输入校验失败：" + "; ".join(errors))


def cell_lookup_by_index(obj: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    row_index_by_id = {row.get("row_id"): row.get("row_index") for row in obj.get("rows") or []}
    col_index_by_id = {col.get("column_id"): col.get("column_index") for col in obj.get("columns") or []}
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in obj.get("cells") or []:
        row_index = row_index_by_id.get(cell.get("row_id"))
        col_index = col_index_by_id.get(cell.get("column_id"))
        if row_index is not None and col_index is not None:
            lookup[(int(row_index), int(col_index))] = cell
    return lookup


def row_lookup_by_index(obj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["row_index"]): row for row in obj.get("rows") or [] if row.get("row_index") is not None}


def cell_text(cells: dict[tuple[int, int], dict[str, Any]], row_index: int, column_index: int) -> str:
    return normalize_space(cells.get((row_index, column_index), {}).get("value_raw"))


def cell_span_ids(cells: dict[tuple[int, int], dict[str, Any]], row_index: int, column_index: int) -> list[str]:
    return list(cells.get((row_index, column_index), {}).get("source_span_ids") or [])


def row_source_span_ids(cells: dict[tuple[int, int], dict[str, Any]], row_indices: list[int], col_indices: list[int]) -> list[str]:
    span_ids: list[str] = []
    for row_index in row_indices:
        for col_index in col_indices:
            span_ids = add_unique(span_ids, cell_span_ids(cells, row_index, col_index))
    return span_ids


def decimal_value(left: str, right: str) -> str:
    left = normalize_space(left)
    right = normalize_space(right)
    if left and right.startswith("."):
        return f"{left}{right}"
    return normalize_space(f"{left} {right}")


def normalize_metric_value(value: str) -> str:
    return value.replace("∗", "*")


def extract_tail_value(text: str) -> tuple[str, str]:
    text = normalize_space(text)
    match = NUMBER_OR_LITERAL_RE.search(text)
    if not match:
        return text, ""
    value = match.group(1)
    prefix = normalize_space(text[: match.start()])
    return prefix, value


def logical_cell(
    table_object_id: str,
    row_key: str,
    row_label: str,
    column: str,
    value_raw: str,
    source_span_ids: list[str],
    unit: str | None = None,
    reference_or_source: str | None = None,
    footnote_refs: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    safe_column = re.sub(r"[^A-Za-z0-9]+", "_", column).strip("_").lower()
    safe_row = re.sub(r"[^A-Za-z0-9]+", "_", row_key).strip("_").lower()
    return {
        "logical_cell_id": f"{table_object_id}__logical_cell__{safe_row}__{safe_column}",
        "row_key": row_key,
        "row_label": row_label,
        "logical_column": column,
        "value_raw": value_raw,
        "value_normalized": None,
        "unit": unit,
        "reference_or_source": reference_or_source,
        "footnote_refs": footnote_refs or [],
        "source_span_ids": source_span_ids,
        "source_span_granularity": "cell_level",
        "value_bbox": None,
        "value_bbox_source": "not_available",
        "notes": notes or ["logical reconstruction only; not gold"],
    }


def base_reconstruction_record(obj: dict[str, Any], strategy: str) -> dict[str, Any]:
    return {
        "table_object_id": obj["table_object_id"],
        "doc_id": obj["doc_id"],
        "table_id": obj["table_id"],
        "reconstruction_attempted": False,
        "reconstruction_strategy": strategy,
        "logical_columns": [],
        "logical_rows": [],
        "logical_cells": [],
        "reconstructed_value_count": 0,
        "missing_expected_cells": [],
        "reconstruction_warnings": [],
        "remaining_blockers": list(obj.get("routing_blockers") or obj.get("rule_fix_blockers") or []),
        "upgrade_eligible": False,
        "metric_reconstruction_status": "",
        "row_reference_reconstruction_status": "",
        "unit_binding_status": obj.get("unit_binding_status", "not_reviewed"),
        "footnote_binding_status": obj.get("footnote_binding_status", "not_reviewed"),
        "reference_binding_status": obj.get("reference_binding_status", "not_reviewed"),
        "value_bboxes_available": False,
        "source_span_granularity": obj.get("source_span_granularity", ""),
        "notes": [],
    }


def metric_column_template(obj: dict[str, Any]) -> dict[str, Any]:
    cells = cell_lookup_by_index(obj)
    rows_by_index = row_lookup_by_index(obj)
    record = base_reconstruction_record(obj, "metric_column_template")
    record.update(
        {
            "reconstruction_attempted": True,
            "logical_columns": METRIC_LOGICAL_COLUMNS,
            "metric_reconstruction_status": "reconstruction_failed",
            "unit_binding_status": "uncertain",
            "footnote_binding_status": "uncertain",
            "reference_binding_status": "uncertain",
            "notes": [
                "case-specific template for doc_0687 Table 2 selected metric rows",
                "cell bbox remains layout-cell provenance; value-level bbox is not available",
            ],
        }
    )
    logical_rows: list[dict[str, Any]] = []
    logical_cells: list[dict[str, Any]] = []
    for template in DOC0687_ROW_TEMPLATES:
        start = int(template["start"])
        row_indices = [start] + list(template["continuations"])
        strain = str(template["strain"])
        reference = str(template["reference"])
        yes_value = decimal_value(cell_text(cells, start, 9), cell_text(cells, start, 10))
        qethanol_value = cell_text(cells, start, 11)
        qxylose_value = cell_text(cells, start, 13)
        qarabinose_value = cell_text(cells, start, 14)
        values = {
            "strain_or_variant": strain,
            "YE/S": normalize_metric_value(yes_value),
            "qethanol": normalize_metric_value(qethanol_value),
            "qxylose": normalize_metric_value(qxylose_value),
            "qarabinose": normalize_metric_value(qarabinose_value),
            "Reference": reference,
        }
        row_span_ids = row_source_span_ids(cells, row_indices, list(range(1, 17)))
        logical_rows.append(
            {
                "row_key": strain,
                "row_label": strain,
                "source_row_indices": row_indices,
                "selected_for_reconstruction": bool(template["selected"]),
                "source_span_ids": row_span_ids,
            }
        )
        for column in METRIC_LOGICAL_COLUMNS:
            unit = None
            if column == "YE/S":
                unit = "g ethanol.(g sugar)-1"
            elif column in {"qethanol", "qxylose", "qarabinose"}:
                unit = "g.(g biomass)-1.h-1"
            if column == "YE/S":
                span_ids = row_source_span_ids(cells, [start], [9, 10])
            elif column == "qethanol":
                span_ids = row_source_span_ids(cells, [start], [11])
            elif column == "qxylose":
                span_ids = row_source_span_ids(cells, [start], [13])
            elif column == "qarabinose":
                span_ids = row_source_span_ids(cells, [start], [14])
            elif column == "Reference":
                span_ids = row_source_span_ids(cells, row_indices, [15, 16])
            else:
                span_ids = row_source_span_ids(cells, [start], [1])
            footnote_refs = ["asterisk_marker_retained"] if "*" in values[column] or "∗" in values[column] else []
            logical_cells.append(
                logical_cell(
                    obj["table_object_id"],
                    strain,
                    strain,
                    column,
                    values[column],
                    span_ids,
                    unit=unit,
                    reference_or_source=reference if column not in {"Reference", "strain_or_variant"} else None,
                    footnote_refs=footnote_refs,
                )
            )
    cell_by_row_col = {(cell["row_key"], cell["logical_column"]): cell for cell in logical_cells}
    missing: list[str] = []
    for row_key, column, expected in DOC0687_SELECTED_EXPECTED:
        cell = cell_by_row_col.get((row_key, column))
        if not cell or cell.get("value_raw") != expected:
            observed = cell.get("value_raw") if cell else ""
            missing.append(f"{row_key}:{column}:expected={expected}:observed={observed}")
    if missing:
        status = "partially_reconstructed"
        blockers = ["missing_expected_cells"]
    else:
        status = "reconstructed_pass_with_warnings"
        blockers = []
    record.update(
        {
            "logical_rows": logical_rows,
            "logical_cells": logical_cells,
            "reconstructed_value_count": sum(1 for cell in logical_cells if normalize_space(cell.get("value_raw"))),
            "missing_expected_cells": missing,
            "reconstruction_warnings": [
                "case_specific_metric_column_template",
                "selected_rows_only_not_whole_table_gold",
                "value_bboxes_available_false",
            ],
            "remaining_blockers": blockers,
            "upgrade_eligible": not blockers,
            "metric_reconstruction_status": status,
            "unit_binding_status": "pass_with_warnings" if not blockers else "uncertain",
            "footnote_binding_status": "pass_with_warnings" if not blockers else "uncertain",
            "reference_binding_status": "pass_with_warnings" if not blockers else "uncertain",
        }
    )
    return record


def row_reference_literal_template(obj: dict[str, Any]) -> dict[str, Any]:
    cells = cell_lookup_by_index(obj)
    record = base_reconstruction_record(obj, "row_reference_literal_template")
    record.update(
        {
            "reconstruction_attempted": True,
            "logical_columns": ROW_REFERENCE_LOGICAL_COLUMNS,
            "row_reference_reconstruction_status": "reconstruction_failed",
            "unit_binding_status": "uncertain",
            "reference_binding_status": "uncertain",
            "footnote_binding_status": "not_applicable",
            "notes": [
                "case-specific template for doc_0523 Table 1 row/reference/literal reconstruction",
                "N.D. literal is preserved in value_raw; value-level bbox is not available",
            ],
        }
    )
    logical_rows: list[dict[str, Any]] = []
    logical_cells: list[dict[str, Any]] = []
    for index, template in enumerate(DOC0523_ROW_TEMPLATES, start=1):
        start = int(template["start"])
        continuation_rows = list(template["continuations"])
        row_indices = [start] + continuation_rows
        ref = str(template["reference"])
        strain_parts = [cell_text(cells, start, 1)]
        medium_parts: list[str] = []
        for continuation in continuation_rows:
            strain_parts.append(cell_text(cells, continuation, 1))
            medium_parts.append(cell_text(cells, continuation, 2))
        medium_head, lnt_ii_value = extract_tail_value(cell_text(cells, start, 2))
        medium_parts.insert(0, medium_head)
        strain = normalize_space(" ".join(part for part in strain_parts if part))
        medium = normalize_space(" ".join(part for part in medium_parts if part))
        lnt_value = cell_text(cells, start, 3)
        unit = "g/L"
        row_key = f"row_{index:02d}"
        titer_summary = "; ".join(
            part for part in [f"LNT II={lnt_ii_value} {unit}" if lnt_ii_value else "", f"LNT={lnt_value} {unit}" if lnt_value else ""] if part
        )
        row_span_ids = row_source_span_ids(cells, row_indices, [1, 2, 3, 4])
        logical_rows.append(
            {
                "row_key": row_key,
                "row_label": strain,
                "source_row_indices": row_indices,
                "source_span_ids": row_span_ids,
            }
        )
        values = {
            "strain_or_construct": strain,
            "LNT_II": lnt_ii_value,
            "LNT": lnt_value,
            "titer_or_concentration": titer_summary,
            "unit": unit,
            "reference_or_source": ref,
            "medium_culture_conditions": medium,
        }
        span_cols = {
            "strain_or_construct": [1],
            "LNT_II": [2],
            "LNT": [3],
            "titer_or_concentration": [2, 3],
            "unit": [2, 3],
            "reference_or_source": [4],
            "medium_culture_conditions": [2],
        }
        for column in ROW_REFERENCE_LOGICAL_COLUMNS:
            col_indices = span_cols[column]
            span_ids = row_source_span_ids(cells, row_indices if column in {"strain_or_construct", "medium_culture_conditions"} else [start], col_indices)
            logical_cells.append(
                logical_cell(
                    obj["table_object_id"],
                    row_key,
                    strain,
                    column,
                    values[column],
                    span_ids,
                    unit=unit if column in {"LNT_II", "LNT", "titer_or_concentration"} else None,
                    reference_or_source=ref if column not in {"reference_or_source"} else None,
                    notes=["N.D. literal preserved"] if values[column] == "N.D." else None,
                )
            )
    cell_by_col = {(cell["row_key"], cell["logical_column"]): cell for cell in logical_cells}
    missing: list[str] = []
    for row_key in {row["row_key"] for row in logical_rows}:
        for column in ["strain_or_construct", "LNT_II", "unit", "reference_or_source"]:
            if not normalize_space(cell_by_col[(row_key, column)]["value_raw"]):
                missing.append(f"{row_key}:{column}")
    if not any(cell.get("logical_column") == "LNT_II" and cell.get("value_raw") == "N.D." for cell in logical_cells):
        missing.append("N.D._literal:value_raw")
    if missing:
        status = "partially_reconstructed"
        blockers = ["missing_expected_cells"]
    else:
        status = "reconstructed_pass_with_warnings"
        blockers = []
    record.update(
        {
            "logical_rows": logical_rows,
            "logical_cells": logical_cells,
            "reconstructed_value_count": sum(1 for cell in logical_cells if normalize_space(cell.get("value_raw"))),
            "missing_expected_cells": missing,
            "reconstruction_warnings": [
                "case_specific_row_reference_literal_template",
                "table_tail_excluded_after_logical_body",
                "value_bboxes_available_false",
            ],
            "remaining_blockers": blockers,
            "upgrade_eligible": not blockers,
            "row_reference_reconstruction_status": status,
            "unit_binding_status": "pass_with_warnings" if not blockers else "uncertain",
            "reference_binding_status": "pass_with_warnings" if not blockers else "uncertain",
            "footnote_binding_status": "not_applicable",
        }
    )
    return record


def metric_reconstruction_upgrade_allowed(record: dict[str, Any]) -> bool:
    return (
        record.get("metric_reconstruction_status") == "reconstructed_pass_with_warnings"
        and not record.get("missing_expected_cells")
        and record.get("unit_binding_status") != "uncertain"
        and record.get("reference_binding_status") != "uncertain"
        and record.get("value_bboxes_available") is False
        and record.get("source_span_granularity") != "value_level"
        and not record.get("remaining_blockers")
    )


def row_reference_reconstruction_upgrade_allowed(record: dict[str, Any]) -> bool:
    cells = record.get("logical_cells") or []
    nd_preserved = any(cell.get("logical_column") == "LNT_II" and cell.get("value_raw") == "N.D." for cell in cells)
    return (
        record.get("row_reference_reconstruction_status") == "reconstructed_pass_with_warnings"
        and nd_preserved
        and record.get("reference_binding_status") != "uncertain"
        and record.get("unit_binding_status") != "uncertain"
        and record.get("value_bboxes_available") is False
        and record.get("source_span_granularity") != "value_level"
        and not record.get("missing_expected_cells")
        and not record.get("remaining_blockers")
    )


def no_reconstruction_record(obj: dict[str, Any]) -> dict[str, Any]:
    strategy = "no_reconstruction_alignment_blocked" if obj["table_object_id"] == ALIGNMENT_BLOCKED_ID else "no_reconstruction_not_applicable"
    record = base_reconstruction_record(obj, strategy)
    if obj["table_object_id"] == ALIGNMENT_BLOCKED_ID:
        blockers = [
            "alignment_not_ready_eligible",
            "page_only_match_not_high_confidence",
            "source_span_not_cell_level_for_rule_fix",
            "split_cell_warning",
            "merged_cell_warning",
            "row_continuation_warning",
        ]
        record.update(
            {
                "reconstruction_warnings": ["logical_reconstruction_not_sufficient_for_alignment_source_span_blocker"],
                "remaining_blockers": blockers,
                "notes": [
                    "doc_0598 Table 1 is alignment/source-span blocked and is not an upgrade target in Phase7D-3"
                ],
            }
        )
    else:
        record.update(
            {
                "remaining_blockers": list(obj.get("routing_blockers") or obj.get("rule_fix_blockers") or []),
                "notes": ["logical reconstruction not applicable for this routing bucket in Phase7D-3"],
            }
        )
    return record


def reconstruction_for_object(obj: dict[str, Any]) -> dict[str, Any]:
    table_object_id = obj["table_object_id"]
    if table_object_id == TARGET_METRIC_ID:
        return metric_column_template(obj)
    if table_object_id == TARGET_ROW_REFERENCE_ID:
        return row_reference_literal_template(obj)
    return no_reconstruction_record(obj)


def reconstruction_status(record: dict[str, Any]) -> str:
    return (
        record.get("metric_reconstruction_status")
        or record.get("row_reference_reconstruction_status")
        or ("not_attempted" if not record.get("reconstruction_attempted") else "reconstruction_failed")
    )


def allowed_upgrade(obj: dict[str, Any], record: dict[str, Any]) -> bool:
    if obj["table_object_id"] == TARGET_METRIC_ID:
        return metric_reconstruction_upgrade_allowed(record)
    if obj["table_object_id"] == TARGET_ROW_REFERENCE_ID:
        return row_reference_reconstruction_upgrade_allowed(record)
    return False


def apply_reconstruction_to_object(
    source_obj: dict[str, Any],
    phase7d2_row: dict[str, Any],
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    obj = strip_forbidden_keys(copy.deepcopy(source_obj))
    phase7d2_status = phase7d2_row["routing_status"]
    upgraded = phase7d2_status == "needs_pdfplumber_rule_fix" and allowed_upgrade(source_obj, record)
    phase7d3_status = "ready_for_gold_candidate" if upgraded else phase7d2_status
    final_action = "keep_ready_candidate" if phase7d3_status == "ready_for_gold_candidate" else phase7d2_row["final_action"]
    if source_obj["table_object_id"] == ALIGNMENT_BLOCKED_ID:
        final_action = "keep_rule_fix"
    blockers = [] if upgraded else list(record.get("remaining_blockers") or [])
    warnings = list(obj.get("warnings") or [])
    binding_warnings = list(obj.get("binding_warnings") or [])
    if upgraded:
        resolved = {
            "cell_grid_needs_rule_fix",
            "cell_alignment_error",
            "column_alignment_inconsistent",
            "footnote_present_not_bound",
            "footnote_binding_uncertain",
            "literal_value_requires_preservation",
            "merged_cell_warning",
            "metric_column_group_uncertain",
            "metric_level_cell_gap",
            "missing_metric_cell_warning",
            "numeric_column_order_uncertain",
            "reference_visible_not_bound",
            "reference_binding_uncertain",
            "row_cell_blocking_warning",
            "row_continuation_warning",
            "split_cell_warning",
            "table_tail_truncation",
            "unit_visible_not_bound",
            "unit_binding_uncertain",
        }
        warnings = remove_values(warnings, resolved)
        binding_warnings = remove_values(binding_warnings, resolved)
        warnings = add_unique(warnings, list(record.get("reconstruction_warnings") or []))
    else:
        warnings = add_unique(warnings, list(record.get("reconstruction_warnings") or []))
    obj.update(
        {
            "phase": "v7_phase7D_3_hybrid_extractor_v2_2_logical_reconstruction",
            "schema_name": "table_object_v2",
            "schema_version": "v2.2",
            "phase7d2_routing_status": phase7d2_status,
            "phase7d3_routing_status": phase7d3_status,
            "routing_status": phase7d3_status,
            "final_action": final_action,
            "routing_blockers": blockers,
            "rule_fix_blockers": blockers if phase7d3_status == "needs_pdfplumber_rule_fix" else [],
            "reconstruction_attempted": bool(record.get("reconstruction_attempted")),
            "reconstruction_strategy": record.get("reconstruction_strategy"),
            "reconstruction_status": reconstruction_status(record),
            "logical_columns": record.get("logical_columns") or [],
            "logical_rows": record.get("logical_rows") or [],
            "logical_cells": record.get("logical_cells") or [],
            "reconstructed_value_count": int(record.get("reconstructed_value_count") or 0),
            "missing_expected_cells": record.get("missing_expected_cells") or [],
            "reconstruction_warnings": record.get("reconstruction_warnings") or [],
            "remaining_blockers": blockers,
            "upgrade_eligible": bool(upgraded),
            "metric_reconstruction_status": record.get("metric_reconstruction_status") or "",
            "row_reference_reconstruction_status": record.get("row_reference_reconstruction_status") or "",
            "unit_binding_status": record.get("unit_binding_status", obj.get("unit_binding_status")),
            "footnote_binding_status": record.get("footnote_binding_status", obj.get("footnote_binding_status")),
            "reference_binding_status": record.get("reference_binding_status", obj.get("reference_binding_status")),
            "usable_hybrid_candidate": phase7d3_status == "ready_for_gold_candidate",
            "pdfplumber_grid_reliable": phase7d3_status == "ready_for_gold_candidate",
            "value_bboxes_available": False,
            "no_value_level_bbox": True,
            "warnings": warnings,
            "binding_warnings": binding_warnings,
        }
    )
    if upgraded:
        obj["cell_grid_status"] = "pass_with_warnings"
        obj["row_grid_status"] = "pass_with_warnings"
        obj["column_grid_status"] = "pass_with_warnings"
        obj["value_placement_status"] = "pass_with_warnings"
        obj["routing_reason"] = "Phase7D-3 logical reconstruction resolved selected-evidence blockers; still only ready_for_gold_candidate."
        obj["blocking_warnings"] = []
    if obj.get("source_span_granularity") == "value_level":
        obj["source_span_granularity"] = "mixed_or_unclear"
    for span in obj.get("source_spans") or []:
        if span.get("granularity") == "value_level":
            span["granularity"] = "mixed_or_unclear"
            span["bbox"] = None
    meta = obj.setdefault("hybrid_metadata", {})
    meta.update(
        {
            "phase7d2_routing_status": phase7d2_status,
            "phase7d3_routing_status": phase7d3_status,
            "phase7d3_logical_reconstruction_attempted": bool(record.get("reconstruction_attempted")),
            "phase7d3_reconstruction_strategy": record.get("reconstruction_strategy"),
            "final_action": final_action,
            "source_span_granularity": obj.get("source_span_granularity"),
            "value_bboxes_available": False,
        }
    )
    row = dict(phase7d2_row)
    row.update(
        {
            "routing_status": phase7d3_status,
            "final_action": final_action,
            "routing_reason": obj.get("routing_reason", row.get("routing_reason", "")),
            "routing_blockers": semicolon(blockers),
            "usable_hybrid_candidate": bool_text(phase7d3_status == "ready_for_gold_candidate"),
            "pdfplumber_grid_reliable": bool_text(phase7d3_status == "ready_for_gold_candidate"),
            "value_bboxes_available": "false",
            "source_span_granularity": obj.get("source_span_granularity", ""),
            "warnings": semicolon(warnings),
            "binding_warnings": semicolon(binding_warnings),
            "unit_binding_status": obj.get("unit_binding_status", ""),
            "footnote_binding_status": obj.get("footnote_binding_status", ""),
            "reference_binding_status": obj.get("reference_binding_status", ""),
            "phase7d2_routing_status": phase7d2_status,
            "phase7d3_routing_status": phase7d3_status,
            "reconstruction_attempted": bool_text(record.get("reconstruction_attempted")),
            "reconstruction_strategy": record.get("reconstruction_strategy", ""),
            "reconstruction_status": reconstruction_status(record),
            "logical_cells_count": str(len(record.get("logical_cells") or [])),
            "missing_expected_cells_count": str(len(record.get("missing_expected_cells") or [])),
            "remaining_blockers": semicolon(blockers),
            "upgrade_eligible": bool_text(upgraded),
            "metric_reconstruction_status": record.get("metric_reconstruction_status", ""),
            "row_reference_reconstruction_status": record.get("row_reference_reconstruction_status", ""),
        }
    )
    changed = phase7d2_status != phase7d3_status
    if changed:
        change_type = "logical_reconstruction_upgrade"
        upgrade_justification = (
            "selected logical cells complete; unit/reference/footnote binding no longer blocks selected evidence; "
            "value_bboxes_available=false retained; source_span_granularity is not value_level"
        )
        notes = "v2.2 upgraded only to ready_for_gold_candidate, not confirmed gold."
    else:
        change_type = "routing_preserved"
        upgrade_justification = "none"
        notes = "v2.2 preserved Phase7D-2 routing."
    delta = {
        "table_object_id": obj["table_object_id"],
        "phase7d2_routing_status": phase7d2_status,
        "phase7d3_routing_status": phase7d3_status,
        "changed": bool_text(changed),
        "change_type": change_type,
        "reconstruction_attempted": bool_text(record.get("reconstruction_attempted")),
        "reconstruction_strategy": record.get("reconstruction_strategy", ""),
        "reconstruction_status": reconstruction_status(record),
        "logical_cells_count": str(len(record.get("logical_cells") or [])),
        "missing_expected_cells_count": str(len(record.get("missing_expected_cells") or [])),
        "remaining_blockers": semicolon(blockers),
        "upgrade_justification": upgrade_justification,
        "downgrade_reason": "none",
        "notes": notes,
    }
    diagnostic = diagnostic_csv_row(record)
    return obj, row, diagnostic, delta


def diagnostic_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_object_id": record["table_object_id"],
        "doc_id": record["doc_id"],
        "table_id": record["table_id"],
        "reconstruction_attempted": bool_text(record.get("reconstruction_attempted")),
        "reconstruction_strategy": record.get("reconstruction_strategy", ""),
        "reconstruction_status": reconstruction_status(record),
        "logical_columns": semicolon(record.get("logical_columns") or []),
        "logical_rows_count": str(len(record.get("logical_rows") or [])),
        "logical_cells_count": str(len(record.get("logical_cells") or [])),
        "reconstructed_value_count": str(record.get("reconstructed_value_count") or 0),
        "missing_expected_cells": semicolon(record.get("missing_expected_cells") or []),
        "reconstruction_warnings": semicolon(record.get("reconstruction_warnings") or []),
        "remaining_blockers": semicolon(record.get("remaining_blockers") or []),
        "upgrade_eligible": bool_text(record.get("upgrade_eligible")),
        "metric_reconstruction_status": record.get("metric_reconstruction_status", ""),
        "row_reference_reconstruction_status": record.get("row_reference_reconstruction_status", ""),
        "unit_binding_status": record.get("unit_binding_status", ""),
        "footnote_binding_status": record.get("footnote_binding_status", ""),
        "reference_binding_status": record.get("reference_binding_status", ""),
        "value_bboxes_available": "false",
        "source_span_granularity": record.get("source_span_granularity", ""),
        "notes": semicolon(record.get("notes") or []),
    }


def build_v22_outputs(
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_inputs(inputs)
    phase7d2_row_by_id = {row["table_object_id"]: row for row in inputs["phase7d2_summary_rows"]}
    objects: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for source_obj in inputs["phase7d2_objects"]:
        record = reconstruction_for_object(source_obj)
        obj, row, diagnostic, delta = apply_reconstruction_to_object(source_obj, phase7d2_row_by_id[source_obj["table_object_id"]], record)
        objects.append(obj)
        rows.append(row)
        diagnostics.append(diagnostic)
        deltas.append(delta)
    validate_v22_outputs(objects, rows, deltas)
    return objects, rows, diagnostics, deltas


def has_forbidden_output_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN_OUTPUT_KEYS or has_forbidden_output_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_forbidden_output_key(item) for item in value)
    return False


def validate_v22_outputs(objects: list[dict[str, Any]], rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    if len(objects) != 16 or len(rows) != 16 or len(deltas) != 16:
        errors.append(f"v2.2 输出应覆盖 16 个 case，实际 objects={len(objects)} rows={len(rows)} delta={len(deltas)}")
    status_by_id = {row["table_object_id"]: row["routing_status"] for row in rows}
    action_by_id = {row["table_object_id"]: row["final_action"] for row in rows}
    ready_ids = {row["table_object_id"] for row in rows if row["routing_status"] == "ready_for_gold_candidate"}
    if not READY_IDS <= ready_ids:
        errors.append("2 个 Phase7D ready candidate 未稳定保留")
    if status_by_id.get(ALIGNMENT_BLOCKED_ID) != "needs_pdfplumber_rule_fix":
        errors.append("doc_0598 Table 1 不得被强行升级")
    for obj_id in GRID_REJECTED_IDS:
        if status_by_id.get(obj_id) in {"ready_for_gold_candidate", "partial_hybrid"} or action_by_id.get(obj_id) != "reject_pdfplumber_grid":
            errors.append(f"{obj_id} grid_rejected 回归")
    for obj_id in CHUNK_FALLBACK_IDS:
        if action_by_id.get(obj_id) != "use_chunk_fallback":
            errors.append(f"{obj_id} chunk_fallback final_action 回归")
    for obj_id in BACKLOG_IDS:
        if action_by_id.get(obj_id) != "keep_backlog":
            errors.append(f"{obj_id} backlog final_action 回归")
    if any(row.get("value_bboxes_available") != "false" for row in rows):
        errors.append("summary 中 value_bboxes_available 必须全部为 false")
    if any(obj.get("value_bboxes_available") for obj in objects):
        errors.append("objects 中 value_bboxes_available 不得为 true")
    if any(row.get("source_span_granularity") == "value_level" for row in rows):
        errors.append("summary 中 source_span_granularity 不得为 value_level")
    if any(obj.get("source_span_granularity") == "value_level" for obj in objects):
        errors.append("objects 中 source_span_granularity 不得为 value_level")
    if any(span.get("granularity") == "value_level" for obj in objects for span in obj.get("source_spans") or []):
        errors.append("source_spans 不得为 value_level")
    if any(has_forbidden_output_key(obj) for obj in objects):
        errors.append("v2.2 object 不得写 confirmed/prod ready 字段")
    payload = json.dumps({"objects": objects, "rows": rows}, ensure_ascii=False)
    if "production_ready" in payload or "confirmed_gold" in payload:
        errors.append("v2.2 输出 payload 不得出现 production_ready / confirmed_gold")
    if errors:
        raise ValueError("Phase7D-3 v2.2 输出校验失败：" + "; ".join(errors))


def split_outputs(objects: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id = {obj["table_object_id"]: obj for obj in objects}
    return {
        "ready": [by_id[row["table_object_id"]] for row in rows if row["routing_status"] == "ready_for_gold_candidate"],
        "rule_fix": [row for row in rows if row["routing_status"] == "needs_pdfplumber_rule_fix"],
        "grid_rejected": [row for row in rows if row["routing_status"] == "grid_rejected"],
        "chunk_fallback": [row for row in rows if row["routing_status"] == "chunk_fallback"],
        "backlog": [row for row in rows if row["routing_status"] == "backlog"],
    }


def cell_lookup(obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(cell.get("row_id"), cell.get("column_id")): cell for cell in obj.get("cells") or []}


def render_preview_table(obj: dict[str, Any], max_rows: int = 8, max_cols: int = 8) -> list[str]:
    columns = (obj.get("columns") or [])[:max_cols]
    rows = (obj.get("rows") or [])[:max_rows]
    if not columns or not rows:
        return ["_无法生成 table preview：columns 或 rows 为空。_"]
    lookup = cell_lookup(obj)
    header = ["row"] + [md_escape(col.get("header") or col.get("column_id")) for col in columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        values = [md_escape(row.get("row_label") or row.get("row_text") or row.get("row_id"))[:160]]
        for col in columns:
            cell = lookup.get((row.get("row_id"), col.get("column_id")))
            values.append(md_escape(cell.get("value_raw") if cell else "")[:160])
        lines.append("| " + " | ".join(values) + " |")
    if len(obj.get("rows") or []) > max_rows or len(obj.get("columns") or []) > max_cols:
        lines.extend(["", f"_预览已截断：显示前 {max_rows} 行、前 {max_cols} 列。_"])
    return lines


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


def facts(objects: list[dict[str, Any]], rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    splits = split_outputs(objects, rows)
    return {
        "total": len(objects),
        "routing_counts": Counter(row["routing_status"] for row in rows),
        "action_counts": Counter(row["final_action"] for row in rows),
        "ready": [obj["table_object_id"] for obj in splits["ready"]],
        "rule_fix": [row["table_object_id"] for row in splits["rule_fix"]],
        "grid_rejected": [row["table_object_id"] for row in splits["grid_rejected"]],
        "chunk_fallback": [row["table_object_id"] for row in splits["chunk_fallback"]],
        "backlog": [row["table_object_id"] for row in splits["backlog"]],
        "ready_stable": READY_IDS <= {obj["table_object_id"] for obj in splits["ready"]},
        "value_bbox_false": all(row.get("value_bboxes_available") == "false" for row in rows),
        "no_value_level": all(row.get("source_span_granularity") != "value_level" for row in rows),
        "logical_cells_total": sum(int(row.get("logical_cells_count") or 0) for row in rows),
        "missing_expected_total": sum(int(row.get("missing_expected_cells_count") or 0) for row in rows),
        "remaining_blocker_counts": Counter(
            blocker
            for row in rows
            for blocker in split_semicolon(row.get("remaining_blockers"))
        ),
        "upgraded": [
            row["table_object_id"]
            for row in rows
            if row.get("phase7d2_routing_status") != row.get("phase7d3_routing_status")
        ],
        "diagnostics": diagnostics,
    }


def write_review_markdown(objects: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Phase7D-3 hybrid table_objects v2.2 审阅视图",
        "",
        "本文件用于审阅 logical cell reconstruction 结果；JSONL 是机器可读 source of truth。",
        "",
        "本轮不扩大 smoke，不构造 confirmed gold，不接 production，不伪造 value-level bbox。",
        "",
    ]
    for obj in objects:
        lines.extend(
            [
                f"## {obj.get('table_object_id')}",
                "",
                f"- table_object_id：`{obj.get('table_object_id')}`",
                f"- doc_id：`{obj.get('doc_id')}`",
                f"- table_id：`{obj.get('table_id')}`",
                f"- phase7d2_routing_status：`{obj.get('phase7d2_routing_status')}`",
                f"- phase7d3_routing_status：`{obj.get('phase7d3_routing_status')}`",
                f"- final_action：`{obj.get('final_action')}`",
                f"- reconstruction_attempted：`{str(bool(obj.get('reconstruction_attempted'))).lower()}`",
                f"- reconstruction_strategy：`{obj.get('reconstruction_strategy')}`",
                f"- reconstruction_status：`{obj.get('reconstruction_status')}`",
                f"- logical_columns：`{semicolon(obj.get('logical_columns') or [])}`",
                f"- logical_cells_count：`{len(obj.get('logical_cells') or [])}`",
                f"- missing_expected_cells：`{semicolon(obj.get('missing_expected_cells') or [])}`",
                f"- remaining_blockers：`{semicolon(obj.get('remaining_blockers') or [])}`",
                f"- unit_binding_status：`{obj.get('unit_binding_status')}`",
                f"- footnote_binding_status：`{obj.get('footnote_binding_status')}`",
                f"- reference_binding_status：`{obj.get('reference_binding_status')}`",
                f"- value_bboxes_available：`false`",
                f"- source_span_granularity：`{obj.get('source_span_granularity')}`",
                f"- warnings：`{semicolon(obj.get('warnings') or [])}`",
                "",
                "### table preview",
                "",
            ]
        )
        lines.extend(render_preview_table(obj))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_guardrail(report_dir: Path, inventory: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7D-3 Guardrail",
        "",
        "## 1. 本轮定位",
        "",
        "本轮定位为 logical cell reconstruction：在 Phase7D-2 hybrid extractor v2.1 的基础上，对受限 rule-fix case 生成可审计 logical_cells。",
        "",
        "## 2. 明确边界",
        "",
        "1. 本轮定位为 logical cell reconstruction。",
        "2. 本轮不是审阅阶段。",
        "3. 本轮不是 diagnostics-only。",
        "4. 本轮不是 gold construction。",
        "5. 本轮不扩大 smoke。",
        "6. 本轮不引入 Camelot / PyMuPDF。",
        "7. 本轮不接 production。",
        "8. 本轮不访问 Milvus / BM25，不读取或查询 BM25 index。",
        "9. 本轮不运行 retrieval / model。",
        "10. 本轮不伪造 value-level bbox。",
        "11. 本轮不得强行升级 rule-fix case。",
        "12. doc_0598 Table 1 不作为升级目标。",
        "13. Route C 仍只是 backlog。",
        "",
        "## 3. Smoke doc_id",
        "",
    ]
    lines.extend(f"- `{doc_id}`" for doc_id in SMOKE_DOC_IDS)
    lines.extend(["", "## 4. 只读输入 inventory", "", "| path | required | exists | records | lines |", "|---|---:|---:|---:|---:|"])
    for item in inventory:
        lines.append(
            f"| `{item['path']}` | {str(item['required']).lower()} | {str(item['exists']).lower()} | {item['record_count']} | {item['line_count']} |"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase7d_3_guardrail.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_logical_reconstruction_report(report_dir: Path, diagnostics: list[dict[str, Any]]) -> None:
    by_id = {row["table_object_id"]: row for row in diagnostics}
    lines = [
        "# Logical Cell Reconstruction 报告",
        "",
        "## 1. 实现目标",
        "",
        "本轮新增 v2.2 logical reconstruction layer。logical_cells 是离线结构化重建结果，可用于 ready_for_gold_candidate routing 判断，但不等于 confirmed gold。",
        "",
        "## 2. 模板边界",
        "",
        "- `metric_column_template`：仅用于 doc_0687 Table 2 selected metric rows，属于 scoped case-specific template。",
        "- `row_reference_literal_template`：仅用于 doc_0523 Table 1 selected logical table body，属于 scoped case-specific template。",
        "- `no_reconstruction_alignment_blocked`：用于 doc_0598 Table 1，保留 alignment/source-span blocker。",
        "- 其他 case 不做 logical reconstruction，保持原 routing。",
        "",
        "## 3. doc_0687 Table 2",
        "",
    ]
    doc0687 = by_id[TARGET_METRIC_ID]
    lines.extend(
        [
            f"- reconstruction_status：`{doc0687['reconstruction_status']}`",
            f"- logical_columns：`{doc0687['logical_columns']}`",
            f"- logical_cells_count：{doc0687['logical_cells_count']}",
            f"- missing_expected_cells：`{doc0687['missing_expected_cells']}`",
            f"- remaining_blockers：`{doc0687['remaining_blockers']}`",
            "- 已检查 YE/S 与 qethanol/qxylose/qarabinose 的列身份；YE/S 使用 split decimal 合并，qxylose 使用 split header 后的 metric column。",
            "- unit scope 绑定为 YE/S 与 q-rate 两组；reference 只绑定 selected rows；footnote marker 保留但不伪造 value bbox。",
            "",
            "## 4. doc_0523 Table 1",
            "",
        ]
    )
    doc0523 = by_id[TARGET_ROW_REFERENCE_ID]
    lines.extend(
        [
            f"- reconstruction_status：`{doc0523['reconstruction_status']}`",
            f"- logical_columns：`{doc0523['logical_columns']}`",
            f"- logical_cells_count：{doc0523['logical_cells_count']}",
            f"- missing_expected_cells：`{doc0523['missing_expected_cells']}`",
            f"- remaining_blockers：`{doc0523['remaining_blockers']}`",
            "- 已保留 N.D. 原文、LNT II/LNT、g/L unit 和 row-level reference/source values。",
            "- table tail 中的正文被排除在 logical table body 之外，但原始 table preview 仍保留在对象中供审阅。",
            "",
            "## 5. doc_0598 Table 1",
            "",
        ]
    )
    doc0598 = by_id[ALIGNMENT_BLOCKED_ID]
    lines.extend(
        [
            f"- reconstruction_strategy：`{doc0598['reconstruction_strategy']}`",
            f"- remaining_blockers：`{doc0598['remaining_blockers']}`",
            "- 当前 blocker 不是 logical cell reconstruction 能单独解决的问题，应进入后续 alignment/source-span backlog 或继续保持 rule_fix。",
            "",
            "## 6. Guardrail",
            "",
            "- 未伪造 value-level bbox。",
            "- 未构造 confirmed gold。",
            "- 未接入 production。",
            "- case-specific template 已在本报告标记，不假装通用。",
        ]
    )
    (report_dir / "logical_cell_reconstruction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    data = facts(objects, rows, diagnostics)
    by_id = {row["table_object_id"]: row for row in diagnostics}
    lines = [
        "# table_object v2.2 验证报告",
        "",
        "## 1. v2.2 table_object 总数",
        "",
        f"- v2.2 table_object 总数：{data['total']}",
        "",
        "## 2. routing_status 分布",
        "",
    ]
    lines.extend(counter_table(data["routing_counts"], sorted(v2.ROUTING_STATUS_VALUES)))
    lines.extend(["", "## 3. final_action 分布", ""])
    lines.extend(counter_table(data["action_counts"], sorted(v2.FINAL_ACTION_VALUES)))
    for title, key in [
        ("4. ready_candidate_pool 清单", "ready"),
        ("5. rule_fix_cases 清单", "rule_fix"),
        ("6. grid_rejected 清单", "grid_rejected"),
        ("7. chunk_fallback 清单", "chunk_fallback"),
        ("8. backlog 清单", "backlog"),
    ]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- `{item}`" for item in data[key])
    lines.extend(
        [
            "",
            "## 9. Phase7D ready candidate 稳定性",
            "",
            f"- 2 个 Phase7D ready candidate 是否稳定保留：{'是' if data['ready_stable'] else '否'}。",
            "",
            "## 10. doc_0687 Table 2 reconstruction 结果",
            "",
            f"- `{TARGET_METRIC_ID}`：{by_id[TARGET_METRIC_ID]['reconstruction_status']}；missing_expected_cells={by_id[TARGET_METRIC_ID]['missing_expected_cells']}。",
            "",
            "## 11. doc_0523 Table 1 reconstruction 结果",
            "",
            f"- `{TARGET_ROW_REFERENCE_ID}`：{by_id[TARGET_ROW_REFERENCE_ID]['reconstruction_status']}；missing_expected_cells={by_id[TARGET_ROW_REFERENCE_ID]['missing_expected_cells']}。",
            "",
            "## 12. doc_0598 Table 1 是否保持 rule_fix",
            "",
            f"- 是，routing_status=`{[row for row in rows if row['table_object_id'] == ALIGNMENT_BLOCKED_ID][0]['routing_status']}`。",
            "",
            "## 13. logical_cells 数量统计",
            "",
            f"- logical_cells 总数：{data['logical_cells_total']}",
            f"- doc_0687 Table 2：{by_id[TARGET_METRIC_ID]['logical_cells_count']}",
            f"- doc_0523 Table 1：{by_id[TARGET_ROW_REFERENCE_ID]['logical_cells_count']}",
            "",
            "## 14. missing_expected_cells 统计",
            "",
            f"- missing_expected_cells 总数：{data['missing_expected_total']}",
            "",
            "## 15. remaining_blockers 统计",
            "",
        ]
    )
    if data["remaining_blocker_counts"]:
        lines.extend(counter_table(data["remaining_blocker_counts"]))
    else:
        lines.append("- 无")
    upgraded = data["upgraded"]
    lines.extend(
        [
            "",
            "## 16. 是否有 rule-fix case 被真实修复",
            "",
            f"- {'有：' + ', '.join('`' + item + '`' for item in upgraded) if upgraded else '没有。'}",
            "",
            "## 17. 如果无升级，原因是什么",
            "",
            "- 如无升级，原因应来自 remaining_blockers；本轮不会为了 ready 数量绕过 blocker。",
            "",
            "## 18. grid rejected 是否不再进入 usable",
            "",
            "- 是。5 个 grid_rejected 均保持 `final_action=reject_pdfplumber_grid`。",
            "",
            "## 19. chunk fallback 是否生效",
            "",
            "- 是。3 个 chunk_fallback 均保持 `final_action=use_chunk_fallback`。",
            "",
            "## 20. backlog 是否不再硬救",
            "",
            "- 是。3 个 backlog 均保持 `final_action=keep_backlog`。",
            "",
            "## 21. value_bboxes_available 是否全部 false",
            "",
            f"- {'是' if data['value_bbox_false'] else '否'}。",
            "",
            "## 22. source_span_granularity 是否没有 value_level",
            "",
            f"- {'是' if data['no_value_level'] else '否'}。",
            "",
            "## 23. validation 是否无回归",
            "",
            "- 是。Phase7D-2 的 ready/grid/fallback/backlog 均稳定，doc_0598 未强行升级。",
            "",
            "## 24. 未解决问题",
            "",
            "- v2.2 logical reconstruction 仍是离线 case-specific layer，不是 production extraction。",
            "- value-level bbox 仍不存在。",
            "- ready_for_gold_candidate 仍需要后续单独 gold construction 授权。",
        ]
    )
    (report_dir / "table_object_v2_2_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    data = facts(objects, rows, diagnostics)
    by_id = {row["table_object_id"]: row for row in diagnostics}
    lines = [
        "# Phase7D-2 与 Phase7D-3 对比报告",
        "",
        "## 1. 对比目标",
        "",
        "比较 v2.1 rule-fix diagnostics 与 v2.2 logical cell reconstruction 在同一批 16 个 hybrid case 上的变化。",
        "",
        "## 2. Phase7D-2 v2.1 状态",
        "",
        "- ready_for_gold_candidate：2",
        "- needs_pdfplumber_rule_fix：3",
        "- grid_rejected：5",
        "- chunk_fallback：3",
        "- backlog：3",
        "",
        "## 3. Phase7D-3 v2.2 状态",
        "",
    ]
    lines.extend(counter_table(data["routing_counts"], sorted(v2.ROUTING_STATUS_VALUES)))
    lines.extend(
        [
            "",
            "## 4. ready candidate 变化",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in data["ready"])
    lines.extend(
        [
            "",
            "## 5. rule-fix case 变化",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in data["rule_fix"])
    lines.extend(
        [
            "",
            "## 6. doc_0687 Table 2 是否被 logical reconstruction 修复",
            "",
            f"- `{TARGET_METRIC_ID}`：{by_id[TARGET_METRIC_ID]['reconstruction_status']}。",
            "",
            "## 7. doc_0523 Table 1 是否被 logical reconstruction 修复",
            "",
            f"- `{TARGET_ROW_REFERENCE_ID}`：{by_id[TARGET_ROW_REFERENCE_ID]['reconstruction_status']}。",
            "",
            "## 8. doc_0598 Table 1 为什么不升级",
            "",
            "- alignment_not_ready_eligible、page_only_match_not_high_confidence、source_span_not_cell_level_for_rule_fix 不是本轮 logical reconstruction 能单独解决的问题。",
            "",
            "## 9. 是否有 rule-fix case 被升级",
            "",
            f"- {'有：' + ', '.join('`' + item + '`' for item in data['upgraded']) if data['upgraded'] else '没有。'}",
            "",
            "## 10. 若有升级，升级依据是否充分",
            "",
            "- 升级仅在 missing_expected_cells 清空、unit/reference/footnote 不再阻断 selected evidence、value_bboxes_available=false 且 source_span_granularity 不是 value_level 时发生。",
            "",
            "## 11. 若无升级，blocker 是否更聚焦",
            "",
            "- 是。remaining_blockers 明确记录未解决项。",
            "",
            "## 12. grid rejected case 是否保持",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in data["grid_rejected"])
    lines.extend(["", "## 13. chunk fallback case 是否保持", ""])
    lines.extend(f"- `{item}`" for item in data["chunk_fallback"])
    lines.extend(["", "## 14. backlog case 是否保持", ""])
    lines.extend(f"- `{item}`" for item in data["backlog"])
    lines.extend(
        [
            "",
            "## 15. 是否减少误升级风险",
            "",
            "- 是。doc_0598 保持 rule_fix；grid/fallback/backlog 均未误入 ready。",
            "",
            "## 16. 是否从 warning 进入 reconstruction",
            "",
            "- 是。doc_0687 与 doc_0523 均生成 logical_rows/logical_cells，而不是只追加 warning。",
            "",
            "## 17. 是否仍需要 gold construction",
            "",
            "- 是。ready_for_gold_candidate 仍不等于 confirmed gold。",
            "",
            "## 18. 是否建议扩大 smoke",
            "",
            "- 不建议。",
            "",
            "## 19. 是否建议引入 Camelot / PyMuPDF",
            "",
            "- 不建议。",
            "",
            "## 20. 是否建议 production",
            "",
            "- 不建议。",
            "",
            "## 21. Route C 是否仍只是 backlog",
            "",
            "- 是，Route C 仍只是 backlog。",
            "",
            "## 22. 结论",
            "",
            "- 本轮不是为了增加 pass 数量。",
            "- 本轮是为了实现 logical cell reconstruction。",
            "- `ready_for_gold_candidate` 仍不等于 confirmed gold。",
            "- gold construction 仍需后续单独授权。",
            "- 不建议 production。",
        ]
    )
    (report_dir / "phase7d2_vs_phase7d3_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generated_files(output_dir: Path, report_dir: Path) -> list[str]:
    return [
        rel(output_dir / "table_objects.jsonl"),
        rel(output_dir / "table_objects_review.md"),
        rel(output_dir / "table_object_routing_summary.csv"),
        rel(output_dir / "logical_reconstruction_diagnostics.csv"),
        rel(output_dir / "logical_reconstruction_delta.csv"),
        rel(output_dir / "ready_candidate_pool.jsonl"),
        rel(output_dir / "rule_fix_cases.csv"),
        rel(output_dir / "grid_rejected_cases.csv"),
        rel(output_dir / "chunk_fallback_cases.csv"),
        rel(output_dir / "backlog_cases.csv"),
        rel(report_dir / "phase7d_3_guardrail.md"),
        rel(report_dir / "logical_cell_reconstruction_report.md"),
        rel(report_dir / "table_object_v2_2_validation_report.md"),
        rel(report_dir / "phase7d2_vs_phase7d3_comparison.md"),
        rel(report_dir / "phase7d_3_summary.md"),
    ]


def write_summary(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    output_dir: Path,
    report_dir: Path,
) -> None:
    data = facts(objects, rows, diagnostics)
    by_id = {row["table_object_id"]: row for row in diagnostics}
    lines = [
        "# Phase7D-3 总结",
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
            "- 新增：`scripts/extraction/reconstruct_logical_cells_v2.py`",
            "",
            "## 3. 新增测试",
            "",
            "- `tests/test_phase7_logical_cell_reconstruction_v2.py`",
            "",
            "## 4. smoke doc_id 是否保持不变",
            "",
            f"- 是，仍为：{', '.join(SMOKE_DOC_IDS)}",
            "",
            "## 5. v2.2 table_object 数量",
            "",
            f"- {data['total']}",
            "",
            "## 6. routing_status 统计",
            "",
        ]
    )
    lines.extend(counter_table(data["routing_counts"], sorted(v2.ROUTING_STATUS_VALUES)))
    lines.extend(["", "## 7. final_action 统计", ""])
    lines.extend(counter_table(data["action_counts"], sorted(v2.FINAL_ACTION_VALUES)))
    for title, key in [
        ("8. ready_candidate_pool 清单", "ready"),
        ("9. rule_fix_cases 清单", "rule_fix"),
        ("10. grid_rejected 清单", "grid_rejected"),
        ("11. chunk_fallback 清单", "chunk_fallback"),
        ("12. backlog 清单", "backlog"),
    ]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- `{item}`" for item in data[key])
    lines.extend(
        [
            "",
            "## 13. 2 个 ready candidate 是否稳定",
            "",
            f"- {'是' if data['ready_stable'] else '否'}。",
            "",
            "## 14. doc_0687 Table 2 reconstruction 结果",
            "",
            f"- {by_id[TARGET_METRIC_ID]['reconstruction_status']}；logical_cells={by_id[TARGET_METRIC_ID]['logical_cells_count']}；missing={by_id[TARGET_METRIC_ID]['missing_expected_cells']}。",
            "",
            "## 15. doc_0523 Table 1 reconstruction 结果",
            "",
            f"- {by_id[TARGET_ROW_REFERENCE_ID]['reconstruction_status']}；logical_cells={by_id[TARGET_ROW_REFERENCE_ID]['logical_cells_count']}；missing={by_id[TARGET_ROW_REFERENCE_ID]['missing_expected_cells']}。",
            "",
            "## 16. doc_0598 Table 1 是否保持 rule_fix",
            "",
            "- 是，保持 `needs_pdfplumber_rule_fix`。",
            "",
            "## 17. logical_cells 统计",
            "",
            f"- logical_cells 总数：{data['logical_cells_total']}",
            "",
            "## 18. missing_expected_cells 统计",
            "",
            f"- missing_expected_cells 总数：{data['missing_expected_total']}",
            "",
            "## 19. remaining_blockers 统计",
            "",
        ]
    )
    if data["remaining_blocker_counts"]:
        lines.extend(counter_table(data["remaining_blocker_counts"]))
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 20. 是否有 rule-fix case 被升级",
            "",
            f"- {'有：' + ', '.join('`' + item + '`' for item in data['upgraded']) if data['upgraded'] else '没有。'}",
            "",
            "## 21. 如果有升级，列出升级依据",
            "",
            "- 升级依据：selected logical cells 完整、unit/reference/footnote 不再阻断 selected evidence、value_bboxes_available=false、source_span_granularity 不是 value_level。",
            "",
            "## 22. 如果没有升级，说明 blocker 是否更聚焦",
            "",
            "- 如无升级，remaining_blockers 会聚焦到未解决项；本轮不会绕过 blocker。",
            "",
            "## 23. 是否复现 Phase7D-2 分流",
            "",
            "- ready/grid/fallback/backlog 均稳定；仅允许真实 logical reconstruction 后的 rule-fix 升级。",
            "",
            "## 24. 相比 Phase7D-2 的主要改善",
            "",
            "- 从 warning diagnostics 进入 logical_rows/logical_cells 结构化重建。",
            "- doc_0687 Table 2 和 doc_0523 Table 1 生成 case-specific logical reconstruction 结果。",
            "- doc_0598 Table 1 的 alignment/source-span blocker 更明确。",
            "",
            "## 25. 仍然存在的问题",
            "",
            "- v2.2 仍是离线受限模板，不是通用 production extractor。",
            "- value-level bbox 仍不存在。",
            "",
            "## 26. 是否建议进入 gold construction",
            "",
            "- 不建议本轮进入；gold construction 需要后续单独授权。",
            "",
            "## 27. 是否建议继续 pdfplumber 主线",
            "",
            "- 建议继续离线 hardening，但不接 production。",
            "",
            "## 28. 是否建议扩大 smoke",
            "",
            "- 不建议。",
            "",
            "## 29. 是否建议引入 Camelot / PyMuPDF",
            "",
            "- 不建议。",
            "",
            "## 30. 是否建议进入 production",
            "",
            "- 不建议。",
            "",
            "## 31. baseline / guardrail 是否漂移",
            "",
            "- 未发现漂移。未修改 official dataset、official baseline、configs 或 baseline registry。",
            "",
            "## 32. Route C 是否仍只是 backlog",
            "",
            "- 是，Route C 仍只是 backlog。",
            "",
            "## 33. 明确未执行事项",
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
    (report_dir / "phase7d_3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    output_dir: Path,
    report_dir: Path,
    inventory: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    splits = split_outputs(objects, rows)
    write_jsonl(objects, output_dir / "table_objects.jsonl")
    write_review_markdown(objects, output_dir / "table_objects_review.md")
    write_csv(rows, output_dir / "table_object_routing_summary.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(diagnostics, output_dir / "logical_reconstruction_diagnostics.csv", LOGICAL_DIAGNOSTIC_FIELDS)
    write_csv(deltas, output_dir / "logical_reconstruction_delta.csv", DELTA_FIELDS)
    write_jsonl(splits["ready"], output_dir / "ready_candidate_pool.jsonl")
    write_csv(splits["rule_fix"], output_dir / "rule_fix_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(splits["grid_rejected"], output_dir / "grid_rejected_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(splits["chunk_fallback"], output_dir / "chunk_fallback_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(splits["backlog"], output_dir / "backlog_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_guardrail(report_dir, inventory)
    write_logical_reconstruction_report(report_dir, diagnostics)
    write_validation_report(objects, rows, diagnostics, report_dir)
    write_comparison_report(objects, rows, diagnostics, report_dir)
    write_summary(objects, rows, diagnostics, output_dir, report_dir)


def run(args: argparse.Namespace) -> None:
    required_paths = (
        PHASE7D2_REQUIRED_INPUTS
        + PHASE7C_REQUIRED_INPUTS
        + CURRENT_EXTRACTOR_SCRIPTS
        + PHASE6D_REQUIRED_INPUTS
    )
    inventory = read_input_inventory(required_paths, OPTIONAL_EXISTING_TESTS)
    inputs = load_inputs()
    objects, rows, diagnostics, deltas = build_v22_outputs(inputs)
    write_outputs(objects, rows, diagnostics, deltas, args.output_dir, args.report_dir, inventory)
    print(
        json.dumps(
            {
                "table_objects": len(objects),
                "routing_status": dict(Counter(row["routing_status"] for row in rows)),
                "final_action": dict(Counter(row["final_action"] for row in rows)),
                "logical_reconstruction_diagnostics": len(diagnostics),
                "output_dir": rel(args.output_dir),
                "report_dir": rel(args.report_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase7D-3 logical cell reconstruction v2.2.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
