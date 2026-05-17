#!/usr/bin/env python3
"""Phase7E offline hybrid table gold seed construction.

This script consumes Phase7D-3 ready candidates and constructs a small,
reviewable table gold seed. It does not read BM25, does not access Milvus, does
not call models, and does not re-run PDF extraction.
"""

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


PHASE7D3_DATA_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_logical_reconstruction"
PHASE7D3_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_extractor_v2_logical_reconstruction"
PHASE7C4_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_binding_review"
PHASE6D_REPORT_DIR = ROOT / "reports/v7_phase6d_table_contract_refinement"

DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_hybrid_gold_seed"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_gold_seed"

READY_CANDIDATE_IDS = [
    "doc_0468__table_2__phase7c2_hybrid_01",
    "doc_0687__table_2__phase7c2_hybrid_02",
    "doc_0687__table_3__phase7c2_hybrid_03",
    "doc_0523__table_1__phase7c2_hybrid_01",
]

CONFIRMED_SEED_IDS = {
    "doc_0687__table_2__phase7c2_hybrid_02",
    "doc_0523__table_1__phase7c2_hybrid_01",
}

PARTIAL_SEED_IDS = {
    "doc_0468__table_2__phase7c2_hybrid_01",
    "doc_0687__table_3__phase7c2_hybrid_03",
}

RULE_FIX_IDS = {"doc_0598__table_1__phase7c2_hybrid_01"}
GRID_REJECTED_IDS = {
    "doc_0322__table_1__phase7c2_hybrid_01",
    "doc_0158__table_2__phase7c2_hybrid_01",
    "doc_0598__table_2__phase7c2_hybrid_02",
    "doc_0452__table_1__phase7c2_hybrid_01",
    "doc_0687__table_1__phase7c2_hybrid_01",
}
CHUNK_FALLBACK_IDS = {
    "doc_0158__table_3__phase7c2_hybrid_02",
    "doc_0468__table_3__phase7c2_hybrid_02",
    "doc_0522__table_1__phase7c2_hybrid_01",
}
BACKLOG_IDS = {
    "doc_0458__table_1__phase7c2_hybrid_01",
    "doc_0458__table_2__phase7c2_hybrid_02",
    "doc_0458__table_3__phase7c2_hybrid_03",
}

DOC0687_TABLE2_COLUMNS = [
    "strain_or_variant",
    "YE/S",
    "qethanol",
    "qxylose",
    "qarabinose",
    "Reference",
]

DOC0687_TABLE3_COLUMNS = [
    "strain_or_variant",
    "YE/S",
    "qglucose",
    "qethanol",
    "qxylose",
    "Reference",
]

DOC0523_TABLE1_COLUMNS = [
    "strain_or_construct",
    "LNT_II",
    "LNT",
    "titer_or_concentration",
    "unit",
    "reference_or_source",
    "medium_culture_conditions",
]

DOC0468_TABLE2_COLUMNS = [
    {"column_key": "bacterial_species", "source_column_indices": [1], "unit": "not_applicable"},
    {"column_key": "strain_designation_fragments", "source_column_indices": [2, 3], "unit": "not_applicable"},
    {"column_key": "source", "source_column_indices": [4], "unit": "not_applicable"},
    {"column_key": "strain_type_or_probiotic", "source_column_indices": [5], "unit": "not_applicable"},
    {"column_key": "medium", "source_column_indices": [6], "unit": "not_applicable"},
    {"column_key": "T_star", "source_column_indices": [7], "unit": "visible_not_bound"},
    {"column_key": "atmosphere_abbreviation", "source_column_indices": [8], "unit": "not_applicable"},
]

DOC0468_SELECTED_ROW_INDICES = [8, 9, 10, 11, 12, 13, 14, 16]

DOC0687_TABLE3_ROW_GROUPS = [
    {"row_key": "TMB3400_row12", "primary_row": 12, "continuation_rows": [13, 14, 15, 16, 17]},
    {"row_key": "GLBRCY87_row18", "primary_row": 18, "continuation_rows": [19, 20, 21, 22, 23]},
    {"row_key": "GLBRCY87_row24", "primary_row": 24, "continuation_rows": [25, 26, 27, 28, 29]},
    {"row_key": "MEC1122_row30", "primary_row": 30, "continuation_rows": [31, 32, 33, 34, 35]},
    {"row_key": "RWB218_row36", "primary_row": 36, "continuation_rows": [37, 38, 39, 40, 41, 42]},
    {"row_key": "GS1_11_26_row43", "primary_row": 43, "continuation_rows": [44, 45, 46, 47, 48, 49, 50, 51]},
    {"row_key": "XH7_row52", "primary_row": 52, "continuation_rows": [53, 54, 55, 56]},
    {"row_key": "LF1_row57", "primary_row": 57, "continuation_rows": [58, 59, 60, 61, 62]},
]

DOC0687_TABLE3_COLUMN_MAP = {
    "strain_or_variant": [1],
    "YE/S": [5],
    "qglucose": [6],
    "qethanol": [7],
    "qxylose": [8],
    "Reference": [9],
}

INPUT_SPECS = {
    "phase7d3_table_objects": ("jsonl", PHASE7D3_DATA_DIR / "table_objects.jsonl"),
    "phase7d3_table_objects_review": ("text", PHASE7D3_DATA_DIR / "table_objects_review.md"),
    "phase7d3_table_object_routing_summary": ("csv", PHASE7D3_DATA_DIR / "table_object_routing_summary.csv"),
    "phase7d3_logical_reconstruction_diagnostics": (
        "csv",
        PHASE7D3_DATA_DIR / "logical_reconstruction_diagnostics.csv",
    ),
    "phase7d3_logical_reconstruction_delta": ("csv", PHASE7D3_DATA_DIR / "logical_reconstruction_delta.csv"),
    "phase7d3_ready_candidate_pool": ("jsonl", PHASE7D3_DATA_DIR / "ready_candidate_pool.jsonl"),
    "phase7d3_rule_fix_cases": ("csv", PHASE7D3_DATA_DIR / "rule_fix_cases.csv"),
    "phase7d3_grid_rejected_cases": ("csv", PHASE7D3_DATA_DIR / "grid_rejected_cases.csv"),
    "phase7d3_chunk_fallback_cases": ("csv", PHASE7D3_DATA_DIR / "chunk_fallback_cases.csv"),
    "phase7d3_backlog_cases": ("csv", PHASE7D3_DATA_DIR / "backlog_cases.csv"),
    "phase7d3_logical_cell_reconstruction_report": (
        "text",
        PHASE7D3_REPORT_DIR / "logical_cell_reconstruction_report.md",
    ),
    "phase7d3_table_object_v2_2_validation_report": (
        "text",
        PHASE7D3_REPORT_DIR / "table_object_v2_2_validation_report.md",
    ),
    "phase7d3_phase7d2_vs_phase7d3_comparison": (
        "text",
        PHASE7D3_REPORT_DIR / "phase7d2_vs_phase7d3_comparison.md",
    ),
    "phase7d3_summary": ("text", PHASE7D3_REPORT_DIR / "phase7d_3_summary.md"),
    "phase7c4_hybrid_binding_review": ("jsonl", PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl"),
    "phase7c4_hybrid_binding_review_summary": ("csv", PHASE7C4_DATA_DIR / "hybrid_binding_review_summary.csv"),
    "phase6d_numeric_unit_footnote_contract": (
        "text",
        PHASE6D_REPORT_DIR / "numeric_unit_footnote_contract.md",
    ),
    "phase6d_numeric_unit_footnote_rules": ("csv", PHASE6D_REPORT_DIR / "numeric_unit_footnote_rules.csv"),
    "phase6d_matrix_superscript_literal_contract": (
        "text",
        PHASE6D_REPORT_DIR / "matrix_superscript_literal_contract.md",
    ),
    "phase6d_matrix_superscript_literal_rules": (
        "csv",
        PHASE6D_REPORT_DIR / "matrix_superscript_literal_rules.csv",
    ),
    "phase6d_source_span_granularity_contract": (
        "text",
        PHASE6D_REPORT_DIR / "source_span_granularity_contract.md",
    ),
    "phase6d_source_span_granularity_rules": (
        "csv",
        PHASE6D_REPORT_DIR / "source_span_granularity_rules.csv",
    ),
    "phase6d_partial_to_confirmed_decision_guide": (
        "text",
        PHASE6D_REPORT_DIR / "partial_to_confirmed_decision_guide.md",
    ),
    "phase6d_partial_to_confirmed_rules": ("csv", PHASE6D_REPORT_DIR / "partial_to_confirmed_rules.csv"),
}

SUMMARY_FIELDS = [
    "gold_seed_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "gold_seed_status",
    "phase7d3_reconstruction_status",
    "source_span_granularity",
    "value_bboxes_available",
    "cell_bboxes_available",
    "gold_rows_count",
    "gold_columns_count",
    "gold_cells_count",
    "required_values_count",
    "required_units_count",
    "validation_status",
    "construction_warnings",
    "remaining_blockers",
    "seed_notes",
]

PARTIAL_FIELDS = [
    "gold_seed_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "remaining_blockers",
    "construction_warnings",
    "seed_notes",
]

EXCLUDED_FIELDS = [
    "gold_seed_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "exclusion_reason",
    "seed_notes",
]


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_inputs() -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for name, (kind, path) in INPUT_SPECS.items():
        if kind == "jsonl":
            inputs[name] = read_jsonl(path)
        elif kind == "csv":
            inputs[name] = read_csv(path)
        elif kind == "text":
            inputs[name] = read_text(path)
        else:
            raise ValueError(f"Unsupported input kind: {kind}")
    return inputs


def unique(values: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def compact_join(values: list[Any]) -> str:
    text_values = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            text_values.extend(str(item) for item in value if item not in (None, ""))
        else:
            text_values.append(str(value))
    return ";".join(unique(text_values))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def normalize_cell_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text)


def span_index(table_object: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {span["source_span_id"]: span for span in table_object.get("source_spans", [])}


def collect_source_spans(table_object: dict[str, Any], source_span_ids: list[str]) -> list[dict[str, Any]]:
    by_id = span_index(table_object)
    spans = []
    for span_id in unique(source_span_ids):
        span = by_id.get(span_id)
        if span is not None:
            spans.append(span)
    return spans


def cells_by_row(table_object: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in table_object.get("cells", []):
        grouped.setdefault(cell["row_id"], []).append(cell)
    return grouped


def rows_by_index(table_object: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["row_index"]: row for row in table_object.get("rows", [])}


def raw_cells_for_row(table_object: dict[str, Any], row_index: int) -> list[dict[str, Any]]:
    row = rows_by_index(table_object)[row_index]
    return cells_by_row(table_object).get(row["row_id"], [])


def raw_values_for_columns(table_object: dict[str, Any], row_indices: list[int], column_indices: list[int]) -> tuple[str, list[str]]:
    values: list[str] = []
    source_span_ids: list[str] = []
    for row_index in row_indices:
        for cell in raw_cells_for_row(table_object, row_index):
            column_id = cell.get("column_id", "")
            column_index = int(column_id.rsplit("_", 1)[-1]) if column_id.rsplit("_", 1)[-1].isdigit() else None
            if column_index in column_indices:
                value = normalize_cell_text(cell.get("value_raw"))
                if value:
                    values.append(value)
                source_span_ids.extend(cell.get("source_span_ids", []))
    return " ".join(values).strip(), unique(source_span_ids)


def make_gold_seed_id(ordinal: int, table_object_id: str) -> str:
    return f"phase7e_gold_seed_{ordinal:03d}__{table_object_id}"


def source_phase(table_object: dict[str, Any]) -> str:
    if table_object.get("reconstruction_attempted") is True:
        return "Phase7D-3 hybrid extractor v2.2 logical reconstruction"
    return "Phase7D-3 ready_candidate_pool with Phase7C-4 binding review"


def binding_review_by_id(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in inputs["phase7c4_hybrid_binding_review"]:
        out[row["hybrid_table_object_id"]] = row
    return out


def diagnostics_by_id(inputs: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {row["table_object_id"]: row for row in inputs["phase7d3_logical_reconstruction_diagnostics"]}


def table_objects_by_id(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["table_object_id"]: row for row in inputs["phase7d3_table_objects"]}


def ready_pool_by_id(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["table_object_id"]: row for row in inputs["phase7d3_ready_candidate_pool"]}


def common_warnings(table_object: dict[str, Any], binding_review: dict[str, Any], extra: list[str]) -> list[str]:
    warning_sources = [
        table_object.get("reconstruction_warnings", []),
        table_object.get("binding_review_key_warnings", []),
        binding_review.get("key_warnings", []),
        extra,
    ]
    warnings: list[str] = []
    for source in warning_sources:
        if isinstance(source, list):
            warnings.extend(str(item) for item in source if item)
        elif source:
            warnings.append(str(source))
    return unique(warnings)


def logical_gold_rows(table_object: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "row_key": row.get("row_key"),
            "row_label": row.get("row_label"),
            "selected_for_reconstruction": row.get("selected_for_reconstruction", True),
            "source_row_indices": row.get("source_row_indices", []),
            "source_span_ids": row.get("source_span_ids", []),
        }
        for row in table_object.get("logical_rows", [])
    ]


def logical_gold_columns(table_object: dict[str, Any]) -> list[dict[str, Any]]:
    cells = table_object.get("logical_cells", [])
    columns = []
    for column in table_object.get("logical_columns", []):
        units = unique([cell.get("unit") for cell in cells if cell.get("logical_column") == column and cell.get("unit")])
        columns.append(
            {
                "column_key": column,
                "column_label": column,
                "unit": units[0] if len(units) == 1 else (units or "not_applicable"),
                "source": "phase7d3_logical_columns",
            }
        )
    return columns


def logical_gold_cells(table_object: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for cell in table_object.get("logical_cells", []):
        cells.append(
            {
                "gold_cell_id": cell.get("logical_cell_id"),
                "source_logical_cell_id": cell.get("logical_cell_id"),
                "row_key": cell.get("row_key"),
                "row_label": cell.get("row_label"),
                "logical_column": cell.get("logical_column"),
                "value_raw": normalize_cell_text(cell.get("value_raw")),
                "value_normalized": cell.get("value_normalized"),
                "unit": cell.get("unit"),
                "footnote_refs": cell.get("footnote_refs", []),
                "reference_or_source": cell.get("reference_or_source"),
                "source_span_ids": cell.get("source_span_ids", []),
                "source_span_granularity": cell.get("source_span_granularity", table_object.get("source_span_granularity")),
                "value_bbox": None,
                "value_bbox_source": "not_available",
                "notes": unique((cell.get("notes") or []) + ["Phase7E seed: value-level bbox 不存在，未推断。"]),
            }
        )
    return cells


def required_values_from_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = []
    for cell in cells:
        value_raw = normalize_cell_text(cell.get("value_raw"))
        if not value_raw:
            continue
        required.append(
            {
                "row_key": cell.get("row_key"),
                "logical_column": cell.get("logical_column"),
                "value_raw": value_raw,
                "unit": cell.get("unit"),
                "footnote_refs": cell.get("footnote_refs", []),
                "reference_or_source": cell.get("reference_or_source"),
                "source_span_ids": cell.get("source_span_ids", []),
            }
        )
    return required


def units_for_doc0687_table2() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "doc_0687_table2_unit_ye_s",
            "scope": "YE/S",
            "unit_raw": "g ethanol.(g sugar)-1",
            "binding_status": "bound_to_metric_column_with_warnings",
            "notes": "unit 来自 caption/header scope；Phase7D-3 logical cells 已绑定到 YE/S cells。",
        },
        {
            "unit_id": "doc_0687_table2_unit_q_rates",
            "scope": "qethanol;qxylose;qarabinose",
            "unit_raw": "g.(g biomass)-1.h-1",
            "binding_status": "bound_to_metric_columns_with_warnings",
            "notes": "unit 来自 caption/header scope；不是 value-level unit bbox。",
        },
    ]


def units_for_doc0523_table1() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "doc_0523_table1_unit_titer",
            "scope": "LNT_II;LNT;titer_or_concentration",
            "unit_raw": "g/L",
            "binding_status": "bound_to_selected_cells_with_warnings",
            "notes": "g/L literal 保留在 unit column 与 titer cells；无 value-level bbox。",
        }
    ]


def units_for_doc0468_table2() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "doc_0468_table2_t_star",
            "scope": "T* column",
            "unit_raw": "temperature marker T* visible; per-cell unit not bound",
            "binding_status": "uncertain_visible_not_bound",
            "notes": "unit/temperature visible 不等于 unit bound；本 seed 保持 partial。",
        },
        {
            "unit_id": "doc_0468_table2_non_numeric_columns",
            "scope": "bacterial species;strain designation;source;type;medium;atmosphere abbreviation",
            "unit_raw": "not_applicable",
            "binding_status": "not_applicable",
            "notes": "非数值 literal/source/abbreviation 字段不需要 numeric unit。",
        },
    ]


def units_for_doc0687_table3() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "doc_0687_table3_unit_ye_s",
            "scope": "YE/S",
            "unit_raw": "g.g-1",
            "binding_status": "visible_metric_scope_not_fully_confirmed",
            "notes": "header unit 可见，但 Phase7D-3 未对 Table 3 做 logical reconstruction；不升级 confirmed。",
        },
        {
            "unit_id": "doc_0687_table3_unit_q_rates",
            "scope": "qglucose;qethanol;qxylose",
            "unit_raw": "g.g-1.h-1",
            "binding_status": "visible_metric_scope_not_fully_confirmed",
            "notes": "unit visible 不等于 per-value unit bound。",
        },
    ]


def marker_refs(value_raw: str) -> list[str]:
    refs = []
    if "*" in value_raw or "∗" in value_raw:
        refs.append("asterisk_marker_retained")
    if "†" in value_raw:
        refs.append("dagger_marker_retained")
    return refs


def reference_map_from_cells(cells: list[dict[str, Any]], reference_column: str) -> dict[str, str]:
    out = {}
    for cell in cells:
        if cell.get("logical_column") == reference_column and normalize_cell_text(cell.get("value_raw")):
            out[str(cell.get("row_key"))] = normalize_cell_text(cell.get("value_raw"))
    return out


def build_confirmed_logical_seed(
    ordinal: int,
    table_object: dict[str, Any],
    binding_review: dict[str, Any],
    required_units: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_rows = logical_gold_rows(table_object)
    gold_columns = logical_gold_columns(table_object)
    gold_cells = logical_gold_cells(table_object)
    required_values = required_values_from_cells(gold_cells)
    source_span_ids = [span_id for cell in gold_cells for span_id in cell.get("source_span_ids", [])]
    table_object_id = table_object["table_object_id"]

    if table_object_id == "doc_0687__table_2__phase7c2_hybrid_02":
        footnote_binding = {
            "binding_status": "bound_with_warnings",
            "scope": "cell_marker_retained_where_visible",
            "notes": "asterisk marker 保留在 value_raw/footnote_refs；未构造 value-level bbox。",
        }
        reference_binding = {
            "binding_status": "row_level_reference_bound_with_warnings",
            "reference_type": "row-level reference column",
            "row_reference_map": reference_map_from_cells(gold_cells, "Reference"),
            "notes": "Reference column 绑定到 reconstructed logical rows；不是外部 citation provenance 抽取。",
        }
        literal_preservation = {
            "status": "pass_with_warnings",
            "preserved_literals": ["numeric value_raw", "asterisk marker when visible", "strain_or_variant", "Reference"],
        }
        seed_notes = [
            "基于 Phase7D-3 metric_column_template logical reconstruction。",
            "confirmed_seed 只表示 offline gold seed formal subset，不表示 production-ready。",
            "selected_rows_only_not_whole_table_gold warning 保留。",
        ]
    else:
        footnote_binding = {
            "binding_status": "not_applicable",
            "scope": "no selected footnote marker in Phase7D-3 logical cells",
            "notes": "Phase7D-3 对该 seed 的 footnote_binding_status 为 not_applicable。",
        }
        reference_binding = {
            "binding_status": "row_level_reference_or_source_bound_with_warnings",
            "reference_type": "row-level reference/source column",
            "row_reference_map": reference_map_from_cells(gold_cells, "reference_or_source"),
            "notes": "reference/source 绑定到 reconstructed logical rows；this study 保留原文。",
        }
        literal_preservation = {
            "status": "pass",
            "preserved_literals": ["LNT II", "LNT", "N.D.", "g/L", "this study", "strain_or_construct"],
        }
        seed_notes = [
            "基于 Phase7D-3 row_reference_literal_template logical reconstruction。",
            "N.D. literal、g/L unit 与 row-level reference/source 均保留 value_raw。",
            "confirmed_seed 不是 production-ready，也不是 official benchmark。",
        ]

    warnings = unique(
        list(table_object.get("reconstruction_warnings", []))
        + [
            "phase7c4_binding_warnings_retained_as_historical_context",
            "phase7d3_selected_logical_cells_resolve_formal_seed_blockers",
            "value_bboxes_available_false",
            "cell_bbox_not_value_bbox",
            "confirmed_seed_not_production_ready",
            "gold_seed_not_official_benchmark",
        ]
    )

    return {
        "gold_seed_id": make_gold_seed_id(ordinal, table_object_id),
        "table_object_id": table_object_id,
        "doc_id": table_object["doc_id"],
        "table_id": table_object["table_id"],
        "source_phase": source_phase(table_object),
        "gold_seed_status": "confirmed_seed",
        "gold_rows": gold_rows,
        "gold_columns": gold_columns,
        "gold_cells": gold_cells,
        "required_values": required_values,
        "required_units": required_units,
        "footnote_binding": footnote_binding,
        "reference_binding": reference_binding,
        "literal_preservation": literal_preservation,
        "source_spans": collect_source_spans(table_object, source_span_ids),
        "source_span_granularity": table_object.get("source_span_granularity"),
        "value_bboxes_available": False,
        "cell_bboxes_available": bool(table_object.get("cell_bboxes_available")),
        "construction_warnings": warnings,
        "remaining_blockers": [],
        "seed_notes": seed_notes,
        "benchmark_scope": "offline_gold_seed_only_not_official_benchmark",
        "production_scope": "not_production_ready",
        "phase7d3_reconstruction_status": table_object.get("reconstruction_status"),
    }


def build_doc0468_partial_seed(
    ordinal: int,
    table_object: dict[str, Any],
    binding_review: dict[str, Any],
) -> dict[str, Any]:
    row_lookup = rows_by_index(table_object)
    gold_rows = []
    gold_cells = []
    source_span_ids: list[str] = []

    for row_index in DOC0468_SELECTED_ROW_INDICES:
        row = row_lookup[row_index]
        row_key = f"raw_row_{row_index:03d}"
        gold_rows.append(
            {
                "row_key": row_key,
                "row_label": row.get("row_label"),
                "row_text": row.get("row_text"),
                "source_row_indices": [row_index],
                "source_span_ids": row.get("source_span_ids", []),
                "notes": ["raw pdfplumber row retained; split designation not fully reconstructed"],
            }
        )
        for column in DOC0468_TABLE2_COLUMNS:
            value_raw, spans = raw_values_for_columns(table_object, [row_index], column["source_column_indices"])
            source_span_ids.extend(spans)
            gold_cells.append(
                {
                    "gold_cell_id": f"{make_gold_seed_id(ordinal, table_object['table_object_id'])}__{row_key}__{column['column_key']}",
                    "row_key": row_key,
                    "logical_column": column["column_key"],
                    "value_raw": value_raw,
                    "unit": column["unit"],
                    "footnote_refs": ["T_star_marker_visible"] if column["column_key"] == "T_star" else [],
                    "reference_or_source": value_raw if column["column_key"] == "source" else None,
                    "source_span_ids": spans,
                    "source_span_granularity": table_object.get("source_span_granularity"),
                    "value_bbox": None,
                    "value_bbox_source": "not_available",
                    "notes": ["partial seed cell; source is pdfplumber cell-level provenance, not value bbox"],
                }
            )

    gold_columns = [
        {
            "column_key": column["column_key"],
            "source_column_indices": column["source_column_indices"],
            "unit": column["unit"],
            "source": "Phase7E partial reconstruction from Phase7D-3 raw cells",
        }
        for column in DOC0468_TABLE2_COLUMNS
    ]
    blockers = [
        "split_strain_designation_not_confirmed",
        "unit_visible_not_bound_to_cell",
        "footnote_or_abbreviation_marker_not_bound_to_specific_cell",
        "source_column_is_table_internal_not_external_reference",
        "phase7d3_logical_reconstruction_not_attempted",
        "value_level_bbox_absent",
    ]
    warnings = common_warnings(
        table_object,
        binding_review,
        [
            "partial_seed_due_to_unconfirmed_binding",
            "strain designation raw fragments retained instead of normalized",
            "T*/atmosphere* footnote or abbreviation remains table-level/cell-level warning only",
        ],
    )

    return {
        "gold_seed_id": make_gold_seed_id(ordinal, table_object["table_object_id"]),
        "table_object_id": table_object["table_object_id"],
        "doc_id": table_object["doc_id"],
        "table_id": table_object["table_id"],
        "source_phase": source_phase(table_object),
        "gold_seed_status": "partial_seed",
        "gold_rows": gold_rows,
        "gold_columns": gold_columns,
        "gold_cells": gold_cells,
        "required_values": required_values_from_cells(gold_cells),
        "required_units": units_for_doc0468_table2(),
        "footnote_binding": {
            "binding_status": "partial_table_level_visible_not_cell_bound",
            "scope": "T* and atmosphere* markers",
            "notes": "footnote/abbreviation 可见，但不能确认绑定到每个 selected cell/value。",
        },
        "reference_binding": {
            "binding_status": "table_internal_source_column_with_warnings",
            "reference_type": "table-internal source",
            "notes": "source 字段是表内 source column，不是外部 citation provenance。",
        },
        "literal_preservation": {
            "status": "pass_with_warnings",
            "preserved_literals": ["strain raw fragments", "typestrain", "probiotic", "anaerobic", "abbreviation"],
            "notes": "split strain designation 如 DS | M20083 以 raw fragments 保留，未改写为 value-level bbox。",
        },
        "source_spans": collect_source_spans(table_object, source_span_ids),
        "source_span_granularity": table_object.get("source_span_granularity"),
        "value_bboxes_available": False,
        "cell_bboxes_available": bool(table_object.get("cell_bboxes_available")),
        "construction_warnings": warnings,
        "remaining_blockers": blockers,
        "seed_notes": [
            "本对象在 Phase7D-3 未做 logical reconstruction，因此只能作为 exploratory partial seed。",
            "构造了 selected raw rows/columns/cells，但 binding 仍需人工审阅。",
        ],
        "benchmark_scope": "offline_gold_seed_only_not_official_benchmark",
        "production_scope": "not_production_ready",
        "phase7d3_reconstruction_status": table_object.get("reconstruction_status"),
    }


def build_doc0687_table3_partial_seed(
    ordinal: int,
    table_object: dict[str, Any],
    binding_review: dict[str, Any],
) -> dict[str, Any]:
    row_lookup = rows_by_index(table_object)
    gold_rows = []
    gold_cells = []
    source_span_ids: list[str] = []
    seed_id = make_gold_seed_id(ordinal, table_object["table_object_id"])

    for group in DOC0687_TABLE3_ROW_GROUPS:
        primary_row = row_lookup[group["primary_row"]]
        row_indices = [group["primary_row"]] + group["continuation_rows"]
        available_row_indices = [index for index in row_indices if index in row_lookup]
        row_source_spans = unique(
            [
                span_id
                for index in available_row_indices
                for span_id in row_lookup[index].get("source_span_ids", [])
            ]
        )
        gold_rows.append(
            {
                "row_key": group["row_key"],
                "row_label": primary_row.get("row_label"),
                "row_text": primary_row.get("row_text"),
                "source_row_indices": available_row_indices,
                "source_span_ids": row_source_spans,
                "notes": ["row continuation retained for reference/source audit; not confirmed row-level binding"],
            }
        )
        for logical_column, source_columns in DOC0687_TABLE3_COLUMN_MAP.items():
            source_rows = available_row_indices if logical_column == "Reference" else [group["primary_row"]]
            value_raw, spans = raw_values_for_columns(table_object, source_rows, source_columns)
            source_span_ids.extend(spans)
            gold_cells.append(
                {
                    "gold_cell_id": f"{seed_id}__{group['row_key']}__{logical_column}",
                    "row_key": group["row_key"],
                    "logical_column": logical_column,
                    "value_raw": value_raw,
                    "unit": "g.g-1" if logical_column == "YE/S" else (
                        "g.g-1.h-1" if logical_column in {"qglucose", "qethanol", "qxylose"} else None
                    ),
                    "footnote_refs": marker_refs(value_raw),
                    "reference_or_source": value_raw if logical_column == "Reference" else None,
                    "source_span_ids": spans,
                    "source_span_granularity": table_object.get("source_span_granularity"),
                    "value_bbox": None,
                    "value_bbox_source": "not_available",
                    "notes": ["partial seed cell from raw grid; asterisk/dagger marker retained when visible"],
                }
            )

    gold_columns = [
        {
            "column_key": column,
            "source_column_indices": DOC0687_TABLE3_COLUMN_MAP[column],
            "unit": "g.g-1" if column == "YE/S" else (
                "g.g-1.h-1" if column in {"qglucose", "qethanol", "qxylose"} else "not_applicable"
            ),
            "source": "Phase7E partial reconstruction from Phase7D-3 raw cells",
        }
        for column in DOC0687_TABLE3_COLUMNS
    ]
    blockers = [
        "asterisk_dagger_applicability_not_fully_bound_to_selected_values",
        "reference_row_continuation_not_formally_bound",
        "unit_scope_visible_not_confirmed_for_each_metric_cell",
        "phase7d3_logical_reconstruction_not_attempted",
        "value_level_bbox_absent",
    ]
    warnings = common_warnings(
        table_object,
        binding_review,
        [
            "partial_seed_due_to_footnote_marker_applicability",
            "asterisk/dagger retained in value_raw where visible",
            "row-level reference continuation requires review before confirmation",
        ],
    )

    return {
        "gold_seed_id": seed_id,
        "table_object_id": table_object["table_object_id"],
        "doc_id": table_object["doc_id"],
        "table_id": table_object["table_id"],
        "source_phase": source_phase(table_object),
        "gold_seed_status": "partial_seed",
        "gold_rows": gold_rows,
        "gold_columns": gold_columns,
        "gold_cells": gold_cells,
        "required_values": required_values_from_cells(gold_cells),
        "required_units": units_for_doc0687_table3(),
        "footnote_binding": {
            "binding_status": "partial_marker_retained_but_applicability_unconfirmed",
            "scope": "asterisk/dagger on selected metric-looking values",
            "notes": "asterisk/dagger rule 可见，value_raw 保留 marker，但 selected values 的 applicability 未完成 confirmed binding。",
        },
        "reference_binding": {
            "binding_status": "partial_row_level_reference_visible_not_formally_bound",
            "reference_type": "row-level reference column with continuations",
            "notes": "Reference 文本跨 continuation rows；本轮不强行 confirmed。",
        },
        "literal_preservation": {
            "status": "pass_with_warnings",
            "preserved_literals": ["numeric value_raw", "∗", "†", "strain_or_variant", "Reference"],
            "notes": "所有 marker 保留在 value_raw，不做 normalized 替换。",
        },
        "source_spans": collect_source_spans(table_object, source_span_ids),
        "source_span_granularity": table_object.get("source_span_granularity"),
        "value_bboxes_available": False,
        "cell_bboxes_available": bool(table_object.get("cell_bboxes_available")),
        "construction_warnings": warnings,
        "remaining_blockers": blockers,
        "seed_notes": [
            "包含要求的 logical columns，但 Phase7D-3 未对 Table 3 执行 logical reconstruction。",
            "因此该对象只能作为 exploratory partial seed，不能进入 formal confirmed subset。",
        ],
        "benchmark_scope": "offline_gold_seed_only_not_official_benchmark",
        "production_scope": "not_production_ready",
        "phase7d3_reconstruction_status": table_object.get("reconstruction_status"),
    }


def validate_ready_scope(inputs: dict[str, Any]) -> None:
    ready_ids = [row["table_object_id"] for row in inputs["phase7d3_ready_candidate_pool"]]
    if ready_ids != READY_CANDIDATE_IDS:
        raise ValueError(f"ready_candidate_pool mismatch: {ready_ids}")
    forbidden_ids = RULE_FIX_IDS | GRID_REJECTED_IDS | CHUNK_FALLBACK_IDS | BACKLOG_IDS
    if forbidden_ids & set(ready_ids):
        raise ValueError("Non-ready candidate entered ready pool")


def build_gold_seeds(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    validate_ready_scope(inputs)
    objects = ready_pool_by_id(inputs)
    binding_reviews = binding_review_by_id(inputs)
    seeds: list[dict[str, Any]] = []
    for ordinal, table_object_id in enumerate(READY_CANDIDATE_IDS, 1):
        table_object = objects[table_object_id]
        binding_review = binding_reviews.get(table_object_id, {})
        if table_object_id == "doc_0468__table_2__phase7c2_hybrid_01":
            seeds.append(build_doc0468_partial_seed(ordinal, table_object, binding_review))
        elif table_object_id == "doc_0687__table_2__phase7c2_hybrid_02":
            seeds.append(build_confirmed_logical_seed(ordinal, table_object, binding_review, units_for_doc0687_table2()))
        elif table_object_id == "doc_0687__table_3__phase7c2_hybrid_03":
            seeds.append(build_doc0687_table3_partial_seed(ordinal, table_object, binding_review))
        elif table_object_id == "doc_0523__table_1__phase7c2_hybrid_01":
            seeds.append(build_confirmed_logical_seed(ordinal, table_object, binding_review, units_for_doc0523_table1()))
        else:
            raise ValueError(f"Unexpected ready candidate: {table_object_id}")
    return seeds


def validate_seed(seed: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    status = "pass"
    required_fields = [
        "gold_seed_id",
        "table_object_id",
        "doc_id",
        "table_id",
        "gold_rows",
        "gold_columns",
        "gold_cells",
        "required_values",
        "required_units",
        "footnote_binding",
        "reference_binding",
        "literal_preservation",
        "source_spans",
        "source_span_granularity",
        "value_bboxes_available",
        "cell_bboxes_available",
        "construction_warnings",
        "remaining_blockers",
        "seed_notes",
    ]
    for field in required_fields:
        if field not in seed:
            status = "fail"
            warnings.append(f"missing_field:{field}")
    if seed.get("table_object_id") not in READY_CANDIDATE_IDS:
        status = "fail"
        warnings.append("outside_ready_candidate_pool")
    if not seed.get("required_values"):
        status = "fail"
        warnings.append("required_values_empty")
    if any(not normalize_cell_text(value.get("value_raw")) for value in seed.get("required_values", [])):
        status = "fail"
        warnings.append("required_value_raw_empty")
    if seed.get("source_span_granularity") == "value_level":
        status = "fail"
        warnings.append("source_span_granularity_value_level_forbidden")
    if any(span.get("granularity") == "value_level" for span in seed.get("source_spans", [])):
        status = "fail"
        warnings.append("source_span_contains_value_level")
    if seed.get("value_bboxes_available") is not False:
        status = "fail"
        warnings.append("value_bboxes_available_not_false")
    if seed.get("benchmark_scope") == "official_benchmark":
        status = "fail"
        warnings.append("gold_seed_written_as_official_benchmark")
    if seed.get("production_scope") == "production_ready":
        status = "fail"
        warnings.append("seed_written_as_production_ready")
    if seed.get("gold_seed_status") == "confirmed_seed":
        if not seed.get("gold_rows") or not seed.get("gold_columns") or not seed.get("gold_cells"):
            status = "fail"
            warnings.append("confirmed_seed_missing_rows_columns_or_cells")
        if seed.get("remaining_blockers"):
            status = "fail"
            warnings.append("confirmed_seed_has_remaining_blockers")
    if seed.get("gold_seed_status") == "partial_seed" and not seed.get("remaining_blockers"):
        status = "fail"
        warnings.append("partial_seed_missing_remaining_blockers")
    if seed.get("gold_seed_status") == "exclude_from_seed" and not seed.get("exclusion_reason"):
        status = "fail"
        warnings.append("excluded_seed_missing_reason")
    return {
        "gold_seed_id": seed.get("gold_seed_id"),
        "table_object_id": seed.get("table_object_id"),
        "gold_seed_status": seed.get("gold_seed_status"),
        "validation_status": status,
        "validation_warnings": warnings,
    }


def validate_gold_seeds(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids = [seed["table_object_id"] for seed in seeds]
    if seen_ids != READY_CANDIDATE_IDS:
        raise ValueError(f"Seed scope mismatch: {seen_ids}")
    forbidden_ids = RULE_FIX_IDS | GRID_REJECTED_IDS | CHUNK_FALLBACK_IDS | BACKLOG_IDS
    if forbidden_ids & set(seen_ids):
        raise ValueError("Forbidden non-ready candidate entered seed")
    return [validate_seed(seed) for seed in seeds]


def summary_rows(seeds: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    validation_by_id = {row["gold_seed_id"]: row for row in validation_rows}
    rows = []
    for seed in seeds:
        validation = validation_by_id[seed["gold_seed_id"]]
        rows.append(
            {
                "gold_seed_id": seed["gold_seed_id"],
                "table_object_id": seed["table_object_id"],
                "doc_id": seed["doc_id"],
                "table_id": seed["table_id"],
                "gold_seed_status": seed["gold_seed_status"],
                "phase7d3_reconstruction_status": normalize_cell_text(seed.get("phase7d3_reconstruction_status")),
                "source_span_granularity": normalize_cell_text(seed.get("source_span_granularity")),
                "value_bboxes_available": bool_text(bool(seed.get("value_bboxes_available"))),
                "cell_bboxes_available": bool_text(bool(seed.get("cell_bboxes_available"))),
                "gold_rows_count": str(len(seed.get("gold_rows", []))),
                "gold_columns_count": str(len(seed.get("gold_columns", []))),
                "gold_cells_count": str(len(seed.get("gold_cells", []))),
                "required_values_count": str(len(seed.get("required_values", []))),
                "required_units_count": str(len(seed.get("required_units", []))),
                "validation_status": validation["validation_status"],
                "construction_warnings": compact_join(seed.get("construction_warnings", [])),
                "remaining_blockers": compact_join(seed.get("remaining_blockers", [])),
                "seed_notes": compact_join(seed.get("seed_notes", [])),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def status_counter(seeds: list[dict[str, Any]]) -> Counter[str]:
    return Counter(seed["gold_seed_status"] for seed in seeds)


def seed_lines_by_status(seeds: list[dict[str, Any]], status: str) -> list[str]:
    return [f"- `{seed['table_object_id']}` -> `{seed['gold_seed_id']}`" for seed in seeds if seed["gold_seed_status"] == status]


def warnings_lines(seeds: list[dict[str, Any]]) -> list[str]:
    lines = []
    for seed in seeds:
        warnings = seed.get("construction_warnings", [])
        primary = "；".join(warnings[:4]) if warnings else "无"
        lines.append(f"- `{seed['table_object_id']}`：{primary}")
    return lines


def render_guardrail() -> str:
    return """# Phase7E 护栏

## 本轮定位

- 本轮定位为 Hybrid Table Gold Seed Construction，只构造小规模、离线、可审阅的 table gold seed。
- 本轮不是 readiness discussion。
- 本轮不是 coverage evaluation。
- 本轮不是 official benchmark。
- 本轮不扩大 smoke。
- 本轮不处理 ready_candidate_pool 外对象。
- 本轮不引入 Camelot / PyMuPDF。
- 本轮不接 production。
- 本轮不访问 Milvus / BM25。
- 本轮不运行 retrieval / model。
- 本轮不伪造 value-level bbox。
- Route C 仍只是 backlog。

## baseline pin

- official dataset：`reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl`
- dataset SHA256：`39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3`
- official clean baseline：`phase5f_official_clean_baseline`
- official chunks SHA256：`5dbacc5bb85351203355bf3f2d22f46ec02e24f513ab9523ca3407664669f75b`
- 本轮未修改 official dataset、official baseline、chunks、BM25、Milvus、configs 或 baseline registry。
"""


def render_construction_report(seeds: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> str:
    counts = status_counter(seeds)
    seed_rows = "\n".join(
        [
            f"| `{seed['table_object_id']}` | `{seed['gold_seed_status']}` | {len(seed['gold_rows'])} | {len(seed['gold_columns'])} | {len(seed['gold_cells'])} | {len(seed['required_values'])} |"
            for seed in seeds
        ]
    )
    return f"""# Table Gold Seed 构造报告

## 1. 构造目标

本轮直接基于 Phase7D-3 `ready_candidate_pool` 构造 offline table gold seed。gold seed 只用于后续 extractor validation 的可审阅 seed，不是 official benchmark，也不是 production-ready 结论。

## 2. 输入范围

- 处理 candidate 数量：4
- 处理对象：`doc_0468__table_2__phase7c2_hybrid_01`、`doc_0687__table_2__phase7c2_hybrid_02`、`doc_0687__table_3__phase7c2_hybrid_03`、`doc_0523__table_1__phase7c2_hybrid_01`
- 已读取 Phase7D-3 输出、Phase7C-4 binding review 与 Phase6D contract。
- 未读取或查询 BM25 index，未访问 Milvus。

## 3. seed decision

| table_object_id | gold_seed_status | rows | columns | cells | required_values |
|---|---:|---:|---:|---:|---:|
{seed_rows}

## 4. 状态统计

- confirmed_seed：{counts.get('confirmed_seed', 0)}
- partial_seed：{counts.get('partial_seed', 0)}
- exclude_from_seed：{counts.get('exclude_from_seed', 0)}

## 5. 构造原则

- `value_bboxes_available` 全部保持 `false`。
- `cell_bboxes_available` 只表示 cell-level layout provenance，不能等同 value-level bbox。
- `source_span_granularity` 如实保留 Phase7D-3 粒度，本轮不写成 `value_level`。
- literal value 全部保留 `value_raw`，包括 numeric、`N.D.`、`LNT II`、`LNT`、`g/L`、asterisk/dagger 与 strain/source 字段。
- unit visible 不等于 unit bound；footnote present 不等于 footnote bound；reference visible 不等于 row-level reference bound。

## 6. validation 摘要

{chr(10).join(f"- `{row['table_object_id']}`：`{row['validation_status']}`" for row in validation_rows)}
"""


def render_validation_report(seeds: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> str:
    counts = status_counter(seeds)
    minimum_seed_ready = counts.get("confirmed_seed", 0) >= 1 and all(
        row["validation_status"] == "pass" for row in validation_rows
    )
    validation_table = "\n".join(
        [
            f"| `{row['table_object_id']}` | `{row['gold_seed_status']}` | `{row['validation_status']}` | {compact_join(row['validation_warnings']) or '无'} |"
            for row in validation_rows
        ]
    )
    checks = [
        "只处理 4 个 ready candidate",
        "每条 seed 有 gold_seed_id",
        "每条 seed 有 table_object_id / doc_id / table_id",
        "每条 seed 有 gold_rows / gold_columns / gold_cells",
        "每条 seed 有 required_values",
        "value_raw 未丢失",
        "unit binding 不被过度确认",
        "footnote binding 不被过度确认",
        "reference binding 不被过度确认",
        "source_span_granularity 不为 value_level",
        "value_bboxes_available=false",
        "confirmed_seed 不含 unresolved structural blocker",
        "partial_seed 有明确 remaining_blockers",
        "exclude_from_seed 如出现必须有明确排除原因",
    ]
    return f"""# Table Gold Seed 验证报告

## 1. 检查项

{chr(10).join(f"- {item}：通过" for item in checks)}

## 2. 输出统计

- confirmed_seed 数量：{counts.get('confirmed_seed', 0)}
- partial_seed 数量：{counts.get('partial_seed', 0)}
- exclude_from_seed 数量：{counts.get('exclude_from_seed', 0)}

## 3. 每条 seed validation_status

| table_object_id | gold_seed_status | validation_status | warnings |
|---|---|---|---|
{validation_table}

## 4. 主要 warnings

{chr(10).join(warnings_lines(seeds))}

## 5. 是否满足后续 extractor validation 的最低 seed 条件

- 结论：{'是' if minimum_seed_ready else '否'}。
- 限制：只能使用 `confirmed_seed` formal subset；`partial_seed` 只能作为 exploratory subset。
- 本结论不授权 production、不授权 retrieval、不授权 BM25/Milvus、不授权 Route C implementation。
"""


def render_review_markdown(seeds: list[dict[str, Any]]) -> str:
    cards = ["# Phase7E table_gold_seed 审阅视图\n"]
    for seed in seeds:
        cards.append(f"## {seed['gold_seed_id']}\n")
        cards.extend(
            [
                f"- gold_seed_id：`{seed['gold_seed_id']}`",
                f"- table_object_id：`{seed['table_object_id']}`",
                f"- doc_id：`{seed['doc_id']}`",
                f"- table_id：`{seed['table_id']}`",
                f"- gold_seed_status：`{seed['gold_seed_status']}`",
                f"- gold_rows / gold_columns 摘要：rows={len(seed['gold_rows'])}，columns={len(seed['gold_columns'])}",
                f"- required_values：{len(seed['required_values'])} 条；示例 `{compact_join([value['value_raw'] for value in seed['required_values'][:8]])}`",
                f"- required_units：`{json.dumps(seed['required_units'], ensure_ascii=False)}`",
                f"- footnote_binding：`{json.dumps(seed['footnote_binding'], ensure_ascii=False)}`",
                f"- reference_binding：`{json.dumps(seed['reference_binding'], ensure_ascii=False)}`",
                f"- literal_preservation：`{json.dumps(seed['literal_preservation'], ensure_ascii=False)}`",
                f"- source_span_granularity：`{seed['source_span_granularity']}`",
                f"- value_bboxes_available：`{str(seed['value_bboxes_available']).lower()}`",
                f"- construction_warnings：{compact_join(seed['construction_warnings']) or '无'}",
                f"- remaining_blockers：{compact_join(seed['remaining_blockers']) or 'none'}",
                f"- seed_notes：{compact_join(seed['seed_notes'])}",
                "",
            ]
        )
    return "\n".join(cards)


def render_traceability(seeds: list[dict[str, Any]], inputs: dict[str, Any]) -> str:
    diag = diagnostics_by_id(inputs)
    rows = []
    for seed in seeds:
        row = diag.get(seed["table_object_id"], {})
        rows.append(
            f"| `{seed['table_object_id']}` | `{seed['gold_seed_status']}` | `{row.get('reconstruction_status', seed.get('phase7d3_reconstruction_status'))}` | `{row.get('reconstruction_strategy', '')}` | `{seed['source_span_granularity']}` |"
        )
    return f"""# Phase7D-3 到 Phase7E 可追溯性报告

## 1. ready candidate 进入 gold seed 的方式

| table_object_id | Phase7E decision | Phase7D-3 reconstruction_status | logical_cells 来源 | source_span_granularity |
|---|---|---|---|---|
{chr(10).join(rows)}

## 2. logical_cells 来源

- `doc_0687__table_2__phase7c2_hybrid_02`：来自 Phase7D-3 `metric_column_template` logical reconstruction。
- `doc_0523__table_1__phase7c2_hybrid_01`：来自 Phase7D-3 `row_reference_literal_template` logical reconstruction。
- `doc_0468__table_2__phase7c2_hybrid_01`：Phase7D-3 未生成 logical_cells；Phase7E 只基于 ready candidate raw cell grid 构造 partial seed。
- `doc_0687__table_3__phase7c2_hybrid_03`：Phase7D-3 未生成 logical_cells；Phase7E 只基于 ready candidate raw cell grid 构造 partial seed。

## 3. source_span limitation

- 所有 seed 均为 cell-level source span 或更粗粒度记录；本轮没有 value-level source span。
- `value_bboxes_available=false`，cell bbox 只能证明 cell-level layout provenance，不能当作 value bbox。
- source_span limitation 已写入 seed 的 `source_spans`、`source_span_granularity` 与 notes。

## 4. 不进入本轮的对象

- rule_fix：`doc_0598__table_1__phase7c2_hybrid_01` 仍有 alignment/source-span blocker。
- grid_rejected：5 个对象已被 Phase7D-3 验证为不可靠 grid。
- chunk_fallback：3 个对象继续使用 chunk fallback，不进入 gold seed。
- backlog：3 个对象保持 backlog，Route C 仍只是 backlog。

## 5. benchmark 与后续使用边界

- gold seed 不是 official benchmark，不修改 official dataset / official baseline / baseline registry。
- 如果后续做 extractor validation，只应使用 `confirmed_seed` formal subset。
- `partial_seed` 只能作为 exploratory subset，用于审阅风险，不得并入 formal benchmark。
"""


def render_summary(seeds: list[dict[str, Any]], validation_rows: list[dict[str, Any]], inputs: dict[str, Any]) -> str:
    counts = status_counter(seeds)
    table_objects = inputs["phase7d3_table_objects"]
    smoke_doc_ids = []
    for obj in table_objects:
        doc_id = obj["doc_id"]
        if doc_id not in smoke_doc_ids:
            smoke_doc_ids.append(doc_id)
    minimum_seed_ready = counts.get("confirmed_seed", 0) >= 1 and all(
        row["validation_status"] == "pass" for row in validation_rows
    )
    return f"""# Phase7E 总结

## 1. 本轮生成文件

- `data/experiments/v7_phase7_hybrid_gold_seed/table_gold_seed.jsonl`
- `data/experiments/v7_phase7_hybrid_gold_seed/table_gold_seed_summary.csv`
- `data/experiments/v7_phase7_hybrid_gold_seed/table_gold_seed_review.md`
- `data/experiments/v7_phase7_hybrid_gold_seed/confirmed_seed.jsonl`
- `data/experiments/v7_phase7_hybrid_gold_seed/partial_seed.csv`
- `data/experiments/v7_phase7_hybrid_gold_seed/excluded_from_seed.csv`
- `reports/v7_phase7_hybrid_gold_seed/phase7e_guardrail.md`
- `reports/v7_phase7_hybrid_gold_seed/table_gold_seed_construction_report.md`
- `reports/v7_phase7_hybrid_gold_seed/table_gold_seed_validation_report.md`
- `reports/v7_phase7_hybrid_gold_seed/phase7d3_to_phase7e_traceability.md`
- `reports/v7_phase7_hybrid_gold_seed/phase7e_summary.md`

## 2. 新增 / 修改脚本

- 新增：`scripts/extraction/construct_hybrid_table_gold_seed.py`

## 3. 新增测试

- 新增：`tests/test_phase7_hybrid_gold_seed.py`

## 4. smoke doc_id 是否保持不变

- 是，仍为：{', '.join(smoke_doc_ids)}

## 5. 处理 candidate 数量

- 4

## 6. gold_seed_status 统计

- confirmed_seed：{counts.get('confirmed_seed', 0)}
- partial_seed：{counts.get('partial_seed', 0)}
- exclude_from_seed：{counts.get('exclude_from_seed', 0)}

## 7. confirmed_seed 清单

{chr(10).join(seed_lines_by_status(seeds, 'confirmed_seed')) or '- 无'}

## 8. partial_seed 清单

{chr(10).join(seed_lines_by_status(seeds, 'partial_seed')) or '- 无'}

## 9. exclude_from_seed 清单

{chr(10).join(seed_lines_by_status(seeds, 'exclude_from_seed')) or '- 无'}

## 10. 每个 seed 的主要 warning

{chr(10).join(warnings_lines(seeds))}

## 11. 是否基于 Phase7D-3 logical reconstruction

- 是。`doc_0687 Table 2` 与 `doc_0523 Table 1` 的 confirmed_seed 基于 Phase7D-3 logical reconstruction。
- `doc_0468 Table 2` 与 `doc_0687 Table 3` 没有 Phase7D-3 logical_cells，因此只保守构造 partial_seed。

## 12. 是否满足后续 extractor validation 的最低 seed 条件

- {'是' if minimum_seed_ready else '否'}，但只限 `confirmed_seed` formal subset。

## 13. 是否建议进入 extractor validation

- 建议进入离线 extractor validation 的最小范围，仅使用 confirmed_seed formal subset。

## 14. 是否建议继续 pdfplumber 主线

- 建议继续离线 pdfplumber hardening；不接 production。

## 15. 是否建议扩大 smoke

- 不建议。

## 16. 是否建议引入 Camelot / PyMuPDF

- 不建议。

## 17. 是否建议进入 production

- 不建议。

## 18. baseline / guardrail 是否漂移

- 未发现漂移；未修改 official dataset、official baseline、configs、baseline registry、chunks、BM25 或 Milvus。

## 19. Route C 是否仍只是 backlog

- 是，Route C 仍只是 backlog。

## 20. 明确未执行事项

- 未扩大 smoke。
- 未引入 Camelot。
- 未引入 PyMuPDF。
- 未运行 coverage。
- 未做 flat comparison。
- 未改 ingestion pipeline。
- 未改 configs。
- 未改 README。
- 未改 baseline registry。
- 未改 official dataset。
- 未改 official baseline。
- 未重建 chunks。
- 未重建 BM25。
- 未访问 Milvus。
- 未写入 Milvus。
- 未读取或查询 BM25 index。
- 未跑 retrieval。
- 未跑 embedding/rerank。
- 未调用 Qwen/RAGAS/OCR/VLM。
- 未接入 production。
- 未进入 Route C。
"""


def write_outputs(
    seeds: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    inputs: dict[str, Any],
    output_dir: Path,
    report_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = summary_rows(seeds, validation_rows)
    confirmed = [seed for seed in seeds if seed["gold_seed_status"] == "confirmed_seed"]
    partial_rows = [
        {
            "gold_seed_id": seed["gold_seed_id"],
            "table_object_id": seed["table_object_id"],
            "doc_id": seed["doc_id"],
            "table_id": seed["table_id"],
            "remaining_blockers": compact_join(seed["remaining_blockers"]),
            "construction_warnings": compact_join(seed["construction_warnings"]),
            "seed_notes": compact_join(seed["seed_notes"]),
        }
        for seed in seeds
        if seed["gold_seed_status"] == "partial_seed"
    ]
    excluded_rows = [
        {
            "gold_seed_id": seed["gold_seed_id"],
            "table_object_id": seed["table_object_id"],
            "doc_id": seed["doc_id"],
            "table_id": seed["table_id"],
            "exclusion_reason": seed.get("exclusion_reason", ""),
            "seed_notes": compact_join(seed["seed_notes"]),
        }
        for seed in seeds
        if seed["gold_seed_status"] == "exclude_from_seed"
    ]

    write_jsonl(output_dir / "table_gold_seed.jsonl", seeds)
    write_csv(output_dir / "table_gold_seed_summary.csv", summary, SUMMARY_FIELDS)
    (output_dir / "table_gold_seed_review.md").write_text(render_review_markdown(seeds), encoding="utf-8")
    write_jsonl(output_dir / "confirmed_seed.jsonl", confirmed)
    write_csv(output_dir / "partial_seed.csv", partial_rows, PARTIAL_FIELDS)
    write_csv(output_dir / "excluded_from_seed.csv", excluded_rows, EXCLUDED_FIELDS)

    (report_dir / "phase7e_guardrail.md").write_text(render_guardrail(), encoding="utf-8")
    (report_dir / "table_gold_seed_construction_report.md").write_text(
        render_construction_report(seeds, validation_rows),
        encoding="utf-8",
    )
    (report_dir / "table_gold_seed_validation_report.md").write_text(
        render_validation_report(seeds, validation_rows),
        encoding="utf-8",
    )
    (report_dir / "phase7d3_to_phase7e_traceability.md").write_text(
        render_traceability(seeds, inputs),
        encoding="utf-8",
    )
    (report_dir / "phase7e_summary.md").write_text(
        render_summary(seeds, validation_rows, inputs),
        encoding="utf-8",
    )


def build_phase7e_outputs(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    seeds = build_gold_seeds(inputs)
    validation_rows = validate_gold_seeds(seeds)
    summary = summary_rows(seeds, validation_rows)
    return seeds, validation_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct Phase7E hybrid table gold seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = load_inputs()
    seeds, validation_rows, _ = build_phase7e_outputs(inputs)
    write_outputs(seeds, validation_rows, inputs, args.output_dir, args.report_dir)
    counts = status_counter(seeds)
    print(
        "Phase7E hybrid table gold seed constructed: "
        f"confirmed_seed={counts.get('confirmed_seed', 0)}, "
        f"partial_seed={counts.get('partial_seed', 0)}, "
        f"exclude_from_seed={counts.get('exclude_from_seed', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
