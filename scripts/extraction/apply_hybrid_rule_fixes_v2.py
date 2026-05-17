#!/usr/bin/env python3
"""Phase7D-2 hybrid extractor v2.1 rule-fix layer.

This script does not re-extract PDFs. It reads Phase7D v2 artifacts plus the
Phase7C-2/C-3/C-4 sidecars, applies deterministic structure diagnostics for the
three rule-fix cases, and materializes a v2.1 output set over the same 16 cases.
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

from scripts.extraction import align_chunk_pdfplumber_tables as align_gate
from scripts.extraction import run_hybrid_table_extractor_v2 as v2


SMOKE_DOC_IDS = list(v2.SMOKE_DOC_IDS)
READY_IDS = set(v2.READY_IDS)
RULE_FIX_IDS = set(v2.RULE_FIX_IDS)
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

PHASE7D_DATA_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2"
PHASE7D_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_extractor_v2"
PHASE7C4_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_binding_review"
PHASE7C4_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_binding_review"
PHASE7C2_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_pilot_v2"
PHASE7C3_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_gate_hardening"
PHASE6D_REPORT_DIR = ROOT / "reports/v7_phase6d_table_contract_refinement"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_rule_fix"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_hybrid_extractor_v2_rule_fix"

PHASE7D_REQUIRED_INPUTS = [
    PHASE7D_DATA_DIR / "table_objects.jsonl",
    PHASE7D_DATA_DIR / "table_objects_review.md",
    PHASE7D_DATA_DIR / "table_object_routing_summary.csv",
    PHASE7D_DATA_DIR / "ready_candidate_pool.jsonl",
    PHASE7D_DATA_DIR / "rule_fix_cases.csv",
    PHASE7D_DATA_DIR / "grid_rejected_cases.csv",
    PHASE7D_DATA_DIR / "chunk_fallback_cases.csv",
    PHASE7D_DATA_DIR / "backlog_cases.csv",
    PHASE7D_REPORT_DIR / "extractor_v2_change_log.md",
    PHASE7D_REPORT_DIR / "table_object_v2_validation_report.md",
    PHASE7D_REPORT_DIR / "phase7c4_vs_phase7d_comparison.md",
    PHASE7D_REPORT_DIR / "phase7d_summary.md",
]

PHASE7C4_REQUIRED_INPUTS = [
    PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_binding_review_summary.csv",
    PHASE7C4_DATA_DIR / "hybrid_candidates_ready_for_gold.jsonl",
    PHASE7C4_DATA_DIR / "hybrid_candidates_needing_rule_fix.csv",
    PHASE7C4_REPORT_DIR / "phase7c_4_summary.md",
    PHASE7C4_REPORT_DIR / "gold_readiness_decision.md",
]

PHASE7C2_C3_REQUIRED_INPUTS = [
    PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl",
    PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv",
    PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl",
]

CURRENT_EXTRACTOR_SCRIPTS = [
    ROOT / "scripts/extraction/extract_tables_pdfplumber_v1.py",
    ROOT / "scripts/extraction/align_chunk_pdfplumber_tables.py",
    ROOT / "scripts/extraction/build_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/validate_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/render_hybrid_table_objects_markdown.py",
    ROOT / "scripts/extraction/run_hybrid_table_extractor_v2.py",
]

OPTIONAL_EXISTING_TESTS = [
    ROOT / "tests/test_phase7_hybrid_alignment_gate.py",
    ROOT / "tests/test_phase7_source_review_gate.py",
    ROOT / "tests/test_phase7_hybrid_binding_review.py",
    ROOT / "tests/test_phase7_hybrid_extractor_v2.py",
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

DIAGNOSTIC_FIELDS = [
    "table_object_id",
    "doc_id",
    "table_id",
    "original_routing_status",
    "primary_rule_fix_reason",
    "split_cell_detected",
    "merged_cell_detected",
    "row_continuation_detected",
    "column_alignment_inconsistent",
    "metric_level_cell_gap_detected",
    "numeric_column_order_uncertain",
    "missing_metric_cell_warning",
    "unit_binding_warning",
    "footnote_binding_warning",
    "reference_binding_warning",
    "literal_preservation_warning",
    "rule_fix_attempted",
    "rule_fix_applied",
    "rule_fix_blockers",
    "candidate_upgrade_eligible",
    "notes",
]

DELTA_FIELDS = [
    "table_object_id",
    "phase7d_routing_status",
    "phase7d2_routing_status",
    "changed",
    "change_type",
    "fix_applied",
    "remaining_blockers",
    "upgrade_justification",
    "downgrade_reason",
    "notes",
]

EXTRA_SUMMARY_FIELDS = [
    "phase7d_routing_status",
    "phase7d2_routing_status",
    "rule_fix_attempted",
    "rule_fix_applied",
    "remaining_blockers",
    "candidate_upgrade_eligible",
    "split_cell_warning",
    "merged_cell_warning",
    "row_continuation_warning",
    "metric_level_cell_gap",
    "numeric_column_order_uncertain",
    "missing_metric_cell_warning",
    "metric_column_group_uncertain",
    "literal_preservation_status",
]
ROUTING_SUMMARY_FIELDS = list(dict.fromkeys(v2.ROUTING_SUMMARY_FIELDS + EXTRA_SUMMARY_FIELDS))

STRUCTURAL_WARNING_KEYS = [
    "split_cell_warning",
    "merged_cell_warning",
    "row_continuation_warning",
    "column_alignment_inconsistent",
    "cell_grid_needs_rule_fix",
]
METRIC_WARNING_KEYS = [
    "metric_level_cell_gap",
    "numeric_column_order_uncertain",
    "missing_metric_cell_warning",
    "metric_column_group_uncertain",
]
BINDING_WARNING_KEYS = [
    "unit_visible_not_bound",
    "unit_binding_uncertain",
    "footnote_present_not_bound",
    "footnote_binding_uncertain",
    "reference_visible_not_bound",
    "reference_binding_uncertain",
    "internal_reference_column",
    "external_citation_not_supported",
    "literal_value_requires_preservation",
]

FORBIDDEN_OUTPUT_KEYS = set(v2.FORBIDDEN_OUTPUT_KEYS) | {
    "confirmed_gold",
    "production_ready",
}

NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
DECIMAL_LEFT_RE = re.compile(r"^[+-]?\d+$")
DECIMAL_RIGHT_RE = re.compile(r"^\.\d+(?:[∗*†A-Za-z]*)?$")
DNA_MERGED_RE = re.compile(r"^[A-Za-z0-9_.-]{2,18}\s+[ACGTURYKMSWBDHVN]{18,}", re.IGNORECASE)
LITERAL_RE = re.compile(r"\b(?:N\.?D\.?|NT|NC|not detected|not tested|not calculable)\b", re.IGNORECASE)

KNOWN_JOINED_TERMS = {
    "location",
    "reference",
    "xylose",
    "qarabinose",
    "pentose",
    "fermentation",
    "strategy",
    "modifications",
    "keygenetic",
    "primersequence",
    "primername",
    "mediumcultureconditions",
    "yes",
    "qxylose",
}
METRIC_TERMS = {
    "ye/s",
    "yes",
    "qethanol",
    "qxylose",
    "qarabinose",
    "qglucose",
    "titer",
    "titre",
    "lnt",
    "lntii",
}
EXPECTED_METRIC_ORDER = ["yes", "qethanol", "qxylose", "qarabinose"]


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


def compact_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_space(value).lower())


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def semicolon(values: list[Any]) -> str:
    cleaned = [normalize_space(value) for value in values if normalize_space(value)]
    return ";".join(cleaned) if cleaned else "none"


def add_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for item in additions:
        if item and item not in result:
            result.append(item)
    return result


def split_semicolon(value: Any) -> list[str]:
    text = normalize_space(value)
    if not text or text == "none":
        return []
    return [item for item in text.split(";") if item]


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


def read_input_inventory(required_paths: list[Path], optional_paths: list[Path] | None = None) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    missing_required: list[str] = []
    optional_paths = optional_paths or []
    for path in required_paths + optional_paths:
        required = path in required_paths
        if not path.exists():
            if required:
                missing_required.append(rel(path))
            inventory.append(
                {
                    "path": rel(path),
                    "exists": False,
                    "required": required,
                    "line_count": 0,
                    "record_count": 0,
                    "bytes": 0,
                }
            )
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
                "required": required,
                "line_count": line_count,
                "record_count": record_count,
                "bytes": path.stat().st_size,
            }
        )
    if missing_required:
        raise FileNotFoundError("缺少 Phase7D-2 必读输入：" + "; ".join(missing_required))
    return inventory


def strip_forbidden_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_forbidden_keys(item) for key, item in value.items() if key not in FORBIDDEN_OUTPUT_KEYS}
    if isinstance(value, list):
        return [strip_forbidden_keys(item) for item in value]
    return value


def md_escape(value: Any) -> str:
    return normalize_space(value).replace("|", "\\|")


def load_inputs() -> dict[str, Any]:
    return {
        "phase7d_objects": load_jsonl(PHASE7D_DATA_DIR / "table_objects.jsonl"),
        "phase7d_summary_rows": load_csv(PHASE7D_DATA_DIR / "table_object_routing_summary.csv"),
        "phase7d_rule_fix_rows": load_csv(PHASE7D_DATA_DIR / "rule_fix_cases.csv"),
        "phase7d_ready_rows": load_jsonl(PHASE7D_DATA_DIR / "ready_candidate_pool.jsonl"),
        "phase7d_grid_rows": load_csv(PHASE7D_DATA_DIR / "grid_rejected_cases.csv"),
        "phase7d_fallback_rows": load_csv(PHASE7D_DATA_DIR / "chunk_fallback_cases.csv"),
        "phase7d_backlog_rows": load_csv(PHASE7D_DATA_DIR / "backlog_cases.csv"),
        "binding_review_rows": load_jsonl(PHASE7C4_DATA_DIR / "hybrid_binding_review.jsonl"),
        "binding_summary_rows": load_csv(PHASE7C4_DATA_DIR / "hybrid_binding_review_summary.csv"),
        "c4_ready_rows": load_jsonl(PHASE7C4_DATA_DIR / "hybrid_candidates_ready_for_gold.jsonl"),
        "c4_rule_fix_rows": load_csv(PHASE7C4_DATA_DIR / "hybrid_candidates_needing_rule_fix.csv"),
        "c2_hybrid_objects": load_jsonl(PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl"),
        "c2_raw_tables": load_jsonl(PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl"),
        "c2_alignment_rows": load_csv(PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv"),
        "c3_decision_rows": load_jsonl(PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl"),
        "c3_gated_objects": load_jsonl(PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl"),
    }


def validate_inputs(inputs: dict[str, Any]) -> None:
    object_ids = {row["table_object_id"] for row in inputs["phase7d_objects"]}
    summary_ids = {row["table_object_id"] for row in inputs["phase7d_summary_rows"]}
    c2_ids = {row["table_object_id"] for row in inputs["c2_hybrid_objects"]}
    c3_decision_ids = {row["hybrid_table_object_id"] for row in inputs["c3_decision_rows"]}
    binding_ids = {row["hybrid_table_object_id"] for row in inputs["binding_review_rows"]}
    doc_ids = {row.get("doc_id") for row in inputs["phase7d_objects"]}
    errors: list[str] = []
    if len(object_ids) != 16 or len(summary_ids) != 16:
        errors.append(f"Phase7D v2 输入应覆盖 16 个对象，实际 objects={len(object_ids)} summary={len(summary_ids)}")
    if object_ids != summary_ids:
        errors.append("Phase7D table_objects 与 routing summary 覆盖不一致")
    if object_ids != c2_ids or object_ids != c3_decision_ids:
        errors.append("Phase7D 与 Phase7C-2/C-3 case 覆盖不一致")
    if binding_ids != READY_IDS | RULE_FIX_IDS:
        errors.append("Phase7C-4 binding review 应只覆盖 2 ready + 3 rule_fix")
    if {row["table_object_id"] for row in inputs["phase7d_grid_rows"]} != GRID_REJECTED_IDS:
        errors.append("Phase7D grid_rejected 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d_fallback_rows"]} != CHUNK_FALLBACK_IDS:
        errors.append("Phase7D chunk_fallback 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d_backlog_rows"]} != BACKLOG_IDS:
        errors.append("Phase7D backlog 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d_rule_fix_rows"]} != RULE_FIX_IDS:
        errors.append("Phase7D rule_fix 清单漂移")
    if {row["table_object_id"] for row in inputs["phase7d_ready_rows"]} != READY_IDS:
        errors.append("Phase7D ready pool 漂移")
    if doc_ids != set(SMOKE_DOC_IDS):
        errors.append(f"smoke doc_id 漂移：{sorted(doc_ids)}")
    if errors:
        raise ValueError("Phase7D-2 输入校验失败：" + "; ".join(errors))


def raw_table_for_object(
    obj: dict[str, Any],
    c2_object_by_id: dict[str, dict[str, Any]],
    raw_by_pdf_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    meta = obj.get("hybrid_metadata") or {}
    pdf_id = meta.get("pdfplumber_table_id")
    if not pdf_id:
        c2_obj = c2_object_by_id.get(obj.get("table_object_id", "")) or {}
        pdf_id = (c2_obj.get("hybrid_metadata") or {}).get("pdfplumber_table_id")
    return raw_by_pdf_id.get(pdf_id or "")


def evidence_append(evidence: list[str], message: str, limit: int = 6) -> None:
    if len(evidence) < limit:
        evidence.append(message)


def non_empty_cells(row: list[Any]) -> list[str]:
    return [normalize_space(cell) for cell in row if normalize_space(cell)]


def joined_compact(cells: list[str]) -> str:
    return compact_token("".join(cells))


def is_decimal_split(left: str, right: str) -> bool:
    return bool(DECIMAL_LEFT_RE.fullmatch(left) and DECIMAL_RIGHT_RE.fullmatch(right))


def detect_split_cells(raw_table: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not raw_table:
        return False, ["no_raw_pdfplumber_table"]
    rows = raw_table.get("rows") or []
    evidence: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        values = [normalize_space(cell) for cell in row]
        for col_index in range(len(values) - 1):
            left = values[col_index]
            right = values[col_index + 1]
            if not left or not right:
                continue
            pair = joined_compact([left, right])
            if is_decimal_split(left, right) or pair in KNOWN_JOINED_TERMS:
                evidence_append(evidence, f"r{row_index}c{col_index + 1}-c{col_index + 2}:{left}|{right}")
        for col_index in range(len(values) - 2):
            part = values[col_index : col_index + 3]
            if not all(part):
                continue
            triple = joined_compact(part)
            if triple in KNOWN_JOINED_TERMS:
                evidence_append(evidence, f"r{row_index}c{col_index + 1}-c{col_index + 3}:{'|'.join(part)}")
    return bool(evidence), evidence or ["no_split_cell_signal"]


def has_metric_context(text: str) -> bool:
    compact = compact_token(text)
    lowered = normalize_space(text).lower()
    return any(term in compact or term in lowered for term in METRIC_TERMS)


def detect_merged_cells(raw_table: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not raw_table:
        return False, ["no_raw_pdfplumber_table"]
    evidence: list[str] = []
    for row_index, row in enumerate(raw_table.get("rows") or [], start=1):
        values = [normalize_space(cell) for cell in row]
        for col_index, cell in enumerate(values, start=1):
            if not cell:
                continue
            compact = compact_token(cell)
            lowered = cell.lower()
            if "primername" in compact and "primersequence" in compact:
                evidence_append(evidence, f"r{row_index}c{col_index}:primer header merged")
            elif DNA_MERGED_RE.search(cell):
                evidence_append(evidence, f"r{row_index}c{col_index}:primer name and sequence merged")
            elif has_metric_context(cell) and (
                bool(NUMERIC_RE.search(cell)) or bool(LITERAL_RE.search(cell))
            ) and len(non_empty_cells(values)) >= 2:
                evidence_append(evidence, f"r{row_index}c{col_index}:metric/context/value share one cell")
            elif any(key in compact for key in ["mediumcultureconditions", "hoststraincharacteristics"]):
                evidence_append(evidence, f"r{row_index}c{col_index}:multiple logical labels share one cell")
            elif len(cell) >= 34 and bool(NUMERIC_RE.search(cell)) and re.search(r"[A-Za-z]", cell):
                evidence_append(evidence, f"r{row_index}c{col_index}:text and numeric value merged")
    return bool(evidence), evidence or ["no_merged_cell_signal"]


def detect_row_continuation(raw_table: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not raw_table:
        return False, ["no_raw_pdfplumber_table"]
    evidence: list[str] = []
    previous_first = ""
    for row_index, row in enumerate(raw_table.get("rows") or [], start=1):
        values = [normalize_space(cell) for cell in row]
        non_empty = non_empty_cells(values)
        if not non_empty:
            continue
        first = values[0] if values else ""
        if row_index > 1 and not first and any(values[1:]):
            evidence_append(evidence, f"r{row_index}:empty row label continues previous logical row")
        elif (
            row_index > 1
            and first
            and previous_first
            and len(first) <= 14
            and len(previous_first) >= 18
            and len(non_empty) <= max(2, len(values) // 2)
        ):
            evidence_append(evidence, f"r{row_index}:{first} likely continues previous row label")
        previous_first = first or previous_first
    return bool(evidence), evidence or ["no_row_continuation_signal"]


def sentence_like_row(row: list[Any]) -> bool:
    text = normalize_space(" ".join(str(cell) for cell in row if normalize_space(cell)))
    if len(text) < 70:
        return False
    return bool(re.search(r"[.;:,)]", text)) or len(non_empty_cells(row)) <= 2


def detect_column_alignment_inconsistent(
    raw_table: dict[str, Any] | None,
    split_detected: bool,
    merged_detected: bool,
    continuation_detected: bool,
) -> tuple[bool, list[str]]:
    if not raw_table:
        return False, ["no_raw_pdfplumber_table"]
    evidence: list[str] = []
    column_count = int(raw_table.get("column_count") or 0)
    empty_ratio = float(raw_table.get("empty_cell_ratio") or 0.0)
    rows = raw_table.get("rows") or []
    if column_count >= 12 and (split_detected or empty_ratio >= 0.45):
        evidence_append(evidence, f"column_count={column_count} with split/empty grid")
    if empty_ratio >= 0.45 and (split_detected or continuation_detected):
        evidence_append(evidence, f"empty_cell_ratio={empty_ratio:.4f} with continuation/split")
    if split_detected and merged_detected:
        evidence_append(evidence, "split and merged cell signals coexist")
    body_like_count = sum(1 for row in rows if sentence_like_row(row))
    if body_like_count >= 5:
        evidence_append(evidence, f"body_like_rows_inside_grid={body_like_count}")
    flattened = normalize_space(" ".join(" ".join(str(cell) for cell in row) for row in rows)).lower()
    if "doi.org" in flattened or "journal of" in flattened or "figure" in flattened:
        evidence_append(evidence, "page body/header/tail text appears inside cell grid")
    return bool(evidence), evidence or ["no_column_alignment_inconsistency_signal"]


def header_compact_text(raw_table: dict[str, Any] | None, max_rows: int = 8) -> str:
    if not raw_table:
        return ""
    header_rows = raw_table.get("rows") or []
    text = " ".join(" ".join(str(cell) for cell in row) for row in header_rows[:max_rows])
    return compact_token(text)


def detect_decimal_split_values(raw_table: dict[str, Any] | None) -> bool:
    if not raw_table:
        return False
    for row in raw_table.get("rows") or []:
        values = [normalize_space(cell) for cell in row]
        for col_index in range(len(values) - 1):
            if is_decimal_split(values[col_index], values[col_index + 1]):
                return True
    return False


def detect_metric_checks(
    raw_table: dict[str, Any] | None,
    split_detected: bool,
    merged_detected: bool,
) -> dict[str, Any]:
    if not raw_table:
        return {
            "metric_level_cell_gap": False,
            "numeric_column_order_uncertain": False,
            "missing_metric_cell_warning": False,
            "metric_column_group_uncertain": False,
            "evidence": ["no_raw_pdfplumber_table"],
        }
    rows = raw_table.get("rows") or []
    full_text = normalize_space(" ".join(" ".join(str(cell) for cell in row) for row in rows))
    full_compact = compact_token(full_text)
    header_compact = header_compact_text(raw_table)
    metric_present = any(term in full_compact for term in METRIC_TERMS)
    decimal_split = detect_decimal_split_values(raw_table)
    metric_order_positions = [header_compact.find(term) for term in EXPECTED_METRIC_ORDER if term in header_compact]
    expected_order_unstable = bool(metric_present and len(metric_order_positions) < 2)
    if len(metric_order_positions) >= 2:
        expected_order_unstable = metric_order_positions != sorted(metric_order_positions)

    merged_metric_value = False
    metric_evidence: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        values = [normalize_space(cell) for cell in row]
        for col_index, cell in enumerate(values, start=1):
            if not cell:
                continue
            if has_metric_context(cell) and (NUMERIC_RE.search(cell) or LITERAL_RE.search(cell)):
                merged_metric_value = True
                evidence_append(metric_evidence, f"r{row_index}c{col_index}:metric context and value are not isolated")
        non_empty_numeric = [cell for cell in values if NUMERIC_RE.search(cell)]
        if metric_present and len(non_empty_numeric) >= 3 and any(not cell for cell in values):
            evidence_append(metric_evidence, f"r{row_index}:numeric sequence crosses empty/split cells")

    metric_column_group_uncertain = bool(metric_present and (split_detected or expected_order_unstable))
    numeric_column_order_uncertain = bool(metric_present and (metric_column_group_uncertain or decimal_split))
    metric_level_cell_gap = bool(metric_present and (numeric_column_order_uncertain or merged_metric_value or merged_detected))
    missing_metric_cell_warning = bool(metric_present and (decimal_split or (metric_level_cell_gap and expected_order_unstable)))
    if decimal_split:
        evidence_append(metric_evidence, "decimal value split across adjacent cells")
    if expected_order_unstable:
        evidence_append(metric_evidence, "expected metric column order cannot be reconstructed from header grid")
    if metric_column_group_uncertain:
        evidence_append(metric_evidence, "metric column group is split or incomplete")
    return {
        "metric_level_cell_gap": metric_level_cell_gap,
        "numeric_column_order_uncertain": numeric_column_order_uncertain,
        "missing_metric_cell_warning": missing_metric_cell_warning,
        "metric_column_group_uncertain": metric_column_group_uncertain,
        "evidence": metric_evidence or ["no_metric_level_gap_signal"],
    }


def all_object_warnings(obj: dict[str, Any], row: dict[str, Any] | None = None) -> set[str]:
    warnings = set(obj.get("warnings") or [])
    warnings.update(obj.get("binding_warnings") or [])
    if row:
        warnings.update(split_semicolon(row.get("warnings", "")))
        warnings.update(split_semicolon(row.get("binding_warnings", "")))
        warnings.update(split_semicolon(row.get("routing_blockers", "")))
    return warnings


def binding_review_id(row: dict[str, Any]) -> str:
    return str(row.get("hybrid_table_object_id") or row.get("table_object_id") or "")


def normalize_binding_statuses(
    obj: dict[str, Any],
    binding_review: dict[str, Any] | None,
    raw_table: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
    binding_review = binding_review or {}
    statuses = {
        "unit_binding_status": obj.get("unit_binding_status") or binding_review.get("unit_binding_status") or "not_reviewed",
        "footnote_binding_status": obj.get("footnote_binding_status")
        or binding_review.get("footnote_binding_status")
        or "not_reviewed",
        "reference_binding_status": obj.get("reference_binding_status")
        or binding_review.get("reference_binding_status")
        or "not_reviewed",
        "literal_preservation_status": binding_review.get("literal_preservation_status")
        or obj.get("literal_preservation_status")
        or "not_applicable",
    }
    warnings: list[str] = []
    if binding_review.get("unit_visible") and not binding_review.get("unit_bound"):
        statuses["unit_binding_status"] = "uncertain"
        warnings.extend(["unit_visible_not_bound", "unit_binding_uncertain"])
    if binding_review.get("footnote_present") and not binding_review.get("footnote_bound"):
        statuses["footnote_binding_status"] = "uncertain"
        warnings.extend(["footnote_present_not_bound", "footnote_binding_uncertain"])
    if binding_review.get("reference_visible") and not binding_review.get("row_level_reference_bound"):
        statuses["reference_binding_status"] = "uncertain"
        warnings.extend(
            [
                "reference_visible_not_bound",
                "reference_binding_uncertain",
                "internal_reference_column",
                "external_citation_not_supported",
            ]
        )
    raw_text = normalize_space(
        " ".join(" ".join(str(cell) for cell in row) for row in ((raw_table or {}).get("rows") or []))
    )
    if LITERAL_RE.search(raw_text):
        if statuses["literal_preservation_status"] in {"not_applicable", "not_reviewed", ""}:
            statuses["literal_preservation_status"] = "pass_with_warnings"
        warnings.append("literal_value_requires_preservation")
    return statuses, sorted(set(warnings))


def candidate_upgrade_eligible(
    phase7d_row: dict[str, Any],
    blockers: list[str],
    statuses: dict[str, str],
) -> bool:
    if phase7d_row.get("routing_status") != "needs_pdfplumber_rule_fix":
        return False
    if blockers:
        return False
    alignment_status = phase7d_row.get("alignment_status", "")
    alignment_confidence = phase7d_row.get("alignment_confidence", "")
    if not align_gate.alignment_allows_ready_candidate(alignment_status, alignment_confidence):
        return False
    blocking_statuses = {"fail", "uncertain", "not_reviewed", "not_evaluable"}
    for key in [
        "unit_binding_status",
        "footnote_binding_status",
        "reference_binding_status",
        "literal_preservation_status",
    ]:
        if statuses.get(key) in blocking_statuses:
            return False
    return True


def build_rule_fix_diagnostic(
    obj: dict[str, Any],
    phase7d_row: dict[str, Any],
    binding_review: dict[str, Any] | None,
    raw_table: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = all_object_warnings(obj, phase7d_row)
    split_detected, split_evidence = detect_split_cells(raw_table)
    merged_detected, merged_evidence = detect_merged_cells(raw_table)
    continuation_detected, continuation_evidence = detect_row_continuation(raw_table)
    column_inconsistent, column_evidence = detect_column_alignment_inconsistent(
        raw_table,
        split_detected or "split_cell_warning" in existing,
        merged_detected or "merged_cell_warning" in existing,
        continuation_detected or "row_continuation_warning" in existing,
    )
    metric_checks = detect_metric_checks(
        raw_table,
        split_detected or "split_cell_warning" in existing,
        merged_detected or "merged_cell_warning" in existing,
    )
    statuses, binding_warnings = normalize_binding_statuses(obj, binding_review, raw_table)

    split_warning = split_detected or "split_cell_warning" in existing
    merged_warning = merged_detected or "merged_cell_warning" in existing
    row_warning = continuation_detected or "row_continuation_warning" in existing
    column_warning = column_inconsistent or "column_alignment_inconsistent" in existing
    metric_gap = metric_checks["metric_level_cell_gap"] or "metric_level_cell_gap" in existing
    numeric_order = metric_checks["numeric_column_order_uncertain"] or "numeric_column_order_uncertain" in existing
    missing_metric = metric_checks["missing_metric_cell_warning"] or "missing_metric_cell_warning" in existing
    metric_group = metric_checks["metric_column_group_uncertain"] or "metric_column_group_uncertain" in existing

    unit_warning = any(item in binding_warnings or item in existing for item in ["unit_visible_not_bound", "unit_binding_uncertain"])
    footnote_warning = any(
        item in binding_warnings or item in existing
        for item in ["footnote_present_not_bound", "footnote_binding_uncertain"]
    )
    reference_warning = any(
        item in binding_warnings or item in existing
        for item in ["reference_visible_not_bound", "reference_binding_uncertain"]
    )
    literal_warning = any(
        item in binding_warnings or item in existing
        for item in ["literal_value_requires_preservation", "raw_literal_lost", "literal_definition_missing"]
    )

    blockers = []
    for name, active in [
        ("split_cell_warning", split_warning),
        ("merged_cell_warning", merged_warning),
        ("row_continuation_warning", row_warning),
        ("column_alignment_inconsistent", column_warning),
        ("metric_level_cell_gap", metric_gap),
        ("numeric_column_order_uncertain", numeric_order),
        ("missing_metric_cell_warning", missing_metric),
        ("metric_column_group_uncertain", metric_group),
        ("unit_binding_uncertain", unit_warning),
        ("footnote_binding_uncertain", footnote_warning),
        ("reference_binding_uncertain", reference_warning),
    ]:
        if active:
            blockers.append(name)
    if phase7d_row.get("source_span_granularity") != "cell_level":
        blockers.append("source_span_not_cell_level_for_rule_fix")
    blockers.extend(split_semicolon(phase7d_row.get("routing_blockers", "")))
    blockers = sorted(set(blockers))
    eligible = candidate_upgrade_eligible(phase7d_row, blockers, statuses)
    notes = [
        "v2.1 规则已检测结构 blocker；未重构 logical cells。",
        "candidate_upgrade_eligible=true 也不等于 confirmed gold。",
    ]
    if blockers:
        notes.append("仍有 blocker，保持 needs_pdfplumber_rule_fix。")
    else:
        notes.append("未发现 blocker；可进入 ready routing 检查。")
    evidence = {
        "split_cell_evidence": split_evidence,
        "merged_cell_evidence": merged_evidence,
        "row_continuation_evidence": continuation_evidence,
        "column_alignment_evidence": column_evidence,
        "metric_evidence": metric_checks["evidence"],
        "normalized_binding_warnings": binding_warnings,
    }
    diagnostic = {
        "table_object_id": obj.get("table_object_id", ""),
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "original_routing_status": phase7d_row.get("routing_status", ""),
        "primary_rule_fix_reason": phase7d_row.get("rule_fix_scope")
        or semicolon((binding_review or {}).get("key_warnings") or []),
        "split_cell_detected": split_warning,
        "merged_cell_detected": merged_warning,
        "row_continuation_detected": row_warning,
        "column_alignment_inconsistent": column_warning,
        "metric_level_cell_gap_detected": metric_gap,
        "numeric_column_order_uncertain": numeric_order,
        "missing_metric_cell_warning": missing_metric,
        "unit_binding_warning": unit_warning,
        "footnote_binding_warning": footnote_warning,
        "reference_binding_warning": reference_warning,
        "literal_preservation_warning": literal_warning,
        "rule_fix_attempted": True,
        "rule_fix_applied": False,
        "rule_fix_blockers": blockers,
        "candidate_upgrade_eligible": eligible,
        "notes": " ".join(notes),
        "statuses": statuses,
        "binding_warnings": binding_warnings,
        "metric_column_group_uncertain": metric_group,
        "cell_grid_needs_rule_fix": bool(
            split_warning or merged_warning or row_warning or column_warning or metric_gap or numeric_order
        ),
        "evidence": evidence,
    }
    return diagnostic, evidence


def blank_diagnostic(
    obj: dict[str, Any],
    phase7d_row: dict[str, Any],
    binding_review: dict[str, Any] | None,
    raw_table: dict[str, Any] | None,
) -> dict[str, Any]:
    statuses, binding_warnings = normalize_binding_statuses(obj, binding_review, raw_table)
    existing = all_object_warnings(obj, phase7d_row)
    return {
        "table_object_id": obj.get("table_object_id", ""),
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "original_routing_status": phase7d_row.get("routing_status", ""),
        "primary_rule_fix_reason": "not_in_rule_fix_scope",
        "split_cell_detected": "split_cell_warning" in existing,
        "merged_cell_detected": "merged_cell_warning" in existing,
        "row_continuation_detected": "row_continuation_warning" in existing,
        "column_alignment_inconsistent": "column_alignment_inconsistent" in existing,
        "metric_level_cell_gap_detected": "metric_level_cell_gap" in existing,
        "numeric_column_order_uncertain": "numeric_column_order_uncertain" in existing,
        "missing_metric_cell_warning": "missing_metric_cell_warning" in existing,
        "unit_binding_warning": any(item in existing or item in binding_warnings for item in ["unit_visible_not_bound", "unit_binding_uncertain"]),
        "footnote_binding_warning": any(item in existing or item in binding_warnings for item in ["footnote_present_not_bound", "footnote_binding_uncertain"]),
        "reference_binding_warning": any(item in existing or item in binding_warnings for item in ["reference_visible_not_bound", "reference_binding_uncertain"]),
        "literal_preservation_warning": any(item in existing or item in binding_warnings for item in ["literal_value_requires_preservation"]),
        "rule_fix_attempted": False,
        "rule_fix_applied": False,
        "rule_fix_blockers": [],
        "candidate_upgrade_eligible": False,
        "notes": "非本轮 3 个 rule-fix case；仅做 binding warning normalization，不改变 routing。",
        "statuses": statuses,
        "binding_warnings": binding_warnings,
        "metric_column_group_uncertain": "metric_column_group_uncertain" in existing,
        "cell_grid_needs_rule_fix": "cell_grid_needs_rule_fix" in existing,
        "evidence": {},
    }


def diagnostic_csv_row(diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_object_id": diagnostic["table_object_id"],
        "doc_id": diagnostic["doc_id"],
        "table_id": diagnostic["table_id"],
        "original_routing_status": diagnostic["original_routing_status"],
        "primary_rule_fix_reason": diagnostic["primary_rule_fix_reason"],
        "split_cell_detected": bool_text(diagnostic["split_cell_detected"]),
        "merged_cell_detected": bool_text(diagnostic["merged_cell_detected"]),
        "row_continuation_detected": bool_text(diagnostic["row_continuation_detected"]),
        "column_alignment_inconsistent": bool_text(diagnostic["column_alignment_inconsistent"]),
        "metric_level_cell_gap_detected": bool_text(diagnostic["metric_level_cell_gap_detected"]),
        "numeric_column_order_uncertain": bool_text(diagnostic["numeric_column_order_uncertain"]),
        "missing_metric_cell_warning": bool_text(diagnostic["missing_metric_cell_warning"]),
        "unit_binding_warning": bool_text(diagnostic["unit_binding_warning"]),
        "footnote_binding_warning": bool_text(diagnostic["footnote_binding_warning"]),
        "reference_binding_warning": bool_text(diagnostic["reference_binding_warning"]),
        "literal_preservation_warning": bool_text(diagnostic["literal_preservation_warning"]),
        "rule_fix_attempted": bool_text(diagnostic["rule_fix_attempted"]),
        "rule_fix_applied": bool_text(diagnostic["rule_fix_applied"]),
        "rule_fix_blockers": semicolon(diagnostic["rule_fix_blockers"]),
        "candidate_upgrade_eligible": bool_text(diagnostic["candidate_upgrade_eligible"]),
        "notes": diagnostic["notes"],
    }


def source_spans_without_value_level(obj: dict[str, Any]) -> None:
    if obj.get("source_span_granularity") == "value_level":
        obj["source_span_granularity"] = "mixed_or_unclear"
    for span in obj.get("source_spans") or []:
        if span.get("granularity") == "value_level":
            span["granularity"] = "mixed_or_unclear"
            span["bbox"] = None


def warning_names_from_diagnostic(diagnostic: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    mapping = [
        ("split_cell_detected", "split_cell_warning"),
        ("merged_cell_detected", "merged_cell_warning"),
        ("row_continuation_detected", "row_continuation_warning"),
        ("column_alignment_inconsistent", "column_alignment_inconsistent"),
        ("cell_grid_needs_rule_fix", "cell_grid_needs_rule_fix"),
        ("metric_level_cell_gap_detected", "metric_level_cell_gap"),
        ("numeric_column_order_uncertain", "numeric_column_order_uncertain"),
        ("missing_metric_cell_warning", "missing_metric_cell_warning"),
        ("metric_column_group_uncertain", "metric_column_group_uncertain"),
    ]
    for key, warning in mapping:
        if diagnostic.get(key):
            warnings.append(warning)
    if diagnostic.get("unit_binding_warning"):
        warnings.extend(["unit_visible_not_bound", "unit_binding_uncertain"])
    if diagnostic.get("footnote_binding_warning"):
        warnings.extend(["footnote_present_not_bound", "footnote_binding_uncertain"])
    if diagnostic.get("reference_binding_warning"):
        warnings.extend(["reference_visible_not_bound", "reference_binding_uncertain"])
    if diagnostic.get("literal_preservation_warning"):
        warnings.append("literal_value_requires_preservation")
    warnings.extend(diagnostic.get("binding_warnings") or [])
    warnings.extend(["value_level_bbox_absent", "cell_bbox_not_value_bbox"])
    return sorted(set(warnings))


def final_route_from_diagnostic(phase7d_row: dict[str, Any], diagnostic: dict[str, Any]) -> tuple[str, str, str]:
    original_status = phase7d_row.get("routing_status", "")
    original_action = phase7d_row.get("final_action", "")
    if original_status == "needs_pdfplumber_rule_fix" and diagnostic.get("candidate_upgrade_eligible"):
        return (
            "ready_for_gold_candidate",
            "keep_ready_candidate",
            "v2.1 rule-fix 检测未发现剩余 blocker；仅可进入 ready_for_gold_candidate，不等于 confirmed gold。",
        )
    return original_status, original_action, phase7d_row.get("routing_reason", "")


def build_v21_object_and_row(
    source_obj: dict[str, Any],
    phase7d_row: dict[str, Any],
    diagnostic: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    obj = strip_forbidden_keys(copy.deepcopy(source_obj))
    phase7d_status = phase7d_row.get("routing_status", "")
    phase7d2_status, final_action, routing_reason = final_route_from_diagnostic(phase7d_row, diagnostic)
    blockers = list(diagnostic.get("rule_fix_blockers") or [])
    if phase7d_status != "needs_pdfplumber_rule_fix":
        blockers = split_semicolon(phase7d_row.get("routing_blockers", ""))
    statuses = diagnostic.get("statuses") or {}
    warnings = add_unique(list(obj.get("warnings") or []), warning_names_from_diagnostic(diagnostic))
    binding_warnings = add_unique(list(obj.get("binding_warnings") or []), warning_names_from_diagnostic(diagnostic))
    source_spans_without_value_level(obj)
    obj.update(
        {
            "phase": "v7_phase7D_2_hybrid_extractor_v2_1",
            "schema_name": "table_object_v2",
            "schema_version": "v2.1",
            "phase7d_routing_status": phase7d_status,
            "phase7d2_routing_status": phase7d2_status,
            "routing_status": phase7d2_status,
            "routing_reason": routing_reason,
            "final_action": final_action,
            "routing_blockers": blockers,
            "rule_fix_attempted": bool(diagnostic.get("rule_fix_attempted")),
            "rule_fix_applied": bool(diagnostic.get("rule_fix_applied")),
            "rule_fix_blockers": blockers,
            "candidate_upgrade_eligible": bool(diagnostic.get("candidate_upgrade_eligible")),
            "split_cell_warning": bool(diagnostic.get("split_cell_detected")),
            "merged_cell_warning": bool(diagnostic.get("merged_cell_detected")),
            "row_continuation_warning": bool(diagnostic.get("row_continuation_detected")),
            "column_alignment_inconsistent": bool(diagnostic.get("column_alignment_inconsistent")),
            "cell_grid_needs_rule_fix": bool(diagnostic.get("cell_grid_needs_rule_fix")),
            "metric_level_cell_gap": bool(diagnostic.get("metric_level_cell_gap_detected")),
            "numeric_column_order_uncertain": bool(diagnostic.get("numeric_column_order_uncertain")),
            "missing_metric_cell_warning": bool(diagnostic.get("missing_metric_cell_warning")),
            "metric_column_group_uncertain": bool(diagnostic.get("metric_column_group_uncertain")),
            "unit_binding_status": statuses.get("unit_binding_status", obj.get("unit_binding_status", "not_reviewed")),
            "footnote_binding_status": statuses.get(
                "footnote_binding_status", obj.get("footnote_binding_status", "not_reviewed")
            ),
            "reference_binding_status": statuses.get(
                "reference_binding_status", obj.get("reference_binding_status", "not_reviewed")
            ),
            "literal_preservation_status": statuses.get(
                "literal_preservation_status", obj.get("literal_preservation_status", "not_applicable")
            ),
            "binding_warnings": binding_warnings,
            "warnings": warnings,
            "usable_hybrid_candidate": phase7d2_status == "ready_for_gold_candidate",
            "pdfplumber_grid_reliable": phase7d2_status == "ready_for_gold_candidate",
            "value_bboxes_available": False,
            "no_value_level_bbox": True,
        }
    )
    if obj.get("source_span_granularity") == "value_level":
        obj["source_span_granularity"] = "mixed_or_unclear"
    meta = obj.setdefault("hybrid_metadata", {})
    meta.update(
        {
            "phase7d2_rule_fix_applied": bool(diagnostic.get("rule_fix_applied")),
            "phase7d_routing_status": phase7d_status,
            "phase7d2_routing_status": phase7d2_status,
            "final_action": final_action,
            "value_bboxes_available": False,
            "source_span_granularity": obj.get("source_span_granularity"),
        }
    )
    row = dict(phase7d_row)
    row.update(
        {
            "routing_status": phase7d2_status,
            "final_action": final_action,
            "routing_reason": routing_reason,
            "routing_blockers": semicolon(blockers),
            "usable_hybrid_candidate": bool_text(phase7d2_status == "ready_for_gold_candidate"),
            "value_bboxes_available": "false",
            "source_span_granularity": obj.get("source_span_granularity", phase7d_row.get("source_span_granularity", "")),
            "warnings": semicolon(warnings),
            "binding_warnings": semicolon(binding_warnings),
            "unit_binding_status": obj["unit_binding_status"],
            "footnote_binding_status": obj["footnote_binding_status"],
            "reference_binding_status": obj["reference_binding_status"],
            "phase7d_routing_status": phase7d_status,
            "phase7d2_routing_status": phase7d2_status,
            "rule_fix_attempted": bool_text(diagnostic.get("rule_fix_attempted")),
            "rule_fix_applied": bool_text(diagnostic.get("rule_fix_applied")),
            "remaining_blockers": semicolon(blockers),
            "candidate_upgrade_eligible": bool_text(diagnostic.get("candidate_upgrade_eligible")),
            "split_cell_warning": bool_text(diagnostic.get("split_cell_detected")),
            "merged_cell_warning": bool_text(diagnostic.get("merged_cell_detected")),
            "row_continuation_warning": bool_text(diagnostic.get("row_continuation_detected")),
            "metric_level_cell_gap": bool_text(diagnostic.get("metric_level_cell_gap_detected")),
            "numeric_column_order_uncertain": bool_text(diagnostic.get("numeric_column_order_uncertain")),
            "missing_metric_cell_warning": bool_text(diagnostic.get("missing_metric_cell_warning")),
            "metric_column_group_uncertain": bool_text(diagnostic.get("metric_column_group_uncertain")),
            "literal_preservation_status": obj["literal_preservation_status"],
        }
    )
    changed = phase7d_status != phase7d2_status
    delta = {
        "table_object_id": obj["table_object_id"],
        "phase7d_routing_status": phase7d_status,
        "phase7d2_routing_status": phase7d2_status,
        "changed": bool_text(changed),
        "change_type": "routing_changed" if changed else "routing_preserved",
        "fix_applied": bool_text(diagnostic.get("rule_fix_applied")),
        "remaining_blockers": semicolon(blockers),
        "upgrade_justification": (
            "所有 blocker 已由 v2.1 规则消除；仍只是 ready_for_gold_candidate。"
            if changed and phase7d2_status == "ready_for_gold_candidate"
            else "none"
        ),
        "downgrade_reason": "none",
        "notes": (
            "v2.1 保持 Phase7D routing；新增结构化 rule-fix diagnostics 和 binding warning normalization。"
            if not changed
            else "v2.1 routing 发生变化，详见 upgrade_justification。"
        ),
    }
    return obj, row, delta


def build_v21_outputs(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_inputs(inputs)
    phase7d_row_by_id = {row["table_object_id"]: row for row in inputs["phase7d_summary_rows"]}
    binding_by_id = {binding_review_id(row): row for row in inputs["binding_review_rows"]}
    c2_object_by_id = {row["table_object_id"]: row for row in inputs["c2_hybrid_objects"]}
    raw_by_pdf_id = {row["pdfplumber_table_id"]: row for row in inputs["c2_raw_tables"]}
    objects: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for source_obj in inputs["phase7d_objects"]:
        table_object_id = source_obj["table_object_id"]
        phase7d_row = phase7d_row_by_id[table_object_id]
        binding = binding_by_id.get(table_object_id)
        raw_table = raw_table_for_object(source_obj, c2_object_by_id, raw_by_pdf_id)
        if table_object_id in RULE_FIX_IDS:
            diagnostic, _ = build_rule_fix_diagnostic(source_obj, phase7d_row, binding, raw_table)
            diagnostics.append(diagnostic_csv_row(diagnostic))
        else:
            diagnostic = blank_diagnostic(source_obj, phase7d_row, binding, raw_table)
        obj, row, delta = build_v21_object_and_row(source_obj, phase7d_row, diagnostic)
        objects.append(obj)
        rows.append(row)
        deltas.append(delta)
    validate_v21_outputs(objects, rows, deltas)
    return objects, rows, diagnostics, deltas


def has_forbidden_output_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN_OUTPUT_KEYS or has_forbidden_output_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_forbidden_output_key(item) for item in value)
    return False


def validate_v21_outputs(objects: list[dict[str, Any]], rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    if len(objects) != 16 or len(rows) != 16 or len(deltas) != 16:
        errors.append(f"v2.1 输出应覆盖 16 个 case，实际 objects={len(objects)} rows={len(rows)} delta={len(deltas)}")
    object_ids = {obj["table_object_id"] for obj in objects}
    row_ids = {row["table_object_id"] for row in rows}
    if object_ids != row_ids:
        errors.append("v2.1 object 与 summary row 覆盖不一致")
    status_by_id = {row["table_object_id"]: row["routing_status"] for row in rows}
    action_by_id = {row["table_object_id"]: row["final_action"] for row in rows}
    if {obj_id for obj_id, status in status_by_id.items() if status == "ready_for_gold_candidate"} != READY_IDS:
        errors.append("2 个 ready candidate 未稳定保留，或出现额外 ready")
    if any(status_by_id.get(obj_id) == "ready_for_gold_candidate" for obj_id in RULE_FIX_IDS):
        errors.append("rule-fix case 被误标 ready")
    for obj_id in RULE_FIX_IDS:
        if status_by_id.get(obj_id) != "needs_pdfplumber_rule_fix":
            errors.append(f"{obj_id} 未稳定保持 needs_pdfplumber_rule_fix")
    for obj_id in GRID_REJECTED_IDS:
        if status_by_id.get(obj_id) == "ready_for_gold_candidate" or action_by_id.get(obj_id) != "reject_pdfplumber_grid":
            errors.append(f"{obj_id} grid_rejected 回归")
    for obj_id in CHUNK_FALLBACK_IDS:
        if action_by_id.get(obj_id) != "use_chunk_fallback":
            errors.append(f"{obj_id} chunk_fallback final_action 回归")
    for obj_id in BACKLOG_IDS:
        if action_by_id.get(obj_id) != "keep_backlog":
            errors.append(f"{obj_id} backlog final_action 回归")
    if any(row["value_bboxes_available"] != "false" for row in rows):
        errors.append("summary 中 value_bboxes_available 必须全部为 false")
    if any(obj.get("value_bboxes_available") for obj in objects):
        errors.append("objects 中 value_bboxes_available 不得为 true")
    if any(row.get("source_span_granularity") == "value_level" for row in rows):
        errors.append("summary 中 source_span_granularity 不得为 value_level")
    if any(obj.get("source_span_granularity") == "value_level" for obj in objects):
        errors.append("objects 中 source_span_granularity 不得为 value_level")
    if any(has_forbidden_output_key(obj) for obj in objects):
        errors.append("v2.1 object 不得写 confirmed/prod ready 字段")
    payload = json.dumps({"objects": objects, "rows": rows}, ensure_ascii=False)
    if "production_ready" in payload or "confirmed_gold" in payload:
        errors.append("v2.1 输出 payload 不得出现 production_ready / confirmed_gold")
    if errors:
        raise ValueError("Phase7D-2 v2.1 输出校验失败：" + "; ".join(errors))


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
        "ready_stable": READY_IDS == {obj["table_object_id"] for obj in splits["ready"]},
        "rule_fix_stable": RULE_FIX_IDS == {row["table_object_id"] for row in splits["rule_fix"]},
        "all_value_bbox_false": all(row["value_bboxes_available"] == "false" for row in rows),
        "no_value_level": all(row.get("source_span_granularity") != "value_level" for row in rows),
        "diagnostics": diagnostics,
        "warning_counts": {
            "split_cell_warning": sum(parse_bool(row.get("split_cell_warning")) for row in rows),
            "merged_cell_warning": sum(parse_bool(row.get("merged_cell_warning")) for row in rows),
            "row_continuation_warning": sum(parse_bool(row.get("row_continuation_warning")) for row in rows),
            "metric_level_cell_gap": sum(parse_bool(row.get("metric_level_cell_gap")) for row in rows),
            "numeric_column_order_uncertain": sum(parse_bool(row.get("numeric_column_order_uncertain")) for row in rows),
            "missing_metric_cell_warning": sum(parse_bool(row.get("missing_metric_cell_warning")) for row in rows),
            "unit_binding_warning": sum("unit_visible_not_bound" in row.get("binding_warnings", "") for row in rows),
            "footnote_binding_warning": sum("footnote_present_not_bound" in row.get("binding_warnings", "") for row in rows),
            "reference_binding_warning": sum("reference_visible_not_bound" in row.get("binding_warnings", "") for row in rows),
        },
    }


def write_review_markdown(objects: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Phase7D-2 hybrid table_objects v2.1 审阅视图",
        "",
        "本文件用于审阅 v2.1 rule-fix diagnostics 与 routing 回归；JSONL 是机器可读 source of truth。",
        "",
        "本轮不扩大 smoke，不构造 confirmed gold，不接 production，不伪造 value-level bbox。",
        "",
    ]
    for obj in objects:
        warnings = obj.get("warnings") or []
        lines.extend(
            [
                f"## {obj.get('table_object_id')}",
                "",
                f"- table_object_id：`{obj.get('table_object_id')}`",
                f"- doc_id：`{obj.get('doc_id')}`",
                f"- table_id：`{obj.get('table_id')}`",
                f"- phase7d_routing_status：`{obj.get('phase7d_routing_status')}`",
                f"- phase7d2_routing_status：`{obj.get('phase7d2_routing_status')}`",
                f"- final_action：`{obj.get('final_action')}`",
                f"- rule_fix_applied：`{str(bool(obj.get('rule_fix_applied'))).lower()}`",
                f"- remaining_blockers：`{semicolon(obj.get('rule_fix_blockers') or obj.get('routing_blockers') or [])}`",
                f"- split_cell_warning：`{str(bool(obj.get('split_cell_warning'))).lower()}`",
                f"- merged_cell_warning：`{str(bool(obj.get('merged_cell_warning'))).lower()}`",
                f"- row_continuation_warning：`{str(bool(obj.get('row_continuation_warning'))).lower()}`",
                f"- metric_level_cell_gap：`{str(bool(obj.get('metric_level_cell_gap'))).lower()}`",
                f"- numeric_column_order_uncertain：`{str(bool(obj.get('numeric_column_order_uncertain'))).lower()}`",
                f"- unit_binding_status：`{obj.get('unit_binding_status')}`",
                f"- footnote_binding_status：`{obj.get('footnote_binding_status')}`",
                f"- reference_binding_status：`{obj.get('reference_binding_status')}`",
                f"- value_bboxes_available：`false`",
                f"- source_span_granularity：`{obj.get('source_span_granularity')}`",
                f"- warnings：`{semicolon(warnings)}`",
                "",
                "### 表格预览",
                "",
            ]
        )
        lines.extend(render_preview_table(obj))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_guardrail(report_dir: Path, inventory: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7D-2 护栏",
        "",
        "## 1. 本轮定位",
        "",
        "本轮定位为工程规则修复：在 Phase7D hybrid extractor v2 的 16 个 table_objects 上增加 v2.1 rule-fix diagnostics、warning normalization 和 routing 回归保护。",
        "",
        "## 2. 明确边界",
        "",
        "1. 本轮是工程规则修复。",
        "2. 本轮不是审阅阶段。",
        "3. 本轮不是 gold construction。",
        "4. 本轮不扩大 smoke，仍限定既有 9 个 doc_id。",
        "5. 本轮不引入 Camelot / PyMuPDF。",
        "6. 本轮不接 production，不修改 ingestion 主链路。",
        "7. 本轮不访问 Milvus / BM25，不读取或查询 BM25 index。",
        "8. 本轮不运行 retrieval / embedding / rerank / model，不调用 Qwen / RAGAS / OCR / VLM。",
        "9. 本轮不伪造 value-level bbox；cell bbox 不能写成 value bbox。",
        "10. 本轮不得为了增加 ready 数量强行升级 rule-fix case。",
        "11. Route C 仍只是 backlog。",
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
    (report_dir / "phase7d_2_guardrail.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rule_fix_implementation_report(report_dir: Path, diagnostics: list[dict[str, Any]]) -> None:
    lines = [
        "# 规则修复实施报告",
        "",
        "## 1. 实现目标",
        "",
        "本轮将 split cell、merged cell、row continuation、column alignment、metric-level completeness 与 binding warning normalization 固化为 v2.1 可重跑规则。规则用于识别 blocker 和防止误升级，不用于构造 gold。",
        "",
        "## 2. 规则实现口径",
        "",
        "- split cell：检测相邻 narrow/fragment cells 组成 decimal、Location、Reference、qxylose、YE/S 等 logical token 的情况。",
        "- merged cell：检测一个 cell 中同时包含 logical label/value、primer name + sequence、medium/culture + titer value 等结构。",
        "- row continuation：检测 row label 为空或短 suffix 续接上一 logical row 的情况。",
        "- column alignment：检测高 empty ratio、多列拆分、正文/页脚混入 grid、split 与 merged 同时出现等风险。",
        "- metric-level completeness：检测 YE/S、qethanol、qxylose、qarabinose、titer 等 metric header/value 是否可稳定绑定。",
        "- binding normalization：遵守 Phase6D contract，unit/footnote/reference 可见不等于已绑定，literal 原文必须保留。",
        "",
        "## 3. 3 个 rule-fix case 诊断",
        "",
        "| table_object_id | split | merged | row_continuation | metric_gap | numeric_uncertain | missing_metric | rule_fix_applied | candidate_upgrade_eligible | blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in diagnostics:
        lines.append(
            f"| `{row['table_object_id']}` | {row['split_cell_detected']} | {row['merged_cell_detected']} | {row['row_continuation_detected']} | {row['metric_level_cell_gap_detected']} | {row['numeric_column_order_uncertain']} | {row['missing_metric_cell_warning']} | {row['rule_fix_applied']} | {row['candidate_upgrade_eligible']} | {row['rule_fix_blockers']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 实施结果",
            "",
            "- 3 个 rule-fix case 均完成 diagnostics。",
            "- 本轮没有 case 被真实修复到可安全升级 ready，因此 `rule_fix_applied=false`，routing 继续保持 `needs_pdfplumber_rule_fix`。",
            "- 改善点是 blocker 不再只来自人工文字说明，而是由可解释的结构规则复现。",
            "- `candidate_upgrade_eligible=true` 只代表满足 ready routing 的前置检查，不等于 confirmed gold；本轮实际没有 eligible case。",
            "",
            "## 5. 未做事项",
            "",
            "- 未构造 value-level bbox。",
            "- 未构造 confirmed gold。",
            "- 未接入 production。",
            "- 未扩大 smoke。",
        ]
    )
    (report_dir / "rule_fix_implementation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    data = facts(objects, rows, diagnostics)
    warning_counts = data["warning_counts"]
    lines = [
        "# table_object v2.1 验证报告",
        "",
        "## 1. v2.1 table_object 总数",
        "",
        f"- v2.1 table_object 总数：{data['total']}",
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
            "## 9. ready candidate 稳定性",
            "",
            f"- 2 个 ready candidate 是否稳定保留：{'是' if data['ready_stable'] else '否'}。",
            "",
            "## 10. rule-fix case 处理结果",
            "",
            "- 3 个 rule-fix case 均执行 v2.1 diagnostics。",
            "- 3 个 rule-fix case 均未被真实修复到可安全升级，继续稳定标记为 `needs_pdfplumber_rule_fix`。",
            "",
            "## 11. split / merged cell 检测结果",
            "",
            f"- split_cell_warning 数量：{warning_counts['split_cell_warning']}",
            f"- merged_cell_warning 数量：{warning_counts['merged_cell_warning']}",
            f"- row_continuation_warning 数量：{warning_counts['row_continuation_warning']}",
            "",
            "## 12. metric-level gap 检测结果",
            "",
            f"- metric_level_cell_gap 数量：{warning_counts['metric_level_cell_gap']}",
            f"- numeric_column_order_uncertain 数量：{warning_counts['numeric_column_order_uncertain']}",
            f"- missing_metric_cell_warning 数量：{warning_counts['missing_metric_cell_warning']}",
            "",
            "## 13. binding warning normalization 结果",
            "",
            f"- unit binding warning 数量：{warning_counts['unit_binding_warning']}",
            f"- footnote binding warning 数量：{warning_counts['footnote_binding_warning']}",
            f"- reference binding warning 数量：{warning_counts['reference_binding_warning']}",
            "- literal preservation status 已写入 v2.1 对象；N.D. / NT / NC 等 literal 不做规范化替换。",
            "",
            "## 14. grid rejected 是否不再进入 usable",
            "",
            "- 是。5 个 grid_rejected 均保持 `final_action=reject_pdfplumber_grid`，未进入 ready 或 usable hybrid。",
            "",
            "## 15. chunk fallback 是否生效",
            "",
            "- 是。3 个 chunk_fallback 均保持 `final_action=use_chunk_fallback`。",
            "",
            "## 16. backlog 是否不再硬救",
            "",
            "- 是。3 个 backlog 均保持 `final_action=keep_backlog`，Route C 仍只是 backlog。",
            "",
            "## 17. value bbox 与 source_span",
            "",
            f"- value_bboxes_available 是否全部 false：{'是' if data['all_value_bbox_false'] else '否'}。",
            f"- source_span_granularity 是否没有 value_level：{'是' if data['no_value_level'] else '否'}。",
            "",
            "## 18. validation 是否无回归",
            "",
            "- 是。v2.1 复现 Phase7D 的 2 ready / 3 rule_fix / 5 grid_rejected / 3 chunk_fallback / 3 backlog 分流。",
            "",
            "## 19. confirmed/prod guardrail",
            "",
            "- 未写入 `confirmed_gold`。",
            "- 未写入 `production_ready`。",
            "- `ready_for_gold_candidate` 仍不等于 confirmed gold。",
            "",
            "## 20. 未解决问题",
            "",
            "- 3 个 rule-fix case 的 logical cell reconstruction 仍未完成。",
            "- doc_0687 Table 2 的 metric-level cells 仍有拆分和缺失风险。",
            "- doc_0598 Table 1 仍有 page_only alignment blocker 与 primer row continuation blocker。",
            "- doc_0523 Table 1 仍有 medium/culture 与 titer value 合并、reference 跨行和表尾正文混入问题。",
        ]
    )
    (report_dir / "table_object_v2_1_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    data = facts(objects, rows, diagnostics)
    lines = [
        "# Phase7D 与 Phase7D-2 对比报告",
        "",
        "## 1. 对比目标",
        "",
        "本报告比较 Phase7D extractor v2 与 Phase7D-2 extractor v2.1 在同一批 16 个 hybrid case 上的 routing、rule-fix warning 和 guardrail 是否保持一致。",
        "",
        "## 2. Phase7D v2 状态",
        "",
        "- ready_for_gold_candidate：2",
        "- needs_pdfplumber_rule_fix：3",
        "- grid_rejected：5",
        "- chunk_fallback：3",
        "- backlog：3",
        "",
        "## 3. Phase7D-2 v2.1 状态",
        "",
    ]
    lines.extend(counter_table(data["routing_counts"], sorted(v2.ROUTING_STATUS_VALUES)))
    lines.extend(["", "## 4. ready candidate 变化", ""])
    lines.extend(f"- `{item}`：保持 ready_for_gold_candidate。" for item in data["ready"])
    lines.extend(["", "## 5. rule-fix case 变化", ""])
    for row in diagnostics:
        lines.append(f"- `{row['table_object_id']}`：保持 needs_pdfplumber_rule_fix；remaining_blockers=`{row['rule_fix_blockers']}`。")
    lines.extend(
        [
            "",
            "## 6. 是否有 rule-fix case 被真实修复",
            "",
            "- 没有。3 个 case 均仍有 blocker，未升级 ready。",
            "",
            "## 7. 若无升级，blocker 是否更准确",
            "",
            "- 是。v2.1 将 split/merged/row continuation/metric gap/binding warning 写入结构化 diagnostics 和 table_object 字段。",
            "",
            "## 8. grid rejected case 是否保持",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in data["grid_rejected"])
    lines.extend(["", "## 9. chunk fallback case 是否保持", ""])
    lines.extend(f"- `{item}`" for item in data["chunk_fallback"])
    lines.extend(["", "## 10. backlog case 是否保持", ""])
    lines.extend(f"- `{item}`" for item in data["backlog"])
    lines.extend(
        [
            "",
            "## 11. 是否减少误升级风险",
            "",
            "- 是。v2.1 要求所有 structural/binding blocker 清零后才允许 rule-fix case 进入 ready routing。",
            "",
            "## 12. 是否将 rule-fix warning 写成 extractor 逻辑",
            "",
            "- 是。split_cell_warning、merged_cell_warning、row_continuation_warning、column_alignment_inconsistent、metric_level_cell_gap、numeric_column_order_uncertain、missing_metric_cell_warning 和 binding warnings 已由脚本生成。",
            "",
            "## 13. 是否仍需要 gold construction",
            "",
            "- 是。ready_for_gold_candidate 仍不等于 confirmed gold；gold construction 仍需后续单独授权。",
            "",
            "## 14. 是否建议扩大 smoke",
            "",
            "- 不建议。本轮只验证既定 16 个 case。",
            "",
            "## 15. 是否建议引入 Camelot / PyMuPDF",
            "",
            "- 不建议本轮引入。",
            "",
            "## 16. 是否建议 production",
            "",
            "- 不建议。value-level bbox 仍不存在，rule-fix case 未修复完成。",
            "",
            "## 17. Route C 是否仍只是 backlog",
            "",
            "- 是。Route C 仍只是 backlog。",
            "",
            "## 18. 结论",
            "",
            "- 本轮不是为了增加 pass 数量。",
            "- 本轮是为了实现 rule-fix logic。",
            "- `ready_for_gold_candidate` 仍不等于 confirmed gold。",
            "- gold construction 仍需后续单独授权。",
            "- 不建议 production。",
        ]
    )
    (report_dir / "phase7d_vs_phase7d2_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generated_files(output_dir: Path, report_dir: Path) -> list[str]:
    return [
        rel(output_dir / "table_objects.jsonl"),
        rel(output_dir / "table_objects_review.md"),
        rel(output_dir / "table_object_routing_summary.csv"),
        rel(output_dir / "rule_fix_diagnostics.csv"),
        rel(output_dir / "rule_fix_delta.csv"),
        rel(output_dir / "ready_candidate_pool.jsonl"),
        rel(output_dir / "rule_fix_cases.csv"),
        rel(output_dir / "grid_rejected_cases.csv"),
        rel(output_dir / "chunk_fallback_cases.csv"),
        rel(output_dir / "backlog_cases.csv"),
        rel(report_dir / "phase7d_2_guardrail.md"),
        rel(report_dir / "rule_fix_implementation_report.md"),
        rel(report_dir / "table_object_v2_1_validation_report.md"),
        rel(report_dir / "phase7d_vs_phase7d2_comparison.md"),
        rel(report_dir / "phase7d_2_summary.md"),
    ]


def write_summary(
    objects: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    output_dir: Path,
    report_dir: Path,
) -> None:
    data = facts(objects, rows, diagnostics)
    warning_counts = data["warning_counts"]
    lines = [
        "# Phase7D-2 总结",
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
            "- 新增：`scripts/extraction/apply_hybrid_rule_fixes_v2.py`",
            "",
            "## 3. 新增测试",
            "",
            "- `tests/test_phase7_hybrid_extractor_v2_rule_fix.py`",
            "",
            "## 4. smoke doc_id 是否保持不变",
            "",
            f"- 是，仍为：{', '.join(SMOKE_DOC_IDS)}",
            "",
            "## 5. v2.1 table_object 数量",
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
            "## 13. ready candidate 稳定性",
            "",
            f"- 2 个 ready candidate 是否稳定：{'是' if data['ready_stable'] else '否'}。",
            "",
            "## 14. rule-fix case 状态",
            "",
            "- 3 个 rule-fix case 均完成 diagnostics。",
            "- 3 个 rule-fix case 均未升级，继续稳定标记为 `needs_pdfplumber_rule_fix`。",
            "",
            "## 15. split / merged cell warning 统计",
            "",
            f"- split_cell_warning：{warning_counts['split_cell_warning']}",
            f"- merged_cell_warning：{warning_counts['merged_cell_warning']}",
            f"- row_continuation_warning：{warning_counts['row_continuation_warning']}",
            "",
            "## 16. metric-level gap warning 统计",
            "",
            f"- metric_level_cell_gap：{warning_counts['metric_level_cell_gap']}",
            f"- numeric_column_order_uncertain：{warning_counts['numeric_column_order_uncertain']}",
            f"- missing_metric_cell_warning：{warning_counts['missing_metric_cell_warning']}",
            "",
            "## 17. binding warning 统计",
            "",
            f"- unit binding warning：{warning_counts['unit_binding_warning']}",
            f"- footnote binding warning：{warning_counts['footnote_binding_warning']}",
            f"- reference binding warning：{warning_counts['reference_binding_warning']}",
            "",
            "## 18. 是否复现 Phase7D 分流",
            "",
            "- 是，复现 2 ready / 3 rule_fix / 5 grid_rejected / 3 chunk_fallback / 3 backlog。",
            "",
            "## 19. 相比 Phase7D 的主要改善",
            "",
            "- rule-fix blocker 从人工说明固化为可重跑结构规则。",
            "- binding warning normalization 明确写入 v2.1 对象。",
            "- rule_fix_delta.csv 显式记录 routing 是否变化和剩余 blocker。",
            "",
            "## 20. 仍然存在的问题",
            "",
            "- 3 个 rule-fix case 仍没有完成 logical cell reconstruction。",
            "- value-level bbox 仍不存在。",
            "- ready_for_gold_candidate 仍需要后续单独 gold construction。",
            "",
            "## 21. 是否建议进入 gold construction",
            "",
            "- 不建议本轮进入；后续如授权，只能从 ready_candidate_pool 单独推进。",
            "",
            "## 22. 是否建议继续 pdfplumber 主线",
            "",
            "- 建议继续离线 hardening，但不接 production。",
            "",
            "## 23. 是否建议扩大 smoke",
            "",
            "- 不建议。",
            "",
            "## 24. 是否建议引入 Camelot / PyMuPDF",
            "",
            "- 不建议本轮引入。",
            "",
            "## 25. 是否建议进入 production",
            "",
            "- 不建议。",
            "",
            "## 26. baseline / guardrail 是否漂移",
            "",
            "- 未发现漂移。未修改 official dataset、official baseline、configs 或 baseline registry。",
            "",
            "## 27. Route C 是否仍只是 backlog",
            "",
            "- 是，Route C 仍只是 backlog。",
            "",
            "## 28. 明确未执行事项",
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
    (report_dir / "phase7d_2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    write_csv(diagnostics, output_dir / "rule_fix_diagnostics.csv", DIAGNOSTIC_FIELDS)
    write_csv(deltas, output_dir / "rule_fix_delta.csv", DELTA_FIELDS)
    write_jsonl(splits["ready"], output_dir / "ready_candidate_pool.jsonl")
    write_csv(splits["rule_fix"], output_dir / "rule_fix_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(splits["grid_rejected"], output_dir / "grid_rejected_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(splits["chunk_fallback"], output_dir / "chunk_fallback_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_csv(splits["backlog"], output_dir / "backlog_cases.csv", ROUTING_SUMMARY_FIELDS)
    write_guardrail(report_dir, inventory)
    write_rule_fix_implementation_report(report_dir, diagnostics)
    write_validation_report(objects, rows, diagnostics, report_dir)
    write_comparison_report(objects, rows, diagnostics, report_dir)
    write_summary(objects, rows, diagnostics, output_dir, report_dir)


def run(args: argparse.Namespace) -> None:
    required_paths = (
        PHASE7D_REQUIRED_INPUTS
        + PHASE7C4_REQUIRED_INPUTS
        + PHASE7C2_C3_REQUIRED_INPUTS
        + CURRENT_EXTRACTOR_SCRIPTS
        + PHASE6D_REQUIRED_INPUTS
    )
    inventory = read_input_inventory(required_paths, OPTIONAL_EXISTING_TESTS)
    inputs = load_inputs()
    objects, rows, diagnostics, deltas = build_v21_outputs(inputs)
    write_outputs(objects, rows, diagnostics, deltas, args.output_dir, args.report_dir, inventory)
    print(
        json.dumps(
            {
                "table_objects": len(objects),
                "routing_status": dict(Counter(row["routing_status"] for row in rows)),
                "final_action": dict(Counter(row["final_action"] for row in rows)),
                "rule_fix_diagnostics": len(diagnostics),
                "output_dir": rel(args.output_dir),
                "report_dir": rel(args.report_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Phase7D-2 hybrid extractor v2.1 rule fixes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
