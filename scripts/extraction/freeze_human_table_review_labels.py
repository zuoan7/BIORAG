#!/usr/bin/env python3
"""Freeze Phase7G human table review labels without modifying the source sheet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REVIEW_PACK_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_human_review_label_freeze"

REVIEW_DECISION_VALUES = {
    "accept_confirmed_seed_candidate",
    "accept_partial_seed_candidate",
    "reject_boundary",
    "reject_grid",
    "needs_rule_fix",
    "backlog",
    "skip",
}
CORE_OK_VALUES = {"yes", "no", "unclear"}
OPTIONAL_OK_VALUES = {"yes", "warning", "no", "not_applicable", "unchecked"}

CORE_HUMAN_FIELDS = ["review_decision", "boundary_ok", "grid_ok", "key_values_ok"]
OPTIONAL_HUMAN_FIELDS = ["unit_or_note_ok", "reference_ok"]
HUMAN_FIELDS = CORE_HUMAN_FIELDS + OPTIONAL_HUMAN_FIELDS + ["review_notes"]

FROZEN_FIELDS = [
    "candidate_id",
    "review_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "review_notes",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "risk_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
    "auto_score",
    "routing_status",
    "table_type_tags",
    "warnings_summary",
    "core_fields_complete",
    "label_status",
    "missing_core_fields",
    "invalid_fields",
    "contradiction_flags",
    "whitespace_normalized_fields",
    "enum_normalized_fields",
]

UNREVIEWED_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "missing_core_fields",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "risk_tags",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def strip_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_enum(value: Any) -> str:
    return strip_value(value).lower()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def by_candidate(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("candidate_id", ""): row for row in rows if row.get("candidate_id")}


def validate_row(row: dict[str, str]) -> list[str]:
    invalid: list[str] = []
    decision = row.get("review_decision", "")
    if decision and decision not in REVIEW_DECISION_VALUES:
        invalid.append("review_decision")
    for field in ["boundary_ok", "grid_ok", "key_values_ok"]:
        value = row.get(field, "")
        if value and value not in CORE_OK_VALUES:
            invalid.append(field)
    for field in OPTIONAL_HUMAN_FIELDS:
        value = row.get(field, "")
        if value and value not in OPTIONAL_OK_VALUES:
            invalid.append(field)
    return invalid


def contradiction_flags(row: dict[str, str]) -> list[str]:
    decision = row.get("review_decision", "")
    boundary = row.get("boundary_ok", "")
    grid = row.get("grid_ok", "")
    key_values = row.get("key_values_ok", "")
    flags: list[str] = []
    if decision == "accept_confirmed_seed_candidate" and (
        boundary != "yes" or grid != "yes" or key_values != "yes"
    ):
        flags.append("confirmed_candidate_without_yes_yes_yes")
    if decision == "accept_partial_seed_candidate" and boundary == "yes" and grid == "yes" and key_values == "yes":
        flags.append("partial_candidate_with_yes_yes_yes")
    if decision == "reject_boundary" and boundary == "yes":
        flags.append("reject_boundary_but_boundary_yes")
    if decision == "reject_grid" and grid == "yes":
        flags.append("reject_grid_but_grid_yes")
    if decision in {"backlog", "skip"} and boundary == "yes" and grid == "yes" and key_values == "yes":
        flags.append("deferred_but_core_all_yes")
    return flags


def normalize_review_row(
    row: dict[str, str],
    index_lookup: dict[str, dict[str, str]],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    normalized = dict(row)
    whitespace_fields: list[str] = []
    enum_fields: list[str] = []
    for field in HUMAN_FIELDS:
        raw = row.get(field, "")
        if field == "review_notes":
            value = strip_value(raw)
        else:
            value = normalize_enum(raw)
            if strip_value(raw) != value:
                enum_fields.append(field)
        if raw != value:
            whitespace_fields.append(field)
        normalized[field] = value

    index_row = index_lookup.get(normalized.get("candidate_id", ""), {})
    pool_row = pool_lookup.get(normalized.get("candidate_id", ""), {})
    for field in FROZEN_FIELDS:
        if field in normalized:
            continue
        normalized[field] = index_row.get(field, pool_row.get(field, ""))

    missing_core = [field for field in CORE_HUMAN_FIELDS if not normalized.get(field, "")]
    invalid = validate_row(normalized)
    conflicts = contradiction_flags(normalized)
    core_complete = not missing_core
    if invalid:
        label_status = "invalid_label"
    elif core_complete:
        label_status = "complete_review"
    else:
        label_status = "unreviewed"

    normalized["core_fields_complete"] = str(core_complete).lower()
    normalized["label_status"] = label_status
    normalized["missing_core_fields"] = ";".join(missing_core)
    normalized["invalid_fields"] = ";".join(invalid)
    normalized["contradiction_flags"] = ";".join(conflicts)
    normalized["whitespace_normalized_fields"] = ";".join(whitespace_fields)
    normalized["enum_normalized_fields"] = ";".join(enum_fields)
    return normalized


def inspect_xlsx_status(xlsx_path: Path, csv_complete_count: int, review_csv_path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "xlsx_path": rel(xlsx_path),
        "xlsx_exists": str(xlsx_path.exists()).lower(),
        "xlsx_complete_review_count": "",
        "xlsx_likely_not_updated": "unknown",
        "xlsx_mtime_before_csv": "",
    }
    if not xlsx_path.exists():
        status["xlsx_likely_not_updated"] = "not_present"
        return status

    status["xlsx_mtime_before_csv"] = str(xlsx_path.stat().st_mtime <= review_csv_path.stat().st_mtime).lower()
    try:
        rows = read_review_rows_from_minimal_xlsx(xlsx_path)
        xlsx_complete = sum(1 for row in rows if all(row.get(field, "") for field in CORE_HUMAN_FIELDS))
        status["xlsx_complete_review_count"] = str(xlsx_complete)
        status["xlsx_likely_not_updated"] = str(xlsx_complete < csv_complete_count).lower()
    except (KeyError, ET.ParseError, zipfile.BadZipFile):
        status["xlsx_likely_not_updated"] = "unreadable"
    return status


def read_review_rows_from_minimal_xlsx(path: Path) -> list[dict[str, str]]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(xml)
    sheet_rows = root.findall(".//main:sheetData/main:row", ns)
    parsed: list[list[str]] = []
    for sheet_row in sheet_rows:
        values: list[str] = []
        for cell in sheet_row.findall("main:c", ns):
            text_node = cell.find("main:is/main:t", ns)
            values.append(text_node.text if text_node is not None and text_node.text is not None else "")
        parsed.append(values)
    if not parsed:
        return []
    header = parsed[0]
    return [dict(zip(header, values)) for values in parsed[1:]]


def count_by(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row.get(field, "") or "empty") for row in rows)


def add_counter_rows(
    summary: list[dict[str, Any]],
    section: str,
    metric: str,
    counter: Counter[str],
    notes: str = "",
) -> None:
    for key, value in counter.most_common():
        summary.append({"section": section, "metric": metric, "key": key, "value": value, "notes": notes})


def build_audit_summary(
    frozen_rows: list[dict[str, Any]],
    review_rows_total: int,
    candidate_pool_total: int,
    xlsx_status: dict[str, Any],
) -> list[dict[str, Any]]:
    complete_rows = [row for row in frozen_rows if row["core_fields_complete"] == "true"]
    unreviewed_rows = [row for row in frozen_rows if row["core_fields_complete"] != "true"]
    invalid_rows = [row for row in frozen_rows if row["label_status"] == "invalid_label"]
    whitespace_rows = [row for row in frozen_rows if row.get("whitespace_normalized_fields")]
    summary: list[dict[str, Any]] = [
        {"section": "input", "metric": "review_rows_total", "key": "all", "value": review_rows_total, "notes": ""},
        {"section": "input", "metric": "candidate_pool_rows_total", "key": "all", "value": candidate_pool_total, "notes": ""},
        {
            "section": "review_completion",
            "metric": "complete_review_count",
            "key": "all",
            "value": len(complete_rows),
            "notes": "按四个核心字段非空定义",
        },
        {
            "section": "review_completion",
            "metric": "unreviewed_count",
            "key": "all",
            "value": len(unreviewed_rows),
            "notes": "包含任一核心字段为空的候选",
        },
        {
            "section": "review_completion",
            "metric": "possible_missing_30th_review",
            "key": "all",
            "value": str(len(complete_rows) == 29 and len(unreviewed_rows) >= 1).lower(),
            "notes": "若预期 30 条已审，则仅记录不阻断",
        },
        {
            "section": "label_quality",
            "metric": "invalid_label_count",
            "key": "all",
            "value": len(invalid_rows),
            "notes": "非法枚举进入 invalid_label",
        },
        {
            "section": "label_quality",
            "metric": "contradiction_count",
            "key": "all",
            "value": sum(1 for row in frozen_rows if row.get("contradiction_flags")),
            "notes": "review_decision 与 boundary/grid/key_values 的矛盾",
        },
        {
            "section": "normalization",
            "metric": "whitespace_normalized_row_count",
            "key": "all",
            "value": len(whitespace_rows),
            "notes": "至少一个人工字段被 strip 或枚举规范化",
        },
    ]
    for key, value in xlsx_status.items():
        summary.append({"section": "xlsx_sidecar", "metric": key, "key": "review_sheet", "value": value, "notes": "仅记录，不依赖 xlsx"})

    add_counter_rows(summary, "distribution_complete", "review_decision", count_by(complete_rows, "review_decision"))
    add_counter_rows(summary, "distribution_complete", "review_priority", count_by(complete_rows, "review_priority"))
    add_counter_rows(summary, "distribution_all", "review_priority", count_by(frozen_rows, "review_priority"))
    add_counter_rows(summary, "distribution_all", "unit_or_note_ok", count_by(frozen_rows, "unit_or_note_ok"))
    add_counter_rows(summary, "distribution_all", "reference_ok", count_by(frozen_rows, "reference_ok"))

    cross = Counter(
        f"{row.get('suggested_decision') or 'empty'} -> {row.get('review_decision') or 'empty'}"
        for row in complete_rows
    )
    add_counter_rows(summary, "cross_table_complete", "suggested_decision_vs_review_decision", cross)

    priority_decision = Counter(
        f"{row.get('review_priority') or 'empty'} -> {row.get('review_decision') or 'empty'}"
        for row in frozen_rows
    )
    add_counter_rows(summary, "priority_result_distribution", "review_priority_vs_review_decision", priority_decision)

    missing_by_priority = Counter(row.get("review_priority") or "empty" for row in unreviewed_rows)
    add_counter_rows(summary, "missing_core_fields", "unreviewed_by_priority", missing_by_priority)
    return summary


def freeze_review_labels(
    review_labels_path: Path,
    review_pack_index_path: Path,
    candidate_pool_path: Path,
    output_dir: Path,
    review_sheet_xlsx_path: Path | None = None,
) -> dict[str, Any]:
    review_labels_path = resolve(review_labels_path)
    review_pack_index_path = resolve(review_pack_index_path)
    candidate_pool_path = resolve(candidate_pool_path)
    output_dir = resolve(output_dir)
    review_sheet_xlsx_path = review_sheet_xlsx_path or review_labels_path.with_name("review_sheet.xlsx")
    review_sheet_xlsx_path = resolve(review_sheet_xlsx_path)

    review_rows = load_csv(review_labels_path)
    index_rows = load_csv(review_pack_index_path)
    pool_rows = load_csv(candidate_pool_path)
    index_lookup = by_candidate(index_rows)
    pool_lookup = by_candidate(pool_rows)

    frozen_rows = [normalize_review_row(row, index_lookup, pool_lookup) for row in review_rows]
    complete_count = sum(1 for row in frozen_rows if row["core_fields_complete"] == "true")
    xlsx_status = inspect_xlsx_status(review_sheet_xlsx_path, complete_count, review_labels_path)
    audit_rows = build_audit_summary(frozen_rows, len(review_rows), len(pool_rows), xlsx_status)
    unreviewed_rows = [row for row in frozen_rows if row["core_fields_complete"] != "true"]

    write_csv(frozen_rows, output_dir / "frozen_review_labels.csv", FROZEN_FIELDS)
    write_jsonl(frozen_rows, output_dir / "frozen_review_labels.jsonl")
    write_csv(audit_rows, output_dir / "label_audit_summary.csv", ["section", "metric", "key", "value", "notes"])
    write_csv(unreviewed_rows, output_dir / "unreviewed_candidates.csv", UNREVIEWED_FIELDS)

    return {
        "frozen_rows": frozen_rows,
        "audit_rows": audit_rows,
        "complete_review_count": complete_count,
        "unreviewed_count": len(unreviewed_rows),
        "invalid_label_count": sum(1 for row in frozen_rows if row["label_status"] == "invalid_label"),
        "whitespace_normalized_row_count": sum(1 for row in frozen_rows if row.get("whitespace_normalized_fields")),
        "xlsx_status": xlsx_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-labels", type=Path, default=DEFAULT_REVIEW_PACK_DIR / "review_labels_template.csv")
    parser.add_argument("--review-pack-index", type=Path, default=DEFAULT_REVIEW_PACK_DIR / "review_pack_index.csv")
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_REVIEW_PACK_DIR / "candidate_pool_scored.csv")
    parser.add_argument("--review-sheet-xlsx", type=Path, default=DEFAULT_REVIEW_PACK_DIR / "review_sheet.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_review_labels(
        review_labels_path=args.review_labels,
        review_pack_index_path=args.review_pack_index,
        candidate_pool_path=args.candidate_pool,
        output_dir=args.output_dir,
        review_sheet_xlsx_path=args.review_sheet_xlsx,
    )
    print(
        "frozen_review_labels: "
        f"complete={result['complete_review_count']} "
        f"unreviewed={result['unreviewed_count']} "
        f"invalid={result['invalid_label_count']}"
    )


if __name__ == "__main__":
    main()
