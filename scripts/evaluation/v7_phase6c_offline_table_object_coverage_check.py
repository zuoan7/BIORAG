#!/usr/bin/env python3
"""Minimal isolated offline table_object coverage check for BIORAG v7-phase6C-5."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

TABLE_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase6c_table_object_expanded_sample/table_objects.jsonl"
)
ROW_CELL_GOLD_REFINED_PATH = (
    ROOT
    / "data/experiments/v7_phase6c_table_object_expanded_sample/row_cell_gold_refined.jsonl"
)
GOLD_REFINEMENT_SUMMARY_PATH = (
    ROOT
    / "reports/v7_phase6c_expanded_table_object_offline_sample/gold_refinement_summary.csv"
)
OUTPUT_DIR = ROOT / "results/v7_phase6c_table_object_expanded_sample"
OUTPUT_JSON = OUTPUT_DIR / "offline_coverage_check_results.json"
OUTPUT_CSV = OUTPUT_DIR / "offline_coverage_check_results.csv"

OFFICIAL_BASELINE_NAME = "phase5f_official_clean_baseline"
OFFICIAL_DATASET_SHA256 = (
    "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
)

CONFIRMED_GOLD_IDS = {
    "gold_doc_0687_table2_p5c3_0003",
    "gold_doc_0458_table3_p5c5_0071",
}
PARTIAL_EXPLORATORY_GOLD_IDS = {
    "gold_doc_0687_table3_p5c3_0001",
    "gold_doc_0424_table4_p5c3_0012",
}
GOLD_ORDER = [
    "gold_doc_0687_table2_p5c3_0003",
    "gold_doc_0458_table3_p5c5_0071",
    "gold_doc_0687_table3_p5c3_0001",
    "gold_doc_0424_table4_p5c3_0012",
]

COVERAGE_FIELDS = [
    "table_object_source_coverage",
    "row_gold_coverage",
    "cell_gold_coverage",
    "value_coverage",
    "unit_binding_coverage",
    "footnote_reference_coverage",
    "source_span_coverage",
    "evidence_completeness",
    "answerability_calibration",
]
OUTPUT_FIELDS = [
    "gold_id",
    "gold_status",
    "subset",
    "table_object_id",
    "sample_id",
    *COVERAGE_FIELDS,
    "coverage_status",
    "warnings",
    "notes",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_refinement_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["gold_id"]: row for row in csv.DictReader(handle)}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def normalize_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_text(value).rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def scalar_matches(left: Any, right: Any) -> bool:
    left_number = normalize_number(left)
    right_number = normalize_number(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) < 1e-9
    return normalize_text(left) == normalize_text(right)


def value_matches(required_value: Any, candidate_value: Any) -> bool:
    if isinstance(required_value, list):
        if not isinstance(candidate_value, list):
            return False
        if len(required_value) != len(candidate_value):
            return False
        return all(scalar_matches(left, right) for left, right in zip(required_value, candidate_value))

    if isinstance(candidate_value, list):
        return any(scalar_matches(required_value, item) for item in candidate_value)

    return scalar_matches(required_value, candidate_value)


def raw_value_in_candidate(required_raw: Any, candidate_raw: Any) -> bool:
    required = normalize_text(required_raw)
    candidate = normalize_text(candidate_raw)
    if not required or not candidate:
        return False
    if required == candidate:
        return True
    if re.fullmatch(r"[<>]?\d+(?:\.\d+)?%?", required):
        pattern = rf"(?<![\d.]){re.escape(required)}(?![\d.])"
        return re.search(pattern, candidate) is not None
    return required in candidate


def source_span_ids(items: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("source_span_id")) for item in items if item.get("source_span_id")}


def item_ids(items: list[dict[str, Any]], key: str) -> set[str]:
    return {str(item.get(key)) for item in items if item.get(key)}


def has_source_span_ref(item: dict[str, Any], table_source_span_ids: set[str]) -> bool:
    source_span = item.get("source_span")
    return isinstance(source_span, str) and source_span in table_source_span_ids


def status_has_uncertainty(value: Any) -> bool:
    text = normalize_text(value).lower()
    if not text or "not_applicable" in text:
        return False
    markers = (
        "uncertain",
        "partial",
        "not_bound",
        "not safely",
        "requires",
        "require ",
        "low",
    )
    return any(marker in text for marker in markers)


def coverage_from_counts(total: int, covered: int, uncertain: bool = False) -> str:
    if total == 0:
        return "not_applicable"
    if covered == total and not uncertain:
        return "covered"
    if covered > 0:
        return "partially_covered"
    if uncertain:
        return "uncertain"
    return "not_covered"


def subset_for_gold(gold: dict[str, Any]) -> str:
    gold_id = gold.get("gold_id")
    if gold_id in CONFIRMED_GOLD_IDS:
        return "confirmed"
    if gold_id in PARTIAL_EXPLORATORY_GOLD_IDS:
        return "partial_exploratory"
    raise ValueError(f"gold_id outside C-5 scope: {gold_id}")


def candidates_for_cell(
    required_cell: dict[str, Any], table_cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    exact = [
        cell for cell in table_cells if cell.get("cell_id") == required_cell.get("cell_id")
    ]
    if exact:
        return exact
    row_index = required_cell.get("row_index")
    return [cell for cell in table_cells if cell.get("row_index") == row_index]


def cell_value_matches(required_cell: dict[str, Any], table_cell: dict[str, Any]) -> bool:
    raw_ok = raw_value_in_candidate(required_cell.get("value_raw"), table_cell.get("value_raw"))
    normalized_ok = value_matches(
        required_cell.get("value_normalized"), table_cell.get("value_normalized")
    )
    return raw_ok or normalized_ok


def evaluate_table_object_source(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    if table_object is None:
        notes.append("缺少对应 table_object，无法评估 table/source/object 覆盖。")
        return "not_covered"

    checks = [
        table_object.get("table_object_id") == gold.get("table_object_id"),
        gold.get("sample_id") in table_object.get("sample_ids", []),
        table_object.get("baseline_name") == OFFICIAL_BASELINE_NAME,
        table_object.get("dataset_sha256") == OFFICIAL_DATASET_SHA256,
        bool(table_object.get("source_spans")),
        bool(table_object.get("source_block_ids")),
        bool(table_object.get("chunk_ids")),
        table_object.get("table_boundary_status") == "frozen",
        table_object.get("source_relation_confidence") in {"high", "medium"},
    ]
    if all(checks):
        return "covered"
    if any(checks):
        notes.append("table_object 的 source/object pin 或 provenance 字段存在部分缺口。")
        return "partially_covered"
    return "not_covered"


def evaluate_rows(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_rows = gold.get("required_rows", [])
    if not required_rows:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_rows = {row.get("row_id"): row for row in table_object.get("rows", [])}
    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    covered = 0
    missing: list[str] = []
    for row in required_rows:
        table_row = table_rows.get(row.get("row_id"))
        if table_row and row.get("source_span_id") in table_source_ids and has_source_span_ref(
            table_row, table_source_ids
        ):
            covered += 1
        else:
            missing.append(str(row.get("row_id")))
    if missing:
        notes.append(f"row_gold_coverage 缺口：{','.join(missing)}。")
    return coverage_from_counts(len(required_rows), covered)


def evaluate_cells(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = gold.get("required_cells", [])
    if not required_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = table_object.get("cells", [])
    exact_ids = item_ids(table_cells, "cell_id")
    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    covered = 0
    fallback_used = 0
    missing: list[str] = []

    for required_cell in required_cells:
        cell_id = required_cell.get("cell_id")
        candidates = candidates_for_cell(required_cell, table_cells)
        exact_match = cell_id in exact_ids
        matched = any(
            cell_value_matches(required_cell, candidate)
            and has_source_span_ref(candidate, table_source_ids)
            for candidate in candidates
        )
        if matched and exact_match:
            covered += 1
        elif matched:
            covered += 1
            fallback_used += 1
        else:
            missing.append(str(cell_id))

    if fallback_used:
        notes.append(
            f"{fallback_used} 个 required cells 只能由 row-level flattened cell 表达，未形成 metric-level cell。"
        )
    if missing:
        notes.append(f"cell_gold_coverage 缺口：{','.join(missing)}。")
    return coverage_from_counts(len(required_cells), covered, uncertain=fallback_used > 0)


def evaluate_values(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = gold.get("required_cells", [])
    if not required_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = table_object.get("cells", [])
    covered = 0
    missing: list[str] = []
    for required_cell in required_cells:
        candidates = candidates_for_cell(required_cell, table_cells)
        if any(cell_value_matches(required_cell, candidate) for candidate in candidates):
            covered += 1
        else:
            missing.append(str(required_cell.get("cell_id")))

    if missing:
        notes.append(f"value_coverage 缺口：{','.join(missing)}。")
    return coverage_from_counts(len(required_cells), covered)


def evaluate_units(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_units = gold.get("required_units", [])
    required_cells = gold.get("required_cells", [])
    unit_cells = [cell for cell in required_cells if cell.get("unit")]
    if not required_units and not unit_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = table_object.get("cells", [])
    table_columns = table_object.get("columns", [])
    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    visible_unit_source = any(
        unit.get("source_span_id") in table_source_ids for unit in required_units
    ) or any(column.get("unit") for column in table_columns)
    uncertain = any(
        status_has_uncertainty(unit.get("binding_status"))
        or status_has_uncertainty(unit.get("scope"))
        or normalize_text(unit.get("confidence")).lower() not in {"", "high"}
        for unit in required_units
    )
    uncertain = uncertain or "unit_scope_uncertain" in table_object.get("warnings", [])
    uncertain = uncertain or "unit_uncertain" in gold.get("warnings", [])

    covered = 0
    for required_cell in unit_cells:
        candidates = candidates_for_cell(required_cell, table_cells)
        if any(scalar_matches(required_cell.get("unit"), candidate.get("unit")) for candidate in candidates):
            covered += 1

    if unit_cells:
        if covered != len(unit_cells):
            notes.append(
                f"unit_binding_coverage 不完整：{covered}/{len(unit_cells)} 个 required cell units 绑定。"
            )
        if covered == len(unit_cells):
            return coverage_from_counts(len(unit_cells), covered, uncertain=uncertain)
        if visible_unit_source:
            return "partially_covered"
        return coverage_from_counts(len(unit_cells), covered, uncertain=uncertain)

    if visible_unit_source and uncertain:
        notes.append("unit source 可见，但 refined gold 或 table_object 仍标注 unit binding 不确定。")
        return "partially_covered"
    if visible_unit_source:
        return "covered"
    return "not_covered"


def evaluate_footnotes_references(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_footnotes = gold.get("required_footnotes", [])
    required_references = gold.get("required_references", [])
    total = len(required_footnotes) + len(required_references)
    if total == 0:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    table_footnotes = {
        footnote.get("footnote_id"): footnote for footnote in table_object.get("footnotes", [])
    }
    table_references = {
        reference.get("reference_id"): reference for reference in table_object.get("references", [])
    }
    covered = 0
    uncertain = False
    missing: list[str] = []
    table_warnings = set(table_object.get("warnings", []))
    if required_footnotes and "footnote_binding_uncertain" in table_warnings:
        uncertain = True
    if required_references and "reference_binding_uncertain" in table_warnings:
        uncertain = True

    for footnote in required_footnotes:
        table_footnote = table_footnotes.get(footnote.get("footnote_id"))
        binding_status = footnote.get("binding_status")
        if table_footnote and footnote.get("source_span_id") in table_source_ids:
            covered += 1
        else:
            missing.append(str(footnote.get("footnote_id")))
        if status_has_uncertainty(binding_status):
            uncertain = True

    for reference in required_references:
        table_reference = table_references.get(reference.get("reference_id"))
        if table_reference and reference.get("source_span_id") in table_source_ids:
            covered += 1
        else:
            missing.append(str(reference.get("reference_id")))
        if status_has_uncertainty(reference.get("binding_status")):
            uncertain = True

    if missing:
        notes.append(f"footnote_reference_coverage 缺口：{','.join(missing)}。")
    if uncertain:
        notes.append("footnote/reference binding 含 partial 或 uncertain 标注，不能升级为 fully covered。")
    return coverage_from_counts(total, covered, uncertain=uncertain)


def evaluate_source_spans(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    if table_object is None:
        return "not_covered"

    gold_span_ids = source_span_ids(gold.get("source_spans", []))
    table_span_ids = source_span_ids(table_object.get("source_spans", []))
    evidence_items: list[dict[str, Any]] = []
    evidence_items.extend(gold.get("required_rows", []))
    evidence_items.extend(gold.get("required_cells", []))
    evidence_items.extend(gold.get("required_units", []))
    evidence_items.extend(gold.get("required_footnotes", []))
    evidence_items.extend(gold.get("required_references", []))
    if not evidence_items:
        return "not_applicable"

    covered = 0
    missing: list[str] = []
    for item in evidence_items:
        span_id = item.get("source_span_id")
        if span_id in gold_span_ids and span_id in table_span_ids:
            covered += 1
        else:
            missing.append(str(span_id))
    if missing:
        notes.append(f"source_span_coverage 缺口：{','.join(missing)}。")
    return coverage_from_counts(len(evidence_items), covered)


def evaluate_answerability(
    gold: dict[str, Any],
    subset: str,
    refinement_row: dict[str, str] | None,
    notes: list[str],
) -> str:
    if refinement_row is None:
        notes.append("缺少 gold_refinement_summary.csv 对应记录。")
        return "uncertain"

    gold_status = gold.get("gold_status")
    refined_status = refinement_row.get("refined_gold_status")
    consistency_status = refinement_row.get("consistency_status")
    blocking_warnings = normalize_text(refinement_row.get("blocking_warnings_remaining"))

    if consistency_status == "fail":
        notes.append("refined gold consistency_status=fail，不能评估。")
        return "not_covered"
    if subset == "confirmed":
        required_true_fields = [
            "has_expected_answer",
            "has_required_cells",
            "has_source_spans",
            "has_key_cell_source_span",
            "has_required_units_if_needed",
            "has_required_footnotes_if_needed",
            "has_required_references_if_needed",
        ]
        if gold_status == "confirmed_gold" and refined_status == "confirmed_gold" and not blocking_warnings:
            if all(refinement_row.get(field) == "true" for field in required_true_fields):
                return "covered"
        notes.append("confirmed answerability calibration 不满足 refined summary 条件。")
        return "uncertain"

    if subset == "partial_exploratory" and gold_status == "partial_gold" and refined_status == "partial_gold":
        return "covered"

    notes.append("subset 与 refined gold_status 不一致。")
    return "not_covered"


def evidence_completeness(metrics: dict[str, str], warnings: list[str], subset: str) -> str:
    core_metrics = [
        "table_object_source_coverage",
        "row_gold_coverage",
        "cell_gold_coverage",
        "value_coverage",
        "source_span_coverage",
    ]
    if any(metrics[name] == "not_covered" for name in core_metrics):
        return "not_covered"
    if any(metrics[name] in {"partially_covered", "uncertain"} for name in core_metrics):
        return "partially_covered"

    optional_metrics = ["unit_binding_coverage", "footnote_reference_coverage"]
    if any(metrics[name] == "not_covered" for name in optional_metrics):
        return "not_covered"
    if any(metrics[name] in {"partially_covered", "uncertain"} for name in optional_metrics):
        return "partially_covered"

    if subset == "confirmed" and warnings:
        return "covered_with_minor_warnings"
    return "covered"


def coverage_status(metrics: dict[str, str], subset: str) -> str:
    if metrics["table_object_source_coverage"] == "not_covered":
        return "not_evaluable"
    if metrics["evidence_completeness"] == "not_covered":
        return "fail"
    if metrics["evidence_completeness"] == "partially_covered":
        return "partial"
    if subset == "partial_exploratory":
        return "partial"
    if metrics["evidence_completeness"] == "covered_with_minor_warnings":
        return "pass_with_warnings"
    return "pass"


def unique_warnings(*warning_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for warnings in warning_lists:
        for warning in warnings:
            if warning and warning not in seen:
                seen.add(warning)
                ordered.append(warning)
    return ordered


def evaluate_gold(
    gold: dict[str, Any],
    table_objects: dict[str, dict[str, Any]],
    refinement_summary: dict[str, dict[str, str]],
) -> dict[str, Any]:
    subset = subset_for_gold(gold)
    table_object = table_objects.get(gold.get("table_object_id"))
    refinement_row = refinement_summary.get(gold.get("gold_id"))
    notes: list[str] = []

    metrics = {
        "table_object_source_coverage": evaluate_table_object_source(gold, table_object, notes),
        "row_gold_coverage": evaluate_rows(gold, table_object, notes),
        "cell_gold_coverage": evaluate_cells(gold, table_object, notes),
        "value_coverage": evaluate_values(gold, table_object, notes),
        "unit_binding_coverage": evaluate_units(gold, table_object, notes),
        "footnote_reference_coverage": evaluate_footnotes_references(gold, table_object, notes),
        "source_span_coverage": evaluate_source_spans(gold, table_object, notes),
        "answerability_calibration": evaluate_answerability(
            gold, subset, refinement_row, notes
        ),
    }
    warnings = unique_warnings(
        gold.get("warnings", []),
        table_object.get("warnings", []) if table_object else [],
    )
    metrics["evidence_completeness"] = evidence_completeness(metrics, warnings, subset)
    status = coverage_status(metrics, subset)

    if subset == "partial_exploratory":
        notes.append("partial_gold 仅作为 exploratory coverage observation，不进入 formal conclusion。")
    if warnings:
        notes.append("warnings 保留为 coverage 风险说明，不作为 retrieval 或 QA 分数。")

    return {
        "gold_id": gold.get("gold_id"),
        "gold_status": gold.get("gold_status"),
        "subset": subset,
        "table_object_id": gold.get("table_object_id"),
        "sample_id": gold.get("sample_id"),
        **metrics,
        "coverage_status": status,
        "warnings": warnings,
        "notes": "；".join(notes),
    }


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["warnings"] = ";".join(record.get("warnings", []))
            writer.writerow(row)


def confirmed_c5_pass(records: list[dict[str, Any]]) -> bool:
    confirmed = [record for record in records if record["subset"] == "confirmed"]
    if len(confirmed) != 2:
        return False
    for record in confirmed:
        if record["coverage_status"] in {"fail", "not_evaluable"}:
            return False
        for field in (
            "table_object_source_coverage",
            "row_gold_coverage",
            "cell_gold_coverage",
            "value_coverage",
            "source_span_coverage",
        ):
            if record[field] != "covered":
                return False
        if record["evidence_completeness"] not in {"covered", "covered_with_minor_warnings"}:
            return False
    return True


def main() -> None:
    table_objects_list = load_jsonl(TABLE_OBJECTS_PATH)
    gold_rows = load_jsonl(ROW_CELL_GOLD_REFINED_PATH)
    refinement_summary = load_refinement_summary(GOLD_REFINEMENT_SUMMARY_PATH)

    table_objects = {item["table_object_id"]: item for item in table_objects_list}
    gold_by_id = {item["gold_id"]: item for item in gold_rows}
    missing = [gold_id for gold_id in GOLD_ORDER if gold_id not in gold_by_id]
    if missing:
        raise SystemExit(f"Missing expected C-5 gold rows: {', '.join(missing)}")

    records = [
        evaluate_gold(gold_by_id[gold_id], table_objects, refinement_summary)
        for gold_id in GOLD_ORDER
    ]
    write_json(OUTPUT_JSON, records)
    write_csv(OUTPUT_CSV, records)

    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"confirmed_subset_c5_pass={str(confirmed_c5_pass(records)).lower()}")


if __name__ == "__main__":
    main()
