#!/usr/bin/env python3
"""Phase7F gold seed validation for hybrid extractor v2.2.

This is a seed-level offline checker. It reads frozen Phase7E gold seed records
and frozen Phase7D-3 table_objects, then validates only the confirmed seed
formal subset. Partial seeds are emitted as exploratory observations and are not
included in the formal overall status.
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


GOLD_SEED_DIR = ROOT / "data/experiments/v7_phase7_hybrid_gold_seed"
EXTRACTOR_DIR = ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_logical_reconstruction"
DEFAULT_OUTPUT_DIR = ROOT / "results/v7_phase7_gold_seed_validation"

FORMAL_CONFIRMED_SEED_IDS = [
    "phase7e_gold_seed_002__doc_0687__table_2__phase7c2_hybrid_02",
    "phase7e_gold_seed_004__doc_0523__table_1__phase7c2_hybrid_01",
]

PARTIAL_EXPLORATORY_SEED_IDS = [
    "phase7e_gold_seed_001__doc_0468__table_2__phase7c2_hybrid_01",
    "phase7e_gold_seed_003__doc_0687__table_3__phase7c2_hybrid_03",
]

PROD_READY_MARKER = "production" + "_ready"

COVERAGE_VALUES = {
    "covered",
    "covered_with_warnings",
    "partially_covered",
    "not_covered",
    "not_applicable",
    "not_evaluable",
}

OVERALL_STATUS_VALUES = {"pass", "pass_with_warnings", "partial", "fail", "not_evaluable"}

RESULT_FIELDS = [
    "gold_seed_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "validation_subset",
    "row_coverage",
    "column_coverage",
    "cell_coverage",
    "required_value_coverage",
    "unit_binding_coverage",
    "footnote_binding_coverage",
    "reference_binding_coverage",
    "literal_preservation_coverage",
    "source_span_coverage",
    "bbox_provenance_check",
    "overall_validation_status",
    "required_values_total",
    "required_values_matched",
    "gold_cells_total",
    "gold_cells_matched",
    "warnings",
    "missing_items",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def split_semicolon(value: str | None) -> list[str]:
    if not value or value == "none":
        return []
    return [part for part in value.split(";") if part]


def load_inputs(
    gold_seed_path: Path = GOLD_SEED_DIR / "table_gold_seed.jsonl",
    confirmed_seed_path: Path = GOLD_SEED_DIR / "confirmed_seed.jsonl",
    partial_seed_path: Path = GOLD_SEED_DIR / "partial_seed.csv",
    table_objects_path: Path = EXTRACTOR_DIR / "table_objects.jsonl",
) -> dict[str, Any]:
    seeds = read_jsonl(gold_seed_path)
    confirmed = read_jsonl(confirmed_seed_path)
    partial = read_csv(partial_seed_path)
    table_objects = read_jsonl(table_objects_path)

    confirmed_ids = [seed["gold_seed_id"] for seed in confirmed]
    partial_ids = [row["gold_seed_id"] for row in partial]
    if confirmed_ids != FORMAL_CONFIRMED_SEED_IDS:
        raise ValueError(f"Unexpected confirmed seed ids: {confirmed_ids}")
    if partial_ids != PARTIAL_EXPLORATORY_SEED_IDS:
        raise ValueError(f"Unexpected partial seed ids: {partial_ids}")

    return {
        "seeds": seeds,
        "confirmed_seed_ids": confirmed_ids,
        "partial_seed_ids": partial_ids,
        "partial_seed_rows": partial,
        "table_objects": table_objects,
    }


def logical_cell_index(table_object: dict[str, Any] | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not table_object:
        return index
    for cell in table_object.get("logical_cells") or []:
        key = (str(cell.get("row_key")), str(cell.get("logical_column")))
        index.setdefault(key, []).append(cell)
    return index


def find_matching_cell(
    item: dict[str, Any],
    index: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    require_value: bool = True,
) -> dict[str, Any] | None:
    candidates = index.get((str(item.get("row_key")), str(item.get("logical_column"))), [])
    if not require_value:
        return candidates[0] if candidates else None
    expected = normalize_text(item.get("value_raw"))
    for cell in candidates:
        if normalize_text(cell.get("value_raw")) == expected:
            return cell
    return None


def coverage_from_counts(matched: int, total: int) -> str:
    if total == 0:
        return "not_applicable"
    if matched == total:
        return "covered"
    if matched == 0:
        return "not_covered"
    return "partially_covered"


def status_has_warning(status: str | None) -> bool:
    return "warning" in normalize_text(status).lower()


def append_warning(warnings: list[str], value: str) -> None:
    if value and value not in warnings:
        warnings.append(value)


def validate_rows(seed: dict[str, Any], table_object: dict[str, Any] | None, missing_items: list[dict[str, Any]]) -> str:
    if not table_object:
        return "not_evaluable"
    expected = {str(row.get("row_key")) for row in seed.get("gold_rows") or []}
    actual = {str(row.get("row_key")) for row in table_object.get("logical_rows") or []}
    missing = sorted(expected - actual)
    for row_key in missing:
        missing_items.append({"kind": "gold_row", "row_key": row_key})
    return coverage_from_counts(len(expected) - len(missing), len(expected))


def validate_columns(seed: dict[str, Any], table_object: dict[str, Any] | None, missing_items: list[dict[str, Any]]) -> str:
    if not table_object:
        return "not_evaluable"
    expected = {str(column.get("column_key")) for column in seed.get("gold_columns") or []}
    actual = {str(column) for column in table_object.get("logical_columns") or []}
    missing = sorted(expected - actual)
    for column_key in missing:
        missing_items.append({"kind": "gold_column", "logical_column": column_key})
    return coverage_from_counts(len(expected) - len(missing), len(expected))


def validate_cells(
    seed: dict[str, Any],
    table_object: dict[str, Any] | None,
    missing_items: list[dict[str, Any]],
) -> tuple[str, int, list[dict[str, Any]]]:
    if not table_object:
        return "not_evaluable", 0, []
    index = logical_cell_index(table_object)
    mappings: list[dict[str, Any]] = []
    for gold_cell in seed.get("gold_cells") or []:
        match = find_matching_cell(gold_cell, index)
        if match:
            mappings.append(
                {
                    "gold_cell_id": gold_cell.get("gold_cell_id"),
                    "row_key": gold_cell.get("row_key"),
                    "logical_column": gold_cell.get("logical_column"),
                    "value_raw": gold_cell.get("value_raw"),
                    "logical_cell_id": match.get("logical_cell_id"),
                }
            )
        else:
            missing_items.append(
                {
                    "kind": "gold_cell",
                    "row_key": gold_cell.get("row_key"),
                    "logical_column": gold_cell.get("logical_column"),
                    "value_raw": gold_cell.get("value_raw"),
                }
            )
    total = len(seed.get("gold_cells") or [])
    return coverage_from_counts(len(mappings), total), len(mappings), mappings


def validate_required_values(
    seed: dict[str, Any],
    table_object: dict[str, Any] | None,
    missing_items: list[dict[str, Any]],
) -> tuple[str, int, list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    if not table_object:
        return "not_evaluable", 0, [], {}
    index = logical_cell_index(table_object)
    mappings: list[dict[str, Any]] = []
    matches_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for required in seed.get("required_values") or []:
        match = find_matching_cell(required, index)
        key = (
            str(required.get("row_key")),
            str(required.get("logical_column")),
            normalize_text(required.get("value_raw")),
        )
        if match:
            mapping = {
                "row_key": required.get("row_key"),
                "logical_column": required.get("logical_column"),
                "value_raw": required.get("value_raw"),
                "logical_cell_id": match.get("logical_cell_id"),
                "source_span_ids": match.get("source_span_ids") or [],
            }
            mappings.append(mapping)
            matches_by_key[key] = match
        else:
            missing_items.append(
                {
                    "kind": "required_value",
                    "row_key": required.get("row_key"),
                    "logical_column": required.get("logical_column"),
                    "value_raw": required.get("value_raw"),
                }
            )
    total = len(seed.get("required_values") or [])
    return coverage_from_counts(len(mappings), total), len(mappings), mappings, matches_by_key


def validate_units(
    seed: dict[str, Any],
    matches_by_key: dict[tuple[str, str, str], dict[str, Any]],
    missing_items: list[dict[str, Any]],
) -> str:
    required_units = seed.get("required_units") or []
    expected_values = [
        value
        for value in seed.get("required_values") or []
        if normalize_text(value.get("unit")) and normalize_text(value.get("unit")) != "not_applicable"
    ]
    if not required_units and not expected_values:
        return "not_applicable"

    missing = 0
    for value in expected_values:
        key = (str(value.get("row_key")), str(value.get("logical_column")), normalize_text(value.get("value_raw")))
        match = matches_by_key.get(key)
        if not match or normalize_text(match.get("unit")) != normalize_text(value.get("unit")):
            missing += 1
            missing_items.append(
                {
                    "kind": "unit_binding",
                    "row_key": value.get("row_key"),
                    "logical_column": value.get("logical_column"),
                    "expected_unit": value.get("unit"),
                    "value_raw": value.get("value_raw"),
                }
            )

    coverage = coverage_from_counts(len(expected_values) - missing, len(expected_values))
    if coverage == "covered" and any(status_has_warning(unit.get("binding_status")) for unit in required_units):
        return "covered_with_warnings"
    return coverage


def validate_footnotes(
    seed: dict[str, Any],
    matches_by_key: dict[tuple[str, str, str], dict[str, Any]],
    missing_items: list[dict[str, Any]],
) -> str:
    binding = seed.get("footnote_binding") or {}
    if binding.get("binding_status") == "not_applicable":
        return "not_applicable"

    values_with_footnotes = [value for value in seed.get("required_values") or [] if value.get("footnote_refs")]
    if not values_with_footnotes:
        return "covered_with_warnings" if status_has_warning(binding.get("binding_status")) else "not_applicable"

    missing = 0
    for value in values_with_footnotes:
        key = (str(value.get("row_key")), str(value.get("logical_column")), normalize_text(value.get("value_raw")))
        match = matches_by_key.get(key)
        expected = set(value.get("footnote_refs") or [])
        actual = set((match or {}).get("footnote_refs") or [])
        if not match or not expected <= actual:
            missing += 1
            missing_items.append(
                {
                    "kind": "footnote_binding",
                    "row_key": value.get("row_key"),
                    "logical_column": value.get("logical_column"),
                    "expected_footnote_refs": sorted(expected),
                    "value_raw": value.get("value_raw"),
                }
            )
    coverage = coverage_from_counts(len(values_with_footnotes) - missing, len(values_with_footnotes))
    if coverage == "covered" and status_has_warning(binding.get("binding_status")):
        return "covered_with_warnings"
    return coverage


def validate_references(
    seed: dict[str, Any],
    table_object: dict[str, Any] | None,
    missing_items: list[dict[str, Any]],
) -> str:
    binding = seed.get("reference_binding") or {}
    if binding.get("binding_status") == "not_applicable":
        return "not_applicable"
    row_reference_map = binding.get("row_reference_map") or {}
    if not row_reference_map:
        return "not_applicable"
    if not table_object:
        return "not_evaluable"

    index = logical_cell_index(table_object)
    missing = 0
    for row_key, expected_reference in row_reference_map.items():
        reference_cells = index.get((str(row_key), "Reference"), []) + index.get((str(row_key), "reference_or_source"), [])
        reference_match = any(normalize_text(cell.get("value_raw")) == normalize_text(expected_reference) for cell in reference_cells)
        row_cells = [cell for cell in table_object.get("logical_cells") or [] if str(cell.get("row_key")) == str(row_key)]
        inherited_match = any(
            normalize_text(cell.get("reference_or_source")) == normalize_text(expected_reference) for cell in row_cells
        )
        if not reference_match and not inherited_match:
            missing += 1
            missing_items.append(
                {
                    "kind": "reference_binding",
                    "row_key": row_key,
                    "expected_reference": expected_reference,
                }
            )
    coverage = coverage_from_counts(len(row_reference_map) - missing, len(row_reference_map))
    if coverage == "covered" and status_has_warning(binding.get("binding_status")):
        return "covered_with_warnings"
    return coverage


def validate_literal_preservation(
    seed: dict[str, Any],
    required_value_coverage: str,
    missing_items: list[dict[str, Any]],
) -> str:
    values = seed.get("required_values") or []
    missing_raw = [
        {"kind": "literal_value_raw", "row_key": value.get("row_key"), "logical_column": value.get("logical_column")}
        for value in values
        if normalize_text(value.get("value_raw")) == ""
    ]
    missing_items.extend(missing_raw)
    if missing_raw:
        return "not_covered"
    if required_value_coverage not in {"covered", "covered_with_warnings"}:
        return required_value_coverage
    status = (seed.get("literal_preservation") or {}).get("status")
    return "covered_with_warnings" if status_has_warning(status) else "covered"


def validate_source_span(
    seed: dict[str, Any],
    table_object: dict[str, Any] | None,
    matches_by_key: dict[tuple[str, str, str], dict[str, Any]],
    missing_items: list[dict[str, Any]],
) -> str:
    if not table_object:
        return "not_evaluable"
    if seed.get("source_span_granularity") == "value_level" or table_object.get("source_span_granularity") == "value_level":
        missing_items.append({"kind": "source_span_granularity", "reason": "value_level_not_available_in_phase7f"})
        return "not_covered"

    missing = 0
    warning = False
    for value in seed.get("required_values") or []:
        key = (str(value.get("row_key")), str(value.get("logical_column")), normalize_text(value.get("value_raw")))
        match = matches_by_key.get(key)
        expected = set(value.get("source_span_ids") or [])
        actual = set((match or {}).get("source_span_ids") or [])
        if not match or not expected or not actual or not expected <= actual:
            missing += 1
            missing_items.append(
                {
                    "kind": "source_span",
                    "row_key": value.get("row_key"),
                    "logical_column": value.get("logical_column"),
                    "value_raw": value.get("value_raw"),
                }
            )
        if match and match.get("source_span_granularity") != seed.get("source_span_granularity"):
            warning = True
    coverage = coverage_from_counts(len(seed.get("required_values") or []) - missing, len(seed.get("required_values") or []))
    if coverage == "covered" and warning:
        return "covered_with_warnings"
    return coverage


def validate_bbox_provenance(
    seed: dict[str, Any],
    table_object: dict[str, Any] | None,
    warnings: list[str],
    missing_items: list[dict[str, Any]],
) -> str:
    if not table_object:
        return "not_evaluable"
    if seed.get("value_bboxes_available") is not False or table_object.get("value_bboxes_available") is not False:
        missing_items.append({"kind": "bbox_provenance", "reason": "value_bboxes_available_must_remain_false"})
        return "not_covered"
    cells = list(seed.get("gold_cells") or []) + list(table_object.get("logical_cells") or [])
    forged = [
        {
            "row_key": cell.get("row_key"),
            "logical_column": cell.get("logical_column"),
            "value_raw": cell.get("value_raw"),
        }
        for cell in cells
        if cell.get("value_bbox") is not None or cell.get("value_bbox_source") not in {None, "not_available"}
    ]
    if forged:
        missing_items.extend({"kind": "bbox_provenance", **item} for item in forged)
        return "not_covered"
    append_warning(warnings, "value_bboxes_available=false 按 Phase7F guardrail 保留，cell bbox 不解释为 value bbox")
    return "covered_with_warnings"


def derive_overall_status(result: dict[str, Any], validation_subset: str) -> str:
    if validation_subset == "exploratory_partial":
        if result["required_values_matched"] == 0:
            return "not_evaluable"
        return "partial"

    blocking_fields = [
        "row_coverage",
        "column_coverage",
        "cell_coverage",
        "required_value_coverage",
        "literal_preservation_coverage",
        "source_span_coverage",
        "bbox_provenance_check",
    ]
    allowed_binding_fields = [
        "unit_binding_coverage",
        "footnote_binding_coverage",
        "reference_binding_coverage",
    ]
    pass_values = {"covered", "covered_with_warnings"}
    pass_or_na_values = {"covered", "covered_with_warnings", "not_applicable"}

    if any(result[field] == "not_evaluable" for field in blocking_fields + allowed_binding_fields):
        return "not_evaluable"
    if any(result[field] not in pass_values for field in blocking_fields):
        return "fail"
    if any(result[field] not in pass_or_na_values for field in allowed_binding_fields):
        return "fail"
    if any(result[field] == "covered_with_warnings" for field in blocking_fields + allowed_binding_fields):
        return "pass_with_warnings"
    if result["warnings"]:
        return "pass_with_warnings"
    return "pass"


def validation_subset_for_seed(seed: dict[str, Any], confirmed_ids: set[str], partial_ids: set[str]) -> str:
    seed_id = seed.get("gold_seed_id")
    if seed_id in confirmed_ids:
        return "formal_confirmed"
    if seed_id in partial_ids:
        return "exploratory_partial"
    return "out_of_scope"


def validate_seed(
    seed: dict[str, Any],
    table_object: dict[str, Any] | None,
    validation_subset: str,
    partial_seed_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    warnings = [warning for warning in seed.get("construction_warnings") or [] if PROD_READY_MARKER not in warning]
    if seed.get("gold_seed_status") == "confirmed_seed":
        append_warning(warnings, "confirmed_seed 不是 production readiness 结论")
    missing_items: list[dict[str, Any]] = []

    row_coverage = validate_rows(seed, table_object, missing_items)
    column_coverage = validate_columns(seed, table_object, missing_items)
    cell_coverage, gold_cells_matched, gold_cell_mappings = validate_cells(seed, table_object, missing_items)
    (
        required_value_coverage,
        required_values_matched,
        required_value_mappings,
        matches_by_key,
    ) = validate_required_values(seed, table_object, missing_items)
    unit_binding_coverage = validate_units(seed, matches_by_key, missing_items)
    footnote_binding_coverage = validate_footnotes(seed, matches_by_key, missing_items)
    reference_binding_coverage = validate_references(seed, table_object, missing_items)
    literal_preservation_coverage = validate_literal_preservation(seed, required_value_coverage, missing_items)
    source_span_coverage = validate_source_span(seed, table_object, matches_by_key, missing_items)
    bbox_provenance_check = validate_bbox_provenance(seed, table_object, warnings, missing_items)

    if validation_subset == "exploratory_partial" and partial_seed_row:
        append_warning(warnings, f"partial_seed exploratory only: {partial_seed_row.get('remaining_blockers', '')}")
    if table_object and not table_object.get("logical_cells"):
        append_warning(warnings, "Phase7D-3 table_object 未生成 logical_cells，本轮只作 exploratory 记录")

    result: dict[str, Any] = {
        "gold_seed_id": seed.get("gold_seed_id"),
        "table_object_id": seed.get("table_object_id"),
        "doc_id": seed.get("doc_id"),
        "table_id": seed.get("table_id"),
        "validation_subset": validation_subset,
        "row_coverage": row_coverage,
        "column_coverage": column_coverage,
        "cell_coverage": cell_coverage,
        "required_value_coverage": required_value_coverage,
        "unit_binding_coverage": unit_binding_coverage,
        "footnote_binding_coverage": footnote_binding_coverage,
        "reference_binding_coverage": reference_binding_coverage,
        "literal_preservation_coverage": literal_preservation_coverage,
        "source_span_coverage": source_span_coverage,
        "bbox_provenance_check": bbox_provenance_check,
        "overall_validation_status": "not_evaluable",
        "required_values_total": len(seed.get("required_values") or []),
        "required_values_matched": required_values_matched,
        "gold_cells_total": len(seed.get("gold_cells") or []),
        "gold_cells_matched": gold_cells_matched,
        "warnings": warnings,
        "missing_items": missing_items,
        "required_value_mappings": required_value_mappings,
        "gold_cell_mappings": gold_cell_mappings,
        "source_span_granularity": seed.get("source_span_granularity"),
        "value_bboxes_available": seed.get("value_bboxes_available"),
    }
    result["overall_validation_status"] = derive_overall_status(result, validation_subset)
    assert result["overall_validation_status"] in OVERALL_STATUS_VALUES
    for field in RESULT_FIELDS:
        if field.endswith("coverage") or field == "bbox_provenance_check":
            assert result[field] in COVERAGE_VALUES
    return result


def compute_formal_overall(results: list[dict[str, Any]]) -> dict[str, Any]:
    formal = [result for result in results if result["validation_subset"] == "formal_confirmed"]
    status_counts = Counter(result["overall_validation_status"] for result in formal)
    if not formal:
        status = "not_evaluable"
    elif all(result["overall_validation_status"] in {"pass", "pass_with_warnings"} for result in formal):
        status = "pass_with_warnings" if any(result["overall_validation_status"] == "pass_with_warnings" for result in formal) else "pass"
    elif any(result["overall_validation_status"] == "fail" for result in formal):
        status = "fail"
    elif any(result["overall_validation_status"] == "partial" for result in formal):
        status = "partial"
    else:
        status = "not_evaluable"

    coverage_fields = [
        "required_value_coverage",
        "cell_coverage",
        "unit_binding_coverage",
        "footnote_binding_coverage",
        "reference_binding_coverage",
        "literal_preservation_coverage",
        "source_span_coverage",
        "bbox_provenance_check",
    ]
    coverage_counts = {
        field: dict(Counter(result[field] for result in formal))
        for field in coverage_fields
    }
    return {
        "formal_overall_status": status,
        "formal_seed_count": len(formal),
        "formal_pass_like_count": sum(
            1 for result in formal if result["overall_validation_status"] in {"pass", "pass_with_warnings"}
        ),
        "status_counts": dict(status_counts),
        "coverage_counts": coverage_counts,
    }


def build_validation_results(inputs: dict[str, Any]) -> dict[str, Any]:
    confirmed_ids = set(inputs["confirmed_seed_ids"])
    partial_ids = set(inputs["partial_seed_ids"])
    partial_rows = {row["gold_seed_id"]: row for row in inputs["partial_seed_rows"]}
    table_objects = {obj["table_object_id"]: obj for obj in inputs["table_objects"]}

    results: list[dict[str, Any]] = []
    for seed in inputs["seeds"]:
        validation_subset = validation_subset_for_seed(seed, confirmed_ids, partial_ids)
        if validation_subset == "out_of_scope":
            continue
        table_object = table_objects.get(seed.get("table_object_id"))
        results.append(
            validate_seed(
                seed,
                table_object,
                validation_subset,
                partial_seed_row=partial_rows.get(seed.get("gold_seed_id")),
            )
        )

    formal_overall = compute_formal_overall(results)
    partial_results = [result for result in results if result["validation_subset"] == "exploratory_partial"]
    return {
        "phase": "v7_phase7F_gold_seed_validation",
        "validation_scope": "seed_level_offline_validation_not_official_benchmark",
        "formal_subset_policy": "confirmed_seed_only",
        "partial_subset_policy": "exploratory_only_excluded_from_formal",
        "production_readiness_note": "本轮不是 production readiness。",
        "retrieval_model_note": "本轮未运行 retrieval、RAG、RAGAS、Qwen、OCR 或 VLM。",
        "formal_confirmed_overall": formal_overall,
        "partial_exploratory_count": len(partial_results),
        "results": results,
    }


def flat_result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: result.get(field) for field in RESULT_FIELDS} for result in results]


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    inputs = load_inputs()
    payload = build_validation_results(inputs)
    results = payload["results"]

    write_json(output_dir / "gold_seed_validation_results.json", payload)
    write_csv(output_dir / "gold_seed_validation_results.csv", flat_result_rows(results), RESULT_FIELDS)
    write_csv(
        output_dir / "formal_confirmed_validation_results.csv",
        flat_result_rows([result for result in results if result["validation_subset"] == "formal_confirmed"]),
        RESULT_FIELDS,
    )
    write_csv(
        output_dir / "partial_seed_exploratory_results.csv",
        flat_result_rows([result for result in results if result["validation_subset"] == "exploratory_partial"]),
        RESULT_FIELDS,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase7D-3 hybrid extractor output against Phase7E gold seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.output_dir)
    print(json.dumps(payload["formal_confirmed_overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
