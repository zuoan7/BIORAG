#!/usr/bin/env python3
"""Minimal isolated offline coverage rerun for BIORAG v7-phase6C-5R."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_CHECKER_PATH = ROOT / "scripts/evaluation/v7_phase6c_offline_table_object_coverage_check.py"

TABLE_OBJECTS_REFINED_PATH = (
    ROOT
    / "data/experiments/v7_phase6c_table_object_expanded_sample/table_objects_refined.jsonl"
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
OUTPUT_JSON = OUTPUT_DIR / "offline_coverage_check_rerun_results.json"
OUTPUT_CSV = OUTPUT_DIR / "offline_coverage_check_rerun_results.csv"


def load_base_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase6c_c5_checker", BASE_CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load base checker: {BASE_CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confirmed_c5r_pass(records: list[dict[str, Any]]) -> bool:
    confirmed = [record for record in records if record["subset"] == "confirmed"]
    if len(confirmed) != 2:
        return False

    required_covered = (
        "table_object_source_coverage",
        "row_gold_coverage",
        "cell_gold_coverage",
        "value_coverage",
        "source_span_coverage",
    )
    required_classified = (
        "unit_binding_coverage",
        "footnote_reference_coverage",
    )
    for record in confirmed:
        if record["coverage_status"] in {"fail", "partial", "not_evaluable"}:
            return False
        if any(record[field] != "covered" for field in required_covered):
            return False
        if any(record[field] not in {"covered", "not_applicable"} for field in required_classified):
            return False
        if record["evidence_completeness"] not in {"covered", "covered_with_minor_warnings"}:
            return False
        if record["answerability_calibration"] != "covered":
            return False
    return True


def main() -> None:
    checker = load_base_checker()

    table_objects_list = checker.load_jsonl(TABLE_OBJECTS_REFINED_PATH)
    gold_rows = checker.load_jsonl(ROW_CELL_GOLD_REFINED_PATH)
    refinement_summary = checker.load_refinement_summary(GOLD_REFINEMENT_SUMMARY_PATH)

    table_objects = {item["table_object_id"]: item for item in table_objects_list}
    gold_by_id = {item["gold_id"]: item for item in gold_rows}
    missing = [gold_id for gold_id in checker.GOLD_ORDER if gold_id not in gold_by_id]
    if missing:
        raise SystemExit(f"Missing expected C-5R gold rows: {', '.join(missing)}")

    records = [
        checker.evaluate_gold(gold_by_id[gold_id], table_objects, refinement_summary)
        for gold_id in checker.GOLD_ORDER
    ]
    checker.write_json(OUTPUT_JSON, records)
    checker.write_csv(OUTPUT_CSV, records)

    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"confirmed_subset_c5r_pass={str(confirmed_c5r_pass(records)).lower()}")


if __name__ == "__main__":
    main()
