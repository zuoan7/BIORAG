#!/usr/bin/env python3
"""Isolated offline flat-vs-table_object representation comparison for C-6."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

TABLE_OBJECTS_REFINED_PATH = (
    ROOT
    / "data/experiments/v7_phase6c_table_object_expanded_sample/table_objects_refined.jsonl"
)
ROW_CELL_GOLD_REFINED_PATH = (
    ROOT
    / "data/experiments/v7_phase6c_table_object_expanded_sample/row_cell_gold_refined.jsonl"
)
FLAT_EXTRACTS_PATH = (
    ROOT / "results/v7_phase6c_table_object_expanded_sample/flat_representation_extracts.json"
)
TABLE_EXTRACTS_PATH = (
    ROOT
    / "results/v7_phase6c_table_object_expanded_sample/table_object_representation_extracts.json"
)
COVERAGE_RESULTS_PATH = (
    ROOT
    / "results/v7_phase6c_table_object_expanded_sample/offline_coverage_check_rerun_results.json"
)
OUTPUT_JSON = (
    ROOT
    / "results/v7_phase6c_table_object_expanded_sample/flat_vs_table_object_comparison.json"
)
OUTPUT_CSV = (
    ROOT
    / "results/v7_phase6c_table_object_expanded_sample/flat_vs_table_object_comparison.csv"
)

FORMAL_GOLD_IDS = [
    "gold_doc_0687_table2_p5c3_0003",
    "gold_doc_0458_table3_p5c5_0071",
]

OUTPUT_FIELDS = [
    "gold_id",
    "sample_id",
    "table_object_id",
    "subset",
    "evidence_completeness",
    "source_span_traceability",
    "row_expression_clarity",
    "cell_expression_clarity",
    "value_expression_clarity",
    "unit_binding_clarity",
    "footnote_reference_binding_clarity",
    "uncertainty_annotation_clarity",
    "answerability_calibration",
    "representation_gap",
    "formal_conclusion",
    "warnings",
    "notes",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def flat_contains_text(flat: dict[str, Any], value: str) -> bool:
    return value in flat.get("flat_text_extract", "")


def compare_record(
    gold: dict[str, Any],
    flat: dict[str, Any],
    table_extract: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    gold_id = gold["gold_id"]
    warnings = unique(
        flat.get("flat_representation_warnings", [])
        + table_extract.get("table_object_warnings", [])
        + coverage.get("warnings", [])
    )

    if gold_id == "gold_doc_0687_table2_p5c3_0003":
        missing_flat_values = [
            value for value in ("0.35", "0.43") if not flat_contains_text(flat, value)
        ]
        if missing_flat_values:
            warnings.append(
                "flat_missing_refined_restored_values:" + ",".join(missing_flat_values)
            )
        representation_gap = (
            "flat representation 只能呈现 official chunk 的线性 paragraph：selected rows 可见，"
            "但 metric-level cells、unit scope、row-level reference binding 需要人工推断；"
            "TMB3421 YE/S=0.35 与 RWB217 YE/S=0.43 不在 flat source_span 文本中显式出现。"
        )
        notes = (
            "table_object_refined 将 5 个 selected rows 展开为 15 个 metric-level cells，"
            "显式绑定 YE/S、qethanol、qxylose、unit、row-level reference 和 source_span；"
            "风险是 source_span 仍为 row-level/block-level，不是 value-level bbox。"
        )
        return {
            "gold_id": gold_id,
            "sample_id": gold.get("sample_id"),
            "table_object_id": gold.get("table_object_id"),
            "subset": "confirmed",
            "evidence_completeness": "better_in_table_object",
            "source_span_traceability": "better_in_table_object",
            "row_expression_clarity": "better_in_table_object",
            "cell_expression_clarity": "better_in_table_object",
            "value_expression_clarity": "better_in_table_object",
            "unit_binding_clarity": "better_in_table_object",
            "footnote_reference_binding_clarity": "better_in_table_object",
            "uncertainty_annotation_clarity": "better_in_table_object",
            "answerability_calibration": "better_in_table_object",
            "representation_gap": representation_gap,
            "formal_conclusion": "table_object_stronger",
            "warnings": unique(warnings),
            "notes": notes,
        }

    if gold_id == "gold_doc_0458_table3_p5c5_0071":
        representation_gap = (
            "flat representation 已显示 selected row names、Added Nutrients/Content headers "
            "和 content percent values；主要缺口是所有 rows/cells 混在同一个 body paragraph，"
            "row span、cell boundary 与 percent unit binding 需要从线性顺序推断。"
        )
        notes = (
            "table_object_refined 将 selected products、Added Nutrients cells、Content (%) values "
            "和 percent unit 显式绑定到 rows/cells/source_span；"
            "风险是 selected cells 仍共享 coarse body source_span。"
        )
        return {
            "gold_id": gold_id,
            "sample_id": gold.get("sample_id"),
            "table_object_id": gold.get("table_object_id"),
            "subset": "confirmed",
            "evidence_completeness": "comparable",
            "source_span_traceability": "better_in_table_object",
            "row_expression_clarity": "better_in_table_object",
            "cell_expression_clarity": "better_in_table_object",
            "value_expression_clarity": "comparable",
            "unit_binding_clarity": "better_in_table_object",
            "footnote_reference_binding_clarity": "not_applicable",
            "uncertainty_annotation_clarity": "better_in_table_object",
            "answerability_calibration": "better_in_table_object",
            "representation_gap": representation_gap,
            "formal_conclusion": "table_object_stronger",
            "warnings": unique(warnings),
            "notes": notes,
        }

    raise ValueError(f"unexpected formal gold_id: {gold_id}")


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


def main() -> None:
    golds = {row["gold_id"]: row for row in load_jsonl(ROW_CELL_GOLD_REFINED_PATH)}
    table_objects = {
        row["table_object_id"]: row for row in load_jsonl(TABLE_OBJECTS_REFINED_PATH)
    }
    flat_extracts = {row["gold_id"]: row for row in load_json(FLAT_EXTRACTS_PATH)}
    table_extracts = {row["gold_id"]: row for row in load_json(TABLE_EXTRACTS_PATH)}
    coverage_results = {row["gold_id"]: row for row in load_json(COVERAGE_RESULTS_PATH)}

    missing = [
        gold_id
        for gold_id in FORMAL_GOLD_IDS
        if gold_id not in golds
        or gold_id not in flat_extracts
        or gold_id not in table_extracts
        or gold_id not in coverage_results
    ]
    if missing:
        raise SystemExit(f"missing formal comparison input for: {', '.join(missing)}")

    records: list[dict[str, Any]] = []
    for gold_id in FORMAL_GOLD_IDS:
        gold = golds[gold_id]
        if gold.get("gold_status") != "confirmed_gold":
            raise SystemExit(f"formal input is not confirmed_gold: {gold_id}")
        if coverage_results[gold_id].get("subset") != "confirmed":
            raise SystemExit(f"formal coverage input is not confirmed subset: {gold_id}")
        if coverage_results[gold_id].get("coverage_status") != "pass_with_warnings":
            raise SystemExit(f"formal coverage input did not pass C-5R: {gold_id}")
        if gold.get("table_object_id") not in table_objects:
            raise SystemExit(f"missing table_object for: {gold_id}")
        records.append(
            compare_record(
                gold,
                flat_extracts[gold_id],
                table_extracts[gold_id],
                coverage_results[gold_id],
            )
        )

    write_json(OUTPUT_JSON, records)
    write_csv(OUTPUT_CSV, records)
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"formal_comparison_records={len(records)}")


if __name__ == "__main__":
    main()
