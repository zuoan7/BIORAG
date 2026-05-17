#!/usr/bin/env python3
"""Isolated offline flat-vs-table_object representation comparison for B-5."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

TABLE_OBJECTS_PATH = ROOT / "data/experiments/v7_phase6b_table_object_pilot/table_objects.jsonl"
ROW_CELL_GOLD_PATH = ROOT / "data/experiments/v7_phase6b_table_object_pilot/row_cell_gold.jsonl"
FLAT_EXTRACTS_PATH = ROOT / "results/v7_phase6b_table_object_pilot/flat_representation_extracts.json"
TABLE_EXTRACTS_PATH = (
    ROOT / "results/v7_phase6b_table_object_pilot/table_object_representation_extracts.json"
)
COVERAGE_RESULTS_PATH = ROOT / "results/v7_phase6b_table_object_pilot/offline_coverage_results.json"
OUTPUT_JSON = ROOT / "results/v7_phase6b_table_object_pilot/flat_vs_table_object_comparison.json"
OUTPUT_CSV = ROOT / "results/v7_phase6b_table_object_pilot/flat_vs_table_object_comparison.csv"

FORMAL_GOLD_IDS = [
    "gold_doc_0452_table1_p5c5_0027",
    "gold_doc_0468_table3_p5c2_0024",
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


def flat_has_all_values(flat: dict[str, Any], table_extract: dict[str, Any]) -> bool:
    text = flat.get("flat_text_extract", "")
    for value in table_extract.get("table_object_values", []):
        raw = value.get("value_raw")
        if raw is not None and str(raw) not in text:
            return False
    return True


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

    values_visible = flat_has_all_values(flat, table_extract)
    has_structured_cells = bool(table_extract.get("table_object_cells"))
    has_per_cell_source_spans = all(
        cell.get("source_span") for cell in table_extract.get("table_object_cells", [])
    )
    has_units = bool(table_extract.get("table_object_units"))
    has_footnotes = bool(table_extract.get("table_object_footnotes"))

    if gold_id == "gold_doc_0452_table1_p5c5_0027":
        notes = (
            "flat 文本包含 primer rows、forward/reverse header、sequence values 和 abbreviation note，"
            "但 row-column-cell 关系需要从相邻 chunk 的线性顺序推断；"
            "table_object 将每个 primer sequence 绑定到 row、forward/reverse column、value 和 source_span。"
        )
        representation_gap = (
            "flat representation 的主要缺口是方向与 sequence cell 的结构化绑定不显式，"
            "且 body chunk 原本含后续非表格正文，需要依赖 frozen boundary 截断。"
        )
        unit_binding_clarity = "not_applicable"
    elif gold_id == "gold_doc_0468_table3_p5c2_0024":
        notes = (
            "flat 文本完整包含 Table 3 的 rows、Bimuno GOS/GOS-p values、not detected 原文、"
            "area% footnote 和 abbreviation definitions；table_object 进一步显式表达 row、cell、unit、"
            "footnote 与 source_span 绑定。"
        )
        representation_gap = (
            "flat representation 的主要缺口是 composition value 与 column/unit/footnote 的绑定依赖线性文本推断；"
            "table_object 将 area% 和 footnote a 绑定到 composition cells。"
        )
        unit_binding_clarity = "better_in_table_object" if has_units else "unclear"
    else:
        raise ValueError(f"unexpected formal gold_id: {gold_id}")

    value_expression = "better_in_table_object" if values_visible and has_structured_cells else "unclear"
    if not values_visible:
        value_expression = "better_in_table_object"

    return {
        "gold_id": gold_id,
        "sample_id": gold.get("sample_id"),
        "table_object_id": gold.get("table_object_id"),
        "subset": "confirmed",
        "evidence_completeness": "better_in_table_object",
        "source_span_traceability": (
            "better_in_table_object" if has_per_cell_source_spans else "comparable"
        ),
        "row_expression_clarity": "better_in_table_object",
        "cell_expression_clarity": "better_in_table_object",
        "value_expression_clarity": value_expression,
        "unit_binding_clarity": unit_binding_clarity,
        "footnote_reference_binding_clarity": (
            "better_in_table_object" if has_footnotes else "not_applicable"
        ),
        "uncertainty_annotation_clarity": "better_in_table_object",
        "answerability_calibration": "comparable",
        "representation_gap": representation_gap,
        "formal_conclusion": "table_object_stronger",
        "warnings": warnings,
        "notes": notes,
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


def main() -> None:
    golds = {row["gold_id"]: row for row in load_jsonl(ROW_CELL_GOLD_PATH)}
    table_objects = {row["table_object_id"]: row for row in load_jsonl(TABLE_OBJECTS_PATH)}
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
