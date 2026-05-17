#!/usr/bin/env python3
"""Isolated offline table_object coverage evaluation for BIORAG v7-phase6B-4."""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

TABLE_OBJECTS_PATH = ROOT / "data/experiments/v7_phase6b_table_object_pilot/table_objects.jsonl"
ROW_CELL_GOLD_PATH = ROOT / "data/experiments/v7_phase6b_table_object_pilot/row_cell_gold.jsonl"
CONSISTENCY_SUMMARY_PATH = (
    ROOT
    / "reports/v7_phase6b_table_object_offline_pilot/row_cell_gold_consistency_summary.csv"
)
OUTPUT_DIR = ROOT / "results/v7_phase6b_table_object_pilot"
OUTPUT_JSON = OUTPUT_DIR / "offline_coverage_results.json"
OUTPUT_CSV = OUTPUT_DIR / "offline_coverage_results.csv"

OFFICIAL_BASELINE_NAME = "phase5f_official_clean_baseline"
OFFICIAL_DATASET_SHA256 = (
    "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
)

CONFIRMED_GOLD_IDS = {
    "gold_doc_0452_table1_p5c5_0027",
    "gold_doc_0468_table3_p5c2_0024",
}
PARTIAL_EXPLORATORY_GOLD_IDS = {
    "gold_doc_0522_table1_p5c3_0008",
    "gold_doc_0523_table1_p5c5_0023",
}
GOLD_ORDER = [
    "gold_doc_0452_table1_p5c5_0027",
    "gold_doc_0468_table3_p5c2_0024",
    "gold_doc_0522_table1_p5c3_0008",
    "gold_doc_0523_table1_p5c5_0023",
]

OUTPUT_FIELDS = [
    "gold_id",
    "gold_status",
    "subset",
    "table_object_id",
    "sample_id",
    "table_object_source_coverage",
    "row_gold_coverage",
    "cell_gold_coverage",
    "value_coverage",
    "unit_binding_coverage",
    "footnote_reference_coverage",
    "source_span_coverage",
    "evidence_completeness",
    "answerability_calibration",
    "coverage_status",
    "warnings",
    "notes",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_consistency_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["gold_id"]: row for row in csv.DictReader(handle)}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split())


def same_scalar(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    return normalize_text(left) == normalize_text(right)


def list_ids(items: list[dict[str, Any]], key: str) -> set[str]:
    return {str(item.get(key)) for item in items if item.get(key)}


def has_source_span(item: dict[str, Any]) -> bool:
    source_span = item.get("source_span")
    return isinstance(source_span, dict) and bool(
        source_span.get("text_span") or source_span.get("block_id") or source_span.get("chunk_id")
    )


def subset_for_gold(gold: dict[str, Any]) -> str:
    gold_id = gold["gold_id"]
    if gold_id in CONFIRMED_GOLD_IDS:
        return "confirmed"
    if gold_id in PARTIAL_EXPLORATORY_GOLD_IDS:
        return "partial_exploratory"
    raise ValueError(f"gold_id outside B-4 scope: {gold_id}")


def status_has_uncertainty(status: Any) -> bool:
    text = normalize_text(status).lower()
    markers = ("uncertain", "unresolved", "partial", "not stable", "not safely")
    return any(marker in text for marker in markers)


def coverage_from_counts(total: int, covered: int, uncertain: bool = False) -> str:
    if total == 0:
        return "not_applicable"
    if covered == total and not uncertain:
        return "covered"
    if covered == total and uncertain:
        return "partially_covered"
    if covered > 0:
        return "partially_covered"
    return "not_covered"


def evaluate_table_object_source(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    if table_object is None:
        notes.append("table_object 缺失，无法进行 source/object 覆盖判断。")
        return "not_covered"

    required_checks = [
        table_object.get("table_object_id") == gold.get("table_object_id"),
        table_object.get("doc_id") == gold.get("doc_id"),
        gold.get("sample_id") in table_object.get("sample_ids", []),
        table_object.get("baseline_name") == OFFICIAL_BASELINE_NAME,
        gold.get("baseline_name") == OFFICIAL_BASELINE_NAME,
        table_object.get("dataset_sha256") == OFFICIAL_DATASET_SHA256,
        gold.get("dataset_sha256") == OFFICIAL_DATASET_SHA256,
        bool(table_object.get("source_spans")),
        table_object.get("table_boundary_status") in {"frozen", "stable"},
        table_object.get("source_relation_confidence") in {"high", "medium"},
    ]
    if all(required_checks):
        return "covered"

    notes.append("table_object 的 source/object 关系存在缺口或 pin 不一致。")
    if any(required_checks):
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
    covered = 0
    for row in required_rows:
        table_row = table_rows.get(row.get("row_id"))
        if table_row and has_source_span(table_row):
            covered += 1

    if covered != len(required_rows):
        notes.append(f"row coverage 不完整：{covered}/{len(required_rows)} 个 required rows 可追溯。")
    return coverage_from_counts(len(required_rows), covered)


def evaluate_cells(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = gold.get("required_cells", [])
    if not required_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = {cell.get("cell_id"): cell for cell in table_object.get("cells", [])}
    covered = 0
    uncertain = False
    for cell in required_cells:
        table_cell = table_cells.get(cell.get("cell_id"))
        if table_cell and has_source_span(table_cell):
            covered += 1
        if normalize_text(cell.get("confidence")).lower() not in {"", "high"}:
            uncertain = True
        if status_has_uncertainty(cell.get("notes")):
            uncertain = True

    if covered != len(required_cells):
        notes.append(f"cell coverage 不完整：{covered}/{len(required_cells)} 个 required cells 可追溯。")
    return coverage_from_counts(len(required_cells), covered, uncertain=uncertain)


def evaluate_values(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = gold.get("required_cells", [])
    if not required_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = {cell.get("cell_id"): cell for cell in table_object.get("cells", [])}
    covered = 0
    uncertain = False
    for cell in required_cells:
        table_cell = table_cells.get(cell.get("cell_id"))
        if not table_cell:
            continue
        raw_matches = same_scalar(cell.get("value_raw"), table_cell.get("value_raw"))
        normalized_matches = same_scalar(
            cell.get("value_normalized"), table_cell.get("value_normalized")
        )
        if raw_matches or normalized_matches:
            covered += 1
        if status_has_uncertainty(cell.get("notes")):
            uncertain = True

    if covered != len(required_cells):
        notes.append(f"value coverage 不完整：{covered}/{len(required_cells)} 个 required values 匹配。")
    return coverage_from_counts(len(required_cells), covered, uncertain=uncertain)


def evaluate_units(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_units = gold.get("required_units", [])
    required_cells = gold.get("required_cells", [])
    if not required_units and not any(cell.get("unit") for cell in required_cells):
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = {cell.get("cell_id"): cell for cell in table_object.get("cells", [])}
    uncertain = any(status_has_uncertainty(unit.get("status")) for unit in required_units)
    covered = 0
    total = 0

    for cell in required_cells:
        required_unit = cell.get("unit")
        if not required_unit:
            continue
        total += 1
        table_cell = table_cells.get(cell.get("cell_id"))
        if table_cell and same_scalar(required_unit, table_cell.get("unit")):
            covered += 1

    if total == 0 and uncertain:
        notes.append("unit 已明确标注为不确定，不能升级为 covered。")
        return "uncertain"
    if total == 0:
        return "not_applicable"
    if covered != total:
        notes.append(f"unit binding 不完整：{covered}/{total} 个 required cell units 匹配。")
    return coverage_from_counts(total, covered, uncertain=uncertain)


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

    table_footnote_ids = list_ids(table_object.get("footnotes", []), "footnote_id")
    table_reference_ids = list_ids(table_object.get("references", []), "reference_id")
    covered = 0
    uncertain = False

    for footnote in required_footnotes:
        if footnote.get("footnote_id") in table_footnote_ids:
            covered += 1
        if status_has_uncertainty(footnote.get("binding_status")):
            uncertain = True

    for reference in required_references:
        if reference.get("reference_id") in table_reference_ids:
            covered += 1
        elif status_has_uncertainty(reference.get("binding_status")):
            covered += 1
            uncertain = True
        if status_has_uncertainty(reference.get("binding_status")):
            uncertain = True

    if covered != total:
        notes.append(
            f"footnote/reference coverage 不完整：{covered}/{total} 个 required items 覆盖。"
        )
    if uncertain:
        notes.append("footnote/reference 存在明确的不确定绑定，保持 partial/uncertain 分类。")
    return coverage_from_counts(total, covered, uncertain=uncertain)


def evaluate_source_spans(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    if table_object is None:
        return "not_covered"

    gold_source_span_ids = list_ids(gold.get("source_spans", []), "source_span_id")
    required_cells = gold.get("required_cells", [])
    required_footnotes = gold.get("required_footnotes", [])
    required_references = gold.get("required_references", [])
    total = len(required_cells) + len(required_footnotes) + len(required_references)
    if total == 0:
        return "not_applicable"

    table_cells = {cell.get("cell_id"): cell for cell in table_object.get("cells", [])}
    table_footnotes = {
        footnote.get("footnote_id"): footnote for footnote in table_object.get("footnotes", [])
    }
    table_references = {
        reference.get("reference_id"): reference for reference in table_object.get("references", [])
    }

    covered = 0
    uncertain = False
    for cell in required_cells:
        table_cell = table_cells.get(cell.get("cell_id"))
        if cell.get("source_span_id") in gold_source_span_ids and table_cell and has_source_span(table_cell):
            covered += 1

    for footnote in required_footnotes:
        table_footnote = table_footnotes.get(footnote.get("footnote_id"))
        if (
            footnote.get("source_span_id") in gold_source_span_ids
            and table_footnote
            and has_source_span(table_footnote)
        ):
            covered += 1
        if status_has_uncertainty(footnote.get("binding_status")):
            uncertain = True

    for reference in required_references:
        table_reference = table_references.get(reference.get("reference_id"))
        if (
            reference.get("source_span_id") in gold_source_span_ids
            and table_reference
            and has_source_span(table_reference)
        ):
            covered += 1
        elif (
            reference.get("source_span_id") in gold_source_span_ids
            and status_has_uncertainty(reference.get("binding_status"))
        ):
            covered += 1
            uncertain = True
        if status_has_uncertainty(reference.get("binding_status")):
            uncertain = True

    if covered != total:
        notes.append(f"source_span coverage 不完整：{covered}/{total} 个 required spans 可追溯。")
    return coverage_from_counts(total, covered, uncertain=uncertain)


def evaluate_answerability(
    gold: dict[str, Any], subset: str, consistency_row: dict[str, str] | None, notes: list[str]
) -> str:
    gold_status = gold.get("gold_status")
    if subset == "confirmed" and gold_status != "confirmed_gold":
        notes.append("confirmed subset 与 gold_status 不一致。")
        return "not_covered"
    if subset == "partial_exploratory" and gold_status != "partial_gold":
        notes.append("partial exploratory subset 与 gold_status 不一致。")
        return "not_covered"
    if consistency_row is None:
        notes.append("缺少 row_cell_gold_consistency_summary 记录。")
        return "uncertain"

    manual_review_required = normalize_text(consistency_row.get("manual_review_required")).lower()
    consistency_status = normalize_text(consistency_row.get("consistency_status")).lower()
    if consistency_status == "fail":
        notes.append("一致性摘要为 fail，不能作为 B-4 可评估输入。")
        return "not_covered"
    if subset == "partial_exploratory" and manual_review_required == "true":
        return "covered"
    if subset == "confirmed" and manual_review_required == "false":
        return "covered"
    return "partially_covered"


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
    if any(metrics[name] in {"partially_covered", "uncertain"} for name in optional_metrics):
        return "partially_covered"
    if any(metrics[name] == "not_covered" for name in optional_metrics):
        return "not_covered"
    if subset == "confirmed" and warnings:
        return "covered_with_minor_warnings"
    return "covered"


def coverage_status(metrics: dict[str, str], subset: str) -> str:
    if metrics["table_object_source_coverage"] == "not_covered":
        return "not_evaluable"
    if metrics["evidence_completeness"] == "not_covered":
        return "fail"
    if subset == "partial_exploratory":
        if metrics["evidence_completeness"] in {"partially_covered", "covered_with_minor_warnings"}:
            return "partial"
        return "pass_with_warnings"
    if metrics["evidence_completeness"] == "covered_with_minor_warnings":
        return "pass_with_warnings"
    if metrics["evidence_completeness"] == "covered":
        return "pass"
    return "partial"


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
    consistency_summary: dict[str, dict[str, str]],
) -> dict[str, Any]:
    subset = subset_for_gold(gold)
    table_object = table_objects.get(gold.get("table_object_id"))
    consistency_row = consistency_summary.get(gold.get("gold_id"))
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
            gold, subset, consistency_row, notes
        ),
    }

    warnings = unique_warnings(
        gold.get("warnings", []),
        table_object.get("warnings", []) if table_object else [],
    )
    metrics["evidence_completeness"] = evidence_completeness(metrics, warnings, subset)
    status = coverage_status(metrics, subset)

    if subset == "partial_exploratory":
        notes.append("partial_gold 仅作为 exploratory coverage，不参与 B-4 formal benchmark。")
    if warnings:
        notes.append("warnings 已保留为非阻断或探索性风险信号。")

    record: dict[str, Any] = {
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
    return record


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
    table_objects_list = load_jsonl(TABLE_OBJECTS_PATH)
    gold_rows = load_jsonl(ROW_CELL_GOLD_PATH)
    consistency_summary = load_consistency_summary(CONSISTENCY_SUMMARY_PATH)

    table_objects = {item["table_object_id"]: item for item in table_objects_list}
    gold_by_id = {item["gold_id"]: item for item in gold_rows}
    missing = [gold_id for gold_id in GOLD_ORDER if gold_id not in gold_by_id]
    if missing:
        raise SystemExit(f"Missing expected B-4 gold rows: {', '.join(missing)}")

    records = [
        evaluate_gold(gold_by_id[gold_id], table_objects, consistency_summary)
        for gold_id in GOLD_ORDER
    ]
    write_json(OUTPUT_JSON, records)
    write_csv(OUTPUT_CSV, records)

    confirmed = [record for record in records if record["subset"] == "confirmed"]
    confirmed_ready = all(
        record["coverage_status"] not in {"fail", "not_evaluable"}
        and record["table_object_source_coverage"] == "covered"
        and record["row_gold_coverage"] == "covered"
        and record["cell_gold_coverage"] == "covered"
        and record["value_coverage"] == "covered"
        and record["source_span_coverage"] == "covered"
        and record["evidence_completeness"] in {"covered", "covered_with_minor_warnings"}
        for record in confirmed
    )
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"confirmed_subset_b4_pass={str(confirmed_ready).lower()}")


if __name__ == "__main__":
    main()
