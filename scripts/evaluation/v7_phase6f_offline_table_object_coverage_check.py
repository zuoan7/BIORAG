#!/usr/bin/env python3
"""Minimal isolated offline coverage check for BIORAG v7-phase6F-5.

This checker reads only the Phase6F-3 table objects, the Phase6F-4 row/cell
gold, and the two F-3/F-4 summary CSVs. It does not run retrieval, models,
embedding, rerank, BM25, Milvus, OCR, VLM, RAGAS, or production code.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

TABLE_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase6f_table_object_fresh_batch/table_objects.jsonl"
)
ROW_CELL_GOLD_PATH = (
    ROOT / "data/experiments/v7_phase6f_table_object_fresh_batch/row_cell_gold.jsonl"
)
GOLD_CONSISTENCY_SUMMARY_PATH = (
    ROOT / "reports/v7_phase6f_fresh_candidate_sampling/row_cell_gold_consistency_summary.csv"
)
TABLE_VALIDATION_SUMMARY_PATH = (
    ROOT / "reports/v7_phase6f_fresh_candidate_sampling/table_object_validation_summary.csv"
)

OUTPUT_DIR = ROOT / "results/v7_phase6f_table_object_fresh_batch"
OUTPUT_JSON = OUTPUT_DIR / "offline_coverage_check_results.json"
OUTPUT_CSV = OUTPUT_DIR / "offline_coverage_check_results.csv"

OFFICIAL_BASELINE_NAME = "phase5f_official_clean_baseline"
OFFICIAL_DATASET_SHA256 = (
    "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
)
OFFICIAL_CHUNKS_SHA256 = (
    "5dbacc5bb85351203355bf3f2d22f46ec02e24f513ab9523ca3407664669f75b"
)

FORMAL_CONFIRMED_GOLD_IDS = {
    "gold_doc_0322_table1_f6f_0001",
    "gold_doc_0598_table2_f6f_0005",
}
EXCLUDED_FROM_FORMAL_GOLD_IDS = {
    "gold_doc_0158_table3_f6f_0004",
}
GOLD_ORDER = [
    "gold_doc_0322_table1_f6f_0001",
    "gold_doc_0158_table3_f6f_0004",
    "gold_doc_0598_table2_f6f_0005",
]

OUTPUT_FIELDS = [
    "gold_id",
    "table_object_id",
    "sample_id",
    "doc_id",
    "table_id",
    "subset",
    "gold_status",
    "coverage_status",
    "table_object_source_coverage",
    "row_gold_coverage",
    "column_gold_coverage",
    "cell_gold_coverage",
    "value_coverage",
    "unit_binding_coverage",
    "literal_binding_coverage",
    "footnote_reference_coverage",
    "source_span_coverage",
    "evidence_completeness",
    "answerability_calibration",
    "blocking_warnings",
    "nonblocking_warnings",
    "notes",
]

DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).translate(DASH_TRANSLATION)
    return " ".join(text.split())


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalized_path(value: Any) -> tuple[str, ...]:
    return tuple(normalize_text(item) for item in as_list(value))


def semicolon_values(value: Any) -> list[str]:
    if value in (None, "", "none"):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "", "none")]
    return [item for item in str(value).split(";") if item and item != "none"]


def unique_values(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
    return result


def source_span_ids(source_spans: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("source_span_id"))
        for item in source_spans
        if item.get("source_span_id")
    }


def item_has_source_span(item: dict[str, Any], table_source_ids: set[str]) -> bool:
    span_id = item.get("source_span_id")
    return bool(span_id and span_id in table_source_ids)


def raw_matches(required: Any, candidate: Any) -> bool:
    required_text = normalize_text(required)
    candidate_text = normalize_text(candidate)
    return bool(required_text and candidate_text and required_text == candidate_text)


def value_matches(required_cell: dict[str, Any], table_cell: dict[str, Any]) -> bool:
    if raw_matches(required_cell.get("value_raw"), table_cell.get("value_raw")):
        return True
    return normalize_text(required_cell.get("value_normalized")) == normalize_text(
        table_cell.get("value_normalized")
    )


def has_table_row_level_limitation(gold: dict[str, Any], table_object: dict[str, Any]) -> bool:
    if gold.get("source_span_granularity") == "table_row_level":
        return True
    return table_object.get("source_span_granularity") == "table_row_level"


def coverage_from_counts(
    total: int, covered: int, with_warning: bool = False, notes: list[str] | None = None
) -> str:
    if total == 0:
        return "not_applicable"
    if covered == total:
        return "covered_with_warnings" if with_warning else "covered"
    if covered > 0:
        if notes is not None:
            notes.append(f"partial coverage count: {covered}/{total}.")
        return "partially_covered"
    return "not_covered"


def subset_for_gold(gold: dict[str, Any]) -> str:
    gold_id = gold.get("gold_id")
    if gold_id in FORMAL_CONFIRMED_GOLD_IDS:
        return "formal_confirmed"
    if gold_id in EXCLUDED_FROM_FORMAL_GOLD_IDS:
        return "excluded_from_formal"
    raise ValueError(f"gold_id outside F-5 scope: {gold_id}")


def table_cells_by_id(table_object: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if table_object is None:
        return {}
    return {str(cell.get("cell_id")): cell for cell in table_object.get("cells", [])}


def evaluate_table_object_source(
    gold: dict[str, Any],
    table_object: dict[str, Any] | None,
    validation_row: dict[str, str] | None,
    notes: list[str],
) -> str:
    if table_object is None or validation_row is None:
        notes.append("missing table_object or table_object validation summary row.")
        return "not_covered"

    checks = [
        table_object.get("table_object_id") == gold.get("table_object_id"),
        table_object.get("sample_id") == gold.get("sample_id"),
        table_object.get("doc_id") == gold.get("doc_id"),
        normalize_text(table_object.get("table_id")) == normalize_text(gold.get("table_id")),
        table_object.get("baseline_name") == OFFICIAL_BASELINE_NAME,
        table_object.get("dataset_sha256") == OFFICIAL_DATASET_SHA256,
        table_object.get("chunks_sha256") == OFFICIAL_CHUNKS_SHA256,
        bool(table_object.get("source_spans")),
        validation_row.get("validation_status") in {"pass", "pass_with_warnings"},
    ]
    if all(checks):
        return "covered_with_warnings"
    if any(checks):
        notes.append("table_object source/object pins are only partially covered.")
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

    table_rows = {str(row.get("row_id")): row for row in table_object.get("rows", [])}
    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    covered = 0
    missing: list[str] = []
    for row in required_rows:
        table_row = table_rows.get(str(row.get("row_id")))
        if table_row and normalized_path(table_row.get("row_header_path")) == normalized_path(
            row.get("row_header_path")
        ) and item_has_source_span(table_row, table_source_ids):
            covered += 1
        else:
            missing.append(str(row.get("row_id")))
    if missing:
        notes.append(f"row coverage missing: {','.join(missing)}.")
    return coverage_from_counts(len(required_rows), covered)


def evaluate_columns(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_columns = gold.get("required_columns", [])
    if not required_columns:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_columns = {
        str(column.get("column_id")): column for column in table_object.get("columns", [])
    }
    covered = 0
    missing: list[str] = []
    for column in required_columns:
        table_column = table_columns.get(str(column.get("column_id")))
        if not table_column:
            missing.append(str(column.get("column_id")))
            continue
        header_matches = normalize_text(table_column.get("header_text")) == normalize_text(
            column.get("header_text")
        )
        unit_matches = True
        if column.get("unit"):
            unit_matches = normalize_text(table_column.get("unit")) == normalize_text(
                column.get("unit")
            )
        if header_matches and unit_matches:
            covered += 1
        else:
            missing.append(str(column.get("column_id")))
    if missing:
        notes.append(f"column coverage missing: {','.join(missing)}.")
    return coverage_from_counts(len(required_columns), covered)


def evaluate_cells(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = gold.get("required_cells", [])
    if not required_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = table_cells_by_id(table_object)
    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    covered = 0
    missing: list[str] = []
    for cell in required_cells:
        table_cell = table_cells.get(str(cell.get("cell_id")))
        if not table_cell:
            missing.append(str(cell.get("cell_id")))
            continue
        row_matches = normalized_path(table_cell.get("row_header_path")) == normalized_path(
            cell.get("row_header_path")
        )
        column_matches = normalized_path(table_cell.get("col_header_path")) == normalized_path(
            cell.get("col_header_path")
        )
        if row_matches and column_matches and item_has_source_span(table_cell, table_source_ids):
            covered += 1
        else:
            missing.append(str(cell.get("cell_id")))
    if missing:
        notes.append(f"cell coverage missing: {','.join(missing)}.")
    return coverage_from_counts(len(required_cells), covered)


def evaluate_values(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = gold.get("required_cells", [])
    if not required_cells:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = table_cells_by_id(table_object)
    covered = 0
    missing: list[str] = []
    for cell in required_cells:
        table_cell = table_cells.get(str(cell.get("cell_id")))
        if table_cell and value_matches(cell, table_cell):
            covered += 1
        else:
            missing.append(str(cell.get("cell_id")))
    if missing:
        notes.append(f"value coverage missing: {','.join(missing)}.")
    return coverage_from_counts(len(required_cells), covered)


def evaluate_units(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_cells = [cell for cell in gold.get("required_cells", []) if cell.get("unit")]
    required_columns = [column for column in gold.get("required_columns", []) if column.get("unit")]
    if not required_cells and not required_columns and not gold.get("unit"):
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_cells = table_cells_by_id(table_object)
    table_columns = {
        str(column.get("column_id")): column for column in table_object.get("columns", [])
    }
    total = len(required_cells) + len(required_columns)
    covered = 0
    missing: list[str] = []
    for cell in required_cells:
        table_cell = table_cells.get(str(cell.get("cell_id")))
        if table_cell and normalize_text(table_cell.get("unit")) == normalize_text(cell.get("unit")):
            covered += 1
        else:
            missing.append(str(cell.get("cell_id")))
    for column in required_columns:
        table_column = table_columns.get(str(column.get("column_id")))
        if table_column and normalize_text(table_column.get("unit")) == normalize_text(
            column.get("unit")
        ):
            covered += 1
        else:
            missing.append(str(column.get("column_id")))
    if missing:
        notes.append(f"unit binding missing: {','.join(missing)}.")
    return coverage_from_counts(total, covered)


def evaluate_literals(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    required_markers = [normalize_text(item) for item in as_list(gold.get("literal_marker"))]
    required_markers = [item for item in required_markers if item]
    if not required_markers:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_markers = {normalize_text(item) for item in as_list(table_object.get("literal_markers"))}
    table_text_parts: list[str] = [normalize_text(table_object.get("table_text_raw"))]
    table_text_parts.extend(normalize_text(value) for value in as_list(table_object.get("value_raw")))
    for cell in table_object.get("cells", []):
        table_text_parts.append(normalize_text(cell.get("value_raw")))
        table_text_parts.extend(normalize_text(item) for item in as_list(cell.get("literal_marker")))
    joined = " ".join(part for part in table_text_parts if part)

    covered = 0
    missing: list[str] = []
    for marker in required_markers:
        if marker in table_markers or marker in joined:
            covered += 1
            continue
        if marker in {"mean", "SD"}:
            has_mean_sd = any(
                isinstance(cell.get("value_normalized"), dict)
                and "mean" in cell["value_normalized"]
                and "sd" in cell["value_normalized"]
                for cell in table_object.get("cells", [])
            )
            has_plus_minus = any("\u00b1" in normalize_text(cell.get("value_raw")) for cell in table_object.get("cells", []))
            if has_mean_sd or has_plus_minus:
                covered += 1
                continue
        missing.append(marker)
    if missing:
        notes.append(f"literal binding missing: {','.join(missing)}.")
    return coverage_from_counts(len(required_markers), covered)


def evaluate_footnotes_references(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    binding = gold.get("footnote_or_reference_binding", {})
    required_footnotes = binding.get("footnotes", [])
    required_references = binding.get("references", [])
    total = len(required_footnotes) + len(required_references)
    if total == 0:
        return "not_applicable"
    if table_object is None:
        return "not_covered"

    table_source_ids = source_span_ids(table_object.get("source_spans", []))
    table_footnotes = {
        str(footnote.get("footnote_id")): footnote
        for footnote in table_object.get("footnotes", [])
    }
    table_references = {
        str(reference.get("reference_id")): reference
        for reference in table_object.get("references", [])
    }
    covered = 0
    missing: list[str] = []
    for footnote in required_footnotes:
        table_footnote = table_footnotes.get(str(footnote.get("footnote_id")))
        if table_footnote and item_has_source_span(table_footnote, table_source_ids):
            covered += 1
        else:
            missing.append(str(footnote.get("footnote_id")))
    for reference in required_references:
        table_reference = table_references.get(str(reference.get("reference_id")))
        if table_reference and item_has_source_span(table_reference, table_source_ids):
            covered += 1
        else:
            missing.append(str(reference.get("reference_id")))
    if missing:
        notes.append(f"footnote/reference binding missing: {','.join(missing)}.")
    with_warning = bool(
        {
            "footnote_binding_uncertain",
            "reference_binding_uncertain",
            "reference_source_column_not_external_citation_gold",
        }.intersection(set(table_object.get("warnings", [])) | set(gold.get("nonblocking_warnings", [])))
    )
    return coverage_from_counts(total, covered, with_warning=with_warning)


def evaluate_source_spans(
    gold: dict[str, Any], table_object: dict[str, Any] | None, notes: list[str]
) -> str:
    if table_object is None:
        return "not_covered"

    gold_span_ids = source_span_ids(gold.get("source_spans", []))
    table_span_ids = source_span_ids(table_object.get("source_spans", []))
    if not gold_span_ids:
        return "not_covered"

    covered = len(gold_span_ids.intersection(table_span_ids))
    if covered != len(gold_span_ids):
        missing = sorted(gold_span_ids - table_span_ids)
        notes.append(f"source span missing: {','.join(missing)}.")
    with_warning = has_table_row_level_limitation(gold, table_object)
    if with_warning:
        notes.append("source_span is table_row_level only; no value-level bbox.")
    return coverage_from_counts(len(gold_span_ids), covered, with_warning=with_warning)


def evaluate_answerability(
    gold: dict[str, Any],
    consistency_row: dict[str, str] | None,
    subset: str,
    notes: list[str],
) -> str:
    if subset != "formal_confirmed":
        return "not_applicable"
    if consistency_row is None:
        notes.append("missing gold consistency summary row.")
        return "not_covered"
    required_yes_fields = [
        "has_expected_answer",
        "has_required_rows",
        "has_required_columns",
        "has_required_cells",
        "has_required_values",
        "has_value_raw",
        "has_unit_or_literal",
        "has_footnote_or_reference_binding",
        "has_source_spans",
        "source_span_limitation_recorded",
    ]
    if gold.get("gold_status") != "confirmed_gold":
        return "not_covered"
    if semicolon_values(consistency_row.get("blocking_warnings")):
        return "not_covered"
    if all(consistency_row.get(field) == "yes" for field in required_yes_fields):
        if consistency_row.get("consistency_status") == "pass_with_warnings":
            return "covered_with_warnings"
        return "covered"
    notes.append("answerability calibration is incomplete in consistency summary.")
    return "partially_covered"


def evidence_completeness(metrics: dict[str, str], subset: str) -> str:
    if subset != "formal_confirmed":
        return "not_evaluable"

    blocking_fields = [
        "table_object_source_coverage",
        "row_gold_coverage",
        "column_gold_coverage",
        "cell_gold_coverage",
        "value_coverage",
        "source_span_coverage",
    ]
    optional_binding_fields = [
        "unit_binding_coverage",
        "literal_binding_coverage",
        "footnote_reference_coverage",
        "answerability_calibration",
    ]
    if any(metrics[field] == "not_covered" for field in blocking_fields):
        return "not_covered"
    if any(metrics[field] == "partially_covered" for field in blocking_fields):
        return "partially_covered"
    if any(metrics[field] == "not_covered" for field in optional_binding_fields):
        return "not_covered"
    if any(metrics[field] == "partially_covered" for field in optional_binding_fields):
        return "partially_covered"
    if any(metrics[field] == "covered_with_warnings" for field in metrics):
        return "covered_with_minor_warnings"
    return "covered"


def coverage_status(metrics: dict[str, str], subset: str) -> str:
    if subset != "formal_confirmed":
        return "not_evaluable"
    completeness = metrics["evidence_completeness"]
    if completeness == "not_evaluable":
        return "not_evaluable"
    if completeness == "not_covered":
        return "fail"
    if completeness == "partially_covered":
        return "partial"
    if completeness == "covered_with_minor_warnings":
        return "pass_with_warnings"
    return "pass"


def excluded_record(gold: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    metrics = {
        "table_object_source_coverage": "not_applicable",
        "row_gold_coverage": "not_applicable",
        "column_gold_coverage": "not_applicable",
        "cell_gold_coverage": "not_applicable",
        "value_coverage": "not_applicable",
        "unit_binding_coverage": "not_applicable",
        "literal_binding_coverage": "not_applicable",
        "footnote_reference_coverage": "not_applicable",
        "source_span_coverage": "not_applicable",
        "evidence_completeness": "not_evaluable",
        "answerability_calibration": "not_applicable",
    }
    notes.append("excluded from formal subset because gold_status is needs_manual_review.")
    return {
        "gold_id": gold.get("gold_id"),
        "table_object_id": gold.get("table_object_id"),
        "sample_id": gold.get("sample_id"),
        "doc_id": gold.get("doc_id"),
        "table_id": gold.get("table_id"),
        "subset": "excluded_from_formal",
        "gold_status": gold.get("gold_status"),
        "coverage_status": "not_evaluable",
        **metrics,
        "blocking_warnings": gold.get("blocking_warnings", []),
        "nonblocking_warnings": gold.get("nonblocking_warnings", []),
        "notes": " ".join(notes),
    }


def evaluate_gold(
    gold: dict[str, Any],
    table_objects: dict[str, dict[str, Any]],
    consistency_summary: dict[str, dict[str, str]],
    validation_summary: dict[str, dict[str, str]],
) -> dict[str, Any]:
    subset = subset_for_gold(gold)
    notes: list[str] = []
    if subset == "excluded_from_formal":
        return excluded_record(gold, notes)

    table_object = table_objects.get(str(gold.get("table_object_id")))
    consistency_row = consistency_summary.get(str(gold.get("gold_id")))
    validation_row = validation_summary.get(str(gold.get("table_object_id")))

    metrics = {
        "table_object_source_coverage": evaluate_table_object_source(
            gold, table_object, validation_row, notes
        ),
        "row_gold_coverage": evaluate_rows(gold, table_object, notes),
        "column_gold_coverage": evaluate_columns(gold, table_object, notes),
        "cell_gold_coverage": evaluate_cells(gold, table_object, notes),
        "value_coverage": evaluate_values(gold, table_object, notes),
        "unit_binding_coverage": evaluate_units(gold, table_object, notes),
        "literal_binding_coverage": evaluate_literals(gold, table_object, notes),
        "footnote_reference_coverage": evaluate_footnotes_references(
            gold, table_object, notes
        ),
        "source_span_coverage": evaluate_source_spans(gold, table_object, notes),
        "answerability_calibration": evaluate_answerability(
            gold, consistency_row, subset, notes
        ),
    }
    metrics["evidence_completeness"] = evidence_completeness(metrics, subset)
    status = coverage_status(metrics, subset)

    blocking_warnings = unique_values(
        gold.get("blocking_warnings", []),
        semicolon_values(consistency_row.get("blocking_warnings") if consistency_row else ""),
        semicolon_values(validation_row.get("blocking_warnings") if validation_row else ""),
    )
    nonblocking_warnings = unique_values(
        gold.get("nonblocking_warnings", []),
        table_object.get("warnings", []) if table_object else [],
        semicolon_values(consistency_row.get("nonblocking_warnings") if consistency_row else ""),
        semicolon_values(validation_row.get("nonblocking_warnings") if validation_row else ""),
    )
    if nonblocking_warnings:
        notes.append("nonblocking warnings retained; coverage is not production readiness.")

    return {
        "gold_id": gold.get("gold_id"),
        "table_object_id": gold.get("table_object_id"),
        "sample_id": gold.get("sample_id"),
        "doc_id": gold.get("doc_id"),
        "table_id": gold.get("table_id"),
        "subset": subset,
        "gold_status": gold.get("gold_status"),
        "coverage_status": status,
        **metrics,
        "blocking_warnings": blocking_warnings,
        "nonblocking_warnings": nonblocking_warnings,
        "notes": " ".join(notes),
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
            row["blocking_warnings"] = ";".join(record.get("blocking_warnings", []))
            row["nonblocking_warnings"] = ";".join(record.get("nonblocking_warnings", []))
            writer.writerow(row)


def formal_subset_passes(records: list[dict[str, Any]]) -> bool:
    formal = [record for record in records if record["subset"] == "formal_confirmed"]
    if len(formal) != 2:
        return False
    return all(record["coverage_status"] in {"pass", "pass_with_warnings"} for record in formal)


def main() -> None:
    table_objects_list = load_jsonl(TABLE_OBJECTS_PATH)
    gold_rows = load_jsonl(ROW_CELL_GOLD_PATH)
    consistency_summary = load_csv_by_key(GOLD_CONSISTENCY_SUMMARY_PATH, "gold_id")
    validation_summary = load_csv_by_key(TABLE_VALIDATION_SUMMARY_PATH, "table_object_id")

    table_objects = {str(item["table_object_id"]): item for item in table_objects_list}
    gold_by_id = {str(item["gold_id"]): item for item in gold_rows}
    missing = [gold_id for gold_id in GOLD_ORDER if gold_id not in gold_by_id]
    if missing:
        raise SystemExit(f"Missing expected F-5 gold rows: {', '.join(missing)}")

    records = [
        evaluate_gold(gold_by_id[gold_id], table_objects, consistency_summary, validation_summary)
        for gold_id in GOLD_ORDER
    ]
    write_json(OUTPUT_JSON, records)
    write_csv(OUTPUT_CSV, records)

    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"formal_subset_passes={str(formal_subset_passes(records)).lower()}")


if __name__ == "__main__":
    main()
