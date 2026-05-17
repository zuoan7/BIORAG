#!/usr/bin/env python3
"""Isolated offline flat-vs-table_object representation comparison for Phase6F-6.

This script is intentionally narrow. It reads only the F-6 representation
extracts, F-5 coverage results, and F-4 row/cell gold. It does not perform
retrieval, embedding, reranking, model calls, BM25 access, Milvus access, or
coverage reruns.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

FLAT_EXTRACTS = ROOT / "results/v7_phase6f_table_object_fresh_batch/flat_representation_extracts.json"
TABLE_OBJECT_EXTRACTS = ROOT / "results/v7_phase6f_table_object_fresh_batch/table_object_representation_extracts.json"
COVERAGE_RESULTS = ROOT / "results/v7_phase6f_table_object_fresh_batch/offline_coverage_check_results.json"
ROW_CELL_GOLD = ROOT / "data/experiments/v7_phase6f_table_object_fresh_batch/row_cell_gold.jsonl"

OUTPUT_JSON = ROOT / "results/v7_phase6f_table_object_fresh_batch/flat_vs_table_object_comparison.json"
OUTPUT_CSV = ROOT / "results/v7_phase6f_table_object_fresh_batch/flat_vs_table_object_comparison.csv"

EXPECTED_FORMAL_GOLD_IDS = {
    "gold_doc_0322_table1_f6f_0001",
    "gold_doc_0598_table2_f6f_0005",
}

ALLOWED_CONCLUSIONS = {
    "table_object_stronger",
    "flat_sufficient",
    "mixed_or_inconclusive",
    "not_evaluable",
}

CSV_FIELDS = [
    "gold_id",
    "table_object_id",
    "sample_id",
    "doc_id",
    "table_id",
    "subset",
    "flat_evidence_completeness",
    "table_object_evidence_completeness",
    "flat_row_identity_clarity",
    "table_object_row_identity_clarity",
    "flat_column_identity_clarity",
    "table_object_column_identity_clarity",
    "flat_cell_identity_clarity",
    "table_object_cell_identity_clarity",
    "flat_value_clarity",
    "table_object_value_clarity",
    "flat_unit_binding_clarity",
    "table_object_unit_binding_clarity",
    "flat_literal_binding_clarity",
    "table_object_literal_binding_clarity",
    "flat_footnote_reference_binding_clarity",
    "table_object_footnote_reference_binding_clarity",
    "flat_source_span_clarity",
    "table_object_source_span_clarity",
    "flat_warning_visibility",
    "table_object_warning_visibility",
    "answerability_calibration_delta",
    "formal_conclusion",
    "main_reason",
    "limitations",
    "notes",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_extract_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    records = payload if isinstance(payload, list) else payload.get("records", [])
    return {record["gold_id"]: record for record in records}


def load_gold_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["gold_id"]] = record
    return records


def require_formal_subset(coverage_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formal = [record for record in coverage_records if record.get("subset") == "formal_confirmed"]
    formal_ids = {record["gold_id"] for record in formal}
    if formal_ids != EXPECTED_FORMAL_GOLD_IDS:
        raise SystemExit(f"Unexpected formal subset: {sorted(formal_ids)}")
    if any(record.get("gold_status") != "confirmed_gold" for record in formal):
        raise SystemExit("Formal subset contains non-confirmed gold.")
    if any(record.get("coverage_status") not in {"pass", "pass_with_warnings"} for record in formal):
        raise SystemExit("Formal subset contains coverage status outside pass/pass_with_warnings.")
    return sorted(formal, key=lambda item: item["gold_id"])


def has_explicit_cell_binding(cells: list[dict[str, Any]]) -> bool:
    required = {"cell_id", "row_header_path", "col_header_path", "value_raw", "source_span_id"}
    return bool(cells) and all(required.issubset(cell.keys()) for cell in cells)


def unit_clarity(flat: dict[str, Any], table: dict[str, Any]) -> tuple[str, str]:
    flat_units = flat.get("flat_visible_units") or []
    table_units = table.get("table_object_units") or []
    if not flat_units and not table_units:
        return "not_applicable", "not_applicable"
    if flat_units and table_units:
        return "unit_visible_caption_or_flat_text_level", "unit_bound_to_columns_or_cells_with_scope"
    if flat_units:
        return "unit_visible_but_not_structured", "unit_not_structured"
    return "unit_not_visible_in_flat_extract", "unit_bound_in_table_object"


def footnote_reference_clarity(flat: dict[str, Any], table: dict[str, Any]) -> tuple[str, str]:
    flat_refs = flat.get("flat_visible_footnotes_or_references") or []
    binding = table.get("table_object_footnotes_or_references") or {}
    has_table_binding = bool(binding.get("footnotes") or binding.get("references") or binding.get("gold_binding"))
    if flat_refs and has_table_binding:
        return "visible_but_not_structured_binding", "binding_scope_explicit_with_limitations"
    if flat_refs:
        return "visible_but_not_structured_binding", "not_structured"
    if has_table_binding:
        return "not_visible_or_not_applicable", "binding_scope_explicit"
    return "not_applicable", "not_applicable"


def warning_values(table: dict[str, Any]) -> list[str]:
    warnings = table.get("table_object_warnings") or {}
    values: list[str] = []
    for item in warnings.values():
        if isinstance(item, list):
            values.extend(str(value) for value in item)
    return sorted(set(values))


def compare_record(
    coverage: dict[str, Any],
    gold: dict[str, Any],
    flat: dict[str, Any],
    table: dict[str, Any],
) -> dict[str, Any]:
    table_cells = table.get("table_object_cells") or []
    flat_values_visible = bool((flat.get("flat_value_presence_crosscheck") or {}).get("all_required_values_visible_in_flat_source_text"))
    explicit_cells = has_explicit_cell_binding(table_cells)
    flat_unit, table_unit = unit_clarity(flat, table)
    flat_note_ref, table_note_ref = footnote_reference_clarity(flat, table)
    table_warnings = warning_values(table)

    if explicit_cells and coverage.get("coverage_status") in {"pass", "pass_with_warnings"}:
        conclusion = "table_object_stronger"
    elif flat_values_visible and not explicit_cells:
        conclusion = "flat_sufficient"
    else:
        conclusion = "mixed_or_inconclusive"

    if conclusion not in ALLOWED_CONCLUSIONS:
        raise SystemExit(f"Disallowed conclusion for {coverage['gold_id']}: {conclusion}")

    if coverage["gold_id"] == "gold_doc_0322_table1_f6f_0001":
        reason = (
            "flat text 可见 4 个 energy source rows、2 个 titer headers、8 个 mean±SD values 和 g/L；"
            "table_object 进一步把这些证据拆成 row_id、column_id、cell_id、unit scope、replicate note binding 和 source_span_id。"
        )
    else:
        reason = (
            "flat text 可见 Strains/Plasmids rows、Reference or source header、construct literals 和 FRT/nt note；"
            "table_object 进一步显式区分 row/source-description cell、literal marker、reference/source column limitation 和 abbreviation note scope。"
        )

    limitations = [
        "formal subset 只有 2 条 fresh confirmed gold",
        "source_span 仍为 table_row_level",
        "无 value-level bbox",
        "本轮没有 retrieval、BM25、Milvus、Qwen、RAGAS、OCR、VLM 或 coverage rerun",
        "comparison 只支持 evidence representation-level conclusion",
    ]

    return {
        "gold_id": coverage["gold_id"],
        "table_object_id": table["table_object_id"],
        "sample_id": coverage["sample_id"],
        "doc_id": coverage["doc_id"],
        "table_id": coverage["table_id"],
        "subset": "formal_confirmed",
        "flat_evidence_completeness": "covered_in_flat_text_with_binding_limitations" if flat_values_visible else "not_fully_visible",
        "table_object_evidence_completeness": coverage["evidence_completeness"],
        "flat_row_identity_clarity": "visible_in_single_flat_text_block",
        "table_object_row_identity_clarity": "explicit_row_ids_and_header_paths",
        "flat_column_identity_clarity": "visible_headers_but_binding_inferred_from_sequence",
        "table_object_column_identity_clarity": "explicit_column_ids_indices_headers_and_binding_status",
        "flat_cell_identity_clarity": "not_explicit_cell_objects",
        "table_object_cell_identity_clarity": "explicit_cell_ids_with_row_column_paths" if explicit_cells else "not_evaluable",
        "flat_value_clarity": "value_raw_visible_in_flat_text" if flat_values_visible else "value_visibility_incomplete",
        "table_object_value_clarity": "value_raw_preserved_in_cells",
        "flat_unit_binding_clarity": flat_unit,
        "table_object_unit_binding_clarity": table_unit,
        "flat_literal_binding_clarity": "literals_visible_but_untyped",
        "table_object_literal_binding_clarity": "literal_markers_preserved_with_context",
        "flat_footnote_reference_binding_clarity": flat_note_ref,
        "table_object_footnote_reference_binding_clarity": table_note_ref,
        "flat_source_span_clarity": "chunk_block_table_row_level_only",
        "table_object_source_span_clarity": "source_span_ids_per_header_row_or_note_but_table_row_level_only",
        "flat_warning_visibility": "warnings_not_encoded_in_flat_text",
        "table_object_warning_visibility": "warnings_preserved_in_extract_and_f5_coverage" if table_warnings else "no_warnings_recorded",
        "answerability_calibration_delta": "table_object_improves_binding_and_uncertainty_visibility_without_changing_answerability",
        "formal_conclusion": conclusion,
        "main_reason": reason,
        "limitations": limitations,
        "notes": (
            f"F-4 gold_status={gold.get('gold_status')}; "
            f"F-5 coverage_status={coverage.get('coverage_status')}; "
            "结果不是 production readiness，也不是 Route C readiness。"
        ),
    }


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def main() -> None:
    flat_records = load_extract_records(FLAT_EXTRACTS)
    table_records = load_extract_records(TABLE_OBJECT_EXTRACTS)
    coverage_records = load_json(COVERAGE_RESULTS)
    gold_records = load_gold_records(ROW_CELL_GOLD)
    formal_coverage = require_formal_subset(coverage_records)

    output_records = []
    for coverage in formal_coverage:
        gold_id = coverage["gold_id"]
        gold = gold_records.get(gold_id)
        flat = flat_records.get(gold_id)
        table = table_records.get(gold_id)
        if not gold or not flat or not table:
            raise SystemExit(f"Missing input record for {gold_id}")
        if gold.get("gold_status") != "confirmed_gold":
            raise SystemExit(f"Gold is not confirmed and cannot enter formal subset: {gold_id}")
        output_records.append(compare_record(coverage, gold, flat, table))

    OUTPUT_JSON.write_text(json.dumps(output_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in output_records:
            writer.writerow({field: csv_value(record.get(field)) for field in CSV_FIELDS})

    counts: dict[str, int] = {}
    for record in output_records:
        counts[record["formal_conclusion"]] = counts.get(record["formal_conclusion"], 0) + 1
    print(json.dumps({"records": len(output_records), "formal_conclusion_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
