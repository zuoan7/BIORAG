#!/usr/bin/env python3
"""Validate Phase7Q table citation schema prototype examples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_schema_prototype"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_schema_prototype"
REPORT_DIR = ROOT / "reports/v7_phase7_table_citation_schema_prototype"

SCHEMA_PATH = DATA_DIR / "table_evidence_citation_schema.json"
EXAMPLES_PATH = DATA_DIR / "citation_prototype_examples.jsonl"
RESULTS_PATH = RESULTS_DIR / "schema_validation_results.csv"
REPORT_PATH = REPORT_DIR / "schema_validation_report.md"

VALID_CITATION_SCOPES = {"table", "row", "cell_group"}
VALID_TABLE_UNIT_TYPES = {"table_unit", "row_unit", "cell_group_unit"}
VALID_SOURCE_SPAN_GRANULARITIES = {
    "table",
    "row",
    "cell_group",
    "cell_level",
    "table_row_level",
}
VALID_QUOTE_SCOPES = {
    "table_summary",
    "row_summary",
    "cell_group_summary",
}
VALID_BINDING_REVIEW_LEVELS = {"warning", "reviewed", "verified"}
VALID_BBOX_VERIFICATION_LEVELS = {"none", "table", "cell", "value"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def text_bool(value: bool) -> str:
    return "true" if value else "false"


def nested_get(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def has_required_shape(citation: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required_paths = [
        ("citation_type",),
        ("citation_id",),
        ("doc_id",),
        ("canonical_source",),
        ("canonical_source", "paper_title"),
        ("canonical_source", "source_file"),
        ("canonical_source", "doi"),
        ("canonical_source", "pmid"),
        ("table_scope",),
        ("table_scope", "table_id"),
        ("table_scope", "table_caption"),
        ("table_scope", "page_start"),
        ("table_scope", "page_end"),
        ("evidence_scope",),
        ("evidence_scope", "table_unit_type"),
        ("evidence_scope", "citation_scope"),
        ("evidence_scope", "row_label"),
        ("evidence_scope", "header_path"),
        ("evidence_scope", "source_span_granularity"),
        ("quote",),
        ("quote", "text"),
        ("quote", "quote_scope"),
        ("provenance_debug",),
        ("provenance_debug", "source_csv_path"),
        ("provenance_debug", "source_pdf_crop_path"),
        ("provenance_debug", "source_markdown_path"),
        ("provenance_debug", "table_index_unit_id"),
        ("provenance_debug", "seed_id"),
        ("provenance_debug", "candidate_id"),
        ("limitations",),
        ("limitations", "production_ready"),
        ("limitations", "index_unit_status"),
        ("limitations", "value_bboxes_available"),
        ("limitations", "cell_bboxes_available"),
        ("limitations", "binding_review_level"),
        ("limitations", "bbox_verification_level"),
        ("limitations", "value_level_citation_claim_allowed"),
    ]
    for path in required_paths:
        current: Any = citation
        missing = False
        for key in path:
            if not isinstance(current, dict) or key not in current:
                missing = True
                break
            current = current[key]
        if missing:
            failures.append(f"missing_required:{'.'.join(path)}")
    return failures


def string_or_null(value: Any) -> bool:
    return value is None or isinstance(value, str)


def int_or_null(value: Any) -> bool:
    return value is None or isinstance(value, int)


def validate_schema_object(citation: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures = has_required_shape(citation)
    warnings: list[str] = []

    if citation.get("citation_type") != "table_evidence":
        failures.append("citation_type_not_table_evidence")

    if not isinstance(citation.get("citation_id"), str) or not citation.get("citation_id"):
        failures.append("citation_id_missing_or_not_string")
    if not isinstance(citation.get("doc_id"), str) or not citation.get("doc_id"):
        failures.append("doc_id_missing_or_not_string")

    canonical = citation.get("canonical_source") or {}
    for field in ("paper_title", "source_file", "doi", "pmid"):
        if field in canonical and not string_or_null(canonical.get(field)):
            failures.append(f"canonical_source.{field}_not_string_or_null")

    table_scope = citation.get("table_scope") or {}
    if not isinstance(table_scope.get("table_id"), str) or not table_scope.get("table_id"):
        failures.append("table_scope.table_id_missing_or_not_string")
    if "table_caption" in table_scope and not string_or_null(table_scope.get("table_caption")):
        failures.append("table_scope.table_caption_not_string_or_null")
    for field in ("page_start", "page_end"):
        if field in table_scope and not int_or_null(table_scope.get(field)):
            failures.append(f"table_scope.{field}_not_integer_or_null")

    evidence = citation.get("evidence_scope") or {}
    table_unit_type = evidence.get("table_unit_type")
    citation_scope = evidence.get("citation_scope")
    if table_unit_type not in VALID_TABLE_UNIT_TYPES:
        failures.append("invalid_table_unit_type")
    if citation_scope not in VALID_CITATION_SCOPES:
        failures.append("invalid_citation_scope")
    if citation_scope == "value":
        failures.append("citation_scope_value_forbidden")
    if "row_label" in evidence and not string_or_null(evidence.get("row_label")):
        failures.append("evidence_scope.row_label_not_string_or_null")
    header_path = evidence.get("header_path")
    if not isinstance(header_path, list) or not all(isinstance(item, str) for item in header_path):
        failures.append("evidence_scope.header_path_not_string_array")
    if evidence.get("source_span_granularity") not in VALID_SOURCE_SPAN_GRANULARITIES:
        failures.append("invalid_source_span_granularity")

    quote = citation.get("quote") or {}
    if not isinstance(quote.get("text"), str) or not quote.get("text"):
        failures.append("quote.text_missing_or_not_string")
    if quote.get("quote_scope") not in VALID_QUOTE_SCOPES:
        failures.append("invalid_quote_scope")

    provenance = citation.get("provenance_debug") or {}
    for field in (
        "source_csv_path",
        "source_pdf_crop_path",
        "source_markdown_path",
        "table_index_unit_id",
        "seed_id",
        "candidate_id",
    ):
        if field in provenance and not string_or_null(provenance.get(field)):
            failures.append(f"provenance_debug.{field}_not_string_or_null")

    source_file = canonical.get("source_file")
    source_csv_path = provenance.get("source_csv_path")
    source_pdf_crop_path = provenance.get("source_pdf_crop_path")
    if source_csv_path and source_file == source_csv_path:
        failures.append("source_csv_path_in_canonical_source")
    if source_pdf_crop_path and source_file == source_pdf_crop_path:
        failures.append("source_pdf_crop_path_in_canonical_source")
    if isinstance(source_file, str) and source_file.endswith(".csv"):
        failures.append("canonical_source.source_file_looks_like_csv")
    if isinstance(source_file, str) and source_file.endswith((".png", ".jpg", ".jpeg")):
        failures.append("canonical_source.source_file_looks_like_crop")

    limitations = citation.get("limitations") or {}
    if limitations.get("value_level_citation_claim_allowed") is not False:
        failures.append("value_level_citation_claim_allowed_not_false")
    if limitations.get("value_bboxes_available") is False and (
        limitations.get("value_level_citation_claim_allowed") is not False
    ):
        failures.append("value_bbox_false_but_value_claim_allowed")
    if limitations.get("production_ready") is not False:
        failures.append("production_ready_not_false_for_phase7q")
    if limitations.get("index_unit_status") != "preview_only":
        failures.append("index_unit_status_not_preview_only_for_phase7q")
    if limitations.get("binding_review_level") not in VALID_BINDING_REVIEW_LEVELS:
        failures.append("invalid_binding_review_level")
    if limitations.get("bbox_verification_level") not in VALID_BBOX_VERIFICATION_LEVELS:
        failures.append("invalid_bbox_verification_level")
    if not isinstance(limitations.get("value_bboxes_available"), bool):
        failures.append("value_bboxes_available_not_boolean")
    cell_bboxes = limitations.get("cell_bboxes_available")
    if cell_bboxes is not None and not isinstance(cell_bboxes, bool):
        failures.append("cell_bboxes_available_not_boolean_or_null")
    if limitations.get("production_ready") is False:
        warnings.append("production_ready_false_blocks_formal_citation")
    if limitations.get("index_unit_status") == "preview_only":
        warnings.append("preview_only_blocks_formal_citation")
    if limitations.get("binding_review_level") == "warning":
        warnings.append("binding_warning_level")
    if limitations.get("value_bboxes_available") is False:
        warnings.append("value_bboxes_unavailable")

    return failures, warnings


def validate_example(example: dict[str, Any]) -> dict[str, Any]:
    citation = example.get("schema_object")
    failures: list[str] = []
    warnings: list[str] = []
    if not isinstance(citation, dict):
        failures.append("schema_object_missing_or_not_object")
        citation = {}
    else:
        schema_failures, schema_warnings = validate_schema_object(citation)
        failures.extend(schema_failures)
        warnings.extend(schema_warnings)

    context = example.get("example_context") or {}
    if context.get("query_type") == "non_table_query":
        failures.append("non_table_query_blocks_table_citation")
    if context.get("retrieved_chunk_object_type") not in (None, "table_index_unit"):
        failures.append("non_table_chunk_blocks_table_citation")

    formal_allowed = not failures and nested_get(citation, ("limitations", "production_ready")) is True
    if nested_get(citation, ("limitations", "index_unit_status")) == "preview_only":
        formal_allowed = False
    if nested_get(citation, ("limitations", "production_ready")) is False:
        formal_allowed = False

    expected_status = example.get("expected_validation_status")
    actual_status = "blocked" if failures else "pass_with_warnings" if warnings else "pass"
    expected_blocked = expected_status == "blocked"
    blocked_flag_matches = expected_blocked == bool(failures)
    expected_formal_allowed = bool(example.get("formal_citation_allowed"))
    formal_allowed_matches = expected_formal_allowed == formal_allowed

    check_pass = blocked_flag_matches and formal_allowed_matches
    if expected_status and expected_status != actual_status:
        check_pass = False
        failures.append(f"expected_status_mismatch:{expected_status}!={actual_status}")

    debug_only = bool(example.get("debug_provenance_only"))
    debug_only_matches = debug_only == (not formal_allowed and not failures)
    if example.get("example_type") in {"malformed_blocked", "non_table_query_blocked"}:
        debug_only_matches = debug_only is False

    if not debug_only_matches:
        check_pass = False
        failures.append("debug_provenance_only_flag_mismatch")

    return {
        "example_id": example.get("example_id", ""),
        "example_type": example.get("example_type", ""),
        "expected_validation_status": expected_status or "",
        "actual_validation_status": actual_status,
        "blocked": text_bool(bool(failures)),
        "formal_citation_allowed": text_bool(formal_allowed),
        "debug_provenance_only": text_bool(debug_only),
        "block_reason": ";".join(failures),
        "warning_reason": ";".join(warnings),
        "check_pass": text_bool(check_pass),
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for row in rows if row["check_pass"] == "true")
    blocked_count = sum(1 for row in rows if row["actual_validation_status"] == "blocked")
    warning_count = sum(1 for row in rows if row["actual_validation_status"] == "pass_with_warnings")
    validation_status = "pass_with_warnings" if pass_count == len(rows) else "blocked"
    return {
        "validation_status": validation_status,
        "example_count": len(rows),
        "pass_count": pass_count,
        "blocked_count": blocked_count,
        "pass_with_warnings_count": warning_count,
    }


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase7Q Schema Validation Report",
        "",
        f"- validation_status: `{summary['validation_status']}`",
        f"- example_count: {summary['example_count']}",
        f"- pass_count: {summary['pass_count']}",
        f"- blocked_count: {summary['blocked_count']}",
        f"- pass_with_warnings_count: {summary['pass_with_warnings_count']}",
        f"- output: `results/v7_phase7_table_citation_schema_prototype/schema_validation_results.csv`",
        "",
        "## Checks",
        "",
        "The validator checks required fields, `citation_type=table_evidence`, legal citation scopes, no `value` scope, CSV/crop path exclusion from `canonical_source`, value-level claim blocking, `production_ready=false`, `index_unit_status=preview_only`, required limitations, and blocked example labels.",
        "",
        "## Results",
        "",
        "| example_id | actual_validation_status | formal_citation_allowed | check_pass | block_reason | warning_reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {example_id} | {actual_validation_status} | {formal_citation_allowed} | {check_pass} | {block_reason} | {warning_reason} |".format(
                example_id=row["example_id"],
                actual_validation_status=row["actual_validation_status"],
                formal_citation_allowed=row["formal_citation_allowed"],
                check_pass=row["check_pass"],
                block_reason=(row["block_reason"] or "-").replace("|", "\\|"),
                warning_reason=(row["warning_reason"] or "-").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The schema prototype passes structural validation with warnings because every valid example remains `production_ready=false`, `index_unit_status=preview_only`, value bboxes are unavailable, and formal production citation remains blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def write_validation_artifacts(
    examples_path: Path = EXAMPLES_PATH,
    results_path: Path = RESULTS_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    examples = load_jsonl(examples_path)
    rows = [validate_example(example) for example in examples]
    fields = [
        "example_id",
        "example_type",
        "expected_validation_status",
        "actual_validation_status",
        "blocked",
        "formal_citation_allowed",
        "debug_provenance_only",
        "block_reason",
        "warning_reason",
        "check_pass",
    ]
    write_csv(results_path, rows, fields)
    summary = summarize_results(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rows, summary), encoding="utf-8")
    return summary


def main() -> int:
    load_json(SCHEMA_PATH)
    summary = write_validation_artifacts()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["validation_status"] == "pass_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
