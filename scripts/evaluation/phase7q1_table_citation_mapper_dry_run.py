#!/usr/bin/env python3
"""Run Phase7Q-1 table citation mapper dry-run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_binder_prototype_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_binder_prototype_dry_run"
REPORT_DIR = ROOT / "reports/v7_phase7_table_citation_binder_prototype_dry_run"

FIXTURE_PATH = DATA_DIR / "mapper_input_fixture.jsonl"
MAPPED_PATH = RESULTS_DIR / "mapped_table_evidence_citations.jsonl"
BLOCKED_PATH = RESULTS_DIR / "mapper_blocked_records.jsonl"
RESULTS_PATH = RESULTS_DIR / "mapper_dry_run_results.csv"
REPORT_PATH = REPORT_DIR / "mapper_dry_run_report.md"

VALID_TABLE_UNIT_TYPES = {"table_unit", "row_unit", "cell_group_unit"}
VALID_CITATION_SCOPES = {"table", "row", "cell_group"}
UNIT_TO_SCOPE = {
    "table_unit": "table",
    "row_unit": "row",
    "cell_group_unit": "cell_group",
}
UNIT_TO_QUOTE_SCOPE = {
    "table_unit": "table_summary",
    "row_unit": "row_summary",
    "cell_group_unit": "cell_group_summary",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def text_bool(value: bool) -> str:
    return "true" if value else "false"


def is_debug_path(value: Any, provenance: dict[str, Any]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value in {
        provenance.get("source_csv_path"),
        provenance.get("source_pdf_crop_path"),
        provenance.get("source_markdown_path"),
    }:
        return True
    lowered = value.lower()
    return lowered.endswith((".csv", ".png", ".jpg", ".jpeg"))


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def bool_from_metadata(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def binding_review_level(metadata: dict[str, Any]) -> str:
    value = str(metadata.get("binding_review_level") or metadata.get("binding_review_limitation") or "")
    if value in {"reviewed", "verified"}:
        return value
    return "warning"


def source_span_granularity(metadata: dict[str, Any], table_unit_type: str) -> str:
    value = metadata.get("source_span_granularity")
    if value in {"table", "row", "cell_group", "cell_level", "table_row_level"}:
        return value
    if table_unit_type == "table_unit":
        return "table"
    if table_unit_type == "row_unit":
        return "table_row_level"
    return "cell_group"


def citation_id(fixture: dict[str, Any]) -> str:
    return f"phase7q1::{fixture['fixture_id']}"


def block_record(
    fixture: dict[str, Any],
    block_reasons: list[str],
    warning_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture["fixture_id"],
        "fixture_type": fixture["fixture_type"],
        "query_id": fixture["query_id"],
        "query_type": fixture["query_type"],
        "mapper_status": "blocked",
        "block_reasons": block_reasons,
        "warning_reasons": warning_reasons or [],
        "expected_mapper_status": fixture["expected_mapper_status"],
        "expected_block_reason_contains": fixture["expected_block_reason_contains"],
        "formal_citation_allowed": False,
        "debug_provenance_only": False,
    }


def map_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    chunk = fixture.get("retrieved_chunk") or {}
    metadata = chunk.get("metadata") or {}
    block_reasons: list[str] = []
    warning_reasons: list[str] = []

    object_type = metadata.get("object_type")
    if object_type != "table_index_unit":
        block_reasons.append("normal_chunk_not_table_evidence")

    if fixture.get("query_type") == "non_table_query":
        block_reasons.append("non_table_query_blocks_table_citation")

    doc_id = first_nonempty(metadata.get("doc_id"), chunk.get("doc_id"))
    table_id = metadata.get("table_id")
    table_unit_type = metadata.get("table_unit_type")
    quote_text = first_nonempty(metadata.get("retrieval_text"), chunk.get("text"))

    if not doc_id:
        block_reasons.append("missing_doc_id")
    if object_type == "table_index_unit" and not table_id:
        block_reasons.append("missing_table_id")
    if object_type == "table_index_unit" and table_unit_type not in VALID_TABLE_UNIT_TYPES:
        block_reasons.append("invalid_table_unit_type")
    if object_type == "table_index_unit" and not quote_text:
        block_reasons.append("missing_quote_text")

    forced_scope = metadata.get("forced_citation_scope")
    citation_scope = forced_scope or UNIT_TO_SCOPE.get(str(table_unit_type), "")
    if citation_scope == "value":
        block_reasons.append("citation_scope_value_forbidden")
    if object_type == "table_index_unit" and citation_scope not in VALID_CITATION_SCOPES:
        block_reasons.append("invalid_citation_scope")

    provenance = {
        "source_csv_path": metadata.get("source_csv_path"),
        "source_pdf_crop_path": metadata.get("source_pdf_crop_path"),
        "source_markdown_path": metadata.get("source_markdown_path"),
        "table_index_unit_id": metadata.get("table_index_unit_id"),
        "seed_id": metadata.get("seed_id"),
        "candidate_id": metadata.get("candidate_id"),
    }

    if block_reasons:
        return block_record(fixture, block_reasons)

    candidate_source_file = chunk.get("source_file")
    canonical_source_file = metadata.get("canonical_source_file")
    if not canonical_source_file and not is_debug_path(candidate_source_file, provenance):
        canonical_source_file = candidate_source_file
    if is_debug_path(canonical_source_file, provenance):
        canonical_source_file = None
    if not canonical_source_file:
        warning_reasons.append("canonical_source_file_unresolved")

    production_ready = bool_from_metadata(metadata.get("production_ready"), default=False)
    index_unit_status = metadata.get("index_unit_status") or "preview_only"
    value_bboxes_available = bool_from_metadata(
        metadata.get("value_bboxes_available"), default=False
    )
    if production_ready is False:
        warning_reasons.append("production_ready_false_blocks_formal_citation")
    if index_unit_status == "preview_only":
        warning_reasons.append("preview_only_blocks_formal_citation")
    if value_bboxes_available is False:
        warning_reasons.append("value_bboxes_unavailable")
    review_level = binding_review_level(metadata)
    if review_level == "warning":
        warning_reasons.append("binding_warning_level")

    citation = {
        "citation_type": "table_evidence",
        "citation_id": citation_id(fixture),
        "doc_id": str(doc_id),
        "canonical_source": {
            "paper_title": metadata.get("paper_title"),
            "source_file": canonical_source_file,
            "doi": metadata.get("doi"),
            "pmid": metadata.get("pmid"),
        },
        "table_scope": {
            "table_id": str(table_id),
            "table_caption": metadata.get("caption"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
        },
        "evidence_scope": {
            "table_unit_type": table_unit_type,
            "citation_scope": citation_scope,
            "row_label": metadata.get("row_label"),
            "header_path": metadata.get("header_path") or [],
            "source_span_granularity": source_span_granularity(metadata, str(table_unit_type)),
        },
        "quote": {
            "text": str(quote_text),
            "quote_scope": UNIT_TO_QUOTE_SCOPE[str(table_unit_type)],
        },
        "provenance_debug": provenance,
        "limitations": {
            "production_ready": False,
            "index_unit_status": "preview_only",
            "value_bboxes_available": False,
            "cell_bboxes_available": metadata.get("cell_bboxes_available"),
            "binding_review_level": review_level,
            "bbox_verification_level": "table" if metadata.get("cell_bboxes_available") else "none",
            "value_level_citation_claim_allowed": False,
        },
    }
    formal_allowed = False
    return {
        "fixture_id": fixture["fixture_id"],
        "fixture_type": fixture["fixture_type"],
        "query_id": fixture["query_id"],
        "query_type": fixture["query_type"],
        "mapper_status": "mapped_with_warnings" if warning_reasons else "mapped",
        "warning_reasons": warning_reasons,
        "block_reasons": [],
        "formal_citation_allowed": formal_allowed,
        "debug_provenance_only": not formal_allowed,
        "expected_mapper_status": fixture["expected_mapper_status"],
        "expected_block_reason_contains": fixture["expected_block_reason_contains"],
        "schema_object": citation,
    }


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    expected = result.get("expected_mapper_status")
    expected_block = result.get("expected_block_reason_contains") or ""
    block_reason = ";".join(result.get("block_reasons") or [])
    status_ok = result["mapper_status"] == expected
    if expected == "mapped_with_warnings":
        status_ok = result["mapper_status"] in {"mapped", "mapped_with_warnings"}
    reason_ok = True
    if expected_block:
        reason_ok = expected_block in block_reason
    return {
        "fixture_id": result["fixture_id"],
        "fixture_type": result["fixture_type"],
        "query_type": result["query_type"],
        "mapper_status": result["mapper_status"],
        "expected_mapper_status": expected,
        "formal_citation_allowed": text_bool(bool(result.get("formal_citation_allowed"))),
        "debug_provenance_only": text_bool(bool(result.get("debug_provenance_only"))),
        "block_reason": block_reason,
        "warning_reason": ";".join(result.get("warning_reasons") or []),
        "expectation_pass": text_bool(status_ok and reason_ok),
    }


def render_report(rows: list[dict[str, Any]]) -> str:
    mapped_count = sum(1 for row in rows if row["mapper_status"] in {"mapped", "mapped_with_warnings"})
    blocked_count = sum(1 for row in rows if row["mapper_status"] == "blocked")
    expectation_pass = sum(1 for row in rows if row["expectation_pass"] == "true")
    lines = [
        "# Phase7Q-1 Mapper Dry-Run Report",
        "",
        f"- input_count: {len(rows)}",
        f"- mapped_count: {mapped_count}",
        f"- blocked_count: {blocked_count}",
        f"- expectation_pass_count: {expectation_pass}",
        "- production_citation_count: 0",
        "- answer_generated: false",
        "",
        "| fixture_id | mapper_status | formal_citation_allowed | debug_provenance_only | expectation_pass | block_reason | warning_reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {fixture_id} | {mapper_status} | {formal_citation_allowed} | {debug_provenance_only} | {expectation_pass} | {block_reason} | {warning_reason} |".format(
                fixture_id=row["fixture_id"],
                mapper_status=row["mapper_status"],
                formal_citation_allowed=row["formal_citation_allowed"],
                debug_provenance_only=row["debug_provenance_only"],
                expectation_pass=row["expectation_pass"],
                block_reason=(row["block_reason"] or "-").replace("|", "\\|"),
                warning_reason=(row["warning_reason"] or "-").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "The mapper is intentionally standalone and does not import or modify production `CitationBinder`. CSV/crop source paths are retained only under `provenance_debug`; unresolved canonical source files remain `null` rather than being replaced by debug paths.",
        ]
    )
    return "\n".join(lines)


def run_mapper_dry_run() -> dict[str, Any]:
    fixtures = load_jsonl(FIXTURE_PATH)
    results = [map_fixture(fixture) for fixture in fixtures]
    mapped = [result for result in results if result["mapper_status"] in {"mapped", "mapped_with_warnings"}]
    blocked = [result for result in results if result["mapper_status"] == "blocked"]
    rows = [result_row(result) for result in results]

    write_jsonl(MAPPED_PATH, mapped)
    write_jsonl(BLOCKED_PATH, blocked)
    write_csv(
        RESULTS_PATH,
        rows,
        [
            "fixture_id",
            "fixture_type",
            "query_type",
            "mapper_status",
            "expected_mapper_status",
            "formal_citation_allowed",
            "debug_provenance_only",
            "block_reason",
            "warning_reason",
            "expectation_pass",
        ],
    )
    write_text(REPORT_PATH, render_report(rows))
    return {
        "input_count": len(results),
        "mapped_count": len(mapped),
        "blocked_count": len(blocked),
        "expectation_pass_count": sum(1 for row in rows if row["expectation_pass"] == "true"),
    }


def main() -> int:
    summary = run_mapper_dry_run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["expectation_pass_count"] == summary["input_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
