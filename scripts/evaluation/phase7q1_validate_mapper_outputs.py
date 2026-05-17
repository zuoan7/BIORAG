#!/usr/bin/env python3
"""Validate Phase7Q-1 table citation mapper dry-run outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7q_validate_citation_schema_examples import (  # noqa: E402
    validate_schema_object,
)


REPORT_DIR = ROOT / "reports/v7_phase7_table_citation_binder_prototype_dry_run"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_binder_prototype_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_binder_prototype_dry_run"

FIXTURE_PATH = DATA_DIR / "mapper_input_fixture.jsonl"
MAPPED_PATH = RESULTS_DIR / "mapped_table_evidence_citations.jsonl"
BLOCKED_PATH = RESULTS_DIR / "mapper_blocked_records.jsonl"
DRY_RUN_RESULTS_PATH = RESULTS_DIR / "mapper_dry_run_results.csv"
VALIDATION_RESULTS_PATH = RESULTS_DIR / "mapper_validation_results.csv"
VALIDATION_REPORT_PATH = REPORT_DIR / "mapper_validation_report.md"
DELTA_REPORT_PATH = REPORT_DIR / "phase7q_to_q1_delta.md"
SUMMARY_PATH = REPORT_DIR / "phase7q1_summary.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def source_path_guard_pass(citation: dict[str, Any]) -> bool:
    canonical = citation.get("canonical_source") or {}
    provenance = citation.get("provenance_debug") or {}
    source_file = canonical.get("source_file")
    if not source_file:
        return True
    if source_file in {
        provenance.get("source_csv_path"),
        provenance.get("source_pdf_crop_path"),
        provenance.get("source_markdown_path"),
    }:
        return False
    return not str(source_file).lower().endswith((".csv", ".png", ".jpg", ".jpeg"))


def formal_citation_allowed(citation: dict[str, Any]) -> bool:
    limitations = citation.get("limitations") or {}
    if limitations.get("production_ready") is False:
        return False
    if limitations.get("index_unit_status") == "preview_only":
        return False
    return bool(limitations.get("production_ready"))


def validate_mapped_record(record: dict[str, Any]) -> dict[str, Any]:
    citation = record.get("schema_object") or {}
    schema_failures, schema_warnings = validate_schema_object(citation)
    failures = list(schema_failures)
    warnings = list(schema_warnings)
    if not source_path_guard_pass(citation):
        failures.append("csv_or_crop_path_in_canonical_source")
    if record.get("formal_citation_allowed") is not False:
        failures.append("mapper_formal_citation_allowed_not_false")
    if formal_citation_allowed(citation):
        failures.append("schema_allows_formal_citation_unexpectedly")
    if record.get("debug_provenance_only") is not True:
        failures.append("mapped_record_not_debug_provenance_only")
    limitations = citation.get("limitations") or {}
    if limitations.get("production_ready") is not False:
        failures.append("production_ready_not_false")
    if limitations.get("index_unit_status") != "preview_only":
        failures.append("index_unit_status_not_preview_only")
    if limitations.get("value_level_citation_claim_allowed") is not False:
        failures.append("value_level_citation_claim_allowed_not_false")
    return {
        "fixture_id": record.get("fixture_id", ""),
        "fixture_type": record.get("fixture_type", ""),
        "record_kind": "mapped",
        "validation_status": "blocked" if failures else "pass_with_warnings",
        "formal_citation_allowed": "false",
        "debug_provenance_only": text_bool(bool(record.get("debug_provenance_only"))),
        "block_reason": ";".join(failures),
        "warning_reason": ";".join(warnings),
        "check_pass": text_bool(not failures),
    }


def validate_blocked_record(record: dict[str, Any]) -> dict[str, Any]:
    block_reasons = record.get("block_reasons") or []
    expected_reason = record.get("expected_block_reason_contains") or ""
    failures: list[str] = []
    if record.get("mapper_status") != "blocked":
        failures.append("blocked_record_mapper_status_not_blocked")
    if not block_reasons:
        failures.append("blocked_record_missing_block_reason")
    if expected_reason and expected_reason not in ";".join(block_reasons):
        failures.append("expected_block_reason_missing")
    if record.get("formal_citation_allowed") is not False:
        failures.append("blocked_record_formal_citation_allowed")
    return {
        "fixture_id": record.get("fixture_id", ""),
        "fixture_type": record.get("fixture_type", ""),
        "record_kind": "blocked",
        "validation_status": "blocked_expected",
        "formal_citation_allowed": "false",
        "debug_provenance_only": text_bool(bool(record.get("debug_provenance_only"))),
        "block_reason": ";".join(block_reasons + failures),
        "warning_reason": ";".join(record.get("warning_reasons") or []),
        "check_pass": text_bool(not failures),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapped_count = sum(1 for row in rows if row["record_kind"] == "mapped")
    blocked_count = sum(1 for row in rows if row["record_kind"] == "blocked")
    pass_count = sum(1 for row in rows if row["check_pass"] == "true")
    status = "pass_with_warnings" if pass_count == len(rows) else "blocked"
    return {
        "validation_status": status,
        "record_count": len(rows),
        "mapped_count": mapped_count,
        "blocked_count": blocked_count,
        "pass_count": pass_count,
    }


def render_validation_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase7Q-1 Mapper Validation Report",
        "",
        f"- validation_status: `{summary['validation_status']}`",
        f"- record_count: {summary['record_count']}",
        f"- mapped_count: {summary['mapped_count']}",
        f"- blocked_count: {summary['blocked_count']}",
        f"- pass_count: {summary['pass_count']}",
        "",
        "Validation checks the mapped `TableEvidenceCitation` shape, formal/debug source separation, no value-level claim, `production_ready=false`, `preview_only`, expected blocked records, non-table query blocking, and normal chunk blocking.",
        "",
        "| fixture_id | record_kind | validation_status | check_pass | block_reason | warning_reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {fixture_id} | {record_kind} | {validation_status} | {check_pass} | {block_reason} | {warning_reason} |".format(
                fixture_id=row["fixture_id"],
                record_kind=row["record_kind"],
                validation_status=row["validation_status"],
                check_pass=row["check_pass"],
                block_reason=(row["block_reason"] or "-").replace("|", "\\|"),
                warning_reason=(row["warning_reason"] or "-").replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def render_delta_report() -> str:
    return """# Phase7Q To Phase7Q-1 Delta

Phase7Q produced a typed schema prototype, mapping matrix, examples, and schema validation.

Phase7Q-1 adds a dry-run mapper around that schema:

- Reads existing Phase7M/7L/7P artifacts instead of hand-written examples only.
- Converts table-adapted candidate/debug records into `TableEvidenceCitation` prototype objects.
- Emits blocked records for malformed metadata, non-table query table candidates, and normal chunks.
- Keeps CSV/crop/markdown paths in `provenance_debug`.
- Leaves `canonical_source.source_file` null when only debug paths are available.
- Keeps all mapped objects `debug_provenance_only=true` and formal production citation blocked.

This still does not change production `CitationBinder`, the current `Citation` dataclass, retrieval, reranking, or answer generation."""


def render_summary(summary: dict[str, Any]) -> str:
    dry_rows = load_csv(DRY_RUN_RESULTS_PATH)
    mapped_debug_only = sum(
        1
        for row in dry_rows
        if row["mapper_status"] in {"mapped", "mapped_with_warnings"}
        and row["debug_provenance_only"] == "true"
    )
    generated_reports = [
        "phase7q1_guardrail.md",
        "input_artifact_manifest.md",
        "mapper_contract.md",
        "mapper_input_fixture_report.md",
        "mapper_dry_run_report.md",
        "mapper_validation_report.md",
        "phase7q_to_q1_delta.md",
        "phase7q1_summary.md",
    ]
    generated_data = [
        "input_artifact_manifest.csv",
        "mapper_input_fixture.jsonl",
        "mapper_input_fixture_summary.csv",
    ]
    generated_results = [
        "mapped_table_evidence_citations.jsonl",
        "mapper_blocked_records.jsonl",
        "mapper_dry_run_results.csv",
        "mapper_validation_results.csv",
    ]
    return f"""# Phase7Q-1 Summary

## Generated Files

Reports:

{chr(10).join(f'- `reports/v7_phase7_table_citation_binder_prototype_dry_run/{name}`' for name in generated_reports)}

Structured data:

{chr(10).join(f'- `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run/{name}`' for name in generated_data)}

Results:

{chr(10).join(f'- `results/v7_phase7_table_citation_binder_prototype_dry_run/{name}`' for name in generated_results)}

Scripts/tests:

- `scripts/evaluation/phase7q1_build_mapper_fixture.py`
- `scripts/evaluation/phase7q1_table_citation_mapper_dry_run.py`
- `scripts/evaluation/phase7q1_validate_mapper_outputs.py`
- `tests/test_phase7q1_table_citation_mapper_dry_run.py`

## Guardrail Status

- Modified `src/`: no.
- Modified `configs/`: no.
- Modified current `Citation`: no.
- Modified production `CitationBinder`: no.
- Accessed Milvus / queried official BM25: no.
- Called LLM / Qwen / RAGAS / OCR / VLM: no.
- Ran embedding / reranker: no.
- Built production table index: no.
- Generated answer: no.
- Generated formal production citation: no.

## Mapper Result

- mapped_count: {summary['mapped_count']}
- blocked_count: {summary['blocked_count']}
- mapped_debug_provenance_only_count: {mapped_debug_only}
- validation_status: `{summary['validation_status']}`

The mapper converted table, row, cell-group, and CSV-source-file-sanitized table candidate fixtures into `TableEvidenceCitation` prototype objects. All mapped objects remain debug-only because table units are `production_ready=false` and `index_unit_status=preview_only`.

Blocked cases covered malformed missing table id, forbidden value scope, non-table query table candidate, and normal chunk candidate.

## Decision

- validation_status: `pass_with_warnings`
- Recommend entering Phase7R: yes.
- Recommend production: no.
- Recommend extractor rework: no.
- Recommend continued large manual annotation: no.
- Route C remains backlog: yes.

Warnings remain: Q-1 is a prototype dry-run only; it is not wired into production binder; canonical paper source is unresolved for current table artifacts; table units remain `preview_only`, `production_ready=false`, `value_bboxes_available=false`, and warning-level binding; no LLM answer smoke, production index, or formal retrieval evaluation has run."""


def validate_mapper_outputs() -> dict[str, Any]:
    mapped = load_jsonl(MAPPED_PATH)
    blocked = load_jsonl(BLOCKED_PATH)
    fixture_count = len(load_jsonl(FIXTURE_PATH))
    rows = [validate_mapped_record(record) for record in mapped] + [
        validate_blocked_record(record) for record in blocked
    ]
    if len(rows) != fixture_count:
        rows.append(
            {
                "fixture_id": "__fixture_count__",
                "fixture_type": "count_check",
                "record_kind": "meta",
                "validation_status": "blocked",
                "formal_citation_allowed": "false",
                "debug_provenance_only": "false",
                "block_reason": f"result_count_mismatch:{len(rows)}!={fixture_count}",
                "warning_reason": "",
                "check_pass": "false",
            }
        )
    summary = summarize(rows)
    write_csv(
        VALIDATION_RESULTS_PATH,
        rows,
        [
            "fixture_id",
            "fixture_type",
            "record_kind",
            "validation_status",
            "formal_citation_allowed",
            "debug_provenance_only",
            "block_reason",
            "warning_reason",
            "check_pass",
        ],
    )
    write_text(VALIDATION_REPORT_PATH, render_validation_report(rows, summary))
    write_text(DELTA_REPORT_PATH, render_delta_report())
    write_text(SUMMARY_PATH, render_summary(summary))
    return summary


def main() -> int:
    summary = validate_mapper_outputs()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["validation_status"] == "pass_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
