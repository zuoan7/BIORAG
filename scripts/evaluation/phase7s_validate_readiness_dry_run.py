#!/usr/bin/env python3
"""Validate Phase7S canonical source and production readiness dry-run outputs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7s_production_readiness_gate_dry_run import (  # noqa: E402
    BLOCKER_SUMMARY_PATH,
    GATE_DRY_RUN_PATH,
    render_summary,
)


REPORT_DIR = ROOT / "reports/v7_phase7_table_production_readiness_dry_run"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_production_readiness_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_production_readiness_dry_run"

UNIT_JSONL_PATH = (
    ROOT
    / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
MANIFEST_PATH = DATA_DIR / "canonical_source_manifest.draft.jsonl"
CANONICAL_SUMMARY_PATH = DATA_DIR / "canonical_source_resolution_summary.csv"
VALIDATION_SUMMARY_PATH = RESULTS_DIR / "phase7s_validation_summary.csv"
PHASE7S_SUMMARY_PATH = REPORT_DIR / "phase7s_summary.md"

DEBUG_PATH_KEYS = {
    "source_csv_path",
    "source_pdf_crop_path",
    "source_markdown_path",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} did not parse to an object")
            rows.append(value)
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


def check_row(check_id: str, passed: bool, details: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "check_pass": text_bool(passed),
        "status": "pass" if passed else "fail",
        "details": details,
    }


def input_preflight_executed() -> tuple[bool, str]:
    rows = load_csv(CANONICAL_SUMMARY_PATH)
    ok = any(
        row.get("metric") == "input_preflight"
        and row.get("key") == "pass"
        and row.get("count") == "1"
        for row in rows
    )
    return ok, "input_preflight pass row present" if ok else "input_preflight pass row missing"


def unit_counts_match() -> tuple[bool, str]:
    units = load_jsonl(UNIT_JSONL_PATH)
    manifest = load_jsonl(MANIFEST_PATH)
    gates = load_csv(GATE_DRY_RUN_PATH)
    ok = len(units) == len(manifest) == len(gates)
    return (
        ok,
        f"units={len(units)},manifest={len(manifest)},gate_rows={len(gates)}",
    )


def canonical_manifest_parseable() -> tuple[bool, str]:
    rows = load_jsonl(MANIFEST_PATH)
    required = {
        "table_index_unit_id",
        "doc_id",
        "table_id",
        "table_caption",
        "page",
        "candidate_id",
        "seed_id",
        "canonical_source_status",
        "canonical_source_confidence",
        "canonical_source_fields_available",
        "canonical_source_missing_fields",
        "formal_source_allowed",
        "debug_provenance_paths",
        "notes",
    }
    missing_rows = [
        row.get("table_index_unit_id", f"row_{index}")
        for index, row in enumerate(rows)
        if not required.issubset(row)
    ]
    return (
        not missing_rows,
        "manifest rows parse and include required fields"
        if not missing_rows
        else "missing_fields_for=" + ",".join(missing_rows[:10]),
    )


def gate_dry_run_parseable() -> tuple[bool, str]:
    rows = load_csv(GATE_DRY_RUN_PATH)
    required = {
        "table_index_unit_id",
        "doc_id",
        "table_id",
        "unit_type",
        "gate_status",
        "failed_gates",
        "blocker_category",
        "blocker_reason",
        "can_be_fixed_by_metadata",
        "requires_binder_integration",
        "requires_production_index_build",
        "requires_future_canary_or_answer_smoke",
        "recommended_next_action",
    }
    missing = required - set(rows[0]) if rows else required
    return (
        bool(rows) and not missing,
        "gate dry-run CSV parseable"
        if rows and not missing
        else "missing_columns=" + ",".join(sorted(missing)),
    )


def debug_paths_not_formal_source() -> tuple[bool, str]:
    rows = load_jsonl(MANIFEST_PATH)
    failures: list[str] = []
    for row in rows:
        debug_paths = row.get("debug_provenance_paths") or {}
        if not isinstance(debug_paths, dict):
            failures.append(f"{row.get('table_index_unit_id')}:debug_paths_not_object")
            continue
        if not set(debug_paths).issubset(DEBUG_PATH_KEYS):
            failures.append(f"{row.get('table_index_unit_id')}:unexpected_debug_path_keys")
        row_without_debug = {
            key: value for key, value in row.items() if key != "debug_provenance_paths"
        }
        serialized = json.dumps(row_without_debug, ensure_ascii=False)
        for path_value in debug_paths.values():
            if path_value and path_value in serialized:
                failures.append(f"{row.get('table_index_unit_id')}:debug_path_in_formal_fields")
                break
        if row.get("formal_source_allowed") is True:
            failures.append(f"{row.get('table_index_unit_id')}:formal_source_allowed_true")
    return (
        not failures,
        "CSV/crop/markdown paths are debug provenance only"
        if not failures
        else ";".join(failures[:10]),
    )


def preview_and_production_ready_blocked() -> tuple[bool, str]:
    rows = load_csv(GATE_DRY_RUN_PATH)
    preview_missing = [
        row["table_index_unit_id"]
        for row in rows
        if "index_unit_status_not_preview_only" not in row["failed_gates"]
    ]
    production_ready_missing = [
        row["table_index_unit_id"]
        for row in rows
        if "production_ready_true_independent_gate" not in row["failed_gates"]
    ]
    failures: list[str] = []
    if preview_missing:
        failures.append("preview_gate_missing=" + ",".join(preview_missing[:10]))
    if production_ready_missing:
        failures.append(
            "production_ready_gate_missing=" + ",".join(production_ready_missing[:10])
        )
    return (
        not failures,
        "preview_only and production_ready=false blocked"
        if not failures
        else ";".join(failures),
    )


def value_bboxes_block_value_level_claim() -> tuple[bool, str]:
    rows = load_csv(GATE_DRY_RUN_PATH)
    missing = [
        row["table_index_unit_id"]
        for row in rows
        if "value_level_citation_disabled_due_to_value_bboxes_available=false"
        not in row["blocker_reason"]
    ]
    return (
        not missing,
        "value_bboxes_available=false disables value-level citation claim"
        if not missing
        else "missing_value_bbox_guard=" + ",".join(missing[:10]),
    )


def blocker_classification_nonempty() -> tuple[bool, str]:
    rows = load_csv(GATE_DRY_RUN_PATH)
    allowed = {
        "data_blocker",
        "schema_or_binder_blocker",
        "operational_blocker",
        "expected_preview_blocker",
        "not_evaluable",
    }
    invalid = [
        row["table_index_unit_id"]
        for row in rows
        if row.get("blocker_category") not in allowed
    ]
    empty_reasons = [
        row["table_index_unit_id"]
        for row in rows
        if not row.get("blocker_reason")
    ]
    return (
        not invalid and not empty_reasons,
        "blocker_category and blocker_reason populated"
        if not invalid and not empty_reasons
        else f"invalid={invalid[:10]};empty_reasons={empty_reasons[:10]}",
    )


def no_src_or_configs_modified() -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "status", "--short", "--", "src", "configs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = proc.stdout.strip()
    return output == "", output or "src/ and configs/ clean"


def no_forbidden_runtime_markers() -> tuple[bool, str]:
    guardrail = (REPORT_DIR / "phase7s_guardrail.md").read_text(encoding="utf-8")
    summary = PHASE7S_SUMMARY_PATH.read_text(encoding="utf-8")
    required = [
        "Accessed Milvus / official BM25: no",
        "Ran embedding / reranker / LLM: no",
        "Built production table index: no",
    ]
    missing = [item for item in required if item not in summary]
    if "Do not access Milvus" not in guardrail or "Do not read or query official BM25" not in guardrail:
        missing.append("guardrail_missing_milvus_bm25_freeze")
    return (
        not missing,
        "forbidden runtime actions remain disabled"
        if not missing
        else "missing=" + ";".join(missing),
    )


def run_validation() -> list[dict[str, str]]:
    checks = [
        ("input_preflight_executed", input_preflight_executed),
        ("unit_counts_match", unit_counts_match),
        ("canonical_source_manifest_parseable", canonical_manifest_parseable),
        ("production_readiness_gate_dry_run_parseable", gate_dry_run_parseable),
        ("csv_crop_markdown_not_formal_source", debug_paths_not_formal_source),
        ("preview_and_production_ready_blocked", preview_and_production_ready_blocked),
        ("value_bboxes_block_value_level_claim", value_bboxes_block_value_level_claim),
        ("blocker_classification_nonempty", blocker_classification_nonempty),
        ("no_src_or_configs_modified", no_src_or_configs_modified),
        ("no_milvus_bm25_embedding_reranker_llm", no_forbidden_runtime_markers),
    ]
    rows: list[dict[str, str]] = []
    for check_id, check_fn in checks:
        try:
            passed, details = check_fn()
        except Exception as exc:  # pragma: no cover - reported in CSV
            rows.append(check_row(check_id, False, f"exception:{exc}"))
        else:
            rows.append(check_row(check_id, passed, details))
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    pass_count = sum(1 for row in rows if row["check_pass"] == "true")
    status = "pass_with_warnings" if pass_count == len(rows) else "blocked"
    return {
        "validation_status": status,
        "check_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
    }


def write_validation_artifacts() -> dict[str, Any]:
    rows = run_validation()
    summary = summarize(rows)
    write_csv(
        VALIDATION_SUMMARY_PATH,
        rows
        + [
            {
                "check_id": "validation_status",
                "check_pass": text_bool(summary["validation_status"] == "pass_with_warnings"),
                "status": summary["validation_status"],
                "details": (
                    f"pass_count={summary['pass_count']}/{summary['check_count']}"
                ),
            }
        ],
        ["check_id", "check_pass", "status", "details"],
    )
    blocker_summary = load_csv(BLOCKER_SUMMARY_PATH)
    write_text(PHASE7S_SUMMARY_PATH, render_summary(blocker_summary, summary["validation_status"]))
    return summary


def main() -> None:
    summary = write_validation_artifacts()
    print(f"validation_status={summary['validation_status']}")
    print(f"pass_count={summary['pass_count']}/{summary['check_count']}")


if __name__ == "__main__":
    main()
