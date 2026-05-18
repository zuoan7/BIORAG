#!/usr/bin/env python3
"""Validate Phase7R production table index proposal artifacts."""

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

from scripts.evaluation.phase7r_build_table_index_production_proposal import (  # noqa: E402
    REPORT_PATHS,
    STRUCTURED_PATHS,
    render_summary,
)


REPORT_DIR = ROOT / "reports/v7_phase7_table_index_production_proposal"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_index_production_proposal"
RESULTS_DIR = ROOT / "results/v7_phase7_table_index_production_proposal"
VALIDATION_RESULTS_PATH = RESULTS_DIR / "phase7r_validation_results.csv"
VALIDATION_REPORT_PATH = REPORT_DIR / "phase7r_validation_report.md"
SUMMARY_PATH = REPORT_DIR / "phase7r_summary.md"

REQUIRED_PROMOTION_GATES = {
    "canonical_paper_source_resolved",
    "csv_crop_path_not_formal_source",
    "table_id_caption_page_valid",
    "table_row_cell_group_scope_valid",
    "citation_scope_not_value",
    "value_level_citation_disabled_unless_value_bboxes_verified",
    "binding_review_at_least_reviewed",
    "source_span_granularity_explicit",
    "production_ready_true_independent_gate",
    "index_unit_status_not_preview_only",
    "non_table_query_guard_enforced",
    "rollback_metadata_present",
    "typed_citation_schema_available",
    "metadata_contract_valid",
    "checksum_build_manifest_valid",
}

REQUIRED_ROLLBACK_SCENARIOS = {
    "flag_disabled",
    "table_index_unavailable",
    "table_index_schema_mismatch",
    "canonical_source_manifest_missing",
    "metadata_contract_fail",
    "citation_guard_fail",
    "production_ready_guard_fail",
    "preview_only_guard_fail",
    "reranker_high_score_bypass_attempt",
    "active_build_pointer_rollback",
    "hard_disable_to_normal_only",
}

REQUIRED_ROLLOUT_STAGES = {
    "disabled",
    "shadow_index_build",
    "shadow_retrieval_debug",
    "active_merge_dry_run",
    "support_pack_dry_run",
    "canary_no_answer",
    "canary_answer_gated",
    "production",
}

PREVIEW_BLOCKING_GATES = {
    "canonical_paper_source_resolved",
    "binding_review_at_least_reviewed",
    "production_ready_true_independent_gate",
    "index_unit_status_not_preview_only",
    "rollback_metadata_present",
    "typed_citation_schema_available",
    "metadata_contract_valid",
    "checksum_build_manifest_valid",
}

MANIFEST_REQUIRED_ARTIFACTS = {
    "production_table_units_jsonl",
    "table_unit_schema_manifest",
    "table_evidence_citation_schema_manifest",
    "canonical_source_manifest",
    "debug_provenance_manifest",
    "validation_report",
    "promotion_approval_record",
    "rollback_record",
    "checksum_manifest",
}

MANIFEST_REQUIRED_BUILD_FIELDS = {
    "table_index_version",
    "table_index_build_id",
    "table_index_unit_schema_version",
    "source_corpus_snapshot_id",
    "canonical_source_manifest_id",
    "promotion_approval_id",
    "rollback_manifest_id",
    "table_index_quality_gate_status",
    "checksum_manifest_id",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not parse to an object")
    return value


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


def required_reports_exist() -> tuple[bool, str]:
    missing = [
        str(REPORT_DIR / path)
        for path in REPORT_PATHS
        if not (REPORT_DIR / path).exists()
    ]
    return not missing, "missing=" + ";".join(missing) if missing else "all required reports exist"


def required_structured_files_parse() -> tuple[bool, str]:
    errors: list[str] = []
    for name in STRUCTURED_PATHS:
        path = DATA_DIR / name
        if not path.exists():
            errors.append(f"missing:{path}")
            continue
        try:
            if name.endswith(".json"):
                load_json(path)
            elif name.endswith(".csv"):
                rows = load_csv(path)
                if not rows:
                    errors.append(f"empty_csv:{path}")
            else:
                errors.append(f"unsupported_template_type:{path}")
        except Exception as exc:  # pragma: no cover - surfaced in CSV result
            errors.append(f"parse_error:{path}:{exc}")
    return not errors, ";".join(errors) if errors else "all required CSV/JSON templates parse"


def manifest_template_valid() -> tuple[bool, str]:
    manifest = load_json(DATA_DIR / "production_index_artifact_manifest_template.json")
    artifacts = manifest.get("artifacts")
    build_metadata = manifest.get("build_metadata")
    source_metadata = manifest.get("source_corpus_snapshot_metadata")
    if not isinstance(artifacts, dict):
        return False, "artifacts object missing"
    if not isinstance(build_metadata, dict):
        return False, "build_metadata object missing"
    if not isinstance(source_metadata, dict):
        return False, "source_corpus_snapshot_metadata object missing"
    missing_artifacts = MANIFEST_REQUIRED_ARTIFACTS - set(artifacts)
    missing_fields = MANIFEST_REQUIRED_BUILD_FIELDS - set(build_metadata)
    policy = manifest.get("formal_source_policy") or {}
    promotion = manifest.get("promotion_policy") or {}
    failures: list[str] = []
    if missing_artifacts:
        failures.append("missing_artifacts=" + ",".join(sorted(missing_artifacts)))
    if missing_fields:
        failures.append("missing_build_fields=" + ",".join(sorted(missing_fields)))
    if policy.get("csv_crop_markdown_paths_debug_only") is not True:
        failures.append("csv_crop_markdown_paths_debug_only_not_true")
    if promotion.get("preview_only_units_blocked") is not True:
        failures.append("preview_only_units_blocked_not_true")
    if promotion.get("production_ready_true_requires_independent_quality_gate") is not True:
        failures.append("independent_quality_gate_not_required")
    return not failures, ";".join(failures) if failures else "manifest template valid"


def promotion_gate_matrix_covers_required_blockers() -> tuple[bool, str]:
    rows = load_csv(DATA_DIR / "promotion_gate_matrix.csv")
    gate_ids = {row["gate_id"] for row in rows}
    missing = REQUIRED_PROMOTION_GATES - gate_ids
    required_not_true = [
        row["gate_id"]
        for row in rows
        if row["gate_id"] in REQUIRED_PROMOTION_GATES
        and row.get("required_for_promotion") != "true"
    ]
    return (
        not missing and not required_not_true,
        (
            "missing="
            + ",".join(sorted(missing))
            + ";required_not_true="
            + ",".join(sorted(required_not_true))
            if missing or required_not_true
            else "required promotion gates covered"
        ),
    )


def rollback_matrix_covers_hard_disable() -> tuple[bool, str]:
    rows = load_csv(DATA_DIR / "rollback_scenario_matrix.csv")
    scenario_ids = {row["scenario_id"] for row in rows}
    missing = REQUIRED_ROLLBACK_SCENARIOS - scenario_ids
    hard_disable = next(
        (row for row in rows if row["scenario_id"] == "hard_disable_to_normal_only"), None
    )
    failures: list[str] = []
    if missing:
        failures.append("missing=" + ",".join(sorted(missing)))
    if hard_disable is None:
        failures.append("hard_disable_to_normal_only_missing")
    else:
        expected_false_fields = [
            "table_branch_executed_after_rollback",
            "table_support_selected_after_rollback",
            "table_citation_emitted_after_rollback",
            "answer_visible_table_evidence_after_rollback",
        ]
        for field in expected_false_fields:
            if hard_disable.get(field) != "false":
                failures.append(f"{field}_not_false")
        if hard_disable.get("active_build_pointer_state") != "disabled":
            failures.append("active_build_pointer_state_not_disabled")
        if hard_disable.get("normal_only_restored") != "true":
            failures.append("normal_only_not_restored")
    return not failures, ";".join(failures) if failures else "hard disable rollback covered"


def rollout_stage_matrix_blocks_production_by_default() -> tuple[bool, str]:
    rows = load_csv(DATA_DIR / "rollout_stage_matrix.csv")
    stages = {row["stage"] for row in rows}
    missing = REQUIRED_ROLLOUT_STAGES - stages
    production = next((row for row in rows if row["stage"] == "production"), None)
    canary_no_answer = next((row for row in rows if row["stage"] == "canary_no_answer"), None)
    canary_answer_gated = next((row for row in rows if row["stage"] == "canary_answer_gated"), None)
    failures: list[str] = []
    if missing:
        failures.append("missing=" + ",".join(sorted(missing)))
    for stage_name, row in [
        ("production", production),
        ("canary_no_answer", canary_no_answer),
        ("canary_answer_gated", canary_answer_gated),
    ]:
        if row is None:
            failures.append(f"{stage_name}_missing")
            continue
        if row.get("phase7r_execution_allowed") != "false":
            failures.append(f"{stage_name}_phase7r_execution_not_false")
        if row.get("default_state") != "blocked":
            failures.append(f"{stage_name}_default_state_not_blocked")
    return not failures, ";".join(failures) if failures else "production/canary blocked by default"


def citation_readiness_blocks_preview_and_formal_misuse() -> tuple[bool, str]:
    text = (REPORT_DIR / "citation_readiness_coupling.md").read_text(encoding="utf-8").lower()
    required_snippets = [
        "no typed schema, no formal table citation",
        "no canonical source, no formal table citation",
        "preview_only blocks formal citation",
        "production_ready=false blocks formal citation",
        "mapper dry-run is not production binder",
        "citation_scope=value is forbidden",
        "binding warning-level blocks production-ready citation",
        "phase7q schema prototype pass does not equal production citation ready",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    return not missing, "missing=" + ";".join(missing) if missing else "citation readiness blockers present"


def citation_readiness_blocks_csv_crop_formal_source() -> tuple[bool, str]:
    text = (REPORT_DIR / "citation_readiness_coupling.md").read_text(encoding="utf-8").lower()
    snippets = [
        "csv/crop stay debug-only",
        "must not enter canonical_source or public citation.source_file",
    ]
    missing = [snippet for snippet in snippets if snippet not in text]
    return not missing, "missing=" + ";".join(missing) if missing else "CSV/crop formal source blocked"


def current_preview_units_fail_promotion_gate() -> tuple[bool, str]:
    rows = load_csv(DATA_DIR / "promotion_gate_matrix.csv")
    failed_blocking = {
        row["gate_id"]
        for row in rows
        if row.get("current_preview_status") == "fail"
        and row.get("blocks_current_preview_units") == "true"
    }
    missing = PREVIEW_BLOCKING_GATES - failed_blocking
    return (
        not missing,
        "missing_preview_blockers=" + ",".join(sorted(missing))
        if missing
        else "current preview units fail required promotion blockers",
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


def run_validation() -> list[dict[str, str]]:
    checks = [
        ("required_reports_exist", required_reports_exist),
        ("required_csv_json_templates_parse", required_structured_files_parse),
        ("manifest_template_json_valid", manifest_template_valid),
        ("promotion_gate_matrix_covers_required_blockers", promotion_gate_matrix_covers_required_blockers),
        ("rollback_matrix_covers_hard_disable", rollback_matrix_covers_hard_disable),
        ("rollout_stage_matrix_blocks_production_by_default", rollout_stage_matrix_blocks_production_by_default),
        ("citation_readiness_blocks_preview_formal_misuse", citation_readiness_blocks_preview_and_formal_misuse),
        ("citation_readiness_blocks_csv_crop_formal_source", citation_readiness_blocks_csv_crop_formal_source),
        ("current_preview_units_fail_promotion_gate", current_preview_units_fail_promotion_gate),
        ("no_src_or_configs_modified", no_src_or_configs_modified),
    ]
    rows: list[dict[str, str]] = []
    for check_id, check_fn in checks:
        try:
            passed, details = check_fn()
        except Exception as exc:  # pragma: no cover - surfaced in CSV result
            rows.append(check_row(check_id, False, f"exception:{exc}"))
        else:
            rows.append(check_row(check_id, passed, details))
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    pass_count = sum(1 for row in rows if row["check_pass"] == "true")
    validation_status = "pass_with_warnings" if pass_count == len(rows) else "blocked"
    return {
        "validation_status": validation_status,
        "check_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
    }


def render_validation_report(rows: list[dict[str, str]], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase7R Validation Report",
        "",
        f"- validation_status: `{summary['validation_status']}`",
        f"- check_count: {summary['check_count']}",
        f"- pass_count: {summary['pass_count']}",
        f"- fail_count: {summary['fail_count']}",
        f"- output: `{VALIDATION_RESULTS_PATH.relative_to(ROOT)}`",
        "",
        "The validator checks proposal artifact self-consistency only. It does not build an index, access retrieval stores, run embedding, run reranker, call LLMs, or modify production code/configuration.",
        "",
        "| check_id | check_pass | status | details |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {check_id} | {check_pass} | {status} | {details} |".format(
                check_id=row["check_id"],
                check_pass=row["check_pass"],
                status=row["status"],
                details=row["details"].replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def write_validation_artifacts() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_validation()
    summary = summarize(rows)
    write_csv(
        VALIDATION_RESULTS_PATH,
        rows,
        ["check_id", "check_pass", "status", "details"],
    )
    write_text(VALIDATION_REPORT_PATH, render_validation_report(rows, summary))
    write_text(SUMMARY_PATH, render_summary(summary["validation_status"]))
    return summary


def main() -> None:
    summary = write_validation_artifacts()
    print(f"validation_status={summary['validation_status']}")
    print(f"pass_count={summary['pass_count']}/{summary['check_count']}")


if __name__ == "__main__":
    main()
