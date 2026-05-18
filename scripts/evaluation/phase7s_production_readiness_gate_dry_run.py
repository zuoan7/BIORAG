#!/usr/bin/env python3
"""Phase7S production readiness gate dry-run for preview table units."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports/v7_phase7_table_production_readiness_dry_run"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_production_readiness_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_production_readiness_dry_run"

UNIT_JSONL_PATH = (
    ROOT
    / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
PHASE7R_GATE_MATRIX_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_production_proposal/promotion_gate_matrix.csv"
)
MANIFEST_PATH = DATA_DIR / "canonical_source_manifest.draft.jsonl"
GATE_DRY_RUN_PATH = RESULTS_DIR / "production_readiness_gate_dry_run.csv"
BLOCKER_SUMMARY_PATH = RESULTS_DIR / "production_readiness_blocker_summary.csv"
GATE_REPORT_PATH = REPORT_DIR / "production_readiness_gate_dry_run_report.md"
SUMMARY_REPORT_PATH = REPORT_DIR / "phase7s_summary.md"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def text_bool(value: bool) -> str:
    return "true" if value else "false"


def nested_get(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def manifest_by_unit_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {as_text(record.get("table_index_unit_id")): record for record in records}


def clean_caption(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("[TABLE CAPTION]", "").strip()


def provenance_paths(unit: dict[str, Any]) -> dict[str, str]:
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), dict) else {}
    return {
        key: as_text(provenance.get(key))
        for key in ("source_csv_path", "source_pdf_crop_path", "source_markdown_path")
        if as_text(provenance.get(key))
    }


def formal_source_candidate(unit: dict[str, Any]) -> str:
    for candidate in (
        unit.get("canonical_source_file"),
        unit.get("source_file"),
        nested_get(unit, ("metadata", "canonical_source_file")),
        nested_get(unit, ("metadata", "source_file")),
    ):
        text = as_text(candidate)
        if text:
            return text
    return ""


def is_debug_artifact_path(path: str, paths: dict[str, str]) -> bool:
    if not path:
        return False
    lowered = path.lower()
    if path in set(paths.values()):
        return True
    return lowered.endswith((".csv", ".png", ".jpg", ".jpeg", ".md"))


def unit_binding_level(unit: dict[str, Any]) -> str:
    guardrail = unit.get("guardrail") if isinstance(unit.get("guardrail"), dict) else {}
    binding_limitation = as_text(guardrail.get("binding_review_limitation"))
    reference_ok = as_text(guardrail.get("reference_ok"))
    unit_or_note_ok = as_text(guardrail.get("unit_or_note_ok"))
    if "verified" in {binding_limitation, reference_ok, unit_or_note_ok}:
        return "verified"
    if "reviewed" in {binding_limitation, reference_ok, unit_or_note_ok}:
        return "reviewed"
    return "warning"


def derived_citation_scope(unit_type: str) -> str:
    return {
        "table_unit": "table",
        "row_unit": "row",
        "cell_group_unit": "cell_group",
    }.get(unit_type, "")


def scope_shape_valid(unit: dict[str, Any]) -> bool:
    unit_type = as_text(unit.get("unit_type"))
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    if unit_type not in {"table_unit", "row_unit", "cell_group_unit"}:
        return False
    if unit_type == "table_unit":
        return True
    if not as_text(metadata.get("row_label")):
        return False
    header_path = metadata.get("header_path")
    return isinstance(header_path, list) and bool(header_path)


def source_span_granularity(unit: dict[str, Any]) -> str:
    return as_text(nested_get(unit, ("provenance", "source_span_granularity")))


def evaluate_unit(
    unit: dict[str, Any],
    canonical_record: dict[str, Any] | None,
) -> dict[str, Any]:
    unit_id = as_text(unit.get("table_index_unit_id"))
    unit_type = as_text(unit.get("unit_type"))
    guardrail = unit.get("guardrail") if isinstance(unit.get("guardrail"), dict) else {}
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), dict) else {}
    paths = provenance_paths(unit)
    formal_source = formal_source_candidate(unit)

    failed_gates: list[str] = []
    reasons: list[str] = []

    canonical_status = (
        as_text(canonical_record.get("canonical_source_status")) if canonical_record else "not_evaluable"
    )
    if canonical_status != "resolved_from_existing_metadata":
        failed_gates.append("canonical_paper_source_resolved")
        reasons.append(f"canonical_source_status={canonical_status}")

    if formal_source and is_debug_artifact_path(formal_source, paths):
        failed_gates.append("csv_crop_path_not_formal_source")
        reasons.append("formal_source_candidate_is_debug_artifact_path")

    table_scope_present = bool(
        as_text(unit.get("table_id")) and clean_caption(unit.get("caption")) and as_text(metadata.get("page"))
    )
    if not table_scope_present:
        failed_gates.append("table_id_caption_page_valid")
        reasons.append("missing_table_id_caption_or_page")

    if not scope_shape_valid(unit):
        failed_gates.append("table_row_cell_group_scope_valid")
        reasons.append("invalid_table_row_or_cell_group_scope_shape")

    citation_scope = derived_citation_scope(unit_type)
    if citation_scope == "value" or citation_scope not in {"table", "row", "cell_group"}:
        failed_gates.append("citation_scope_not_value")
        reasons.append(f"invalid_or_value_citation_scope={citation_scope or 'missing'}")

    value_bboxes_available = parse_bool(provenance.get("value_bboxes_available"))
    value_level_claim_allowed = False
    if value_level_claim_allowed and not value_bboxes_available:
        failed_gates.append("value_level_citation_disabled_unless_value_bboxes_verified")
        reasons.append("value_level_claim_allowed_without_value_bboxes")
    elif not value_bboxes_available:
        reasons.append("value_level_citation_disabled_due_to_value_bboxes_available=false")

    binding_level = unit_binding_level(unit)
    if binding_level not in {"reviewed", "verified"}:
        failed_gates.append("binding_review_at_least_reviewed")
        reasons.append(f"binding_review_level={binding_level}")

    span = source_span_granularity(unit)
    if not span:
        failed_gates.append("source_span_granularity_explicit")
        reasons.append("source_span_granularity_missing")

    production_ready = parse_bool(guardrail.get("production_ready"))
    if production_ready is not True:
        failed_gates.append("production_ready_true_independent_gate")
        reasons.append("production_ready=false_or_missing")

    index_unit_status = as_text(guardrail.get("index_unit_status"))
    if index_unit_status == "preview_only" or not index_unit_status:
        failed_gates.append("index_unit_status_not_preview_only")
        reasons.append(f"index_unit_status={index_unit_status or 'missing'}")

    failed_gates.append("typed_citation_schema_available")
    reasons.append("typed_schema_prototype_exists_but_not_integrated_into_production_binder")

    failed_gates.append("metadata_contract_valid")
    reasons.append("production_build_version_approval_confidence_metadata_missing")

    failed_gates.append("rollback_metadata_present")
    reasons.append("rollback_manifest_id_missing")

    failed_gates.append("checksum_build_manifest_valid")
    reasons.append("checksum_manifest_id_missing")

    failed_gates = list(dict.fromkeys(failed_gates))
    gate_status = "blocked" if failed_gates else "pass"
    if not unit_id or canonical_record is None:
        gate_status = "not_evaluable" if not failed_gates else "blocked"

    can_be_fixed_by_metadata = any(
        gate
        in {
            "canonical_paper_source_resolved",
            "table_id_caption_page_valid",
            "table_row_cell_group_scope_valid",
            "source_span_granularity_explicit",
        }
        for gate in failed_gates
    )
    requires_binder_integration = "typed_citation_schema_available" in failed_gates
    requires_production_index_build = any(
        gate
        in {
            "production_ready_true_independent_gate",
            "index_unit_status_not_preview_only",
            "metadata_contract_valid",
            "rollback_metadata_present",
            "checksum_build_manifest_valid",
        }
        for gate in failed_gates
    )
    requires_future_canary_or_answer_smoke = True

    if "canonical_paper_source_resolved" in failed_gates:
        blocker_category = "data_blocker"
        recommended_next_action = "resolve_canonical_source_manifest_before_promotion"
    elif requires_binder_integration:
        blocker_category = "schema_or_binder_blocker"
        recommended_next_action = "integrate_typed_table_citation_schema_before_formal_citation"
    elif requires_production_index_build:
        blocker_category = "expected_preview_blocker"
        recommended_next_action = "run_independent_quality_gate_and_future_production_index_build"
    elif gate_status == "not_evaluable":
        blocker_category = "not_evaluable"
        recommended_next_action = "repair_missing_unit_identity_or_manifest_record"
    else:
        blocker_category = "operational_blocker"
        recommended_next_action = "run future canary and answer smoke only after earlier gates pass"

    return {
        "table_index_unit_id": unit_id,
        "doc_id": as_text(unit.get("doc_id")),
        "table_id": as_text(unit.get("table_id")),
        "unit_type": unit_type,
        "gate_status": gate_status,
        "failed_gates": ";".join(failed_gates),
        "blocker_category": blocker_category,
        "blocker_reason": ";".join(reasons),
        "can_be_fixed_by_metadata": text_bool(can_be_fixed_by_metadata),
        "requires_binder_integration": text_bool(requires_binder_integration),
        "requires_production_index_build": text_bool(requires_production_index_build),
        "requires_future_canary_or_answer_smoke": text_bool(requires_future_canary_or_answer_smoke),
        "recommended_next_action": recommended_next_action,
    }


def top_failed_gates(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for gate in str(row.get("failed_gates", "")).split(";"):
            if gate:
                counts[gate] += 1
    return counts


def blocker_summary_rows(
    units: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_counts = Counter(record["canonical_source_status"] for record in canonical_records)
    gate_counts = Counter(row["gate_status"] for row in gate_rows)
    category_counts = Counter(row["blocker_category"] for row in gate_rows)
    failed_gate_counts = top_failed_gates(gate_rows)
    rows: list[dict[str, Any]] = [
        {
            "metric": "unit_count",
            "key": "total_units",
            "count": len(units),
            "notes": "Total eligible units processed by Phase7S dry-run.",
        },
        {
            "metric": "canonical_source",
            "key": "resolved_from_existing_metadata",
            "count": canonical_counts.get("resolved_from_existing_metadata", 0),
            "notes": "Units with confirmed canonical source fields from existing metadata.",
        },
        {
            "metric": "canonical_source",
            "key": "partial_metadata_only",
            "count": canonical_counts.get("partial_metadata_only", 0),
            "notes": "Units with doc/table/caption/page but missing canonical paper source.",
        },
        {
            "metric": "canonical_source",
            "key": "unresolved_missing_canonical_source",
            "count": canonical_counts.get("unresolved_missing_canonical_source", 0),
            "notes": "Units missing enough canonical source metadata.",
        },
        {
            "metric": "gate_status",
            "key": "pass",
            "count": gate_counts.get("pass", 0),
            "notes": "Phase7S does not require any unit to pass production gate.",
        },
        {
            "metric": "gate_status",
            "key": "blocked",
            "count": gate_counts.get("blocked", 0),
            "notes": "Expected for current preview units.",
        },
        {
            "metric": "gate_status",
            "key": "not_evaluable",
            "count": gate_counts.get("not_evaluable", 0),
            "notes": "Units not evaluable by the dry-run.",
        },
    ]
    for category in (
        "data_blocker",
        "schema_or_binder_blocker",
        "operational_blocker",
        "expected_preview_blocker",
        "not_evaluable",
    ):
        rows.append(
            {
                "metric": "blocker_category",
                "key": category,
                "count": category_counts.get(category, 0),
                "notes": "Primary blocker category distribution.",
            }
        )
    for gate, count in failed_gate_counts.most_common():
        rows.append(
            {
                "metric": "top_failed_gate",
                "key": gate,
                "count": count,
                "notes": "Failed gate count across units.",
            }
        )
    rows.extend(
        [
            {
                "metric": "fix_path",
                "key": "can_be_fixed_by_metadata_or_source_resolution",
                "count": sum(1 for row in gate_rows if row["can_be_fixed_by_metadata"] == "true"),
                "notes": "Units requiring canonical source or local metadata repair.",
            },
            {
                "metric": "fix_path",
                "key": "requires_binder_integration",
                "count": sum(1 for row in gate_rows if row["requires_binder_integration"] == "true"),
                "notes": "Units blocked by production binder/schema integration gap.",
            },
            {
                "metric": "fix_path",
                "key": "requires_production_index_build",
                "count": sum(1 for row in gate_rows if row["requires_production_index_build"] == "true"),
                "notes": "Units requiring future production build/promotion metadata.",
            },
            {
                "metric": "fix_path",
                "key": "requires_future_canary_or_answer_smoke",
                "count": sum(
                    1
                    for row in gate_rows
                    if row["requires_future_canary_or_answer_smoke"] == "true"
                ),
                "notes": "Production still requires future canary/answer smoke after earlier blockers clear.",
            },
        ]
    )
    return rows


def metric_count(rows: list[dict[str, Any]], metric: str, key: str) -> int:
    for row in rows:
        if row["metric"] == metric and row["key"] == key:
            return int(row["count"])
    return 0


def render_gate_report(summary_rows: list[dict[str, Any]]) -> str:
    total = metric_count(summary_rows, "unit_count", "total_units")
    blocked = metric_count(summary_rows, "gate_status", "blocked")
    passed = metric_count(summary_rows, "gate_status", "pass")
    not_evaluable = metric_count(summary_rows, "gate_status", "not_evaluable")
    top_gates = [
        row for row in summary_rows if row["metric"] == "top_failed_gate"
    ][:8]
    top_lines = "\n".join(f"- {row['key']}: {row['count']}" for row in top_gates)
    return f"""# Production Readiness Gate Dry-Run Report

Phase7S converted the Phase7R promotion gate matrix into a unit-level dry-run checker for the current preview-eligible table units.

- total_units: {total}
- gate_pass: {passed}
- gate_blocked: {blocked}
- gate_not_evaluable: {not_evaluable}

Top failed gates:

{top_lines}

Current preview units are expected to be blocked. The dry-run succeeds when those blockers are explicit, classified, and explainable.

CSV/crop/markdown paths were not accepted as formal source. `preview_only`, `production_ready=false`, warning-level binding, missing canonical source, missing production manifest metadata, and missing production binder integration remain production blockers."""


def render_summary(summary_rows: list[dict[str, Any]], validation_status: str = "pending") -> str:
    top_gates = [row for row in summary_rows if row["metric"] == "top_failed_gate"][:8]
    top_gate_lines = "\n".join(f"- {row['key']}: {row['count']}" for row in top_gates)
    category_lines = "\n".join(
        f"- {row['key']}: {row['count']}"
        for row in summary_rows
        if row["metric"] == "blocker_category"
    )
    return f"""# Phase7S Summary

## 1. Generated Files

Reports:

- `reports/v7_phase7_table_production_readiness_dry_run/phase7s_guardrail.md`
- `reports/v7_phase7_table_production_readiness_dry_run/canonical_source_resolution_report.md`
- `reports/v7_phase7_table_production_readiness_dry_run/production_readiness_gate_dry_run_report.md`
- `reports/v7_phase7_table_production_readiness_dry_run/phase7s_summary.md`

Structured outputs:

- `data/experiments/v7_phase7_table_production_readiness_dry_run/canonical_source_manifest.draft.jsonl`
- `data/experiments/v7_phase7_table_production_readiness_dry_run/canonical_source_resolution_summary.csv`
- `results/v7_phase7_table_production_readiness_dry_run/production_readiness_gate_dry_run.csv`
- `results/v7_phase7_table_production_readiness_dry_run/production_readiness_blocker_summary.csv`
- `results/v7_phase7_table_production_readiness_dry_run/phase7s_validation_summary.csv`

Scripts/tests:

- `scripts/evaluation/phase7s_canonical_source_dry_run.py`
- `scripts/evaluation/phase7s_production_readiness_gate_dry_run.py`
- `scripts/evaluation/phase7s_validate_readiness_dry_run.py`
- `tests/test_phase7s_production_readiness_dry_run.py`

## 2. Guardrail Status

- Modified `src/` / `configs/`: no.
- Accessed Milvus / official BM25: no.
- Ran embedding / reranker / LLM: no.
- Built production table index: no.
- Generated answer or formal production citation: no.
- Entered Route C implementation: no.

## 3. Canonical Source Resolution Statistics

- total_units: {metric_count(summary_rows, 'unit_count', 'total_units')}
- resolved_from_existing_metadata: {metric_count(summary_rows, 'canonical_source', 'resolved_from_existing_metadata')}
- partial_metadata_only: {metric_count(summary_rows, 'canonical_source', 'partial_metadata_only')}
- unresolved_missing_canonical_source: {metric_count(summary_rows, 'canonical_source', 'unresolved_missing_canonical_source')}

## 4. Production Readiness Gate Statistics

- gate_pass: {metric_count(summary_rows, 'gate_status', 'pass')}
- gate_blocked: {metric_count(summary_rows, 'gate_status', 'blocked')}
- gate_not_evaluable: {metric_count(summary_rows, 'gate_status', 'not_evaluable')}

## 5. Blocker Category Distribution

{category_lines}

## 6. Top Failed Gates

{top_gate_lines}

## 7. Fix Path Counts

- Metadata/source resolution fixable: {metric_count(summary_rows, 'fix_path', 'can_be_fixed_by_metadata_or_source_resolution')}
- Requires binder integration: {metric_count(summary_rows, 'fix_path', 'requires_binder_integration')}
- Requires production index build: {metric_count(summary_rows, 'fix_path', 'requires_production_index_build')}
- Requires future canary / answer smoke: {metric_count(summary_rows, 'fix_path', 'requires_future_canary_or_answer_smoke')}

## 8. Validation And Decisions

- validation_status: `{validation_status}`
- Recommend Phase7T: yes, Feature-Flagged Implementation Scaffold / No Production Rollout.
- Recommend production: no.
- Recommend directly building production index: no.
- Recommend extractor rework: no.
- Recommend continued large manual annotation: no.
- Route C remains backlog: yes."""


def run(validation_status: str = "pending") -> dict[str, Any]:
    ensure_dirs()
    units = load_jsonl(UNIT_JSONL_PATH)
    canonical_records = load_jsonl(MANIFEST_PATH)
    canonical_by_id = manifest_by_unit_id(canonical_records)
    _phase7r_gate_rows = load_csv(PHASE7R_GATE_MATRIX_PATH)

    gate_rows = [
        evaluate_unit(unit, canonical_by_id.get(as_text(unit.get("table_index_unit_id"))))
        for unit in units
    ]
    summary = blocker_summary_rows(units, canonical_records, gate_rows)

    write_csv(
        GATE_DRY_RUN_PATH,
        gate_rows,
        [
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
        ],
    )
    write_csv(BLOCKER_SUMMARY_PATH, summary, ["metric", "key", "count", "notes"])
    write_text(GATE_REPORT_PATH, render_gate_report(summary))
    write_text(SUMMARY_REPORT_PATH, render_summary(summary, validation_status))

    return {
        "unit_count": len(units),
        "gate_status_counts": Counter(row["gate_status"] for row in gate_rows),
        "blocker_category_counts": Counter(row["blocker_category"] for row in gate_rows),
        "top_failed_gates": top_failed_gates(gate_rows),
    }


def main() -> None:
    result = run()
    print(f"production_readiness_gate_records={result['unit_count']}")
    for status, count in sorted(result["gate_status_counts"].items()):
        print(f"{status}={count}")


if __name__ == "__main__":
    main()
