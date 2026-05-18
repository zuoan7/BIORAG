#!/usr/bin/env python3
"""Phase7S canonical source resolution dry-run for current preview table units."""

from __future__ import annotations

import csv
import json
import sys
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
UNIT_CSV_PATH = (
    ROOT
    / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.csv"
)

REQUIRED_INPUTS = [
    ROOT / "reports/v7_phase7_table_index_production_proposal/phase7r_summary.md",
    ROOT / "reports/v7_phase7_table_index_production_proposal/promotion_gate_matrix.md",
    ROOT / "reports/v7_phase7_table_index_production_proposal/citation_readiness_coupling.md",
    ROOT / "reports/v7_phase7_table_index_production_proposal/production_index_artifact_manifest.md",
    ROOT / "reports/v7_phase7_table_index_production_proposal/promotion_rollback_design.md",
    ROOT / "data/experiments/v7_phase7_table_index_production_proposal/promotion_gate_matrix.csv",
    ROOT
    / "data/experiments/v7_phase7_table_index_production_proposal/production_index_artifact_manifest_template.json",
    ROOT
    / "data/experiments/v7_phase7_table_citation_schema_prototype/table_evidence_citation_schema.json",
    ROOT / "data/experiments/v7_phase7_table_citation_schema_prototype/citation_mapping_matrix.csv",
    ROOT / "reports/v7_phase7_table_citation_schema_prototype/phase7q_summary.md",
    UNIT_JSONL_PATH,
    UNIT_CSV_PATH,
    ROOT / "configs/baseline_registry.yaml",
]

MANIFEST_PATH = DATA_DIR / "canonical_source_manifest.draft.jsonl"
SUMMARY_CSV_PATH = DATA_DIR / "canonical_source_resolution_summary.csv"
GUARDRAIL_REPORT_PATH = REPORT_DIR / "phase7s_guardrail.md"
RESOLUTION_REPORT_PATH = REPORT_DIR / "canonical_source_resolution_report.md"
MISSING_INPUTS_REPORT_PATH = REPORT_DIR / "missing_inputs_report.md"

DEBUG_PATH_KEYS = (
    "source_csv_path",
    "source_pdf_crop_path",
    "source_markdown_path",
)


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def input_preflight() -> tuple[bool, list[str], dict[str, int]]:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_INPUTS if not path.exists()]
    counts = {
        "eligible_jsonl_units": 0,
        "eligible_csv_units": 0,
    }
    if UNIT_JSONL_PATH.exists():
        counts["eligible_jsonl_units"] = len(load_jsonl(UNIT_JSONL_PATH))
    if UNIT_CSV_PATH.exists():
        counts["eligible_csv_units"] = len(load_csv(UNIT_CSV_PATH))
    if counts["eligible_jsonl_units"] and counts["eligible_csv_units"]:
        if counts["eligible_jsonl_units"] != counts["eligible_csv_units"]:
            missing.append(
                "eligible_unit_count_mismatch:"
                f"jsonl={counts['eligible_jsonl_units']},csv={counts['eligible_csv_units']}"
            )
    return not missing, missing, counts


def render_missing_inputs_report(missing: list[str], counts: dict[str, int]) -> str:
    lines = [
        "# Phase7S Missing Inputs Report",
        "",
        "Input preflight failed. Phase7S dry-run stopped and did not fabricate missing inputs.",
        "",
        f"- eligible_jsonl_units: {counts.get('eligible_jsonl_units', 0)}",
        f"- eligible_csv_units: {counts.get('eligible_csv_units', 0)}",
        "",
        "## Missing Or Invalid Inputs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines)


def render_guardrail() -> str:
    return """# Phase7S Guardrail

Phase7S is a dry-run, not a production implementation.

Guardrails:

- Do not build a production table index.
- Do not upgrade preview units.
- Do not generate formal production citations.
- Do not generate answers.
- Do not modify `src/`.
- Do not modify `configs/`.
- Do not modify ingestion pipeline.
- Do not access Milvus.
- Do not read or query official BM25.
- Do not run embedding, reranker, Qwen, LLM, RAGAS, OCR, or VLM.
- Do not enter Route C implementation.

Route C remains backlog. Phase7S may only read local artifacts, build a draft canonical source manifest, run a production readiness gate dry-run, classify blockers, validate the dry-run outputs, and report go/no-go decisions."""


def clean_caption(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("[TABLE CAPTION]", "").strip()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def nested_get(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def debug_paths(unit: dict[str, Any]) -> dict[str, str]:
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), dict) else {}
    return {
        key: as_text(provenance.get(key))
        for key in DEBUG_PATH_KEYS
        if as_text(provenance.get(key))
    }


def formal_source_candidate(unit: dict[str, Any]) -> str:
    candidates = [
        unit.get("canonical_source_file"),
        unit.get("source_file"),
        nested_get(unit, ("metadata", "canonical_source_file")),
        nested_get(unit, ("metadata", "source_file")),
    ]
    for candidate in candidates:
        text = as_text(candidate)
        if text:
            return text
    return ""


def paper_title_candidate(unit: dict[str, Any]) -> str:
    candidates = [
        unit.get("paper_title"),
        nested_get(unit, ("metadata", "paper_title")),
        nested_get(unit, ("canonical_source", "paper_title")),
    ]
    for candidate in candidates:
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


def available_and_missing_fields(
    unit: dict[str, Any],
    formal_source: str,
    paper_title: str,
) -> tuple[list[str], list[str]]:
    fields = {
        "doc_id": as_text(unit.get("doc_id")),
        "table_id": as_text(unit.get("table_id")),
        "table_caption": clean_caption(unit.get("caption")),
        "page": as_text(nested_get(unit, ("metadata", "page"))),
        "paper_title": paper_title,
        "canonical_source_file": formal_source,
        "doi": as_text(unit.get("doi") or nested_get(unit, ("metadata", "doi"))),
        "pmid": as_text(unit.get("pmid") or nested_get(unit, ("metadata", "pmid"))),
    }
    available = [key for key, value in fields.items() if value]
    missing = [key for key, value in fields.items() if not value]
    return available, missing


def resolve_unit(unit: dict[str, Any]) -> dict[str, Any]:
    paths = debug_paths(unit)
    formal_source = formal_source_candidate(unit)
    paper_title = paper_title_candidate(unit)
    formal_is_debug = is_debug_artifact_path(formal_source, paths)
    available, missing = available_and_missing_fields(unit, formal_source, paper_title)

    doc_id = as_text(unit.get("doc_id"))
    table_id = as_text(unit.get("table_id"))
    caption = clean_caption(unit.get("caption"))
    page = as_text(nested_get(unit, ("metadata", "page")))
    has_scope = bool(doc_id and table_id and caption and page)

    if formal_source and paper_title and has_scope and not formal_is_debug:
        status = "resolved_from_existing_metadata"
        confidence = 0.85
        formal_source_allowed = True
        notes = "Existing metadata includes canonical paper source fields and table scope."
    elif formal_is_debug and not (paper_title and has_scope):
        status = "blocked_debug_path_only"
        confidence = 0.05
        formal_source_allowed = False
        notes = "Only debug artifact paths are available; they remain debug provenance only."
    elif has_scope:
        status = "partial_metadata_only"
        confidence = 0.45
        formal_source_allowed = False
        notes = "Existing metadata has doc/table/caption/page but lacks confirmed canonical paper source fields."
    elif doc_id or table_id or caption or page or paths:
        status = "unresolved_missing_canonical_source"
        confidence = 0.15
        formal_source_allowed = False
        notes = "Metadata is insufficient for canonical paper source resolution."
    else:
        status = "not_evaluable"
        confidence = 0.0
        formal_source_allowed = False
        notes = "Unit lacks enough local metadata to evaluate canonical source resolution."

    return {
        "table_index_unit_id": as_text(unit.get("table_index_unit_id")),
        "doc_id": doc_id,
        "table_id": table_id,
        "table_caption": caption,
        "page": page,
        "candidate_id": as_text(unit.get("candidate_id")),
        "seed_id": as_text(unit.get("seed_id")),
        "canonical_source_status": status,
        "canonical_source_confidence": confidence,
        "canonical_source_fields_available": available,
        "canonical_source_missing_fields": missing,
        "formal_source_allowed": formal_source_allowed,
        "debug_provenance_paths": paths,
        "notes": notes,
    }


def summary_rows(records: list[dict[str, Any]], preflight_counts: dict[str, int]) -> list[dict[str, Any]]:
    status_counts = Counter(record["canonical_source_status"] for record in records)
    rows: list[dict[str, Any]] = [
        {
            "metric": "input_preflight",
            "key": "pass",
            "count": 1,
            "notes": "All required Phase7S inputs were present.",
        },
        {
            "metric": "unit_count",
            "key": "eligible_jsonl_units",
            "count": preflight_counts.get("eligible_jsonl_units", len(records)),
            "notes": "Input units read from JSONL.",
        },
        {
            "metric": "unit_count",
            "key": "eligible_csv_units",
            "count": preflight_counts.get("eligible_csv_units", len(records)),
            "notes": "Input units read from CSV.",
        },
        {
            "metric": "unit_count",
            "key": "manifest_records",
            "count": len(records),
            "notes": "Records written to canonical_source_manifest.draft.jsonl.",
        },
    ]
    for status in (
        "resolved_from_existing_metadata",
        "partial_metadata_only",
        "unresolved_missing_canonical_source",
        "blocked_debug_path_only",
        "not_evaluable",
    ):
        rows.append(
            {
                "metric": "canonical_source_status",
                "key": status,
                "count": status_counts.get(status, 0),
                "notes": "Canonical source resolution dry-run status count.",
            }
        )
    rows.extend(
        [
            {
                "metric": "formal_source_allowed",
                "key": "true",
                "count": sum(1 for record in records if record["formal_source_allowed"] is True),
                "notes": "Expected to be zero unless canonical paper source is confirmed.",
            },
            {
                "metric": "formal_source_allowed",
                "key": "false",
                "count": sum(1 for record in records if record["formal_source_allowed"] is False),
                "notes": "CSV/crop/markdown paths remain debug-only.",
            },
        ]
    )
    return rows


def render_resolution_report(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    status_counts = {row["key"]: row["count"] for row in rows if row["metric"] == "canonical_source_status"}
    formal_allowed = next(
        (row["count"] for row in rows if row["metric"] == "formal_source_allowed" and row["key"] == "true"),
        0,
    )
    return f"""# Canonical Source Resolution Report

Phase7S resolved current eligible table units using only existing local metadata and artifacts. It did not query DOI/PMID services and did not read retrieval indexes.

- input_units: {len(records)}
- resolved_from_existing_metadata: {status_counts.get('resolved_from_existing_metadata', 0)}
- partial_metadata_only: {status_counts.get('partial_metadata_only', 0)}
- unresolved_missing_canonical_source: {status_counts.get('unresolved_missing_canonical_source', 0)}
- blocked_debug_path_only: {status_counts.get('blocked_debug_path_only', 0)}
- not_evaluable: {status_counts.get('not_evaluable', 0)}
- formal_source_allowed: {formal_allowed}

The current units expose `doc_id`, `table_id`, caption, page, seed, candidate, and debug provenance paths. They do not expose confirmed canonical paper source fields, so unresolved or partial records must not be promoted to formal production citation.

`source_csv_path`, `source_pdf_crop_path`, and `source_markdown_path` are written only inside `debug_provenance_paths` in `canonical_source_manifest.draft.jsonl`. They are not used as formal citation sources."""


def run() -> dict[str, Any]:
    ensure_dirs()
    preflight_ok, missing, counts = input_preflight()
    if not preflight_ok:
        write_text(MISSING_INPUTS_REPORT_PATH, render_missing_inputs_report(missing, counts))
        raise SystemExit(2)

    units = load_jsonl(UNIT_JSONL_PATH)
    records = [resolve_unit(unit) for unit in units]
    rows = summary_rows(records, counts)

    write_text(GUARDRAIL_REPORT_PATH, render_guardrail())
    write_jsonl(MANIFEST_PATH, records)
    write_csv(SUMMARY_CSV_PATH, rows, ["metric", "key", "count", "notes"])
    write_text(RESOLUTION_REPORT_PATH, render_resolution_report(records, rows))
    return {
        "unit_count": len(records),
        "status_counts": Counter(record["canonical_source_status"] for record in records),
        "formal_source_allowed_count": sum(
            1 for record in records if record["formal_source_allowed"] is True
        ),
    }


def main() -> None:
    result = run()
    print(f"canonical_source_manifest_records={result['unit_count']}")
    for status, count in sorted(result["status_counts"].items()):
        print(f"{status}={count}")


if __name__ == "__main__":
    main()
