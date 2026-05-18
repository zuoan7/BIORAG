from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.table_preview import TablePreviewCandidateProvider


PHASE_DIR = "v7_phase7_table_preview_final_acceptance"
UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
PHASE7V_FIXTURE_PATH = (
    ROOT / "data/experiments/v7_phase7_table_preview_type_aware_merge/ab_query_fixture.jsonl"
)
DATA_DIR = ROOT / f"data/experiments/{PHASE_DIR}"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"
QUERY_SET_PATH = DATA_DIR / "final_acceptance_query_set.jsonl"

CORE_TABLE_QUERY_TYPES = {"table_lookup", "row_lookup", "metric_lookup"}
TABLE_QUERY_TYPES = CORE_TABLE_QUERY_TYPES | {"source_or_reference_lookup"}
QUERY_TYPE_QUOTAS = {
    "table_lookup": 7,
    "row_lookup": 8,
    "metric_lookup": 8,
    "source_or_reference_lookup": 2,
}
NON_TABLE_CONTROLS = [
    (
        "Summarize the study motivation and biological system discussed in the paper.",
        "doc_0075",
    ),
    (
        "Explain the experimental context and main objective of the induction study.",
        "doc_0600",
    ),
    (
        "What organism context is described in the protein localization work?",
        "doc_0066",
    ),
    (
        "Summarize the microbiota study background and objective.",
        "doc_0261",
    ),
    (
        "What is the main research question behind the primer design work?",
        "doc_0066",
    ),
    (
        "Explain the pathway engineering context without reporting structured numeric details.",
        "doc_0074",
    ),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_final_query_set(
    *,
    source_fixture_path: Path = PHASE7V_FIXTURE_PATH,
    units_path: Path = UNITS_PATH,
    output_path: Path = QUERY_SET_PATH,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    units = load_jsonl(units_path)
    unit_by_id = {str(unit.get("table_index_unit_id", "")): unit for unit in units}
    source_rows = load_jsonl(source_fixture_path)
    provider = TablePreviewCandidateProvider(str(units_path))

    selected: list[dict[str, Any]] = []
    for query_type, quota in QUERY_TYPE_QUOTAS.items():
        candidates = [row for row in source_rows if row.get("query_type") == query_type]
        accepted = []
        for row in candidates:
            unit = unit_by_id.get(str(row.get("expected_table_index_unit_id", "")))
            if not unit:
                continue
            query_text = _natural_query_text(row, unit)
            if _expected_seen_at_20(query_text, row, provider):
                accepted.append((row, query_text))
            if len(accepted) >= quota:
                break
        if len(accepted) < quota:
            raise RuntimeError(
                f"Only {len(accepted)} preview-visible {query_type} queries for quota {quota}"
            )
        selected.extend(_query_row(row, query_text) for row, query_text in accepted)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        row = dict(row)
        row["query_id"] = f"phase7x_final_query_{idx:03d}"
        rows.append(row)

    for offset, (query_text, doc_id) in enumerate(NON_TABLE_CONTROLS, start=len(rows) + 1):
        rows.append(
            {
                "query_id": f"phase7x_final_query_{offset:03d}",
                "source_query_id": "",
                "query_text": query_text,
                "query_type": "non_table_control",
                "expected_doc_id": doc_id,
                "expected_table_id": "",
                "expected_table_index_unit_id": "",
                "expected_unit_type": "none",
                "expected_row_label": "",
                "query_notes": "phase7x final non-table control",
            }
        )

    write_jsonl(output_path, rows)
    summary = validate_query_set_payload(rows=rows, units=units, query_set_path=output_path)
    summary["source_fixture_path"] = str(source_fixture_path)
    summary["units_path"] = str(units_path)
    if summary_path is None:
        summary_path = RESULTS_DIR / "final_acceptance_query_set_summary.json"
    write_json(summary_path, summary)
    write_query_set_report(summary, REPORTS_DIR / "final_acceptance_query_set_report.md")
    return summary


def validate_query_set_payload(
    *,
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    query_set_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(row.get("query_type", "") for row in rows)
    unit_ids = {str(unit.get("table_index_unit_id", "")) for unit in units}
    table_like_count = sum(counts.get(query_type, 0) for query_type in TABLE_QUERY_TYPES)
    non_table_count = counts.get("non_table_control", 0)

    if not 20 <= table_like_count <= 30:
        errors.append(f"expected 20-30 table-like queries, got {table_like_count}")
    if not 5 <= non_table_count <= 8:
        errors.append(f"expected 5-8 non-table controls, got {non_table_count}")
    for query_type in QUERY_TYPE_QUOTAS:
        if counts.get(query_type, 0) <= 0:
            errors.append(f"missing query type {query_type}")

    for row in rows:
        query_id = str(row.get("query_id", ""))
        query_type = row.get("query_type")
        if query_type in TABLE_QUERY_TYPES:
            expected_unit_id = str(row.get("expected_table_index_unit_id", ""))
            if expected_unit_id not in unit_ids:
                errors.append(f"{query_id} expected unit not found in preview units")
            if row.get("expected_unit_type") not in {
                "table_unit",
                "row_unit",
                "cell_group_unit",
            }:
                errors.append(f"{query_id} invalid expected_unit_type")
            if not row.get("expected_table_id"):
                errors.append(f"{query_id} missing expected_table_id")
        elif query_type == "non_table_control":
            if row.get("expected_unit_type") != "none":
                errors.append(f"{query_id} non-table control must use expected_unit_type=none")
            if re.search(r"\b(table|row|column|metric|value|yield|titer|residual)\b", row["query_text"], re.I):
                errors.append(f"{query_id} non-table control contains table-like trigger")
        else:
            errors.append(f"{query_id} unknown query_type={query_type!r}")

    return {
        "pass": not errors,
        "errors": errors,
        "query_set_path": str(query_set_path),
        "preview_unit_count": len(units),
        "query_count": len(rows),
        "table_like_query_count": table_like_count,
        "non_table_control_count": non_table_count,
        "query_type_counts": dict(sorted(counts.items())),
    }


def ensure_query_set(query_set_path: Path = QUERY_SET_PATH) -> list[dict[str, Any]]:
    if not query_set_path.exists():
        build_final_query_set(output_path=query_set_path)
    return load_jsonl(query_set_path)


def write_query_set_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7X Final Query Set",
        "",
        f"- status: {'pass' if summary['pass'] else 'fail'}",
        f"- query_count: {summary['query_count']}",
        f"- table_like_query_count: {summary['table_like_query_count']}",
        f"- non_table_control_count: {summary['non_table_control_count']}",
        f"- query_type_counts: {summary['query_type_counts']}",
        f"- query_set_path: {summary['query_set_path']}",
    ]
    if summary["errors"]:
        lines.extend(["", "## Errors", *[f"- {error}" for error in summary["errors"]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _query_row(source_row: dict[str, Any], query_text: str) -> dict[str, Any]:
    return {
        "query_id": "",
        "source_query_id": source_row.get("query_id", ""),
        "query_text": query_text,
        "query_type": source_row["query_type"],
        "expected_doc_id": source_row.get("expected_doc_id", ""),
        "expected_table_id": source_row.get("expected_table_id", ""),
        "expected_table_index_unit_id": source_row.get("expected_table_index_unit_id", ""),
        "expected_unit_type": source_row.get("expected_unit_type", ""),
        "expected_row_label": source_row.get("expected_row_label", ""),
        "query_notes": "phase7x final acceptance query",
    }


def _natural_query_text(row: dict[str, Any], unit: dict[str, Any]) -> str:
    query_type = row.get("query_type", "")
    caption = _clean_caption(unit.get("caption"))
    table_id = str(unit.get("table_id") or row.get("expected_table_id") or "the table")
    row_label = str(row.get("expected_row_label") or unit.get("metadata", {}).get("row_label") or "")
    header = _primary_header(unit)

    if query_type == "table_lookup":
        return f"Which table reports {caption}?"
    if query_type == "row_lookup":
        return f"Which table row reports {row_label} for {header} in {table_id}?"
    if query_type == "metric_lookup":
        return f"Find metric evidence for {row_label} values in {table_id}, especially {header}."
    if query_type == "source_or_reference_lookup":
        subject = row_label or caption
        return f"What source table supports the row {subject} in {table_id}?"
    return str(row.get("query_text", ""))


def _expected_seen_at_20(
    query_text: str,
    row: dict[str, Any],
    provider: TablePreviewCandidateProvider,
) -> bool:
    expected_unit_id = str(row.get("expected_table_index_unit_id", ""))
    candidates = provider.search(query_text, top_k=20)
    return expected_unit_id in {
        str(candidate.chunk.metadata.get("table_index_unit_id", "")) for candidate in candidates
    }


def _primary_header(unit: dict[str, Any]) -> str:
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    for key in ("cell_group_values", "row_values"):
        values = metadata.get(key)
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, dict):
                    continue
                header = str(item.get("column_header") or "").strip()
                if header and header.lower() not in {"col_001", "col_000"}:
                    return header
    header_path = metadata.get("header_path")
    if isinstance(header_path, list):
        for path in header_path:
            if isinstance(path, list) and path:
                header = str(path[-1]).strip()
                if header and header.lower() not in {"col_001", "col_000"}:
                    return header
    return "the reported measurement"


def _clean_caption(value: Any) -> str:
    text = str(value or "")
    text = text.replace("[TABLE CAPTION]", "").strip()
    text = " ".join(text.split())
    return text.rstrip(".")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase7X final acceptance query set.")
    parser.add_argument("--source-fixture-path", type=_path_arg, default=PHASE7V_FIXTURE_PATH)
    parser.add_argument("--units-path", type=_path_arg, default=UNITS_PATH)
    parser.add_argument("--output-path", type=_path_arg, default=QUERY_SET_PATH)
    parser.add_argument(
        "--summary-path",
        type=_path_arg,
        default=RESULTS_DIR / "final_acceptance_query_set_summary.json",
    )
    args = parser.parse_args()
    summary = build_final_query_set(
        source_fixture_path=args.source_fixture_path,
        units_path=args.units_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
