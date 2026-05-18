from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7v_fast_ab_smoke import (
    FIXTURE_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    TABLE_QUERY_TYPES,
    CitationBinder,
    SupportItem,
    ensure_ab_fixture,
    evidence_candidate_from_chunk,
    normal_retrieved,
    preview_chunks,
    preview_config,
    write_csv,
    write_json,
)
from src.synbio_rag.application.table_preview import apply_table_preview


OUTPUT_CSV = RESULTS_DIR / "citation_guard_regression.csv"
OUTPUT_REPORT = REPORTS_DIR / "citation_guard_regression_report.md"


def run_citation_guard_regression(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_csv: Path = OUTPUT_CSV,
    output_report: Path = OUTPUT_REPORT,
    max_queries: int = 12,
) -> dict[str, Any]:
    queries = [row for row in ensure_ab_fixture(fixture_path) if row["query_type"] in TABLE_QUERY_TYPES]
    config = preview_config(enabled=True, merge_enabled=True, strategy="type_aware_merge_v1")
    binder = CitationBinder()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in queries[:max_queries]:
        output, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=config,
        )
        for idx, chunk in enumerate(preview_chunks(output)[:2], start=1):
            evidence_id = f"E{idx}"
            candidate = evidence_candidate_from_chunk(evidence_id, chunk)
            support = [SupportItem(evidence_id, candidate, 0.9, ["selected_preview_table"])]
            candidates = binder.build_citation_candidates(support)
            answer, citations, citation_debug = binder.bind(
                f"Preview-only table evidence [{evidence_id}].",
                support,
                citation_candidates=candidates,
            )
            drop_reasons = citation_debug.get("drop_reasons_by_evidence_id", {})
            source_csv_path = candidate.metadata.get("source_csv_path", "")
            source_pdf_crop_path = candidate.metadata.get("source_pdf_crop_path", "")
            record = {
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "merge_mode": debug.get("mode", ""),
                "chunk_id": chunk.chunk_id,
                "table_index_unit_id": chunk.metadata.get("table_index_unit_id", ""),
                "source_file": candidate.source_file,
                "source_csv_path": source_csv_path,
                "source_pdf_crop_path": source_pdf_crop_path,
                "citation_count": len(citations),
                "answer_contains_formal_marker": "[1]" in answer,
                "blocked_evidence_ids": ";".join(citation_debug.get("blocked_evidence_ids", [])),
                "drop_reason": drop_reasons.get(evidence_id, ""),
                "candidate_eligible": candidates[0].citation_eligible,
                "csv_path_in_source_file": bool(source_csv_path and candidate.source_file == source_csv_path),
                "crop_path_in_source_file": bool(
                    source_pdf_crop_path and candidate.source_file == source_pdf_crop_path
                ),
            }
            records.append(record)
            if record["citation_count"] != 0:
                errors.append(f"{chunk.chunk_id} produced formal citation")
            if record["answer_contains_formal_marker"]:
                errors.append(f"{chunk.chunk_id} kept formal citation marker")
            if record["drop_reason"] != "table_preview_formal_citation_blocked":
                errors.append(f"{chunk.chunk_id} drop reason was {record['drop_reason']!r}")
            if record["candidate_eligible"] is not False:
                errors.append(f"{chunk.chunk_id} citation candidate remained eligible")
            if record["csv_path_in_source_file"] or record["crop_path_in_source_file"]:
                errors.append(f"{chunk.chunk_id} leaked debug path into citation source_file")

    summary = {
        "pass": not errors,
        "errors": errors,
        "record_count": len(records),
        "formal_citation_count": sum(int(row["citation_count"]) for row in records),
        "drop_reason": "table_preview_formal_citation_blocked",
        "records_path": str(output_csv),
        "report_path": str(output_report),
    }
    write_csv(
        output_csv,
        records,
        [
            "query_id",
            "query_type",
            "merge_mode",
            "chunk_id",
            "table_index_unit_id",
            "source_file",
            "source_csv_path",
            "source_pdf_crop_path",
            "citation_count",
            "answer_contains_formal_marker",
            "blocked_evidence_ids",
            "drop_reason",
            "candidate_eligible",
            "csv_path_in_source_file",
            "crop_path_in_source_file",
        ],
    )
    write_json(output_csv.with_suffix(".summary.json"), summary)
    _write_report(summary, output_report)
    return summary


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7V-fast Citation Guard Regression",
        "",
        f"- pass: {summary['pass']}",
        f"- formal_citation_count: {summary['formal_citation_count']}",
        f"- required_drop_reason: {summary['drop_reason']}",
        f"- record_count: {summary['record_count']}",
        "",
        "Preview table candidates remain blocked from formal citations; CSV and crop paths stay debug provenance only.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7V-fast citation guard regression.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--output-csv", type=_path_arg, default=OUTPUT_CSV)
    parser.add_argument("--output-report", type=_path_arg, default=OUTPUT_REPORT)
    parser.add_argument("--max-queries", type=int, default=12)
    args = parser.parse_args()
    summary = run_citation_guard_regression(
        fixture_path=args.fixture_path,
        output_csv=args.output_csv,
        output_report=args.output_report,
        max_queries=args.max_queries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
