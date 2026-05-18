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
    ensure_ab_fixture,
    normal_retrieved,
    preview_chunks,
    preview_config,
    write_csv,
    write_json,
)
from src.synbio_rag.application.table_preview import apply_table_preview


OUTPUT_CSV = RESULTS_DIR / "rollback_regression.csv"
OUTPUT_REPORT = REPORTS_DIR / "rollback_regression_report.md"


class ForbiddenPreviewProvider:
    def __init__(self) -> None:
        self.called = False

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.called = True
        raise AssertionError("preview provider must not run when TABLE_PREVIEW_ENABLED=false")


def run_rollback_regression(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_csv: Path = OUTPUT_CSV,
    output_report: Path = OUTPUT_REPORT,
) -> dict[str, Any]:
    queries = ensure_ab_fixture(fixture_path)
    config = preview_config(enabled=False, merge_enabled=False, strategy="type_aware_merge_v1")
    provider = ForbiddenPreviewProvider()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in queries:
        retrieved = normal_retrieved(query["query_id"])
        output, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=retrieved,
            config=config,
            provider=provider,  # type: ignore[arg-type]
        )
        record = {
            "query_id": query["query_id"],
            "query_type": query["query_type"],
            "enabled": debug.get("enabled", None),
            "mode": debug.get("mode", ""),
            "reason": debug.get("reason", ""),
            "table_branch_executed": debug.get("table_branch_executed", None),
            "table_candidates_in_rerank_input": debug.get("table_candidates_in_rerank_input", None),
            "input_chunk_ids": ";".join(chunk.chunk_id for chunk in retrieved),
            "output_chunk_ids": ";".join(chunk.chunk_id for chunk in output),
            "preview_output_count": len(preview_chunks(output)),
        }
        records.append(record)
        if record["enabled"] is not False:
            errors.append(f"{query['query_id']} did not report enabled=false")
        if record["table_branch_executed"] is not False:
            errors.append(f"{query['query_id']} executed table branch while disabled")
        if record["table_candidates_in_rerank_input"] is not False:
            errors.append(f"{query['query_id']} marked preview rerank input while disabled")
        if record["input_chunk_ids"] != record["output_chunk_ids"]:
            errors.append(f"{query['query_id']} changed normal-only retrieval output")
        if int(record["preview_output_count"]) != 0:
            errors.append(f"{query['query_id']} emitted preview chunks while disabled")
    if provider.called:
        errors.append("preview provider was called while TABLE_PREVIEW_ENABLED=false")

    summary = {
        "pass": not errors,
        "errors": errors,
        "query_count": len(records),
        "provider_called": provider.called,
        "table_loader_executed": provider.called,
        "table_candidates_in_rerank_input_count": sum(
            1 for row in records if row["table_candidates_in_rerank_input"] is True
        ),
        "preview_output_count": sum(int(row["preview_output_count"]) for row in records),
        "normal_only_restored": not errors,
        "records_path": str(output_csv),
        "report_path": str(output_report),
    }
    write_csv(
        output_csv,
        records,
        [
            "query_id",
            "query_type",
            "enabled",
            "mode",
            "reason",
            "table_branch_executed",
            "table_candidates_in_rerank_input",
            "input_chunk_ids",
            "output_chunk_ids",
            "preview_output_count",
        ],
    )
    write_json(output_csv.with_suffix(".summary.json"), summary)
    _write_report(summary, output_report)
    return summary


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7V-fast Rollback Regression",
        "",
        f"- pass: {summary['pass']}",
        f"- query_count: {summary['query_count']}",
        f"- table_loader_executed: {summary['table_loader_executed']}",
        f"- table_candidates_in_rerank_input_count: {summary['table_candidates_in_rerank_input_count']}",
        f"- preview_output_count: {summary['preview_output_count']}",
        f"- normal_only_restored: {summary['normal_only_restored']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7V-fast rollback regression.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--output-csv", type=_path_arg, default=OUTPUT_CSV)
    parser.add_argument("--output-report", type=_path_arg, default=OUTPUT_REPORT)
    args = parser.parse_args()
    summary = run_rollback_regression(
        fixture_path=args.fixture_path,
        output_csv=args.output_csv,
        output_report=args.output_report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
