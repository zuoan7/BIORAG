from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7x_final_build_query_set import QUERY_SET_PATH, TABLE_QUERY_TYPES, load_jsonl
from scripts.evaluation.phase7x_final_mainchain_ab_acceptance import (
    AB_RESULTS_PATH,
    AB_SUMMARY_PATH,
)


PHASE_DIR = "v7_phase7_table_preview_final_acceptance"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"
ANSWER_RESULTS_PATH = RESULTS_DIR / "answer_acceptance_results.csv"
ANSWER_SUMMARY_PATH = RESULTS_DIR / "answer_acceptance_summary.json"
ANSWER_REPORT_PATH = REPORTS_DIR / "answer_acceptance_report.md"
REVIEW_CARDS_PATH = REPORTS_DIR / "final_acceptance_review_cards.md"

FIELDNAMES = [
    "query_id",
    "query_type",
    "status",
    "skipped_reason",
    "normal_answer_generated",
    "preview_answer_generated",
    "preview_answer_uses_table_evidence",
    "non_table_answer_preview_leak",
    "formal_table_citation_count",
    "csv_crop_in_formal_citation",
    "answer_contains_debug_path",
    "answer_improvement_label",
    "normal_answer_preview",
    "preview_answer_preview",
    "preview_support_table_index_unit_ids",
    "preview_support_table_doc_ids",
    "preview_support_table_ids",
    "preview_support_row_labels",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_answer_acceptance(
    *,
    query_set_path: Path = QUERY_SET_PATH,
    ab_results_path: Path = AB_RESULTS_PATH,
    ab_summary_path: Path = AB_SUMMARY_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    max_table_like: int = 12,
    max_non_table: int = 4,
) -> dict[str, Any]:
    queries = load_jsonl(query_set_path)
    ab_summary = json.loads(ab_summary_path.read_text(encoding="utf-8"))
    if ab_summary.get("real_backend_status") == "blocked":
        summary = _skipped_summary(ab_summary)
        write_csv(results_dir / "answer_acceptance_results.csv", [], FIELDNAMES)
        write_json(results_dir / "answer_acceptance_summary.json", summary)
        write_answer_report(summary, reports_dir / "answer_acceptance_report.md")
        write_review_cards([], reports_dir / "final_acceptance_review_cards.md")
        return summary

    ab_rows = load_csv(ab_results_path)
    by_query: dict[str, dict[str, dict[str, str]]] = {}
    for row in ab_rows:
        by_query.setdefault(row["query_id"], {})[row["mode"]] = row

    selected = [
        query for query in queries if query["query_type"] in TABLE_QUERY_TYPES
    ][:max_table_like]
    selected.extend(
        [query for query in queries if query["query_type"] == "non_table_control"][:max_non_table]
    )

    records: list[dict[str, Any]] = []
    for query in selected:
        pair = by_query.get(query["query_id"], {})
        normal = pair.get("normal_only", {})
        preview = pair.get("table_preview_default_on", {})
        if not normal or not preview:
            records.append(_missing_record(query))
            continue
        is_non_table = query["query_type"] == "non_table_control"
        formal_count = int(normal.get("formal_table_citation_count") or 0) + int(
            preview.get("formal_table_citation_count") or 0
        )
        csv_crop_leak = _truthy(normal.get("csv_crop_in_formal_citation")) or _truthy(
            preview.get("csv_crop_in_formal_citation")
        )
        answer_debug_path = _answer_contains_debug_path(normal) or _answer_contains_debug_path(preview)
        preview_uses_table = _truthy(preview.get("answer_uses_table_evidence"))
        non_table_leak = is_non_table and (
            preview_uses_table or _truthy(preview.get("support_contains_table_preview"))
        )
        status = "pass"
        skipped_reason = ""
        if normal.get("mode_status") != "pass" or preview.get("mode_status") != "pass":
            status = "failed"
            skipped_reason = "mainchain_mode_error"
        elif not _truthy(preview.get("answer_generated")):
            status = "failed"
            skipped_reason = "preview_answer_not_generated"
        records.append(
            {
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "status": status,
                "skipped_reason": skipped_reason,
                "normal_answer_generated": _truthy(normal.get("answer_generated")),
                "preview_answer_generated": _truthy(preview.get("answer_generated")),
                "preview_answer_uses_table_evidence": preview_uses_table,
                "non_table_answer_preview_leak": non_table_leak,
                "formal_table_citation_count": formal_count,
                "csv_crop_in_formal_citation": csv_crop_leak,
                "answer_contains_debug_path": answer_debug_path,
                "answer_improvement_label": preview.get("answer_improvement_label", "skipped"),
                "normal_answer_preview": normal.get("answer_text_preview", ""),
                "preview_answer_preview": preview.get("answer_text_preview", ""),
                "preview_support_table_index_unit_ids": preview.get(
                    "support_table_index_unit_ids", ""
                ),
                "preview_support_table_doc_ids": preview.get("support_table_doc_ids", ""),
                "preview_support_table_ids": preview.get("support_table_ids", ""),
                "preview_support_row_labels": preview.get("support_row_labels", ""),
            }
        )

    summary = _summarize(records=records, ab_summary=ab_summary)
    write_csv(results_dir / "answer_acceptance_results.csv", records, FIELDNAMES)
    write_json(results_dir / "answer_acceptance_summary.json", summary)
    write_answer_report(summary, reports_dir / "answer_acceptance_report.md")
    write_review_cards(records, reports_dir / "final_acceptance_review_cards.md")
    return summary


def _summarize(*, records: list[dict[str, Any]], ab_summary: dict[str, Any]) -> dict[str, Any]:
    table_records = [row for row in records if row["query_type"] in TABLE_QUERY_TYPES]
    non_table_records = [row for row in records if row["query_type"] == "non_table_control"]
    status_counts = Counter(row["status"] for row in records)
    answer_labels = Counter(row["answer_improvement_label"] for row in table_records)
    table_using_count = sum(
        1 for row in table_records if _truthy(row["preview_answer_uses_table_evidence"])
    )
    formal_count = sum(int(row["formal_table_citation_count"]) for row in records)
    csv_crop_leak_count = sum(1 for row in records if _truthy(row["csv_crop_in_formal_citation"]))
    debug_path_leak_count = sum(1 for row in records if _truthy(row["answer_contains_debug_path"]))
    non_table_leak_count = sum(
        1 for row in non_table_records if _truthy(row["non_table_answer_preview_leak"])
    )
    pass_conditions = {
        "answer_smoke_not_blocked": ab_summary.get("real_backend_status") != "blocked",
        "answer_crash_zero": status_counts.get("failed", 0) == 0,
        "table_like_majority_uses_preview_evidence": table_using_count > len(table_records) / 2,
        "preview_better_more_than_worse": answer_labels.get("preview_better", 0)
        > answer_labels.get("preview_worse", 0),
        "non_table_answer_preview_leak_zero": non_table_leak_count == 0,
        "formal_table_citation_count_zero": formal_count == 0,
        "csv_crop_formal_citation_leak_zero": csv_crop_leak_count == 0,
        "answer_debug_path_leak_zero": debug_path_leak_count == 0,
    }
    failed_conditions = [name for name, ok in pass_conditions.items() if not ok]
    return {
        "validation_status": "pass" if not failed_conditions else "fail",
        "pass": not failed_conditions,
        "errors": failed_conditions,
        "query_count": len(records),
        "table_like_query_count": len(table_records),
        "non_table_control_count": len(non_table_records),
        "executed_count": status_counts.get("pass", 0),
        "failed_count": status_counts.get("failed", 0),
        "skipped_provider_unavailable_count": status_counts.get(
            "skipped_provider_unavailable", 0
        ),
        "answers_using_table_evidence_count": table_using_count,
        "answer_improvement_counts": dict(answer_labels),
        "non_table_answer_preview_leak_count": non_table_leak_count,
        "formal_table_citation_count": formal_count,
        "csv_crop_formal_citation_leak_count": csv_crop_leak_count,
        "answer_debug_path_leak_count": debug_path_leak_count,
        "pass_conditions": pass_conditions,
        "records_path": str(ANSWER_RESULTS_PATH),
        "review_cards_path": str(REVIEW_CARDS_PATH),
    }


def _skipped_summary(ab_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_status": "skipped_provider_unavailable",
        "pass": False,
        "errors": ["mainchain_ab_blocked"],
        "query_count": 0,
        "table_like_query_count": 0,
        "non_table_control_count": 0,
        "executed_count": 0,
        "failed_count": 0,
        "skipped_provider_unavailable_count": 1,
        "answers_using_table_evidence_count": 0,
        "answer_improvement_counts": {},
        "non_table_answer_preview_leak_count": 0,
        "formal_table_citation_count": 0,
        "csv_crop_formal_citation_leak_count": 0,
        "answer_debug_path_leak_count": 0,
        "pass_conditions": {"answer_smoke_not_blocked": False},
        "mainchain_real_backend_error": ab_summary.get("real_backend_error", ""),
        "records_path": str(ANSWER_RESULTS_PATH),
        "review_cards_path": str(REVIEW_CARDS_PATH),
    }


def _missing_record(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "status": "failed",
        "skipped_reason": "missing_ab_pair",
        "normal_answer_generated": False,
        "preview_answer_generated": False,
        "preview_answer_uses_table_evidence": False,
        "non_table_answer_preview_leak": False,
        "formal_table_citation_count": 0,
        "csv_crop_in_formal_citation": False,
        "answer_contains_debug_path": False,
        "answer_improvement_label": "skipped",
        "normal_answer_preview": "",
        "preview_answer_preview": "",
        "preview_support_table_index_unit_ids": "",
        "preview_support_table_doc_ids": "",
        "preview_support_table_ids": "",
        "preview_support_row_labels": "",
    }


def write_answer_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7X Final Answer Acceptance",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- query_count: {summary['query_count']}",
        f"- table_like_query_count: {summary['table_like_query_count']}",
        f"- non_table_control_count: {summary['non_table_control_count']}",
        f"- executed_count: {summary['executed_count']}",
        f"- failed_count: {summary['failed_count']}",
        f"- answers_using_table_evidence_count: {summary['answers_using_table_evidence_count']}",
        f"- answer_improvement_counts: {summary['answer_improvement_counts']}",
        f"- non_table_answer_preview_leak_count: {summary['non_table_answer_preview_leak_count']}",
        f"- formal_table_citation_count: {summary['formal_table_citation_count']}",
        f"- csv_crop_formal_citation_leak_count: {summary['csv_crop_formal_citation_leak_count']}",
        f"- answer_debug_path_leak_count: {summary['answer_debug_path_leak_count']}",
        f"- review_cards_path: {summary['review_cards_path']}",
    ]
    if summary["errors"]:
        lines.extend(["", "## Errors", *[f"- {error}" for error in summary["errors"]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_cards(records: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Phase7X Final Acceptance Review Cards", ""]
    for idx, row in enumerate(records[:15], start=1):
        lines.extend(
            [
                f"## Card {idx}: {row['query_id']}",
                "",
                f"- query_type: {row['query_type']}",
                f"- system_label: {row['answer_improvement_label']}",
                f"- citation_guard_status: formal_table_citation_count={row['formal_table_citation_count']}; csv_crop_in_formal_citation={row['csv_crop_in_formal_citation']}",
                f"- table_evidence_unit_ids: {row['preview_support_table_index_unit_ids']}",
                f"- table_evidence_doc_ids: {row['preview_support_table_doc_ids']}",
                f"- table_evidence_table_ids: {row['preview_support_table_ids']}",
                f"- table_evidence_row_labels: {row['preview_support_row_labels']}",
                "",
                "### Normal-only Answer",
                "",
                row["normal_answer_preview"] or "(empty)",
                "",
                "### Table-preview Answer",
                "",
                row["preview_answer_preview"] or "(empty)",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _answer_contains_debug_path(row: dict[str, str]) -> bool:
    text = " ".join(
        [
            row.get("answer_text_preview", ""),
            row.get("citation_source_files", ""),
            row.get("support_text_preview", ""),
        ]
    ).lower()
    return any(suffix in text for suffix in (".csv", ".png", ".jpg", ".jpeg", ".md"))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7X final answer acceptance.")
    parser.add_argument("--query-set-path", type=_path_arg, default=QUERY_SET_PATH)
    parser.add_argument("--ab-results-path", type=_path_arg, default=AB_RESULTS_PATH)
    parser.add_argument("--ab-summary-path", type=_path_arg, default=AB_SUMMARY_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    parser.add_argument("--max-table-like", type=int, default=12)
    parser.add_argument("--max-non-table", type=int, default=4)
    args = parser.parse_args()
    summary = run_answer_acceptance(
        query_set_path=args.query_set_path,
        ab_results_path=args.ab_results_path,
        ab_summary_path=args.ab_summary_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        max_table_like=args.max_table_like,
        max_non_table=args.max_non_table,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
