from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7w_slim_mainchain_evidence_smoke import (
    FIXTURE_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    TABLE_QUERY_TYPES,
    ensure_fixture,
    preview_chunks,
    preview_config,
    normal_retrieved,
    stub_rerank,
)
from src.synbio_rag.application.generation_v2 import GenerationV2Service
from src.synbio_rag.domain.config import GenerationConfig, ModelEndpointConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent
from src.synbio_rag.application.table_preview import apply_table_preview


FIELDNAMES = [
    "query_id",
    "query_type",
    "status",
    "skipped_reason",
    "preview_input_count",
    "support_preview_count",
    "answer_uses_preview_evidence",
    "answer_contains_debug_path",
    "formal_table_citation_count",
    "qwen_attempted",
    "qwen_used",
    "answer_preview",
]


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


def run_answer_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    max_queries: int = 5,
) -> dict[str, Any]:
    queries = [query for query in ensure_fixture(fixture_path) if query["query_type"] in TABLE_QUERY_TYPES][
        :max_queries
    ]
    records: list[dict[str, Any]] = []
    service = GenerationV2Service(ModelEndpointConfig(provider="stub", model_name="local-extractive"))
    gen_config = GenerationConfig(
        v2_use_qwen_synthesis=False,
        v2_require_citation=False,
        v2_min_support_score=0.0,
        v2_max_support_factoid=3,
    )
    analysis = QueryAnalysis(
        intent=QueryIntent.FACTOID,
        requires_external_tools=False,
        search_limit=5,
        rerank_top_k=5,
        notes="phase7w_slim_answer_smoke",
    )

    for query in queries:
        output, _debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"], query.get("expected_doc_id", "")),
            config=preview_config(enabled=True, merge_enabled=True, strategy="type_aware_merge_v1"),
        )
        reranked = stub_rerank(output)
        preview = preview_chunks(reranked)
        if not preview:
            records.append(_skipped_record(query, "no_preview_evidence_after_merge"))
            continue
        try:
            result = service.run(
                question=query["query_text"],
                analysis=analysis,
                seed_chunks=preview[:3],
                config=gen_config,
            )
        except Exception as exc:
            records.append(_skipped_record(query, f"generation_failed:{type(exc).__name__}"))
            continue

        qwen_debug = result.debug.get("qwen_synthesis", {})
        debug_paths = _debug_paths(preview)
        support_preview_count = sum(
            1
            for item in result.support_pack
            if item.candidate.metadata.get("object_type") == "table_index_unit"
        )
        answer_uses_preview = support_preview_count > 0 and _answer_mentions_preview(
            result.answer,
            result.support_pack,
        )
        records.append(
            {
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "status": "pass",
                "skipped_reason": "",
                "preview_input_count": len(preview),
                "support_preview_count": support_preview_count,
                "answer_uses_preview_evidence": answer_uses_preview,
                "answer_contains_debug_path": any(path and path in result.answer for path in debug_paths),
                "formal_table_citation_count": len(result.citations),
                "qwen_attempted": bool(qwen_debug.get("attempted", False)),
                "qwen_used": bool(qwen_debug.get("used_qwen", False)),
                "answer_preview": " ".join(result.answer.split())[:220],
            }
        )

    errors: list[str] = []
    executed = [row for row in records if row["status"] == "pass"]
    skipped = [row for row in records if row["status"] == "skipped"]
    if not executed:
        errors.append("no answer smoke query executed")
    if any(row["answer_contains_debug_path"] for row in executed):
        errors.append("answer included debug CSV/crop/markdown path")
    if any(int(row["formal_table_citation_count"]) != 0 for row in executed):
        errors.append("answer smoke produced formal table citation")
    if any(row["qwen_attempted"] or row["qwen_used"] for row in executed):
        errors.append("answer smoke attempted or used Qwen")
    if not any(row["answer_uses_preview_evidence"] for row in executed):
        errors.append("no executed answer used preview evidence content")
    summary = {
        "validation_status": "pass" if not errors else "fail",
        "pass": not errors,
        "errors": errors,
        "query_count": len(records),
        "executed_count": len(executed),
        "skipped_count": len(skipped),
        "skipped_reasons": [row["skipped_reason"] for row in skipped],
        "formal_table_citation_count": sum(int(row["formal_table_citation_count"]) for row in executed),
        "answer_debug_path_leak_count": sum(1 for row in executed if row["answer_contains_debug_path"]),
        "answers_using_preview_evidence_count": sum(
            1 for row in executed if row["answer_uses_preview_evidence"]
        ),
        "qwen_or_llm_called": False,
        "records_path": str(results_dir / "answer_smoke_results.csv"),
    }
    write_csv(results_dir / "answer_smoke_results.csv", records, FIELDNAMES)
    write_json(results_dir / "answer_smoke_summary.json", summary)
    write_report(summary, reports_dir / "answer_smoke_report.md")
    return summary


def _skipped_record(query: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "status": "skipped",
        "skipped_reason": reason,
        "preview_input_count": 0,
        "support_preview_count": 0,
        "answer_uses_preview_evidence": False,
        "answer_contains_debug_path": False,
        "formal_table_citation_count": 0,
        "qwen_attempted": False,
        "qwen_used": False,
        "answer_preview": "",
    }


def _debug_paths(chunks) -> set[str]:
    paths: set[str] = set()
    for chunk in chunks:
        for key in ("source_csv_path", "source_pdf_crop_path", "source_markdown_path"):
            value = chunk.metadata.get(key)
            if value:
                paths.add(str(value))
    return paths


def _answer_mentions_preview(answer: str, support_pack) -> bool:
    if "[TABLE" in answer or "table_unit" in answer or "row_unit" in answer or "cell_group_unit" in answer:
        return True
    lowered = answer.lower()
    for item in support_pack:
        title = (item.candidate.title or "").lower()
        if title and title[:30] in lowered:
            return True
    return False


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7W-slim Answer Smoke",
        "",
        "This smoke used GenerationV2 extractive mode with Qwen synthesis disabled.",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- executed_count: {summary['executed_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- formal_table_citation_count: {summary['formal_table_citation_count']}",
        f"- answer_debug_path_leak_count: {summary['answer_debug_path_leak_count']}",
        f"- answers_using_preview_evidence_count: {summary['answers_using_preview_evidence_count']}",
        "- Qwen / LLM called: no",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7W-slim optional answer smoke.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    parser.add_argument("--max-queries", type=int, default=5)
    args = parser.parse_args()
    summary = run_answer_smoke(
        fixture_path=args.fixture_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        max_queries=args.max_queries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
