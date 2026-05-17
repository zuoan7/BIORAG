#!/usr/bin/env python3
"""Render Phase7J table retrieval wiring preview evidence cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPK_JSONL = (
    ROOT / "results/v7_phase7_table_retrieval_wiring_preview/topk_evidence_units.jsonl"
)
DEFAULT_REPORT_PATH = (
    ROOT / "reports/v7_phase7_table_retrieval_wiring_preview/retrieval_evidence_cards.md"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def md_escape(value: Any) -> str:
    return normalize(value).replace("|", "\\|")


def compact(value: Any, limit: int = 420) -> str:
    text = normalize(value)
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_target(record: dict[str, Any]) -> str:
    parts = [
        f"seed={record.get('expected_seed_id', '')}",
        f"doc={record.get('expected_doc_id', '')}",
        f"table={record.get('expected_table_id', '')}",
        f"unit_type={record.get('expected_unit_type', '')}",
        f"unit_id={record.get('expected_table_index_unit_id', '')}",
    ]
    if record.get("expected_row_label"):
        parts.append(f"row_label={record['expected_row_label']}")
    return "; ".join(parts)


def render_evidence_cards(
    topk_jsonl: Path = DEFAULT_TOPK_JSONL,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    topk_jsonl = resolve_path(topk_jsonl)
    report_path = resolve_path(report_path)
    records = load_jsonl(topk_jsonl)

    lines = [
        "# Phase7J 检索证据卡",
        "",
        "## 定位",
        "",
        "本文件用于人工检查 Phase7J offline table retrieval wiring preview 的 top-k evidence 展示形态。它不评价 LLM answer，不调用 Qwen，不运行 RAGAS，不构成正式 retrieval evaluation 或 benchmark。",
        "",
        "## Guardrail",
        "",
        "- evidence 只来自 Phase7I-1 eligible units。",
        "- 本轮只做 isolated lexical dry-run。",
        "- production_ready 必须为 false。",
        "- value_bboxes_available 必须为 false。",
        "- 不声称 value-level bbox，不接 production。",
        "",
    ]

    for record in records:
        lines.extend(
            [
                f"## {record.get('query_id', '')}",
                "",
                f"- query_text：{normalize(record.get('query_text'))}",
                f"- query_type：`{record.get('query_type', '')}`",
                f"- 期望目标：`{expected_target(record)}`",
                "",
                "| rank | match_status | unit_type | unit_id | doc_id | table_id | row_label | score | value_bboxes_available | production_ready |",
                "| ---: | --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for row in record.get("top_k") or []:
            lines.append(
                "| {rank} | `{status}` | `{unit_type}` | `{unit_id}` | `{doc}` | `{table}` | {row_label} | {score} | `{value_bbox}` | `{prod}` |".format(
                    rank=row.get("rank", ""),
                    status=md_escape(row.get("match_status", "")),
                    unit_type=md_escape(row.get("matched_unit_type", "")),
                    unit_id=md_escape(row.get("matched_table_index_unit_id", "")),
                    doc=md_escape(row.get("matched_doc_id", "")),
                    table=md_escape(row.get("matched_table_id", "")),
                    row_label=md_escape(row.get("matched_row_label", "")),
                    score=md_escape(row.get("score", "")),
                    value_bbox=md_escape(row.get("value_bboxes_available", "")),
                    prod=md_escape(row.get("production_ready", "")),
                )
            )
        lines.extend(["", "### Evidence Text", ""])
        for row in record.get("top_k") or []:
            lines.extend(
                [
                    f"**rank {row.get('rank', '')} / {row.get('match_status', '')}**",
                    "",
                    f"- evidence_text：{compact(row.get('evidence_text'))}",
                    f"- source_csv_path：`{row.get('source_csv_path', '')}`",
                    f"- source_pdf_crop_path：`{row.get('source_pdf_crop_path', '')}`",
                    f"- guardrail warnings：`{row.get('guardrail_limitation', record.get('guardrail_limitation', ''))}`",
                    "",
                ]
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "query_count": len(records),
        "report_path": rel(report_path),
        "source_path": rel(topk_jsonl),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-jsonl", type=Path, default=DEFAULT_TOPK_JSONL)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_evidence_cards(topk_jsonl=args.topk_jsonl, report_path=args.report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
