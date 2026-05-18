#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit parsed_raw_v4 two-column recall diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counter(items: list[str]) -> dict[str, int]:
    return dict(Counter(items))


def page_row(item: dict[str, Any]) -> dict[str, Any]:
    likely_fallback = bool(item.get("likely_two_column_but_fallback"))
    return {
        "page": item.get("page"),
        "selected_order_strategy": item.get("selected_order_strategy", "unknown"),
        "layout_reason": item.get("reason", ""),
        "body_line_count": item.get("body_line_count", 0),
        "left_line_count": item.get("left_line_count", 0),
        "right_line_count": item.get("right_line_count", 0),
        "span_line_count": item.get("span_line_count", 0),
        "left_x_median": item.get("left_x_median"),
        "right_x_median": item.get("right_x_median"),
        "column_gap_estimate": item.get("column_gap_estimate"),
        "page_width": item.get("page_width"),
        "is_references_page": bool(item.get("is_references_page")),
        "is_front_matter_like": bool(item.get("is_front_matter_like")),
        "is_figure_table_heavy_page": bool(item.get("is_figure_table_heavy_page")),
        "likely_two_column_but_fallback": likely_fallback,
        "fallback_reason": item.get("fallback_reason") or (item.get("reason") if likely_fallback else ""),
    }


def audit_doc(path: Path) -> dict[str, Any]:
    data = load_json(path)
    diagnostics = data.get("diagnostics", {})
    layout_items = diagnostics.get("page_layout_diagnostics", []) or []
    pages = [page_row(item) for item in layout_items]
    order_dist = _counter([row["selected_order_strategy"] for row in pages])
    reason_dist = _counter([row["layout_reason"] for row in pages])
    two_column_pages = diagnostics.get("two_column_pages", []) or []
    likely_fallback_pages = [
        row["page"] for row in pages
        if row["likely_two_column_but_fallback"]
    ]

    return {
        "doc_id": data.get("doc_id", path.stem),
        "source_file": data.get("source_file"),
        "total_pages": data.get("total_pages", len(data.get("pages", []))),
        "current_two_column_pages": len(two_column_pages),
        "current_two_column_page_numbers": two_column_pages,
        "current_single_column_yx_pages": order_dist.get("single_column_yx", 0),
        "document_two_column_prior": bool(diagnostics.get("document_two_column_prior")),
        "document_two_column_confidence": diagnostics.get("document_two_column_confidence", 0.0),
        "document_two_column_prior_diagnostics": diagnostics.get("document_two_column_prior_diagnostics", {}),
        "likely_two_column_but_fallback_pages": likely_fallback_pages,
        "order_strategy_distribution": order_dist,
        "layout_reason_distribution": reason_dist,
        "pages": pages,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    order = Counter()
    reasons = Counter()
    for row in rows:
        order.update(row["order_strategy_distribution"])
        reasons.update(row["layout_reason_distribution"])

    likely_pages = [
        {"doc_id": row["doc_id"], "page": page}
        for row in rows
        for page in row["likely_two_column_but_fallback_pages"]
    ]
    return {
        "sample_count": len(rows),
        "total_pages": sum(int(row["total_pages"] or 0) for row in rows),
        "docs_with_document_two_column_prior": sum(1 for row in rows if row["document_two_column_prior"]),
        "total_two_column_pages": sum(int(row["current_two_column_pages"]) for row in rows),
        "total_single_column_yx_pages": sum(int(row["current_single_column_yx_pages"]) for row in rows),
        "total_likely_two_column_but_fallback_pages": len(likely_pages),
        "likely_two_column_but_fallback_pages": likely_pages,
        "order_strategy_distribution": dict(order),
        "layout_reason_distribution": dict(reasons),
    }


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]], parsed_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# v4 Phase1d Two-column Recall Audit")
    lines.append("")
    lines.append(f"- parsed_dir: `{parsed_dir}`")
    lines.append(f"- sample_count: {summary['sample_count']}")
    lines.append(f"- total_pages: {summary['total_pages']}")
    lines.append(f"- docs_with_document_two_column_prior: {summary['docs_with_document_two_column_prior']}")
    lines.append(f"- total_two_column_pages: {summary['total_two_column_pages']}")
    lines.append(f"- total_single_column_yx_pages: {summary['total_single_column_yx_pages']}")
    lines.append(f"- total_likely_two_column_but_fallback_pages: {summary['total_likely_two_column_but_fallback_pages']}")
    lines.append(f"- order_strategy_distribution: `{summary['order_strategy_distribution']}`")
    lines.append(f"- layout_reason_distribution: `{summary['layout_reason_distribution']}`")
    lines.append("")

    lines.append("## Document Summary")
    lines.append("")
    lines.append("| doc_id | pages | prior | confidence | two_column | single_column_yx | likely_fallback | top reasons |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        top_reasons = ", ".join(
            f"{reason}:{count}"
            for reason, count in Counter(row["layout_reason_distribution"]).most_common(4)
        )
        lines.append(
            "| {doc_id} | {pages} | {prior} | {conf} | {two} | {single} | {fallback} | {reasons} |".format(
                doc_id=row["doc_id"],
                pages=row["total_pages"],
                prior=row["document_two_column_prior"],
                conf=row["document_two_column_confidence"],
                two=row["current_two_column_pages"],
                single=row["current_single_column_yx_pages"],
                fallback=len(row["likely_two_column_but_fallback_pages"]),
                reasons=top_reasons or "-",
            )
        )
    lines.append("")

    lines.append("## Likely Two-column But Fallback")
    lines.append("")
    fallback_rows = [
        (row, page)
        for row in rows
        for page in row["pages"]
        if page["likely_two_column_but_fallback"]
    ]
    if not fallback_rows:
        lines.append("- none")
    else:
        lines.append("| doc_id | page | reason | fallback_reason | body | L | R | SPAN | gap |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|")
        for row, page in fallback_rows[:80]:
            lines.append(
                "| {doc_id} | {page} | {reason} | {fallback} | {body} | {left} | {right} | {span} | {gap} |".format(
                    doc_id=row["doc_id"],
                    page=page["page"],
                    reason=page["layout_reason"],
                    fallback=page["fallback_reason"],
                    body=page["body_line_count"],
                    left=page["left_line_count"],
                    right=page["right_line_count"],
                    span=page["span_line_count"],
                    gap=page["column_gap_estimate"],
                )
            )
    lines.append("")

    lines.append("## Manual Spot Check Candidates")
    lines.append("")
    lines.append("- Newly enabled two-column pages: sample the first 3-5 `two_column_span_left_right` pages per prior-positive doc.")
    lines.append("- High-risk exemptions: page 1/front matter, references pages, and figure/table-heavy pages with `single_column_yx`.")
    lines.append("- Remaining fallbacks: inspect every page listed in `Likely Two-column But Fallback`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parsed_raw_v4 two-column recall")
    parser.add_argument("--parsed_dir", required=True, help="Directory containing parsed_raw_v4 JSON files")
    parser.add_argument("--report", required=True, help="Markdown report path")
    parser.add_argument("--json", required=True, help="JSON report path")
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    json_paths = sorted(path for path in parsed_dir.glob("*.json") if path.is_file())
    rows = [audit_doc(path) for path in json_paths]
    summary = build_summary(rows)
    output = {"summary": summary, "documents": rows}

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, rows, parsed_dir), encoding="utf-8")
    print(f"[OK] wrote {report_path} and {json_path}")


if __name__ == "__main__":
    main()
