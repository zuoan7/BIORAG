#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spot-check parsed_raw_v4 two-column reading-order risks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    return text[:limit]


def _bbox(block: dict[str, Any]) -> list[float]:
    bbox = block.get("bbox") or []
    if len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(v) for v in bbox[:4]]


def _page_height(page: dict[str, Any], layout: dict[str, Any] | None) -> float:
    if layout and layout.get("page_height"):
        return float(layout["page_height"])
    blocks = page.get("blocks") or []
    max_y = max((_bbox(block)[3] for block in blocks), default=0.0)
    return max(max_y, 1.0)


def _page_width(page: dict[str, Any], layout: dict[str, Any] | None) -> float:
    if layout and layout.get("page_width"):
        return float(layout["page_width"])
    blocks = page.get("blocks") or []
    max_x = max((_bbox(block)[2] for block in blocks), default=0.0)
    return max(max_x, 1.0)


def _text_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block for block in (page.get("blocks") or [])
        if block.get("type") == "text" and str(block.get("text", "")).strip()
    ]


def _count_column_transitions(blocks: list[dict[str, Any]]) -> int:
    last = ""
    transitions = 0
    for block in sorted(blocks, key=lambda item: int(item.get("reading_order") or 0)):
        column = block.get("column")
        if column not in {"L", "R"}:
            continue
        if last and column != last:
            transitions += 1
        last = column
    return transitions


def _same_column_y_inversions(blocks: list[dict[str, Any]]) -> int:
    inversions = 0
    last_y_by_column: dict[str, float] = {}
    for block in sorted(blocks, key=lambda item: int(item.get("reading_order") or 0)):
        column = str(block.get("column") or "")
        if column not in {"L", "R"}:
            continue
        y0 = _bbox(block)[1]
        if column in last_y_by_column and y0 + 2.0 < last_y_by_column[column]:
            inversions += 1
        last_y_by_column[column] = y0
    return inversions


def _mid_span_blocks(
    blocks: list[dict[str, Any]],
    page_height: float,
) -> list[dict[str, Any]]:
    column_blocks = [block for block in blocks if block.get("column") in {"L", "R"}]
    if not column_blocks:
        return []
    min_col_y = min(_bbox(block)[1] for block in column_blocks)
    max_col_y = max(_bbox(block)[3] for block in column_blocks)
    top_limit = min_col_y + page_height * 0.06
    bottom_limit = max_col_y - page_height * 0.04
    spans = [
        block for block in blocks
        if block.get("column") not in {"L", "R"}
        and top_limit < _bbox(block)[1] < bottom_limit
    ]
    return spans


def _looks_like_region_anchor_block(block: dict[str, Any], page_width: float) -> bool:
    text = " ".join(str(block.get("text") or "").split())
    if not text or text.isdigit():
        return False
    lower = text.lower()
    if lower.startswith(("fig.", "fig ", "figure", "table")):
        return True
    heading_words = {
        "abstract",
        "introduction",
        "results",
        "discussion",
        "methods",
        "materials and methods",
        "conclusion",
        "conclusions",
        "funding",
        "acknowledgements",
    }
    if lower.strip(" .:") in heading_words:
        return True
    bbox = _bbox(block)
    width = max(0.0, bbox[2] - bbox[0])
    size = float(block.get("size") or 0.0)
    if width >= page_width * 0.68 and size >= 9.5 and 3 <= len(text.split()) <= 35:
        return True
    return False


def _span_displacement_count(blocks: list[dict[str, Any]], mid_spans: list[dict[str, Any]]) -> int:
    if not mid_spans:
        return 0
    column_orders = [
        int(block.get("reading_order") or 0)
        for block in blocks
        if block.get("column") in {"L", "R"}
    ]
    if not column_orders:
        return 0
    max_column_order = max(column_orders)
    return sum(
        1 for block in mid_spans
        if int(block.get("reading_order") or 0) > max_column_order
    )


def _caption_image_distance_risks(page: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = page.get("blocks") or []
    images = [block for block in blocks if block.get("type") == "image"]
    if not images:
        return []
    risks: list[dict[str, Any]] = []
    for block in _text_blocks(page):
        text = str(block.get("text") or "")
        if not text.lower().startswith(("fig.", "fig ", "figure", "table")):
            continue
        y0 = _bbox(block)[1]
        nearest = min(abs(y0 - _bbox(image)[1]) for image in images)
        if nearest > 180:
            risks.append({
                "block_id": block.get("block_id"),
                "text_preview": _preview(text),
                "nearest_image_y_distance": round(nearest, 2),
            })
    return risks


def _block_example(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block.get("block_id"),
        "text_preview": _preview(str(block.get("text") or "")),
        "bbox": block.get("bbox"),
        "column": block.get("column"),
        "reading_order": block.get("reading_order"),
    }


def audit_page(
    doc_id: str,
    page: dict[str, Any],
    layout: dict[str, Any] | None,
) -> dict[str, Any]:
    text_blocks = _text_blocks(page)
    page_height = _page_height(page, layout)
    page_width = _page_width(page, layout)
    mid_spans = _mid_span_blocks(text_blocks, page_height)
    mid_anchor_spans = [
        block for block in mid_spans
        if _looks_like_region_anchor_block(block, page_width)
    ]
    column_transitions = _count_column_transitions(text_blocks)
    y_inversions = _same_column_y_inversions(text_blocks)
    displaced_spans = _span_displacement_count(text_blocks, mid_anchor_spans)
    image_blocks = [block for block in (page.get("blocks") or []) if block.get("type") == "image"]
    caption_image_risks = _caption_image_distance_risks(page)

    selected_strategy = str((layout or {}).get("selected_order_strategy") or "unknown")
    reason = str((layout or {}).get("reason") or "")
    is_single_fallback = selected_strategy == "single_column_yx"
    lr_count = sum(1 for block in text_blocks if block.get("column") in {"L", "R"})
    risk_flags: list[str] = []
    if is_single_fallback and lr_count >= 12 and column_transitions >= 6:
        risk_flags.append("single_column_interleaved_lr")
    if displaced_spans:
        risk_flags.append("mid_anchor_displaced_after_columns")
    elif mid_anchor_spans and selected_strategy != "two_column_region_mixed":
        risk_flags.append("mid_anchor_requires_region_check")
    if (layout or {}).get("likely_two_column_but_fallback"):
        risk_flags.append("likely_two_column_but_fallback")
    if reason in {"document_prior_but_exempt_figure_table", "figure_table_heavy_page"}:
        risk_flags.append("figure_table_heavy_fallback")
    if image_blocks and selected_strategy.startswith("two_column"):
        risk_flags.append("two_column_page_with_image_blocks")
    if caption_image_risks:
        risk_flags.append("caption_image_distance_risk")
    if y_inversions:
        risk_flags.append("same_column_y_inversion")

    return {
        "doc_id": doc_id,
        "page": page.get("page"),
        "selected_order_strategy": selected_strategy,
        "layout_reason": reason,
        "is_two_column": bool((layout or {}).get("is_two_column")),
        "document_two_column_prior": bool((layout or {}).get("document_two_column_prior")),
        "likely_two_column_but_fallback": bool((layout or {}).get("likely_two_column_but_fallback")),
        "body_line_count": (layout or {}).get("body_line_count"),
        "left_line_count": (layout or {}).get("left_line_count"),
        "right_line_count": (layout or {}).get("right_line_count"),
        "span_line_count": (layout or {}).get("span_line_count"),
        "page_width": page_width,
        "page_height": page_height,
        "text_block_count": len(text_blocks),
        "image_block_count": len(image_blocks),
        "column_transition_count": column_transitions,
        "same_column_y_inversion_count": y_inversions,
        "mid_span_block_count": len(mid_spans),
        "mid_anchor_span_block_count": len(mid_anchor_spans),
        "span_displacement_count": displaced_spans,
        "caption_image_distance_risk_count": len(caption_image_risks),
        "risk_flags": risk_flags,
        "mid_span_examples": [_block_example(block) for block in mid_spans[:5]],
        "mid_anchor_span_examples": [_block_example(block) for block in mid_anchor_spans[:5]],
        "caption_image_distance_examples": caption_image_risks[:5],
    }


def audit_doc(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    doc_id = str(data.get("doc_id") or path.stem)
    layout_by_page = {
        int(item.get("page")): item
        for item in (data.get("diagnostics", {}).get("page_layout_diagnostics") or [])
        if item.get("page") is not None
    }
    pages = [
        audit_page(doc_id, page, layout_by_page.get(int(page.get("page") or 0)))
        for page in data.get("pages", [])
    ]
    risk_pages = [page for page in pages if page["risk_flags"]]
    return {
        "doc_id": doc_id,
        "source_file": data.get("source_file"),
        "total_pages": data.get("total_pages", len(pages)),
        "document_two_column_prior": bool(data.get("diagnostics", {}).get("document_two_column_prior")),
        "order_strategy_distribution": dict(Counter(page["selected_order_strategy"] for page in pages)),
        "layout_reason_distribution": dict(Counter(page["layout_reason"] for page in pages)),
        "risk_flag_distribution": dict(Counter(flag for page in pages for flag in page["risk_flags"])),
        "risk_page_count": len(risk_pages),
        "risk_pages": risk_pages,
        "pages": pages,
    }


def summarize(documents: list[dict[str, Any]]) -> dict[str, Any]:
    order = Counter()
    reasons = Counter()
    risks = Counter()
    for doc in documents:
        order.update(doc["order_strategy_distribution"])
        reasons.update(doc["layout_reason_distribution"])
        risks.update(doc["risk_flag_distribution"])
    return {
        "doc_count": len(documents),
        "total_pages": sum(int(doc.get("total_pages") or 0) for doc in documents),
        "risk_page_count": sum(int(doc["risk_page_count"]) for doc in documents),
        "order_strategy_distribution": dict(order),
        "layout_reason_distribution": dict(reasons),
        "risk_flag_distribution": dict(risks),
    }


def render_report(summary: dict[str, Any], documents: list[dict[str, Any]], parsed_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# v5 Two-column Order Spot-check")
    lines.append("")
    lines.append(f"- parsed_dir: `{parsed_dir}`")
    lines.append(f"- doc_count: {summary['doc_count']}")
    lines.append(f"- total_pages: {summary['total_pages']}")
    lines.append(f"- risk_page_count: {summary['risk_page_count']}")
    lines.append(f"- order_strategy_distribution: `{summary['order_strategy_distribution']}`")
    lines.append(f"- layout_reason_distribution: `{summary['layout_reason_distribution']}`")
    lines.append(f"- risk_flag_distribution: `{summary['risk_flag_distribution']}`")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append("- The current clean/chunk stages consume parser reading order and do not re-order blocks.")
    lines.append("- `single_column_interleaved_lr` flags pages where y/x sorting alternates left and right columns.")
    lines.append("- `mid_anchor_displaced_after_columns` flags heading/caption-like SPAN blocks visually in the middle of a two-column body but ordered after both columns.")
    lines.append("- `figure_table_heavy_fallback` flags pages where mixed layout was exempted from two-column ordering.")
    lines.append("")
    lines.append("## Document Risk Summary")
    lines.append("")
    lines.append("| doc_id | pages | prior | risk_pages | top risks | strategies |")
    lines.append("|---|---:|---|---:|---|---|")
    for doc in documents:
        top_risks = ", ".join(f"{k}:{v}" for k, v in Counter(doc["risk_flag_distribution"]).most_common(4)) or "-"
        strategies = ", ".join(f"{k}:{v}" for k, v in Counter(doc["order_strategy_distribution"]).most_common())
        lines.append(
            f"| {doc['doc_id']} | {doc['total_pages']} | {doc['document_two_column_prior']} | "
            f"{doc['risk_page_count']} | {top_risks} | {strategies} |"
        )
    lines.append("")
    lines.append("## Spot-check Pages")
    lines.append("")
    risk_pages = [
        page for doc in documents for page in doc["risk_pages"]
    ]
    if not risk_pages:
        lines.append("- none")
    else:
        lines.append("| doc_id | page | strategy | reason | flags | L/R/SPAN | transitions | mid_anchor | examples |")
        lines.append("|---|---:|---|---|---|---:|---:|---:|---|")
        for page in risk_pages[:120]:
            examples = "; ".join(
                example["text_preview"] for example in page["mid_anchor_span_examples"][:2]
            ) or "-"
            lines.append(
                "| {doc_id} | {page_no} | {strategy} | {reason} | {flags} | "
                "{left}/{right}/{span} | {transitions} | {mid_span} | {examples} |".format(
                    doc_id=page["doc_id"],
                    page_no=page["page"],
                    strategy=page["selected_order_strategy"],
                    reason=page["layout_reason"],
                    flags=", ".join(page["risk_flags"]),
                    left=page.get("left_line_count"),
                    right=page.get("right_line_count"),
                    span=page.get("span_line_count"),
                    transitions=page["column_transition_count"],
                    mid_span=page["mid_anchor_span_block_count"],
                    examples=examples.replace("|", "\\|"),
                )
            )
    lines.append("")
    lines.append("## Parser Change Plan")
    lines.append("")
    lines.append("1. Keep document-level two-column prior and conservative exemptions.")
    lines.append("2. Replace page-level `top SPAN -> left -> right -> bottom SPAN` sorting with region-aware sorting for two-column pages.")
    lines.append("3. Treat wide SPAN headings/captions as vertical region anchors so middle captions or headings stay between surrounding text regions.")
    lines.append("4. Preserve single-column fallback for front matter, references, and pages without enough left/right evidence.")
    lines.append("5. Re-run this script after parser changes and compare risk flags before entering clean/chunk work.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parsed_raw_v4 two-column reading-order risks")
    parser.add_argument("--parsed_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    docs = [audit_doc(path) for path in sorted(parsed_dir.glob("*.json")) if path.is_file()]
    summary = summarize(docs)
    output = {"summary": summary, "documents": docs}

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, docs, parsed_dir), encoding="utf-8")
    print(f"[OK] wrote {report_path} and {json_path}")


if __name__ == "__main__":
    main()
