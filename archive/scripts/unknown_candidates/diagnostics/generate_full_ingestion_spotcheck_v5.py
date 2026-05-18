#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a manual spot-check list for the v5 mini ingestion regression."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


KNOWN_RISK_PAGES = [
    ("doc_0235", 16),
    ("doc_0430", 20),
    ("doc_0005", 11),
    ("doc_0083", 7),
    ("doc_0163", 4),
]

PARSER_RISK_PRIORITY = {
    "likely_two_column_but_fallback": 0,
    "figure_table_heavy_fallback": 1,
    "caption_image_distance_risk": 2,
    "two_column_page_with_image_blocks": 3,
    "mid_anchor_requires_region_check": 4,
}

TABLE_TEXT_RISK = re.compile(
    r"\[TABLE TEXT\]\s+(?:was mixed|were mixed|incubated|After identification|48%\s+byproduct|gDW-1|NGAM|at a compound annual growth rate)",
    re.I,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _preview(text: str, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _md(text: Any) -> str:
    return str(text or "").replace("|", "\\|")


def _chunk_page(chunk: dict[str, Any]) -> int | None:
    pages = chunk.get("page_numbers") or []
    if pages:
        try:
            return int(pages[0])
        except (TypeError, ValueError):
            return None
    page_start = chunk.get("page_start")
    return int(page_start) if page_start is not None else None


def _chunk_item(chunk: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "doc_id": chunk.get("doc_id"),
        "page": _chunk_page(chunk),
        "chunk_id": chunk.get("chunk_id"),
        "section": chunk.get("section"),
        "reason": reason,
        "text_preview": _preview(chunk.get("text", "")),
        "source_block_ids": chunk.get("source_block_ids", [])[:12],
        "block_types": chunk.get("block_types", []),
        "pdf_page": _chunk_page(chunk),
    }


def _parser_risk_pages(two_column_json: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for doc in two_column_json.get("documents", []) or []:
        for page in doc.get("risk_pages", []) or []:
            flags = page.get("risk_flags", []) or []
            if not any(flag in PARSER_RISK_PRIORITY for flag in flags):
                continue
            priority = min(PARSER_RISK_PRIORITY.get(flag, 99) for flag in flags)
            known_bonus = -1 if (page.get("doc_id"), page.get("page")) in KNOWN_RISK_PAGES else 0
            pages.append({**page, "_priority": priority + known_bonus})
    pages.sort(key=lambda item: (item["_priority"], str(item.get("doc_id")), int(item.get("page") or 0)))

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for doc_id, page_num in KNOWN_RISK_PAGES:
        for page in pages:
            key = (str(page.get("doc_id")), int(page.get("page") or 0))
            if key == (doc_id, page_num) and key not in seen:
                selected.append(page)
                seen.add(key)
                break
    for page in pages:
        key = (str(page.get("doc_id")), int(page.get("page") or 0))
        if key in seen:
            continue
        selected.append(page)
        seen.add(key)
        if len(selected) >= 20:
            break
    return selected[:20]


def _table_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        chunk for chunk in chunks
        if chunk.get("contains_table_caption") or chunk.get("contains_table_text")
        or "[TABLE CAPTION]" in (chunk.get("text") or "")
        or "[TABLE TEXT]" in (chunk.get("text") or "")
    ]

    def priority(chunk: dict[str, Any]) -> tuple[int, int, str]:
        text = chunk.get("text", "") or ""
        both = "[TABLE CAPTION]" in text and "[TABLE TEXT]" in text
        risk = bool(TABLE_TEXT_RISK.search(text))
        long_chunk = len(text) > 1800
        score = 0
        if risk:
            score -= 20
        if both:
            score -= 10
        if long_chunk:
            score -= 3
        if chunk.get("contains_table_text"):
            score -= 1
        return (score, int(chunk.get("chunk_index") or 0), str(chunk.get("chunk_id") or ""))

    return [_chunk_item(chunk, "table_caption_or_table_text") for chunk in sorted(candidates, key=priority)[:10]]


def _figure_chunks(chunks: list[dict[str, Any]], risk_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risk_page_keys = {(page.get("doc_id"), int(page.get("page") or 0)) for page in risk_pages}
    candidates = [
        chunk for chunk in chunks
        if chunk.get("contains_figure_caption") or "[FIGURE CAPTION]" in (chunk.get("text") or "")
    ]

    def priority(chunk: dict[str, Any]) -> tuple[int, int, str]:
        page_key = (chunk.get("doc_id"), _chunk_page(chunk) or -1)
        score = -10 if page_key in risk_page_keys else 0
        return (score, int(chunk.get("chunk_index") or 0), str(chunk.get("chunk_id") or ""))

    return [_chunk_item(chunk, "figure_caption") for chunk in sorted(candidates, key=priority)[:10]]


def render_report(result: dict[str, Any]) -> str:
    lines = ["# v5 Full Ingestion Manual Spot Check", ""]
    lines.append("## A. Parser Reading Order Pages")
    lines.append("")
    parser_pages = result["parser_reading_order_pages"]
    if not parser_pages:
        lines.append("- none")
    else:
        lines.append("| doc_id | page | strategy | reason | flags | block_id/chunk_id | text_preview | source_block_ids | block_types | pdf_page |")
        lines.append("|---|---:|---|---|---|---|---|---|---|---:|")
        for page in parser_pages:
            examples = page.get("mid_anchor_span_examples") or page.get("caption_image_distance_examples") or []
            text_preview = _preview("; ".join(str(ex.get("text_preview", "")) for ex in examples[:2])) or "-"
            block_id = ", ".join(str(ex.get("block_id")) for ex in examples[:2] if ex.get("block_id")) or "-"
            lines.append(
                f"| {page.get('doc_id')} | {page.get('page')} | {page.get('selected_order_strategy')} | "
                f"{page.get('layout_reason')} | {', '.join(page.get('risk_flags', []))} | {block_id} | "
                f"{_md(text_preview)} | - | - | {page.get('page')} |"
            )
    lines.append("")
    lines.append("## B. Table Chunk Checks")
    lines.append("")
    _append_chunk_table(lines, result["table_chunk_checks"])
    lines.append("")
    lines.append("## C. Figure Chunk Checks")
    lines.append("")
    _append_chunk_table(lines, result["figure_chunk_checks"])
    return "\n".join(lines) + "\n"


def _append_chunk_table(lines: list[str], chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        lines.append("- none")
        return
    lines.append("| doc_id | page | chunk_id | section | reason | text_preview | source_block_ids | block_types | pdf_page |")
    lines.append("|---|---:|---|---|---|---|---|---|---:|")
    for item in chunks:
        lines.append(
            f"| {item.get('doc_id')} | {item.get('page')} | {item.get('chunk_id')} | "
            f"{_md(item.get('section'))} | {item.get('reason')} | "
            f"{_md(item.get('text_preview'))} | "
            f"{', '.join(map(str, item.get('source_block_ids', [])))} | "
            f"{', '.join(map(str, item.get('block_types', [])))} | {item.get('pdf_page')} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v5 full ingestion manual spot-check report")
    parser.add_argument("--two_column_json", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    two_column = _load_json(Path(args.two_column_json))
    chunks = _load_chunks(Path(args.chunks_jsonl))
    parser_pages = _parser_risk_pages(two_column)
    result = {
        "parser_reading_order_pages": parser_pages,
        "table_chunk_checks": _table_chunks(chunks),
        "figure_chunk_checks": _figure_chunks(chunks, parser_pages),
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"[OK] wrote {report_path} and {json_path}")


if __name__ == "__main__":
    main()
