#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate parsed_raw_v4 outputs against old parsed_raw JSON.

This script is audit-only. It does not touch ingestion clean/chunk/import,
retrieval, generation, reranker, Milvus, BM25, or datasets.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


UNICODE_SPACE_CHARS_RE = re.compile(r"[\u00a0\u202f\u2009\u2007]")
DASH_CHARS_RE = re.compile(r"[\u2013\u2014\u2212]")

JOURNAL_PREPROOF_NOISE_PATTERNS = [
    ("journal_preproof_exact_line", re.compile(r"^journal pre-proofs?$")),
    ("journal_preproof_front_matter_disclaimer", re.compile(r"^this is a pdf file of an article")),
    ("journal_preproof_metadata_line", re.compile(r"^PII:\s*")),
    ("journal_preproof_metadata_line", re.compile(r"^DOI:\s*")),
    ("journal_preproof_metadata_line", re.compile(r"^Reference:\s*")),
    ("journal_preproof_metadata_line", re.compile(r"^To appear in:\s*")),
    ("journal_preproof_metadata_line", re.compile(r"^Received Date:\s*")),
    ("journal_preproof_metadata_line", re.compile(r"^Revised Date:\s*")),
    ("journal_preproof_metadata_line", re.compile(r"^Accepted Date:\s*")),
    ("journal_preproof_front_matter_disclaimer", re.compile(r"^please cite this article as:")),
    ("journal_preproof_front_matter_disclaimer", re.compile(r"^this manuscript has been accepted")),
    ("journal_preproof_front_matter_disclaimer", re.compile(r"^the manuscript will undergo copyediting")),
]

FIGURE_CAPTION_RE = re.compile(
    r"^(?:Supplementary\s+)?(?:Fig\.?|Figure)\s+S?\d+[A-Za-z]?"
    r"(?:\s*[\.\:\-]|\s|$)",
    re.I,
)

TABLE_CAPTION_RE = re.compile(
    r"^(?:Supplementary\s+)?Table\s+S?\d+[A-Za-z]?"
    r"(?:\s*[\.\:\-]|\s|$)",
    re.I,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_dir(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists() or not path.is_dir():
        return {}

    docs: dict[str, dict[str, Any]] = {}
    for json_path in sorted(path.glob("*.json")):
        try:
            data = load_json(json_path)
        except json.JSONDecodeError:
            continue
        doc_id = str(data.get("doc_id") or json_path.stem)
        docs[doc_id] = data
    return docs


def get_pages(data: dict[str, Any]) -> list[dict[str, Any]]:
    pages = data.get("pages", [])
    return pages if isinstance(pages, list) else []


def page_text(page: dict[str, Any]) -> str:
    text = page.get("text", "")
    return text if isinstance(text, str) else ""


def normalize_residual_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = UNICODE_SPACE_CHARS_RE.sub(" ", text)
    text = DASH_CHARS_RE.sub("-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lstrip("#").strip().lower()


def match_journal_preproof_noise(text: str) -> tuple[bool, str | None]:
    normalized = normalize_residual_text(text)
    for pattern_name, pattern in JOURNAL_PREPROOF_NOISE_PATTERNS:
        if pattern_name == "journal_preproof_metadata_line":
            raw = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lstrip("#").strip()
            if pattern.search(raw):
                return True, pattern_name
            continue
        if pattern.search(normalized):
            return True, pattern_name
    return False, None


def preview_text(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def total_text_chars(data: dict[str, Any]) -> int:
    return sum(len(page_text(page)) for page in get_pages(data))


def empty_page_count(data: dict[str, Any]) -> int:
    return sum(1 for page in get_pages(data) if not page_text(page).strip())


def total_pages(data: dict[str, Any]) -> int:
    value = data.get("total_pages")
    if isinstance(value, int):
        return value
    return len(get_pages(data))


def line_count_matching_noise(text: str) -> int:
    count = 0
    for line in text.splitlines():
        matched, _pattern_name = match_journal_preproof_noise(line)
        if matched:
            count += 1
    return count


def page_noise_hits(doc_id: str, data: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    residual_examples: list[dict[str, Any]] = []
    for page in get_pages(data):
        page_no = page.get("page")
        for line in page_text(page).splitlines():
            matched, pattern_name = match_journal_preproof_noise(line)
            if not matched:
                continue
            residual_examples.append({
                "doc_id": doc_id,
                "page": page_no,
                "field": "page_text",
                "block_id": None,
                "raw_text_preview": preview_text(line),
                "normalized_text": normalize_residual_text(line),
                "bbox": None,
                "column": None,
                "reading_order": None,
                "size": None,
                "matched_pattern": pattern_name,
                "is_in_diagnostics_only": False,
            })

    hits = [
        {"page": page_no, "count": count}
        for page_no, count in Counter(item["page"] for item in residual_examples).items()
    ]
    return len(residual_examples), hits, residual_examples


def block_noise_hits(doc_id: str, data: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    residual_examples: list[dict[str, Any]] = []
    for page in get_pages(data):
        page_no = page.get("page")
        for block in page.get("blocks", []) or []:
            text = block.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            for line in text.splitlines() or [text]:
                matched, pattern_name = match_journal_preproof_noise(line)
                if not matched:
                    continue
                residual_examples.append({
                    "doc_id": doc_id,
                    "page": page_no,
                    "field": "block_text",
                    "block_id": block.get("block_id"),
                    "raw_text_preview": preview_text(line),
                    "normalized_text": normalize_residual_text(line),
                    "bbox": block.get("bbox"),
                    "column": block.get("column"),
                    "reading_order": block.get("reading_order"),
                    "size": block.get("size"),
                    "matched_pattern": pattern_name,
                    "is_in_diagnostics_only": False,
                })

    hits = [
        {
            "page": item["page"],
            "block_id": item.get("block_id"),
            "count": 1,
        }
        for item in residual_examples
    ]
    return len(residual_examples), hits, residual_examples


def diagnostics_noise_count(data: dict[str, Any]) -> int:
    diagnostics = data.get("diagnostics", {}) or {}
    examples = diagnostics.get("stripped_noise_examples", []) or []
    return sum(
        1 for example in examples
        if isinstance(example, str)
        and match_journal_preproof_noise(example)[0]
    )


def diagnostics_noise_examples(doc_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = data.get("diagnostics", {}) or {}
    examples = diagnostics.get("stripped_noise_examples", []) or []
    rows: list[dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, str):
            continue
        matched, pattern_name = match_journal_preproof_noise(example)
        if not matched:
            continue
        rows.append({
            "doc_id": doc_id,
            "page": None,
            "field": "diagnostics",
            "block_id": None,
            "raw_text_preview": preview_text(example),
            "normalized_text": normalize_residual_text(example),
            "bbox": None,
            "column": None,
            "reading_order": None,
            "size": None,
            "matched_pattern": pattern_name,
            "is_in_diagnostics_only": True,
        })
    return rows


def count_caption_candidates_in_text(data: dict[str, Any]) -> tuple[int, int]:
    figure_count = 0
    table_count = 0
    for page in get_pages(data):
        for line in page_text(page).splitlines():
            stripped = line.strip().lstrip("#").strip()
            if FIGURE_CAPTION_RE.match(stripped):
                figure_count += 1
            if TABLE_CAPTION_RE.match(stripped):
                table_count += 1
    return figure_count, table_count


def order_strategy_distribution(data: dict[str, Any]) -> dict[str, int]:
    diagnostics = data.get("diagnostics", {}) or {}
    layout = diagnostics.get("page_layout_diagnostics", []) or []
    return dict(Counter(
        str(item.get("selected_order_strategy", "unknown"))
        for item in layout
        if isinstance(item, dict)
    ))


def reason_distribution(data: dict[str, Any]) -> dict[str, int]:
    diagnostics = data.get("diagnostics", {}) or {}
    layout = diagnostics.get("page_layout_diagnostics", []) or []
    return dict(Counter(
        str(item.get("reason", "unknown"))
        for item in layout
        if isinstance(item, dict)
    ))


def empty_pages(data: dict[str, Any]) -> list[int]:
    return [
        int(page.get("page") or index + 1)
        for index, page in enumerate(get_pages(data))
        if not page_text(page).strip()
    ]


def analyze_doc(doc_id: str, new_data: dict[str, Any], old_data: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = new_data.get("diagnostics", {}) or {}
    old_chars = total_text_chars(old_data) if old_data else None
    new_chars = total_text_chars(new_data)
    if old_chars and old_chars > 0:
        char_delta_ratio = round((new_chars - old_chars) / old_chars, 4)
    else:
        char_delta_ratio = None

    old_empty_count = empty_page_count(old_data) if old_data else None
    new_empty_count = empty_page_count(new_data)
    old_figure_count, old_table_count = count_caption_candidates_in_text(old_data) if old_data else (None, None)
    new_figure_count = int(diagnostics.get("figure_caption_candidate_count", 0) or 0)
    new_table_count = int(diagnostics.get("table_caption_candidate_count", 0) or 0)
    if new_figure_count == 0 and new_table_count == 0:
        fallback_figures, fallback_tables = count_caption_candidates_in_text(new_data)
        new_figure_count = fallback_figures
        new_table_count = fallback_tables

    text_noise_count, text_noise_pages, page_residual_examples = page_noise_hits(doc_id, new_data)
    block_noise_count, block_noise_pages, block_residual_examples = block_noise_hits(doc_id, new_data)
    diagnostics_residual_examples = diagnostics_noise_examples(doc_id, new_data)
    two_column_pages = diagnostics.get("two_column_pages", []) or []
    reason_dist = reason_distribution(new_data)

    potential_regressions: list[str] = []
    manual_review_pages: list[dict[str, Any]] = []

    if char_delta_ratio is not None and char_delta_ratio < -0.3:
        potential_regressions.append("text_char_delta_ratio_lt_-0.3")
        manual_review_pages.append({"doc_id": doc_id, "page": None, "reason": "large_text_char_drop"})

    if old_empty_count is not None and new_empty_count > old_empty_count:
        potential_regressions.append("new_empty_page_count_gt_old")
        for page_no in empty_pages(new_data):
            manual_review_pages.append({"doc_id": doc_id, "page": page_no, "reason": "new_empty_page"})

    if old_figure_count is not None and old_figure_count >= 3 and new_figure_count < old_figure_count * 0.5:
        potential_regressions.append("figure_caption_candidate_count_dropped")
    if old_table_count is not None and old_table_count >= 3 and new_table_count < old_table_count * 0.5:
        potential_regressions.append("table_caption_candidate_count_dropped")

    page_total = max(total_pages(new_data), 1)
    if len(two_column_pages) >= 5 and len(two_column_pages) / page_total > 0.7:
        potential_regressions.append("high_two_column_page_ratio_manual_check")

    if reason_dist:
        top_reason, top_count = max(reason_dist.items(), key=lambda item: item[1])
        if top_reason not in {"left_right_distribution_clear", "not_enough_body_lines"} and page_total >= 3:
            if top_count / page_total >= 0.8:
                potential_regressions.append(f"layout_reason_concentrated:{top_reason}")

    if text_noise_count:
        potential_regressions.append("journal_preproof_noise_residual_in_page_text")
        for hit in text_noise_pages:
            manual_review_pages.append({"doc_id": doc_id, "page": hit["page"], "reason": "journal_preproof_text_residual"})

    if block_noise_count:
        potential_regressions.append("journal_preproof_noise_residual_in_block_text")
        for hit in block_noise_pages:
            manual_review_pages.append({
                "doc_id": doc_id,
                "page": hit["page"],
                "reason": "journal_preproof_block_residual",
                "block_id": hit.get("block_id"),
            })

    for page_no in two_column_pages[:8]:
        manual_review_pages.append({"doc_id": doc_id, "page": page_no, "reason": "two_column_order_spot_check"})

    return {
        "doc_id": doc_id,
        "total_pages": total_pages(new_data),
        "text_char_count_old": old_chars,
        "text_char_count_new": new_chars,
        "text_char_delta_ratio": char_delta_ratio,
        "empty_page_count_old": old_empty_count,
        "empty_page_count": new_empty_count,
        "parser_stage": new_data.get("parser_stage"),
        "two_column_pages": two_column_pages,
        "order_strategy_distribution": order_strategy_distribution(new_data),
        "layout_reason_distribution": reason_dist,
        "image_block_count": int(diagnostics.get("image_block_count", 0) or 0),
        "figure_caption_candidate_count_old": old_figure_count,
        "figure_caption_candidate_count": new_figure_count,
        "table_caption_candidate_count_old": old_table_count,
        "table_caption_candidate_count": new_table_count,
        "journal_preproof_noise_in_page_text_count": text_noise_count,
        "journal_preproof_noise_in_block_text_count": block_noise_count,
        "journal_preproof_noise_in_diagnostics_count": len(diagnostics_residual_examples),
        "stripped_noise_line_count": int(diagnostics.get("stripped_noise_line_count", 0) or 0),
        "residual_examples": page_residual_examples + block_residual_examples,
        "diagnostics_noise_examples": diagnostics_residual_examples,
        "potential_regressions": potential_regressions,
        "manual_review_pages": manual_review_pages,
    }


def summarize(results: list[dict[str, Any]], has_old_compare: bool, old_dir_missing: bool) -> dict[str, Any]:
    potential_regression_docs = [
        item for item in results
        if item["potential_regressions"]
    ]
    fail_risks = {
        "text_char_delta_ratio_lt_-0.3",
        "new_empty_page_count_gt_old",
        "journal_preproof_noise_residual_in_page_text",
    }
    has_fail = any(
        any(risk in fail_risks for risk in item["potential_regressions"])
        for item in results
    )
    if has_fail:
        conclusion = "fail"
    elif potential_regression_docs or not has_old_compare:
        conclusion = "warning"
    else:
        conclusion = "pass"

    return {
        "sample_count": len(results),
        "has_old_new_compare": has_old_compare,
        "old_dir_missing_or_unusable": old_dir_missing,
        "conclusion": conclusion,
        "total_pages": sum(item["total_pages"] for item in results),
        "total_image_block_count": sum(item["image_block_count"] for item in results),
        "total_two_column_pages": sum(len(item["two_column_pages"]) for item in results),
        "total_figure_caption_candidate_count": sum(item["figure_caption_candidate_count"] for item in results),
        "total_table_caption_candidate_count": sum(item["table_caption_candidate_count"] for item in results),
        "total_journal_preproof_noise_in_page_text_count": sum(
            item["journal_preproof_noise_in_page_text_count"] for item in results
        ),
        "total_journal_preproof_noise_in_block_text_count": sum(
            item["journal_preproof_noise_in_block_text_count"] for item in results
        ),
        "total_journal_preproof_noise_in_diagnostics_count": sum(
            item["journal_preproof_noise_in_diagnostics_count"] for item in results
        ),
        "total_stripped_noise_line_count": sum(item["stripped_noise_line_count"] for item in results),
        "potential_regression_doc_count": len(potential_regression_docs),
        "order_strategy_distribution": aggregate_counter(
            item["order_strategy_distribution"] for item in results
        ),
        "layout_reason_distribution": aggregate_counter(
            item["layout_reason_distribution"] for item in results
        ),
        "repeated_block_residuals": aggregate_repeated_block_residuals(results),
        "residual_examples": [
            example
            for item in results
            for example in item.get("residual_examples", [])
        ],
        "diagnostics_noise_examples": [
            example
            for item in results
            for example in item.get("diagnostics_noise_examples", [])
        ],
    }


def aggregate_counter(items: list[dict[str, int]] | Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(item)
    return dict(counter)


def bbox_band(bbox: Any) -> str:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return "unknown"
    try:
        y0 = float(bbox[1])
    except (TypeError, ValueError):
        return "unknown"
    if y0 < 160:
        return "top"
    if y0 > 620:
        return "bottom"
    return "middle"


def aggregate_repeated_block_residuals(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        for example in item.get("residual_examples", []):
            if example.get("field") != "block_text":
                continue
            key = str(example.get("normalized_text") or "")
            if not key:
                continue
            grouped.setdefault(key, []).append(example)

    rows: list[dict[str, Any]] = []
    for normalized_text, examples in grouped.items():
        band_counts = Counter(bbox_band(example.get("bbox")) for example in examples)
        common_band = band_counts.most_common(1)[0][0] if band_counts else "unknown"
        page_keys = {
            (example.get("doc_id"), example.get("page"))
            for example in examples
        }
        rows.append({
            "normalized_text": normalized_text,
            "doc_ids": sorted({str(example.get("doc_id")) for example in examples}),
            "page_count": len(page_keys),
            "occurrence_count": len(examples),
            "example_pages": [
                {"doc_id": example.get("doc_id"), "page": example.get("page")}
                for example in examples[:12]
            ],
            "example_block_ids": [
                example.get("block_id")
                for example in examples[:12]
                if example.get("block_id")
            ],
            "common_bbox_band": common_band,
        })

    return sorted(rows, key=lambda row: (-row["occurrence_count"], row["normalized_text"]))


def markdown_table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def render_report(summary: dict[str, Any], results: list[dict[str, Any]], old_dir: Path | None, new_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# v4 Parsed Raw Parser Validation")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("本报告只验证 `parsed_raw_v4` parser 在样本上的效果，不修改 clean/chunk/retrieval/generation/reranker/Milvus/BM25/dataset。")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- new_dir: `{new_dir}`")
    lines.append(f"- old_dir: `{old_dir}`" if old_dir else "- old_dir: not provided")
    lines.append(f"- 本次验证样本数量: {summary['sample_count']}")
    lines.append(f"- 是否有 old/new 对比: {'yes' if summary['has_old_new_compare'] else 'no'}")
    if summary["old_dir_missing_or_unusable"]:
        lines.append("- 说明: old_dir 不存在或不可用，本次只能分析 new_dir，无法计算 old/new 回归差异。")
    lines.append("")
    lines.append("## Overall Conclusion")
    lines.append("")
    lines.append(f"总体结论: **{summary['conclusion']}**")
    lines.append("")
    lines.append("汇总指标:")
    lines.append("")
    lines.append(markdown_table_row(["metric", "value"]))
    lines.append(markdown_table_row(["---", "---"]))
    for key in [
        "total_pages",
        "total_two_column_pages",
        "total_image_block_count",
        "total_figure_caption_candidate_count",
        "total_table_caption_candidate_count",
        "total_journal_preproof_noise_in_page_text_count",
        "total_journal_preproof_noise_in_block_text_count",
        "total_journal_preproof_noise_in_diagnostics_count",
        "total_stripped_noise_line_count",
        "potential_regression_doc_count",
    ]:
        lines.append(markdown_table_row([key, summary[key]]))
    lines.append("")
    lines.append("## Dual Column Detection And Ordering")
    lines.append("")
    lines.append(f"- two_column_pages total: {summary['total_two_column_pages']}")
    lines.append(f"- order_strategy_distribution: `{summary['order_strategy_distribution']}`")
    lines.append(f"- layout_reason_distribution: `{summary['layout_reason_distribution']}`")
    lines.append("")
    lines.append("## Journal Pre-proof Noise")
    lines.append("")
    lines.append(f"- pages[].text residual count: {summary['total_journal_preproof_noise_in_page_text_count']}")
    lines.append(f"- pages[].blocks[].text residual count: {summary['total_journal_preproof_noise_in_block_text_count']}")
    lines.append(f"- diagnostics stripped example count: {summary['total_journal_preproof_noise_in_diagnostics_count']}")
    lines.append(f"- stripped_noise_line_count: {summary['total_stripped_noise_line_count']}")
    lines.append("")
    lines.append("注意: diagnostics 中的 stripped examples 是审计记录，不计为正文残留。")
    lines.append("")
    lines.append("### Residual Examples")
    lines.append("")
    residual_examples = summary.get("residual_examples", [])
    if not residual_examples:
        lines.append("No page_text or block_text residual examples found.")
    else:
        lines.append(markdown_table_row([
            "doc_id",
            "page",
            "field",
            "block_id",
            "matched_pattern",
            "normalized_text",
            "bbox",
            "column",
            "reading_order",
            "size",
            "raw_text_preview",
        ]))
        lines.append(markdown_table_row(["---"] * 11))
        for example in residual_examples[:80]:
            lines.append(markdown_table_row([
                example.get("doc_id"),
                example.get("page"),
                example.get("field"),
                example.get("block_id") or "-",
                example.get("matched_pattern"),
                example.get("normalized_text"),
                example.get("bbox") or "-",
                example.get("column") or "-",
                example.get("reading_order") or "-",
                example.get("size") or "-",
                example.get("raw_text_preview"),
            ]))
        if len(residual_examples) > 80:
            lines.append("")
            lines.append(f"Truncated residual examples to 80 rows from {len(residual_examples)} total. Full list is in JSON output.")
    lines.append("")
    lines.append("### Repeated Block Residuals")
    lines.append("")
    repeated_rows = summary.get("repeated_block_residuals", [])
    if not repeated_rows:
        lines.append("No repeated block_text residual clusters found.")
    else:
        lines.append(markdown_table_row([
            "normalized_text",
            "doc_ids",
            "page_count",
            "occurrence_count",
            "example_block_ids",
            "common_bbox_band",
        ]))
        lines.append(markdown_table_row(["---"] * 6))
        for row in repeated_rows[:40]:
            lines.append(markdown_table_row([
                row.get("normalized_text"),
                ", ".join(row.get("doc_ids", [])),
                row.get("page_count"),
                row.get("occurrence_count"),
                ", ".join(row.get("example_block_ids", [])),
                row.get("common_bbox_band"),
            ]))
    lines.append("")
    lines.append("## Figure/Table Caption Candidates")
    lines.append("")
    lines.append(f"- figure_caption_candidate_count: {summary['total_figure_caption_candidate_count']}")
    lines.append(f"- table_caption_candidate_count: {summary['total_table_caption_candidate_count']}")
    lines.append("")
    lines.append("## Image Blocks")
    lines.append("")
    lines.append(f"- image_block_count: {summary['total_image_block_count']}")
    lines.append("")
    lines.append("## Per-Doc Metrics")
    lines.append("")
    lines.append(markdown_table_row([
        "doc_id",
        "pages",
        "old_chars",
        "new_chars",
        "delta",
        "empty_new",
        "two_col_pages",
        "images",
        "fig",
        "table",
        "noise_text",
        "noise_blocks",
        "risks",
    ]))
    lines.append(markdown_table_row(["---"] * 13))
    for item in results:
        lines.append(markdown_table_row([
            item["doc_id"],
            item["total_pages"],
            item["text_char_count_old"],
            item["text_char_count_new"],
            item["text_char_delta_ratio"],
            item["empty_page_count"],
            len(item["two_column_pages"]),
            item["image_block_count"],
            item["figure_caption_candidate_count"],
            item["table_caption_candidate_count"],
            item["journal_preproof_noise_in_page_text_count"],
            item["journal_preproof_noise_in_block_text_count"],
            ", ".join(item["potential_regressions"]) or "-",
        ]))
    lines.append("")
    lines.append("## Potential Regressions")
    lines.append("")
    regression_rows = [
        item for item in results
        if item["potential_regressions"]
    ]
    if not regression_rows:
        lines.append("No potential regressions flagged by the configured rules.")
    else:
        for item in regression_rows:
            lines.append(f"- `{item['doc_id']}`: {', '.join(item['potential_regressions'])}")
    lines.append("")
    lines.append("## Manual Spot Check List")
    lines.append("")
    review_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any, Any]] = set()
    for item in results:
        for row in item["manual_review_pages"]:
            key = (row.get("doc_id"), row.get("page"), row.get("reason"), row.get("block_id"))
            if key not in seen:
                seen.add(key)
                review_rows.append(row)
    if not review_rows:
        lines.append("No mandatory manual spot checks from validation rules.")
    else:
        lines.append(markdown_table_row(["doc_id", "page", "reason", "block_id"]))
        lines.append(markdown_table_row(["---", "---", "---", "---"]))
        for row in review_rows[:80]:
            lines.append(markdown_table_row([
                row.get("doc_id"),
                row.get("page"),
                row.get("reason"),
                row.get("block_id", "-"),
            ]))
        if len(review_rows) > 80:
            lines.append("")
            lines.append(f"Truncated manual list to 80 rows from {len(review_rows)} total. Full list is in JSON output.")
    lines.append("")
    lines.append("## Phase2 Recommendation")
    lines.append("")
    if summary["conclusion"] == "fail":
        lines.append("不建议直接进入 v4 Phase2。先人工抽检并处理 fail 风险，尤其是正文残留噪声、空页增加或文本量大幅下降。")
    elif summary["conclusion"] == "warning":
        lines.append("可以有条件进入 v4 Phase2，但建议先抽检本报告列出的 doc/page，并确认 warnings 是否为可接受的保守 parser 行为。")
    else:
        lines.append("建议进入 v4 Phase2。当前验证规则未发现明显 parser 回归。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parsed_raw_v4 outputs")
    parser.add_argument("--old_dir", help="Old parsed_raw directory", default=None)
    parser.add_argument("--new_dir", required=True, help="New parsed_raw_v4 directory")
    parser.add_argument("--report", required=True, help="Markdown report path")
    parser.add_argument("--json", dest="json_path", required=True, help="JSON report path")
    args = parser.parse_args()

    old_dir = Path(args.old_dir).resolve() if args.old_dir else None
    new_dir = Path(args.new_dir).resolve()
    report_path = Path(args.report).resolve()
    json_path = Path(args.json_path).resolve()

    new_docs = load_json_dir(new_dir)
    old_docs = load_json_dir(old_dir)
    old_dir_missing = bool(old_dir and not old_docs)
    has_old_compare = bool(old_docs)

    results = [
        analyze_doc(doc_id, new_data, old_docs.get(doc_id))
        for doc_id, new_data in sorted(new_docs.items())
    ]
    summary = summarize(results, has_old_compare, old_dir_missing)
    output = {
        "summary": summary,
        "docs": results,
        "old_dir": str(old_dir) if old_dir else None,
        "new_dir": str(new_dir),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, results, old_dir, new_dir), encoding="utf-8")
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"validated_docs={summary['sample_count']}")
    print(f"conclusion={summary['conclusion']}")
    print(f"report={report_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
