#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate parsed_clean output produced from parsed_raw_v4."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.clean_parsed_structure import (
    collect_repeated_metadata_keys,
    detect_figure_caption,
    detect_table_caption,
    is_front_matter_metadata_line,
    is_journal_preproof_clean_noise,
    is_numbered_reference_entry,
    looks_like_body_sentence_continuation,
    looks_like_front_matter_affiliation_metadata,
    looks_like_marginal_access_banner,
    normalize_pdf_text,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for page in data.get("pages", []) or []
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def text_char_count(data: dict[str, Any]) -> int:
    return sum(len(page.get("text", "") or "") for page in data.get("pages", []) or [])


def _preview(text: str, limit: int = 180) -> str:
    text = normalize_pdf_text(text)
    return text[:limit]


def _block_example(doc_id: str, page: int, block: dict[str, Any], reason: str) -> dict[str, Any]:
    metadata = block.get("metadata", {}) or {}
    return {
        "doc_id": doc_id,
        "page": page,
        "block_id": block.get("block_id"),
        "source_block_id": metadata.get("source_block_id"),
        "current_type": block.get("type"),
        "raw_text_preview": _preview(block.get("text", "")),
        "text_preview": _preview(block.get("text", "")),
        "bbox": metadata.get("bbox"),
        "column": metadata.get("column"),
        "reading_order": metadata.get("reading_order"),
        "matched_reason": reason,
    }


def raw_caption_candidates(raw_data: dict[str, Any]) -> tuple[int, int, dict[str, str]]:
    diagnostics = raw_data.get("diagnostics", {}) or {}
    fig_count = int(diagnostics.get("figure_caption_candidate_count", 0) or 0)
    tbl_count = int(diagnostics.get("table_caption_candidate_count", 0) or 0)
    candidates: dict[str, str] = {}

    scanned_fig = 0
    scanned_tbl = 0
    for block in iter_blocks(raw_data):
        if block.get("type") != "text":
            continue
        text = normalize_pdf_text(block.get("text", ""))
        source_id = block.get("block_id")
        if detect_figure_caption(text):
            scanned_fig += 1
            if source_id:
                candidates[source_id] = "figure_caption"
        elif detect_table_caption(text):
            scanned_tbl += 1
            if source_id:
                candidates[source_id] = "table_caption"

    return max(fig_count, scanned_fig), max(tbl_count, scanned_tbl), candidates


def analyze_doc(doc_id: str, raw_data: dict[str, Any], clean_data: dict[str, Any]) -> dict[str, Any]:
    raw_blocks = iter_blocks(raw_data)
    clean_blocks = iter_blocks(clean_data)
    raw_text_blocks = [block for block in raw_blocks if block.get("type") == "text"]
    type_counts = Counter(block.get("type", "unknown") for block in clean_blocks)
    total_pages = int(clean_data.get("total_pages") or len(clean_data.get("pages", []) or []))

    raw_fig_count, raw_tbl_count, raw_caption_by_source = raw_caption_candidates(raw_data)
    clean_caption_sources = {
        (block.get("metadata", {}) or {}).get("source_block_id"): block.get("type")
        for block in clean_blocks
        if block.get("type") in {"figure_caption", "table_caption"}
    }
    repeated_metadata_keys = collect_repeated_metadata_keys(clean_data.get("pages", []) or [])

    caption_miss_examples: list[dict[str, Any]] = []
    for source_id, expected_type in raw_caption_by_source.items():
        actual_type = clean_caption_sources.get(source_id)
        if actual_type != expected_type and len(caption_miss_examples) < 12:
            raw_block = next((b for b in raw_blocks if b.get("block_id") == source_id), {})
            caption_miss_examples.append({
                "doc_id": doc_id,
                "page": raw_block.get("page"),
                "source_block_id": source_id,
                "expected_type": expected_type,
                "actual_type": actual_type,
                "raw_text_preview": _preview(raw_block.get("text", "")),
            })

    metadata_examples: list[dict[str, Any]] = []
    numbered_ref_examples: list[dict[str, Any]] = []
    text_type_examples: list[dict[str, Any]] = []
    journal_page_examples: list[dict[str, Any]] = []
    journal_block_examples: list[dict[str, Any]] = []
    layout_missing_examples: list[dict[str, Any]] = []
    affiliation_metadata_examples: list[dict[str, Any]] = []
    marginal_banner_examples: list[dict[str, Any]] = []
    false_table_text_body_sentence_examples: list[dict[str, Any]] = []

    metadata_as_paragraph_count = 0
    front_matter_affiliation_as_paragraph_count = 0
    correspondence_as_paragraph_count = 0
    correspondence_as_table_text_count = 0
    marginal_banner_as_paragraph_count = 0
    marginal_banner_as_table_text_count = 0
    marginal_banner_as_references_count = 0
    false_table_text_body_sentence_count = 0
    numbered_reference_as_paragraph_count = 0
    clean_block_journal_count = 0
    metadata_retention_denominator = 0
    metadata_retention_numerator = 0

    for page in clean_data.get("pages", []) or []:
        page_num = int(page.get("page") or 0)
        page_text = page.get("text", "") or ""
        if any(is_journal_preproof_clean_noise(line) for line in page_text.splitlines()):
            if len(journal_page_examples) < 8:
                journal_page_examples.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "raw_text_preview": _preview(page_text),
                    "reason": "journal_preproof_noise_in_page_text",
                })

        for block in page.get("blocks", []) or []:
            block_type = block.get("type", "")
            text = block.get("text", "") or ""
            metadata = block.get("metadata", {}) or {}
            is_affiliation_metadata, affiliation_reason = looks_like_front_matter_affiliation_metadata(
                text, page_num, metadata
            )
            is_marginal_banner, marginal_reason = looks_like_marginal_access_banner(
                text, metadata, repeated_metadata_keys
            )

            if block_type == "text" and len(text_type_examples) < 12:
                text_type_examples.append(_block_example(doc_id, page_num, block, "clean_type_text_residual"))

            if text and is_journal_preproof_clean_noise(text):
                clean_block_journal_count += 1
                if len(journal_block_examples) < 8:
                    journal_block_examples.append(_block_example(doc_id, page_num, block, "journal_preproof_noise_in_block_text"))

            if block_type == "paragraph" and is_front_matter_metadata_line(text, page_num, total_pages):
                metadata_as_paragraph_count += 1
                if len(metadata_examples) < 12:
                    metadata_examples.append(_block_example(doc_id, page_num, block, "front_matter_metadata_as_paragraph"))

            if block_type == "paragraph" and is_affiliation_metadata:
                if affiliation_reason == "correspondence_metadata":
                    correspondence_as_paragraph_count += 1
                else:
                    front_matter_affiliation_as_paragraph_count += 1
                if len(affiliation_metadata_examples) < 12:
                    affiliation_metadata_examples.append(_block_example(doc_id, page_num, block, affiliation_reason))

            if block_type == "table_text" and is_affiliation_metadata and affiliation_reason == "correspondence_metadata":
                correspondence_as_table_text_count += 1
                if len(affiliation_metadata_examples) < 12:
                    affiliation_metadata_examples.append(_block_example(doc_id, page_num, block, affiliation_reason))

            if block_type in {"paragraph", "table_text", "references"} and is_marginal_banner:
                if block_type == "paragraph":
                    marginal_banner_as_paragraph_count += 1
                elif block_type == "table_text":
                    marginal_banner_as_table_text_count += 1
                elif block_type == "references":
                    marginal_banner_as_references_count += 1
                if len(marginal_banner_examples) < 12:
                    marginal_banner_examples.append(_block_example(doc_id, page_num, block, marginal_reason))

            if block_type == "table_text" and looks_like_body_sentence_continuation(text):
                false_table_text_body_sentence_count += 1
                if len(false_table_text_body_sentence_examples) < 12:
                    false_table_text_body_sentence_examples.append(_block_example(doc_id, page_num, block, "body_sentence_continuation_as_table_text"))

            if block_type == "paragraph" and (
                is_numbered_reference_entry(text)
                or re.match(r"^\s*\[\d{1,3}\]\s+[A-Z][A-Za-z]", text)
            ):
                numbered_reference_as_paragraph_count += 1
                if len(numbered_ref_examples) < 12:
                    numbered_ref_examples.append(_block_example(doc_id, page_num, block, "numbered_reference_as_paragraph"))

            if block_type not in {"metadata", "noise", "image"} and raw_text_blocks:
                metadata_retention_denominator += 1
                if (
                    metadata.get("source_block_id")
                    and "bbox" in metadata
                    and "column" in metadata
                    and "reading_order" in metadata
                ):
                    metadata_retention_numerator += 1
                elif len(layout_missing_examples) < 12:
                    layout_missing_examples.append(_block_example(doc_id, page_num, block, "layout_metadata_missing"))

    clean_page_journal_count = len(journal_page_examples)
    raw_chars = text_char_count(raw_data)
    clean_chars = text_char_count(clean_data)
    text_delta = (clean_chars - raw_chars) / raw_chars if raw_chars else 0.0
    caption_raw_total = raw_fig_count + raw_tbl_count
    caption_clean_total = type_counts.get("figure_caption", 0) + type_counts.get("table_caption", 0)
    caption_conversion_rate = caption_clean_total / caption_raw_total if caption_raw_total else 1.0
    retention_rate = (
        metadata_retention_numerator / metadata_retention_denominator
        if metadata_retention_denominator
        else 1.0
    )

    risks: list[str] = []
    if type_counts.get("text", 0) > 0:
        risks.append("clean_type_text_count_gt_0")
    if clean_page_journal_count > 0:
        risks.append("journal_preproof_noise_in_clean_page_text")
    if clean_block_journal_count > 0:
        risks.append("journal_preproof_noise_in_clean_block_text")
    if retention_rate < 0.95:
        risks.append("layout_metadata_retention_rate_lt_0.95")
    if text_delta < -0.3:
        risks.append("text_char_delta_raw_to_clean_lt_-0.3")
    if caption_raw_total >= 5 and caption_conversion_rate < 0.8:
        risks.append("caption_conversion_rate_low")
    if numbered_reference_as_paragraph_count > 0:
        risks.append("numbered_reference_as_paragraph")
    if metadata_as_paragraph_count > 0:
        risks.append("metadata_as_paragraph")
    if front_matter_affiliation_as_paragraph_count > 0:
        risks.append("front_matter_affiliation_as_paragraph")
    if correspondence_as_paragraph_count > 0:
        risks.append("correspondence_as_paragraph")
    if correspondence_as_table_text_count > 0:
        risks.append("correspondence_as_table_text")
    if marginal_banner_as_paragraph_count > 0:
        risks.append("marginal_banner_as_paragraph")
    if marginal_banner_as_table_text_count > 0:
        risks.append("marginal_banner_as_table_text")
    if marginal_banner_as_references_count > 0:
        risks.append("marginal_banner_as_references")
    if false_table_text_body_sentence_count > 0:
        risks.append("false_table_text_body_sentence")

    return {
        "doc_id": doc_id,
        "total_pages": total_pages,
        "raw_block_count": len(raw_blocks),
        "raw_text_block_count": len(raw_text_blocks),
        "clean_block_count": len(clean_blocks),
        "clean_type_text_count": type_counts.get("text", 0),
        "clean_paragraph_count": type_counts.get("paragraph", 0),
        "clean_references_count": type_counts.get("references", 0),
        "clean_metadata_count": type_counts.get("metadata", 0),
        "clean_figure_caption_count": type_counts.get("figure_caption", 0),
        "clean_table_caption_count": type_counts.get("table_caption", 0),
        "clean_table_text_count": type_counts.get("table_text", 0),
        "raw_figure_caption_candidate_count": raw_fig_count,
        "raw_table_caption_candidate_count": raw_tbl_count,
        "raw_to_clean_caption_conversion_rate": round(caption_conversion_rate, 3),
        "layout_metadata_retention_rate": round(retention_rate, 4),
        "journal_preproof_noise_in_clean_page_text_count": clean_page_journal_count,
        "journal_preproof_noise_in_clean_block_text_count": clean_block_journal_count,
        "potential_metadata_as_paragraph_count": metadata_as_paragraph_count,
        "front_matter_affiliation_as_paragraph_count": front_matter_affiliation_as_paragraph_count,
        "correspondence_as_paragraph_count": correspondence_as_paragraph_count,
        "correspondence_as_table_text_count": correspondence_as_table_text_count,
        "marginal_banner_as_paragraph_count": marginal_banner_as_paragraph_count,
        "marginal_banner_as_table_text_count": marginal_banner_as_table_text_count,
        "marginal_banner_as_references_count": marginal_banner_as_references_count,
        "false_table_text_body_sentence_count": false_table_text_body_sentence_count,
        "false_table_text_body_sentence_examples": false_table_text_body_sentence_examples,
        "affiliation_metadata_examples": affiliation_metadata_examples,
        "marginal_banner_examples": marginal_banner_examples,
        "numbered_reference_as_paragraph_count": numbered_reference_as_paragraph_count,
        "text_char_delta_raw_to_clean": round(text_delta, 4),
        "empty_clean_page_count": sum(1 for p in clean_data.get("pages", []) or [] if not (p.get("text", "") or "").strip()),
        "potential_regressions": risks,
        "examples": {
            "type_text_residual": text_type_examples,
            "metadata_as_paragraph": metadata_examples,
            "numbered_reference_as_paragraph": numbered_ref_examples,
            "caption_candidate_not_converted": caption_miss_examples,
            "layout_metadata_missing": layout_missing_examples,
            "journal_page_text": journal_page_examples,
            "journal_block_text": journal_block_examples,
            "affiliation_metadata": affiliation_metadata_examples,
            "marginal_banner": marginal_banner_examples,
            "false_table_text_body_sentence": false_table_text_body_sentence_examples,
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    total_retention_weight = sum(row["clean_block_count"] for row in rows)
    weighted_retention = (
        sum(row["layout_metadata_retention_rate"] * row["clean_block_count"] for row in rows) / total_retention_weight
        if total_retention_weight
        else 1.0
    )
    raw_caption_total = sum(row["raw_figure_caption_candidate_count"] + row["raw_table_caption_candidate_count"] for row in rows)
    clean_caption_total = sum(row["clean_figure_caption_count"] + row["clean_table_caption_count"] for row in rows)
    caption_rate = clean_caption_total / raw_caption_total if raw_caption_total else 1.0
    potential_docs = [row for row in rows if row["potential_regressions"]]
    conclusion = "pass" if not potential_docs else "fail"
    return {
        "sample_doc_count": sample_count,
        "total_pages": sum(row["total_pages"] for row in rows),
        "raw_block_count": sum(row["raw_block_count"] for row in rows),
        "raw_text_block_count": sum(row["raw_text_block_count"] for row in rows),
        "clean_block_count": sum(row["clean_block_count"] for row in rows),
        "clean_type_text_count": sum(row["clean_type_text_count"] for row in rows),
        "clean_paragraph_count": sum(row["clean_paragraph_count"] for row in rows),
        "clean_references_count": sum(row["clean_references_count"] for row in rows),
        "clean_metadata_count": sum(row["clean_metadata_count"] for row in rows),
        "clean_figure_caption_count": sum(row["clean_figure_caption_count"] for row in rows),
        "clean_table_caption_count": sum(row["clean_table_caption_count"] for row in rows),
        "clean_table_text_count": sum(row["clean_table_text_count"] for row in rows),
        "raw_figure_caption_candidate_count": sum(row["raw_figure_caption_candidate_count"] for row in rows),
        "raw_table_caption_candidate_count": sum(row["raw_table_caption_candidate_count"] for row in rows),
        "raw_to_clean_caption_conversion_rate": round(caption_rate, 3),
        "layout_metadata_retention_rate": round(weighted_retention, 4),
        "journal_preproof_noise_in_clean_page_text_count": sum(row["journal_preproof_noise_in_clean_page_text_count"] for row in rows),
        "journal_preproof_noise_in_clean_block_text_count": sum(row["journal_preproof_noise_in_clean_block_text_count"] for row in rows),
        "potential_metadata_as_paragraph_count": sum(row["potential_metadata_as_paragraph_count"] for row in rows),
        "front_matter_affiliation_as_paragraph_count": sum(row["front_matter_affiliation_as_paragraph_count"] for row in rows),
        "correspondence_as_paragraph_count": sum(row["correspondence_as_paragraph_count"] for row in rows),
        "correspondence_as_table_text_count": sum(row["correspondence_as_table_text_count"] for row in rows),
        "marginal_banner_as_paragraph_count": sum(row["marginal_banner_as_paragraph_count"] for row in rows),
        "marginal_banner_as_table_text_count": sum(row["marginal_banner_as_table_text_count"] for row in rows),
        "marginal_banner_as_references_count": sum(row["marginal_banner_as_references_count"] for row in rows),
        "false_table_text_body_sentence_count": sum(row["false_table_text_body_sentence_count"] for row in rows),
        "numbered_reference_as_paragraph_count": sum(row["numbered_reference_as_paragraph_count"] for row in rows),
        "potential_regression_doc_count": len(potential_docs),
        "text_char_delta_raw_to_clean_min": min((row["text_char_delta_raw_to_clean"] for row in rows), default=0.0),
        "empty_clean_page_count": sum(row["empty_clean_page_count"] for row in rows),
        "conclusion": conclusion,
    }


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]], raw_dir: Path, clean_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# v4 Phase2a Parsed Clean Validation")
    lines.append("")
    lines.append(f"- raw_dir: `{raw_dir}`")
    lines.append(f"- clean_dir: `{clean_dir}`")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    lines.append("| doc_id | raw_text_blocks | clean_text | refs | metadata | fig | table | table_text | corr_para | marginal | false_table_sentence | retention | risks |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {doc} | {raw_text} | {clean_text} | {refs} | {meta} | {fig} | {tbl} | {table_text} | {corr_para} | {marginal} | {false_table} | {retention} | {risks} |".format(
                doc=row["doc_id"],
                raw_text=row["raw_text_block_count"],
                clean_text=row["clean_type_text_count"],
                refs=row["clean_references_count"],
                meta=row["clean_metadata_count"],
                fig=row["clean_figure_caption_count"],
                tbl=row["clean_table_caption_count"],
                table_text=row["clean_table_text_count"],
                corr_para=row["correspondence_as_paragraph_count"] + row["correspondence_as_table_text_count"],
                marginal=(
                    row["marginal_banner_as_paragraph_count"]
                    + row["marginal_banner_as_table_text_count"]
                    + row["marginal_banner_as_references_count"]
                ),
                false_table=row["false_table_text_body_sentence_count"],
                retention=row["layout_metadata_retention_rate"],
                risks=", ".join(row["potential_regressions"]) or "-",
            )
        )
    lines.append("")

    lines.append("## Examples")
    lines.append("")
    for label in [
        "type_text_residual",
        "metadata_as_paragraph",
        "numbered_reference_as_paragraph",
        "caption_candidate_not_converted",
        "layout_metadata_missing",
        "journal_page_text",
        "journal_block_text",
        "affiliation_metadata",
        "marginal_banner",
        "false_table_text_body_sentence",
    ]:
        examples = [ex for row in rows for ex in row["examples"].get(label, [])]
        lines.append(f"### {label}")
        if not examples:
            lines.append("- none")
        else:
            for ex in examples[:20]:
                lines.append(f"- `{ex.get('doc_id')}` p{ex.get('page')} `{ex.get('block_id') or ex.get('source_block_id')}`: {ex.get('raw_text_preview')}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parsed_clean output from parsed_raw_v4")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    rows: list[dict[str, Any]] = []
    for clean_path in sorted(clean_dir.glob("*.json")):
        raw_path = raw_dir / clean_path.name
        if not raw_path.exists():
            continue
        clean_data = load_json(clean_path)
        raw_data = load_json(raw_path)
        rows.append(analyze_doc(clean_data.get("doc_id", clean_path.stem), raw_data, clean_data))

    summary = summarize(rows)
    output = {"summary": summary, "docs": rows, "raw_dir": str(raw_dir), "clean_dir": str(clean_dir)}

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, rows, raw_dir, clean_dir), encoding="utf-8")
    print(f"validated_docs={len(rows)}")
    print(f"conclusion={summary['conclusion']}")
    print(f"report={report_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
