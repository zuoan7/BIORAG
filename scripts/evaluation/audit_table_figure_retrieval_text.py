#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static retrieval_text audit for table/figure focused chunks.

This script does not call Milvus, embeddings, rerankers, OCR, or any retrieval
pipeline. It only reads parsed_clean JSON, chunks.jsonl, and the Phase 4B audit
directory to assess retrieval_text quality and unmatched evidence examples.
"""

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

from scripts.ingestion.audit_table_figure_evidence import (
    EVIDENCE_TYPES,
    build_chunk_index,
    compact_text,
    doc_context,
    evidence_pack_for,
    find_chunks_for_item,
    iter_blocks,
    iter_json_files,
    load_chunks,
    load_json,
    make_sample,
)


FALSE_TABLE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[TABLE CAPTION\]\s*)?"
    r"table\s+s?\d+[.:]?\s+(?:the\s+)?[A-Z]\.?\s*$",
    re.I,
)
FALSE_FIGURE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[FIGURE CAPTION\]\s*)?(?:fig(?:ure)?\.?)\s+s?\d+[A-Z]?[.:]?\s*$",
    re.I,
)
MARKER_PATTERN = re.compile(r"\[(?:TABLE CAPTION|TABLE TEXT|FIGURE CAPTION)\]")
TABLE_EVIDENCE_PREFIX = "[Table Evidence]"
FIGURE_EVIDENCE_PREFIX = "[Figure Evidence]"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def clean_body(text: Any) -> str:
    normalized = compact_text(str(text or ""), 100000)
    normalized = MARKER_PATTERN.sub(" ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def length_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "avg": None}
    ordered = sorted(values)

    def percentile(pct: float) -> int:
        index = round((len(ordered) - 1) * pct)
        return int(ordered[index])

    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": int(ordered[-1]),
        "avg": sum(ordered) / len(ordered),
    }


def alpha_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(1 for char in chars if char.isalpha()) / len(chars)


def is_numeric_only(text: str) -> bool:
    body = clean_body(text)
    if not body:
        return False
    numeric_chars = "[0-9\\s.,;:()/%+\\-\\u2013\\u2014\\u2264\\u2265=<>\\u00b1\\u00d7*/_]+"
    remainder = re.sub(numeric_chars, "", body)
    return not remainder.strip()


def evidence_flags(chunk: dict[str, Any]) -> tuple[bool, bool, bool]:
    return (
        bool(chunk.get("contains_table_caption")),
        bool(chunk.get("contains_table_text")),
        bool(chunk.get("contains_figure_caption")),
    )


def is_table_focused(chunk: dict[str, Any]) -> bool:
    table_caption, table_text, _figure_caption = evidence_flags(chunk)
    return table_caption or table_text


def is_figure_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_figure_caption"))


def is_evidence_chunk(chunk: dict[str, Any]) -> bool:
    return is_table_focused(chunk) or is_figure_focused(chunk)


def has_context_label(retrieval_text: str, label: str) -> bool:
    return re.search(rf"(^|\n){re.escape(label)}\s*:", retrieval_text, re.I) is not None


def body_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", clean_body(text)))


def retrieval_text_token_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def is_parser_false_caption(chunk_or_text: dict[str, Any] | str, evidence_type: str | None = None) -> bool:
    text = chunk_or_text.get("text", "") if isinstance(chunk_or_text, dict) else str(chunk_or_text)
    body = clean_body(text)
    if evidence_type == "figure_caption":
        return bool(FALSE_FIGURE_CAPTION_PATTERN.match(body))
    if evidence_type == "table_caption":
        return bool(FALSE_TABLE_CAPTION_PATTERN.match(body))
    if bool(FALSE_TABLE_CAPTION_PATTERN.match(body)):
        return True
    return bool(FALSE_FIGURE_CAPTION_PATTERN.match(body))


def chunk_quality(chunk: dict[str, Any]) -> dict[str, Any]:
    table_caption, table_text, figure_caption = evidence_flags(chunk)
    retrieval_text = str(chunk.get("retrieval_text", ""))
    text = str(chunk.get("text", ""))
    body = clean_body(text)
    body_words = body_word_count(body)
    retrieval_len = len(retrieval_text)
    text_len = len(text)
    section = str(chunk.get("section", ""))

    reasons = []
    if retrieval_len < 30:
        reasons.append("retrieval_text_lt30")
    if retrieval_len < 50:
        reasons.append("retrieval_text_lt50")
    if section == "Title":
        reasons.append("section_title")
    if section == "Unknown":
        reasons.append("section_unknown")
    if table_caption and "[TABLE CAPTION]" not in retrieval_text:
        reasons.append("missing_table_caption_marker")
    if table_text and "[TABLE TEXT]" not in retrieval_text:
        reasons.append("missing_table_text_marker")
    if figure_caption and "[FIGURE CAPTION]" not in retrieval_text:
        reasons.append("missing_figure_caption_marker")
    if not has_context_label(retrieval_text, "section"):
        reasons.append("missing_section_context")
    if not has_context_label(retrieval_text, "source_file"):
        reasons.append("missing_source_file_context")
    if not has_context_label(retrieval_text, "doc_id"):
        reasons.append("missing_doc_id_context")
    if alpha_ratio(body) < 0.15:
        reasons.append("low_alpha_body")
    if is_numeric_only(body):
        reasons.append("numeric_only_body")
    if body_words <= 3:
        reasons.append("single_word_or_numbering_body")
    if table_caption and not table_text and len(body) < 50:
        reasons.append("caption_only_table_short")
    if figure_caption and len(body) < 50:
        reasons.append("figure_caption_short_fragment")
    if table_caption and is_parser_false_caption(text, "table_caption"):
        reasons.append("parser_false_table_caption")
    if figure_caption and is_parser_false_caption(text, "figure_caption"):
        reasons.append("parser_false_figure_caption")

    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "source_file": chunk.get("source_file"),
        "section": section,
        "page_numbers": chunk.get("page_numbers", []),
        "block_types": chunk.get("block_types", []),
        "evidence_types": chunk.get("evidence_types", []),
        "contains_table_caption": table_caption,
        "contains_table_text": table_text,
        "contains_figure_caption": figure_caption,
        "retrieval_text_len": retrieval_len,
        "text_len": text_len,
        "token_count": chunk.get("token_count"),
        "body_alpha_ratio": alpha_ratio(body),
        "body_word_count": body_words,
        "has_table_caption_marker": "[TABLE CAPTION]" in retrieval_text,
        "has_table_text_marker": "[TABLE TEXT]" in retrieval_text,
        "has_figure_caption_marker": "[FIGURE CAPTION]" in retrieval_text,
        "has_table_evidence_prefix": TABLE_EVIDENCE_PREFIX in retrieval_text,
        "has_figure_evidence_prefix": FIGURE_EVIDENCE_PREFIX in retrieval_text,
        "has_page_context": has_context_label(retrieval_text, "page"),
        "has_section_context": has_context_label(retrieval_text, "section"),
        "has_source_file_context": has_context_label(retrieval_text, "source_file"),
        "has_doc_id_context": has_context_label(retrieval_text, "doc_id"),
        "retrieval_text_token_count": retrieval_text_token_count(retrieval_text),
        "retrieval_text_preview": compact_text(retrieval_text, 420),
        "text_preview": compact_text(text, 360),
        "suspicious_reasons": reasons,
    }


def judge_caption_only_table(quality: dict[str, Any]) -> str:
    reasons = set(quality["suspicious_reasons"])
    if "parser_false_table_caption" in reasons:
        return "parser_false_caption"
    if "caption_only_table_short" in reasons or "single_word_or_numbering_body" in reasons:
        return "too_short_or_fragment"
    if "section_title" in reasons or "section_unknown" in reasons:
        return "needs_context_enrichment"
    return "useful_caption_only"


def judge_title_evidence_chunk(quality: dict[str, Any]) -> dict[str, Any]:
    retrieval_usable = quality["retrieval_text_len"] >= 50 and not {
        "parser_false_table_caption",
        "parser_false_figure_caption",
        "single_word_or_numbering_body",
    }.intersection(quality["suspicious_reasons"])
    return {
        "likely_upstream_section_path_missing": quality["section"] == "Title",
        "retrieval_text_usable": retrieval_usable,
        "needs_phase4d_context_enrichment": quality["section"] in {"Title", "Unknown"},
        "needs_section_metadata_cleanup": quality["section"] in {"Title", "Unknown"},
    }


def classify_unmatched(sample: dict[str, Any]) -> tuple[str, str]:
    evidence_type = str(sample.get("evidence_type", ""))
    text = str(sample.get("text_preview", ""))
    previous_text = str(sample.get("previous_block_preview", ""))
    next_text = str(sample.get("next_block_preview", ""))
    joined = " ".join([text, previous_text, next_text])

    if evidence_type == "table_caption" and is_parser_false_caption(text, "table_caption"):
        return "likely_false_caption", "ignore_as_parser_false_positive"
    if evidence_type == "table_caption" and re.search(r"\b(?:E\.|E\. coli|K\.|K\. phaffii)\b", joined):
        if len(clean_body(text)) < 40:
            return "likely_false_caption", "track_for_parser_cleanup"
    if len(clean_body(text)) < 35:
        return "insufficient_context", "track_for_parser_cleanup"
    if evidence_type == "figure_caption" and len(clean_body(text)) >= 80:
        return "likely_valid_caption_dropped", "fix_chunk_filtering_later"
    if evidence_type == "table_caption" and len(clean_body(text)) >= 80:
        return "likely_valid_caption_dropped", "inspect_pdf_manually"
    return "unknown", "inspect_pdf_manually"


def collect_unmatched(parsed_clean_dir: Path, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_index = build_chunk_index(chunks)
    unmatched = []
    for path in iter_json_files(parsed_clean_dir):
        data = load_json(path)
        doc_id = str(data.get("doc_id", path.stem))
        source_file = str(data.get("source_file", path.name))
        context = doc_context(iter_blocks(data), doc_id)
        evidence_pack = evidence_pack_for(data)
        for unit in evidence_pack.get("evidence_units", []) or []:
            if not isinstance(unit, dict):
                continue
            unit_type = str(unit.get("type", ""))
            if unit_type not in EVIDENCE_TYPES:
                continue
            enriched = dict(unit, doc_id=unit.get("doc_id", doc_id), source_file=unit.get("source_file", source_file))
            if find_chunks_for_item(enriched, chunk_index):
                continue
            sample = make_sample(enriched, context, chunk_index)
            category, recommendation = classify_unmatched(sample)
            sample["judgement_category"] = category
            sample["recommendation"] = recommendation
            sample["requires_special_attention"] = sample.get("doc_id") == "doc_0367"
            unmatched.append(sample)
    return unmatched


def chunk_sample(quality: dict[str, Any], judgement: str | None = None) -> dict[str, Any]:
    item = {
        "chunk_id": quality["chunk_id"],
        "doc_id": quality["doc_id"],
        "source_file": quality["source_file"],
        "section": quality["section"],
        "page_numbers": quality["page_numbers"],
        "block_types": quality["block_types"],
        "evidence_types": quality["evidence_types"],
        "contains_table_caption": quality["contains_table_caption"],
        "contains_table_text": quality["contains_table_text"],
        "contains_figure_caption": quality["contains_figure_caption"],
        "retrieval_text_len": quality["retrieval_text_len"],
        "text_len": quality["text_len"],
        "token_count": quality["token_count"],
        "suspicious_reasons": quality["suspicious_reasons"],
        "retrieval_text_preview": quality["retrieval_text_preview"],
        "text_preview": quality["text_preview"],
    }
    if judgement:
        item["judgement"] = judgement
    return item


def load_phase4b_stats(table_figure_audit_dir: Path) -> dict[str, Any]:
    stats_path = table_figure_audit_dir / "table_figure_stats.json"
    if not stats_path.is_file():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))


def collect_audit(
    chunks_jsonl: Path,
    parsed_clean_dir: Path,
    table_figure_audit_dir: Path,
    sample_per_type: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    chunks = load_chunks(chunks_jsonl)
    phase4b_stats = load_phase4b_stats(table_figure_audit_dir)

    evidence_chunks = [chunk for chunk in chunks if is_evidence_chunk(chunk)]
    table_chunks = [chunk for chunk in chunks if is_table_focused(chunk)]
    figure_chunks = [chunk for chunk in chunks if is_figure_focused(chunk)]
    caption_only_table_chunks = [
        chunk for chunk in table_chunks
        if chunk.get("contains_table_caption") and not chunk.get("contains_table_text")
    ]
    caption_with_table_text_chunks = [
        chunk for chunk in table_chunks
        if chunk.get("contains_table_caption") and chunk.get("contains_table_text")
    ]
    orphan_table_text_chunks = [
        chunk for chunk in table_chunks
        if chunk.get("contains_table_text") and not chunk.get("contains_table_caption")
    ]

    qualities = [chunk_quality(chunk) for chunk in evidence_chunks]
    quality_by_id = {quality["chunk_id"]: quality for quality in qualities}
    reason_counts = Counter(
        reason
        for quality in qualities
        for reason in quality["suspicious_reasons"]
    )
    suspicious = [quality for quality in qualities if quality["suspicious_reasons"]]

    table_qualities = [quality_by_id[chunk.get("chunk_id")] for chunk in table_chunks]
    figure_qualities = [quality_by_id[chunk.get("chunk_id")] for chunk in figure_chunks]
    caption_only_qualities = [quality_by_id[chunk.get("chunk_id")] for chunk in caption_only_table_chunks]
    title_unknown_qualities = [
        quality for quality in qualities
        if quality["section"] in {"Title", "Unknown"}
    ]

    unmatched = collect_unmatched(parsed_clean_dir, chunks)
    unmatched_counts = Counter(item["judgement_category"] for item in unmatched)
    unmatched_recommendations = Counter(item["recommendation"] for item in unmatched)

    caption_judgements = Counter(judge_caption_only_table(quality) for quality in caption_only_qualities)
    title_review = [
        {**chunk_sample(quality), **judge_title_evidence_chunk(quality)}
        for quality in title_unknown_qualities
    ]
    paragraph_chunks_with_evidence_prefix = [
        chunk for chunk in chunks
        if "paragraph" in (chunk.get("block_types", []) or [])
        and (
            TABLE_EVIDENCE_PREFIX in str(chunk.get("retrieval_text", ""))
            or FIGURE_EVIDENCE_PREFIX in str(chunk.get("retrieval_text", ""))
        )
    ]
    retrieval_text_token_counts = [quality["retrieval_text_token_count"] for quality in qualities]

    stats = {
        "inputs": {
            "chunks_jsonl": str(chunks_jsonl),
            "parsed_clean_dir": str(parsed_clean_dir),
            "table_figure_audit_dir": str(table_figure_audit_dir),
            "sample_per_type": sample_per_type,
        },
        "phase4b_baseline": {
            "chunk_count": phase4b_stats.get("chunk_count"),
            "table_figure_integrity": phase4b_stats.get("integrity_checks", {}),
            "table_figure_chunks": phase4b_stats.get("chunks", {}),
        },
        "counts": {
            "total_chunks": len(chunks),
            "table_focused_chunk_count": len(table_chunks),
            "figure_focused_chunk_count": len(figure_chunks),
            "caption_only_table_chunk_count": len(caption_only_table_chunks),
            "caption_with_table_text_chunk_count": len(caption_with_table_text_chunks),
            "orphan_table_text_chunk_count": len(orphan_table_text_chunks),
            "figure_caption_chunk_count": len(figure_chunks),
            "section_title_evidence_chunk_count": sum(1 for quality in qualities if quality["section"] == "Title"),
            "section_unknown_evidence_chunk_count": sum(1 for quality in qualities if quality["section"] == "Unknown"),
            "retrieval_text_lt30_count": sum(1 for quality in qualities if quality["retrieval_text_len"] < 30),
            "retrieval_text_lt50_count": sum(1 for quality in qualities if quality["retrieval_text_len"] < 50),
            "caption_only_table_retrieval_text_lt50_count": sum(
                1 for quality in caption_only_qualities if quality["retrieval_text_len"] < 50
            ),
            "figure_focused_retrieval_text_lt50_count": sum(
                1 for quality in figure_qualities if quality["retrieval_text_len"] < 50
            ),
            "caption_only_table_short_or_fragment_count": sum(
                1 for quality in caption_only_qualities
                if "caption_only_table_short" in quality["suspicious_reasons"]
            ),
            "figure_caption_short_fragment_count": sum(
                1 for quality in figure_qualities
                if "figure_caption_short_fragment" in quality["suspicious_reasons"]
            ),
            "short_caption_fragment_count": sum(
                1 for quality in qualities
                if "caption_only_table_short" in quality["suspicious_reasons"]
                or "figure_caption_short_fragment" in quality["suspicious_reasons"]
                or "single_word_or_numbering_body" in quality["suspicious_reasons"]
            ),
            "low_alpha_chunk_count": sum(1 for quality in qualities if "low_alpha_body" in quality["suspicious_reasons"]),
            "numeric_only_chunk_count": sum(1 for quality in qualities if "numeric_only_body" in quality["suspicious_reasons"]),
            "suspected_false_caption_chunk_count": sum(
                1 for quality in qualities
                if "parser_false_table_caption" in quality["suspicious_reasons"]
                or "parser_false_figure_caption" in quality["suspicious_reasons"]
            ),
            "suspected_false_caption_count": sum(
                1 for quality in qualities
                if "parser_false_table_caption" in quality["suspicious_reasons"]
                or "parser_false_figure_caption" in quality["suspicious_reasons"]
            ),
            "needs_context_enrichment_count": sum(
                1 for quality in qualities
                if (
                    quality["page_numbers"]
                    and not quality["has_page_context"]
                )
                or not quality["has_section_context"]
                or quality["section"] in {"Title", "Unknown"}
            ),
            "paragraph_chunks_with_evidence_prefix_count": len(paragraph_chunks_with_evidence_prefix),
            "suspicious_chunk_count": len(suspicious),
        },
        "context_fields": {
            "has_table_caption_marker_count": sum(1 for quality in qualities if quality["has_table_caption_marker"]),
            "has_table_text_marker_count": sum(1 for quality in qualities if quality["has_table_text_marker"]),
            "has_figure_caption_marker_count": sum(1 for quality in qualities if quality["has_figure_caption_marker"]),
            "has_page_context_count": sum(1 for quality in qualities if quality["has_page_context"]),
            "has_section_context_count": sum(1 for quality in qualities if quality["has_section_context"]),
            "has_source_file_context_count": sum(1 for quality in qualities if quality["has_source_file_context"]),
            "has_doc_id_context_count": sum(1 for quality in qualities if quality["has_doc_id_context"]),
            "has_table_evidence_prefix_count": sum(1 for quality in qualities if quality["has_table_evidence_prefix"]),
            "has_figure_evidence_prefix_count": sum(1 for quality in qualities if quality["has_figure_evidence_prefix"]),
            "paragraph_chunks_with_evidence_prefix_count": len(paragraph_chunks_with_evidence_prefix),
            "retrieval_text_token_count_total": int(sum(retrieval_text_token_counts)),
            "retrieval_text_token_count_avg": (
                sum(retrieval_text_token_counts) / len(retrieval_text_token_counts)
                if retrieval_text_token_counts else None
            ),
        },
        "distributions": {
            "retrieval_text_length": length_summary([quality["retrieval_text_len"] for quality in qualities]),
            "retrieval_text_token_count": length_summary(retrieval_text_token_counts),
            "text_length": length_summary([quality["text_len"] for quality in qualities]),
            "token_count": length_summary([
                int(quality["token_count"])
                for quality in qualities
                if quality["token_count"] is not None and str(quality["token_count"]).isdigit()
            ]),
            "table_retrieval_text_length": length_summary([quality["retrieval_text_len"] for quality in table_qualities]),
            "figure_retrieval_text_length": length_summary([quality["retrieval_text_len"] for quality in figure_qualities]),
        },
        "suspicious_reason_counts": dict(sorted(reason_counts.items())),
        "caption_only_table_judgement_counts": dict(sorted(caption_judgements.items())),
        "unmatched_evidence": {
            "total": len(unmatched),
            "category_counts": dict(sorted(unmatched_counts.items())),
            "recommendation_counts": dict(sorted(unmatched_recommendations.items())),
            "doc_0367_items": [
                item for item in unmatched
                if item.get("doc_id") == "doc_0367"
            ],
        },
    }

    samples = {
        "caption_only_table": [
            chunk_sample(quality, judge_caption_only_table(quality))
            for quality in caption_only_qualities[:sample_per_type]
        ],
        "caption_only_table_suspicious": [
            chunk_sample(quality, judge_caption_only_table(quality))
            for quality in caption_only_qualities
            if judge_caption_only_table(quality) != "useful_caption_only"
        ][:sample_per_type],
        "figure_focused": [
            chunk_sample(quality)
            for quality in figure_qualities[:sample_per_type]
        ],
        "suspicious_chunks": [
            chunk_sample(quality)
            for quality in suspicious[:sample_per_type]
        ],
        "title_unknown_evidence_chunks": title_review[:sample_per_type],
    }
    return stats, samples, unmatched


def render_chunk_item(item: dict[str, Any]) -> list[str]:
    lines = [
        f"- chunk_id: `{item.get('chunk_id', '')}`",
        f"  doc_id: `{item.get('doc_id', '')}`",
        f"  source_file: `{item.get('source_file', '')}`",
        f"  section: `{item.get('section', '')}`",
        f"  page_numbers: `{json.dumps(item.get('page_numbers', []), ensure_ascii=False)}`",
        f"  block_types: `{json.dumps(item.get('block_types', []), ensure_ascii=False)}`",
        f"  evidence_types: `{json.dumps(item.get('evidence_types', []), ensure_ascii=False)}`",
        f"  contains_table_caption: `{item.get('contains_table_caption')}`",
        f"  contains_table_text: `{item.get('contains_table_text')}`",
        f"  contains_figure_caption: `{item.get('contains_figure_caption')}`",
    ]
    if item.get("judgement"):
        lines.append(f"  judgement: `{item.get('judgement')}`")
    if "retrieval_text_usable" in item:
        lines.append(f"  retrieval_text_usable: `{item.get('retrieval_text_usable')}`")
        lines.append(f"  likely_upstream_section_path_missing: `{item.get('likely_upstream_section_path_missing')}`")
        lines.append(f"  needs_phase4d_context_enrichment: `{item.get('needs_phase4d_context_enrichment')}`")
        lines.append(f"  needs_section_metadata_cleanup: `{item.get('needs_section_metadata_cleanup')}`")
    lines.extend([
        f"  lengths: `retrieval={item.get('retrieval_text_len')}, text={item.get('text_len')}, tokens={item.get('token_count')}`",
        f"  suspicious_reasons: `{json.dumps(item.get('suspicious_reasons', []), ensure_ascii=False)}`",
        f"  retrieval_text preview: {item.get('retrieval_text_preview', '')}",
        f"  text preview: {item.get('text_preview', '')}",
        "",
    ])
    return lines


def render_unmatched_item(item: dict[str, Any]) -> list[str]:
    lines = [
        f"- doc_id: `{item.get('doc_id', '')}`",
        f"  source_file: `{item.get('source_file', '')}`",
        f"  page: `{item.get('page', '')}`",
        f"  block_id: `{item.get('block_id', '')}`",
        f"  block_type: `{item.get('block_type', '')}`",
        f"  evidence_type: `{item.get('evidence_type', '')}`",
        f"  section_path: `{json.dumps(item.get('section_path', []), ensure_ascii=False)}`",
        f"  text preview: {item.get('text_preview', '')}",
        f"  previous block preview: {item.get('previous_block_preview', '')}",
        f"  next block preview: {item.get('next_block_preview', '')}",
        f"  judgement_category: `{item.get('judgement_category', '')}`",
        f"  recommendation: `{item.get('recommendation', '')}`",
    ]
    if item.get("requires_special_attention"):
        lines.append("  special_attention: `doc_0367 unmatched figure_caption`")
    lines.append("")
    return lines


def render_samples(stats: dict[str, Any], samples: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Phase 4C Static Retrieval Text Samples",
        "",
        f"- chunks_jsonl: `{stats['inputs']['chunks_jsonl']}`",
        f"- parsed_clean_dir: `{stats['inputs']['parsed_clean_dir']}`",
        "",
    ]
    sections = [
        ("caption_only_table", "Caption-Only Table Chunk Samples"),
        ("caption_only_table_suspicious", "Caption-Only Table Suspicious Samples"),
        ("figure_focused", "Figure-Focused Chunk Samples"),
        ("title_unknown_evidence_chunks", "Section Title/Unknown Evidence Chunks"),
        ("suspicious_chunks", "Suspicious Evidence Chunks"),
    ]
    for key, title in sections:
        lines.extend([f"## {title}", ""])
        if not samples.get(key):
            lines.extend(["No samples.", ""])
            continue
        for item in samples[key]:
            lines.extend(render_chunk_item(item))
    return "\n".join(lines).rstrip() + "\n"


def render_unmatched_review(unmatched: list[dict[str, Any]]) -> str:
    counts = Counter(item["judgement_category"] for item in unmatched)
    recommendations = Counter(item["recommendation"] for item in unmatched)
    lines = [
        "# Phase 4C Unmatched Evidence Review",
        "",
        f"- total_unmatched: {len(unmatched)}",
        f"- category_counts: `{json.dumps(dict(sorted(counts.items())), ensure_ascii=False)}`",
        f"- recommendation_counts: `{json.dumps(dict(sorted(recommendations.items())), ensure_ascii=False)}`",
        "",
        "## Items",
        "",
    ]
    for item in unmatched:
        lines.extend(render_unmatched_item(item))
    return "\n".join(lines).rstrip() + "\n"


def render_summary(stats: dict[str, Any]) -> str:
    counts = stats["counts"]
    unmatched = stats["unmatched_evidence"]
    lines = [
        "# Phase 4C Static Retrieval Text Audit Summary",
        "",
        "## Inputs",
        "",
        f"- chunks_jsonl: `{stats['inputs']['chunks_jsonl']}`",
        f"- parsed_clean_dir: `{stats['inputs']['parsed_clean_dir']}`",
        f"- table_figure_audit_dir: `{stats['inputs']['table_figure_audit_dir']}`",
        "",
        "## Static Retrieval Text",
        "",
        f"- table_focused_chunk_count: {counts['table_focused_chunk_count']}",
        f"- figure_focused_chunk_count: {counts['figure_focused_chunk_count']}",
        f"- caption_only_table_chunk_count: {counts['caption_only_table_chunk_count']}",
        f"- caption_with_table_text_chunk_count: {counts['caption_with_table_text_chunk_count']}",
        f"- orphan_table_text_chunk_count: {counts['orphan_table_text_chunk_count']}",
        f"- figure_caption_chunk_count: {counts['figure_caption_chunk_count']}",
        f"- retrieval_text_lt30_count: {counts['retrieval_text_lt30_count']}",
        f"- retrieval_text_lt50_count: {counts['retrieval_text_lt50_count']}",
        f"- caption_only_table_retrieval_text_lt50_count: {counts['caption_only_table_retrieval_text_lt50_count']}",
        f"- figure_focused_retrieval_text_lt50_count: {counts['figure_focused_retrieval_text_lt50_count']}",
        f"- section_title_evidence_chunk_count: {counts['section_title_evidence_chunk_count']}",
        f"- section_unknown_evidence_chunk_count: {counts['section_unknown_evidence_chunk_count']}",
        f"- suspected_false_caption_chunk_count: {counts['suspected_false_caption_chunk_count']}",
        f"- short_caption_fragment_count: {counts['short_caption_fragment_count']}",
        f"- needs_context_enrichment_count: {counts['needs_context_enrichment_count']}",
        f"- paragraph_chunks_with_evidence_prefix_count: {counts['paragraph_chunks_with_evidence_prefix_count']}",
        f"- suspicious_chunk_count: {counts['suspicious_chunk_count']}",
        "",
        "## Context Fields",
        "",
        f"- has_page_context_count: {stats['context_fields']['has_page_context_count']}",
        f"- has_section_context_count: {stats['context_fields']['has_section_context_count']}",
        f"- has_source_file_context_count: {stats['context_fields']['has_source_file_context_count']}",
        f"- has_doc_id_context_count: {stats['context_fields']['has_doc_id_context_count']}",
        f"- has_table_evidence_prefix_count: {stats['context_fields']['has_table_evidence_prefix_count']}",
        f"- has_figure_evidence_prefix_count: {stats['context_fields']['has_figure_evidence_prefix_count']}",
        f"- paragraph_chunks_with_evidence_prefix_count: {stats['context_fields']['paragraph_chunks_with_evidence_prefix_count']}",
        f"- retrieval_text_token_count_total: {stats['context_fields']['retrieval_text_token_count_total']}",
        f"- retrieval_text_token_count_avg: {stats['context_fields']['retrieval_text_token_count_avg']}",
        "",
        "## Caption-Only Table Judgements",
        "",
        f"- judgement_counts: `{json.dumps(stats['caption_only_table_judgement_counts'], ensure_ascii=False)}`",
        "",
        "## Unmatched Evidence",
        "",
        f"- total: {unmatched['total']}",
        f"- category_counts: `{json.dumps(unmatched['category_counts'], ensure_ascii=False)}`",
        f"- recommendation_counts: `{json.dumps(unmatched['recommendation_counts'], ensure_ascii=False)}`",
        f"- doc_0367_unmatched_count: {len(unmatched['doc_0367_items'])}",
        "",
        "## Distributions",
        "",
        f"- retrieval_text_length: `{json.dumps(stats['distributions']['retrieval_text_length'], ensure_ascii=False)}`",
        f"- retrieval_text_token_count: `{json.dumps(stats['distributions']['retrieval_text_token_count'], ensure_ascii=False)}`",
        f"- text_length: `{json.dumps(stats['distributions']['text_length'], ensure_ascii=False)}`",
        f"- token_count: `{json.dumps(stats['distributions']['token_count'], ensure_ascii=False)}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    output_dir: Path,
    stats: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    unmatched: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_text_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "retrieval_text_samples.md").write_text(
        render_samples(stats, samples),
        encoding="utf-8",
    )
    (output_dir / "unmatched_evidence_review.md").write_text(
        render_unmatched_review(unmatched),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary(stats),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static audit of table/figure focused chunk retrieval_text quality."
    )
    parser.add_argument("--chunks_jsonl", required=True, help="Path to chunks.jsonl.")
    parser.add_argument("--parsed_clean_dir", required=True, help="Directory containing parsed_clean JSON files.")
    parser.add_argument(
        "--table_figure_audit_dir",
        required=True,
        help="Directory containing Phase 4B table/figure audit outputs.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for Phase 4C audit outputs.")
    parser.add_argument("--sample_per_type", type=int, default=30, help="Samples per category.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_jsonl = Path(args.chunks_jsonl).resolve()
    parsed_clean_dir = Path(args.parsed_clean_dir).resolve()
    table_figure_audit_dir = Path(args.table_figure_audit_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not chunks_jsonl.is_file():
        raise SystemExit(f"[ERROR] chunks_jsonl does not exist: {chunks_jsonl}")
    if not parsed_clean_dir.is_dir():
        raise SystemExit(f"[ERROR] parsed_clean_dir does not exist: {parsed_clean_dir}")
    if not table_figure_audit_dir.is_dir():
        raise SystemExit(f"[ERROR] table_figure_audit_dir does not exist: {table_figure_audit_dir}")
    if args.sample_per_type < 0:
        raise SystemExit("[ERROR] sample_per_type must be >= 0")

    stats, samples, unmatched = collect_audit(
        chunks_jsonl=chunks_jsonl,
        parsed_clean_dir=parsed_clean_dir,
        table_figure_audit_dir=table_figure_audit_dir,
        sample_per_type=args.sample_per_type,
    )
    write_outputs(output_dir, stats, samples, unmatched)

    print(f"Wrote {output_dir / 'retrieval_text_stats.json'}")
    print(f"Wrote {output_dir / 'retrieval_text_samples.md'}")
    print(f"Wrote {output_dir / 'unmatched_evidence_review.md'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(
        "Summary: "
        f"table_focused={stats['counts']['table_focused_chunk_count']}, "
        f"figure_focused={stats['counts']['figure_focused_chunk_count']}, "
        f"suspicious={stats['counts']['suspicious_chunk_count']}, "
        f"unmatched={stats['unmatched_evidence']['total']}"
    )


if __name__ == "__main__":
    main()
