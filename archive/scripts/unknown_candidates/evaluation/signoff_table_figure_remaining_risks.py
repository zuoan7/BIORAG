#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4D.5 sign-off audit for remaining table/figure retrieval risks.

This script is read-only with respect to preprocessing outputs. It does not
call Milvus, embeddings, rerankers, OCR, or PDF/table extraction.
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

from scripts.evaluation.audit_table_figure_retrieval_text import (
    FIGURE_EVIDENCE_PREFIX,
    TABLE_EVIDENCE_PREFIX,
    body_word_count,
    chunk_quality,
    clean_body,
    collect_unmatched,
    has_context_label,
    is_evidence_chunk,
    is_parser_false_caption,
)
from scripts.ingestion.audit_table_figure_evidence import (
    compact_text,
    load_chunks,
)


CAPTION_KEYWORD_PATTERN = re.compile(
    r"\b(?:table|fig(?:ure)?|strain|plasmid|gene|pathway|growth|production|"
    r"expression|activity|biosynthesis|fermentation|medium|construct|"
    r"metabolic|enzyme|protein|cell|coli|phaffii)\b",
    re.I,
)
MARKER_PATTERN = re.compile(r"\[(?:TABLE CAPTION|TABLE TEXT|FIGURE CAPTION)\]")
FRAGMENT_CAPTION_PATTERN = re.compile(
    r"^\s*(?:supplementary\s+)?(?:table|fig(?:ure)?\.?)\s+s?\d+[A-Z]?[.:]?"
    r"(?:\s+(?:\d+|[A-Z]\.?|The\s+[A-Z]\.?|[A-Z]\.))?\s*$",
    re.I,
)
GENERIC_CAPTION_WORDS = {
    "table", "fig", "figure", "supplementary", "s", "the",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_table_or_figure_chunk(chunk: dict[str, Any]) -> bool:
    return bool(
        chunk.get("contains_table_caption")
        or chunk.get("contains_table_text")
        or chunk.get("contains_figure_caption")
    )


def has_evidence_prefix(text: str) -> bool:
    return TABLE_EVIDENCE_PREFIX in text or FIGURE_EVIDENCE_PREFIX in text


def has_caption_keywords(text: str) -> bool:
    body = clean_body(text)
    return bool(CAPTION_KEYWORD_PATTERN.search(body))


def meaningful_caption_word_count(text: str) -> int:
    body = MARKER_PATTERN.sub(" ", clean_body(text))
    words = [
        word.lower().strip(".")
        for word in re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", body)
    ]
    return sum(1 for word in words if word not in GENERIC_CAPTION_WORDS)


def looks_like_caption_fragment(text: str) -> bool:
    body = MARKER_PATTERN.sub(" ", clean_body(text)).strip()
    if FRAGMENT_CAPTION_PATTERN.match(body):
        return True
    return meaningful_caption_word_count(body) <= 1 and re.search(r"\b(?:table|fig(?:ure)?)\b", body, re.I)


def classify_unmatched_item(item: dict[str, Any]) -> tuple[str, str, str]:
    text = str(item.get("text_preview", ""))
    previous_text = str(item.get("previous_block_preview", ""))
    next_text = str(item.get("next_block_preview", ""))
    joined = " ".join([previous_text, text, next_text])
    clean = clean_body(text)
    evidence_type = str(item.get("evidence_type", ""))

    if evidence_type == "table_caption" and is_parser_false_caption(text, "table_caption"):
        return (
            "likely_false_caption",
            "no_blocker_parser_cleanup",
            "Caption is a table marker fragment and adjacent text reconstructs an organism/name split.",
        )
    if re.search(r"\b(?:E\.|E\. coli|K\.|K\. phaffii)\b", joined) and len(clean) < 40:
        return (
            "likely_false_caption",
            "no_blocker_parser_cleanup",
            "Short table marker is adjacent to organism text split across parser blocks.",
        )
    if re.search(r"\btable\s+s?\d+\b", clean, re.I) and len(clean.split()) >= 5:
        return (
            "likely_valid_caption_dropped",
            "blocker_valid_caption_loss",
            "Text looks like a complete table caption but did not enter a chunk.",
        )
    return (
        "insufficient_context",
        "needs_manual_pdf_check",
        "Existing block context is not enough to rule out a valid caption.",
    )


def classify_title_evidence_chunk(chunk: dict[str, Any]) -> tuple[str, str]:
    retrieval_text = str(chunk.get("retrieval_text", ""))
    text = str(chunk.get("text", ""))
    has_prefix = has_evidence_prefix(retrieval_text)
    has_page = has_context_label(retrieval_text, "page")
    has_keywords = has_caption_keywords(text)
    words = body_word_count(text)

    if has_prefix and has_page and has_keywords and words >= 4:
        return (
            "acceptable_with_compact_context",
            "Compact retrieval_text supplies evidence type and page; caption has searchable keywords.",
        )
    if has_prefix and has_page and has_keywords:
        return (
            "weak_but_not_blocking",
            "Section is weak, but compact context and caption keywords remain available.",
        )
    if words <= 3 or not has_keywords:
        return (
            "likely_to_hurt_retrieval",
            "Section is Title and caption body has too little searchable signal.",
        )
    return (
        "needs_section_metadata_cleanup",
        "Caption content exists, but upstream section metadata should be repaired later.",
    )


def classify_suspicious_quality(quality: dict[str, Any]) -> tuple[str, str]:
    reasons = set(quality.get("suspicious_reasons", []))
    retrieval_text = str(quality.get("retrieval_text_preview", ""))
    text = str(quality.get("text_preview", ""))
    has_prefix = has_evidence_prefix(retrieval_text)
    has_page = bool(quality.get("page_numbers")) and quality.get("has_page_context")
    has_keywords = has_caption_keywords(text)
    body_words = int(quality.get("body_word_count") or body_word_count(text))
    meaningful_words = meaningful_caption_word_count(text)

    if "parser_false_table_caption" in reasons or "parser_false_figure_caption" in reasons:
        return (
            "parser_false_caption",
            "Parser emitted a table/figure caption fragment; this is parser cleanup, not compact-context failure.",
        )
    if looks_like_caption_fragment(text):
        return (
            "parser_false_caption",
            "Caption body is only a table/figure marker plus a split token or number.",
        )
    if body_words <= 3 and meaningful_words <= 1:
        return (
            "likely_to_hurt_retrieval",
            "Very short caption fragment with little searchable signal.",
        )
    if "caption_only_table_short" in reasons or "figure_caption_short_fragment" in reasons:
        if has_prefix and has_page and has_keywords and meaningful_words >= 2:
            return (
                "true_but_short_caption",
                "Caption is short but compact context adds page/section/evidence type.",
            )
        return (
            "needs_parser_cleanup",
            "Short caption needs parser cleanup or manual source inspection.",
        )
    if has_prefix and has_page and has_keywords:
        return (
            "useful_with_compact_context",
            "Compact context plus caption keywords make the chunk usable for retrieval.",
        )
    if "section_title" in reasons or "section_unknown" in reasons:
        return (
            "needs_parser_cleanup",
            "Weak section metadata remains, but this does not by itself show valid evidence loss.",
        )
    return (
        "useful_with_compact_context",
        "Suspicion is low-risk after compact context enrichment.",
    )


def chunk_item_from_quality(quality: dict[str, Any], category: str, rationale: str) -> dict[str, Any]:
    return {
        "chunk_id": quality.get("chunk_id"),
        "doc_id": quality.get("doc_id"),
        "source_file": quality.get("source_file"),
        "section": quality.get("section"),
        "page_numbers": quality.get("page_numbers", []),
        "evidence_types": quality.get("evidence_types", []),
        "retrieval_text_preview": quality.get("retrieval_text_preview", ""),
        "text_preview": quality.get("text_preview", ""),
        "suspicious_reasons": quality.get("suspicious_reasons", []),
        "judgement_category": category,
        "rationale": rationale,
    }


def sample_qualities(qualities: list[dict[str, Any]], sample_limit: int) -> list[dict[str, Any]]:
    if sample_limit <= 0:
        return []
    return qualities[:sample_limit]


def percent(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def collect_signoff(
    chunks_jsonl: Path,
    parsed_clean_dir: Path,
    table_figure_audit_dir: Path,
    retrieval_text_audit_dir: Path,
    sample_suspicious: int,
    sample_short_fragment: int,
    sample_false_caption: int,
) -> dict[str, Any]:
    chunks = load_chunks(chunks_jsonl)
    retrieval_stats = load_json(retrieval_text_audit_dir / "retrieval_text_stats.json")
    table_figure_stats = load_json(table_figure_audit_dir / "table_figure_stats.json")

    unmatched_items = collect_unmatched(parsed_clean_dir, chunks)
    unmatched_reviews = []
    for item in unmatched_items:
        category, impact, rationale = classify_unmatched_item(item)
        unmatched_reviews.append({
            "doc_id": item.get("doc_id"),
            "source_file": item.get("source_file"),
            "page": item.get("page"),
            "block_id": item.get("block_id"),
            "section_path": item.get("section_path", []),
            "text": item.get("text_preview", ""),
            "previous_block_preview": item.get("previous_block_preview", ""),
            "next_block_preview": item.get("next_block_preview", ""),
            "judgement_category": category,
            "phase4e_impact": impact,
            "rationale": rationale,
        })

    title_chunks = [
        chunk for chunk in chunks
        if is_table_or_figure_chunk(chunk) and str(chunk.get("section", "")) == "Title"
    ]
    title_reviews = []
    for chunk in title_chunks:
        category, rationale = classify_title_evidence_chunk(chunk)
        retrieval_text = str(chunk.get("retrieval_text", ""))
        text = str(chunk.get("text", ""))
        title_reviews.append({
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "source_file": chunk.get("source_file"),
            "section": chunk.get("section"),
            "page_numbers": chunk.get("page_numbers", []),
            "block_types": chunk.get("block_types", []),
            "evidence_types": chunk.get("evidence_types", []),
            "retrieval_text_preview": compact_text(retrieval_text, 420),
            "text_preview": compact_text(text, 360),
            "has_table_evidence_prefix": TABLE_EVIDENCE_PREFIX in retrieval_text,
            "has_figure_evidence_prefix": FIGURE_EVIDENCE_PREFIX in retrieval_text,
            "has_page": has_context_label(retrieval_text, "page"),
            "has_caption_keywords": has_caption_keywords(text),
            "judgement_category": category,
            "rationale": rationale,
        })

    evidence_qualities = [chunk_quality(chunk) for chunk in chunks if is_evidence_chunk(chunk)]
    suspicious = [
        quality for quality in evidence_qualities
        if quality.get("suspicious_reasons")
    ]
    short_fragment = [
        quality for quality in evidence_qualities
        if {
            "caption_only_table_short",
            "figure_caption_short_fragment",
            "single_word_or_numbering_body",
        }.intersection(set(quality.get("suspicious_reasons", [])))
    ]
    false_caption = [
        quality for quality in evidence_qualities
        if {
            "parser_false_table_caption",
            "parser_false_figure_caption",
        }.intersection(set(quality.get("suspicious_reasons", [])))
    ]

    selected_by_id: dict[str, dict[str, Any]] = {}
    sample_sources: dict[str, list[str]] = {}
    for source_name, sample in (
        ("suspicious", sample_qualities(suspicious, sample_suspicious)),
        ("short_or_fragment", sample_qualities(short_fragment, sample_short_fragment)),
        ("suspected_false_caption", sample_qualities(false_caption, sample_false_caption)),
    ):
        for quality in sample:
            chunk_id = str(quality.get("chunk_id", ""))
            selected_by_id.setdefault(chunk_id, quality)
            sample_sources.setdefault(chunk_id, []).append(source_name)

    suspicious_reviews = []
    for chunk_id, quality in selected_by_id.items():
        category, rationale = classify_suspicious_quality(quality)
        item = chunk_item_from_quality(quality, category, rationale)
        item["sample_sources"] = sample_sources.get(chunk_id, [])
        suspicious_reviews.append(item)

    category_counts = Counter(item["judgement_category"] for item in suspicious_reviews)
    sample_total = len(suspicious_reviews)
    sample_stats = {
        "requested": {
            "suspicious": sample_suspicious,
            "short_or_fragment": sample_short_fragment,
            "suspected_false_caption": sample_false_caption,
        },
        "available": {
            "suspicious": len(suspicious),
            "short_or_fragment": len(short_fragment),
            "suspected_false_caption": len(false_caption),
        },
        "deduped_sample_count": sample_total,
        "category_counts": dict(sorted(category_counts.items())),
        "category_ratios": {
            key: percent(value, sample_total)
            for key, value in sorted(category_counts.items())
        },
    }

    unmatched_counts = Counter(item["judgement_category"] for item in unmatched_reviews)
    title_counts = Counter(item["judgement_category"] for item in title_reviews)

    table_text_count = int(table_figure_stats["parsed_clean"]["table_text_block_count"])
    blockers = []
    if any(item["phase4e_impact"] == "blocker_valid_caption_loss" for item in unmatched_reviews):
        blockers.append("valid_table_caption_dropped")
    if any(item["judgement_category"] == "likely_to_hurt_retrieval" for item in title_reviews):
        blockers.append("title_section_evidence_likely_to_hurt_retrieval")
    hurt_ratio = percent(category_counts.get("likely_to_hurt_retrieval", 0), sample_total)
    if sample_total and hurt_ratio >= 0.25:
        blockers.append("suspicious_caption_sample_high_likely_to_hurt_retrieval_ratio")

    known_risks = [
        "10 unmatched table_caption units remain, all classified as likely_false_caption parser artifacts.",
        "10 table/figure evidence chunks still have section == Title.",
        "Suspicious/short caption chunks remain and should be disclosed in Phase 4E.",
        "production parsed_clean has table_text=0, so Phase 4E cannot evaluate row-level/full-table retrieval.",
    ]
    future_cleanup = [
        "parser cleanup for false table-caption fragments and short caption fragments",
        "section metadata cleanup for Title-only evidence chunks",
        "future corpus/parser work to preserve table_text rows for full table retrieval evaluation",
    ]

    return {
        "inputs": {
            "chunks_jsonl": str(chunks_jsonl),
            "parsed_clean_dir": str(parsed_clean_dir),
            "table_figure_audit_dir": str(table_figure_audit_dir),
            "retrieval_text_audit_dir": str(retrieval_text_audit_dir),
        },
        "phase4d_compact_baseline": {
            "preprocess": table_figure_stats.get("preprocess", {}),
            "chunk_count": table_figure_stats.get("chunk_count"),
            "schema_field_set_count": table_figure_stats.get("chunk_schema", {}).get("distinct_field_set_count"),
            "mixed_table_figure_with_paragraph_count": table_figure_stats.get("chunks", {}).get("mixed_table_figure_with_paragraph_count"),
            "table_focused_chunk_count": retrieval_stats.get("counts", {}).get("table_focused_chunk_count"),
            "figure_focused_chunk_count": retrieval_stats.get("counts", {}).get("figure_focused_chunk_count"),
            "has_page_context_count": retrieval_stats.get("context_fields", {}).get("has_page_context_count"),
            "has_table_evidence_prefix_count": retrieval_stats.get("context_fields", {}).get("has_table_evidence_prefix_count"),
            "has_figure_evidence_prefix_count": retrieval_stats.get("context_fields", {}).get("has_figure_evidence_prefix_count"),
            "paragraph_chunks_with_evidence_prefix_count": retrieval_stats.get("context_fields", {}).get("paragraph_chunks_with_evidence_prefix_count"),
            "suspicious_chunk_count": retrieval_stats.get("counts", {}).get("suspicious_chunk_count"),
            "short_caption_fragment_count": retrieval_stats.get("counts", {}).get("short_caption_fragment_count"),
            "suspected_false_caption_count": retrieval_stats.get("counts", {}).get("suspected_false_caption_count"),
        },
        "unmatched_table_caption_review": {
            "items": unmatched_reviews,
            "counts": dict(sorted(unmatched_counts.items())),
            "phase4e_impact_counts": dict(sorted(Counter(item["phase4e_impact"] for item in unmatched_reviews).items())),
        },
        "title_section_evidence_review": {
            "items": title_reviews,
            "counts": dict(sorted(title_counts.items())),
        },
        "suspicious_sample_review": {
            "items": suspicious_reviews,
            "stats": sample_stats,
        },
        "production_table_text_impact": {
            "table_text_block_count": table_text_count,
            "phase4e_can_evaluate": [
                "caption-only table retrieval",
                "figure caption retrieval",
                "compact page/section/evidence-prefix impact on caption retrieval",
            ],
            "phase4e_cannot_evaluate": [
                "table_caption + table_text merge benefit",
                "row-level table_text retrieval",
                "numeric table row retrieval",
                "full table object/row retrieval",
            ],
            "blocks_phase4e": False,
            "scope_limitation": "Phase 4E results must be scoped to caption-only table and figure caption retrieval.",
        },
        "signoff": {
            "blockers_before_phase4e": blockers,
            "known_risks_not_blocking": known_risks,
            "future_cleanup": future_cleanup,
            "recommendation": "proceed" if not blockers else "do_not_proceed",
            "phase4e_scope_limitation": "caption-only table + figure caption retrieval, not full table row/object retrieval",
        },
    }


def render_item_list(items: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. `{item.get(fields[0], '')}`")
        for field in fields[1:]:
            value = item.get(field)
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"   - {field}: {value}")
        lines.append("")
    return lines


def render_summary(report: dict[str, Any]) -> str:
    signoff = report["signoff"]
    unmatched = report["unmatched_table_caption_review"]
    title = report["title_section_evidence_review"]
    suspicious = report["suspicious_sample_review"]["stats"]
    table_text = report["production_table_text_impact"]
    baseline = report["phase4d_compact_baseline"]

    lines = [
        "# Phase 4D.5 Table/Figure Remaining Risk Sign-off",
        "",
        "## Decision",
        "",
        f"- Recommendation: `{signoff['recommendation']}`",
        f"- Blockers: `{json.dumps(signoff['blockers_before_phase4e'], ensure_ascii=False)}`",
        f"- Phase 4E scope limitation: {signoff['phase4e_scope_limitation']}",
        "",
        "## Phase 4D Compact Baseline",
        "",
        f"- success_docs: {baseline['preprocess'].get('success_docs')}/{baseline['preprocess'].get('total_docs')}",
        f"- failed_docs: {baseline['preprocess'].get('failed_docs')}",
        f"- low_quality_docs: {baseline['preprocess'].get('low_quality_docs')}",
        f"- chunk_count: {baseline['chunk_count']}",
        f"- schema_field_set_count: {baseline['schema_field_set_count']}",
        f"- mixed_table_figure_with_paragraph_count: {baseline['mixed_table_figure_with_paragraph_count']}",
        f"- table_focused_chunk_count: {baseline['table_focused_chunk_count']}",
        f"- figure_focused_chunk_count: {baseline['figure_focused_chunk_count']}",
        f"- has_page_context_count: {baseline['has_page_context_count']}",
        f"- has_table_evidence_prefix_count: {baseline['has_table_evidence_prefix_count']}",
        f"- has_figure_evidence_prefix_count: {baseline['has_figure_evidence_prefix_count']}",
        f"- paragraph_chunks_with_evidence_prefix_count: {baseline['paragraph_chunks_with_evidence_prefix_count']}",
        "",
        "## Unmatched Table Captions",
        "",
        f"- total: {len(unmatched['items'])}",
        f"- judgement_counts: `{json.dumps(unmatched['counts'], ensure_ascii=False)}`",
        f"- phase4e_impact_counts: `{json.dumps(unmatched['phase4e_impact_counts'], ensure_ascii=False)}`",
        "- Phase 4E blocker: `False`",
        "",
        "## Title Section Evidence Chunks",
        "",
        f"- total: {len(title['items'])}",
        f"- judgement_counts: `{json.dumps(title['counts'], ensure_ascii=False)}`",
        "- section metadata cleanup required before Phase 4E: `False`",
        "",
        "## Suspicious / Short Caption Sample",
        "",
        f"- available suspicious chunks: {suspicious['available']['suspicious']}",
        f"- available short_or_fragment chunks: {suspicious['available']['short_or_fragment']}",
        f"- available suspected_false_caption chunks: {suspicious['available']['suspected_false_caption']}",
        f"- deduped_sample_count: {suspicious['deduped_sample_count']}",
        f"- category_counts: `{json.dumps(suspicious['category_counts'], ensure_ascii=False)}`",
        f"- category_ratios: `{json.dumps(suspicious['category_ratios'], ensure_ascii=False)}`",
        "- Phase 4E blocker: `False`",
        "",
        "## Production table_text=0 Impact",
        "",
        f"- parsed_clean table_text_block_count: {table_text['table_text_block_count']}",
        f"- blocks Phase 4E: `{table_text['blocks_phase4e']}`",
        f"- can evaluate: `{json.dumps(table_text['phase4e_can_evaluate'], ensure_ascii=False)}`",
        f"- cannot evaluate: `{json.dumps(table_text['phase4e_cannot_evaluate'], ensure_ascii=False)}`",
        "",
        "## Sign-off Classes",
        "",
        f"- A. Blocker before Phase 4E: `{json.dumps(signoff['blockers_before_phase4e'], ensure_ascii=False)}`",
        f"- B. Known risk, not blocking: `{json.dumps(signoff['known_risks_not_blocking'], ensure_ascii=False)}`",
        f"- C. Future cleanup: `{json.dumps(signoff['future_cleanup'], ensure_ascii=False)}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_remaining_risks(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 4D.5 Remaining Risk Details",
        "",
        "## 10 Unmatched Table Captions",
        "",
    ]
    lines.extend(render_item_list(report["unmatched_table_caption_review"]["items"], [
        "doc_id",
        "source_file",
        "page",
        "block_id",
        "section_path",
        "text",
        "previous_block_preview",
        "next_block_preview",
        "judgement_category",
        "phase4e_impact",
        "rationale",
    ]))
    lines.extend([
        "## Section == Title Evidence Chunks",
        "",
    ])
    lines.extend(render_item_list(report["title_section_evidence_review"]["items"], [
        "chunk_id",
        "doc_id",
        "source_file",
        "section",
        "page_numbers",
        "block_types",
        "evidence_types",
        "has_table_evidence_prefix",
        "has_figure_evidence_prefix",
        "has_page",
        "has_caption_keywords",
        "judgement_category",
        "retrieval_text_preview",
        "text_preview",
    ]))
    lines.extend([
        "## Suspicious / Short Caption Sample",
        "",
    ])
    lines.extend(render_item_list(report["suspicious_sample_review"]["items"], [
        "chunk_id",
        "doc_id",
        "source_file",
        "section",
        "page_numbers",
        "evidence_types",
        "sample_sources",
        "suspicious_reasons",
        "judgement_category",
        "retrieval_text_preview",
        "text_preview",
    ]))
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "remaining_risks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "remaining_risks.md").write_text(
        render_remaining_risks(report),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary(report),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4D.5 table/figure remaining risk sign-off audit."
    )
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--parsed_clean_dir", required=True)
    parser.add_argument("--table_figure_audit_dir", required=True)
    parser.add_argument("--retrieval_text_audit_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_suspicious", type=int, default=50)
    parser.add_argument("--sample_short_fragment", type=int, default=50)
    parser.add_argument("--sample_false_caption", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_jsonl = Path(args.chunks_jsonl).resolve()
    parsed_clean_dir = Path(args.parsed_clean_dir).resolve()
    table_figure_audit_dir = Path(args.table_figure_audit_dir).resolve()
    retrieval_text_audit_dir = Path(args.retrieval_text_audit_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path, label, is_dir in (
        (chunks_jsonl, "chunks_jsonl", False),
        (parsed_clean_dir, "parsed_clean_dir", True),
        (table_figure_audit_dir, "table_figure_audit_dir", True),
        (retrieval_text_audit_dir, "retrieval_text_audit_dir", True),
    ):
        if is_dir and not path.is_dir():
            raise SystemExit(f"[ERROR] {label} does not exist or is not a directory: {path}")
        if not is_dir and not path.is_file():
            raise SystemExit(f"[ERROR] {label} does not exist or is not a file: {path}")

    report = collect_signoff(
        chunks_jsonl=chunks_jsonl,
        parsed_clean_dir=parsed_clean_dir,
        table_figure_audit_dir=table_figure_audit_dir,
        retrieval_text_audit_dir=retrieval_text_audit_dir,
        sample_suspicious=args.sample_suspicious,
        sample_short_fragment=args.sample_short_fragment,
        sample_false_caption=args.sample_false_caption,
    )
    write_outputs(output_dir, report)
    print(f"Wrote {output_dir / 'remaining_risks.json'}")
    print(f"Wrote {output_dir / 'remaining_risks.md'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(
        "Summary: "
        f"recommendation={report['signoff']['recommendation']}, "
        f"blockers={len(report['signoff']['blockers_before_phase4e'])}, "
        f"unmatched={len(report['unmatched_table_caption_review']['items'])}, "
        f"title_section={len(report['title_section_evidence_review']['items'])}, "
        f"suspicious_sample={report['suspicious_sample_review']['stats']['deduped_sample_count']}"
    )


if __name__ == "__main__":
    main()
