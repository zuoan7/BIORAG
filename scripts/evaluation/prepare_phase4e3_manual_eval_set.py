#!/usr/bin/env python3
"""Prepare Phase 4E-3 manual/semi-manual retrieval eval candidates.

This script only reads an existing chunks.jsonl and writes candidate review
artifacts. It does not build indexes, call an LLM, or run retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MARKER_RE = re.compile(r"\[(?:TABLE CAPTION|TABLE TEXT|FIGURE CAPTION)\]\s*")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'._+-]{2,}|\d+(?:\.\d+)?%?")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
FIGURE_NUMBER_RE = re.compile(r"\b(?:fig(?:ure)?\.?)\s*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)
TABLE_NUMBER_RE = re.compile(r"\btable\s*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "been",
    "being",
    "between",
    "caption",
    "cell",
    "cells",
    "characteristic",
    "characteristics",
    "does",
    "during",
    "each",
    "fig",
    "figure",
    "from",
    "into",
    "list",
    "lists",
    "main",
    "paper",
    "present",
    "presents",
    "report",
    "reports",
    "show",
    "shown",
    "shows",
    "study",
    "table",
    "that",
    "the",
    "these",
    "this",
    "used",
    "using",
    "was",
    "were",
    "what",
    "where",
    "which",
    "with",
    "source",
    "sources",
    "genotype",
    "description",
    "descriptions",
}
LOW_INFORMATION_TERMS = STOPWORDS | {
    "continued",
    "data",
    "different",
    "results",
    "supplemental",
    "supplementary",
}

NUMBER_ONLY_CAPTION_PATTERN = re.compile(
    r"^\s*(?:table|fig(?:ure)?\.?)\s+s?[A-Za-z]?\d+[A-Za-z]?[.:|() ]*\s*$",
    re.I,
)
CONTINUED_ONLY_CAPTION_PATTERN = re.compile(
    r"^\s*(?:table|fig(?:ure)?\.?)\s+s?[A-Za-z]?\d+[A-Za-z]?[.:|() ]*"
    r"(?:continued|cont\.?)\s*$",
    re.I,
)
FALSE_TABLE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[TABLE CAPTION\]\s*)?table\s+s?\d+[.:]?\s+(?:the\s+)?[A-Z]\.?\s*$",
    re.I,
)
FALSE_FIGURE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[FIGURE CAPTION\]\s*)?(?:fig(?:ure)?\.?)\s+s?\d+[A-Z]?[.:]?\s*$",
    re.I,
)
GENERIC_TABLE_CAPTION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bstrains?\s+(?:and|,)\s+plasmids?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\bstrains?\s+(?:and|,)\s+plasmids?\s+used\s+in\s+(?:this|the)\s+(?:work|report)\b",
        r"\bstrains?,\s+genes?,\s+and\s+plasmids?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\b(?:main\s+)?strains?\s+used\s+in\s+(?:this|the)\s+(?:study|work|report)\b",
        r"\bstrains?\s+relevant\s+genotype\s+and\s+description\b",
        r"\bprimers?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\b(?:oligonucleotide\s+)?primers?\s+used\s+for\b",
        r"\boligonucleotides?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\bequipment\s+and\s+facilities\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\bsupplementary\s+table\s+s?\d+[.:]?\s*\d?\.?\s*$",
    )
]
GENERIC_ANCHOR_TERMS = {
    "strain",
    "strains",
    "plasmid",
    "plasmids",
    "primer",
    "primers",
    "oligonucleotide",
    "oligonucleotides",
    "relevant",
    "work",
    "report",
    "used",
}
BOILERPLATE_SECTION_RE = re.compile(
    r"\b(references|acknowledg|credit authorship|declaration|conflict of interest)\b",
    re.I,
)


def clean_body(text: Any) -> str:
    body = MARKER_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", body).strip()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                chunks.append(item)
    return chunks


def is_table_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption") or chunk.get("contains_table_text"))


def is_figure_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_figure_caption"))


def is_evidence_chunk(chunk: dict[str, Any]) -> bool:
    return is_table_focused(chunk) or is_figure_focused(chunk)


def is_likely_false_caption(chunk: dict[str, Any]) -> bool:
    body = clean_body(chunk.get("text", ""))
    if chunk.get("contains_table_caption") and FALSE_TABLE_CAPTION_PATTERN.match(body):
        return True
    if chunk.get("contains_figure_caption") and FALSE_FIGURE_CAPTION_PATTERN.match(body):
        return True
    return False


def is_doc0367_figure5_chunk(chunk: dict[str, Any]) -> bool:
    return (
        chunk.get("doc_id") == "doc_0367"
        and is_figure_focused(chunk)
        and re.search(r"\bfig(?:ure)?\.?\s*5\b", clean_body(chunk.get("text", "")), re.I)
        is not None
    )


def word_tokens(text: Any) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(clean_body(text))]


def discriminative_terms(text: Any, max_terms: int = 12) -> list[str]:
    terms: list[str] = []
    for token in TOKEN_RE.findall(clean_body(text)):
        lowered = token.lower().strip(".,;:()[]{}")
        if len(lowered) < 3 or lowered in LOW_INFORMATION_TERMS:
            continue
        if lowered.replace(".", "").isdigit():
            continue
        if lowered not in terms:
            terms.append(lowered)
        if len(terms) >= max_terms:
            break
    return terms


def visible_anchor_terms(text: Any, max_terms: int = 3) -> list[str]:
    candidates: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for index, token in enumerate(TOKEN_RE.findall(clean_body(text))):
        stripped = token.strip(".,;:()[]{}")
        lowered = stripped.lower()
        if len(lowered) < 3 or lowered in LOW_INFORMATION_TERMS:
            continue
        if lowered in GENERIC_ANCHOR_TERMS:
            continue
        if lowered.replace(".", "").isdigit():
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        score = len(stripped)
        if re.search(r"\d", stripped):
            score += 8
        if re.search(r"[-+_/]", stripped):
            score += 5
        if re.search(r"[A-Z]", stripped) and re.search(r"[a-z]", stripped):
            score += 4
        if stripped.isupper() and len(stripped) > 3:
            score += 3
        candidates.append((-score, index, lowered, stripped))
    candidates.sort()
    return [item[3] for item in candidates[:max_terms]]


def caption_exclusion_reasons(chunk: dict[str, Any], sample_type: str) -> list[str]:
    body = clean_body(chunk.get("text", ""))
    reasons: list[str] = []
    if is_likely_false_caption(chunk):
        reasons.append("false_caption")
    if NUMBER_ONLY_CAPTION_PATTERN.match(body) or CONTINUED_ONLY_CAPTION_PATTERN.match(body):
        reasons.append("fragment_caption")
    if len(discriminative_terms(body)) < 3:
        reasons.append("fragment_caption")
    if sample_type == "table" and any(pattern.search(body) for pattern in GENERIC_TABLE_CAPTION_PATTERNS):
        reasons.append("generic_caption")
    if len(word_tokens(body)) < 8:
        reasons.append("fragment_caption")
    return sorted(set(reasons))


def normal_exclusion_reasons(chunk: dict[str, Any]) -> list[str]:
    body = clean_body(chunk.get("text", ""))
    section = str(chunk.get("section") or "")
    reasons: list[str] = []
    if is_evidence_chunk(chunk):
        reasons.append("evidence_chunk")
    if int(chunk.get("token_count") or 0) < 80:
        reasons.append("too_short")
    if chunk.get("contains_references") or BOILERPLATE_SECTION_RE.search(section) or BOILERPLATE_SECTION_RE.search(body[:200]):
        reasons.append("boilerplate_or_references")
    if len(discriminative_terms(body)) < 3:
        reasons.append("insufficient_anchor_terms")
    return sorted(set(reasons))


def longest_common_token_span(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_token in left:
        current = [0] * (len(right) + 1)
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[idx] = previous[idx - 1] + 1
                best = max(best, current[idx])
        previous = current
    return best


def caption_overlap(query: str, caption: str) -> tuple[float, int, str]:
    query_tokens = word_tokens(query)
    caption_tokens = word_tokens(caption)
    if not caption_tokens:
        return 0.0, 0, "low"
    overlap = sum(1 for token in query_tokens if token in set(caption_tokens))
    ratio = overlap / len(caption_tokens)
    span = longest_common_token_span(query_tokens, caption_tokens)
    caption_norm = " ".join(caption_tokens)
    query_norm = " ".join(query_tokens)
    if caption_norm and caption_norm in query_norm:
        return ratio, span, "high"
    if ratio >= 0.75 or span >= 8:
        return ratio, span, "high"
    if ratio >= 0.40 or span >= 4:
        return ratio, span, "medium"
    return ratio, span, "low"


def figure_compare_terms(text: str) -> list[str]:
    match = re.search(r"\bcomparison\s+of\s+(.+?)\s+(?:to|with|and)\s+(.+?)(?:[).,;:]|\s+[A-Z][a-z])", text, re.I)
    if not match:
        match = re.search(r"\bcompare[sd]?\s+(.+?)\s+(?:to|with|and)\s+(.+?)(?:[).,;:]|\s+[A-Z][a-z])", text, re.I)
    if not match:
        return []
    terms = []
    for group in match.groups():
        candidates = visible_anchor_terms(group, max_terms=1)
        if candidates:
            terms.append(candidates[0])
    return terms[:2]


def make_query(sample_type: str, caption: str, anchors: list[str], index: int) -> tuple[str, str]:
    anchor_text = " ".join(anchors[:3])
    if sample_type == "table":
        styles = [
            ("table_where_summarized", f"Where are {anchor_text} summarized in a table?"),
            ("table_reports", f"Which table reports {anchor_text}?"),
            ("table_lists", f"Which table lists {anchor_text}?"),
        ]
    elif sample_type == "figure":
        compare_terms = figure_compare_terms(caption)
        if len(compare_terms) >= 2:
            return (
                f"Which figure compares {compare_terms[0]} with {compare_terms[1]}?",
                "figure_compares",
            )
        styles = [
            ("figure_shows", f"Which figure shows {anchor_text}?"),
            ("figure_presents", f"Which figure presents {anchor_text}?"),
            ("figure_method_or_condition", f"Which figure shows the result for {anchor_text}?"),
        ]
    else:
        styles = [
            ("normal_report", f"What does the study report about {anchor_text}?"),
            ("normal_method_or_pathway", f"Which method or pathway was used for {anchor_text}?"),
            ("normal_optimized", f"What was optimized to improve {anchor_text}?"),
        ]
    style, query = styles[index % len(styles)]
    return query, style


def chunk_score(chunk: dict[str, Any], sample_type: str) -> tuple[int, int, str]:
    body = clean_body(chunk.get("text", ""))
    terms = discriminative_terms(body)
    domain_bonus = sum(1 for term in terms if re.search(r"\d|[-+_/]|[A-Z]", term))
    if sample_type == "normal":
        domain_bonus += 2 if str(chunk.get("doc_id", "")) else 0
    return (len(terms) + domain_bonus, int(chunk.get("token_count") or 0), str(chunk.get("chunk_id", "")))


def sample_base(
    sample_id: str,
    sample_type: str,
    query: str,
    query_style: str,
    chunk: dict[str, Any],
    anchors: list[str],
    caption: str,
    eligibility_reason: str,
) -> dict[str, Any]:
    overlap_ratio, span, copy_risk = caption_overlap(query, caption)
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "query": query,
        "target_doc_id": chunk.get("doc_id", ""),
        "target_chunk_id": chunk.get("chunk_id", ""),
        "target_source_file": chunk.get("source_file", ""),
        "target_page_numbers": chunk.get("page_numbers") or [],
        "target_section": chunk.get("section", ""),
        "target_evidence_types": chunk.get("evidence_types") or [],
        "target_block_types": chunk.get("block_types") or [],
        "target_text_preview": clean_body(chunk.get("text", ""))[:500],
        "target_caption": caption,
        "anchor_terms": anchors,
        "query_style": query_style,
        "caption_token_overlap_ratio": overlap_ratio,
        "longest_common_token_span": span,
        "caption_copy_risk": copy_risk,
        "eligibility_reason": eligibility_reason,
        "review_status": "needs_review",
        "review_notes": "",
    }


def add_candidates(
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    sample_type: str,
    target_count: int,
    max_per_doc: int,
    per_doc_counts: Counter[str],
    excluded: Counter[str],
) -> None:
    candidates = [chunk for chunk in chunks if is_table_focused(chunk)] if sample_type == "table" else [
        chunk for chunk in chunks if is_figure_focused(chunk)
    ]
    candidates = sorted(candidates, key=lambda chunk: chunk_score(chunk, sample_type), reverse=True)
    selected = 0
    for chunk in candidates:
        if selected >= target_count:
            return
        if sample_type == "figure" and is_doc0367_figure5_chunk(chunk):
            continue
        doc_id = str(chunk.get("doc_id", ""))
        if per_doc_counts[doc_id] >= max_per_doc:
            continue
        reasons = caption_exclusion_reasons(chunk, sample_type)
        if reasons:
            for reason in reasons:
                excluded[reason] += 1
            continue
        caption = clean_body(chunk.get("text", ""))
        anchors = visible_anchor_terms(caption, max_terms=3)
        if len(anchors) < 2:
            excluded["insufficient_anchor_terms"] += 1
            continue
        query, query_style = make_query(sample_type, caption, anchors, selected)
        overlap_ratio, span, copy_risk = caption_overlap(query, caption)
        if copy_risk == "high":
            excluded["caption_copy"] += 1
            continue
        selected += 1
        per_doc_counts[doc_id] += 1
        samples.append(
            sample_base(
                f"p4e3_{sample_type}_{selected:04d}",
                sample_type,
                query,
                query_style,
                chunk,
                anchors,
                caption,
                "high-information caption with non-generic deterministic anchor query",
            )
        )


def add_normal_candidates(
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    target_count: int,
    max_per_doc: int,
    per_doc_counts: Counter[str],
    excluded: Counter[str],
) -> None:
    evidence_docs = {str(chunk.get("doc_id", "")) for chunk in chunks if is_evidence_chunk(chunk)}
    paragraph_chunks = [chunk for chunk in chunks if not is_evidence_chunk(chunk)]
    pools = [
        [chunk for chunk in paragraph_chunks if str(chunk.get("doc_id", "")) in evidence_docs],
        [chunk for chunk in paragraph_chunks if str(chunk.get("doc_id", "")) not in evidence_docs],
        paragraph_chunks,
    ]
    selected_ids: set[str] = set()
    selected = 0
    target_first_pool = max(target_count // 2, 1)
    for pool_idx, pool in enumerate(pools):
        pool_target = target_first_pool if pool_idx == 0 else target_count
        ordered = sorted(pool, key=lambda chunk: chunk_score(chunk, "normal"), reverse=True)
        for chunk in ordered:
            if selected >= target_count:
                return
            if pool_idx == 0 and selected >= pool_target:
                break
            chunk_id = str(chunk.get("chunk_id", ""))
            doc_id = str(chunk.get("doc_id", ""))
            if not chunk_id or chunk_id in selected_ids or per_doc_counts[doc_id] >= max_per_doc:
                continue
            reasons = normal_exclusion_reasons(chunk)
            if reasons:
                for reason in reasons:
                    excluded[f"normal_{reason}"] += 1
                continue
            body = clean_body(chunk.get("text", ""))
            anchors = visible_anchor_terms(body, max_terms=3)
            if len(anchors) < 2:
                excluded["normal_insufficient_anchor_terms"] += 1
                continue
            query, query_style = make_query("normal", "", anchors, selected)
            selected += 1
            selected_ids.add(chunk_id)
            per_doc_counts[doc_id] += 1
            samples.append(
                sample_base(
                    f"p4e3_normal_{selected:04d}",
                    "normal",
                    query,
                    query_style,
                    chunk,
                    anchors,
                    "",
                    "paragraph-heavy non-table/figure chunk for normal retrieval takeover control",
                )
            )


def add_doc0367_sanity_anchor(samples: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> int:
    candidates = sorted(
        [chunk for chunk in chunks if is_doc0367_figure5_chunk(chunk)],
        key=lambda chunk: str(chunk.get("chunk_id", "")),
    )
    if not candidates:
        return 0
    chunk = candidates[0]
    caption = clean_body(chunk.get("text", ""))
    anchors = ["Opto-T7RNAPs", "paT7P-148"]
    query = "Which figure compares Opto-T7RNAPs with paT7P-148?"
    samples.append(
        sample_base(
            "p4e3_sanity_anchor_0001",
            "sanity_anchor",
            query,
            "doc_0367_figure5_sanity_anchor",
            chunk,
            anchors,
            caption,
            "required doc_0367 Figure 5 sanity anchor; reported separately from main denominator",
        )
    )
    return 1


def per_doc_distribution(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        doc_id = str(sample.get("target_doc_id", ""))
        sample_type = str(sample.get("sample_type", ""))
        distribution[doc_id][sample_type] += 1
        distribution[doc_id]["total"] += 1
    return {doc_id: dict(counter) for doc_id, counter in sorted(distribution.items())}


def full_caption_exclusion_audit(chunks: list[dict[str, Any]]) -> Counter[str]:
    audit: Counter[str] = Counter()
    for sample_type, predicate in (
        ("table", is_table_focused),
        ("figure", is_figure_focused),
    ):
        for chunk in chunks:
            if not predicate(chunk):
                continue
            reasons = caption_exclusion_reasons(chunk, sample_type)
            for reason in reasons:
                audit[reason] += 1
    return audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review(path: Path, samples: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 4E-3 Candidate Eval Set Review",
        "",
        "This is a candidate set for manual review, not an approved benchmark.",
        "",
        "## Summary",
        "",
        f"- table candidates: {summary['table_candidate_count']}",
        f"- figure candidates: {summary['figure_candidate_count']}",
        f"- normal candidates: {summary['normal_candidate_count']}",
        f"- sanity anchors: {summary['sanity_anchor_count']}",
        f"- high caption-copy risk count: {summary['high_caption_copy_risk_count']}",
        f"- production table_text chunks: {summary['production_table_text_count']}",
        "",
        "## Review Instructions",
        "",
        "- Approve only samples whose query is answerable from the target chunk without relying on exact caption wording.",
        "- Reject false captions, fragment captions, generic table/figure questions, boilerplate normal paragraphs, and full-caption-copy queries.",
        "- After review, write approved rows to `reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl`.",
        "",
    ]
    for sample_type in ("table", "figure", "normal", "sanity_anchor"):
        items = [sample for sample in samples if sample["sample_type"] == sample_type]
        lines.extend([f"## {sample_type}", ""])
        for sample in items:
            lines.extend(
                [
                    f"### {sample['sample_id']} | `{sample['target_chunk_id']}`",
                    "",
                    f"- review_status: `{sample['review_status']}`",
                    f"- query: {sample['query']}",
                    f"- anchors: `{sample['anchor_terms']}`",
                    f"- caption_copy_risk: `{sample['caption_copy_risk']}` overlap={sample['caption_token_overlap_ratio']:.3f} span={sample['longest_common_token_span']}",
                    f"- target_doc_id: `{sample['target_doc_id']}` source: `{sample['target_source_file']}` pages: `{sample['target_page_numbers']}`",
                    f"- evidence_types: `{sample['target_evidence_types']}` block_types: `{sample['target_block_types']}`",
                    f"- target_preview: {sample['target_text_preview']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase 4E-3 Eval Set Candidates",
                "",
                "This directory contains a candidate manual/semi-manual retrieval eval set. It is not an approved benchmark.",
                "",
                "Required review workflow:",
                "",
                "1. Inspect `candidate_eval_set_review.md`.",
                "2. Edit `review_status` outside this candidate artifact, or copy approved rows into a new approved file.",
                "3. Save the approved eval set as `reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl`.",
                "4. Run retrieval-only evaluation only after the approved file exists.",
                "",
                "Scope limitations:",
                "",
                "- Current production parsed_clean chunks have `contains_table_text=0`, so table candidates are primarily caption-only table retrieval samples.",
                "- This set does not represent full table row, table object, OCR, or structured figure/object retrieval.",
                "- Query generation is deterministic and template-based; no Qwen, LLM, reranker, OCR, pdfplumber, or camelot was used.",
                "- `sanity_anchor` samples, including doc_0367 Figure 5, should be reported separately unless a later approved eval explicitly includes them in the main denominator.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_summary(
    chunks: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    excluded: Counter[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    table = [sample for sample in samples if sample["sample_type"] == "table"]
    figure = [sample for sample in samples if sample["sample_type"] == "figure"]
    normal = [sample for sample in samples if sample["sample_type"] == "normal"]
    anchors = [sample for sample in samples if sample["sample_type"] == "sanity_anchor"]
    high_copy = [sample for sample in samples if sample["caption_copy_risk"] == "high"]
    full_exclusion_audit = full_caption_exclusion_audit(chunks)
    return {
        "inputs": {
            "chunks_jsonl": str(args.chunks_jsonl),
            "target_table": args.target_table,
            "target_figure": args.target_figure,
            "target_normal": args.target_normal,
            "max_per_doc": args.max_per_doc,
        },
        "table_candidate_count": len(table),
        "figure_candidate_count": len(figure),
        "normal_candidate_count": len(normal),
        "sanity_anchor_count": len(anchors),
        "excluded_false_caption_count": full_exclusion_audit.get("false_caption", 0),
        "excluded_fragment_caption_count": full_exclusion_audit.get("fragment_caption", 0),
        "excluded_generic_caption_count": full_exclusion_audit.get("generic_caption", 0),
        "excluded_caption_copy_count": excluded.get("caption_copy", 0),
        "full_candidate_excluded_by_reason": dict(sorted(full_exclusion_audit.items())),
        "selection_excluded_by_reason": dict(sorted(excluded.items())),
        "avg_query_caption_overlap_table": (
            sum(sample["caption_token_overlap_ratio"] for sample in table) / len(table) if table else 0.0
        ),
        "avg_query_caption_overlap_figure": (
            sum(sample["caption_token_overlap_ratio"] for sample in figure) / len(figure) if figure else 0.0
        ),
        "high_caption_copy_risk_count": len(high_copy),
        "per_doc_distribution": per_doc_distribution(samples),
        "production_table_text_count": sum(1 for chunk in chunks if chunk.get("contains_table_text")),
        "scope_limitation": (
            "candidate eval set only; production table_text=0, so table eval candidates are "
            "caption-only table retrieval and do not cover table row/object retrieval"
        ),
        "approved_eval_set_path": "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
        "formal_eval_status": "candidate_set_generated_waiting_for_manual_approval",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--target_table", type=int, default=60)
    parser.add_argument("--target_figure", type=int, default=60)
    parser.add_argument("--target_normal", type=int, default=60)
    parser.add_argument("--max_per_doc", type=int, default=3)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    per_doc_counts: Counter[str] = Counter()

    add_candidates(samples, chunks, "table", args.target_table, args.max_per_doc, per_doc_counts, excluded)
    add_candidates(samples, chunks, "figure", args.target_figure, args.max_per_doc, per_doc_counts, excluded)
    add_normal_candidates(samples, chunks, args.target_normal, args.max_per_doc, per_doc_counts, excluded)
    sanity_anchor_count = add_doc0367_sanity_anchor(samples, chunks)
    if sanity_anchor_count:
        excluded["sanity_anchor_added"] += sanity_anchor_count

    summary = build_summary(chunks, samples, excluded, args)
    write_jsonl(args.output_dir / "candidate_eval_set.jsonl", samples)
    (args.output_dir / "candidate_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review(args.output_dir / "candidate_eval_set_review.md", samples, summary)
    write_readme(args.output_dir / "README.md")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "table_candidate_count": summary["table_candidate_count"],
                "figure_candidate_count": summary["figure_candidate_count"],
                "normal_candidate_count": summary["normal_candidate_count"],
                "sanity_anchor_count": summary["sanity_anchor_count"],
                "formal_eval_status": summary["formal_eval_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
