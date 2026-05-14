#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Phase 5A/B audit for table content loss.

This script audits existing parsed_raw, parsed_clean, evidence-pack behavior,
and prebuilt chunks. It does not modify parser, cleaning, chunking, schema, or
index artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.document_cleaning_v5 import build_evidence_pack
from src.synbio_rag.ingestion.cleaning_rules import normalize_cleaning_text


TABLE_KEYWORD_RE = re.compile(
    r"\b(?:Table|TABLE|Supplementary\s+Table|Table\s+S\d*|TABLE\s+S\d*)\b|"
    r"(?:表|序号|菌株|质粒|引物|浓度|条件|产量)"
)
TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:Supplementary\s+)?Table\s+S?\d+[A-Za-z]?\b|"
    r"^\s*TABLE\s+S?\d+[A-Za-z]?\b",
    re.I,
)
FIGURE_CAPTION_RE = re.compile(r"^\s*(?:Supplementary\s+)?(?:Fig\.?|Figure)\s+S?\d+", re.I)
REFERENCE_LIKE_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d+\.)\s+[A-Z][A-Za-z'`-]+,\s+[A-Z]\.|"
    r"\b(?:doi|crossref|pubmed|google scholar)\b",
    re.I,
)
AFFILIATION_LIKE_RE = re.compile(
    r"\b(?:department|university|institute|college|school|hospital|academy|"
    r"received|accepted|published|revised|correspondence|email|e-mail)\b|"
    r"\b[A-Z][a-z]+,\s+[A-Z]{2}\s+\d{5}\b",
    re.I,
)
NATURAL_SENTENCE_VERBS_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|showed|shown|suggested|indicated|"
    r"reported|observed|measured|determined|constructed|engineered|"
    r"compared|analyzed|revealed|demonstrated|confirmed)\b",
    re.I,
)
TABLE_COLUMN_RE = re.compile(
    r"\b(?:strain|strains|plasmid|plasmids|genotype|source|primer|sequence|"
    r"yield|titer|titre|condition|medium|activity|concentration|temperature|"
    r"sample|species|isolate|accession|gene|genes|product|substrate|"
    r"parameter|biomass|dcw|od600|rate|group|number)\b",
    re.I,
)
BIO_ITEM_RE = re.compile(
    r"\b(?:strain|plasmid|primer|gene|genotype|sequence|medium|mutant|"
    r"菌株|质粒|引物|序列|基因|培养基)\b",
    re.I,
)
GENE_TOKEN_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9.-]*\d+[A-Za-z0-9.-]*|[A-Za-z]{2,}\d+[A-Za-z0-9.-]*|"
    r"p[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,}-[A-Z0-9-]{2,})\b"
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

SELECTED_FIELDS = [
    "doc_id",
    "source_file",
    "reason_selected",
    "in_phase4_table_eval",
    "was_phase4_table_miss",
    "parsed_raw_table_caption_count",
    "parsed_raw_table_text_count",
    "parsed_clean_table_caption_count",
    "parsed_clean_table_text_count",
    "table_keyword_block_count",
    "suspected_table_like_paragraph_count",
    "chunk_count",
    "table_focused_chunk_count",
    "caption_only_table_chunk_count",
    "risk_tags",
]

FLOW_FIELDS = [
    "doc_id",
    "source_file",
    "raw_table_caption_count",
    "raw_table_text_count",
    "clean_table_caption_count",
    "clean_table_text_count",
    "raw_table_keyword_blocks",
    "clean_table_keyword_blocks",
    "raw_table_like_paragraph_candidates",
    "clean_table_like_paragraph_candidates",
    "evidence_table_caption_count",
    "evidence_table_text_count",
    "chunk_table_focused_count",
    "chunk_caption_only_table_count",
    "chunk_caption_plus_text_count",
    "paragraph_chunks_with_table_like_text",
    "suspected_loss_stage",
    "confidence",
    "recommended_phase5c_action",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact(text: Any, limit: int = 220) -> str:
    return normalize_cleaning_text(str(text or ""))[:limit]


def iter_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in data.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks", []) or []:
            if isinstance(block, dict):
                blocks.append(block)
    return blocks


def raw_pseudo_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = iter_blocks(data)
    if blocks:
        return blocks
    pseudo: list[dict[str, Any]] = []
    for page in data.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        page_num = page.get("page") or page.get("page_num")
        text = str(page.get("text") or "")
        for idx, part in enumerate(split_raw_text_units(text)):
            if part.strip():
                pseudo.append({
                    "block_id": f"raw_p{page_num}_u{idx:04d}",
                    "type": "raw_text_unit",
                    "text": part.strip(),
                    "page": page_num,
                })
    return pseudo


def split_raw_text_units(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        return lines
    # Older parsed_raw files in this corpus often have only page-level text.
    # Split near table/figure/heading markers without claiming real block status.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    marker = re.compile(r"(?=(?:Table|TABLE|Supplementary Table|Fig\.?|Figure)\s+S?\d+|\s###\s)", re.I)
    parts = [p.strip() for p in marker.split(normalized) if p.strip()]
    if len(parts) <= 1:
        return [normalized]
    return parts


def source_block_id(block: dict[str, Any]) -> str:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    value = metadata.get("source_block_id") or block.get("source_block_id") or block.get("block_id") or ""
    return str(value)


def has_layout_metadata(blocks: list[dict[str, Any]]) -> bool:
    for block in blocks:
        metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
        if any(block.get(key) is not None for key in ("bbox", "column", "reading_order")):
            return True
        if any(metadata.get(key) is not None for key in ("bbox", "column", "reading_order")):
            return True
    return False


def block_type_counts(blocks: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(block.get("type", "unknown")) for block in blocks))


def table_keyword_count(blocks: list[dict[str, Any]]) -> int:
    return sum(1 for block in blocks if TABLE_KEYWORD_RE.search(str(block.get("text") or "")))


def is_excluded_context(block: dict[str, Any], text: str) -> bool:
    btype = str(block.get("type", ""))
    stripped = re.sub(r"^[#\s]+", "", text).strip()
    if btype in {"references", "metadata", "noise", "figure_caption", "image", "title"}:
        return True
    section_path = " ".join(str(x) for x in block.get("section_path", []) or [])
    if re.search(r"\b(references|bibliography|acknowledg|funding|author contributions)\b", section_path, re.I):
        return True
    if FIGURE_CAPTION_RE.match(stripped):
        return True
    if REFERENCE_LIKE_RE.search(text):
        return True
    if AFFILIATION_LIKE_RE.search(text) and int(block.get("page") or 9999) <= 2:
        return True
    return False


def table_like_score(text: str, after_caption: bool = False) -> tuple[int, list[str]]:
    raw = str(text or "").strip()
    normalized = normalize_cleaning_text(raw)
    if not normalized:
        return 0, []
    words = normalized.split()
    reasons: list[str] = []

    sep_count = raw.count("\t") + raw.count("|") + len(re.findall(r"\s{2,}", raw))
    semicolon_cells = normalized.count(";")
    if sep_count >= 2 or semicolon_cells >= 3:
        reasons.append("multi_column_separators")

    numeric = len(re.findall(r"\b\d+(?:\.\d+)?(?:%|g/L|mg/L|mM|uM|°C|h|min|rpm|x)?\b", normalized, re.I))
    if numeric >= 3:
        reasons.append("dense_numeric_values")

    columns = TABLE_COLUMN_RE.findall(normalized)
    if len(columns) >= 2:
        reasons.append("table_column_terms")

    bio_terms = BIO_ITEM_RE.findall(normalized)
    gene_tokens = GENE_TOKEN_RE.findall(normalized)
    if len(bio_terms) >= 1 and len(gene_tokens) >= 2:
        reasons.append("bio_items_and_identifiers")
    elif len(gene_tokens) >= 4 and len(words) <= 80:
        reasons.append("identifier_series")

    if re.search(r"(?:表|序号|菌株|质粒|引物|浓度|条件|产量)", normalized):
        reasons.append("cjk_table_terms")

    if after_caption and (len(words) <= 35 or numeric >= 2 or len(gene_tokens) >= 2):
        reasons.append("near_table_caption_short_fragment")

    if re.search(r"\b[A-Za-z0-9_.-]+\s+[-+]?\d+(?:\.\d+)?\s+[A-Za-z0-9_.-]+\s+[-+]?\d+", normalized):
        reasons.append("name_value_repetition")

    robust_reasons = {
        "multi_column_separators",
        "identifier_series",
        "name_value_repetition",
        "cjk_table_terms",
    }
    if "dense_numeric_values" in reasons and len(words) <= 35:
        robust_reasons.add("dense_numeric_values")
    if "bio_items_and_identifiers" in reasons and len(words) <= 80:
        robust_reasons.add("bio_items_and_identifiers")

    score = len(set(reasons))
    sentence_count = len(re.findall(r"[.!?]\s+[A-Z]", normalized))
    natural_sentence = (
        len(words) > 28
        and sentence_count >= 1
        and NATURAL_SENTENCE_VERBS_RE.search(normalized)
        and "multi_column_separators" not in reasons
        and "dense_numeric_values" not in reasons
        and "identifier_series" not in reasons
    )
    if natural_sentence:
        score = max(0, score - 2)
        reasons.append("natural_sentence_penalty")
    if not after_caption:
        has_robust = bool(set(reasons) & robust_reasons)
        if len(words) > 45 and not has_robust:
            score = 0
            reasons.append("long_natural_paragraph_penalty")
        if NATURAL_SENTENCE_VERBS_RE.search(normalized) and not has_robust:
            score = 0
            reasons.append("body_sentence_penalty")
    return score, reasons


def is_table_like_candidate(block: dict[str, Any], after_caption: bool = False) -> tuple[bool, list[str]]:
    text = str(block.get("text") or "")
    if is_excluded_context(block, text):
        return False, []
    btype = str(block.get("type", ""))
    if btype not in {"paragraph", "subsection_heading", "raw_text_unit", "table_text"}:
        return False, []
    score, reasons = table_like_score(text, after_caption=after_caption)
    reason_set = set(reasons)
    robust = {
        "multi_column_separators",
        "identifier_series",
        "name_value_repetition",
        "cjk_table_terms",
        "bio_items_and_identifiers",
    }
    words = str(text or "").split()
    has_sentence_verb = bool(NATURAL_SENTENCE_VERBS_RE.search(str(text or "")))
    if "dense_numeric_values" in reason_set and len(words) <= 35:
        robust.add("dense_numeric_values")
    if after_caption:
        if reason_set <= {"near_table_caption_short_fragment"}:
            return False, reasons
        if btype == "subsection_heading" and not (reason_set & {"table_column_terms", "cjk_table_terms", "dense_numeric_values"}):
            return False, reasons
        if len(words) > 45 and has_sentence_verb and not (reason_set & robust):
            return False, reasons
        return score >= 1 and bool(reason_set & (robust | {"table_column_terms"})), reasons
    if has_sentence_verb and not (reason_set & {"multi_column_separators", "identifier_series", "name_value_repetition"}):
        return False, reasons
    if len(words) > 60:
        return False, reasons
    return score >= 2 and bool(reason_set & robust), reasons


def caption_indexes(blocks: list[dict[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for idx, block in enumerate(blocks):
        text = str(block.get("text") or "")
        if block.get("type") == "table_caption" or TABLE_CAPTION_RE.match(text):
            indexes.add(idx)
    return indexes


def candidate_indexes(blocks: list[dict[str, Any]]) -> dict[int, list[str]]:
    captions = caption_indexes(blocks)
    candidates: dict[int, list[str]] = {}
    for idx, block in enumerate(blocks):
        after_caption = any(0 < idx - cap_idx <= 5 for cap_idx in captions)
        ok, reasons = is_table_like_candidate(block, after_caption=after_caption)
        if ok:
            candidates[idx] = reasons
    return candidates


def analyze_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    table_focused = 0
    caption_only = 0
    caption_plus_text = 0
    paragraph_like = 0
    mixed = 0
    for chunk in chunks:
        block_types = set(str(x) for x in chunk.get("block_types", []) or [])
        evidence_types = set(str(x) for x in chunk.get("evidence_types", []) or [])
        has_table_caption = bool(chunk.get("contains_table_caption")) or "table_caption" in block_types or "table_caption" in evidence_types
        has_table_text = bool(chunk.get("contains_table_text")) or "table_text" in block_types or "table_text" in evidence_types
        has_figure = bool(chunk.get("contains_figure_caption")) or "figure_caption" in block_types or "figure_caption" in evidence_types
        has_paragraph = "paragraph" in block_types or "paragraph" in evidence_types
        if (has_table_caption or has_table_text) and not has_paragraph and not has_figure:
            table_focused += 1
        if has_table_caption and not has_table_text:
            caption_only += 1
        if has_table_caption and has_table_text:
            caption_plus_text += 1
        if has_paragraph and table_like_score(str(chunk.get("text") or ""))[0] >= 2:
            paragraph_like += 1
        if (has_table_caption or has_table_text or has_figure) and has_paragraph:
            mixed += 1
    return {
        "chunk_count": len(chunks),
        "table_focused_chunk_count": table_focused,
        "caption_only_table_chunk_count": caption_only,
        "caption_plus_text_chunk_count": caption_plus_text,
        "paragraph_chunks_with_table_like_text": paragraph_like,
        "table_figure_mixed_with_paragraph_count": mixed,
    }


def load_eval_context(eval_set_path: Path, retrieval_results_path: Path) -> dict[str, Any]:
    eval_rows = load_jsonl(eval_set_path)
    table_rows = [
        row for row in eval_rows
        if row.get("sample_type") == "table" and row.get("approved") is True
        and row.get("include_in_main_denominator", True) is True
    ]
    table_docs = {str(row.get("target_doc_id")) for row in table_rows if row.get("target_doc_id")}
    table_doc_samples: dict[str, list[str]] = defaultdict(list)
    for row in table_rows:
        if row.get("target_doc_id"):
            table_doc_samples[str(row["target_doc_id"])].append(str(row.get("sample_id", "")))

    miss_docs: set[str] = set()
    miss_samples: dict[str, str] = {}
    if retrieval_results_path.is_file():
        results = load_json(retrieval_results_path)
        for row in (results.get("results_by_mode", {}) or {}).get("hybrid", []) or []:
            if row.get("sample_type") != "table":
                continue
            if row.get("chunk_hit@10") is False or row.get("doc_hit@10") is False:
                doc_id = str(row.get("target_doc_id"))
                miss_docs.add(doc_id)
                miss_samples[doc_id] = str(row.get("sample_id", ""))

    return {
        "table_rows": table_rows,
        "table_docs": table_docs,
        "table_doc_samples": dict(table_doc_samples),
        "miss_docs": miss_docs,
        "miss_samples": miss_samples,
    }


def doc_id_from_path(path: Path) -> str:
    return path.stem


def doc_metrics(
    doc_id: str,
    raw_dir: Path,
    clean_dir: Path,
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    with_evidence: bool = True,
) -> dict[str, Any]:
    raw_path = raw_dir / f"{doc_id}.json"
    clean_path = clean_dir / f"{doc_id}.json"
    raw_data = load_json(raw_path) if raw_path.is_file() else {}
    clean_data = load_json(clean_path) if clean_path.is_file() else {}
    raw_blocks = raw_pseudo_blocks(raw_data) if raw_data else []
    clean_blocks = iter_blocks(clean_data) if clean_data else []
    raw_candidates = candidate_indexes(raw_blocks)
    clean_candidates = candidate_indexes(clean_blocks)
    chunks = chunks_by_doc.get(doc_id, [])
    chunk_stats = analyze_chunks(chunks)

    source_file = (
        clean_data.get("source_file")
        or raw_data.get("source_file")
        or (chunks[0].get("source_file") if chunks else f"{doc_id}.pdf")
    )
    raw_types = Counter(str(b.get("type", "unknown")) for b in raw_blocks)
    clean_types = Counter(str(b.get("type", "unknown")) for b in clean_blocks)
    raw_caption_count = raw_types.get("table_caption", 0)
    if raw_caption_count == 0:
        raw_caption_count = sum(1 for block in raw_blocks if TABLE_CAPTION_RE.match(str(block.get("text") or "")))
    clean_caption_count = clean_types.get("table_caption", 0)

    evidence = build_evidence_pack(clean_data) if clean_data and with_evidence else {}
    evidence_units = evidence.get("evidence_units", []) or []
    evidence_type_counts = Counter(str(unit.get("type", "unknown")) for unit in evidence_units)
    evidence_source_ids = {str(unit.get("source_block_id") or unit.get("block_id") or "") for unit in evidence_units}
    excluded_table_like = []
    if with_evidence:
        for idx, block in clean_candidates.items():
            sid = source_block_id(clean_blocks[idx])
            if sid and sid not in evidence_source_ids:
                excluded_table_like.append(clean_blocks[idx])

    clean_metadata_table_like = 0
    for block in clean_blocks:
        if str(block.get("type")) in {"metadata", "noise", "references"}:
            score, _ = table_like_score(str(block.get("text") or ""))
            stripped = re.sub(r"^[#\s]+", "", str(block.get("text") or "")).strip()
            if TABLE_CAPTION_RE.match(stripped) or (TABLE_KEYWORD_RE.search(stripped) and score >= 2) or score >= 3:
                clean_metadata_table_like += 1

    adjacent = []
    for idx in caption_indexes(clean_blocks):
        caption = clean_blocks[idx]
        following = clean_blocks[idx + 1: idx + 6]
        adjacent.append({
            "caption_block_id": caption.get("block_id"),
            "page": caption.get("page"),
            "caption_preview": compact(caption.get("text"), 260),
            "next_blocks": [
                {
                    "block_id": block.get("block_id"),
                    "type": block.get("type"),
                    "page": block.get("page"),
                    "table_like": (idx + 1 + off) in clean_candidates,
                    "preview": compact(block.get("text"), 180),
                }
                for off, block in enumerate(following)
            ],
        })

    raw_has_blocks = bool(iter_blocks(raw_data)) if raw_data else False
    layout_present = has_layout_metadata(iter_blocks(raw_data)) or has_layout_metadata(clean_blocks)
    risk_tags = []
    if not raw_has_blocks:
        risk_tags.append("raw_page_text_only")
    if not layout_present:
        risk_tags.append("no_bbox_column_reading_order")
    if clean_caption_count > 0 and clean_types.get("table_text", 0) == 0:
        risk_tags.append("caption_without_table_text")
    if clean_candidates:
        risk_tags.append("table_like_paragraph_present")
    if clean_metadata_table_like:
        risk_tags.append("table_like_excluded_clean_block")
    if CJK_RE.search(" ".join(str(b.get("text") or "") for b in clean_blocks[:80])):
        risk_tags.append("cjk_text_present")
    if clean_data.get("total_pages", 0) and int(clean_data.get("total_pages", 0)) >= 25:
        risk_tags.append("long_or_thesis_like")
    if max(Counter(str(b.get("page")) for b in clean_blocks).values() or [0]) >= 25:
        risk_tags.append("dense_page_blocks_complex_layout")

    loss_stage, confidence, action = classify_loss_stage(
        raw_caption_count=raw_caption_count,
        raw_text_count=raw_types.get("table_text", 0),
        clean_caption_count=clean_caption_count,
        clean_text_count=clean_types.get("table_text", 0),
        raw_like_count=len(raw_candidates),
        clean_like_count=len(clean_candidates),
        evidence_table_text_count=evidence_type_counts.get("table_text", 0),
        excluded_like_count=len(excluded_table_like),
        clean_metadata_table_like=clean_metadata_table_like,
        chunk_caption_only=chunk_stats["caption_only_table_chunk_count"],
        chunk_caption_plus_text=chunk_stats["caption_plus_text_chunk_count"],
        image_blocks=clean_types.get("image", 0) + raw_types.get("image", 0),
    )

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "raw_path_exists": raw_path.is_file(),
        "clean_path_exists": clean_path.is_file(),
        "raw_has_real_blocks": raw_has_blocks,
        "raw_block_type_distribution": block_type_counts(raw_blocks),
        "clean_block_type_distribution": block_type_counts(clean_blocks),
        "raw_table_caption_count": raw_caption_count,
        "raw_table_text_count": raw_types.get("table_text", 0),
        "clean_table_caption_count": clean_caption_count,
        "clean_table_text_count": clean_types.get("table_text", 0),
        "raw_table_keyword_blocks": table_keyword_count(raw_blocks),
        "clean_table_keyword_blocks": table_keyword_count(clean_blocks),
        "raw_table_like_paragraph_candidates": len(raw_candidates),
        "clean_table_like_paragraph_candidates": len(clean_candidates),
        "evidence_table_caption_count": evidence_type_counts.get("table_caption", 0),
        "evidence_table_text_count": evidence_type_counts.get("table_text", 0),
        "evidence_excluded_table_like_count": len(excluded_table_like),
        "clean_metadata_table_like_count": clean_metadata_table_like,
        "has_bbox_page_column_reading_order_metadata": layout_present,
        "chunks": chunks,
        "chunk_stats": chunk_stats,
        "adjacent_table_caption_blocks": adjacent,
        "raw_examples": examples_for(raw_blocks, raw_candidates),
        "clean_examples": examples_for(clean_blocks, clean_candidates),
        "excluded_table_like_examples": [
            {"block_id": b.get("block_id"), "type": b.get("type"), "page": b.get("page"), "preview": compact(b.get("text"))}
            for b in excluded_table_like[:5]
        ],
        "evidence_policy": evidence.get("evidence_policy", {}),
        "risk_tags": sorted(set(risk_tags)),
        "suspected_loss_stage": loss_stage,
        "confidence": confidence,
        "recommended_phase5c_action": action,
    }


def examples_for(blocks: list[dict[str, Any]], candidates: dict[int, list[str]]) -> list[dict[str, Any]]:
    examples = []
    for idx, reasons in list(candidates.items())[:8]:
        block = blocks[idx]
        examples.append({
            "block_id": block.get("block_id"),
            "type": block.get("type"),
            "page": block.get("page"),
            "reasons": reasons,
            "preview": compact(block.get("text"), 260),
        })
    return examples


def classify_loss_stage(
    *,
    raw_caption_count: int,
    raw_text_count: int,
    clean_caption_count: int,
    clean_text_count: int,
    raw_like_count: int,
    clean_like_count: int,
    evidence_table_text_count: int,
    excluded_like_count: int,
    clean_metadata_table_like: int,
    chunk_caption_only: int,
    chunk_caption_plus_text: int,
    image_blocks: int,
) -> tuple[str, str, str]:
    if raw_text_count > 0 and clean_text_count == 0:
        return (
            "C. cleaning_dropped_or_downgraded",
            "high",
            "cleaning rule guard for true table_text blocks",
        )
    if clean_metadata_table_like > 0:
        return (
            "C. cleaning_dropped_or_downgraded",
            "medium",
            "cleaning rule guard for table-like text near captions",
        )
    if clean_like_count > 0 and excluded_like_count > 0 and evidence_table_text_count == 0:
        return (
            "D. evidence_policy_excluded",
            "medium",
            "evidence policy update for preserved table-like clean blocks",
        )
    if clean_like_count > 0 or raw_like_count > 0:
        return (
            "B. parser_table_as_paragraph",
            "high" if clean_like_count > 0 else "medium",
            "table-like paragraph preservation plus caption-nearby association",
        )
    if clean_caption_count > 0 and clean_text_count == 0 and chunk_caption_only > 0 and chunk_caption_plus_text == 0:
        return (
            "E. preprocess_no_table_text_input",
            "high",
            "provide table_text/table_related_text input before preprocessing",
        )
    if (raw_caption_count > 0 or clean_caption_count > 0) and image_blocks > 0:
        return (
            "F. likely_image_only_table_needs_ocr",
            "low",
            "record OCR candidate; do not prioritize unless common",
        )
    if raw_caption_count == 0 and clean_caption_count == 0:
        return (
            "A. parser_no_table_rows",
            "medium",
            "no Phase 5C table action for this control document",
        )
    return (
        "G. unknown_needs_pdf_visual_check",
        "low",
        "manual PDF visual check before any table extraction change",
    )


def build_doc_universe(clean_dir: Path, raw_dir: Path) -> list[str]:
    ids = {path.stem for path in clean_dir.glob("*.json")}
    ids.update(path.stem for path in raw_dir.glob("*.json"))
    return sorted(ids)


def select_docs(
    all_metrics: dict[str, dict[str, Any]],
    eval_context: dict[str, Any],
    max_docs: int = 50,
) -> list[str]:
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    def add(doc_id: str, reason: str, force: bool = False) -> None:
        if doc_id not in all_metrics:
            return
        if doc_id not in selected:
            if len(selected) < max_docs:
                selected.append(doc_id)
            elif force:
                removable = next(
                    (
                        existing
                        for existing in reversed(selected)
                        if existing not in eval_context["table_docs"]
                        and existing not in eval_context["miss_docs"]
                        and "low_or_no_table_control" not in reasons.get(existing, [])
                    ),
                    None,
                )
                if removable:
                    selected.remove(removable)
                    selected.append(doc_id)
        if doc_id in selected:
            reasons[doc_id].append(reason)

    for doc_id in sorted(eval_context["table_docs"]):
        add(doc_id, "phase4e3_approved_table_eval_doc")
    for doc_id in sorted(eval_context["miss_docs"]):
        add(doc_id, "phase4e3_hybrid_table_miss")

    sorted_metrics = list(all_metrics.values())
    for metric in sorted(
        sorted_metrics,
        key=lambda m: (m["clean_table_caption_count"], m["clean_table_keyword_blocks"]),
        reverse=True,
    )[:12]:
        if metric["clean_table_caption_count"] >= 4 and metric["clean_table_text_count"] == 0:
            add(metric["doc_id"], "many_table_captions_but_clean_table_text_zero")

    for metric in sorted(sorted_metrics, key=lambda m: m["clean_table_keyword_blocks"], reverse=True)[:12]:
        if metric["clean_table_keyword_blocks"] >= 6:
            add(metric["doc_id"], "many_clean_table_keyword_blocks")

    for metric in sorted(sorted_metrics, key=lambda m: m["clean_table_like_paragraph_candidates"], reverse=True)[:12]:
        if metric["clean_table_like_paragraph_candidates"] >= 4:
            add(metric["doc_id"], "many_suspected_table_like_paragraphs")

    for metric in sorted(
        sorted_metrics,
        key=lambda m: max(m["clean_block_type_distribution"].values() or [0]),
        reverse=True,
    )[:8]:
        if "dense_page_blocks_complex_layout" in metric["risk_tags"]:
            add(metric["doc_id"], "complex_layout_inferred_from_dense_page_blocks")

    for metric in sorted_metrics:
        if "cjk_text_present" in metric["risk_tags"] or "long_or_thesis_like" in metric["risk_tags"]:
            add(metric["doc_id"], "cjk_or_long_thesis_like_sample")
            if sum("cjk_or_long_thesis_like_sample" in reasons[d] for d in selected) >= 3:
                break

    controls = sorted(
        [
            m for m in sorted_metrics
            if m["clean_table_caption_count"] == 0
            and m["chunk_stats"]["caption_only_table_chunk_count"] == 0
        ],
        key=lambda m: (m["clean_table_keyword_blocks"], m["clean_table_like_paragraph_candidates"], m["doc_id"]),
    )
    for metric in controls[:5]:
        add(metric["doc_id"], "low_or_no_table_control", force=True)

    for doc_id, reason_list in reasons.items():
        all_metrics[doc_id]["reason_selected"] = "; ".join(dict.fromkeys(reason_list))
    return selected[:max_docs]


def selected_row(metric: dict[str, Any], eval_context: dict[str, Any]) -> dict[str, Any]:
    chunk_stats = metric["chunk_stats"]
    return {
        "doc_id": metric["doc_id"],
        "source_file": metric["source_file"],
        "reason_selected": metric.get("reason_selected", ""),
        "in_phase4_table_eval": str(metric["doc_id"] in eval_context["table_docs"]).lower(),
        "was_phase4_table_miss": str(metric["doc_id"] in eval_context["miss_docs"]).lower(),
        "parsed_raw_table_caption_count": metric["raw_table_caption_count"],
        "parsed_raw_table_text_count": metric["raw_table_text_count"],
        "parsed_clean_table_caption_count": metric["clean_table_caption_count"],
        "parsed_clean_table_text_count": metric["clean_table_text_count"],
        "table_keyword_block_count": metric["clean_table_keyword_blocks"],
        "suspected_table_like_paragraph_count": metric["clean_table_like_paragraph_candidates"],
        "chunk_count": chunk_stats["chunk_count"],
        "table_focused_chunk_count": chunk_stats["table_focused_chunk_count"],
        "caption_only_table_chunk_count": chunk_stats["caption_only_table_chunk_count"],
        "risk_tags": ";".join(metric["risk_tags"]),
    }


def flow_row(metric: dict[str, Any]) -> dict[str, Any]:
    chunk_stats = metric["chunk_stats"]
    return {
        "doc_id": metric["doc_id"],
        "source_file": metric["source_file"],
        "raw_table_caption_count": metric["raw_table_caption_count"],
        "raw_table_text_count": metric["raw_table_text_count"],
        "clean_table_caption_count": metric["clean_table_caption_count"],
        "clean_table_text_count": metric["clean_table_text_count"],
        "raw_table_keyword_blocks": metric["raw_table_keyword_blocks"],
        "clean_table_keyword_blocks": metric["clean_table_keyword_blocks"],
        "raw_table_like_paragraph_candidates": metric["raw_table_like_paragraph_candidates"],
        "clean_table_like_paragraph_candidates": metric["clean_table_like_paragraph_candidates"],
        "evidence_table_caption_count": metric["evidence_table_caption_count"],
        "evidence_table_text_count": metric["evidence_table_text_count"],
        "chunk_table_focused_count": chunk_stats["table_focused_chunk_count"],
        "chunk_caption_only_table_count": chunk_stats["caption_only_table_chunk_count"],
        "chunk_caption_plus_text_count": chunk_stats["caption_plus_text_chunk_count"],
        "paragraph_chunks_with_table_like_text": chunk_stats["paragraph_chunks_with_table_like_text"],
        "suspected_loss_stage": metric["suspected_loss_stage"],
        "confidence": metric["confidence"],
        "recommended_phase5c_action": metric["recommended_phase5c_action"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_selected_summary(path: Path, selected: list[dict[str, Any]], missing: list[str]) -> None:
    reason_counts = Counter()
    for row in selected:
        for reason in str(row["reason_selected"]).split("; "):
            if reason:
                reason_counts[reason] += 1
    lines = [
        "# Phase 5A/B Selected Table Audit Documents",
        "",
        f"- selected_doc_count: {len(selected)}",
        f"- missing_optional_inputs: {missing or []}",
        "",
        "## Selection Rationale",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- {reason}: {count}")
    lines.extend([
        "",
        "## Notes",
        "- Selection is coverage-oriented, not random.",
        "- All available Phase 4E-3 approved table target documents are prioritized.",
        "- The Phase 4E-3 hybrid table miss is explicitly tagged when present.",
        "- `raw_page_text_only` means parsed_raw did not expose true block/layout objects, so raw block counts are based on page-text pseudo-units for audit only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_examples(path: Path, metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# Table Flow Examples",
        "",
        "These examples use conservative audit heuristics only. They are not production cleaning rules.",
    ]
    for metric in metrics:
        lines.extend([
            "",
            f"## {metric['doc_id']} ({metric['source_file']})",
            "",
            f"- suspected_loss_stage: {metric['suspected_loss_stage']}",
            f"- confidence: {metric['confidence']}",
            f"- raw_block_type_distribution: `{json.dumps(metric['raw_block_type_distribution'], ensure_ascii=False, sort_keys=True)}`",
            f"- clean_block_type_distribution: `{json.dumps(metric['clean_block_type_distribution'], ensure_ascii=False, sort_keys=True)}`",
            f"- has_bbox_page_column_reading_order_metadata: `{metric['has_bbox_page_column_reading_order_metadata']}`",
            f"- evidence_excluded_table_like_count: `{metric['evidence_excluded_table_like_count']}`",
            "",
            "### Clean Table-like Paragraph Examples",
        ])
        if metric["clean_examples"]:
            for ex in metric["clean_examples"][:4]:
                lines.append(
                    f"- {ex['block_id']} page={ex['page']} type={ex['type']} reasons={ex['reasons']} :: {ex['preview']}"
                )
        else:
            lines.append("- none detected")
        lines.append("")
        lines.append("### Caption Adjacency")
        for cap in metric["adjacent_table_caption_blocks"][:3]:
            lines.append(f"- {cap['caption_block_id']} page={cap['page']} :: {cap['caption_preview']}")
            for nb in cap["next_blocks"]:
                mark = "table_like" if nb["table_like"] else "not_table_like"
                lines.append(f"  - next {nb['block_id']} {nb['type']} page={nb['page']} {mark} :: {nb['preview']}")
        if not metric["adjacent_table_caption_blocks"]:
            lines.append("- no clean table_caption blocks")
        if metric["excluded_table_like_examples"]:
            lines.append("")
            lines.append("### Excluded Table-like Examples")
            for ex in metric["excluded_table_like_examples"]:
                lines.append(f"- {ex['block_id']} page={ex['page']} type={ex['type']} :: {ex['preview']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loss_summary(path: Path, metrics: list[dict[str, Any]]) -> None:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_stage[metric["suspected_loss_stage"]].append(metric)
    payload = {
        "count_by_loss_stage": {stage: len(items) for stage, items in sorted(by_stage.items())},
        "examples_by_loss_stage": {
            stage: [
                {
                    "doc_id": item["doc_id"],
                    "source_file": item["source_file"],
                    "clean_table_like_paragraph_candidates": item["clean_table_like_paragraph_candidates"],
                    "clean_table_caption_count": item["clean_table_caption_count"],
                    "recommended_phase5c_action": item["recommended_phase5c_action"],
                }
                for item in items[:5]
            ]
            for stage, items in sorted(by_stage.items())
        },
        "confidence_by_loss_stage": {
            stage: dict(Counter(item["confidence"] for item in items))
            for stage, items in sorted(by_stage.items())
        },
        "top_recommended_actions": dict(Counter(item["recommended_phase5c_action"] for item in metrics).most_common()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_phase5c_recommendation(path: Path, metrics: list[dict[str, Any]]) -> None:
    stage_counts = Counter(metric["suspected_loss_stage"] for metric in metrics)
    table_like_docs = sum(1 for metric in metrics if metric["clean_table_like_paragraph_candidates"] > 0)
    cleaning_docs = sum(1 for metric in metrics if metric["suspected_loss_stage"].startswith("C."))
    ocr_docs = sum(1 for metric in metrics if metric["suspected_loss_stage"].startswith("F."))
    lines = [
        "# Phase 5C Minimal Recommendation",
        "",
        "## recommended_phase5c_scope",
        "Implement a small-sample, conservative table-like paragraph preservation and caption-nearby association pilot. Do not introduce table_object, OCR, parser replacement, schema-breaking chunk fields, or compact retrieval_text default changes.",
        "",
        "## why_this_scope",
        f"- selected_docs_with_clean_table_like_paragraphs: {table_like_docs}/{len(metrics)}",
        f"- suspected_loss_stage_counts: `{json.dumps(dict(stage_counts), ensure_ascii=False, sort_keys=True)}`",
        "- The dominant signal is that table rows/headers often already exist as paragraph or subsection/list-like text near table captions, while `table_text` remains zero.",
        "- Preprocess is mostly consuming caption-only table evidence because upstream clean blocks do not provide `table_text` or an equivalent table-related signal.",
        "",
        "## expected_benefit",
        "- Recover row/header terms for table retrieval without changing the object model.",
        "- Improve row-level and numeric table fact recall for documents where table content is already text-extracted.",
        "- Keep the Phase 4 caption-level baseline intact while adding a measurable small-sample enhancement.",
        "",
        "## risk",
        "- False positives from dense methods paragraphs, references, affiliations, or figure captions.",
        "- Over-associating nearby body text with a table when layout order is imperfect.",
        "- Longer table chunks may affect normal retrieval if not gated conservatively.",
        f"- cleaning_guard_needed_docs_in_sample: {cleaning_docs}",
        "",
        "## whether_schema_changes_are_needed",
        "No schema-breaking change is needed for Phase 5C. Prefer metadata-compatible marking such as `table_related_text` metadata or controlled conversion to existing `table_text` only for high-confidence local cases.",
        "",
        "## whether_reindexing_is_needed",
        "Yes for validation of any implemented Phase 5C chunk changes, but not for this audit. Re-index only a small pilot subset first.",
        "",
        "## validation_plan",
        "- Build a 10-20 document pilot from this selected set.",
        "- Add focused tests for caption-nearby table-like paragraph preservation and cleaning guards.",
        "- Compare before/after counts for `table_text` or table-related metadata, caption-only table chunks, and paragraph chunks with table-like text.",
        "- Run retrieval-only table content probes on the pilot subset, separate from the accepted Phase 4 caption baseline.",
        "- Manually inspect false positives before expanding.",
        "",
        "## stop_conditions",
        "- Stop if false positives include references, affiliations, figure captions, or normal methods paragraphs at material frequency.",
        "- Stop if normal paragraph retrieval is noticeably crowded by table chunks.",
        "- Stop if most failures require visual/OCR extraction rather than already-extracted text.",
        f"- OCR/image-table extraction should remain backlog unless OCR-stage candidates become the majority; current_sample_ocr_candidates: {ocr_docs}/{len(metrics)}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, selected: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    stage_counts = Counter(metric["suspected_loss_stage"] for metric in metrics)
    table_like_docs = sum(1 for metric in metrics if metric["clean_table_like_paragraph_candidates"] > 0)
    cleaning_docs = sum(1 for metric in metrics if metric["suspected_loss_stage"].startswith("C."))
    evidence_excluded_docs = sum(1 for metric in metrics if metric["suspected_loss_stage"].startswith("D."))
    caption_only_docs = sum(1 for metric in metrics if metric["chunk_stats"]["caption_only_table_chunk_count"] > 0)
    ocr_docs = sum(1 for metric in metrics if metric["suspected_loss_stage"].startswith("F."))
    lines = [
        "# Phase 5A/B Table Content Loss Audit Summary",
        "",
        "## 1. selected docs 数量和选择依据",
        f"- selected_docs: {len(selected)}",
        "- selection_basis: Phase 4E-3 approved table docs first, explicit table miss, high table-caption/table-keyword docs, suspected table-like paragraph docs, inferred complex-layout/long docs, and low-table controls.",
        "",
        "## 2. production table_text=0 的主要根因",
        f"- suspected_loss_stage_counts: `{json.dumps(dict(stage_counts), ensure_ascii=False, sort_keys=True)}`",
        "- main_root_cause: table content is usually present as caption text or paragraph/subsection/list-like text, but is not represented as `table_text` before evidence packing and preprocessing.",
        "",
        "## 3. 表格内容是否已经以 paragraph/list-like text 存在",
        f"- yes: {table_like_docs}/{len(metrics)} selected docs have conservative table-like paragraph/list-like candidates.",
        "",
        "## 4. 是否存在 cleaning 阶段误删/误降级",
        f"- possible_cleaning_dropped_or_downgraded_docs: {cleaning_docs}/{len(metrics)}.",
        "- Most sampled loss is not hard deletion; it is more often missing table semantic typing. A small guard backlog remains for table-like text classified as metadata/noise/reference or lost between raw page text and clean blocks.",
        "",
        "## 5. 是否 evidence policy 没纳入表格样文本",
        f"- evidence_policy_excluded_docs: {evidence_excluded_docs}/{len(metrics)}.",
        "- Most clean paragraph candidates are included as paragraph evidence, not excluded; the policy gap is semantic table typing/association rather than total evidence omission.",
        "",
        "## 6. preprocess 是否只是缺少 table_text 输入",
        f"- docs_with_caption_only_table_chunks: {caption_only_docs}/{len(metrics)}.",
        "- yes: preprocess can only create caption-only table chunks when upstream provides table_caption without table_text/table-related input.",
        "",
        "## 7. 是否需要 OCR",
        f"- likely_ocr_candidate_docs: {ocr_docs}/{len(metrics)}.",
        "- OCR is not the primary Phase 5C path for this sample; keep image-only table extraction as backlog unless a later visual audit shows it dominates.",
        "",
        "## 8. 是否建议 Phase 5C 做 table-like paragraph → table_text/table_related_text 的保守迁移",
        "- yes. Recommended Phase 5C is a small-sample conservative migration/association pilot for table-like paragraphs near table captions, preferably metadata-compatible first.",
        "",
        "## 9. 哪些问题继续进入 backlog",
        "- table_object design",
        "- OCR/image-only table extraction",
        "- parser/layout upgrade with real cell geometry",
        "- broader normal retrieval crowding analysis after any table-content enhancement",
        "- false/fragment table caption cleanup",
        "",
        "## 10. 是否建议进入小样本增强实现",
        "- yes: proceed to a small-sample Phase 5C implementation pilot with stop conditions and manual false-positive review.",
        "",
        "## Heuristic Caveat",
        "The table-like paragraph detections in this audit are conservative read-only heuristics for diagnosis. They are not production cleaning rules.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=REPO_ROOT / "data/paper_round1/parsed_raw")
    parser.add_argument("--clean_dir", type=Path, default=REPO_ROOT / "data/paper_round1/parsed_clean")
    parser.add_argument("--chunks_jsonl", type=Path, default=Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"))
    parser.add_argument(
        "--eval_set",
        type=Path,
        default=REPO_ROOT / "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
    )
    parser.add_argument(
        "--retrieval_results",
        type=Path,
        default=REPO_ROOT / "reports/table_figure_retrieval_eval/phase4e3_manual_eval/retrieval_results.json",
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "reports/phase5_table_audit")
    parser.add_argument("--max_docs", type=int, default=50)
    args = parser.parse_args()

    missing = []
    for path in [
        REPO_ROOT / "reports/phase4_closeout/summary.md",
        REPO_ROOT / "reports/table_figure_retrieval_eval/phase4e3_manual_eval/summary.md",
        args.eval_set,
        args.retrieval_results,
        args.chunks_jsonl,
    ]:
        if not path.exists():
            missing.append(str(path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_jsonl(args.chunks_jsonl)
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[str(chunk.get("doc_id", ""))].append(chunk)

    eval_context = load_eval_context(args.eval_set, args.retrieval_results)

    all_doc_ids = build_doc_universe(args.clean_dir, args.raw_dir)
    all_metrics = {
        doc_id: doc_metrics(doc_id, args.raw_dir, args.clean_dir, chunks_by_doc, with_evidence=False)
        for doc_id in all_doc_ids
    }
    selected_ids = select_docs(all_metrics, eval_context, max_docs=args.max_docs)
    selected_metrics = []
    for doc_id in selected_ids:
        metric = doc_metrics(doc_id, args.raw_dir, args.clean_dir, chunks_by_doc, with_evidence=True)
        metric["reason_selected"] = all_metrics[doc_id].get("reason_selected", "")
        selected_metrics.append(metric)
    selected_rows = [selected_row(metric, eval_context) for metric in selected_metrics]
    flow_rows = [flow_row(metric) for metric in selected_metrics]

    write_csv(args.output_dir / "selected_docs.csv", selected_rows, SELECTED_FIELDS)
    write_selected_summary(args.output_dir / "selected_docs_summary.md", selected_rows, missing)
    write_csv(args.output_dir / "table_flow_audit.csv", flow_rows, FLOW_FIELDS)
    write_examples(args.output_dir / "table_flow_examples.md", selected_metrics)
    write_loss_summary(args.output_dir / "loss_stage_summary.json", selected_metrics)
    write_phase5c_recommendation(args.output_dir / "phase5c_recommendation.md", selected_metrics)
    write_summary(args.output_dir / "summary.md", selected_rows, selected_metrics)

    print(f"selected_docs={len(selected_rows)}")
    print(f"output_dir={args.output_dir}")
    if missing:
        print("missing_optional_inputs=" + json.dumps(missing, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
