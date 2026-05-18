#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate enriched chunks produced from parsed_clean_v4."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTAMINATION_PATTERNS = {
    "journal_preproof": re.compile(r"journal\s+pre-proof|this\s+is\s+a\s+pdf\s+file\s+of\s+an\s+article", re.I),
    "correspondence": re.compile(r"\b(?:\*?\s*correspondence\s*:|to\s+whom\s+correspondence|correspondence\s+may\s+also)", re.I),
    "marginal_banner": re.compile(r"at\s+University\s+of\s+Hawaii\s+at\s+Manoa\s+Library\s+on\s+June\s+16,\s+2015", re.I),
}

FALSE_TABLE_TEXT_BODY_PATTERNS = [
    re.compile(r"\[TABLE TEXT\]\s+was\s+mixed\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+were\s+mixed\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+incubated\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+After\s+identification\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+48%\s+byproduct\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+gDW-1\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+NGAM\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+at\s+a\s+compound\s+annual\s+growth\s+rate", re.I),
    re.compile(r"\[TABLE TEXT\]\s+gradient\s+from\s+\d+", re.I),
    re.compile(r"\[TABLE TEXT\]\s+umn,\s*\d+", re.I),
    re.compile(r"\[TABLE TEXT\]\s+from\s+\d+(?:\.\d+)?%\s+[A-Z]?", re.I),
    re.compile(r"\[TABLE TEXT\]\s+tryptone\b", re.I),
]

# Phase3a-hotfix4: new false table_text detection categories
FALSE_TABLE_TEXT_GENERAL_BODY_PATTERNS = [
    # Body sentences that start like prose but are marked [TABLE TEXT]
    re.compile(r"\[TABLE TEXT\]\s+confined\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+analysis\s+of\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+when\s+paired\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+in\s+the\s+cytosol\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+in\s+S\.\s", re.I),
    re.compile(r"\[TABLE TEXT\]\s+in\s+titers?\s+up\s+to\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+IgG\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+GAM\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+that\s+the\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+by\s+the\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+we\s+(?:next|also|further|observed)\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+these\s+results\b", re.I),
    re.compile(r"\[TABLE TEXT\]\s+this\s+experiment\b", re.I),
]

TABLE_CONTEXT_LEAKAGE_PATTERNS = [
    # [TABLE TEXT] followed by body prose with citation, long sentence, or narrative structure
    # Only flag when the text after [TABLE TEXT] clearly reads as continuous body prose (≥50 chars)
    re.compile(r"\[TABLE TEXT\][^\[]*?[.!?]\s+(?:The|This|These|We)\b[^\[]*?\[\d+\]", re.I),
    re.compile(r"\[TABLE TEXT\][^\[]*?[.!?]\s+(?:In\s+this\s+study|Our\s+results)\b", re.I),
]

FRAGMENTED_TABLE_MARKER_PATTERNS = [
    # 4+ consecutive single-word [TABLE TEXT] lines (likely parser fragmentation of body text)
    re.compile(r"(?:\[TABLE TEXT\]\s+\w+\s*\n){4,}", re.I),
]

COVER_METADATA_PATTERNS = [
    re.compile(r"This\s+is\s+a\s+PDF\s+of\s+an\s+article\s+that\s+has\s+undergone\s+enhancements\s+after\s+acceptance", re.I),
    re.compile(r"This\s+version\s+will\s+undergo\s+additional\s+copyediting", re.I),
    re.compile(r"Version\s+of\s+Record", re.I),
    re.compile(r"Accepted\s+Manuscript", re.I),
    re.compile(r"S1096-7176", re.I),
    re.compile(r"\bYMBEN\b", re.I),
    re.compile(r"To\s+appear\s+in\s*:", re.I),
    re.compile(r"Please\s+cite\s+this\s+article\s+as", re.I),
    re.compile(r"in\s+its\s+final\s+form,\s+but\s+we\s+are\s+providing\s+this\s+version", re.I),
    re.compile(r"this\s+early\s+version\s+to\s+give\s+early\s+visibility\s+of\s+the\s+article", re.I),
    re.compile(r"Please\s+note\s+that", re.I),
    re.compile(r"Please\s+also\s+note\s+that", re.I),
    re.compile(r"errors\s+may\s+be\s+discovered\s+which\s+could\s+affect\s+the\s+content", re.I),
    re.compile(r"all\s+legal\s+disclaimers\s+that\s+apply\s+to\s+the\s+journal", re.I),
    re.compile(r"disclaimers\s+that\s+apply\s+to\s+the\s+journal\s+pertain", re.I),
    re.compile(r"\b(?:Investigation|Formal\s+analysis|Conceptualization|Supervision|Writing\s*-\s*original\s+draft|Writing\s*-\s*review\s*&\s*editing)\b", re.I),
]

RUNNING_HEADER_FOOTER_PATTERNS = [
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.I),
    re.compile(r"Vol\.\s*\d+,\s*No\.\s*\d+", re.I),
    re.compile(r"Trends\s+in\s+Biotechnology,\s+September\s+2025,\s+Vol\.\s*43,\s+No\.\s*9", re.I),
    re.compile(r"Biotechnology\s+and\s+Bioengineering,\s+Vol\.\s*110,\s+No\.\s*3", re.I),
    re.compile(r"Barrero\s+et\s+al\.\s+Microb\s+Cell\s+Fact", re.I),
    re.compile(r"\bOPEN\s+ACCESS\b", re.I),
    re.compile(r"J\.\s+Biochem\.\s+143,\s+187-197\s+\(2008\)", re.I),
]

ANNOTATION_NOISE_PATTERNS = [
    re.compile(r"表达\s*Fam20C"),
    re.compile(r"是否有尝试"),
    re.compile(r"共表达\s*\?{2,}"),
    re.compile(r"[\u4e00-\u9fff]{2,}.{0,20}\?{2,}"),
]

DOC0005_PAGE11_EXPECTED_TERMS = [
    "Table 3 continued",
    "gm_orf2729",
    "Encoding protein",
    "Cytochrome c551",
    "Malate dehydrogenase (quinone)",
    "Fig. 6 Putative lignin degradation pathways",
]

DOC0005_PAGE11_FORBIDDEN_TERMS = [
    "Page 11 of 14",
    "Zhu et al. Biotechnol Biofuels (2017) 10:44",
]

DOC0163_FALSE_TABLE_MARKERS = [
    "[TABLE TEXT] gradient from 15 to 80% acetonitrile",
    "[TABLE TEXT] umn, 100 Å, 1.8 μm",
    "[TABLE TEXT] from 3.5% B",
]

DOC0005_MEDIUM_FALSE_TABLE_MARKERS = [
    "[TABLE TEXT] tryptone, 1 g yeast extract",
    "[TABLE TEXT] (NH4)2SO4, 1 mM MgSO4",
]

DOC0200_COVER_FORBIDDEN_TERMS = [
    "Metabolic Engineering\n\n23 November 2023",
    "in its final form, but we are providing this version",
    "Please note that",
    "Investigation (lead, equal)",
    "Formal analysis; Writing - original draft",
    "Writing - review & editing",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def iter_clean_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for page in data.get("pages", []) or []
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def _preview(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _chunk_example(chunk: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "doc_id": chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "block_types": chunk.get("block_types", []),
        "text_preview": _preview(chunk.get("text", "")),
        "matched_reason": reason,
    }


def _first_matching_pattern(patterns: list[re.Pattern], text: str) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


def _clean_source_id(block: dict[str, Any]) -> str | None:
    metadata = block.get("metadata", {}) or {}
    return metadata.get("source_block_id") or block.get("block_id")


def analyze(clean_dir: Path, chunks_jsonl: Path) -> dict[str, Any]:
    chunks = load_chunks(chunks_jsonl)
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.get("doc_id", "")].append(chunk)

    clean_docs = [load_json(path) for path in sorted(clean_dir.glob("*.json"))]
    clean_by_doc = {doc.get("doc_id", Path(doc.get("source_file", "doc")).stem): doc for doc in clean_docs}

    clean_counts = Counter()
    clean_source_by_type: dict[str, set[str]] = defaultdict(set)
    for doc in clean_docs:
        doc_id = doc.get("doc_id", "")
        for block in iter_clean_blocks(doc):
            btype = block.get("type", "")
            clean_counts[btype] += 1
            source_id = _clean_source_id(block)
            if source_id:
                clean_source_by_type[btype].add(f"{doc_id}:{source_id}")

    chunk_type_counts = Counter()
    chunk_source_ids: set[str] = set()
    chunk_blocks_with_source = 0
    chunk_blocks_total = 0
    layout_with_metadata = 0
    layout_denominator = 0
    chunks_with_source_block_ids = 0
    chunks_missing_section_count = 0
    chunks_missing_page_count = 0
    total_text_chars = 0

    examples = {
        "metadata_in_chunk_text_examples": [],
        "references_in_chunk_text_examples": [],
        "caption_loss_examples": [],
        "table_text_loss_examples": [],
        "missing_source_block_id_examples": [],
        "suspicious_chunk_examples": [],
        "false_table_text_body_sentence_examples": [],
        "false_table_text_general_body_continuation_examples": [],
        "table_context_leakage_examples": [],
        "fragmented_table_marker_examples": [],
        "cover_metadata_in_chunk_text_examples": [],
        "running_header_footer_in_chunk_text_examples": [],
        "annotation_noise_in_chunk_text_examples": [],
        "doc_0005_page11_missing_table_evidence_examples": [],
        "doc_0005_page11_header_footer_leak_examples": [],
        "doc_0163_false_table_marker_examples": [],
        "doc_0005_medium_false_table_marker_examples": [],
        "doc_0200_cover_metadata_leak_examples": [],
    }

    contamination_counts = Counter()
    risk_doc_ids: set[str] = set()
    doc0005_searchable = ""
    doc0163_searchable = ""
    doc0200_searchable = ""
    doc0005_chunks: list[dict[str, Any]] = []
    doc0163_chunks: list[dict[str, Any]] = []
    doc0200_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text", "") or ""
        retrieval_text = chunk.get("retrieval_text", "") or ""
        total_text_chars += len(text)
        if chunk.get("source_block_ids"):
            chunks_with_source_block_ids += 1
        if not chunk.get("section"):
            chunks_missing_section_count += 1
        if chunk.get("page_start") is None or chunk.get("page_end") is None:
            chunks_missing_page_count += 1

        block_types = set(chunk.get("block_types", []) or [])
        chunk_type_counts.update(block_types)
        if chunk.get("contains_figure_caption"):
            chunk_type_counts["chunk_contains_figure_caption"] += 1
        if chunk.get("contains_table_caption"):
            chunk_type_counts["chunk_contains_table_caption"] += 1
        if chunk.get("contains_table_text"):
            chunk_type_counts["chunk_contains_table_text"] += 1

        if "metadata" in block_types or chunk.get("contains_metadata"):
            contamination_counts["metadata_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["metadata_in_chunk_text_examples"]) < 12:
                examples["metadata_in_chunk_text_examples"].append(_chunk_example(chunk, "metadata_block_type_in_chunk"))
        if "noise" in block_types or chunk.get("contains_noise"):
            contamination_counts["noise_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
        if "image" in block_types or chunk.get("contains_image"):
            contamination_counts["image_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
        if "references" in block_types or chunk.get("contains_references"):
            contamination_counts["references_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["references_in_chunk_text_examples"]) < 12:
                examples["references_in_chunk_text_examples"].append(_chunk_example(chunk, "references_block_type_in_chunk"))

        searchable_text = f"{text}\n{retrieval_text}"
        if chunk.get("doc_id") == "doc_0005":
            doc0005_searchable += "\n" + searchable_text
            doc0005_chunks.append(chunk)
        if chunk.get("doc_id") == "doc_0163":
            doc0163_searchable += "\n" + searchable_text
            doc0163_chunks.append(chunk)
        if chunk.get("doc_id") == "doc_0200":
            doc0200_searchable += "\n" + searchable_text
            doc0200_chunks.append(chunk)
        for label, pattern in CONTAMINATION_PATTERNS.items():
            if pattern.search(searchable_text):
                contamination_counts[f"{label}_in_chunk_text"] += 1
                risk_doc_ids.add(str(chunk.get("doc_id") or ""))
                if len(examples["suspicious_chunk_examples"]) < 12:
                    examples["suspicious_chunk_examples"].append(_chunk_example(chunk, label))

        false_table_match = _first_matching_pattern(FALSE_TABLE_TEXT_BODY_PATTERNS, searchable_text)
        if false_table_match:
            contamination_counts["false_table_text_body_sentence"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["false_table_text_body_sentence_examples"]) < 12:
                examples["false_table_text_body_sentence_examples"].append(
                    _chunk_example(chunk, f"false_table_text_body_sentence:{false_table_match}")
                )

        general_body_match = _first_matching_pattern(FALSE_TABLE_TEXT_GENERAL_BODY_PATTERNS, searchable_text)
        if general_body_match:
            contamination_counts["false_table_text_general_body_continuation"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["false_table_text_general_body_continuation_examples"]) < 12:
                examples["false_table_text_general_body_continuation_examples"].append(
                    _chunk_example(chunk, f"general_body_continuation:{general_body_match}")
                )

        context_leak_match = _first_matching_pattern(TABLE_CONTEXT_LEAKAGE_PATTERNS, searchable_text)
        if context_leak_match:
            contamination_counts["table_context_leakage"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["table_context_leakage_examples"]) < 12:
                examples["table_context_leakage_examples"].append(
                    _chunk_example(chunk, f"table_context_leakage:{context_leak_match}")
                )

        fragmented_match = _first_matching_pattern(FRAGMENTED_TABLE_MARKER_PATTERNS, searchable_text)
        if fragmented_match:
            contamination_counts["fragmented_table_marker"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["fragmented_table_marker_examples"]) < 12:
                examples["fragmented_table_marker_examples"].append(
                    _chunk_example(chunk, f"fragmented_table_marker:{fragmented_match}")
                )

        cover_match = _first_matching_pattern(COVER_METADATA_PATTERNS, searchable_text)
        if cover_match:
            contamination_counts["cover_metadata_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["cover_metadata_in_chunk_text_examples"]) < 12:
                examples["cover_metadata_in_chunk_text_examples"].append(
                    _chunk_example(chunk, f"cover_metadata:{cover_match}")
                )

        running_match = _first_matching_pattern(RUNNING_HEADER_FOOTER_PATTERNS, searchable_text)
        if running_match:
            contamination_counts["running_header_footer_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["running_header_footer_in_chunk_text_examples"]) < 12:
                examples["running_header_footer_in_chunk_text_examples"].append(
                    _chunk_example(chunk, f"running_header_footer:{running_match}")
                )

        annotation_match = _first_matching_pattern(ANNOTATION_NOISE_PATTERNS, searchable_text)
        if annotation_match:
            contamination_counts["annotation_noise_in_chunk_text"] += 1
            risk_doc_ids.add(str(chunk.get("doc_id") or ""))
            if len(examples["annotation_noise_in_chunk_text_examples"]) < 12:
                examples["annotation_noise_in_chunk_text_examples"].append(
                    _chunk_example(chunk, f"annotation_noise:{annotation_match}")
                )

        source_ids = chunk.get("source_block_ids", []) or []
        chunk_source_ids.update(f"{chunk.get('doc_id', '')}:{source_id}" for source_id in source_ids)
        for meta in chunk.get("source_block_metadata", []) or []:
            chunk_blocks_total += 1
            if meta.get("source_block_id"):
                chunk_blocks_with_source += 1
            elif len(examples["missing_source_block_id_examples"]) < 12:
                examples["missing_source_block_id_examples"].append(_chunk_example(chunk, "missing_source_block_id"))
            if meta.get("type") not in {"metadata", "noise", "image"}:
                layout_denominator += 1
                if meta.get("bbox") is not None and meta.get("column") is not None and meta.get("reading_order") is not None:
                    layout_with_metadata += 1

    missing_doc0005_terms = [
        term for term in DOC0005_PAGE11_EXPECTED_TERMS
        if term.lower() not in doc0005_searchable.lower()
    ]
    leaked_doc0005_terms = [
        term for term in DOC0005_PAGE11_FORBIDDEN_TERMS
        if term.lower() in doc0005_searchable.lower()
    ]
    if missing_doc0005_terms:
        contamination_counts["table_evidence_expected_terms_missing"] = len(missing_doc0005_terms)
        risk_doc_ids.add("doc_0005")
        for term in missing_doc0005_terms:
            examples["doc_0005_page11_missing_table_evidence_examples"].append({
                "doc_id": "doc_0005",
                "matched_reason": f"missing_expected_term:{term}",
                "text_preview": "",
            })
    if leaked_doc0005_terms:
        contamination_counts["table_header_footer_leak"] = len(leaked_doc0005_terms)
        risk_doc_ids.add("doc_0005")
        for term in leaked_doc0005_terms:
            match_chunk = next(
                (chunk for chunk in doc0005_chunks if term.lower() in f"{chunk.get('text','')}\n{chunk.get('retrieval_text','')}".lower()),
                {},
            )
            examples["doc_0005_page11_header_footer_leak_examples"].append(
                _chunk_example(match_chunk, f"forbidden_doc0005_term:{term}") if match_chunk else {
                    "doc_id": "doc_0005",
                    "matched_reason": f"forbidden_doc0005_term:{term}",
                    "text_preview": "",
                }
            )

    doc0163_false_markers = [
        term for term in DOC0163_FALSE_TABLE_MARKERS
        if term.lower() in doc0163_searchable.lower()
    ]
    doc0005_medium_false_markers = [
        term for term in DOC0005_MEDIUM_FALSE_TABLE_MARKERS
        if term.lower() in doc0005_searchable.lower()
    ]
    doc0200_cover_leaks = [
        term for term in DOC0200_COVER_FORBIDDEN_TERMS
        if term.lower() in doc0200_searchable.lower()
    ]
    if doc0163_false_markers:
        contamination_counts["doc_0163_false_table_marker"] = len(doc0163_false_markers)
        risk_doc_ids.add("doc_0163")
        for term in doc0163_false_markers:
            match_chunk = next(
                (chunk for chunk in doc0163_chunks if term.lower() in f"{chunk.get('text','')}\n{chunk.get('retrieval_text','')}".lower()),
                {},
            )
            examples["doc_0163_false_table_marker_examples"].append(
                _chunk_example(match_chunk, f"doc_0163_false_table_marker:{term}") if match_chunk else {
                    "doc_id": "doc_0163",
                    "matched_reason": f"doc_0163_false_table_marker:{term}",
                    "text_preview": "",
                }
            )
    if doc0005_medium_false_markers:
        contamination_counts["doc_0005_medium_false_table_marker"] = len(doc0005_medium_false_markers)
        risk_doc_ids.add("doc_0005")
        for term in doc0005_medium_false_markers:
            match_chunk = next(
                (chunk for chunk in doc0005_chunks if term.lower() in f"{chunk.get('text','')}\n{chunk.get('retrieval_text','')}".lower()),
                {},
            )
            examples["doc_0005_medium_false_table_marker_examples"].append(
                _chunk_example(match_chunk, f"doc_0005_medium_false_table_marker:{term}") if match_chunk else {
                    "doc_id": "doc_0005",
                    "matched_reason": f"doc_0005_medium_false_table_marker:{term}",
                    "text_preview": "",
                }
            )
    if doc0200_cover_leaks:
        contamination_counts["doc_0200_cover_metadata_leak"] = len(doc0200_cover_leaks)
        risk_doc_ids.add("doc_0200")
        for term in doc0200_cover_leaks:
            match_chunk = next(
                (chunk for chunk in doc0200_chunks if term.lower() in f"{chunk.get('text','')}\n{chunk.get('retrieval_text','')}".lower()),
                {},
            )
            examples["doc_0200_cover_metadata_leak_examples"].append(
                _chunk_example(match_chunk, f"doc_0200_cover_metadata_leak:{term}") if match_chunk else {
                    "doc_id": "doc_0200",
                    "matched_reason": f"doc_0200_cover_metadata_leak:{term}",
                    "text_preview": "",
                }
            )

    retained_by_type = {
        btype: len(clean_source_by_type[btype] & chunk_source_ids)
        for btype in ("figure_caption", "table_caption", "table_text")
    }
    caption_loss = (
        clean_counts["figure_caption"] - retained_by_type["figure_caption"]
        + clean_counts["table_caption"] - retained_by_type["table_caption"]
    )
    table_text_loss = clean_counts["table_text"] - retained_by_type["table_text"]

    for btype in ("figure_caption", "table_caption"):
        missing = sorted(clean_source_by_type[btype] - chunk_source_ids)
        for source_id in missing[:8]:
            examples["caption_loss_examples"].append({"source_block_id": source_id, "type": btype})
    for source_id in sorted(clean_source_by_type["table_text"] - chunk_source_ids)[:8]:
        examples["table_text_loss_examples"].append({"source_block_id": source_id, "type": "table_text"})

    source_retention_den = sum(
        len(clean_source_by_type[btype])
        for btype in ("title", "section_heading", "subsection_heading", "paragraph", "figure_caption", "table_caption", "table_text")
    )
    source_retention = len(chunk_source_ids) / source_retention_den if source_retention_den else 1.0
    block_type_coverage = (
        sum(retained_by_type.values())
        / max(clean_counts["figure_caption"] + clean_counts["table_caption"] + clean_counts["table_text"], 1)
    )
    layout_retention = layout_with_metadata / layout_denominator if layout_denominator else 1.0

    risks = []
    if source_retention < 0.95:
        risks.append("source_block_id_retention_rate_lt_0.95")
    if layout_retention < 0.95:
        risks.append("layout_metadata_retention_rate_lt_0.95")
    for key in (
        "metadata_in_chunk_text",
        "noise_in_chunk_text",
        "image_in_chunk_text",
        "references_in_chunk_text",
        "journal_preproof_in_chunk_text",
        "correspondence_in_chunk_text",
        "marginal_banner_in_chunk_text",
        "cover_metadata_in_chunk_text",
        "running_header_footer_in_chunk_text",
        "annotation_noise_in_chunk_text",
    ):
        if contamination_counts[key] > 0:
            risks.append(key)
    if caption_loss > 0:
        risks.append("caption_loss")
    if table_text_loss > max(5, int(clean_counts["table_text"] * 0.10)):
        risks.append("table_text_loss")
    if contamination_counts["false_table_text_body_sentence"] > 0:
        risks.append("false_table_text_body_sentence")
    if contamination_counts["false_table_text_general_body_continuation"] > 0:
        risks.append("false_table_text_general_body_continuation")
    # These are warning-level only — they flag potential patterns for human review
    # but do not block pass/fail because real table cells can contain sentences
    # and single-word table values (gene names, numbers, etc.)
    if contamination_counts["table_context_leakage"] > 5:
        risks.append("table_context_leakage")
    if contamination_counts["fragmented_table_marker"] > 12:
        risks.append("fragmented_table_marker")
    if contamination_counts["table_evidence_expected_terms_missing"] > 0:
        risks.append("table_evidence_expected_terms_missing")
    if contamination_counts["table_header_footer_leak"] > 0:
        risks.append("table_header_footer_leak")
    if contamination_counts["doc_0163_false_table_marker"] > 0:
        risks.append("doc_0163_false_table_marker")
    if contamination_counts["doc_0005_medium_false_table_marker"] > 0:
        risks.append("doc_0005_medium_false_table_marker")
    if contamination_counts["doc_0200_cover_metadata_leak"] > 0:
        risks.append("doc_0200_cover_metadata_leak")

    summary = {
        "doc_count": len(clean_docs),
        "chunk_count": len(chunks),
        "total_chunk_text_chars": total_text_chars,
        "chunk_with_source_block_ids_count": chunks_with_source_block_ids,
        "source_block_id_retention_rate": round(source_retention, 4),
        "block_type_coverage_rate": round(block_type_coverage, 4),
        "layout_metadata_retention_rate": round(layout_retention, 4),
        "chunks_with_figure_caption": chunk_type_counts["chunk_contains_figure_caption"],
        "chunks_with_table_caption": chunk_type_counts["chunk_contains_table_caption"],
        "chunks_with_table_text": chunk_type_counts["chunk_contains_table_text"],
        "raw_clean_figure_caption_count": clean_counts["figure_caption"],
        "raw_clean_table_caption_count": clean_counts["table_caption"],
        "raw_clean_table_text_count": clean_counts["table_text"],
        "chunk_figure_caption_count": retained_by_type["figure_caption"],
        "chunk_table_caption_count": retained_by_type["table_caption"],
        "chunk_table_text_count": retained_by_type["table_text"],
        "caption_loss_count": caption_loss,
        "table_text_loss_count": table_text_loss,
        "metadata_in_chunk_text_count": contamination_counts["metadata_in_chunk_text"],
        "noise_in_chunk_text_count": contamination_counts["noise_in_chunk_text"],
        "image_in_chunk_text_count": contamination_counts["image_in_chunk_text"],
        "references_in_chunk_text_count": contamination_counts["references_in_chunk_text"],
        "journal_preproof_in_chunk_text_count": contamination_counts["journal_preproof_in_chunk_text"],
        "correspondence_in_chunk_text_count": contamination_counts["correspondence_in_chunk_text"],
        "marginal_banner_in_chunk_text_count": contamination_counts["marginal_banner_in_chunk_text"],
        "cover_metadata_in_chunk_text_count": contamination_counts["cover_metadata_in_chunk_text"],
        "running_header_footer_in_chunk_text_count": contamination_counts["running_header_footer_in_chunk_text"],
        "annotation_noise_in_chunk_text_count": contamination_counts["annotation_noise_in_chunk_text"],
        "false_table_text_body_sentence_count": contamination_counts["false_table_text_body_sentence"],
        "false_table_text_general_body_continuation_count": contamination_counts["false_table_text_general_body_continuation"],
        "table_context_leakage_count": contamination_counts["table_context_leakage"],
        "fragmented_table_marker_count": contamination_counts["fragmented_table_marker"],
        "table_evidence_expected_terms_missing_count": contamination_counts["table_evidence_expected_terms_missing"],
        "table_header_footer_leak_count": contamination_counts["table_header_footer_leak"],
        "doc_0005_page11_expected_terms_missing": missing_doc0005_terms,
        "doc_0005_page11_forbidden_terms_leaked": leaked_doc0005_terms,
        "doc_0163_false_table_marker_count": contamination_counts["doc_0163_false_table_marker"],
        "doc_0005_medium_false_table_marker_count": contamination_counts["doc_0005_medium_false_table_marker"],
        "doc_0200_cover_metadata_leak_count": contamination_counts["doc_0200_cover_metadata_leak"],
        "doc_0163_false_table_markers": doc0163_false_markers,
        "doc_0005_medium_false_table_markers": doc0005_medium_false_markers,
        "doc_0200_cover_metadata_leaks": doc0200_cover_leaks,
        "chunks_missing_section_count": chunks_missing_section_count,
        "chunks_missing_page_count": chunks_missing_page_count,
        "potential_regression_doc_count": len({doc_id for doc_id in risk_doc_ids if doc_id}),
        "potential_regressions": risks,
        "conclusion": "pass" if not risks else "fail",
    }
    return {
        "summary": summary,
        "examples": examples,
        "clean_dir": str(clean_dir),
        "chunks_jsonl": str(chunks_jsonl),
        "doc_ids": sorted(clean_by_doc),
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = ["# v4 Phase3a Chunk Validation", ""]
    lines.append(f"- clean_dir: `{result['clean_dir']}`")
    lines.append(f"- chunks_jsonl: `{result['chunks_jsonl']}`")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    for label, examples in result["examples"].items():
        lines.append(f"### {label}")
        if not examples:
            lines.append("- none")
        else:
            for ex in examples[:20]:
                lines.append(f"- `{ex.get('doc_id', '')}` `{ex.get('chunk_id') or ex.get('source_block_id')}`: {ex.get('text_preview', ex.get('type', ''))}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v4 enriched chunks")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    result = analyze(Path(args.clean_dir), Path(args.chunks_jsonl))

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")

    print(f"doc_count={result['summary']['doc_count']}")
    print(f"chunk_count={result['summary']['chunk_count']}")
    print(f"conclusion={result['summary']['conclusion']}")
    print(f"report={report_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
