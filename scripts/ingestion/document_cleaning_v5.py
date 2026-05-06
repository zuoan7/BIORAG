#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document cleaning v5 evidence-pack helpers.

This module is the contract boundary between parsed_clean blocks and RAG
chunking. It does not parse PDFs and it does not run retrieval. Its job is to
turn classified clean blocks into explicit evidence units, while preserving
source/layout provenance and excluding non-evidence material.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


SCHEMA_VERSION = "document_cleaning_v5"
EVIDENCE_STAGE = "evidence_pack_v5"

INCLUDED_BLOCK_TYPES = {
    "title",
    "section_heading",
    "subsection_heading",
    "paragraph",
    "figure_caption",
    "table_caption",
    "table_text",
}

EXCLUDED_BLOCK_TYPES = {
    "metadata",
    "noise",
    "image",
    "references",
}

FORBIDDEN_EVIDENCE_TYPES = EXCLUDED_BLOCK_TYPES | {"text", "unknown"}

EVIDENCE_MARKERS = {
    "section_heading": "##",
    "subsection_heading": "###",
    "figure_caption": "[FIGURE CAPTION]",
    "table_caption": "[TABLE CAPTION]",
    "table_text": "[TABLE TEXT]",
}

CONTAMINATION_PATTERNS = {
    "journal_preproof": re.compile(
        r"journal\s+pre-proof|this\s+is\s+a\s+pdf\s+file\s+of\s+an\s+article",
        re.I,
    ),
    "correspondence": re.compile(
        r"\b(?:\*?\s*correspondence\s*:|to\s+whom\s+correspondence|correspondence\s+may\s+also)",
        re.I,
    ),
    "marginal_banner": re.compile(
        r"at\s+University\s+of\s+Hawaii\s+at\s+Manoa\s+Library\s+on\s+June\s+16,\s+2015",
        re.I,
    ),
    "cover_metadata": re.compile(
        r"S1096-7176|\bYMBEN\b|Accepted\s+Manuscript|Version\s+of\s+Record|"
        r"this\s+is\s+a\s+pdf\s+of\s+an\s+article|"
        r"this\s+version\s+will\s+undergo\s+additional\s+copyediting|"
        r"of\s+a\s+cover\s+page\s+and\s+metadata|"
        r"during\s+the\s+production\s+process,\s+errors\s+may\s+be\s+discovered|"
        r"in\s+its\s+final\s+form,\s+but\s+we\s+are\s+providing\s+this\s+version|"
        r"this\s+early\s+version\s+to\s+give\s+early\s+visibility\s+of\s+the\s+article|"
        r"please\s+note\s+that.*early\s+visibility\s+of\s+the\s+article|"
        r"please\s+also\s+note\s+that.*during\s+the\s+production\s+process|"
        r"errors\s+may\s+be\s+discovered\s+which\s+could\s+affect\s+the\s+content|"
        r"disclaimers\s+that\s+apply\s+to\s+the\s+journal\s+pertain|"
        r"all\s+legal\s+disclaimers\s+that\s+apply\s+to\s+the\s+journal|"
        r"^Metabolic\s+Engineering$|^\d{1,2}\s+November\s+2023$|"
        r"\b(?:Investigation|Formal\s+analysis|Conceptualization|Supervision|Writing\s*-\s*original\s+draft|"
        r"Writing\s*-\s*review\s*&\s*editing|Methodology|Validation|Visualization|Funding\s+acquisition)\b",
        re.I,
    ),
    "running_header_footer": re.compile(
        r"^page\s+\d+\s+of\s+\d+$|^open\s+access$|"
        r"this\s+article\s+is\s+available\s+from\s*:|"
        r"this\s+is\s+an\s+open\s+access\s+article\s+distributed\s+under|"
        r"open\s+access\s+this\s+article\s+is\s+licensed\s+under|"
        r"creative\s+commons\s+attribution|"
        r"(?:which\s+)?permits\s+unrestricted\s+use,\s+distribution,\s+and\s+reproduction|"
        r"provided\s+the\s+original\s+work\s+is\s+properly\s+cited|"
        r"^©\s+The\s+Author\(s\)|"
        r"vol\.\s*\d+,\s*no\.\s*\d+(?:,\s*\d{4})?|"
        r"^j\.\s+biochem\.|"
        r"biotechnology\s+and\s+bioengineering,\s+vol\.\s*110,\s+no\.\s*3|"
        r"^barrero\s+et\s+al\.\s+microb\s+cell\s+fact\s+\(\d{4}\)\s+\d+:\d+$|"
        r"^zhu\s+et\s+al\.\s+biotechnol\s+biofuels\s+\(\d{4}\)\s+\d+:\d+$",
        re.I,
    ),
    "annotation_noise": re.compile(
        r"表达\s*Fam20C|是否有尝试|共表达\s*\?{2,}|[\u4e00-\u9fff]{2,}.{0,20}\?{2,}"
    ),
}

BODY_CONTINUATION_START = re.compile(
    r"^(?:at|and|that|which|while|with|however|therefore|these|this|the|"
    r"particularly|in\s+addition|to|for|from|by|of|as|bon|after|was|were|"
    r"incubated|gDW-1|NGAM|\d+%\s+byproduct|"
    r"confined|analysis\s+of|when\s+paired|in\s+the\s+cytosol|"
    r"in\s+S\.|in\s+titers?\s+up\s+to|IgG\b|GAM\b|strains?\s+were|"
    r"when\s+grown|in\s+lysogeny|that\s+the|by\s+the|"
    r"this\s+experiment|these\s+results|we\s+(?:next|also|further|observed))\b",
    re.I,
)

PROTOCOL_RECIPE_CONTINUATION_PATTERN = re.compile(
    r"^(?:gradient\s+from|from\s+\d+(?:\.\d+)?%?\s+[A-Z]?|umn,\s*\d+|column,\s*\d+|"
    r"tryptone\b|yeast\s+extract\b|\(?NH4\)?2SO4\b|KH2PO4\b|MgSO4\b|NaCl\b|FePO4\b)",
    re.I,
)

PROTOCOL_RECIPE_KEYWORD_PATTERN = re.compile(
    r"\b(?:acetonitrile|formic\s+acid|gradient|flow\s+rate|Waters|ACN|tryptone|"
    r"yeast\s+extract|sterile\s+seawater|mineral\s+medium|KH2PO4|MgSO4|NaCl|FePO4|KOH)\b",
    re.I,
)

TABLE_EVIDENCE_KEYWORDS = re.compile(
    r"\b(?:primer|sequence|strain|plasmid|titer|titre|od600|gene|product|"
    r"sample|medium|substrate|yield|activity|concentration|temperature|"
    r"time|condition|species|isolate|accession)\b",
    re.I,
)

SENTENCE_VERBS = re.compile(
    r"\b(?:is|are|was|were|has|have|had|showed|shown|suggested|indicated|"
    r"reached|contains|contained|contains?|selected|incubated|degraded|"
    r"confirmed|increased|decreased|reduced|"
    r"confined|colonize[sd]?|paired|grown|mixed|identified|screened|"
    r"evaluated|characterized|determined|measured|expressed|engineered|"
    r"constructed|generated|transformed|enhanced|improved|enabled|allowed?|"
    r"provided?|yielded|obtained|compared|analyzed|detected|revealed|described)\b",
    re.I,
)


def normalize_text(text: str) -> str:
    """Normalize PDF text for stable evidence keys and contamination checks."""
    text = unicodedata.normalize("NFKC", text or "")
    for ch in ("\u00a0", "\u202f", "\u2009", "\u2007"):
        text = text.replace(ch, " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return re.sub(r"\s+", " ", text).strip()


def _preview(text: str, limit: int = 180) -> str:
    return normalize_text(text)[:limit]


def iter_clean_blocks(clean_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for page in clean_data.get("pages", []) or []
        if isinstance(page, dict)
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def _source_metadata(block: dict[str, Any]) -> dict[str, Any]:
    metadata = block.get("metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def source_block_id(block: dict[str, Any]) -> str | None:
    metadata = _source_metadata(block)
    return metadata.get("source_block_id") or block.get("source_block_id") or block.get("block_id")


def block_layout(block: dict[str, Any]) -> dict[str, Any]:
    metadata = _source_metadata(block)
    layout = {
        "bbox": metadata.get("bbox", block.get("bbox")),
        "column": metadata.get("column", block.get("column")),
        "reading_order": metadata.get("reading_order", block.get("reading_order")),
        "page": block.get("page"),
        "size": metadata.get("size", block.get("size")),
        "is_bold": metadata.get("is_bold", block.get("is_bold")),
        "is_italic": metadata.get("is_italic", block.get("is_italic")),
    }
    return {k: v for k, v in layout.items() if v is not None}


def evidence_display_text(block_type: str, text: str) -> str:
    text = normalize_text(text)
    if block_type == "title":
        return f"# {text.lstrip('#').strip()}"
    if block_type == "section_heading":
        return f"## {text.lstrip('#').strip()}"
    if block_type == "subsection_heading":
        return f"### {text.lstrip('#').strip()}"
    marker = EVIDENCE_MARKERS.get(block_type)
    if marker:
        return f"{marker} {text}"
    return text


def looks_like_tabular_evidence(text: str) -> bool:
    """Conservative table-row/header heuristic used before demoting table_text."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    words = normalized.split()
    if re.search(r"\[\d+(?:,\s*\d+)*\]", normalized):
        return False
    if re.search(r"[.!?]\s+[A-Z]", normalized):
        return False
    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?(?:%|x|°C|g|mg|ml|h|min)?\b", normalized, re.I))
    separators = normalized.count("\t") + normalized.count("|") + normalized.count(";")
    keyword_hit = bool(TABLE_EVIDENCE_KEYWORDS.search(normalized))
    short_header = len(words) <= 8 and keyword_hit and not re.search(r"[.!?]\s+[A-Z]", normalized)
    dense_values = numeric_tokens >= 3 and len(words) <= 18
    separated_cells = separators >= 2 and len(words) <= 24
    return short_header or dense_values or separated_cells


def looks_like_false_table_text_body_sentence(text: str, strong_table_context: bool = False) -> bool:
    """Detect table_text labels that are actually body sentence continuations."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    words = normalized.split()
    if (
        PROTOCOL_RECIPE_CONTINUATION_PATTERN.match(normalized)
        and PROTOCOL_RECIPE_KEYWORD_PATTERN.search(normalized)
        and not re.search(r"\b(?:this\s+study|harboring|plasmid|strain|FBA|RBA|in\s+vivo)\b", normalized, re.I)
    ):
        return not strong_table_context
    if len(words) < 6:
        return False
    if re.match(r"^(?:was|were|incubated|after)\b", normalized, re.I):
        return True
    if re.match(r"^(?:\d+%\s+byproduct|gDW-1|NGAM)\b", normalized, re.I) and re.search(
        r"\b(?:we|assumed|compared|for\s+growth|carbon-limited|chemostats)\b",
        normalized,
        re.I,
    ):
        return True
    if looks_like_tabular_evidence(normalized):
        return False
    lower_start = bool(re.match(r"^[a-z]", normalized))
    body_start = bool(BODY_CONTINUATION_START.search(normalized))
    has_sentence_boundary = bool(re.search(r"[.!?]\s+(?:The|This|These|We|In|Results?|However)\b", normalized))
    has_citation = bool(re.search(r"\[\d+(?:,\s*\d+)*\]", normalized))
    has_verb = bool(SENTENCE_VERBS.search(normalized))
    alpha_chars = sum(1 for ch in normalized if ch.isalpha())
    alpha_ratio = alpha_chars / max(len(normalized), 1)

    if alpha_ratio < 0.45:
        return False
    if (lower_start or body_start) and (has_sentence_boundary or has_citation or has_verb):
        return True
    if has_sentence_boundary and (has_verb or len(words) >= 8):
        return True
    # Single-sentence body text that ends with sentence-end punctuation
    if (lower_start or body_start) and re.search(r"[.!?]$", normalized) and len(words) >= 6:
        return True
    return False


def is_contaminated_evidence_text(text: str) -> tuple[bool, str]:
    normalized = normalize_text(text)
    for label, pattern in CONTAMINATION_PATTERNS.items():
        if pattern.search(normalized):
            return True, label
    return False, ""


def _new_group_id(doc_id: str, kind: str, index: int) -> str:
    return f"{doc_id}_{kind}{index:04d}"


def _block_key(block: dict[str, Any]) -> str:
    sid = source_block_id(block)
    if sid:
        return str(sid)
    if block.get("block_id"):
        return str(block["block_id"])
    return normalize_text(block.get("text", ""))[:120]


def build_evidence_pack(clean_data: dict[str, Any]) -> dict[str, Any]:
    """Build a v5 evidence-pack document from parsed_clean blocks."""
    doc_id = clean_data.get("doc_id") or "unknown_doc"
    source_file = clean_data.get("source_file", "")
    total_pages = clean_data.get("total_pages", len(clean_data.get("pages", []) or []))
    clean_blocks = iter_clean_blocks(clean_data)

    evidence_units: list[dict[str, Any]] = []
    excluded_block_counts: Counter[str] = Counter()
    excluded_examples: list[dict[str, Any]] = []
    excluded_image_blocks: list[dict[str, Any]] = []
    clean_type_counts = Counter(block.get("type", "unknown") for block in clean_blocks)

    table_group_index = 0
    figure_group_index = 0
    last_table_group_id: str | None = None
    last_table_page: int | None = None

    for block in clean_blocks:
        block_type = block.get("type", "unknown")
        text = normalize_text(block.get("text", ""))
        if block_type in EXCLUDED_BLOCK_TYPES or block_type not in INCLUDED_BLOCK_TYPES:
            excluded_block_counts[block_type] += 1
            if block_type == "image":
                excluded_image_blocks.append({
                    "block_id": block.get("block_id"),
                    "source_block_id": source_block_id(block),
                    "page": block.get("page"),
                    "layout": block_layout(block),
                    "image_metadata": _source_metadata(block),
                })
            if len(excluded_examples) < 40:
                excluded_examples.append({
                    "block_id": block.get("block_id"),
                    "source_block_id": source_block_id(block),
                    "type": block_type,
                    "page": block.get("page"),
                    "text_preview": _preview(text),
                    "layout": block_layout(block),
                })
            continue
        if not text:
            excluded_block_counts["empty_evidence_text"] += 1
            continue
        contaminated, contamination_reason = is_contaminated_evidence_text(text)
        if contaminated:
            excluded_block_counts[f"contamination:{contamination_reason}"] += 1
            if len(excluded_examples) < 40:
                excluded_examples.append({
                    "block_id": block.get("block_id"),
                    "source_block_id": source_block_id(block),
                    "type": block_type,
                    "page": block.get("page"),
                    "text_preview": _preview(text),
                    "layout": block_layout(block),
                    "reason": contamination_reason,
                })
            continue

        source_clean_block_type = block_type
        evidence_type_override = None
        strong_table_context = bool(
            last_table_group_id
            and isinstance(block.get("page"), int)
            and (last_table_page is None or int(block.get("page")) <= last_table_page + 1)
        )
        if block_type == "table_text" and looks_like_false_table_text_body_sentence(text, strong_table_context):
            block_type = "paragraph"
            evidence_type_override = "table_text_body_sentence_to_paragraph"

        page = block.get("page")
        table_group_id = None
        figure_group_id = None
        if block_type == "table_caption":
            table_group_index += 1
            table_group_id = _new_group_id(doc_id, "table", table_group_index)
            last_table_group_id = table_group_id
            last_table_page = int(page) if isinstance(page, int) else None
        elif block_type == "table_text":
            if (
                last_table_group_id
                and isinstance(page, int)
                and (last_table_page is None or page <= last_table_page + 1)
            ):
                table_group_id = last_table_group_id
            else:
                table_group_index += 1
                table_group_id = _new_group_id(doc_id, "table", table_group_index)
                last_table_group_id = table_group_id
            last_table_page = int(page) if isinstance(page, int) else last_table_page
        elif block_type == "figure_caption":
            figure_group_index += 1
            figure_group_id = _new_group_id(doc_id, "figure", figure_group_index)

        unit_index = len(evidence_units) + 1
        layout = block_layout(block)
        unit = {
            "evidence_id": f"{doc_id}_ev{unit_index:05d}",
            "schema_version": SCHEMA_VERSION,
            "type": block_type,
            "evidence_type": block_type,
            "text": text,
            "display_text": evidence_display_text(block_type, text),
            "doc_id": doc_id,
            "source_file": source_file,
            "page": page,
            "section_path": block.get("section_path", []) or [],
            "block_id": block.get("block_id"),
            "source_block_id": source_block_id(block),
            "source_block_ids": [source_block_id(block)] if source_block_id(block) else [],
            "block_ids": [block.get("block_id")] if block.get("block_id") else [],
            "layout": layout,
            "bbox": layout.get("bbox"),
            "column": layout.get("column"),
            "reading_order": layout.get("reading_order"),
            "table_group_id": table_group_id,
            "figure_group_id": figure_group_id,
            "metadata": {
                "source_block_metadata": _source_metadata(block),
                "source_clean_block_type": source_clean_block_type,
                "source_clean_block_key": _block_key(block),
                "evidence_type_override": evidence_type_override,
            },
        }
        evidence_units.append(unit)

    validation_summary = validate_evidence_pack_units(clean_data, evidence_units)

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "total_pages": total_pages,
        "parser_stage": EVIDENCE_STAGE,
        "source_parser_stage": clean_data.get("parser_stage", ""),
        "schema_version": SCHEMA_VERSION,
        "evidence_policy": {
            "included_block_types": sorted(INCLUDED_BLOCK_TYPES),
            "excluded_block_types": sorted(EXCLUDED_BLOCK_TYPES),
            "references_policy": "excluded_by_default",
            "image_policy": "pymupdf_placeholder_metadata_no_ocr",
        },
        "layout_profile": _layout_profile(clean_blocks),
        "clean_type_counts": dict(clean_type_counts),
        "excluded_block_counts": dict(excluded_block_counts),
        "excluded_block_examples": excluded_examples,
        "excluded_image_blocks": excluded_image_blocks,
        "evidence_units": evidence_units,
        "validation_summary": validation_summary,
    }


def _layout_profile(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    columns = Counter()
    pages_with_blocks = set()
    blocks_with_bbox = 0
    for block in blocks:
        layout = block_layout(block)
        page = layout.get("page")
        if page is not None:
            pages_with_blocks.add(page)
        column = layout.get("column")
        if column:
            columns[str(column)] += 1
        bbox = layout.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            blocks_with_bbox += 1
    return {
        "pages_with_blocks": len(pages_with_blocks),
        "column_distribution": dict(columns),
        "blocks_with_bbox": blocks_with_bbox,
        "block_count": len(blocks),
    }


def validate_evidence_pack_units(
    clean_data: dict[str, Any],
    evidence_units: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_blocks = iter_clean_blocks(clean_data)
    included_clean_blocks = [
        b for b in clean_blocks
        if b.get("type") in INCLUDED_BLOCK_TYPES and normalize_text(b.get("text", ""))
    ]
    evidence_types = Counter(unit.get("type", "unknown") for unit in evidence_units)
    forbidden_units = [
        unit for unit in evidence_units
        if unit.get("type") in FORBIDDEN_EVIDENCE_TYPES
    ]
    missing_source = [unit for unit in evidence_units if not unit.get("source_block_id")]
    missing_layout = [
        unit for unit in evidence_units
        if unit.get("type") not in {"title", "section_heading", "subsection_heading"}
        and not (isinstance(unit.get("bbox"), list) and unit.get("column") is not None)
    ]
    contamination = []
    for unit in evidence_units:
        text = unit.get("text", "")
        for name, pattern in CONTAMINATION_PATTERNS.items():
            if pattern.search(text):
                contamination.append({
                    "evidence_id": unit.get("evidence_id"),
                    "type": unit.get("type"),
                    "page": unit.get("page"),
                    "matched_reason": name,
                    "text_preview": _preview(text),
                })
                break

    expected_keys = {_block_key(block) for block in included_clean_blocks}
    actual_keys = {
        ((unit.get("metadata") or {}).get("source_clean_block_key") or unit.get("source_block_id"))
        for unit in evidence_units
    }
    lost_keys = sorted(k for k in expected_keys if k and k not in actual_keys)

    return {
        "clean_included_block_count": len(included_clean_blocks),
        "evidence_unit_count": len(evidence_units),
        "evidence_type_counts": dict(evidence_types),
        "forbidden_evidence_unit_count": len(forbidden_units),
        "source_block_id_retention_rate": (
            1.0 - (len(missing_source) / len(evidence_units)) if evidence_units else 1.0
        ),
        "layout_metadata_retention_rate": (
            1.0 - (len(missing_layout) / len(evidence_units)) if evidence_units else 1.0
        ),
        "contamination_count": len(contamination),
        "lost_included_block_count": len(lost_keys),
        "missing_source_examples": missing_source[:20],
        "missing_layout_examples": missing_layout[:20],
        "contamination_examples": contamination[:20],
        "lost_included_block_keys": lost_keys[:50],
    }


def evidence_pack_to_pages(evidence_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert evidence units to pages shaped like parsed_clean pages[].blocks."""
    pages: dict[int, list[dict[str, Any]]] = {}
    for unit in evidence_data.get("evidence_units", []) or []:
        if not isinstance(unit, dict):
            continue
        page_num = unit.get("page") or 1
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = 1
        metadata = dict(unit.get("metadata", {}) or {})
        source_meta = dict(metadata.get("source_block_metadata", {}) or {})
        for key in ("bbox", "column", "reading_order"):
            if unit.get(key) is not None:
                source_meta.setdefault(key, unit.get(key))
        source_meta.setdefault("source_block_id", unit.get("source_block_id"))
        source_meta.setdefault("evidence_id", unit.get("evidence_id"))
        if unit.get("table_group_id"):
            source_meta["table_group_id"] = unit.get("table_group_id")
        if unit.get("figure_group_id"):
            source_meta["figure_group_id"] = unit.get("figure_group_id")
        block = {
            "block_id": unit.get("evidence_id") or unit.get("block_id"),
            "type": unit.get("evidence_type") or unit.get("type"),
            "text": unit.get("text", ""),
            "section_path": unit.get("section_path", []) or [],
            "page": page_num,
            "metadata": source_meta,
        }
        pages.setdefault(page_num, []).append(block)

    result = []
    for page_num in sorted(pages):
        blocks = pages[page_num]
        result.append({
            "page": page_num,
            "text": "\n\n".join(evidence_display_text(b.get("type", "paragraph"), b.get("text", "")) for b in blocks),
            "blocks": blocks,
        })
    return result
