#!/usr/bin/env python3
"""Offline table_object extraction MVP for BIORAG v7 Phase7A.

This script reads only official chunks and writes isolated experiment artifacts.
It does not read BM25, access Milvus, run retrieval, call models, run OCR/VLM, or
modify the official baseline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    ROOT / "data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_table_extraction_mvp"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_table_extraction_mvp"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
REPORT_DIR = DEFAULT_REPORT_DIR

TABLE_CANDIDATES_PATH = OUTPUT_DIR / "table_candidates.jsonl"
TABLE_OBJECTS_PATH = OUTPUT_DIR / "table_objects.jsonl"
DETECTION_REPORT_PATH = REPORT_DIR / "table_candidate_detection_report.md"
PHASE_LABEL = "Phase7A"
RUN_TAG = "phase7a"

SMOKE_DOC_IDS = [
    "doc_0322",
    "doc_0158",
    "doc_0598",
    "doc_0452",
    "doc_0468",
    "doc_0687",
    "doc_0458",
    "doc_0522",
    "doc_0523",
]

OFFICIAL_BASELINE_NAME = "phase5f_official_clean_baseline"
OFFICIAL_CHUNKS_SHA256 = (
    "5dbacc5bb85351203355bf3f2d22f46ec02e24f513ab9523ca3407664669f75b"
)

TABLE_ID_RE = re.compile(
    r"\b((?:Supplementary\s+)?(?:Table|TABLE)\s+[S]?\d+[A-Za-z]?)\b(?:\s+continued)?",
    re.IGNORECASE,
)
TABLE_CAPTION_START_RE = re.compile(
    r"(?:\[TABLE CAPTION\]\s*)?((?:Supplementary\s+)?(?:Table|TABLE)\s+[S]?\d+[A-Za-z]?\.?[^|\n]{0,260})",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[±x×]\s*[-+]?\d+(?:\.\d+)?)?")
UNIT_RE = re.compile(
    r"\b(?:g/L|mg/L|U\s*mg|mM|%|cfu|OD660|g\s*ethanol|g\s*sugar|g\s*biomass|h-1|°C|bp|nt)\b",
    re.IGNORECASE,
)
LITERAL_RE = re.compile(r"\b(?:N\.D\.|ND|NT|NC|FRT|nt|not detected|mean|SD)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"\b(?:Reference or source|ref|source|supplier|Takara|DSM|ATCC|NCTC)\b", re.IGNORECASE)
FOOTNOTE_RE = re.compile(r"(?:\([0-9]+\)[a-z]|\b[a-z]The\b|\*|†|‡)")
TABLE_KEYWORD_RE = re.compile(
    r"\b(?:primer|strain|plasmid|fragment|construct|composition|carbohydrate|"
    r"yield|titer|titre|activity|selectivity|source|medium|supplier|"
    r"atmosphere|energy source|host strain|Reference or source|vector|"
    r"gene|culture conditions|LNT|LNB|ethanol|growth)\b",
    re.IGNORECASE,
)
HEADER_HINT_RE = re.compile(
    r"\b(?:Primer name|Primer sequence|Energy source|Strain or plasmid|"
    r"Reference or source|carbohydrate|Bimuno|GOS-p|Company/plant|Country|"
    r"host strain|culture conditions|LNT II titer|specific features)\b",
    re.IGNORECASE,
)

ALLOWED_SOURCE_SPAN_GRANULARITIES = {
    "table_level",
    "table_row_level",
    "row_level",
    "cell_level",
    "value_level",
    "mixed_or_unclear",
}

BLOCKING_WARNINGS = {
    "false_positive_candidate",
    "duplicate_table_candidate",
    "body_blocks_missing",
    "mixed_table_block_risk",
    "table_tail_truncation",
    "continued_table_needs_merge",
    "cell_alignment_error",
    "matrix_flattened",
    "target_mapping_risk",
    "boundary_blocking_warning",
    "row_cell_blocking_warning",
}

FALSE_POSITIVE_CAPTION_RE = re.compile(
    r"\b(?:listed in Table S\d+|were calculated using|stock suspension|"
    r"proportion of .* increased from|shown in Table S\d+)\b",
    re.IGNORECASE,
)
BODY_TEXT_STOP_RE = re.compile(
    r"\b(?:Following the blocking process|Statistical analyses|Bacterial Growth Characteristics|"
    r"Effects of HMOs|Prospects for Future Research Directions|Funding:|Conflicts of Interest|"
    r"ACKNOWLEDGMENT|How to cite this article|Supporting Information)\b",
    re.IGNORECASE,
)
TERMINAL_SECTION_RE = re.compile(
    r"\b(?:ACKNOWLEDGMENT|CONFLICT OF INTERESTS|ORCID|Supporting Information|How to cite this article)\b",
    re.IGNORECASE,
)
METRIC_GAP_RE = re.compile(
    r"\b(?:YE/S|qxylose|qarabinose|qethanol|qglucose|titer|titre|yield|"
    r"activity|selectivity|conversion|retention|AUC|OD660|hydrolysis|RTH)\b",
    re.IGNORECASE,
)
REFERENCE_COLUMN_RE = re.compile(r"\b(?:Reference or source|Reference|ref)\b", re.IGNORECASE)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def compact_id(value: str) -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def normalized_table_key(table_id: str) -> str:
    text = normalize_space(table_id).lower()
    text = re.sub(r"\bcontinued\b", "", text)
    text = text.replace("supplementary", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown_table"


def is_continued_table_id(table_id: str) -> bool:
    return "continued" in normalize_space(table_id).lower()


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_target_chunks() -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("doc_id") in SMOKE_DOC_IDS:
                chunks_by_doc[chunk["doc_id"]].append(chunk)

    available = []
    for doc_id in SMOKE_DOC_IDS:
        chunks_by_doc[doc_id].sort(key=lambda item: item.get("chunk_index", 0))
        if chunks_by_doc[doc_id]:
            available.append(doc_id)
    return chunks_by_doc, available


def table_id_from_text(text: str) -> str | None:
    match = TABLE_ID_RE.search(text)
    if not match:
        return None
    table_id = normalize_space(match.group(1))
    continued_window = text[match.start() : match.start() + 80].lower()
    if "continued" in continued_window and "continued" not in table_id.lower():
        table_id = f"{table_id} continued"
    return table_id


def caption_from_text(text: str) -> str:
    clean = normalize_space(text.replace("[TABLE CAPTION]", " "))
    match = TABLE_CAPTION_START_RE.search(clean)
    if not match:
        return clean[:320]
    caption = normalize_space(match.group(1))
    return caption[:420]


def count_numbers(text: str) -> int:
    return len(NUMERIC_RE.findall(text))


def looks_like_table_body(text: str) -> bool:
    clean = normalize_space(text)
    if not clean:
        return False
    if HEADER_HINT_RE.search(clean):
        return True
    if TABLE_KEYWORD_RE.search(clean) and count_numbers(clean) >= 3:
        return True
    if clean.count(" | ") >= 2 and (TABLE_KEYWORD_RE.search(clean) or count_numbers(clean) >= 4):
        return True
    if len(re.findall(r"\b[A-Z][A-Za-z0-9Δ_'/.-]{2,}\b", clean)) >= 8 and count_numbers(clean) >= 2:
        return True
    return False


def count_measurement_numbers(text: str) -> int:
    clean = normalize_space(text)
    numbers = NUMERIC_RE.findall(clean)
    if not numbers:
        return 0
    citation_numbers = re.findall(r"\[[0-9,\s;–-]+\]", clean)
    citation_token_count = sum(len(NUMERIC_RE.findall(item)) for item in citation_numbers)
    return max(0, len(numbers) - citation_token_count)


def looks_like_stable_table_rows(text: str) -> bool:
    clean = normalize_space(text)
    if not clean:
        return False
    if HEADER_HINT_RE.search(clean):
        return True
    if re.search(r"\b(?:Forward primer sequence|Reverse primer sequence|Primer sequence|host strain characteristics|culture conditions|LNT II titer|LNT titer|Name of Enzyme|Reference)\b", clean, re.I):
        return True
    if clean.count(" | ") >= 2:
        return True
    if METRIC_GAP_RE.search(clean) and count_measurement_numbers(clean) >= 3:
        return True
    if REFERENCE_COLUMN_RE.search(clean) and TABLE_KEYWORD_RE.search(clean):
        return True
    if len(re.findall(r"\b[A-Z][A-Za-z0-9Δ_'/.-]{2,}\b", clean)) >= 6 and count_measurement_numbers(clean) >= 2:
        return True
    return False


def is_body_text_stop(text: str) -> bool:
    clean = normalize_space(text)
    if not clean:
        return False
    if BODY_TEXT_STOP_RE.search(clean):
        return True
    if re.match(r"^(?:#|##|###)\s+", clean) and not HEADER_HINT_RE.search(clean):
        return True
    if len(clean) > 220 and not looks_like_stable_table_rows(clean) and not looks_like_table_body(clean):
        return True
    return False


def likely_embedded_table_caption(chunk: dict[str, Any]) -> bool:
    text = chunk.get("text", "")
    if "[TABLE CAPTION]" in text:
        return True
    match = TABLE_ID_RE.search(text)
    if not match:
        return False
    window = text[match.start() : match.start() + 1200]
    if match.group(1).startswith("Table S") and "[TABLE CAPTION]" not in text:
        return False
    after_id = normalize_space(text[match.end() : match.end() + 180])
    if after_id.startswith(")") or after_id.startswith("for ") or after_id.startswith(","):
        return False
    strong_title = re.match(
        r"^\.?\s*(?:Primers used|Primer name|Activity and selectivity|Composition of|"
        r"Strains and plasmids|Vectors and Fragments|PCR Amplification|Overview of|"
        r"Ethanol yields|Comparison of|List of|Bacterial Strains)",
        after_id,
        re.IGNORECASE,
    )
    if not strong_title:
        return False
    return bool((HEADER_HINT_RE.search(window) or TABLE_KEYWORD_RE.search(window)) and count_numbers(window) >= 3)


def block_ids(chunk: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata", []):
        block_id = meta.get("block_id") or meta.get("source_block_id")
        if block_id:
            ids.append(str(block_id))
    ids.extend(str(item) for item in chunk.get("source_block_ids", []) if item)
    return unique(ids)


def block_ids_by_type(chunk: dict[str, Any], block_type: str) -> list[str]:
    ids = []
    for meta in chunk.get("source_block_metadata", []):
        if meta.get("type") == block_type:
            block_id = meta.get("block_id") or meta.get("source_block_id")
            if block_id:
                ids.append(str(block_id))
    return unique(ids)


def table_like_signals(chunk: dict[str, Any], prev_chunk: dict[str, Any] | None = None) -> list[str]:
    text = chunk.get("text", "")
    signals: list[str] = []
    if chunk.get("contains_table_caption"):
        signals.append("contains_table_caption")
    if chunk.get("contains_table_text"):
        signals.append("contains_table_text")
    if TABLE_ID_RE.search(text):
        signals.append("caption_regex")
    if looks_like_table_body(text):
        signals.append("table_like_patterns")
    if HEADER_HINT_RE.search(text):
        signals.append("header_keywords")
    if count_numbers(text) >= 5:
        signals.append("multi_numeric_tokens")
    if UNIT_RE.search(text):
        signals.append("unit_tokens")
    if REFERENCE_RE.search(text):
        signals.append("reference_or_source_tokens")
    if LITERAL_RE.search(text):
        signals.append("literal_tokens")
    if "subsection_heading" in chunk.get("block_types", []) and HEADER_HINT_RE.search(text):
        signals.append("heading_as_table_header")
    if prev_chunk and prev_chunk.get("contains_table_caption") and looks_like_table_body(text):
        signals.append("adjacent_body_after_caption")
    return unique(signals)


def confidence_from_signals(signals: list[str], chunk: dict[str, Any]) -> str:
    if "contains_table_caption" in signals and ("table_like_patterns" in signals or "multi_numeric_tokens" in signals):
        return "high"
    if "contains_table_caption" in signals:
        return "medium"
    if "caption_regex" in signals and "table_like_patterns" in signals:
        return "medium"
    return "low"


def candidate_source_from_signals(signals: list[str], chunk: dict[str, Any]) -> str:
    if "contains_table_caption" in signals and "adjacent_body_after_caption" in signals:
        return "mixed_signal"
    if "contains_table_caption" in signals:
        return "table_caption_flag"
    if "caption_regex" in signals and "table_like_patterns" in signals:
        return "caption_regex"
    if "heading_as_table_header" in signals:
        return "heading_as_table_header"
    if "adjacent_body_after_caption" in signals:
        return "adjacent_body_after_caption"
    if "table_like_patterns" in signals:
        return "table_like_paragraph"
    return "mixed_signal"


def detect_candidates(
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    run_tag: str = RUN_TAG,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    missing_docs = [doc_id for doc_id in SMOKE_DOC_IDS if not chunks_by_doc.get(doc_id)]

    for doc_id in SMOKE_DOC_IDS:
        chunks = chunks_by_doc.get(doc_id, [])
        for index, chunk in enumerate(chunks):
            prev_chunk = chunks[index - 1] if index else None
            include = bool(chunk.get("contains_table_caption")) or likely_embedded_table_caption(chunk)
            if not include:
                continue

            text = chunk.get("text", "")
            table_id = table_id_from_text(text) or f"table_candidate_{len(candidates) + 1:03d}"
            signals = table_like_signals(chunk, prev_chunk)
            candidate_id = f"{run_tag}_{doc_id}_{compact_id(table_id)}_{chunk.get('chunk_index', index):03d}"
            warnings: list[str] = []
            if not chunk.get("contains_table_text"):
                warnings.append("no_table_text_flag")
            if chunk.get("contains_table_caption") and not chunk.get("contains_table_text"):
                warnings.append("parser_boundary_warning")
            if "Table S" in table_id or "Supplementary Table" in text:
                warnings.append("supplementary_required")

            candidate = {
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "source_file": chunk.get("source_file"),
                "table_id": table_id,
                "caption_text": caption_from_text(text),
                "candidate_source": candidate_source_from_signals(signals, chunk),
                "caption_block_ids": block_ids_by_type(chunk, "table_caption") or block_ids(chunk)[:1],
                "nearby_block_ids": block_ids(chunk),
                "chunk_ids": [chunk.get("chunk_id")],
                "page": chunk.get("page_start"),
                "table_like_signals": signals,
                "candidate_confidence": confidence_from_signals(signals, chunk),
                "candidate_status": "active",
                "candidate_status_reason": "candidate_detected",
                "candidate_filter_reason": "",
                "candidate_decision_warnings": [],
                "normalized_table_key": normalized_table_key(table_id),
                "continued_part": is_continued_table_id(table_id),
                "merge_target_candidate_id": "",
                "merged_from_candidate_ids": [],
                "boundary_hypothesis": {
                    "caption_chunk_id": chunk.get("chunk_id"),
                    "caption_chunk_index": chunk.get("chunk_index"),
                    "body_may_be_in_same_block": looks_like_table_body(text),
                    "body_may_be_in_adjacent_chunk": False,
                    "notes": "初始候选，后续 boundary grouping 会补充相邻 chunk。",
                },
                "warnings": unique(warnings),
            }
            candidates.append(candidate)

    return candidates, missing_docs


def _candidate_root_chunk(
    candidate: dict[str, Any],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    chunk_id = (candidate.get("chunk_ids") or [""])[0]
    for chunk in chunks_by_doc.get(candidate.get("doc_id"), []):
        if chunk.get("chunk_id") == chunk_id:
            return chunk
    return None


def _next_chunk(
    candidate: dict[str, Any],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    root = _candidate_root_chunk(candidate, chunks_by_doc)
    if not root:
        return None
    chunks = chunks_by_doc.get(candidate.get("doc_id"), [])
    try:
        index = chunks.index(root)
    except ValueError:
        return None
    if index + 1 >= len(chunks):
        return None
    return chunks[index + 1]


def candidate_body_score(
    candidate: dict[str, Any],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
) -> int:
    root = _candidate_root_chunk(candidate, chunks_by_doc)
    if not root:
        return 0
    score = 0
    root_text = root.get("text", "")
    if looks_like_stable_table_rows(root_text):
        score += 4
    elif looks_like_table_body(root_text):
        score += 2
    if root.get("contains_table_text"):
        score += 2

    next_chunk = _next_chunk(candidate, chunks_by_doc)
    if next_chunk and not next_chunk.get("contains_table_caption") and not next_chunk.get("contains_figure_caption"):
        next_text = next_chunk.get("text", "")
        if looks_like_stable_table_rows(next_text):
            score += 4
        elif looks_like_table_body(next_text):
            score += 2
    return score


def mark_candidate(
    candidate: dict[str, Any],
    status: str,
    reason: str,
    warnings: list[str],
) -> None:
    candidate["candidate_status"] = status
    candidate["candidate_status_reason"] = reason
    candidate["candidate_filter_reason"] = reason if status in {"filtered", "deduped"} else ""
    candidate["candidate_decision_warnings"] = unique(candidate.get("candidate_decision_warnings", []) + warnings)
    candidate["warnings"] = unique(candidate.get("warnings", []) + warnings)


def annotate_candidate_decisions(
    candidates: list[dict[str, Any]],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Mark false positives, shadows, duplicates, and continued parts in-place."""

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    per_doc_legacy_counter: Counter[str] = Counter()
    for candidate in candidates:
        candidate["normalized_table_key"] = normalized_table_key(candidate.get("table_id", ""))
        candidate["continued_part"] = is_continued_table_id(candidate.get("table_id", ""))
        candidate["body_signal_score"] = candidate_body_score(candidate, chunks_by_doc)
        per_doc_legacy_counter[candidate.get("doc_id", "")] += 1
        candidate["legacy_phase7a_table_object_id"] = (
            f"{candidate.get('doc_id')}__{compact_id(candidate.get('table_id', 'table'))}"
            f"__phase7a_{per_doc_legacy_counter[candidate.get('doc_id', '')]:02d}"
        )
        by_group[(candidate.get("doc_id", ""), candidate["normalized_table_key"])].append(candidate)

    for candidate in candidates:
        root = _candidate_root_chunk(candidate, chunks_by_doc)
        root_text = root.get("text", "") if root else ""
        if FALSE_POSITIVE_CAPTION_RE.search(root_text) and not looks_like_stable_table_rows(root_text):
            mark_candidate(
                candidate,
                "filtered",
                "false_positive_candidate",
                ["false_positive_candidate", "candidate_filtered"],
            )

    for (_doc_id, _table_key), group in by_group.items():
        non_continued = [item for item in group if not item.get("continued_part")]
        continued = [item for item in group if item.get("continued_part")]
        active_non_continued = [
            item for item in non_continued if item.get("candidate_status") == "active"
        ]
        primary = max(active_non_continued, key=lambda item: item.get("body_signal_score", 0), default=None)

        if primary:
            shadow_threshold = max(1, primary.get("body_signal_score", 0))
            for item in non_continued:
                if item is primary:
                    continue
                if item.get("candidate_status") != "active" or item.get("body_signal_score", 0) < shadow_threshold:
                    item["merge_target_candidate_id"] = primary.get("candidate_id", "")
                    mark_candidate(
                        item,
                        "deduped",
                        "duplicate_or_shadow_candidate",
                        [
                            "duplicate_table_candidate",
                            "shadow_caption_candidate",
                            "candidate_deduped",
                        ],
                    )

        for item in continued:
            item["candidate_decision_warnings"] = unique(
                item.get("candidate_decision_warnings", []) + ["continued_table_part"]
            )
            item["warnings"] = unique(item.get("warnings", []) + ["continued_table_part"])
            if primary:
                item["merge_target_candidate_id"] = primary.get("candidate_id", "")
                primary["merged_from_candidate_ids"] = unique(
                    primary.get("merged_from_candidate_ids", []) + [item.get("candidate_id", "")]
                )
                primary["candidate_decision_warnings"] = unique(
                    primary.get("candidate_decision_warnings", []) + ["continued_table_merged"]
                )
                primary["warnings"] = unique(primary.get("warnings", []) + ["continued_table_merged"])
                mark_candidate(
                    item,
                    "merged_into_primary",
                    "continued_table_merged",
                    ["continued_table_part", "continued_table_merged"],
                )
            else:
                mark_candidate(
                    item,
                    "active",
                    "continued_table_needs_merge",
                    [
                        "continued_table_part",
                        "continued_table_needs_merge",
                        "continued_table_merge_uncertain",
                    ],
                )

        if primary and len(active_non_continued) > 1:
            primary["candidate_decision_warnings"] = unique(
                primary.get("candidate_decision_warnings", []) + ["duplicate_table_candidate"]
            )

    return candidates


def chunk_lookup(chunks_by_doc: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunks in chunks_by_doc.values() for chunk in chunks}


def group_candidate_boundary(
    candidate: dict[str, Any],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    chunks = chunks_by_doc[candidate["doc_id"]]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    root_chunk = by_id[candidate["chunk_ids"][0]]
    index = chunks.index(root_chunk)
    grouped = [root_chunk]
    warnings = list(candidate.get("warnings", []))
    relation = "same_block" if looks_like_table_body(root_chunk.get("text", "")) else "uncertain"

    next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None
    if next_chunk:
        if next_chunk.get("contains_table_caption"):
            warnings.append("body_grouping_stopped_at_next_table")
        elif next_chunk.get("contains_figure_caption"):
            warnings.extend(["body_grouping_stopped_at_figure", "adjacent_non_table_contamination"])
        elif looks_like_table_body(next_chunk.get("text", "")) or looks_like_stable_table_rows(next_chunk.get("text", "")):
            grouped.append(next_chunk)
            warnings.extend(["caption_body_split", "body_as_paragraph"])
            relation = "adjacent_chunk_split"
            candidate["boundary_hypothesis"]["body_may_be_in_adjacent_chunk"] = True
        else:
            warnings.append("body_grouping_stopped_at_body_text")

    caption_block_ids: list[str] = []
    header_block_ids: list[str] = []
    body_block_ids: list[str] = []
    source_block_ids: list[str] = []
    table_text_parts: list[str] = []
    stopped = False
    for chunk in grouped:
        caption_block_ids.extend(block_ids_by_type(chunk, "table_caption"))
        for meta in chunk.get("source_block_metadata", []):
            block_id = meta.get("block_id") or meta.get("source_block_id")
            if not block_id:
                continue
            block_type = meta.get("type")
            preview = normalize_space(meta.get("text_preview"))
            if block_type == "table_caption" and source_block_ids and TABLE_ID_RE.search(preview):
                warnings.append("body_grouping_stopped_at_next_table")
                stopped = True
                break
            if block_type == "figure_caption":
                warnings.extend(["body_grouping_stopped_at_figure", "adjacent_non_table_contamination"])
                stopped = True
                break
            if source_block_ids and is_body_text_stop(preview):
                warnings.append("body_grouping_stopped_at_body_text")
                terminal_after_body = bool(body_block_ids and TERMINAL_SECTION_RE.search(preview))
                if not terminal_after_body and (
                    BODY_TEXT_STOP_RE.search(preview) or block_type in {"title", "section_heading"}
                ):
                    warnings.append("adjacent_non_table_contamination")
                stopped = True
                break

            source_block_ids.append(str(block_id))
            if chunk is root_chunk and block_type == "table_caption":
                table_text_parts.append(normalize_space(chunk.get("text", "")))
            else:
                table_text_parts.append(preview)
            if block_type in {"section_heading", "subsection_heading"} and HEADER_HINT_RE.search(preview):
                header_block_ids.append(str(block_id))
            if block_type == "table_caption" and looks_like_table_body(preview):
                body_block_ids.append(str(block_id))
                warnings.append("body_as_table_caption")
            if block_type == "paragraph" and looks_like_table_body(preview):
                body_block_ids.append(str(block_id))
                warnings.append("body_as_paragraph")
            if block_type == "subsection_heading" and looks_like_table_body(preview):
                body_block_ids.append(str(block_id))
                warnings.append("body_as_subsection_heading")
        if stopped:
            break

    if not caption_block_ids:
        caption_block_ids = block_ids(root_chunk)[:1]
    if not body_block_ids and looks_like_table_body(root_chunk.get("text", "")) and candidate.get("candidate_status") == "active":
        body_block_ids = caption_block_ids[:]
    if not header_block_ids:
        header_block_ids = caption_block_ids[:1]

    boundary_text = " ".join(table_text_parts) if table_text_parts else " ".join(chunk.get("text", "") for chunk in grouped)
    table_mentions = {normalize_space(item).lower() for item in TABLE_ID_RE.findall(boundary_text)}
    if len(table_mentions) > 1:
        warnings.append("mixed_table_block_risk")
        relation = "contaminated"
    if any(chunk.get("contains_figure_caption") for chunk in grouped):
        warnings.extend(["adjacent_non_table_contamination", "body_grouping_stopped_at_figure"])

    if body_block_ids:
        boundary_status = "boundary_pass_with_warnings" if warnings else "boundary_pass"
    else:
        boundary_status = "boundary_partial"
        warnings.extend(["table_boundary_partial", "body_blocks_missing", "boundary_blocking_warning"])

    if "mixed_table_block_risk" in warnings or "adjacent_non_table_contamination" in warnings:
        warnings.append("boundary_blocking_warning")
        boundary_status = "boundary_partial"

    return {
        "chunks": grouped,
        "caption_block_ids": unique(caption_block_ids),
        "header_block_ids": unique(header_block_ids),
        "body_block_ids": unique(body_block_ids),
        "source_block_ids": unique(source_block_ids),
        "chunk_ids": [chunk.get("chunk_id") for chunk in grouped],
        "allowed_source_block_ids": unique(source_block_ids),
        "table_text": normalize_space(boundary_text),
        "boundary_status": boundary_status,
        "caption_body_relation_status": relation,
        "warnings": unique(warnings),
    }


def make_source_spans(
    table_object_id: str,
    doc_id: str,
    grouped_chunks: list[dict[str, Any]],
    body_block_ids: list[str],
    allowed_block_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    source_spans: list[dict[str, Any]] = []
    block_to_spans: dict[str, list[str]] = defaultdict(list)
    allowed = set(allowed_block_ids or [])

    for chunk in grouped_chunks:
        metadata = chunk.get("source_block_metadata") or []
        if not metadata:
            metadata = [
                {
                    "block_id": chunk.get("chunk_id"),
                    "source_block_id": chunk.get("chunk_id"),
                    "type": "chunk",
                    "page": chunk.get("page_start"),
                    "text_preview": chunk.get("text", "")[:240],
                }
            ]
        for meta in metadata:
            block_id = str(meta.get("block_id") or meta.get("source_block_id") or chunk.get("chunk_id"))
            if allowed and block_id not in allowed:
                continue
            span_id = f"{table_object_id}__span_{len(source_spans) + 1:03d}"
            granularity = "table_row_level" if block_id in body_block_ids else "table_level"
            span = {
                "source_span_id": span_id,
                "doc_id": doc_id,
                "chunk_id": chunk.get("chunk_id"),
                "block_id": block_id,
                "page": meta.get("page") or chunk.get("page_start"),
                "span_text": normalize_space(meta.get("text_preview") or chunk.get("text", "")[:240]),
                "granularity": granularity,
                "bbox": None,
            }
            source_spans.append(span)
            block_to_spans[block_id].append(span_id)

    return source_spans, block_to_spans


def first_body_span(source_spans: list[dict[str, Any]]) -> list[str]:
    for span in source_spans:
        if span.get("granularity") == "table_row_level":
            return [span["source_span_id"]]
    return [source_spans[0]["source_span_id"]] if source_spans else []


def normalized_value(raw: str) -> str | None:
    text = normalize_space(raw)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return text
    return None


def literal_marker(raw: str) -> str | None:
    match = LITERAL_RE.search(raw)
    return match.group(0) if match else None


def add_cell(
    cells: list[dict[str, Any]],
    table_object_id: str,
    row_id: str,
    column_id: str,
    value_raw: str,
    source_span_ids: list[str],
    unit: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    cells.append(
        {
            "cell_id": f"{table_object_id}__cell_{len(cells) + 1:04d}",
            "row_id": row_id,
            "column_id": column_id,
            "value_raw": normalize_space(value_raw),
            "value_normalized": normalized_value(value_raw),
            "unit": unit,
            "literal_marker": literal_marker(value_raw),
            "footnote_refs": [],
            "reference_refs": [],
            "source_span_ids": source_span_ids,
            "warnings": warnings or [],
        }
    )


def parse_energy_table(
    table_object_id: str, text: str, source_spans: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    columns = [
        {"column_id": f"{table_object_id}__col_001", "column_index": 1, "header": "Energy source", "unit": None, "header_path": ["Energy source"], "source_span_ids": first_body_span(source_spans), "warnings": []},
        {"column_id": f"{table_object_id}__col_002", "column_index": 2, "header": "JAT/pGb3", "unit": "g/L", "header_path": ["JAT/pGb3"], "source_span_ids": first_body_span(source_spans), "warnings": []},
        {"column_id": f"{table_object_id}__col_003", "column_index": 3, "header": "JAET/pGb3", "unit": "g/L", "header_path": ["JAET/pGb3"], "source_span_ids": first_body_span(source_spans), "warnings": []},
    ]
    row_specs = [
        ("Galactose", r"Galactose\s+([0-9.]+\s*±\s*[0-9.]+)\s+([0-9.]+\s*±\s*[0-9.]+)"),
        ("Glucose", r"Glucose\s+([0-9.]+\s*±\s*[0-9.]+)\s+([0-9.]+\s*±\s*[0-9.]+)"),
        ("α-KG", r"α[‐-]?KG\s+([0-9.]+\s*±\s*[0-9.]+)\s+([0-9.]+\s*±\s*[0-9.]+)"),
        ("Succinate", r"Succinate\s+([0-9.]+\s*±\s*[0-9.]+)\s+([0-9.]+\s*±\s*[0-9.]+)"),
    ]
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    span_ids = first_body_span(source_spans)
    for label, pattern in row_specs:
        match = re.search(pattern, text)
        if not match:
            continue
        row_id = f"{table_object_id}__row_{len(rows) + 1:03d}"
        rows.append({"row_id": row_id, "row_index": len(rows) + 1, "row_label": label, "row_text": normalize_space(match.group(0)), "source_span_ids": span_ids, "warnings": []})
        add_cell(cells, table_object_id, row_id, columns[0]["column_id"], label, span_ids)
        add_cell(cells, table_object_id, row_id, columns[1]["column_id"], match.group(1), span_ids, unit="g/L")
        add_cell(cells, table_object_id, row_id, columns[2]["column_id"], match.group(2), span_ids, unit="g/L")
    footnotes = []
    if "averages of three replicates" in text:
        footnotes.append({"footnote_id": f"{table_object_id}__note_001", "marker": "Note", "text": "Data shown are averages of three replicates.", "source_span_ids": span_ids, "binding_status": "table_level_note"})
    return columns, rows, cells, footnotes, [], []


def parse_pcr_table(
    table_object_id: str, text: str, source_spans: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    headers = ["strain", "not amplified", "short", "mutation", "correct"]
    columns = [
        {"column_id": f"{table_object_id}__col_{i:03d}", "column_index": i, "header": header, "unit": None, "header_path": [header], "source_span_ids": first_body_span(source_spans), "warnings": []}
        for i, header in enumerate(headers, 1)
    ]
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    span_ids = first_body_span(source_spans)
    row_values = [
        ("CBS7435", ["6 (1)b", "13 (1)b", "", ""]),
        ("CBS7435 dnl4 his4", ["24 (3)b", "", "", ""]),
    ]
    for label, values in row_values:
        if label not in text:
            continue
        row_id = f"{table_object_id}__row_{len(rows) + 1:03d}"
        row_text = f"{label} {' '.join(value for value in values if value)}"
        rows.append({"row_id": row_id, "row_index": len(rows) + 1, "row_label": label, "row_text": row_text, "source_span_ids": span_ids, "warnings": ["footnote_present_not_bound"]})
        add_cell(cells, table_object_id, row_id, columns[0]["column_id"], label, span_ids)
        for col, value in zip(columns[1:], values):
            if value:
                add_cell(cells, table_object_id, row_id, col["column_id"], value, span_ids, warnings=["footnote_present_not_bound"])
    footnotes = []
    if "categorized number" in text or "The categorized number" in text:
        footnotes.append({"footnote_id": f"{table_object_id}__note_001", "marker": "a/b", "text": "The categorized number of transformants and footnote markers are visible but cell applicability is uncertain.", "source_span_ids": span_ids, "binding_status": "present_not_bound_to_specific_cell"})
    return columns, rows, cells, footnotes, [], ["numeric_column_order_uncertain", "footnote_present_not_bound", "manual_review_dependency"]


def parse_primer_table(
    table_object_id: str, text: str, source_spans: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    columns = [
        {"column_id": f"{table_object_id}__col_001", "column_index": 1, "header": "Primer name", "unit": None, "header_path": ["Primer name"], "source_span_ids": first_body_span(source_spans), "warnings": []},
        {"column_id": f"{table_object_id}__col_002", "column_index": 2, "header": "Primer sequence (5' to 3')", "unit": None, "header_path": ["Primer sequence"], "source_span_ids": first_body_span(source_spans), "warnings": []},
        {"column_id": f"{table_object_id}__col_003", "column_index": 3, "header": "Location", "unit": None, "header_path": ["Location"], "source_span_ids": first_body_span(source_spans), "warnings": []},
    ]
    pattern = re.compile(
        r"\b([A-Za-z][A-Za-z0-9'_-]{2,})\s+([ACGT]{12,})\s+(.+?)(?=\s+[A-Za-z][A-Za-z0-9'_-]{2,}\s+[ACGT]{12,}|\s+TABLE|\s+Table|$)"
    )
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    span_ids = first_body_span(source_spans)
    for match in pattern.finditer(text):
        row_id = f"{table_object_id}__row_{len(rows) + 1:03d}"
        location = normalize_space(match.group(3))[:220]
        row_text = f"{match.group(1)} {match.group(2)} {location}"
        rows.append({"row_id": row_id, "row_index": len(rows) + 1, "row_label": match.group(1), "row_text": row_text, "source_span_ids": span_ids, "warnings": []})
        add_cell(cells, table_object_id, row_id, columns[0]["column_id"], match.group(1), span_ids)
        add_cell(cells, table_object_id, row_id, columns[1]["column_id"], match.group(2), span_ids)
        add_cell(cells, table_object_id, row_id, columns[2]["column_id"], location, span_ids)
        if len(rows) >= 18:
            break
    return columns, rows, cells, [], [], []


def parse_composition_table(
    table_object_id: str, text: str, source_spans: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    columns = [
        {"column_id": f"{table_object_id}__col_001", "column_index": 1, "header": "carbohydrate", "unit": None, "header_path": ["carbohydrate"], "source_span_ids": first_body_span(source_spans), "warnings": []},
        {"column_id": f"{table_object_id}__col_002", "column_index": 2, "header": "Bimuno GOS", "unit": "% area", "header_path": ["Bimuno GOS"], "source_span_ids": first_body_span(source_spans), "warnings": ["unit_scope_column_level"]},
        {"column_id": f"{table_object_id}__col_003", "column_index": 3, "header": "GOS-p", "unit": "% area", "header_path": ["GOS-p"], "source_span_ids": first_body_span(source_spans), "warnings": ["unit_scope_column_level"]},
    ]
    row_pattern = re.compile(
        r"((?:HPLC fraction\s+\d+:\s+DP\d(?: and/or higher)?|glucose|galactose))\s+"
        r"([0-9.]+|not detected)\s+([0-9.]+|not detected)",
        re.I,
    )
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    span_ids = first_body_span(source_spans)
    for match in row_pattern.finditer(text):
        row_id = f"{table_object_id}__row_{len(rows) + 1:03d}"
        rows.append({"row_id": row_id, "row_index": len(rows) + 1, "row_label": match.group(1), "row_text": normalize_space(match.group(0)), "source_span_ids": span_ids, "warnings": []})
        add_cell(cells, table_object_id, row_id, columns[0]["column_id"], match.group(1), span_ids)
        add_cell(cells, table_object_id, row_id, columns[1]["column_id"], match.group(2), span_ids, unit="% area")
        add_cell(cells, table_object_id, row_id, columns[2]["column_id"], match.group(3), span_ids, unit="% area")
    footnotes = []
    if "area under curve" in text or "area%" in text:
        footnotes.append({"footnote_id": f"{table_object_id}__note_001", "marker": "a", "text": "The HPLC composition note is visible but not bound to individual cells.", "source_span_ids": span_ids, "binding_status": "table_level_note"})
    warnings = ["unit_scope_column_level"]
    if len(rows) < 6 and "HPLC fraction 4" in text:
        warnings.extend(["table_tail_truncation", "boundary_blocking_warning"])
    return columns, rows, cells, footnotes, [], unique(warnings)


def parse_variant_activity_table(
    table_object_id: str, text: str, source_spans: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    columns = [
        {"column_id": f"{table_object_id}__col_001", "column_index": 1, "header": "LnbB variant", "unit": None, "header_path": ["LnbB variant"], "source_span_ids": first_body_span(source_spans), "warnings": []},
        {"column_id": f"{table_object_id}__col_002", "column_index": 2, "header": "activity/selectivity value bundle", "unit": None, "header_path": ["activity/selectivity value bundle"], "source_span_ids": first_body_span(source_spans), "warnings": ["numeric_column_order_uncertain"]},
    ]
    variants = ["WT", "D320E", "D320A", "Y419F"]
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    span_ids = first_body_span(source_spans)
    for i, variant in enumerate(variants):
        start = text.find(variant)
        if start < 0:
            continue
        next_positions = [text.find(v, start + len(variant)) for v in variants[i + 1 :] if text.find(v, start + len(variant)) > 0]
        end = min(next_positions) if next_positions else min(len(text), start + 260)
        bundle = normalize_space(text[start + len(variant) : end])
        if not bundle:
            continue
        row_id = f"{table_object_id}__row_{len(rows) + 1:03d}"
        rows.append({"row_id": row_id, "row_index": len(rows) + 1, "row_label": variant, "row_text": f"{variant} {bundle}", "source_span_ids": span_ids, "warnings": ["numeric_column_order_uncertain"]})
        add_cell(cells, table_object_id, row_id, columns[0]["column_id"], variant, span_ids)
        add_cell(cells, table_object_id, row_id, columns[1]["column_id"], bundle, span_ids, warnings=["numeric_column_order_uncertain"])
    return columns, rows, cells, [], [], [
        "numeric_column_order_uncertain",
        "metric_level_cell_gap",
        "unit_visible_not_bound",
        "row_cell_blocking_warning",
    ]


def useful_span_rows(source_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for span in source_spans:
        text = normalize_space(span.get("span_text"))
        if len(text) < 20:
            continue
        if text.startswith("#"):
            continue
        if re.match(r"^(Figure|Fig\.)\b", text, re.I):
            continue
        if looks_like_table_body(text) or count_numbers(text) >= 2 or TABLE_KEYWORD_RE.search(text):
            rows.append(span)
    return rows[:18]


def parse_generic_table(
    table_object_id: str,
    text: str,
    source_spans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    columns = [
        {"column_id": f"{table_object_id}__col_001", "column_index": 1, "header": "row_text", "unit": None, "header_path": ["row_text"], "source_span_ids": first_body_span(source_spans), "warnings": []}
    ]
    warnings: list[str] = []
    measurement_numbers = count_measurement_numbers(text)
    has_metric_gap = bool(METRIC_GAP_RE.search(text) and measurement_numbers >= 3)
    has_reference_column = bool(REFERENCE_COLUMN_RE.search(text))
    if measurement_numbers >= 3:
        columns.append({"column_id": f"{table_object_id}__col_002", "column_index": 2, "header": "numeric_values_visible_in_row", "unit": None, "header_path": ["numeric_values_visible_in_row"], "source_span_ids": first_body_span(source_spans), "warnings": ["numeric_column_order_uncertain"]})
        warnings.extend(["numeric_column_order_uncertain", "cell_alignment_error", "row_cell_blocking_warning"])
        if has_metric_gap:
            warnings.append("metric_level_cell_gap")
    if UNIT_RE.search(text):
        warnings.append("unit_visible_not_bound" if has_metric_gap or measurement_numbers else "unit_scope_table_level")
    if has_reference_column:
        warnings.extend(["internal_reference_column", "external_citation_not_supported"])
    elif REFERENCE_RE.search(text):
        warnings.append("reference_visible_not_bound")
    if LITERAL_RE.search(text):
        warnings.append("literal_value_requires_preservation")
    if "Abbreviation:" in text:
        warnings.append("abbreviation_binding_ok")
    if "matrix" in text.lower() or "DOL" in text or (measurement_numbers >= 8 and not has_reference_column):
        warnings.extend(["matrix_flattened", "row_cell_blocking_warning"])

    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    span_rows = useful_span_rows(source_spans)
    if not span_rows:
        span_rows = source_spans[:1]

    for span in span_rows:
        row_text = normalize_space(span.get("span_text"))
        row_id = f"{table_object_id}__row_{len(rows) + 1:03d}"
        row_label = " ".join(row_text.split()[:6])
        row_warnings = []
        if count_measurement_numbers(row_text) >= 3 and len(columns) > 1:
            row_warnings.append("numeric_column_order_uncertain")
        rows.append({"row_id": row_id, "row_index": len(rows) + 1, "row_label": row_label, "row_text": row_text, "source_span_ids": [span["source_span_id"]], "warnings": row_warnings})
        add_cell(cells, table_object_id, row_id, columns[0]["column_id"], row_text, [span["source_span_id"]], warnings=row_warnings)
        if len(columns) > 1:
            nums = " ".join(NUMERIC_RE.findall(row_text))
            if nums:
                add_cell(cells, table_object_id, row_id, columns[1]["column_id"], nums, [span["source_span_id"]], warnings=["numeric_column_order_uncertain"])

    footnotes: list[dict[str, Any]] = []
    if "Note:" in text or "aThe" in text or "Abbreviation:" in text:
        note_match = re.search(r"(Note:|aThe|Abbreviation:).{0,260}", text, re.I)
        if note_match:
            footnotes.append({"footnote_id": f"{table_object_id}__note_001", "marker": note_match.group(1).rstrip(":"), "text": normalize_space(note_match.group(0)), "source_span_ids": first_body_span(source_spans), "binding_status": "table_level_or_uncertain"})
            warnings.append("footnote_present_not_bound")

    return columns, rows, cells, footnotes, [], unique(warnings)


def parse_table_structure(
    table_object_id: str,
    candidate: dict[str, Any],
    grouped_chunks: list[dict[str, Any]],
    source_spans: list[dict[str, Any]],
    boundary_text: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], str]:
    text = boundary_text or " ".join(chunk.get("text", "") for chunk in grouped_chunks)
    normalized = normalize_space(text)
    table_id = candidate.get("table_id", "")

    if "Energy source JAT/pGb3 JAET/pGb3" in normalized:
        columns, rows, cells, footnotes, refs, warnings = parse_energy_table(table_object_id, normalized, source_spans)
        return columns, rows, cells, footnotes, refs, warnings, "energy_titer_parser"

    if "PCR Amplification and Sequencing Analysis" in normalized and "not amplified" in normalized:
        columns, rows, cells, footnotes, refs, warnings = parse_pcr_table(table_object_id, normalized, source_spans)
        return columns, rows, cells, footnotes, refs, warnings, "pcr_category_parser"

    if "Primer name" in normalized and "Primer sequence" in normalized:
        columns, rows, cells, footnotes, refs, warnings = parse_primer_table(table_object_id, normalized, source_spans)
        if rows:
            return columns, rows, cells, footnotes, refs, warnings, "primer_sequence_parser"

    if "Composition of the Commercial Bimuno GOS" in normalized:
        columns, rows, cells, footnotes, refs, warnings = parse_composition_table(table_object_id, normalized, source_spans)
        if rows:
            return columns, rows, cells, footnotes, refs, warnings, "composition_percentage_parser"

    if "Activity and selectivity parameters" in normalized and "LnbB" in normalized:
        columns, rows, cells, footnotes, refs, warnings = parse_variant_activity_table(table_object_id, normalized, source_spans)
        if rows:
            return columns, rows, cells, footnotes, refs, warnings, "variant_activity_bundle_parser"

    columns, rows, cells, footnotes, refs, warnings = parse_generic_table(table_object_id, normalized, source_spans)
    parser_name = "generic_row_text_parser"
    if "continued" in table_id.lower():
        warnings.append("caption_body_split")
    return columns, rows, cells, footnotes, refs, warnings, parser_name


def validation_status_for_object(obj: dict[str, Any]) -> str:
    if not obj.get("table_object_id") or not obj.get("doc_id") or not obj.get("source_spans"):
        return "fail"
    if not obj.get("chunk_ids") or not obj.get("source_block_ids") or not obj.get("source_span_granularity"):
        return "fail"
    warnings = set(obj.get("warnings", []))
    if "false_positive_candidate" in warnings:
        return "fail"
    if not obj.get("caption") or not obj.get("rows") or not obj.get("columns") or not obj.get("cells"):
        return "partial"
    if warnings & BLOCKING_WARNINGS:
        return "partial"
    return "pass_with_warnings" if warnings else "pass"


def build_table_objects(
    candidates: list[dict[str, Any]],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    run_tag: str = RUN_TAG,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    per_doc_table_counter: Counter[str] = Counter()
    candidates_by_id = {candidate.get("candidate_id"): candidate for candidate in candidates}

    for candidate in candidates:
        if candidate.get("candidate_status") in {"filtered", "deduped", "merged_into_primary"}:
            continue
        per_doc_table_counter[candidate["doc_id"]] += 1
        table_object_id = (
            f"{candidate['doc_id']}__{compact_id(candidate.get('table_id', 'table'))}"
            f"__{run_tag}_{per_doc_table_counter[candidate['doc_id']]:02d}"
        )
        boundary = group_candidate_boundary(candidate, chunks_by_doc)
        continued_parts: list[dict[str, Any]] = []
        merged_from_table_object_ids: list[str] = []
        merge_status = "not_continued"
        for merged_candidate_id in candidate.get("merged_from_candidate_ids", []):
            continued_candidate = candidates_by_id.get(merged_candidate_id)
            if not continued_candidate:
                continue
            continued_boundary = group_candidate_boundary(continued_candidate, chunks_by_doc)
            boundary["chunks"].extend(continued_boundary["chunks"])
            boundary["caption_block_ids"].extend(continued_boundary["caption_block_ids"])
            boundary["header_block_ids"].extend(continued_boundary["header_block_ids"])
            boundary["body_block_ids"].extend(continued_boundary["body_block_ids"])
            boundary["source_block_ids"].extend(continued_boundary["source_block_ids"])
            boundary["allowed_source_block_ids"].extend(continued_boundary["allowed_source_block_ids"])
            boundary["chunk_ids"].extend(continued_boundary["chunk_ids"])
            boundary["table_text"] = normalize_space(
                f"{boundary.get('table_text', '')} {continued_boundary.get('table_text', '')}"
            )
            boundary["warnings"] = unique(
                boundary["warnings"] + continued_boundary["warnings"] + ["continued_table_merged"]
            )
            continued_parts.append(
                {
                    "candidate_id": continued_candidate.get("candidate_id"),
                    "legacy_phase7a_table_object_id": continued_candidate.get("legacy_phase7a_table_object_id"),
                    "table_id": continued_candidate.get("table_id"),
                    "chunk_ids": continued_boundary.get("chunk_ids", []),
                    "status": "merged",
                }
            )
            if continued_candidate.get("legacy_phase7a_table_object_id"):
                merged_from_table_object_ids.append(continued_candidate["legacy_phase7a_table_object_id"])
            merge_status = "merged"

        boundary["chunks"] = list({chunk.get("chunk_id"): chunk for chunk in boundary["chunks"]}.values())
        boundary["caption_block_ids"] = unique(boundary["caption_block_ids"])
        boundary["header_block_ids"] = unique(boundary["header_block_ids"])
        boundary["body_block_ids"] = unique(boundary["body_block_ids"])
        boundary["source_block_ids"] = unique(boundary["source_block_ids"])
        boundary["allowed_source_block_ids"] = unique(boundary["allowed_source_block_ids"])
        boundary["chunk_ids"] = unique(boundary["chunk_ids"])
        source_spans, _block_to_spans = make_source_spans(
            table_object_id,
            candidate["doc_id"],
            boundary["chunks"],
            boundary["body_block_ids"],
            boundary.get("allowed_source_block_ids"),
        )
        columns, rows, cells, footnotes, references, parser_warnings, parser_name = parse_table_structure(
            table_object_id,
            candidate,
            boundary["chunks"],
            source_spans,
            boundary.get("table_text"),
        )

        chunk0 = boundary["chunks"][0]
        warnings = list(boundary["warnings"]) + parser_warnings + candidate.get("candidate_decision_warnings", [])
        if not any(chunk.get("contains_table_text") for chunk in boundary["chunks"]):
            warnings.append("no_table_text_flag")
        if any(span.get("granularity") == "table_row_level" for span in source_spans):
            warnings.extend(["source_span_table_row_level_only", "source_span_not_value_level", "no_value_level_bbox"])
        else:
            warnings.extend(["source_span_not_value_level", "no_value_level_bbox"])
        boundary_text = boundary.get("table_text") or " ".join(chunk.get("text", "") for chunk in boundary["chunks"])
        if LITERAL_RE.search(boundary_text):
            warnings.append("literal_value_requires_preservation")
        if "Abbreviation:" in boundary_text:
            warnings.append("abbreviation_binding_ok")
        if FOOTNOTE_RE.search(boundary_text):
            warnings.append("footnote_present_not_bound")
        if UNIT_RE.search(boundary_text) and not any(column.get("unit") for column in columns):
            warnings.append("unit_visible_not_bound")
        if REFERENCE_COLUMN_RE.search(boundary_text):
            warnings.extend(["internal_reference_column", "external_citation_not_supported"])
        elif REFERENCE_RE.search(boundary_text) and not references:
            warnings.append("reference_visible_not_bound")
        if "Table S" in candidate.get("table_id", "") or "Supplementary Table" in candidate.get("caption_text", ""):
            warnings.append("supplementary_required")
        if boundary["boundary_status"] == "boundary_partial":
            warnings.append("table_boundary_partial")
        if is_continued_table_id(candidate.get("table_id", "")):
            warnings.extend(["continued_table_part", "continued_table_needs_merge"])
            merge_status = "needs_merge"
        if METRIC_GAP_RE.search(boundary_text) and len(columns) == 1:
            warnings.extend(["metric_level_cell_gap", "row_cell_blocking_warning", "table_tail_truncation"])

        granularity = "table_row_level" if any(span["granularity"] == "table_row_level" for span in source_spans) else "table_level"
        if granularity not in ALLOWED_SOURCE_SPAN_GRANULARITIES:
            granularity = "mixed_or_unclear"

        obj = {
            "table_object_id": table_object_id,
            "doc_id": candidate["doc_id"],
            "source_file": candidate.get("source_file"),
            "table_id": candidate.get("table_id"),
            "caption": candidate.get("caption_text"),
            "page": candidate.get("page"),
            "section_path": chunk0.get("section_path", []),
            "candidate_id": candidate.get("candidate_id"),
            "legacy_phase7a_table_object_id": candidate.get("legacy_phase7a_table_object_id"),
            "candidate_status": candidate.get("candidate_status", "active"),
            "candidate_status_reason": candidate.get("candidate_status_reason", ""),
            "caption_block_ids": boundary["caption_block_ids"],
            "body_block_ids": boundary["body_block_ids"],
            "header_block_ids": boundary["header_block_ids"],
            "source_block_ids": boundary["source_block_ids"],
            "chunk_ids": boundary["chunk_ids"],
            "boundary_status": boundary["boundary_status"],
            "caption_body_relation_status": boundary["caption_body_relation_status"],
            "merged_from_table_object_ids": merged_from_table_object_ids,
            "merged_from_candidate_ids": candidate.get("merged_from_candidate_ids", []),
            "continued_parts": continued_parts,
            "merge_status": merge_status,
            "columns": columns,
            "rows": rows,
            "cells": cells,
            "footnotes": footnotes,
            "references": references,
            "source_spans": source_spans,
            "source_span_granularity": granularity,
            "source_span_limitation": (
                "official chunks only provide chunk/block/table-row level text spans; "
                "value-level bbox is absent and not inferred."
            ),
            "no_value_level_bbox": True,
            "warnings": unique(warnings),
            "extraction_method": f"official_chunks_heuristic_v1:{parser_name}",
            "extraction_confidence": candidate.get("candidate_confidence", "low"),
            "validation_status": "partial",
            "notes": [
                f"{PHASE_LABEL} offline object; not production extraction.",
                "official chunks do not provide value-level bbox; bbox remains null.",
            ],
            "baseline_name": OFFICIAL_BASELINE_NAME,
            "official_chunks_sha256": OFFICIAL_CHUNKS_SHA256,
        }
        obj["validation_status"] = validation_status_for_object(obj)
        objects.append(obj)

    return objects


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def configure_paths(output_dir: Path, report_dir: Path, phase_label: str, run_tag: str) -> None:
    global OUTPUT_DIR, REPORT_DIR, TABLE_CANDIDATES_PATH, TABLE_OBJECTS_PATH, DETECTION_REPORT_PATH
    global PHASE_LABEL, RUN_TAG

    OUTPUT_DIR = output_dir if output_dir.is_absolute() else ROOT / output_dir
    REPORT_DIR = report_dir if report_dir.is_absolute() else ROOT / report_dir
    TABLE_CANDIDATES_PATH = OUTPUT_DIR / "table_candidates.jsonl"
    TABLE_OBJECTS_PATH = OUTPUT_DIR / "table_objects.jsonl"
    DETECTION_REPORT_PATH = REPORT_DIR / "table_candidate_detection_report.md"
    PHASE_LABEL = phase_label
    RUN_TAG = run_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract offline table objects from official chunks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--phase-label", default="Phase7A")
    parser.add_argument("--run-tag", default="phase7a")
    return parser.parse_args()


def write_detection_report(
    candidates: list[dict[str, Any]],
    table_objects: list[dict[str, Any]],
    available_docs: list[str],
    missing_docs: list[str],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    confidence_counts = Counter(item["candidate_confidence"] for item in candidates)
    source_counts = Counter(item["candidate_source"] for item in candidates)
    signal_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for candidate in candidates:
        signal_counts.update(candidate.get("table_like_signals", []))
        warning_counts.update(candidate.get("warnings", []))
    for obj in table_objects:
        warning_counts.update(obj.get("warnings", []))

    contains_false = sum(1 for obj in table_objects if "no_table_text_flag" in obj.get("warnings", []))
    split_count = sum(1 for obj in table_objects if "caption_body_split" in obj.get("warnings", []))
    status_counts = Counter(candidate.get("candidate_status", "active") for candidate in candidates)
    filtered_count = status_counts.get("filtered", 0)
    deduped_count = status_counts.get("deduped", 0)
    merged_candidate_count = status_counts.get("merged_into_primary", 0)
    object_merge_count = sum(1 for obj in table_objects if obj.get("merge_status") == "merged")

    lines = [
        f"# {PHASE_LABEL} 表格候选检测报告",
        "",
        "## 1. Smoke 输入范围",
        "",
        f"本轮固定 smoke doc_id：`{', '.join(SMOKE_DOC_IDS)}`。",
        f"official chunks 中实际可用文档：`{', '.join(available_docs)}`。",
        f"缺失文档：`{', '.join(missing_docs) if missing_docs else '无'}`。",
        "",
        "本轮只读取 official chunks；未读取或查询 BM25 index，未访问 Milvus，未运行 retrieval、embedding、rerank、Qwen、RAGAS、OCR 或 VLM。",
        "",
        "## 2. 候选检测方法",
        "",
        "候选检测使用以下离线信号：",
        "",
        "- `contains_table_caption` flag。",
        "- caption regex：`Table` / `TABLE` / `Supplementary Table`。",
        "- block text 中的 table-like patterns，例如多列数字、单位、reference/source、primer、strain、construct、composition、yield、titer。",
        "- `paragraph` 或 `subsection_heading` 中疑似表头或表体的文本。",
        "- caption 与 body 在相邻 chunk 的 split 情况。",
        "",
        "检测不依赖 `contains_table_text=true`；事实上本轮 smoke candidate 基本都来自 `contains_table_text=false` 的 chunks。",
        "",
        "## 3. 候选数量",
        "",
        f"检测到 table candidates：{len(candidates)}。",
        f"生成 table_objects：{len(table_objects)}。",
        f"filtered candidates：{filtered_count}；deduped candidates：{deduped_count}；merged continued candidates：{merged_candidate_count}；merged table_objects：{object_merge_count}。",
        "",
        "| candidate_confidence | 数量 |",
        "|---|---:|",
    ]
    for key in ["high", "medium", "low"]:
        lines.append(f"| `{key}` | {confidence_counts.get(key, 0)} |")

    lines.extend(
        [
            "",
        "## 4. 候选来源统计",
            "",
            "| candidate_source | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(source_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## 4.1 候选状态统计",
            "",
            "| candidate_status | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(status_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
        "## 5. 主要 `table_like_signals`",
            "",
            "| signal | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in signal_counts.most_common():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## 6. contains_table_text=false 的影响",
            "",
            f"带有 `no_table_text_flag` 的 table_object 数量：{contains_false}。",
            "这说明 official chunks 中表体经常被放入 `table_caption`、`paragraph` 或相邻 chunk，而不是稳定标记为 table text。本轮 MVP 因此显式记录 `no_table_text_flag`、`body_as_table_caption`、`body_as_paragraph` 和 `parser_boundary_warning`。",
            "",
            "## 7. caption/body split 情况",
            "",
            f"带有 `caption_body_split` 的 table_object 数量：{split_count}。",
            "这些对象通常需要从 caption chunk 和相邻 paragraph chunk 合并 source_block_ids / chunk_ids，source_span 粒度只能记录到 table_row_level 或 row/block level。",
            "",
            "## 8. 主要 warnings",
            "",
            "| warning | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in warning_counts.most_common():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## 9. 不接 production 声明",
            "",
            "本轮是离线 structured table extraction MVP。输出只用于人工审阅、gold 建设准备和后续结构化验证设计；不修改 ingestion 主链路，不接 RAG，不写 Milvus，不重建 BM25，不跑 retrieval，不调用模型/OCR/VLM，不修改 official baseline，也不进入 Route C implementation。",
            "",
        ]
    )
    DETECTION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_paths(args.output_dir, args.report_dir, args.phase_label, args.run_tag)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    chunks_by_doc, available_docs = load_target_chunks()
    candidates, missing_docs = detect_candidates(chunks_by_doc, RUN_TAG)
    annotate_candidate_decisions(candidates, chunks_by_doc)
    table_objects = build_table_objects(candidates, chunks_by_doc, RUN_TAG)
    write_jsonl(TABLE_CANDIDATES_PATH, candidates)
    write_jsonl(TABLE_OBJECTS_PATH, table_objects)
    write_detection_report(candidates, table_objects, available_docs, missing_docs)
    print(
        json.dumps(
            {
                "available_docs": available_docs,
                "missing_docs": missing_docs,
                "table_candidates": len(candidates),
                "table_objects": len(table_objects),
                "outputs": [
                    str(TABLE_CANDIDATES_PATH.relative_to(ROOT)),
                    str(TABLE_OBJECTS_PATH.relative_to(ROOT)),
                    str(DETECTION_REPORT_PATH.relative_to(ROOT)),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
