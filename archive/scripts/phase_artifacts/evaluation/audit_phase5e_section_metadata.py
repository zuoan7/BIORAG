#!/usr/bin/env python3
"""Phase 5E-1 read-only section metadata audit.

This script reads parsed_clean, chunk JSONL files, and prior reports, then writes
audit artifacts under reports/phase5e_section_metadata_audit. It does not modify
parsed_clean, chunks, indexes, schemas, or cleaning code.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PARSED_CLEAN_DIR = ROOT / "data/paper_round1/parsed_clean"
OUT_DIR = ROOT / "reports/phase5e_section_metadata_audit"

DATASETS = {
    "phase4d_compact": Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
    "phase5c4_full_enhanced": Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    "phase5d3_caption_cleanup": Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
}

SOURCE_REPORTS = [
    ROOT / "reports/phase4_closeout/summary.md",
    ROOT / "reports/phase5c8_closeout/summary.md",
    ROOT / "reports/phase5d_closeout/summary.md",
]

TARGET_MAPPING_FILES = [
    ROOT / "reports/phase5c5_full_retrieval_ab/target_mapping_audit.csv",
    ROOT / "reports/phase5c3_table_expansion/retrieval_ab/target_mapping_audit.csv",
    ROOT / "reports/phase5c2_table_retrieval_ab/stable_target_mapping/target_mapping_audit.csv",
]

PROTECTED_CAPTION_FILES = [
    ROOT / "reports/phase5d_caption_cleanup_experiment/protected_caption_check.csv",
    ROOT / "reports/phase5d_caption_cleanup_signoff/protected_caption_review.csv",
    ROOT / "reports/phase5d_caption_cleanup_audit/protected_short_captions.csv",
]

WEAK_SECTION_NAMES = {"title", "unknown"}
EXCLUDED_SECTION_PATTERNS = [
    r"^references?$",
    r"^acknowledg(e)?ments?$",
    r"^author contributions?$",
    r"^correspondence$",
    r"^funding$",
    r"^metadata$",
    r"^title$",
    r"^journal preproof$",
    r"^running (header|footer)$",
    r"^supporting information$",
    r"^supplementary (data|information|materials?)$",
    r"^open$",
    r"^research$",
    r"^review$",
    r"^copyright:?$",
    r"^correction statement:?$",
    r"^full terms",
    r"^journal homepage",
    r"^issn:",
    r"^doi:",
    r"^received:",
    r"^revised:",
    r"^accepted:",
    r"^published:",
    r"^edited by:",
    r"^academic editor:",
    r"^citation:",
    r"^number of figures:",
    r"^running title:",
    r"^correspondence author:",
    r"^address correspondence",
]
CAPTION_HEADING_RE = re.compile(r"^(fig(?:ure)?\.?|table)\s*(s?\d+|[ivxlcdm]+)\b", re.I)
SECTION_LINE_RE = re.compile(r"(?im)^section:\s*(.*?)\s*$")
CANONICAL_SHORT_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "method",
    "materials",
    "materials and methods",
    "results",
    "discussion",
    "results and discussion",
    "conclusion",
    "conclusions",
}
METADATA_TEXT_RE = re.compile(
    r"(doi:|received:|revised:|accepted:|published:|citation:|copyright|"
    r"journal homepage|full terms|issn:|edited by:|academic editor:|"
    r"correspondence author|address correspondence|wileyonlinelibrary|"
    r"frontiers in|open access)",
    re.I,
)


@dataclass
class BlockRef:
    doc_id: str
    block_id: str
    index: int
    page: int | None
    block_type: str
    text: str
    section_path: list[str]


@dataclass
class DocIndex:
    blocks: list[BlockRef]
    by_id: dict[str, BlockRef]
    headings: list[BlockRef]


def norm_ws(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def preview(value: Any, limit: int = 220) -> str:
    text = norm_ws(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def string_list(value: Any) -> list[str]:
    return [norm_ws(v) for v in list_value(value) if norm_ws(v)]


def json_list(value: Any) -> str:
    return json.dumps(list_value(value), ensure_ascii=False)


def clean_heading_text(text: str) -> str:
    text = norm_ws(text)
    text = re.sub(r"^\[.*?CAPTION\]\s*", "", text, flags=re.I)
    text = re.sub(r"^#+\s*", "", text).strip()
    return text


def is_weak_section_name(section: Any) -> bool:
    sec = norm_ws(section)
    return not sec or sec.lower() in WEAK_SECTION_NAMES


def is_section_path_empty(path: Any) -> bool:
    return len(string_list(path)) == 0


def is_section_path_only_title(path: Any) -> bool:
    parts = string_list(path)
    return len(parts) == 1 and parts[0].lower() == "title"


def credible_section_text(text: Any) -> bool:
    heading = clean_heading_text(norm_ws(text))
    if not heading:
        return False
    low = heading.lower().strip(" .:")
    if low in WEAK_SECTION_NAMES:
        return False
    for pattern in EXCLUDED_SECTION_PATTERNS:
        if re.match(pattern, low, re.I):
            return False
    if CAPTION_HEADING_RE.match(heading):
        return False
    if re.match(r"^表\s*\d+", heading):
        return False
    if METADATA_TEXT_RE.search(heading):
        return False
    if len(heading) > 140 or len(heading.split()) > 18:
        return False
    if len(heading) <= 2:
        return False
    if re.match(r"^[A-Z]$", heading):
        return False
    if re.match(r"^[a-z]+$", heading) and low not in CANONICAL_SHORT_HEADINGS:
        return False
    if re.match(r"^[A-Z][a-z]+$", heading) and low not in CANONICAL_SHORT_HEADINGS:
        return False
    if re.search(r"\b(university|department|institute|school|college|center|centre|laboratory|academy)\b", heading, re.I):
        return False
    if re.search(r"\b(print|online|homepage|publication date|press|springer|elsevier|wiley|taylor)\b", heading, re.I):
        return False
    if re.search(r"\b\d{4}\b", heading) and low not in CANONICAL_SHORT_HEADINGS:
        return False
    if re.search(r"\b(μmol|od600|cfu/ml)\b", heading, re.I):
        return False
    if re.search(r"\b(is|are|was|were|has|have|had|led|resulted|shown|based|redundant|inhibitors)\b", heading, re.I):
        if low not in CANONICAL_SHORT_HEADINGS:
            return False
    if re.match(r"^\d+[A-Z]?,", heading):
        return False
    if re.match(r"^\d+\s+[A-ZX\.\-]{4,}(\s+[A-ZX\.\-]{4,}){2,}$", heading):
        return False
    if re.match(r"^[A-ZX\.\-]{8,}(\s+[A-ZX\.\-]{4,}){2,}$", heading):
        return False
    return True


def is_excluded_context(text: Any) -> bool:
    heading = clean_heading_text(norm_ws(text))
    low = heading.lower().strip(" .:")
    if not heading:
        return True
    for pattern in EXCLUDED_SECTION_PATTERNS:
        if re.match(pattern, low, re.I):
            return True
    return False


def is_heading_block(block: BlockRef) -> bool:
    block_type = block.block_type.lower()
    if block_type == "title":
        return False
    if "caption" in block_type:
        return False
    if "heading" in block_type:
        return True
    return clean_heading_text(block.text).startswith("##")


def latest_credible_path_item(path: Any) -> str | None:
    for item in reversed(string_list(path)):
        cleaned = clean_heading_text(item)
        if credible_section_text(cleaned):
            return cleaned
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parsed_index(parsed_dir: Path) -> dict[str, DocIndex]:
    docs: dict[str, DocIndex] = {}
    for path in sorted(parsed_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_id = data.get("doc_id") or path.stem
        blocks: list[BlockRef] = []
        by_id: dict[str, BlockRef] = {}
        headings: list[BlockRef] = []
        for page_obj in data.get("pages", []):
            page = page_obj.get("page")
            for raw in page_obj.get("blocks", []) or []:
                block_id = norm_ws(raw.get("block_id"))
                if not block_id:
                    continue
                block = BlockRef(
                    doc_id=doc_id,
                    block_id=block_id,
                    index=len(blocks),
                    page=page if page is not None else raw.get("page"),
                    block_type=norm_ws(raw.get("type")),
                    text=norm_ws(raw.get("text")),
                    section_path=string_list(raw.get("section_path")),
                )
                blocks.append(block)
                by_id[block_id] = block
                if is_heading_block(block):
                    headings.append(block)
        docs[doc_id] = DocIndex(blocks=blocks, by_id=by_id, headings=headings)
    return docs


def chunk_types(chunk: dict[str, Any]) -> set[str]:
    types = set()
    for key in ("block_types", "evidence_types"):
        for value in string_list(chunk.get(key)):
            types.add(value.lower())
    for flag, name in [
        ("contains_table_caption", "table_caption"),
        ("contains_table_text", "table_text"),
        ("contains_table_related", "table_related"),
        ("contains_figure_caption", "figure_caption"),
        ("contains_metadata", "metadata"),
        ("contains_references", "reference"),
        ("contains_noise", "noise"),
    ]:
        if chunk.get(flag):
            types.add(name)
    return types


def has_table_or_figure_evidence(chunk: dict[str, Any]) -> bool:
    types = chunk_types(chunk)
    return bool(types & {"table_caption", "table_text", "table_related", "figure_caption"})


def is_metadata_like(chunk: dict[str, Any]) -> bool:
    types = chunk_types(chunk)
    text = norm_ws(chunk.get("text")).lower()
    if chunk.get("contains_metadata") or chunk.get("contains_references") or chunk.get("contains_noise"):
        return True
    if types & {"metadata", "reference", "references", "noise", "title"}:
        return True
    if METADATA_TEXT_RE.search(text[:500]):
        return True
    if "abstract" in text[:120] or "keywords:" in text[:160]:
        return True
    pages = [p for p in list_value(chunk.get("page_numbers")) if isinstance(p, int)]
    if pages and min(pages) == 1 and is_weak_section_name(chunk.get("section")):
        return True
    return False


def retrieval_uses_weak_section(chunk: dict[str, Any]) -> bool:
    retrieval_text = chunk.get("retrieval_text") or ""
    section = norm_ws(chunk.get("section"))
    matches = [m.group(1).strip() for m in SECTION_LINE_RE.finditer(str(retrieval_text))]
    if not matches:
        return False
    if is_weak_section_name(section):
        return any(is_weak_section_name(match) or match == section for match in matches)
    return False


def weak_reasons(chunk: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    section = norm_ws(chunk.get("section"))
    section_path = chunk.get("section_path")
    if not section:
        reasons.append("section_empty")
    elif section.lower() == "title":
        reasons.append("section_title")
    elif section.lower() == "unknown":
        reasons.append("section_unknown")
    if is_section_path_empty(section_path):
        reasons.append("section_path_empty")
    elif is_section_path_only_title(section_path):
        reasons.append("section_path_only_title")
    if reasons and has_table_or_figure_evidence(chunk):
        reasons.append("evidence_no_explainable_body_section")
    return reasons


def weak_severity(chunk: dict[str, Any], reasons: list[str]) -> str:
    if not reasons:
        return ""
    if has_table_or_figure_evidence(chunk) and retrieval_uses_weak_section(chunk):
        return "high"
    if is_metadata_like(chunk):
        return "low"
    types = chunk_types(chunk)
    if "paragraph" in types:
        return "medium"
    if has_table_or_figure_evidence(chunk):
        return "medium"
    return "low"


def bucket_names(chunk: dict[str, Any]) -> list[str]:
    types = chunk_types(chunk)
    buckets: list[str] = []
    for name in ["paragraph", "table_caption", "table_text", "table_related", "figure_caption"]:
        if name in types:
            buckets.append(name)
    if len(types - {"metadata", "reference", "references", "noise"}) > 1:
        buckets.append("mixed_evidence")
    if chunk.get("contains_metadata") or "metadata" in types:
        buckets.append("metadata")
    if chunk.get("contains_references") or "reference" in types or "references" in types:
        buckets.append("reference")
    if chunk.get("contains_noise") or "noise" in types:
        buckets.append("noise")
    if not buckets:
        buckets.append("other")
    return buckets


def analyze_distribution(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(chunks)
    section_counts = Counter(norm_ws(c.get("section")) or "<empty>" for c in chunks)
    weak_rows = []
    type_stats = defaultdict(lambda: Counter(total=0, weak=0, high=0, medium=0, low=0))
    doc_stats = defaultdict(lambda: Counter(total=0, weak=0, high=0, medium=0, low=0))
    title_count = unknown_count = empty_count = 0
    path_only_title_count = path_empty_count = 0

    for chunk in chunks:
        section = norm_ws(chunk.get("section"))
        if section.lower() == "title":
            title_count += 1
        if section.lower() == "unknown":
            unknown_count += 1
        if not section:
            empty_count += 1
        if is_section_path_only_title(chunk.get("section_path")):
            path_only_title_count += 1
        if is_section_path_empty(chunk.get("section_path")):
            path_empty_count += 1

        reasons = weak_reasons(chunk)
        severity = weak_severity(chunk, reasons)
        doc_id = norm_ws(chunk.get("doc_id")) or "<missing_doc_id>"
        doc_stats[doc_id]["total"] += 1
        for bucket in bucket_names(chunk):
            type_stats[bucket]["total"] += 1
        if reasons:
            weak_rows.append((chunk, reasons, severity))
            doc_stats[doc_id]["weak"] += 1
            if severity:
                doc_stats[doc_id][severity] += 1
            for bucket in bucket_names(chunk):
                type_stats[bucket]["weak"] += 1
                if severity:
                    type_stats[bucket][severity] += 1

    severity_counts = Counter(sev for _, _, sev in weak_rows)
    return {
        "total_chunks": total,
        "section_distribution": dict(section_counts.most_common()),
        "section_title_count": title_count,
        "section_unknown_count": unknown_count,
        "section_empty_or_null_count": empty_count,
        "section_path_only_title_count": path_only_title_count,
        "section_path_empty_or_null_count": path_empty_count,
        "weak_section_chunks": len(weak_rows),
        "severity_distribution": dict(severity_counts),
        "weak_by_block_or_evidence_type": {k: dict(v) for k, v in sorted(type_stats.items())},
        "weak_by_doc_id": {k: dict(v) for k, v in sorted(doc_stats.items()) if v["weak"]},
        "top_docs_by_weak_section_count": [
            {"doc_id": doc_id, **dict(counter)}
            for doc_id, counter in sorted(
                doc_stats.items(), key=lambda item: (-item[1]["weak"], item[0])
            )
            if counter["weak"]
        ][:30],
    }


def weak_chunk_row(chunk: dict[str, Any], reasons: list[str], severity: str) -> dict[str, Any]:
    return {
        "doc_id": norm_ws(chunk.get("doc_id")),
        "source_file": norm_ws(chunk.get("source_file")),
        "chunk_id": norm_ws(chunk.get("chunk_id")),
        "section": norm_ws(chunk.get("section")),
        "section_path": json_list(chunk.get("section_path")),
        "chunk_type": norm_ws(chunk.get("chunk_type")),
        "content_kind": norm_ws(chunk.get("content_kind")),
        "block_types": json_list(chunk.get("block_types")),
        "evidence_types": json_list(chunk.get("evidence_types")),
        "contains_table_caption": bool(chunk.get("contains_table_caption")),
        "contains_table_text": bool(chunk.get("contains_table_text")),
        "contains_table_related": bool(chunk.get("contains_table_related")),
        "contains_figure_caption": bool(chunk.get("contains_figure_caption")),
        "page_numbers": json_list(chunk.get("page_numbers")),
        "source_block_ids": json_list(chunk.get("source_block_ids") or chunk.get("block_ids")),
        "text_preview": preview(chunk.get("text"), 320),
        "retrieval_text_preview": preview(chunk.get("retrieval_text"), 360),
        "weak_section_reason": ";".join(reasons),
        "severity": severity,
    }


def candidate_from_chunk_path(chunk: dict[str, Any]) -> dict[str, Any] | None:
    candidate = latest_credible_path_item(chunk.get("section_path"))
    if not candidate:
        return None
    return {
        "candidate_section": candidate,
        "candidate_source": "section_path_parent",
        "candidate_block_id": "",
        "candidate_distance_blocks": "",
        "candidate_page_distance": "",
        "nearby_heading_preview": candidate,
    }


def block_context_for_chunk(chunk: dict[str, Any], doc_index: DocIndex | None) -> tuple[list[BlockRef], int | None, int | None]:
    if doc_index is None:
        return [], None, None
    block_ids = string_list(chunk.get("source_block_ids") or chunk.get("block_ids"))
    blocks = [doc_index.by_id[b] for b in block_ids if b in doc_index.by_id]
    if not blocks:
        return [], None, None
    first = min(blocks, key=lambda b: b.index)
    pages = [b.page for b in blocks if b.page is not None]
    page = min(pages) if pages else first.page
    return blocks, first.index, page


def block_distance(first_index: int | None, candidate: BlockRef) -> int | None:
    if first_index is None:
        return None
    return abs(first_index - candidate.index)


def page_distance(page: int | None, candidate: BlockRef) -> int | None:
    if page is None or candidate.page is None:
        return None
    return abs(page - candidate.page)


def confidence_for_distance(source: str, distance: int | None, pages: int | None) -> str:
    if source == "section_path_parent":
        return "high"
    distance = 999999 if distance is None else distance
    pages = 999999 if pages is None else pages
    if source in {"previous_heading", "same_page_heading"} and distance <= 20 and pages <= 1:
        return "high"
    if distance <= 60 and pages <= 3:
        return "medium"
    if distance <= 120 and pages <= 5:
        return "low"
    return "none"


def repair_recommendation(chunk: dict[str, Any], candidate: dict[str, Any] | None, confidence: str, severity: str) -> tuple[str, str]:
    if candidate is None or confidence == "none":
        return "do_not_repair", "no credible nearby body heading found"
    if is_metadata_like(chunk):
        return "do_not_repair", "chunk is title/front matter/metadata/reference/noise-like"
    candidate_section = candidate.get("candidate_section", "")
    if is_excluded_context(candidate_section):
        return "do_not_repair", "candidate is an excluded non-body section"
    if confidence == "high":
        return "safe_to_repair", f"{candidate.get('candidate_source')} gives a nearby credible body heading"
    if confidence == "medium":
        return "needs_manual_check", "candidate is plausible but not close enough for automatic repair"
    if severity == "high":
        return "needs_manual_check", "weak evidence chunk has only a low-confidence nearby heading"
    return "do_not_repair", "candidate confidence is too low for a non-critical chunk"


def find_repair_candidate(chunk: dict[str, Any], parsed_indexes: dict[str, DocIndex], severity: str) -> dict[str, Any]:
    doc_id = norm_ws(chunk.get("doc_id"))
    doc_index = parsed_indexes.get(doc_id)
    blocks, first_index, page = block_context_for_chunk(chunk, doc_index)

    candidate = candidate_from_chunk_path(chunk)
    if candidate is None and doc_index is not None and first_index is not None:
        previous = [
            h
            for h in doc_index.headings
            if h.index < first_index and first_index - h.index <= 120 and credible_section_text(h.text)
        ]
        if previous:
            h = max(previous, key=lambda b: b.index)
            candidate = {
                "candidate_section": clean_heading_text(h.text),
                "candidate_source": "previous_heading",
                "candidate_block_id": h.block_id,
                "candidate_distance_blocks": block_distance(first_index, h),
                "candidate_page_distance": page_distance(page, h),
                "nearby_heading_preview": preview(h.text, 180),
            }

    if candidate is None and doc_index is not None and page is not None:
        same_page = [
            h
            for h in doc_index.headings
            if h.page == page and credible_section_text(h.text)
        ]
        if same_page:
            h = min(same_page, key=lambda b: block_distance(first_index, b) or 999999)
            candidate = {
                "candidate_section": clean_heading_text(h.text),
                "candidate_source": "same_page_heading",
                "candidate_block_id": h.block_id,
                "candidate_distance_blocks": block_distance(first_index, h),
                "candidate_page_distance": page_distance(page, h),
                "nearby_heading_preview": preview(h.text, 180),
            }

    if candidate is None and doc_index is not None and first_index is not None:
        lo = max(0, first_index - 10)
        hi = min(len(doc_index.blocks), first_index + 11)
        nearby = doc_index.blocks[lo:hi]
        path_candidates = []
        for block in nearby:
            item = latest_credible_path_item(block.section_path)
            if item:
                path_candidates.append((block, item))
        if path_candidates:
            h, item = min(path_candidates, key=lambda pair: block_distance(first_index, pair[0]) or 999999)
            candidate = {
                "candidate_section": item,
                "candidate_source": "nearby_block",
                "candidate_block_id": h.block_id,
                "candidate_distance_blocks": block_distance(first_index, h),
                "candidate_page_distance": page_distance(page, h),
                "nearby_heading_preview": item,
            }

    if candidate is None:
        confidence = "none"
        recommendation, rationale = repair_recommendation(chunk, None, confidence, severity)
        candidate = {
            "candidate_section": "",
            "candidate_source": "none",
            "candidate_block_id": "",
            "candidate_distance_blocks": "",
            "candidate_page_distance": "",
            "nearby_heading_preview": "",
        }
    else:
        confidence = confidence_for_distance(
            str(candidate.get("candidate_source")),
            candidate.get("candidate_distance_blocks") if isinstance(candidate.get("candidate_distance_blocks"), int) else None,
            candidate.get("candidate_page_distance") if isinstance(candidate.get("candidate_page_distance"), int) else None,
        )
        recommendation, rationale = repair_recommendation(chunk, candidate, confidence, severity)

    return {
        "doc_id": doc_id,
        "chunk_id": norm_ws(chunk.get("chunk_id")),
        "current_section": norm_ws(chunk.get("section")),
        "current_section_path": json_list(chunk.get("section_path")),
        "candidate_section": candidate.get("candidate_section", ""),
        "candidate_source": candidate.get("candidate_source", "none"),
        "candidate_block_id": candidate.get("candidate_block_id", ""),
        "candidate_distance_blocks": candidate.get("candidate_distance_blocks", ""),
        "candidate_page_distance": candidate.get("candidate_page_distance", ""),
        "confidence": confidence,
        "repair_recommendation": recommendation,
        "rationale": rationale,
        "nearby_heading_preview": candidate.get("nearby_heading_preview", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_section_distribution_md(path: Path, primary: dict[str, Any], all_stats: dict[str, Any], missing: list[str]) -> None:
    lines = [
        "# Phase 5E Section Metadata Distribution",
        "",
        "Primary audit dataset: `phase4d_compact`.",
        "",
        "## Input Availability",
    ]
    for name, dataset_path in DATASETS.items():
        status = "present" if dataset_path.exists() else "missing"
        lines.append(f"- {name}: {status} `{dataset_path}`")
    for report in SOURCE_REPORTS:
        status = "present" if report.exists() else "missing"
        lines.append(f"- {report.relative_to(ROOT)}: {status}")
    if missing:
        lines.extend(["", "Missing optional inputs:", *[f"- {item}" for item in missing]])

    lines.extend(
        [
            "",
            "## Primary Counts",
            f"- total_chunks: {primary['total_chunks']}",
            f"- weak_section_chunks: {primary['weak_section_chunks']}",
            f"- section == Title: {primary['section_title_count']}",
            f"- section == Unknown: {primary['section_unknown_count']}",
            f"- section empty/null: {primary['section_empty_or_null_count']}",
            f"- section_path only Title: {primary['section_path_only_title_count']}",
            f"- section_path empty/null: {primary['section_path_empty_or_null_count']}",
            "",
            "## Severity",
        ]
    )
    for key in ["high", "medium", "low"]:
        lines.append(f"- {key}: {primary.get('severity_distribution', {}).get(key, 0)}")

    lines.extend(["", "## Weak By Block/Evidence Type", "", "| type | total | weak | high | medium | low |", "|---|---:|---:|---:|---:|---:|"])
    for bucket, stats in primary["weak_by_block_or_evidence_type"].items():
        lines.append(
            f"| {bucket} | {stats.get('total', 0)} | {stats.get('weak', 0)} | "
            f"{stats.get('high', 0)} | {stats.get('medium', 0)} | {stats.get('low', 0)} |"
        )

    lines.extend(["", "## Top Docs By Weak Section Count", "", "| doc_id | total | weak | high | medium | low |", "|---|---:|---:|---:|---:|---:|"])
    for row in primary["top_docs_by_weak_section_count"][:20]:
        lines.append(
            f"| {row['doc_id']} | {row.get('total', 0)} | {row.get('weak', 0)} | "
            f"{row.get('high', 0)} | {row.get('medium', 0)} | {row.get('low', 0)} |"
        )

    lines.extend(["", "## Dataset Comparison", "", "| dataset | total | weak | Title | Unknown | empty | high |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, stats in all_stats.items():
        lines.append(
            f"| {name} | {stats.get('total_chunks', 0)} | {stats.get('weak_section_chunks', 0)} | "
            f"{stats.get('section_title_count', 0)} | {stats.get('section_unknown_count', 0)} | "
            f"{stats.get('section_empty_or_null_count', 0)} | {stats.get('severity_distribution', {}).get('high', 0)} |"
        )

    lines.extend(["", "## Top Section Values", "", "| section | count |", "|---|---:|"])
    for section, count in list(primary["section_distribution"].items())[:30]:
        lines.append(f"| {section} | {count} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_target_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in TARGET_MAPPING_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


def chunk_lookup_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {norm_ws(c.get("chunk_id")): c for c in chunks}


def chunk_lookup_by_block(chunks: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = norm_ws(chunk.get("doc_id"))
        for block_id in string_list(chunk.get("source_block_ids") or chunk.get("block_ids")):
            lookup[(doc_id, block_id)].append(chunk)
    return lookup


def protected_chunk_status(
    dataset: str,
    chunk: dict[str, Any] | None,
    candidates: dict[str, dict[str, Any]],
    category: str,
    sample_id: str,
    query_type: str = "",
    source: str = "",
) -> dict[str, Any]:
    if chunk is None:
        return {
            "category": category,
            "sample_id": sample_id,
            "query_type": query_type,
            "dataset": dataset,
            "source": source,
            "doc_id": "",
            "chunk_id": "",
            "section": "",
            "section_path": "[]",
            "weak_section": "",
            "severity": "",
            "candidate_section": "",
            "repair_recommendation": "",
            "retrieval_text_uses_weak_section": "",
            "retrieval_explainability_impact": "not_located",
            "text_preview": "",
        }
    reasons = weak_reasons(chunk)
    severity = weak_severity(chunk, reasons)
    candidate = candidates.get(norm_ws(chunk.get("chunk_id")), {})
    uses_weak = retrieval_uses_weak_section(chunk)
    if reasons and has_table_or_figure_evidence(chunk):
        impact = "yes_weak_evidence_section_in_retrieval_text" if uses_weak else "possible_weak_evidence_section"
    elif reasons:
        impact = "possible_weak_context"
    else:
        impact = "no"
    return {
        "category": category,
        "sample_id": sample_id,
        "query_type": query_type,
        "dataset": dataset,
        "source": source,
        "doc_id": norm_ws(chunk.get("doc_id")),
        "chunk_id": norm_ws(chunk.get("chunk_id")),
        "section": norm_ws(chunk.get("section")),
        "section_path": json_list(chunk.get("section_path")),
        "weak_section": bool(reasons),
        "severity": severity,
        "candidate_section": candidate.get("candidate_section", ""),
        "repair_recommendation": candidate.get("repair_recommendation", ""),
        "retrieval_text_uses_weak_section": uses_weak,
        "retrieval_explainability_impact": impact,
        "text_preview": preview(chunk.get("text"), 260),
    }


def find_doc0367_figure5(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for chunk in chunks:
        if norm_ws(chunk.get("doc_id")) != "doc_0367":
            continue
        text = norm_ws(chunk.get("text"))
        if re.search(r"\b(Figure|Fig\.?)\s*5\b", text, re.I):
            hits.append(chunk)
    hits.sort(key=lambda c: (not bool(c.get("contains_figure_caption")), norm_ws(c.get("chunk_id"))))
    return hits


def build_protected_sample_outputs(
    all_chunks: dict[str, list[dict[str, Any]]],
    primary_candidates: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    lookups = {name: chunk_lookup_by_id(chunks) for name, chunks in all_chunks.items()}
    block_lookups = {name: chunk_lookup_by_block(chunks) for name, chunks in all_chunks.items()}

    candidate_maps = {name: {} for name in all_chunks}
    candidate_maps["phase4d_compact"] = primary_candidates

    for dataset, chunks in all_chunks.items():
        for index, chunk in enumerate(find_doc0367_figure5(chunks), start=1):
            rows.append(
                protected_chunk_status(
                    dataset,
                    chunk,
                    candidate_maps.get(dataset, {}),
                    "doc_0367_figure5",
                    f"doc_0367_figure5_{index}",
                    "figure_caption" if chunk.get("contains_figure_caption") else "figure_context",
                    "direct_chunk_text_search",
                )
            )

    target_rows = load_target_rows()
    seen_targets = set()
    for target in target_rows:
        sample_id = norm_ws(target.get("sample_id"))
        query_type = norm_ws(target.get("query_type"))
        source = norm_ws(target.get("_source"))
        chunk_fields = [
            ("phase4d_compact", "corrected_baseline_target_chunk_id"),
            ("phase5c4_full_enhanced", "corrected_enhanced_target_chunk_id"),
            ("phase4d_compact", "original_baseline_target_chunk_id"),
            ("phase5c4_full_enhanced", "original_enhanced_target_chunk_id"),
        ]
        for dataset, field in chunk_fields:
            chunk_id = norm_ws(target.get(field))
            key = (sample_id, query_type, source, dataset, chunk_id)
            if not chunk_id or key in seen_targets:
                continue
            seen_targets.add(key)
            chunk = lookups.get(dataset, {}).get(chunk_id)
            rows.append(
                protected_chunk_status(
                    dataset,
                    chunk,
                    candidate_maps.get(dataset, {}),
                    "phase4_5c_eval_target",
                    sample_id,
                    query_type,
                    source,
                )
            )

    for path in PROTECTED_CAPTION_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for i, row in enumerate(reader, start=1):
                doc_id = norm_ws(row.get("doc_id"))
                block_id = norm_ws(row.get("block_id"))
                protect_reason = norm_ws(row.get("protect_reason"))
                if not doc_id or not block_id:
                    continue
                for dataset in ("phase4d_compact", "phase5d3_caption_cleanup"):
                    chunks = block_lookups.get(dataset, {}).get((doc_id, block_id), [])
                    chunk = chunks[0] if chunks else None
                    rows.append(
                        protected_chunk_status(
                            dataset,
                            chunk,
                            candidate_maps.get(dataset, {}),
                            "phase5d_protected_caption",
                            f"{path.stem}_{i}",
                            protect_reason,
                            str(path.relative_to(ROOT)),
                        )
                    )

    # Markdown summary.
    doc0367_rows = [r for r in rows if r["category"] == "doc_0367_figure5"]
    primary_doc0367 = [
        r
        for r in doc0367_rows
        if r["dataset"] == "phase4d_compact" and r["query_type"] == "figure_caption"
    ]
    chosen_doc0367 = primary_doc0367[0] if primary_doc0367 else (doc0367_rows[0] if doc0367_rows else None)

    target_rows_out = [r for r in rows if r["category"] == "phase4_5c_eval_target" and r["chunk_id"]]
    by_query = defaultdict(lambda: Counter(total=0, weak=0, weak_impact=0))
    for row in target_rows_out:
        query_type = row["query_type"] or "unknown"
        by_query[query_type]["total"] += 1
        if row["weak_section"] == "True" or row["weak_section"] is True:
            by_query[query_type]["weak"] += 1
        if str(row["retrieval_explainability_impact"]).startswith("yes"):
            by_query[query_type]["weak_impact"] += 1

    lines = [
        "# Protected Sample Section Check",
        "",
        "## doc_0367 Figure 5",
    ]
    if chosen_doc0367:
        lines.extend(
            [
                f"- current dataset: {chosen_doc0367['dataset']}",
                f"- chunk_id: {chosen_doc0367['chunk_id']}",
                f"- section: {chosen_doc0367['section']}",
                f"- section_path: {chosen_doc0367['section_path']}",
                f"- weak_section: {chosen_doc0367['weak_section']}",
                f"- severity: {chosen_doc0367['severity']}",
                f"- candidate_section: {chosen_doc0367['candidate_section'] or 'none'}",
                f"- repair_recommendation: {chosen_doc0367['repair_recommendation'] or 'none'}",
                f"- retrieval_text_explainability_impact: {chosen_doc0367['retrieval_explainability_impact']}",
            ]
        )
    else:
        lines.append("- not located")

    lines.extend(["", "## Key Target Weak Section Ratios", "", "| query_type | located_rows | weak | weak_ratio | retrieval_text_impact |", "|---|---:|---:|---:|---:|"])
    for query_type, counts in sorted(by_query.items()):
        total = counts["total"]
        weak = counts["weak"]
        ratio = weak / total if total else 0.0
        lines.append(f"| {query_type} | {total} | {weak} | {ratio:.3f} | {counts['weak_impact']} |")

    protected_caption_rows = [r for r in rows if r["category"] == "phase5d_protected_caption" and r["chunk_id"]]
    weak_caption = sum(1 for r in protected_caption_rows if r["weak_section"] == "True" or r["weak_section"] is True)
    impact_caption = sum(1 for r in protected_caption_rows if str(r["retrieval_explainability_impact"]).startswith("yes"))
    lines.extend(
        [
            "",
            "## Phase 5D Protected Caption Rows",
            f"- located_rows: {len(protected_caption_rows)}",
            f"- weak_section_rows: {weak_caption}",
            f"- retrieval_text_impact_rows: {impact_caption}",
            "",
            "Interpretation: rows with `yes_weak_evidence_section_in_retrieval_text` have evidence-context explainability impact; rows with `no` do not.",
        ]
    )

    return rows, "\n".join(lines) + "\n"


def example_block(title: str, rows: list[dict[str, Any]], candidates: dict[str, dict[str, Any]], limit: int) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("- none")
        lines.append("")
        return lines
    for i, row in enumerate(rows[:limit], start=1):
        chunk_id = row["chunk_id"]
        candidate = candidates.get(chunk_id, {})
        lines.extend(
            [
                f"### {i}. {row['doc_id']} / {chunk_id}",
                f"- current_section: {row['section']}",
                f"- section_path: {row['section_path']}",
                f"- candidate_section: {candidate.get('candidate_section', '') or 'none'}",
                f"- block_types: {row['block_types']}",
                f"- evidence_types: {row['evidence_types']}",
                f"- text_preview: {row['text_preview']}",
                f"- nearby_heading_preview: {candidate.get('nearby_heading_preview', '') or 'none'}",
                f"- recommendation: {candidate.get('repair_recommendation', '') or 'none'}",
                f"- rationale: {candidate.get('rationale', '') or 'none'}",
                "",
            ]
        )
    return lines


def write_examples(path: Path, weak_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    candidates = {r["chunk_id"]: r for r in candidate_rows}
    high = [r for r in weak_rows if r["severity"] == "high"]
    medium = [r for r in weak_rows if r["severity"] == "medium"]
    safe = [r for r in weak_rows if candidates.get(r["chunk_id"], {}).get("repair_recommendation") == "safe_to_repair"]
    do_not = [r for r in weak_rows if candidates.get(r["chunk_id"], {}).get("repair_recommendation") == "do_not_repair"]
    manual = [r for r in weak_rows if candidates.get(r["chunk_id"], {}).get("repair_recommendation") == "needs_manual_check"]
    lines = ["# Phase 5E Section Metadata Examples", ""]
    lines += example_block("High Severity Weak Section Chunks", high, candidates, 20)
    lines += example_block("Medium Severity Weak Section Chunks", medium, candidates, 20)
    lines += example_block("Safe To Repair Candidates", safe, candidates, 20)
    lines += example_block("Do Not Repair Cases", do_not, candidates, 20)
    lines += example_block("Needs Manual Check Cases", manual, candidates, 10)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_repair_strategy(path: Path, primary: dict[str, Any], recommendation_counts: Counter, high_safe: int) -> None:
    weak = primary["weak_section_chunks"]
    high = primary.get("severity_distribution", {}).get("high", 0)
    safe = recommendation_counts.get("safe_to_repair", 0)
    manual = recommendation_counts.get("needs_manual_check", 0)
    do_not = recommendation_counts.get("do_not_repair", 0)
    recommend_5e2 = high > 0 and safe > 0 and high_safe > 0
    lines = [
        "# Phase 5E Repair Strategy Proposal",
        "",
        "## 1. Should Phase 5E-2 Proceed?",
        "",
        f"Recommendation: {'yes' if recommend_5e2 else 'no'}.",
        "",
        f"Rationale: weak_section_chunks={weak}, high_severity={high}, safe_to_repair={safe}, high_severity_safe_to_repair={high_safe}, needs_manual_check={manual}, do_not_repair={do_not}.",
        "",
        "## 2. Repairable Scenarios",
        "",
        "- `section == Title` where a previous credible heading is nearby.",
        "- table/figure evidence chunks with weak section where same-page or nearby parsed_clean blocks carry a credible body section.",
        "- `section_path == [\"Title\"]` where parsed_clean has a non-Title heading within a short block/page distance.",
        "",
        "## 3. Scenarios That Should Not Be Repaired",
        "",
        "- No nearby credible heading exists.",
        "- The nearby heading is References, Acknowledgements, Author Contributions, Correspondence, Funding, Metadata, Title, journal preproof noise, or running header/footer.",
        "- Candidate requires a long cross-page inference.",
        "- Candidate can only be guessed from keywords in the paragraph.",
        "- Front matter/title/abstract boundary is unclear.",
        "",
        "## 4. Recommended Handling For A Later Implementation",
        "",
        "- Do not modify original parsed_clean `section_path`.",
        "- Add optional repair metadata on chunk metadata or retrieval-section selection only.",
        "- Preserve `original_section` and original `section_path`.",
        "- Record `section_repair_rule_id`.",
        "- Record `section_repair_confidence`.",
        "- Keep the behavior default-off or under an experiment output path.",
        "- Keep audit-first review before any production path change.",
        "",
        "## 5. Stop Conditions",
        "",
        "- `safe_to_repair` count is too small to justify implementation.",
        "- `needs_manual_check` dominates the candidate pool.",
        "- False repair risk is high due to noisy nearby headings.",
        "- No material retrieval_text explainability impact is found.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    primary: dict[str, Any],
    recommendation_counts: Counter,
    protected_rows: list[dict[str, Any]],
    high_safe: int,
) -> None:
    high = primary.get("severity_distribution", {}).get("high", 0)
    medium = primary.get("severity_distribution", {}).get("medium", 0)
    low = primary.get("severity_distribution", {}).get("low", 0)
    table_figure_weak = 0
    for bucket in ["table_caption", "table_text", "table_related", "figure_caption"]:
        table_figure_weak += primary["weak_by_block_or_evidence_type"].get(bucket, {}).get("weak", 0)
    weak = primary["weak_section_chunks"]
    evidence_concentrated = table_figure_weak >= (weak * 0.5) if weak else False

    protected_weak = [
        r for r in protected_rows if r.get("weak_section") == "True" or r.get("weak_section") is True
    ]
    protected_impact = [
        r for r in protected_rows if str(r.get("retrieval_explainability_impact", "")).startswith("yes")
    ]
    doc0367 = [
        r
        for r in protected_rows
        if r.get("category") == "doc_0367_figure5"
        and r.get("dataset") == "phase4d_compact"
        and r.get("query_type") == "figure_caption"
    ]
    if not doc0367:
        doc0367 = [r for r in protected_rows if r.get("category") == "doc_0367_figure5"]
    doc0367_row = doc0367[0] if doc0367 else None
    recommend_5e2 = high > 0 and recommendation_counts.get("safe_to_repair", 0) > 0 and high_safe > 0
    recommended_scope = (
        "limit to high-severity evidence chunks with high-confidence nearby body headings"
        if recommend_5e2
        else "no implementation scope recommended from this audit"
    )
    no_repair_reason = (
        ""
        if recommend_5e2
        else "safe high-impact repair candidates were insufficient or evidence impact was low"
    )

    lines = [
        "# Phase 5E-1 Section Metadata Weakness Audit Summary",
        "",
        f"1. total chunks: {primary['total_chunks']}",
        f"2. weak section chunks: {weak}",
        f"3. section == Title / Unknown / empty: {primary['section_title_count']} / {primary['section_unknown_count']} / {primary['section_empty_or_null_count']}",
        f"4. high/medium/low severity: {high} / {medium} / {low}",
        f"5. weak section mainly concentrated in table/figure evidence: {'yes' if evidence_concentrated else 'no'} (table/figure weak bucket hits={table_figure_weak})",
        f"6. safe_to_repair / needs_manual_check / do_not_repair: {recommendation_counts.get('safe_to_repair', 0)} / {recommendation_counts.get('needs_manual_check', 0)} / {recommendation_counts.get('do_not_repair', 0)}",
        f"7. protected samples affected: {'yes' if protected_weak else 'no'} (weak_rows={len(protected_weak)}, retrieval_text_impact_rows={len(protected_impact)})",
    ]
    if doc0367_row:
        lines.append(
            "8. doc_0367 Figure 5 section: "
            f"{doc0367_row['section']}; weak={doc0367_row['weak_section']}; "
            f"candidate={doc0367_row['candidate_section'] or 'none'}; "
            f"recommendation={doc0367_row['repair_recommendation'] or 'none'}"
        )
    else:
        lines.append("8. doc_0367 Figure 5 section: not located")
    lines.extend(
        [
            f"9. recommend Phase 5E-2: {'yes' if recommend_5e2 else 'no'}",
            f"10. recommended repair scope: {recommended_scope}",
            f"11. if not recommended, reason: {no_repair_reason or 'not applicable'}",
            "12. need rebuild index: no",
            "13. need schema change: no",
            "14. need Qwen/RAGAS: no",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    all_chunks: dict[str, list[dict[str, Any]]] = {}
    all_stats: dict[str, Any] = {}
    for name, path in DATASETS.items():
        if not path.exists():
            missing.append(str(path))
            continue
        chunks = load_jsonl(path)
        all_chunks[name] = chunks
        all_stats[name] = analyze_distribution(chunks)

    for report in SOURCE_REPORTS:
        if not report.exists():
            missing.append(str(report))
    for report in SOURCE_REPORTS:
        if report.exists():
            # Read to satisfy audit input requirements and record availability.
            _ = report.read_text(encoding="utf-8", errors="replace")

    if "phase4d_compact" not in all_chunks:
        raise SystemExit("Missing required primary chunks: /tmp/biorag_phase4d_compact_chunks/chunks.jsonl")

    parsed_indexes = build_parsed_index(PARSED_CLEAN_DIR)
    primary_chunks = all_chunks["phase4d_compact"]
    primary_stats = all_stats["phase4d_compact"]

    weak_rows_for_csv: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for chunk in primary_chunks:
        reasons = weak_reasons(chunk)
        if not reasons:
            continue
        severity = weak_severity(chunk, reasons)
        weak_rows_for_csv.append(weak_chunk_row(chunk, reasons, severity))
        candidate_rows.append(find_repair_candidate(chunk, parsed_indexes, severity))

    candidate_by_chunk = {row["chunk_id"]: row for row in candidate_rows}
    recommendation_counts = Counter(row["repair_recommendation"] for row in candidate_rows)
    high_safe = sum(
        1
        for row in weak_rows_for_csv
        if row["severity"] == "high"
        and candidate_by_chunk.get(row["chunk_id"], {}).get("repair_recommendation") == "safe_to_repair"
    )

    distribution = {
        "primary_dataset": "phase4d_compact",
        "inputs": {
            "parsed_clean_dir": str(PARSED_CLEAN_DIR),
            "datasets": {name: str(path) for name, path in DATASETS.items()},
            "source_reports": [str(path) for path in SOURCE_REPORTS],
            "missing_optional_inputs": missing,
        },
        "datasets": all_stats,
    }
    (OUT_DIR / "section_distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_section_distribution_md(OUT_DIR / "section_distribution.md", primary_stats, all_stats, missing)

    weak_fields = [
        "doc_id",
        "source_file",
        "chunk_id",
        "section",
        "section_path",
        "chunk_type",
        "content_kind",
        "block_types",
        "evidence_types",
        "contains_table_caption",
        "contains_table_text",
        "contains_table_related",
        "contains_figure_caption",
        "page_numbers",
        "source_block_ids",
        "text_preview",
        "retrieval_text_preview",
        "weak_section_reason",
        "severity",
    ]
    write_csv(OUT_DIR / "weak_section_chunks.csv", weak_rows_for_csv, weak_fields)

    candidate_fields = [
        "doc_id",
        "chunk_id",
        "current_section",
        "current_section_path",
        "candidate_section",
        "candidate_source",
        "candidate_block_id",
        "candidate_distance_blocks",
        "candidate_page_distance",
        "confidence",
        "repair_recommendation",
        "rationale",
    ]
    write_csv(OUT_DIR / "section_repair_candidates.csv", candidate_rows, candidate_fields)

    protected_rows, protected_md = build_protected_sample_outputs(all_chunks, candidate_by_chunk)
    protected_fields = [
        "category",
        "sample_id",
        "query_type",
        "dataset",
        "source",
        "doc_id",
        "chunk_id",
        "section",
        "section_path",
        "weak_section",
        "severity",
        "candidate_section",
        "repair_recommendation",
        "retrieval_text_uses_weak_section",
        "retrieval_explainability_impact",
        "text_preview",
    ]
    write_csv(OUT_DIR / "protected_sample_section_check.csv", protected_rows, protected_fields)
    (OUT_DIR / "protected_sample_section_check.md").write_text(protected_md, encoding="utf-8")

    write_examples(OUT_DIR / "examples.md", weak_rows_for_csv, candidate_rows)
    write_repair_strategy(OUT_DIR / "repair_strategy_proposal.md", primary_stats, recommendation_counts, high_safe)
    write_summary(OUT_DIR / "summary.md", primary_stats, recommendation_counts, protected_rows, high_safe)

    print(json.dumps({
        "out_dir": str(OUT_DIR),
        "total_chunks": primary_stats["total_chunks"],
        "weak_section_chunks": primary_stats["weak_section_chunks"],
        "section_title_count": primary_stats["section_title_count"],
        "section_unknown_count": primary_stats["section_unknown_count"],
        "section_empty_or_null_count": primary_stats["section_empty_or_null_count"],
        "high_severity": primary_stats.get("severity_distribution", {}).get("high", 0),
        "safe_to_repair": recommendation_counts.get("safe_to_repair", 0),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
