#!/usr/bin/env python3
"""Phase 5D-1 read-only audit for false or fragment table/figure captions."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PARSED_RAW_DIR = ROOT / "data/paper_round1/parsed_raw"
PARSED_CLEAN_DIR = ROOT / "data/paper_round1/parsed_clean"
PHASE4D_CHUNKS = Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
PHASE5C_CHUNKS = Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl")
OUT_DIR = ROOT / "reports/phase5d_caption_cleanup_audit"

REQUIRED_REPORTS = [
    ROOT / "reports/phase4_closeout/summary.md",
    ROOT / "reports/table_figure_retrieval_text/phase4d_signoff/remaining_risks.md",
    ROOT / "reports/table_figure_retrieval_eval/phase4e1_failure_diagnosis/failure_ledger.csv",
    ROOT / "reports/phase5c8_closeout/backlog.md",
]

APPROVED_EVAL_FILES = [
    ROOT / "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
    ROOT / "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/sanity_anchor.jsonl",
]

CAPTION_TYPES = {"table_caption", "figure_caption"}
SHORT_CHAR_LIMIT = 80
SHORT_WORD_LIMIT = 12

CAPTION_PREFIX_RE = re.compile(
    r"^\s*(table|fig(?:ure)?)\.?\s+(s?\d+[a-z]?)\s*[:.]?\s*",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(
    r"^\s*(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*$",
    re.IGNORECASE,
)
SINGLE_LETTER_RE = re.compile(
    r"^\s*(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*([A-Z])\s*\.?\s*$",
    re.IGNORECASE,
)
ARTICLE_SINGLE_LETTER_RE = re.compile(
    r"^\s*(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*(the|a|an)\s+([A-Z])\s*\.?\s*$",
    re.IGNORECASE,
)
SUPPLEMENTARY_REFERENCE_RE = re.compile(
    r"\b(supplementary|supporting information)\s+(table|fig(?:ure)?)\s+s?\d+\b",
    re.IGNORECASE,
)

ORGANISM_CONTINUATIONS = {
    "coli",
    "pneumoniae",
    "cerevisiae",
    "phaffii",
    "pastoris",
    "subtilis",
    "lactis",
    "meliloti",
    "typhimurium",
    "aureus",
    "aeruginosa",
    "putida",
    "glutamicum",
}

SEMANTIC_ANCHORS = {
    "workflow",
    "overview",
    "strains",
    "strain",
    "plasmids",
    "plasmid",
    "primers",
    "primer",
    "pathway",
    "model",
    "schematic",
    "alignment",
    "time course",
    "summary",
    "list",
    "comparison",
    "effect",
    "effects",
    "analysis",
    "assay",
    "structure",
    "sequence",
    "expression",
    "production",
    "biosynthesis",
    "fermentation",
    "growth",
    "metabolism",
    "activity",
    "distribution",
    "abundance",
    "profile",
    "profiles",
    "parameters",
    "characterization",
    "characterisation",
    "genes",
    "gene",
    "protein",
    "proteins",
    "enzyme",
    "enzymes",
    "cell",
    "cells",
    "mutant",
    "mutants",
    "construct",
    "constructs",
    "design",
    "network",
    "taxonomy",
    "phylogeny",
}

LOW_INFORMATION_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "with",
    "from",
    "by",
    "as",
    "at",
}


@dataclass
class CaptionBlock:
    doc_id: str
    source_file: str
    block_id: str
    block_type: str
    caption_text: str
    page: int | None
    section_path: list[str]
    previous_block_preview: str
    next_block_preview: str
    nearby_text_preview: str


def norm_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def preview_block(block: dict[str, Any] | None, limit: int = 180) -> str:
    if not block:
        return ""
    text = norm_space(block.get("text"))
    prefix = norm_space(block.get("type"))
    if len(text) > limit:
        text = text[: limit - 1] + "..."
    return f"{prefix}: {text}" if prefix else text


def caption_body(text: str) -> str:
    return CAPTION_PREFIX_RE.sub("", norm_space(text), count=1).strip()


def alpha_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", text)


def is_short_caption(text: str) -> bool:
    clean = norm_space(text)
    return len(clean) <= SHORT_CHAR_LIMIT or len(alpha_tokens(clean)) <= SHORT_WORD_LIMIT


def has_semantic_anchor(text: str) -> bool:
    lower = norm_space(text).lower()
    if any(anchor in lower for anchor in SEMANTIC_ANCHORS):
        return True
    body = caption_body(text)
    informative = [
        token
        for token in alpha_tokens(body)
        if token.lower() not in LOW_INFORMATION_WORDS and not re.fullmatch(r"s?\d+[a-z]?", token.lower())
    ]
    return len(informative) >= 3


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_blocks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in doc.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            if "page" not in block:
                block = dict(block)
                block["page"] = page.get("page")
            blocks.append(block)
    return blocks


def load_caption_blocks(parsed_dir: Path) -> list[CaptionBlock]:
    captions: list[CaptionBlock] = []
    for path in sorted(parsed_dir.glob("*.json")):
        doc = load_json(path)
        doc_id = norm_space(doc.get("doc_id")) or path.stem
        source_file = norm_space(doc.get("source_file")) or f"{doc_id}.pdf"
        blocks = iter_blocks(doc)
        for idx, block in enumerate(blocks):
            block_type = norm_space(block.get("type"))
            if block_type not in CAPTION_TYPES:
                continue
            prev_block = blocks[idx - 1] if idx > 0 else None
            next_block = blocks[idx + 1] if idx + 1 < len(blocks) else None
            near_bits = [preview_block(prev_block, 120), preview_block(block, 120), preview_block(next_block, 120)]
            captions.append(
                CaptionBlock(
                    doc_id=doc_id,
                    source_file=source_file,
                    block_id=norm_space(block.get("block_id")),
                    block_type=block_type,
                    caption_text=norm_space(block.get("text")),
                    page=block.get("page"),
                    section_path=list(block.get("section_path") or []),
                    previous_block_preview=preview_block(prev_block),
                    next_block_preview=preview_block(next_block),
                    nearby_text_preview=" | ".join(bit for bit in near_bits if bit),
                )
            )
    return captions


def count_parsed_captions(parsed_dir: Path) -> dict[str, Any]:
    counts = Counter()
    docs = 0
    if not parsed_dir.exists():
        return {"exists": False, "doc_count": 0, "counts": {}}
    for path in sorted(parsed_dir.glob("*.json")):
        docs += 1
        doc = load_json(path)
        for block in iter_blocks(doc):
            block_type = norm_space(block.get("type"))
            if block_type in CAPTION_TYPES:
                counts[block_type] += 1
    return {"exists": True, "doc_count": docs, "counts": dict(counts)}


def is_caption_only_table_chunk(chunk: dict[str, Any]) -> bool:
    evidence = set(chunk.get("evidence_types") or [])
    block_types = set(chunk.get("block_types") or [])
    has_table_caption = bool(chunk.get("contains_table_caption")) or "table_caption" in evidence
    has_table_text = bool(chunk.get("contains_table_text")) or "table_text" in evidence or "table" in evidence
    non_caption_evidence = evidence - {"table_caption"}
    non_caption_blocks = block_types - {"table_caption"}
    return has_table_caption and not has_table_text and not non_caption_evidence and not non_caption_blocks


def load_chunk_stats(path: Path, label: str) -> tuple[dict[str, Any], dict[tuple[str, str], list[str]]]:
    stats: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "chunk_count": 0,
        "caption_only_table_chunk_count": 0,
        "figure_focused_chunk_count": 0,
        "table_focused_chunk_count": 0,
        "chunks_with_table_caption": 0,
        "chunks_with_figure_caption": 0,
    }
    block_to_chunks: dict[tuple[str, str], list[str]] = defaultdict(list)
    if not path.exists():
        return stats, block_to_chunks

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            chunk = json.loads(line)
            stats["chunk_count"] += 1
            doc_id = norm_space(chunk.get("doc_id"))
            chunk_id = norm_space(chunk.get("chunk_id"))
            evidence = set(chunk.get("evidence_types") or [])
            block_types = set(chunk.get("block_types") or [])
            has_table_caption = bool(chunk.get("contains_table_caption")) or "table_caption" in evidence
            has_figure_caption = bool(chunk.get("contains_figure_caption")) or "figure_caption" in evidence
            has_table_text = bool(chunk.get("contains_table_text")) or bool(
                {"table_text", "table"} & evidence
            )
            if has_table_caption:
                stats["chunks_with_table_caption"] += 1
            if has_figure_caption:
                stats["chunks_with_figure_caption"] += 1
            if is_caption_only_table_chunk(chunk):
                stats["caption_only_table_chunk_count"] += 1
            if has_figure_caption or "figure_caption" in block_types:
                stats["figure_focused_chunk_count"] += 1
            if has_table_caption or has_table_text or bool({"table_caption", "table_text", "table"} & block_types):
                stats["table_focused_chunk_count"] += 1
            block_ids = chunk.get("source_block_ids") or chunk.get("block_ids") or []
            for block_id in block_ids:
                block_to_chunks[(doc_id, norm_space(block_id))].append(f"{label}:{chunk_id}")
    return stats, block_to_chunks


def load_approved_targets() -> dict[str, Any]:
    target_chunk_ids: set[str] = set()
    target_docs: set[str] = set()
    target_captions: list[str] = []
    missing: list[str] = []
    for path in APPROVED_EVAL_FILES:
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("approved") is False:
                    continue
                target_chunk_ids.add(norm_space(row.get("target_chunk_id")))
                target_docs.add(norm_space(row.get("target_doc_id")))
                caption = norm_space(row.get("target_caption"))
                if caption:
                    target_captions.append(caption)
    return {
        "target_chunk_ids": target_chunk_ids,
        "target_docs": target_docs,
        "target_captions": target_captions,
        "missing": missing,
    }


def broken_organism_context(caption: CaptionBlock) -> bool:
    match = ARTICLE_SINGLE_LETTER_RE.match(caption.caption_text) or SINGLE_LETTER_RE.match(caption.caption_text)
    if not match:
        return False
    combined = f"{caption.previous_block_preview} {caption.next_block_preview}".lower()
    next_text = caption.next_block_preview.lower()
    starts_with_continuation = any(re.search(rf"\b{re.escape(name)}\b", next_text[:80]) for name in ORGANISM_CONTINUATIONS)
    has_list_context = bool(
        re.search(
            r"\b(strains?|plasmids?|media|culture|listed|summari[sz]ed|shown|reported)\b",
            combined,
        )
    )
    return starts_with_continuation or has_list_context


def classify_candidate(caption: CaptionBlock) -> tuple[str | None, str | None, str]:
    text = norm_space(caption.caption_text)
    body = caption_body(text)
    body_tokens = alpha_tokens(body)

    if ARTICLE_SINGLE_LETTER_RE.match(text) and broken_organism_context(caption):
        return (
            "broken_organism_or_abbreviation_prefix",
            "high",
            "Table/Figure marker plus article and single-letter prefix; adjacent text indicates organism/name split.",
        )
    if SINGLE_LETTER_RE.match(text) and broken_organism_context(caption):
        return (
            "broken_organism_or_abbreviation_prefix",
            "high",
            "Table/Figure marker plus single-letter prefix; adjacent text indicates organism/name split.",
        )
    if ARTICLE_SINGLE_LETTER_RE.match(text):
        return (
            "article_plus_single_letter_fragment",
            "high",
            "Caption has only marker plus article and one letter.",
        )
    if SINGLE_LETTER_RE.match(text):
        return (
            "single_letter_fragment",
            "high",
            "Caption has only marker plus one-letter body.",
        )
    if NUMBER_ONLY_RE.match(text):
        return (
            "number_only_caption",
            "high",
            "Caption is only a table/figure number marker; also a generic template caption.",
        )
    if SUPPLEMENTARY_REFERENCE_RE.search(text) and is_short_caption(text):
        return (
            "supplementary_reference_fragment",
            "medium",
            "Short caption-like block appears to be a supplementary table/figure reference.",
        )
    if is_short_caption(text) and not has_semantic_anchor(text):
        confidence = "medium" if len(body_tokens) <= 2 else "low"
        return (
            "very_short_no_semantic_anchor",
            confidence,
            "Short caption has no evident entity, method, noun, or metric anchor.",
        )
    return None, None, ""


def protect_caption(
    caption: CaptionBlock,
    chunk_ids: list[str],
    approved_target_chunks: set[str],
) -> tuple[str | None, str | None]:
    text = norm_space(caption.caption_text)
    short = is_short_caption(text)
    approved_hit = any(chunk.split(":", 1)[-1] in approved_target_chunks for chunk in chunk_ids)
    if caption.doc_id == "doc_0367" and re.match(r"^\s*fig(?:ure)?\.?\s*5\b", text, re.IGNORECASE):
        return (
            "doc_0367_figure5_known_good_anchor",
            "Would damage the Phase 4/5 known-good sanity anchor if demoted.",
        )
    if approved_hit:
        return (
            "approved_eval_target_chunk",
            "Approved eval target evidence should not be demoted without manual review.",
        )
    if short and has_semantic_anchor(text):
        return (
            "short_caption_with_semantic_anchor",
            "Could lose a valid concise title or table/figure heading.",
        )
    if short:
        body = caption_body(text)
        informative = [
            token
            for token in alpha_tokens(body)
            if token.lower() not in LOW_INFORMATION_WORDS and not re.fullmatch(r"s?\d+[a-z]?", token.lower())
        ]
        if len(informative) >= 3:
            return (
                "short_caption_with_complete_title_information",
                "Could demote a compact but meaningful caption title.",
            )
    return None, None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in selected:
        values = []
        for field in fields:
            value = norm_space(row.get(field, ""))
            value = value.replace("|", "\\|")
            if len(value) > 120:
                value = value[:119] + "..."
            values.append(value)
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def recommended_label(row: dict[str, Any]) -> str:
    if row.get("confidence") == "high":
        return "safe_to_demote"
    if row.get("category") == "protected":
        return "keep_as_caption"
    if row.get("category") == "known_good":
        return "keep_as_caption"
    return "needs_manual_check"


def sample_rows(
    candidates: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    known_good: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()

    def add(rows: list[dict[str, Any]], category: str, limit: int) -> None:
        added = 0
        for row in rows:
            key = (row.get("doc_id", ""), row.get("block_id", ""), category)
            if key in used:
                continue
            new_row = dict(row)
            new_row["category"] = category
            new_row["recommended_label"] = recommended_label(new_row)
            if not new_row.get("rationale"):
                new_row["rationale"] = new_row.get("risk_note") or new_row.get("protect_reason") or ""
            sample.append(new_row)
            used.add(key)
            added += 1
            if added >= limit:
                break

    add(
        [
            row
            for row in candidates
            if row["confidence"] == "high" and row["block_type"] == "table_caption"
        ],
        "high_confidence_false_table_caption",
        30,
    )
    add(
        [
            row
            for row in candidates
            if row["confidence"] == "high" and row["block_type"] == "figure_caption"
        ],
        "high_confidence_false_figure_caption",
        20,
    )
    add(
        [row for row in candidates if row["confidence"] == "medium"],
        "medium_confidence_uncertain_caption",
        30,
    )
    add(protected, "protected_short_caption", 20)
    add(known_good, "known_good_caption", 10)
    return sample


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_status = {
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): path.exists()
        for path in REQUIRED_REPORTS
    }
    parsed_raw_counts = count_parsed_captions(PARSED_RAW_DIR)
    parsed_clean_counts = count_parsed_captions(PARSED_CLEAN_DIR)
    captions = load_caption_blocks(PARSED_CLEAN_DIR)

    chunk_stats: list[dict[str, Any]] = []
    combined_block_to_chunks: dict[tuple[str, str], list[str]] = defaultdict(list)
    for label, path in [("phase4d", PHASE4D_CHUNKS), ("phase5c4", PHASE5C_CHUNKS)]:
        stats, mapping = load_chunk_stats(path, label)
        chunk_stats.append(stats)
        for key, values in mapping.items():
            combined_block_to_chunks[key].extend(values)

    approved = load_approved_targets()
    approved_target_chunks = approved["target_chunk_ids"]

    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    known_good: list[dict[str, Any]] = []
    doc_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    short_caption_count = 0
    generic_caption_count = 0

    for caption in captions:
        doc_distribution[caption.doc_id][caption.block_type] += 1
        if is_short_caption(caption.caption_text):
            short_caption_count += 1
        if NUMBER_ONLY_RE.match(caption.caption_text):
            generic_caption_count += 1

        chunk_ids = sorted(set(combined_block_to_chunks.get((caption.doc_id, caption.block_id), [])))
        candidate_rule, confidence, risk_note = classify_candidate(caption)
        protect_reason, risk_if_demoted = protect_caption(caption, chunk_ids, approved_target_chunks)

        if candidate_rule:
            candidates.append(
                {
                    "doc_id": caption.doc_id,
                    "source_file": caption.source_file,
                    "block_id": caption.block_id,
                    "block_type": caption.block_type,
                    "caption_text": caption.caption_text,
                    "page": caption.page,
                    "section_path": json.dumps(caption.section_path, ensure_ascii=False),
                    "candidate_rule": candidate_rule,
                    "confidence": confidence,
                    "previous_block_preview": caption.previous_block_preview,
                    "next_block_preview": caption.next_block_preview,
                    "nearby_text_preview": caption.nearby_text_preview,
                    "appears_in_chunks": bool(chunk_ids),
                    "chunk_id": ";".join(chunk_ids),
                    "risk_note": risk_note,
                }
            )
            doc_distribution[caption.doc_id]["candidate"] += 1
            doc_distribution[caption.doc_id][f"candidate_{caption.block_type}"] += 1
            doc_distribution[caption.doc_id][f"candidate_{confidence}"] += 1
            continue

        if protect_reason:
            protected.append(
                {
                    "doc_id": caption.doc_id,
                    "block_id": caption.block_id,
                    "caption_text": caption.caption_text,
                    "protect_reason": protect_reason,
                    "risk_if_demoted": risk_if_demoted,
                    "source_file": caption.source_file,
                    "block_type": caption.block_type,
                    "page": caption.page,
                    "candidate_rule": "",
                    "confidence": "",
                    "previous_block_preview": caption.previous_block_preview,
                    "next_block_preview": caption.next_block_preview,
                    "nearby_text_preview": caption.nearby_text_preview,
                    "appears_in_chunks": bool(chunk_ids),
                    "chunk_id": ";".join(chunk_ids),
                    "rationale": risk_if_demoted,
                }
            )

        approved_hit = any(chunk.split(":", 1)[-1] in approved_target_chunks for chunk in chunk_ids)
        if approved_hit or (
            caption.doc_id == "doc_0367"
            and re.match(r"^\s*fig(?:ure)?\.?\s*5\b", caption.caption_text, re.IGNORECASE)
        ):
            known_good.append(
                {
                    "doc_id": caption.doc_id,
                    "source_file": caption.source_file,
                    "block_id": caption.block_id,
                    "block_type": caption.block_type,
                    "caption_text": caption.caption_text,
                    "page": caption.page,
                    "section_path": json.dumps(caption.section_path, ensure_ascii=False),
                    "candidate_rule": "",
                    "confidence": "",
                    "previous_block_preview": caption.previous_block_preview,
                    "next_block_preview": caption.next_block_preview,
                    "nearby_text_preview": caption.nearby_text_preview,
                    "appears_in_chunks": bool(chunk_ids),
                    "chunk_id": ";".join(chunk_ids),
                    "rationale": "Approved Phase 4E-3 eval target or doc_0367 Figure 5 sanity anchor.",
                }
            )

    candidates.sort(key=lambda row: (row["confidence"] != "high", row["block_type"], row["doc_id"], row["block_id"]))
    protected.sort(key=lambda row: (row["doc_id"], row["block_id"]))
    known_good.sort(key=lambda row: (row["doc_id"], row["block_id"]))

    confidence_counts = Counter(row["confidence"] for row in candidates)
    rule_counts = Counter(row["candidate_rule"] for row in candidates)
    candidate_type_counts = Counter(row["block_type"] for row in candidates)
    candidates_in_chunks = sum(1 for row in candidates if row["appears_in_chunks"])

    doc_rows = []
    for doc_id, counts in doc_distribution.items():
        doc_rows.append(
            {
                "doc_id": doc_id,
                "table_caption": counts.get("table_caption", 0),
                "figure_caption": counts.get("figure_caption", 0),
                "candidate_count": counts.get("candidate", 0),
                "candidate_table_caption": counts.get("candidate_table_caption", 0),
                "candidate_figure_caption": counts.get("candidate_figure_caption", 0),
                "candidate_high": counts.get("candidate_high", 0),
                "candidate_medium": counts.get("candidate_medium", 0),
                "candidate_low": counts.get("candidate_low", 0),
            }
        )
    doc_rows.sort(key=lambda row: (-row["candidate_count"], row["doc_id"]))

    distribution = {
        "input_status": {
            "parsed_raw_dir": {"path": str(PARSED_RAW_DIR), "exists": PARSED_RAW_DIR.exists()},
            "parsed_clean_dir": {"path": str(PARSED_CLEAN_DIR), "exists": PARSED_CLEAN_DIR.exists()},
            "chunks": chunk_stats,
            "reports": report_status,
            "approved_eval_missing": approved["missing"],
        },
        "parsed_raw_caption_counts": parsed_raw_counts,
        "parsed_clean_caption_counts": parsed_clean_counts,
        "caption_distribution": {
            "table_caption_count": parsed_clean_counts["counts"].get("table_caption", 0),
            "figure_caption_count": parsed_clean_counts["counts"].get("figure_caption", 0),
            "short_caption_count": short_caption_count,
            "generic_caption_count": generic_caption_count,
            "fragment_caption_candidate_count": len(candidates),
            "confidence_counts": dict(confidence_counts),
            "candidate_rule_counts": dict(rule_counts),
            "candidate_block_type_counts": dict(candidate_type_counts),
            "candidates_appearing_in_chunks": candidates_in_chunks,
            "protected_short_caption_count": len(protected),
        },
        "doc_level_distribution": doc_rows,
    }

    (OUT_DIR / "caption_distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    candidate_fields = [
        "doc_id",
        "source_file",
        "block_id",
        "block_type",
        "caption_text",
        "page",
        "section_path",
        "candidate_rule",
        "confidence",
        "previous_block_preview",
        "next_block_preview",
        "nearby_text_preview",
        "appears_in_chunks",
        "chunk_id",
        "risk_note",
    ]
    write_csv(OUT_DIR / "false_caption_candidates.csv", candidates, candidate_fields)

    protected_fields = ["doc_id", "block_id", "caption_text", "protect_reason", "risk_if_demoted"]
    write_csv(OUT_DIR / "protected_short_captions.csv", protected, protected_fields)

    signoff = sample_rows(candidates, protected, known_good)
    signoff_fields = [
        "category",
        "doc_id",
        "block_id",
        "caption_text",
        "candidate_rule",
        "confidence",
        "previous_block_preview",
        "next_block_preview",
        "recommended_label",
        "rationale",
    ]
    write_csv(OUT_DIR / "signoff_sample.csv", signoff, signoff_fields)

    chunk_lines = []
    for stats in chunk_stats:
        if not stats["exists"]:
            chunk_lines.append(f"- {stats['label']}: missing ({stats['path']})")
            continue
        chunk_lines.extend(
            [
                f"- {stats['label']} chunk_count: {stats['chunk_count']}",
                f"- {stats['label']} caption-only table chunk count: {stats['caption_only_table_chunk_count']}",
                f"- {stats['label']} table-focused chunk count: {stats['table_focused_chunk_count']}",
                f"- {stats['label']} figure-focused chunk count: {stats['figure_focused_chunk_count']}",
            ]
        )

    top_doc_lines = [
        f"- {row['doc_id']}: candidates={row['candidate_count']}, table={row['candidate_table_caption']}, figure={row['candidate_figure_caption']}"
        for row in doc_rows[:15]
        if row["candidate_count"]
    ]
    report_missing = [path for path, exists in report_status.items() if not exists]

    distribution_md = "\n".join(
        [
            "# Phase 5D-1 Caption Distribution Audit",
            "",
            "## Input status",
            f"- parsed_raw exists: {PARSED_RAW_DIR.exists()}",
            f"- parsed_clean exists: {PARSED_CLEAN_DIR.exists()}",
            f"- missing requested reports: {', '.join(report_missing) if report_missing else 'none'}",
            "",
            "## Parsed caption counts",
            f"- table_caption count: {distribution['caption_distribution']['table_caption_count']}",
            f"- figure_caption count: {distribution['caption_distribution']['figure_caption_count']}",
            f"- short caption count: {short_caption_count}",
            f"- generic caption count: {generic_caption_count}",
            f"- fragment caption candidate count: {len(candidates)}",
            "",
            "## Chunk counts",
            *chunk_lines,
            "",
            "## Candidate distribution",
            f"- by confidence: {dict(confidence_counts)}",
            f"- by block_type: {dict(candidate_type_counts)}",
            f"- by rule: {dict(rule_counts)}",
            f"- candidates appearing in chunks: {candidates_in_chunks}",
            "",
            "## Top documents by candidate count",
            *(top_doc_lines or ["- none"]),
            "",
            "## Doc-level distribution sample",
            markdown_table(
                doc_rows,
                [
                    "doc_id",
                    "table_caption",
                    "figure_caption",
                    "candidate_count",
                    "candidate_table_caption",
                    "candidate_figure_caption",
                    "candidate_high",
                    "candidate_medium",
                    "candidate_low",
                ],
                limit=40,
            ),
            "",
        ]
    )
    (OUT_DIR / "caption_distribution.md").write_text(distribution_md, encoding="utf-8")

    signoff_md_sections = [
        "# Phase 5D-2 Sign-off Sample",
        "",
        "Recommended labels are audit-only suggestions; no parser or chunk data has been changed.",
        "",
        markdown_table(signoff, signoff_fields, limit=None),
        "",
    ]
    (OUT_DIR / "signoff_sample.md").write_text("\n".join(signoff_md_sections), encoding="utf-8")

    table_vs_figure = "table_caption" if candidate_type_counts["table_caption"] >= candidate_type_counts["figure_caption"] else "figure_caption"
    top_docs_summary = ", ".join(
        f"{row['doc_id']}({row['candidate_count']})"
        for row in doc_rows[:10]
        if row["candidate_count"]
    )
    summary = "\n".join(
        [
            "# Phase 5D-1 Parser False / Fragment Caption Audit Summary",
            "",
            "## Required answers",
            f"1. table_caption / figure_caption counts: {distribution['caption_distribution']['table_caption_count']} / {distribution['caption_distribution']['figure_caption_count']}.",
            f"2. false / fragment caption candidate count: {len(candidates)}.",
            f"3. confidence distribution: high={confidence_counts.get('high', 0)}, medium={confidence_counts.get('medium', 0)}, low={confidence_counts.get('low', 0)}.",
            f"4. main false caption patterns: {', '.join(f'{rule}={count}' for rule, count in rule_counts.most_common(8))}.",
            f"5. most concentrated docs: {top_docs_summary}.",
            f"6. dominant affected caption type: {table_vs_figure} (table={candidate_type_counts.get('table_caption', 0)}, figure={candidate_type_counts.get('figure_caption', 0)}).",
            f"7. chunk impact: {candidates_in_chunks} candidates appear in at least one audited chunk set.",
            f"8. protected short caption count: {len(protected)}.",
            "9. recommendation for Phase 5D-2 manual sign-off: yes.",
            "10. recommendation to directly implement cleanup now: no; sign-off should happen first.",
            "11. Phase 5D-2 should review the signoff_sample files, especially high-confidence table fragments, high-confidence figure number-only captions, medium-confidence very_short_no_semantic_anchor rows, protected short captions, and known-good approved eval anchors.",
            "",
            "## Output files",
            "- caption_distribution.json",
            "- caption_distribution.md",
            "- false_caption_candidates.csv",
            "- protected_short_captions.csv",
            "- signoff_sample.csv",
            "- signoff_sample.md",
            "- summary.md",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    print(json.dumps(distribution["caption_distribution"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
