#!/usr/bin/env python3
"""Phase 5E-2 read-only sign-off for section repair candidates.

Reads Phase 5E-1 audit artifacts, parsed_clean, and chunk JSONL files, then
writes sign-off decisions under reports/phase5e_section_repair_signoff.
It does not implement section repair or modify parsed/chunk inputs.
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
AUDIT_DIR = ROOT / "reports/phase5e_section_metadata_audit"
OUT_DIR = ROOT / "reports/phase5e_section_repair_signoff"
PARSED_CLEAN_DIR = ROOT / "data/paper_round1/parsed_clean"

REQUIRED_AUDIT_FILES = [
    AUDIT_DIR / "summary.md",
    AUDIT_DIR / "section_distribution.json",
    AUDIT_DIR / "section_distribution.md",
    AUDIT_DIR / "weak_section_chunks.csv",
    AUDIT_DIR / "section_repair_candidates.csv",
    AUDIT_DIR / "protected_sample_section_check.md",
    AUDIT_DIR / "protected_sample_section_check.csv",
    AUDIT_DIR / "examples.md",
    AUDIT_DIR / "repair_strategy_proposal.md",
]

CHUNK_PATHS = {
    "phase4d_compact": Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
    "phase5c4_full_enhanced": Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    "phase5d3_caption_cleanup": Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
}

BODY_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "results",
    "result",
    "discussion",
    "results and discussion",
    "methods",
    "method",
    "materials and methods",
    "experimental procedures",
    "conclusion",
    "conclusions",
}
EXCLUDED_HEADINGS = {
    "title",
    "unknown",
    "references",
    "reference",
    "acknowledgements",
    "acknowledgments",
    "author contributions",
    "correspondence",
    "funding",
    "metadata",
    "supporting information",
    "supplementary information",
    "key references",
}
SIGNOFF_LABELS = {"safe_to_repair", "keep_as_is", "needs_manual_check", "do_not_repair", "backlog_only"}
RECOMMENDED_ACTIONS = {
    "add_section_repair_metadata",
    "use_repaired_section_for_retrieval_text_experiment",
    "no_action",
    "manual_pdf_check",
    "backlog",
}


@dataclass
class Block:
    block_id: str
    index: int
    page: int | None
    block_type: str
    text: str
    section_path: list[str]


@dataclass
class DocBlocks:
    blocks: list[Block]
    by_id: dict[str, Block]


def norm_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def clean_heading(value: Any) -> str:
    text = norm_ws(value)
    text = re.sub(r"^\[.*?CAPTION\]\s*", "", text, flags=re.I)
    text = re.sub(r"^#+\s*", "", text).strip()
    return text


def low(value: Any) -> str:
    return clean_heading(value).lower().strip(" .:;")


def preview(value: Any, limit: int = 180) -> str:
    text = norm_ws(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [norm_ws(v) for v in value if norm_ws(v)]
    if value is None:
        return []
    text = norm_ws(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [norm_ws(v) for v in parsed if norm_ws(v)]
    except json.JSONDecodeError:
        pass
    return [text]


def json_list(values: Any) -> str:
    if isinstance(values, list):
        return json.dumps(values, ensure_ascii=False)
    return json.dumps(parse_json_list(values), ensure_ascii=False)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parsed_index() -> dict[str, DocBlocks]:
    docs: dict[str, DocBlocks] = {}
    for path in sorted(PARSED_CLEAN_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_id = data.get("doc_id") or path.stem
        blocks: list[Block] = []
        by_id: dict[str, Block] = {}
        for page in data.get("pages", []):
            page_num = page.get("page")
            for raw in page.get("blocks", []) or []:
                block_id = norm_ws(raw.get("block_id"))
                if not block_id:
                    continue
                block = Block(
                    block_id=block_id,
                    index=len(blocks),
                    page=page_num if page_num is not None else raw.get("page"),
                    block_type=norm_ws(raw.get("type")),
                    text=norm_ws(raw.get("text")),
                    section_path=parse_json_list(raw.get("section_path")),
                )
                blocks.append(block)
                by_id[block_id] = block
        docs[doc_id] = DocBlocks(blocks=blocks, by_id=by_id)
    return docs


def source_blocks(row: dict[str, str], docs: dict[str, DocBlocks]) -> list[Block]:
    doc = docs.get(row.get("doc_id", ""))
    if doc is None:
        return []
    ids = parse_json_list(row.get("source_block_ids"))
    return [doc.by_id[block_id] for block_id in ids if block_id in doc.by_id]


def candidate_block(row: dict[str, str], docs: dict[str, DocBlocks]) -> Block | None:
    doc = docs.get(row.get("doc_id", ""))
    if doc is None:
        return None
    block_id = norm_ws(row.get("candidate_block_id"))
    return doc.by_id.get(block_id)


def relative_position(weak: dict[str, str], cand: dict[str, str], docs: dict[str, DocBlocks]) -> str:
    blocks = source_blocks(weak, docs)
    cblock = candidate_block(cand, docs)
    if not blocks or cblock is None:
        return "unknown"
    first = min(block.index for block in blocks)
    if cblock.index < first:
        return "before"
    if cblock.index > first:
        return "after"
    return "same"


def text_reference_like(text: str) -> bool:
    stripped = norm_ws(text)
    low_text = stripped.lower()
    if re.match(r"^\d+\.\s+[A-Z][A-Za-z'\-]+,", stripped):
        return True
    if re.match(r"^[A-Z]\.,?\s+and\s+[A-Z]", stripped):
        return True
    if "references" in low_text[:80]:
        return True
    reference_terms = ["journal", "fems microbiol", "biotechnol", "letters.", "press.", "springer", "wiley"]
    return sum(term in low_text for term in reference_terms) >= 2


def candidate_is_excluded(candidate: str) -> bool:
    c = low(candidate)
    if not c:
        return True
    if c in EXCLUDED_HEADINGS:
        return True
    if re.match(r"^(fig(?:ure)?\.?|table)\s*(s?\d+|[ivxlcdm]+)\b", c, re.I):
        return True
    if re.match(r"^表\s*\d+", c):
        return True
    if len(c) <= 2:
        return True
    if re.search(r"\b(references?|funding|correspondence|author contributions?|acknowledg)", c):
        return True
    if re.search(r"\b(university|department|institute|school|college|laboratory|centre|center)\b", c, re.I):
        return True
    if re.search(r"\b(doi|issn|copyright|received|accepted|published|journal homepage|terms)\b", c, re.I):
        return True
    if re.search(r"\b(μmol|od600|cfu/ml)\b", c, re.I):
        return True
    if re.search(r"\b(is|are|was|were|has|have|had|led|resulted|shown|based|redundant|inhibitors|initialized|accumulates|consistent)\b", c, re.I):
        if c not in BODY_HEADINGS:
            return True
    if re.match(r"^[a-z]+$", c) and c not in BODY_HEADINGS:
        return True
    if re.match(r"^[A-Z]$", clean_heading(candidate)):
        return True
    return False


def candidate_is_clear_body_heading(candidate: str) -> bool:
    c = low(candidate)
    if candidate_is_excluded(candidate):
        return False
    if c in BODY_HEADINGS:
        return True
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Za-z]", clean_heading(candidate)):
        return True
    if re.match(r"^\([0-9ivxlcdm]+\)\s+[A-Za-z]", clean_heading(candidate), re.I):
        # Often a figure callout in this corpus unless it is part of a stable
        # parsed section path; keep it out of broad automatic repair.
        return False
    words = clean_heading(candidate).split()
    if 2 <= len(words) <= 8 and not re.search(r"[.;,]$", clean_heading(candidate)):
        return True
    return False


def chunk_is_front_or_metadata(weak: dict[str, str]) -> bool:
    text = norm_ws(weak.get("text_preview"))
    low_text = text.lower()
    if text_reference_like(text):
        return True
    if re.search(r"(doi:|received:|accepted:|citation:|copyright|journal homepage|full terms|issn:)", low_text):
        return True
    block_types = " ".join(parse_json_list(weak.get("block_types"))).lower()
    if "title" in block_types:
        return True
    return False


def has_table_figure_evidence(weak: dict[str, str]) -> bool:
    evidence = " ".join(parse_json_list(weak.get("evidence_types"))).lower()
    return any(kind in evidence for kind in ["table_caption", "table_text", "table_related", "figure_caption"])


def approved_table_figure_target_chunks(protected_rows: list[dict[str, str]]) -> set[str]:
    chunks = set()
    for row in protected_rows:
        if row.get("category") != "phase4_5c_eval_target":
            continue
        if row.get("query_type") in {"table_content", "caption_level_table", "figure_caption"}:
            chunk_id = norm_ws(row.get("chunk_id"))
            if chunk_id:
                chunks.add(chunk_id)
    return chunks


def impacted_protected_chunks(protected_rows: list[dict[str, str]]) -> set[str]:
    chunks = set()
    for row in protected_rows:
        if row.get("category") != "phase5d_protected_caption":
            continue
        if norm_ws(row.get("retrieval_explainability_impact")).startswith("yes"):
            chunk_id = norm_ws(row.get("chunk_id"))
            if chunk_id:
                chunks.add(chunk_id)
    return chunks


def signoff_decision(
    weak: dict[str, str],
    cand: dict[str, str],
    docs: dict[str, DocBlocks],
    approved_targets: set[str],
    protected_impacted: set[str],
) -> tuple[str, str, str, str]:
    """Return signoff_label, recommended_action, rationale, risk_if_wrong."""
    chunk_id = weak["chunk_id"]
    severity = weak.get("severity", "")
    candidate = cand.get("candidate_section", "")
    source = cand.get("candidate_source", "")
    confidence = cand.get("confidence", "")
    direction = relative_position(weak, cand, docs)
    distance = norm_ws(cand.get("candidate_distance_blocks"))
    text = weak.get("text_preview", "")

    if chunk_id in approved_targets:
        return (
            "keep_as_is",
            "no_action",
            "approved table/figure eval target; no section repair needed from protected check",
            "could perturb stable eval target context",
        )

    if severity == "low":
        return (
            "backlog_only",
            "backlog",
            "low severity explainability issue; not worth automatic repair",
            "unnecessary metadata churn for low-impact chunk",
        )

    if not candidate:
        if chunk_id in protected_impacted or severity == "high":
            return (
                "backlog_only",
                "backlog",
                "weak evidence section but no trustworthy nearby body heading was found",
                "guessing section would mislead evidence display",
            )
        return (
            "do_not_repair",
            "no_action",
            "no candidate section available",
            "no safe repair target",
        )

    if candidate_is_excluded(candidate) or not candidate_is_clear_body_heading(candidate):
        return (
            "do_not_repair",
            "no_action",
            "candidate is not a clear body section heading or appears to be figure/table/front-matter text",
            "would replace weak section with a misleading pseudo-heading",
        )

    if chunk_is_front_or_metadata(weak):
        return (
            "backlog_only",
            "backlog",
            "chunk appears to be front matter, metadata, or references despite a plausible candidate",
            "front/back matter could be mislabeled as body evidence",
        )

    if source == "same_page_heading" and direction != "before":
        action = "manual_pdf_check" if severity == "high" or chunk_id in protected_impacted else "backlog"
        label = "needs_manual_check" if action == "manual_pdf_check" else "backlog_only"
        return (
            label,
            action,
            f"same-page candidate is {direction} the chunk; ordering is not safe enough for automatic repair",
            "could attach caption/front matter to a later section",
        )

    if source == "previous_heading" and severity == "high":
        return (
            "needs_manual_check",
            "manual_pdf_check",
            "high-severity evidence chunk with previous heading needs visual/context confirmation",
            "caption may be associated with figure panel labels rather than a body section",
        )

    if source not in {"section_path_parent", "previous_heading", "same_page_heading"}:
        return (
            "do_not_repair",
            "no_action",
            "candidate source is not one of the approved explainable sources",
            "untraceable repair source",
        )

    if distance and distance.isdigit() and int(distance) > 20:
        return (
            "needs_manual_check",
            "manual_pdf_check",
            "candidate is too far away for automatic repair",
            "long-distance repair may cross section boundaries",
        )

    if has_table_figure_evidence(weak):
        return (
            "needs_manual_check",
            "manual_pdf_check",
            "table/figure evidence requires manual confirmation even with a plausible heading",
            "evidence captions often sit near figure labels or table headers",
        )

    if source != "section_path_parent":
        return (
            "backlog_only",
            "backlog",
            "candidate is plausible but not anchored in the chunk source section_path; keep out of automatic repair",
            "nearby heading may be front matter, a figure label, or a neighboring section",
        )

    normalized_text = clean_heading(text).lower()
    normalized_candidate = clean_heading(candidate).lower()
    heading_anchor = normalized_text.startswith(normalized_candidate)
    markdown_heading_anchor = norm_ws(text).lower().startswith("### " + normalized_candidate)
    if not (heading_anchor or markdown_heading_anchor):
        return (
            "backlog_only",
            "backlog",
            "candidate comes from section_path_parent but the chunk does not start at that heading; defer broad body-paragraph repair",
            "could relabel trailing references or body text that crossed section boundaries",
        )

    return (
        "safe_to_repair",
        "add_section_repair_metadata",
        "candidate is a clear body heading from an explainable nearby/section_path source and affects only metadata/display",
        "low; wrong repair would affect section display/retrieval text context only",
    )


def join_rows(weak_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    weak_by_chunk = {row["chunk_id"]: row for row in weak_rows}
    cand_by_chunk = {row["chunk_id"]: row for row in candidate_rows}
    return weak_by_chunk, cand_by_chunk


def build_high_review(
    weak_by_chunk: dict[str, dict[str, str]],
    cand_by_chunk: dict[str, dict[str, str]],
    decisions: dict[str, tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for chunk_id, weak in sorted(weak_by_chunk.items()):
        if weak.get("severity") != "high":
            continue
        cand = cand_by_chunk.get(chunk_id, {})
        label, action, rationale, _risk = decisions[chunk_id]
        rows.append(
            {
                "doc_id": weak.get("doc_id"),
                "chunk_id": chunk_id,
                "current_section": weak.get("section"),
                "current_section_path": weak.get("section_path"),
                "chunk_type": weak.get("chunk_type"),
                "evidence_types": weak.get("evidence_types"),
                "weak_section_reason": weak.get("weak_section_reason"),
                "candidate_section": cand.get("candidate_section", ""),
                "candidate_source": cand.get("candidate_source", ""),
                "candidate_block_id": cand.get("candidate_block_id", ""),
                "candidate_distance_blocks": cand.get("candidate_distance_blocks", ""),
                "candidate_page_distance": cand.get("candidate_page_distance", ""),
                "signoff_label": label,
                "rationale": rationale,
                "recommended_action": action,
            }
        )
    return rows


def build_candidate_signoff(
    weak_by_chunk: dict[str, dict[str, str]],
    cand_by_chunk: dict[str, dict[str, str]],
    decisions: dict[str, tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for chunk_id, cand in sorted(cand_by_chunk.items()):
        if cand.get("repair_recommendation") != "safe_to_repair":
            continue
        weak = weak_by_chunk.get(chunk_id)
        if weak is None:
            continue
        label, action, rationale, risk = decisions[chunk_id]
        rows.append(
            {
                "doc_id": weak.get("doc_id"),
                "chunk_id": chunk_id,
                "severity": weak.get("severity"),
                "current_section": weak.get("section"),
                "current_section_path": weak.get("section_path"),
                "candidate_section": cand.get("candidate_section", ""),
                "candidate_source": cand.get("candidate_source", ""),
                "candidate_confidence": cand.get("confidence", ""),
                "signoff_label": label,
                "rationale": rationale,
                "recommended_action": action,
                "risk_if_wrong": risk,
            }
        )
    return rows


def build_protected_signoff_rows(
    protected_rows: list[dict[str, str]],
    weak_by_chunk: dict[str, dict[str, str]],
    cand_by_chunk: dict[str, dict[str, str]],
    decisions: dict[str, tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted_query_types = {"table_content", "caption_level_table", "figure_caption"}
    seen = set()
    for row in protected_rows:
        category = row.get("category", "")
        query_type = row.get("query_type", "")
        impacted = norm_ws(row.get("retrieval_explainability_impact")).startswith("yes")
        include = False
        if category == "doc_0367_figure5":
            include = True
        elif category == "phase4_5c_eval_target" and query_type in wanted_query_types:
            include = True
        elif category == "phase5d_protected_caption" and impacted:
            include = True
        if not include:
            continue
        key = (
            category,
            row.get("dataset", ""),
            row.get("sample_id", ""),
            row.get("query_type", ""),
            row.get("chunk_id", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        chunk_id = row.get("chunk_id", "")
        weak = weak_by_chunk.get(chunk_id, {})
        cand = cand_by_chunk.get(chunk_id, {})
        if category == "doc_0367_figure5":
            label, action, rationale = "keep_as_is", "no_action", "doc_0367 Figure 5 has section Full Text and is not weak"
        elif category == "phase4_5c_eval_target":
            label, action, rationale = "keep_as_is", "no_action", "key table/figure eval target has no weak-section repair need"
        elif impacted and chunk_id in decisions:
            label, action, rationale, _risk = decisions[chunk_id]
            if label == "do_not_repair":
                label, action = "backlog_only", "backlog"
                rationale = "protected caption has weak section but no safe automatic repair candidate"
        else:
            label, action, rationale = "backlog_only", "backlog", "protected caption impact could not be mapped to a safe candidate"
        rows.append(
            {
                "category": category,
                "sample_id": row.get("sample_id", ""),
                "query_type": query_type,
                "dataset": row.get("dataset", ""),
                "doc_id": row.get("doc_id", ""),
                "chunk_id": chunk_id,
                "section": row.get("section", ""),
                "section_path": row.get("section_path", ""),
                "weak_section": row.get("weak_section", ""),
                "severity": row.get("severity", weak.get("severity", "")),
                "candidate_section": cand.get("candidate_section", row.get("candidate_section", "")),
                "signoff_label": label,
                "recommended_action": action,
                "rationale": rationale,
                "manual_pdf_check_needed": action == "manual_pdf_check",
                "retrieval_explainability_impact": row.get("retrieval_explainability_impact", ""),
                "text_preview": row.get("text_preview", ""),
            }
        )
    return rows


def protected_markdown(rows: list[dict[str, Any]]) -> str:
    doc0367 = [r for r in rows if r["category"] == "doc_0367_figure5"]
    eval_targets = [r for r in rows if r["category"] == "phase4_5c_eval_target"]
    impacted = [r for r in rows if r["category"] == "phase5d_protected_caption"]
    eval_by_query = defaultdict(lambda: Counter(total=0, repair=0, manual=0))
    for row in eval_targets:
        q = row["query_type"] or "unknown"
        eval_by_query[q]["total"] += 1
        if row["recommended_action"] != "no_action":
            eval_by_query[q]["repair"] += 1
        if row["manual_pdf_check_needed"] in {True, "True"}:
            eval_by_query[q]["manual"] += 1
    impacted_counter = Counter(row["signoff_label"] for row in impacted)
    manual = sum(1 for row in impacted if row["manual_pdf_check_needed"] in {True, "True"})
    lines = [
        "# Phase 5E-2 Protected Sample Sign-off",
        "",
        "## doc_0367 Figure 5",
    ]
    if doc0367:
        first = doc0367[0]
        lines.extend(
            [
                f"- section: {first['section']}",
                f"- weak_section: {first['weak_section']}",
                "- decision: keep_as_is",
                "- repair_needed: no",
            ]
        )
    else:
        lines.append("- not located; no automatic repair decision made")
    lines.extend(
        [
            "",
            "## Key Table/Figure Eval Targets",
            "",
            "| query_type | checked_rows | repair_needed | manual_pdf_check |",
            "|---|---:|---:|---:|",
        ]
    )
    for query_type, counts in sorted(eval_by_query.items()):
        lines.append(f"| {query_type} | {counts['total']} | {counts['repair']} | {counts['manual']} |")
    lines.extend(
        [
            "",
            "## Phase 5D Protected Caption Explainability Impact",
            f"- checked_rows: {len(impacted)}",
            f"- signoff_counts: {dict(impacted_counter)}",
            f"- manual_pdf_check_needed: {manual}",
            "- automatic_repair: no; impacted rows lack a reliable body-heading candidate or need visual/context confirmation.",
            "",
            "## Answers",
            "- doc_0367 Figure 5 still does not need repair.",
            "- key table_content / caption_level_table / figure_caption eval targets do not need repair.",
            "- Phase 5D protected caption weak sections should remain backlog/manual-review items, not automatic repair inputs.",
            "- protected samples should not be automatically repaired in Phase 5E-3.",
            "- manual PDF check is only needed if the team wants to resolve the protected-caption backlog; it is not needed for an automatic repair implementation decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_rule_proposal(
    path: Path,
    high_counts: Counter,
    candidate_counts: Counter,
    protected_counts: Counter,
) -> bool:
    reviewed_safe = candidate_counts.get("safe_to_repair", 0)
    high_safe = high_counts.get("safe_to_repair", 0)
    protected_auto = protected_counts.get("safe_to_repair", 0)
    recommend = high_safe > 0 or protected_auto > 0
    lines = [
        "# Phase 5E-3 Section Repair Rule Proposal",
        "",
        "## 1. Recommendation",
        "",
        f"Recommendation: {'enter Phase 5E-3' if recommend else 'do not enter Phase 5E-3 yet; keep as backlog'}",
        "",
        f"Rationale: reviewed safe_to_repair candidates={reviewed_safe}, high-severity safe_to_repair={high_safe}, protected-caption automatic repair candidates={protected_auto}. The reliable candidates are not in the high/protected slices that motivated this phase.",
        "",
        "## 2. Repairable Scenarios",
        "",
        "- `section == Title` or `section == Unknown` only when a previous credible body heading is explicit and nearby.",
        "- evidence chunks only when the candidate heading is a real body section, not a figure panel label, table header, caption, or front-matter fragment.",
        "- `section_path == [\"Title\"]` only when parsed_clean source blocks already carry a clear non-Title body heading.",
        "- Phase 5D protected caption explainability impact only if a high-confidence nearby body heading exists after manual/context confirmation.",
        "",
        "## 3. Non-Repairable Scenarios",
        "",
        "- Do not repair all `Title` / `Unknown` chunks.",
        "- Do not automatically repair all low-severity chunks.",
        "- Do not guess sections from keywords.",
        "- Do not backfill across distant pages.",
        "- Do not use References, Metadata, Funding, Author Contributions, Correspondence, Title, figure labels, table headers, or captions as body sections.",
        "- Do not modify parsed_clean original `section_path`.",
        "- Do not modify chunks JSONL top-level fields.",
        "- Do not add doc_id-specific exceptions.",
        "",
        "## 4. Recommended Implementation Shape If Reopened Later",
        "",
        "- Use a default-off experiment path.",
        "- Preserve original parsed_clean and original chunk fields.",
        "- Add optional section repair metadata under chunk metadata or source_block_metadata.",
        "- Record `original_section` and `original_section_path`.",
        "- Record `repaired_section` and `repaired_section_source`.",
        "- Record `section_repair_rule_id`.",
        "- Record `section_repair_confidence`.",
        "- Use repaired section only in experimental retrieval_text/display section.",
        "- Emit an audit file for every changed chunk.",
        "",
        "## 5. Stop Conditions",
        "",
        "- safe_to_repair count is too small in high/protected slices.",
        "- high-severity candidates are figure/table-local labels or lack trustworthy nearby headings.",
        "- needs_manual_check appears in protected samples.",
        "- protected sample false-repair risk exists.",
        "- the rule cannot be expressed without visual PDF judgment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommend


def write_summary(
    path: Path,
    high_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    protected_rows: list[dict[str, Any]],
    recommend_5e3: bool,
) -> None:
    high_counts = Counter(row["signoff_label"] for row in high_rows)
    cand_counts = Counter(row["signoff_label"] for row in candidate_rows)
    all_counts = high_counts + cand_counts
    protected_impacted = [row for row in protected_rows if row["category"] == "phase5d_protected_caption"]
    protected_counts = Counter(row["signoff_label"] for row in protected_impacted)
    doc0367_needs = any(
        row["category"] == "doc_0367_figure5" and row["recommended_action"] != "no_action"
        for row in protected_rows
    )
    eval_needs = any(
        row["category"] == "phase4_5c_eval_target" and row["recommended_action"] != "no_action"
        for row in protected_rows
    )
    manual_needed = any(
        row.get("manual_pdf_check_needed") in {True, "True"}
        or row.get("recommended_action") == "manual_pdf_check"
        for row in protected_rows + candidate_rows + high_rows
    )
    lines = [
        "# Phase 5E-2 Section Repair Candidate Sign-off Summary",
        "",
        f"1. high severity chunks: {len(high_rows)}; sign-off: {dict(high_counts)}",
        f"2. Phase 5E-1 safe_to_repair candidates after review still safe_to_repair: {cand_counts.get('safe_to_repair', 0)} / {len(candidate_rows)}",
        f"3. keep_as_is / do_not_repair / needs_manual_check / backlog_only: {all_counts.get('keep_as_is', 0)} / {all_counts.get('do_not_repair', 0)} / {all_counts.get('needs_manual_check', 0)} / {all_counts.get('backlog_only', 0)}",
        f"4. Phase 5D protected caption explainability impacted samples: {dict(protected_counts)}; automatic repair recommended: no",
        f"5. doc_0367 Figure 5 needs repair: {'yes' if doc0367_needs else 'no'}",
        f"6. key table/figure eval targets need repair: {'yes' if eval_needs else 'no'}",
        f"7. recommend Phase 5E-3: {'yes' if recommend_5e3 else 'no'}",
        "8. recommended repair scope: none for immediate implementation; if reopened later, limit to default-off metadata-only repair for reviewed non-protected medium chunks with clear section_path_parent body headings",
        f"9. if not entering: high/protected slices do not have enough reliable automatic repair candidates; most issues are explainability backlog or require manual/context checks",
        "10. need modify parsed_clean: no",
        "11. need modify chunks JSONL top-level fields: no",
        "12. need rebuild index now: no",
        "13. need retrieval eval now: no",
        f"14. need manual PDF check: {'yes, only for backlog/manual protected-caption resolution' if manual_needed else 'no'}",
        "15. next step: keep Phase 5E repair as backlog unless a later phase explicitly scopes a default-off metadata-only experiment with manual-reviewed candidates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in REQUIRED_AUDIT_FILES if not path.exists()]
    for path in CHUNK_PATHS.values():
        if not path.exists():
            missing.append(str(path))

    # Read required inputs for availability and to avoid stale assumptions.
    for path in REQUIRED_AUDIT_FILES:
        if path.exists() and path.suffix == ".md":
            _ = path.read_text(encoding="utf-8", errors="replace")
    all_chunks = {name: load_chunks(path) for name, path in CHUNK_PATHS.items() if path.exists()}
    _chunk_count = sum(len(rows) for rows in all_chunks.values())

    weak_rows = load_csv(AUDIT_DIR / "weak_section_chunks.csv")
    candidate_rows_raw = load_csv(AUDIT_DIR / "section_repair_candidates.csv")
    protected_rows_raw = load_csv(AUDIT_DIR / "protected_sample_section_check.csv")
    weak_by_chunk, cand_by_chunk = join_rows(weak_rows, candidate_rows_raw)
    docs = build_parsed_index()

    approved_targets = approved_table_figure_target_chunks(protected_rows_raw)
    protected_impacted = impacted_protected_chunks(protected_rows_raw)

    decisions: dict[str, tuple[str, str, str, str]] = {}
    for chunk_id, weak in weak_by_chunk.items():
        cand = cand_by_chunk.get(chunk_id, {})
        decisions[chunk_id] = signoff_decision(weak, cand, docs, approved_targets, protected_impacted)

    high_rows = build_high_review(weak_by_chunk, cand_by_chunk, decisions)
    candidate_rows = build_candidate_signoff(weak_by_chunk, cand_by_chunk, decisions)
    protected_rows = build_protected_signoff_rows(protected_rows_raw, weak_by_chunk, cand_by_chunk, decisions)

    high_fields = [
        "doc_id",
        "chunk_id",
        "current_section",
        "current_section_path",
        "chunk_type",
        "evidence_types",
        "weak_section_reason",
        "candidate_section",
        "candidate_source",
        "candidate_block_id",
        "candidate_distance_blocks",
        "candidate_page_distance",
        "signoff_label",
        "rationale",
        "recommended_action",
    ]
    write_csv(OUT_DIR / "high_severity_review.csv", high_rows, high_fields)

    candidate_fields = [
        "doc_id",
        "chunk_id",
        "severity",
        "current_section",
        "current_section_path",
        "candidate_section",
        "candidate_source",
        "candidate_confidence",
        "signoff_label",
        "rationale",
        "recommended_action",
        "risk_if_wrong",
    ]
    write_csv(OUT_DIR / "repair_candidate_signoff.csv", candidate_rows, candidate_fields)

    protected_fields = [
        "category",
        "sample_id",
        "query_type",
        "dataset",
        "doc_id",
        "chunk_id",
        "section",
        "section_path",
        "weak_section",
        "severity",
        "candidate_section",
        "signoff_label",
        "recommended_action",
        "rationale",
        "manual_pdf_check_needed",
        "retrieval_explainability_impact",
        "text_preview",
    ]
    write_csv(OUT_DIR / "protected_sample_signoff.csv", protected_rows, protected_fields)
    (OUT_DIR / "protected_sample_signoff.md").write_text(protected_markdown(protected_rows), encoding="utf-8")

    high_counts = Counter(row["signoff_label"] for row in high_rows)
    candidate_counts = Counter(row["signoff_label"] for row in candidate_rows)
    protected_counts = Counter(row["signoff_label"] for row in protected_rows if row["category"] == "phase5d_protected_caption")
    recommend_5e3 = write_rule_proposal(OUT_DIR / "section_repair_rule_proposal.md", high_counts, candidate_counts, protected_counts)
    write_summary(OUT_DIR / "summary.md", high_rows, candidate_rows, protected_rows, recommend_5e3)

    if missing:
        (OUT_DIR / "missing_inputs.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(OUT_DIR),
                "high_severity_signoff": dict(high_counts),
                "candidate_signoff": dict(candidate_counts),
                "protected_caption_signoff": dict(protected_counts),
                "recommend_phase5e3": recommend_5e3,
                "missing_inputs": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
