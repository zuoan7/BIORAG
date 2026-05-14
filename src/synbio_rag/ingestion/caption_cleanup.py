from __future__ import annotations

import copy
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


CAPTION_TYPES = {"table_caption", "figure_caption"}
SAFE_LABEL = "safe_to_demote"
PHASE = "phase5d3"

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
    "jejuni",
    "breve",
}

CAPTION_PREFIX_RE = re.compile(
    r"^\s*(?:supplementary\s+)?(?:table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*",
    re.IGNORECASE,
)
ARTICLE_PLUS_SINGLE_LETTER_RE = re.compile(
    r"^\s*(?:table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*(?:the|a|an)\s+[A-Z]\s*\.?\s*$",
    re.IGNORECASE,
)
SINGLE_LETTER_RE = re.compile(
    r"^\s*(?:table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*[A-Z]\s*\.?\s*$",
    re.IGNORECASE,
)
PAGE_HEADER_RE = re.compile(
    r"^\s*fig(?:ure)?\.?\s+\d+\s+\d+\s+of\s+\d+\s*$",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(
    r"^\s*(?:supplementary\s+)?(?:table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*$",
    re.IGNORECASE,
)
CONTINUED_RE = re.compile(r"\b(cont\.?|continued)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SignoffDecision:
    doc_id: str
    block_id: str
    block_type: str
    caption_text: str
    candidate_rule: str
    confidence: str
    label: str
    rationale: str
    recommended_cleanup_action: str
    risk_if_wrong: str


@dataclass
class CleanupAuditRow:
    doc_id: str
    source_file: str
    block_id: str
    original_block_type: str
    new_block_type: str
    caption_text: str
    cleanup_action: str
    caption_cleanup_rule_id: str
    caption_cleanup_reason: str
    confidence: str
    protected_hit: bool
    signoff_label: str
    nearby_preview: str
    safe_to_demote: bool


@dataclass
class CleanupStats:
    total_docs: int = 0
    processed_docs: int = 0
    success_docs: int = 0
    failed_docs: int = 0
    cleanup_candidates_seen: int = 0
    demoted_count: int = 0
    skipped_eval_only_noise_count: int = 0
    skipped_uncertain_count: int = 0
    skipped_protected_count: int = 0
    skipped_manual_pdf_check_count: int = 0
    skipped_keep_as_caption_count: int = 0
    skipped_rule_guard_count: int = 0
    table_caption_before: int = 0
    table_caption_after: int = 0
    figure_caption_before: int = 0
    figure_caption_after: int = 0
    protected_caption_violations: int = 0
    doc_0367_figure5_preserved: bool = True
    approved_eval_target_preserved: bool | None = None
    demoted_by_rule: Counter[str] = field(default_factory=Counter)
    skipped_by_label: Counter[str] = field(default_factory=Counter)
    skipped_by_rule: Counter[str] = field(default_factory=Counter)
    failed_docs_list: list[str] = field(default_factory=list)


def norm_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def block_key(doc_id: str, block_id: str) -> tuple[str, str]:
    return norm_space(doc_id), norm_space(block_id)


def read_signoff_decisions(path: Path) -> dict[tuple[str, str], SignoffDecision]:
    decisions: dict[tuple[str, str], SignoffDecision] = {}
    if not path.exists():
        return decisions
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            decision = SignoffDecision(
                doc_id=norm_space(row.get("doc_id")),
                block_id=norm_space(row.get("block_id")),
                block_type=norm_space(row.get("block_type")),
                caption_text=norm_space(row.get("caption_text")),
                candidate_rule=norm_space(row.get("candidate_rule")),
                confidence=norm_space(row.get("confidence")),
                label=norm_space(row.get("label")),
                rationale=norm_space(row.get("rationale")),
                recommended_cleanup_action=norm_space(row.get("recommended_cleanup_action")),
                risk_if_wrong=norm_space(row.get("risk_if_wrong")),
            )
            decisions[block_key(decision.doc_id, decision.block_id)] = decision
    return decisions


def read_protected_keys(paths: Iterable[Path]) -> set[tuple[str, str]]:
    protected: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                review_label = norm_space(row.get("review_label"))
                if review_label == "should_be_candidate":
                    continue
                doc_id = norm_space(row.get("doc_id"))
                block_id = norm_space(row.get("block_id"))
                if doc_id and block_id:
                    protected.add(block_key(doc_id, block_id))
    return protected


def flatten_blocks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in doc.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            blocks.append(block)
    return blocks


def caption_body(text: str) -> str:
    return CAPTION_PREFIX_RE.sub("", norm_space(text), count=1).strip()


def has_semantic_anchor(text: str) -> bool:
    lower = norm_space(text).lower()
    return any(anchor in lower for anchor in SEMANTIC_ANCHORS)


def _text_from_preview(preview: str) -> str:
    preview = norm_space(preview)
    if ":" in preview:
        return preview.split(":", 1)[1].strip()
    return preview


def _starts_with_organism_continuation(text: str) -> bool:
    lower = norm_space(text).lower()
    return any(re.match(rf"^{re.escape(term)}\b", lower) for term in ORGANISM_CONTINUATIONS)


def _has_list_context(text: str) -> bool:
    return bool(re.search(r"\b(listed|shown|summari[sz]ed|provided|reported)\s+in\b", text, re.IGNORECASE))


def _previous_next(blocks: list[dict[str, Any]], index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    prev_block = blocks[index - 1] if index > 0 else None
    next_block = blocks[index + 1] if index + 1 < len(blocks) else None
    return prev_block, next_block


def _preview(block: dict[str, Any] | None, limit: int = 180) -> str:
    if not block:
        return ""
    btype = norm_space(block.get("type"))
    text = norm_space(block.get("text"))
    if len(text) > limit:
        text = text[: limit - 1] + "..."
    return f"{btype}: {text}" if btype else text


def _strong_organism_continuation(
    block: dict[str, Any],
    prev_block: dict[str, Any] | None,
    next_block: dict[str, Any] | None,
) -> bool:
    next_text = norm_space(next_block.get("text") if next_block else "")
    prev_text = norm_space(prev_block.get("text") if prev_block else "")
    if _starts_with_organism_continuation(next_text):
        return True
    if norm_space(block.get("type")) == "table_caption" and _has_list_context(prev_text):
        return True
    return False


def _is_descriptive_caption(text: str) -> bool:
    body = caption_body(text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", body)
    return has_semantic_anchor(text) or len(words) >= 4


def should_demote_caption(
    block: dict[str, Any],
    decision: SignoffDecision,
    prev_block: dict[str, Any] | None,
    next_block: dict[str, Any] | None,
    *,
    protected_hit: bool,
) -> tuple[bool, str, str, str]:
    """Return (should_demote, action, rule_id, reason)."""
    text = norm_space(block.get("text"))
    btype = norm_space(block.get("type"))

    if protected_hit:
        return False, "skip_protected", "", "Block is present in protected review/list."
    if decision.label != SAFE_LABEL:
        return False, f"skip_{decision.label}", "", f"Sign-off label is {decision.label}."
    if btype not in CAPTION_TYPES:
        return False, "skip_not_caption", "", f"Current block type is {btype}."
    if text != decision.caption_text:
        return False, "skip_text_mismatch", "", "Parsed block text differs from sign-off text."
    if _is_descriptive_caption(text) and not PAGE_HEADER_RE.match(text):
        return False, "skip_descriptive_caption", "", "Caption has semantic/descriptive content."
    if NUMBER_ONLY_RE.match(text):
        return False, "skip_number_only_guard", "", "Broad number-only caption cleanup is disabled."
    if CONTINUED_RE.search(text):
        return False, "skip_continued_guard", "", "Continued captions are protected by default."

    rule = decision.candidate_rule
    if rule == "article_plus_single_letter_fragment":
        if ARTICLE_PLUS_SINGLE_LETTER_RE.match(text) and _strong_organism_continuation(block, prev_block, next_block):
            return (
                True,
                "demote_to_paragraph",
                "phase5d3_article_plus_single_letter_fragment",
                "Sign-off safe_to_demote and nearby text supports organism/name continuation fragment.",
            )
        return False, "skip_rule_guard", "", "Article-plus-single-letter rule did not pass nearby continuation guard."

    if rule == "broken_organism_or_abbreviation_prefix":
        if (
            (SINGLE_LETTER_RE.match(text) or ARTICLE_PLUS_SINGLE_LETTER_RE.match(text))
            and _strong_organism_continuation(block, prev_block, next_block)
        ):
            return (
                True,
                "demote_to_paragraph",
                "phase5d3_broken_organism_or_abbreviation_prefix",
                "Sign-off safe_to_demote and nearby text strongly indicates split organism/abbreviation.",
            )
        return False, "skip_rule_guard", "", "Broken-organism rule did not pass single-letter and nearby continuation guards."

    if rule == "very_short_no_semantic_anchor":
        if PAGE_HEADER_RE.match(text):
            return (
                True,
                "demote_to_paragraph",
                "phase5d3_page_header_footer_caption_fragment",
                "Sign-off safe_to_demote and caption matches exact page-header/footer fragment pattern.",
            )
        return False, "skip_rule_guard", "", "Broad very_short_no_semantic_anchor cleanup is disabled."

    if rule == "supplementary_reference_fragment":
        if (
            (SINGLE_LETTER_RE.match(text) or ARTICLE_PLUS_SINGLE_LETTER_RE.match(text))
            and _strong_organism_continuation(block, prev_block, next_block)
        ):
            return (
                True,
                "demote_to_paragraph",
                "phase5d3_supplementary_fragment_with_continuation",
                "Sign-off safe_to_demote and supplementary fragment has strong organism continuation evidence.",
            )
        return False, "skip_rule_guard", "", "Supplementary fragment lacks strong continuation evidence for parsed cleanup."

    return False, "skip_rule_guard", "", f"Rule {rule} is not enabled in conservative mode."


def apply_cleanup_to_doc(
    doc: dict[str, Any],
    decisions: dict[tuple[str, str], SignoffDecision],
    protected_keys: set[tuple[str, str]],
) -> tuple[dict[str, Any], list[CleanupAuditRow], CleanupStats]:
    doc_id = norm_space(doc.get("doc_id"))
    source_file = norm_space(doc.get("source_file"))
    new_doc = copy.deepcopy(doc)
    blocks = flatten_blocks(new_doc)
    stats = CleanupStats(total_docs=1, processed_docs=1, success_docs=1)

    for block in blocks:
        btype = norm_space(block.get("type"))
        if btype == "table_caption":
            stats.table_caption_before += 1
            stats.table_caption_after += 1
        elif btype == "figure_caption":
            stats.figure_caption_before += 1
            stats.figure_caption_after += 1

    audit_rows: list[CleanupAuditRow] = []
    for index, block in enumerate(blocks):
        block_id = norm_space(block.get("block_id") or block.get("id"))
        key = block_key(doc_id, block_id)
        decision = decisions.get(key)
        if not decision:
            continue

        stats.cleanup_candidates_seen += 1
        protected_hit = key in protected_keys
        prev_block, next_block = _previous_next(blocks, index)
        original_type = norm_space(block.get("type"))
        should_demote, action, rule_id, reason = should_demote_caption(
            block,
            decision,
            prev_block,
            next_block,
            protected_hit=protected_hit,
        )
        new_type = original_type
        if should_demote:
            metadata = dict(block.get("metadata", {}) or {})
            metadata.update(
                {
                    "original_block_type": original_type,
                    "caption_cleanup_rule_id": rule_id,
                    "caption_cleanup_reason": reason,
                    "caption_cleanup_confidence": decision.confidence,
                    "caption_cleanup_action": action,
                    "caption_cleanup_enabled": True,
                    "caption_cleanup_phase": PHASE,
                    "protected_caption": False,
                }
            )
            block["metadata"] = metadata
            block["type"] = "paragraph"
            new_type = "paragraph"
            stats.demoted_count += 1
            stats.demoted_by_rule[rule_id] += 1
            if original_type == "table_caption":
                stats.table_caption_after -= 1
            elif original_type == "figure_caption":
                stats.figure_caption_after -= 1
        else:
            stats.skipped_by_label[decision.label] += 1
            if protected_hit:
                stats.skipped_protected_count += 1
                stats.skipped_by_rule["protected_hit"] += 1
            elif decision.label == "eval_only_noise":
                stats.skipped_eval_only_noise_count += 1
            elif decision.label == "uncertain":
                stats.skipped_uncertain_count += 1
            elif decision.label == "needs_manual_pdf_check":
                stats.skipped_manual_pdf_check_count += 1
            elif decision.label == "keep_as_caption":
                stats.skipped_keep_as_caption_count += 1
            elif decision.label == SAFE_LABEL:
                stats.skipped_rule_guard_count += 1
                stats.skipped_by_rule[action] += 1

        audit_rows.append(
            CleanupAuditRow(
                doc_id=doc_id,
                source_file=source_file,
                block_id=block_id,
                original_block_type=original_type,
                new_block_type=new_type,
                caption_text=norm_space(block.get("text")),
                cleanup_action=action,
                caption_cleanup_rule_id=rule_id,
                caption_cleanup_reason=reason,
                confidence=decision.confidence,
                protected_hit=protected_hit,
                signoff_label=decision.label,
                nearby_preview=" | ".join(part for part in (_preview(prev_block), _preview(next_block)) if part),
                safe_to_demote=decision.label == SAFE_LABEL,
            )
        )

    if doc_id == "doc_0367":
        stats.doc_0367_figure5_preserved = _doc_0367_figure5_preserved(blocks)

    return new_doc, audit_rows, stats


def _doc_0367_figure5_preserved(blocks: list[dict[str, Any]]) -> bool:
    for block in blocks:
        text = norm_space(block.get("text"))
        if re.match(r"^\s*figure\s+5\b", text, re.IGNORECASE) and "Opto-T7RNAP" in text:
            return norm_space(block.get("type")) == "figure_caption"
    return True


def merge_stats(target: CleanupStats, source: CleanupStats) -> None:
    for field_name in (
        "total_docs",
        "processed_docs",
        "success_docs",
        "failed_docs",
        "cleanup_candidates_seen",
        "demoted_count",
        "skipped_eval_only_noise_count",
        "skipped_uncertain_count",
        "skipped_protected_count",
        "skipped_manual_pdf_check_count",
        "skipped_keep_as_caption_count",
        "skipped_rule_guard_count",
        "table_caption_before",
        "table_caption_after",
        "figure_caption_before",
        "figure_caption_after",
        "protected_caption_violations",
    ):
        setattr(target, field_name, getattr(target, field_name) + getattr(source, field_name))
    target.doc_0367_figure5_preserved = target.doc_0367_figure5_preserved and source.doc_0367_figure5_preserved
    target.demoted_by_rule.update(source.demoted_by_rule)
    target.skipped_by_label.update(source.skipped_by_label)
    target.skipped_by_rule.update(source.skipped_by_rule)
    target.failed_docs_list.extend(source.failed_docs_list)


def count_caption_types_in_dir(input_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted(input_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        for block in flatten_blocks(doc):
            btype = norm_space(block.get("type"))
            if btype in CAPTION_TYPES:
                counts[btype] += 1
    return counts


def audit_row_to_dict(row: CleanupAuditRow) -> dict[str, Any]:
    return {
        "doc_id": row.doc_id,
        "source_file": row.source_file,
        "block_id": row.block_id,
        "original_block_type": row.original_block_type,
        "new_block_type": row.new_block_type,
        "caption_text": row.caption_text,
        "cleanup_action": row.cleanup_action,
        "caption_cleanup_rule_id": row.caption_cleanup_rule_id,
        "caption_cleanup_reason": row.caption_cleanup_reason,
        "confidence": row.confidence,
        "protected_hit": str(row.protected_hit).lower(),
        "signoff_label": row.signoff_label,
        "nearby_preview": row.nearby_preview,
        "safe_to_demote": str(row.safe_to_demote).lower(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json_doc(path: Path, doc: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
