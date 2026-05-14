#!/usr/bin/env python3
"""Phase 5D-2 read-only sign-off review for false/fragment captions."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "reports/phase5d_caption_cleanup_audit"
OUT_DIR = ROOT / "reports/phase5d_caption_cleanup_signoff"

INPUTS = {
    "summary": AUDIT_DIR / "summary.md",
    "false_caption_candidates": AUDIT_DIR / "false_caption_candidates.csv",
    "protected_short_captions": AUDIT_DIR / "protected_short_captions.csv",
    "signoff_sample_md": AUDIT_DIR / "signoff_sample.md",
    "signoff_sample_csv": AUDIT_DIR / "signoff_sample.csv",
    "caption_distribution_json": AUDIT_DIR / "caption_distribution.json",
    "caption_distribution_md": AUDIT_DIR / "caption_distribution.md",
}

ALLOWED_LABELS = {
    "safe_to_demote",
    "keep_as_caption",
    "needs_manual_pdf_check",
    "eval_only_noise",
    "uncertain",
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
    "promoters",
    "classification",
    "guidelines",
    "endpoints",
    "scores",
    "curves",
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
    r"^\s*(supplementary\s+)?(table|fig(?:ure)?)\.?\s+(s?\d+[a-z]?)\s*[:.]?\s*",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(
    r"^\s*(supplementary\s+)?(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*$",
    re.IGNORECASE,
)
SINGLE_OR_ARTICLE_LETTER_RE = re.compile(
    r"^\s*(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*((the|a|an)\s+)?[A-Z]\s*\.?\s*$",
    re.IGNORECASE,
)
CONTINUED_RE = re.compile(r"\b(cont\.?|continued)\b", re.IGNORECASE)
PAGE_HEADER_RE = re.compile(r"^\s*fig(?:ure)?\.?\s+\d+\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)


def norm_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def preview_text(preview: str) -> str:
    text = norm_space(preview)
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def caption_body(text: str) -> str:
    return CAPTION_PREFIX_RE.sub("", norm_space(text), count=1).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", text)


def has_semantic_anchor(text: str) -> bool:
    lower = norm_space(text).lower()
    return any(anchor in lower for anchor in SEMANTIC_ANCHORS)


def starts_with_continuation(preview: str) -> bool:
    text = preview_text(preview).lower()
    return any(re.match(rf"^{re.escape(term)}\b", text) for term in ORGANISM_CONTINUATIONS)


def has_list_context(preview: str) -> bool:
    lower = preview.lower()
    return bool(re.search(r"\b(listed|shown|summari[sz]ed|provided|reported)\s+in\b", lower))


def next_looks_like_caption_body(row: dict[str, str]) -> bool:
    nxt = preview_text(row.get("next_block_preview", ""))
    if not nxt:
        return False
    if has_semantic_anchor(nxt):
        return True
    return len(words(nxt)) >= 6 and not re.match(r"^\d+(\.\d+)*\b", nxt)


def continuation_or_reference_fragment(text: str) -> bool:
    return bool(CONTINUED_RE.search(text) or re.search(r"\bfig(?:ure)?\.\s*$", text, re.IGNORECASE))


def supplementary_in_text_reference(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in [
            "provided in supplementary",
            "presented in supplementary",
            "given supplementary",
            "data are in",
            "are in given supplementary",
            "given in the text",
            "subsequently",
        ]
    )


def classify_candidate(row: dict[str, str]) -> tuple[str, str, str, str]:
    rule = row.get("candidate_rule", "")
    text = norm_space(row.get("caption_text", ""))
    prev = row.get("previous_block_preview", "")
    nxt = row.get("next_block_preview", "")
    block_type = row.get("block_type", "")

    if rule == "article_plus_single_letter_fragment":
        return (
            "safe_to_demote",
            "Caption is only a table/figure marker plus article and one letter; nearby text shows the organism/name continues outside the caption block.",
            "demote_fragment_caption_preserve_text_with_cleanup_metadata",
            "If wrong, a real panel-level caption marker could lose caption evidence; current nearby preview makes that unlikely.",
        )

    if rule == "broken_organism_or_abbreviation_prefix":
        if starts_with_continuation(nxt) or (block_type == "table_caption" and has_list_context(prev)):
            return (
                "safe_to_demote",
                "Nearby preview shows a split organism or abbreviation prefix, e.g. E. + coli / S. + cerevisiae, rather than a standalone caption.",
                "demote_fragment_caption_preserve_text_with_cleanup_metadata",
                "If wrong, a real caption mentioning an abbreviated organism could be demoted; require the continuation evidence in 5D-3.",
            )
        return (
            "needs_manual_pdf_check",
            "The block is single-letter-like, but the preview does not strongly prove an organism/name split.",
            "do_not_clean_until_pdf_layout_check",
            "Demoting without visual/layout confirmation could remove a valid panel label or real short caption.",
        )

    if rule == "number_only_caption":
        if block_type == "table_caption" and next_looks_like_caption_body(row):
            return (
                "keep_as_caption",
                "The number-only block appears to be the table marker for a real caption whose title/body is in the next block.",
                "protect_from_length_or_number_only_cleanup",
                "Demoting this could lose table-caption evidence unless a future merger preserves the adjacent title.",
            )
        if next_looks_like_caption_body(row):
            return (
                "eval_only_noise",
                "The standalone number is noisy as an eval/probe target, but the next block appears to contain a real figure caption body.",
                "exclude_as_eval_target_or_require_caption_merge_before_cleanup",
                "Direct demotion could split a real caption marker from its descriptive body.",
            )
        return (
            "needs_manual_pdf_check",
            "Number-only captions need PDF/layout confirmation because they may be true caption markers with visually adjacent titles.",
            "do_not_clean_until_pdf_layout_check",
            "A valid caption marker could be removed from parsed caption evidence.",
        )

    if rule == "supplementary_reference_fragment":
        if starts_with_continuation(nxt) or SINGLE_OR_ARTICLE_LETTER_RE.match(text):
            return (
                "safe_to_demote",
                "Supplementary caption candidate also shows a split organism/abbreviation continuation in nearby text.",
                "demote_fragment_caption_preserve_text_with_cleanup_metadata",
                "If wrong, supplemental table evidence could be demoted; require nearby continuation evidence.",
            )
        if supplementary_in_text_reference(text):
            return (
                "eval_only_noise",
                "Text reads like an in-text reference to supplementary material rather than a standalone parsed caption.",
                "exclude_from_eval_targets; consider cleanup only with explicit in_text_reference rule",
                "Cleaning too broadly could demote real supplementary table/figure captions.",
            )
        if has_semantic_anchor(text) and len(words(caption_body(text))) >= 3:
            return (
                "keep_as_caption",
                "Despite being supplementary, the block has enough descriptive title content to be a real caption.",
                "protect_supplementary_caption_with_descriptive_title",
                "Demotion would lose real supplementary figure/table caption evidence.",
            )
        return (
            "uncertain",
            "Supplementary reference/caption boundary is ambiguous from parsed text alone.",
            "do_not_clean_without_manual_review",
            "Could either preserve noise or demote real supplemental caption evidence.",
        )

    if rule == "very_short_no_semantic_anchor":
        if PAGE_HEADER_RE.match(text):
            return (
                "safe_to_demote",
                "Caption text matches a page header/footer pattern rather than a content caption.",
                "demote_page_header_like_caption_preserve_text_with_cleanup_metadata",
                "If wrong, a continuation caption could be demoted; require exact page-count pattern.",
            )
        if starts_with_continuation(nxt) and SINGLE_OR_ARTICLE_LETTER_RE.match(text):
            return (
                "safe_to_demote",
                "Very short caption is also an organism/abbreviation split with nearby continuation evidence.",
                "demote_fragment_caption_preserve_text_with_cleanup_metadata",
                "If wrong, a valid abbreviated caption could be demoted.",
            )
        if re.search(r"\b(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*\d+\s*\.?\s*$", text, re.IGNORECASE):
            return (
                "eval_only_noise",
                "The block is a table/figure marker plus a section/list number; useful to filter from probes but not enough for automatic parsed cleanup.",
                "exclude_from_eval_targets; no parsed cleanup by default",
                "Could be a real caption continuation marker or layout artifact.",
            )
        if continuation_or_reference_fragment(text):
            return (
                "eval_only_noise",
                "Continuation/in-text-reference marker is too weak to use as a caption-level eval target.",
                "exclude_from_eval_targets; no parsed cleanup by default",
                "Some continued captions are valid and should not be demoted automatically.",
            )
        if has_semantic_anchor(text) or len(words(caption_body(text))) >= 4:
            return (
                "keep_as_caption",
                "Short text still has enough descriptive content to plausibly be a real caption.",
                "protect_short_descriptive_caption",
                "Demotion could remove concise but real caption evidence.",
            )
        return (
            "uncertain",
            "Short caption lacks anchors, but nearby parsed text is insufficient for safe cleanup.",
            "do_not_clean_without_more_review",
            "Could preserve noise, or could demote a valid compact caption.",
        )

    return (
        "uncertain",
        "No sign-off rule matched this candidate.",
        "do_not_clean_without_more_review",
        "Unknown failure mode.",
    )


def infer_protected_type(row: dict[str, str]) -> str:
    text = row.get("caption_text", "").lower().strip()
    if text.startswith(("table", "supplementary table", "tbl")):
        return "table_caption"
    if text.startswith(("fig", "figure", "supplementary figure")):
        return "figure_caption"
    return "unknown"


def evenly_sample(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    if count <= 1:
        return rows[:count]
    step = (len(rows) - 1) / (count - 1)
    picked = []
    seen = set()
    for i in range(count):
        idx = round(i * step)
        if idx in seen:
            continue
        picked.append(rows[idx])
        seen.add(idx)
    return picked


def review_protected(row: dict[str, str]) -> tuple[str, str, str]:
    text = norm_space(row.get("caption_text", ""))
    reason = row.get("protect_reason", "")
    body = caption_body(text)

    if "approved_eval_target" in reason or "doc_0367" in reason:
        return (
            "correctly_protected",
            "Approved/known-good evidence should remain protected unless manually superseded.",
            "approved_eval_or_known_good_anchor",
        )
    if NUMBER_ONLY_RE.match(text) or SINGLE_OR_ARTICLE_LETTER_RE.match(text):
        return (
            "should_be_candidate",
            "Sample has a pure number or single-letter fragment shape despite appearing in protected output.",
            "avoid_number_only_or_single_letter_protection_without_context",
        )
    if has_semantic_anchor(text):
        anchors = sorted(anchor for anchor in SEMANTIC_ANCHORS if anchor in text.lower())
        return (
            "correctly_protected",
            f"Contains semantic caption anchor(s): {', '.join(anchors[:5])}.",
            "semantic_anchor_words",
        )
    if len(words(body)) >= 4:
        return (
            "correctly_protected",
            "Short but descriptive enough to be a real caption title/body.",
            "minimum_descriptive_content",
        )
    return (
        "uncertain",
        "Short caption has limited content and no configured semantic anchor in the sampled review.",
        "manual_review_for_sparse_short_caption",
    )


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = norm_space(row.get(field, "")).replace("|", "\\|")
            if len(value) > 130:
                value = value[:129] + "..."
            values.append(value)
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name, path in INPUTS.items() if not path.exists()]

    candidates = read_csv(INPUTS["false_caption_candidates"])
    protected = read_csv(INPUTS["protected_short_captions"])

    decisions: list[dict[str, Any]] = []
    for row in candidates:
        label, rationale, action, risk = classify_candidate(row)
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unexpected label {label}")
        decisions.append(
            {
                "doc_id": row.get("doc_id", ""),
                "block_id": row.get("block_id", ""),
                "block_type": row.get("block_type", ""),
                "caption_text": row.get("caption_text", ""),
                "candidate_rule": row.get("candidate_rule", ""),
                "confidence": row.get("confidence", ""),
                "label": label,
                "rationale": rationale,
                "recommended_cleanup_action": action,
                "risk_if_wrong": risk,
            }
        )

    decision_fields = [
        "doc_id",
        "block_id",
        "block_type",
        "caption_text",
        "candidate_rule",
        "confidence",
        "label",
        "rationale",
        "recommended_cleanup_action",
        "risk_if_wrong",
    ]
    write_csv(OUT_DIR / "signoff_decisions.csv", decisions, decision_fields)

    protected_by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in protected:
        protected_by_type[infer_protected_type(row)].append(row)

    protected_sample = []
    protected_sample.extend(evenly_sample(protected_by_type["table_caption"], 30))
    protected_sample.extend(evenly_sample(protected_by_type["figure_caption"], 30))
    if len(protected_by_type["table_caption"]) < 30:
        protected_sample.extend(protected_by_type.get("unknown", []))

    protected_reviews: list[dict[str, Any]] = []
    for row in protected_sample:
        inferred_type = infer_protected_type(row)
        label, rationale, pattern = review_protected(row)
        protected_reviews.append(
            {
                "doc_id": row.get("doc_id", ""),
                "block_id": row.get("block_id", ""),
                "inferred_block_type": inferred_type,
                "caption_text": row.get("caption_text", ""),
                "protect_reason": row.get("protect_reason", ""),
                "review_label": label,
                "rationale": rationale,
                "recommended_5d3_protection_pattern": pattern,
            }
        )

    protected_fields = [
        "doc_id",
        "block_id",
        "inferred_block_type",
        "caption_text",
        "protect_reason",
        "review_label",
        "rationale",
        "recommended_5d3_protection_pattern",
    ]
    write_csv(OUT_DIR / "protected_caption_review.csv", protected_reviews, protected_fields)

    label_counts = Counter(row["label"] for row in decisions)
    pattern_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    high_counts = Counter()
    for row in decisions:
        pattern_label_counts[row["candidate_rule"]][row["label"]] += 1
        if row["confidence"] == "high":
            high_counts[row["label"]] += 1

    protected_review_counts = Counter(row["review_label"] for row in protected_reviews)
    protected_type_counts = Counter(row["inferred_block_type"] for row in protected_reviews)
    safe_rules = [
        "article_plus_single_letter_fragment.",
        "broken_organism_or_abbreviation_prefix only when nearby text strongly shows the organism/abbreviation continuation, such as E. + coli, S. + cerevisiae, C. + jejuni, B. + breve, or a table-list context ending in 'listed/shown/summarized in'.",
        "very_short page-header/footer forms only with exact patterns such as 'FIGURE 3 5 of 12'.",
        "supplementary_reference_fragment only when it has the same strong organism-continuation evidence; in-text supplementary references should be filtered from eval targets first, not parsed-cleaned by default.",
    ]
    do_not_clean = [
        "short but meaningful captions with semantic anchors.",
        "captions containing Workflow, Overview, Strains, Plasmids, Primers, Pathway, Schematic, Model, Alignment, Time course, Summary, Comparison, Effect, Analysis, Assay, Expression, Production, Growth, Activity, or similar domain nouns.",
        "approved eval target captions and the doc_0367 Figure 5 sanity anchor.",
        "number-only captions when the adjacent block looks like a real caption body unless a future merger preserves both marker and title.",
        "continued captions such as 'Figure 3. Cont.' or 'Table 1 (continued)' by default.",
        "captions with enough descriptive content even if they are short.",
    ]

    proposal = "\n".join(
        [
            "# Phase 5D-3 Cleanup Rule Proposal",
            "",
            "## Safe Rules",
            *(f"- {rule}" for rule in safe_rules),
            "",
            "## Do-Not-Clean Rules",
            *(f"- {rule}" for rule in do_not_clean),
            "",
            "## Recommended Action",
            "- Do not delete blocks.",
            "- Prefer demote/retype or metadata marking over destructive cleanup.",
            "- Preserve the original text exactly.",
            "- Record `original_block_type`.",
            "- Record `caption_cleanup_rule_id`.",
            "- Record `caption_cleanup_reason`.",
            "- Keep downstream schema compatibility; do not change chunk JSONL main fields.",
            "- Add an audit output listing every demoted block and rule id.",
            "- Run 5D-3 on an experimental output path first; default should remain off/no until reviewed.",
            "",
            "## Stop Conditions",
            "- Stop if safe_to_demote count drops materially after implementation dry-run.",
            "- Stop if any approved eval target or doc_0367 Figure 5 is touched.",
            "- Stop if protected-caption review reveals broad semantic-anchor false positives.",
            "- Stop if number-only or continued captions require PDF layout decisions to express safely.",
            "- Stop if implementation requires OCR, PDF visual extraction, schema changes, or index rebuild as a prerequisite.",
            "",
        ]
    )
    (OUT_DIR / "cleanup_rule_proposal.md").write_text(proposal, encoding="utf-8")

    pattern_rows = []
    for pattern, counts in sorted(pattern_label_counts.items()):
        pattern_rows.append(
            {
                "candidate_rule": pattern,
                "safe_to_demote": counts.get("safe_to_demote", 0),
                "keep_as_caption": counts.get("keep_as_caption", 0),
                "needs_manual_pdf_check": counts.get("needs_manual_pdf_check", 0),
                "eval_only_noise": counts.get("eval_only_noise", 0),
                "uncertain": counts.get("uncertain", 0),
            }
        )

    high_total = sum(high_counts.values())
    high_safe = high_counts.get("safe_to_demote", 0)
    high_safe_ratio = high_safe / high_total if high_total else 0.0
    recommend_5d3 = (
        label_counts.get("safe_to_demote", 0) >= 40
        and high_total > 0
        and high_safe_ratio >= 0.5
        and protected_review_counts.get("should_be_candidate", 0) <= 2
    )
    needs_pdf = (
        label_counts.get("needs_manual_pdf_check", 0) > 0
        or label_counts.get("uncertain", 0) > 0
    )

    summary_lines = [
        "# Phase 5D-2 False / Fragment Caption Sign-off Summary",
        "",
        "## Input Status",
        f"- missing Phase 5D-1 files: {', '.join(missing) if missing else 'none'}",
        "",
        "## Required Answers",
        f"1. total candidate count: {len(decisions)}.",
        f"2. safe_to_demote count: {label_counts.get('safe_to_demote', 0)}.",
        f"3. keep_as_caption count: {label_counts.get('keep_as_caption', 0)}.",
        f"4. needs_manual_pdf_check count: {label_counts.get('needs_manual_pdf_check', 0)}.",
        f"5. eval_only_noise count: {label_counts.get('eval_only_noise', 0)}.",
        f"6. uncertain count: {label_counts.get('uncertain', 0)}.",
        "7. pattern label distribution:",
        markdown_table(
            pattern_rows,
            [
                "candidate_rule",
                "safe_to_demote",
                "keep_as_caption",
                "needs_manual_pdf_check",
                "eval_only_noise",
                "uncertain",
            ],
        ),
        f"8. high-confidence safe_to_demote ratio: {high_safe}/{high_total} = {high_safe_ratio:.1%}.",
        f"9. protected short caption review: {dict(protected_review_counts)} across {dict(protected_type_counts)}; protected short captions are common and mostly real, so length-only cleanup is unsafe.",
        f"10. recommendation to enter Phase 5D-3: {'yes' if recommend_5d3 else 'no'}; use an experimental default-off path first.",
        "11. implementable rules: article_plus_single_letter_fragment; broken organism/abbreviation prefix with strong nearby continuation evidence; exact page-header/footer fragments; supplementary fragments only with strong organism-continuation evidence.",
        "12. rules not implementable now: broad number_only_caption cleanup; broad very_short_no_semantic_anchor cleanup; continued caption cleanup; any cleanup of approved eval targets, doc_0367 Figure 5, or semantic-anchor captions.",
        "13. 5D-3 default enabled: no; default should remain off/no until dry-run and sign-off.",
        f"14. PDF manual check needed: {'yes for the explicit needs_manual_pdf_check rows and optional follow-up for uncertain rows, but not as a prerequisite for the narrow safe rules' if needs_pdf else 'no'}.",
        "",
        "## Protected Review Conclusion",
        "- Protected short captions are numerous and mostly meaningful in the sampled review.",
        "- This confirms that caption length alone must not drive cleanup.",
        "- 5D-3 protection should include semantic anchors, approved eval targets, doc_0367 Figure 5, descriptive body length, and continued-caption guards.",
        "",
        "## Output Files",
        "- signoff_decisions.csv",
        "- protected_caption_review.csv",
        "- cleanup_rule_proposal.md",
        "- summary.md",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "candidate_count": len(decisions),
                "label_counts": dict(label_counts),
                "high_safe_ratio": f"{high_safe}/{high_total}",
                "protected_review_counts": dict(protected_review_counts),
                "recommend_phase5d3": recommend_5d3,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
