from __future__ import annotations

import re
from collections import Counter

from ...domain.config import GenerationConfig
from .models import SupportItem


_EN_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'_.-]*", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{1,4}")


def _select_summary(
    question: str,
    scored: list[SupportItem],
    config: GenerationConfig,
    finalizer,
) -> tuple[list[SupportItem], dict[str, object]]:
    min_summary_support = 2
    preferred_max_support = min(config.v2_max_support_summary, 3)
    ranked_all = sorted(scored, key=_summary_rank_key)
    # Phase 21A-9M: compute median support_score for quality filter bypass
    all_scores = sorted(item.support_score for item in ranked_all)
    scored_median = all_scores[len(all_scores) // 2] if all_scores else 0.0
    quality_pool: list[SupportItem] = []
    excluded: list[dict[str, str]] = []
    top_score_count = max(2, min(4, len(ranked_all)))
    for rank_index, item in enumerate(ranked_all):
        eligible, reasons = _evaluate_summary_quality(
            item,
            rank_index=rank_index,
            top_score_count=top_score_count,
            scored_median=scored_median,
        )
        if not eligible:
            excluded.append({"evidence_id": item.evidence_id, "reason": reasons[0] if reasons else "low_quality"})
            continue
        for reason in reasons:
            item.reasons.append(reason)
        quality_pool.append(item)

    ranked = sorted(quality_pool, key=_summary_rank_key)
    deduped_ranked: list[SupportItem] = []
    for item in ranked:
        duplicate_reason = _find_summary_duplicate_reason(item, deduped_ranked)
        if duplicate_reason:
            item.reasons.append("duplicate_filtered")
            excluded.append({"evidence_id": item.evidence_id, "reason": duplicate_reason})
            continue
        deduped_ranked.append(item)

    selected: list[SupportItem] = []
    per_doc: Counter[str] = Counter()
    qualified_docs = {item.candidate.doc_id for item in deduped_ranked}
    same_doc_allowed = _is_explicit_single_doc_summary(question, deduped_ranked) or _all_high_quality_same_doc(deduped_ranked)
    max_per_doc = 3 if same_doc_allowed else 2

    for item in deduped_ranked:
        if len(selected) >= min_summary_support:
            break
        if per_doc[item.candidate.doc_id] >= max_per_doc:
            continue
        if _should_defer_for_doc_diversity(item, selected, deduped_ranked, max_per_doc):
            continue
        if len(selected) < min_summary_support:
            item.reasons.append("summary_min_support_fill")
        else:
            item.reasons.append("summary_secondary_support")
        if len(qualified_docs) > 1 and per_doc[item.candidate.doc_id] == 0:
            item.reasons.append("summary_doc_diversity")
        if same_doc_allowed and len(qualified_docs) == 1:
            item.reasons.append("summary_same_doc_allowed")
        selected.append(item)
        per_doc[item.candidate.doc_id] += 1
    if len(selected) < min_summary_support and len(selected) < min(len(deduped_ranked), preferred_max_support):
        for item in deduped_ranked:
            if len(selected) >= min(min_summary_support, len(deduped_ranked)) or len(selected) >= preferred_max_support:
                break
            if item.evidence_id in {selected_item.evidence_id for selected_item in selected}:
                continue
            if per_doc[item.candidate.doc_id] >= max_per_doc:
                continue
            item.reasons.append("summary_min_support_fill")
            if same_doc_allowed and len(qualified_docs) == 1:
                item.reasons.append("summary_same_doc_allowed")
            selected.append(item)
            per_doc[item.candidate.doc_id] += 1

    for item in deduped_ranked:
        if len(selected) >= preferred_max_support:
            break
        if item.evidence_id in {selected_item.evidence_id for selected_item in selected}:
            continue
        if per_doc[item.candidate.doc_id] >= max_per_doc:
            continue
        item.reasons.append("summary_secondary_support")
        if same_doc_allowed and len(qualified_docs) == 1:
            item.reasons.append("summary_same_doc_allowed")
        selected.append(item)
        per_doc[item.candidate.doc_id] += 1

    insufficient_qualified = len(deduped_ranked) < min_summary_support
    if insufficient_qualified:
        for item in selected:
            item.reasons.append("insufficient_qualified_summary_support")
            item.reasons.append("insufficient_summary_support")

    finalized = finalizer(selected, "summary_selection")

    # Diagnostic counts
    selected_sections = [_summary_section_bucket(item.candidate.section) for item in finalized]
    abstract_concl_count = sum(1 for s in selected_sections if s in ("abstract", "conclusion", "conclusions"))
    fragment_body_count = sum(1 for s in selected_sections if s in ("results", "discussion", "introduction"))
    bibliography_count = sum(1 for item in finalized if item.candidate.features.get("bibliography_like"))

    debug = {
        "is_summary": True,
        "candidate_count": len(scored),
        "qualified_count": len(deduped_ranked),
        "selected_count": len(finalized),
        "target_min_support": min_summary_support,
        "max_support": config.v2_max_support_summary,
        "selected_evidence_ids": [item.evidence_id for item in finalized],
        "excluded": excluded,
        "section_distribution": dict(Counter(selected_sections)),
        "doc_distribution": dict(Counter(item.candidate.doc_id for item in finalized)),
        "insufficient_qualified_summary_support": insufficient_qualified,
        "summary_section_boost_applied": True,
        "summary_source_section_distribution": dict(Counter(selected_sections)),
        "abstract_or_conclusion_support_count": abstract_concl_count,
        "fragmentary_body_support_count": fragment_body_count,
        "bibliography_like_chunk_count": bibliography_count,
    }
    return finalized, debug


def _summary_rank_key(item: SupportItem) -> tuple[int, int, int, float]:
    return (
        _section_priority(item.candidate.section),
        0 if item.candidate.features.get("has_result_terms") else 1,
        0 if item.candidate.features.get("has_numeric") else 1,
        -item.support_score,
    )


def _section_priority(section: str) -> int:
    """Summary route section priority: Abstract/Conclusion > Results+Discussion > Results > Discussion > Intro > other.

    Reason: summary answers should be grounded in summary-level sections (Abstract/Conclusion)
    rather than fragmented body-text (Results/Discussion/Introduction). Body sections are still
    eligible but rank lower, preventing them from pushing out more synthesis-friendly evidence.
    """
    lowered = (section or "").lower()
    # Summary-level sections (highest priority for synthesis)
    if "abstract" in lowered:
        return 0
    if "conclusion" in lowered:
        return 1
    # Combined result+discussion — still good
    if "result" in lowered and "discussion" in lowered:
        return 2
    # Full text — parser fallback, treat as body evidence
    if "full text" in lowered:
        return 2
    # Standalone results — moderate
    if "result" in lowered:
        return 3
    # Standalone discussion — lower
    if "discussion" in lowered:
        return 4
    # Introduction — background, not synthesis-friendly
    if "introduction" in lowered:
        return 5
    # Reference/bibliography lists — lowest
    if "reference" in lowered or "bibliograph" in lowered:
        return 7
    return 6


def _evaluate_summary_quality(
    item: SupportItem,
    *,
    rank_index: int,
    top_score_count: int,
    scored_median: float | None = None,
) -> tuple[bool, list[str]]:
    section = (item.candidate.section or "").lower()
    if any(token in section for token in ("reference", "bibliograph")):
        return False, ["references_section"]
    if any(token in section for token in ("acknowledg", "author", "content")):
        return False, ["low_value_section"]
    if "list" in section and ("figure" in section or "table" in section):
        return False, ["figure_table_list"]
    text_length = int(item.candidate.features.get("text_length") or len(item.candidate.text or ""))
    if text_length < 12:
        return False, ["too_short"]

    # Phase 21A-9M: bypass quality filter for evidence with support_score above median
    if scored_median is not None and item.support_score >= scored_median:
        return True, ["summary_quality_pass", "high_score_bypass"]

    reasons = ["summary_quality_pass"]
    section_bucket = _summary_section_bucket(item.candidate.section)
    reasons.append(f"summary_section:{section_bucket}")
    has_query_overlap = "query_overlap" in item.reasons
    if has_query_overlap:
        reasons.append("query_overlap")
    has_structured = any(
        item.candidate.features.get(name) for name in ("has_table_text", "has_table_caption", "has_figure_caption")
    )
    if item.candidate.features.get("has_result_terms"):
        reasons.append("has_result_terms")
    if item.candidate.features.get("has_numeric"):
        reasons.append("has_numeric")
    if has_structured:
        reasons.append("has_table_or_figure")
    top_ranked = rank_index < top_score_count
    if has_query_overlap or _section_priority(item.candidate.section) <= 4 or item.candidate.features.get("has_result_terms") or item.candidate.features.get("has_numeric") or has_structured or top_ranked:
        return True, list(dict.fromkeys(reasons))
    return False, ["low_quality"]


def _is_explicit_single_doc_summary(question: str, items: list[SupportItem]) -> bool:
    question_lower = question.lower()
    if any(phrase in question_lower for phrase in ("this paper", "this article", "single paper", "这篇论文", "该论文", "这篇文章")):
        return True
    return bool(items) and len({item.candidate.doc_id for item in items[:3]}) == 1


def _all_high_quality_same_doc(items: list[SupportItem]) -> bool:
    return len(items) >= 2 and len({item.candidate.doc_id for item in items[:3]}) == 1


def _summary_section_bucket(section: str) -> str:
    lowered = (section or "").lower()
    if "result" in lowered and "discussion" in lowered:
        return "results_and_discussion"
    if "result" in lowered:
        return "results"
    if "discussion" in lowered:
        return "discussion"
    if "abstract" in lowered:
        return "abstract"
    if "introduction" in lowered:
        return "introduction"
    if "reference" in lowered or "bibliograph" in lowered:
        return "references"
    if "author" in lowered:
        return "author"
    if "title" in lowered:
        return "title"
    return "other"


def _should_defer_for_doc_diversity(
    item: SupportItem,
    selected: list[SupportItem],
    ranked: list[SupportItem],
    max_per_doc: int,
) -> bool:
    if len(selected) >= 2 or not selected:
        return False
    selected_docs = {selected_item.candidate.doc_id for selected_item in selected}
    if item.candidate.doc_id not in selected_docs:
        return False
    remaining_other_doc = any(
        candidate.candidate.doc_id not in selected_docs
        and candidate.evidence_id not in {selected_item.evidence_id for selected_item in selected}
        for candidate in ranked
    )
    return remaining_other_doc and sum(1 for selected_item in selected if selected_item.candidate.doc_id == item.candidate.doc_id) < max_per_doc


def _find_summary_duplicate_reason(item: SupportItem, selected: list[SupportItem]) -> str | None:
    for existing in selected:
        if item.candidate.chunk_id == existing.candidate.chunk_id:
            return "duplicate_chunk"
        if (
            item.candidate.doc_id == existing.candidate.doc_id
            and _summary_section_bucket(item.candidate.section) == _summary_section_bucket(existing.candidate.section)
            and _token_overlap_ratio(item.candidate.text, existing.candidate.text) >= 0.8
        ):
            return "duplicate"
    return None


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _tokenize(text: str) -> list[str]:
    english = [token.lower() for token in _EN_TOKEN_PATTERN.findall(text)]
    cjk = _CJK_PATTERN.findall(text)
    return english + cjk
