from __future__ import annotations

from collections import Counter

from ...domain.config import GenerationConfig
from ...domain.schemas import QueryIntent
from .models import SupportItem


def _ensure_protected_support_seeds(
    scored: list[SupportItem],
    selected: list[SupportItem],
    config: GenerationConfig,
) -> list[SupportItem]:
    """Ensure top-N protected rerank seeds are in selected support.

    Protected seeds are identified by their rerank_rank metadata (set by pipleine).
    Returns a modified selected list with protected seeds inserted if missing.
    """
    protect_k = config.v2_protect_support_seeds_top_k
    if protect_k <= 0 or not scored:
        return selected

    # Identify protected candidates: rerank_rank <= protect_k
    protected = []
    for item in scored:
        rank = item.candidate.metadata.get("rerank_rank", 999)
        if isinstance(rank, (int, float)) and int(rank) <= protect_k:
            protected.append(item)

    if not protected:
        return selected

    selected_ids = {s.evidence_id for s in selected}
    to_insert = [p for p in protected if p.evidence_id not in selected_ids]

    if not to_insert:
        return selected

    # Get max support size from existing selected
    max_size = len(selected) if selected else max(config.v2_max_support_factoid, 1)

    # Insert protected seeds at front, keep existing selected
    result = list(to_insert[:protect_k])
    protected_docs = {item.candidate.doc_id for item in result}
    remaining = [s for s in selected if s.evidence_id not in {r.evidence_id for r in result}]
    # Phase 21A-9G: diversity-aware truncation — prefer docs not yet represented
    distinct_first = [s for s in remaining if s.candidate.doc_id not in protected_docs]
    same_doc = [s for s in remaining if s.candidate.doc_id in protected_docs]
    for s in distinct_first:
        result.append(s)
        protected_docs.add(s.candidate.doc_id)
    for s in same_doc:
        if len(result) >= max_size:
            break
        result.append(s)

    return result[:max_size]


def _retain_doc_diversity(
    selected: list[SupportItem],
    scored: list[SupportItem],
    all_scored: list[SupportItem],
    config: GenerationConfig,
) -> list[SupportItem]:
    """After primary selection, retain doc diversity by limiting same-doc duplicates.

    If a doc appears 2+ times in selected AND an unselected item from a
    different doc has a comparable support score, replace the lowest-scoring
    duplicate.  When selected is empty, add the best available item as a
    support floor so that citation binding has at least one candidate.
    """
    if not selected:
        # Support floor: when selection is empty (all below min_score),
        # add the single best-scored item from all_scored so citation
        # binding has at least one candidate.
        if all_scored:
            best = max(all_scored, key=lambda item: item.support_score)
            best.reasons.append("support_floor_empty_selection")
            return [best]
        return selected

    doc_counts: Counter[str] = Counter(item.candidate.doc_id for item in selected)
    overcrowded = sorted(
        [doc for doc, cnt in doc_counts.items() if cnt >= 2],
        key=lambda d: min(item.support_score for item in selected if item.candidate.doc_id == d),
    )

    if not overcrowded:
        return selected

    selected_ids = {item.evidence_id for item in selected}
    result = list(selected)

    for doc in overcrowded:
        doc_items = [(i, result[i]) for i in range(len(result)) if result[i].candidate.doc_id == doc]
        if len(doc_items) < 2:
            continue
        doc_items.sort(key=lambda x: x[1].support_score)
        lowest_idx, lowest = doc_items[0]

        # Collect alternatives from docs not in selected
        current_docs = {item.candidate.doc_id for item in result}
        alternatives = [
            item for item in all_scored
            if item.evidence_id not in selected_ids
            and item.candidate.doc_id not in current_docs
        ]
        if not alternatives:
            break

        alternatives.sort(key=lambda item: item.support_score, reverse=True)
        best_alt = alternatives[0]

        # Require alternative score >= 85% of the replaced item's score
        if best_alt.support_score < lowest.support_score * 0.85:
            continue

        best_alt.reasons.append("doc_diversity_retention_swap")
        result[lowest_idx] = best_alt
        selected_ids.add(best_alt.evidence_id)

    return result


def _apply_close_margin_capacity_plus_one(
    selected: list[SupportItem],
    scored: list[SupportItem],
    config: GenerationConfig,
    intent,
) -> list[SupportItem]:
    """When selection is at capacity and a distinct-doc candidate has a score
    within 20% of the lowest selected item, expand capacity by one.

    Constraints:
    - Only when selected is non-empty (not negative route).
    - Candidate must be from a doc not already in selected.
    - Candidate score must be >= 80% of the lowest selected score.
    - At most one extra item is added.
    - Does not replace existing items.
    """
    if not selected:
        return selected

    if intent == QueryIntent.NEGATIVE:
        return selected

    selected_ids = {item.evidence_id for item in selected}
    selected_docs = {item.candidate.doc_id for item in selected}
    selected_scores = [item.support_score for item in selected]
    lowest_selected = min(selected_scores)
    threshold = lowest_selected * 0.80

    # Find distinct-doc candidates above the close-margin threshold
    candidates = [
        item for item in scored
        if item.evidence_id not in selected_ids
        and item.candidate.doc_id not in selected_docs
        and item.support_score >= threshold
    ]
    if not candidates:
        return selected

    candidates.sort(key=lambda item: item.support_score, reverse=True)
    best = candidates[0]
    best.reasons.append("close_margin_capacity_plus_one")
    return selected + [best]
