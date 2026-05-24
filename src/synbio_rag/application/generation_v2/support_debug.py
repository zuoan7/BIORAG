from __future__ import annotations

from ...domain.config import GenerationConfig
from .models import EvidenceCandidate, SupportItem


def _build_selection_debug(
    *,
    candidates: list[EvidenceCandidate],
    all_scored: list[SupportItem],
    scored: list[SupportItem],
    below_min_score: list[SupportItem],
    selected_before_protection: list[SupportItem],
    selected: list[SupportItem],
    config: GenerationConfig,
    intent: str,
) -> dict[str, object]:
    selected_ids = {item.evidence_id for item in selected}
    selected_before_ids = {item.evidence_id for item in selected_before_protection}
    below_min_ids = {item.evidence_id for item in below_min_score}
    scored_by_id = {item.evidence_id: item for item in all_scored}
    drop_reasons: dict[str, str] = {}
    for candidate in candidates:
        if candidate.evidence_id in selected_ids:
            continue
        if candidate.evidence_id in below_min_ids:
            drop_reasons[candidate.evidence_id] = "score_too_low"
            continue
        item = scored_by_id.get(candidate.evidence_id)
        if item and any("duplicate_filtered" in reason for reason in item.reasons):
            drop_reasons[candidate.evidence_id] = "duplicate_chunk_id"
            continue
        drop_reasons[candidate.evidence_id] = "support_pack_size_limit"

    protected_ids = []
    for item in scored:
        rank = item.candidate.metadata.get("rerank_rank", 999)
        if isinstance(rank, (int, float)) and int(rank) <= config.v2_protect_support_seeds_top_k:
            protected_ids.append(item.evidence_id)
    inserted_ids = [
        item.evidence_id
        for item in selected
        if item.evidence_id not in selected_before_ids and item.evidence_id in protected_ids
    ]
    return {
        "intent": intent,
        "candidate_count": len(candidates),
        "scored_count": len(all_scored),
        "eligible_count": len(scored),
        "below_min_score_count": len(below_min_score),
        "selected_before_protection_evidence_ids": [item.evidence_id for item in selected_before_protection],
        "selected_evidence_ids": [item.evidence_id for item in selected],
        "protected_seed_evidence_ids": protected_ids,
        "protected_seed_inserted_evidence_ids": inserted_ids,
        "drop_reasons_by_evidence_id": drop_reasons,
        "support_score_ranking": _support_score_ranking(
            all_scored=all_scored,
            selected_ids=selected_ids,
            selected_before_ids=selected_before_ids,
            protected_ids=set(protected_ids),
            inserted_ids=set(inserted_ids),
            drop_reasons=drop_reasons,
        ),
    }


def _support_score_ranking(
    *,
    all_scored: list[SupportItem],
    selected_ids: set[str],
    selected_before_ids: set[str],
    protected_ids: set[str],
    inserted_ids: set[str],
    drop_reasons: dict[str, str],
) -> list[dict[str, object]]:
    ranked = sorted(all_scored, key=lambda item: item.support_score, reverse=True)
    rows: list[dict[str, object]] = []
    for rank, item in enumerate(ranked, start=1):
        metadata = item.candidate.metadata or {}
        rerank_rank = metadata.get("rerank_rank")
        rows.append(
            {
                "support_rank": rank,
                "evidence_id": item.evidence_id,
                "chunk_id": item.candidate.chunk_id,
                "parent_chunk_id": _parent_chunk_id(item.candidate.chunk_id),
                "doc_id": item.candidate.doc_id,
                "section": item.candidate.section,
                "support_score": round(float(item.support_score), 6),
                "rerank_rank": int(rerank_rank) if isinstance(rerank_rank, (int, float)) else None,
                "rerank_score": round(float(item.candidate.rerank_score or 0.0), 6),
                "selected": item.evidence_id in selected_ids,
                "selected_before_protection": item.evidence_id in selected_before_ids,
                "protected_seed": item.evidence_id in protected_ids,
                "inserted_by_protection": item.evidence_id in inserted_ids,
                "drop_reason": drop_reasons.get(item.evidence_id, ""),
                "matched_child_chunk_ids": _matched_child_chunk_ids(metadata),
                "reasons": list(item.reasons),
            }
        )
    return rows


def _parent_chunk_id(chunk_id: object) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def _matched_child_chunk_ids(metadata: dict) -> list[str]:
    value = metadata.get("matched_child_chunk_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    value = metadata.get("matched_child_chunk_id")
    if value:
        return [str(value)]
    return []
