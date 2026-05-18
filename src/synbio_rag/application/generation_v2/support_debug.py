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
    return {
        "intent": intent,
        "candidate_count": len(candidates),
        "scored_count": len(all_scored),
        "eligible_count": len(scored),
        "below_min_score_count": len(below_min_score),
        "selected_before_protection_evidence_ids": [item.evidence_id for item in selected_before_protection],
        "selected_evidence_ids": [item.evidence_id for item in selected],
        "protected_seed_evidence_ids": protected_ids,
        "protected_seed_inserted_evidence_ids": [
            item.evidence_id for item in selected if item.evidence_id not in selected_before_ids and item.evidence_id in protected_ids
        ],
        "drop_reasons_by_evidence_id": drop_reasons,
    }
