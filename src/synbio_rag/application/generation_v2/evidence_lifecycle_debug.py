from __future__ import annotations

from collections import Counter
from typing import Any

from ...domain.schemas import Citation, RetrievedChunk
from .models import EvidenceCandidate, SupportItem

ALLOWED_DROP_REASONS = {
    "not_in_input",
    "duplicate_chunk_id",
    "topk_cutoff",
    "score_too_low",
    "doc_cap",
    "context_budget",
    "metadata_missing",
    "text_missing",
    "not_citation_eligible",
    "answer_mode_filtered",
    "plan_mode_filtered",
    "branch_coverage_filtered",
    "partial_mode_filtered",
    "citation_marker_not_used",
    "answer_text_not_supported_by_chunk",
    "quote_missing",
    "unsupported_evidence_type",
    "unknown",
    "parent_expansion_budget",
    "support_pack_size_limit",
    "selected_support_not_referenced",
    "citation_binder_input_missing",
    "citation_output_limit",
    "comparison_branch_missing",
    "doc_local_evidence_miss",
}


def normalize_drop_reason(reason: str | None) -> str:
    value = str(reason or "unknown")
    return value if value in ALLOWED_DROP_REASONS else "unknown"


def chunk_ids(chunks: list[Any]) -> list[str]:
    return [str(getattr(chunk, "chunk_id", "") or "") for chunk in chunks]


def doc_ids(chunks: list[Any]) -> list[str]:
    result: list[str] = []
    for chunk in chunks:
        doc_id = getattr(chunk, "doc_id", "") or ""
        if not doc_id:
            candidate = getattr(chunk, "candidate", None)
            if candidate is not None:
                doc_id = getattr(candidate, "doc_id", "") or ""
        result.append(str(doc_id))
    return result


def protected_seed_chunk_ids(chunks: list[Any], protect_top_k: int | None = None) -> list[str]:
    ids: list[str] = []
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        rank = metadata.get("rerank_rank")
        if isinstance(rank, (int, float)) and (protect_top_k is None or int(rank) <= protect_top_k):
            ids.append(str(getattr(chunk, "chunk_id", "") or ""))
    return ids


def stage_debug_from_chunks(
    *,
    input_chunks: list[RetrievedChunk],
    output_chunks: list[RetrievedChunk],
    protected_ids: set[str] | None = None,
    default_drop_reason: str = "unknown",
) -> dict[str, Any]:
    protected_ids = protected_ids or set()
    input_ids = chunk_ids(input_chunks)
    output_ids = chunk_ids(output_chunks)
    output_set = set(output_ids)
    dropped = [chunk_id for chunk_id in input_ids if chunk_id not in output_set]
    drop_reasons = {chunk_id: normalize_drop_reason(default_drop_reason) for chunk_id in dropped}
    protected_kept = [chunk_id for chunk_id in protected_ids if chunk_id in output_set]
    protected_dropped = [chunk_id for chunk_id in protected_ids if chunk_id not in output_set]
    for chunk_id in protected_dropped:
        drop_reasons.setdefault(chunk_id, normalize_drop_reason(default_drop_reason))
    return {
        "input_count": len(input_chunks),
        "output_count": len(output_chunks),
        "kept_chunk_ids": output_ids,
        "doc_ids": doc_ids(output_chunks),
        "dropped_chunk_ids": dropped,
        "drop_reasons": drop_reasons,
        "protected_seed_kept_count": len(protected_kept),
        "protected_seed_dropped_count": len(protected_dropped),
        "protected_seed_kept_chunk_ids": sorted(protected_kept),
        "protected_seed_dropped_chunk_ids": sorted(protected_dropped),
    }


def rerank_output_debug(
    reranked: list[RetrievedChunk],
    *,
    protected_ids: set[str],
) -> dict[str, Any]:
    return {
        "input_count": len(reranked),
        "output_count": len(reranked),
        "chunk_ids": chunk_ids(reranked),
        "doc_ids": doc_ids(reranked),
        "protected_seed_count": len(protected_ids),
        "protected_seed_chunk_ids": sorted(protected_ids),
    }


def support_input_debug(candidates: list[EvidenceCandidate]) -> dict[str, Any]:
    protected = protected_seed_chunk_ids(candidates)
    return {
        "input_count": len(candidates),
        "chunk_ids": chunk_ids(candidates),
        "doc_ids": doc_ids(candidates),
        "protected_seed_count": len(protected),
        "protected_seed_chunk_ids": protected,
    }


def selected_support_debug(
    *,
    candidates: list[EvidenceCandidate],
    support_pack: list[SupportItem],
    selector_debug: dict[str, Any],
    answer_mode: str,
    plan_mode: str,
    support_pack_size: int,
) -> dict[str, Any]:
    candidate_by_eid = {candidate.evidence_id: candidate for candidate in candidates}
    selected_ids = {item.evidence_id for item in support_pack}
    selected_chunk_ids = [item.candidate.chunk_id for item in support_pack]
    dropped_chunk_ids: list[str] = []
    drop_reasons: dict[str, str] = {}
    selector_reasons = selector_debug.get("drop_reasons_by_evidence_id", {}) if selector_debug else {}
    for candidate in candidates:
        if candidate.evidence_id in selected_ids:
            continue
        dropped_chunk_ids.append(candidate.chunk_id)
        drop_reasons[candidate.chunk_id] = normalize_drop_reason(
            selector_reasons.get(candidate.evidence_id, "support_pack_size_limit")
        )
    protected_ids = set(protected_seed_chunk_ids(candidates))
    protected_kept = [chunk_id for chunk_id in protected_ids if chunk_id in selected_chunk_ids]
    protected_dropped = [chunk_id for chunk_id in protected_ids if chunk_id not in selected_chunk_ids]
    for chunk_id in protected_dropped:
        drop_reasons.setdefault(chunk_id, "support_pack_size_limit")
    return {
        "input_count": len(candidates),
        "output_count": len(support_pack),
        "kept_chunk_ids": selected_chunk_ids,
        "doc_ids": doc_ids(support_pack),
        "dropped_chunk_ids": dropped_chunk_ids,
        "drop_reasons": drop_reasons,
        "protected_seed_kept_count": len(protected_kept),
        "protected_seed_dropped_count": len(protected_dropped),
        "protected_seed_kept_chunk_ids": sorted(protected_kept),
        "protected_seed_dropped_chunk_ids": sorted(protected_dropped),
        "support_pack_size": support_pack_size,
        "answer_mode": answer_mode,
        "plan_mode": plan_mode,
        "selector_debug": selector_debug,
        "dropped_evidence_ids": [
            evidence_id for evidence_id, candidate in candidate_by_eid.items() if candidate.chunk_id in dropped_chunk_ids
        ],
    }


def citation_candidates_debug(
    support_pack: list[SupportItem],
    citation_candidates: list[Any] | None = None,
) -> dict[str, Any]:
    if citation_candidates is not None:
        protected = [
            c.chunk_id for c in citation_candidates
            if getattr(c, "is_protected_seed", False)
        ]
        drop_reasons: dict[str, str] = {}
        for c in citation_candidates:
            if getattr(c, "drop_reason", ""):
                drop_reasons[c.chunk_id] = c.drop_reason
        return {
            "input_count": len(support_pack),
            "output_count": len(citation_candidates),
            "chunk_ids": [c.chunk_id for c in citation_candidates],
            "doc_ids": [c.doc_id for c in citation_candidates],
            "protected_seed_count": len(protected),
            "protected_seed_chunk_ids": protected,
            "drop_reasons": drop_reasons,
            "citation_eligible_count": sum(1 for c in citation_candidates if getattr(c, "citation_eligible", False)),
        }
    protected = [
        item.candidate.chunk_id
        for item in support_pack
        if isinstance(item.candidate.metadata.get("rerank_rank"), (int, float))
    ]
    return {
        "input_count": len(support_pack),
        "chunk_ids": [item.candidate.chunk_id for item in support_pack],
        "doc_ids": [item.candidate.doc_id for item in support_pack],
        "protected_seed_count": len(protected),
        "protected_seed_chunk_ids": protected,
    }


def citation_output_debug(
    *,
    support_pack: list[SupportItem],
    citations: list[Citation],
    citation_debug: dict[str, Any],
    plan_mode: str,
) -> dict[str, Any]:
    cited_chunk_ids = [citation.chunk_id for citation in citations]
    cited_doc_ids = [citation.doc_id for citation in citations]
    cited_set = set(cited_chunk_ids)
    uncited = [item.candidate.chunk_id for item in support_pack if item.candidate.chunk_id not in cited_set]
    ordered_evidence_ids = set(citation_debug.get("ordered_evidence_ids", []))
    drop_reasons: dict[str, str] = {}
    partial_mode_uncited: list[str] = []
    for item in support_pack:
        if item.candidate.chunk_id in cited_set:
            continue
        # Determine the specific reason — never overridden by plan_mode alone
        if item.evidence_id not in ordered_evidence_ids:
            reason = "citation_marker_not_used"
        else:
            reason = "selected_support_not_referenced"
        drop_reasons[item.candidate.chunk_id] = reason
        # partial_mode marks which uncited items sit in a partial-mode plan;
        # it is a context label, not a replacement for the specific reason.
        if plan_mode == "partial":
            partial_mode_uncited.append(item.candidate.chunk_id)
    return {
        "output_count": len(citations),
        "cited_chunk_ids": cited_chunk_ids,
        "cited_doc_ids": cited_doc_ids,
        "uncited_selected_support_chunk_ids": uncited,
        "drop_reasons": drop_reasons,
        "partial_mode_uncited_chunk_ids": partial_mode_uncited,
        "partial_mode": plan_mode == "partial",
        "ordered_evidence_ids": list(citation_debug.get("ordered_evidence_ids", [])),
        "invalid_evidence_ids": list(citation_debug.get("invalid_evidence_ids", [])),
    }


def drop_reason_distribution(debug: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for stage in ("final_chunks", "selected_support", "citation_output"):
        for reason in (debug.get(stage, {}).get("drop_reasons", {}) or {}).values():
            counter[normalize_drop_reason(reason)] += 1
    return dict(counter)
