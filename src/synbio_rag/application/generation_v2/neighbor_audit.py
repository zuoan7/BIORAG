from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ...domain.config import GenerationConfig
from ...domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from .models import AnswerPlan, EvidenceCandidate

_LOW_QUALITY_SECTIONS = frozenset({
    "references", "bibliography", "acknowledgements", "acknowledgement",
    "acknowledgments", "acknowledgment", "author information", "authors",
    "author contributions", "conflict of interest", "conflicts of interest",
    "funding", "supporting information", "supplementary", "appendix",
    "contents", "table of contents", "title",
})

# Default context_only unless strong semantic signal present
_WEAK_SECTIONS = frozenset({
    "materials and methods", "methods", "introduction",
    "experimental", "experimental section",
})

_HIGH_VALUE_SECTIONS = frozenset({
    "results", "results and discussion", "discussion",
    "abstract", "conclusion", "conclusions",
    "results & discussion", "key findings",
})

_NO_SUPPORT_REFUSE_REASONS = frozenset({
    "no_support_pack", "existence_no_support", "no_support",
    "empty_support", "summary_no_support",
})

_RESULT_RE = re.compile(
    r"\b(result|results|showed|demonstrated|increased|decreased|yield|titer|production"
    r"|fold|efficiency|improvement|activity|expression|secretion)\b"
    r"|结果|显示|提高|降低|产量|滴度|表达|分泌|效率|改善|增加|减少",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")


def _get_seed_score(
    ev: EvidenceCandidate | None,
    seed_chunk: RetrievedChunk,
) -> tuple[float, str]:
    """Return (score, source_field_name) using priority: rerank_score > fusion_score > vector_score > bm25_score."""
    # Priority: rerank_score first (most reliable post-rerank signal)
    for attr in ("rerank_score", "fusion_score", "vector_score", "bm25_score"):
        # check EvidenceCandidate first
        if ev is not None:
            val = getattr(ev, attr, None)
            if val is not None and val > 0.0:
                return float(val), attr
        # fallback to seed_chunk
        val = getattr(seed_chunk, attr, None)
        if val is not None and val > 0.0:
            return float(val), attr
    return 0.0, "missing"


@dataclass
class NeighborCandidate:
    evidence_id: str
    chunk_id: str
    doc_id: str
    section: str
    text_preview: str
    source_seed_chunk_id: str
    source_seed_evidence_id: str | None
    distance: int
    abs_distance: int
    direction: str
    source_seed_score: float | None
    source_seed_score_source: str
    score_raw: float
    score_after_decay: float
    neighbor_score: float
    inherited_seed_score: bool
    citable: bool
    promotion_status: str
    promotion_reasons: list[str] = field(default_factory=list)
    exclusion_reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    linked_seed_chunk_ids: list[str] = field(default_factory=list)
    score_floor: float = 0.05
    score_passed: bool = False
    semantic_gate_passed: bool = False
    section_gate_passed: bool = False
    blocked_by_refusal: bool = False
    promotion_decision_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "source_seed_chunk_id": self.source_seed_chunk_id,
            "source_seed_evidence_id": self.source_seed_evidence_id,
            "distance": self.distance,
            "abs_distance": self.abs_distance,
            "direction": self.direction,
            "source_seed_score": self.source_seed_score,
            "source_seed_score_source": self.source_seed_score_source,
            "score_raw": round(self.score_raw, 6),
            "score_after_decay": round(self.score_after_decay, 6),
            "neighbor_score": round(self.neighbor_score, 6),
            "inherited_seed_score": self.inherited_seed_score,
            "citable": self.citable,
            "promotion_status": self.promotion_status,
            "promotion_reasons": list(self.promotion_reasons),
            "exclusion_reasons": list(self.exclusion_reasons),
            "features": dict(self.features),
            "text_preview": self.text_preview,
            "linked_seed_chunk_ids": list(self.linked_seed_chunk_ids),
            "score_floor": self.score_floor,
            "score_passed": self.score_passed,
            "semantic_gate_passed": self.semantic_gate_passed,
            "section_gate_passed": self.section_gate_passed,
            "blocked_by_refusal": self.blocked_by_refusal,
            "promotion_decision_reason": self.promotion_decision_reason,
        }


@dataclass
class NeighborAuditResult:
    enabled: bool
    window: int
    promotion_enabled: bool
    promotion_dry_run: bool
    candidate_count: int
    dry_run_promoted_count: int
    excluded_count: int
    candidates: list[NeighborCandidate] = field(default_factory=list)
    by_seed: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "window": self.window,
            "promotion_enabled": self.promotion_enabled,
            "promotion_dry_run": self.promotion_dry_run,
            "candidate_count": self.candidate_count,
            "dry_run_promoted_count": self.dry_run_promoted_count,
            "excluded_count": self.excluded_count,
            "by_seed": self.by_seed,
            "candidates": [c.to_dict() for c in self.candidates],
            "summary": self.summary,
        }


class NeighborAuditEngine:
    """
    Selective neighbor audit for generation_v2.
    Dry-run only: never mutates support_pack, citations, or Qwen input.
    """

    def __init__(
        self,
        chunk_index: dict[str, RetrievedChunk],
        position_index: dict[str, tuple[str, int]],
        doc_chunks: dict[str, list[RetrievedChunk]],
    ) -> None:
        self._by_id = chunk_index
        self._positions = position_index
        self._doc_chunks = doc_chunks

    def run(
        self,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
        candidates: list[EvidenceCandidate],
        config: GenerationConfig,
        plan: AnswerPlan | None = None,
    ) -> NeighborAuditResult:
        if not config.v2_enable_neighbor_audit:
            return NeighborAuditResult(
                enabled=False,
                window=config.v2_neighbor_window,
                promotion_enabled=config.v2_enable_neighbor_promotion,
                promotion_dry_run=config.v2_neighbor_promotion_dry_run,
                candidate_count=0,
                dry_run_promoted_count=0,
                excluded_count=0,
            )

        # Determine if this is a no-support/refuse case
        refusal_blocked = _is_refusal_blocked(plan)

        seed_evidence_map: dict[str, EvidenceCandidate] = {c.chunk_id: c for c in candidates}
        seed_chunk_ids: set[str] = {chunk.chunk_id for chunk in seed_chunks}

        neighbor_links: dict[str, list[tuple[RetrievedChunk, int]]] = defaultdict(list)
        for seed in seed_chunks:
            pos = self._positions.get(seed.chunk_id)
            if not pos:
                continue
            doc_id, idx = pos
            doc_list = self._doc_chunks.get(doc_id) or []
            for dist in range(1, config.v2_neighbor_window + 1):
                for signed, nb_idx in ((-dist, idx - dist), (dist, idx + dist)):
                    if 0 <= nb_idx < len(doc_list):
                        nb_chunk = doc_list[nb_idx]
                        if nb_chunk.chunk_id not in seed_chunk_ids:
                            neighbor_links[nb_chunk.chunk_id].append((seed, signed))

        all_candidates: list[NeighborCandidate] = []
        for nb_chunk_id, links in neighbor_links.items():
            nb_chunk = self._by_id.get(nb_chunk_id)
            if nb_chunk is None:
                continue
            links_sorted = sorted(links, key=lambda t: abs(t[1]))
            primary_seed, primary_dist = links_sorted[0]
            primary_ev = seed_evidence_map.get(primary_seed.chunk_id)

            seed_score, score_source = _get_seed_score(primary_ev, primary_seed)
            abs_dist = abs(primary_dist)
            decay = (
                config.v2_neighbor_score_decay_distance1
                if abs_dist == 1
                else config.v2_neighbor_score_decay_distance2
            )
            score_after_decay = seed_score * decay
            feature_bonus = 0.0
            features = _extract_features(nb_chunk, question, analysis)
            if score_source != "missing":
                feature_bonus = _feature_bonus(features)
            neighbor_score = min(
                score_after_decay + feature_bonus * seed_score * 0.1,
                score_after_decay * 1.3,
            )

            score_floor = config.v2_neighbor_min_promotion_score
            score_passed = neighbor_score >= score_floor and score_source != "missing"

            promotion_reasons: list[str] = []
            exclusion_reasons: list[str] = []
            status, semantic_gate, section_gate, decision_reason = _classify_promotion(
                nb_chunk=nb_chunk,
                features=features,
                score_passed=score_passed,
                score_source=score_source,
                refusal_blocked=refusal_blocked,
                exclusion_reasons=exclusion_reasons,
                promotion_reasons=promotion_reasons,
            )
            if refusal_blocked and status == "dry_run_promoted":
                status = "context_only"
                exclusion_reasons.append("blocked_by_no_support_refusal")
                decision_reason = "blocked_by_refusal"

            nc = NeighborCandidate(
                evidence_id=f"NB_{nb_chunk_id}",
                chunk_id=nb_chunk_id,
                doc_id=nb_chunk.doc_id,
                section=nb_chunk.section or "",
                text_preview=(nb_chunk.text or "")[:200],
                source_seed_chunk_id=primary_seed.chunk_id,
                source_seed_evidence_id=primary_ev.evidence_id if primary_ev else None,
                distance=primary_dist,
                abs_distance=abs_dist,
                direction="prev" if primary_dist < 0 else "next",
                source_seed_score=round(seed_score, 6) if score_source != "missing" else None,
                source_seed_score_source=score_source,
                score_raw=round(seed_score, 6),
                score_after_decay=round(score_after_decay, 6),
                neighbor_score=round(neighbor_score, 6),
                inherited_seed_score=False,
                citable=False,
                promotion_status=status,
                promotion_reasons=promotion_reasons,
                exclusion_reasons=exclusion_reasons,
                features=features,
                linked_seed_chunk_ids=[s.chunk_id for s, _ in links],
                score_floor=score_floor,
                score_passed=score_passed,
                semantic_gate_passed=semantic_gate,
                section_gate_passed=section_gate,
                blocked_by_refusal=refusal_blocked,
                promotion_decision_reason=decision_reason,
            )
            all_candidates.append(nc)

        seed_counts: dict[str, int] = defaultdict(int)
        final_candidates: list[NeighborCandidate] = []
        for nc in sorted(all_candidates, key=lambda c: -c.neighbor_score):
            if seed_counts[nc.source_seed_chunk_id] < config.v2_max_neighbors_per_seed:
                final_candidates.append(nc)
                seed_counts[nc.source_seed_chunk_id] += 1

        dry_run_promoted = [c for c in final_candidates if c.promotion_status == "dry_run_promoted"]
        excluded = [c for c in final_candidates if c.promotion_status == "excluded"]

        by_seed = _build_by_seed(final_candidates, seed_chunks)
        summary = _build_summary(final_candidates, analysis, refusal_blocked, config.v2_neighbor_min_promotion_score)

        return NeighborAuditResult(
            enabled=True,
            window=config.v2_neighbor_window,
            promotion_enabled=config.v2_enable_neighbor_promotion,
            promotion_dry_run=config.v2_neighbor_promotion_dry_run,
            candidate_count=len(final_candidates),
            dry_run_promoted_count=len(dry_run_promoted),
            excluded_count=len(excluded),
            candidates=final_candidates,
            by_seed=by_seed,
            summary=summary,
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_refusal_blocked(plan: AnswerPlan | None) -> bool:
    if plan is None:
        return False
    if plan.mode == "refuse":
        return True
    if plan.reason in _NO_SUPPORT_REFUSE_REASONS:
        return True
    return False


def _extract_features(chunk: RetrievedChunk, question: str, analysis: QueryAnalysis) -> dict[str, Any]:
    text = chunk.text or ""
    lower_text = text.lower()
    lower_section = (chunk.section or "").strip().lower()
    q_lower = question.lower()

    has_numeric = bool(_DIGIT_RE.search(text))
    has_result_terms = bool(_RESULT_RE.search(text))
    has_table = "[table text]" in lower_text or "table_text" in lower_text
    has_table_caption = "[table caption]" in lower_text or "table caption" in lower_text
    has_figure = "[figure caption]" in lower_text or "fig." in lower_text

    q_tokens = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", q_lower))
    text_tokens = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", lower_text))
    query_overlap_count = len(q_tokens & text_tokens)
    query_overlap = query_overlap_count >= 2

    is_high_value_section = lower_section in _HIGH_VALUE_SECTIONS
    is_low_quality_section = lower_section in _LOW_QUALITY_SECTIONS
    is_weak_section = lower_section in _WEAK_SECTIONS

    summary_overlap = False
    if analysis.intent == QueryIntent.SUMMARY:
        summary_overlap = has_result_terms or query_overlap

    comparison_branch_overlap = False
    if analysis.intent == QueryIntent.COMPARISON:
        comparison_branch_overlap = query_overlap_count >= 3

    return {
        "has_numeric": has_numeric,
        "has_result_terms": has_result_terms,
        "has_table": has_table,
        "has_table_caption": has_table_caption,
        "has_figure": has_figure,
        "query_overlap": query_overlap,
        "query_overlap_count": query_overlap_count,
        "summary_overlap": summary_overlap,
        "comparison_branch_overlap": comparison_branch_overlap,
        "is_high_value_section": is_high_value_section,
        "is_low_quality_section": is_low_quality_section,
        "is_weak_section": is_weak_section,
        "section_lower": lower_section,
        "text_length": len(text),
    }


def _feature_bonus(features: dict[str, Any]) -> float:
    bonus = 0.0
    if features.get("has_numeric"):
        bonus += 0.3
    if features.get("has_result_terms"):
        bonus += 0.3
    if features.get("has_table") or features.get("has_table_caption"):
        bonus += 0.2
    if features.get("has_figure"):
        bonus += 0.1
    if features.get("is_high_value_section"):
        bonus += 0.2
    if features.get("query_overlap"):
        bonus += 0.2
    return min(bonus, 1.3)


def _classify_promotion(
    nb_chunk: RetrievedChunk,
    features: dict[str, Any],
    score_passed: bool,
    score_source: str,
    refusal_blocked: bool,
    exclusion_reasons: list[str],
    promotion_reasons: list[str],
) -> tuple[str, bool, bool, str]:
    """Return (status, semantic_gate_passed, section_gate_passed, decision_reason)."""

    # hard exclusion: low quality section
    if features.get("is_low_quality_section"):
        exclusion_reasons.append("low_quality_section")
        return "excluded", False, False, "low_quality_section"

    # hard exclusion: too short
    if features.get("text_length", 0) < 50:
        exclusion_reasons.append("text_too_short")
        return "excluded", False, False, "text_too_short"

    # hard exclusion: missing score
    if score_source == "missing":
        exclusion_reasons.append("missing_score")
        return "excluded", False, False, "missing_score"

    # section gate
    section_gate_passed = not features.get("is_weak_section", False)

    # semantic gate: at least one STRONG signal required
    semantic_gate_passed = False
    if features.get("query_overlap"):
        promotion_reasons.append("query_overlap")
        semantic_gate_passed = True
    if features.get("summary_overlap") and features.get("query_overlap"):
        # summary_overlap alone (without query_overlap) is not sufficient
        if "summary_overlap" not in promotion_reasons:
            promotion_reasons.append("summary_overlap")
    if features.get("comparison_branch_overlap"):
        promotion_reasons.append("branch_overlap")
        semantic_gate_passed = True

    # rich content is a bonus but only when semantic gate also passes
    if features.get("has_table") or features.get("has_table_caption") or features.get("has_figure"):
        if semantic_gate_passed:
            promotion_reasons.append("rich_content")

    # result/numeric: bonus only, NOT sufficient alone
    if (features.get("has_result_terms") or features.get("has_numeric")) and semantic_gate_passed:
        promotion_reasons.append("result_or_numeric")

    # high value section: bonus only, NOT sufficient alone
    if features.get("is_high_value_section") and semantic_gate_passed:
        promotion_reasons.append("high_value_section")

    # for weak sections: require STRONG overlap (query_overlap_count >= 3)
    if features.get("is_weak_section"):
        strong_overlap = features.get("query_overlap_count", 0) >= 3
        if not strong_overlap:
            # demote to context_only regardless of other signals
            exclusion_reasons.clear()
            promotion_reasons.clear()
            return "context_only", False, False, "weak_section_insufficient_overlap"

    if not score_passed:
        exclusion_reasons.append("score_below_floor")
        return "context_only", semantic_gate_passed, section_gate_passed, "score_below_floor"

    if not semantic_gate_passed:
        exclusion_reasons.append("no_semantic_signal")
        return "context_only", False, section_gate_passed, "no_semantic_signal"

    return "dry_run_promoted", True, section_gate_passed, "semantic_and_score_passed"


def _build_by_seed(candidates: list[NeighborCandidate], seed_chunks: list[RetrievedChunk]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in seed_chunks:
        related = [c for c in candidates if seed.chunk_id in c.linked_seed_chunk_ids]
        result[seed.chunk_id] = {
            "seed_doc_id": seed.doc_id,
            "seed_section": seed.section or "",
            "neighbor_ids": [c.chunk_id for c in related],
            "dry_run_promoted_ids": [c.chunk_id for c in related if c.promotion_status == "dry_run_promoted"],
        }
    return result


def _build_summary(
    candidates: list[NeighborCandidate],
    analysis: QueryAnalysis,
    refusal_blocked: bool,
    score_floor: float,
) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    by_section: dict[str, int] = defaultdict(int)
    score_sources: dict[str, int] = defaultdict(int)
    promoted_by_reason: dict[str, int] = defaultdict(int)
    excluded_by_reason: dict[str, int] = defaultdict(int)
    context_only_by_reason: dict[str, int] = defaultdict(int)

    for c in candidates:
        by_status[c.promotion_status] += 1
        by_section[c.section or "unknown"] += 1
        score_sources[c.source_seed_score_source] += 1
        if c.promotion_status == "dry_run_promoted":
            for r in c.promotion_reasons:
                promoted_by_reason[r] += 1
        elif c.promotion_status == "excluded":
            for r in c.exclusion_reasons:
                excluded_by_reason[r] += 1
        else:  # context_only
            for r in c.exclusion_reasons:
                context_only_by_reason[r] += 1

    no_support_blocked_count = sum(1 for c in candidates if c.blocked_by_refusal)
    false_positive_risk_count = sum(
        1 for c in candidates
        if c.promotion_status == "dry_run_promoted"
        and c.source_seed_score_source == "missing"
    )

    potential_summary_boost = sum(
        1 for c in candidates
        if c.promotion_status == "dry_run_promoted" and c.features.get("summary_overlap")
    )
    potential_comparison_boost = sum(
        1 for c in candidates
        if c.promotion_status == "dry_run_promoted" and c.features.get("comparison_branch_overlap")
    )

    return {
        "by_status": dict(by_status),
        "by_section": dict(by_section),
        "low_quality_excluded": sum(1 for c in candidates if "low_quality_section" in c.exclusion_reasons),
        "potential_summary_boost_count": potential_summary_boost,
        "potential_comparison_boost_count": potential_comparison_boost,
        "score_source_distribution": dict(score_sources),
        "score_floor": score_floor,
        "promoted_by_reason": dict(promoted_by_reason),
        "excluded_by_reason": dict(excluded_by_reason),
        "context_only_by_reason": dict(context_only_by_reason),
        "no_support_blocked_count": no_support_blocked_count,
        "false_positive_risk_count": false_positive_risk_count,
    }
