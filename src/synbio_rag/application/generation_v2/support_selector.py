from __future__ import annotations

import re
from collections import Counter

from ...domain.config import GenerationConfig
from ...domain.schemas import QueryAnalysis, QueryIntent
from .branch_parser import parse_comparison_branches
from .comparison_coverage import score_branch_support
from .models import EvidenceCandidate, SupportItem
from .summary_support_selector import (
    _all_high_quality_same_doc,
    _evaluate_summary_quality,
    _find_summary_duplicate_reason,
    _is_explicit_single_doc_summary,
    _section_priority,
    _select_summary,
    _should_defer_for_doc_diversity,
    _summary_rank_key,
    _summary_section_bucket,
    _token_overlap_ratio,
)
from .support_debug import _build_selection_debug
from .support_retention import (
    _apply_close_margin_capacity_plus_one,
    _ensure_protected_support_seeds,
    _retain_doc_diversity,
    _retain_matched_child_support_candidates,
)

_EN_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'_.-]*", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{1,4}")
_TABLE_LABEL_PATTERN = re.compile(r"(table\s*\d+|表\s*\d+|table\b|表\b)", re.IGNORECASE)
_FIGURE_LABEL_PATTERN = re.compile(r"(figure\s*\d+|fig\.\s*\d+|fig\s*\d+|图\s*\d+|figure\b|fig\.\b|图\b)", re.IGNORECASE)

class SupportPackSelector:
    def __init__(self) -> None:
        self.last_summary_selection_debug: dict[str, object] = {"is_summary": False}
        self.last_selection_debug: dict[str, object] = {}

    def select(
        self,
        question: str,
        analysis: QueryAnalysis,
        candidates: list[EvidenceCandidate],
        config: GenerationConfig,
    ) -> list[SupportItem]:
        self.last_summary_selection_debug = {"is_summary": False}
        self.last_selection_debug = {}

        # Phase 21A-9I: negative/no-answer guard — skip support selection entirely
        if analysis.intent == QueryIntent.NEGATIVE:
            self.last_selection_debug = {
                "intent": "negative",
                "candidate_count": len(candidates),
                "eligible_count": 0,
                "selected_evidence_ids": [],
                "negative_guard": True,
                "drop_reasons_by_evidence_id": {},
            }
            return []

        all_scored = [self._to_support_item(question, candidate) for candidate in candidates]
        below_min_score = [
            item for item in all_scored if item.support_score < config.v2_min_support_score
        ]
        scored = [item for item in all_scored if item.support_score >= config.v2_min_support_score]

        intent = analysis.intent
        if intent in {QueryIntent.FACTOID, QueryIntent.UNKNOWN}:
            selected = self._select_factoid(question, scored, config)
        elif intent == QueryIntent.SUMMARY:
            selected, debug = _select_summary(question, scored, config, self._finalize)
            self.last_summary_selection_debug = debug
        elif intent == QueryIntent.COMPARISON:
            selected = self._select_comparison(question, scored, config)
        else:
            selected = self._select_factoid(question, scored, config)

        # Phase 15C: ensure protected rerank seeds are in selected support
        selected_before_protection = list(selected)
        if config.v2_protect_support_seeds_enabled:
            selected = _ensure_protected_support_seeds(scored, selected, config)
        # Phase 21A-9G: retain doc diversity after seed protection
        selected = _retain_doc_diversity(selected, scored, all_scored, config)
        # Phase 21A-9M: close-margin distinct-doc capacity+1
        selected = _apply_close_margin_capacity_plus_one(selected, scored, config, analysis.intent)
        selected = _retain_matched_child_support_candidates(selected, scored, config, analysis.intent)
        self.last_selection_debug = _build_selection_debug(
            candidates=candidates,
            all_scored=all_scored,
            scored=scored,
            below_min_score=below_min_score,
            selected_before_protection=selected_before_protection,
            selected=selected,
            config=config,
            intent=analysis.intent.value,
        )
        return selected

    def _to_support_item(self, question: str, candidate: EvidenceCandidate) -> SupportItem:
        reasons = list(candidate.reasons)
        score = self._base_score(candidate)
        section_type = str(candidate.features.get("section_type", ""))
        if "result" in section_type:
            score += 0.25
            reasons.append("section_bonus:results")
        elif "discussion" in section_type:
            score += 0.18
            reasons.append("section_bonus:discussion")
        elif "abstract" in section_type:
            score += 0.08
            reasons.append("section_bonus:abstract")
        elif "reference" in section_type or "bibliograph" in section_type:
            score -= 0.30
            reasons.append("section_penalty:references")
        if _is_bibliography_like(candidate.text):
            score -= 0.25
            reasons.append("section_penalty:bibliography_like")
        for feature_name, bonus in (
            ("has_numeric", 0.08),
            ("has_result_terms", 0.10),
            ("has_table_text", 0.18),
            ("has_table_caption", 0.14),
            ("has_figure_caption", 0.14),
        ):
            if candidate.features.get(feature_name):
                score += bonus
                reasons.append(f"feature_bonus:{feature_name}")
        overlap = _query_overlap(question, candidate.text)
        if overlap > 0:
            score += min(overlap * 0.3, 0.3)
            reasons.append(f"query_overlap:{overlap:.2f}")
            reasons.append("query_overlap")
        if _question_mentions_table(question) and (
            candidate.features.get("has_table_text") or candidate.features.get("has_table_caption")
        ):
            score += 0.30
            reasons.append("question_targets_table")
        if _question_mentions_figure(question) and candidate.features.get("has_figure_caption"):
            score += 0.30
            reasons.append("question_targets_figure")
        return SupportItem(evidence_id=candidate.evidence_id, candidate=candidate, support_score=score, reasons=reasons)

    def _select_factoid(
        self,
        question: str,
        scored: list[SupportItem],
        config: GenerationConfig,
    ) -> list[SupportItem]:
        ranked = sorted(scored, key=lambda item: item.support_score, reverse=True)
        if _question_mentions_table(question):
            ranked = sorted(
                ranked,
                key=lambda item: (
                    not (
                        item.candidate.features.get("has_table_text")
                        or item.candidate.features.get("has_table_caption")
                    ),
                    -item.support_score,
                ),
            )
        elif _question_mentions_figure(question):
            ranked = sorted(
                ranked,
                key=lambda item: (
                    not item.candidate.features.get("has_figure_caption"),
                    -item.support_score,
                ),
            )
        selected, per_doc = _select_with_doc_diversity(
            ranked=ranked,
            max_total=config.v2_max_support_factoid,
            max_per_doc=2,
            route_name="factoid_top_score",
            finalizer=self._finalize,
        )
        return selected

    def _select_comparison(self, question: str, scored: list[SupportItem], config: GenerationConfig) -> list[SupportItem]:
        parse_result = parse_comparison_branches(question)
        branches = parse_result.branches if parse_result.parse_ok else []
        ranked = sorted(scored, key=lambda item: item.support_score, reverse=True)
        selected: list[SupportItem] = []
        seen_ids: set[str] = set()

        if branches and config.v2_enable_comparison_coverage:
            for branch in branches:
                best_item: SupportItem | None = None
                best_status = "missing"
                best_confidence = -1.0
                for item in ranked:
                    if item.evidence_id in seen_ids:
                        continue
                    assessment = score_branch_support(branch, item)
                    if assessment.status == "missing":
                        continue
                    if (
                        _status_rank(assessment.status) > _status_rank(best_status)
                        or (
                            assessment.status == best_status
                            and assessment.confidence > best_confidence
                        )
                    ):
                        best_item = item
                        best_status = assessment.status
                        best_confidence = assessment.confidence
                if best_item is None:
                    continue
                best_item.reasons.append(f"comparison_branch:{branch}")
                best_item.reasons.append(f"comparison_branch_status:{best_status}")
                selected.append(best_item)
                seen_ids.add(best_item.evidence_id)
        elif branches:
            for branch in branches:
                branch_match = next(
                    (
                        item
                        for item in ranked
                        if item.evidence_id not in seen_ids and _branch_matches(branch, item.candidate)
                    ),
                    None,
                )
                if branch_match is None:
                    continue
                branch_match.reasons.append(f"comparison_branch:{branch}")
                selected.append(branch_match)
                seen_ids.add(branch_match.evidence_id)
        ranked_diverse = sorted(
            ranked,
            key=lambda item: (
                _doc_seen_rank(selected, item.candidate.doc_id),
                -item.support_score,
            ),
        )
        for item in ranked_diverse:
            if len(selected) >= config.v2_max_support_comparison:
                break
            if item.evidence_id in seen_ids:
                continue
            if not branches and len({s.candidate.doc_id for s in selected}) == 0:
                item.reasons.append("comparison_top_support")
            elif not branches:
                item.reasons.append("comparison_doc_diversity")
                item.reasons.append(f"comparison_parse:{parse_result.reason}")
            selected.append(item)
            seen_ids.add(item.evidence_id)
        return self._finalize(selected, "comparison_selection")

    def _finalize(self, items: list[SupportItem], rule: str) -> list[SupportItem]:
        finalized: list[SupportItem] = []
        for item in items:
            finalized.append(
                SupportItem(
                    evidence_id=item.evidence_id,
                    candidate=item.candidate,
                    support_score=item.support_score,
                    reasons=list(dict.fromkeys([rule, *item.reasons])),
                )
            )
        return finalized

    def _base_score(self, candidate: EvidenceCandidate) -> float:
        if candidate.rerank_score:
            return candidate.rerank_score
        if candidate.fusion_score:
            return candidate.fusion_score
        return max(candidate.vector_score, candidate.bm25_score)


def _is_results_or_discussion(section: str) -> bool:
    lowered = (section or "").lower()
    return "result" in lowered or "discussion" in lowered


def _is_abstract(section: str) -> bool:
    return "abstract" in (section or "").lower()


def _question_mentions_table(question: str) -> bool:
    return bool(_TABLE_LABEL_PATTERN.search(question))


def _question_mentions_figure(question: str) -> bool:
    return bool(_FIGURE_LABEL_PATTERN.search(question))


def _query_overlap(question: str, text: str) -> float:
    question_tokens = set(_tokenize(question))
    text_tokens = set(_tokenize(text))
    if not question_tokens or not text_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / len(question_tokens)


def _tokenize(text: str) -> list[str]:
    english = [token.lower() for token in _EN_TOKEN_PATTERN.findall(text)]
    cjk = _CJK_PATTERN.findall(text)
    return english + cjk


def _branch_matches(branch: str, candidate: EvidenceCandidate) -> bool:
    branch_lower = branch.lower()
    haystack = " ".join(
        [
            candidate.title.lower(),
            candidate.section.lower(),
            candidate.text.lower(),
            " ".join(str(value).lower() for value in candidate.metadata.values()),
        ]
    )
    if re.fullmatch(r"[a-z0-9_.-]{1,4}", branch_lower):
        return bool(re.search(rf"\b{re.escape(branch_lower)}\b", haystack))
    return branch_lower in haystack


def _doc_seen_rank(selected: list[SupportItem], doc_id: str) -> int:
    return sum(1 for item in selected if item.candidate.doc_id == doc_id)


def _status_rank(status: str) -> int:
    if status == "direct":
        return 2
    if status == "indirect":
        return 1
    return 0


def _is_bibliography_like(text: str) -> bool:
    """Detect bibliography/reference-list chunks (DOI URLs, citation lists, author lists)."""
    if not text:
        return False
    lowered = text.lower()
    doi_count = len(re.findall(r"https?://doi\.org", lowered))
    if doi_count >= 2:
        return True
    # Many author-year patterns in sequence → likely bibliography
    et_al_patterns = len(re.findall(r"et\s+al\.?\s*,?\s*\d{4}", lowered))
    if et_al_patterns >= 3:
        return True
    # Long sequence of references like "[1]...[2]...[3]..."
    ref_tags = len(re.findall(r"\[\d+(?:,\s*\d+)*\]", lowered))
    if ref_tags >= 5:
        return True
    # Dense http URLs (reference link farms)
    if lowered.count("http") > 10:
        return True
    return False


def _select_with_doc_diversity(
    *,
    ranked: list[SupportItem],
    max_total: int,
    max_per_doc: int,
    route_name: str,
    finalizer,
) -> tuple[list[SupportItem], Counter]:
    """Select up to max_total items from ranked, with per-doc diversity cap.

    Fair selection: picks the top-scoring item first, then iterates through
    remaining items in score order, skipping items that would exceed the per-doc
    cap. Falls back to allowing same-doc overflow if fewer than max_total
    distinct docs are available.
    """
    from collections import Counter
    selected: list[SupportItem] = []
    per_doc: Counter[str] = Counter()
    distinct_docs = len({item.candidate.doc_id for item in ranked})

    # Phase 1: score-ordered selection with diversity
    for item in ranked:
        if len(selected) >= max_total:
            break
        if per_doc[item.candidate.doc_id] >= max_per_doc:
            continue
        selected.append(item)
        per_doc[item.candidate.doc_id] += 1

    # Phase 2: if not enough items and all docs exhausted at max_per_doc,
    # allow overflow (not enough distinct docs)
    if len(selected) < max_total and distinct_docs * max_per_doc < max_total:
        overflow_needed = max_total - len(selected)
        for item in ranked:
            if overflow_needed <= 0:
                break
            if item in selected:
                continue
            selected.append(item)
            per_doc[item.candidate.doc_id] += 1
            overflow_needed -= 1

    return finalizer(selected, route_name), per_doc

