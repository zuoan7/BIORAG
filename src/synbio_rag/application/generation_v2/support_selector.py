from __future__ import annotations

import re
from collections import Counter

from ...domain.config import GenerationConfig
from ...domain.schemas import QueryAnalysis, QueryIntent
from .branch_parser import parse_comparison_branches
from .comparison_coverage import score_branch_support
from .models import EvidenceCandidate, SupportItem

_EN_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'_.-]*", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{1,4}")
_TABLE_LABEL_PATTERN = re.compile(r"(table\s*\d+|表\s*\d+|table\b|表\b)", re.IGNORECASE)
_FIGURE_LABEL_PATTERN = re.compile(r"(figure\s*\d+|fig\.\s*\d+|fig\s*\d+|图\s*\d+|figure\b|fig\.\b|图\b)", re.IGNORECASE)

class SupportPackSelector:
    def select(
        self,
        question: str,
        analysis: QueryAnalysis,
        candidates: list[EvidenceCandidate],
        config: GenerationConfig,
    ) -> list[SupportItem]:
        scored = [self._to_support_item(question, candidate) for candidate in candidates]
        scored = [item for item in scored if item.support_score >= config.v2_min_support_score]

        intent = analysis.intent
        if intent in {QueryIntent.FACTOID, QueryIntent.UNKNOWN}:
            return self._select_factoid(question, scored, config)
        if intent == QueryIntent.SUMMARY:
            return self._select_summary(scored, config)
        if intent == QueryIntent.COMPARISON:
            return self._select_comparison(question, scored, config)
        return self._select_factoid(question, scored, config)

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
        return self._finalize(ranked[: config.v2_max_support_factoid], "factoid_top_score")

    def _select_summary(self, scored: list[SupportItem], config: GenerationConfig) -> list[SupportItem]:
        min_summary_support = 2
        quality_pool = [item for item in scored if _is_summary_eligible(item)]
        ranked = sorted(quality_pool, key=_summary_rank_key)
        selected: list[SupportItem] = []
        per_doc: Counter[str] = Counter()
        max_per_doc = 3 if (_is_explicit_single_doc_summary(ranked) or _all_high_quality_same_doc(ranked)) else 2
        for item in ranked:
            if len(selected) >= config.v2_max_support_summary:
                break
            if per_doc[item.candidate.doc_id] >= max_per_doc:
                continue
            if len(selected) < min_summary_support:
                item.reasons.append("summary_min_support_fill")
            selected.append(item)
            per_doc[item.candidate.doc_id] += 1
        if len(selected) == 1:
            selected[0].reasons.append("insufficient_summary_support")
        return self._finalize(selected, "summary_doc_cap")

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


def _summary_rank_key(item: SupportItem) -> tuple[int, int, int, float]:
    return (
        _section_priority(item.candidate.section),
        0 if item.candidate.features.get("has_result_terms") else 1,
        0 if item.candidate.features.get("has_numeric") else 1,
        -item.support_score,
    )


def _section_priority(section: str) -> int:
    lowered = (section or "").lower()
    if "result" in lowered and "discussion" in lowered:
        return 0
    if "result" in lowered:
        return 1
    if "discussion" in lowered:
        return 2
    if "abstract" in lowered:
        return 3
    if "introduction" in lowered:
        return 4
    return 5


def _is_summary_eligible(item: SupportItem) -> bool:
    section = (item.candidate.section or "").lower()
    if any(token in section for token in ("reference", "bibliograph", "acknowledg", "author", "title", "content")):
        item.reasons.append("excluded:references")
        return False
    text_length = int(item.candidate.features.get("text_length") or len(item.candidate.text or ""))
    if text_length < 12:
        item.reasons.append("excluded:short_text")
        return False
    if "query_overlap" in item.reasons:
        return True
    if _section_priority(item.candidate.section) <= 4:
        return True
    if item.candidate.features.get("has_result_terms") or item.candidate.features.get("has_numeric"):
        return True
    if any(item.candidate.features.get(name) for name in ("has_table_text", "has_table_caption", "has_figure_caption")):
        return True
    return item.support_score >= 0.7


def _is_explicit_single_doc_summary(items: list[SupportItem]) -> bool:
    return bool(items) and len({item.candidate.doc_id for item in items[:3]}) == 1


def _all_high_quality_same_doc(items: list[SupportItem]) -> bool:
    return len(items) >= 2 and len({item.candidate.doc_id for item in items[:3]}) == 1
