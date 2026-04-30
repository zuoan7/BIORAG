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
    def __init__(self) -> None:
        self.last_summary_selection_debug: dict[str, object] = {"is_summary": False}

    def select(
        self,
        question: str,
        analysis: QueryAnalysis,
        candidates: list[EvidenceCandidate],
        config: GenerationConfig,
    ) -> list[SupportItem]:
        self.last_summary_selection_debug = {"is_summary": False}
        scored = [self._to_support_item(question, candidate) for candidate in candidates]
        scored = [item for item in scored if item.support_score >= config.v2_min_support_score]

        intent = analysis.intent
        if intent in {QueryIntent.FACTOID, QueryIntent.UNKNOWN}:
            return self._select_factoid(question, scored, config)
        if intent == QueryIntent.SUMMARY:
            selected, debug = self._select_summary(question, scored, config)
            self.last_summary_selection_debug = debug
            return selected
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
        return self._finalize(ranked[: config.v2_max_support_factoid], "factoid_top_score")

    def _select_summary(
        self,
        question: str,
        scored: list[SupportItem],
        config: GenerationConfig,
    ) -> tuple[list[SupportItem], dict[str, object]]:
        min_summary_support = 2
        preferred_max_support = min(config.v2_max_support_summary, 3)
        ranked_all = sorted(scored, key=_summary_rank_key)
        quality_pool: list[SupportItem] = []
        excluded: list[dict[str, str]] = []
        top_score_count = max(2, min(4, len(ranked_all)))
        for rank_index, item in enumerate(ranked_all):
            eligible, reasons = _evaluate_summary_quality(
                item,
                rank_index=rank_index,
                top_score_count=top_score_count,
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

        finalized = self._finalize(selected, "summary_selection")

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
) -> tuple[bool, list[str]]:
    section = (item.candidate.section or "").lower()
    if any(token in section for token in ("reference", "bibliograph")):
        return False, ["references_section"]
    if any(token in section for token in ("acknowledg", "author", "title", "content")):
        return False, ["low_value_section"]
    if "list" in section and ("figure" in section or "table" in section):
        return False, ["figure_table_list"]
    text_length = int(item.candidate.features.get("text_length") or len(item.candidate.text or ""))
    if text_length < 12:
        return False, ["too_short"]

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
