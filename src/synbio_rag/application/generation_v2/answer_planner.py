from __future__ import annotations

import re

from ...domain.config import GenerationConfig
from ...domain.schemas import QueryAnalysis, QueryIntent
from .branch_parser import parse_comparison_branches
from .comparison_coverage import build_comparison_coverage
from .guardrails import detect_existence_question, evaluate_existence_support
from .models import AnswerPlan, ComparisonCoverage, EvidenceCandidate, SupportItem

_TABLE_LABEL_PATTERN = re.compile(r"(table\s*\d+|表\s*\d+|table\b|表\b)", re.IGNORECASE)
_FIGURE_LABEL_PATTERN = re.compile(r"(figure\s*\d+|fig\.\s*\d+|fig\s*\d+|图\s*\d+|figure\b|fig\.\b|图\b)", re.IGNORECASE)

class AnswerPlanner:
    def __init__(self) -> None:
        self.last_existence_guardrail: dict[str, object] = {}

    def plan(
        self,
        question: str,
        analysis: QueryAnalysis,
        support_pack: list[SupportItem],
        candidates: list[EvidenceCandidate] | None = None,
        config: GenerationConfig | None = None,
    ) -> AnswerPlan:
        existence_signal = detect_existence_question(question)
        existence_assessment = evaluate_existence_support(question, support_pack, candidates)
        self.last_existence_guardrail = {
            "is_existence_question": existence_signal.is_existence_question,
            "detection_reason": existence_signal.reason,
            "matched_patterns": list(existence_signal.matched_patterns),
            "target_terms": list(existence_signal.target_terms),
            "support_status": existence_assessment.support_status,
            "support_reason": existence_assessment.reason,
            "matched_core_terms": list(existence_assessment.matched_core_terms),
            "missing_core_terms": list(existence_assessment.missing_core_terms),
            "weak_signals": list(existence_assessment.weak_signals),
        }

        if not support_pack:
            if existence_signal.is_existence_question:
                return AnswerPlan(
                    mode="refuse",
                    reason="existence_no_support",
                    allowed_scope=["insufficient_library_evidence"],
                )
            return AnswerPlan(mode="refuse", reason="no_support_pack", allowed_scope=[])

        comparison_coverage: ComparisonCoverage | None = None
        parse_result = parse_comparison_branches(question) if analysis.intent == QueryIntent.COMPARISON else None
        branches = parse_result.branches if parse_result and parse_result.parse_ok else []
        if (
            analysis.intent == QueryIntent.COMPARISON
            and branches
            and (config is None or config.v2_enable_comparison_coverage)
        ):
            comparison_coverage = build_comparison_coverage(question, branches, support_pack, candidates)
        covered_branches = list(comparison_coverage.covered_branches) if comparison_coverage else [
            branch for branch in branches if any(_branch_matches(branch, item) for item in support_pack)
        ]
        missing_branches = list(comparison_coverage.missing_branches) if comparison_coverage else [
            branch for branch in branches if branch not in covered_branches
        ]
        is_structured_question = _question_mentions_table(question) or _question_mentions_figure(question)
        has_structured_support = any(_has_structured_match(question, item) for item in support_pack)

        if is_structured_question:
            if has_structured_support:
                mode = "full" if not missing_branches else "partial"
                base_plan = AnswerPlan(
                    mode=mode,
                    reason="structured_evidence_found",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["structured_evidence_only"],
                    comparison_coverage=comparison_coverage,
                )
            else:
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="missing_explicit_structured_evidence",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["plain_text_support_only"],
                    comparison_coverage=comparison_coverage,
                )
        elif analysis.intent in {QueryIntent.FACTOID, QueryIntent.UNKNOWN}:
            base_plan = AnswerPlan(
                mode="full",
                reason="factoid_supported",
                covered_branches=covered_branches,
                missing_branches=missing_branches,
                allowed_scope=["supported_facts"],
                comparison_coverage=comparison_coverage,
            )
        elif analysis.intent == QueryIntent.SUMMARY:
            base_plan = AnswerPlan(
                mode="full" if len(support_pack) >= 2 else "partial",
                reason="summary_support_count",
                covered_branches=covered_branches,
                missing_branches=missing_branches,
                allowed_scope=["supported_summary"],
                comparison_coverage=comparison_coverage,
            )
        elif analysis.intent == QueryIntent.COMPARISON:
            if branches:
                if comparison_coverage:
                    statuses = {entry.status for entry in comparison_coverage.branch_evidence}
                    direct_branches = [entry.branch for entry in comparison_coverage.branch_evidence if entry.status == "direct"]
                    indirect_branches = [entry.branch for entry in comparison_coverage.branch_evidence if entry.status == "indirect"]
                    unique_direct_evidence_ids = {
                        evidence_id
                        for entry in comparison_coverage.branch_evidence
                        if entry.status == "direct"
                        for evidence_id in entry.evidence_ids
                    }
                    if support_pack and statuses == {"direct"} and len(unique_direct_evidence_ids) >= len(branches):
                        mode = "full"
                        reason = "comparison_branch_coverage"
                    elif support_pack and (direct_branches or indirect_branches):
                        mode = "partial"
                        reason = "comparison_branch_coverage" if direct_branches else "comparison_indirect_support"
                    elif support_pack:
                        mode = "partial"
                        reason = "comparison_indirect_support"
                    else:
                        mode = "refuse"
                        reason = "no_support_pack"
                    allowed_scope = ["supported_comparison"]
                    if mode == "partial":
                        allowed_scope.extend(
                            ["branch_limited_comparison", "missing_or_indirect_branches_must_be_disclosed"]
                        )
                elif covered_branches and not missing_branches:
                    mode = "full"
                    reason = "comparison_branch_coverage"
                    allowed_scope = ["supported_comparison"]
                elif covered_branches or _has_meaningful_comparison_support(support_pack):
                    mode = "partial"
                    reason = "comparison_branch_coverage"
                    allowed_scope = ["supported_comparison"]
                else:
                    mode = "refuse"
                    reason = "comparison_branch_coverage"
                    allowed_scope = ["supported_comparison"]
                base_plan = AnswerPlan(
                    mode=mode,
                    reason=reason,
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=allowed_scope,
                    comparison_coverage=comparison_coverage,
                )
            else:
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="comparison_evidence_incomplete",
                    allowed_scope=["supported_comparison"],
                    comparison_coverage=comparison_coverage,
                )
        elif analysis.intent == QueryIntent.EXPERIMENT:
            base_plan = AnswerPlan(
                mode="partial",
                reason="experiment_queries_only_allow_summary",
                covered_branches=covered_branches,
                missing_branches=missing_branches,
                allowed_scope=["existing_evidence_summary_only"],
                comparison_coverage=comparison_coverage,
            )
        else:
            base_plan = AnswerPlan(
                mode="partial",
                reason="fallback_partial",
                allowed_scope=["supported_content"],
                comparison_coverage=comparison_coverage,
            )

        if not existence_signal.is_existence_question:
            return base_plan
        if existence_assessment.support_status == "none":
            return AnswerPlan(
                mode="refuse",
                reason="existence_no_support",
                covered_branches=list(base_plan.covered_branches),
                missing_branches=list(base_plan.missing_branches),
                allowed_scope=["insufficient_library_evidence"],
                comparison_coverage=base_plan.comparison_coverage,
            )
        if existence_assessment.support_status == "weak":
            if not support_pack:
                return AnswerPlan(mode="refuse", reason="existence_no_support", allowed_scope=["insufficient_library_evidence"])
            return AnswerPlan(
                mode="partial",
                reason="existence_weak_support",
                covered_branches=list(base_plan.covered_branches),
                missing_branches=list(base_plan.missing_branches),
                allowed_scope=[
                    "only_report_limited_or_indirect_evidence",
                    "do_not_claim_library_contains_requested_item",
                ],
                comparison_coverage=base_plan.comparison_coverage,
            )
        return base_plan


def _question_mentions_table(question: str) -> bool:
    return bool(_TABLE_LABEL_PATTERN.search(question))


def _question_mentions_figure(question: str) -> bool:
    return bool(_FIGURE_LABEL_PATTERN.search(question))


def _has_structured_match(question: str, item: SupportItem) -> bool:
    if _question_mentions_table(question):
        return bool(
            item.candidate.features.get("has_table_text") or item.candidate.features.get("has_table_caption")
        )
    if _question_mentions_figure(question):
        return bool(item.candidate.features.get("has_figure_caption"))
    return False


def _branch_matches(branch: str, item: SupportItem) -> bool:
    haystack = " ".join(
        [
            item.candidate.title.lower(),
            item.candidate.section.lower(),
            item.candidate.text.lower(),
            " ".join(str(value).lower() for value in item.candidate.metadata.values()),
        ]
    )
    branch_lower = branch.lower()
    if re.fullmatch(r"[a-z0-9_.-]{1,4}", branch_lower):
        return bool(re.search(rf"\b{re.escape(branch_lower)}\b", haystack))
    return branch_lower in haystack


def _has_meaningful_comparison_support(support_pack: list[SupportItem]) -> bool:
    return any(_is_meaningful_support(item) for item in support_pack)


def _is_meaningful_support(item: SupportItem) -> bool:
    section = (item.candidate.section or "").lower()
    if any(token in section for token in ("reference", "bibliograph", "acknowledg", "author", "title", "content")):
        return False
    text_length = int(item.candidate.features.get("text_length") or len(item.candidate.text or ""))
    return text_length >= 12 and item.support_score > 0.0
