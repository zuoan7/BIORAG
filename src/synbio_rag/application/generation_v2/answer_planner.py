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
        self.last_comparison_coverage_debug: dict[str, object] = {"reason": "not_comparison_intent", "parse_ok": False}
        self.last_summary_plan_debug: dict[str, object] = {"is_summary": False}

    def plan(
        self,
        question: str,
        analysis: QueryAnalysis,
        support_pack: list[SupportItem],
        candidates: list[EvidenceCandidate] | None = None,
        config: GenerationConfig | None = None,
    ) -> AnswerPlan:
        self.last_comparison_coverage_debug = {"reason": "not_comparison_intent", "parse_ok": False}
        self.last_summary_plan_debug = {"is_summary": False}
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
            if analysis.intent == QueryIntent.SUMMARY:
                return AnswerPlan(mode="refuse", reason="summary_no_support", allowed_scope=["supported_summary"])
            return AnswerPlan(mode="refuse", reason="no_support_pack", allowed_scope=[])

        comparison_coverage: ComparisonCoverage | None = None
        parse_result = parse_comparison_branches(question) if analysis.intent == QueryIntent.COMPARISON else None
        branches = parse_result.branches if parse_result and parse_result.parse_ok else []
        if analysis.intent == QueryIntent.COMPARISON:
            if config is not None and not config.v2_enable_comparison_coverage:
                self.last_comparison_coverage_debug = {"reason": "coverage_disabled", "parse_ok": False}
            elif not parse_result or not parse_result.parse_ok:
                self.last_comparison_coverage_debug = {
                    "reason": "branch_parse_failed",
                    "parse_ok": False,
                    "branches": list(parse_result.branches) if parse_result else [],
                    "parse_failure_reason": parse_result.reason if parse_result else "missing_parse_result",
                    "parser_patterns_tried": list(parse_result.parser_patterns_tried) if parse_result else [],
                }
            elif not branches:
                self.last_comparison_coverage_debug = {"reason": "no_valid_branches", "parse_ok": False}
            elif not support_pack:
                self.last_comparison_coverage_debug = {"reason": "empty_support_pack", "parse_ok": True, "branches": list(branches)}
            else:
                try:
                    comparison_coverage = build_comparison_coverage(question, branches, support_pack, candidates)
                    self.last_comparison_coverage_debug = comparison_coverage.to_dict()
                except Exception:
                    self.last_comparison_coverage_debug = {"reason": "coverage_builder_error", "parse_ok": False}
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
            self.last_summary_plan_debug = _build_summary_plan_debug(support_pack)
            if not support_pack:
                base_plan = AnswerPlan(
                    mode="refuse",
                    reason="summary_no_support",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["supported_summary"],
                    comparison_coverage=comparison_coverage,
                )
            elif len(support_pack) == 1:
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="summary_support_count",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["supported_summary", "limited_summary_single_evidence"],
                    comparison_coverage=comparison_coverage,
                )
            elif self.last_summary_plan_debug.get("insufficient_qualified_summary_support"):
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="summary_support_count",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["supported_summary", "limited_summary_single_evidence"],
                    comparison_coverage=comparison_coverage,
                )
            elif self.last_summary_plan_debug.get("abstract_only"):
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="summary_abstract_only",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["supported_summary"],
                    comparison_coverage=comparison_coverage,
                )
            elif self.last_summary_plan_debug.get("single_doc_limited"):
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="summary_single_doc_limited",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["supported_summary"],
                    comparison_coverage=comparison_coverage,
                )
            elif self.last_summary_plan_debug.get("low_diversity"):
                base_plan = AnswerPlan(
                    mode="partial",
                    reason="summary_low_diversity",
                    covered_branches=covered_branches,
                    missing_branches=missing_branches,
                    allowed_scope=["supported_summary"],
                    comparison_coverage=comparison_coverage,
                )
            else:
                base_plan = AnswerPlan(
                    mode="full",
                    reason="summary_multi_support",
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
                    unique_direct_primary_ids = {
                        evidence_id
                        for entry in comparison_coverage.branch_evidence
                        if entry.status == "direct"
                        for evidence_id in entry.primary_evidence_ids
                    }
                    if (
                        support_pack
                        and statuses == {"direct"}
                        and comparison_coverage.reason != "shared_evidence_limited_comparison"
                        and len(unique_direct_primary_ids) >= len(branches)
                    ):
                        mode = "partial"
                        reason = "direct_all_branches_independent"
                    elif support_pack and comparison_coverage.reason == "shared_evidence_limited_comparison":
                        mode = "partial"
                        reason = "shared_evidence_limited_comparison"
                    elif support_pack and (direct_branches or indirect_branches):
                        mode = "partial"
                        if direct_branches and indirect_branches:
                            reason = "branch_partial_support"
                        elif direct_branches:
                            reason = "comparison_branch_coverage"
                        else:
                            reason = "comparison_indirect_support"
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
                        if reason == "shared_evidence_limited_comparison":
                            allowed_scope.append("shared_evidence_not_full_comparison")
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


def _build_summary_plan_debug(support_pack: list[SupportItem]) -> dict[str, object]:
    section_buckets = [_summary_section_bucket(item.candidate.section) for item in support_pack]
    doc_ids = [item.candidate.doc_id for item in support_pack]
    unique_docs = set(doc_ids)
    unique_sections = set(section_buckets)
    abstract_only = bool(section_buckets) and unique_sections == {"abstract"}
    low_diversity = len(unique_sections) <= 1 and len(unique_docs) <= 1
    return {
        "is_summary": True,
        "support_pack_count": len(support_pack),
        "doc_count": len(unique_docs),
        "section_count": len(unique_sections),
        "abstract_only": abstract_only,
        "single_doc_limited": len(unique_docs) == 1 and len(support_pack) >= 2,
        "low_diversity": low_diversity and not abstract_only,
        "insufficient_qualified_summary_support": any(
            "insufficient_qualified_summary_support" in item.reasons for item in support_pack
        ),
    }


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
    return "other"
