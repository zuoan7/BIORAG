from __future__ import annotations

from .models import AnswerPlan, SupportItem
from ...domain.config import GenerationConfig
from ...domain.schemas import Citation


class FinalValidator:
    def validate(
        self,
        answer: str,
        citations: list[Citation],
        plan: AnswerPlan,
        support_pack: list[SupportItem],
        config: GenerationConfig,
    ) -> tuple[str, AnswerPlan, dict]:
        valid_chunk_ids = {item.candidate.chunk_id for item in support_pack}
        filtered_citations = [citation for citation in citations if citation.chunk_id in valid_chunk_ids]
        debug = {
            "zero_citation_guardrail_triggered": False,
            "citation_count_before": len(citations),
            "citation_count_after": len(filtered_citations),
            "dropped_citation_count": len(citations) - len(filtered_citations),
        }
        citations[:] = filtered_citations

        if plan.mode == "full" and plan.missing_branches:
            plan = AnswerPlan(
                mode="partial",
                reason=plan.reason,
                covered_branches=plan.covered_branches,
                missing_branches=plan.missing_branches,
                allowed_scope=plan.allowed_scope,
                comparison_coverage=plan.comparison_coverage,
            )
            debug["downgraded_full_to_partial"] = True
        else:
            debug["downgraded_full_to_partial"] = False

        if config.v2_require_citation and plan.mode != "refuse" and not citations:
            debug["zero_citation_guardrail_triggered"] = True
            return (
                "当前知识库证据不足，无法基于已检索证据回答该问题。",
                AnswerPlan(
                    mode="refuse",
                    reason="post_generation_zero_citation_refuse",
                    covered_branches=plan.covered_branches,
                    missing_branches=plan.missing_branches,
                    allowed_scope=[],
                    comparison_coverage=plan.comparison_coverage,
                ),
                debug,
            )
        return answer, plan, debug
