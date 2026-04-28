from __future__ import annotations

from dataclasses import asdict

from ...domain.config import GenerationConfig, ModelEndpointConfig
from ...domain.schemas import ConversationTurn, QueryAnalysis, RetrievedChunk
from .answer_builder import ExtractiveAnswerBuilder
from .answer_planner import AnswerPlanner
from .citation_binder import CitationBinder
from .evidence_ledger import EvidenceLedgerBuilder
from .models import GenerationV2Result
from .neighbor_audit import NeighborAuditEngine
from .qwen_synthesizer import QwenSynthesizer
from .support_selector import SupportPackSelector
from .validator import FinalValidator


class GenerationV2Service:
    def __init__(
        self,
        llm_config: ModelEndpointConfig | None = None,
        synthesizer: QwenSynthesizer | None = None,
        neighbor_audit_engine: NeighborAuditEngine | None = None,
    ) -> None:
        self.ledger_builder = EvidenceLedgerBuilder()
        self.support_selector = SupportPackSelector()
        self.answer_planner = AnswerPlanner()
        self.answer_builder = ExtractiveAnswerBuilder()
        self.synthesizer = synthesizer or QwenSynthesizer(llm_config)
        self.citation_binder = CitationBinder()
        self.validator = FinalValidator()
        # optional neighbor audit engine; None means "no corpus index available"
        self.neighbor_audit_engine: NeighborAuditEngine | None = neighbor_audit_engine

    def run(
        self,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
        config: GenerationConfig,
        history: list[ConversationTurn] | None = None,
    ) -> GenerationV2Result:
        candidates = self.ledger_builder.build(question, analysis, seed_chunks)
        support_pack = self.support_selector.select(question, analysis, candidates, config)
        plan = self.answer_planner.plan(question, analysis, support_pack, candidates, config)
        extractive_answer = self.answer_builder.build(question, analysis, plan, support_pack)
        existence_guardrail = dict(self.answer_planner.last_existence_guardrail)
        qwen_attempted = bool(
            config.v2_use_qwen_synthesis and plan.mode != "refuse" and bool(support_pack)
        )
        synthesis_result = self.synthesizer.synthesize(
            question=question,
            plan=plan,
            support_pack=support_pack,
            extractive_answer=extractive_answer,
            config=config,
            existence_guardrail=existence_guardrail,
        )
        draft_answer = synthesis_result.answer
        answer, citations, citation_debug = self.citation_binder.bind(draft_answer, support_pack)
        answer, plan, validator_debug = self.validator.validate(answer, citations, plan, support_pack, config)
        qwen_output_evidence_ids = citation_debug.get("ordered_evidence_ids", [])

        # neighbor audit (dry-run only; never mutates support_pack/citations/answer)
        if self.neighbor_audit_engine is not None and config.v2_enable_neighbor_audit:
            neighbor_audit_result = self.neighbor_audit_engine.run(
                question=question,
                analysis=analysis,
                seed_chunks=seed_chunks,
                candidates=candidates,
                config=config,
                plan=plan,
            )
        else:
            from .neighbor_audit import NeighborAuditResult
            neighbor_audit_result = NeighborAuditResult(
                enabled=False,
                window=config.v2_neighbor_window,
                promotion_enabled=config.v2_enable_neighbor_promotion,
                promotion_dry_run=config.v2_neighbor_promotion_dry_run,
                candidate_count=0,
                dry_run_promoted_count=0,
                excluded_count=0,
            )

        support_selection_debug = {
            "candidate_ids": [candidate.evidence_id for candidate in candidates],
            "selected_evidence_ids": [item.evidence_id for item in support_pack],
            "citation_binding": citation_debug,
            "summary_selection": dict(getattr(self.support_selector, "last_summary_selection_debug", {"is_summary": False})),
        }
        comparison_coverage_debug = (
            plan.comparison_coverage.to_dict()
            if plan.comparison_coverage
            else dict(getattr(self.answer_planner, "last_comparison_coverage_debug", {"reason": "not_comparison_intent", "parse_ok": False}))
        )
        summary_selection_debug = dict(getattr(self.support_selector, "last_summary_selection_debug", {"is_summary": False}))
        # 当最终为 refuse 且原因属于 no-support 类时，覆写 summary_selection 避免与 summary_plan 冲突
        _refuse_no_support_reasons = {"no_support_pack", "existence_no_support", "no_support", "empty_support", "summary_no_support"}
        if plan.mode == "refuse" and (
            plan.reason in _refuse_no_support_reasons
            or (not support_pack and summary_selection_debug.get("is_summary"))
        ):
            summary_selection_debug = {
                "is_summary": summary_selection_debug.get("is_summary", False),
                "skipped": True,
                "skip_reason": "refuse_or_existence_no_support",
            }
        summary_plan_debug = dict(getattr(self.answer_planner, "last_summary_plan_debug", {"is_summary": False}))
        debug = {
            "generation_version": "v2",
            "neighbor_expansion_used": False,
            "external_tools_used": False,
            "history_used": bool(config.v2_use_history and history),
            "candidate_count": len(candidates),
            "support_pack_count": len(support_pack),
            "answer_mode": plan.mode,
            "refuse_reason": plan.reason if plan.mode == "refuse" else "",
            "covered_branches": list(plan.covered_branches),
            "missing_branches": list(plan.missing_branches),
            "support_selection_debug": support_selection_debug,
            "citation_count": len(citations),
            "validator_debug": validator_debug,
            "existence_guardrail": existence_guardrail,
            "comparison_coverage": comparison_coverage_debug,
            "summary_selection": summary_selection_debug,
            "summary_plan": summary_plan_debug,
            "qwen_synthesis": {
                "enabled": bool(config.v2_use_qwen_synthesis),
                "attempted": qwen_attempted,
                "used_qwen": synthesis_result.used_qwen,
                "fallback_used": synthesis_result.fallback_used,
                "fallback_reason": synthesis_result.fallback_reason,
                "validation_flags": list(synthesis_result.validation_flags),
                "validation_details": dict(synthesis_result.validation_details),
                "disallowed_evidence_ids": list(
                    synthesis_result.validation_details.get("disallowed_evidence_ids", [])
                ),
                "raw_output_preview": synthesis_result.raw_output_preview,
                "input_evidence_ids": [item.evidence_id for item in support_pack],
                "output_evidence_ids": qwen_output_evidence_ids,
                "allowed_citation_evidence_ids": (
                    list(plan.comparison_coverage.allowed_citation_evidence_ids)
                    if plan.comparison_coverage
                    else []
                ),
                "citation_validation_policy": (
                    "comparison_allowed_subset"
                    if plan.comparison_coverage and plan.comparison_coverage.allowed_citation_evidence_ids
                    else "exact_set"
                ),
            },
            "support_pack": [
                {
                    "evidence_id": item.evidence_id,
                    "chunk_id": item.candidate.chunk_id,
                    "doc_id": item.candidate.doc_id,
                    "section": item.candidate.section,
                    "support_score": item.support_score,
                    "reasons": list(item.reasons),
                }
                for item in support_pack
            ],
            "answer_plan": plan.to_dict(),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "history_turn_count": len(history or []),
            "neighbor_audit": neighbor_audit_result.to_dict(),
        }
        return GenerationV2Result(
            answer=answer,
            citations=list(citations),
            answer_plan=plan,
            support_pack=support_pack,
            debug=debug,
            confidence_chunks=[RetrievedChunk(**asdict(chunk)) for chunk in seed_chunks],
        )
