from __future__ import annotations

import re

from dataclasses import asdict

from ...domain.config import GenerationConfig, ModelEndpointConfig
from ...domain.schemas import ConversationTurn, QueryAnalysis, RetrievedChunk
from .answer_builder import ExtractiveAnswerBuilder
from .answer_planner import AnswerPlanner
from .citation_binder import CitationBinder
from .evidence_ledger import EvidenceLedgerBuilder
from .models import EvidenceCandidate, GenerationV2Result, SupportItem
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

        # Fix B: limited_support_pack fallback when support_pack=0 but candidates exist
        limited_support_pack_used = False
        limited_support_pack_reason = ""
        matched_question_entities: list[str] = []
        selected_limited_support_chunk_ids: list[str] = []
        if not support_pack and candidates:
            limited = _build_limited_support_pack(question, candidates, max_chunks=3)
            if limited:
                support_pack = limited
                limited_support_pack_used = True
                limited_support_pack_reason = "support_pack_empty_entity_fallback"
                matched_question_entities = _extract_matched_entities(question, candidates)
                selected_limited_support_chunk_ids = [item.candidate.chunk_id for item in limited]

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
            "limited_support_pack_used": limited_support_pack_used,
            "limited_support_pack_reason": limited_support_pack_reason,
            "matched_question_entities": matched_question_entities,
            "selected_limited_support_chunk_ids": selected_limited_support_chunk_ids,
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


# ── Fix B: limited_support_pack helpers ───────────────────────────

_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]{2,}(?:[A-Z][a-z]+)*\d*[A-Z]?)\b"
    r"|(?:CRISPR|Cas\d+|FadL|ABC|MFS|RND|MATE|SMR|TRAP|UGGT|HAC1"
    r"|pfkA|zwf|Fam20[A-C]?|Neu5Ac|GDP|UDP|CMP|ATP|NADH|NADPH"
    r"|SpMAE|TsaM|TsaT)",
    re.IGNORECASE,
)
# Longer CJK spans (3-4 gram) capture meaningful terms better than 2-gram
_CJK_3GRAM_RE = re.compile(r"[\u4e00-\u9fff]{3}")
_CJK_4GRAM_RE = re.compile(r"[\u4e00-\u9fff]{4}")
_SECTION_NAMES = {
    "abstract", "introduction", "results", "discussion", "conclusion",
    "conclusions", "methods", "full text", "title", "background",
    "results and discussion", "materials and methods",
}


def _extract_question_entities(question: str) -> set[str]:
    """Extract key entities from question for entity-match fallback.

    Uses English domain terms, longer CJK spans (3-4 gram),
    and candidate title/section field matching to bridge Chinese-English terminology gaps.
    """
    entities: set[str] = set()
    for m in _ENTITY_RE.finditer(question):
        token = m.group()
        if token.lower() not in _SECTION_NAMES and len(token) >= 3:
            entities.add(token.lower())
    # 4-gram CJK (priority — more specific)
    for m in _CJK_4GRAM_RE.finditer(question):
        entities.add(m.group().lower())
    # 3-gram CJK (fallback)
    for m in _CJK_3GRAM_RE.finditer(question):
        entities.add(m.group().lower())
    return entities


def _extract_matched_entities(question: str,
                               candidates: list[EvidenceCandidate]) -> list[str]:
    """Return which question entities matched in candidates."""
    q_entities = _extract_question_entities(question)
    all_text = " ".join(c.text or "" for c in candidates).lower()
    return sorted(e for e in q_entities if e in all_text)


def _build_limited_support_pack(
    question: str,
    candidates: list[EvidenceCandidate],
    max_chunks: int = 3,
) -> list[SupportItem]:
    """When support_pack is empty but candidates have entity hits or minimal rerank scores,
    build a conservative limited_support_pack from top entity-matching or top-scored chunks."""
    q_entities = _extract_question_entities(question)

    # Score each candidate by entity match count (check text + title)
    entity_scored: list[tuple[int, EvidenceCandidate]] = []
    for candidate in candidates:
        text = (candidate.text or "").lower()
        title = (candidate.title or "").lower()
        haystack = text + " " + title
        hits = sum(1 for e in q_entities if e in haystack)
        if hits > 0:
            entity_scored.append((hits, candidate))

    # Fallback: if no entity matches, use top-ranked candidates by rerank score.
    # Candidates were already filtered by retrieval+rerank pipeline, so even
    # low absolute scores indicate relative relevance.
    if not entity_scored:
        ranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or c.fusion_score or 0.0,
            reverse=True,
        )
        # Only use candidates with valid scores (not None/0 for both score types)
        ranked = [c for c in ranked if (c.rerank_score or c.fusion_score or 0.0) != 0.0]
        if not ranked:
            return []
        limited: list[SupportItem] = []
        for candidate in ranked[:max_chunks]:
            score = candidate.rerank_score or candidate.fusion_score or 0.0
            limited.append(SupportItem(
                evidence_id=candidate.evidence_id,
                candidate=candidate,
                support_score=score,
                reasons=["limited_support_pack_fallback", f"rank_fallback_score:{score:.3f}"],
            ))
        return limited

    # Take top-N by entity hits
    entity_scored.sort(key=lambda x: x[0], reverse=True)
    limited: list[SupportItem] = []
    for hits, candidate in entity_scored[:max_chunks]:
        support_score = candidate.rerank_score or candidate.fusion_score or 0.0
        if support_score == 0.0:
            support_score = 0.3  # minimal score for limited pack
        limited.append(SupportItem(
            evidence_id=candidate.evidence_id,
            candidate=candidate,
            support_score=support_score,
            reasons=["limited_support_pack_fallback", f"entity_hits:{hits}"],
        ))

    return limited
