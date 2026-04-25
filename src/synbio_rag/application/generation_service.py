from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..domain.config import Round8PolicyConfig
from ..domain.schemas import (
    Citation,
    ConversationTurn,
    QueryAnalysis,
    QueryIntent,
    RetrievedChunk,
)
from ..infrastructure.clients.openai_compatible import OpenAICompatibleClient, extract_json_block

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+|[\u4e00-\u9fff]")
_UNCERTAINTY_TERMS = ("may", "might", "unclear", "possible", "可能", "尚不清楚", "提示")

logger = logging.getLogger(__name__)

_NO_EVIDENCE_REFUSAL = "当前检索结果没有提供足够可引用证据，无法基于文库可靠回答。"


@dataclass
class EvidenceAssessment:
    level: str
    citation_count: int
    unique_doc_count: int
    score_strength: float
    keyword_overlap: float
    branch_coverage: float
    negative_signal: float
    should_refuse: bool
    partial_only: bool
    reason: str
    hard_refusal_reason: str = ""
    hard_refusal_disabled_by_round8: bool = False
    should_refuse_original: bool = False
    should_refuse_final: bool = False
    final_answer_mode: str = "full"
    claim_fallback_enabled: bool = False
    fallback_method: str = ""
    evidence_unit_count: int = 0
    supported_claim_count: int = 0
    recovered_by_claim_fallback: bool = False
    evidence_units: list[dict[str, Any]] = field(default_factory=list)
    support_pack: list[dict[str, Any]] = field(default_factory=list)
    support_pack_count: int = 0
    support_pack_doc_ids: list[str] = field(default_factory=list)
    support_pack_chunk_ids: list[str] = field(default_factory=list)
    empty_support_pack_guardrail_triggered: bool = False
    zero_citation_guardrail_triggered: bool = False
    fallback_guardrail_failed: bool = False
    refusal_reason: str = ""
    candidate_support_pack_count: int = 0
    limited_partial_compare_triggered: bool = False
    covered_branch_count: int = 0
    missing_branch_count: int = 0
    missing_branch_labels: list[str] = field(default_factory=list)
    missing_branch_disclosed: bool = False
    unsupported_branch_claim_count: int = 0
    unsupported_branch_claims: list[str] = field(default_factory=list)
    covered_branch_but_refuse: bool = False
    branch_support_pack_counts: dict[str, int] = field(default_factory=dict)
    branch_citable_quote_counts: dict[str, int] = field(default_factory=dict)
    citation_limit_applied: bool = False
    citation_budget: dict[str, Any] = field(default_factory=dict)
    citation_count_before_budget: int = 0
    citation_count_after_budget: int = 0


# ---------------------------------------------------------------------------
# Phase A: Evidence Extraction prompt
# ---------------------------------------------------------------------------
_PHASE_A_SYSTEM = (
    "你是证据抽取助手。从给定的证据片段中，针对用户问题提取关键事实点。\n"
    "要求：\n"
    "1. 每个事实点必须直接来自证据片段，标注来源编号 [X]。\n"
    "2. 如果问题的某个方面没有证据支持，在 missing 列表中标注。\n"
    "3. 不要添加证据之外的任何信息或推断。\n"
    "4. 尽量提取具体数据（数值、百分比、倍数变化等）。\n"
    "5. 首先判断证据片段的主题是否与问题的核心主题一致。\n"
    "   - 如果证据片段讨论的主题（如某种化合物、某个生物过程）与问题询问的主题不同，\n"
    "     则 relevant 设为 false，claims 为空。\n"
    "   - 例如：问题问 CAR-T 临床试验，但证据讲的是人乳寡糖合成，则 relevant=false。\n"
    "   - 例如：问题问 mRNA 疫苗，但证据讲的是糖基化工程，则 relevant=false。\n"
    "   - 如果证据和问题的核心主题一致（即使只覆盖部分方面），relevant=true。\n"
    "6. 严格区分「实质性内容」和「参考文献列表/间接引用」：\n"
    "   - 参考文献列表的典型特征：连续编号（如 109. 110. 111.）、作者 et al.、\n"
    "     期刊名+卷号+页码（如 N. Engl. J. Med. 384, 252-260）、URL 链接。\n"
    "   - 如果某个事实点的唯一来源是参考文献列表中的论文标题，而证据片段正文\n"
    "     没有对该论文的内容做任何展开描述（没有数据、没有结论引用、没有方法描述），\n"
    "     则该事实点 status 设为 \"unsupported\"，因为文库中只有引用条目，\n"
    "     没有该论文的实际内容。\n"
    "   - 只有证据片段正文（非参考文献列表部分）包含问题所需的实质性内容\n"
    "     （数据、机制描述、实验结果）时，才标记 status=\"supported\"。\n"
    '输出严格 JSON 格式：\n'
    '{"relevant": true, "claims": [{"fact": "...", "source": [1,2], "status": "supported"}], '
    '"missing": ["问题中未被证据覆盖的方面"]}'
)

_PHASE_A_USER = (
    "请从以下证据中，针对问题提取所有关键事实点。\n\n"
    "{context}\n\n"
    "问题: {question}\n\n"
    "请直接输出 JSON，不要添加其他文字。"
)

# ---------------------------------------------------------------------------
# Phase B: Answer Synthesis prompts — per route type
# ---------------------------------------------------------------------------
_PHASE_B_SYSTEM_BASE = (
    "你是合成生物学企业知识助手。\n"
    "严格规则：\n"
    "1. 只能使用下方提供的已确认事实点来回答，不允许引入任何新信息。\n"
    "2. 每个结论性句子后必须标注来源编号，如 [1][3]。没有来源编号支撑的句子必须删除。\n"
    "3. 不要用外部常识补全，不要把推测写成确定结论。\n"
    "4. 直接回答问题，不要输出与问题无关的前言或寒暄。\n"
    "5. 如果有证据缺失的方面，在回答最后用一句话简要提及即可，不要展开。\n"
    "6. 优先使用证据原文措辞，不要改写或扩展原文含义。\n"
    "7. 回答范围严格限定在已确认事实点内，不要推断因果关系或补充机制细节。\n"
    "8. 用中文回答。专有名词（基因名、蛋白名、化合物名等）保留英文原文。\n"
)

_PHASE_B_TEMPLATES = {
    QueryIntent.FACTOID: (
        _PHASE_B_SYSTEM_BASE +
        "回答要求：先给出直接结论，再用证据支撑，引用来源编号。\n"
    ),
    QueryIntent.SUMMARY: (
        _PHASE_B_SYSTEM_BASE +
        "回答要求：按逻辑层次组织回答（如先总结核心结论，再展开支撑细节），每个要点标注来源编号。不要使用固定的段落标题。\n"
    ),
    QueryIntent.COMPARISON: (
        _PHASE_B_SYSTEM_BASE +
        "回答要求：分别列出相同点和不同点，每项标注来源编号。证据不足的维度不要展开推测。\n"
    ),
    QueryIntent.EXPERIMENT: (
        _PHASE_B_SYSTEM_BASE +
        "回答要求：说明核心机制或方法，按步骤或因果链组织，标注来源编号。\n"
    ),
}

_PHASE_B_COMPARISON_V2 = (
    _PHASE_B_SYSTEM_BASE +
    "比较类问题专项规则（v2）：\n"
    "1. 比较对象可以来自同一篇文献，也可以来自多篇文献。不要仅因为所有证据都来自同一个 doc_id 就拒答或说『证据不足』。\n"
    "2. 只要上下文中包含两个或多个比较对象、实验组、条件、菌株、基因、启动子、代谢路径或表型的信息，就应基于这些证据进行比较。\n"
    "3. 如果某个比较对象缺少证据，请明确说明该对象证据不足，并只回答已有证据支持的部分。\n"
    "4. 不得根据常识或模型先验补全未被上下文支持的差异。\n"
    "回答结构：\n"
    "  1. 比较对象及其证据（按对象分条列出，每条标注来源编号）\n"
    "  2. 主要差异或共同点\n"
    "  3. 证据限制（说明哪些比较对象或维度证据不足）\n"
)

_PHASE_B_USER = (
    "已确认事实：\n{claims_text}\n\n"
    "证据缺失方面：{missing_text}\n\n"
    "原始证据片段（请尽量使用原文措辞）：\n{context}\n\n"
    "问题: {question}\n\n"
    "请直接回答问题。"
)


class QwenChatGenerator:

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model_name: str = "qwen-plus",
        temperature: float = 0.1,
        round8_config: Round8PolicyConfig | None = None,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.round8_config = round8_config or Round8PolicyConfig()
        self.client = OpenAICompatibleClient(api_base, api_key, timeout_seconds=60)

    def generate(
        self,
        question: str,
        context: str,
        chunks: list[RetrievedChunk],
        analysis: QueryAnalysis | None = None,
        history: list[ConversationTurn] | None = None,
        assessment: EvidenceAssessment | None = None,
    ) -> str:
        del context
        assessment = assessment or self.assess_evidence(question, chunks, analysis=analysis)
        support_context = _build_support_context(assessment.support_pack)
        if assessment.should_refuse:
            return _build_refusal_answer(assessment)
        if assessment.final_answer_mode == "limited_partial_compare":
            return _build_limited_partial_compare_answer(assessment)
        if assessment.recovered_by_claim_fallback:
            return _build_fallback_answer(question, assessment)
        if self.client.is_enabled():
            try:
                return self._generate_two_phase(
                    question, support_context, assessment, analysis, history, chunks,
                )
            except Exception:
                logger.exception("Two-phase generation failed, falling back")
        if assessment.partial_only:
            return _build_limited_answer(question, assessment)
        return _build_supported_answer(assessment)

    # ------------------------------------------------------------------
    # Two-phase generation core
    # ------------------------------------------------------------------
    def _generate_two_phase(
        self,
        question: str,
        context: str,
        assessment: EvidenceAssessment,
        analysis: QueryAnalysis | None,
        history: list[ConversationTurn] | None,
        chunks: list[RetrievedChunk] | None = None,
    ) -> str:
        claims_data = self._phase_a_extract(question, context)
        logger.debug(
            "Phase A claims for [%s]: %s",
            question[:60],
            json.dumps(claims_data, ensure_ascii=False)[:800],
        )

        should_abstain = _phase_a_quality_gate(
            claims_data, question, chunks or [],
        )
        if should_abstain:
            missing = claims_data.get("missing", [])
            missing_text = "、".join(missing) if missing else "问题的核心方面"
            logger.info("Phase A quality gate: abstaining — %s", missing_text)
            return f"当前检索到的证据与问题直接相关性不足，无法可靠作答。未覆盖：{missing_text}。"

        validated, rejected = _validate_claims(
            claims_data.get("claims", []), question,
        )
        logger.info(
            "Claim validation: %d valid, %d rejected out of %d total",
            len(validated), len(rejected),
            len(claims_data.get("claims", [])),
        )
        if rejected:
            for r in rejected:
                logger.debug("Rejected claim: %s", json.dumps(r, ensure_ascii=False)[:200])

        if not validated:
            missing = claims_data.get("missing", [])
            missing_text = "、".join(missing) if missing else "问题的核心方面"
            return f"当前检索到的证据与问题直接相关性不足，无法可靠作答。未覆盖：{missing_text}。"

        validated_data = {
            "claims": validated,
            "missing": claims_data.get("missing", []),
        }
        intent = analysis.intent if analysis else QueryIntent.UNKNOWN
        return self._phase_b_synthesize(question, validated_data, intent, history, context)

    def _phase_a_extract(self, question: str, context: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": _PHASE_A_SYSTEM},
            {
                "role": "user",
                "content": _PHASE_A_USER.format(context=context, question=question),
            },
        ]
        raw = self.client.chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=0.05,
        )
        try:
            return extract_json_block(raw)
        except (ValueError, Exception):
            logger.warning("Phase A JSON parse failed, using raw text fallback")
            return {
                "claims": [{"fact": raw.strip(), "source": [], "status": "supported"}],
                "missing": [],
            }

    def _phase_b_synthesize(
        self,
        question: str,
        claims_data: dict[str, Any],
        intent: QueryIntent,
        history: list[ConversationTurn] | None,
        context: str = "",
    ) -> str:
        claims = claims_data.get("claims", [])
        missing = claims_data.get("missing", [])

        supported = [c for c in claims if c.get("status") == "supported"]
        if not supported:
            missing_text = "、".join(missing) if missing else "所有方面"
            return f"证据不足。当前证据未能覆盖以下方面：{missing_text}。"

        claims_lines = []
        for i, claim in enumerate(supported, 1):
            sources = claim.get("source", [])
            source_str = "".join(f"[{s}]" for s in sources) if sources else ""
            claims_lines.append(f"{i}. {claim['fact']} {source_str}")
        claims_text = "\n".join(claims_lines)

        missing_text = "、".join(missing) if missing else "无"

        use_comparison_v2 = (
            intent == QueryIntent.COMPARISON
            and self.round8_config.enable_round8_policy
            and self.round8_config.enable_comparison_prompt_v2
        )
        if use_comparison_v2:
            system_prompt = _PHASE_B_COMPARISON_V2
        else:
            system_prompt = _PHASE_B_TEMPLATES.get(intent, _PHASE_B_TEMPLATES[QueryIntent.FACTOID])
        user_content = _PHASE_B_USER.format(
            claims_text=claims_text,
            missing_text=missing_text,
            question=question,
            context=context,
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history[-6:]:
                messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_content})

        return self.client.chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

    # ------------------------------------------------------------------
    # Evidence assessment (unchanged from Round 4)
    # ------------------------------------------------------------------
    def assess_evidence(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        analysis: QueryAnalysis | None = None,
    ) -> EvidenceAssessment:
        if not chunks:
            return EvidenceAssessment(
                level="none",
                citation_count=0,
                unique_doc_count=0,
                score_strength=0.0,
                keyword_overlap=0.0,
                branch_coverage=0.0,
                negative_signal=0.0,
                should_refuse=True,
                partial_only=False,
                reason="当前知识库中没有检索到可支撑该问题的相关片段。",
                hard_refusal_reason="no_retrieved_chunks",
                should_refuse_original=True,
                should_refuse_final=True,
                final_answer_mode="refuse",
                refusal_reason="empty_support_refuse",
            )

        top_chunks = chunks[:5]
        top_scores = [_normalize_score(max(chunk.rerank_score, chunk.vector_score, chunk.fusion_score)) for chunk in top_chunks]
        score_strength = sum(top_scores) / len(top_scores)
        unique_doc_count = len({chunk.doc_id for chunk in top_chunks})
        keyword_overlap = _keyword_overlap(question, top_chunks)
        evidence_signal = _evidence_feature_signal(top_chunks)
        branch_coverage = _branch_coverage(question, top_chunks, analysis)
        negative_signal = _negative_signal(question, top_chunks)
        citation_count = len(top_chunks)
        level = "strong"
        should_refuse = False
        partial_only = False
        reason = ""
        hard_refusal_reason = ""
        refusal_reason = ""
        fallback_guardrail_failed = False
        covered_branch_but_refuse = False

        intent = analysis.intent if analysis else QueryIntent.UNKNOWN
        (
            score_strength_threshold,
            partial_boundary,
            _,
            evidence_signal_weak,
            branch_coverage_threshold,
        ) = _round8_route_thresholds(analysis, self.round8_config)

        if citation_count == 0 or score_strength < score_strength_threshold or (keyword_overlap < 0.05 and evidence_signal < evidence_signal_weak):
            level = "none"
            should_refuse = True
            reason = "检索到的证据与问题的直接相关性不足，无法可靠作答。"
            hard_refusal_reason = "weak_evidence"
            refusal_reason = "weak_evidence"
        elif analysis and analysis.intent == QueryIntent.COMPARISON and branch_coverage < branch_coverage_threshold:
            level = "partial"
            should_refuse = True
            reason = "比较问题所需的多个分支证据覆盖不足，无法完成完整对比。"
            hard_refusal_reason = "comparison_branch_coverage"
            refusal_reason = "comparison_branch_coverage"
        elif unique_doc_count == 1 and analysis and analysis.intent == QueryIntent.COMPARISON:
            level = "partial"
            should_refuse = True
            reason = "比较问题目前只命中单一文档来源，缺少交叉证据。"
            hard_refusal_reason = "comparison_single_doc"
            refusal_reason = "comparison_single_doc"
        elif negative_signal > 0.55:
            level = "partial"
            partial_only = True
            reason = "当前证据中混入较多综述或应用类内容，只能给出有限结论。"
        elif score_strength < partial_boundary or (keyword_overlap < 0.1 and evidence_signal < 0.5):
            level = "partial"
            partial_only = True
            reason = "证据只覆盖了问题的一部分，回答需要保留不确定性。"

        should_refuse_original = should_refuse
        hard_refusal_disabled_by_round8 = False
        if (
            self.round8_config.enable_round8_policy
            and self.round8_config.disable_comparison_single_doc_hard_refusal
            and hard_refusal_reason == "comparison_single_doc"
        ):
            should_refuse = False
            hard_refusal_disabled_by_round8 = True

        claim_fallback_enabled = (
            self.round8_config.enable_round8_policy
            and self.round8_config.enable_claim_fallback
            and self.round8_config.enable_partial_answer
        )
        evidence_units: list[dict[str, Any]] = []
        recovered_by_claim_fallback = False
        supported_claim_count = 0
        fallback_method = ""
        support_pack: list[dict[str, Any]] = []
        if claim_fallback_enabled and should_refuse_original and should_refuse:
            fallback_method = "extractive_keyword_entity_parent_score"
            evidence_units = _extract_evidence_units(question, top_chunks, analysis, self.round8_config)
            supported_claim_count = len(evidence_units)
            support_pack = build_support_pack(top_chunks, evidence_units, top_chunks)
            if (
                citation_count > 0
                and _claim_fallback_satisfies_route(supported_claim_count, analysis)
                and len(support_pack) >= 1
                and _has_citable_support(support_pack)
            ):
                should_refuse = False
                recovered_by_claim_fallback = True
                partial_only = True
            elif _claim_fallback_satisfies_route(supported_claim_count, analysis):
                reason = _NO_EVIDENCE_REFUSAL
                refusal_reason = "fallback_without_citable_support"
                fallback_guardrail_failed = True
        if not support_pack:
            support_pack = build_support_pack(top_chunks, evidence_units, top_chunks)

        empty_support_pack_guardrail_triggered = False
        if not support_pack:
            should_refuse = True
            partial_only = False
            reason = _NO_EVIDENCE_REFUSAL
            refusal_reason = "empty_support_refuse"
            empty_support_pack_guardrail_triggered = True

        (
            covered_branch_count,
            missing_branch_count,
            missing_branch_labels,
            branch_support_pack_counts,
            branch_citable_quote_counts,
        ) = _comparison_branch_support(question, support_pack, analysis)
        candidate_support_pack_count = len(support_pack)
        limited_partial_compare_triggered = False
        allow_partial_coverage = 1
        has_covered_branch_citable_support = any(count >= 1 for count in branch_citable_quote_counts.values())
        if (
            self.round8_config.enable_round8_policy
            and self.round8_config.enable_partial_answer
            and analysis
            and analysis.intent == QueryIntent.COMPARISON
            and candidate_support_pack_count >= 1
            and covered_branch_count >= allow_partial_coverage
            and has_covered_branch_citable_support
            and missing_branch_count > 0
        ):
            should_refuse = False
            partial_only = False
            limited_partial_compare_triggered = True
            reason = "当前检索结果只覆盖了部分比较分支。"
            refusal_reason = ""
        elif (
            analysis
            and analysis.intent == QueryIntent.COMPARISON
            and candidate_support_pack_count == 0
        ):
            should_refuse = True
            partial_only = False
            reason = _NO_EVIDENCE_REFUSAL
            refusal_reason = "empty_support_refuse"
        elif (
            analysis
            and analysis.intent == QueryIntent.COMPARISON
            and candidate_support_pack_count > 0
            and covered_branch_count == 0
        ):
            should_refuse = True
            partial_only = False
            reason = "比较问题的所有分支都缺少可引用证据，无法完成比较。"
            refusal_reason = "missing_all_branches_refuse"
        elif (
            analysis
            and analysis.intent == QueryIntent.COMPARISON
            and should_refuse
            and covered_branch_count >= allow_partial_coverage
            and has_covered_branch_citable_support
        ):
            refusal_reason = "covered_branch_but_refuse"
            covered_branch_but_refuse = True

        citation_budget = _citation_budget_for_intent(intent)
        support_pack_doc_ids = _unique_str_values(item.get("doc_id") for item in support_pack)
        support_pack_chunk_ids = _unique_str_values(item.get("chunk_id") for item in support_pack)

        final_answer_mode = _final_answer_mode(
            should_refuse,
            partial_only,
            recovered_by_claim_fallback,
            analysis,
            limited_partial_compare_triggered=limited_partial_compare_triggered,
        )

        return EvidenceAssessment(
            level=level,
            citation_count=citation_count,
            unique_doc_count=unique_doc_count,
            score_strength=round(score_strength, 4),
            keyword_overlap=round(keyword_overlap, 4),
            branch_coverage=round(branch_coverage, 4),
            negative_signal=round(negative_signal, 4),
            should_refuse=should_refuse,
            partial_only=partial_only,
            reason=reason,
            hard_refusal_reason=hard_refusal_reason,
            hard_refusal_disabled_by_round8=hard_refusal_disabled_by_round8,
            should_refuse_original=should_refuse_original,
            should_refuse_final=should_refuse,
            final_answer_mode=final_answer_mode,
            claim_fallback_enabled=claim_fallback_enabled,
            fallback_method=fallback_method,
            evidence_unit_count=len(evidence_units),
            supported_claim_count=supported_claim_count,
            recovered_by_claim_fallback=recovered_by_claim_fallback,
            evidence_units=evidence_units,
            support_pack=support_pack,
            support_pack_count=len(support_pack),
            candidate_support_pack_count=candidate_support_pack_count,
            support_pack_doc_ids=support_pack_doc_ids,
            support_pack_chunk_ids=support_pack_chunk_ids,
            empty_support_pack_guardrail_triggered=empty_support_pack_guardrail_triggered,
            fallback_guardrail_failed=fallback_guardrail_failed,
            refusal_reason=refusal_reason,
            limited_partial_compare_triggered=limited_partial_compare_triggered,
            covered_branch_count=covered_branch_count,
            missing_branch_count=missing_branch_count,
            missing_branch_labels=missing_branch_labels,
            missing_branch_disclosed=limited_partial_compare_triggered and missing_branch_count > 0,
            unsupported_branch_claim_count=0,
            unsupported_branch_claims=[],
            covered_branch_but_refuse=covered_branch_but_refuse,
            branch_support_pack_counts=branch_support_pack_counts,
            branch_citable_quote_counts=branch_citable_quote_counts,
            citation_budget=citation_budget,
            citation_count_before_budget=len(support_pack),
        )

    # ------------------------------------------------------------------
    # Citation building
    # ------------------------------------------------------------------
    def build_citations(self, chunks: list[RetrievedChunk], assessment: EvidenceAssessment | None = None) -> list[Citation]:
        if assessment and assessment.support_pack:
            selected_pack = _apply_citation_budget(assessment.support_pack, assessment)
            assessment.citation_count_before_budget = len(assessment.support_pack)
            assessment.citation_count_after_budget = len(selected_pack)
            assessment.citation_limit_applied = len(selected_pack) < len(assessment.support_pack)
            citations: list[Citation] = []
            for item in selected_pack:
                citations.append(
                    Citation(
                        chunk_id=str(item.get("chunk_id") or ""),
                        doc_id=str(item.get("doc_id") or ""),
                        title=str(item.get("title") or item.get("doc_id") or ""),
                        source_file=str(item.get("source_file") or ""),
                        section=str(item.get("section") or ""),
                        page_start=item.get("page_start"),
                        page_end=item.get("page_end"),
                        score=float(item.get("score") or 0.0),
                        quote=_compress_text(str(item.get("quote") or ""), limit=1200),
                    )
                )
            return citations
        selected_chunks = chunks[:3]
        citations: list[Citation] = []
        for chunk in selected_chunks:
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    source_file=chunk.source_file,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    score=chunk.rerank_score or chunk.vector_score,
                    quote=_compress_text(chunk.text, limit=1200),
                )
            )
        return citations

    def validate_generated_answer(
        self,
        answer: str,
        citations: list[Citation],
        assessment: EvidenceAssessment,
    ) -> str:
        answer = _sanitize_missing_branch_claims(answer, assessment)
        if assessment.final_answer_mode in {"full", "partial", "concise", "limited_partial_compare"} and len(citations) == 0:
            assessment.should_refuse = True
            assessment.should_refuse_final = True
            assessment.partial_only = False
            assessment.final_answer_mode = "refuse"
            assessment.reason = _NO_EVIDENCE_REFUSAL
            assessment.refusal_reason = "post_generation_zero_citation_refuse"
            assessment.zero_citation_guardrail_triggered = True
            assessment.citation_count_after_budget = 0
            return _NO_EVIDENCE_REFUSAL
        return answer


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compress_text(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _build_refusal_answer(assessment: EvidenceAssessment) -> str:
    if assessment.refusal_reason in {
        "empty_support_refuse",
        "missing_all_branches_refuse",
        "fallback_without_citable_support",
        "post_generation_zero_citation_refuse",
    }:
        return _NO_EVIDENCE_REFUSAL
    if assessment.reason == _NO_EVIDENCE_REFUSAL:
        return _NO_EVIDENCE_REFUSAL
    return f"证据不足。{assessment.reason}"


def _build_limited_answer(
    question: str,
    assessment: EvidenceAssessment,
) -> str:
    del question
    support_pack = assessment.support_pack[:3]
    if not support_pack:
        return _build_refusal_answer(assessment)
    snippets = []
    for index, item in enumerate(support_pack, start=1):
        title = str(item.get("title") or item.get("doc_id") or "文献")
        quote = _compress_text(str(item.get("quote") or ""), 140)
        snippets.append(f"[{index}] {title}：{quote}")
    detail = "；".join(snippets)
    return f"证据有限。{assessment.reason} 当前只能确认：{detail}。"


def _build_limited_partial_compare_answer(assessment: EvidenceAssessment) -> str:
    support_pack = assessment.support_pack
    if not support_pack:
        return _build_refusal_answer(assessment)

    covered_labels = [
        label for label, count in assessment.branch_citable_quote_counts.items()
        if count >= 1
    ] or [
        label for label, count in assessment.branch_support_pack_counts.items()
        if count >= 1
    ] or ["已覆盖分支"]
    covered_items = _branch_evidence_map(assessment)
    conclusion_parts: list[str] = []
    lines = ["有限比较结论：", f"当前证据只覆盖了 {'、'.join(covered_labels)}。"]
    for label in covered_labels[:2]:
        item = (covered_items.get(label) or support_pack[:1])[0]
        conclusion_parts.append(f"{label} 的已检索证据显示 {_compress_text(str(item.get('quote') or ''), 72)}[{_support_citation_marker(item)}]")
    lines.append(f"可以确认：{'；'.join(conclusion_parts)}。")
    if assessment.missing_branch_labels:
        lines.append(
            f"{'、'.join(assessment.missing_branch_labels)} 当前没有足够可引用证据，因此不能比较其机制、产量、优势或差异。"
        )
    lines.extend(["", "证据："])
    for label in covered_labels:
        item = (covered_items.get(label) or support_pack[:1])[0]
        lines.append(
            f"- {label}: {_compress_text(str(item.get('quote') or ''), 120)} [{_support_citation_marker(item)}]"
        )
    if assessment.missing_branch_labels:
        lines.extend(["", "证据限制：", f"- 未覆盖：{'、'.join(assessment.missing_branch_labels)}"])
        assessment.missing_branch_disclosed = True
    return "\n".join(lines)


def _build_supported_answer(assessment: EvidenceAssessment) -> str:
    support_pack = assessment.support_pack[:3]
    if not support_pack:
        return _NO_EVIDENCE_REFUSAL
    lead = "基于当前知识库证据，"
    summary_bits = []
    for index, item in enumerate(support_pack, start=1):
        section = str(item.get("section") or "Full Text")
        title = str(item.get("title") or item.get("doc_id") or "文献")
        summary_bits.append(f"{title}（{section}）指出：{_compress_text(str(item.get('quote') or ''))}[{index}]")
    return f"{lead}{'；'.join(summary_bits)}。"


def _build_fallback_answer(question: str, assessment: EvidenceAssessment) -> str:
    del question
    support_pack = assessment.support_pack[:5]
    if not support_pack:
        return _build_refusal_answer(assessment)

    if assessment.final_answer_mode == "concise":
        conclusion_lead = "基于当前检索到的证据，可以得出以下结论："
        evidence_lead = "支撑证据："
    else:
        conclusion_lead = "基于当前检索到的部分证据，可以得出如下有限结论："
        evidence_lead = "支撑证据（仅覆盖部分方面）："

    doc_ids = sorted({str(item.get("doc_id") or "") for item in support_pack if item.get("doc_id")})
    doc_str = "、".join(doc_ids) if doc_ids else "已检索到的文献"

    lines = [conclusion_lead, ""]
    lines.append(evidence_lead)
    for index, item in enumerate(support_pack, start=1):
        text = _compress_text(str(item.get("quote") or ""), limit=220)
        source = str(item.get("doc_id") or item.get("chunk_id") or "").strip()
        suffix = f"（来源：{source}）" if source else ""
        lines.append(f"  {index}. {text}[{index}]{suffix}")
    lines.append("")
    lines.append(f"证据限制：以上内容仅来自 {doc_str} 中已命中的证据句。未被证据覆盖的比较对象、细节或结论不做推断和补全。")
    return "\n".join(lines)


def _branch_evidence_map(assessment: EvidenceAssessment) -> dict[str, list[dict[str, Any]]]:
    branch_map: dict[str, list[dict[str, Any]]] = {}
    for item in assessment.support_pack:
        for label in item.get("branch_labels") or []:
            branch_map.setdefault(str(label), []).append(item)
    for items in branch_map.values():
        items.sort(key=lambda entry: float(entry.get("score") or 0.0), reverse=True)
    return branch_map


def _support_citation_marker(item: dict[str, Any]) -> str:
    return str(item.get("doc_id") or item.get("source_file") or "citation")


def _sanitize_missing_branch_claims(answer: str, assessment: EvidenceAssessment) -> str:
    if not answer or not assessment.missing_branch_labels:
        return answer

    lines = answer.splitlines() or [answer]
    allowed_tokens = ("证据不足", "未覆盖", "不能推断", "没有足够可引用证据", "未提供", "未检索到", "证据限制")
    unsupported: list[str] = []
    rewritten: list[str] = []
    seen_missing_labels: set[str] = set()

    for raw_line in lines:
        line = raw_line
        for label in assessment.missing_branch_labels:
            if label not in raw_line:
                continue
            if any(token in raw_line for token in allowed_tokens):
                seen_missing_labels.add(label)
                continue
            unsupported.append(label)
            line = f"{label} 当前没有足够可引用证据，不能推断其机制、产量、优势或差异。"
            seen_missing_labels.add(label)
        rewritten.append(line)

    unsupported = _unique_str_values(unsupported)
    if assessment.missing_branch_labels and len(seen_missing_labels) < len(assessment.missing_branch_labels):
        missing_text = "、".join(label for label in assessment.missing_branch_labels if label not in seen_missing_labels)
        rewritten.extend(["", "证据限制：", f"- 未覆盖：{missing_text}"])
        seen_missing_labels.update(label for label in assessment.missing_branch_labels if label not in seen_missing_labels)

    normalized_answer = "\n".join(rewritten)
    assessment.unsupported_branch_claims = []
    assessment.unsupported_branch_claim_count = 0
    assessment.missing_branch_disclosed = all(label in seen_missing_labels for label in assessment.missing_branch_labels)
    return normalized_answer


def _comparison_branch_support(
    question: str,
    support_pack: list[dict[str, Any]],
    analysis: QueryAnalysis | None,
) -> tuple[int, int, list[str], dict[str, int], dict[str, int]]:
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return 0, 0, [], {}, {}
    branch_labels = _infer_comparison_branches(question)
    if not branch_labels:
        target_count = _infer_branch_target_count(question)
        covered_generic = min(
            target_count,
            len({str(item.get("doc_id") or "") for item in support_pack if item.get("doc_id")}),
        )
        missing_count = max(target_count - covered_generic, 0)
        missing_labels = [f"未覆盖分支 {index}" for index in range(1, missing_count + 1)]
        return covered_generic, missing_count, missing_labels, {}, {}

    branch_support_pack_counts: dict[str, int] = {label: 0 for label in branch_labels}
    branch_citable_quote_counts: dict[str, int] = {label: 0 for label in branch_labels}
    for item in support_pack:
        haystack = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("section") or ""),
                str(item.get("quote") or ""),
            ]
        )
        matched_labels = []
        for label in branch_labels:
            if _branch_matches_text(label, haystack):
                branch_support_pack_counts[label] += 1
                if item.get("doc_id") and item.get("source_file") and item.get("quote"):
                    branch_citable_quote_counts[label] += 1
                matched_labels.append(label)
        if matched_labels:
            item["branch_labels"] = matched_labels

    if not any(count > 0 for count in branch_support_pack_counts.values()) and support_pack:
        best_label = ""
        best_score = 0.0
        for label in branch_labels:
            label_score = max(_branch_match_score(label, " ".join([
                str(item.get("title") or ""),
                str(item.get("section") or ""),
                str(item.get("quote") or ""),
            ])) for item in support_pack)
            if label_score > best_score:
                best_score = label_score
                best_label = label
        if best_label and best_score >= 0.2:
            branch_support_pack_counts[best_label] = len(support_pack)
            branch_citable_quote_counts[best_label] = len(
                [item for item in support_pack if item.get("doc_id") and item.get("source_file") and item.get("quote")]
            )
            for item in support_pack:
                item["branch_labels"] = [best_label]

    covered_labels = [label for label, count in branch_support_pack_counts.items() if count > 0]
    missing_labels = [label for label, count in branch_support_pack_counts.items() if count == 0]
    return len(covered_labels), len(missing_labels), missing_labels, {
        label: count for label, count in branch_support_pack_counts.items()
    }, {
        label: count for label, count in branch_citable_quote_counts.items()
    }


def _infer_branch_target_count(question: str) -> int:
    if "三篇" in question or "三类" in question or "三种" in question:
        return 3
    if "两类" in question or "两种" in question or "两篇" in question:
        return 2
    return 2


def _infer_comparison_branches(question: str) -> list[str]:
    prompt = re.split(r"[。？！?]", question, maxsplit=1)[0]
    if "：" in prompt or ":" in prompt:
        tail = re.split(r"[:：]", prompt, maxsplit=1)[1]
        tail = re.sub(r"(一类是|另一类是|第三类是|第三种是|第一类是|第二类是)", "|", tail)
        tail = re.sub(r"(以及|以及基于|及)", "、", tail)
        candidates = [
            part.strip(" ，,。；;")
            for part in re.split(r"[、，,|]", tail)
            if part.strip(" ，,。；;")
        ]
        return _dedupe_nontrivial_branches(candidates)

    match = re.search(
        r"(?:比较文库中|文库中|比较)?\s*(.+?)(?:和|与)(.+?)(?:在|作为|两种|两类|的|策略|工作|研究|表现|优势|角色|功能|结果|异同|差异)",
        prompt,
    )
    if match:
        return _dedupe_nontrivial_branches([match.group(1), match.group(2)])
    return []


def _dedupe_nontrivial_branches(candidates: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = re.sub(r"^(比较文库中|文库中|比较|一类是|另一类是)\s*", "", item).strip()
        text = re.sub(r"(请说明.*|请比较.*)$", "", text).strip(" ：:，,。；;")
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned[:4]


def _branch_matches_text(label: str, text: str) -> bool:
    if label.lower() in text.lower():
        return True
    return _branch_match_score(label, text) >= 0.45


def _branch_match_score(label: str, text: str) -> float:
    entity_overlap = _entity_overlap(label, text)
    term_overlap = _term_overlap(label, text)
    return max(entity_overlap, term_overlap)


def build_support_pack(
    top_contexts: list[RetrievedChunk],
    evidence_units: list[dict[str, Any]],
    citation_candidates: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    chunk_map = {chunk.chunk_id: chunk for chunk in citation_candidates}
    support_pack: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for unit in evidence_units:
        chunk = chunk_map.get(str(unit.get("chunk_id") or ""))
        source_file = str(unit.get("source_file") or (chunk.source_file if chunk else ""))
        section = str(unit.get("section") or (chunk.section if chunk else ""))
        title = str(unit.get("title") or (chunk.title if chunk else unit.get("doc_id") or ""))
        quote = str(unit.get("text") or unit.get("quote") or "").strip()
        key = (str(unit.get("chunk_id") or ""), re.sub(r"\s+", " ", quote))
        if not quote or not source_file or not unit.get("doc_id") or not unit.get("chunk_id") or key in seen_keys:
            continue
        seen_keys.add(key)
        support_pack.append(
            {
                "doc_id": str(unit.get("doc_id") or ""),
                "source_file": source_file,
                "chunk_id": str(unit.get("chunk_id") or ""),
                "section": section,
                "quote": quote,
                "score": float(unit.get("score") or 0.0),
                "title": title,
                "page_start": chunk.page_start if chunk else None,
                "page_end": chunk.page_end if chunk else None,
                "support_type": "evidence_unit",
                "branch_labels": _support_branch_labels(str(unit.get("text") or quote)),
            }
        )

    for chunk in top_contexts[:5]:
        quote = _compress_text(chunk.text, limit=1200)
        key = (chunk.chunk_id, re.sub(r"\s+", " ", quote))
        if not quote or not chunk.doc_id or not chunk.source_file or key in seen_keys:
            continue
        seen_keys.add(key)
        support_pack.append(
            {
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "quote": quote,
                "score": _normalize_score(max(chunk.rerank_score, chunk.vector_score, chunk.fusion_score)),
                "title": chunk.title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "support_type": "retrieved_chunk",
                "branch_labels": _support_branch_labels(chunk.text),
            }
        )

    support_pack.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return support_pack


def _build_support_context(support_pack: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(support_pack, start=1):
        title = str(item.get("title") or item.get("doc_id") or "文献")
        section = str(item.get("section") or "Full Text")
        source_file = str(item.get("source_file") or "")
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        lines.append(f"[{index}] {title} | {section} | {source_file}\n{quote}")
    return "\n\n".join(lines)


def _has_citable_support(support_pack: list[dict[str, Any]]) -> bool:
    return any(
        item.get("doc_id") and item.get("chunk_id") and item.get("source_file") and item.get("quote")
        for item in support_pack
    )


def _unique_str_values(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _citation_budget_for_intent(intent: QueryIntent) -> dict[str, Any]:
    if intent == QueryIntent.FACTOID:
        return {"max_total": 4, "min_per_answer": 1}
    if intent == QueryIntent.SUMMARY:
        return {"max_total": 8, "min_per_key_claim": 1}
    if intent == QueryIntent.COMPARISON:
        return {"max_total": 10, "max_per_branch": 3, "min_per_branch": 1}
    return {"max_total": 6, "min_per_answer": 1}


def _support_branch_labels(text: str) -> list[str]:
    return sorted(_entity_terms(text) | _anchor_terms(text))[:8]


def _apply_citation_budget(
    support_pack: list[dict[str, Any]],
    assessment: EvidenceAssessment,
) -> list[dict[str, Any]]:
    budget = assessment.citation_budget or _citation_budget_for_intent(QueryIntent.UNKNOWN)
    max_total = int(budget.get("max_total") or len(support_pack) or 0)
    if len(support_pack) <= max_total:
        return support_pack

    if assessment.final_answer_mode == "refuse":
        return []

    if assessment.recovered_by_claim_fallback:
        extractive = [item for item in support_pack if item.get("support_type") == "evidence_unit"]
        if extractive:
            support_pack = extractive + [item for item in support_pack if item.get("support_type") != "evidence_unit"]

    if budget.get("min_per_branch") and assessment.final_answer_mode in {"full", "partial", "limited_partial_compare"}:
        selected: list[dict[str, Any]] = []
        used_keys: set[tuple[str, str]] = set()
        per_branch: dict[str, int] = {}
        for item in support_pack:
            branches = item.get("branch_labels") or []
            if not branches:
                continue
            for branch in branches:
                if per_branch.get(branch, 0) >= int(budget.get("max_per_branch") or max_total):
                    continue
                key = (str(item.get("chunk_id") or ""), str(item.get("quote") or ""))
                if key in used_keys:
                    continue
                used_keys.add(key)
                selected.append(item)
                per_branch[branch] = per_branch.get(branch, 0) + 1
                break
            if len(selected) >= max_total:
                return selected[:max_total]
        for item in support_pack:
            key = (str(item.get("chunk_id") or ""), str(item.get("quote") or ""))
            if key in used_keys:
                continue
            selected.append(item)
            used_keys.add(key)
            if len(selected) >= max_total:
                break
        return selected[:max_total]

    return support_pack[:max_total]


def _normalize_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return max(0.0, min(score / 5.0, 1.0))


def _final_answer_mode(
    should_refuse: bool,
    partial_only: bool,
    recovered_by_claim_fallback: bool,
    analysis: QueryAnalysis | None,
    limited_partial_compare_triggered: bool = False,
) -> str:
    if should_refuse:
        return "refuse"
    if limited_partial_compare_triggered:
        return "limited_partial_compare"
    if recovered_by_claim_fallback:
        if analysis and analysis.intent == QueryIntent.FACTOID:
            return "concise"
        return "partial"
    if partial_only:
        return "partial"
    return "full"


def _claim_fallback_satisfies_route(
    supported_claim_count: int,
    analysis: QueryAnalysis | None,
) -> bool:
    intent = analysis.intent if analysis else QueryIntent.UNKNOWN
    if intent == QueryIntent.FACTOID:
        return supported_claim_count >= 1
    if intent in {QueryIntent.SUMMARY, QueryIntent.COMPARISON, QueryIntent.EXPERIMENT}:
        return supported_claim_count >= 2
    return supported_claim_count >= 2


def _round8_route_thresholds(
    analysis: QueryAnalysis | None,
    round8_config: Round8PolicyConfig | None = None,
) -> tuple[float, float, float, float, float]:
    config = round8_config or Round8PolicyConfig()
    use_route_thresholds = config.enable_round8_policy and config.enable_route_specific_thresholds
    ep = config.evidence_policy
    if not use_route_thresholds:
        return 0.38, 0.50, 0.30, 0.35, 0.55

    intent = analysis.intent if analysis else QueryIntent.UNKNOWN
    if intent == QueryIntent.FACTOID:
        return (
            ep.score_strength_factoid,
            ep.partial_boundary_factoid,
            ep.evidence_unit_threshold_factoid,
            ep.evidence_signal_weak_threshold,
            ep.comparison_branch_coverage_threshold,
        )
    if intent == QueryIntent.SUMMARY:
        return (
            ep.score_strength_summary,
            ep.partial_boundary_summary,
            ep.evidence_unit_threshold_summary,
            ep.evidence_signal_weak_threshold,
            ep.comparison_branch_coverage_threshold,
        )
    if intent == QueryIntent.COMPARISON:
        return (
            ep.score_strength_comparison,
            ep.partial_boundary_comparison,
            ep.evidence_unit_threshold_comparison,
            ep.evidence_signal_weak_threshold,
            ep.comparison_branch_coverage_threshold,
        )
    return (
        ep.score_strength_factoid,
        ep.partial_boundary_factoid,
        ep.evidence_unit_threshold_factoid,
        ep.evidence_signal_weak_threshold,
        ep.comparison_branch_coverage_threshold,
    )


def _extract_evidence_units(
    question: str,
    chunks: list[RetrievedChunk],
    analysis: QueryAnalysis | None,
    round8_config: Round8PolicyConfig | None = None,
) -> list[dict[str, Any]]:
    _, _, threshold, _, _ = _round8_route_thresholds(analysis, round8_config)
    scored: list[dict[str, Any]] = []
    for chunk in chunks[:5]:
        parent_score = _normalize_score(max(chunk.rerank_score, chunk.vector_score, chunk.fusion_score))
        for sentence in _split_evidence_sentences(chunk.text):
            lexical = _term_overlap(question, sentence)
            entity = _entity_overlap(question, sentence)
            score = (0.40 * lexical) + (0.35 * entity) + (0.25 * parent_score)
            if score < threshold:
                continue
            scored.append(
                {
                    "text": sentence,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "section": chunk.section,
                    "title": chunk.title,
                    "score": round(score, 4),
                    "lexical_overlap": round(lexical, 4),
                    "entity_overlap": round(entity, 4),
                    "parent_chunk_score": round(parent_score, 4),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    deduped: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    seen_chunks: set[str] = set()
    for item in scored:
        normalized = re.sub(r"\W+", "", str(item["text"]).lower())[:160]
        if normalized in seen_texts:
            continue
        chunk_id = str(item["chunk_id"])
        if chunk_id in seen_chunks and len(deduped) >= 2:
            continue
        seen_texts.add(normalized)
        seen_chunks.add(chunk_id)
        deduped.append(item)
        if len(deduped) >= 5:
            break
    return deduped


def _split_evidence_sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    pieces = [
        item.strip(" ;；")
        for item in re.split(r"(?<=[。！？!?])\s+|(?<=[.!?])\s+|[；;]\s*", cleaned)
        if item.strip(" ;；")
    ]
    sentences: list[str] = []
    for piece in pieces:
        if len(piece) <= 360:
            sentences.append(piece)
            continue
        clauses = [item.strip(" ,，") for item in re.split(r"(?<=,)\s+|，\s*", piece) if item.strip(" ,，")]
        buffer = ""
        for clause in clauses:
            candidate = f"{buffer}，{clause}" if buffer else clause
            if len(candidate) <= 300:
                buffer = candidate
            else:
                if buffer:
                    sentences.append(buffer)
                buffer = clause
        if buffer:
            sentences.append(buffer)
    return [sentence for sentence in sentences if len(sentence) >= 24][:80]


def _term_overlap(left: str, right: str) -> float:
    left_terms = _anchor_terms(left)
    if not left_terms:
        return 0.0
    right_terms = _anchor_terms(right)
    return len(left_terms & right_terms) / len(left_terms)


def _entity_overlap(left: str, right: str) -> float:
    left_entities = _entity_terms(left)
    if not left_entities:
        return _term_overlap(left, right)
    right_entities = _entity_terms(right)
    return len(left_entities & right_entities) / len(left_entities)


def _text_terms(text: str) -> set[str]:
    normalized = _normalize_text_for_terms(text)
    ascii_terms = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_+'′-]*", normalized)
        if len(token) >= 2 and token not in _TERM_STOPWORDS
    }
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_bigrams = {
        "".join(cjk_chars[index:index + 2])
        for index in range(max(len(cjk_chars) - 1, 0))
    }
    return ascii_terms | cjk_bigrams


def _anchor_terms(text: str) -> set[str]:
    normalized = _normalize_text_for_terms(text)
    ascii_terms = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_+'′-]*", normalized)
        if len(token) >= 2 and token not in _TERM_STOPWORDS
    }
    # When Chinese query terms have been normalized into English anchors, prefer
    # those anchors over noisy CJK bigrams for evidence-unit scoring.
    if ascii_terms:
        return ascii_terms
    return _text_terms(text)


def _entity_terms(text: str) -> set[str]:
    normalized = _normalize_text_for_terms(text)
    entities = set(
        re.findall(
            r"\b[a-z]{1,10}\d{1,6}[a-z0-9_+'′-]*\b|"
            r"\b\d+(?:\.\d+)?\s*(?:%|g/l|mg/l|mg|mmol|um|μm|fold|h|copy/μl|copies/μl)\b|"
            r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b|"
            r"\b[a-z]{2,24}ase\b",
            normalized,
        )
    )
    cjk_entities = re.findall(
        r"[\u4e00-\u9fff]{2,}(?:酶|酸|菌|株|通路|路径|启动子|表达盒|产量|滴度|前体|分泌|调控|检测|传感器|蛋白)",
        normalized,
    )
    entities.update(cjk_entities)
    return {entity.strip() for entity in entities if entity.strip() and entity not in _TERM_STOPWORDS}


def _normalize_text_for_terms(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("’", "'").replace("′", "'")
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    for source, target in _TERM_TRANSLATION_MAP.items():
        if source in normalized:
            normalized = f"{normalized} {target}"
    normalized = re.sub(r"\b2\s*-\s*fl\b", "2'-fl", normalized)
    normalized = re.sub(r"\b6\s*-\s*sl\b", "6'-sl", normalized)
    return normalized


_TERM_TRANSLATION_MAP = {
    "细菌": "bacterial bacteria",
    "摄取": "uptake entrance transport",
    "外排": "efflux export",
    "芳香化合物": "aromatic compounds",
    "膜转运": "membrane transporter transporters",
    "转运家族": "transporter family families",
    "转运体": "transporter transporters",
    "通道": "channels pores",
    "骨桥蛋白": "osteopontin opn",
    "天然矿化调控": "natural mineralization biomineralization mineralization",
    "半理性设计": "semi-rational design semirational engineering",
    "设计目标": "design goal design target objective",
    "磷酸化位点": "phosphorylation sites phosphorylated residues",
    "磷酸化残基": "phosphorylated residues serine threonine",
    "簇状": "cluster clustered clusters",
    "骨矿化": "bone mineralization mineralized",
    "羟基磷灰石": "hydroxyapatite crystallinity crystal",
    "毕赤酵母": "pichia pastoris",
    "甲醇": "methanol",
    "苹果酸": "malic acid",
    "代谢通量": "metabolic flux",
    "重分配": "redistribution reallocating reallocation",
    "生产": "production produce",
    "环状排列": "circularly permuted circular permutation",
    "环状": "circularly permuted circular",
    "荧光蛋白": "fluorescent protein autofluorescent proteins gfp",
    "指示器": "indicator reporter probe gefi",
    "设计原理": "design principle",
    "配体检测": "ligand detection sensing",
    "检测": "detection sensing assay",
    "改善骨质疏松": "osteoporosis bone loss bone microstructure",
    "骨质疏松": "osteoporosis bone loss",
    "体外发酵": "in vitro fermentation",
    "肠道菌群": "gut microflora microbiota bacterial communities",
    "健康效应": "health effect",
    "双歧杆菌": "bifidobacterium bifidobacteria",
    "人乳糖缀合物": "human milk glycoconjugates",
    "人乳寡糖": "hmo human milk oligosaccharides",
    "糖蛋白": "glycoprotein glycoproteins",
    "分泌信号": "secretion signal signal peptide leader sequence",
    "干扰素": "interferon",
    "乳铁蛋白": "lactoferrin",
    "唾液酸化": "sialylated sialylation",
    "非唾液酸化": "non-sialylated non sialylated",
    "免疫功能": "immune function immunological",
    "抗体": "antibody antibodies",
    "优势": "advantages",
}


_TERM_STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "these",
    "those",
    "study",
    "result",
    "results",
    "paper",
    "文库",
    "比较",
    "总结",
    "说明",
    "当前",
    "研究",
}


def _keyword_overlap(question: str, chunks: list[RetrievedChunk]) -> float:
    q_tokens = {token.lower() for token in _TOKEN_RE.findall(question) if len(token.strip()) > 1}
    if not q_tokens:
        return 0.0
    evidence_tokens = {
        token.lower()
        for chunk in chunks
        for token in _TOKEN_RE.findall(" ".join([chunk.title or "", chunk.section or "", chunk.text[:800]]))
        if len(token.strip()) > 1
    }
    return len(q_tokens & evidence_tokens) / max(len(q_tokens), 1)


def _evidence_feature_signal(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    signals: list[float] = []
    for chunk in chunks:
        features = chunk.metadata.get("evidence_features") or {}
        if features:
            feature_hits = sum(
                1
                for key in ("numeric", "result", "definition")
                if bool(features.get(key))
            )
            section_bonus = float(features.get("section_bonus") or 0.0)
            signals.append(min(1.0, feature_hits / 3 + section_bonus))
            continue
        text = " ".join([chunk.section or "", chunk.text[:1200]]).lower()
        feature_hits = 0
        if re.search(r"\d+", text):
            feature_hits += 1
        if any(term in text for term in ("yield", "production", "titer", "result", "increase", "decrease")):
            feature_hits += 1
        if any(term in text for term in (" is ", "defined as", "acts as", "functions as")):
            feature_hits += 1
        signals.append(feature_hits / 3)
    return sum(signals) / len(signals)


def _branch_coverage(question: str, chunks: list[RetrievedChunk], analysis: QueryAnalysis | None) -> float:
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return 1.0
    lowered = question.lower()
    branches = []
    if "6′-sl" in lowered or "6'-sl" in lowered:
        branches.append(("6′-sl", "cmp-neu5ac", "sialyltransferase"))
    if "2′-fl" in lowered or "2'-fl" in lowered:
        branches.append(("2′-fl", "gdp-l-fucose", "fucosyltransferase"))
    if "wcfb" in lowered:
        branches.append(("wcfb",))
    if "salvage" in lowered or "补料" in question:
        branches.append(("salvage", "gdp-l-fucose"))
    if "染色体整合" in question or "chromosomal integration" in lowered:
        branches.append(("chromosomally", "integration", "染色体整合"))
    if not branches:
        return 1.0
    covered = 0
    for branch in branches:
        if any(any(term in " ".join([chunk.title.lower(), chunk.text[:600].lower()]) for term in branch) for chunk in chunks):
            covered += 1
    return covered / len(branches)


def _negative_signal(question: str, chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    negatives = 0
    for chunk in chunks:
        haystack = " ".join([chunk.title.lower(), chunk.text[:400].lower()])
        if any(term in haystack for term in ("review", "progress", "covid", "sars", "protective effect", "inflammation")):
            negatives += 1
    return negatives / len(chunks)



_BIBLIO_RE = re.compile(
    r"(?:et\s+al\.)|"
    r"(?:\b(?:J|N)\.\s*(?:Engl|Am|Biol)\.)|"
    r"(?:\b\d{4}\b.*\b\d{1,4}[-–]\d{1,4}\b)|"
    r"(?:卷\d+)|"
    r"(?:Vol\.\s*\d+)",
)


def _is_bibliographic_claim(fact: str) -> bool:
    hits = len(_BIBLIO_RE.findall(fact))
    words = len(fact.split())
    return hits >= 2 or (hits >= 1 and words < 40)


def _validate_claims(
    claims: list[dict[str, Any]],
    question: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not claims:
        return [], []

    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for claim in claims:
        if claim.get("status") != "supported":
            rejected.append(claim)
            continue

        sources = claim.get("source", [])
        if not sources:
            rejected.append(claim)
            continue

        if _is_bibliographic_claim(claim.get("fact", "")):
            logger.info("Rejecting bibliographic claim: %s", claim.get("fact", "")[:80])
            rejected.append(claim)
            continue

        validated.append(claim)

    return validated, rejected


def _phase_a_quality_gate(
    claims_data: dict[str, Any],
    question: str,
    chunks: list[RetrievedChunk],
) -> bool:
    """Return True if Phase A output indicates we should abstain."""
    if claims_data.get("relevant") is False:
        return True

    claims = claims_data.get("claims", [])
    supported = [c for c in claims if c.get("status") == "supported"]
    if not supported:
        return True

    with_source = [c for c in supported if c.get("source")]
    if not with_source:
        return True

    return False
