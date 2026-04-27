from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ...domain.config import GenerationConfig, ModelEndpointConfig
from ...infrastructure.clients.openai_compatible import OpenAICompatibleClient
from .models import AnswerPlan, SupportItem

_EVIDENCE_REF_RE = re.compile(r"\[(E\d+)\]")
_PARTIAL_OVERCLAIM_PATTERNS = (
    "完整回答如下",
    "可以完整回答",
    "可以确定文库中完整包含",
    "两者均已充分证明",
    "可以确认文库中完整包含",
    "完整比较",
    "完整回答",
    "充分证明",
    "均已充分支持",
    "可以确定",
    "完全支持",
    "文库已经证明",
    "明确证明两者",
    "无需额外证据",
    "已完整覆盖",
    "fully supported",
    "complete comparison",
    "conclusively proves",
    "sufficient evidence for all",
    "definitely confirms",
)
_EXISTENCE_OVERCLAIM_PATTERNS = (
    "文库中有该详细方案",
    "文库中有完整详细方案",
    "已提供完整方案",
    "可以确认包含",
    "可以确认文库中有",
)
_LIMITATION_TERMS = ("不足", "缺失", "无法", "不能完整比较", "不能逐分支", "不能确认", "证据限制")
_INDIRECT_LIMITATION_TERMS = ("间接", "有限比较", "不能完整比较", "不能逐分支", "证据限制")
_PARTIAL_LIMITATION_PATTERNS = (
    "有限比较",
    "有限回答",
    "只能支持部分",
    "当前证据只能",
    "当前文库只能",
    "在文库所支持的范围内",
    "证据不足以",
    "不能完整",
    "不能逐分支",
    "不能逐分支完整比较",
    "无法完整",
    "缺乏直接",
    "仅提供间接",
    "部分支持",
    "证据分布不均衡",
    "尚不能",
    "不能确认",
    "limited comparison",
    "partial support",
    "insufficient evidence",
    "indirect evidence",
    "cannot fully",
    "not enough evidence",
)
_NEGATING_PREFIXES = ("不能", "无法", "尚不能", "不可", "不应", "not ", "cannot ", "can't ")


@dataclass
class SynthesisResult:
    answer: str
    used_qwen: bool
    fallback_used: bool
    fallback_reason: str
    raw_output_preview: str
    validation_flags: list[str] = field(default_factory=list)
    validation_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QwenSynthesizer:
    def __init__(
        self,
        llm_config: ModelEndpointConfig | None = None,
        *,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        resolved = llm_config or ModelEndpointConfig(model_name="qwen-plus")
        self.model_name = resolved.model_name
        self.temperature = resolved.temperature
        self.client = client or OpenAICompatibleClient(
            resolved.api_base,
            resolved.api_key,
            timeout_seconds=resolved.timeout_seconds,
        )

    def synthesize(
        self,
        question: str,
        plan: AnswerPlan,
        support_pack: list[SupportItem],
        extractive_answer: str,
        config: GenerationConfig,
        existence_guardrail: dict[str, Any] | None = None,
    ) -> SynthesisResult:
        if not config.v2_use_qwen_synthesis:
            return SynthesisResult(
                answer=extractive_answer,
                used_qwen=False,
                fallback_used=False,
                fallback_reason="disabled",
                raw_output_preview="",
            )
        if plan.mode == "refuse" or not support_pack:
            return SynthesisResult(
                answer=extractive_answer,
                used_qwen=False,
                fallback_used=True,
                fallback_reason="refuse_or_empty_support",
                raw_output_preview="",
            )
        if not self.client.is_enabled():
            return SynthesisResult(
                answer=extractive_answer,
                used_qwen=False,
                fallback_used=True,
                fallback_reason="client_not_configured",
                raw_output_preview="",
            )

        original_timeout = getattr(self.client, "timeout_seconds", None)
        if hasattr(self.client, "timeout_seconds"):
            self.client.timeout_seconds = config.v2_qwen_synthesis_timeout_seconds
        try:
            raw_output = self.client.chat_completion(
                model=self.model_name,
                messages=_build_messages(
                    question=question,
                    plan=plan,
                    support_pack=support_pack,
                    extractive_answer=extractive_answer,
                    config=config,
                    existence_guardrail=existence_guardrail or {},
                ),
                temperature=self.temperature,
            )
        except Exception:
            return SynthesisResult(
                answer=extractive_answer,
                used_qwen=False,
                fallback_used=True,
                fallback_reason="client_error",
                raw_output_preview="",
            )
        finally:
            if hasattr(self.client, "timeout_seconds"):
                self.client.timeout_seconds = original_timeout

        synthesized_answer = str(raw_output or "").strip()
        is_valid, validation_flags = validate_synthesized_answer(
            synthesized_answer,
            plan,
            support_pack,
            config,
            extractive_answer=extractive_answer,
            existence_guardrail=existence_guardrail or {},
        )
        validation_details = getattr(validate_synthesized_answer, "last_details", {})
        if not is_valid:
            return SynthesisResult(
                answer=extractive_answer,
                used_qwen=False,
                fallback_used=True,
                fallback_reason="validation_failed",
                raw_output_preview=synthesized_answer[:300],
                validation_flags=validation_flags,
                validation_details=validation_details,
            )

        return SynthesisResult(
            answer=synthesized_answer,
            used_qwen=True,
            fallback_used=False,
            fallback_reason="",
            raw_output_preview=synthesized_answer[:300],
            validation_flags=validation_flags,
            validation_details=validation_details,
        )


def validate_synthesized_answer(
    answer: str,
    plan: AnswerPlan,
    support_pack: list[SupportItem],
    config: GenerationConfig,
    *,
    extractive_answer: str,
    existence_guardrail: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    flags: list[str] = []
    details: dict[str, Any] = {}
    support_ids = {item.evidence_id for item in support_pack}
    output_refs = _extract_evidence_ids(answer)
    extractive_refs = _extract_evidence_ids(extractive_answer)
    comparison_coverage = plan.comparison_coverage
    allowed_ids = set((comparison_coverage.allowed_citation_evidence_ids if comparison_coverage else []) or [])
    exact_set_policy = not (
        plan.mode in {"full", "partial"}
        and comparison_coverage
        and comparison_coverage.parse_ok
        and allowed_ids
    )
    comparison_policy = "exact_set" if exact_set_policy else "comparison_allowed_subset"
    details["comparison_policy"] = comparison_policy

    invalid_refs = [ref for ref in output_refs if ref not in support_ids]
    if invalid_refs:
        flags.append("invalid_citation_ids")
        details["invalid_evidence_ids"] = invalid_refs
    disallowed_refs = [ref for ref in output_refs if ref not in allowed_ids] if not exact_set_policy else []
    if disallowed_refs:
        flags.append("comparison_disallowed_citation")
        details["disallowed_evidence_ids"] = disallowed_refs
    details["citation_subset_ok"] = not disallowed_refs if not exact_set_policy else True

    if len(answer) > config.v2_qwen_synthesis_max_output_chars:
        flags.append("output_too_long")

    if plan.mode != "refuse" and not output_refs:
        flags.append("missing_citations")

    if extractive_refs and not output_refs:
        flags.append("dropped_all_citations")
    if exact_set_policy and extractive_refs and set(output_refs) != set(extractive_refs):
        flags.append("citation_set_changed")
    if not exact_set_policy and comparison_coverage:
        direct_groups = [
            set(entry.evidence_ids)
            for entry in comparison_coverage.branch_evidence
            if entry.status == "direct" and entry.evidence_ids
        ]
        for group in direct_groups:
            if not (group & set(output_refs)):
                flags.append("missing_direct_branch_citation")
                break

    if plan.mode == "partial" and _find_non_negated_patterns(answer, _PARTIAL_OVERCLAIM_PATTERNS):
        flags.append("partial_overclaim")
    partial_limit_terms_found = [pattern for pattern in _PARTIAL_LIMITATION_PATTERNS if pattern in answer]
    partial_overclaim_terms_found = _find_non_negated_patterns(answer, _PARTIAL_OVERCLAIM_PATTERNS)
    details["partial_limit_terms_found"] = partial_limit_terms_found
    details["partial_overclaim_terms_found"] = partial_overclaim_terms_found
    if plan.mode == "partial":
        partial_tone_ok = bool(partial_limit_terms_found) and not partial_overclaim_terms_found
        if not partial_tone_ok:
            flags.append("partial_abstention_tone")
            details["partial_tone_decision"] = "fail"
        else:
            details["partial_tone_decision"] = "pass"
    else:
        details["partial_tone_decision"] = "not_applicable"

    if plan.reason == "existence_weak_support" and any(pattern in answer for pattern in _EXISTENCE_OVERCLAIM_PATTERNS):
        flags.append("existence_overclaim")

    requires_branch_disclosure = bool(plan.missing_branches)
    requires_indirect_disclosure = bool(
        comparison_coverage
        and any(entry.status == "indirect" for entry in comparison_coverage.branch_evidence)
    )
    if requires_branch_disclosure and not any(term in answer for term in _LIMITATION_TERMS):
        flags.append("missing_branch_limit_not_disclosed")
    if requires_indirect_disclosure and not any(term in answer for term in _INDIRECT_LIMITATION_TERMS):
        flags.append("indirect_branch_limit_not_disclosed")

    guardrail = existence_guardrail or {}
    if (
        plan.reason == "existence_weak_support"
        and guardrail.get("is_existence_question")
        and guardrail.get("support_status") == "weak"
        and "不能确认文库中包含" not in answer
    ):
        flags.append("missing_existence_weak_disclaimer")

    validate_synthesized_answer.last_details = details
    return not flags, flags


def _build_messages(
    *,
    question: str,
    plan: AnswerPlan,
    support_pack: list[SupportItem],
    extractive_answer: str,
    config: GenerationConfig,
    existence_guardrail: dict[str, Any],
) -> list[dict[str, str]]:
    evidence_blocks = []
    for item in support_pack:
        text = _compress_text(item.candidate.text, config.v2_qwen_synthesis_max_chars_per_evidence)
        evidence_blocks.append(
            f"[{item.evidence_id}] doc_id={item.candidate.doc_id}; section={item.candidate.section}\n{text}"
        )

    existence_lines: list[str] = []
    if existence_guardrail.get("is_existence_question"):
        target_terms = "、".join(existence_guardrail.get("target_terms") or []) or "目标资料"
        existence_lines.extend(
            [
                f"- existence_question: true",
                f"- support_status: {existence_guardrail.get('support_status') or 'unknown'}",
                f"- target_terms: {target_terms}",
                f"- support_reason: {existence_guardrail.get('support_reason') or ''}",
                f"- missing_core_terms: {'、'.join(existence_guardrail.get('missing_core_terms') or []) or '无'}",
            ]
        )

    limitation_line = "无"
    if plan.comparison_coverage and plan.comparison_coverage.branch_evidence:
        branch_limits = []
        for entry in plan.comparison_coverage.branch_evidence:
            if entry.status == "missing":
                branch_limits.append(f"{entry.branch}: 缺少直接证据")
            elif entry.status == "indirect":
                branch_limits.append(f"{entry.branch}: 仅间接支持")
        if branch_limits:
            limitation_line = "；".join(branch_limits)
    elif plan.missing_branches:
        limitation_line = f"未覆盖分支：{'、'.join(plan.missing_branches)}"
    elif plan.reason == "existence_weak_support":
        limitation_line = "当前文库只能提供间接或弱相关证据，不能确认文库中包含用户所要求的资料/数据/方案。"
    elif plan.mode == "partial":
        limitation_line = f"证据限制：{plan.reason or 'partial_support'}"

    user_prompt = "\n".join(
        [
            f"问题：{question}",
            f"答案模式：{plan.mode}",
            f"计划原因：{plan.reason}",
            f"已覆盖分支：{'、'.join(plan.covered_branches) or '无'}",
            f"未覆盖分支：{'、'.join(plan.missing_branches) or '无'}",
            f"允许范围：{'、'.join(plan.allowed_scope) or '无'}",
            f"证据限制：{limitation_line}",
            f"允许引用证据：{'、'.join(plan.comparison_coverage.allowed_citation_evidence_ids) if plan.comparison_coverage else 'extractive_answer 原有 [E#] 集合'}",
            "可引用证据：",
            "\n\n".join(evidence_blocks),
            "当前抽取式草稿：",
            extractive_answer,
            "existence_guardrail：",
            "\n".join(existence_lines) if existence_lines else "- existence_question: false",
            "请生成更自然的中文回答。必须保留引用编号 [E#]。",
        ]
    )

    return [
        {
            "role": "system",
            "content": (
                "你是合成生物学知识库问答助手。你只能改写和组织给定证据，不得新增事实。\n"
                "严格要求：\n"
                "1. 你只能基于【可引用证据】回答。\n"
                "2. 每个实质性结论必须带 [E#]。\n"
                "3. 只能使用列出的 [E#]，不得创造新编号。\n"
                "4. 不得使用文库外知识。\n"
                "5. 不得改变 answer_mode。\n"
                "6. 如果 mode=partial，必须明确说明证据限制。\n"
                "7. 只有在 existence_guardrail.is_existence_question=true 且 support_status=weak 时，才允许写“当前文库只能提供间接或弱相关证据，不能确认文库中包含用户所要求的资料/数据/方案”。\n"
                "8. 如果 existence_guardrail.is_existence_question=false，不要引入“文库中未提供/文库中没有/不能确认文库中包含”这类 existence 判断。\n"
                "9. 如果 covered_branches 为空但 missing_branches 非空，不能写成两个分支都已被证实；只能说当前证据不能逐分支完整对齐。\n"
                "10. 如果是 comparison 且给出了允许引用证据，只能使用允许集合内的 [E#]；不得新增其他 [E#]。\n"
                "11. 输出中文。\n"
                "12. 不要输出 JSON。\n"
                "13. 不要输出 markdown 表格，除非问题明确要求表格。\n"
                "14. 不要输出没有 citation 的事实句。"
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _compress_text(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _extract_evidence_ids(answer: str) -> list[str]:
    ordered: list[str] = []
    for match in _EVIDENCE_REF_RE.finditer(answer or ""):
        evidence_id = match.group(1)
        if evidence_id not in ordered:
            ordered.append(evidence_id)
    return ordered


def _find_non_negated_patterns(text: str, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(re.escape(pattern), text or "", flags=re.IGNORECASE):
            prefix = (text or "")[max(0, match.start() - 8) : match.start()].lower()
            if any(neg.lower() in prefix for neg in _NEGATING_PREFIXES):
                continue
            if pattern not in found:
                found.append(pattern)
            break
    return found
