from __future__ import annotations

from ...domain.schemas import QueryAnalysis, QueryIntent
from .guardrails import detect_existence_question
from .models import AnswerPlan, SupportItem


class ExtractiveAnswerBuilder:
    def build(
        self,
        question: str,
        analysis: QueryAnalysis,
        plan: AnswerPlan,
        support_pack: list[SupportItem],
    ) -> str:
        existence_signal = detect_existence_question(question)
        target_terms = "、".join(existence_signal.target_terms[:4]) if existence_signal.target_terms else "目标资料"
        if plan.mode == "refuse":
            if plan.reason == "existence_no_support":
                return (
                    "当前知识库证据不足，无法确认文库中包含该问题所要求的资料/数据/方案。"
                    f"已检索证据未能直接覆盖：{target_terms}。因此不能基于当前文库回答为“有”。"
                )
            reason = f"原因：{plan.reason}。" if plan.reason else ""
            return f"当前知识库证据不足，无法基于已检索证据回答该问题。{reason}".strip()

        lines: list[str] = []
        if analysis.intent == QueryIntent.EXPERIMENT:
            lines.append("不能据此生成新的实验方案或 protocol；以下仅为文献/知识库中已有信息的归纳。")
        comparison_coverage = plan.comparison_coverage if analysis.intent == QueryIntent.COMPARISON else None
        if plan.reason == "existence_weak_support":
            lines.append(
                "当前知识库只检索到间接或弱相关证据，不能确认文库中包含该问题所要求的资料/数据/方案。"
            )
            lines.append(
                f"以下证据仅能说明相关背景，不能证明文库中已有完整的{target_terms}。"
            )
        elif comparison_coverage and comparison_coverage.parse_ok and comparison_coverage.branch_evidence:
            if comparison_coverage.missing_branches or any(
                entry.status == "indirect" for entry in comparison_coverage.branch_evidence
            ):
                lines.append("根据当前知识库证据，可以进行有限比较：")
            else:
                lines.append("根据当前知识库证据，可以进行比较：")
            support_by_id = {item.evidence_id: item for item in support_pack}
            for branch_info in comparison_coverage.branch_evidence:
                evidence_refs = [
                    f"[{evidence_id}]"
                    for evidence_id in branch_info.evidence_ids
                    if evidence_id in support_by_id
                ]
                evidence_text = ""
                if evidence_refs:
                    evidence_text = "、".join(evidence_refs)
                if branch_info.status == "direct":
                    item = support_by_id.get(branch_info.evidence_ids[0]) if branch_info.evidence_ids else None
                    summary = _summarize(item) if item else "当前证据提供了该分支的直接信息。"
                    lines.append(f"关于{branch_info.branch}，证据显示：{summary} {evidence_text}".strip())
                elif branch_info.status == "indirect":
                    item = support_by_id.get(branch_info.evidence_ids[0]) if branch_info.evidence_ids else None
                    summary = _summarize(item) if item else "当前证据仅提供相关背景。"
                    lines.append(f"关于{branch_info.branch}，当前证据只提供间接线索：{summary} {evidence_text}".strip())
                else:
                    lines.append(f"关于{branch_info.branch}，当前 support pack 未提供直接证据。")
            if comparison_coverage.missing_branches or any(
                entry.status == "indirect" for entry in comparison_coverage.branch_evidence
            ):
                lines.append("因此不能把缺失或仅间接支持的分支推断为已被文库完整支持。")
        elif plan.mode == "full":
            lines.append("根据当前知识库证据：")
        else:
            lines.append("当前知识库只能支持部分回答：")
            if plan.covered_branches or plan.missing_branches:
                covered = "、".join(plan.covered_branches) if plan.covered_branches else "无"
                missing = "、".join(plan.missing_branches) if plan.missing_branches else "无"
                lines.append(f"已覆盖：{covered}；未覆盖：{missing}。")
            elif plan.reason:
                lines.append(f"证据限制：{plan.reason}。")

        for item in support_pack[:3]:
            lines.append(f"{_summarize(item)} [{item.evidence_id}]")
        return "\n".join(lines)


def _summarize(item: SupportItem) -> str:
    text = " ".join((item.candidate.text or "").replace("\n", " ").split())
    if len(text) > 160:
        text = text[:157].rstrip() + "..."
    section = item.candidate.section or "unknown"
    return f"{section} 证据显示：{text}"
