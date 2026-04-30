from __future__ import annotations

import re

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
            if plan.mode == "full":
                lines.append("根据当前知识库证据，可以进行比较：")
            else:
                lines.append("根据当前知识库证据，只能进行有限比较：")
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
                    summary = _summarize(item) if item else "当前证据直接涉及该分支。"
                    lines.append(f"关于{branch_info.branch}，证据直接涉及：{summary} {evidence_text}".strip())
                elif branch_info.status == "indirect":
                    item = support_by_id.get(branch_info.evidence_ids[0]) if branch_info.evidence_ids else None
                    summary = _summarize(item) if item else "当前证据仅提供相关背景。"
                    lines.append(f"关于{branch_info.branch}，当前证据只提供间接线索：{summary} {evidence_text}".strip())
                else:
                    lines.append(f"关于{branch_info.branch}，当前 support pack 未提供直接证据。")
            if plan.reason == "shared_evidence_limited_comparison":
                lines.append("部分分支依赖相同或重叠证据，不能视为两个分支都已有独立充分支持。")
            if comparison_coverage.missing_branches or any(
                entry.status == "indirect" for entry in comparison_coverage.branch_evidence
            ) or plan.reason == "shared_evidence_limited_comparison":
                lines.append("因此不能把缺失或仅间接支持的分支推断为已被文库完整支持。")
        elif analysis.intent == QueryIntent.SUMMARY and plan.mode == "full":
            lines.append("根据当前知识库证据，可作如下总结：")
        elif analysis.intent == QueryIntent.SUMMARY and plan.mode == "partial":
            lines.append("当前知识库只能支持有限总结。")
            if len(support_pack) == 1:
                lines.append("当前只检索到一条较直接的合格证据，因此以下总结只能覆盖该证据涉及的范围。")
            elif plan.reason == "summary_abstract_only":
                lines.append("当前合格证据主要来自摘要层面，缺少更完整的结果或讨论支撑，因此不能视为完整综述。")
            elif plan.reason in {"summary_single_doc_limited", "summary_low_diversity"}:
                lines.append("当前证据可以支持部分总结，但证据集中在有限文献或章节中，不能视为完整综述。")
            else:
                lines.append("当前知识库只能支持有限总结，因为检索到的合格证据较少。")
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

        # Phase 9: summary route — produce structured supported claims instead of raw evidence snippets
        if analysis.intent == QueryIntent.SUMMARY:
            claims, limitations = _build_summary_claims(support_pack, plan)
            lines.append("")
            lines.extend(claims)
            if limitations:
                lines.append("")
                lines.append("证据限制：")
                lines.extend(f"- {lim}" for lim in limitations)
        else:
            for item in support_pack[:3]:
                lines.append(f"{_summarize(item)} [{item.evidence_id}]")
        return "\n".join(lines)


def _summarize(item: SupportItem) -> str:
    text = " ".join((item.candidate.text or "").replace("\n", " ").split())
    if len(text) > 160:
        text = text[:157].rstrip() + "..."
    section = item.candidate.section or "unknown"
    return f"{section} 证据显示：{text}"


# ── Phase 9: Summary supported-claims builder ────────────────────

_CLAIM_SECTION_PRIORITY = {
    "abstract": 0, "conclusion": 0, "conclusions": 0,
    "results and discussion": 1, "results": 2, "discussion": 3,
    "introduction": 4, "background": 4,
}

_BIBLIOGRAPHY_SECTIONS = {"references", "bibliography", "acknowledgements", "author information"}


def _build_summary_claims(
    support_pack: list[SupportItem],
    plan: AnswerPlan,
) -> tuple[list[str], list[str]]:
    """Build structured supported claims and evidence limitations for summary route.

    Returns (claims, limitations) where each claim is a simple sentence
    with a citation marker, and each limitation is a one-line caveat.
    """
    claims: list[str] = []
    limitations: list[str] = []

    # Rank items by section quality (prefer summary sections)
    ranked = sorted(
        support_pack,
        key=lambda item: (
            _CLAIM_SECTION_PRIORITY.get(
                (item.candidate.section or "").lower(), 5
            ),
            -(item.support_score or 0),
        ),
    )

    # Track which docs and sections are covered
    covered_docs: set[str] = set()
    covered_sections: set[str] = set()
    seen_text_fingerprints: set[str] = set()

    max_claims = min(len(ranked), 5)
    claim_count = 0

    for item in ranked:
        if claim_count >= max_claims:
            break

        section = (item.candidate.section or "").lower()
        doc_id = item.candidate.doc_id or ""

        # Skip bibliography/reference sections
        if any(bib in section for bib in _BIBLIOGRAPHY_SECTIONS):
            continue

        # Skip near-duplicate claims (text overlap > 80%)
        text = " ".join((item.candidate.text or "").replace("\n", " ").split())
        fingerprint = " ".join(text.split()[:10])  # first 10 words as fingerprint
        if fingerprint in seen_text_fingerprints:
            continue
        seen_text_fingerprints.add(fingerprint)

        # Build a concise claim from the evidence
        claim_text = _make_claim(text, section, doc_id, item.evidence_id)
        if claim_text:
            claims.append(claim_text)
            claim_count += 1
            covered_docs.add(doc_id)
            covered_sections.add(section)

    # Build limitations
    if len(support_pack) == 0:
        limitations.append("未检索到合格证据。")
    elif len(support_pack) <= 2:
        limitations.append(f"仅有 {len(support_pack)} 条合格证据，覆盖面有限。")
    elif plan.mode == "partial":
        limitations.append("当前证据不足以构成完整综述。")

    if covered_sections and all(s in {"introduction", "background"} for s in covered_sections):
        limitations.append("主要证据来自 Introduction/Background，缺少 Results 原文支撑。")

    if plan.mode == "partial" and plan.reason == "summary_abstract_only":
        limitations.append("证据主要来自摘要，缺少详细结果或讨论。")

    return claims, limitations


def _make_claim(text: str, section: str, doc_id: str, evidence_id: str) -> str | None:
    """Extract a concise claim from evidence text. Returns a one-sentence claim with citation.

    Conservative: if text is too fragmented or bibliography-like, return None.
    """
    # Clean text
    text = " ".join(text.replace("\n", " ").split())
    if len(text) < 30:
        return None

    # Skip bibliography-like content
    if _is_bibliography_line(text):
        return None

    # Build short claim: truncate to a reasonable sentence length
    # Try to end at a sentence boundary
    claim = text[:200].rstrip()
    # Find last sentence-ending punctuation within range
    for sep in [". ", "。", ".\n", ".\r"]:
        last_dot = claim.rfind(sep)
        if last_dot > 60:
            claim = claim[:last_dot + len(sep)].rstrip()
            break
    else:
        if len(text) > 200:
            claim = text[:197].rstrip() + "..."

    # Add section+source label
    section_label = _section_short(section)
    source = f"[{doc_id}]" if doc_id else ""
    return f"- {section_label}: {claim} [{evidence_id}] {source}".strip()


def _section_short(section: str) -> str:
    """Short section label for claims."""
    mapping = {
        "abstract": "摘要", "conclusion": "结论", "conclusions": "结论",
        "results": "结果", "discussion": "讨论",
        "results and discussion": "结果与讨论",
        "introduction": "引言", "background": "背景",
        "methods": "方法", "materials and methods": "方法",
    }
    return mapping.get(section.lower(), section)


def _is_bibliography_line(text: str) -> bool:
    """Check if text looks like a bibliography/reference entry."""
    lowered = text.lower()
    if lowered.startswith("http") or lowered.startswith("doi"):
        return True
    if re.match(r"^\d+\.\s", lowered):  # numbered reference
        return True
    if re.match(r"^\[\d+\]", lowered):  # [1] style reference
        return True
    return False
