from __future__ import annotations

from src.synbio_rag.application.generation_v2.answer_planner import AnswerPlanner
from src.synbio_rag.application.generation_v2.guardrails import (
    detect_existence_question,
    evaluate_existence_support,
)
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent


def _analysis(intent: QueryIntent) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, requires_external_tools=False, search_limit=10, rerank_top_k=5)


def _candidate(
    evidence_id: str,
    text: str,
    *,
    doc_id: str = "doc1",
    section: str = "Results",
    rerank: float = 0.8,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        chunk_id=f"{evidence_id.lower()}_chunk",
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section=section,
        text=text,
        page_start=None,
        page_end=None,
        vector_score=0.0,
        bm25_score=0.0,
        rerank_score=rerank,
        fusion_score=0.0,
        metadata={},
        features={
            "has_table_text": False,
            "has_table_caption": False,
            "has_figure_caption": False,
            "has_numeric": any(char.isdigit() for char in text),
            "has_result_terms": any(term in text.lower() for term in ("result", "yield", "production", "strategy", "control")),
            "section_type": section.lower(),
            "text_length": len(text),
        },
        reasons=["seed_chunk", f"section:{section.lower()}"],
    )


def _support_item(candidate: EvidenceCandidate) -> SupportItem:
    return SupportItem(
        evidence_id=candidate.evidence_id,
        candidate=candidate,
        support_score=max(candidate.rerank_score, 0.1),
        reasons=["selected"],
    )


def test_detect_existence_question_positive_cases() -> None:
    cases = [
        "文库中是否有关于工业规模 PHA 发酵工艺中溶解氧控制策略的详细方案？",
        "当前文库里是否有关于 CAR-T 治疗 DLBCL III 期临床试验的系统综述？如果没有，请说明证据不足。",
        "文库中是否有关于 mRNA 疫苗 III 期临床试验结果的数据？如果没有，请明确说明证据不足。",
    ]

    for question in cases:
        signal = detect_existence_question(question)
        assert signal.is_existence_question is True
        assert signal.target_terms


def test_detect_existence_question_negative_cases() -> None:
    cases = [
        "FAM20A 是否调控 FAM20C 定位？",
        "2′-FL 是否通过调节肠道菌群改善骨质疏松？",
        "比较 A 和 B 的机制差异。",
    ]

    for question in cases:
        signal = detect_existence_question(question)
        assert signal.is_existence_question is False


def test_existence_guardrail_weak_support_blocks_full() -> None:
    planner = AnswerPlanner()
    question = "文库中是否有关于工业规模 PHA 发酵工艺中溶解氧控制策略的详细方案？请基于文库内容回答。"
    candidate = _candidate(
        "E1",
        "L-精氨酸 5 L 发酵罐的发酵条件和培养参数优化，讨论了补料与发酵条件。",
        doc_id="doc_arg",
        section="Abstract",
    )
    support_pack = [_support_item(candidate)]

    assessment = evaluate_existence_support(question, support_pack)
    plan = planner.plan(question, _analysis(QueryIntent.FACTOID), support_pack, [candidate])

    assert assessment.support_status == "weak"
    assert "PHA" in assessment.missing_core_terms
    assert plan.mode in {"partial", "refuse"}
    assert plan.mode != "full"
    assert plan.reason == "existence_weak_support"


def test_existence_guardrail_strong_support_allows_full() -> None:
    planner = AnswerPlanner()
    question = "文库中是否有关于工业规模 PHA 发酵工艺中溶解氧控制策略的详细方案？"
    candidate = _candidate(
        "E1",
        "This industrial-scale PHA fermentation process describes a dissolved oxygen control strategy for large-scale production. The detailed control strategy adjusts DO setpoints during industrial-scale fermentation.",
        doc_id="doc_pha",
        section="Results",
    )
    support_pack = [_support_item(candidate)]

    assessment = evaluate_existence_support(question, support_pack)
    plan = planner.plan(question, _analysis(QueryIntent.FACTOID), support_pack, [candidate])

    assert assessment.support_status == "strong"
    assert "PHA" in assessment.matched_core_terms
    assert plan.mode == "full"


def test_existence_guardrail_no_support_refuses() -> None:
    planner = AnswerPlanner()

    plan = planner.plan(
        "文库中是否有关于 mRNA 疫苗 III 期临床试验结果的数据？如果没有，请明确说明证据不足。",
        _analysis(QueryIntent.FACTOID),
        [],
        [],
    )

    assert plan.mode == "refuse"
    assert plan.reason == "existence_no_support"


def test_non_existence_factoid_still_allows_full() -> None:
    planner = AnswerPlanner()
    candidate = _candidate(
        "E1",
        "FAM20A regulates FAM20C localization in the Golgi apparatus according to the reported results.",
    )
    support_pack = [_support_item(candidate)]

    plan = planner.plan("FAM20A 是否调控 FAM20C 定位？", _analysis(QueryIntent.FACTOID), support_pack, [candidate])

    assert plan.mode == "full"
    assert plan.reason == "factoid_supported"


def test_comparison_hotfix_still_returns_partial_with_missing_branches() -> None:
    planner = AnswerPlanner()
    candidate = _candidate(
        "E1",
        "This study describes methanol induction timing and ATP balance during Pichia expression.",
        doc_id="doc_0074",
        section="Results and Discussion",
    )
    support_pack = [_support_item(candidate)]

    plan = planner.plan(
        "比较 Pichia 中两类提升表达的策略差异：一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数。",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
        [candidate],
    )

    assert plan.mode == "partial"
    assert plan.reason == "comparison_branch_coverage"
