from __future__ import annotations

from src.synbio_rag.application.generation_v2.answer_planner import AnswerPlanner
from src.synbio_rag.application.generation_v2.comparison_coverage import (
    build_comparison_coverage,
    extract_branch_terms,
)
from src.synbio_rag.application.generation_v2.models import AnswerPlan, ComparisonCoverage, SupportItem
from src.synbio_rag.application.generation_v2.qwen_synthesizer import validate_synthesized_answer
from src.synbio_rag.domain.config import GenerationConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent

from tests.test_generation_v2 import _analysis, _candidate


def _support(
    evidence_id: str,
    text: str,
    *,
    doc_id: str = "doc1",
    section: str = "Results",
    score: float = 0.9,
) -> SupportItem:
    return SupportItem(
        evidence_id=evidence_id,
        candidate=_candidate(
            evidence_id,
            text,
            doc_id=doc_id,
            section=section,
            rerank=score,
            features={"has_numeric": True, "has_result_terms": True},
        ),
        support_score=score,
        reasons=["comparison_selection"],
    )


def test_extract_branch_terms_keeps_entities_and_filters_generic_terms() -> None:
    assert "E. coli" in extract_branch_terms("调控 E. coli 唾液酸代谢")
    assert "唾液酸代谢" in extract_branch_terms("调控 E. coli 唾液酸代谢")
    assert "Neu5Ac" in extract_branch_terms("被改造成 Neu5Ac 传感器")
    assert "传感器" in extract_branch_terms("被改造成 Neu5Ac 传感器")
    terms = extract_branch_terms("直接增加 AOX1 启动子调控表达盒拷贝数")
    assert "AOX1" in terms
    assert "启动子" in terms
    assert "表达盒" in terms
    assert "拷贝数" in terms
    generic_terms = extract_branch_terms("作用机制和用途")
    assert "作用" not in generic_terms
    assert "机制" not in generic_terms
    assert "用途" not in generic_terms


def test_build_comparison_coverage_marks_direct_support() -> None:
    support_pack = [
        _support(
            "E1",
            "NanR-based Neu5Ac biosensor with mKate2 reporter and promoter engineering improved signal output.",
        )
    ]

    coverage = build_comparison_coverage(
        "比较两类角色",
        ["被改造成 Neu5Ac 传感器"],
        support_pack,
    )

    assert coverage.branch_evidence[0].status == "direct"
    assert coverage.branch_evidence[0].evidence_ids == ["E1"]


def test_build_comparison_coverage_marks_indirect_support() -> None:
    support_pack = [
        _support(
            "E1",
            "Methanol induction timing in Pichia expression improved protein production during fed-batch fermentation.",
        )
    ]

    coverage = build_comparison_coverage(
        "比较两类策略",
        ["优化甲醇诱导时机和能量利用"],
        support_pack,
    )

    assert coverage.branch_evidence[0].status == "indirect"


def test_build_comparison_coverage_marks_missing_support() -> None:
    support_pack = [
        _support("E1", "L-arginine production in a 5 L fermenter under optimized aeration.")
    ]

    coverage = build_comparison_coverage(
        "比较两类角色",
        ["被改造成 Neu5Ac 传感器"],
        support_pack,
    )

    assert coverage.branch_evidence[0].status == "missing"
    assert coverage.branch_evidence[0].evidence_ids == []


def test_comparison_planner_stays_partial_when_one_branch_missing() -> None:
    planner = AnswerPlanner()
    support_pack = [
        _support(
            "E1",
            "NanR regulates E. coli sialic acid metabolism and improved downstream catabolic response.",
        )
    ]

    plan = planner.plan(
        "NanR 在天然调控和工程应用中的两种角色：一类是调控 E. coli 唾液酸代谢，另一类是被改造成 Neu5Ac 传感器",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
        config=GenerationConfig(v2_enable_comparison_coverage=True),
    )

    assert plan.mode == "partial"
    assert "调控 E. coli 唾液酸代谢" in plan.covered_branches
    assert "被改造成 Neu5Ac 传感器" in plan.missing_branches
    assert plan.reason == "comparison_branch_coverage"


def test_comparison_planner_does_not_upgrade_full_when_branches_share_one_direct_evidence() -> None:
    planner = AnswerPlanner()
    support_pack = [
        _support(
            "E1",
            "Methanol induction optimization with energy utilization analysis and AOX1 promoter multi-copy expression cassette copy number increase improved production.",
        )
    ]

    plan = planner.plan(
        "比较 Pichia 中两类提升表达的策略差异：一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数。",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
        config=GenerationConfig(v2_enable_comparison_coverage=True),
    )

    assert plan.mode == "partial"
    assert plan.comparison_coverage is not None
    assert plan.comparison_coverage.allowed_citation_evidence_ids == ["E1"]


def test_allowed_citation_set_is_support_pack_subset() -> None:
    support_pack = [
        _support("E1", "NanR regulates E. coli sialic acid metabolism."),
        _support("E2", "Neu5Ac biosensor with mKate2 reporter enabled sensing."),
    ]
    coverage = build_comparison_coverage(
        "比较两类角色",
        ["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        support_pack,
    )

    support_ids = {item.evidence_id for item in support_pack}
    assert set(coverage.allowed_citation_evidence_ids) <= support_ids
    assert "E1" in coverage.allowed_citation_evidence_ids
    assert "E2" in coverage.allowed_citation_evidence_ids


def test_qwen_validator_allows_comparison_allowed_subset() -> None:
    support_pack = [
        _support("E1", "NanR regulates E. coli sialic acid metabolism."),
        _support("E2", "Neu5Ac biosensor with mKate2 reporter enabled sensing."),
    ]
    coverage = ComparisonCoverage(
        parse_ok=True,
        branches=["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        branch_evidence=[
            build_comparison_coverage("", ["调控 E. coli 唾液酸代谢"], support_pack).branch_evidence[0],
            build_comparison_coverage("", ["被改造成 Neu5Ac 传感器"], support_pack).branch_evidence[0],
        ],
        covered_branches=["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        missing_branches=[],
        allowed_citation_evidence_ids=["E1", "E2"],
        reason="all_branches_direct",
    )
    plan = AnswerPlan(
        mode="partial",
        reason="comparison_branch_coverage",
        covered_branches=coverage.covered_branches,
        missing_branches=[],
        allowed_scope=["branch_limited_comparison"],
        comparison_coverage=coverage,
    )

    is_valid, flags = validate_synthesized_answer(
        "关于天然调控的证据如下 [E1]；关于工程应用的证据如下 [E2]，但当前仍需保留分支限制说明 [E1]",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="当前证据还不能逐分支完整比较 [E1]",
        existence_guardrail={},
    )

    assert is_valid is True
    assert flags == []


def test_qwen_validator_rejects_comparison_out_of_set_or_hidden_limits() -> None:
    support_pack = [
        _support("E1", "Methanol induction timing improved expression in Pichia."),
    ]
    coverage = build_comparison_coverage(
        "比较两类策略",
        ["优化甲醇诱导时机和能量利用", "直接增加 AOX1 启动子调控表达盒拷贝数"],
        support_pack,
    )
    plan = AnswerPlan(
        mode="partial",
        reason="comparison_branch_coverage",
        covered_branches=coverage.covered_branches,
        missing_branches=coverage.missing_branches,
        allowed_scope=["branch_limited_comparison"],
        comparison_coverage=coverage,
    )

    invalid_answer_ok, invalid_flags = validate_synthesized_answer(
        "两类策略都已充分证明 [E99]",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="当前证据还不能逐分支完整比较 [E1]",
        existence_guardrail={},
    )
    hidden_limit_ok, hidden_limit_flags = validate_synthesized_answer(
        "第一类策略得到支持 [E1]",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="当前证据还不能逐分支完整比较 [E1]",
        existence_guardrail={},
    )

    assert invalid_answer_ok is False
    assert "invalid_citation_ids" in invalid_flags or "comparison_disallowed_citation" in invalid_flags
    assert hidden_limit_ok is False
    assert (
        "missing_branch_limit_not_disclosed" in hidden_limit_flags
        or "indirect_branch_limit_not_disclosed" in hidden_limit_flags
        or "partial_overclaim" in hidden_limit_flags
    )


def test_nanr_style_support_no_longer_leaves_all_branches_uncovered() -> None:
    support_pack = [
        _support("E1", "NanR regulates E. coli sialic acid metabolism through transcriptional control."),
        _support("E2", "Neu5Ac biosensor with mKate2 reporter was engineered for sensing."),
    ]

    coverage = build_comparison_coverage(
        "NanR 在天然调控和工程应用中的两种角色：一类是调控 E. coli 唾液酸代谢，另一类是被改造成 Neu5Ac 传感器",
        ["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        support_pack,
    )

    assert coverage.covered_branches
    assert any(entry.status in {"direct", "indirect"} for entry in coverage.branch_evidence)


def test_aox1_copy_number_branch_can_match_branch_evidence() -> None:
    support_pack = [
        _support(
            "E1",
            "AOX1 promoter multi-copy integration increased expression cassette copy number and improved protein expression.",
        )
    ]

    coverage = build_comparison_coverage(
        "比较两类策略",
        ["直接增加 AOX1 启动子调控表达盒拷贝数"],
        support_pack,
    )

    assert coverage.branch_evidence[0].status in {"direct", "indirect"}
    assert "E1" in coverage.allowed_citation_evidence_ids
