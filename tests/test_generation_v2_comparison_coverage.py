from __future__ import annotations

from src.synbio_rag.application.generation_v2.answer_planner import AnswerPlanner
from src.synbio_rag.application.generation_v2.comparison_coverage import (
    build_comparison_coverage,
    extract_branch_terms,
)
from src.synbio_rag.application.generation_v2.models import AnswerPlan, ComparisonCoverage, SupportItem
from src.synbio_rag.application.generation_v2.qwen_synthesizer import validate_synthesized_answer
from src.synbio_rag.domain.config import GenerationConfig
from src.synbio_rag.domain.schemas import QueryIntent

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
    assert "表达盒拷贝数" in terms or "拷贝数" in terms
    generic_terms = extract_branch_terms("作用机制和用途")
    assert "作用" not in generic_terms
    assert "机制" not in generic_terms
    assert "用途" not in generic_terms


def test_direct_is_stricter_for_methanol_energy_branch() -> None:
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


def test_aox1_copy_number_branch_requires_specific_combo_for_direct() -> None:
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

    assert coverage.branch_evidence[0].status == "direct"
    assert coverage.branch_evidence[0].primary_evidence_ids == ["E1"]


def test_shared_evidence_penalty_prevents_full_comparison() -> None:
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
    assert plan.reason == "shared_evidence_limited_comparison"
    assert plan.comparison_coverage is not None
    assert plan.comparison_coverage.reason == "shared_evidence_limited_comparison"


def test_secondary_evidence_expansion_adds_branch_relevant_support_only() -> None:
    support_pack = [
        _support("E3", "NanR- Neu5Ac binding attenuated DNA affinity and affected sialic acid metabolism in E. coli.", score=1.0),
        _support("E4", "The Discussion section explains NanR repression and the sialoregulon in E. coli catabolism.", section="Discussion", score=0.8),
        _support("E9", "Unrelated L-arginine fermenter optimization in 5 L tanks.", doc_id="doc9", score=0.7),
    ]

    coverage = build_comparison_coverage(
        "NanR 在天然调控中的角色",
        ["调控 E. coli 唾液酸代谢"],
        support_pack,
    )

    branch = coverage.branch_evidence[0]
    assert branch.primary_evidence_ids == ["E3"]
    assert "E4" in branch.secondary_evidence_ids
    assert "E9" not in coverage.allowed_citation_evidence_ids
    assert "E3" in coverage.allowed_citation_evidence_ids
    assert "E4" in coverage.allowed_citation_evidence_ids


def test_ent007_style_support_keeps_regulation_and_sensor_evidence() -> None:
    support_pack = [
        _support("E1", "NanR-based Neu5Ac biosensor with mKate2 reporter and promoter engineering improved signal output.", section="Abstract", score=0.95),
        _support("E3", "Neu5Ac binding to NanR attenuated DNA binding and changed sialic acid metabolism readouts.", section="Results", score=1.0),
        _support("E4", "Discussion of NanR repression, the sialoregulon, and E. coli sialic acid catabolism.", section="Discussion", score=0.92),
        _support("E2", "Neu5Ac synthesis host background in E. coli.", section="Introduction", score=0.5),
    ]

    coverage = build_comparison_coverage(
        "NanR 在天然调控和工程应用中的两种角色：一类是调控 E. coli 唾液酸代谢，另一类是被改造成 Neu5Ac 传感器",
        ["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        support_pack,
    )

    regulation = coverage.branch_evidence[0]
    sensor = coverage.branch_evidence[1]
    assert regulation.status in {"direct", "indirect"}
    assert "E3" in regulation.evidence_ids
    assert "E4" in coverage.allowed_citation_evidence_ids
    assert sensor.status == "direct"
    assert "E1" in coverage.allowed_citation_evidence_ids


def test_ent020_style_support_spreads_allowed_set_across_branches() -> None:
    planner = AnswerPlanner()
    support_pack = [
        _support("E3", "Multi-copy strains changed glycerol consumption, methanol response and some energy-related pathways.", section="Results and Discussion", score=1.0),
        _support("E2", "rMeOH, OUR, NADH and ATP regeneration quantified energy utilization during methanol induction.", section="Results and Discussion", score=0.85),
        _support("E1", "AOX1 promoter multi-copy expression cassette copy number increase improved production.", section="Results", score=0.9),
        _support("E4", "Methanol-inducible AOX1 promoter and multi-copy cassette design changed expression levels.", section="Results and Discussion", score=0.82),
        _support("E9", "Unrelated oxygen transfer optimization for another strain.", score=0.4),
    ]

    plan = planner.plan(
        "比较 Pichia 中两类提升表达的策略差异：一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数。",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
        config=GenerationConfig(v2_enable_comparison_coverage=True),
    )

    assert plan.mode == "partial"
    assert plan.comparison_coverage is not None
    coverage = plan.comparison_coverage
    assert "E2" in coverage.allowed_citation_evidence_ids
    assert "E1" in coverage.allowed_citation_evidence_ids or "E4" in coverage.allowed_citation_evidence_ids
    assert "E9" not in coverage.allowed_citation_evidence_ids


def test_coverage_empty_reason_is_recorded_for_parse_failure_and_disabled() -> None:
    planner = AnswerPlanner()
    support_pack = [_support("E1", "Methanol induction timing and ATP balance.")]

    planner.plan(
        "比较策略差异",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
        config=GenerationConfig(v2_enable_comparison_coverage=True),
    )
    assert planner.last_comparison_coverage_debug["reason"] == "branch_parse_failed"

    planner.plan(
        "比较 Pichia 中两类提升表达的策略差异：一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数。",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
        config=GenerationConfig(v2_enable_comparison_coverage=False),
    )
    assert planner.last_comparison_coverage_debug["reason"] == "coverage_disabled"


def test_qwen_validator_allows_allowed_secondary_evidence_and_rejects_disallowed() -> None:
    support_pack = [
        _support("E3", "NanR attenuated DNA binding in the presence of Neu5Ac and affected sialic acid metabolism.", score=1.0),
        _support("E4", "Discussion of NanR repression and sialoregulon behavior in E. coli.", section="Discussion", score=0.85),
        _support("E1", "Neu5Ac biosensor with mKate2 reporter enabled sensing.", score=0.9),
        _support("E9", "Unrelated L-arginine fermentation background.", score=0.3),
    ]
    coverage = build_comparison_coverage(
        "NanR 在天然调控和工程应用中的两种角色：一类是调控 E. coli 唾液酸代谢，另一类是被改造成 Neu5Ac 传感器",
        ["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        support_pack,
    )
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        covered_branches=coverage.covered_branches,
        missing_branches=coverage.missing_branches,
        allowed_scope=["branch_limited_comparison"],
        comparison_coverage=coverage,
    )

    valid_ok, valid_flags = validate_synthesized_answer(
        "关于天然调控分支，当前证据只提供间接线索 [E3][E4]；关于工程应用分支，证据直接涉及 Neu5Ac 传感器 [E1]，因此只能做有限比较 [E3]",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较： [E3][E1]",
        existence_guardrail={},
    )
    invalid_ok, invalid_flags = validate_synthesized_answer(
        "两类分支都已充分证明 [E9]",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较： [E3][E1]",
        existence_guardrail={},
    )

    assert valid_ok is True
    assert valid_flags == []
    assert invalid_ok is False
    assert "comparison_disallowed_citation" in invalid_flags
    assert validate_synthesized_answer.last_details["disallowed_evidence_ids"] == ["E9"]
