from __future__ import annotations

from src.synbio_rag.application.generation_v2.models import (
    AnswerPlan,
    BranchEvidence,
    ComparisonCoverage,
    EvidenceCandidate,
    SupportItem,
)
from src.synbio_rag.application.generation_v2.qwen_synthesizer import QwenSynthesizer, validate_synthesized_answer
from src.synbio_rag.application.generation_v2.service import GenerationV2Service
from src.synbio_rag.domain.config import GenerationConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


def _analysis(intent: QueryIntent = QueryIntent.FACTOID) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, requires_external_tools=False, search_limit=10, rerank_top_k=5)


def _candidate(
    evidence_id: str = "E1",
    *,
    doc_id: str = "doc_0001",
    section: str = "Results",
    text: str = "PHA fermentation results show dissolved oxygen control improved yield.",
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
        vector_score=0.2,
        bm25_score=0.1,
        rerank_score=0.9,
        fusion_score=0.8,
        metadata={},
        features={
            "has_table_text": False,
            "has_table_caption": False,
            "has_figure_caption": False,
            "has_numeric": True,
            "has_result_terms": True,
            "section_type": section.lower(),
            "text_length": len(text),
        },
        reasons=["seed_chunk", f"section:{section.lower()}"],
    )


def _support_item(
    evidence_id: str = "E1",
    *,
    doc_id: str = "doc_0001",
    section: str = "Results",
    text: str = "PHA fermentation results show dissolved oxygen control improved yield.",
) -> SupportItem:
    return SupportItem(
        evidence_id=evidence_id,
        candidate=_candidate(evidence_id=evidence_id, doc_id=doc_id, section=section, text=text),
        support_score=0.9,
        reasons=["selected_support"],
    )


def _chunk(text: str = "PHA fermentation results show dissolved oxygen control improved yield.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_1",
        doc_id="doc_0001",
        source_file="doc_0001.pdf",
        title="title-doc_0001",
        section="Results",
        text=text,
        rerank_score=0.9,
        fusion_score=0.8,
    )


class FakeChatClient:
    def __init__(self, output: str, *, enabled: bool = True) -> None:
        self.output = output
        self.enabled = enabled
        self.calls = 0
        self.timeout_seconds = 30

    def is_enabled(self) -> bool:
        return self.enabled

    def chat_completion(self, **_: object) -> str:
        self.calls += 1
        return self.output


class ExplodingChatClient(FakeChatClient):
    def __init__(self) -> None:
        super().__init__("", enabled=True)

    def chat_completion(self, **_: object) -> str:
        raise AssertionError("Qwen client should not be called")


class DummyLedger:
    def __init__(self, candidates: list[EvidenceCandidate]) -> None:
        self.candidates = candidates

    def build(self, question: str, analysis: QueryAnalysis, seed_chunks: list[RetrievedChunk]) -> list[EvidenceCandidate]:
        del question, analysis, seed_chunks
        return self.candidates


class DummySelector:
    def __init__(self, support_pack: list[SupportItem]) -> None:
        self.support_pack = support_pack

    def select(
        self,
        question: str,
        analysis: QueryAnalysis,
        candidates: list[EvidenceCandidate],
        config: GenerationConfig,
    ) -> list[SupportItem]:
        del question, analysis, candidates, config
        return self.support_pack


class DummyPlanner:
    def __init__(self, plan: AnswerPlan, *, existence_guardrail: dict | None = None) -> None:
        self.plan_value = plan
        self.last_existence_guardrail = existence_guardrail or {}

    def plan(
        self,
        question: str,
        analysis: QueryAnalysis,
        support_pack: list[SupportItem],
        candidates: list[EvidenceCandidate] | None = None,
        config: GenerationConfig | None = None,
    ) -> AnswerPlan:
        del question, analysis, support_pack, candidates, config
        return self.plan_value


class DummyBuilder:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def build(
        self,
        question: str,
        analysis: QueryAnalysis,
        plan: AnswerPlan,
        support_pack: list[SupportItem],
    ) -> str:
        del question, analysis, plan, support_pack
        return self.answer


def _service_with(
    *,
    qwen_output: str | None,
    plan: AnswerPlan,
    support_pack: list[SupportItem],
    extractive_answer: str = "根据当前知识库证据：结果支持该结论 [E1]",
    existence_guardrail: dict | None = None,
) -> GenerationV2Service:
    client = FakeChatClient(qwen_output or "") if qwen_output is not None else ExplodingChatClient()
    service = GenerationV2Service(synthesizer=QwenSynthesizer(client=client))
    service.ledger_builder = DummyLedger([item.candidate for item in support_pack])
    service.support_selector = DummySelector(support_pack)
    service.answer_planner = DummyPlanner(plan, existence_guardrail=existence_guardrail)
    service.answer_builder = DummyBuilder(extractive_answer)
    return service


def _comparison_coverage(*, allowed_ids: list[str], missing_branches: list[str] | None = None) -> ComparisonCoverage:
    missing = missing_branches or []
    covered = ["branch_a"] if missing else ["branch_a", "branch_b"]
    branch_evidence = [
        BranchEvidence(
            branch="branch_a",
            status="direct",
            evidence_ids=["E1"],
            primary_evidence_ids=["E1"],
        ),
        BranchEvidence(
            branch="branch_b",
            status="indirect" if missing else "direct",
            evidence_ids=["E2"],
            primary_evidence_ids=["E2"],
        ),
    ]
    return ComparisonCoverage(
        parse_ok=True,
        branches=["branch_a", "branch_b"],
        branch_evidence=branch_evidence,
        covered_branches=covered,
        missing_branches=missing,
        allowed_citation_evidence_ids=allowed_ids,
        reason="branch_partial_support" if missing else "all_branches_direct",
    )


def test_qwen_synthesis_disabled_skips_client_and_reports_debug() -> None:
    service = _service_with(
        qwen_output=None,
        plan=AnswerPlan(mode="full", reason="factoid_supported"),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="PHA 的结论是什么？",
        analysis=_analysis(),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=False),
    )

    assert result.answer == "根据当前知识库证据：结果支持该结论 [1]"
    assert result.debug["qwen_synthesis"]["enabled"] is False
    assert result.debug["qwen_synthesis"]["attempted"] is False
    assert result.debug["qwen_synthesis"]["used_qwen"] is False
    assert result.debug["qwen_synthesis"]["fallback_reason"] == "disabled"


def test_qwen_synthesis_refuse_does_not_call_qwen() -> None:
    synthesizer = QwenSynthesizer(client=ExplodingChatClient())
    result = synthesizer.synthesize(
        question="文库中是否有该方案？",
        plan=AnswerPlan(mode="refuse", reason="existence_no_support"),
        support_pack=[],
        extractive_answer="当前知识库证据不足。",
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.answer == "当前知识库证据不足。"
    assert result.used_qwen is False
    assert result.fallback_used is True
    assert result.fallback_reason == "refuse_or_empty_support"


def test_qwen_synthesis_success_binds_internal_citation_ids() -> None:
    service = _service_with(
        qwen_output="根据当前证据，可以回答该问题，关键结果如下 [E1]",
        plan=AnswerPlan(mode="full", reason="factoid_supported"),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="PHA 的结论是什么？",
        analysis=_analysis(),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.answer == "根据当前证据，可以回答该问题，关键结果如下 [1]"
    assert len(result.citations) == 1
    assert result.debug["qwen_synthesis"]["used_qwen"] is True
    assert result.debug["qwen_synthesis"]["fallback_used"] is False


def test_qwen_synthesis_falls_back_on_invalid_citation_ids() -> None:
    service = _service_with(
        qwen_output="该结论来自证据 [E99]",
        plan=AnswerPlan(mode="full", reason="factoid_supported"),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="PHA 的结论是什么？",
        analysis=_analysis(),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.answer == "根据当前知识库证据：结果支持该结论 [1]"
    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "invalid_citation_ids" in result.debug["qwen_synthesis"]["validation_flags"]


def test_qwen_synthesis_falls_back_when_output_has_no_citation() -> None:
    service = _service_with(
        qwen_output="根据当前证据，可以回答该问题。",
        plan=AnswerPlan(mode="full", reason="factoid_supported"),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="PHA 的结论是什么？",
        analysis=_analysis(),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.answer == "根据当前知识库证据：结果支持该结论 [1]"
    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "missing_citations" in result.debug["qwen_synthesis"]["validation_flags"]


def test_qwen_synthesis_falls_back_when_citation_set_changes() -> None:
    service = _service_with(
        qwen_output="根据当前证据，可以回答该问题 [E1]",
        plan=AnswerPlan(mode="full", reason="factoid_supported"),
        support_pack=[_support_item(), _support_item("E2", doc_id="doc_0002")],
        extractive_answer="根据当前知识库证据：结果支持该结论 [E1]，补充证据如下 [E2]",
    )

    result = service.run(
        question="PHA 的结论是什么？",
        analysis=_analysis(),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "citation_set_changed" in result.debug["qwen_synthesis"]["validation_flags"]


def test_qwen_synthesis_partial_cannot_be_rewritten_as_full() -> None:
    service = _service_with(
        qwen_output="可以完整回答如下：该问题已经得到完整证明 [E1]",
        plan=AnswerPlan(mode="partial", reason="summary_support_count"),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="总结该问题",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "partial_overclaim" in result.debug["qwen_synthesis"]["validation_flags"]


def test_qwen_synthesis_partial_non_existence_cannot_shift_to_refusal_tone() -> None:
    service = _service_with(
        qwen_output="文库中未提供该问题的直接证据，因此无法基于当前证据总结 [E1]",
        plan=AnswerPlan(mode="partial", reason="summary_support_count"),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="总结 Pichia pastoris 中提高氧化戊糖磷酸途径通量为何会提升重组蛋白产量。",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "partial_abstention_tone" in result.debug["qwen_synthesis"]["validation_flags"]


def test_validate_synthesized_answer_allows_partial_comparison_with_limit_terms() -> None:
    support_pack = [_support_item("E1"), _support_item("E2", doc_id="doc_0002")]
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        covered_branches=["branch_a", "branch_b"],
        missing_branches=[],
        comparison_coverage=_comparison_coverage(allowed_ids=["E1", "E2"]),
    )

    ok, flags = validate_synthesized_answer(
        "在文库所支持的范围内，可进行有限比较：关于分支A，证据显示相关结果 [E1]；关于分支B，当前证据提供部分支持 [E2]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E1][E2]",
        existence_guardrail={},
    )

    assert ok is True
    assert flags == []
    details = validate_synthesized_answer.last_details
    assert "在文库所支持的范围内" in details["partial_limit_terms_found"]
    assert "有限比较" in details["partial_limit_terms_found"]
    assert details["partial_tone_decision"] == "pass"
    assert details["comparison_policy"] == "comparison_allowed_subset"
    assert details["citation_subset_ok"] is True


def test_validate_synthesized_answer_rejects_partial_without_limit_terms() -> None:
    support_pack = [_support_item("E1"), _support_item("E2", doc_id="doc_0002")]
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        comparison_coverage=_comparison_coverage(allowed_ids=["E1", "E2"]),
    )

    ok, flags = validate_synthesized_answer(
        "证据显示两类分支分别具有相关结果 [E1][E2]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E1][E2]",
        existence_guardrail={},
    )

    assert ok is False
    assert "partial_abstention_tone" in flags
    assert validate_synthesized_answer.last_details["partial_tone_decision"] == "fail"


def test_validate_synthesized_answer_rejects_partial_with_overclaim_terms() -> None:
    support_pack = [_support_item("E1"), _support_item("E2", doc_id="doc_0002")]
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        comparison_coverage=_comparison_coverage(allowed_ids=["E1", "E2"]),
    )

    ok, flags = validate_synthesized_answer(
        "在文库所支持的范围内，可以完整比较，两者均已充分证明 [E1][E2]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E1][E2]",
        existence_guardrail={},
    )

    assert ok is False
    assert "partial_abstention_tone" in flags
    assert "partial_overclaim" in flags
    details = validate_synthesized_answer.last_details
    assert "可以完整比较" in details["partial_overclaim_terms_found"] or "完整比较" in details["partial_overclaim_terms_found"]


def test_validate_synthesized_answer_keeps_comparison_allowed_subset_strict() -> None:
    support_pack = [
        _support_item("E1"),
        _support_item("E2", doc_id="doc_0002"),
        _support_item("E3", doc_id="doc_0003"),
    ]
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        comparison_coverage=_comparison_coverage(allowed_ids=["E1", "E2"]),
    )

    ok, flags = validate_synthesized_answer(
        "在文库所支持的范围内，可进行有限比较 [E3]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E1][E2]",
        existence_guardrail={},
    )

    assert ok is False
    assert "comparison_disallowed_citation" in flags
    assert validate_synthesized_answer.last_details["disallowed_evidence_ids"] == ["E3"]


def test_validate_synthesized_answer_ent007_style_partial_comparison_passes() -> None:
    support_pack = [
        _support_item("E1"),
        _support_item("E3", doc_id="doc_0003"),
        _support_item("E4", doc_id="doc_0004"),
    ]
    coverage = ComparisonCoverage(
        parse_ok=True,
        branches=["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        branch_evidence=[
            BranchEvidence(branch="调控 E. coli 唾液酸代谢", status="direct", evidence_ids=["E3", "E4"], primary_evidence_ids=["E4"]),
            BranchEvidence(branch="被改造成 Neu5Ac 传感器", status="direct", evidence_ids=["E1"], primary_evidence_ids=["E1"]),
        ],
        covered_branches=["调控 E. coli 唾液酸代谢", "被改造成 Neu5Ac 传感器"],
        missing_branches=[],
        allowed_citation_evidence_ids=["E3", "E4", "E1"],
        reason="all_branches_direct",
    )
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        covered_branches=coverage.covered_branches,
        comparison_coverage=coverage,
    )

    ok, flags = validate_synthesized_answer(
        "在文库所支持的范围内，可进行有限比较：天然调控分支的证据显示相关机制 [E3][E4]；工程应用分支有直接传感器证据 [E1]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E3][E4][E1]",
        existence_guardrail={},
    )

    assert ok is True
    assert flags == []


def test_validate_synthesized_answer_ent084_style_disallowed_citation_still_fails() -> None:
    support_pack = [
        _support_item("E1"),
        _support_item("E2", doc_id="doc_0002"),
        _support_item("E3", doc_id="doc_0003"),
        _support_item("E4", doc_id="doc_0004"),
    ]
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        comparison_coverage=_comparison_coverage(allowed_ids=["E1", "E2"]),
    )

    ok, flags = validate_synthesized_answer(
        "在文库所支持的范围内，可进行有限比较 [E1][E3][E4]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E1][E2]",
        existence_guardrail={},
    )

    assert ok is False
    assert "comparison_disallowed_citation" in flags


def test_qwen_synthesis_existence_weak_support_cannot_claim_full_existence() -> None:
    service = _service_with(
        qwen_output="文库中有完整详细方案，相关证据如下 [E1]",
        plan=AnswerPlan(mode="partial", reason="existence_weak_support"),
        support_pack=[_support_item(text="L-arginine fermentation in a 5 L tank.")],
        existence_guardrail={
            "is_existence_question": True,
            "support_status": "weak",
            "target_terms": ["PHA", "工业规模", "溶解氧", "控制策略"],
            "support_reason": "core_terms_not_sufficiently_covered",
            "missing_core_terms": ["PHA", "工业规模", "控制策略"],
        },
    )

    result = service.run(
        question="文库中是否有关于工业规模 PHA 发酵工艺中溶解氧控制策略的详细方案？",
        analysis=_analysis(),
        seed_chunks=[_chunk("L-arginine fermentation in a 5 L tank.")],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "existence_overclaim" in result.debug["qwen_synthesis"]["validation_flags"]


def test_qwen_synthesis_requires_missing_branch_disclosure() -> None:
    service = _service_with(
        qwen_output="证据说明其中一个策略提高了表达水平 [E1]",
        plan=AnswerPlan(
            mode="partial",
            reason="comparison_branch_coverage",
            covered_branches=["优化甲醇诱导时机和能量利用"],
            missing_branches=["直接增加 AOX1 启动子调控表达盒拷贝数"],
        ),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="比较两类策略差异",
        analysis=_analysis(QueryIntent.COMPARISON),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["qwen_synthesis"]["fallback_used"] is True
    assert "missing_branch_limit_not_disclosed" in result.debug["qwen_synthesis"]["validation_flags"]


def test_qwen_synthesis_preserves_comparison_partial_with_citations() -> None:
    service = _service_with(
        qwen_output="当前证据还不能逐分支完整比较，但已检索到一类策略的支持信息 [E1]",
        plan=AnswerPlan(
            mode="partial",
            reason="comparison_branch_coverage",
            covered_branches=["优化甲醇诱导时机和能量利用"],
            missing_branches=["直接增加 AOX1 启动子调控表达盒拷贝数"],
        ),
        support_pack=[_support_item()],
    )

    result = service.run(
        question="比较两类策略差异",
        analysis=_analysis(QueryIntent.COMPARISON),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["answer_mode"] == "partial"
    assert result.debug["qwen_synthesis"]["used_qwen"] is True
    assert len(result.citations) > 0


# ── Stage 2D.1: summary partial validator 测试 ──────────────────────────────

def test_validate_summary_partial_with_limit_terms_passes() -> None:
    """summary partial 输出含明确限制语 → validator 通过，partial_tone_category=summary"""
    support_pack = [_support_item("E1")]
    plan = AnswerPlan(mode="partial", reason="summary_support_count")

    ok, flags = validate_synthesized_answer(
        "当前知识库仅提供有限支持，仅检索到一条直接相关证据，因此以下总结仅基于该证据所能覆盖的范围。"
        "现有证据指出异源蛋白表达会对宿主造成代谢负担，表现为比生长速率下降 [E1]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="证据表明代谢负担存在 [E1]",
        existence_guardrail={},
    )

    assert ok is True
    assert flags == []
    details = validate_synthesized_answer.last_details
    assert len(details["partial_limit_terms_found"]) > 0
    assert details["partial_tone_decision"] == "pass"
    assert details["partial_tone_category"] == "summary"


def test_validate_summary_partial_without_limit_terms_fails() -> None:
    """summary partial 输出无限制语 → 触发 partial_abstention_tone"""
    support_pack = [_support_item("E1")]
    plan = AnswerPlan(mode="partial", reason="summary_support_count")

    ok, flags = validate_synthesized_answer(
        "该机制已被证实：氧化戊糖磷酸途径通量提高促进了 NADPH 再生，从而为重组蛋白折叠提供更多还原力 [E1]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="证据表明代谢负担存在 [E1]",
        existence_guardrail={},
    )

    assert ok is False
    assert "partial_abstention_tone" in flags
    assert validate_synthesized_answer.last_details["partial_tone_decision"] == "fail"


def test_validate_summary_partial_with_overclaim_fails() -> None:
    """summary partial 含 overclaim → validator 拒绝，即使含限制语前缀"""
    support_pack = [_support_item("E1")]
    plan = AnswerPlan(mode="partial", reason="summary_support_count")

    ok, flags = validate_synthesized_answer(
        "证据有限，但可以完整总结该机制：机制已经完整阐明，所有环节均已充分证明 [E1]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="证据表明代谢负担存在 [E1]",
        existence_guardrail={},
    )

    assert ok is False
    assert "partial_overclaim" in flags


def test_validate_summary_partial_tone_category_comparison_unchanged() -> None:
    """comparison partial 含 comparison 限制语 → partial_tone_category=comparison"""
    support_pack = [_support_item("E1"), _support_item("E2", doc_id="doc_0002")]
    plan = AnswerPlan(
        mode="partial",
        reason="branch_partial_support",
        covered_branches=["branch_a", "branch_b"],
        missing_branches=[],
        comparison_coverage=_comparison_coverage(allowed_ids=["E1", "E2"]),
    )

    ok, flags = validate_synthesized_answer(
        "在文库所支持的范围内，可进行有限比较：关于分支A，证据显示相关结果 [E1]；关于分支B，部分支持 [E2]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="根据当前知识库证据，只能进行有限比较 [E1][E2]",
        existence_guardrail={},
    )

    assert ok is True
    details = validate_synthesized_answer.last_details
    assert details["partial_tone_category"] == "comparison"


def test_validate_summary_partial_debug_has_tone_category() -> None:
    """validation_details 应包含 partial_tone_category 字段"""
    support_pack = [_support_item("E1")]
    plan = AnswerPlan(mode="partial", reason="summary_support_count")

    validate_synthesized_answer(
        "证据有限，当前知识库仅提供有限支持，现有研究结果 [E1]。",
        plan,
        support_pack,
        GenerationConfig(v2_use_qwen_synthesis=True),
        extractive_answer="[E1]",
        existence_guardrail={},
    )

    details = validate_synthesized_answer.last_details
    assert "partial_tone_category" in details


# ── Stage 2D.1: summary debug skip 测试 ─────────────────────────────────────

def test_service_summary_selection_skipped_on_existence_refuse() -> None:
    """existence/no-support refusal 场景下 summary_selection debug 应有 skipped=True"""
    from src.synbio_rag.application.generation_v2.models import EvidenceCandidate
    from src.synbio_rag.domain.schemas import QueryIntent

    service = _service_with(
        qwen_output=None,
        plan=AnswerPlan(mode="refuse", reason="existence_no_support"),
        support_pack=[],
        extractive_answer="当前知识库证据不足，无法确认。",
        existence_guardrail={"is_existence_question": True, "support_status": "no_support"},
    )

    result = service.run(
        question="文库中是否有关于 PPP 通量提升的详细方案？",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    ss = result.debug["summary_selection"]
    sp = result.debug["summary_plan"]
    assert ss.get("skipped") is True
    assert "skip_reason" in ss
    # summary_selection 与 summary_plan 不冲突：两者均应 is_summary 一致或 skipped 有解释
    if sp.get("is_summary") is False:
        assert ss.get("is_summary") is False or ss.get("skipped") is True


def test_service_answer_mode_unchanged_on_refuse() -> None:
    """debug 修复不能改变最终 answer_mode"""
    service = _service_with(
        qwen_output=None,
        plan=AnswerPlan(mode="refuse", reason="no_support_pack"),
        support_pack=[],
        extractive_answer="当前知识库不足以回答该问题。",
    )

    result = service.run(
        question="总结某机制",
        analysis=_analysis(),
        seed_chunks=[_chunk()],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["answer_mode"] == "refuse"
    assert result.debug["summary_selection"].get("skipped") is True


def test_qwen_synthesis_preserves_existence_guardrail_language() -> None:
    service = _service_with(
        qwen_output="当前文库只能提供间接或弱相关证据，不能确认文库中包含用户所要求的资料/数据/方案。现有背景证据如下 [E1]",
        plan=AnswerPlan(mode="partial", reason="existence_weak_support"),
        support_pack=[_support_item(text="L-arginine fermentation in a 5 L tank.")],
        existence_guardrail={
            "is_existence_question": True,
            "support_status": "weak",
            "target_terms": ["PHA", "工业规模", "溶解氧", "控制策略"],
            "support_reason": "core_terms_not_sufficiently_covered",
            "missing_core_terms": ["PHA", "工业规模", "控制策略"],
        },
    )

    result = service.run(
        question="文库中是否有关于工业规模 PHA 发酵工艺中溶解氧控制策略的详细方案？",
        analysis=_analysis(),
        seed_chunks=[_chunk("L-arginine fermentation in a 5 L tank.")],
        config=GenerationConfig(v2_use_qwen_synthesis=True),
    )

    assert result.debug["answer_mode"] == "partial"
    assert "不能确认文库中包含用户所要求的资料/数据/方案" in result.answer
    assert len(result.citations) == 1
