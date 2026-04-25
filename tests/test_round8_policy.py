from src.synbio_rag.application import generation_service
from src.synbio_rag.application.generation_service import QwenChatGenerator
from src.synbio_rag.domain.config import Round8PolicyConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


def _chunk(
    text: str = "Engineered E. coli strains were compared under different culture conditions. "
    "The results showed higher titer and yield in the optimized condition.",
    doc_id: str = "doc_001",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}_chunk_1",
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title="Synthetic biology comparison study",
        section="Results",
        text=text,
        vector_score=score,
        rerank_score=score,
        fusion_score=score,
    )


def _comparison_analysis() -> QueryAnalysis:
    return QueryAnalysis(
        intent=QueryIntent.COMPARISON,
        requires_external_tools=False,
        search_limit=10,
        rerank_top_k=5,
    )


def test_round8_flag_disables_comparison_single_doc_hard_refusal() -> None:
    generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            disable_comparison_single_doc_hard_refusal=True,
        )
    )

    assessment = generator.assess_evidence(
        "比较优化条件和对照条件下 engineered E. coli 的 titer 与 yield。",
        [_chunk()],
        analysis=_comparison_analysis(),
    )

    assert assessment.hard_refusal_reason == "comparison_single_doc"
    assert assessment.should_refuse_original is True
    assert assessment.hard_refusal_disabled_by_round8 is True
    assert assessment.should_refuse_final is False
    assert assessment.final_answer_mode == "full"


def test_round7_default_keeps_comparison_single_doc_hard_refusal() -> None:
    generator = QwenChatGenerator()

    assessment = generator.assess_evidence(
        "比较优化条件和对照条件下 engineered E. coli 的 titer 与 yield。",
        [_chunk()],
        analysis=_comparison_analysis(),
    )

    assert assessment.hard_refusal_reason == "comparison_single_doc"
    assert assessment.should_refuse_original is True
    assert assessment.hard_refusal_disabled_by_round8 is False
    assert assessment.should_refuse_final is True
    assert assessment.final_answer_mode == "refuse"


def test_round8_limited_partial_compare_recovers_partial_branch_coverage() -> None:
    generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            disable_comparison_single_doc_hard_refusal=True,
            enable_partial_answer=True,
        )
    )

    assessment = generator.assess_evidence(
        "比较 6'-SL 和 2'-FL 的合成路径。",
        [_chunk(text="6'-SL synthesis used Neu5Ac and a sialyltransferase strategy.")],
        analysis=_comparison_analysis(),
    )

    assert assessment.hard_refusal_reason == "comparison_branch_coverage"
    assert assessment.hard_refusal_disabled_by_round8 is False
    assert assessment.should_refuse_final is False
    assert assessment.final_answer_mode == "limited_partial_compare"
    assert assessment.limited_partial_compare_triggered is True
    assert assessment.covered_branch_count == 1
    assert assessment.missing_branch_count == 1
    answer = generator.generate(
        "比较 6'-SL 和 2'-FL 的合成路径。",
        "",
        [_chunk(text="6'-SL synthesis used Neu5Ac and a sialyltransferase strategy.")],
        analysis=_comparison_analysis(),
        assessment=assessment,
    )
    assert "有限比较结论" in answer
    assert "可以确认" in answer
    assert "证据限制" in answer


def test_round8_claim_fallback_recovers_factoid_with_supported_sentence() -> None:
    generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            enable_claim_fallback=True,
            enable_partial_answer=True,
        )
    )
    analysis = QueryAnalysis(
        intent=QueryIntent.FACTOID,
        requires_external_tools=False,
        search_limit=10,
        rerank_top_k=5,
    )

    assessment = generator.assess_evidence(
        "CRISPR-TMSD 的检测下限是多少 copy/uL？",
        [
            _chunk(
                text=(
                    "The CRISPR-TMSD assay combines Cas12a cis-cleavage with a TMSD reporter. "
                    "After RPA amplification, CRISPR-TMSD reached a detection limit of 1 copy/uL."
                ),
                doc_id="doc_017",
                score=0.2,
            )
        ],
        analysis=analysis,
    )

    assert assessment.should_refuse_original is True
    assert assessment.recovered_by_claim_fallback is True
    assert assessment.should_refuse_final is False
    assert assessment.final_answer_mode == "concise"
    assert assessment.supported_claim_count >= 1
    assert assessment.fallback_method == "extractive_keyword_entity_parent_score"


def test_round8_claim_fallback_does_not_recover_unrelated_evidence() -> None:
    generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            enable_claim_fallback=True,
            enable_partial_answer=True,
        )
    )
    analysis = QueryAnalysis(
        intent=QueryIntent.FACTOID,
        requires_external_tools=False,
        search_limit=10,
        rerank_top_k=5,
    )

    assessment = generator.assess_evidence(
        "CAR-T 治疗弥漫大 B 细胞淋巴瘤 III 期临床试验有什么结论？",
        [
            _chunk(
                text="Engineered E. coli produced 2'-FL through fucosyltransferase pathway optimization.",
                doc_id="doc_011",
                score=0.2,
            )
        ],
        analysis=analysis,
    )

    assert assessment.should_refuse_original is True
    assert assessment.recovered_by_claim_fallback is False
    assert assessment.should_refuse_final is True
    assert assessment.evidence_unit_count == 0


def test_round8_route_specific_thresholds_relax_comparison_claim_fallback(monkeypatch) -> None:
    question = "比较 2'-FL 和 6'-SL 的合成路径差异。"
    chunks = [
        _chunk(
            text="2'-FL synthesis pathway evidence.",
            doc_id="doc_2fl",
            score=0.2,
        ),
        _chunk(
            text="6'-SL synthesis pathway evidence.",
            doc_id="doc_6sl",
            score=0.2,
        ),
    ]
    monkeypatch.setattr(generation_service, "_term_overlap", lambda left, right: 0.55)
    monkeypatch.setattr(generation_service, "_entity_overlap", lambda left, right: 0.05)

    round82_generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            disable_comparison_single_doc_hard_refusal=True,
            enable_claim_fallback=True,
            enable_partial_answer=True,
        )
    )
    round83_generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            disable_comparison_single_doc_hard_refusal=True,
            enable_claim_fallback=True,
            enable_partial_answer=True,
            enable_route_specific_thresholds=True,
        )
    )

    assessment_round82 = round82_generator.assess_evidence(
        question,
        chunks,
        analysis=_comparison_analysis(),
    )
    assessment_round83 = round83_generator.assess_evidence(
        question,
        chunks,
        analysis=_comparison_analysis(),
    )

    assert assessment_round82.should_refuse_original is True
    assert assessment_round82.recovered_by_claim_fallback is False
    assert assessment_round82.supported_claim_count == 0

    assert assessment_round83.should_refuse_original is True
    assert assessment_round83.recovered_by_claim_fallback is True
    assert assessment_round83.supported_claim_count >= 2
    assert assessment_round83.should_refuse_final is False


def test_empty_support_pack_triggers_refusal_guardrail(monkeypatch) -> None:
    generator = QwenChatGenerator(
        round8_config=Round8PolicyConfig(
            enable_round8_policy=True,
            disable_comparison_single_doc_hard_refusal=True,
        )
    )
    monkeypatch.setattr(generation_service, "build_support_pack", lambda top_contexts, evidence_units, citation_candidates: [])

    assessment = generator.assess_evidence(
        "解释 engineered E. coli 优化条件下的产量变化。",
        [_chunk(doc_id="doc_guardrail", score=0.9)],
        analysis=QueryAnalysis(
            intent=QueryIntent.SUMMARY,
            requires_external_tools=False,
            search_limit=10,
            rerank_top_k=5,
        ),
    )

    assert assessment.should_refuse_final is True
    assert assessment.final_answer_mode == "refuse"
    assert assessment.refusal_reason == "empty_support_refuse"
    assert assessment.empty_support_pack_guardrail_triggered is True


def test_zero_citation_after_generation_replaced_with_refusal() -> None:
    generator = QwenChatGenerator()
    assessment = generation_service.EvidenceAssessment(
        level="strong",
        citation_count=2,
        unique_doc_count=2,
        score_strength=0.8,
        keyword_overlap=0.6,
        branch_coverage=0.8,
        negative_signal=0.0,
        should_refuse=False,
        partial_only=False,
        reason="",
        should_refuse_final=False,
        final_answer_mode="full",
        support_pack_count=1,
    )

    answer = generator.validate_generated_answer("这是一个没有 citation 的回答。", [], assessment)

    assert answer == "当前检索结果没有提供足够可引用证据，无法基于文库可靠回答。"
    assert assessment.final_answer_mode == "refuse"
    assert assessment.should_refuse_final is True
    assert assessment.refusal_reason == "post_generation_zero_citation_refuse"
    assert assessment.zero_citation_guardrail_triggered is True


def test_missing_branch_claims_are_sanitized() -> None:
    generator = QwenChatGenerator()
    assessment = generation_service.EvidenceAssessment(
        level="partial",
        citation_count=2,
        unique_doc_count=1,
        score_strength=0.8,
        keyword_overlap=0.6,
        branch_coverage=0.5,
        negative_signal=0.0,
        should_refuse=False,
        partial_only=False,
        reason="",
        should_refuse_final=False,
        final_answer_mode="limited_partial_compare",
        missing_branch_labels=["骨桥蛋白磷酸化"],
        support_pack_count=1,
    )

    answer = generator.validate_generated_answer(
        "有限比较结论：可以确认天然矿化调控相关证据。骨桥蛋白磷酸化会提高矿化效率。",
        [
            generation_service.Citation(
                chunk_id="c1",
                doc_id="d1",
                title="t",
                source_file="d1.pdf",
                section="Results",
                page_start=None,
                page_end=None,
                score=0.8,
                quote="q",
            )
        ],
        assessment,
    )

    assert "骨桥蛋白磷酸化 当前没有足够可引用证据" in answer
    assert assessment.unsupported_branch_claim_count == 0
    assert assessment.missing_branch_disclosed is True
