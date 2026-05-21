from __future__ import annotations

from types import SimpleNamespace

from src.synbio_rag.application.generation_v2.answer_planner import AnswerPlanner
from src.synbio_rag.application.generation_v2.branch_parser import parse_comparison_branches
from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.evidence_ledger import EvidenceLedgerBuilder
from src.synbio_rag.application.generation_v2.models import AnswerPlan, EvidenceCandidate, SupportItem
from src.synbio_rag.application.generation_v2.support_selector import SupportPackSelector
from src.synbio_rag.application.generation_v2.validator import FinalValidator
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import GenerationConfig, RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc1",
    section: str = "Results",
    rerank: float = 0.0,
    fusion: float = 0.0,
    vector: float = 0.0,
    bm25: float = 0.0,
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section=section,
        text=text,
        rerank_score=rerank,
        fusion_score=fusion,
        vector_score=vector,
        bm25_score=bm25,
        metadata=metadata or {},
    )


def _analysis(intent: QueryIntent) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, requires_external_tools=False, search_limit=10, rerank_top_k=5)


def _candidate(
    evidence_id: str,
    text: str,
    *,
    doc_id: str = "doc1",
    section: str = "Results",
    rerank: float = 0.0,
    fusion: float = 0.0,
    vector: float = 0.0,
    bm25: float = 0.0,
    features: dict | None = None,
    metadata: dict | None = None,
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
        vector_score=vector,
        bm25_score=bm25,
        rerank_score=rerank,
        fusion_score=fusion,
        metadata=metadata or {},
        features={
            "has_table_text": False,
            "has_table_caption": False,
            "has_figure_caption": False,
            "has_numeric": False,
            "has_result_terms": False,
            "section_type": section.lower(),
            "text_length": len(text),
            **(features or {}),
        },
        reasons=["seed_chunk", f"section:{section.lower()}"],
    )


def test_evidence_ledger_builder_extracts_features() -> None:
    builder = EvidenceLedgerBuilder()
    chunks = [
        _chunk(
            "c1",
            "[TABLE CAPTION] Table 2 growth yield\n[TABLE TEXT] strain A 12.4 g/L\n[FIGURE CAPTION] Figure 1 result",
            metadata={"table_text": "strain A 12.4 g/L"},
        )
    ]

    candidates = builder.build("Table 2 里有什么结果？", _analysis(QueryIntent.FACTOID), chunks)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.evidence_id == "E1"
    assert candidate.features["has_numeric"] is True
    assert candidate.features["has_table_text"] is True
    assert candidate.features["has_table_caption"] is True
    assert candidate.features["has_figure_caption"] is True
    assert candidate.features["has_result_terms"] is True
    assert "has_numeric" in candidate.reasons


def test_evidence_ledger_uses_child_focused_generation_text() -> None:
    builder = EvidenceLedgerBuilder()
    parent_text = " ".join(f"unrelated_parent_intro_{i}" for i in range(80))
    parent_text += " matched_parent_context " + " ".join(f"context_{i}" for i in range(80, 170))
    chunk = _chunk(
        "parent_chunk_1",
        parent_text,
        metadata={
            "matched_child_snippets": [
                {
                    "chunk_id": "parent_chunk_1::child003",
                    "text": "specific matched child evidence shows 12 mg/L titer in strain A",
                    "child_start_token": 90,
                    "child_end_token": 99,
                    "block_types": ["paragraph"],
                    "evidence_types": ["paragraph"],
                }
            ]
        },
    )

    candidate = builder.build("strain A titer?", _analysis(QueryIntent.FACTOID), [chunk])[0]

    assert candidate.chunk_id == "parent_chunk_1"
    assert "matched_child_evidence:" in candidate.text
    assert "specific matched child evidence shows 12 mg/L titer" in candidate.text
    assert "unrelated_parent_intro_0" not in candidate.text
    assert candidate.metadata["parent_child_generation_view_used"] is True
    assert candidate.metadata["generation_evidence_role"] == "matched_child_focused_evidence"
    assert candidate.metadata["matched_child_chunk_ids"] == ["parent_chunk_1::child003"]
    assert "parent_child_focused_evidence" in candidate.reasons


def test_evidence_ledger_child_markers_drive_table_and_figure_features() -> None:
    builder = EvidenceLedgerBuilder()
    chunk = _chunk(
        "parent_chunk_1",
        " ".join(f"parent{i}" for i in range(120)),
        metadata={
            "matched_child_snippets": [
                {
                    "chunk_id": "parent_chunk_1::child_table",
                    "text": "strain A 12.4 g/L; strain B 9.1 g/L",
                    "child_start_token": 40,
                    "child_end_token": 52,
                    "block_types": ["table_text"],
                    "evidence_types": ["table_text"],
                    "contains_table_text": True,
                },
                {
                    "chunk_id": "parent_chunk_1::child_figure",
                    "text": "Figure 2 shows the pathway overview.",
                    "child_start_token": 70,
                    "child_end_token": 78,
                    "block_types": ["figure_caption"],
                    "evidence_types": ["figure_caption"],
                    "contains_figure_caption": True,
                },
            ]
        },
    )

    candidate = builder.build(
        "Table and figure evidence?",
        _analysis(QueryIntent.FACTOID),
        [chunk],
    )[0]

    assert "[table text]" in candidate.text
    assert "[figure caption]" in candidate.text
    assert candidate.features["has_table_text"] is True
    assert candidate.features["has_figure_caption"] is True


def test_evidence_ledger_without_matched_children_keeps_parent_text_behavior() -> None:
    builder = EvidenceLedgerBuilder()
    chunk = _chunk("parent_chunk_1", "parent text with result 10 g/L")

    candidate = builder.build("result?", _analysis(QueryIntent.FACTOID), [chunk])[0]

    assert candidate.text == "parent text with result 10 g/L"
    assert candidate.metadata["parent_child_generation_view_used"] is False
    assert candidate.metadata["generation_evidence_role"] == "parent_text"


def test_support_selector_factoid_prefers_highest_score() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_factoid=1)
    candidates = [
        _candidate("E1", "low score evidence", rerank=0.2),
        _candidate("E2", "high score evidence", rerank=0.9),
    ]

    support = selector.select("高分证据是什么？", _analysis(QueryIntent.FACTOID), candidates, config)

    assert [item.evidence_id for item in support] == ["E2"]


def test_support_selector_table_query_prefers_child_focused_table_evidence() -> None:
    builder = EvidenceLedgerBuilder()
    chunks = [
        _chunk(
            "parent_table",
            " ".join(f"parent_table_context_{i}" for i in range(120)),
            rerank=0.35,
            metadata={
                "matched_child_snippets": [
                    {
                        "chunk_id": "parent_table::child001",
                        "text": "Table 2: strain A produced 12.4 g/L.",
                        "child_start_token": 50,
                        "child_end_token": 60,
                        "block_types": ["table_text"],
                        "evidence_types": ["table_text"],
                        "contains_table_text": True,
                    }
                ]
            },
        ),
        _chunk(
            "plain_parent",
            "high score paragraph without table evidence",
            doc_id="doc2",
            rerank=0.95,
        ),
    ]
    candidates = builder.build("Table 2 的产量是多少？", _analysis(QueryIntent.FACTOID), chunks)

    support = SupportPackSelector().select(
        "Table 2 的产量是多少？",
        _analysis(QueryIntent.FACTOID),
        candidates,
        GenerationConfig(v2_max_support_factoid=1),
    )

    assert support[0].candidate.chunk_id == "parent_table"
    assert "parent_child_focused_evidence" in support[0].reasons


def test_support_selector_figure_query_prefers_child_focused_figure_caption() -> None:
    builder = EvidenceLedgerBuilder()
    chunks = [
        _chunk(
            "parent_figure",
            " ".join(f"parent_figure_context_{i}" for i in range(120)),
            rerank=0.35,
            metadata={
                "matched_child_snippets": [
                    {
                        "chunk_id": "parent_figure::child001",
                        "text": "Figure 3 illustrates the engineered pathway.",
                        "child_start_token": 50,
                        "child_end_token": 60,
                        "block_types": ["figure_caption"],
                        "evidence_types": ["figure_caption"],
                        "contains_figure_caption": True,
                    }
                ]
            },
        ),
        _chunk(
            "plain_parent",
            "high score paragraph without figure evidence",
            doc_id="doc2",
            rerank=0.95,
        ),
    ]
    candidates = builder.build("Figure 3 说明了什么？", _analysis(QueryIntent.FACTOID), chunks)

    support = SupportPackSelector().select(
        "Figure 3 说明了什么？",
        _analysis(QueryIntent.FACTOID),
        candidates,
        GenerationConfig(v2_max_support_factoid=1),
    )

    assert support[0].candidate.chunk_id == "parent_figure"
    assert "parent_child_focused_evidence" in support[0].reasons


def test_support_selector_summary_prefers_doc_diversity_before_second_same_doc() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_summary=5)
    candidates = [
        _candidate("E1", "doc1 result one", doc_id="doc1", rerank=0.9),
        _candidate("E2", "doc1 result two", doc_id="doc1", rerank=0.85),
        _candidate("E3", "doc2 result three", doc_id="doc2", rerank=0.8),
        _candidate("E4", "doc3 result four", doc_id="doc3", rerank=0.75),
    ]

    support = selector.select("总结结果", _analysis(QueryIntent.SUMMARY), candidates, config)

    assert len([item for item in support if item.candidate.doc_id == "doc1"]) == 1
    assert {"doc1", "doc2"}.issubset({item.candidate.doc_id for item in support})


def test_support_selector_summary_prefers_at_least_two_eligible_items() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_summary=5)
    candidates = [
        _candidate("E1", "results evidence with yield increase and numeric 12.5 g/L", rerank=0.9, features={"has_numeric": True, "has_result_terms": True}),
        _candidate("E2", "discussion evidence explaining improved production performance", section="Discussion", rerank=0.82, features={"has_result_terms": True}),
        _candidate("E3", "author affiliation list", section="Author Information", rerank=0.99),
    ]

    support = selector.select("总结这篇论文的结果", _analysis(QueryIntent.SUMMARY), candidates, config)

    assert len(support) >= 2
    assert any("summary_min_support_fill" in item.reasons for item in support)


def test_support_selector_summary_filters_reference_and_author_noise_before_fill() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_summary=5)
    candidates = [
        _candidate("E1", "results evidence with improved titer and 8.2 g/L output", rerank=0.85, features={"has_numeric": True, "has_result_terms": True}),
        _candidate("E2", "Smith et al. 2024 reference list item", section="References", rerank=0.95),
        _candidate("E3", "author list and affiliations", section="Author Contributions", rerank=0.94),
        _candidate("E4", "paper title page only", section="Title", rerank=0.93),
    ]

    support = selector.select("总结这篇论文的结果", _analysis(QueryIntent.SUMMARY), candidates, config)

    assert {item.candidate.section for item in support} == {"Results", "Title"}
    assert all(item.candidate.section not in {"References", "Author Contributions"} for item in support)


def test_support_selector_comparison_covers_both_branches() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_comparison=4)
    candidates = [
        _candidate("E1", "A strain increased yield to 10 g/L.", rerank=0.8),
        _candidate("E2", "B strain increased yield to 8 g/L.", rerank=0.82),
    ]

    support = selector.select("比较 A 和 B 的产量", _analysis(QueryIntent.COMPARISON), candidates, config)

    selected_reasons = " ".join(" ".join(item.reasons) for item in support)
    assert "comparison_branch:A" in selected_reasons
    assert "comparison_branch:B" in selected_reasons


def test_parse_comparison_branches_supports_chinese_patterns() -> None:
    simple = parse_comparison_branches("比较 A 和 B")
    assert simple.parse_ok is True
    assert simple.branches == ["A", "B"]

    with_yu = parse_comparison_branches("比较 A 与 B")
    assert with_yu.parse_ok is True
    assert with_yu.branches == ["A", "B"]

    vs_form = parse_comparison_branches("A vs B")
    assert vs_form.parse_ok is True
    assert vs_form.branches == ["A", "B"]

    two_class = parse_comparison_branches("一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数")
    assert two_class.parse_ok is True
    assert len(two_class.branches) == 2

    nanr = parse_comparison_branches("NanR 在天然调控和工程应用中的两种角色：一类是调控 E. coli 唾液酸代谢，另一类是被改造成 Neu5Ac 传感器")
    assert nanr.parse_ok is True
    assert len(nanr.branches) == 2


def test_parse_comparison_branches_rejects_generic_or_too_long_branches() -> None:
    generic = parse_comparison_branches("两种角色：一种是策略，另一种是机制")
    assert generic.parse_ok is False

    too_long = parse_comparison_branches(
        "两种方案：一种是" + "非常长的描述" * 30 + "，另一种是提高产量"
    )
    assert too_long.parse_ok is False


def test_answer_planner_refuses_when_support_pack_empty() -> None:
    planner = AnswerPlanner()

    plan = planner.plan("这个问题没有证据吗？", _analysis(QueryIntent.FACTOID), [])

    assert plan.mode == "refuse"


def test_answer_planner_comparison_stays_partial_with_meaningful_support() -> None:
    planner = AnswerPlanner()
    support_pack = [
        SupportItem(
            evidence_id="E1",
            candidate=_candidate(
                "E1",
                "This study describes methanol induction timing and ATP balance during Pichia expression.",
                doc_id="doc_0074",
                section="Results and Discussion",
                rerank=0.8,
                features={"has_numeric": True, "has_result_terms": True},
            ),
            support_score=0.8,
            reasons=["comparison_selection"],
        )
    ]

    plan = planner.plan(
        "比较 Pichia 中两类提升表达的策略差异：一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数。",
        _analysis(QueryIntent.COMPARISON),
        support_pack,
    )

    assert plan.mode == "partial"
    assert plan.reason == "comparison_branch_coverage"
    assert plan.missing_branches


def test_citation_binder_maps_internal_ids_from_support_pack_only() -> None:
    binder = CitationBinder()
    support_pack = [
        SupportItem(
            evidence_id="E1",
            candidate=_candidate("E1", "evidence one text"),
            support_score=0.9,
            reasons=["picked"],
        )
    ]

    answer, citations, debug = binder.bind("答案引用 [E1]，忽略 [E9]。", support_pack)

    assert "[1]" in answer
    assert "[E9]" not in answer
    assert len(citations) == 1
    assert citations[0].chunk_id == "e1_chunk"
    assert debug["invalid_evidence_ids"] == ["E9"]


def test_citation_binder_keeps_parent_chunk_id_and_quotes_child_focused_text() -> None:
    binder = CitationBinder()
    candidate = EvidenceCandidate(
        evidence_id="E1",
        chunk_id="parent_chunk_1",
        doc_id="doc1",
        source_file="doc1.pdf",
        title="title-doc1",
        section="Results",
        text=(
            "matched_child_evidence:\n"
            "matched child: parent_chunk_1::child002 [table text]\n"
            "child table evidence 12 mg/L"
        ),
        page_start=3,
        page_end=3,
        vector_score=0.1,
        bm25_score=0.1,
        rerank_score=0.8,
        fusion_score=0.7,
        metadata={
            "parent_child_generation_view_used": True,
            "matched_child_chunk_ids": ["parent_chunk_1::child002"],
            "generation_evidence_role": "matched_child_focused_evidence",
            "parent_text_preview": "unrelated parent opening",
        },
        features={},
        reasons=["seed_chunk", "parent_child_focused_evidence"],
    )
    support_pack = [SupportItem("E1", candidate, 0.9, ["picked"])]

    answer, citations, debug = binder.bind("答案引用 [E1]。", support_pack)

    assert "[1]" in answer
    assert citations[0].chunk_id == "parent_chunk_1"
    assert "child table evidence 12 mg/L" in citations[0].quote
    assert "unrelated parent opening" not in citations[0].quote
    citation_debug = debug["citation_candidates"][0]
    assert citation_debug["parent_child_generation_view_used"] is True
    assert citation_debug["matched_child_chunk_ids"] == ["parent_chunk_1::child002"]
    assert citation_debug["generation_evidence_role"] == "matched_child_focused_evidence"


def test_final_validator_refuses_non_refuse_answer_without_citation() -> None:
    validator = FinalValidator()
    plan = AnswerPlan(mode="full", reason="supported", allowed_scope=["supported_facts"])
    support_pack = [
        SupportItem(
            evidence_id="E1",
            candidate=_candidate("E1", "text"),
            support_score=0.8,
            reasons=["picked"],
        )
    ]

    answer, validated_plan, debug = validator.validate(
        "根据证据回答。",
        [],
        plan,
        support_pack,
        GenerationConfig(v2_require_citation=True),
    )

    assert validated_plan.mode == "refuse"
    assert "证据不足" in answer
    assert debug["zero_citation_guardrail_triggered"] is True


def test_pipeline_uses_generation_v2_flow() -> None:
    pipeline = SynBioRAGPipeline.__new__(SynBioRAGPipeline)
    pipeline.settings = Settings(
        retrieval=RetrievalConfig(final_top_k=2),
        generation=GenerationConfig(),
    )
    pipeline.router = SimpleNamespace(analyze=lambda question: _analysis(QueryIntent.FACTOID))
    retrieved = [_chunk("c1", "seed text", rerank=0.9)]
    pipeline._search_with_filter_fallback = lambda **kwargs: (retrieved, {"selected": "original", "attempts": []})
    pipeline._rewrite_svc = SimpleNamespace(
        rewrite=lambda question, is_negative=False: (
            question,
            SimpleNamespace(to_dict=lambda: {}, query_rewrite_mode="off"),
        )
    )
    pipeline.reranker = SimpleNamespace(rerank=lambda *args, **kwargs: retrieved, last_debug={"rerank": True})
    pipeline.retriever = SimpleNamespace(last_debug={"retrieval": True})

    pipeline.confidence_scorer = SimpleNamespace(score=lambda chunks: 0.77, needs_external_tool=lambda confidence: False)
    pipeline.generator_v2 = SimpleNamespace(
        run=lambda **kwargs: SimpleNamespace(
            answer="根据当前知识库证据：[1]",
            citations=[],
            debug={"generation_version": "v2"},
        )
    )

    response = pipeline.answer("问题是什么？")

    assert response.debug["neighbor_expansion"]["reason"] == "generation_v2_seed_only_or_replaced_by_parent_expansion"
    assert response.debug["generation_v2"]["generation_version"] == "v2"


def test_generation_config_is_v2_only() -> None:
    settings = Settings()
    assert not hasattr(settings.generation, "version")
