from __future__ import annotations

from src.synbio_rag.application.generation_v2.answer_planner import AnswerPlanner
from src.synbio_rag.application.generation_v2.support_selector import SupportPackSelector
from src.synbio_rag.domain.config import GenerationConfig
from src.synbio_rag.domain.schemas import QueryIntent

from tests.test_generation_v2 import _analysis, _candidate


def test_summary_selector_prefers_multiple_qualified_evidence() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_summary=5)
    candidates = [
        _candidate(
            "E1",
            "Results showed yield increased to 12.5 g/L and production improved across replicates.",
            section="Results",
            doc_id="doc1",
            rerank=0.9,
            features={"has_numeric": True, "has_result_terms": True},
        ),
        _candidate(
            "E2",
            "Discussion connected the improved production to pathway balancing and fermentation stability.",
            section="Discussion",
            doc_id="doc2",
            rerank=0.85,
            features={"has_result_terms": True},
        ),
        _candidate(
            "E3",
            "Abstract summarized higher titer and better pathway performance in the engineered host.",
            section="Abstract",
            doc_id="doc3",
            rerank=0.8,
            features={"has_result_terms": True},
        ),
    ]

    support = selector.select("总结该主题的主要结果", _analysis(QueryIntent.SUMMARY), candidates, config)
    debug = selector.last_summary_selection_debug

    assert len(support) >= 2
    assert debug["qualified_count"] >= 2
    assert any("summary_selection" in item.reasons for item in support)
    assert any("summary_min_support_fill" in item.reasons for item in support)


def test_summary_selector_keeps_single_qualified_evidence_without_noise_fill() -> None:
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_summary=5)
    candidates = [
        _candidate(
            "E1",
            "Results showed improved titer to 8.2 g/L and increased production yield in fermentation.",
            section="Results",
            rerank=0.84,
            features={"has_numeric": True, "has_result_terms": True},
        ),
        _candidate("E2", "Smith et al. 2024 reference list item", section="References", rerank=0.98),
        _candidate("E3", "author affiliations", section="Author Information", rerank=0.95),
        _candidate("E4", "short note", section="Discussion", rerank=0.92),
        _candidate("E5", "generic background without topic alignment", section="Methods", rerank=0.2),
    ]

    support = selector.select("总结这篇论文的主要结果", _analysis(QueryIntent.SUMMARY), candidates, config)
    debug = selector.last_summary_selection_debug

    assert len(support) == 1
    assert support[0].evidence_id == "E1"
    assert "insufficient_qualified_summary_support" in support[0].reasons
    assert debug["insufficient_qualified_summary_support"] is True
    assert all(item.evidence_id != "E2" for item in support)


def test_summary_selector_excludes_high_scoring_references() -> None:
    selector = SupportPackSelector()
    candidates = [
        _candidate(
            "E1",
            "Results showed 10.3 g/L production with improved yield.",
            section="Results",
            rerank=0.8,
            features={"has_numeric": True, "has_result_terms": True},
        ),
        _candidate("E2", "Reference entry with DOI and author names", section="References", rerank=0.99),
    ]

    support = selector.select("总结主要结果", _analysis(QueryIntent.SUMMARY), candidates, GenerationConfig())

    assert [item.evidence_id for item in support] == ["E1"]


def test_summary_selector_prefers_doc_diversity_when_multiple_docs_qualify() -> None:
    selector = SupportPackSelector()
    candidates = [
        _candidate(
            "E1",
            "Results showed pathway balancing increased yield to 11.2 g/L.",
            section="Results",
            doc_id="doc1",
            rerank=0.95,
            features={"has_numeric": True, "has_result_terms": True},
        ),
        _candidate(
            "E2",
            "Discussion connected expression tuning to secretion gains in a separate study.",
            section="Discussion",
            doc_id="doc2",
            rerank=0.9,
            features={"has_result_terms": True},
        ),
        _candidate(
            "E3",
            "Another doc1 result chunk with overlapping pathway balancing details and the same yield trend.",
            section="Results",
            doc_id="doc1",
            rerank=0.88,
            features={"has_result_terms": True},
        ),
    ]

    support = selector.select("总结该主题的证据", _analysis(QueryIntent.SUMMARY), candidates, GenerationConfig(v2_max_support_summary=5))

    assert len({item.candidate.doc_id for item in support[:2]}) >= 2


def test_summary_selector_allows_multiple_same_doc_evidence_when_only_one_doc_qualifies() -> None:
    selector = SupportPackSelector()
    candidates = [
        _candidate(
            "E1",
            "Results showed secretion yield increased to 10.1 g/L after HAC1 overexpression.",
            section="Results",
            doc_id="doc1",
            rerank=0.92,
            features={"has_numeric": True, "has_result_terms": True},
        ),
        _candidate(
            "E2",
            "Discussion explained why HAC1 overexpression improved folding capacity and secretion.",
            section="Discussion",
            doc_id="doc1",
            rerank=0.88,
            features={"has_result_terms": True},
        ),
        _candidate(
            "E3",
            "Abstract summarized the secretion enhancement and improved protein folding outcome.",
            section="Abstract",
            doc_id="doc1",
            rerank=0.8,
            features={"has_result_terms": True},
        ),
    ]

    support = selector.select("总结这篇论文的结果", _analysis(QueryIntent.SUMMARY), candidates, GenerationConfig(v2_max_support_summary=5))

    assert len(support) >= 2
    assert len({item.candidate.doc_id for item in support}) == 1
    assert any("summary_same_doc_allowed" in item.reasons for item in support)


def test_summary_planner_marks_abstract_only_as_partial() -> None:
    planner = AnswerPlanner()
    support_pack = [
        selector_item
        for selector_item in SupportPackSelector().select(
            "总结该主题",
            _analysis(QueryIntent.SUMMARY),
            [
                _candidate("E1", "Abstract summarized improved titer and growth.", section="Abstract", doc_id="doc1", rerank=0.8, features={"has_result_terms": True}),
                _candidate("E2", "Abstract summarized another improved phenotype.", section="Abstract", doc_id="doc2", rerank=0.78, features={"has_result_terms": True}),
            ],
            GenerationConfig(v2_max_support_summary=5),
        )
    ]

    plan = planner.plan("总结该主题", _analysis(QueryIntent.SUMMARY), support_pack)

    assert plan.mode == "partial"
    assert plan.reason == "summary_abstract_only"


def test_summary_planner_respects_support_count_and_quality() -> None:
    planner = AnswerPlanner()

    refuse_plan = planner.plan("总结该主题", _analysis(QueryIntent.SUMMARY), [])
    assert refuse_plan.mode == "refuse"
    assert refuse_plan.reason == "summary_no_support"

    single_support = [
        SupportPackSelector().select(
            "总结该主题",
            _analysis(QueryIntent.SUMMARY),
            [
                _candidate(
                    "E1",
                    "Results showed improved titer to 7.8 g/L with better production.",
                    section="Results",
                    doc_id="doc1",
                    rerank=0.9,
                    features={"has_numeric": True, "has_result_terms": True},
                )
            ],
            GenerationConfig(v2_max_support_summary=5),
        )[0]
    ]
    partial_plan = planner.plan("总结该主题", _analysis(QueryIntent.SUMMARY), single_support)
    assert partial_plan.mode == "partial"
    assert partial_plan.reason == "summary_support_count"

    selector = SupportPackSelector()
    multi_support = selector.select(
        "总结该主题的主要结果",
        _analysis(QueryIntent.SUMMARY),
        [
            _candidate("E1", "Results showed 12.5 g/L yield and stronger production.", section="Results", doc_id="doc1", rerank=0.92, features={"has_numeric": True, "has_result_terms": True}),
            _candidate("E2", "Discussion connected pathway balancing to the improved phenotype.", section="Discussion", doc_id="doc2", rerank=0.87, features={"has_result_terms": True}),
        ],
        GenerationConfig(v2_max_support_summary=5),
    )
    full_or_partial_plan = planner.plan("总结该主题的主要结果", _analysis(QueryIntent.SUMMARY), multi_support)
    assert full_or_partial_plan.reason in {"summary_multi_support", "summary_single_doc_limited", "summary_low_diversity", "summary_abstract_only"}
    assert full_or_partial_plan.mode in {"full", "partial"}


def test_summary_changes_do_not_affect_other_intents() -> None:
    selector = SupportPackSelector()
    factoid_support = selector.select(
        "哪个证据分数最高？",
        _analysis(QueryIntent.FACTOID),
        [
            _candidate("E1", "low score evidence", rerank=0.2),
            _candidate("E2", "high score evidence", rerank=0.9),
        ],
        GenerationConfig(v2_max_support_factoid=1),
    )

    assert [item.evidence_id for item in factoid_support] == ["E2"]
