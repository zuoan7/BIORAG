from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synbio_rag.application.generation_v2.answer_builder import ExtractiveAnswerBuilder
from synbio_rag.application.generation_v2.models import AnswerPlan, EvidenceCandidate, SupportItem
from synbio_rag.domain.schemas import QueryAnalysis, QueryIntent


def test_factoid_uses_claim_citation_lines_without_matched_child_debug() -> None:
    answer = ExtractiveAnswerBuilder().build(
        question="What changed after treatment?",
        analysis=_analysis(QueryIntent.FACTOID),
        plan=AnswerPlan(mode="full", reason=""),
        support_pack=[
            _support_item(
                "E1",
                text=(
                    "matched_child_evidence: matched child: "
                    "doc_001_sec02_chunk03::child004 The treated group decreased "
                    "from baseline after 12 weeks."
                ),
            )
        ],
    )

    assert "matched_child_evidence" not in answer
    assert "matched child:" not in answer
    assert "证据显示" not in answer
    assert "结果：The treated group decreased from baseline after 12 weeks. [E1]" in answer


def test_factoid_keeps_one_citation_marker_per_support_claim() -> None:
    answer = ExtractiveAnswerBuilder().build(
        question="What did the paper report?",
        analysis=_analysis(QueryIntent.FACTOID),
        plan=AnswerPlan(mode="full", reason=""),
        support_pack=[
            _support_item("E1", text="The first result was statistically significant."),
            _support_item("E2", text="The second result was not statistically significant."),
        ],
    )

    assert "[E1]" in answer
    assert "[E2]" in answer
    assert answer.count("[E") == 2


def test_experiment_limit_is_separate_from_evidence_claim() -> None:
    answer = ExtractiveAnswerBuilder().build(
        question="Give me a protocol.",
        analysis=_analysis(QueryIntent.EXPERIMENT),
        plan=AnswerPlan(mode="full", reason=""),
        support_pack=[_support_item("E1", text="The paper used a dilution rate of 0.1 h-1.")],
    )

    lines = answer.splitlines()
    assert lines[0].startswith("使用限制：")
    assert "[E1]" not in lines[0]
    assert any(line.endswith("[E1]") for line in lines[1:])


def test_summary_path_keeps_structured_summary_claims() -> None:
    answer = ExtractiveAnswerBuilder().build(
        question="Summarize the paper.",
        analysis=_analysis(QueryIntent.SUMMARY),
        plan=AnswerPlan(mode="full", reason=""),
        support_pack=[
            _support_item(
                "E1",
                section="Abstract",
                text="This study reports a supported summary claim. It includes enough detail.",
            )
        ],
    )

    assert "根据当前知识库证据，可作如下总结：" in answer
    assert "- 摘要:" in answer
    assert "[E1]" in answer


def _analysis(intent: QueryIntent) -> QueryAnalysis:
    return QueryAnalysis(
        intent=intent,
        requires_external_tools=False,
        search_limit=5,
        rerank_top_k=5,
    )


def _support_item(
    evidence_id: str,
    *,
    text: str,
    section: str = "Results",
) -> SupportItem:
    return SupportItem(
        evidence_id=evidence_id,
        candidate=EvidenceCandidate(
            evidence_id=evidence_id,
            chunk_id=f"doc_001_sec02_chunk03_{evidence_id}",
            doc_id="doc_001",
            source_file="doc_001.pdf",
            title="A paper",
            section=section,
            text=text,
            page_start=None,
            page_end=None,
            vector_score=0.0,
            bm25_score=0.0,
            rerank_score=1.0,
            fusion_score=0.0,
        ),
        support_score=1.0,
        reasons=["test"],
    )
