from __future__ import annotations

from src.synbio_rag.application.generation_service import QwenChatGenerator
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc_0045",
    section: str = "Results",
    rerank: float = 0.8,
    vector: float = 0.7,
    fusion: float = 0.75,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=doc_id,
        section=section,
        text=text,
        rerank_score=rerank,
        vector_score=vector,
        fusion_score=fusion,
    )


def test_table_text_survives_support_pack_selection() -> None:
    generator = QwenChatGenerator()
    analysis = QueryAnalysis(QueryIntent.FACTOID, False, 20, 10)
    chunks = [
        _chunk("body", "General discussion paragraph about glycan engineering and phenotypes.", rerank=0.92),
        _chunk(
            "table",
            "[TABLE CAPTION] Table 5 | Oligonucleotide primer list\n"
            "[TABLE TEXT] OCH1-5int-fw1 GAACTGTGTAACCTTTTAAATGACGGGATCTAAATACGTCATG\n"
            "[TABLE TEXT] OCH1-5int-rv1 CTATTCTCTAGAAAGTATAGGAACTTCGGCTGATGATATTTGCTACGAACACTG",
            section="Methods",
            rerank=0.80,
        ),
    ]

    assessment = generator.assess_evidence(
        "In doc_0045 Table 5, what primer sequences are listed for OCH1-5int-fw1 and OCH1-5int-rv1?",
        chunks,
        analysis=analysis,
    )

    assert assessment.support_pack
    assert assessment.support_pack[0]["chunk_id"] == "table"
    assert "table_text" in assessment.support_pack[0]["block_types"]
    assert "[TABLE TEXT]" in assessment.prompt_support_context


def test_explicit_doc_anchor_keeps_figure_support_in_correct_doc() -> None:
    generator = QwenChatGenerator()
    analysis = QueryAnalysis(QueryIntent.FACTOID, False, 20, 10)
    chunks = [
        _chunk(
            "wrong",
            "[FIGURE CAPTION] Fig. 4. Unrelated figure from another document.",
            doc_id="doc_0060",
            rerank=0.95,
        ),
        _chunk(
            "right",
            "[FIGURE CAPTION] Fig. 4. Microbial composition changes among the top 20 species after fermentation.",
            doc_id="doc_0001",
            section="3. Results and discussion",
            rerank=0.72,
        ),
    ]

    assessment = generator.assess_evidence(
        "What does Figure 4 describe in doc_0001?",
        chunks,
        analysis=analysis,
    )

    assert assessment.support_pack
    assert assessment.support_pack[0]["doc_id"] == "doc_0001"
    assert "figure_caption" in assessment.support_pack[0]["block_types"]


def test_extraction_gap_table_query_refuses_without_matching_table_reference() -> None:
    generator = QwenChatGenerator()
    analysis = QueryAnalysis(QueryIntent.FACTOID, False, 20, 10)
    chunks = [
        _chunk("body1", "Discussion paragraph mentioning strain-specific parameters in general terms.", rerank=0.91),
        _chunk("body2", "Results paragraph describing altered phenotype and growth profile.", rerank=0.88),
    ]

    assessment = generator.assess_evidence(
        "What strain-specific parameters are shown in doc_0045 Table 2?",
        chunks,
        analysis=analysis,
    )

    assert assessment.should_refuse_final is True
    assert assessment.refusal_reason == "missing_structured_reference"
    answer = generator.generate(
        "What strain-specific parameters are shown in doc_0045 Table 2?",
        "",
        chunks,
        analysis=analysis,
        assessment=assessment,
    )
    assert "没有稳定匹配 table 2 的表格正文" in answer.lower()


def test_body_query_is_not_forced_to_use_structured_support() -> None:
    generator = QwenChatGenerator()
    analysis = QueryAnalysis(QueryIntent.FACTOID, False, 20, 10)
    chunks = [
        _chunk(
            "table",
            "[TABLE TEXT] random strain values",
            section="Results",
            rerank=0.82,
        ),
        _chunk(
            "body",
            "### 2.1. Materials Six commercially available HMOs were supplied by Glycom A/S.",
            doc_id="doc_0001",
            section="2. Materials and methods",
            rerank=0.90,
        ),
    ]

    assessment = generator.assess_evidence(
        "What materials were used in the HMO fermentation study in doc_0001?",
        chunks,
        analysis=analysis,
    )

    assert assessment.support_pack[0]["chunk_id"] == "body"

