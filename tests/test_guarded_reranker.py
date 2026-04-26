from __future__ import annotations

from src.synbio_rag.application.rerank_service import _apply_guarded_rerank, _apply_rank1_evidence_guard
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    fusion: float,
    rerank: float,
    vector: float = 0.0,
    bm25: float = 0.0,
    doc_id: str = "doc_0045",
    section: str = "Results",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title="Test",
        section=section,
        text=text,
        vector_score=vector,
        bm25_score=bm25,
        fusion_score=fusion,
        rerank_score=rerank,
    )


def test_guarded_reranker_prefers_complete_table_evidence():
    config = RetrievalConfig()
    chunks = [
        _chunk(
            "table",
            "[TABLE CAPTION] Table 3 | Relative peak areas of identified glycan structures\n"
            "[TABLE TEXT] Man8 0.0 57.3 PpFWK3HRP",
            fusion=0.92,
            rerank=0.45,
            vector=0.70,
        ),
        _chunk(
            "paragraph",
            "The glycan profile discussion mentions Man8 as a major structure in the results, "
            "but does not list the PpFWK3HRP value explicitly.",
            fusion=0.65,
            rerank=0.95,
            vector=0.75,
        ),
    ]

    ranked = _apply_guarded_rerank(
        "In doc_0045 Table 3, what is the relative peak area of Man8 in PpFWK3HRP?",
        chunks,
        config,
    )

    assert ranked[0].chunk_id == "table"
    assert ranked[0].metadata["guarded_keyword_completeness"] > ranked[1].metadata["guarded_keyword_completeness"]
    assert ranked[0].metadata["guarded_marker_score"] > ranked[1].metadata["guarded_marker_score"]


def test_guarded_reranker_does_not_force_table_chunk_for_body_query():
    config = RetrievalConfig()
    chunks = [
        _chunk(
            "table",
            "[TABLE TEXT] PpMutSHRP 2.40 3.07",
            fusion=0.80,
            rerank=0.70,
            vector=0.75,
        ),
        _chunk(
            "body",
            "2.1. Materials. Six commercially available HMOs were supplied by Glycom A/S.",
            fusion=0.82,
            rerank=0.90,
            vector=0.83,
            doc_id="doc_0001",
            section="Materials",
        ),
    ]

    ranked = _apply_guarded_rerank(
        "What materials were used in the HMO fermentation study in doc_0001?",
        chunks,
        config,
    )

    assert ranked[0].chunk_id == "body"
    assert ranked[0].metadata["guarded_marker_score"] == 0.0


def test_rank1_guard_promotes_matching_figure_caption():
    config = RetrievalConfig()
    chunks = [
        _chunk(
            "paragraph",
            "Results paragraph discussing Figure 1 and pathway overview without the full legend.",
            fusion=0.88,
            rerank=0.92,
            vector=0.81,
        ),
        _chunk(
            "figure",
            "[FIGURE CAPTION] Figure 1. Overview of the engineered glycosylation pathway in P. pastoris.",
            fusion=0.83,
            rerank=0.74,
            vector=0.76,
        ),
    ]

    guarded = _apply_guarded_rerank("In doc_0045 Figure 1, what does Figure 1 describe?", chunks, config)
    assert guarded[0].chunk_id == "paragraph"

    protected = _apply_rank1_evidence_guard(guarded, config)
    assert protected[0].chunk_id == "figure"
    assert protected[0].metadata["guarded_rank1_guard_triggered"] is True
    assert protected[0].metadata["guarded_rank1_guard_reason"] == "promoted_complete_evidence"


def test_rank1_guard_promotes_complete_table_over_marker_only_table():
    config = RetrievalConfig()
    chunks = [
        _chunk(
            "marker_only",
            "[TABLE TEXT] Table 6 strain list for engineered strains.",
            fusion=0.91,
            rerank=0.85,
            vector=0.82,
        ),
        _chunk(
            "complete",
            "[TABLE CAPTION] Table 6 | P. pastoris strains\n"
            "[TABLE TEXT] strain name PpMutS PpFWK3 PpMutSHRP PpFWK3HRP",
            fusion=0.84,
            rerank=0.73,
            vector=0.78,
        ),
    ]

    guarded = _apply_guarded_rerank("In doc_0045 Table 6, which Pichia pastoris strains are listed?", chunks, config)
    assert guarded[0].chunk_id == "marker_only"

    protected = _apply_rank1_evidence_guard(guarded, config)
    assert protected[0].chunk_id == "complete"
    assert protected[0].metadata["guarded_rank1_guard_triggered"] is True


def test_rank1_guard_keeps_top1_when_no_complete_evidence_exists():
    config = RetrievalConfig()
    chunks = [
        _chunk(
            "paragraph",
            "Discussion paragraph mentioning Table 2 strain-specific parameters in general terms.",
            fusion=0.89,
            rerank=0.93,
            vector=0.84,
        ),
        _chunk(
            "weak_table",
            "[TABLE TEXT] Table 2 parameter summary.",
            fusion=0.80,
            rerank=0.71,
            vector=0.74,
        ),
    ]

    guarded = _apply_guarded_rerank("What strain-specific parameters are shown in doc_0045 Table 2?", chunks, config)
    protected = _apply_rank1_evidence_guard(guarded, config)

    assert protected[0].chunk_id == "paragraph"
    assert protected[0].metadata["guarded_rank1_guard_triggered"] is False
    assert protected[0].metadata["guarded_rank1_guard_reason"] in {"no_better_complete_evidence", "top1_already_complete"}


def test_rank1_guard_respects_explicit_doc_anchor():
    config = RetrievalConfig()
    chunks = [
        _chunk(
            "correct_doc",
            "[FIGURE CAPTION] Fig. 4. Microbial composition changes among the top 20 species.",
            fusion=0.80,
            rerank=0.70,
            vector=0.72,
            doc_id="doc_0001",
        ),
        _chunk(
            "wrong_doc",
            "[FIGURE CAPTION] Fig. 4. Unrelated figure from another document.",
            fusion=0.78,
            rerank=0.68,
            vector=0.70,
            doc_id="doc_0060",
        ),
    ]

    guarded = _apply_guarded_rerank("What does Figure 4 describe in doc_0001?", chunks, config)
    protected = _apply_rank1_evidence_guard(guarded, config)

    assert protected[0].doc_id == "doc_0001"
    assert protected[0].chunk_id == "correct_doc"
