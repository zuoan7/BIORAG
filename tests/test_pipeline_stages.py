from __future__ import annotations

from types import SimpleNamespace

from src.synbio_rag.application.pipeline_stages import (
    ContextStage,
    GenerationStage,
    RerankStage,
    ResponseStage,
    RetrievalStage,
)
from src.synbio_rag.domain.config import GenerationConfig, RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import (
    QueryAnalysis,
    QueryIntent,
    RAGPipelineResponse,
    RetrievedChunk,
)


def _analysis(intent: QueryIntent = QueryIntent.FACTOID) -> QueryAnalysis:
    return QueryAnalysis(
        intent=intent,
        requires_external_tools=False,
        search_limit=40,
        rerank_top_k=10,
    )


def _chunk(chunk_id: str, *, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        source_file=f"{chunk_id}.pdf",
        title=f"Title {chunk_id}",
        section="Results",
        text=f"Evidence {chunk_id}",
        rerank_score=0.5,
        metadata={} if metadata is None else metadata,
    )


def test_retrieval_stage_keeps_table_preview_debug_when_disabled() -> None:
    settings = Settings(retrieval=RetrievalConfig(table_preview_enabled=False))
    trace = SimpleNamespace(to_dict=lambda: {}, query_rewrite_mode="off")
    retrieved = [_chunk("c1")]
    stage = RetrievalStage(
        settings=settings,
        router=SimpleNamespace(analyze=lambda question: _analysis()),
        rewrite_service=SimpleNamespace(
            rewrite=lambda question, is_negative=False: (question, trace)
        ),
        search_with_filter_fallback=lambda **kwargs: (
            retrieved,
            {"selected": "original", "attempts": []},
        ),
        table_preview_provider=None,
        fallback_pipeline=SimpleNamespace(
            _search_with_filter_fallback=lambda **kwargs: ([], {})
        ),
    )

    result = stage.run(question="what?", filters=None)

    assert result.analysis.intent == QueryIntent.FACTOID
    assert result.retrieved == retrieved
    assert result.retrieval_debug["selected"] == "original"
    assert result.table_preview_debug["enabled"] is False
    assert result.cn_fallback_debug["reason"] == "disabled"


def test_rerank_stage_sets_rerank_rank_on_seed_chunks() -> None:
    settings = Settings(retrieval=RetrievalConfig(final_top_k=2))
    reranked = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    stage = RerankStage(
        settings=settings,
        reranker=SimpleNamespace(rerank=lambda *args, **kwargs: reranked),
    )

    result = stage.run(question="what?", retrieved=reranked, analysis=_analysis())

    assert [chunk.chunk_id for chunk in result.seed_chunks] == ["c1", "c2"]
    assert result.seed_chunks[0].metadata["rerank_rank"] == 1
    assert result.seed_chunks[1].metadata["rerank_rank"] == 2


def test_context_stage_returns_parent_and_lifecycle_debug() -> None:
    seed = _chunk("c1", metadata={"rerank_rank": 1})
    parent_expander = SimpleNamespace(
        expand=lambda **kwargs: (
            kwargs["seed_chunks"],
            {"enabled": True, "reason": "expanded"},
        )
    )
    stage = ContextStage(
        settings=Settings(),
        retriever=SimpleNamespace(last_debug={}),
        parent_expander=parent_expander,
    )

    result = stage.run(
        question="what?",
        analysis=_analysis(),
        seed_chunks=[seed],
        reranked=[seed],
    )

    assert result.final_chunks == [seed]
    assert result.parent_expansion_debug["reason"] == "expanded"
    assert "rerank_output" in result.evidence_lifecycle_debug
    assert "seed_chunks" in result.evidence_lifecycle_debug
    assert "final_chunks" in result.evidence_lifecycle_debug


def test_generation_stage_disables_required_citation_for_merged_table_preview() -> None:
    captured = {}
    settings = Settings(generation=GenerationConfig(v2_require_citation=True))

    def run_generator(**kwargs):
        captured["config"] = kwargs["config"]
        return SimpleNamespace(answer="ok", citations=[], debug={})

    stage = GenerationStage(
        settings=settings,
        confidence_scorer=SimpleNamespace(score=lambda chunks: float(len(chunks))),
        generator_v2=SimpleNamespace(run=run_generator),
    )

    result = stage.run(
        question="what?",
        analysis=_analysis(),
        seed_chunks=[_chunk("c1")],
        final_chunks=[_chunk("c1"), _chunk("c2")],
        table_preview_debug={"merged_count": 1},
        history=[],
    )

    assert result.seed_confidence == 1.0
    assert result.confidence == 2.0
    assert result.table_preview_answer_without_formal_citation is True
    assert captured["config"].v2_require_citation is False
    assert settings.generation.v2_require_citation is True


def test_generation_stage_keeps_required_citation_for_shadow_table_preview() -> None:
    captured = {}
    settings = Settings(generation=GenerationConfig(v2_require_citation=True))

    def run_generator(**kwargs):
        captured["config"] = kwargs["config"]
        return SimpleNamespace(answer="ok", citations=[], debug={})

    stage = GenerationStage(
        settings=settings,
        confidence_scorer=SimpleNamespace(score=lambda chunks: float(len(chunks))),
        generator_v2=SimpleNamespace(run=run_generator),
    )

    result = stage.run(
        question="what?",
        analysis=_analysis(),
        seed_chunks=[_chunk("c1")],
        final_chunks=[_chunk("c1")],
        table_preview_debug={"candidate_count": 1, "merged_count": 0},
        history=[],
    )

    assert result.table_preview_answer_without_formal_citation is False
    assert captured["config"].v2_require_citation is True


def test_response_stage_returns_internal_pipeline_response() -> None:
    settings = Settings()
    trace = SimpleNamespace(to_dict=lambda: {})
    analysis = _analysis()
    chunk = _chunk("c1", metadata={"rerank_rank": 1})
    retrieval = SimpleNamespace(
        analysis=analysis,
        retrieved=[chunk],
        retrieval_debug={"selected": "original", "attempts": []},
        cn_fallback_debug={"triggered": False, "reason": "disabled"},
        table_preview_debug={"enabled": False},
        rewrite_trace=trace,
    )
    rerank = SimpleNamespace(reranked=[chunk])
    context = SimpleNamespace(
        seed_chunks=[chunk],
        final_chunks=[chunk],
        summary_supplement_debug={"used": False},
        parent_expansion_debug={"enabled": False},
        evidence_lifecycle_debug={},
    )
    generation = SimpleNamespace(
        gen_result=SimpleNamespace(answer="ok", citations=[], debug={}),
        seed_confidence=0.5,
        confidence=0.6,
        table_preview_answer_without_formal_citation=False,
    )
    stage = ResponseStage(
        settings=settings,
        retriever=SimpleNamespace(last_debug={"retrieval": True}),
        reranker=SimpleNamespace(last_debug={"rerank": True}),
    )

    response = stage.run(
        retrieval=retrieval,
        rerank=rerank,
        context=context,
        generation=generation,
        session_id="session-1",
        filters=None,
        start_time=0.0,
    )

    assert isinstance(response, RAGPipelineResponse)
    assert not hasattr(response, "used_external_tool")
    assert not hasattr(response, "tool_name")
    assert not hasattr(response, "tool_result")
    assert not hasattr(response, "external_references")
    assert response.debug["generation_v2"]["table_preview_answer_without_formal_citation"] is False
    assert response.debug["retrieval_hits"] == {"retrieval": True}
    assert response.debug["rerank_hits"] == {"rerank": True}
