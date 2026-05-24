from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synbio_rag.application.pipeline import SynBioRAGPipeline  # noqa: E402
from synbio_rag.domain.config import RetrievalConfig, Settings  # noqa: E402
from synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk  # noqa: E402
from synbio_rag.rewrite.query_rewrite_service import RewriteTrace  # noqa: E402

ORIGINAL_QUERY = "中文问题：文中如何调控 acetyl-CoA sensor？"
REWRITTEN_QUERY = "How does the paper regulate the acetyl-CoA sensor?"


def test_pipeline_rerank_uses_rewritten_query_while_analysis_uses_original() -> None:
    router = _RouterSpy()
    rewrite_service = _RewriteServiceStub()
    reranker = _RerankerSpy()
    generator = _GeneratorStub()
    parent_expander = _ParentExpanderSpy()
    search_calls: list[dict[str, Any]] = []

    def search_with_filter_fallback(
        *,
        question: str,
        analysis: QueryAnalysis,
        filters: object | None,
        original_question: str | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        search_calls.append(
            {
                "question": question,
                "analysis": analysis,
                "filters": filters,
                "original_question": original_question,
            }
        )
        return [_chunk("gold"), _chunk("other")], {"selected": "stub"}

    pipeline = object.__new__(SynBioRAGPipeline)
    pipeline.settings = Settings(
        retrieval=RetrievalConfig(
            final_top_k=2,
            table_preview={"table_preview_enabled": False},
        )
    )
    pipeline.router = router
    pipeline._rewrite_svc = rewrite_service
    pipeline.reranker = reranker
    pipeline.retriever = SimpleNamespace(last_debug={})
    pipeline.confidence_scorer = _ConfidenceStub()
    pipeline.generator_v2 = generator
    pipeline.parent_expander = parent_expander
    pipeline.table_preview_provider = None
    pipeline._search_with_filter_fallback = search_with_filter_fallback

    response = pipeline.answer(ORIGINAL_QUERY)

    assert router.questions == [ORIGINAL_QUERY]
    assert rewrite_service.calls == [{"query": ORIGINAL_QUERY, "is_negative": False}]
    assert search_calls[0]["question"] == REWRITTEN_QUERY
    assert search_calls[0]["original_question"] == ORIGINAL_QUERY
    assert reranker.calls[0]["question"] == REWRITTEN_QUERY
    assert reranker.calls[0]["original_question"] == ORIGINAL_QUERY
    assert reranker.calls[0]["rewritten_question"] == REWRITTEN_QUERY
    assert generator.question == ORIGINAL_QUERY
    assert parent_expander.question == ORIGINAL_QUERY
    assert response.debug["query_bundle"] == {
        "original_query": ORIGINAL_QUERY,
        "rewritten_query": REWRITTEN_QUERY,
        "retrieval_query": REWRITTEN_QUERY,
        "rerank_query": REWRITTEN_QUERY,
    }
    assert response.debug["rerank_hits"]["rerank_query"] == REWRITTEN_QUERY


class _RouterSpy:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def analyze(self, question: str) -> QueryAnalysis:
        self.questions.append(question)
        return QueryAnalysis(
            intent=QueryIntent.FACTOID,
            requires_external_tools=False,
            search_limit=2,
            rerank_top_k=2,
        )


class _RewriteServiceStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rewrite(self, original_query: str, is_negative: bool = False) -> tuple[str, RewriteTrace]:
        self.calls.append({"query": original_query, "is_negative": is_negative})
        return (
            REWRITTEN_QUERY,
            RewriteTrace(
                query_rewrite_mode="enabled",
                query_rewrite_enabled=True,
                original_query=original_query,
                rewritten_query=REWRITTEN_QUERY,
                retrieval_query_used="rewritten",
            ),
        )


class _RerankerSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.last_debug: dict[str, object] = {}

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None = None,
        *,
        original_question: str | None = None,
        rewritten_question: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "analysis": analysis,
                "original_question": original_question,
                "rewritten_question": rewritten_question,
            }
        )
        self.last_debug = {
            "original_query": original_question,
            "rewritten_query": rewritten_question,
            "rerank_query": question,
        }
        return chunks[:top_k]


class _ConfidenceStub:
    def score(self, chunks: list[RetrievedChunk]) -> float:
        return 1.0 if chunks else 0.0


class _ParentExpanderSpy:
    def __init__(self) -> None:
        self.question = ""

    def expand(
        self,
        *,
        question: str,
        seed_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        self.question = question
        return list(seed_chunks), {"reason": "stub"}


class _GeneratorStub:
    def __init__(self) -> None:
        self.question = ""

    def run(
        self,
        *,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
        config: object,
        history: object | None,
    ) -> SimpleNamespace:
        self.question = question
        return SimpleNamespace(answer="ok", citations=[], debug={})


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_file=f"{chunk_id}.pdf",
        title=f"Title {chunk_id}",
        section="Results",
        text=f"Evidence text for {chunk_id}",
        rerank_score=1.0,
    )
