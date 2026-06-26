from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.synbio_rag.application.parent_aggregation import (
    aggregate_parent_hits_with_child_scores,
)
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.application.rerank_service import LocalBGERerankerService
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.index.parent_store import ParentRecord, ParentStore


def test_child_probe_scores_only_top_n_without_touching_final_debug() -> None:
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    service = _reranker_service(
        RetrievalConfig(rerank_subquery_aggregate_alpha=0.0),
        scores=[0.2, 0.9],
    )

    service.score_child_probe("query", chunks, top_n=2)

    assert len(service.local_reranker.pairs) == 2
    assert chunks[0].metadata["child_probe_rerank_score"] == 0.2
    assert chunks[0].metadata["child_probe_rerank_rank"] == 2
    assert chunks[1].metadata["child_probe_rerank_score"] == 0.9
    assert chunks[1].metadata["child_probe_rerank_rank"] == 1
    assert "child_probe_rerank_score" not in chunks[2].metadata
    assert [chunk.rerank_score for chunk in chunks] == [0.0, 0.0, 0.0]
    assert service.last_debug == {"final_hits": ["keep"]}


def test_parent_aggregation_collapses_children_and_orders_snippets_by_probe_score() -> None:
    store = _parent_store({"p1": ["c-low", "c-high"]})
    low = _chunk("c-low", fusion_score=0.2, metadata={"child_probe_rerank_score": 0.1})
    high = _chunk("c-high", fusion_score=0.3, metadata={"child_probe_rerank_score": 0.9})

    result = aggregate_parent_hits_with_child_scores(
        [low, high],
        store,
        RetrievalConfig(parent_aggregation_enabled=True),
    )

    assert [chunk.chunk_id for chunk in result] == ["p1"]
    parent = result[0]
    snippets = parent.metadata["matched_child_snippets"]
    assert parent.metadata["matched_child_chunk_ids"] == ["c-high", "c-low"]
    assert [snippet["chunk_id"] for snippet in snippets] == ["c-high", "c-low"]
    assert snippets[0]["child_probe_rerank_score"] == 0.9
    assert snippets[1]["child_probe_rerank_score"] == 0.1
    assert parent.metadata["matched_child_count"] == 2


def test_parent_aggregation_score_promotes_high_probe_multi_child_parent() -> None:
    store = _parent_store({"p-single": ["c-single"], "p-multi": ["c-a", "c-b"]})
    single = _chunk(
        "c-single",
        fusion_score=0.5,
        metadata={"child_probe_rerank_score": 0.6},
    )
    multi_a = _chunk(
        "c-a",
        fusion_score=0.2,
        metadata={"child_probe_rerank_score": 0.8},
    )
    multi_b = _chunk(
        "c-b",
        fusion_score=0.2,
        metadata={"child_probe_rerank_score": 0.7},
    )

    result = aggregate_parent_hits_with_child_scores(
        [single, multi_a, multi_b],
        store,
        RetrievalConfig(parent_aggregation_enabled=True),
    )

    assert [chunk.chunk_id for chunk in result] == ["p-multi", "p-single"]
    assert (
        result[0].metadata["parent_aggregation_score"]
        > result[1].metadata["parent_aggregation_score"]
    )


def test_final_rerank_adds_parent_aggregation_bonus_only_when_enabled() -> None:
    enabled = _reranker_service(
        _bonus_config(parent_aggregation_enabled=True),
        scores=[],
    )
    enabled_chunks = [
        _chunk("p1", metadata={"parent_aggregation_score": 1.0}),
        _chunk("p2", metadata={"parent_aggregation_score": 0.0}),
    ]

    enabled_final = enabled._aggregate_scores(
        ["query"],
        enabled_chunks,
        [[1.0, 1.1]],
        top_k=2,
        analysis=None,
        mode="plain",
    )

    assert [chunk.chunk_id for chunk in enabled_final] == ["p1", "p2"]
    assert enabled_chunks[0].rerank_score == pytest.approx(1.25)
    assert enabled_chunks[0].metadata["parent_aggregation_bonus"] == pytest.approx(0.25)
    assert enabled.last_debug["ranking_trace"][0]["parent_aggregation_bonus"] == 0.25

    disabled = _reranker_service(
        _bonus_config(parent_aggregation_enabled=False),
        scores=[],
    )
    disabled_chunks = [
        _chunk("p1", metadata={"parent_aggregation_score": 1.0}),
        _chunk("p2", metadata={"parent_aggregation_score": 0.0}),
    ]

    disabled_final = disabled._aggregate_scores(
        ["query"],
        disabled_chunks,
        [[1.0, 1.1]],
        top_k=2,
        analysis=None,
        mode="plain",
    )

    assert [chunk.chunk_id for chunk in disabled_final] == ["p2", "p1"]
    assert disabled_chunks[0].rerank_score == pytest.approx(1.0)
    assert "parent_aggregation_bonus" not in disabled_chunks[0].metadata


def test_pipeline_uses_original_materialize_when_parent_aggregation_is_disabled() -> None:
    store = _parent_store({"p1": ["c1"]})
    pipeline = object.__new__(SynBioRAGPipeline)
    pipeline.parent_store = store
    pipeline.settings = SimpleNamespace(
        retrieval=RetrievalConfig(parent_aggregation_enabled=False)
    )
    pipeline.reranker = _ProbeReranker()
    debug: dict[str, object] = {}

    result = pipeline._materialize_parent_child_hits(
        [_chunk("c1")],
        question="query",
        debug=debug,
    )

    assert [chunk.chunk_id for chunk in result] == ["p1"]
    assert pipeline.reranker.calls == []
    assert debug["parent_aggregation_used"] is False
    assert debug["parent_aggregation_reason"] == "disabled"


def test_pipeline_returns_raw_hits_when_parent_store_is_missing() -> None:
    pipeline = object.__new__(SynBioRAGPipeline)
    pipeline.parent_store = None
    debug: dict[str, object] = {}
    chunks = [_chunk("c1")]

    result = pipeline._materialize_parent_child_hits(chunks, question="query", debug=debug)

    assert result is chunks
    assert debug["parent_aggregation_used"] is False
    assert debug["parent_aggregation_reason"] == "parent_store_missing"


class _FakeLocalReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[list[str]] = []

    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        self.pairs = list(pairs)
        return self.scores[: len(pairs)]


class _ProbeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def score_child_probe(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        analysis: object | None = None,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "question": question,
                "chunks": chunks,
                "analysis": analysis,
                "top_n": top_n,
            }
        )
        return chunks


def _reranker_service(
    config: RetrievalConfig,
    scores: list[float],
) -> LocalBGERerankerService:
    service = object.__new__(LocalBGERerankerService)
    service.retrieval_config = config
    service.local_reranker = _FakeLocalReranker(scores)
    service.last_debug = {"final_hits": ["keep"]}
    return service


def _bonus_config(parent_aggregation_enabled: bool) -> RetrievalConfig:
    return RetrievalConfig(
        parent_aggregation_enabled=parent_aggregation_enabled,
        parent_aggregation_rerank_bonus_weight=0.25,
        rerank_subquery_aggregate_alpha=0.0,
        rerank_score_floor_ratio=0.0,
        rerank_strategy_bonus=0.0,
        evidence_numeric_bonus=0.0,
        evidence_result_bonus=0.0,
        evidence_definition_bonus=0.0,
        section_results_bonus=0.0,
        section_discussion_bonus=0.0,
        section_abstract_bonus=0.0,
        section_introduction_penalty=0.0,
        table_text_boost=0.0,
        table_caption_boost=0.0,
        figure_caption_boost=0.0,
    )


def _parent_store(children_by_parent: dict[str, list[str]]) -> ParentStore:
    parents: dict[str, ParentRecord] = {}
    parents_by_chunk: dict[str, list[str]] = {}
    parent_chunk_by_id: dict[str, RetrievedChunk] = {}
    for parent_chunk_id, child_ids in children_by_parent.items():
        record = ParentRecord(
            parent_id=f"{parent_chunk_id}-record",
            parent_type="retrieval_parent",
            doc_id=f"doc-{parent_chunk_id}",
            parent_chunk_id=parent_chunk_id,
            child_chunk_ids=child_ids,
        )
        parents[record.parent_id] = record
        parent_chunk_by_id[parent_chunk_id] = _chunk(
            parent_chunk_id,
            doc_id=record.doc_id,
            text=f"Parent text for {parent_chunk_id}",
        )
        for child_id in child_ids:
            parents_by_chunk.setdefault(child_id, []).append(record.parent_id)
    return ParentStore(
        parents=parents,
        parents_by_chunk=parents_by_chunk,
        parent_chunk_by_id=parent_chunk_by_id,
    )


def _chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc",
    text: str | None = None,
    vector_score: float = 0.0,
    bm25_score: float = 0.0,
    fusion_score: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"Title {doc_id}",
        section="Body",
        text=text or f"Evidence text for {chunk_id}",
        vector_score=vector_score,
        bm25_score=bm25_score,
        fusion_score=fusion_score,
        metadata=dict(metadata or {}),
    )
