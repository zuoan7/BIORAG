from __future__ import annotations

from src.synbio_rag.application.rerank_selection import _finalize_rerank
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk


def test_score_floor_default_min_keep_is_capped_by_top_k() -> None:
    chunks = [
        _chunk("c1", 10.0),
        _chunk("c2", 4.0),
        _chunk("c3", 3.0),
        _chunk("c4", 2.0),
        _chunk("c5", 1.0),
    ]
    debug: dict[str, object] = {}

    final = _finalize_rerank(
        question="question",
        chunks=chunks,
        top_k=4,
        analysis=None,
        config=RetrievalConfig(rerank_score_floor_ratio=0.5),
        mode="plain",
        debug=debug,
    )

    assert [chunk.chunk_id for chunk in final] == ["c1", "c2", "c3", "c4"]
    assert debug["post_floor_chunk_ids"] == ["c1", "c2", "c3", "c4"]
    assert debug["score_floor"] == {
        "enabled": True,
        "ratio": 0.5,
        "top_score": 10.0,
        "floor": 5.0,
        "min_keep": 4,
        "rescued_chunk_ids": ["c2", "c3", "c4"],
        "dropped_chunk_ids": ["c5"],
    }


def test_score_floor_still_drops_below_floor_after_top_k_is_satisfied() -> None:
    chunks = [
        _chunk("c1", 10.0),
        _chunk("c2", 8.0),
        _chunk("c3", 6.0),
        _chunk("c4", 4.0),
        _chunk("c5", 3.0),
    ]
    debug: dict[str, object] = {}

    final = _finalize_rerank(
        question="question",
        chunks=chunks,
        top_k=3,
        analysis=None,
        config=RetrievalConfig(rerank_score_floor_ratio=0.5),
        mode="plain",
        debug=debug,
    )

    assert [chunk.chunk_id for chunk in final] == ["c1", "c2", "c3"]
    assert debug["post_floor_chunk_ids"] == ["c1", "c2", "c3"]
    assert debug["score_floor"]["rescued_chunk_ids"] == []
    assert debug["score_floor"]["dropped_chunk_ids"] == ["c4", "c5"]


def test_score_floor_configured_min_keep_below_top_k_can_drop_long_tail() -> None:
    chunks = [
        _chunk("c1", 10.0),
        _chunk("c2", 4.0),
        _chunk("c3", 3.0),
        _chunk("c4", 2.0),
        _chunk("c5", 1.0),
    ]
    debug: dict[str, object] = {}

    final = _finalize_rerank(
        question="question",
        chunks=chunks,
        top_k=5,
        analysis=None,
        config=RetrievalConfig(
            rerank_score_floor_ratio=0.5,
            rerank_score_floor_min_keep=2,
        ),
        mode="plain",
        debug=debug,
    )

    assert [chunk.chunk_id for chunk in final] == ["c1", "c2"]
    assert debug["post_floor_chunk_ids"] == ["c1", "c2"]
    assert debug["score_floor"]["min_keep"] == 2
    assert debug["score_floor"]["rescued_chunk_ids"] == ["c2"]
    assert debug["score_floor"]["dropped_chunk_ids"] == ["c3", "c4", "c5"]


def test_score_floor_default_b0_keeps_at_least_ten() -> None:
    chunks = [_chunk(f"c{i}", 11.0 - i) for i in range(1, 13)]
    debug: dict[str, object] = {}

    final = _finalize_rerank(
        question="question",
        chunks=chunks,
        top_k=10,
        analysis=None,
        config=RetrievalConfig(rerank_score_floor_ratio=0.9),
        mode="plain",
        debug=debug,
    )

    assert [chunk.chunk_id for chunk in final] == [f"c{i}" for i in range(1, 11)]
    assert debug["score_floor"]["min_keep"] == 10
    assert debug["score_floor"]["rescued_chunk_ids"] == [f"c{i}" for i in range(3, 11)]
    assert debug["score_floor"]["dropped_chunk_ids"] == ["c11", "c12"]


def _chunk(chunk_id: str, rerank_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        source_file=f"{chunk_id}.pdf",
        title=f"Title {chunk_id}",
        section="Results",
        text=f"Evidence text for {chunk_id}",
        rerank_score=rerank_score,
    )
