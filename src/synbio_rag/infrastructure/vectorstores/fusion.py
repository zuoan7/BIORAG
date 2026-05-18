from __future__ import annotations

from ...domain.schemas import RetrievedChunk


def reciprocal_rank_fusion_multi(
    dense_runs: list[tuple[list[RetrievedChunk], float]],
    sparse_runs: list[tuple[list[RetrievedChunk], float]],
    limit: int,
    rrf_k: int,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}

    for dense_results, dense_weight in dense_runs:
        for rank, chunk in enumerate(dense_results, start=1):
            item = merged.get(chunk.chunk_id)
            if item is None:
                item = _clone_chunk(chunk)
                merged[chunk.chunk_id] = item
            item.vector_score = max(item.vector_score, chunk.vector_score)
            item.fusion_score += dense_weight / (rrf_k + rank)

    for sparse_results, sparse_weight in sparse_runs:
        for rank, chunk in enumerate(sparse_results, start=1):
            item = merged.get(chunk.chunk_id)
            if item is None:
                item = _clone_chunk(chunk)
                merged[chunk.chunk_id] = item
            item.bm25_score = max(item.bm25_score, chunk.bm25_score)
            item.fusion_score += sparse_weight / (rrf_k + rank)

    fused = list(merged.values())
    fused.sort(
        key=lambda item: (
            item.fusion_score,
            item.vector_score > 0.0,
            item.vector_score,
            item.bm25_score,
        ),
        reverse=True,
    )
    return fused[:limit]


def _clone_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        title=chunk.title,
        section=chunk.section,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score,
        rerank_score=chunk.rerank_score,
        fusion_score=chunk.fusion_score,
        metadata=dict(chunk.metadata),
    )


def _serialize_hits(chunks: list[RetrievedChunk], score_field: str) -> list[dict]:
    items = []
    for chunk in chunks:
        items.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "score": getattr(chunk, score_field, 0.0),
            }
        )
    return items
