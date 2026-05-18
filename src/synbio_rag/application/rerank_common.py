from __future__ import annotations

from ..domain.schemas import RetrievedChunk


def _rerank_text(chunk: RetrievedChunk) -> str:
    parts = []
    if chunk.title:
        parts.append(f"title: {chunk.title}")
    if chunk.section:
        parts.append(f"section: {chunk.section}")
    if chunk.source_file:
        parts.append(f"source_file: {chunk.source_file}")
    if chunk.doc_id:
        parts.append(f"doc_id: {chunk.doc_id}")
    parts.append(chunk.text)
    return "\n".join(part for part in parts if part)


def _sort_key(chunk: RetrievedChunk) -> tuple[float, float]:
    return (
        float(chunk.rerank_score),
        max(float(chunk.vector_score), float(chunk.fusion_score), float(chunk.bm25_score) * 0.1),
    )


def _guarded_sort_key(chunk: RetrievedChunk) -> tuple[float, float, float]:
    priority = float(chunk.metadata.get("guarded_rank1_priority", 0.0))
    guarded = float(chunk.metadata.get("guarded_score", chunk.rerank_score))
    completeness = float(chunk.metadata.get("guarded_keyword_completeness", 0.0))
    marker = float(chunk.metadata.get("guarded_marker_score", 0.0))
    return (priority, guarded, completeness, marker)


def _normalize_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (value - low) / (high - low)


def _serialize_hits(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "section": chunk.section,
            "score": round(float(chunk.rerank_score), 6),
            "evidence_features": chunk.metadata.get("evidence_features", {}),
        }
        for chunk in chunks
    ]
