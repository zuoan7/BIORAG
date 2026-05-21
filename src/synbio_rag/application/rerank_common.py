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
    focused_text = _parent_child_focused_text(chunk)
    parts.append(focused_text or chunk.text)
    return "\n".join(part for part in parts if part)


def _parent_child_focused_text(chunk: RetrievedChunk) -> str:
    snippets = chunk.metadata.get("matched_child_snippets") or []
    if not isinstance(snippets, list) or not snippets:
        return ""

    evidence_parts = []
    child_ranges: list[tuple[int, int]] = []
    for snippet in snippets[:4]:
        if not isinstance(snippet, dict):
            continue
        text = str(snippet.get("text") or "").strip()
        if not text:
            continue
        start = _safe_int(snippet.get("child_start_token"))
        end = _safe_int(snippet.get("child_end_token"))
        if start is not None and end is not None and end > start:
            child_ranges.append((start, end))
        marker = _snippet_marker(snippet)
        child_id = str(snippet.get("chunk_id") or "").strip()
        label = "matched child"
        if child_id:
            label = f"{label}: {child_id}"
        if marker:
            label = f"{label} {marker}"
        evidence_parts.append(f"{label}\n{_clip_words(text, 220)}")

    if not evidence_parts:
        return ""

    focused_parts = ["matched_child_evidence:", "\n\n".join(evidence_parts)]
    context = _parent_excerpt(chunk.text, child_ranges)
    if context:
        focused_parts.extend(["parent_context:", context])
    return "\n".join(focused_parts)


def _snippet_marker(snippet: dict) -> str:
    markers = []
    block_types = {str(value).lower() for value in snippet.get("block_types") or []}
    evidence_types = {str(value).lower() for value in snippet.get("evidence_types") or []}
    type_names = block_types | evidence_types
    if snippet.get("contains_table_text") or "table_text" in type_names:
        markers.append("[table text]")
    if snippet.get("contains_table_caption") or "table_caption" in type_names:
        markers.append("[table caption]")
    if snippet.get("contains_figure_caption") or "figure_caption" in type_names:
        markers.append("[figure caption]")
    return " ".join(markers)


def _parent_excerpt(text: str, child_ranges: list[tuple[int, int]]) -> str:
    if not text or not child_ranges:
        return ""
    words = text.split()
    if not words:
        return ""

    excerpt_parts = []
    seen: set[tuple[int, int]] = set()
    for start, end in child_ranges[:2]:
        start = max(0, min(start, len(words)))
        end = max(start, min(end, len(words)))
        left = max(0, start - 45)
        right = min(len(words), end + 55)
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        excerpt_parts.append(" ".join(words[left:right]))
    return "\n...\n".join(part for part in excerpt_parts if part)


def _clip_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
