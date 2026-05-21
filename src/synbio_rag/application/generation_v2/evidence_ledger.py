from __future__ import annotations

import re
from typing import Any

from ...domain.schemas import QueryAnalysis, RetrievedChunk
from .models import EvidenceCandidate

_MAX_CHILD_SNIPPETS = 4
_CHILD_SNIPPET_MAX_CHARS = 350
_PARENT_EXCERPT_MAX_CHARS = 500
_RESULT_PATTERN = re.compile(
    r"\b(result|results|showed|demonstrated|increased|decreased|yield|titer|production)\b|结果|显示|提高|降低|产量|滴度",
    re.IGNORECASE,
)
_DIGIT_PATTERN = re.compile(r"\d")


class EvidenceLedgerBuilder:
    def build(
        self,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
    ) -> list[EvidenceCandidate]:
        del question, analysis
        candidates: list[EvidenceCandidate] = []
        for index, chunk in enumerate(seed_chunks, start=1):
            metadata = dict(chunk.metadata or {})
            text, child_view_used, matched_child_chunk_ids = _generation_evidence_text(chunk)
            metadata["generation_evidence_role"] = (
                "matched_child_focused_evidence" if child_view_used else "parent_text"
            )
            metadata["parent_child_generation_view_used"] = child_view_used
            if child_view_used:
                metadata["matched_child_chunk_ids"] = matched_child_chunk_ids
                metadata["parent_text_preview"] = _clip_chars(
                    chunk.text or "",
                    _PARENT_EXCERPT_MAX_CHARS,
                )
            lower_text = text.lower()
            lower_section = (chunk.section or "").strip().lower()
            feature_flags = {
                "has_table_text": _has_table_text(lower_text, metadata, child_view_used),
                "has_table_caption": _has_table_caption(lower_text, metadata, child_view_used),
                "has_figure_caption": _has_figure_caption(lower_text, metadata, child_view_used),
                "has_numeric": bool(_DIGIT_PATTERN.search(text)),
                "has_result_terms": bool(_RESULT_PATTERN.search(text)),
                "section_type": lower_section,
                "text_length": len(text),
            }
            reasons = [f"seed_chunk", f"section:{lower_section or 'unknown'}"]
            if child_view_used:
                reasons.append("parent_child_focused_evidence")
            for feature_name, enabled in feature_flags.items():
                if feature_name in {"section_type", "text_length"} or not enabled:
                    continue
                reasons.append(feature_name)
            candidates.append(
                EvidenceCandidate(
                    evidence_id=f"E{index}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    title=chunk.title,
                    section=chunk.section,
                    text=text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    vector_score=chunk.vector_score,
                    bm25_score=chunk.bm25_score,
                    rerank_score=chunk.rerank_score,
                    fusion_score=chunk.fusion_score,
                    metadata=metadata,
                    features=feature_flags,
                    reasons=reasons,
                )
            )
        return candidates


def _generation_evidence_text(chunk: RetrievedChunk) -> tuple[str, bool, list[str]]:
    metadata = chunk.metadata or {}
    snippets = metadata.get("matched_child_snippets") or []
    if not isinstance(snippets, list) or not snippets:
        return chunk.text or "", False, []

    evidence_parts: list[str] = []
    child_ranges: list[tuple[int, int]] = []
    child_ids: list[str] = []
    for snippet in snippets[:_MAX_CHILD_SNIPPETS]:
        if not isinstance(snippet, dict):
            continue
        raw_text = str(snippet.get("text") or "").strip()
        if not raw_text:
            continue

        child_id = str(snippet.get("chunk_id") or "").strip()
        if child_id:
            child_ids.append(child_id)
        start = _safe_int(snippet.get("child_start_token"))
        end = _safe_int(snippet.get("child_end_token"))
        if start is not None and end is not None and end > start:
            child_ranges.append((start, end))

        label = "matched child"
        if child_id:
            label = f"{label}: {child_id}"
        marker = _snippet_marker(snippet)
        if marker:
            label = f"{label} {marker}"
        evidence_parts.append(f"{label}\n{_clip_chars(raw_text, _CHILD_SNIPPET_MAX_CHARS)}")

    if not evidence_parts:
        return chunk.text or "", False, []

    matched_child_chunk_ids = child_ids or _metadata_child_ids(metadata)[:_MAX_CHILD_SNIPPETS]
    focused_parts = ["matched_child_evidence:", "\n\n".join(evidence_parts)]
    parent_excerpt = _parent_excerpt(chunk.text or "", child_ranges, _PARENT_EXCERPT_MAX_CHARS)
    if parent_excerpt:
        focused_parts.extend(["parent_context_excerpt:", parent_excerpt])
    return "\n".join(focused_parts), True, matched_child_chunk_ids


def _metadata_child_ids(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("matched_child_chunk_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    value = metadata.get("matched_child_chunk_id")
    if value:
        return [str(value)]
    return []


def _snippet_marker(snippet: dict[str, Any]) -> str:
    markers: list[str] = []
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


def _parent_excerpt(text: str, child_ranges: list[tuple[int, int]], max_chars: int) -> str:
    if not text or not child_ranges:
        return ""
    words = text.split()
    if not words:
        return ""

    excerpt_parts: list[str] = []
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
    return _clip_chars("\n...\n".join(part for part in excerpt_parts if part), max_chars)


def _clip_chars(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_table_text(lower_text: str, metadata: dict[str, Any], child_view_used: bool) -> bool:
    return "[table text]" in lower_text or "table_text" in lower_text or (
        not child_view_used and "table_text" in metadata
    )


def _has_table_caption(lower_text: str, metadata: dict[str, Any], child_view_used: bool) -> bool:
    return "[table caption]" in lower_text or "table_caption" in lower_text or (
        not child_view_used and "table_caption" in metadata
    )


def _has_figure_caption(lower_text: str, metadata: dict[str, Any], child_view_used: bool) -> bool:
    return (
        "[figure caption]" in lower_text
        or "fig." in lower_text
        or "figure_caption" in lower_text
        or (not child_view_used and "figure_caption" in metadata)
    )
