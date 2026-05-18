from __future__ import annotations

from ..domain.schemas import RetrievedChunk
from ..infrastructure.index.parent_store import ParentRecord, ParentStore
from .parent_expansion_plan import (
    comparison_caption_allowed,
    explicit_caption_query_type,
    is_caption_seed,
    preferred_evidence_type,
    seed_caption_type,
    weak_caption_reference,
)


def rank_children_for_seed(
    parent_store: ParentStore | None,
    parent: ParentRecord,
    seed: RetrievedChunk,
) -> list[RetrievedChunk]:
    if parent_store is None:
        return []
    children = parent_store.get_children(parent.parent_id)
    if not children or not isinstance(children[0], RetrievedChunk):
        return []
    typed_children = [child for child in children if isinstance(child, RetrievedChunk)]
    if parent.parent_type == "evidence_type_context":
        preferred = preferred_evidence_type("", seed)
        if preferred and parent.evidence_type and parent.evidence_type != preferred:
            return []

    seed_idx = chunk_index(seed)
    if parent.parent_type == "caption_context":
        seed_pages = set(seed.metadata.get("page_numbers", [])) if isinstance(seed.metadata, dict) else set()
        typed_children.sort(
            key=lambda child: (
                0 if child.doc_id == seed.doc_id else 1,
                0 if seed_pages and set(child.metadata.get("page_numbers", [])) & seed_pages else 1,
                abs(chunk_index(child) - seed_idx),
                chunk_index(child),
                child.chunk_id,
            )
        )
    elif parent.parent_type in {"section_path", "chunk_window", "page"}:
        typed_children.sort(key=lambda child: (abs(chunk_index(child) - seed_idx), chunk_index(child), child.chunk_id))
    else:
        typed_children.sort(key=lambda child: (chunk_index(child), child.chunk_id))
    return typed_children


def allow_candidate(
    *,
    seed: RetrievedChunk,
    candidate: RetrievedChunk,
    parent: ParentRecord,
    question: str,
    mode: str,
    parent_type: str,
    seen: set[str],
) -> tuple[bool, str]:
    if candidate.chunk_id in seen:
        return False, "duplicate_chunk"
    if mode == "caption":
        if candidate.doc_id != seed.doc_id:
            return False, "cross_doc"
        caption_query_type = explicit_caption_query_type(question)
        if caption_query_type == "none" and weak_caption_reference(question) and is_caption_seed(seed):
            caption_query_type = seed_caption_type(seed)
        if parent_type == "caption_context":
            if not caption_candidate_matches_type(candidate, caption_query_type):
                return False, "caption_type_mismatch"
        if parent_type == "page" and not same_page(seed, candidate):
            return False, "no_seed_page_numbers"
        if parent_type == "page":
            if not page_candidate_matches_type(candidate, caption_query_type):
                if not has_caption_like_signal(candidate):
                    return False, "page_plain_paragraph"
                return False, "page_type_mismatch"
    if mode == "comparison":
        if parent_type == "caption_context" and not comparison_caption_allowed(question):
            return False, "intent_not_allowed"
    return True, ""


def same_page(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    left_pages = set(left.metadata.get("page_numbers", [])) if isinstance(left.metadata, dict) else set()
    right_pages = set(right.metadata.get("page_numbers", [])) if isinstance(right.metadata, dict) else set()
    return bool(left_pages and right_pages and left_pages & right_pages)


def caption_candidate_matches_type(candidate: RetrievedChunk, query_type: str) -> bool:
    meta = candidate.metadata or {}
    has_table = bool(meta.get("contains_table_caption") or meta.get("contains_table_text"))
    has_figure = bool(meta.get("contains_figure_caption"))
    if query_type == "figure":
        return has_figure
    if query_type == "table":
        return has_table
    if query_type == "mixed":
        return has_table or has_figure
    seed_type = seed_caption_type(candidate)
    return seed_type in {"table", "figure", "mixed"}


def page_candidate_matches_type(candidate: RetrievedChunk, query_type: str) -> bool:
    return caption_candidate_matches_type(candidate, query_type)


def has_caption_like_signal(candidate: RetrievedChunk) -> bool:
    meta = candidate.metadata or {}
    return bool(
        meta.get("contains_table_caption")
        or meta.get("contains_figure_caption")
        or meta.get("contains_table_text")
        or meta.get("contains_image")
    )


def chunk_index(chunk: RetrievedChunk) -> int:
    metadata = chunk.metadata or {}
    value = metadata.get("chunk_index", 0) if isinstance(metadata, dict) else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
