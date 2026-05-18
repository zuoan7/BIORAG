from __future__ import annotations

from collections import Counter

from .query_semantics import _matching_keyword_groups
from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from ..infrastructure.vectorstores.fusion import _clone_chunk


_REVIEW_HINTS = ("review", "综述", "perspective", "overview", "progress")
_TABLE_QUERY_HINTS = ("table", "primer", "sequence", "strain", "vmax", "km", "relative peak area", "glycan", "parameter")
_FIGURE_QUERY_HINTS = ("figure", "fig.", "fig ", "图")
_BODY_EXPAND_SECTIONS: set[str] = {
    "Introduction", "Background", "Methods", "Materials and Methods",
    "Experimental Section", "Experimental Procedures", "Results",
    "Results and Discussion", "Discussion", "Conclusion", "Conclusions",
    "Full Text",
}


def _apply_source_floor(
    expanded: list[RetrievedChunk],
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    config,
) -> list[RetrievedChunk]:
    """Inject top-N single-source candidates that RRF merge may have suppressed.

    Only runs when config.source_floor_enabled=True.
    Does NOT use expected_doc_ids, does NOT boost scores, does NOT bypass reranker.
    """
    if not getattr(config, "source_floor_enabled", False):
        return expanded

    dense_n = getattr(config, "source_floor_dense_top_n", 3)
    bm25_n = getattr(config, "source_floor_bm25_top_n", 3)
    max_total = getattr(config, "source_floor_max_candidates_total", 6)

    existing_ids = {c.chunk_id for c in expanded}
    added_chunks: list[RetrievedChunk] = []
    added_info: list[dict[str, object]] = []

    for src_label, src_hits, top_n in [
        ("dense_floor", dense_results, dense_n),
        ("bm25_floor", sparse_results, bm25_n),
    ]:
        for rank, chunk in enumerate(src_hits[:top_n], start=1):
            if len(added_chunks) >= max_total:
                break
            if chunk.chunk_id in existing_ids:
                continue
            new_chunk = _clone_chunk(chunk)
            new_chunk.metadata["source_floor"] = src_label
            new_chunk.metadata["source_floor_rank"] = rank
            added_chunks.append(new_chunk)
            existing_ids.add(chunk.chunk_id)
            added_info.append({
                "source": src_label,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "rank": rank,
            })
        if len(added_chunks) >= max_total:
            break

    return expanded + added_chunks


def _apply_comparison_diversity(
    chunks: list[RetrievedChunk],
    limit: int,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return chunks[:limit]
    max_per_doc = max(1, config.comparison_max_chunks_per_doc)
    selected: list[RetrievedChunk] = []
    counts: Counter[str] = Counter()
    overflow: list[RetrievedChunk] = []
    for chunk in chunks:
        if counts[chunk.doc_id] < max_per_doc:
            selected.append(chunk)
            counts[chunk.doc_id] += 1
        else:
            overflow.append(chunk)
        if len(selected) >= limit:
            return selected[:limit]
    for chunk in overflow:
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _apply_title_keyword_boost(
    chunks: list[RetrievedChunk],
    question: str,
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    query_groups = _matching_keyword_groups(question)
    if not query_groups:
        return chunks[:]
    boosted: list[RetrievedChunk] = []
    for chunk in chunks:
        title_text = (chunk.title or "").lower()
        section_text = (chunk.section or "").lower()
        body_text = (chunk.text[:600] or "").lower()
        boost = 0.0
        matched_groups = 0
        for alias_group in query_groups:
            title_match = any(term in title_text for term in alias_group)
            abstract_match = section_text == "abstract" and any(term in body_text for term in alias_group)
            if title_match:
                matched_groups += 1
                boost += config.title_keyword_boost
            if abstract_match:
                boost += config.title_keyword_boost * 0.7
        if boost:
            if any(hint in title_text for hint in _REVIEW_HINTS):
                boost -= config.title_keyword_boost * 0.6
            elif matched_groups >= 2:
                boost += config.title_keyword_boost * 0.35
            chunk.fusion_score += max(boost, 0.0)
        boosted.append(chunk)
    boosted.sort(
        key=lambda item: (
            item.fusion_score,
            item.vector_score > 0.0,
            item.vector_score,
            item.bm25_score,
        ),
        reverse=True,
    )
    return boosted


def _apply_structure_marker_boost(
    chunks: list[RetrievedChunk],
    question: str,
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    lowered_question = question.lower()
    wants_table = any(hint in lowered_question for hint in _TABLE_QUERY_HINTS)
    wants_figure = any(hint in lowered_question for hint in _FIGURE_QUERY_HINTS)
    if not wants_table and not wants_figure:
        return chunks[:]
    boosted: list[RetrievedChunk] = []
    for chunk in chunks:
        text = (chunk.text or "").lower()
        boost = 0.0
        if wants_table:
            if "[table text]" in text:
                boost += config.table_text_boost
            if "[table caption]" in text:
                boost += config.table_caption_boost
        if wants_figure and "[figure caption]" in text:
            boost += config.figure_caption_boost
        if boost:
            chunk.fusion_score += boost
        boosted.append(chunk)
    boosted.sort(
        key=lambda item: (
            item.fusion_score,
            item.vector_score > 0.0,
            item.vector_score,
            item.bm25_score,
        ),
        reverse=True,
    )
    return boosted


def _apply_same_doc_body_expansion(
    diversified: list[RetrievedChunk],
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    config,
    question: str = "",
    bm25_retriever=None,
) -> list[RetrievedChunk]:
    """在 hybrid results 中为缺少 body section 的 doc 补入同文档 body chunks。

    优先级: dense/BM25 raw results (免费) > BM25 filtered query (一次查询)
    """
    if not config.same_doc_body_expand_enabled:
        return diversified

    expanded = list(diversified)
    existing_ids = {c.chunk_id for c in expanded}

    docs_in_results: dict[str, int] = {}
    for rank, c in enumerate(diversified):
        if c.doc_id not in docs_in_results:
            docs_in_results[c.doc_id] = rank

    docs_with_body: set[str] = set()
    body_count_per_doc: dict[str, int] = {}
    for c in diversified:
        if c.section in _BODY_EXPAND_SECTIONS:
            docs_with_body.add(c.doc_id)
            body_count_per_doc[c.doc_id] = body_count_per_doc.get(c.doc_id, 0) + 1

    docs_need_expand: list[tuple[str, int]] = []
    for doc_id, rank in docs_in_results.items():
        body_c = body_count_per_doc.get(doc_id, 0)
        # Expand if: doc missing body sections OR has very few body chunks (< 3)
        needs_body = (
            not config.same_doc_body_expand_require_missing_body
            or doc_id not in docs_with_body
            or body_c < 3
        )
        if needs_body and rank < config.same_doc_body_expand_min_doc_rank:
            docs_need_expand.append((doc_id, rank))
    docs_need_expand.sort(key=lambda x: x[1])
    docs_need_expand = docs_need_expand[:config.same_doc_body_expand_top_docs]

    if not docs_need_expand:
        return expanded

    expand_doc_set = {d[0] for d in docs_need_expand}

    # Phase A: scan dense/BM25 raw results for body chunks
    candidate_pool: dict[str, list[RetrievedChunk]] = {}
    for source_results in [dense_results, bm25_results]:
        for c in source_results:
            if c.doc_id in expand_doc_set and c.section in _BODY_EXPAND_SECTIONS:
                if c.chunk_id not in existing_ids:
                    candidate_pool.setdefault(c.doc_id, []).append(c)

    # Phase B: for docs still missing body chunks, query BM25 with doc_id filter
    if bm25_retriever is not None:
        from ..domain.schemas import QueryFilters
        for doc_id, _ in docs_need_expand:
            if doc_id in candidate_pool and len(candidate_pool[doc_id]) >= config.same_doc_body_expand_per_doc:
                continue
            try:
                body_section_list = sorted(_BODY_EXPAND_SECTIONS)
                extra = bm25_retriever.search(
                    question,
                    limit=config.same_doc_body_expand_per_doc * 3,
                    filters=QueryFilters(
                        doc_ids=[doc_id],
                        sections=body_section_list,
                    ),
                )
                for c in extra:
                    if c.section in _BODY_EXPAND_SECTIONS and c.chunk_id not in existing_ids:
                        candidate_pool.setdefault(doc_id, []).append(c)
            except Exception:
                pass

    for doc_chunks in candidate_pool.values():
        doc_chunks.sort(
            key=lambda c: (max(c.vector_score, c.bm25_score * 0.01), c.vector_score, c.bm25_score),
            reverse=True,
        )

    added_total = 0
    per_doc_added: dict[str, int] = {}
    for doc_id, doc_rank in docs_need_expand:
        if added_total >= config.same_doc_body_expand_max_total:
            break
        chunks = candidate_pool.get(doc_id, [])
        for bc in chunks:
            if per_doc_added.get(doc_id, 0) >= config.same_doc_body_expand_per_doc:
                break
            if added_total >= config.same_doc_body_expand_max_total:
                break
            if bc.chunk_id in existing_ids:
                continue

            new_chunk = _clone_chunk(bc)
            new_chunk.metadata["added_by_same_doc_body_expand"] = True
            new_chunk.metadata["expansion_reason"] = (
                f"doc {doc_id} ranked {doc_rank} in hybrid but missing body section"
            )
            new_chunk.metadata["source_doc_signal_rank"] = doc_rank
            new_chunk.metadata["body_section"] = bc.section
            new_chunk.metadata["expansion_score"] = max(bc.vector_score, bc.bm25_score * 0.01)
            expanded.append(new_chunk)
            existing_ids.add(bc.chunk_id)
            added_total += 1
            per_doc_added[doc_id] = per_doc_added.get(doc_id, 0) + 1

    return expanded
