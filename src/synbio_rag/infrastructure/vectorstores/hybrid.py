from __future__ import annotations

from collections import Counter
import re

from ...domain.config import RetrievalConfig
from ...domain.schemas import QueryAnalysis, QueryFilters, QueryIntent, RetrievedChunk


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_COMPARISON_SPLIT_RE = re.compile(r"\s*(?:,|，|、|以及|及|和|与|vs\.?|versus)\s*")
_TITLE_BOOST_KEYWORDS = (
    "6′-sl",
    "6'-sl",
    "2′-fl",
    "2'-fl",
    "wcfb",
    "salvage",
    "crispr-tmsd",
)
_REVIEW_HINTS = ("review", "综述", "perspective", "overview", "progress")
_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "2′-fl": ("2′-fl", "2'-fl", "2-fucosyllactose", "20-fucosyllactose"),
    "2'-fl": ("2′-fl", "2'-fl", "2-fucosyllactose", "20-fucosyllactose"),
    "6′-sl": ("6′-sl", "6'-sl", "6-sialyllactose", "6′-sialyllactose"),
    "6'-sl": ("6′-sl", "6'-sl", "6-sialyllactose", "6′-sialyllactose"),
    "wcfb": ("wcfb",),
    "salvage": ("salvage", "gdp-l-fucose", "gdp fucose"),
    "crispr-tmsd": ("crispr-tmsd", "tmsd", "strand displacement"),
}
_ROUTE_QUERY_MARKERS = ("工程化", "合成路径", "关键前体", "催化步骤", "biosynthesis", "pathway", "precursor")
_ROUTE_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "2′-fl": ("gdp-l-fucose", "gdp-fucose", "alpha-1,2-fucosyltransferase", "fuct", "lactose", "biosynthesis"),
    "2'-fl": ("gdp-l-fucose", "gdp-fucose", "alpha-1,2-fucosyltransferase", "fuct", "lactose", "biosynthesis"),
    "6′-sl": ("cmp-neu5ac", "sialyltransferase", "alpha-2,6-sialyltransferase", "lactose", "biosynthesis"),
    "6'-sl": ("cmp-neu5ac", "sialyltransferase", "alpha-2,6-sialyltransferase", "lactose", "biosynthesis"),
}


class HybridRetriever:
    def __init__(self, config: RetrievalConfig, dense_retriever, bm25_retriever):
        self.config = config
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.last_debug: dict[str, list[dict]] = {}

    def search(
        self,
        question: str,
        limit: int,
        filters: QueryFilters | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> list[RetrievedChunk]:
        query_plan = _build_query_plan(question, analysis, self.config)
        dense_lists: list[list[RetrievedChunk]] = []
        sparse_lists: list[list[RetrievedChunk]] = []
        debug_variants: list[dict[str, object]] = []

        for variant in query_plan:
            dense_results = self.dense_retriever.search(
                variant["query"],
                limit=max(limit, self.config.dense_limit),
                filters=filters,
            )
            dense_lists.append(dense_results)
            sparse_results: list[RetrievedChunk] = []
            if self.config.hybrid_enabled and self.config.bm25_enabled:
                sparse_results = self.bm25_retriever.search(
                    variant["query"],
                    limit=max(limit, self.config.bm25_limit),
                    filters=filters,
                )
            sparse_lists.append(sparse_results)
            debug_variants.append(
                {
                    "query": variant["query"],
                    "weight": variant["weight"],
                    "kind": variant["kind"],
                    "dense_hits": _serialize_hits(dense_results[:3], "vector_score"),
                    "bm25_hits": _serialize_hits(sparse_results[:3], "bm25_score"),
                }
            )

        self.last_debug = {"query_variants": debug_variants}
        dense_results = dense_lists[0] if dense_lists else []
        sparse_results = sparse_lists[0] if sparse_lists else []
        self.last_debug["dense_hits"] = _serialize_hits(dense_results[:5], "vector_score")
        self.last_debug["bm25_hits"] = _serialize_hits(sparse_results[:5], "bm25_score")

        if not self.config.hybrid_enabled or not self.config.bm25_enabled:
            merged_dense = reciprocal_rank_fusion_multi(
                dense_runs=[(hits, float(variant["weight"])) for hits, variant in zip(dense_lists, query_plan)],
                sparse_runs=[],
                limit=limit,
                rrf_k=self.config.rrf_k,
            )
            final_dense = _apply_comparison_diversity(merged_dense, limit, analysis, self.config)
            final_dense = _apply_title_keyword_boost(final_dense, question, self.config)
            self.last_debug["rrf_hits"] = _serialize_hits(final_dense[:5], "fusion_score")
            return final_dense

        dense_weight = self.config.dense_rrf_weight
        bm25_weight = self.config.bm25_rrf_weight
        if _contains_cjk(question):
            bm25_weight *= self.config.cjk_query_bm25_weight
        fused = reciprocal_rank_fusion_multi(
            dense_runs=[
                (hits, dense_weight * float(variant["weight"]))
                for hits, variant in zip(dense_lists, query_plan)
            ],
            sparse_runs=[
                (hits, bm25_weight * float(variant["weight"]))
                for hits, variant in zip(sparse_lists, query_plan)
            ],
            limit=max(limit * 3, limit + 12),
            rrf_k=self.config.rrf_k,
        )
        boosted = _apply_title_keyword_boost(fused, question, self.config)
        diversified = _apply_comparison_diversity(boosted, limit, analysis, self.config)
        self.last_debug["rrf_hits"] = _serialize_hits(diversified[:5], "fusion_score")
        return diversified


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


def _build_query_plan(
    question: str,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
) -> list[dict[str, object]]:
    plan = [{"query": _expand_query_aliases(question), "weight": config.comparison_query_weight, "kind": "original"}]
    plan[0]["query"] = _expand_route_pathway_terms(plan[0]["query"])
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        plan[0]["weight"] = 1.0
        return plan
    for subquery in _extract_comparison_subqueries(question):
        if subquery and subquery != question:
            expanded = _expand_query_aliases(subquery)
            plan.append(
                {
                    "query": _expand_route_pathway_terms(expanded),
                    "weight": config.comparison_subquery_weight,
                    "kind": "subquery",
                }
            )
    return plan


def _extract_comparison_subqueries(question: str) -> list[str]:
    normalized = question.strip()
    if not normalized:
        return []
    prefix_removed = re.sub(
        r"^(?:请|请你|请基于文库|请根据文库)?\s*(?:比较|对比|相比|compare)\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    lead, sep, tail = prefix_removed.partition("：")
    target_span = lead if sep else prefix_removed
    context = tail if sep else ""

    if not context:
        sentence_parts = re.split(r"[。？！?!.]", prefix_removed, maxsplit=1)
        target_span = sentence_parts[0]
        context = sentence_parts[1] if len(sentence_parts) > 1 else ""
    target_span = _trim_target_span(target_span)
    context = _trim_context_span(context)

    objects = _split_comparison_objects(target_span)
    if len(objects) < 2 and sep:
        objects = _split_comparison_objects(context)
        context = lead
    if len(objects) < 2:
        return []

    context_parts = [_remove_leading_compare_tokens(context)]
    if sep:
        context_parts.append(_remove_object_connectors(target_span, objects))
    shared_context = _clean_variant_text(" ".join(part for part in context_parts if part))
    subqueries: list[str] = []
    for obj in objects[:4]:
        pieces = [obj]
        if shared_context:
            pieces.append(shared_context)
        subqueries.append(_clean_variant_text(" ".join(pieces)))
    return [item for item in subqueries if item]


def _split_comparison_objects(text: str) -> list[str]:
    cleaned = _trim_target_span(_clean_variant_text(_remove_leading_compare_tokens(text)))
    if "：" in cleaned:
        cleaned = cleaned.split("：", 1)[1].strip()
    cleaned = re.sub(r"\b(?:两类|三类|四类)\b", " ", cleaned)
    parts = [item.strip(" ;；。") for item in _COMPARISON_SPLIT_RE.split(cleaned) if item.strip(" ;；。")]
    normalized: list[str] = []
    for part in parts:
        part = re.sub(r"\s*(?:两类|三类|四类)\s+[-A-Za-z0-9\u4e00-\u9fff′'/]+$", "", part).strip()
        if len(part) < 2:
            continue
        if any(token in part for token in ("当前文库", "工程化合成路径", "请分别说明", "请说明", "分别优化")):
            continue
        normalized.append(part)
    return normalized


def _remove_leading_compare_tokens(text: str) -> str:
    return re.sub(
        r"^(?:文库中|当前文库里|当前文库中|请|比较|对比|相比|compare)\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _remove_object_connectors(text: str, objects: list[str]) -> str:
    cleaned = text
    for obj in objects:
        cleaned = cleaned.replace(obj, " ")
    cleaned = _COMPARISON_SPLIT_RE.sub(" ", cleaned)
    return _clean_variant_text(cleaned)


def _clean_variant_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("：", " ").replace(":", " ")).strip()


def _trim_target_span(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.split(r"[。？！?!.]", cleaned, maxsplit=1)[0]
    cleaned = re.split(r"(?:在当前文库|在文库|在本研究|请分别说明|请说明|分别说明|说明)", cleaned, maxsplit=1)[0]
    return cleaned.strip(" ，,;；")


def _trim_context_span(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^请(?:分别)?说明", "", cleaned)
    cleaned = re.sub(r"^它们", "", cleaned)
    return _clean_variant_text(cleaned)


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


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _matching_keyword_groups(question: str) -> list[tuple[str, ...]]:
    lowered = question.lower()
    groups: list[tuple[str, ...]] = []
    for keyword in _TITLE_BOOST_KEYWORDS:
        if keyword in lowered:
            groups.append(_KEYWORD_ALIASES.get(keyword, (keyword,)))
    return groups


def _expand_query_aliases(query: str) -> str:
    lowered = query.lower()
    additions: list[str] = []
    seen: set[str] = set()
    for keyword in _TITLE_BOOST_KEYWORDS:
        if keyword not in lowered:
            continue
        for alias in _KEYWORD_ALIASES.get(keyword, (keyword,)):
            if alias in lowered or alias in seen:
                continue
            additions.append(alias)
            seen.add(alias)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _expand_route_pathway_terms(query: str) -> str:
    lowered = query.lower()
    if not any(marker in query for marker in _ROUTE_QUERY_MARKERS):
        return query
    additions: list[str] = []
    seen: set[str] = set()
    for keyword, aliases in _ROUTE_TERM_ALIASES.items():
        if keyword not in lowered:
            continue
        for alias in aliases:
            if alias in lowered or alias in seen:
                continue
            additions.append(alias)
            seen.add(alias)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
