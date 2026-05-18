from __future__ import annotations

from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from ...application.query_semantics import (
    _contains_cjk,
    _expand_query_aliases,
    _expand_route_pathway_terms,
    _extract_comparison_subqueries,
    _mask_organism_abbrevs,
    _split_comparison_objects,
    _unmask_organism_abbrevs,
)
from ...application.retrieval_postprocess import (
    _BODY_EXPAND_SECTIONS,
    _apply_comparison_diversity,
    _apply_same_doc_body_expansion,
    _apply_source_floor,
    _apply_structure_marker_boost,
    _apply_title_keyword_boost,
)
from ...domain.config import RetrievalConfig
from ...domain.schemas import QueryAnalysis, QueryFilters, QueryIntent, RetrievedChunk
from .fusion import _clone_chunk, _serialize_hits, reciprocal_rank_fusion_multi


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
        original_question: str | None = None,
    ) -> list[RetrievedChunk]:
        decomposition_query = original_question or question
        query_plan = _build_query_plan(question, analysis, self.config, decomposition_query)
        debug_plan = {
            "retrieval_query": question,
            "decomposition_query": decomposition_query,
            "query_variants": [],
        }
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
                sparse_query = _expand_alias_query(
                    variant["query"], self.config
                ) if getattr(self.config, "alias_expansion_enabled", False) else variant["query"]
                sparse_results = self.bm25_retriever.search(
                    sparse_query,
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

        debug_plan["query_variants"] = debug_variants
        self.last_debug = debug_plan
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
        boosted = _apply_structure_marker_boost(boosted, question, self.config)
        diversified = _apply_comparison_diversity(boosted, limit, analysis, self.config)
        expanded = _apply_same_doc_body_expansion(
            diversified=diversified,
            dense_results=dense_results,
            bm25_results=sparse_results,
            config=self.config,
            question=question,
            bm25_retriever=self.bm25_retriever,
        )
        expanded = _apply_source_floor(
            expanded=expanded,
            dense_results=dense_results,
            sparse_results=sparse_results,
            config=self.config,
        )
        self.last_debug["rrf_hits"] = _serialize_hits(expanded[:5], "fusion_score")
        self.last_debug["same_doc_body_expand_enabled"] = self.config.same_doc_body_expand_enabled
        self.last_debug["same_doc_body_expand_added"] = len(expanded) - len(diversified)
        return expanded


def _build_query_plan(
    question: str,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
    decomposition_query: str | None = None,
) -> list[dict[str, object]]:
    """Build query plan for hybrid retrieval.

    question: the main retrieval query (may be rewritten EN).
    decomposition_query: optional original query for comparison subquery extraction.
        When QUERY_REWRITE_MODE=enabled, retrieval uses EN query but decomposition
        should use the original CN query which preserves comparison structure.
    """
    plan = [{"query": _expand_query_aliases(question), "weight": config.comparison_query_weight, "kind": "original"}]
    plan[0]["query"] = _expand_route_pathway_terms(plan[0]["query"])
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        plan[0]["weight"] = 1.0
        return plan
    # Use decomposition_query for comparison subquery extraction;
    # falls back to question (retrieval query) if not provided.
    compare_source = decomposition_query if decomposition_query else question
    for subquery in _extract_comparison_subqueries(compare_source):
        if subquery and subquery != compare_source:
            expanded = _expand_query_aliases(subquery)
            plan.append(
                {
                    "query": _expand_route_pathway_terms(expanded),
                    "weight": config.comparison_subquery_weight,
                    "kind": "subquery",
                }
            )
    return plan


# ── Controlled alias expansion (BM25-only, query-time) ────────────

_ALIAS_MAP_CACHE: dict[str, Any] | None = None


def _load_alias_map(config: Any) -> dict[str, Any] | None:
    """Load alias map from YAML file, cached at module level."""
    global _ALIAS_MAP_CACHE
    if _ALIAS_MAP_CACHE is not None:
        return _ALIAS_MAP_CACHE
    map_path = getattr(config, "alias_expansion_map_path", "") or ""
    if not map_path:
        # Default path relative to project root
        map_path = str(Path(__file__).resolve().parents[4] / "src/synbio_rag/resources/retrieval_aliases_v1.yaml")
    try:
        if _HAS_YAML:
            with open(map_path, encoding="utf-8") as f:
                _ALIAS_MAP_CACHE = yaml.safe_load(f)
        else:
            import json
            with open(map_path.replace(".yaml", ".json"), encoding="utf-8") as f:
                _ALIAS_MAP_CACHE = json.load(f)
    except Exception:
        _ALIAS_MAP_CACHE = {}
    return _ALIAS_MAP_CACHE


def _expand_alias_query(query: str, config: Any) -> str:
    """Expand query with controlled BM25-only alias terms.

    Trigger-based: only adds expansions when trigger terms appear in query.
    BM25-only: does not modify dense query or LLM prompt.
    Feature-flagged: only runs when alias_expansion_enabled=True.
    """
    if not getattr(config, "alias_expansion_enabled", False):
        return query

    alias_map = _load_alias_map(config)
    if not alias_map:
        return query

    aliases = alias_map.get("aliases", {})
    if not aliases:
        return query

    allowed_risks = set(getattr(config, "alias_expansion_risk_levels", ["low"]))
    max_entities = getattr(config, "alias_expansion_max_entities_per_query", 3)
    max_per_entity = getattr(config, "alias_expansion_max_expansions_per_entity", 3)
    max_total = getattr(config, "alias_expansion_max_total_terms", 8)

    query_lower = query.lower()
    # Normalize primes and hyphens for trigger matching
    query_normalized = query_lower.replace('\u2019', "'").replace('\u2018', "'").replace('\u2032', "'")

    expansion_terms: list[str] = []
    triggered_count = 0

    for canonical_id, entry in aliases.items():
        if triggered_count >= max_entities:
            break
        risk = entry.get("risk", "medium")
        if risk not in allowed_risks:
            continue

        # Check triggers
        triggered = False
        for lang, terms in (entry.get("triggers") or {}).items():
            for term in terms:
                if term.lower() in query_lower or term.lower() in query_normalized:
                    triggered = True
                    break
            if triggered:
                break
        if not triggered:
            continue

        # Add expansions (capped per entity)
        expansions = entry.get("expansions", [])[:max_per_entity]
        for exp in expansions:
            if exp.lower() in query_lower or exp.lower() in query_normalized:
                continue  # skip if already in query
            expansion_terms.append(exp)

        triggered_count += 1
        if len(expansion_terms) >= max_total:
            break

    expansion_terms = expansion_terms[:max_total]
    if not expansion_terms:
        return query

    return f"{query} {' '.join(expansion_terms)}"
