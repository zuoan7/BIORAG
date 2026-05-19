from __future__ import annotations

from ...domain.config import RetrievalConfig
from ...domain.schemas import QueryAnalysis, QueryFilters, RetrievedChunk
from .fusion import _serialize_hits, reciprocal_rank_fusion_multi


class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        dense_retriever,
        bm25_retriever,
        query_planner=None,
        alias_policy=None,
        postprocessor=None,
    ):
        self.config = config
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.query_planner = query_planner
        self.alias_policy = alias_policy
        self.postprocessor = postprocessor
        self.last_debug: dict[str, object] = {}

    def search(
        self,
        question: str,
        limit: int,
        filters: QueryFilters | None = None,
        analysis: QueryAnalysis | None = None,
        original_question: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.query_planner is None or self.alias_policy is None or self.postprocessor is None:
            raise RuntimeError("HybridRetriever requires injected retrieval policies")
        decomposition_query = original_question or question
        query_plan = self.query_planner.build(question, analysis, self.config, decomposition_query)
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
                sparse_query = self.alias_policy.expand(
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
            final_dense = self.postprocessor.apply_comparison_diversity(
                merged_dense, limit, analysis, self.config
            )
            final_dense = self.postprocessor.apply_title_keyword_boost(
                final_dense, question, self.config
            )
            self.last_debug["rrf_hits"] = _serialize_hits(final_dense[:5], "fusion_score")
            return final_dense

        dense_weight = self.config.dense_rrf_weight
        bm25_weight = self.config.bm25_rrf_weight
        if self.query_planner.contains_cjk(question):
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
        boosted = self.postprocessor.apply_title_keyword_boost(fused, question, self.config)
        boosted = self.postprocessor.apply_structure_marker_boost(boosted, question, self.config)
        diversified = self.postprocessor.apply_comparison_diversity(
            boosted, limit, analysis, self.config
        )
        expanded = self.postprocessor.apply_same_doc_body_expansion(
            diversified=diversified,
            dense_results=dense_results,
            bm25_results=sparse_results,
            config=self.config,
            question=question,
            bm25_retriever=self.bm25_retriever,
        )
        expanded = self.postprocessor.apply_source_floor(
            expanded=expanded,
            dense_results=dense_results,
            sparse_results=sparse_results,
            config=self.config,
        )
        self.last_debug["rrf_hits"] = _serialize_hits(expanded[:5], "fusion_score")
        self.last_debug["same_doc_body_expand_enabled"] = self.config.same_doc_body_expand_enabled
        self.last_debug["same_doc_body_expand_added"] = len(expanded) - len(diversified)
        return expanded
