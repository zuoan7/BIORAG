from __future__ import annotations

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, RetrievedChunk
from .rerank_common import _rerank_text, _serialize_hits
from .rerank_features import (
    _evidence_aware_bonus,
    _route_bonus,
    _strategy_bonus,
    _structure_marker_bonus,
)
from .rerank_query import _build_rerank_queries
from .rerank_selection import (
    _finalize_rerank,
)


class LocalBGERerankerService:
    def __init__(
        self,
        model_path: str = "./models/BAAI/bge-reranker-v2-m3",
        batch_size: int = 8,
        use_fp16: bool = True,
        retrieval_config: RetrievalConfig | None = None,
    ):
        from ..infrastructure.reranker.local_bge import LocalBGEReranker

        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.last_debug: dict[str, object] = {}
        self.local_reranker = LocalBGEReranker(
            model_path=model_path,
            use_fp16=use_fp16,
            batch_size=batch_size,
        )

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None = None,
        mode: str | None = None,
        *,
        original_question: str | None = None,
        rewritten_question: str | None = None,
    ) -> list[RetrievedChunk]:
        self.last_debug = {
            "original_query": original_question or question,
            "rewritten_query": rewritten_question or "",
            "rerank_query": question,
        }
        mode = (mode or self.retrieval_config.rerank_mode or "plain").strip().lower()
        if mode == "off":
            final = chunks[:top_k]
            self.last_debug["mode"] = mode
            self.last_debug["final_hits"] = _serialize_hits(final[:5])
            return final
        queries = _build_rerank_queries(question, analysis)
        self.last_debug["query_variants"] = queries
        self.last_debug["mode"] = mode
        if not chunks:
            self.last_debug["final_hits"] = []
            return []
        return self._rerank_with_local_model(queries, chunks, top_k, analysis, mode)

    def _rerank_with_local_model(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None,
        mode: str,
    ) -> list[RetrievedChunk]:
        texts = [_rerank_text(chunk) for chunk in chunks]
        pairs = [[query, text] for query in queries for text in texts]
        raw_scores = self.local_reranker.score_pairs(pairs)
        scores_by_query: list[list[float]] = []
        cursor = 0
        for _query in queries:
            scores_by_query.append(
                [float(score) for score in raw_scores[cursor : cursor + len(chunks)]]
            )
            cursor += len(chunks)
        return self._aggregate_scores(
            queries, chunks, scores_by_query, top_k, analysis, mode
        )

    def _aggregate_scores(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        scores_by_query: list[list[float]],
        top_k: int,
        analysis: QueryAnalysis | None,
        mode: str,
    ) -> list[RetrievedChunk]:
        alpha = self.retrieval_config.rerank_subquery_aggregate_alpha
        rescored: list[RetrievedChunk] = []
        query_debug: list[dict[str, object]] = []
        for query, scores in zip(queries, scores_by_query):
            ordered = sorted(
                (
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "score": float(score),
                    }
                    for chunk, score in zip(chunks, scores)
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
            query_debug.append({"query": query, "top_hits": ordered[:3]})

        for idx, chunk in enumerate(chunks):
            per_query_scores = [
                float(scores[idx]) for scores in scores_by_query if idx < len(scores)
            ]
            if per_query_scores:
                max_score = max(per_query_scores)
                mean_score = sum(per_query_scores) / len(per_query_scores)
            else:
                max_score = chunk.vector_score
                mean_score = chunk.vector_score
            bonus = _strategy_bonus(queries[0], chunk, self.retrieval_config)
            route_bonus = _route_bonus(queries[0], chunk, self.retrieval_config)
            evidence_bonus = _evidence_aware_bonus(chunk, self.retrieval_config)
            structure_bonus = _structure_marker_bonus(queries[0], chunk, self.retrieval_config)
            chunk.rerank_score = (
                max_score
                + alpha * mean_score
                + bonus
                + route_bonus
                + evidence_bonus
                + structure_bonus
            )
            chunk.metadata["rerank_query_scores"] = [round(score, 6) for score in per_query_scores]
            rescored.append(chunk)
        raw_order = list(rescored)
        selection_debug: dict[str, object] = {}
        final = _finalize_rerank(
            question=queries[0],
            chunks=rescored,
            top_k=top_k,
            analysis=analysis,
            config=self.retrieval_config,
            mode=mode,
            queries=queries,
            debug=selection_debug,
        )
        self.last_debug["query_scores"] = query_debug
        self.last_debug["selection"] = selection_debug
        self.last_debug["ranking_trace"] = _build_ranking_trace(
            raw_order=raw_order,
            selection_debug=selection_debug,
        )
        self.last_debug["final_hits"] = _serialize_hits(final[:5])
        return final


def _build_ranking_trace(
    *,
    raw_order: list[RetrievedChunk],
    selection_debug: dict[str, object],
) -> list[dict[str, object]]:
    pre_floor_rank = _rank_map(selection_debug.get("pre_floor_rank_by_chunk_id"))
    post_floor_rank = _rank_map(selection_debug.get("post_floor_rank_by_chunk_id"))
    final_rank = _rank_map(selection_debug.get("final_rank_by_chunk_id"))
    score_floor = (
        selection_debug.get("score_floor")
        if isinstance(selection_debug.get("score_floor"), dict)
        else {}
    )
    score_floor_dropped = {
        str(chunk_id)
        for chunk_id in (score_floor or {}).get("dropped_chunk_ids", [])
    }
    doc_diversity = (
        selection_debug.get("doc_diversity")
        if isinstance(selection_debug.get("doc_diversity"), dict)
        else {}
    )
    diversity_overflow = {
        str(chunk_id)
        for chunk_id in (doc_diversity or {}).get("overflow_chunk_ids", [])
    }
    comparison_selection = (
        selection_debug.get("comparison_selection")
        if isinstance(selection_debug.get("comparison_selection"), dict)
        else {}
    )
    comparison_applied = bool((comparison_selection or {}).get("applied"))
    top_k = int(selection_debug.get("top_k") or 0)

    trace: list[dict[str, object]] = []
    for raw_rank, chunk in enumerate(raw_order, start=1):
        chunk_id = str(chunk.chunk_id or "")
        final_drop_reason = ""
        if chunk_id not in final_rank:
            if chunk_id in score_floor_dropped:
                final_drop_reason = "score_floor"
            elif chunk_id in diversity_overflow:
                final_drop_reason = "doc_diversity_or_topk"
            elif comparison_applied:
                final_drop_reason = "comparison_selection_or_topk"
            elif post_floor_rank.get(chunk_id) and top_k and int(post_floor_rank[chunk_id]) > top_k:
                final_drop_reason = "final_topk_cutoff"
            elif pre_floor_rank.get(chunk_id) and not post_floor_rank.get(chunk_id):
                final_drop_reason = "score_floor"
            else:
                final_drop_reason = "final_selection_miss"
        trace.append(
            {
                "chunk_id": chunk_id,
                "parent_chunk_id": _parent_chunk_id(chunk_id),
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "section": chunk.section,
                "raw_retrieval_rank": raw_rank,
                "pre_floor_rerank_rank": pre_floor_rank.get(chunk_id),
                "post_floor_rank": post_floor_rank.get(chunk_id),
                "final_top10_rank": final_rank.get(chunk_id),
                "score": _round_float(chunk.rerank_score),
                "rerank_score": _round_float(chunk.rerank_score),
                "vector_score": _round_float(chunk.vector_score),
                "bm25_score": _round_float(chunk.bm25_score),
                "fusion_score": _round_float(chunk.fusion_score),
                "query_scores": list(chunk.metadata.get("rerank_query_scores") or []),
                "survived_score_floor": chunk_id in post_floor_rank,
                "dropped_by_score_floor": chunk_id in score_floor_dropped,
                "doc_diversity_overflow": chunk_id in diversity_overflow,
                "in_final_top10": chunk_id in final_rank,
                "final_drop_reason": final_drop_reason,
            }
        )
    return trace


def _rank_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, rank in value.items():
        if not isinstance(rank, (int, float)):
            continue
        result[str(key)] = int(rank)
    return result


def _parent_chunk_id(chunk_id: object) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def _round_float(value: object) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0
