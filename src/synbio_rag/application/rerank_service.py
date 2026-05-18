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
from .rerank_guarded_policy import _apply_guarded_rerank, _apply_rank1_evidence_guard
from .rerank_query import _build_rerank_queries
from .rerank_selection import (
    _BODY_SECTION_GROUPS,
    _apply_same_doc_body_coverage,
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
    ) -> list[RetrievedChunk]:
        self.last_debug = {}
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
            scores_by_query.append([float(score) for score in raw_scores[cursor : cursor + len(chunks)]])
            cursor += len(chunks)
        return self._aggregate_scores(queries, chunks, scores_by_query, top_k, analysis, mode)

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
            per_query_scores = [float(scores[idx]) for scores in scores_by_query if idx < len(scores)]
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
            chunk.rerank_score = max_score + alpha * mean_score + bonus + route_bonus + evidence_bonus + structure_bonus
            chunk.metadata["rerank_query_scores"] = [round(score, 6) for score in per_query_scores]
            rescored.append(chunk)
        final = _finalize_rerank(
            question=queries[0],
            chunks=rescored,
            top_k=top_k,
            analysis=analysis,
            config=self.retrieval_config,
            mode=mode,
            queries=queries,
        )
        self.last_debug["query_scores"] = query_debug
        self.last_debug["final_hits"] = _serialize_hits(final[:5])
        return final
