from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from ..infrastructure.clients.openai_compatible import OpenAICompatibleClient, extract_json_block
from ..infrastructure.reranker import RerankerServiceClient
from ..infrastructure.vectorstores.hybrid import _extract_comparison_subqueries, _expand_query_aliases


_STRATEGY_GROUPS: dict[str, tuple[str, ...]] = {
    "salvage": ("salvage", "gdp-l-fucose", "gdp fucose", "补料", "补救"),
    "wcfb": ("wcfb", "fucosyltransferase", "末端转移", "岩藻糖基转移酶"),
    "chromosomal integration": (
        "chromosomal integration",
        "chromosomally integrated",
        "chromosomally integration",
        "染色体整合",
        "integrated expression cassette",
    ),
}
_ROUTE_INTENT_MARKERS = ("工程化", "合成路径", "关键前体", "催化步骤", "biosynthesis", "pathway", "precursor")
_ROUTE_POSITIVE_TERMS = (
    "biosynthesis",
    "production",
    "engineered",
    "metabolic engineering",
    "synthesis",
    "pathway",
    "cmp-neu5ac",
    "gdp-l-fucose",
    "gdp-fucose",
    "sialyltransferase",
    "fucosyltransferase",
)
_ROUTE_NEGATIVE_TERMS = (
    "protective effect",
    "covid",
    "sars",
    "necrotizing",
    "inflammation",
    "treatment",
    "review",
    "progress",
    "application",
    "health effects",
)
_NUMERIC_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|g/L|mg/L|mg|g|mM|uM|µM|h|hr|hours?|fold|times?|x\b)",
    flags=re.IGNORECASE,
)
_RESULT_TERMS = (
    "yield",
    "production",
    "produced",
    "titer",
    "titre",
    "result",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "improved",
    "enhanced",
    "产量",
    "产率",
    "滴度",
    "提高",
    "降低",
    "增加",
    "减少",
    "结果",
)
_DEFINITION_TERMS = (
    "defined as",
    "is defined",
    "acts as",
    "act as",
    "functions as",
    "function as",
    "refers to",
    "known as",
    "termed",
    "指",
    "定义为",
)


class QwenReranker:
    """
    演示版支持两种模式:
    - 配置了 API: 使用 OpenAI-compatible chat 接口做 JSON 打分
    - 未配置 API: 使用启发式重排
    """

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model_name: str = "qwen-rerank",
        model_path: str = "",
        service_url: str = "",
        batch_size: int = 8,
        use_fp16: bool = True,
        retrieval_config: RetrievalConfig | None = None,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.client = OpenAICompatibleClient(api_base, api_key, timeout_seconds=30)
        self.service_client = RerankerServiceClient(service_url) if service_url else None
        self.local_reranker: Any | None = None
        self.last_debug: dict[str, object] = {}
        if model_path and Path(model_path).exists():
            try:
                from ..infrastructure.reranker.local_bge import LocalBGEReranker

                self.local_reranker = LocalBGEReranker(
                    model_path=model_path,
                    use_fp16=use_fp16,
                    batch_size=batch_size,
                )
            except Exception:
                self.local_reranker = None

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None = None,
    ) -> list[RetrievedChunk]:
        self.last_debug = {}
        queries = _build_rerank_queries(question, analysis)
        self.last_debug["query_variants"] = queries
        if self.service_client and chunks:
            try:
                return self._rerank_with_service(queries, chunks, top_k, analysis)
            except Exception:
                pass
        if self.local_reranker and chunks:
            try:
                return self._rerank_with_local_model(queries, chunks, top_k, analysis)
            except Exception:
                pass
        if self.client.is_enabled() and chunks:
            try:
                return self._rerank_with_llm(queries, chunks, top_k, analysis)
            except Exception:
                pass
        q_terms = set(" ".join(queries).lower().split())
        rescored: list[RetrievedChunk] = []
        for chunk in chunks:
            overlap = len(q_terms.intersection(_rerank_text(chunk).lower().split()))
            base_score = max(chunk.vector_score, chunk.fusion_score, chunk.bm25_score * 0.1)
            chunk.rerank_score = base_score * 0.7 + min(overlap, 10) * 0.03
            chunk.rerank_score += _strategy_bonus(question, chunk, self.retrieval_config)
            chunk.rerank_score += _evidence_aware_bonus(chunk, self.retrieval_config)
            rescored.append(chunk)
        rescored.sort(key=_sort_key, reverse=True)
        rescored = _apply_rerank_score_floor(rescored, self.retrieval_config)
        final = _apply_rerank_diversity(rescored, top_k, analysis, self.retrieval_config)
        self.last_debug["final_hits"] = _serialize_hits(final[:5])
        return final

    def _rerank_with_service(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None,
    ) -> list[RetrievedChunk]:
        texts = [_rerank_text(chunk) for chunk in chunks]
        scores_by_query = [self.service_client.score(query, texts) for query in queries]
        return self._aggregate_scores(queries, chunks, scores_by_query, top_k, analysis)

    def _rerank_with_local_model(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None,
    ) -> list[RetrievedChunk]:
        texts = [_rerank_text(chunk) for chunk in chunks]
        pairs = [[query, text] for query in queries for text in texts]
        raw_scores = self.local_reranker.score_pairs(pairs)
        scores_by_query: list[list[float]] = []
        cursor = 0
        for _query in queries:
            scores_by_query.append([float(score) for score in raw_scores[cursor : cursor + len(chunks)]])
            cursor += len(chunks)
        return self._aggregate_scores(queries, chunks, scores_by_query, top_k, analysis)

    def _rerank_with_llm(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None,
    ) -> list[RetrievedChunk]:
        scores_by_query = [self._score_with_llm(query, chunks) for query in queries]
        return self._aggregate_scores(queries, chunks, scores_by_query, top_k, analysis)

    def _score_with_llm(self, question: str, chunks: list[RetrievedChunk]) -> list[float]:
        chunk_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_blocks.append(f"id={idx}; text={_rerank_text(chunk)[:1400]}")
        content = self.client.chat_completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是严谨的检索重排序模型。输出 JSON。"},
                {
                    "role": "user",
                    "content": (
                        "请根据用户问题评估每个片段的相关性，"
                        '返回 JSON，格式为 {"items":[{"id":1,"score":0.91}] }。'
                        f"\n用户问题: {question}\n候选片段:\n" + "\n".join(chunk_blocks)
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = extract_json_block(content)
        items = parsed.get("items", [])
        by_id = {int(item["id"]): float(item["score"]) for item in items}
        return [by_id.get(idx, chunks[idx - 1].vector_score) for idx in range(1, len(chunks) + 1)]

    def _aggregate_scores(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        scores_by_query: list[list[float]],
        top_k: int,
        analysis: QueryAnalysis | None,
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
            chunk.rerank_score = max_score + alpha * mean_score + bonus + route_bonus + evidence_bonus
            chunk.metadata["rerank_query_scores"] = [round(score, 6) for score in per_query_scores]
            rescored.append(chunk)

        rescored.sort(key=_sort_key, reverse=True)
        rescored = _apply_rerank_score_floor(rescored, self.retrieval_config)
        final = _apply_comparison_coverage_selection(
            chunks=rescored,
            queries=queries,
            analysis=analysis,
            config=self.retrieval_config,
            top_k=top_k,
        )
        self.last_debug["query_scores"] = query_debug
        self.last_debug["final_hits"] = _serialize_hits(final[:5])
        return final


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


def _apply_rerank_score_floor(chunks: list[RetrievedChunk], config: RetrievalConfig) -> list[RetrievedChunk]:
    if not chunks or config.rerank_score_floor_ratio <= 0:
        return chunks
    top_score = chunks[0].rerank_score
    if top_score <= 0:
        return chunks
    floor = top_score * config.rerank_score_floor_ratio
    return [chunk for chunk in chunks if chunk.rerank_score >= floor]


def _build_rerank_queries(question: str, analysis: QueryAnalysis | None) -> list[str]:
    queries = [_expand_query_aliases(question)]
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return queries
    for subquery in _extract_comparison_subqueries(question):
        expanded = _expand_query_aliases(subquery)
        if expanded and expanded not in queries:
            queries.append(expanded)
    return queries


def _apply_rerank_diversity(
    chunks: list[RetrievedChunk],
    top_k: int,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return chunks[:top_k]
    max_per_doc = max(1, config.comparison_rerank_max_chunks_per_doc)
    selected: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    counts: Counter[str] = Counter()
    for chunk in chunks:
        if counts[chunk.doc_id] < max_per_doc:
            selected.append(chunk)
            counts[chunk.doc_id] += 1
        else:
            overflow.append(chunk)
        if len(selected) >= top_k:
            return selected[:top_k]
    for chunk in overflow:
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def _apply_comparison_coverage_selection(
    chunks: list[RetrievedChunk],
    queries: list[str],
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
    top_k: int,
) -> list[RetrievedChunk]:
    diversified = _apply_rerank_diversity(chunks, top_k=len(chunks), analysis=analysis, config=config)
    if not analysis or analysis.intent != QueryIntent.COMPARISON or len(queries) <= 1:
        return diversified[:top_k]
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    for query_idx in range(1, len(queries)):
        best_chunk = None
        best_score = None
        for chunk in diversified:
            if chunk.doc_id in selected_ids:
                continue
            query_scores = chunk.metadata.get("rerank_query_scores") or []
            if query_idx >= len(query_scores):
                continue
            score = float(query_scores[query_idx])
            if best_score is None or score > best_score:
                best_score = score
                best_chunk = chunk
        if best_chunk is None:
            continue
        if best_score is not None and best_score < 1.0:
            continue
        selected.append(best_chunk)
        selected_ids.add(best_chunk.doc_id)
        if len(selected) >= top_k:
            return selected[:top_k]
    for chunk in diversified:
        if chunk.doc_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.doc_id)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def _strategy_bonus(question: str, chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    lowered_question = question.lower()
    haystack = "\n".join(
        part.lower()
        for part in (chunk.title, chunk.section, chunk.text[:800])
        if part
    )
    bonus = 0.0
    for aliases in _STRATEGY_GROUPS.values():
        if not any(alias.lower() in lowered_question for alias in aliases):
            continue
        if any(alias.lower() in haystack for alias in aliases):
            bonus += config.rerank_strategy_bonus
    return bonus


def _route_bonus(question: str, chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    lowered_question = question.lower()
    if not any(marker.lower() in lowered_question for marker in _ROUTE_INTENT_MARKERS):
        return 0.0
    haystack = "\n".join(
        part.lower()
        for part in (chunk.title, chunk.section, chunk.text[:1200])
        if part
    )
    bonus = 0.0
    positive_hits = sum(1 for term in _ROUTE_POSITIVE_TERMS if term in haystack)
    negative_hits = sum(1 for term in _ROUTE_NEGATIVE_TERMS if term in haystack)
    if positive_hits:
        bonus += min(positive_hits, 3) * (config.rerank_strategy_bonus * 0.4)
    if negative_hits:
        bonus -= min(negative_hits, 2) * (config.rerank_strategy_bonus * 0.5)
    if ("前体" in question or "precursor" in lowered_question) and any(
        term in haystack for term in ("gdp-l-fucose", "gdp-fucose", "cmp-neu5ac")
    ):
        bonus += config.rerank_strategy_bonus * 5.0
    if ("催化步骤" in question or "catalytic" in lowered_question or "末端" in question) and any(
        term in haystack for term in ("fucosyltransferase", "sialyltransferase", "alpha-1,2-ft", "alpha2,6")
    ):
        bonus += config.rerank_strategy_bonus
    if any(term in lowered_question for term in ("2′-fl", "2'-fl")) and any(
        term in haystack for term in ("salvage pathway", "gdp-l-fucose")
    ):
        bonus += config.rerank_strategy_bonus * 0.8
    return bonus


def _evidence_aware_bonus(chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    text = _rerank_text(chunk)
    lowered = f" {text.lower()} "
    numeric_feature = 1.0 if _NUMERIC_RE.search(text) or re.search(r"\d+", text) else 0.0
    result_feature = 1.0 if any(term in lowered for term in _RESULT_TERMS) else 0.0
    definition_feature = 1.0 if any(term in lowered for term in _DEFINITION_TERMS) else 0.0
    section_bonus = _section_bonus(chunk.section, config)
    bonus = (
        config.evidence_numeric_bonus * numeric_feature
        + config.evidence_result_bonus * result_feature
        + config.evidence_definition_bonus * definition_feature
        + section_bonus
    )
    chunk.metadata["evidence_features"] = {
        "numeric": bool(numeric_feature),
        "result": bool(result_feature),
        "definition": bool(definition_feature),
        "section_bonus": round(section_bonus, 4),
        "bonus": round(bonus, 4),
    }
    return bonus


def _section_bonus(section: str, config: RetrievalConfig) -> float:
    normalized = (section or "").strip().lower()
    if "result" in normalized:
        return config.section_results_bonus
    if "discussion" in normalized:
        return config.section_discussion_bonus
    if normalized == "abstract":
        return config.section_abstract_bonus
    if "introduction" in normalized:
        return config.section_introduction_penalty
    return 0.0


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
