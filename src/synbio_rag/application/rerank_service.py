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
_TABLE_QUERY_HINTS = ("table", "primer", "sequence", "strain", "vmax", "km", "relative peak area", "glycan", "parameter")
_FIGURE_QUERY_HINTS = ("figure", "fig.", "fig ", "图")
_DOC_ID_RE = re.compile(r"\bdoc[_-]?\d{4}\b", flags=re.IGNORECASE)
_TABLE_NUMBER_RE = re.compile(r"\btable\s*([a-z]?\d+)\b", flags=re.IGNORECASE)
_FIGURE_NUMBER_RE = re.compile(r"\b(?:figure|fig\.?)\s*([a-z]?\d+)\b", flags=re.IGNORECASE)
_ANCHOR_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'._-]{1,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "describe",
    "described",
    "does",
    "for",
    "in",
    "is",
    "listed",
    "main",
    "of",
    "on",
    "shown",
    "the",
    "their",
    "these",
    "this",
    "to",
    "values",
    "what",
    "which",
}


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
        if self.service_client and chunks:
            try:
                return self._rerank_with_service(queries, chunks, top_k, analysis, mode)
            except Exception:
                pass
        if self.local_reranker and chunks:
            try:
                return self._rerank_with_local_model(queries, chunks, top_k, analysis, mode)
            except Exception:
                pass
        if self.client.is_enabled() and chunks:
            try:
                return self._rerank_with_llm(queries, chunks, top_k, analysis, mode)
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
        final = _finalize_rerank(
            question=question,
            chunks=rescored,
            top_k=top_k,
            analysis=analysis,
            config=self.retrieval_config,
            mode=mode,
        )
        self.last_debug["final_hits"] = _serialize_hits(final[:5])
        return final

    def _rerank_with_service(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None,
        mode: str,
    ) -> list[RetrievedChunk]:
        texts = [_rerank_text(chunk) for chunk in chunks]
        scores_by_query = [self.service_client.score(query, texts) for query in queries]
        return self._aggregate_scores(queries, chunks, scores_by_query, top_k, analysis, mode)

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

    def _rerank_with_llm(
        self,
        queries: list[str],
        chunks: list[RetrievedChunk],
        top_k: int,
        analysis: QueryAnalysis | None,
        mode: str,
    ) -> list[RetrievedChunk]:
        scores_by_query = [self._score_with_llm(query, chunks) for query in queries]
        return self._aggregate_scores(queries, chunks, scores_by_query, top_k, analysis, mode)

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


def _guarded_sort_key(chunk: RetrievedChunk) -> tuple[float, float, float]:
    priority = float(chunk.metadata.get("guarded_rank1_priority", 0.0))
    guarded = float(chunk.metadata.get("guarded_score", chunk.rerank_score))
    completeness = float(chunk.metadata.get("guarded_keyword_completeness", 0.0))
    marker = float(chunk.metadata.get("guarded_marker_score", 0.0))
    return (priority, guarded, completeness, marker)


def _apply_rerank_score_floor(chunks: list[RetrievedChunk], config: RetrievalConfig) -> list[RetrievedChunk]:
    if not chunks or config.rerank_score_floor_ratio <= 0:
        return chunks
    top_score = chunks[0].rerank_score
    if top_score <= 0:
        return chunks
    floor = top_score * config.rerank_score_floor_ratio
    return [chunk for chunk in chunks if chunk.rerank_score >= floor]


def _finalize_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
    mode: str,
    queries: list[str] | None = None,
) -> list[RetrievedChunk]:
    if mode in {"guarded", "guarded_rank1"}:
        profiled = _apply_guarded_rerank(question, chunks, config)
        if mode == "guarded_rank1":
            profiled = _apply_rank1_evidence_guard(profiled, config)
        final = _apply_comparison_coverage_selection(
            chunks=profiled,
            queries=queries or [question],
            analysis=analysis,
            config=config,
            top_k=top_k,
            sort_key=_guarded_sort_key,
        )
        return final
    chunks.sort(key=_sort_key, reverse=True)
    chunks = _apply_rerank_score_floor(chunks, config)
    return _apply_comparison_coverage_selection(
        chunks=chunks,
        queries=queries or [question],
        analysis=analysis,
        config=config,
        top_k=top_k,
    )


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
    sort_key=_sort_key,
) -> list[RetrievedChunk]:
    ordered = sorted(chunks, key=sort_key, reverse=True)
    diversified = _apply_rerank_diversity(ordered, top_k=len(ordered), analysis=analysis, config=config)
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


def _structure_marker_bonus(question: str, chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    lowered_question = question.lower()
    text = _rerank_text(chunk).lower()
    bonus = 0.0
    if any(hint in lowered_question for hint in _TABLE_QUERY_HINTS):
        if "[table text]" in text:
            bonus += config.table_text_boost
        if "[table caption]" in text:
            bonus += config.table_caption_boost
    if any(hint in lowered_question for hint in _FIGURE_QUERY_HINTS) and "[figure caption]" in text:
        bonus += config.figure_caption_boost
    return bonus


def _normalize_query_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("′", "'").replace("’", "'").replace("‘", "'")
    lowered = lowered.replace("–", "-").replace("—", "-").replace("−", "-")
    lowered = lowered.replace("_", " ")
    lowered = lowered.replace("fig.", "figure ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _extract_query_profile(question: str) -> dict[str, object]:
    normalized = _normalize_query_text(question)
    compact = re.sub(r"[\s/_-]+", "", normalized)
    intent = "body"
    if any(hint in normalized for hint in _TABLE_QUERY_HINTS):
        intent = "table"
    elif any(hint in normalized for hint in _FIGURE_QUERY_HINTS):
        intent = "figure"
    anchors: list[str] = []
    for match in _TABLE_NUMBER_RE.findall(normalized):
        anchors.append(f"table {match}")
    for match in _FIGURE_NUMBER_RE.findall(normalized):
        anchors.append(f"figure {match}")
    for token in _ANCHOR_TOKEN_RE.findall(normalized):
        if token in _STOPWORDS:
            continue
        compact_token = re.sub(r"[\s/_-]+", "", token)
        if len(compact_token) <= 1:
            continue
        if (
            any(char.isdigit() for char in compact_token)
            or "-" in token
            or "_" in token
            or compact_token in {"km", "vmax", "hmos", "glycom", "man8", "man7", "man10"}
            or len(compact_token) >= 5
        ):
            anchors.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        anchor_norm = re.sub(r"[\s/_-]+", "", _normalize_query_text(anchor))
        if not anchor_norm or anchor_norm in seen:
            continue
        seen.add(anchor_norm)
        deduped.append(anchor)
    doc_id = None
    doc_match = _DOC_ID_RE.search(question)
    if doc_match:
        doc_id = doc_match.group(0).replace("-", "_").lower()
    return {
        "intent": intent,
        "normalized": normalized,
        "compact": compact,
        "anchors": deduped,
        "doc_id": doc_id,
        "table_refs": _TABLE_NUMBER_RE.findall(normalized),
        "figure_refs": _FIGURE_NUMBER_RE.findall(normalized),
    }


def _keyword_match_score(text: str, anchors: list[str]) -> tuple[float, list[str]]:
    if not anchors:
        return 0.0, []
    normalized = _normalize_query_text(text)
    compact = re.sub(r"[\s/_-]+", "", normalized)
    matched: list[str] = []
    for anchor in anchors:
        anchor_norm = _normalize_query_text(anchor)
        anchor_compact = re.sub(r"[\s/_-]+", "", anchor_norm)
        if anchor_norm in normalized or anchor_compact in compact:
            matched.append(anchor)
            continue
        stripped = anchor_norm.replace("'", "")
        if stripped and stripped in normalized.replace("'", ""):
            matched.append(anchor)
    return min(1.0, len(matched) / max(1, len(anchors))), matched


def _evidence_marker_score(profile: dict[str, object], text: str) -> float:
    lowered = text.lower()
    if profile["intent"] == "table":
        score = 0.0
        if "[table text]" in lowered:
            score += 0.7
        if "[table caption]" in lowered:
            score += 0.4
        return min(score, 1.0)
    if profile["intent"] == "figure":
        return 1.0 if "[figure caption]" in lowered else 0.0
    return 0.0


def _marker_flags(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "table_text": "[table text]" in lowered,
        "table_caption": "[table caption]" in lowered,
        "figure_caption": "[figure caption]" in lowered,
    }


def _reference_match_bonus(profile: dict[str, object], text: str) -> float:
    lowered = text.lower()
    if profile["intent"] == "table":
        refs = profile.get("table_refs") or []
        return 1.0 if any(f"table {ref}".lower() in lowered for ref in refs) else 0.0
    if profile["intent"] == "figure":
        refs = profile.get("figure_refs") or []
        return 1.0 if any(f"figure {ref}".lower() in lowered for ref in refs) else 0.0
    return 0.0


def _doc_route_score(profile: dict[str, object], chunk: RetrievedChunk) -> float:
    if not profile.get("doc_id"):
        return 0.0
    return 1.0 if (chunk.doc_id or "").lower() == profile["doc_id"] else 0.0


def _incomplete_evidence_penalty(
    profile: dict[str, object],
    keyword_completeness: float,
    marker_score: float,
    reference_bonus: float,
) -> float:
    if profile["intent"] == "body":
        return 0.0
    penalty = 0.0
    if keyword_completeness < 0.35:
        penalty += 1.0
    if marker_score <= 0.0 and reference_bonus <= 0.0:
        penalty += 0.35
    return penalty


def _completeness_score(profile: dict[str, object], chunk: RetrievedChunk) -> float:
    keyword = float(chunk.metadata.get("guarded_keyword_completeness", 0.0))
    marker = float(chunk.metadata.get("guarded_marker_score", 0.0))
    reference = float(chunk.metadata.get("guarded_reference_bonus", 0.0))
    doc_score = float(chunk.metadata.get("guarded_doc_score", 0.0))
    flags = chunk.metadata.get("guarded_marker_flags", {}) or {}
    if profile["intent"] == "figure":
        caption_bonus = 0.1 if flags.get("figure_caption") else 0.0
        return min(1.0, 0.20 * keyword + 0.35 * marker + 0.30 * reference + 0.05 * doc_score + caption_bonus)
    if profile["intent"] == "table":
        caption_bonus = 0.12 if flags.get("table_caption") else 0.0
        text_bonus = 0.04 if flags.get("table_text") else 0.0
        return min(1.0, 0.40 * keyword + 0.20 * marker + 0.20 * reference + 0.08 * doc_score + caption_bonus + text_bonus)
    return 0.50 * keyword + 0.35 * reference + 0.15 * doc_score


def _is_complete_rank1_evidence(profile: dict[str, object], chunk: RetrievedChunk) -> bool:
    keyword = float(chunk.metadata.get("guarded_keyword_completeness", 0.0))
    marker = float(chunk.metadata.get("guarded_marker_score", 0.0))
    reference = float(chunk.metadata.get("guarded_reference_bonus", 0.0))
    flags = chunk.metadata.get("guarded_marker_flags", {}) or {}
    if profile["intent"] == "figure":
        return bool(flags.get("figure_caption")) and reference >= 1.0 and keyword >= 0.45
    if profile["intent"] == "table":
        has_caption = bool(flags.get("table_caption"))
        has_text = bool(flags.get("table_text"))
        requires_reference = bool(profile.get("table_refs"))
        reference_ok = reference >= 1.0 if requires_reference else (reference >= 1.0 or has_caption)
        return has_text and keyword >= 0.55 and (has_caption or keyword >= 0.78) and reference_ok
    return True


def _apply_rank1_evidence_guard(
    chunks: list[RetrievedChunk],
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    profile = chunks[0].metadata.get("guarded_profile") or {}
    if profile.get("intent") not in {"table", "figure"}:
        for chunk in chunks:
            chunk.metadata["guarded_rank1_guard_triggered"] = False
            chunk.metadata["guarded_rank1_guard_reason"] = "intent_not_guarded"
        return chunks

    current_top = chunks[0]
    top_complete = _completeness_score(profile, current_top)
    current_top.metadata["guarded_completeness_score"] = round(top_complete, 6)

    for idx, chunk in enumerate(chunks[1:], start=1):
        chunk.metadata["guarded_completeness_score"] = round(_completeness_score(profile, chunk), 6)
        chunk.metadata["guarded_rank1_guard_triggered"] = False
        chunk.metadata["guarded_rank1_guard_reason"] = "not_promoted"
        chunk.metadata["guarded_rank1_promoted_from"] = None

    current_top.metadata["guarded_rank1_guard_triggered"] = False
    current_top.metadata["guarded_rank1_guard_reason"] = "top1_retained"
    current_top.metadata["guarded_rank1_promoted_from"] = None
    current_top.metadata["guarded_rank1_priority"] = 0.0

    if _is_complete_rank1_evidence(profile, current_top):
        current_top.metadata["guarded_rank1_guard_reason"] = "top1_already_complete"
        return chunks

    best_idx: int | None = None
    best_score = top_complete
    top_guarded = float(current_top.metadata.get("guarded_score", current_top.rerank_score))
    for idx, chunk in enumerate(chunks[1:], start=1):
        candidate_complete = float(chunk.metadata.get("guarded_completeness_score", 0.0))
        candidate_guarded = float(chunk.metadata.get("guarded_score", chunk.rerank_score))
        candidate_doc_score = float(chunk.metadata.get("guarded_doc_score", 0.0))
        gain = candidate_complete - top_complete
        if gain < config.guarded_rank1_min_completeness_gain:
            continue
        if profile.get("doc_id") and candidate_doc_score < 1.0:
            continue
        if not _is_complete_rank1_evidence(profile, chunk):
            continue
        top_flags = current_top.metadata.get("guarded_marker_flags", {}) or {}
        candidate_flags = chunk.metadata.get("guarded_marker_flags", {}) or {}
        if profile["intent"] == "figure":
            if not candidate_flags.get("figure_caption"):
                continue
            if candidate_guarded + max(config.guarded_rank1_max_score_gap, 0.65) < top_guarded:
                continue
        else:
            top_has_caption = bool(top_flags.get("table_caption"))
            candidate_has_caption = bool(candidate_flags.get("table_caption"))
            candidate_reference = float(chunk.metadata.get("guarded_reference_bonus", 0.0))
            top_reference = float(current_top.metadata.get("guarded_reference_bonus", 0.0))
            if candidate_guarded + config.guarded_rank1_max_score_gap < top_guarded and not (
                (candidate_has_caption and not top_has_caption)
                or (candidate_reference > top_reference and candidate_complete >= 0.95)
            ):
                continue
        if candidate_complete > best_score:
            best_idx = idx
            best_score = candidate_complete

    if best_idx is None:
        current_top.metadata["guarded_rank1_guard_reason"] = "no_better_complete_evidence"
        return chunks

    promoted = chunks.pop(best_idx)
    promoted.metadata["guarded_rank1_guard_triggered"] = True
    promoted.metadata["guarded_rank1_guard_reason"] = "promoted_complete_evidence"
    promoted.metadata["guarded_rank1_promoted_from"] = best_idx + 1
    promoted.metadata["guarded_rank1_priority"] = 1.0
    current_top.metadata["guarded_rank1_guard_reason"] = "demoted_incomplete_evidence"
    current_top.metadata["guarded_rank1_priority"] = 0.0
    chunks.insert(0, promoted)
    return chunks


def _normalize_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (value - low) / (high - low)


def _apply_guarded_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    profile = _extract_query_profile(question)
    hybrid_scores = [max(float(chunk.fusion_score), float(chunk.vector_score), float(chunk.bm25_score) * 0.1) for chunk in chunks]
    rerank_scores = [float(chunk.rerank_score) for chunk in chunks]
    h_low, h_high = min(hybrid_scores), max(hybrid_scores)
    r_low, r_high = min(rerank_scores), max(rerank_scores)
    for chunk, hybrid_score, rerank_score in zip(chunks, hybrid_scores, rerank_scores):
        text = _rerank_text(chunk)
        flags = _marker_flags(text)
        keyword_completeness, matched_anchors = _keyword_match_score(text, profile["anchors"])
        marker_score = _evidence_marker_score(profile, text)
        reference_bonus = _reference_match_bonus(profile, text)
        doc_score = _doc_route_score(profile, chunk)
        hybrid_norm = _normalize_score(hybrid_score, h_low, h_high)
        rerank_norm = _normalize_score(rerank_score, r_low, r_high)
        penalty = _incomplete_evidence_penalty(
            profile=profile,
            keyword_completeness=keyword_completeness,
            marker_score=marker_score,
            reference_bonus=reference_bonus,
        )
        if profile["intent"] == "body":
            final_score = (
                0.40 * hybrid_norm
                + 0.40 * rerank_norm
                + 0.15 * keyword_completeness
                + 0.05 * doc_score
            )
        else:
            final_score = (
                config.guarded_hybrid_weight * hybrid_norm
                + config.guarded_reranker_weight * rerank_norm
                + config.guarded_keyword_weight * keyword_completeness
                + config.guarded_marker_weight * min(1.0, marker_score + 0.3 * reference_bonus)
                + config.guarded_doc_weight * doc_score
                - config.guarded_incomplete_penalty * penalty
            )
        chunk.metadata["guarded_profile"] = profile
        chunk.metadata["guarded_keyword_completeness"] = round(keyword_completeness, 6)
        chunk.metadata["guarded_marker_score"] = round(marker_score, 6)
        chunk.metadata["guarded_reference_bonus"] = round(reference_bonus, 6)
        chunk.metadata["guarded_doc_score"] = round(doc_score, 6)
        chunk.metadata["guarded_hybrid_norm"] = round(hybrid_norm, 6)
        chunk.metadata["guarded_rerank_norm"] = round(rerank_norm, 6)
        chunk.metadata["guarded_penalty"] = round(penalty, 6)
        chunk.metadata["guarded_matched_anchors"] = matched_anchors
        chunk.metadata["guarded_marker_flags"] = flags
        chunk.metadata["guarded_score"] = round(final_score, 6)
        chunk.metadata["guarded_completeness_score"] = round(_completeness_score(profile, chunk), 6)
        chunk.metadata["guarded_rank1_guard_triggered"] = False
        chunk.metadata["guarded_rank1_guard_reason"] = "not_evaluated"
        chunk.metadata["guarded_rank1_promoted_from"] = None
        chunk.metadata["guarded_rank1_priority"] = 0.0
    chunks.sort(key=_guarded_sort_key, reverse=True)
    return chunks


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
