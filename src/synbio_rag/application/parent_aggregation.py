from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.config import RetrievalConfig
from ..domain.schemas import RetrievedChunk
from ..infrastructure.index.parent_store import ParentStore


@dataclass
class _ParentBucket:
    first_raw_rank: int
    parent: RetrievedChunk
    children: list[RetrievedChunk] = field(default_factory=list)
    snippets: list[dict[str, Any]] = field(default_factory=list)


def aggregate_parent_hits_with_child_scores(
    chunks: list[RetrievedChunk],
    parent_store: ParentStore,
    config: RetrievalConfig,
    debug: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        if debug is not None:
            debug.update(
                {
                    "parent_aggregation_used": True,
                    "parent_aggregation_reason": "empty_input",
                    "parent_candidate_count": 0,
                }
            )
        return []

    buckets: dict[str, _ParentBucket] = {}
    for raw_rank, child in enumerate(chunks, start=1):
        _metadata(child)["raw_retrieval_rank"] = raw_rank
        materialized = parent_store.materialize_parent_hit(child)
        parent_key = str(materialized.chunk_id or child.chunk_id or "")
        if not parent_key:
            continue

        bucket = buckets.get(parent_key)
        if bucket is None:
            bucket = _ParentBucket(
                first_raw_rank=raw_rank,
                parent=materialized,
            )
            buckets[parent_key] = bucket
        else:
            bucket.first_raw_rank = min(bucket.first_raw_rank, raw_rank)
            _merge_parent_scores(bucket.parent, materialized)

        bucket.children.append(child)
        _append_unique_strings(
            bucket.parent.metadata,
            "matched_child_chunk_ids",
            materialized.metadata.get("matched_child_chunk_ids") or [],
        )
        for snippet in materialized.metadata.get("matched_child_snippets") or []:
            if isinstance(snippet, dict):
                bucket.snippets.append(_enrich_snippet(snippet, child, raw_rank))

    features_by_key = {
        parent_key: _bucket_features(bucket)
        for parent_key, bucket in buckets.items()
    }
    best_probe_values = [
        features["best_child_probe_score"]
        for features in features_by_key.values()
    ]
    mean_probe_values = [
        features["mean_top3_child_probe_score"]
        for features in features_by_key.values()
    ]
    fusion_values = [features["best_fusion_score"] for features in features_by_key.values()]

    for parent_key, bucket in buckets.items():
        features = features_by_key[parent_key]
        score = (
            0.55
            * _normalize_feature(features["best_child_probe_score"], best_probe_values)
            + 0.20
            * _normalize_feature(features["mean_top3_child_probe_score"], mean_probe_values)
            + 0.15 * _normalize_feature(features["best_fusion_score"], fusion_values)
            + 0.05 * min(features["matched_child_count"], 4) / 4
            + 0.05 * features["dual_source_hit"]
        )
        _write_parent_metadata(bucket, features, score)

    result = sorted(
        (bucket.parent for bucket in buckets.values()),
        key=lambda chunk: (
            -_float(chunk.metadata.get("parent_aggregation_score")),
            int(chunk.metadata.get("parent_aggregation_first_raw_rank") or 0),
        ),
    )
    if debug is not None:
        debug.update(
            {
                "parent_aggregation_used": True,
                "parent_aggregation_reason": "enabled",
                "child_probe_top_n": int(config.parent_aggregation_child_rerank_top_n),
                "parent_candidate_count": len(result),
                "top_parent_aggregation_hits": [
                    _debug_parent_hit(chunk)
                    for chunk in result[:5]
                ],
            }
        )
    return result


def _bucket_features(bucket: _ParentBucket) -> dict[str, float]:
    probe_scores = [
        _float(_metadata(child).get("child_probe_rerank_score"))
        for child in bucket.children
        if "child_probe_rerank_score" in _metadata(child)
    ]
    top3_probe_scores = sorted(probe_scores, reverse=True)[:3]
    mean_top3 = (
        sum(top3_probe_scores) / len(top3_probe_scores)
        if top3_probe_scores
        else 0.0
    )
    matched_child_ids = _matched_child_ids(bucket)
    has_dense = any(float(child.vector_score or 0.0) > 0.0 for child in bucket.children)
    has_bm25 = any(float(child.bm25_score or 0.0) > 0.0 for child in bucket.children)
    return {
        "best_child_probe_score": max(probe_scores) if probe_scores else 0.0,
        "mean_top3_child_probe_score": mean_top3,
        "best_fusion_score": max(
            [float(child.fusion_score or 0.0) for child in bucket.children] or [0.0]
        ),
        "matched_child_count": float(len(matched_child_ids)),
        "dual_source_hit": 1.0 if has_dense and has_bm25 else 0.0,
    }


def _write_parent_metadata(
    bucket: _ParentBucket,
    features: dict[str, float],
    score: float,
) -> None:
    snippets = _dedupe_snippets(bucket.snippets)
    snippets.sort(
        key=lambda snippet: (
            -_float(snippet.get("child_probe_rerank_score")),
            _int(snippet.get("raw_retrieval_rank"), default=10**9),
        )
    )
    child_ids = [
        str(snippet.get("chunk_id") or "")
        for snippet in snippets
        if str(snippet.get("chunk_id") or "").strip()
    ]
    if not child_ids:
        child_ids = _matched_child_ids(bucket)

    metadata = bucket.parent.metadata
    metadata.update(
        {
            "parent_aggregation_score": round(float(score), 6),
            "parent_aggregation_first_raw_rank": bucket.first_raw_rank,
            "matched_child_count": int(features["matched_child_count"]),
            "best_child_probe_score": round(features["best_child_probe_score"], 6),
            "mean_top3_child_probe_score": round(
                features["mean_top3_child_probe_score"],
                6,
            ),
            "best_child_fusion_score": round(features["best_fusion_score"], 6),
            "dual_source_hit": bool(features["dual_source_hit"]),
        }
    )
    if child_ids:
        metadata["matched_child_chunk_ids"] = child_ids
        metadata["matched_child_snippets"] = snippets


def _merge_parent_scores(current: RetrievedChunk, incoming: RetrievedChunk) -> None:
    current.vector_score = max(float(current.vector_score), float(incoming.vector_score))
    current.bm25_score = max(float(current.bm25_score), float(incoming.bm25_score))
    current.rerank_score = max(float(current.rerank_score), float(incoming.rerank_score))
    current.fusion_score = max(float(current.fusion_score), float(incoming.fusion_score))


def _enrich_snippet(
    snippet: dict[str, Any],
    child: RetrievedChunk,
    raw_rank: int,
) -> dict[str, Any]:
    enriched = dict(snippet)
    child_metadata = _metadata(child)
    enriched["raw_retrieval_rank"] = raw_rank
    for key in (
        "child_probe_rerank_score",
        "child_probe_rerank_rank",
        "child_probe_query_scores",
    ):
        if key in child_metadata:
            enriched[key] = child_metadata[key]
    return enriched


def _dedupe_snippets(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for snippet in snippets:
        chunk_id = str(snippet.get("chunk_id") or "")
        key = chunk_id or str(snippet)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
    return deduped


def _matched_child_ids(bucket: _ParentBucket) -> list[str]:
    ids = [
        str(value)
        for value in bucket.parent.metadata.get("matched_child_chunk_ids") or []
        if str(value or "").strip()
    ]
    for snippet in bucket.snippets:
        chunk_id = str(snippet.get("chunk_id") or "")
        if chunk_id:
            ids.append(chunk_id)
    return _unique_strings(ids)


def _append_unique_strings(metadata: dict[str, Any], key: str, values: list[Any]) -> None:
    merged = _unique_strings(
        [
            str(value)
            for value in list(metadata.get(key) or []) + list(values)
            if str(value or "").strip()
        ]
    )
    metadata[key] = merged


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_feature(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    if high <= low:
        return 1.0 if high != 0.0 else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _metadata(chunk: RetrievedChunk) -> dict[str, Any]:
    if not isinstance(chunk.metadata, dict):
        chunk.metadata = {}
    return chunk.metadata


def _debug_parent_hit(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "parent_aggregation_score": _round_float(
            chunk.metadata.get("parent_aggregation_score")
        ),
        "matched_child_count": int(chunk.metadata.get("matched_child_count") or 0),
        "best_child_probe_score": _round_float(
            chunk.metadata.get("best_child_probe_score")
        ),
        "first_raw_rank": int(
            chunk.metadata.get("parent_aggregation_first_raw_rank") or 0
        ),
    }


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _round_float(value: object) -> float:
    return round(_float(value), 6)
