from __future__ import annotations

from collections import Counter
from typing import Any

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from .rerank_common import _guarded_sort_key, _sort_key
from .rerank_guarded_policy import _apply_guarded_rerank, _apply_rank1_evidence_guard


def _rerank_score_floor_min_keep(config: RetrievalConfig, top_k: int) -> int:
    return min(max(int(config.rerank_score_floor_min_keep), 0), max(int(top_k), 0))


def _apply_rerank_score_floor(
    chunks: list[RetrievedChunk],
    config: RetrievalConfig,
    min_keep: int = 0,
) -> list[RetrievedChunk]:
    if not chunks or config.rerank_score_floor_ratio <= 0:
        return chunks
    top_score = chunks[0].rerank_score
    if top_score <= 0:
        return chunks
    floor = top_score * config.rerank_score_floor_ratio
    kept = [chunk for chunk in chunks if chunk.rerank_score >= floor]
    keep_target = min(max(int(min_keep), 0), len(chunks))
    if len(kept) >= keep_target:
        return kept

    kept_ids = {chunk.chunk_id for chunk in kept}
    for chunk in chunks:
        if len(kept) >= keep_target:
            break
        if chunk.chunk_id in kept_ids:
            continue
        kept.append(chunk)
        kept_ids.add(chunk.chunk_id)
    return kept


_BODY_SECTION_GROUPS: set[str] = {
    "Introduction", "Background", "Methods", "Materials and Methods",
    "Experimental Section", "Experimental Procedures", "Results",
    "Results and Discussion", "Discussion", "Conclusion", "Conclusions",
    "Full Text",
}


def _section_to_body_group(section: str) -> str:
    s = section.lower().strip()
    if s in ("introduction", "background"):
        return "INTRO"
    if s in ("methods", "materials and methods", "experimental section",
             "experimental procedures", "experimental methods"):
        return "METHOD"
    if s in ("results", "results and discussion"):
        return "RESULT"
    if s in ("discussion", "discussion and results"):
        return "DISCUSSION"
    if s in ("conclusion", "conclusions"):
        return "CONCLUSION"
    if s == "full text":
        return "BODY_ANY"
    if s == "abstract":
        return "ABSTRACT"
    if s == "title":
        return "TITLE"
    return "UNKNOWN"


def _apply_same_doc_body_coverage(
    selected: list[RetrievedChunk],
    pre_floor: list[RetrievedChunk],
    top_k: int,
    analysis,
    config,
) -> list[RetrievedChunk]:
    intent = getattr(analysis, "intent", None) if analysis else None
    intent_name = str(intent).split(".")[-1].lower() if intent else ""
    allowed_intents = config.same_doc_body_coverage_intents or ["factoid"]
    if intent_name not in allowed_intents:
        return selected

    margin = config.same_doc_body_coverage_margin
    max_total = config.same_doc_body_coverage_max_total

    pre_floor_ranked: dict[str, int] = {}
    for rank, c in enumerate(pre_floor):
        pre_floor_ranked[c.chunk_id] = rank

    docs_in_selected: dict[str, list[int]] = {}
    for idx, c in enumerate(selected):
        docs_in_selected.setdefault(c.doc_id, []).append(idx)

    def _find_victim(doc_id: str) -> int | None:
        for i in docs_in_selected.get(doc_id, []):
            sec = selected[i].section
            if sec == "Title":
                return i
        for i in docs_in_selected.get(doc_id, []):
            if sec == "Abstract":
                return i
        return None

    replaced = 0

    for doc_id, indices in docs_in_selected.items():
        if replaced >= max_total:
            break
        has_body = any(selected[i].section in _BODY_SECTION_GROUPS for i in indices)
        if has_body:
            continue
        victim_idx = _find_victim(doc_id)
        if victim_idx is None:
            continue
        best = _pick_best_body(pre_floor, doc_id, selected, pre_floor_ranked, top_k, margin)
        if best:
            _apply_replacement(selected, victim_idx, best, doc_id, top_k)
            replaced += 1

    if config.same_doc_section_group_coverage_level2_enabled and replaced < max_total:
        target_groups = {"INTRO", "METHOD"}
        for doc_id, indices in docs_in_selected.items():
            if replaced >= max_total:
                break
            covered_groups = set()
            for i in indices:
                g = _section_to_body_group(selected[i].section)
                if g not in ("TITLE", "ABSTRACT", "UNKNOWN"):
                    covered_groups.add(g)
            missing_groups = target_groups - covered_groups
            if not missing_groups:
                continue

            best = None
            best_rank = None
            for c in pre_floor:
                if c.doc_id != doc_id:
                    continue
                if any(c.chunk_id == s.chunk_id for s in selected):
                    continue
                g = _section_to_body_group(c.section)
                if g not in missing_groups:
                    continue
                rank = pre_floor_ranked.get(c.chunk_id, 999)
                if rank < top_k + margin:
                    if best is None or rank < best_rank:
                        best = c
                        best_rank = rank

            if best is None:
                continue

            victim_idx = _find_victim(doc_id)
            if victim_idx is None:
                continue

            _apply_replacement(selected, victim_idx, best, doc_id, top_k)
            replaced += 1

    return selected


def _pick_best_body(pre_floor, doc_id, selected, pre_floor_ranked, top_k, margin):
    best = None
    best_rank = None
    for c in pre_floor:
        if c.doc_id != doc_id or c.section not in _BODY_SECTION_GROUPS:
            continue
        if any(c.chunk_id == s.chunk_id for s in selected):
            continue
        rank = pre_floor_ranked.get(c.chunk_id, 999)
        if rank < top_k + margin:
            if best is None or rank < best_rank:
                best = c
                best_rank = rank
    return best


def _apply_replacement(selected, victim_idx, new_chunk, doc_id, top_k):
    victim = selected[victim_idx]
    new_chunk.metadata["added_by_body_coverage"] = True
    new_chunk.metadata["body_coverage_reason"] = (
        f"doc {doc_id} in top-{top_k} but missing body coverage"
    )
    new_chunk.metadata["body_coverage_victim"] = victim.chunk_id
    victim.metadata["dropped_by_body_coverage"] = True
    selected[victim_idx] = new_chunk


def _finalize_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
    mode: str,
    queries: list[str] | None = None,
    debug: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    score_floor_min_keep = _rerank_score_floor_min_keep(config, top_k)
    if mode in {"guarded", "guarded_rank1"}:
        profiled = _apply_guarded_rerank(question, chunks, config)
        if mode == "guarded_rank1":
            profiled = _apply_rank1_evidence_guard(profiled, config)
        profiled_ordered = sorted(profiled, key=_guarded_sort_key, reverse=True)
        if debug is not None:
            debug.update(
                {
                    "mode": mode,
                    "top_k": top_k,
                    "pre_floor_chunk_ids": _chunk_ids(profiled_ordered),
                    "pre_floor_rank_by_chunk_id": _rank_by_chunk_id(profiled_ordered),
                    "post_floor_chunk_ids": _chunk_ids(profiled_ordered),
                    "post_floor_rank_by_chunk_id": _rank_by_chunk_id(profiled_ordered),
                    "score_floor": {
                        "enabled": False,
                        "ratio": config.rerank_score_floor_ratio,
                        "floor": None,
                        "min_keep": score_floor_min_keep,
                        "rescued_chunk_ids": [],
                        "dropped_chunk_ids": [],
                    },
                }
            )
        final = _apply_comparison_coverage_selection(
            chunks=profiled,
            queries=queries or [question],
            analysis=analysis,
            config=config,
            top_k=top_k,
            sort_key=_guarded_sort_key,
            debug=debug,
        )
        if debug is not None:
            debug["final_chunk_ids"] = _chunk_ids(final)
            debug["final_rank_by_chunk_id"] = _rank_by_chunk_id(final)
            debug["final_doc_ids"] = [chunk.doc_id for chunk in final]
        return final
    chunks.sort(key=_sort_key, reverse=True)
    pre_floor = list(chunks)
    chunks = _apply_rerank_score_floor(chunks, config, min_keep=score_floor_min_keep)
    post_floor = list(chunks)
    if debug is not None:
        debug.update(
            {
                "mode": mode,
                "top_k": top_k,
                "pre_floor_chunk_ids": _chunk_ids(pre_floor),
                "pre_floor_rank_by_chunk_id": _rank_by_chunk_id(pre_floor),
                "post_floor_chunk_ids": _chunk_ids(post_floor),
                "post_floor_rank_by_chunk_id": _rank_by_chunk_id(post_floor),
                "score_floor": _score_floor_debug(
                    pre_floor,
                    post_floor,
                    config,
                    min_keep=score_floor_min_keep,
                ),
            }
        )
    final = _apply_comparison_coverage_selection(
        chunks=chunks,
        queries=queries or [question],
        analysis=analysis,
        config=config,
        top_k=top_k,
        debug=debug,
    )
    pre_body_coverage = list(final)
    if config.same_doc_body_coverage_enabled:
        final = _apply_same_doc_body_coverage(
            selected=final,
            pre_floor=pre_floor,
            top_k=len(final),
            analysis=analysis,
            config=config,
        )
    if debug is not None:
        debug["pre_body_coverage_final_chunk_ids"] = _chunk_ids(pre_body_coverage)
        debug["same_doc_body_coverage"] = {
            "applied": bool(config.same_doc_body_coverage_enabled),
            "changed": _chunk_ids(pre_body_coverage) != _chunk_ids(final),
            "added_chunk_ids": [
                chunk.chunk_id
                for chunk in final
                if chunk.chunk_id not in {item.chunk_id for item in pre_body_coverage}
            ],
            "dropped_chunk_ids": [
                chunk.chunk_id
                for chunk in pre_body_coverage
                if chunk.chunk_id not in {item.chunk_id for item in final}
            ],
        }
        debug["final_chunk_ids"] = _chunk_ids(final)
        debug["final_rank_by_chunk_id"] = _rank_by_chunk_id(final)
        debug["final_doc_ids"] = [chunk.doc_id for chunk in final]
    return final


def _apply_rerank_diversity(
    chunks: list[RetrievedChunk],
    top_k: int,
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
    debug: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        result = chunks[:top_k]
        if debug is not None:
            debug.update(
                {
                    "applied": False,
                    "reason": "not_comparison_intent",
                    "max_per_doc": None,
                    "overflow_chunk_ids": [],
                    "output_chunk_ids": _chunk_ids(result),
                }
            )
        return result
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
            result = selected[:top_k]
            if debug is not None:
                debug.update(
                    {
                        "applied": True,
                        "max_per_doc": max_per_doc,
                        "overflow_chunk_ids": _chunk_ids(overflow),
                        "output_chunk_ids": _chunk_ids(result),
                        "doc_counts": dict(counts),
                    }
                )
            return result
    for chunk in overflow:
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    result = selected[:top_k]
    if debug is not None:
        debug.update(
            {
                "applied": True,
                "max_per_doc": max_per_doc,
                "overflow_chunk_ids": _chunk_ids(overflow),
                "output_chunk_ids": _chunk_ids(result),
                "doc_counts": dict(counts),
            }
        )
    return result


def _apply_comparison_coverage_selection(
    chunks: list[RetrievedChunk],
    queries: list[str],
    analysis: QueryAnalysis | None,
    config: RetrievalConfig,
    top_k: int,
    sort_key=_sort_key,
    debug: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    ordered = sorted(chunks, key=sort_key, reverse=True)
    diversity_debug: dict[str, Any] = {}
    diversified = _apply_rerank_diversity(
        ordered,
        top_k=len(ordered),
        analysis=analysis,
        config=config,
        debug=diversity_debug,
    )
    if debug is not None:
        debug["ordered_chunk_ids_before_diversity"] = _chunk_ids(ordered)
        debug["doc_diversity"] = diversity_debug
    if not analysis or analysis.intent != QueryIntent.COMPARISON or len(queries) <= 1:
        result = diversified[:top_k]
        if debug is not None:
            debug["comparison_selection"] = {
                "applied": False,
                "reason": "not_comparison_or_single_query",
                "selected_chunk_ids": _chunk_ids(result),
            }
        return result
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    branch_steps: list[dict[str, Any]] = []
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
            branch_steps.append(
                {
                    "query_index": query_idx,
                    "selected_chunk_id": "",
                    "selected_doc_id": "",
                    "best_score": None,
                    "reason": "no_query_score",
                }
            )
            continue
        if best_score is not None and best_score < 1.0:
            branch_steps.append(
                {
                    "query_index": query_idx,
                    "selected_chunk_id": best_chunk.chunk_id,
                    "selected_doc_id": best_chunk.doc_id,
                    "best_score": round(float(best_score), 6),
                    "reason": "score_below_minimum",
                }
            )
            continue
        selected.append(best_chunk)
        selected_ids.add(best_chunk.doc_id)
        branch_steps.append(
            {
                "query_index": query_idx,
                "selected_chunk_id": best_chunk.chunk_id,
                "selected_doc_id": best_chunk.doc_id,
                "best_score": round(float(best_score), 6) if best_score is not None else None,
                "reason": "selected",
            }
        )
        if len(selected) >= top_k:
            result = selected[:top_k]
            if debug is not None:
                debug["comparison_selection"] = {
                    "applied": True,
                    "query_count": len(queries),
                    "branch_steps": branch_steps,
                    "selected_chunk_ids": _chunk_ids(result),
                }
            return result
    for chunk in diversified:
        if chunk.doc_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.doc_id)
        if len(selected) >= top_k:
            break
    result = selected[:top_k]
    if debug is not None:
        debug["comparison_selection"] = {
            "applied": True,
            "query_count": len(queries),
            "branch_steps": branch_steps,
            "selected_chunk_ids": _chunk_ids(result),
        }
    return result


def _chunk_ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [str(chunk.chunk_id or "") for chunk in chunks]


def _rank_by_chunk_id(chunks: list[RetrievedChunk]) -> dict[str, int]:
    return {str(chunk.chunk_id or ""): rank for rank, chunk in enumerate(chunks, start=1)}


def _score_floor_debug(
    pre_floor: list[RetrievedChunk],
    post_floor: list[RetrievedChunk],
    config: RetrievalConfig,
    min_keep: int = 0,
) -> dict[str, Any]:
    if not pre_floor:
        return {
            "enabled": False,
            "ratio": config.rerank_score_floor_ratio,
            "floor": None,
            "min_keep": max(int(min_keep), 0),
            "rescued_chunk_ids": [],
            "dropped_chunk_ids": [],
        }
    top_score = float(pre_floor[0].rerank_score or 0.0)
    enabled = config.rerank_score_floor_ratio > 0 and top_score > 0
    floor = top_score * config.rerank_score_floor_ratio if enabled else None
    post_ids = set(_chunk_ids(post_floor))
    rescued = [
        chunk.chunk_id
        for chunk in pre_floor
        if chunk.chunk_id in post_ids and floor is not None and chunk.rerank_score < floor
    ]
    dropped = [chunk.chunk_id for chunk in pre_floor if chunk.chunk_id not in post_ids]
    return {
        "enabled": enabled,
        "ratio": config.rerank_score_floor_ratio,
        "top_score": round(top_score, 6),
        "floor": round(float(floor), 6) if floor is not None else None,
        "min_keep": max(int(min_keep), 0),
        "rescued_chunk_ids": rescued,
        "dropped_chunk_ids": dropped,
    }
