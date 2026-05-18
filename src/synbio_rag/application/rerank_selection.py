from __future__ import annotations

from collections import Counter

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from .rerank_common import _guarded_sort_key, _sort_key
from .rerank_guarded_policy import _apply_guarded_rerank, _apply_rank1_evidence_guard


def _apply_rerank_score_floor(chunks: list[RetrievedChunk], config: RetrievalConfig) -> list[RetrievedChunk]:
    if not chunks or config.rerank_score_floor_ratio <= 0:
        return chunks
    top_score = chunks[0].rerank_score
    if top_score <= 0:
        return chunks
    floor = top_score * config.rerank_score_floor_ratio
    return [chunk for chunk in chunks if chunk.rerank_score >= floor]


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
    pre_floor = list(chunks)
    chunks = _apply_rerank_score_floor(chunks, config)
    final = _apply_comparison_coverage_selection(
        chunks=chunks,
        queries=queries or [question],
        analysis=analysis,
        config=config,
        top_k=top_k,
    )
    if config.same_doc_body_coverage_enabled:
        final = _apply_same_doc_body_coverage(
            selected=final,
            pre_floor=pre_floor,
            top_k=len(final),
            analysis=analysis,
            config=config,
        )
    return final


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
