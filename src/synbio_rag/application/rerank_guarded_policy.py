from __future__ import annotations

import re

from ..domain.config import RetrievalConfig
from ..domain.schemas import RetrievedChunk
from .rerank_common import _guarded_sort_key, _normalize_score, _rerank_text


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
