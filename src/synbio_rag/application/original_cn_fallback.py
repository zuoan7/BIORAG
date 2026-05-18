from __future__ import annotations


def sanitize_original_cn_fallback_debug(debug: dict) -> dict:
    return {
        key: value
        for key, value in debug.items()
        if key != "merged_candidates"
    }


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def run_original_cn_fallback(
    *,
    question: str,
    retrieval_question: str,
    rewrite_trace,
    retrieved: list,
    analysis,
    filters,
    config,
    pipeline,
) -> dict:
    """Run a small retrieval pass with the original CN query as fallback."""
    debug = {
        "triggered": False,
        "reason": "",
        "fallback_added_count": 0,
        "fallback_added_chunk_ids": [],
        "fallback_added_doc_ids": [],
        "merged_candidates": list(retrieved),
    }

    if not config.original_cn_fallback_enabled:
        debug["reason"] = "disabled"
        return debug

    if config.original_cn_fallback_require_rewrite_enabled:
        rewrite_mode = getattr(
            rewrite_trace,
            "query_rewrite_mode",
            getattr(rewrite_trace, "mode", None),
        )
        is_enabled = str(rewrite_mode).lower() in ("enabled", "shadow")
        if not is_enabled:
            debug["reason"] = "rewrite_not_enabled"
            return debug

    if config.original_cn_fallback_require_cjk and not contains_cjk(question):
        debug["reason"] = "no_cjk_in_original_query"
        return debug

    if config.original_cn_fallback_min_query_diff:
        if question.strip() == retrieval_question.strip():
            debug["reason"] = "query_unchanged_by_rewrite"
            return debug

    try:
        cn_retrieved, _ = pipeline._search_with_filter_fallback(
            question=question,
            analysis=analysis,
            filters=filters,
        )
    except Exception:
        debug["reason"] = "fallback_search_error"
        return debug

    if not cn_retrieved:
        debug["reason"] = "fallback_no_results"
        return debug

    existing_ids = {chunk.chunk_id for chunk in retrieved}
    added = []
    for chunk in cn_retrieved:
        if chunk.chunk_id in existing_ids:
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                chunk.metadata["additional_query_branch"] = "original_cn_fallback"
            continue
        if len(added) >= config.original_cn_fallback_max_total:
            break
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            chunk.metadata["query_branch"] = "original_cn_fallback"
            chunk.metadata["fallback_reason"] = "rewrite_enabled_cjk_query"
        added.append(chunk)

    merged = list(retrieved) + added
    debug["triggered"] = True
    debug["reason"] = "fallback_applied"
    debug["fallback_added_count"] = len(added)
    debug["fallback_added_chunk_ids"] = [c.chunk_id for c in added]
    debug["fallback_added_doc_ids"] = list(dict.fromkeys(c.doc_id for c in added))
    debug["merged_candidates"] = merged

    return debug
