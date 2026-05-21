from __future__ import annotations

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, RetrievedChunk


def build_parent_expansion_debug(
    config: RetrievalConfig,
    seed_chunks: list[RetrievedChunk],
    analysis: QueryAnalysis,
) -> dict:
    return {
        "enabled": config.parent_expansion_enabled,
        "reason": "",
        "input_count": len(seed_chunks),
        "output_count": len(seed_chunks),
        "added_chunk_ids": [],
        "added_parent_ids": [],
        "added_parent_types": [],
        "per_seed_added": {},
        "per_doc_added": {},
        "strategy": analysis.intent.value,
        "effective_intent": "",
        "effective_max_total": 0,
        "effective_per_seed_limit": 0,
        "seed_preservation_enabled": config.parent_expansion_preserve_seed_chunks,
        "effective_max_added": 0,
        "effective_final_context_cap": 0,
        "limit_reason": "",
        "comparison_mode": False,
        "comparison_seed_considered": [],
        "comparison_seed_skipped_by_rank": [],
        "skipped_by_doc_cap": [],
        "selected_parent_types": [],
        "comparison_caption_allowed": False,
        "caption_mode": False,
        "caption_anchor_doc_id": "",
        "same_doc_only": False,
        "same_page_candidates_found": 0,
        "caption_context_candidates_found": 0,
        "caption_context_added": 0,
        "page_context_added": 0,
        "skipped_cross_doc": 0,
        "skipped_after_caption_limit": 0,
        "page_candidates_found": 0,
        "page_candidates_added": 0,
        "page_skipped_reason": "",
        "evidence_candidates_found": 0,
        "evidence_candidates_added": 0,
        "evidence_skipped_reason": "",
        "summary_docs_considered": [],
        "summary_sections_added": [],
        "summary_sections_skipped_existing": [],
        "summary_no_candidate_docs": [],
        "figure_query": False,
        "table_query": False,
        "caption_query_type": "none",
        "caption_mode_trigger_source": "disabled",
        "false_table_trigger_guarded": False,
        "caption_type_filter": "none",
        "caption_candidates_before_type_filter": 0,
        "caption_candidates_filtered_by_type": 0,
        "caption_candidates_added_by_type": 0,
        "caption_seed_docs": [],
        "caption_target_doc_ids": [],
        "skipped_non_target_doc": [],
        "target_doc_selection_reason": "",
        "page_candidates_before_filter": 0,
        "page_candidates_filtered_by_doc": 0,
        "page_candidates_filtered_by_type": 0,
        "page_plain_paragraph_skipped": 0,
        "page_fallback_used": False,
        "primary_doc_window_gating": False,
        "window_target_doc_id": "",
        "window_gating_reason": "",
        "window_skipped_non_target_doc": [],
        "primary_doc_local_context_gating": False,
        "local_context_target_doc_id": "",
        "local_context_gating_reason": "",
        "local_context_skipped_non_target_doc": [],
        "local_context_blocked_parent_types": [],
        "section_path_skipped_non_target_doc": [],
    }


def initialize_optional_debug_reasons(
    *,
    debug: dict,
    config: RetrievalConfig,
    mode: str,
    seed_chunks: list[RetrievedChunk],
    preferred_evidence_type: str,
) -> None:
    if not config.parent_expansion_page_enabled:
        debug["page_skipped_reason"] = "disabled"
    elif mode != "caption":
        debug["page_skipped_reason"] = "intent_not_allowed" if mode == "comparison" else "no_query_trigger"
    elif not any((chunk.metadata or {}).get("page_numbers") for chunk in seed_chunks):
        debug["page_skipped_reason"] = "no_seed_page_numbers"

    if not config.parent_expansion_evidence_enabled:
        debug["evidence_skipped_reason"] = "disabled"
        return
    if mode == "comparison" and not preferred_evidence_type:
        debug["evidence_skipped_reason"] = "intent_not_allowed"
    elif not preferred_evidence_type:
        debug["evidence_skipped_reason"] = "no_query_trigger"


def record_parent_expansion_skip(debug: dict, parent_type: str, reason: str) -> None:
    if reason == "cross_doc":
        debug["skipped_cross_doc"] += 1
        if parent_type == "page":
            debug["page_candidates_filtered_by_doc"] += 1
    if reason == "caption_type_mismatch":
        debug["caption_candidates_filtered_by_type"] += 1
    if reason == "page_type_mismatch":
        debug["page_candidates_filtered_by_type"] += 1
    if reason == "page_plain_paragraph":
        debug["page_plain_paragraph_skipped"] += 1
    if parent_type == "page" and reason and not debug["page_skipped_reason"]:
        debug["page_skipped_reason"] = reason
    if parent_type == "evidence_type_context" and reason and not debug["evidence_skipped_reason"]:
        debug["evidence_skipped_reason"] = reason
