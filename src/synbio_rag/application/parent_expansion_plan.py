from __future__ import annotations

import re

from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


def select_mode(
    question: str,
    seed_chunks: list[RetrievedChunk],
    analysis: QueryAnalysis,
    caption_plan: dict | None = None,
) -> str:
    if analysis.intent == QueryIntent.SUMMARY:
        return "summary"
    if analysis.intent == QueryIntent.COMPARISON:
        return "comparison"
    if (caption_plan or {}).get("caption_mode_enabled"):
        return "caption"
    if seed_chunks and preferred_evidence_type(question, seed_chunks[0]) in {"method", "result", "numeric"}:
        return "method_result"
    return "factoid"


def effective_limits(config: RetrievalConfig, mode: str) -> tuple[int, int, str]:
    configured_total = max(0, int(config.parent_expansion_max_total))
    configured_per_seed = max(0, int(config.parent_expansion_per_seed_limit))
    if mode == "summary":
        return min(configured_total, 12), min(configured_per_seed, 2), "summary_conservative"
    if mode == "comparison":
        return min(configured_total, 8), 1, "comparison_conservative"
    if mode == "caption":
        return min(configured_total, 10), min(configured_per_seed, 1), "caption_same_doc_conservative"
    if mode == "method_result":
        return min(configured_total, 10), min(configured_per_seed, 1), "method_result_conservative"
    return min(configured_total, 10), min(configured_per_seed, 1), "factoid_conservative"


def build_caption_plan(question: str, seed_chunks: list[RetrievedChunk]) -> dict:
    query_type = explicit_caption_query_type(question)
    figure_query = query_type in {"figure", "mixed"}
    table_query = query_type in {"table", "mixed"}
    has_caption_seed = any(is_caption_seed(seed) for seed in seed_chunks)
    weak_seed_fallback = query_type == "none" and has_caption_seed and weak_caption_reference(question)

    if weak_seed_fallback:
        inferred_type = "none"
        for seed in seed_chunks:
            inferred_type = seed_caption_type(seed)
            if inferred_type != "none":
                break
        if inferred_type != "none":
            query_type = inferred_type

    caption_seed_docs = matching_caption_seed_docs(seed_chunks, query_type)
    trigger_source = "disabled"
    enabled = False
    if query_type != "none":
        enabled = True
        trigger_source = "query"
    elif weak_seed_fallback and query_type != "none":
        enabled = True
        trigger_source = "seed_metadata"

    target_doc_ids: list[str] = []
    target_reason = ""
    if enabled and seed_chunks:
        top_two_match_docs: list[str] = []
        for seed in seed_chunks[:2]:
            if seed_matches_caption_query_type(seed, query_type) and seed.doc_id not in top_two_match_docs:
                top_two_match_docs.append(seed.doc_id)
        if len(top_two_match_docs) == 2:
            target_doc_ids = top_two_match_docs[:2]
            target_reason = "top_two_matching_caption_seed_docs"
        elif caption_seed_docs:
            target_doc_ids = [caption_seed_docs[0]]
            target_reason = "matching_caption_seed_doc"
        else:
            target_doc_ids = [seed_chunks[0].doc_id]
            target_reason = "top_rank_seed_doc_fallback"

    return {
        "caption_mode_enabled": enabled,
        "figure_query": figure_query,
        "table_query": table_query,
        "caption_query_type": query_type,
        "caption_mode_trigger_source": trigger_source,
        "false_table_trigger_guarded": false_table_trigger_guarded(question),
        "caption_type_filter": query_type if query_type != "none" else "seed_type_fallback" if weak_seed_fallback else "none",
        "caption_seed_docs": caption_seed_docs,
        "caption_target_doc_ids": target_doc_ids,
        "target_doc_selection_reason": target_reason,
    }


def build_non_caption_window_plan(
    question: str,
    seed_chunks: list[RetrievedChunk],
    mode: str,
    caption_plan: dict,
) -> dict:
    if caption_plan.get("caption_mode_enabled"):
        return {"enabled": False, "target_doc_id": "", "reason": ""}
    if mode not in {"factoid", "method_result"}:
        return {"enabled": False, "target_doc_id": "", "reason": ""}
    doc_ids = [chunk.doc_id for chunk in seed_chunks if chunk.doc_id]
    unique_doc_ids = list(dict.fromkeys(doc_ids))
    if len(unique_doc_ids) <= 1:
        return {"enabled": False, "target_doc_id": "", "reason": ""}
    top_seed = seed_chunks[0]
    preferred = preferred_evidence_type(question, top_seed)
    if not (is_table_hint_or_parameter_query(question) or preferred in {"method", "result", "numeric"}):
        return {"enabled": False, "target_doc_id": "", "reason": ""}
    return {
        "enabled": True,
        "target_doc_id": top_seed.doc_id,
        "reason": "multi_doc_table_hint_or_method_result_primary_doc_only",
    }


def comparison_caption_allowed(question: str) -> bool:
    return explicit_caption_query_type(question) != "none"


def explicit_caption_query_type(question: str) -> str:
    q = question.lower()
    figure_query = bool(
        re.search(r"\bfigure\b", q)
        or re.search(r"\bfig\.\s*\d*", q)
        or re.search(r"\bfig\s+\d+", q)
        or "shown in figure" in q
        or "panel" in q
        or "microscopy" in q
        or "fluorescent" in q
        or "图中" in question
        or "图 " in question
        or re.search(r"图\s*\d+", question)
    )
    table_query = bool(
        re.search(r"\btable\b", q)
        or "tabular" in q
        or "表格" in question
        or re.search(r"表\s*\d+", question)
        or any(
            token in q
            for token in [
                "primer table",
                "sequence table",
                "strain table",
                "parameter table",
                "restriction enzyme table",
            ]
        )
    )
    if figure_query and table_query:
        return "mixed"
    if figure_query:
        return "figure"
    if table_query:
        return "table"
    return "none"


def weak_caption_reference(question: str) -> bool:
    q = question.lower()
    return any(token in q for token in ["shown", "described", "listed", "caption"]) or any(
        token in question for token in ["图中", "表格中", "表中", "图里"]
    )


def false_table_trigger_guarded(question: str) -> bool:
    q = question.lower()
    has_guard_term = any(
        token in q
        for token in [
            "expression",
            "expression cassette",
            "expression vector",
            "phenotypic",
            "phenotype",
        ]
    ) or any(token in question for token in ["表达", "表达盒", "表达载体", "表征", "表型", "表面"])
    return has_guard_term and explicit_caption_query_type(question) == "none"


def matching_caption_seed_docs(seed_chunks: list[RetrievedChunk], query_type: str) -> list[str]:
    docs: list[str] = []
    for seed in seed_chunks:
        if seed_matches_caption_query_type(seed, query_type) and seed.doc_id not in docs:
            docs.append(seed.doc_id)
    return docs


def seed_matches_caption_query_type(seed: RetrievedChunk, query_type: str) -> bool:
    if query_type == "none":
        return is_caption_seed(seed)
    seed_type = seed_caption_type(seed)
    if query_type == "mixed":
        return seed_type in {"figure", "table", "mixed"}
    if query_type == "figure":
        return seed_type in {"figure", "mixed"}
    if query_type == "table":
        return seed_type in {"table", "mixed"}
    return False


def seed_caption_type(seed: RetrievedChunk) -> str:
    meta = seed.metadata or {}
    has_table = bool(meta.get("contains_table_caption") or meta.get("contains_table_text"))
    has_figure = bool(meta.get("contains_figure_caption"))
    if has_table and has_figure:
        return "mixed"
    if has_figure:
        return "figure"
    if has_table:
        return "table"
    return "none"


def is_caption_seed(seed: RetrievedChunk) -> bool:
    meta = seed.metadata or {}
    return bool(meta.get("contains_table_caption") or meta.get("contains_figure_caption") or meta.get("contains_table_text"))


def is_table_hint_or_parameter_query(question: str) -> bool:
    q = question.lower()
    return any(
        token in q
        for token in [
            "primer",
            "sequence",
            "strain",
            "parameter",
            "restriction enzyme",
            "purification",
            "specific activity",
            "activity",
            "screening step",
        ]
    ) or any(token in question for token in ["参数", "引物", "菌株", "酶切", "纯化"])


def preferred_evidence_type(question: str, seed: RetrievedChunk) -> str:
    q = question.lower()
    if any(token in q for token in ["method", "protocol", "strain", "enzyme", "pathway", "方法"]):
        return "method"
    if any(token in q for token in ["result", "yield", "titer", "production", "结果", "产量"]):
        return "result"
    if any(token in q for token in ["fold", "%", "g/l", "mm", "mmol", "g l", "numeric"]):
        return "numeric"
    if any(token in q for token in ["table", "表"]):
        return "table"
    if any(token in q for token in ["figure", "fig.", "图"]):
        return "figure"
    meta_types = seed.metadata.get("evidence_types") if isinstance(seed.metadata, dict) else []
    if isinstance(meta_types, list):
        lowered = [str(v).lower() for v in meta_types]
        for candidate in ("method", "result", "table", "figure", "numeric"):
            if candidate in lowered:
                return candidate
    return ""
