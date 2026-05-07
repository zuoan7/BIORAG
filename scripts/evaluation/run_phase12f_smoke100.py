#!/usr/bin/env python3
"""
Phase 12F Controlled Smoke100: parent_expansion A/B evaluation.

A: parent_expansion_enabled=False, Qwen synthesis=False
B: parent_expansion_enabled=True,  Qwen synthesis=False
All other settings identical.

Usage:
  python scripts/evaluation/run_phase12f_smoke100.py
  python scripts/evaluation/run_phase12f_smoke100.py --dry-run
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_ragas import (
    build_failure_diagnostics,
    build_raw_records,
    compute_retrieval_ledger_summary,
    evaluate_retrieval,
    get_effective_final_answer_mode,
    get_refusal_reason,
    load_records,
)
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

DATASET_PATH = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
OUTPUT_DIR = ROOT / "reports/phase12f_smoke100"


def normalize_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_safe(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_contexts(citations: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for citation in citations:
        quote = (citation.get("quote") or "").strip()
        title = (citation.get("title") or "").strip()
        section = (citation.get("section") or "").strip()
        source_file = (citation.get("source_file") or "").strip()
        context_text = quote or title
        if section:
            context_text = f"{section}: {context_text}" if context_text else section
        if source_file:
            context_text = f"{source_file} | {context_text}" if context_text else source_file
        if context_text:
            contexts.append(context_text)
    return contexts


def mode_bucket(mode: str) -> str:
    lowered = (mode or "").strip().lower()
    if lowered == "refuse":
        return "refuse"
    if "partial" in lowered:
        return "partial"
    return "full"


def failure_category(item: dict[str, Any]) -> str:
    retrieval_eval = item.get("retrieval_eval") or {}
    if retrieval_eval.get("strict_route_match") is False:
        return "route_mismatch"
    if retrieval_eval.get("refusal_no_citation"):
        return "refusal_no_citation"
    if retrieval_eval.get("zero_citation_substantive_answer"):
        return "zero_citation_substantive_answer"
    doc_metrics = retrieval_eval.get("doc_id_metrics") or {}
    if doc_metrics.get("expected") and not doc_metrics.get("hit"):
        return "doc_miss"
    section_metrics = retrieval_eval.get("section_metrics") or {}
    if section_metrics.get("expected") and not section_metrics.get("hit"):
        return "section_miss"
    evidence_metrics = retrieval_eval.get("evidence_metrics") or {}
    if isinstance(evidence_metrics.get("evidence_coverage"), (int, float)) and float(evidence_metrics.get("evidence_coverage")) == 0.0:
        return "evidence_not_supported_by_citations"
    mode = str((item.get("raw_record") or {}).get("answer_mode") or "")
    if mode_bucket(mode) == "partial":
        return "partial_answer"
    if mode_bucket(mode) == "refuse":
        return "refusal_other"
    return "ok"


def is_p0(item: dict[str, Any]) -> bool:
    fc = failure_category(item)
    if fc == "ok":
        return False
    # P0: route_mismatch, refusal_no_citation, zero_citation_substantive_answer, doc_miss
    return fc in ("route_mismatch", "refusal_no_citation", "zero_citation_substantive_answer", "doc_miss")


def is_pass(item: dict[str, Any]) -> bool:
    return failure_category(item) == "ok"


def route_label(item: dict[str, Any]) -> str:
    return str((item.get("api_response") or {}).get("route") or "unknown")


def category_label(item: dict[str, Any]) -> str:
    tags = item.get("dataset_meta", {}).get("tags") or []
    if "comparison" in tags:
        return "comparison"
    if "table" in tags or "figure" in tags or "caption" in tags:
        return "table/figure/caption"
    if "summary" in tags:
        return "summary"
    if "factoid" in tags:
        return "factoid"
    if "method" in tags or "result" in tags or "numeric" in tags:
        return "method/result/numeric"
    return "unknown/other"


def build_settings_a() -> Settings:
    s = Settings.from_env()
    s.audit.enabled = False
    s.generation.version = "v2"
    s.generation.v2_use_qwen_synthesis = False
    s.generation.v2_enable_comparison_coverage = False
    s.generation.v2_enable_neighbor_audit = False
    s.generation.v2_enable_neighbor_promotion = False
    s.generation.v2_include_neighbor_context_in_qwen = False
    s.generation.v2_use_external_tools = False
    s.generation.v2_use_history = False
    s.retrieval.neighbor_expansion_enabled = True
    s.retrieval.parent_expansion_enabled = False
    return s


def build_settings_b() -> Settings:
    s = Settings.from_env()
    s.audit.enabled = False
    s.generation.version = "v2"
    s.generation.v2_use_qwen_synthesis = False
    s.generation.v2_enable_comparison_coverage = False
    s.generation.v2_enable_neighbor_audit = False
    s.generation.v2_enable_neighbor_promotion = False
    s.generation.v2_include_neighbor_context_in_qwen = False
    s.generation.v2_use_external_tools = False
    s.generation.v2_use_history = False
    s.retrieval.neighbor_expansion_enabled = True
    s.retrieval.parent_expansion_enabled = True
    return s


def run_group(label: str, build_fn, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = build_fn()
    pipeline = SynBioRAGPipeline(settings)
    enriched: list[dict[str, Any]] = []
    latencies: list[float] = []

    for idx, item in enumerate(records, start=1):
        filters = QueryFilters(
            tenant_id=item.get("tenant_id", "default"),
            doc_ids=item.get("doc_ids") or [],
            sections=item.get("sections") or [],
            source_files=item.get("source_files") or [],
            min_score=item.get("min_score"),
        )
        t0 = time.perf_counter()
        response = pipeline.answer(
            question=item["question"],
            session_id=f"{label}_{idx:03d}",
            history=None,
            filters=filters,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        api_response = {
            "session_id": response.session_id,
            "answer": response.answer,
            "confidence": response.confidence,
            "route": response.route.value,
            "citations": [citation.__dict__ for citation in response.citations],
            "used_external_tool": response.used_external_tool,
            "tool_name": response.tool_name,
            "tool_result": response.tool_result,
            "external_references": [ref.__dict__ for ref in response.external_references],
            "debug": response.debug,
        }

        enriched.append({
            "id": item.get("id") or f"sample_{idx:03d}",
            "question": item["question"],
            "reference": (item.get("reference") or "").strip(),
            "response": api_response.get("answer", ""),
            "retrieved_contexts": build_contexts(api_response.get("citations") or []),
            "dataset_meta": {
                "tags": item.get("tags") or [],
                "scenario": str(item.get("scenario") or "").strip(),
                "ability": str(item.get("ability") or "").strip(),
                "difficulty": str(item.get("difficulty") or "").strip(),
                "risk_level": str(item.get("risk_level") or "").strip(),
                "expected_behavior": str(item.get("expected_behavior") or "").strip(),
                "expected_doc_ids": item.get("expected_doc_ids") or item.get("doc_ids") or [],
                "accepted_doc_ids": item.get("accepted_doc_ids") or [],
                "doc_ids": item.get("doc_ids") or [],
                "expected_source_files": item.get("expected_source_files") or [],
                "accepted_source_files": item.get("accepted_source_files") or [],
                "source_files": item.get("source_files") or [],
                "expected_sections": item.get("expected_sections") or [],
                "expected_route": str(item.get("expected_route") or "").strip(),
                "accepted_routes": item.get("accepted_routes") or [],
                "expected_min_citations": int(item.get("expected_min_citations", 0) or 0),
                "comparison_branches": item.get("comparison_branches") or [],
                "expected_min_doc_coverage": normalize_int(item.get("expected_min_doc_coverage")),
                "allow_partial_if_doc_coverage": normalize_int(item.get("allow_partial_if_doc_coverage")),
                "allow_partial_answer": bool(item.get("allow_partial_answer")),
                "expected_answer_mode": str(item.get("expected_answer_mode") or "").strip(),
                "notes": (item.get("notes") or "").strip(),
            },
            "api_response": api_response,
        })

    # Run retrieval evaluation
    _, retrieval_summary = evaluate_retrieval(enriched, embeddings=None)
    raw_records = build_raw_records(enriched)

    for item, raw in zip(enriched, raw_records, strict=True):
        item["raw_record"] = raw
        item["retrieval_eval"] = item.get("retrieval_eval", {})
        item["retrieval_eval"]["failure_category"] = failure_category(item)
        item["retrieval_eval"]["refusal_reason"] = get_refusal_reason(item)
        item["retrieval_eval"]["final_answer_mode"] = get_effective_final_answer_mode(item)

    ledger_summary = compute_retrieval_ledger_summary(raw_records)

    latency_sorted = sorted(latencies)
    p95_idx = max(0, int(len(latency_sorted) * 0.95) - 1)

    metrics = {
        "label": label,
        "total": len(enriched),
        "pass_count": sum(1 for item in enriched if is_pass(item)),
        "fail_count": sum(1 for item in enriched if not is_pass(item)),
        "p0_count": sum(1 for item in enriched if is_p0(item)),
        "route_match_rate": retrieval_summary.get("route_match_rate"),
        "doc_id_hit_rate": retrieval_summary.get("doc_id_hit_rate"),
        "section_hit_rate": retrieval_summary.get("section_hit_rate"),
        "min_citation_pass_rate": _min_citation_pass_rate(enriched),
        "zero_citation_count": sum(1 for item in enriched if normalize_int((item.get("raw_record") or {}).get("citation_count")) == 0),
        "avg_citation_count": sum(normalize_int((item.get("raw_record") or {}).get("citation_count")) for item in enriched) / max(1, len(enriched)),
        "avg_retrieved_count": sum(_debug_num(item, "retrieved_count") for item in enriched) / max(1, len(enriched)),
        "avg_reranked_count": sum(_debug_num(item, "reranked_count") for item in enriched) / max(1, len(enriched)),
        "avg_seed_context_count": sum(_debug_num(item, "seed_context_count") for item in enriched) / max(1, len(enriched)),
        "avg_final_context_count": sum(_debug_num(item, "final_context_count") for item in enriched) / max(1, len(enriched)),
        "latency_avg_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "latency_p95_ms": round(latency_sorted[p95_idx] if latency_sorted else 0, 2),
        "mode_counts": {
            "refuse": sum(1 for item in enriched if mode_bucket((item.get("raw_record") or {}).get("answer_mode") or "") == "refuse"),
            "partial": sum(1 for item in enriched if mode_bucket((item.get("raw_record") or {}).get("answer_mode") or "") == "partial"),
            "full": sum(1 for item in enriched if mode_bucket((item.get("raw_record") or {}).get("answer_mode") or "") == "full"),
        },
        "failure_category_distribution": dict(sorted(Counter(
            failure_category(item) for item in enriched if failure_category(item) != "ok"
        ).items())),
        "retrieval_ledger": ledger_summary,
    }
    return enriched, metrics


def _debug_num(item: dict[str, Any], key: str) -> int:
    return normalize_int(((item.get("api_response") or {}).get("debug") or {}).get(key))


def _min_citation_pass_rate(enriched: list[dict[str, Any]]) -> float:
    count = 0
    total = 0
    for item in enriched:
        meta = item.get("dataset_meta") or {}
        expected_min = normalize_int(meta.get("expected_min_citations"))
        actual = normalize_int((item.get("raw_record") or {}).get("citation_count"))
        total += 1
        if actual >= expected_min:
            count += 1
    return round(count / max(1, total), 4)


def extract_parent_expansion_metrics(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract parent_expansion specific metrics from B group enriched records."""
    added_counts = []
    final_ctx_counts = []
    parent_types_all = Counter()
    caption_mode_count = 0
    comparison_mode_count = 0
    local_context_gating_count = 0
    false_table_trigger_guarded_count = 0
    primary_doc_local_context_gating_count = 0
    max_total_truncated = 0
    per_seed_truncated = 0
    enabled_count = 0
    figure_query_count = 0
    table_query_count = 0

    for item in enriched:
        pe_debug = _pe_debug(item)
        if not pe_debug:
            continue
        if pe_debug.get("enabled"):
            enabled_count += 1
        added_ids = pe_debug.get("added_chunk_ids") or []
        added_counts.append(len(added_ids))
        final_ctx_counts.append(normalize_int(
            ((item.get("api_response") or {}).get("debug") or {}).get("final_context_count")
        ))
        for pt in pe_debug.get("added_parent_types") or []:
            parent_types_all[str(pt)] += 1
        for pt in pe_debug.get("selected_parent_types") or []:
            parent_types_all[f"selected:{pt}"] += 1
        if pe_debug.get("caption_mode"):
            caption_mode_count += 1
        if pe_debug.get("comparison_mode"):
            comparison_mode_count += 1
        if pe_debug.get("local_context_gating_reason"):
            local_context_gating_count += 1
        if pe_debug.get("false_table_trigger_guarded"):
            false_table_trigger_guarded_count += 1
        if pe_debug.get("primary_doc_local_context_gating"):
            primary_doc_local_context_gating_count += 1
        if pe_debug.get("figure_query"):
            figure_query_count += 1
        if pe_debug.get("table_query"):
            table_query_count += 1
        if pe_debug.get("reason") == "max_total_reached":
            max_total_truncated += 1
        if pe_debug.get("reason") == "per_seed_limit_reached":
            per_seed_truncated += 1

    sorted_added = sorted(added_counts)
    sorted_final = sorted(final_ctx_counts)

    return {
        "parent_expansion_enabled_count": enabled_count,
        "avg_added_count": round(sum(added_counts) / max(1, len(added_counts)), 3),
        "added_count_p50": _percentile(sorted_added, 50),
        "added_count_p90": _percentile(sorted_added, 90),
        "added_count_max": max(added_counts) if added_counts else 0,
        "final_context_count_p50": _percentile(sorted_final, 50),
        "final_context_count_p90": _percentile(sorted_final, 90),
        "final_context_count_max": max(final_ctx_counts) if final_ctx_counts else 0,
        "parent_types_used": dict(parent_types_all.most_common(20)),
        "caption_mode_count": caption_mode_count,
        "comparison_mode_count": comparison_mode_count,
        "local_context_gating_count": local_context_gating_count,
        "false_table_trigger_guarded_count": false_table_trigger_guarded_count,
        "primary_doc_local_context_gating_count": primary_doc_local_context_gating_count,
        "figure_query_count": figure_query_count,
        "table_query_count": table_query_count,
        "max_total_truncated_count": max_total_truncated,
        "per_seed_limit_truncated_count": per_seed_truncated,
    }


def _pe_debug(item: dict[str, Any]) -> dict[str, Any] | None:
    return ((item.get("api_response") or {}).get("debug") or {}).get("parent_expansion")


def _percentile(sorted_list: list[int], p: int) -> float:
    if not sorted_list:
        return 0
    idx = max(0, min(len(sorted_list) - 1, int(len(sorted_list) * p / 100.0)))
    return float(sorted_list[idx])


# ── Route/Category Split ──────────────────────────────────────────────

def build_split_metrics(enriched_a: list[dict[str, Any]], enriched_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-route/category split metrics with auto-judgment."""
    categories = [
        "factoid", "summary", "comparison", "table/figure/caption",
        "method/result/numeric", "unknown/other",
    ]
    splits = []
    for cat in categories:
        a_items = [it for it in enriched_a if category_label(it) == cat]
        b_items = [it for it in enriched_b if category_label(it) == cat]

        a_by_id = {it["id"]: it for it in a_items}
        b_by_id = {it["id"]: it for it in b_items}

        common_ids = set(a_by_id) & set(b_by_id)
        n = len(common_ids)

        a_p0 = sum(1 for i in common_ids if is_p0(a_by_id[i]))
        b_p0 = sum(1 for i in common_ids if is_p0(b_by_id[i]))
        a_zero = sum(1 for i in common_ids if normalize_int((a_by_id[i].get("raw_record") or {}).get("citation_count")) == 0)
        b_zero = sum(1 for i in common_ids if normalize_int((b_by_id[i].get("raw_record") or {}).get("citation_count")) == 0)
        a_avg_final = sum(_debug_num(a_by_id[i], "final_context_count") for i in common_ids) / max(1, n)
        b_avg_final = sum(_debug_num(b_by_id[i], "final_context_count") for i in common_ids) / max(1, n)
        b_added_counts = [len((_pe_debug(b_by_id[i]) or {}).get("added_chunk_ids") or []) for i in common_ids]
        b_avg_added = sum(b_added_counts) / max(1, n)

        b_types = Counter()
        for i in common_ids:
            for pt in (_pe_debug(b_by_id[i]) or {}).get("added_parent_types") or []:
                b_types[str(pt)] += 1

        # Auto-judgment
        improved = 0
        same = 0
        regressed = 0
        needs_review = 0
        for i in common_ids:
            a_p0_i = is_p0(a_by_id[i])
            b_p0_i = is_p0(b_by_id[i])
            a_pass_i = is_pass(a_by_id[i])
            b_pass_i = is_pass(b_by_id[i])
            a_cit_i = normalize_int((a_by_id[i].get("raw_record") or {}).get("citation_count"))
            b_cit_i = normalize_int((b_by_id[i].get("raw_record") or {}).get("citation_count"))
            a_doc_hit = (a_by_id[i].get("raw_record") or {}).get("doc_hit")
            b_doc_hit = (b_by_id[i].get("raw_record") or {}).get("doc_hit")
            b_added = len((_pe_debug(b_by_id[i]) or {}).get("added_chunk_ids") or [])

            if (not a_pass_i and b_pass_i) or (a_p0_i and not b_p0_i) or (a_cit_i == 0 and b_cit_i > 0) or (not a_doc_hit and b_doc_hit):
                improved += 1
            elif (a_pass_i and not b_pass_i) or (not a_p0_i and b_p0_i) or (a_cit_i > 0 and b_cit_i == 0) or (a_doc_hit and not b_doc_hit):
                regressed += 1
            elif a_p0_i == b_p0_i and abs(b_cit_i - a_cit_i) <= 1:
                same += 1
            else:
                needs_review += 1

        splits.append({
            "category": cat,
            "sample_count": n,
            "A_p0_count": a_p0,
            "B_p0_count": b_p0,
            "A_zero_citation_count": a_zero,
            "B_zero_citation_count": b_zero,
            "A_avg_final_context_count": round(a_avg_final, 2),
            "B_avg_final_context_count": round(b_avg_final, 2),
            "B_avg_added_count": round(b_avg_added, 2),
            "B_parent_types_used": dict(b_types.most_common(10)),
            "auto_likely_improved": improved,
            "auto_likely_same": same,
            "auto_likely_regressed": regressed,
            "auto_needs_manual_review": needs_review,
        })
    return splits


# ── Regression Ledger ──────────────────────────────────────────────────

def build_regression_ledger(enriched_a: list[dict[str, Any]], enriched_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_by_id = {it["id"]: it for it in enriched_a}
    b_by_id = {it["id"]: it for it in enriched_b}
    ledger = []

    for sid in sorted(set(a_by_id) & set(b_by_id)):
        a = a_by_id[sid]
        b = b_by_id[sid]
        a_pass = is_pass(a)
        b_pass = is_pass(b)
        a_p0 = is_p0(a)
        b_p0 = is_p0(b)
        a_cit = normalize_int((a.get("raw_record") or {}).get("citation_count"))
        b_cit = normalize_int((b.get("raw_record") or {}).get("citation_count"))
        a_doc_hit = (a.get("raw_record") or {}).get("doc_hit")
        b_doc_hit = (b.get("raw_record") or {}).get("doc_hit")
        a_sec_hit = (a.get("raw_record") or {}).get("section_hit")
        b_sec_hit = (b.get("raw_record") or {}).get("section_hit")
        b_added = len((_pe_debug(b) or {}).get("added_chunk_ids") or [])
        b_final_ctx = _debug_num(b, "final_context_count")
        b_types = (_pe_debug(b) or {}).get("added_parent_types") or []
        b_false_trigger = (_pe_debug(b) or {}).get("false_table_trigger_guarded", False)
        b_comparison_mode = (_pe_debug(b) or {}).get("comparison_mode", False)

        reasons = []
        priority = "low"

        if a_pass and not b_pass:
            reasons.append("A_pass_B_fail")
            priority = "high"
        if not a_p0 and b_p0:
            reasons.append("A_not_P0_B_P0")
            priority = "high"
        if a_cit > 0 and b_cit == 0:
            reasons.append("citation_dropped_to_zero")
            priority = "high"
        if a_doc_hit and not b_doc_hit:
            reasons.append("doc_hit_lost")
            priority = "high"
        if a_sec_hit and not b_sec_hit:
            reasons.append("section_hit_lost")
            priority = "medium"
        if b_added >= 4 and (not b_pass or b_p0):
            reasons.append("high_added_count_with_degradation")
            priority = max(priority, "medium")
        if b_false_trigger:
            reasons.append("false_table_trigger")
            priority = "high"
        if b_comparison_mode and b_added >= 5:
            reasons.append("comparison_possible_over_expansion")
            priority = max(priority, "medium")
        if b_cit > 0 and a_cit > 0 and b_cit < a_cit * 0.5:
            reasons.append("citation_count_significant_drop")
            priority = max(priority, "medium")
        if b_final_ctx > 12 and b_added >= 3 and (not b_pass or b_p0):
            reasons.append("high_context_noise")
            priority = max(priority, "medium")

        if not reasons:
            continue

        ledger.append({
            "sample_id": sid,
            "question": a["question"][:200],
            "route": route_label(a),
            "category": category_label(a),
            "A_status": "pass" if a_pass else "fail",
            "B_status": "pass" if b_pass else "fail",
            "A_P0": a_p0,
            "B_P0": b_p0,
            "A_doc_hit": a_doc_hit,
            "B_doc_hit": b_doc_hit,
            "A_section_hit": a_sec_hit,
            "B_section_hit": b_sec_hit,
            "A_citation_count": a_cit,
            "B_citation_count": b_cit,
            "B_parent_added_count": b_added,
            "B_parent_types_used": ", ".join(b_types[:5]) if b_types else "",
            "B_final_context_count": b_final_ctx,
            "suspected_reason": "; ".join(reasons),
            "manual_review_priority": priority,
        })

    return ledger


# ── Improvement Ledger ────────────────────────────────────────────────

def build_improvement_ledger(enriched_a: list[dict[str, Any]], enriched_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_by_id = {it["id"]: it for it in enriched_a}
    b_by_id = {it["id"]: it for it in enriched_b}
    ledger = []

    for sid in sorted(set(a_by_id) & set(b_by_id)):
        a = a_by_id[sid]
        b = b_by_id[sid]
        a_pass = is_pass(a)
        b_pass = is_pass(b)
        a_p0 = is_p0(a)
        b_p0 = is_p0(b)
        a_cit = normalize_int((a.get("raw_record") or {}).get("citation_count"))
        b_cit = normalize_int((b.get("raw_record") or {}).get("citation_count"))
        a_doc_hit = (a.get("raw_record") or {}).get("doc_hit")
        b_doc_hit = (b.get("raw_record") or {}).get("doc_hit")
        a_sec_hit = (a.get("raw_record") or {}).get("section_hit")
        b_sec_hit = (b.get("raw_record") or {}).get("section_hit")
        b_added = len((_pe_debug(b) or {}).get("added_chunk_ids") or [])
        b_types = (_pe_debug(b) or {}).get("added_parent_types") or []

        reasons = []
        priority = "low"

        if not a_pass and b_pass:
            reasons.append("A_fail_B_pass")
            priority = "high"
        if a_p0 and not b_p0:
            reasons.append("A_P0_B_not_P0")
            priority = "high"
        if a_cit == 0 and b_cit > 0:
            reasons.append("zero_citation_fixed")
            priority = "high"
        if not a_doc_hit and b_doc_hit:
            reasons.append("doc_hit_gained")
            priority = "high"
        if not a_sec_hit and b_sec_hit:
            reasons.append("section_hit_gained")
            priority = "medium"
        if b_added >= 2 and b_cit > a_cit and b_pass:
            reasons.append("expansion_added_relevant_context")
            priority = max(priority, "medium")
        if b_cit > a_cit + 1 and b_pass:
            reasons.append("citation_count_increased")
            priority = max(priority, "medium")

        if not reasons:
            continue

        ledger.append({
            "sample_id": sid,
            "question": a["question"][:200],
            "route": route_label(a),
            "category": category_label(a),
            "A_status": "pass" if a_pass else "fail",
            "B_status": "pass" if b_pass else "fail",
            "A_P0": a_p0,
            "B_P0": b_p0,
            "A_doc_hit": a_doc_hit,
            "B_doc_hit": b_doc_hit,
            "A_section_hit": a_sec_hit,
            "B_section_hit": b_sec_hit,
            "A_citation_count": a_cit,
            "B_citation_count": b_cit,
            "B_parent_added_count": b_added,
            "B_parent_types_used": ", ".join(b_types[:5]) if b_types else "",
            "improvement_reason": "; ".join(reasons),
            "manual_review_priority": priority,
        })

    return ledger


# ── Manual Review ─────────────────────────────────────────────────────

def build_manual_review(
    enriched_a: list[dict[str, Any]],
    enriched_b: list[dict[str, Any]],
    reg_ledger: list[dict[str, Any]],
    imp_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    a_by_id = {it["id"]: it for it in enriched_a}
    b_by_id = {it["id"]: it for it in enriched_b}

    review_ids: set[str] = set()

    # Regression high priority — all
    for entry in reg_ledger:
        if entry["manual_review_priority"] == "high":
            review_ids.add(entry["sample_id"])

    # Regression medium priority — max 10
    medium_reg = [e for e in reg_ledger if e["manual_review_priority"] == "medium"]
    for entry in medium_reg[:10]:
        review_ids.add(entry["sample_id"])

    # Improvement high priority — max 10
    high_imp = [e for e in imp_ledger if e["manual_review_priority"] == "high"]
    for entry in high_imp[:10]:
        review_ids.add(entry["sample_id"])

    # One representative per route
    routes_seen: dict[str, str] = {}
    for sid in sorted(review_ids):
        route = route_label(a_by_id.get(sid, b_by_id.get(sid, {})))
        if route not in routes_seen:
            routes_seen[route] = sid
    for sid in sorted(set(a_by_id) & set(b_by_id)):
        route = route_label(a_by_id[sid])
        if route not in routes_seen:
            routes_seen[route] = sid

    # Build review entries
    reviews: list[dict[str, Any]] = []
    for sid in sorted(review_ids | set(routes_seen.values())):
        a = a_by_id.get(sid)
        b = b_by_id.get(sid)
        if not a or not b:
            continue

        a_answer = str((a.get("api_response") or {}).get("answer") or "")
        b_answer = str((b.get("api_response") or {}).get("answer") or "")
        a_citations = [c.get("quote", "")[:100] for c in (a.get("api_response") or {}).get("citations") or []]
        b_citations = [c.get("quote", "")[:100] for c in (b.get("api_response") or {}).get("citations") or []]
        b_pe = _pe_debug(b) or {}

        reviews.append({
            "sample_id": sid,
            "question": a["question"][:300],
            "route": route_label(a),
            "category": category_label(a),
            "A_status": "pass" if is_pass(a) else "fail",
            "B_status": "pass" if is_pass(b) else "fail",
            "A_answer_preview": re.sub(r"\s+", " ", a_answer)[:300],
            "B_answer_preview": re.sub(r"\s+", " ", b_answer)[:300],
            "A_citations": a_citations[:5],
            "B_citations": b_citations[:5],
            "B_seed_chunk_ids": b_pe.get("input_count", 0),
            "B_added_chunk_ids": b_pe.get("added_chunk_ids", [])[:10],
            "B_parent_types_used": b_pe.get("added_parent_types", [])[:10],
            "B_effective_intent": b_pe.get("effective_intent", ""),
            "B_comparison_mode": b_pe.get("comparison_mode", False),
            "B_caption_mode": b_pe.get("caption_mode", False),
            "B_false_table_trigger_guarded": b_pe.get("false_table_trigger_guarded", False),
            "B_primary_doc_local_context_gating": b_pe.get("primary_doc_local_context_gating", False),
            "B_local_context_gating_reason": b_pe.get("local_context_gating_reason", ""),
            "B_added_count": len(b_pe.get("added_chunk_ids") or []),
            "B_final_context_count": _debug_num(b, "final_context_count"),
            "judgment": "",  # to be filled by human or auto
            "attribution": "",  # to be filled
        })

    return {"samples": reviews, "total": len(reviews)}


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("Phase 12F Controlled Smoke100 — Dry Run")
        print(f"Dataset: {DATASET_PATH.relative_to(ROOT)}")
        sa = build_settings_a()
        sb = build_settings_b()
        print(f"A: parent_expansion={sa.retrieval.parent_expansion_enabled}, qwen={sa.generation.v2_use_qwen_synthesis}")
        print(f"B: parent_expansion={sb.retrieval.parent_expansion_enabled}, qwen={sb.generation.v2_use_qwen_synthesis}")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(str(DATASET_PATH))
    print(f"Loaded {len(records)} samples from {DATASET_PATH.relative_to(ROOT)}")

    # ── Group A: parent_expansion OFF ──
    print("\n" + "=" * 60)
    print("[Phase12F] Running Group A: parent_expansion OFF ...")
    print("=" * 60)
    t0 = time.perf_counter()
    enriched_a, metrics_a = run_group("A_parent_off", build_settings_a, records)
    elapsed_a = time.perf_counter() - t0
    print(f"Group A completed in {elapsed_a:.1f}s")
    print(f"  P0={metrics_a['p0_count']} pass={metrics_a['pass_count']} fail={metrics_a['fail_count']}")
    print(f"  route={metrics_a['route_match_rate']} doc={metrics_a['doc_id_hit_rate']} section={metrics_a['section_hit_rate']}")

    # ── Group B: parent_expansion ON ──
    print("\n" + "=" * 60)
    print("[Phase12F] Running Group B: parent_expansion ON ...")
    print("=" * 60)
    t0 = time.perf_counter()
    enriched_b, metrics_b = run_group("B_parent_on", build_settings_b, records)
    elapsed_b = time.perf_counter() - t0
    print(f"Group B completed in {elapsed_b:.1f}s")
    print(f"  P0={metrics_b['p0_count']} pass={metrics_b['pass_count']} fail={metrics_b['fail_count']}")
    print(f"  route={metrics_b['route_match_rate']} doc={metrics_b['doc_id_hit_rate']} section={metrics_b['section_hit_rate']}")

    # ── Parent expansion metrics (B only) ──
    pe_metrics = extract_parent_expansion_metrics(enriched_b)
    metrics_b["parent_expansion"] = pe_metrics
    print(f"\nParent Expansion (B): enabled={pe_metrics['parent_expansion_enabled_count']}")
    print(f"  avg_added={pe_metrics['avg_added_count']} p50={pe_metrics['added_count_p50']} p90={pe_metrics['added_count_p90']} max={pe_metrics['added_count_max']}")
    print(f"  caption_mode={pe_metrics['caption_mode_count']} comparison_mode={pe_metrics['comparison_mode_count']}")
    print(f"  false_table_trigger_guarded={pe_metrics['false_table_trigger_guarded_count']}")

    # ── Save raw results ──
    _save_json(OUTPUT_DIR / "smoke100_parent_off.json", _serialize_enriched(enriched_a, metrics_a))
    _save_json(OUTPUT_DIR / "smoke100_parent_on.json", _serialize_enriched(enriched_b, metrics_b))

    # ── A/B metrics ──
    ab_metrics = {
        "A_label": "parent_expansion_off",
        "B_label": "parent_expansion_on",
        "total_samples": len(records),
        "A": {k: v for k, v in metrics_a.items() if k != "retrieval_ledger"},
        "B": {k: v for k, v in metrics_b.items() if k != "retrieval_ledger"},
        "delta": {
            "p0_count": metrics_b["p0_count"] - metrics_a["p0_count"],
            "pass_count": metrics_b["pass_count"] - metrics_a["pass_count"],
            "zero_citation_count": metrics_b["zero_citation_count"] - metrics_a["zero_citation_count"],
            "doc_id_hit_rate": round(float_safe(metrics_b["doc_id_hit_rate"]) - float_safe(metrics_a["doc_id_hit_rate"]), 4),
            "section_hit_rate": round(float_safe(metrics_b["section_hit_rate"]) - float_safe(metrics_a["section_hit_rate"]), 4),
            "route_match_rate": round(float_safe(metrics_b["route_match_rate"]) - float_safe(metrics_a["route_match_rate"]), 4),
            "min_citation_pass_rate": round(float_safe(metrics_b["min_citation_pass_rate"]) - float_safe(metrics_a["min_citation_pass_rate"]), 4),
            "avg_final_context_count": round(metrics_b["avg_final_context_count"] - metrics_a["avg_final_context_count"], 2),
            "latency_avg_ms": round(metrics_b["latency_avg_ms"] - metrics_a["latency_avg_ms"], 2),
            "latency_p95_ms": round(metrics_b["latency_p95_ms"] - metrics_a["latency_p95_ms"], 2),
        },
    }
    _save_json(OUTPUT_DIR / "smoke100_parent_ab_metrics.json", ab_metrics)

    # ── Split metrics ──
    split_metrics = build_split_metrics(enriched_a, enriched_b)

    # ── Ledgers ──
    reg_ledger = build_regression_ledger(enriched_a, enriched_b)
    imp_ledger = build_improvement_ledger(enriched_a, enriched_b)
    manual_review = build_manual_review(enriched_a, enriched_b, reg_ledger, imp_ledger)

    print(f"\nRegression ledger: {len(reg_ledger)} entries")
    for entry in reg_ledger:
        print(f"  [{entry['manual_review_priority']}] {entry['sample_id']}: {entry['suspected_reason'][:120]}")
    print(f"Improvement ledger: {len(imp_ledger)} entries")
    for entry in imp_ledger:
        print(f"  [{entry['manual_review_priority']}] {entry['sample_id']}: {entry['improvement_reason'][:120]}")

    # ── Save CSVs ──
    _save_csv(OUTPUT_DIR / "smoke100_parent_regression_ledger.csv", reg_ledger)
    _save_csv(OUTPUT_DIR / "smoke100_parent_improvement_ledger.csv", imp_ledger)
    _save_json(OUTPUT_DIR / "smoke100_parent_manual_review.json", manual_review)

    # ── Summary MD ──
    summary_md = _build_summary_md(ab_metrics, pe_metrics, split_metrics, reg_ledger, imp_ledger, manual_review)
    (OUTPUT_DIR / "smoke100_parent_ab_summary.md").write_text(summary_md, encoding="utf-8")

    # ── Manual review MD ──
    review_md = _build_review_md(manual_review)
    (OUTPUT_DIR / "smoke100_parent_manual_review.md").write_text(review_md, encoding="utf-8")

    print(f"\nReports saved to {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"  - smoke100_parent_off.json")
    print(f"  - smoke100_parent_on.json")
    print(f"  - smoke100_parent_ab_metrics.json")
    print(f"  - smoke100_parent_ab_summary.md")
    print(f"  - smoke100_parent_regression_ledger.csv")
    print(f"  - smoke100_parent_improvement_ledger.csv")
    print(f"  - smoke100_parent_manual_review.md")

    # ── Acceptance Gate Check ──
    print("\n" + "=" * 60)
    print("ACCEPTANCE GATE CHECK")
    print("=" * 60)
    checks = _acceptance_checks(ab_metrics, pe_metrics, reg_ledger, manual_review)
    for c in checks:
        icon = "✅" if c["pass"] else "❌"
        print(f"  {icon} {c['name']}: {c['value']} (threshold: {c['threshold']})")
    all_pass = all(c["pass"] for c in checks)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'} — {'建议进入 Phase 12G' if all_pass else '不满足验收标准，不要合并 main'}")

    return 0 if all_pass else 1


def _serialize_enriched(enriched: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    # Strip heavy debug to keep json manageable
    slim = []
    for item in enriched:
        slim.append({
            "id": item["id"],
            "question": item["question"][:300],
            "response": (item.get("response") or "")[:500],
            "api_response": {
                "route": (item.get("api_response") or {}).get("route"),
                "answer": ((item.get("api_response") or {}).get("answer") or "")[:500],
                "citations": [
                    {"chunk_id": c.get("chunk_id", ""), "title": c.get("title", ""), "section": c.get("section", ""),
                     "source_file": c.get("source_file", ""), "quote": (c.get("quote") or "")[:200]}
                    for c in (item.get("api_response") or {}).get("citations") or []
                ],
                "debug": {
                    "retrieved_count": _debug_num(item, "retrieved_count"),
                    "reranked_count": _debug_num(item, "reranked_count"),
                    "seed_context_count": _debug_num(item, "seed_context_count"),
                    "final_context_count": _debug_num(item, "final_context_count"),
                    "latency_ms": ((item.get("api_response") or {}).get("debug") or {}).get("latency_ms"),
                    "parent_expansion": {
                        k: v for k, v in (_pe_debug(item) or {}).items()
                        if k in ("enabled", "input_count", "output_count", "added_chunk_ids", "added_parent_types",
                                 "effective_intent", "comparison_mode", "caption_mode", "false_table_trigger_guarded",
                                 "primary_doc_local_context_gating", "local_context_gating_reason", "effective_max_total",
                                 "effective_per_seed_limit", "added_parent_ids")
                    },
                },
            },
            "raw_record": {
                k: v for k, v in (item.get("raw_record") or {}).items()
                if k in ("doc_hit", "section_hit", "citation_count", "answer_mode", "answer_preview",
                         "route_match", "route_matched", "failure_category")
            },
        })
    return {
        "metrics": {k: v for k, v in metrics.items() if k != "retrieval_ledger"},
        "samples": slim,
    }


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        values = []
        for k in keys:
            v = str(row.get(k, "")).replace('"', '""')
            values.append(f'"{v}"')
        lines.append(",".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_summary_md(
    ab: dict, pe: dict, splits: list[dict], reg: list[dict], imp: list[dict], review: dict
) -> str:
    a = ab["A"]
    b = ab["B"]
    d = ab["delta"]
    lines = [
        "# Phase 12F Controlled Smoke100 — A/B Summary",
        "",
        f"**Date**: 2026-05-07",
        f"**Branch**: fix/parsed",
        f"**Dataset**: enterprise_ragas_smoke100.json (N=100)",
        "",
        "## Overall Metrics",
        "",
        "| Metric | A (parent_off) | B (parent_on) | Delta |",
        "|--------|---------------|---------------|-------|",
        f"| total | {a['total']} | {b['total']} | — |",
        f"| pass_count | {a['pass_count']} | {b['pass_count']} | {d['pass_count']:+d} |",
        f"| fail_count | {a['fail_count']} | {b['fail_count']} | {d['pass_count']:+d} |",
        f"| P0_count | {a['p0_count']} | {b['p0_count']} | {d['p0_count']:+d} |",
        f"| route_match_rate | {a['route_match_rate']} | {b['route_match_rate']} | {d['route_match_rate']:+.4f} |",
        f"| doc_id_hit_rate | {a['doc_id_hit_rate']} | {b['doc_id_hit_rate']} | {d['doc_id_hit_rate']:+.4f} |",
        f"| section_hit_rate | {a['section_hit_rate']} | {b['section_hit_rate']} | {d['section_hit_rate']:+.4f} |",
        f"| min_citation_pass_rate | {a['min_citation_pass_rate']} | {b['min_citation_pass_rate']} | {d['min_citation_pass_rate']:+.4f} |",
        f"| zero_citation_count | {a['zero_citation_count']} | {b['zero_citation_count']} | {d['zero_citation_count']:+d} |",
        f"| avg_citation_count | {a['avg_citation_count']:.2f} | {b['avg_citation_count']:.2f} | {b['avg_citation_count'] - a['avg_citation_count']:+.2f} |",
        f"| avg_retrieved_count | {a['avg_retrieved_count']:.1f} | {b['avg_retrieved_count']:.1f} | {b['avg_retrieved_count'] - a['avg_retrieved_count']:+.1f} |",
        f"| avg_reranked_count | {a['avg_reranked_count']:.1f} | {b['avg_reranked_count']:.1f} | {b['avg_reranked_count'] - a['avg_reranked_count']:+.1f} |",
        f"| avg_seed_context_count | {a['avg_seed_context_count']:.1f} | {b['avg_seed_context_count']:.1f} | {b['avg_seed_context_count'] - a['avg_seed_context_count']:+.1f} |",
        f"| avg_final_context_count | {a['avg_final_context_count']:.1f} | {b['avg_final_context_count']:.1f} | {d['avg_final_context_count']:+.1f} |",
        f"| latency_avg_ms | {a['latency_avg_ms']:.0f} | {b['latency_avg_ms']:.0f} | {d['latency_avg_ms']:+.0f} |",
        f"| latency_p95_ms | {a['latency_p95_ms']:.0f} | {b['latency_p95_ms']:.0f} | {d['latency_p95_ms']:+.0f} |",
        "",
        "## Parent Expansion (B Group) Metrics",
        "",
        f"- enabled_count: {pe['parent_expansion_enabled_count']}",
        f"- avg_added_count: {pe['avg_added_count']}",
        f"- added_count p50: {pe['added_count_p50']}",
        f"- added_count p90: {pe['added_count_p90']}",
        f"- added_count max: {pe['added_count_max']}",
        f"- final_context_count p50: {pe['final_context_count_p50']}",
        f"- final_context_count p90: {pe['final_context_count_p90']}",
        f"- final_context_count max: {pe['final_context_count_max']}",
        f"- caption_mode: {pe['caption_mode_count']}",
        f"- comparison_mode: {pe['comparison_mode_count']}",
        f"- local_context_gating: {pe['local_context_gating_count']}",
        f"- false_table_trigger_guarded: {pe['false_table_trigger_guarded_count']}",
        f"- primary_doc_local_context_gating: {pe['primary_doc_local_context_gating_count']}",
        f"- max_total_truncated: {pe['max_total_truncated_count']}",
        f"- per_seed_limit_truncated: {pe['per_seed_limit_truncated_count']}",
        f"- figure_query: {pe['figure_query_count']}",
        f"- table_query: {pe['table_query_count']}",
        "",
        "### Parent Types Used",
        "",
    ]
    for pt, count in pe.get("parent_types_used", {}).items():
        lines.append(f"- {pt}: {count}")
    lines += [
        "",
        "## Route/Category Split",
        "",
        "| Category | N | A_P0 | B_P0 | A_ZeroCit | B_ZeroCit | A_AvgFinal | B_AvgFinal | B_AvgAdded | Improved | Same | Regressed | Review |",
        "|----------|---|------|------|-----------|-----------|-----------|-----------|-----------|----------|------|-----------|--------|",
    ]
    for s in splits:
        lines.append(
            f"| {s['category']} | {s['sample_count']} | {s['A_p0_count']} | {s['B_p0_count']} | "
            f"{s['A_zero_citation_count']} | {s['B_zero_citation_count']} | "
            f"{s['A_avg_final_context_count']} | {s['B_avg_final_context_count']} | {s['B_avg_added_count']} | "
            f"{s['auto_likely_improved']} | {s['auto_likely_same']} | {s['auto_likely_regressed']} | {s['auto_needs_manual_review']} |"
        )
    lines += [
        "",
        "## Regression Ledger Summary",
        "",
        f"- Total entries: {len(reg)}",
    ]
    for prio in ("high", "medium", "low"):
        count = sum(1 for r in reg if r["manual_review_priority"] == prio)
        lines.append(f"- {prio} priority: {count}")
    if reg:
        lines += ["", "### High Priority", ""]
        for r in reg:
            if r["manual_review_priority"] == "high":
                lines.append(f"- **{r['sample_id']}** [{r['route']}/{r['category']}]: {r['suspected_reason']}")

    lines += [
        "",
        "## Improvement Ledger Summary",
        "",
        f"- Total entries: {len(imp)}",
    ]
    for prio in ("high", "medium", "low"):
        count = sum(1 for r in imp if r["manual_review_priority"] == prio)
        lines.append(f"- {prio} priority: {count}")
    if imp:
        lines += ["", "### High Priority", ""]
        for r in imp:
            if r["manual_review_priority"] == "high":
                lines.append(f"- **{r['sample_id']}** [{r['route']}/{r['category']}]: {r['improvement_reason']}")

    lines += [
        "",
        "## Manual Review",
        "",
        f"- Total samples to review: {review['total']}",
    ]
    return "\n".join(lines)


def _build_review_md(review: dict) -> str:
    lines = [
        "# Phase 12F Controlled Smoke100 — Manual Review",
        "",
        f"Total samples to review: {review['total']}",
        "",
        "Each sample requires human judgment: better / same / noisy / worse",
        "",
    ]
    for sample in review["samples"]:
        lines += [
            f"## {sample['sample_id']} — {sample['route']} / {sample['category']}",
            "",
            f"Question: {sample['question']}",
            "",
            f"- A status: {sample['A_status']} | B status: {sample['B_status']}",
            f"- B added: {sample['B_added_count']} | B final_ctx: {sample['B_final_context_count']}",
            f"- B parent_types: {sample['B_parent_types_used']}",
            f"- B effective_intent: {sample['B_effective_intent']}",
            f"- B comparison_mode: {sample['B_comparison_mode']}",
            f"- B caption_mode: {sample['B_caption_mode']}",
            f"- B false_table_trigger: {sample['B_false_table_trigger_guarded']}",
            f"- B primary_doc_gating: {sample['B_primary_doc_local_context_gating']} ({sample['B_local_context_gating_reason']})",
            "",
            "### A Answer Preview",
            f"```\n{sample['A_answer_preview'][:500]}\n```",
            "",
            "### B Answer Preview",
            f"```\n{sample['B_answer_preview'][:500]}\n```",
            "",
            "### A Citations",
        ]
        for i, c in enumerate(sample.get("A_citations", [])[:5], 1):
            lines.append(f"{i}. {c}")
        lines += ["", "### B Citations"]
        for i, c in enumerate(sample.get("B_citations", [])[:5], 1):
            lines.append(f"{i}. {c}")
        lines += [
            "",
            "### Judgment",
            "- [ ] better",
            "- [ ] same",
            "- [ ] noisy",
            "- [ ] worse",
            "",
            "### Attribution",
            "- [ ] expansion_helped",
            "- [ ] expansion_noise",
            "- [ ] seed_miss",
            "- [ ] rerank_miss",
            "- [ ] citation_missing",
            "- [ ] comparison_over_expansion",
            "- [ ] table_figure_false_trigger",
            "- [ ] summary_bad_section",
            "- [ ] generation_not_using_context",
            "- [ ] no_clear_difference",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def _acceptance_checks(ab: dict, pe: dict, reg: list[dict], review: dict) -> list[dict]:
    a = ab["A"]
    b = ab["B"]
    d = ab["delta"]
    reg_high = sum(1 for r in reg if r["manual_review_priority"] == "high")

    # Count worse / noisy from manual review (auto estimate based on regression ledger)
    worse_estimate = sum(1 for r in reg if r["manual_review_priority"] == "high")
    noisy_estimate = sum(1 for r in reg if r["manual_review_priority"] == "medium")
    better_estimate = sum(1 for imp in [])  # will be updated after manual review

    checks = [
        {"name": "B P0_count ≤ A P0_count", "value": f"{b['p0_count']} vs {a['p0_count']}", "pass": b['p0_count'] <= a['p0_count'], "threshold": f"≤ {a['p0_count']}"},
        {"name": "B zero_citation ≤ A zero_citation", "value": f"{b['zero_citation_count']} vs {a['zero_citation_count']}", "pass": b['zero_citation_count'] <= a['zero_citation_count'], "threshold": f"≤ {a['zero_citation_count']}"},
        {"name": "B doc_id_hit_rate ≥ A doc_id_hit_rate", "value": f"{b['doc_id_hit_rate']} vs {a['doc_id_hit_rate']}", "pass": float_safe(b['doc_id_hit_rate']) >= float_safe(a['doc_id_hit_rate']), "threshold": f"≥ {a['doc_id_hit_rate']}"},
        {"name": "B min_citation_pass_rate ≥ A", "value": f"{b['min_citation_pass_rate']} vs {a['min_citation_pass_rate']}", "pass": float_safe(b['min_citation_pass_rate']) >= float_safe(a['min_citation_pass_rate']), "threshold": f"≥ {a['min_citation_pass_rate']}"},
        {"name": "regression high priority ≤ 3", "value": str(reg_high), "pass": reg_high <= 3, "threshold": "≤ 3"},
        {"name": "false_table_trigger_guarded = 0", "value": str(pe.get("false_table_trigger_guarded_count", "?")), "pass": pe.get("false_table_trigger_guarded_count", -1) == 0, "threshold": "0"},
        {"name": "final_context_count p90 < 15", "value": str(pe.get("final_context_count_p90", "?")), "pass": float_safe(pe.get("final_context_count_p90")) < 15, "threshold": "< 15"},
        {"name": "latency p95 delta < 2000ms", "value": f"{d['latency_p95_ms']:+.0f}ms", "pass": d['latency_p95_ms'] < 2000, "threshold": "< +2000ms"},
    ]
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
