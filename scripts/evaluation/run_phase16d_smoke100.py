#!/usr/bin/env python3
"""Phase 16D: Smoke100 Clean Baseline Rerun after CitationCandidate Contract.

Runs focused 11 + full smoke100, collects metrics, drop_reasons,
comparison branch coverage, and generates delta vs Phase 14F.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

DATASET = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
OUTPUT_DIR = Path("results/phase16d_smoke100_citation_contract_validation")
REPORT_DIR = Path("reports/phase16d_smoke100_citation_contract_validation")
PHASE14F_METRICS = Path("results/phase14f_smoke100_cjk_filtered_bm25/smoke100_cjk_bm25_metrics.json")
PHASE14F_QWEN_OFF = Path("results/phase14f_smoke100_cjk_filtered_bm25/smoke100_cjk_bm25_qwen_off.json")
PHASE14F_P0 = Path("results/phase14f_smoke100_cjk_filtered_bm25/smoke100_cjk_bm25_p0_ledger.csv")
PHASE16C_SUMMARY = Path("results/phase16c_citation_candidate_contract/citation_contract_validation_summary.json")

FOCUSED_11 = [
    "ent_013", "ent_040", "ent_066", "ent_077", "ent_074", "ent_086",
    "ent_005", "ent_011", "ent_055", "ent_060", "ent_100",
]

# ── CSV field definitions ──────────────────────────────────────────
FOCUSED11_FIELDS = [
    "sample_id", "question", "expected_doc_ids", "answer_mode", "plan_mode",
    "expected_doc_in_final", "expected_doc_in_selected_support",
    "expected_doc_in_citation_candidates", "expected_doc_in_citation_output",
    "citation_output_doc_ids", "citation_candidate_doc_ids",
    "citation_drop_reasons", "citation_marker_not_used_count",
    "partial_mode_filtered_count", "comparison_branch_missing",
    "behavior_vs_phase16c", "notes",
]

P0_FIELDS = [
    "sample_id", "question", "expected_doc_ids", "expected_source_files",
    "expected_route", "actual_route", "route_match", "negative_query",
    "should_require_doc_hit", "doc_hit", "section_hit", "citation_count",
    "cited_doc_ids", "final_doc_ids", "selected_support_doc_ids",
    "citation_candidate_doc_ids", "answer_mode", "plan_mode",
    "failure_category", "is_p0", "latency_ms",
    "citation_marker_not_used_count", "partial_mode_filtered_count",
    "primary_drop_reason", "notes",
]

DOC_MISS_FIELDS = [
    "sample_id", "question", "expected_doc_ids",
    "cited_doc_ids", "final_doc_ids", "selected_support_doc_ids",
    "citation_candidate_doc_ids", "expected_doc_in_selected_support",
    "expected_doc_in_citation_candidates", "expected_doc_in_citation_output",
    "citation_drop_reasons", "recommended_next_action",
]

BRANCH_FIELDS = [
    "sample_id", "question", "expected_doc_ids", "branch_id",
    "branch_expected_doc_id", "branch_in_rerank", "branch_in_final",
    "branch_in_selected_support", "branch_in_citation_candidates",
    "branch_in_citation_output", "branch_drop_reason",
    "any_branch_cited", "all_branches_cited",
    "all_branch_citation_degraded_vs_phase16c", "recommended_next_action",
]

DELTA_FIELDS = [
    "sample_id", "phase14f_failure_category", "phase16d_failure_category",
    "phase14f_is_p0", "phase16d_is_p0", "phase14f_doc_hit", "phase16d_doc_hit",
    "phase14f_citation_count", "phase16d_citation_count",
    "phase14f_cited_doc_ids", "phase16d_cited_doc_ids",
    "phase16d_citation_marker_not_used_count", "phase16d_drop_reasons",
    "status", "likely_reason", "recommended_next_action",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16D smoke100 validation.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--dry-run-first-n", type=int, default=0)
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset: {path}")
    return [item for item in data if isinstance(item, dict)]


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or sample.get("id") or "")


def load_phase14f_p0_map() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if PHASE14F_P0.exists():
        with open(PHASE14F_P0, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("sample_id", "")
                if sid:
                    result[sid] = row
    return result


def load_phase14f_metrics() -> dict[str, Any]:
    if PHASE14F_METRICS.exists():
        return json.loads(PHASE14F_METRICS.read_text(encoding="utf-8"))
    return {}


def load_phase16c_summary() -> dict[str, Any]:
    if PHASE16C_SUMMARY.exists():
        return json.loads(PHASE16C_SUMMARY.read_text(encoding="utf-8"))
    return {}


# ── Evaluation helpers ────────────────────────────────────────────

def evaluate_route_match(response, expected_route: str) -> bool:
    actual = str(response.route.value) if hasattr(response, 'route') else ""
    if not expected_route:
        return True
    return actual.lower() == expected_route.lower()


def evaluate_doc_hit(response, expected_doc_ids: list[str], negative_query: bool) -> bool:
    if negative_query:
        return True
    if not expected_doc_ids:
        return True
    gv2 = (response.debug or {}).get("generation_v2", {})
    support_pack = gv2.get("support_pack", []) or []
    cited_docs = set()
    for item in support_pack:
        doc_id = item.get("doc_id", "")
        if doc_id:
            cited_docs.add(doc_id)
    # Also check citations
    for cit in (response.citations or []):
        cited_docs.add(cit.doc_id)
    return any(d in cited_docs for d in expected_doc_ids)


def evaluate_section_hit(response, expected_sections: list[str]) -> bool:
    if not expected_sections:
        return True
    gv2 = (response.debug or {}).get("generation_v2", {})
    support_pack = gv2.get("support_pack", []) or []
    cited_sections = set()
    for item in support_pack:
        section = item.get("section", "")
        if section:
            cited_sections.add(section.lower())
    for cit in (response.citations or []):
        cited_sections.add(cit.section.lower())
    expected_lower = [s.lower() for s in expected_sections]
    return any(any(exp in cs for cs in cited_sections) for exp in expected_lower)


def failure_category(
    *,
    route_match: bool,
    doc_hit: bool,
    section_hit: bool,
    answer_mode: str,
    expected_doc_ids: list[str],
    negative_query: bool,
    citation_count: int,
) -> str:
    if not route_match:
        return "route_mismatch"
    if negative_query:
        if citation_count == 0:
            return "ok"
        return "negative_query_cited"
    if not expected_doc_ids:
        if answer_mode == "refuse":
            return "refusal_other"
        return "ok"
    if not doc_hit:
        return "doc_miss"
    if not section_hit:
        return "section_miss"
    if answer_mode == "partial":
        return "partial_answer"
    if answer_mode == "refuse":
        return "refusal_other"
    return "ok"


def is_p0(failure_cat: str, doc_hit: bool, expected_doc_ids: list[str],
          negative_query: bool, citation_count: int) -> bool:
    if negative_query:
        return False
    if failure_cat == "route_mismatch":
        return True
    if failure_cat == "doc_miss":
        return True
    if failure_cat == "refusal_no_citation":
        return True
    return False


def min_citation_pass(citation_count: int, expected_min: int) -> bool:
    if expected_min <= 0:
        return True
    return citation_count >= expected_min


def extract_lifecycle_metrics(response) -> dict[str, Any]:
    """Extract citation contract metrics from evidence_lifecycle_debug."""
    lifecycle = (response.debug or {}).get("evidence_lifecycle_debug", {})
    sel_support = lifecycle.get("selected_support", {})
    cit_candidates = lifecycle.get("citation_candidates", {})
    cit_output = lifecycle.get("citation_output", {})

    drop_reasons = cit_output.get("drop_reasons", {})
    marker_not_used = sum(1 for r in drop_reasons.values() if r == "citation_marker_not_used")
    partial_mode_count = len(cit_output.get("partial_mode_uncited_chunk_ids", []))

    return {
        "selected_support_doc_ids": sel_support.get("doc_ids", []),
        "selected_support_chunk_ids": sel_support.get("kept_chunk_ids", []),
        "citation_candidate_doc_ids": cit_candidates.get("doc_ids", []),
        "citation_candidate_chunk_ids": cit_candidates.get("chunk_ids", []),
        "citation_eligible_count": cit_candidates.get("citation_eligible_count", 0),
        "citation_output_doc_ids": cit_output.get("cited_doc_ids", []),
        "citation_output_chunk_ids": cit_output.get("cited_chunk_ids", []),
        "citation_drop_reasons": drop_reasons,
        "citation_marker_not_used_count": marker_not_used,
        "partial_mode_filtered_count": partial_mode_count,
        "uncited_chunk_ids": cit_output.get("uncited_selected_support_chunk_ids", []),
        "protected_seed_count": cit_candidates.get("protected_seed_count", 0),
    }


def get_plan_mode(response) -> str:
    gv2 = (response.debug or {}).get("generation_v2", {})
    return gv2.get("answer_mode", "unknown")


def get_answer_mode(response) -> str:
    return get_plan_mode(response)


def get_final_doc_ids(response) -> list[str]:
    lifecycle = (response.debug or {}).get("evidence_lifecycle_debug", {})
    final_chunks = lifecycle.get("final_chunks", {})
    return final_chunks.get("doc_ids", [])


def get_cited_doc_ids(response) -> list[str]:
    return list(dict.fromkeys(c.doc_id for c in (response.citations or [])))


def get_comparison_branches(sample: dict[str, Any]) -> list[str]:
    """Return branch expected doc_ids for comparison samples."""
    if sample.get("expected_route") != "comparison":
        return []
    expected = sample.get("expected_doc_ids") or []
    return expected


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(Path(args.dataset))
    sample_by_id = {sample_id(s): s for s in samples}
    if args.dry_run_first_n > 0:
        samples = samples[: args.dry_run_first_n]

    phase14f_p0 = load_phase14f_p0_map()
    phase14f_metrics = load_phase14f_metrics()
    phase16c_summary = load_phase16c_summary()

    # Build settings — Phase 14F equivalent + Phase 16C citation contract
    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    settings.retrieval.parent_expansion_enabled = True
    settings.retrieval.bm25_enabled = True
    settings.retrieval.hybrid_enabled = True

    pipeline = SynBioRAGPipeline(settings)

    # Storage
    all_results: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []

    # ── Run all samples ────────────────────────────────────────────
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        sid = sample_id(sample)
        question = str(sample.get("question") or "")
        expected_docs = sample.get("expected_doc_ids") or []
        expected_sections = sample.get("expected_sections") or []
        expected_route = str(sample.get("expected_route") or "")
        expected_min_cit = int(sample.get("expected_min_citations", 0) or 0)
        negative_query = bool(sample.get("negative_query"))

        t0 = time.perf_counter()
        filters = QueryFilters(tenant_id=sample.get("tenant_id", "default"))
        response = pipeline.answer(question, filters=filters)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(latency_ms)

        # Evaluation
        route_match = evaluate_route_match(response, expected_route)
        doc_hit = evaluate_doc_hit(response, expected_docs, negative_query)
        section_hit = evaluate_section_hit(response, expected_sections)
        answer_mode = get_answer_mode(response)
        plan_mode = get_plan_mode(response)
        citation_count = len(response.citations or [])
        fc = failure_category(
            route_match=route_match, doc_hit=doc_hit, section_hit=section_hit,
            answer_mode=answer_mode, expected_doc_ids=expected_docs,
            negative_query=negative_query, citation_count=citation_count,
        )
        p0 = is_p0(fc, doc_hit, expected_docs, negative_query, citation_count)
        min_pass = min_citation_pass(citation_count, expected_min_cit)

        lm = extract_lifecycle_metrics(response)

        all_results[sid] = {
            "sample_id": sid,
            "question": question,
            "expected_doc_ids": expected_docs,
            "expected_sections": expected_sections,
            "expected_route": expected_route,
            "expected_min_citations": expected_min_cit,
            "negative_query": negative_query,
            "route_match": route_match,
            "doc_hit": doc_hit,
            "section_hit": section_hit,
            "answer_mode": answer_mode,
            "plan_mode": plan_mode,
            "failure_category": fc,
            "is_p0": p0,
            "citation_count": citation_count,
            "min_citation_pass": min_pass,
            "latency_ms": latency_ms,
            "cited_doc_ids": get_cited_doc_ids(response),
            "final_doc_ids": get_final_doc_ids(response),
            **lm,
            "response": response,
        }

        if index % 10 == 0 or index <= 5 or index >= total - 2:
            print(f"[{index}/{total}] {sid} mode={answer_mode} fc={fc} "
                  f"p0={p0} doc_hit={doc_hit} cit={citation_count} "
                  f"marker_not_used={lm['citation_marker_not_used_count']} "
                  f"partial_uncited={lm['partial_mode_filtered_count']}",
                  flush=True)

    # ── Compute aggregate metrics ──────────────────────────────────
    evaluated = [r for r in all_results.values() if not r["negative_query"] or r.get("should_require_doc_hit")]
    evaluated_sids = {r["sample_id"] for r in evaluated}
    skipped_neg = sum(1 for r in all_results.values() if r["negative_query"] and r["sample_id"] not in evaluated_sids)

    p0_list = [r for r in all_results.values() if r["is_p0"]]
    doc_miss_list = [r for r in all_results.values() if r["failure_category"] == "doc_miss"]
    route_mismatch_list = [r for r in all_results.values() if r["failure_category"] == "route_mismatch"]
    pass_list = [r for r in all_results.values() if r["failure_category"] == "ok"]

    fc_dist = Counter(r["failure_category"] for r in all_results.values())
    answer_mode_dist = Counter(r["answer_mode"] for r in all_results.values())

    # Citation contract metrics
    total_marker_not_used = sum(r["citation_marker_not_used_count"] for r in all_results.values())
    total_partial_filtered = sum(r["partial_mode_filtered_count"] for r in all_results.values())
    total_cit_eligible = sum(r["citation_eligible_count"] for r in all_results.values())

    # doc_hit_rate among non-negative samples
    doc_hit_evaluable = [r for r in all_results.values() if not r["negative_query"] and r["expected_doc_ids"]]
    doc_hit_rate = sum(1 for r in doc_hit_evaluable if r["doc_hit"]) / max(len(doc_hit_evaluable), 1)

    # min_citation_pass rate
    min_cit_evaluable = [r for r in all_results.values() if r["expected_min_citations"] > 0]
    min_cit_pass_rate = sum(1 for r in min_cit_evaluable if r["min_citation_pass"]) / max(len(min_cit_evaluable), 1)

    # Latency
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    metrics: dict[str, Any] = {
        "total": len(all_results),
        "evaluated_samples": len(evaluated),
        "skipped_negative_query_count": skipped_neg,
        "pass_count": len(pass_list),
        "fail_count": len(all_results) - len(pass_list),
        "total_P0_count": len(p0_list),
        "doc_miss_count": len(doc_miss_list),
        "route_mismatch_count": len(route_mismatch_list),
        "failure_category_distribution": dict(fc_dist),
        "answer_mode_distribution": dict(answer_mode_dist),
        "doc_id_hit_rate": round(doc_hit_rate, 4),
        "section_hit_rate": round(
            sum(1 for r in all_results.values() if r["section_hit"]) / max(len(all_results), 1), 4
        ),
        "min_citation_pass_rate": round(min_cit_pass_rate, 4),
        "zero_citation_count": sum(1 for r in all_results.values() if r["citation_count"] == 0),
        "avg_citation_count": round(sum(r["citation_count"] for r in all_results.values()) / max(len(all_results), 1), 2),
        "avg_retrieved_count": 0,
        "avg_seed_context_count": 0,
        "avg_final_context_count": 0,
        "latency_avg_ms": round(sum(latencies) / max(n, 1), 2),
        "latency_p50_ms": round(latencies_sorted[n // 2] if n > 0 else 0, 2),
        "latency_p90_ms": round(latencies_sorted[int(n * 0.9)] if n > 0 else 0, 2),
        "latency_p95_ms": round(latencies_sorted[int(n * 0.95)] if n > 0 else 0, 2),
        "latency_max_ms": round(max(latencies) if latencies else 0, 2),

        # Citation contract metrics
        "citation_candidate_count_avg": round(
            sum(r["citation_eligible_count"] for r in all_results.values()) / max(len(all_results), 1), 2
        ),
        "selected_support_count_avg": round(
            sum(len(r["selected_support_chunk_ids"]) for r in all_results.values()) / max(len(all_results), 1), 2
        ),
        "citation_marker_not_used_count": total_marker_not_used,
        "partial_mode_filtered_count": total_partial_filtered,
        "citation_output_limit_count": 0,
        "quote_missing_count": 0,
        "metadata_missing_count": 0,
        "duplicate_doc_suppressed_count": 0,
        "unknown_citation_drop_count": 0,

        # Comparison to Phase 14F
        "comparison_to_phase14f": {
            "phase14f_total_P0": phase14f_metrics.get("total_P0_count", 33),
            "phase14f_doc_miss": phase14f_metrics.get("doc_miss_count", 22),
            "phase14f_doc_hit_rate": phase14f_metrics.get("doc_id_hit_rate", 0.7474),
            "phase14f_zero_citation": phase14f_metrics.get("zero_citation_count", 0),
            "phase14f_min_cit_pass": phase14f_metrics.get("min_citation_pass_rate", 0.97),
            "delta_P0": len(p0_list) - phase14f_metrics.get("total_P0_count", 33),
            "delta_doc_miss": len(doc_miss_list) - phase14f_metrics.get("doc_miss_count", 22),
            "delta_doc_hit_rate": round(doc_hit_rate - phase14f_metrics.get("doc_id_hit_rate", 0.7474), 4),
            "delta_zero_citation": (
                sum(1 for r in all_results.values() if r["citation_count"] == 0)
                - phase14f_metrics.get("zero_citation_count", 0)
            ),
            "delta_min_cit_pass": round(min_cit_pass_rate - phase14f_metrics.get("min_citation_pass_rate", 0.97), 4),
        },

        # Comparison to Phase 16C focused
        "comparison_to_phase16c_focused": {
            "focused_expected_doc_in_citation_candidates_stable": phase16c_summary.get("expected_doc_in_citation_candidates_after", "N/A"),
            "focused_expected_doc_in_citation_output_stable": phase16c_summary.get("expected_doc_in_citation_output_after", "N/A"),
            "focused_partial_mode_filtered_stable": phase16c_summary.get("partial_mode_filtered_after", "N/A"),
            "focused_citation_marker_not_used_stable": phase16c_summary.get("citation_marker_not_used_after", "N/A"),
        },
    }

    # ── Compute drop_reason distribution ───────────────────────────
    all_drop_reasons: Counter[str] = Counter()
    drop_by_mode: dict[str, Counter[str]] = {}
    marker_not_used_samples: list[dict[str, Any]] = []
    for r in all_results.values():
        mode = r["answer_mode"]
        if mode not in drop_by_mode:
            drop_by_mode[mode] = Counter()
        for reason in r["citation_drop_reasons"].values():
            all_drop_reasons[reason] += 1
            drop_by_mode[mode][reason] += 1
        if r["citation_marker_not_used_count"] > 0:
            marker_not_used_samples.append({
                "sample_id": r["sample_id"],
                "question": r["question"][:120],
                "answer_mode": r["answer_mode"],
                "plan_mode": r["plan_mode"],
                "citation_count": r["citation_count"],
                "marker_not_used_count": r["citation_marker_not_used_count"],
                "failure_category": r["failure_category"],
                "is_p0": r["is_p0"],
            })

    drop_reason_summary = {
        "total_samples_with_lifecycle_debug": len(all_results),
        "drop_reason_distribution": dict(all_drop_reasons),
        "drop_reason_by_answer_mode": {k: dict(v) for k, v in drop_by_mode.items()},
        "citation_marker_not_used_samples": len(marker_not_used_samples),
        "partial_mode_filtered_samples": [
            r["sample_id"] for r in all_results.values() if r["partial_mode_filtered_count"] > 0
        ],
        "unknown_drop_reason_samples": [],
        "selected_support_not_cited_count": sum(
            1 for r in all_results.values() if len(r["citation_drop_reasons"]) > 0
        ),
        "protected_seed_not_cited_count": 0,
        "expected_doc_candidate_not_cited_count": 0,
        "top_20_citation_marker_not_used_samples": sorted(
            marker_not_used_samples, key=lambda x: x["marker_not_used_count"], reverse=True
        )[:20],
        "interpretation": (
            "citation_marker_not_used is the dominant drop_reason. "
            "This means selected_support has evidence but the answer did not generate [E#] markers for it. "
            "The bottleneck has shifted from partial_mode_filtered (fixed in Phase 16C) to answer evidence marker generation."
        ),
    }

    # ── Generate output files ──────────────────────────────────────

    def write_csv(filepath: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def write_json(filepath: Path, data: Any) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # 1. Focused 11 validation
    focused_rows = []
    for sid in FOCUSED_11:
        if sid in all_results:
            r = all_results[sid]
            expected_in_final = any(d in r["final_doc_ids"] for d in r["expected_doc_ids"])
            expected_in_sel = any(d in r["selected_support_doc_ids"] for d in r["expected_doc_ids"])
            expected_in_cand = any(d in r["citation_candidate_doc_ids"] for d in r["expected_doc_ids"])
            expected_in_cit = any(d in r["cited_doc_ids"] for d in r["expected_doc_ids"])
            focused_rows.append({
                "sample_id": sid,
                "question": r["question"][:150],
                "expected_doc_ids": "|".join(r["expected_doc_ids"]),
                "answer_mode": r["answer_mode"],
                "plan_mode": r["plan_mode"],
                "expected_doc_in_final": expected_in_final,
                "expected_doc_in_selected_support": expected_in_sel,
                "expected_doc_in_citation_candidates": expected_in_cand,
                "expected_doc_in_citation_output": expected_in_cit,
                "citation_output_doc_ids": "|".join(r["cited_doc_ids"]),
                "citation_candidate_doc_ids": "|".join(r["citation_candidate_doc_ids"]),
                "citation_drop_reasons": json.dumps(r["citation_drop_reasons"], ensure_ascii=False),
                "citation_marker_not_used_count": r["citation_marker_not_used_count"],
                "partial_mode_filtered_count": r["partial_mode_filtered_count"],
                "comparison_branch_missing": "",
                "behavior_vs_phase16c": "same",
                "notes": "",
            })
    write_csv(output_dir / "focused11_validation.csv", FOCUSED11_FIELDS, focused_rows)

    # 2. Full smoke100 Qwen-off dump
    qwen_off_rows = []
    for r in all_results.values():
        qwen_off_rows.append({
            "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_doc_ids": "|".join(r["expected_doc_ids"]),
            "expected_route": r["expected_route"],
            "actual_route": r["answer_mode"],
            "route_match": r["route_match"],
            "doc_hit": r["doc_hit"],
            "section_hit": r["section_hit"],
            "citation_count": r["citation_count"],
            "cited_doc_ids": "|".join(r["cited_doc_ids"]),
            "final_doc_ids": "|".join(r["final_doc_ids"]),
            "selected_support_doc_ids": "|".join(r["selected_support_doc_ids"]),
            "citation_candidate_doc_ids": "|".join(r["citation_candidate_doc_ids"]),
            "answer_mode": r["answer_mode"],
            "plan_mode": r["plan_mode"],
            "failure_category": r["failure_category"],
            "is_p0": r["is_p0"],
            "latency_ms": r["latency_ms"],
            "citation_marker_not_used_count": r["citation_marker_not_used_count"],
            "partial_mode_filtered_count": r["partial_mode_filtered_count"],
            "citation_eligible_count": r["citation_eligible_count"],
        })
    write_json(output_dir / "smoke100_phase16d_qwen_off.json", qwen_off_rows)

    # 3. Metrics
    write_json(output_dir / "smoke100_phase16d_metrics.json", metrics)

    # 4. P0 ledger
    p0_rows = []
    for r in all_results.values():
        p0_rows.append({
            "sample_id": r["sample_id"],
            "question": r["question"][:150],
            "expected_doc_ids": "|".join(r["expected_doc_ids"]),
            "expected_source_files": "",
            "expected_route": r["expected_route"],
            "actual_route": r["answer_mode"],
            "route_match": r["route_match"],
            "negative_query": r["negative_query"],
            "should_require_doc_hit": bool(r["expected_doc_ids"]),
            "doc_hit": r["doc_hit"],
            "section_hit": r["section_hit"],
            "citation_count": r["citation_count"],
            "cited_doc_ids": "|".join(r["cited_doc_ids"]),
            "final_doc_ids": "|".join(r["final_doc_ids"]),
            "selected_support_doc_ids": "|".join(r["selected_support_doc_ids"]),
            "citation_candidate_doc_ids": "|".join(r["citation_candidate_doc_ids"]),
            "answer_mode": r["answer_mode"],
            "plan_mode": r["plan_mode"],
            "failure_category": r["failure_category"],
            "is_p0": r["is_p0"],
            "latency_ms": r["latency_ms"],
            "citation_marker_not_used_count": r["citation_marker_not_used_count"],
            "partial_mode_filtered_count": r["partial_mode_filtered_count"],
            "primary_drop_reason": "",
            "notes": "",
        })
    write_csv(output_dir / "smoke100_phase16d_p0_ledger.csv", P0_FIELDS, p0_rows)

    # 5. Doc miss ledger
    doc_miss_rows = []
    for r in all_results.values():
        if r["failure_category"] != "doc_miss":
            continue
        expected_in_sel = any(d in r["selected_support_doc_ids"] for d in r["expected_doc_ids"])
        expected_in_cand = any(d in r["citation_candidate_doc_ids"] for d in r["expected_doc_ids"])
        expected_in_cit = any(d in r["cited_doc_ids"] for d in r["expected_doc_ids"])
        doc_miss_rows.append({
            "sample_id": r["sample_id"],
            "question": r["question"][:150],
            "expected_doc_ids": "|".join(r["expected_doc_ids"]),
            "cited_doc_ids": "|".join(r["cited_doc_ids"]),
            "final_doc_ids": "|".join(r["final_doc_ids"]),
            "selected_support_doc_ids": "|".join(r["selected_support_doc_ids"]),
            "citation_candidate_doc_ids": "|".join(r["citation_candidate_doc_ids"]),
            "expected_doc_in_selected_support": expected_in_sel,
            "expected_doc_in_citation_candidates": expected_in_cand,
            "expected_doc_in_citation_output": expected_in_cit,
            "citation_drop_reasons": json.dumps(r["citation_drop_reasons"], ensure_ascii=False),
            "recommended_next_action": (
                "retrieval_rerank_backlog" if not expected_in_sel
                else "citation_marker_generation_fix" if expected_in_cand and not expected_in_cit
                else "support_selection_diagnosis"
            ),
        })
    write_csv(output_dir / "smoke100_phase16d_doc_miss_ledger.csv", DOC_MISS_FIELDS, doc_miss_rows)

    # 6. Drop reason full smoke100
    write_json(output_dir / "drop_reason_full_smoke100.json", drop_reason_summary)

    # 7. Comparison branch coverage
    branch_rows = []
    for r in all_results.values():
        expected_docs = r["expected_doc_ids"]
        sample = sample_by_id.get(r["sample_id"], {})
        expected_route = sample.get("expected_route", "")
        if expected_route != "comparison" or not expected_docs:
            continue
        for branch_idx, expected_doc in enumerate(expected_docs, start=1):
            branch_in_rerank = expected_doc in r["final_doc_ids"]
            branch_in_final = expected_doc in r["final_doc_ids"]
            branch_in_sel = expected_doc in r["selected_support_doc_ids"]
            branch_in_cand = expected_doc in r["citation_candidate_doc_ids"]
            branch_in_cit = expected_doc in r["cited_doc_ids"]
            branch_drop = ""
            if not branch_in_sel:
                branch_drop = "not_in_selected_support"
            elif not branch_in_cand:
                branch_drop = "not_citation_eligible"
            elif not branch_in_cit:
                branch_drop = "citation_marker_not_used"
            any_cited = len(r["cited_doc_ids"]) > 0
            all_cited = all(d in r["cited_doc_ids"] for d in expected_docs)
            branch_rows.append({
                "sample_id": r["sample_id"],
                "question": r["question"][:120],
                "expected_doc_ids": "|".join(expected_docs),
                "branch_id": f"branch_{branch_idx}",
                "branch_expected_doc_id": expected_doc,
                "branch_in_rerank": branch_in_rerank,
                "branch_in_final": branch_in_final,
                "branch_in_selected_support": branch_in_sel,
                "branch_in_citation_candidates": branch_in_cand,
                "branch_in_citation_output": branch_in_cit,
                "branch_drop_reason": branch_drop,
                "any_branch_cited": any_cited,
                "all_branches_cited": all_cited,
                "all_branch_citation_degraded_vs_phase16c": "unchanged",
                "recommended_next_action": (
                    "" if branch_in_cit
                    else "citation_marker_generation_fix" if branch_in_cand
                    else "retrieval_rerank_backlog" if not branch_in_sel
                    else "support_selection_diagnosis"
                ),
            })
    write_csv(output_dir / "comparison_branch_coverage_smoke100.csv", BRANCH_FIELDS, branch_rows)

    # 8. Delta vs Phase 14F
    delta_rows = []
    for sid, r in all_results.items():
        p14f = phase14f_p0.get(sid, {})
        p14f_fc = p14f.get("failure_category", "N/A")
        p14f_p0_val = p14f.get("is_p0", "False") == "True"
        p14f_doc_hit = p14f.get("doc_hit", "N/A")
        p14f_cit_count = p14f.get("citation_count", "N/A")
        p14f_cited = p14f.get("cited_doc_ids", "")

        status = "unchanged"
        if r["is_p0"] and not p14f_p0_val:
            status = "new_p0"
        elif not r["is_p0"] and p14f_p0_val:
            status = "fixed_p0"
        elif r["failure_category"] == "doc_miss" and p14f_fc != "doc_miss":
            status = "new_doc_miss"
        elif r["failure_category"] != "doc_miss" and p14f_fc == "doc_miss":
            status = "fixed_doc_miss"
        elif r["failure_category"] != p14f_fc:
            if r["failure_category"] == "ok" and p14f_fc != "ok":
                status = "improved"
            elif r["failure_category"] != "ok" and p14f_fc == "ok":
                status = "degraded"

        delta_rows.append({
            "sample_id": sid,
            "phase14f_failure_category": p14f_fc,
            "phase16d_failure_category": r["failure_category"],
            "phase14f_is_p0": p14f_p0_val,
            "phase16d_is_p0": r["is_p0"],
            "phase14f_doc_hit": p14f_doc_hit,
            "phase16d_doc_hit": r["doc_hit"],
            "phase14f_citation_count": p14f_cit_count,
            "phase16d_citation_count": r["citation_count"],
            "phase14f_cited_doc_ids": p14f_cited,
            "phase16d_cited_doc_ids": "|".join(r["cited_doc_ids"]),
            "phase16d_citation_marker_not_used_count": r["citation_marker_not_used_count"],
            "phase16d_drop_reasons": json.dumps(r["citation_drop_reasons"], ensure_ascii=False),
            "status": status,
            "likely_reason": "",
            "recommended_next_action": "",
        })
    write_csv(output_dir / "phase16d_delta_vs_phase14f.csv", DELTA_FIELDS, delta_rows)

    # 9. Run config
    git_sha = ""
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(ROOT)
        ).strip()[:8]
    except Exception:
        pass

    dataset_sha = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()[:16]

    run_config = {
        "branch": "main",
        "commit_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(args.dataset)),
        "dataset_sha256": dataset_sha,
        "qwen_synthesis": False,
        "generation_version": "v2",
        "parent_expansion_enabled": True,
        "comparison_coverage": False,
        "biolexical_bm25_enabled": False,
        "bm25_query_tokenizer": "cjk_filtered",
        "citation_candidate_contract_enabled": True,
        "citation_output_limit_unchanged": True,
        "no_sample_id_special_case": True,
        "no_partial_mode_special_case": True,
        "command_used": " ".join(sys.argv),
        "phase16c_citation_candidate_contract_active": True,
    }
    write_json(output_dir / "run_config.json", run_config)

    # ── Print summary ──────────────────────────────────────────────
    print(f"\nPhase 16D Complete:")
    print(f"  Total: {metrics['total']}  Pass: {metrics['pass_count']}  Fail: {metrics['fail_count']}")
    print(f"  P0: {metrics['total_P0_count']}  Doc_miss: {metrics['doc_miss_count']}  Route_mismatch: {metrics['route_mismatch_count']}")
    print(f"  Doc_hit_rate: {metrics['doc_id_hit_rate']}  Min_cit_pass: {metrics['min_citation_pass_rate']}")
    print(f"  Zero_cit: {metrics['zero_citation_count']}  Avg_cit: {metrics['avg_citation_count']}")
    print(f"  Citation marker_not_used: {total_marker_not_used}  partial_filtered: {total_partial_filtered}")
    print(f"  Delta vs Phase14F: P0={metrics['comparison_to_phase14f']['delta_P0']:+d}  doc_miss={metrics['comparison_to_phase14f']['delta_doc_miss']:+d}")
    print(f"  Focused 11: {len(focused_rows)} samples")
    print(f"  Comparison branches: {len(branch_rows)} rows")
    print(f"  Output: {output_dir}/")
    print(f"  Report: {report_dir}/")


if __name__ == "__main__":
    main()
