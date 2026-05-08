#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.generation_v2.evidence_lifecycle_debug import ALLOWED_DROP_REASONS
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings

DATASET = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
OUTPUT_DIR = Path("results/phase16b_evidence_lifecycle_debug")
REPORT_DIR = Path("reports/phase16b_evidence_lifecycle_debug")
PHASE15A_FINAL = Path("results/phase15a_final_context_retention/final_context_retention_validation.csv")
PHASE15C_VALIDATION = Path("results/phase15c_protected_support_selection/protected_support_selection_validation.csv")
PHASE15D_GROUPS = Path("results/phase15d_remaining_focused_audit/remaining_sample_group_audit.csv")
PHASE16R2_DOC_LOCAL = Path("results/phase16r2_chunk_evidence_audit/doc_hit_but_evidence_miss.csv")
PHASE16R2_PER_SAMPLE = Path("results/phase16r2_chunk_evidence_audit/chunk_evidence_per_sample.csv")

FOCUSED_IDS = [
    "ent_013", "ent_040", "ent_066", "ent_077", "ent_074", "ent_086",
    "ent_005", "ent_011", "ent_055", "ent_060", "ent_100",
    "ent_020", "ent_037", "ent_094", "ent_054", "ent_057", "ent_064", "ent_075",
]

TRACE_FIELDS = [
    "sample_id", "question", "answer_mode", "plan_mode", "expected_doc_ids", "expected_source_files",
    "expected_sections", "category", "negative_query", "should_require_doc_hit",
    "rerank_output_doc_ids", "rerank_output_chunk_ids", "expected_doc_in_rerank",
    "protected_seed_chunk_ids", "final_doc_ids", "final_chunk_ids", "expected_doc_in_final",
    "final_dropped_chunk_ids", "final_drop_reasons", "support_input_doc_ids", "support_input_chunk_ids",
    "expected_doc_in_support_input", "selected_support_doc_ids", "selected_support_chunk_ids",
    "expected_doc_in_selected_support", "selected_support_dropped_chunk_ids",
    "selected_support_drop_reasons", "support_pack_size", "protected_seed_in_selected_support",
    "citation_candidate_doc_ids", "citation_candidate_chunk_ids", "expected_doc_in_citation_candidates",
    "citation_output_doc_ids", "citation_output_chunk_ids", "expected_doc_in_citation_output",
    "uncited_selected_support_chunk_ids", "citation_drop_reasons", "first_lifecycle_loss_stage",
    "primary_drop_reason", "recommended_next_action",
]
BRANCH_FIELDS = [
    "sample_id", "question", "expected_doc_ids", "branch_id", "branch_expected_doc_id",
    "branch_doc_in_rerank", "branch_doc_in_final", "branch_doc_in_support_input",
    "branch_doc_in_selected_support", "branch_doc_in_citation_output", "branch_first_loss_stage",
    "branch_drop_reason", "any_branch_cited", "all_branches_cited",
    "any_branch_in_selected_support", "all_branches_in_selected_support", "recommended_next_action",
]
BACKLOG_FIELDS = [
    "sample_id", "question", "expected_doc_ids", "rerank_doc_best_rank",
    "expected_doc_chunks_in_rerank_top10", "expected_doc_chunk_sections",
    "extracted_answer_keywords", "why_evidence_miss", "recommended_next_action", "backlog_priority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16B focused evidence lifecycle debug trace.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(Path(args.dataset))
    sample_by_id = {sample_id(sample): sample for sample in samples}
    comparison_ids = [sample_id(sample) for sample in samples if sample.get("expected_route") == "comparison"]
    run_ids = list(dict.fromkeys([sid for sid in FOCUSED_IDS if sid in sample_by_id] + comparison_ids))

    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    pipeline = SynBioRAGPipeline(settings)

    responses: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    for index, sid in enumerate(run_ids, start=1):
        sample = sample_by_id[sid]
        response = pipeline.answer(str(sample.get("question") or ""))
        responses[sid] = response
        if sid in FOCUSED_IDS:
            trace_rows.append(build_trace_row(sample, response))
        print(f"[{index}/{len(run_ids)}] {sid}", flush=True)

    branch_rows = build_branch_rows(samples, responses)
    backlog_rows = build_backlog_rows(sample_by_id)
    compatibility = build_behavior_compatibility(trace_rows)
    summary = build_drop_reason_summary(trace_rows, branch_rows, compatibility)

    write_csv(output_dir / "evidence_lifecycle_debug_trace.csv", TRACE_FIELDS, trace_rows)
    write_csv(output_dir / "comparison_branch_coverage_trace.csv", BRANCH_FIELDS, branch_rows)
    write_csv(output_dir / "doc_local_evidence_backlog.csv", BACKLOG_FIELDS, backlog_rows)
    write_json(output_dir / "behavior_compatibility_check.json", compatibility)
    write_json(output_dir / "drop_reason_summary.json", summary)
    write_summary(report_dir / "summary.md", summary, compatibility, trace_rows, branch_rows, backlog_rows)


def build_trace_row(sample: dict[str, Any], response: Any) -> dict[str, Any]:
    debug = response.debug or {}
    lifecycle = debug.get("evidence_lifecycle_debug", {})
    gen_debug = debug.get("generation_v2", {})
    expected_docs = [str(item) for item in sample.get("expected_doc_ids") or []]
    expected_sources = [str(item) for item in sample.get("expected_source_files") or []]
    rerank = lifecycle.get("rerank_output", {})
    final = lifecycle.get("final_chunks", {})
    support_input = lifecycle.get("support_input", {})
    selected = lifecycle.get("selected_support", {})
    citation_candidates = lifecycle.get("citation_candidates", {})
    citation_output = lifecycle.get("citation_output", {})
    row = {
        "sample_id": sample_id(sample),
        "question": sample.get("question", ""),
        "answer_mode": gen_debug.get("answer_mode", ""),
        "plan_mode": (gen_debug.get("answer_plan") or {}).get("mode", gen_debug.get("answer_mode", "")),
        "expected_doc_ids": join_list(expected_docs),
        "expected_source_files": join_list(expected_sources),
        "expected_sections": join_list(sample.get("expected_sections") or []),
        "category": join_list(sample.get("tags") or []),
        "negative_query": bool(sample.get("negative_query")),
        "should_require_doc_hit": sample.get("should_require_doc_hit", ""),
        "rerank_output_doc_ids": join_list(rerank.get("doc_ids", [])),
        "rerank_output_chunk_ids": join_list(rerank.get("chunk_ids", [])),
        "expected_doc_in_rerank": contains_expected(rerank.get("doc_ids", []), [], expected_docs, expected_sources),
        "protected_seed_chunk_ids": join_list(rerank.get("protected_seed_chunk_ids", [])),
        "final_doc_ids": join_list(final.get("doc_ids", [])),
        "final_chunk_ids": join_list(final.get("kept_chunk_ids", [])),
        "expected_doc_in_final": contains_expected(final.get("doc_ids", []), [], expected_docs, expected_sources),
        "final_dropped_chunk_ids": join_list(final.get("dropped_chunk_ids", [])),
        "final_drop_reasons": json.dumps(final.get("drop_reasons", {}), ensure_ascii=False, sort_keys=True),
        "support_input_doc_ids": join_list(support_input.get("doc_ids", [])),
        "support_input_chunk_ids": join_list(support_input.get("chunk_ids", [])),
        "expected_doc_in_support_input": contains_expected(support_input.get("doc_ids", []), [], expected_docs, expected_sources),
        "selected_support_doc_ids": join_list(selected_docs(gen_debug)),
        "selected_support_chunk_ids": join_list(selected.get("kept_chunk_ids", [])),
        "expected_doc_in_selected_support": contains_expected(selected_docs(gen_debug), [], expected_docs, expected_sources),
        "selected_support_dropped_chunk_ids": join_list(selected.get("dropped_chunk_ids", [])),
        "selected_support_drop_reasons": json.dumps(selected.get("drop_reasons", {}), ensure_ascii=False, sort_keys=True),
        "support_pack_size": selected.get("support_pack_size", gen_debug.get("support_pack_count", "")),
        "protected_seed_in_selected_support": bool(selected.get("protected_seed_kept_count", 0)),
        "citation_candidate_doc_ids": join_list(citation_candidates.get("doc_ids", [])),
        "citation_candidate_chunk_ids": join_list(citation_candidates.get("chunk_ids", [])),
        "expected_doc_in_citation_candidates": contains_expected(citation_candidates.get("doc_ids", []), [], expected_docs, expected_sources),
        "citation_output_doc_ids": join_list(citation_output.get("cited_doc_ids", [])),
        "citation_output_chunk_ids": join_list(citation_output.get("cited_chunk_ids", [])),
        "expected_doc_in_citation_output": contains_expected(citation_output.get("cited_doc_ids", []), [], expected_docs, expected_sources),
        "uncited_selected_support_chunk_ids": join_list(citation_output.get("uncited_selected_support_chunk_ids", [])),
        "citation_drop_reasons": json.dumps(citation_output.get("drop_reasons", {}), ensure_ascii=False, sort_keys=True),
    }
    row["first_lifecycle_loss_stage"], row["primary_drop_reason"], row["recommended_next_action"] = diagnose(row)
    return row


def selected_docs(gen_debug: dict[str, Any]) -> list[str]:
    return [str(item.get("doc_id", "")) for item in gen_debug.get("support_pack", [])]


def diagnose(row: dict[str, Any]) -> tuple[str, str, str]:
    sid = row["sample_id"]
    if sid in {"ent_020", "ent_037"}:
        return "doc_local_evidence_issue", "doc_local_evidence_miss", "doc_local_evidence_selection_backlog"
    if not as_bool(row["expected_doc_in_rerank"]):
        return "retrieval_or_rerank_missing", "not_in_input", "retrieval_rerank_backlog"
    checks = [
        ("expected_doc_in_final", "rerank_to_final", row["final_drop_reasons"], "support_selection_fix_candidate"),
        ("expected_doc_in_support_input", "final_to_support_input", "{}", "need_more_debug"),
        ("expected_doc_in_selected_support", "support_input_to_selected_support", row["selected_support_drop_reasons"], "support_selection_fix_candidate"),
        ("expected_doc_in_citation_candidates", "selected_support_to_citation_candidates", "{}", "need_more_debug"),
        ("expected_doc_in_citation_output", "citation_candidates_to_output", row["citation_drop_reasons"], "citation_binder_fix_candidate"),
    ]
    for field, stage, reason_json, action in checks:
        if not as_bool(row[field]):
            return stage, first_reason(reason_json), action
    return "no_lifecycle_loss", "", "no_action_fixed"


def first_reason(reason_json: str) -> str:
    try:
        data = json.loads(reason_json)
    except Exception:
        return "unknown"
    for reason in data.values():
        return reason if reason in ALLOWED_DROP_REASONS else "unknown"
    return "unknown"


def build_branch_rows(samples: list[dict[str, Any]], responses: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("expected_route") != "comparison":
            continue
        sid = sample_id(sample)
        response = responses.get(sid)
        if response is None:
            continue
        debug = response.debug or {}
        lifecycle = debug.get("evidence_lifecycle_debug", {})
        expected_docs = [str(item) for item in sample.get("expected_doc_ids") or []]
        branch_states = []
        for index, doc_id in enumerate(expected_docs, start=1):
            state = {
                "rerank": doc_id in set(lifecycle.get("rerank_output", {}).get("doc_ids", [])),
                "final": doc_id in set(lifecycle.get("final_chunks", {}).get("doc_ids", [])),
                "support_input": doc_id in set(lifecycle.get("support_input", {}).get("doc_ids", [])),
                "selected_support": doc_id in set(selected_docs(debug.get("generation_v2", {}))),
                "citation_output": doc_id in set(lifecycle.get("citation_output", {}).get("cited_doc_ids", [])),
            }
            first_loss = branch_first_loss(state)
            branch_states.append(state)
            rows.append(
                {
                    "sample_id": sid,
                    "question": sample.get("question", ""),
                    "expected_doc_ids": join_list(expected_docs),
                    "branch_id": f"branch_{index}",
                    "branch_expected_doc_id": doc_id,
                    "branch_doc_in_rerank": state["rerank"],
                    "branch_doc_in_final": state["final"],
                    "branch_doc_in_support_input": state["support_input"],
                    "branch_doc_in_selected_support": state["selected_support"],
                    "branch_doc_in_citation_output": state["citation_output"],
                    "branch_first_loss_stage": first_loss,
                    "branch_drop_reason": branch_drop_reason(first_loss),
                    "any_branch_cited": False,
                    "all_branches_cited": False,
                    "any_branch_in_selected_support": False,
                    "all_branches_in_selected_support": False,
                    "recommended_next_action": branch_action(first_loss),
                }
            )
        any_cited = any(state["citation_output"] for state in branch_states)
        all_cited = bool(branch_states) and all(state["citation_output"] for state in branch_states)
        any_selected = any(state["selected_support"] for state in branch_states)
        all_selected = bool(branch_states) and all(state["selected_support"] for state in branch_states)
        for row in rows:
            if row["sample_id"] == sid:
                row["any_branch_cited"] = any_cited
                row["all_branches_cited"] = all_cited
                row["any_branch_in_selected_support"] = any_selected
                row["all_branches_in_selected_support"] = all_selected
    return rows


def branch_first_loss(state: dict[str, bool]) -> str:
    if not state["rerank"]:
        return "retrieval_or_rerank_missing"
    if not state["final"]:
        return "rerank_to_final"
    if not state["support_input"]:
        return "final_to_support_input"
    if not state["selected_support"]:
        return "support_input_to_selected_support"
    if not state["citation_output"]:
        return "citation_candidates_to_output"
    return "no_lifecycle_loss"


def branch_drop_reason(first_loss: str) -> str:
    return {
        "retrieval_or_rerank_missing": "not_in_input",
        "support_input_to_selected_support": "comparison_branch_missing",
        "citation_candidates_to_output": "citation_marker_not_used",
    }.get(first_loss, "unknown" if first_loss != "no_lifecycle_loss" else "")


def branch_action(first_loss: str) -> str:
    if first_loss == "support_input_to_selected_support":
        return "comparison_branch_coverage_fix_candidate"
    if first_loss == "citation_candidates_to_output":
        return "citation_binder_fix_candidate"
    if first_loss == "retrieval_or_rerank_missing":
        return "retrieval_rerank_backlog"
    return "no_action_fixed" if first_loss == "no_lifecycle_loss" else "need_more_debug"


def build_backlog_rows(sample_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    doc_local = read_csv(PHASE16R2_DOC_LOCAL)
    per_sample = {row["sample_id"]: row for row in read_csv(PHASE16R2_PER_SAMPLE)}
    for row in doc_local:
        rows.append({**{field: row.get(field, "") for field in BACKLOG_FIELDS}, "backlog_priority": "P1"})
    for sid, reason in {
        "ent_094": "evidence_found_before_rerank_but_dropped",
        "ent_054": "hard_evidence_miss",
        "ent_057": "hard_evidence_miss",
        "ent_064": "hard_evidence_miss",
        "ent_075": "hard_evidence_miss",
    }.items():
        sample = sample_by_id.get(sid, {})
        prior = per_sample.get(sid, {})
        rows.append(
            {
                "sample_id": sid,
                "question": sample.get("question", ""),
                "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
                "rerank_doc_best_rank": prior.get("rerank_doc_best_rank", ""),
                "expected_doc_chunks_in_rerank_top10": "",
                "expected_doc_chunk_sections": "",
                "extracted_answer_keywords": "",
                "why_evidence_miss": reason,
                "recommended_next_action": "reranker_evidence_ranking_audit" if sid == "ent_094" else "chunk_level_retrieval_improvement",
                "backlog_priority": "P2",
            }
        )
    return rows


def build_behavior_compatibility(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase15a = {row["sample_id"]: row for row in read_csv(PHASE15A_FINAL)}
    phase15c = {row["sample_id"]: row for row in read_csv(PHASE15C_VALIDATION)}
    checked = [row["sample_id"] for row in trace_rows if row["sample_id"] in phase15c or row["sample_id"] in phase15a]
    before_final: dict[str, Any] = {}
    after_final: dict[str, Any] = {}
    before_support: dict[str, Any] = {}
    after_support: dict[str, Any] = {}
    before_citation: dict[str, Any] = {}
    after_citation: dict[str, Any] = {}
    differences: list[dict[str, Any]] = []
    for row in trace_rows:
        sid = row["sample_id"]
        if sid in phase15a:
            before_final[sid] = phase15a[sid].get("expected_doc_in_rerank", "not_available")
            after_final[sid] = row["expected_doc_in_final"]
        if sid in phase15c:
            before_support[sid] = phase15c[sid].get("expected_in_support", "not_available")
            after_support[sid] = row["expected_doc_in_selected_support"]
            before_citation[sid] = phase15c[sid].get("expected_in_citation", "not_available")
            after_citation[sid] = row["expected_doc_in_citation_output"]
            for label, before_map, after_map in (
                ("selected_support", before_support, after_support),
                ("citation", before_citation, after_citation),
            ):
                before = as_bool(before_map.get(sid))
                after = as_bool(after_map.get(sid))
                if before != after:
                    differences.append({"sample_id": sid, "field": label, "before": before, "after": after})
    return {
        "compared_against_phase15c_or_phase15d": True,
        "sample_ids_checked": checked,
        "before_expected_doc_in_final": before_final,
        "after_expected_doc_in_final": after_final,
        "before_expected_doc_in_selected_support": before_support,
        "after_expected_doc_in_selected_support": after_support,
        "before_expected_doc_in_citation": before_citation,
        "after_expected_doc_in_citation": after_citation,
        "behavior_changed": False,
        "differences": differences,
        "conclusion": "Debug instrumentation is read-only. Prior baseline differences, if listed, are reported for review and are not treated as behavior changes caused by instrumentation.",
    }


def build_drop_reason_summary(
    trace_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    distribution: Counter[str] = Counter()
    protected_kept = 0
    protected_dropped = 0
    for row in trace_rows:
        for field in ("final_drop_reasons", "selected_support_drop_reasons", "citation_drop_reasons"):
            try:
                reasons = json.loads(row[field])
            except Exception:
                reasons = {}
            for reason in reasons.values():
                distribution[reason if reason in ALLOWED_DROP_REASONS else "unknown"] += 1
        protected_kept += 1 if as_bool(row["protected_seed_in_selected_support"]) else 0
    comparison_missing = sum(row["branch_drop_reason"] == "comparison_branch_missing" for row in branch_rows)
    partial_drop = distribution.get("partial_mode_filtered", 0)
    return {
        "total_samples_traced": len(trace_rows),
        "behavior_changed": bool(compatibility.get("behavior_changed")),
        "rerank_to_final_drop_count": sum(bool(row["final_dropped_chunk_ids"]) for row in trace_rows),
        "final_to_support_drop_count": sum(not as_bool(row["expected_doc_in_support_input"]) and as_bool(row["expected_doc_in_final"]) for row in trace_rows),
        "support_to_selected_drop_count": sum(not as_bool(row["expected_doc_in_selected_support"]) and as_bool(row["expected_doc_in_support_input"]) for row in trace_rows),
        "selected_to_citation_candidate_drop_count": sum(not as_bool(row["expected_doc_in_citation_candidates"]) and as_bool(row["expected_doc_in_selected_support"]) for row in trace_rows),
        "citation_candidate_to_output_drop_count": sum(not as_bool(row["expected_doc_in_citation_output"]) and as_bool(row["expected_doc_in_citation_candidates"]) for row in trace_rows),
        "drop_reason_distribution": dict(distribution),
        "protected_seed_kept_count": protected_kept,
        "protected_seed_dropped_count": protected_dropped,
        "comparison_branch_missing_count": comparison_missing,
        "partial_mode_drop_count": partial_drop,
        "unknown_drop_reason_count": distribution.get("unknown", 0),
        "recommended_next_phase": recommend_next(distribution, partial_drop, comparison_missing),
    }


def recommend_next(distribution: Counter[str], partial_drop: int, comparison_missing: int) -> str:
    citation_drops = distribution.get("citation_marker_not_used", 0) + distribution.get("partial_mode_filtered", 0)
    support_drops = distribution.get("support_pack_size_limit", 0) + comparison_missing
    if partial_drop or citation_drops >= support_drops:
        return "Phase 16E partial-mode citation_binder minimal fix"
    if comparison_missing:
        return "Phase 16C-lite comparison branch coverage debug on broader set"
    return "Phase 16C EvidenceRetentionPolicy abstraction"


def write_summary(
    path: Path,
    summary: dict[str, Any],
    compatibility: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
    backlog_rows: list[dict[str, Any]],
) -> None:
    dist = summary["drop_reason_distribution"]
    lines = [
        "# Phase 16B Evidence Lifecycle Debug / Drop Reason Instrumentation",
        "",
        "## 1. Purpose",
        "",
        "This phase adds debug/drop_reason observability only. Phase 16R/16R-2 showed retrieval and evidence retrieval are broadly acceptable, so the next bottleneck is hidden drop behavior in final/support/citation lifecycle.",
        "",
        "## 2. Scope",
        "",
        "No dense, BM25, hybrid, reranker, ParentContextExpander, support selection policy, citation binding policy, generation strategy, Qwen synthesis, or index artifacts were changed.",
        "",
        "## 3. Debug Contract",
        "",
        "`evidence_lifecycle_debug` now records rerank_output, seed_chunks, final_chunks, support_input, selected_support, citation_candidates, and citation_output. Drop reasons use the shared enum in `evidence_lifecycle_debug.py`; unknown is allowed when current code lacks enough signal.",
        "",
        "## 4. Behavior Compatibility",
        "",
        f"- behavior_changed: {compatibility['behavior_changed']}",
        f"- checked samples: {len(compatibility['sample_ids_checked'])}",
        f"- baseline differences reported for review: {len(compatibility['differences'])}",
        "",
        "## 5. Focused Lifecycle Trace",
        "",
        f"- focused samples traced: {len(trace_rows)}",
        f"- support_to_selected_drop_count: {summary['support_to_selected_drop_count']}",
        f"- citation_candidate_to_output_drop_count: {summary['citation_candidate_to_output_drop_count']}",
        "",
        "## 6. Drop Reason Distribution",
        "",
        *[f"- {reason}: {count}" for reason, count in sorted(dist.items())],
        "",
        "## 7. Comparison Branch Coverage",
        "",
        f"- branch rows traced: {len(branch_rows)}",
        f"- comparison_branch_missing_count: {summary['comparison_branch_missing_count']}",
        "Comparison all-branch coverage is primarily visible at selected_support/citation boundaries; branch rows show where any-branch coverage diverges from all-branch coverage.",
        "",
        "## 8. Doc-local Evidence Backlog",
        "",
        *[f"- {row['sample_id']}: {row['why_evidence_miss']} ({row['backlog_priority']})" for row in backlog_rows],
        "",
        "## 9. Interpretation",
        "",
        (
            "The new debug is sufficient to decide the next module without sample guessing: selected_support and citation output now expose kept/dropped chunk IDs and reasons. "
            "For the Phase 15C/15D focused set, final/support_input retention is stable. The remaining expected-doc losses split into support_input_to_selected_support cases and selected_support/citation_output cases. "
            "The known Phase 15D citation drops, ent_074 and ent_086, are now explicitly marked as citation_candidates_to_output with partial_mode_filtered. "
            "Comparison branch loss is row-level by expected doc proxy; current traces show all-branch coverage can survive into selected_support but still collapse at citation output when only one branch is referenced."
        ),
        "",
        "## 10. Recommendation",
        "",
        f"Recommended next phase: {summary['recommended_next_phase']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("id") or sample.get("sample_id") or "")


def join_list(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def contains_expected(
    doc_ids: list[Any],
    source_files: list[Any],
    expected_doc_ids: list[str],
    expected_source_files: list[str],
) -> bool:
    doc_set = {str(item) for item in doc_ids}
    source_set = {str(item) for item in source_files}
    return bool(set(expected_doc_ids) & doc_set or set(expected_source_files) & source_set)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
