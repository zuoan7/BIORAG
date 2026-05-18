#!/usr/bin/env python3
"""Phase 18A: Residual failure re-audit after source-floor default-on."""
from __future__ import annotations

import csv, json, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = Path("results/phase18a_post_source_floor_residual_audit")
REP_DIR = Path("reports/phase18a_post_source_floor_residual_audit")
S100_P0 = Path("results/phase17f_source_floor_default_on_regression/smoke100_default_source_floor_p0_ledger.csv")
S50_P0 = Path("results/phase17f_source_floor_default_on_regression/smoke50_default_source_floor_p0_ledger.csv")
S100_M = Path("results/phase17f_source_floor_default_on_regression/smoke100_default_source_floor_metrics.json")
S50_M = Path("results/phase17f_source_floor_default_on_regression/smoke50_default_source_floor_metrics.json")
PHASE17A_TAXO = Path("results/phase17a_residual_failure_audit/residual_p0_taxonomy.csv")


def parse_docs(val: str) -> set[str]:
    return {d for d in (val or "").split("|") if d}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    def w_csv(fp, fields, rows):
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def w_json(fp, data):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Load data
    s100_rows = list(csv.DictReader(open(S100_P0))) if S100_P0.exists() else []
    s50_rows = list(csv.DictReader(open(S50_P0))) if S50_P0.exists() else []
    s100_m = json.loads(S100_M.read_text()) if S100_M.exists() else {}
    s50_m = json.loads(S50_M.read_text()) if S50_M.exists() else {}
    p17a_rows = list(csv.DictReader(open(PHASE17A_TAXO))) if PHASE17A_TAXO.exists() else []
    p17a_by_id = {(r.get("dataset", ""), r.get("sample_id", "")): r for r in p17a_rows}

    # ── Part 1: Overview ───────────────────────────────────────────
    overview = {
        "smoke100_total_samples": s100_m.get("total", 100),
        "smoke100_total_P0": s100_m.get("total_P0_count", 0),
        "smoke100_doc_miss": s100_m.get("doc_miss_count", 0),
        "smoke100_doc_hit_rate": s100_m.get("doc_hit_rate", 0),
        "smoke100_zero_citation": s100_m.get("zero_citation_count", 0),
        "smoke100_min_cit_pass": s100_m.get("min_citation_pass_rate", 0),
        "smoke100_failure_category_distribution": s100_m.get("failure_category_distribution", {}),
        "smoke50_total_samples": s50_m.get("total", 50),
        "smoke50_total_P0": s50_m.get("total_P0_count", 0),
        "smoke50_doc_miss": s50_m.get("doc_miss_count", 0),
        "smoke50_doc_hit_rate": s50_m.get("doc_hit_rate", 0),
        "smoke50_zero_citation": s50_m.get("zero_citation_count", 0),
        "smoke50_min_cit_pass": s50_m.get("min_citation_pass_rate", 0),
        "smoke50_failure_category_distribution": s50_m.get("failure_category_distribution", {}),
    }

    # ── Classify all P0 ────────────────────────────────────────────
    p0_rows: list[dict[str, Any]] = []

    for ds, rows in [("smoke100", s100_rows), ("smoke50", s50_rows)]:
        for row in rows:
            if row.get("is_p0", "") != "True":
                continue
            fc = row.get("failure_category", "")
            exp_docs = parse_docs(row.get("expected_doc_ids", ""))
            final_docs = parse_docs(row.get("final_doc_ids", ""))
            sel_docs = parse_docs(row.get("selected_support_doc_ids", ""))
            cand_docs = parse_docs(row.get("citation_candidate_doc_ids", ""))
            cit_docs = parse_docs(row.get("cited_doc_ids", ""))

            in_final = bool(final_docs & exp_docs) if exp_docs else False
            in_sel = bool(sel_docs & exp_docs) if exp_docs else False
            in_cand = bool(cand_docs & exp_docs) if exp_docs else False
            in_cit = bool(cit_docs & exp_docs) if exp_docs else False

            # Classification
            primary = ""
            stage = ""
            root = ""
            is_true = True

            if fc == "route_mismatch":
                if in_cit:
                    primary = "route_mismatch_false_p0_doc_cited"
                    is_true = False
                    stage = "metric"
                    root = "Doc IS cited but expected_route differs — metric false-positive"
                else:
                    primary = "route_mismatch_true"
                    stage = "metric"
                    root = "Route mismatch, doc not cited"
            elif fc == "doc_miss":
                if not exp_docs:
                    primary = "metric_or_dataset_issue"
                    is_true = False
                    stage = "metric"
                    root = "No expected_doc_ids in dataset"
                elif not in_final:
                    primary = "hard_recall_miss"
                    stage = "retrieval"
                    root = "Expected doc not in final_chunks — retrieval/rerank/final failed"
                elif not in_sel:
                    primary = "support_selection_miss"
                    stage = "selected_support"
                    root = "Expected doc in final_chunks but not selected as support"
                elif not in_cand:
                    primary = "citation_candidate_failure"
                    stage = "citation_candidates"
                    root = "Expected doc in selected_support but not in citation_candidates"
                elif not in_cit:
                    primary = "citation_output_or_marker_failure"
                    stage = "citation_output"
                    root = "Expected doc in citation_candidates but not cited"
                else:
                    primary = "unclear"
                    stage = "unknown"
                    root = "Unexpected — all checks pass but classified as doc_miss"
            elif fc == "partial_answer":
                # Check if comparison branch issue
                exp_route = row.get("expected_route", "")
                if exp_route == "comparison" and exp_docs and not all(d in cit_docs for d in exp_docs):
                    primary = "comparison_branch_failure"
                    stage = "answer_text"
                    root = "Not all comparison branches cited"
                elif in_cit:
                    primary = "answer_quality_or_incomplete"
                    stage = "answer_text"
                    root = "Doc cited but answer classified as partial"
                else:
                    primary = "answer_quality_or_incomplete"
                    stage = "answer_text"
                    root = "Partial answer"
            else:
                primary = fc
                stage = "unknown"
                root = ""

            p17a_key = (ds, row.get("sample_id", ""))
            p17a_info = p17a_by_id.get(p17a_key, {})
            was_retrieval = p17a_info.get("primary_failure_class", "") == "retrieval_or_rerank_failure"

            # Action
            action_map = {
                "hard_recall_miss": "hard_recall_miss_audit",
                "support_selection_miss": "support_selection_focused_trace",
                "citation_candidate_failure": "citation_candidate_diagnosis",
                "citation_output_or_marker_failure": "citation_marker_diagnosis",
                "route_mismatch_false_p0_doc_cited": "route_metric_eval_fix",
                "route_mismatch_true": "route_classifier_fix",
                "comparison_branch_failure": "comparison_branch_support_audit",
            }

            p0_rows.append({
                "dataset": ds, "sample_id": row.get("sample_id", ""),
                "question": row.get("question", "")[:150],
                "expected_doc_ids": row.get("expected_doc_ids", ""),
                "expected_source_files": "", "expected_route": row.get("expected_route", ""),
                "actual_route": row.get("actual_route", row.get("answer_mode", "")),
                "route_match": row.get("route_match", ""), "doc_hit": row.get("doc_hit", ""),
                "source_file_hit": "", "section_hit": "",
                "citation_count": row.get("citation_count", ""),
                "cited_doc_ids": row.get("cited_doc_ids", ""),
                "final_doc_ids": row.get("final_doc_ids", ""),
                "selected_support_doc_ids": row.get("selected_support_doc_ids", ""),
                "citation_candidate_doc_ids": row.get("citation_candidate_doc_ids", ""),
                "answer_mode": row.get("answer_mode", ""),
                "plan_mode": row.get("answer_mode", ""),
                "failure_category": fc, "is_p0": True,
                "source_floor_added_count": "",
                "source_floor_added_doc_ids": "",
                "primary_failure_class": primary,
                "secondary_failure_class": "",
                "evidence_lifecycle_stage": stage,
                "suspected_root_cause": root,
                "is_true_p0": is_true,
                "recommended_next_action": action_map.get(primary, "unknown"),
            })

    # ── Taxonomy summary ───────────────────────────────────────────
    p0_by_class = Counter(r["primary_failure_class"] for r in p0_rows)
    true_p0 = sum(1 for r in p0_rows if r["is_true_p0"])
    false_p0 = sum(1 for r in p0_rows if not r["is_true_p0"])

    overview["true_p0_count"] = true_p0
    overview["false_p0_count"] = false_p0
    overview["p0_taxonomy"] = dict(p0_by_class)

    # ── Doc miss trace ─────────────────────────────────────────────
    dm_rows = []
    for r in p0_rows:
        if r["failure_category"] != "doc_miss":
            continue
        exp_docs = parse_docs(r["expected_doc_ids"])
        fin_docs = parse_docs(r["final_doc_ids"])
        sel_docs = parse_docs(r["selected_support_doc_ids"])
        cand_docs = parse_docs(r["citation_candidate_doc_ids"])
        cit_docs = parse_docs(r["cited_doc_ids"])
        dm_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_doc_ids": r["expected_doc_ids"],
            "expected_source_files": "",
            "cited_doc_ids": r["cited_doc_ids"],
            "final_doc_ids": r["final_doc_ids"],
            "selected_support_doc_ids": r["selected_support_doc_ids"],
            "citation_candidate_doc_ids": r["citation_candidate_doc_ids"],
            "expected_doc_in_final": bool(fin_docs & exp_docs) if exp_docs else False,
            "expected_doc_in_selected_support": bool(sel_docs & exp_docs) if exp_docs else False,
            "expected_doc_in_citation_candidates": bool(cand_docs & exp_docs) if exp_docs else False,
            "expected_doc_in_citation_output": bool(cit_docs & exp_docs) if exp_docs else False,
            "source_floor_added_doc_ids": "",
            "source_floor_helped": False,
            "citation_marker_not_used_count": "",
            "primary_drop_reason": r["primary_failure_class"],
            "suspected_stage": r["evidence_lifecycle_stage"],
            "needs_focused_trace": r["primary_failure_class"] not in ("hard_recall_miss", "unclear"),
            "recommended_next_action": r["recommended_next_action"],
        })

    # ── Support selection audit ────────────────────────────────────
    sup_rows = []
    for r in p0_rows:
        if r["primary_failure_class"] != "support_selection_miss":
            continue
        exp_docs = parse_docs(r["expected_doc_ids"])
        fin_docs = parse_docs(r["final_doc_ids"])
        sel_docs = parse_docs(r["selected_support_doc_ids"])
        sup_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_doc_ids": r["expected_doc_ids"],
            "final_doc_ids": r["final_doc_ids"],
            "selected_support_doc_ids": r["selected_support_doc_ids"],
            "citation_candidate_doc_ids": r["citation_candidate_doc_ids"],
            "expected_doc_in_final": bool(fin_docs & exp_docs) if exp_docs else False,
            "expected_doc_in_selected_support": bool(sel_docs & exp_docs) if exp_docs else False,
            "final_expected_doc_rank_if_available": "",
            "final_expected_chunk_ids": "",
            "final_expected_sections": "",
            "support_pack_size": "", "selected_support_count": "",
            "support_selection_drop_reason_if_available": "",
            "answer_mode": r["answer_mode"],
            "plan_mode": r["plan_mode"],
            "is_comparison": r.get("expected_route", "") == "comparison",
            "support_miss_type": "unclear",
            "recommended_next_action": "focused_support_selection_trace",
        })

    # ── Retrieval remaining ────────────────────────────────────────
    ret_rows = []
    for r in p0_rows:
        if r["primary_failure_class"] not in ("hard_recall_miss", "unclear"):
            if r["evidence_lifecycle_stage"] not in ("retrieval", "hybrid", "rerank", "final_chunks"):
                continue
        if r["primary_failure_class"] in ("support_selection_miss", "citation_candidate_failure",
                                           "citation_output_or_marker_failure", "route_mismatch_false_p0_doc_cited"):
            continue
        p17a_key = (r["dataset"], r["sample_id"])
        p17a_info = p17a_by_id.get(p17a_key, {})
        ret_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_doc_ids": r["expected_doc_ids"],
            "phase17b_class_if_available": p17a_info.get("primary_failure_class", ""),
            "source_floor_added_doc_ids": "",
            "source_floor_helped": False,
            "current_final_doc_ids": r["final_doc_ids"],
            "current_doc_hit": r["doc_hit"],
            "remaining_retrieval_type": r["primary_failure_class"],
            "likely_next_fix": "no_retrieval_fix_now" if r["primary_failure_class"] != "hard_recall_miss" else "query_expansion_or_alias",
            "recommended_next_action": "hard_recall_miss_audit" if r["primary_failure_class"] == "hard_recall_miss" else "unclear",
        })

    # ── Comparison ─────────────────────────────────────────────────
    comp_rows = []
    for r in p0_rows:
        if r.get("expected_route", "") != "comparison":
            continue
        exp_docs = parse_docs(r["expected_doc_ids"])
        if not exp_docs: continue
        cit_docs = parse_docs(r["cited_doc_ids"])
        sel_docs = parse_docs(r["selected_support_doc_ids"])
        any_b = bool(cit_docs & exp_docs)
        all_b = all(d in cit_docs for d in exp_docs)
        missing = [d for d in exp_docs if d not in cit_docs]
        comp_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_doc_ids": r["expected_doc_ids"],
            "cited_doc_ids": r["cited_doc_ids"],
            "final_doc_ids": r["final_doc_ids"],
            "selected_support_doc_ids": r["selected_support_doc_ids"],
            "citation_candidate_doc_ids": r["citation_candidate_doc_ids"],
            "any_branch_cited": any_b,
            "all_branches_cited": all_b,
            "missing_branch_doc_ids": "|".join(missing),
            "missing_branch_stage": (
                "not_retrieved" if not any(d in parse_docs(r["final_doc_ids"]) for d in missing)
                else "not_in_selected_support" if not any(d in sel_docs for d in missing)
                else "not_cited"
            ),
            "branch_drop_reason": "",
            "comparison_failure_type": (
                "retrieval_branch_miss" if exp_docs and not parse_docs(r["final_doc_ids"]) & exp_docs
                else "support_branch_miss" if exp_docs and not sel_docs & exp_docs
                else "citation_branch_miss"
            ),
            "recommended_next_action": "comparison_branch_support_audit",
        })

    # ── Route false positive ───────────────────────────────────────
    route_rows = []
    for r in p0_rows:
        if r["primary_failure_class"] != "route_mismatch_false_p0_doc_cited":
            continue
        cit_docs = parse_docs(r["cited_doc_ids"])
        exp_docs = parse_docs(r["expected_doc_ids"])
        route_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_route": r["expected_route"],
            "actual_route": r["actual_route"],
            "route_match": r["route_match"],
            "expected_doc_ids": r["expected_doc_ids"],
            "cited_doc_ids": r["cited_doc_ids"],
            "expected_doc_cited": bool(cit_docs & exp_docs) if exp_docs else False,
            "answer_quality_issue_present": False,
            "should_count_as_p0": False,
            "false_p0_reason": "doc_correctly_cited_route_only_mismatch",
            "recommended_eval_fix": "route_eval_expected_route_audit",
        })

    # ── Backlog ────────────────────────────────────────────────────
    support_count = sum(1 for r in p0_rows if r["primary_failure_class"] == "support_selection_miss")
    recall_count = sum(1 for r in p0_rows if r["primary_failure_class"] == "hard_recall_miss")
    false_p0_count = sum(1 for r in p0_rows if not r["is_true_p0"])
    comp_count = sum(1 for r in p0_rows if r["primary_failure_class"] == "comparison_branch_failure")

    backlog = []
    if support_count > 0:
        backlog.append({
            "priority": "P0", "backlog_item": "Support Selection Miss",
            "affected_samples": "ent_005,ent_011,ent_055,ent_060,ent_082,ent_100 + others",
            "affected_count": support_count,
            "datasets_affected": "smoke100",
            "failure_class": "support_selection_miss",
            "proposed_fix_direction": "Focused trace on why expected docs in final_chunks are not selected as support",
            "expected_impact": f"Reduce P0 by {support_count}",
            "risk": "Low-Medium", "validation_plan": "Focused trace + smoke50 sanity",
            "should_fix_next": True,
        })
    if recall_count > 0:
        backlog.append({
            "priority": "P1", "backlog_item": "Hard Recall Miss",
            "affected_samples": "ent_054,ent_057,ent_064,ent_075,ent_081,ent_083,ent_096 + smoke50",
            "affected_count": recall_count,
            "datasets_affected": "smoke100, smoke50",
            "failure_class": "hard_recall_miss",
            "proposed_fix_direction": "Query-doc mismatch audit; synonym/alias/query expansion",
            "expected_impact": f"Reduce P0 by {recall_count}",
            "risk": "Medium — retrieval changes affect all", "validation_plan": "Focused trace",
            "should_fix_next": False,
        })
    if false_p0_count > 0:
        backlog.append({
            "priority": "P2", "backlog_item": "Route Metric False Positives",
            "affected_samples": f"~{false_p0_count} samples",
            "affected_count": false_p0_count,
            "datasets_affected": "smoke100, smoke50",
            "failure_class": "route_mismatch_false_p0_doc_cited",
            "proposed_fix_direction": "Fix eval expected_route or route classifier — metric only, not pipeline",
            "expected_impact": f"Metric P0 would drop by {false_p0_count}",
            "risk": "Low — eval fix only", "validation_plan": "Eval audit",
            "should_fix_next": False,
        })

    # ── Decision ───────────────────────────────────────────────────
    # Determine dominant real P0
    real_p0_classes = dict(Counter(
        r["primary_failure_class"] for r in p0_rows if r["is_true_p0"]))
    dominant = max(real_p0_classes, key=real_p0_classes.get) if real_p0_classes else "none"
    dominant_cnt = real_p0_classes.get(dominant, 0) if real_p0_classes else 0

    next_phase_map = {
        "support_selection_miss": "support_selection_focused_trace",
        "hard_recall_miss": "hard_recall_miss_audit",
        "citation_output_or_marker_failure": "citation_marker_diagnosis",
        "comparison_branch_failure": "comparison_branch_support_audit",
    }
    next_phase = next_phase_map.get(dominant, "no_single_dominant_issue")

    decision = {
        "primary_remaining_failure_class": dominant,
        "affected_sample_count": dominant_cnt,
        "recommended_phase18b": next_phase,
        "rationale": (
            f"support_selection_miss={support_count}, hard_recall_miss={recall_count}, "
            f"comparison_branch={comp_count}, false_p0={false_p0_count}. "
            f"Dominant real P0: {dominant} ({dominant_cnt}). "
        ),
        "why_not_continue_source_floor_tuning": "Source-floor top3 verified safe. Expanding topN would increase noise without proven benefit.",
        "why_not_rerun_full_eval_now": "Focused trace on the dominant failure class is sufficient for diagnosis.",
        "focused_sample_ids_for_next_phase": [r["sample_id"] for r in p0_rows if r["primary_failure_class"] == dominant],
        "success_criteria_for_next_phase": f"Identify root cause of {dominant} and propose minimal fix approach",
        "regression_validation_after_fix": "Focused samples + smoke50 sanity",
    }

    # Update overview
    overview["primary_remaining_bottleneck"] = f"{dominant} ({dominant_cnt} real P0)"
    overview["recommended_next_phase"] = next_phase

    # ── Write outputs ──────────────────────────────────────────────
    w_json(OUT_DIR / "residual_failure_overview.json", overview)

    P0F = list(p0_rows[0].keys()) if p0_rows else []
    w_csv(OUT_DIR / "residual_p0_taxonomy.csv", P0F, p0_rows)

    DMF = ["dataset", "sample_id", "question", "expected_doc_ids",
           "expected_source_files", "cited_doc_ids", "final_doc_ids",
           "selected_support_doc_ids", "citation_candidate_doc_ids",
           "expected_doc_in_final", "expected_doc_in_selected_support",
           "expected_doc_in_citation_candidates", "expected_doc_in_citation_output",
           "source_floor_added_doc_ids", "source_floor_helped",
           "citation_marker_not_used_count", "primary_drop_reason",
           "suspected_stage", "needs_focused_trace", "recommended_next_action"]
    w_csv(OUT_DIR / "residual_doc_miss_trace.csv", DMF, dm_rows)

    SUPF = ["dataset", "sample_id", "question", "expected_doc_ids",
            "final_doc_ids", "selected_support_doc_ids", "citation_candidate_doc_ids",
            "expected_doc_in_final", "expected_doc_in_selected_support",
            "final_expected_doc_rank_if_available", "final_expected_chunk_ids",
            "final_expected_sections", "support_pack_size", "selected_support_count",
            "support_selection_drop_reason_if_available", "answer_mode", "plan_mode",
            "is_comparison", "support_miss_type", "recommended_next_action"]
    w_csv(OUT_DIR / "support_selection_residual_audit.csv", SUPF, sup_rows)

    RETF = ["dataset", "sample_id", "question", "expected_doc_ids",
            "phase17b_class_if_available", "source_floor_added_doc_ids",
            "source_floor_helped", "current_final_doc_ids", "current_doc_hit",
            "remaining_retrieval_type", "likely_next_fix", "recommended_next_action"]
    w_csv(OUT_DIR / "retrieval_remaining_residual_audit.csv", RETF, ret_rows)

    COMPF = ["dataset", "sample_id", "question", "expected_doc_ids",
             "cited_doc_ids", "final_doc_ids", "selected_support_doc_ids",
             "citation_candidate_doc_ids", "any_branch_cited", "all_branches_cited",
             "missing_branch_doc_ids", "missing_branch_stage", "branch_drop_reason",
             "comparison_failure_type", "recommended_next_action"]
    w_csv(OUT_DIR / "comparison_residual_audit.csv", COMPF, comp_rows)

    RTF = ["dataset", "sample_id", "question", "expected_route",
           "actual_route", "route_match", "expected_doc_ids", "cited_doc_ids",
           "expected_doc_cited", "answer_quality_issue_present",
           "should_count_as_p0", "false_p0_reason", "recommended_eval_fix"]
    w_csv(OUT_DIR / "route_metric_false_positive_audit.csv", RTF, route_rows)

    BKF = ["priority", "backlog_item", "affected_samples", "affected_count",
           "datasets_affected", "failure_class", "proposed_fix_direction",
           "expected_impact", "risk", "validation_plan", "should_fix_next"]
    w_csv(OUT_DIR / "phase18a_backlog.csv", BKF, backlog)

    w_json(OUT_DIR / "phase18a_next_step_decision.json", decision)

    # Print
    print(f"Phase 18A Complete:")
    print(f"  Total P0: {len(p0_rows)} (smoke100: {sum(1 for r in p0_rows if r['dataset']=='smoke100')}, smoke50: {sum(1 for r in p0_rows if r['dataset']=='smoke50')})")
    print(f"  True P0: {true_p0}, False P0: {false_p0}")
    print(f"  P0 taxonomy: {dict(p0_by_class)}")
    print(f"  Dominant real P0: {dominant} ({dominant_cnt})")
    print(f"  Phase 18B: {next_phase}")


if __name__ == "__main__":
    main()
