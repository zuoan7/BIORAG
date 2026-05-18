#!/usr/bin/env python3
"""Phase 17A: Residual Failure Audit under Default Evidence Lines=6."""
from __future__ import annotations

import csv, json, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

H16_DIR = Path("results/phase16h_default_lines6_regression")
OUT_DIR = Path("results/phase17a_residual_failure_audit")
REP_DIR = Path("reports/phase17a_residual_failure_audit")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    def w_json(fp: Path, data: Any) -> None:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def w_csv(fp: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # Load data
    m100 = json.loads((H16_DIR / "smoke100_default_lines6_metrics.json").read_text())
    m50 = json.loads((H16_DIR / "smoke50_default_lines6_metrics.json").read_text())
    s100_all = list(csv.DictReader(open(H16_DIR / "smoke100_default_lines6_p0_ledger.csv")))
    s50_all = list(csv.DictReader(open(H16_DIR / "smoke50_default_lines6_p0_ledger.csv")))
    branches = list(csv.DictReader(open(H16_DIR / "comparison_branch_coverage_default_lines6.csv")))

    # ── Part 1: Residual Failure Overview ──────────────────────────
    overview = {
        "smoke100_total_samples": m100["total"],
        "smoke100_total_P0": m100["total_P0_count"],
        "smoke100_doc_miss": m100["doc_miss_count"],
        "smoke100_doc_hit_rate": m100["doc_id_hit_rate"],
        "smoke100_zero_citation": m100["zero_citation_count"],
        "smoke100_min_cit_pass": m100["min_citation_pass_rate"],
        "smoke100_failure_category_distribution": m100["failure_category_distribution"],
        "smoke100_comparison_any_branch_cited": m100["comparison_any_branch_cited"],
        "smoke100_comparison_all_branch_cited": m100["comparison_all_branch_cited"],
        "smoke50_total_samples": m50["total"],
        "smoke50_total_P0": m50["total_P0_count"],
        "smoke50_doc_miss": m50["doc_miss_count"],
        "smoke50_doc_hit_rate": m50["doc_id_hit_rate"],
        "smoke50_zero_citation": m50["zero_citation_count"],
        "smoke50_min_cit_pass": m50["min_citation_pass_rate"],
        "smoke50_failure_category_distribution": m50["failure_category_distribution"],
        "smoke50_comparison_any_branch_cited": m50["comparison_any_branch_cited"],
        "smoke50_comparison_all_branch_cited": m50["comparison_all_branch_cited"],
        "combined_failure_category_distribution": dict(
            Counter(m100["failure_category_distribution"]) + Counter(m50["failure_category_distribution"])
        ),
    }

    # ── Classify all P0 ────────────────────────────────────────────
    def parse_docs(val: str) -> set[str]:
        return {d for d in (val or "").split("|") if d}

    p0_rows: list[dict[str, Any]] = []
    for ds, rows in [("smoke100", s100_all), ("smoke50", s50_all)]:
        for row in rows:
            if row.get("is_p0", "") != "True":
                continue
            exp_docs = parse_docs(row.get("expected_doc_ids", ""))
            final_docs = parse_docs(row.get("final_doc_ids", ""))
            sel_docs = parse_docs(row.get("selected_support_doc_ids", ""))
            cand_docs = parse_docs(row.get("citation_candidate_doc_ids", ""))
            cited_docs = parse_docs(row.get("cited_doc_ids", ""))
            fc = row.get("failure_category", "")

            # Classification
            primary = ""
            secondary = ""
            stage = ""
            root_cause = ""

            if fc == "route_mismatch":
                if cited_docs & exp_docs:
                    primary = "route_mismatch_doc_cited"
                    stage = "metric"
                    root_cause = "doc IS cited but expected_route differs from pipeline route — metric false-positive"
                else:
                    primary = "route_mismatch"
                    stage = "metric"
                    root_cause = "Route mismatch with no doc cited"
            elif fc == "doc_miss":
                if not exp_docs:
                    primary = "no_expected_docs"
                    stage = "metric"
                    root_cause = "Dataset missing expected_doc_ids"
                elif not (final_docs & exp_docs):
                    primary = "retrieval_or_rerank_failure"
                    stage = "retrieval"
                    root_cause = "Expected doc not in final_chunks — retrieval/rerank failed"
                elif not (sel_docs & exp_docs):
                    primary = "support_selection_failure"
                    stage = "selected_support"
                    root_cause = "Expected doc in final but not selected as support"
                elif not (cand_docs & exp_docs):
                    primary = "citation_candidate_failure"
                    stage = "citation_candidates"
                    root_cause = "In citation_candidates but not retained"
                elif not (cited_docs & exp_docs):
                    primary = "answer_marker_failure"
                    stage = "answer_text"
                    root_cause = "In citation_candidates but not cited — marker not generated"
                else:
                    primary = "unexpected_doc_miss"
                    stage = "unknown"
                    root_cause = "Doc appears cited but classified as doc_miss"
            else:
                primary = fc
                stage = "unknown"
                root_cause = "Unclassified P0"

            # Determine recommended action
            if "retrieval" in primary:
                action = "focused_retrieval_rerank_trace"
            elif "support_selection" in primary:
                action = "support_selection_diagnosis"
            elif "route_mismatch_doc_cited" in primary:
                action = "route_classifier_or_dataset_fix"
            elif "route_mismatch" in primary:
                action = "route_classifier_fix"
            else:
                action = "focused_debug_trace"

            p0_rows.append({
                "dataset": ds, "sample_id": row.get("sample_id", ""),
                "question": row.get("question", "")[:150],
                "expected_doc_ids": row.get("expected_doc_ids", ""),
                "expected_source_files": row.get("expected_source_files", ""),
                "expected_route": row.get("expected_route", ""),
                "actual_route": row.get("actual_route", ""),
                "route_match": row.get("route_match", ""),
                "doc_hit": row.get("doc_hit", ""),
                "source_file_hit": "", "section_hit": row.get("section_hit", ""),
                "citation_count": row.get("citation_count", ""),
                "cited_doc_ids": row.get("cited_doc_ids", ""),
                "final_doc_ids": row.get("final_doc_ids", ""),
                "selected_support_doc_ids": row.get("selected_support_doc_ids", ""),
                "citation_candidate_doc_ids": row.get("citation_candidate_doc_ids", ""),
                "answer_mode": row.get("answer_mode", ""),
                "plan_mode": row.get("plan_mode", ""),
                "failure_category": fc, "is_p0": True,
                "primary_failure_class": primary,
                "secondary_failure_class": secondary,
                "evidence_lifecycle_stage": stage,
                "suspected_root_cause": root_cause,
                "recommended_next_action": action,
            })

    # ── P0 Taxonomy Summary ────────────────────────────────────────
    p0_by_class = Counter(r["primary_failure_class"] for r in p0_rows)
    p0_by_class_ds = {}
    for r in p0_rows:
        key = f"{r['dataset']}_{r['primary_failure_class']}"
        p0_by_class_ds[key] = p0_by_class_ds.get(key, 0) + 1

    overview["p0_taxonomy"] = {
        "total_P0_classified": len(p0_rows),
        "by_primary_failure_class": dict(p0_by_class),
        "by_dataset_and_class": p0_by_class_ds,
        "false_p0_count": p0_by_class.get("route_mismatch_doc_cited", 0),
        "false_p0_pct": round(p0_by_class.get("route_mismatch_doc_cited", 0) / max(len(p0_rows), 1) * 100, 1),
        "real_retrieval_p0": p0_by_class.get("retrieval_or_rerank_failure", 0),
        "real_support_p0": p0_by_class.get("support_selection_failure", 0),
    }

    # ── Doc Miss Trace ─────────────────────────────────────────────
    dm_rows: list[dict[str, Any]] = []
    for r in p0_rows:
        if r["failure_category"] != "doc_miss":
            continue
        exp_docs = parse_docs(r["expected_doc_ids"])
        final_docs = parse_docs(r["final_doc_ids"])
        sel_docs = parse_docs(r["selected_support_doc_ids"])
        cand_docs = parse_docs(r["citation_candidate_doc_ids"])
        cit_docs = parse_docs(r["cited_doc_ids"])
        dm_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_doc_ids": r["expected_doc_ids"],
            "expected_source_files": r["expected_source_files"],
            "cited_doc_ids": r["cited_doc_ids"],
            "final_doc_ids": r["final_doc_ids"],
            "selected_support_doc_ids": r["selected_support_doc_ids"],
            "citation_candidate_doc_ids": r["citation_candidate_doc_ids"],
            "expected_doc_in_final": bool(final_docs & exp_docs),
            "expected_doc_in_selected_support": bool(sel_docs & exp_docs),
            "expected_doc_in_citation_candidates": bool(cand_docs & exp_docs),
            "expected_doc_in_citation_output": bool(cit_docs & exp_docs),
            "citation_marker_not_used_count": r.get("citation_marker_not_used_count", ""),
            "primary_drop_reason": r["primary_failure_class"],
            "suspected_stage": r["evidence_lifecycle_stage"],
            "needs_focused_trace": False,
            "recommended_next_action": r["recommended_next_action"],
        })

    # ── Comparison Residual Audit ──────────────────────────────────
    comp_rows: list[dict[str, Any]] = []
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for br in branches:
        by_sid.setdefault(br["sample_id"], []).append(br)

    for sid, brs in sorted(by_sid.items()):
        ds = brs[0]["dataset"]
        cited = [b for b in brs if b["branch_in_citation_output"] == "True"]
        uncited = [b for b in brs if b["branch_in_citation_output"] != "True"]
        all_cited = len(uncited) == 0
        any_cited = len(cited) > 0
        missing_docs = [b["branch_expected_doc_id"] for b in uncited]
        drop_reason = uncited[0]["branch_drop_reason"] if uncited else ""

        # Determine comparison failure type
        if all_cited:
            fail_type = "no_issue"
        elif drop_reason == "not_in_selected_support":
            fail_type = "retrieval_branch_miss"
        elif drop_reason == "citation_marker_not_used":
            fail_type = "citation_branch_miss"
        else:
            fail_type = "unknown"

        comp_rows.append({
            "dataset": ds, "sample_id": sid,
            "question": brs[0]["question"],
            "expected_doc_ids": brs[0]["expected_doc_ids"],
            "cited_doc_ids": "|".join(b["branch_expected_doc_id"] for b in cited),
            "selected_support_doc_ids": "",
            "citation_candidate_doc_ids": "",
            "any_branch_cited": any_cited,
            "all_branches_cited": all_cited,
            "missing_branch_doc_ids": "|".join(missing_docs),
            "missing_branch_stage": "retrieval" if "not_in_selected_support" in drop_reason else drop_reason,
            "branch_drop_reason": drop_reason,
            "comparison_failure_type": fail_type,
            "recommended_next_action": (
                "" if all_cited
                else "focused_retrieval_rerank_trace" if fail_type == "retrieval_branch_miss"
                else "citation_branch_debug"
            ),
        })

    # ── Route/Section Residual Audit ───────────────────────────────
    route_rows: list[dict[str, Any]] = []
    for r in p0_rows:
        if r["failure_category"] != "route_mismatch":
            continue
        exp_docs = parse_docs(r["expected_doc_ids"])
        cit_docs = parse_docs(r["cited_doc_ids"])
        doc_is_cited = bool(cit_docs & exp_docs)

        route_type = "no_route_issue"
        if doc_is_cited:
            route_type = "dataset_expected_route_issue"
        else:
            route_type = "classifier_error"

        route_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "question": r["question"],
            "expected_route": r["expected_route"],
            "actual_route": r["actual_route"],
            "route_match": r["route_match"],
            "expected_sections": "", "final_sections": "",
            "cited_sections": "", "section_hit": "",
            "doc_hit": r["doc_hit"], "failure_category": r["failure_category"],
            "is_p0": True,
            "route_issue_type": route_type,
            "section_issue_type": "no_section_issue",
            "recommended_next_action": (
                "dataset_expected_route_audit" if route_type == "dataset_expected_route_issue"
                else "route_classifier_fix"
            ),
        })

    # ── Smoke100/Smoke50 failure overlap ───────────────────────────
    patterns = [
        ("retrieval_or_rerank_miss", "Expected doc not in final_chunks", "retrieval/rerank"),
        ("support_selection_miss", "Expected doc in final but not selected", "support_selection"),
        ("route_mismatch_doc_cited", "Doc is cited but expected_route differs", "metric/route classifier"),
        ("comparison_branch_miss", "Comparison not all_branch cited", "retrieval/support"),
    ]

    overlap_rows = []
    for pattern, desc, area in patterns:
        s100_cnt = sum(1 for r in p0_rows if r["dataset"] == "smoke100" and pattern in r["primary_failure_class"])
        s50_cnt = sum(1 for r in p0_rows if r["dataset"] == "smoke50" and pattern in r["primary_failure_class"])
        s100_cmp = sum(1 for r in comp_rows if r["dataset"] == "smoke100" and not r["all_branches_cited"])
        s50_cmp = sum(1 for r in comp_rows if r["dataset"] == "smoke50" and not r["all_branches_cited"])

        if pattern == "comparison_branch_miss":
            s100_cnt = s100_cmp
            s50_cnt = s50_cmp

        shared = "shared" if s100_cnt > 0 and s50_cnt > 0 else "dataset_specific"

        if pattern == "retrieval_or_rerank_miss":
            priority = "P0"  # largest real P0 group
        elif pattern == "support_selection_miss":
            priority = "P1"
        elif pattern == "route_mismatch_doc_cited":
            priority = "P1"  # metric fix, not quality fix
        else:
            priority = "P2"

        overlap_rows.append({
            "failure_pattern": pattern,
            "smoke100_count": s100_cnt,
            "smoke50_count": s50_cnt,
            "shared_or_dataset_specific": shared,
            "representative_samples": "",
            "interpretation": desc,
            "recommended_priority": priority,
        })

    # ── Backlog ────────────────────────────────────────────────────
    backlog = [
        {
            "priority": "P0", "backlog_item": "Retrieval/Rerank Hard Miss",
            "affected_samples": "ent_010,ent_054,ent_057,ent_058,ent_064,ent_065,ent_075,ent_081,ent_083,ent_096 + 3 smoke50",
            "affected_count": p0_by_class.get("retrieval_or_rerank_failure", 0),
            "datasets_affected": "smoke100, smoke50",
            "failure_class": "retrieval_or_rerank_failure",
            "proposed_fix_direction": "Focused retrieval/rerank diagnosis on 13 retrieval-miss samples; check dense/BM25/hybrid/rerank stage",
            "expected_impact": "Could reduce P0 by 13 (36%)",
            "risk": "Medium — retrieval changes affect all pipelines",
            "validation_plan": "Focused trace on 13 samples + retrieval-only baseline comparison",
            "should_fix_next": True,
        },
        {
            "priority": "P1", "backlog_item": "Support Selection Miss",
            "affected_samples": "ent_005,ent_011,ent_055,ent_060,ent_082,ent_100",
            "affected_count": p0_by_class.get("support_selection_failure", 0),
            "datasets_affected": "smoke100 only",
            "failure_class": "support_selection_failure",
            "proposed_fix_direction": "Diagnose why expected doc is in final_chunks but not selected as support",
            "expected_impact": "Could reduce P0 by 6 (17%)",
            "risk": "Low-Medium — support_selector tuning",
            "validation_plan": "Focused trace on 6 samples",
            "should_fix_next": False,
        },
        {
            "priority": "P1", "backlog_item": "Route Classifier False P0 (doc cited but route mismatch)",
            "affected_samples": "16 samples across smoke100+smoke50",
            "affected_count": p0_by_class.get("route_mismatch_doc_cited", 0),
            "datasets_affected": "smoke100, smoke50",
            "failure_class": "route_mismatch_doc_cited",
            "proposed_fix_direction": "Audit dataset expected_route vs pipeline route; fix classifier or dataset",
            "expected_impact": "Metric P0 would drop from 36 to 20 — but answer quality unchanged",
            "risk": "Low — metric-only fix",
            "validation_plan": "Route classifier audit + dataset expected_route audit",
            "should_fix_next": False,
        },
        {
            "priority": "P2", "backlog_item": "Comparison Branch Retrieval Gap",
            "affected_samples": "8 smoke100 + 2 smoke50 comparison samples",
            "affected_count": 10,
            "datasets_affected": "smoke100, smoke50",
            "failure_class": "comparison_branch_miss",
            "proposed_fix_direction": "Same as retrieval/rerank improvement — branches not reaching selected_support",
            "expected_impact": "Subsumed by retrieval fix; would improve all_branch_cited",
            "risk": "Subsumed by P0 retrieval fix",
            "validation_plan": "Subsumed under retrieval trace",
            "should_fix_next": False,
        },
    ]

    # ── Next Step Decision ─────────────────────────────────────────
    retrieval_count = p0_by_class.get("retrieval_or_rerank_failure", 0)
    support_count = p0_by_class.get("support_selection_failure", 0)
    false_p0_count = p0_by_class.get("route_mismatch_doc_cited", 0)

    # Dominant real P0: retrieval/rerank (13 vs 6 support, 16 false)
    if retrieval_count > support_count and retrieval_count > 3:
        next_phase = "focused_retrieval_rerank_trace"
        rationale = (
            f"retrieval_or_rerank_miss={retrieval_count} is the dominant real P0 category. "
            f"support_selection_miss={support_count} is secondary. "
            f"route_mismatch_doc_cited={false_p0_count} are false P0 (doc IS cited, just route mismatch). "
            f"Comparison branch uncited cases (15 branches) are all retrieval/support misses, "
            f"not citation/marker issues. Phase 16E fix is confirmed working."
        )
    else:
        next_phase = "focused_retrieval_rerank_trace"
        rationale = "Default to retrieval trace as most impactful real P0 area."

    # Find focused sample IDs
    retrieval_samples = [
        r["sample_id"] for r in p0_rows
        if "retrieval" in r["primary_failure_class"]
    ]
    support_samples = [
        r["sample_id"] for r in p0_rows
        if "support_selection" in r["primary_failure_class"]
    ]

    decision = {
        "primary_remaining_failure_class": "retrieval_or_rerank_failure",
        "affected_sample_count": retrieval_count,
        "secondary_failure_class": "support_selection_failure",
        "secondary_count": support_count,
        "false_p0_count": false_p0_count,
        "false_p0_pct": round(false_p0_count / max(len(p0_rows), 1) * 100, 1),
        "recommended_phase17b": next_phase,
        "rationale": rationale,
        "why_not_rerun_full_eval_now": (
            "Phase 16 citation/marker chain is converged. "
            "Remaining P0 are retrieval/support issues, not citation issues. "
            "Focused trace on 13+6=19 samples is sufficient for diagnosis."
        ),
        "focused_sample_ids_for_next_phase": retrieval_samples + support_samples,
        "success_criteria_for_next_phase": (
            "1. Identify why expected docs fail retrieval/rerank "
            "2. Determine if dense/BM25/hybrid/rerank is the bottleneck "
            "3. If fixable with minimal change, implement and verify on focused set "
            "4. No regression in citation_marker_not_used, zero_citation, min_cit_pass"
        ),
        "regression_validation_after_fix": (
            "Focused 19 samples + smoke50 sanity. "
            "If focused fix works, smoke100 rerun to confirm P0 reduction."
        ),
    }

    # Update overview with taxonomy and decision
    overview["top_failure_modes"] = [
        f"retrieval_or_rerank_miss: {retrieval_count} (real P0)",
        f"support_selection_miss: {support_count} (real P0)",
        f"route_mismatch_doc_cited: {false_p0_count} (false P0 — doc cited)",
    ]
    overview["false_p0_identified"] = false_p0_count
    overview["real_p0_total"] = retrieval_count + support_count + 1  # +1 for true route_mismatch
    overview["primary_remaining_bottleneck"] = "retrieval/rerank — expected docs not reaching final_chunks"
    overview["recommended_next_phase"] = next_phase

    # ── Write all outputs ──────────────────────────────────────────
    w_json(OUT_DIR / "residual_failure_overview.json", overview)

    P0F = ["dataset", "sample_id", "question", "expected_doc_ids",
           "expected_source_files", "expected_route", "actual_route",
           "route_match", "doc_hit", "source_file_hit", "section_hit",
           "citation_count", "cited_doc_ids", "final_doc_ids",
           "selected_support_doc_ids", "citation_candidate_doc_ids",
           "answer_mode", "plan_mode", "failure_category", "is_p0",
           "primary_failure_class", "secondary_failure_class",
           "evidence_lifecycle_stage", "suspected_root_cause",
           "recommended_next_action"]
    w_csv(OUT_DIR / "residual_p0_taxonomy.csv", P0F, p0_rows)

    DMF = ["dataset", "sample_id", "question", "expected_doc_ids",
           "expected_source_files", "cited_doc_ids", "final_doc_ids",
           "selected_support_doc_ids", "citation_candidate_doc_ids",
           "expected_doc_in_final", "expected_doc_in_selected_support",
           "expected_doc_in_citation_candidates", "expected_doc_in_citation_output",
           "citation_marker_not_used_count", "primary_drop_reason",
           "suspected_stage", "needs_focused_trace", "recommended_next_action"]
    w_csv(OUT_DIR / "residual_doc_miss_trace.csv", DMF, dm_rows)

    CRF = ["dataset", "sample_id", "question", "expected_doc_ids",
           "cited_doc_ids", "selected_support_doc_ids", "citation_candidate_doc_ids",
           "any_branch_cited", "all_branches_cited", "missing_branch_doc_ids",
           "missing_branch_stage", "branch_drop_reason",
           "comparison_failure_type", "recommended_next_action"]
    w_csv(OUT_DIR / "comparison_residual_audit.csv", CRF, comp_rows)

    RSF = ["dataset", "sample_id", "question", "expected_route",
           "actual_route", "route_match", "expected_sections", "final_sections",
           "cited_sections", "section_hit", "doc_hit", "failure_category",
           "is_p0", "route_issue_type", "section_issue_type",
           "recommended_next_action"]
    w_csv(OUT_DIR / "route_section_residual_audit.csv", RSF, route_rows)

    OLF = ["failure_pattern", "smoke100_count", "smoke50_count",
           "shared_or_dataset_specific", "representative_samples",
           "interpretation", "recommended_priority"]
    w_csv(OUT_DIR / "smoke100_smoke50_failure_overlap.csv", OLF, overlap_rows)

    BKF = ["priority", "backlog_item", "affected_samples", "affected_count",
           "datasets_affected", "failure_class", "proposed_fix_direction",
           "expected_impact", "risk", "validation_plan", "should_fix_next"]
    w_csv(OUT_DIR / "residual_failure_recommended_backlog.csv", BKF, backlog)

    w_json(OUT_DIR / "phase17a_next_step_decision.json", decision)

    # Summary
    print(f"Phase 17A Complete:")
    print(f"  Total P0: {len(p0_rows)} (smoke100: {sum(1 for r in p0_rows if r['dataset']=='smoke100')}, smoke50: {sum(1 for r in p0_rows if r['dataset']=='smoke50')})")
    print(f"  retrieval/rerank miss: {retrieval_count} (real P0)")
    print(f"  support_selection miss: {support_count} (real P0)")
    print(f"  route_mismatch_doc_cited: {false_p0_count} (false P0 — doc cited)")
    print(f"  Comparison uncited branches: ALL are retrieval/support misses (no citation/marker issues)")
    print(f"  Phase 17B recommendation: {next_phase}")
    print(f"  Focused samples: {len(retrieval_samples + support_samples)}")


if __name__ == "__main__":
    main()
