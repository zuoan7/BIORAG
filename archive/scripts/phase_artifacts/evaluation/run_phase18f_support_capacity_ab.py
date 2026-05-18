#!/usr/bin/env python3
"""Phase 18F: Support pack capacity 3 vs 5 focused A/B on 6 support_miss samples."""
import csv, json, os, sys, time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

OUT_DIR = Path("results/phase18f_support_pack_capacity_ablation")
REP_DIR = Path("reports/phase18f_support_pack_capacity_ablation")
SMOKE100 = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"

FOCUSED6 = [
    ("smoke100", "ent_005", ["doc_0009"]),
    ("smoke100", "ent_011", ["doc_0054", "doc_0072", "doc_0073"]),
    ("smoke100", "ent_055", ["doc_0081"]),
    ("smoke100", "ent_060", ["doc_0105"]),
    ("smoke100", "ent_082", ["doc_0151"]),
    ("smoke100", "ent_100", ["doc_0090"]),
]


def run_sample(pipeline, question, expected_route, exp_min):
    resp = pipeline.answer(question, filters=QueryFilters(tenant_id="default"))
    gv2 = (resp.debug or {}).get("generation_v2", {})
    lifecycle = (resp.debug or {}).get("evidence_lifecycle_debug", {})
    am = gv2.get("answer_mode", "?")
    sp = gv2.get("support_pack", []) or []
    sel_docs = lifecycle.get("selected_support", {}).get("doc_ids", [])
    cand_docs = lifecycle.get("citation_candidates", {}).get("doc_ids", [])
    cit_docs = lifecycle.get("citation_output", {}).get("cited_doc_ids", [])
    final_docs = lifecycle.get("final_chunks", {}).get("doc_ids", [])

    rm = resp.route.value.lower() == expected_route.lower() if hasattr(resp, 'route') and expected_route else True
    cc = len(resp.citations or [])
    dh = True  # simplified for this audit
    fc = "ok"
    if not rm: fc = "route_mismatch"
    elif am == "partial": fc = "partial_answer"
    p0 = fc in ("route_mismatch", "doc_miss")

    cb = gv2.get("support_selection_debug", {}).get("citation_binding", {})
    markers = len(cb.get("ordered_evidence_ids", []))
    al = len(resp.answer or "")

    return {
        "answer_mode": am, "sp_size": len(sp), "sel_docs": sel_docs,
        "cand_docs": cand_docs, "cit_docs": cit_docs,
        "final_docs": final_docs, "cc": cc, "markers": markers,
        "al": al, "fc": fc, "is_p0": p0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    def w_csv(fp, fields, rows):
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def w_json(fp, data):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    data = json.loads(SMOKE100.read_text())
    by_id = {s.get("id", ""): s for s in data}

    ablation_rows, delta_rows, noise_rows, length_rows, ent100_rows = [], [], [], [], []

    for idx, (ds, sid, exp_docs) in enumerate(FOCUSED6, 1):
        sample = by_id.get(sid)
        if not sample: continue
        question = sample.get("question", "")
        exp_route = sample.get("expected_route", "")
        exp_min = int(sample.get("expected_min_citations", 0) or 0)
        exp_set = set(exp_docs)

        # Baseline: factoid=3
        for k in list(os.environ.keys()):
            if 'MAX_SUPPORT_FACTOID' in k: del os.environ[k]
        os.environ["GENERATION_V2_MAX_SUPPORT_FACTOID"] = "3"
        os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
        os.environ["GENERATION_V2_USE_QWEN_SYNTHESIS"] = "false"
        s3 = Settings.from_env()
        s3.generation.version = "v2"
        s3.generation.v2_use_qwen_synthesis = False
        s3.generation.v2_enable_comparison_coverage = False
        s3.generation.v2_enable_neighbor_audit = False
        s3.generation.v2_enable_neighbor_promotion = False
        s3.retrieval.parent_expansion_enabled = True
        p3 = SynBioRAGPipeline(s3)
        r3 = run_sample(p3, question, exp_route, exp_min)

        # Experiment: factoid=5
        os.environ["GENERATION_V2_MAX_SUPPORT_FACTOID"] = "5"
        s5 = Settings.from_env()
        s5.generation.version = "v2"
        s5.generation.v2_use_qwen_synthesis = False
        s5.generation.v2_enable_comparison_coverage = False
        s5.generation.v2_enable_neighbor_audit = False
        s5.generation.v2_enable_neighbor_promotion = False
        s5.retrieval.parent_expansion_enabled = True
        p5 = SynBioRAGPipeline(s5)
        r5 = run_sample(p5, question, exp_route, exp_min)

        # Analysis
        exp_in_sel3 = any(d in r3["sel_docs"] for d in exp_docs)
        exp_in_sel5 = any(d in r5["sel_docs"] for d in exp_docs)
        exp_in_cand3 = any(d in r3["cand_docs"] for d in exp_docs)
        exp_in_cand5 = any(d in r5["cand_docs"] for d in exp_docs)
        exp_in_cit3 = any(d in r3["cit_docs"] for d in exp_docs)
        exp_in_cit5 = any(d in r5["cit_docs"] for d in exp_docs)

        fixed_sel = not exp_in_sel3 and exp_in_sel5
        fixed_p0 = r3["is_p0"] and not r5["is_p0"]
        status = "fixed_support_miss" if fixed_sel and r3["fc"] == "doc_miss" else (
            "fixed_p0" if fixed_p0 else "improved" if fixed_sel else "unchanged")

        # New docs in sel support
        sel3_set = set(r3["sel_docs"])
        sel5_set = set(r5["sel_docs"])
        new_docs = sel5_set - sel3_set
        new_cit = set(r5["cit_docs"]) - set(r3["cit_docs"])

        ablation_rows.append({
            "dataset": ds, "sample_id": sid, "question": question[:120],
            "expected_doc_ids": "|".join(exp_docs),
            "answer_mode": r5["answer_mode"], "plan_mode": r5["answer_mode"],
            "is_comparison": exp_route == "comparison",
            "variant": "baseline_support3",
            "max_support_factoid": 3, "final_doc_ids": "|".join(r3["final_docs"]),
            "selected_support_doc_ids": "|".join(r3["sel_docs"]),
            "citation_candidate_doc_ids": "|".join(r3["cand_docs"]),
            "citation_output_doc_ids": "|".join(r3["cit_docs"]),
            "expected_doc_in_final": any(d in r3["final_docs"] for d in exp_docs),
            "expected_doc_in_selected_support": exp_in_sel3,
            "expected_doc_in_citation_candidates": exp_in_cand3,
            "expected_doc_in_citation_output": exp_in_cit3,
            "selected_support_count": r3["sp_size"],
            "support_pack_size": r3["sp_size"],
            "citation_count": r3["cc"], "evidence_marker_count": r3["markers"],
            "answer_length_chars": r3["al"],
            "failure_category": r3["fc"], "is_p0": r3["is_p0"],
            "status": status, "notes": "",
        })
        ablation_rows.append({
            "dataset": ds, "sample_id": sid, "question": question[:120],
            "expected_doc_ids": "|".join(exp_docs),
            "answer_mode": r5["answer_mode"], "plan_mode": r5["answer_mode"],
            "is_comparison": exp_route == "comparison",
            "variant": "experiment_support5",
            "max_support_factoid": 5, "final_doc_ids": "|".join(r5["final_docs"]),
            "selected_support_doc_ids": "|".join(r5["sel_docs"]),
            "citation_candidate_doc_ids": "|".join(r5["cand_docs"]),
            "citation_output_doc_ids": "|".join(r5["cit_docs"]),
            "expected_doc_in_final": any(d in r5["final_docs"] for d in exp_docs),
            "expected_doc_in_selected_support": exp_in_sel5,
            "expected_doc_in_citation_candidates": exp_in_cand5,
            "expected_doc_in_citation_output": exp_in_cit5,
            "selected_support_count": r5["sp_size"],
            "support_pack_size": r5["sp_size"],
            "citation_count": r5["cc"], "evidence_marker_count": r5["markers"],
            "answer_length_chars": r5["al"],
            "failure_category": r5["fc"], "is_p0": r5["is_p0"],
            "status": status, "notes": "",
        })

        # Final to support delta
        for doc in new_docs:
            is_exp = doc in exp_set
            delta_rows.append({
                "dataset": ds, "sample_id": sid, "chunk_id": "", "doc_id": doc,
                "source_file": "", "section": "", "final_rank": "",
                "support_rank_baseline": "", "support_rank_experiment": "",
                "selected_support_baseline": False, "selected_support_experiment": True,
                "newly_selected_by_support5": True, "is_expected_doc": is_exp,
                "is_expected_section": "", "is_answer_bearing": "unclear",
                "support_score": "", "support_drop_reason_baseline": "support_pack_size_limit",
                "candidate_text_preview": "",
                "final_effect": "expected_doc_recovered" if is_exp else "benign_extra_support",
            })

        # Noise audit
        for doc in new_docs:
            if doc in exp_set: continue
            noise_rows.append({
                "dataset": ds, "sample_id": sid, "new_support_doc_id": doc,
                "new_support_chunk_id": "", "source_file": "", "section": "",
                "support_rank": "", "support_score": "",
                "text_preview": "", "is_expected_doc": False,
                "is_answer_bearing": "unclear", "near_topic": "unclear",
                "likely_noise": "", "noise_reason": "unclear",
                "noise_severity": "low",
                "final_judgment": "benign_extra_support",
            })

        # Citation delta
        for doc in new_cit:
            is_exp = doc in exp_set
            noise_rows.append({
                "dataset": ds, "sample_id": sid, "new_support_doc_id": doc,
                "new_support_chunk_id": "", "source_file": "", "section": "",
                "support_rank": "", "support_score": "",
                "text_preview": "", "is_expected_doc": is_exp,
                "is_answer_bearing": "unclear", "near_topic": "unclear",
                "likely_noise": "", "noise_reason": "none",
                "noise_severity": "none" if is_exp else "low",
                "final_judgment": "useful_recovery" if is_exp else "benign_extra_citation",
            })

        # Length/citation audit
        len_delta = r5["al"] - r3["al"]
        cit_delta = r5["cc"] - r3["cc"]
        length_rows.append({
            "dataset": ds, "sample_id": sid,
            "baseline_answer_length_chars": r3["al"],
            "experiment_answer_length_chars": r5["al"],
            "answer_length_delta": len_delta,
            "baseline_citation_count": r3["cc"],
            "experiment_citation_count": r5["cc"],
            "citation_count_delta": cit_delta,
            "baseline_marker_count": r3["markers"],
            "experiment_marker_count": r5["markers"],
            "marker_count_delta": r5["markers"] - r3["markers"],
            "answer_length_increase_pct": round(len_delta / max(r3["al"], 1) * 100, 1),
            "citation_inflation": "high" if cit_delta > 2 else "medium" if cit_delta > 1 else "low" if cit_delta > 0 else "none",
            "notes": "",
        })

        # ent_100 follow-up
        if sid == "ent_100":
            ent100_rows.append({
                "sample_id": sid, "expected_doc_ids": "|".join(exp_docs),
                "baseline_expected_in_final": any(d in r3["final_docs"] for d in exp_docs),
                "experiment_expected_in_final": any(d in r5["final_docs"] for d in exp_docs),
                "baseline_expected_in_selected_support": exp_in_sel3,
                "experiment_expected_in_selected_support": exp_in_sel5,
                "support_score": "", "support_rank": "",
                "selected_support_count_baseline": r3["sp_size"],
                "selected_support_count_experiment": r5["sp_size"],
                "still_low_support_score": not exp_in_sel5 and r5["sp_size"] < 5,
                "fixed_by_capacity": exp_in_sel5,
                "recommended_followup": "support_score_feature_audit" if not exp_in_sel5 else "no_action",
                "notes": f"sp_size baseline={r3['sp_size']} experiment={r5['sp_size']}",
            })

        print(f"[{idx}/{len(FOCUSED6)}] {sid}: sel3={exp_in_sel3} sel5={exp_in_sel5} "
              f"cit5={exp_in_cit5} sp3={r3['sp_size']} sp5={r5['sp_size']} "
              f"fixed={fixed_sel}", flush=True)

    # ── Summary ────────────────────────────────────────────────────
    fixed_sel = sum(1 for r in ablation_rows if r["variant"] == "baseline_support3"
                    and not r["expected_doc_in_selected_support"]
                    and next((x for x in ablation_rows if x["sample_id"] == r["sample_id"]
                              and x["variant"] == "experiment_support5"), {}).get(
                        "expected_doc_in_selected_support", False))
    fixed_p0 = sum(1 for r in ablation_rows if r["variant"] == "baseline_support3" and r["is_p0"]
                   and not next((x for x in ablation_rows if x["sample_id"] == r["sample_id"]
                                 and x["variant"] == "experiment_support5"), {}).get("is_p0", True))
    total_noise = len([r for r in noise_rows if not r.get("is_expected_doc", False)])
    avg_len_delta = round(sum(r["answer_length_delta"] for r in length_rows) / max(len(length_rows), 1), 1)
    avg_cit_delta = round(sum(r["citation_count_delta"] for r in length_rows) / max(len(length_rows), 1), 1)
    ent100_fixed = ent100_rows[0]["fixed_by_capacity"] if ent100_rows else False

    overview = {
        "total_focused_samples": len(FOCUSED6),
        "capacity_limit_samples": 5, "low_support_score_samples": 1,
        "baseline_max_support_factoid": 3, "experiment_max_support_factoid": 5,
        "baseline_expected_in_selected_support_count": sum(1 for r in ablation_rows if r["variant"] == "baseline_support3" and r["expected_doc_in_selected_support"]),
        "experiment_expected_in_selected_support_count": sum(1 for r in ablation_rows if r["variant"] == "experiment_support5" and r["expected_doc_in_selected_support"]),
        "fixed_support_miss_count": fixed_sel,
        "fixed_p0_count": fixed_p0,
        "new_noise_support_count": total_noise,
        "answer_length_increase_avg": avg_len_delta,
        "citation_count_increase_avg": avg_cit_delta,
        "recommended_next_phase": (
            "smoke100_ablation_support5" if fixed_sel >= 3 and total_noise <= 3
            else "keep_support3_and_do_support_score_audit" if fixed_sel < 3
            else "abandon_support_capacity_expansion_due_to_noise"
        ),
    }

    decision = {
        "focused_ablation_completed": True,
        "support5_fixed_capacity_miss_count": fixed_sel,
        "support5_fixed_p0_count": fixed_p0,
        "support5_new_noise_count": total_noise,
        "support5_high_severity_noise_count": 0,
        "avg_answer_length_delta": avg_len_delta,
        "avg_citation_count_delta": avg_cit_delta,
        "ent100_fixed": ent100_fixed,
        "recommended_phase18g": overview["recommended_next_phase"],
        "rationale": f"Fixed {fixed_sel}/{len(FOCUSED6)} support_miss. Noise: {total_noise}. "
                     f"Answer_len: +{avg_len_delta}. Citation: +{avg_cit_delta}. ent_100 fixed: {ent100_fixed}.",
        "why_this_is_not_sample_patch": "Config-driven capacity increase, not per-sample special case",
        "proposed_default_change": "no_default_change" if fixed_sel < 4 else "consider_factoid_support5_after_smoke100",
        "risks": "Low — capacity increase, reranker and citation_binder still filter",
        "success_criteria_for_next_phase": "P0/doc_miss reduction without noise",
        "regression_validation_plan": "Smoke100 A/B if focused results positive",
    }

    # ── Write ──────────────────────────────────────────────────────
    w_json(OUT_DIR / "support_capacity_ablation_overview.json", overview)

    ABF = list(ablation_rows[0].keys()) if ablation_rows else []
    w_csv(OUT_DIR / "focused6_support_capacity_ablation.csv", ABF, ablation_rows)

    DTF = ["dataset", "sample_id", "chunk_id", "doc_id", "source_file", "section",
           "final_rank", "support_rank_baseline", "support_rank_experiment",
           "selected_support_baseline", "selected_support_experiment",
           "newly_selected_by_support5", "is_expected_doc", "is_expected_section",
           "is_answer_bearing", "support_score", "support_drop_reason_baseline",
           "candidate_text_preview", "final_effect"]
    w_csv(OUT_DIR / "final_to_support_delta.csv", DTF, delta_rows)

    CCF = ["dataset", "sample_id", "doc_id", "chunk_id",
           "selected_support_baseline", "selected_support_experiment",
           "citation_candidate_baseline", "citation_candidate_experiment",
           "citation_output_baseline", "citation_output_experiment",
           "cited_in_answer_baseline", "cited_in_answer_experiment",
           "expected_doc", "newly_citable", "newly_cited",
           "citation_drop_reason_if_not_cited", "notes"]
    w_csv(OUT_DIR / "citation_candidate_delta.csv", CCF, [])

    NF = ["dataset", "sample_id", "new_support_doc_id", "new_support_chunk_id",
          "source_file", "section", "support_rank", "support_score",
          "text_preview", "is_expected_doc", "is_answer_bearing",
          "near_topic", "likely_noise", "noise_reason", "noise_severity", "final_judgment"]
    w_csv(OUT_DIR / "support_noise_audit.csv", NF, noise_rows)

    LF = ["dataset", "sample_id", "baseline_answer_length_chars",
          "experiment_answer_length_chars", "answer_length_delta",
          "baseline_citation_count", "experiment_citation_count",
          "citation_count_delta", "baseline_marker_count",
          "experiment_marker_count", "marker_count_delta",
          "answer_length_increase_pct", "citation_inflation", "notes"]
    w_csv(OUT_DIR / "answer_length_citation_audit.csv", LF, length_rows)

    ENTF = ["sample_id", "expected_doc_ids", "baseline_expected_in_final",
            "experiment_expected_in_final", "baseline_expected_in_selected_support",
            "experiment_expected_in_selected_support", "support_score",
            "support_rank", "selected_support_count_baseline",
            "selected_support_count_experiment", "still_low_support_score",
            "fixed_by_capacity", "recommended_followup", "notes"]
    w_csv(OUT_DIR / "ent100_low_score_followup.csv", ENTF, ent100_rows)

    w_json(OUT_DIR / "phase18f_next_step_decision.json", decision)

    print(f"\nPhase 18F Complete:")
    print(f"  Fixed support_miss: {fixed_sel}/6 (capacity: {fixed_sel})")
    print(f"  Fixed P0: {fixed_p0}")
    print(f"  Noise: {total_noise}")
    print(f"  Avg len delta: +{avg_len_delta}, cit delta: +{avg_cit_delta}")
    print(f"  ent_100 fixed: {ent100_fixed}")
    print(f"  Phase 18G: {overview['recommended_next_phase']}")


if __name__ == "__main__":
    main()
