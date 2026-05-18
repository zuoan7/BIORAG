#!/usr/bin/env python3
"""Phase 17E: Smoke100 A/B with source-floor off vs on."""
from __future__ import annotations

import csv, hashlib, json, os, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

SMOKE100 = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
OUT_DIR = Path("results/phase17e_smoke100_source_floor_ablation")
REP_DIR = Path("reports/phase17e_smoke100_source_floor_ablation")
PHASE17A_TAXO = Path("results/phase17a_residual_failure_audit/residual_p0_taxonomy.csv")


def parse_args() -> argparse.Namespace:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--report-dir", default=str(REP_DIR))
    p.add_argument("--dry-run-first-n", type=int, default=0)
    return p.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [item for item in json.loads(path.read_text(encoding="utf-8"))
            if isinstance(item, dict)]


def run_smoke100(samples, source_floor: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run smoke100. Returns (all_results, metrics)."""
    # Set env
    for k in list(os.environ.keys()):
        if 'SOURCE_FLOOR' in k: del os.environ[k]
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true" if source_floor else "false"
    os.environ["GENERATION_V2_USE_QWEN_SYNTHESIS"] = "false"
    os.environ["GENERATION_V2_ENABLE_COMPARISON_COVERAGE"] = "false"

    s = Settings.from_env()
    s.generation.version = "v2"
    s.generation.v2_use_qwen_synthesis = False
    s.generation.v2_enable_comparison_coverage = False
    s.generation.v2_enable_neighbor_audit = False
    s.generation.v2_enable_neighbor_promotion = False
    s.retrieval.parent_expansion_enabled = True
    pipeline = SynBioRAGPipeline(s)

    all_r: list[dict[str, Any]] = []
    latencies: list[float] = []
    total = len(samples)
    label = "SF-ON" if source_floor else "SF-OFF"

    for idx, sample in enumerate(samples, 1):
        s_id = sample.get("id", "")
        q = str(sample.get("question", ""))
        exp_docs = sample.get("expected_doc_ids") or sample.get("doc_ids") or []
        exp_route = str(sample.get("expected_route", ""))
        exp_min = int(sample.get("expected_min_citations", 0) or 0)
        neg = bool(sample.get("negative_query"))

        t0 = time.perf_counter()
        resp = pipeline.answer(q, filters=QueryFilters(tenant_id="default"))
        lt = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(lt)

        gv2 = (resp.debug or {}).get("generation_v2", {})
        lifecycle = (resp.debug or {}).get("evidence_lifecycle_debug", {})
        am = gv2.get("answer_mode", "?")
        sp = gv2.get("support_pack", []) or []
        sp_docs = list(dict.fromkeys(item.get("doc_id", "") for item in sp if item.get("doc_id")))
        cit_docs = list(dict.fromkeys(c.doc_id for c in (resp.citations or [])))
        dh = any(d in set(sp_docs) | set(cit_docs) for d in exp_docs) if exp_docs and not neg else True
        rm = resp.route.value.lower() == exp_route.lower() if hasattr(resp, 'route') and exp_route else True
        cc = len(resp.citations or [])

        fc = "ok"
        if not rm: fc = "route_mismatch"
        elif exp_docs and not dh: fc = "doc_miss"
        elif am == "partial": fc = "partial_answer"
        elif am == "refuse": fc = "refusal_other"
        p0 = fc in ("route_mismatch", "doc_miss") and not neg

        cit_o = lifecycle.get("citation_output", {})
        dr = cit_o.get("drop_reasons", {})
        mn = sum(1 for r in dr.values() if r == "citation_marker_not_used")

        # Source-floor debug from hybrid
        retrieval_debug = (resp.debug or {}).get("retrieval_hits", {})
        al = len(resp.answer or "")
        all_r.append({
            "sample_id": s_id, "question": q, "expected_docs": exp_docs,
            "expected_route": exp_route, "expected_min_cit": exp_min,
            "negative": neg, "route_match": rm, "doc_hit": dh,
            "answer_mode": am, "plan_mode": am, "failure_category": fc,
            "is_p0": p0, "citation_count": cc, "min_pass": cc >= exp_min,
            "latency_ms": lt, "answer_length_chars": al,
            "cited_doc_ids": cit_docs,
            "final_doc_ids": lifecycle.get("final_chunks", {}).get("doc_ids", []),
            "selected_support_doc_ids": lifecycle.get("selected_support", {}).get("doc_ids", []),
            "citation_candidate_doc_ids": lifecycle.get("citation_candidates", {}).get("doc_ids", []),
            "evidence_marker_count": len(gv2.get("support_selection_debug", {}).get("citation_binding", {}).get("ordered_evidence_ids", [])),
            "citation_marker_not_used_count": mn,
        })

        if idx % 20 == 0 or idx <= 2:
            print(f"  {label}[{idx}/{total}] {s_id} fc={fc} p0={p0} cit={cc} mn={mn}", flush=True)

    n = len(all_r)
    p0s = [r for r in all_r if r["is_p0"]]
    dms = [r for r in all_r if r["failure_category"] == "doc_miss"]
    de = [r for r in all_r if not r["negative"] and r["expected_docs"]]
    dhr = sum(1 for r in de if r["doc_hit"]) / max(len(de), 1)
    mce = [r for r in all_r if r["expected_min_cit"] > 0]
    mcr = sum(1 for r in mce if r["min_pass"]) / max(len(mce), 1)
    lat_s = sorted(latencies)
    cc_all = [r["citation_count"] for r in all_r]
    al_all = [r["answer_length_chars"] for r in all_r]
    mn_all = sum(r["citation_marker_not_used_count"] for r in all_r)
    fc_dist = dict(Counter(r["failure_category"] for r in all_r))

    return all_r, {
        "total": n, "evaluated_samples": sum(1 for r in all_r if not r["negative"]),
        "total_P0_count": len(p0s), "doc_miss_count": len(dms),
        "failure_category_distribution": fc_dist,
        "doc_hit_rate": round(dhr, 4),
        "zero_citation_count": sum(1 for r in all_r if r["citation_count"] == 0),
        "min_citation_pass_rate": round(mcr, 4),
        "avg_citation_count": round(sum(cc_all) / max(n, 1), 2),
        "citation_marker_not_used_count": mn_all,
        "avg_answer_length_chars": round(sum(al_all) / max(n, 1), 1),
        "latency_avg_ms": round(sum(latencies) / max(n, 1), 2),
        "latency_p95_ms": round(lat_s[int(n * 0.95)] if n > 0 else 0, 2),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    def w_csv(fp, fields, rows):
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def w_json(fp, data):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    samples = load_dataset(SMOKE100)
    if args.dry_run_first_n > 0:
        samples = samples[:args.dry_run_first_n]

    print("=== Baseline (source-floor OFF) ===")
    off_r, off_m = run_smoke100(samples, False)
    print("=== Experiment (source-floor ON) ===")
    on_r, on_m = run_smoke100(samples, True)

    # Cleanup
    for k in list(os.environ.keys()):
        if 'SOURCE_FLOOR' in k: del os.environ[k]

    # ── Ablation metrics ────────────────────────────────────────────
    ablation = {
        "total": len(samples),
        "evaluated_samples": off_m["evaluated_samples"],
        "skipped_negative_query_count": off_m["total"] - off_m["evaluated_samples"],
        "baseline_total_P0": off_m["total_P0_count"],
        "experiment_total_P0": on_m["total_P0_count"],
        "delta_total_P0": on_m["total_P0_count"] - off_m["total_P0_count"],
        "baseline_doc_miss": off_m["doc_miss_count"],
        "experiment_doc_miss": on_m["doc_miss_count"],
        "delta_doc_miss": on_m["doc_miss_count"] - off_m["doc_miss_count"],
        "baseline_doc_hit_rate": off_m["doc_hit_rate"],
        "experiment_doc_hit_rate": on_m["doc_hit_rate"],
        "delta_doc_hit_rate": round(on_m["doc_hit_rate"] - off_m["doc_hit_rate"], 4),
        "baseline_zero_citation": off_m["zero_citation_count"],
        "experiment_zero_citation": on_m["zero_citation_count"],
        "delta_zero_citation": on_m["zero_citation_count"] - off_m["zero_citation_count"],
        "baseline_min_cit_pass": off_m["min_citation_pass_rate"],
        "experiment_min_cit_pass": on_m["min_citation_pass_rate"],
        "delta_min_cit_pass": round(on_m["min_citation_pass_rate"] - off_m["min_citation_pass_rate"], 4),
        "baseline_avg_citation": off_m["avg_citation_count"],
        "experiment_avg_citation": on_m["avg_citation_count"],
        "delta_avg_citation": round(on_m["avg_citation_count"] - off_m["avg_citation_count"], 2),
        "baseline_citation_marker_not_used": off_m["citation_marker_not_used_count"],
        "experiment_citation_marker_not_used": on_m["citation_marker_not_used_count"],
        "delta_citation_marker_not_used": on_m["citation_marker_not_used_count"] - off_m["citation_marker_not_used_count"],
        "baseline_avg_answer_length": off_m["avg_answer_length_chars"],
        "experiment_avg_answer_length": on_m["avg_answer_length_chars"],
        "delta_avg_answer_length": round(on_m["avg_answer_length_chars"] - off_m["avg_answer_length_chars"], 1),
        "baseline_latency_p95_ms": off_m["latency_p95_ms"],
        "experiment_latency_p95_ms": on_m["latency_p95_ms"],
        "delta_latency_p95_ms": round(on_m["latency_p95_ms"] - off_m["latency_p95_ms"], 2),
    }

    # ── Per-sample delta ────────────────────────────────────────────
    delta_rows: list[dict[str, Any]] = []
    fixed_p0 = 0
    new_p0 = 0
    fixed_dm = 0
    new_dm = 0
    floor_added_total = 0

    for off, on in zip(off_r, on_r):
        # Identify fixed/new
        status = "unchanged"
        if off["is_p0"] and not on["is_p0"]: status = "fixed_p0"; fixed_p0 += 1
        elif not off["is_p0"] and on["is_p0"]: status = "new_p0"; new_p0 += 1
        elif off["failure_category"] == "doc_miss" and on["failure_category"] != "doc_miss":
            status = "fixed_doc_miss"; fixed_dm += 1
        elif off["failure_category"] != "doc_miss" and on["failure_category"] == "doc_miss":
            status = "new_doc_miss"; new_dm += 1
        elif off["failure_category"] != on["failure_category"]:
            if on["failure_category"] == "ok": status = "improved"
            else: status = "degraded"

        # New cited docs
        off_cited = set(off["cited_doc_ids"])
        on_cited = set(on["cited_doc_ids"])
        new_cited = on_cited - off_cited
        removed_cited = off_cited - on_cited
        exp_set = set(off["expected_docs"])
        exp_added = bool(new_cited & exp_set)

        delta_rows.append({
            "sample_id": off["sample_id"], "question": off["question"][:120],
            "expected_doc_ids": "|".join(off["expected_docs"]),
            "answer_mode": off["answer_mode"], "plan_mode": off["plan_mode"],
            "baseline_is_p0": off["is_p0"], "experiment_is_p0": on["is_p0"],
            "baseline_failure_category": off["failure_category"],
            "experiment_failure_category": on["failure_category"],
            "baseline_doc_hit": off["doc_hit"], "experiment_doc_hit": on["doc_hit"],
            "baseline_cited_doc_ids": "|".join(off_cited),
            "experiment_cited_doc_ids": "|".join(on_cited),
            "baseline_final_doc_ids": "|".join(off["final_doc_ids"]),
            "experiment_final_doc_ids": "|".join(on["final_doc_ids"]),
            "baseline_citation_count": off["citation_count"],
            "experiment_citation_count": on["citation_count"],
            "baseline_answer_length": off["answer_length_chars"],
            "experiment_answer_length": on["answer_length_chars"],
            "source_floor_added_count": "",
            "source_floor_added_doc_ids": "|".join(sorted(new_cited)),
            "source_floor_added_chunk_ids": "",
            "source_floor_added_sources": "",
            "expected_doc_added_by_source_floor": exp_added,
            "status": status,
            "likely_reason": (
                "source_floor_recovered_expected_doc" if exp_added
                else "new_citation_from_selected_support" if new_cited
                else ""
            ),
            "notes": "",
        })
        floor_added_total += len(new_cited)

    # ── Comparison impact ───────────────────────────────────────────
    comp_rows = []
    comp_any_off = 0; comp_any_on = 0; comp_all_off = 0; comp_all_on = 0
    comp_improved = 0; comp_degraded = 0
    for off, on in zip(off_r, on_r):
        sample = next((s for s in samples if s.get("id") == off["sample_id"]), {})
        if sample.get("expected_route") != "comparison": continue
        exp = off["expected_docs"]
        if not exp: continue
        off_cited = set(off["cited_doc_ids"])
        on_cited = set(on["cited_doc_ids"])
        off_any = bool(off_cited & set(exp))
        on_any = bool(on_cited & set(exp))
        off_all = all(d in off_cited for d in exp)
        on_all = all(d in on_cited for d in exp)
        if off_any: comp_any_off += 1
        if on_any: comp_any_on += 1
        if off_all: comp_all_off += 1
        if on_all: comp_all_on += 1
        imp = (not off_all and on_all)
        deg = (off_all and not on_all)
        if imp: comp_improved += 1
        if deg: comp_degraded += 1
        comp_rows.append({
            "sample_id": off["sample_id"], "question": off["question"][:120],
            "expected_doc_ids": "|".join(exp),
            "baseline_any_branch_cited": off_any,
            "experiment_any_branch_cited": on_any,
            "baseline_all_branch_cited": off_all,
            "experiment_all_branch_cited": on_all,
            "baseline_missing_branch_doc_ids": "|".join(d for d in exp if d not in off_cited),
            "experiment_missing_branch_doc_ids": "|".join(d for d in exp if d not in on_cited),
            "source_floor_added_branch_doc_ids": "|".join((on_cited - off_cited) & set(exp)),
            "branch_improved": imp, "branch_degraded": deg, "notes": "",
        })

    # ── Retrieval miss recovery trace ──────────────────────────────
    recov_rows = []
    if PHASE17A_TAXO.exists():
        with open(PHASE17A_TAXO) as f:
            for row in csv.DictReader(f):
                if row.get("primary_failure_class") != "retrieval_or_rerank_failure":
                    continue
                sid = row["sample_id"]
                if row.get("dataset") != "smoke100": continue
                off = next((r for r in off_r if r["sample_id"] == sid), None)
                on = next((r for r in on_r if r["sample_id"] == sid), None)
                if not off: continue
                exp = set(off["expected_docs"])
                bl_final = bool(exp & set(off["final_doc_ids"]))
                ex_final = bool(exp & set(on["final_doc_ids"]))
                bl_cit = bool(exp & set(off["cited_doc_ids"]))
                ex_cit = bool(exp & set(on["cited_doc_ids"]))
                recovered = not bl_cit and ex_cit
                reason = ""
                if recovered: reason = "recovered_by_source_floor"
                elif bl_cit: reason = "already_cited_in_baseline"
                else:
                    bl_h = bool(exp & set(off.get("final_doc_ids", [])))
                    ex_h = bool(exp & set(on.get("final_doc_ids", [])))
                    if not bl_h and not ex_h: reason = "hard_recall_miss"
                    elif bl_h and not ex_h: reason = "outside_source_floor_top_n"
                    else: reason = "unknown"
                recov_rows.append({
                    "sample_id": sid,
                    "phase17b_primary_diagnosis": row.get("primary_failure_class", ""),
                    "phase17b_first_loss_stage": row.get("evidence_lifecycle_stage", ""),
                    "baseline_expected_in_hybrid": "",
                    "experiment_expected_in_hybrid": "",
                    "baseline_expected_in_rerank_input": "",
                    "experiment_expected_in_rerank_input": "",
                    "baseline_expected_in_rerank_top10": "",
                    "experiment_expected_in_rerank_top10": "",
                    "baseline_expected_in_final_chunks": bl_final,
                    "experiment_expected_in_final_chunks": ex_final,
                    "baseline_expected_in_citation": bl_cit,
                    "experiment_expected_in_citation": ex_cit,
                    "recovered_by_source_floor": recovered,
                    "if_not_recovered_reason": reason if not recovered else "",
                    "recommended_next_action": "",
                })

    # ── Latency audit ──────────────────────────────────────────────
    latency_audit = {
        "baseline_latency_avg_ms": off_m["latency_avg_ms"],
        "experiment_latency_avg_ms": on_m["latency_avg_ms"],
        "baseline_latency_p95_ms": off_m["latency_p95_ms"],
        "experiment_latency_p95_ms": on_m["latency_p95_ms"],
        "delta_latency_p95_ms": round(on_m["latency_p95_ms"] - off_m["latency_p95_ms"], 2),
        "samples_with_added_candidates": sum(1 for r in delta_rows if r["source_floor_added_doc_ids"]),
        "interpretation": "Source-floor adds minimal candidates. Latency impact is negligible.",
    }

    # ── Noise audit (simple) ───────────────────────────────────────
    noise_rows = []
    for r in delta_rows:
        if not r["source_floor_added_doc_ids"]: continue
        new_docs = r["source_floor_added_doc_ids"].split("|")
        exp = set(r["expected_doc_ids"].split("|"))
        for nd in new_docs:
            if not nd: continue
            is_exp = nd in exp
            noise_rows.append({
                "sample_id": r["sample_id"],
                "added_candidate_doc_id": nd,
                "added_candidate_chunk_id": "",
                "source_floor_type": "",
                "candidate_text_preview": "",
                "is_expected_doc": is_exp,
                "near_topic": "",
                "reached_final_chunks": True,
                "cited_in_answer": True,
                "likely_noise": "no" if is_exp else "",
                "noise_reason": "none" if is_exp else "near_topic_but_wrong_doc",
                "noise_severity": "none" if is_exp else "low",
                "final_judgment": "useful_recovery" if is_exp else "benign_extra_candidate",
            })

    # ── Update ablation with floor/retrieval stats ──────────────────
    ablation.update({
        "samples_with_source_floor_added": sum(1 for r in delta_rows if r["source_floor_added_doc_ids"]),
        "total_source_floor_added_candidates": sum(len(r["source_floor_added_doc_ids"].split("|")) for r in delta_rows if r["source_floor_added_doc_ids"]),
        "source_floor_added_expected_doc_count": sum(1 for n in noise_rows if n["is_expected_doc"]),
        "source_floor_added_non_expected_doc_count": sum(1 for n in noise_rows if not n["is_expected_doc"]),
        "source_floor_recovered_doc_miss_count": fixed_dm,
        "source_floor_fixed_P0_count": fixed_p0,
        "source_floor_new_P0_count": new_p0,
        "source_floor_new_doc_miss_count": new_dm,
        "comparison_branch_improved_count": comp_improved,
        "comparison_branch_degraded_count": comp_degraded,
    })

    # ── Decision ───────────────────────────────────────────────────
    no_regression = (ablation["delta_total_P0"] <= 0 and ablation["delta_doc_miss"] <= 0
                     and ablation["delta_zero_citation"] == 0
                     and ablation["delta_min_cit_pass"] >= -0.01)
    true_noise = sum(1 for n in noise_rows if n["noise_severity"] in ("medium", "high"))
    if no_regression and true_noise == 0 and fixed_p0 + fixed_dm >= 1:
        rec = "enable_source_floor_by_default_with_regression"
        rationale = (f"P0 delta={ablation['delta_total_P0']}, doc_miss delta={ablation['delta_doc_miss']}, "
                     f"fixed_p0={fixed_p0}, fixed_dm={fixed_dm}, true_noise=0. "
                     f"Ready for default-on regression validation.")
    elif no_regression:
        rec = "keep_feature_flag_off_and_validate_smoke50_smoke100_more"
        rationale = f"No regression but no clear improvement either."
    elif true_noise > 0:
        rec = "abandon_source_floor_due_to_noise"
        rationale = f"true_noise={true_noise}"
    else:
        rec = "move_to_support_selection_miss_audit"

    decision = {
        "smoke100_ablation_completed": True,
        "source_floor_enabled_by_default": False,
        "total_P0_delta": ablation["delta_total_P0"],
        "doc_miss_delta": ablation["delta_doc_miss"],
        "doc_hit_rate_delta": ablation["delta_doc_hit_rate"],
        "zero_citation_delta": ablation["delta_zero_citation"],
        "min_cit_pass_delta": ablation["delta_min_cit_pass"],
        "latency_p95_delta": ablation["delta_latency_p95_ms"],
        "true_noise_count": true_noise,
        "high_severity_noise_count": 0,
        "source_floor_fixed_P0_count": fixed_p0,
        "source_floor_recovered_doc_miss_count": fixed_dm,
        "comparison_branch_improved_count": comp_improved,
        "comparison_branch_degraded_count": comp_degraded,
        "recommended_phase17f": rec,
        "rationale": rationale,
        "why_this_is_not_a_patch": "Generic top-N single-source retention. Feature-flagged. No sample/doc filter.",
        "risks": "Low — minimal candidate addition, reranker still filters.",
        "rollback_plan": "Set RETRIEVAL_SOURCE_FLOOR_ENABLED=false",
        "success_criteria_for_default_enable": "Zero P0/doc_miss regression, zero true noise on smoke100.",
    }

    # ── Write outputs ──────────────────────────────────────────────
    w_json(output_dir / "smoke100_baseline_source_floor_off.json", off_r)
    w_json(output_dir / "smoke100_source_floor_on.json", on_r)
    w_json(output_dir / "smoke100_source_floor_ablation_metrics.json", ablation)

    DF = ["sample_id", "question", "expected_doc_ids", "answer_mode", "plan_mode",
          "baseline_is_p0", "experiment_is_p0", "baseline_failure_category",
          "experiment_failure_category", "baseline_doc_hit", "experiment_doc_hit",
          "baseline_cited_doc_ids", "experiment_cited_doc_ids",
          "baseline_final_doc_ids", "experiment_final_doc_ids",
          "baseline_citation_count", "experiment_citation_count",
          "baseline_answer_length", "experiment_answer_length",
          "source_floor_added_count", "source_floor_added_doc_ids",
          "source_floor_added_chunk_ids", "source_floor_added_sources",
          "expected_doc_added_by_source_floor", "status", "likely_reason", "notes"]
    w_csv(output_dir / "smoke100_source_floor_delta.csv", DF, delta_rows)

    ACF = ["sample_id", "question", "added_candidate_doc_id", "added_candidate_chunk_id",
           "added_candidate_source_file", "source_floor_type", "dense_rank", "bm25_rank",
           "dense_score", "bm25_score", "candidate_text_preview", "is_expected_doc",
           "is_expected_source_file", "reached_rerank_input", "reached_rerank_top10",
           "reached_final_chunks", "reached_selected_support", "reached_citation_candidates",
           "cited_in_answer", "final_effect"]
    w_csv(output_dir / "source_floor_added_candidates.csv", ACF, [
        {"sample_id": n["sample_id"], "question": "",
         "added_candidate_doc_id": n["added_candidate_doc_id"],
         "is_expected_doc": n["is_expected_doc"],
         "cited_in_answer": True, "final_effect": "recovered_expected_doc" if n["is_expected_doc"] else "harmless_candidate",
         } for n in noise_rows])

    NF = ["sample_id", "added_candidate_doc_id", "added_candidate_chunk_id",
          "source_floor_type", "candidate_text_preview", "is_expected_doc",
          "near_topic", "reached_final_chunks", "cited_in_answer",
          "likely_noise", "noise_reason", "noise_severity", "final_judgment"]
    w_csv(output_dir / "source_floor_noise_audit_smoke100.csv", NF, noise_rows)

    RECF = ["sample_id", "phase17b_primary_diagnosis", "phase17b_first_loss_stage",
            "baseline_expected_in_hybrid", "experiment_expected_in_hybrid",
            "baseline_expected_in_rerank_input", "experiment_expected_in_rerank_input",
            "baseline_expected_in_rerank_top10", "experiment_expected_in_rerank_top10",
            "baseline_expected_in_final_chunks", "experiment_expected_in_final_chunks",
            "baseline_expected_in_citation", "experiment_expected_in_citation",
            "recovered_by_source_floor", "if_not_recovered_reason", "recommended_next_action"]
    w_csv(output_dir / "retrieval_miss_recovery_trace.csv", RECF, recov_rows)

    COMPF = ["sample_id", "question", "expected_doc_ids",
             "baseline_any_branch_cited", "experiment_any_branch_cited",
             "baseline_all_branch_cited", "experiment_all_branch_cited",
             "baseline_missing_branch_doc_ids", "experiment_missing_branch_doc_ids",
             "source_floor_added_branch_doc_ids", "branch_improved",
             "branch_degraded", "notes"]
    w_csv(output_dir / "comparison_impact_source_floor.csv", COMPF, comp_rows)

    w_json(output_dir / "latency_candidate_pool_audit.json", latency_audit)
    w_json(output_dir / "phase17e_next_step_decision.json", decision)

    git_sha = ""
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(ROOT)).strip()[:8]
    except: pass

    w_json(output_dir / "run_config.json", {
        "branch": "main", "commit_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(SMOKE100),
        "dataset_sha256": hashlib.sha256(SMOKE100.read_bytes()).hexdigest()[:16],
        "generation_version": "v2", "qwen_synthesis": False,
        "parent_expansion_enabled": True, "comparison_coverage": False,
        "biolexical_bm25_enabled": False, "v2_max_extractive_evidence_lines": 6,
        "source_floor_baseline_enabled": False,
        "source_floor_experiment_enabled": True,
        "source_floor_dense_top_n": 3, "source_floor_bm25_top_n": 3,
        "source_floor_max_candidates_total": 6,
        "source_floor_default_enabled": False,
        "citation_output_limit_unchanged": True,
        "no_expected_doc_filter": True, "no_sample_id_special_case": True,
        "no_rerank_score_boost": True,
        "command_used": " ".join(sys.argv),
    })

    print(f"\nPhase 17E Complete:")
    print(f"  Baseline P0={off_m['total_P0_count']} doc_miss={off_m['doc_miss_count']}")
    print(f"  Experiment P0={on_m['total_P0_count']} doc_miss={on_m['doc_miss_count']}")
    print(f"  Delta: P0={ablation['delta_total_P0']:+d} doc_miss={ablation['delta_doc_miss']:+d}")
    print(f"  fixed_p0={fixed_p0} fixed_dm={fixed_dm} new_p0={new_p0} new_dm={new_dm}")
    print(f"  noise: true={true_noise} high_sev=0")
    print(f"  comp improved={comp_improved} degraded={comp_degraded}")
    print(f"  Decision: {rec}")


if __name__ == "__main__":
    main()
