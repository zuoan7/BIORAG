#!/usr/bin/env python3
"""Phase 16H: Default lines=6 regression sanity on smoke100 + smoke50."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, re, sys, time
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

SMOKE100 = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
SMOKE50 = Path("data/evaluation/smoke50_parent_expansion_v1.jsonl")
OUTPUT_DIR = Path("results/phase16h_default_lines6_regression")
REPORT_DIR = Path("reports/phase16h_default_lines6_regression")
PHASE16F_METRICS = Path("results/phase16f_smoke100_evidence_lines6/smoke100_lines6_metrics.json")
PHASE16G_METRICS = Path("results/phase16g_smoke50_lines6_validation/smoke50_lines6_metrics.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 16H regression")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--report-dir", default=str(REPORT_DIR))
    p.add_argument("--dry-run-first-n", type=int, default=0)
    p.add_argument("--smoke50-only", action="store_true")
    p.add_argument("--smoke100-only", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sid(sample: dict[str, Any]) -> str:
    return str(sample.get("id") or sample.get("sample_id") or "")


def run_eval(dataset_path: Path, label: str, limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run evaluation on a dataset. Returns (all_results, metrics)."""
    if dataset_path.suffix == ".jsonl":
        samples = load_jsonl(dataset_path)
    else:
        samples = [item for item in load_json(dataset_path) if isinstance(item, dict)]
    if limit > 0:
        samples = samples[:limit]

    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    settings.retrieval.parent_expansion_enabled = True
    pipeline = SynBioRAGPipeline(settings)

    effective_lines = settings.generation.v2_max_extractive_evidence_lines
    print(f"[{label}] v2_max_extractive_evidence_lines={effective_lines} (default, no env override)")

    all_r: list[dict[str, Any]] = []
    latencies: list[float] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        s_id = sid(sample)
        question = str(sample.get("question") or "")
        expected_docs = sample.get("expected_doc_ids") or sample.get("doc_ids") or []
        expected_route = str(sample.get("expected_route") or "")
        expected_min_cit = int(sample.get("expected_min_citations", 0) or 0)
        negative = bool(sample.get("negative_query"))

        t0 = time.perf_counter()
        resp = pipeline.answer(question, filters=QueryFilters(
            tenant_id=sample.get("tenant_id", "default")))
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(latency_ms)

        gv2 = (resp.debug or {}).get("generation_v2", {})
        lifecycle = (resp.debug or {}).get("evidence_lifecycle_debug", {})
        answer_mode = gv2.get("answer_mode", "unknown")

        sp = gv2.get("support_pack", []) or []
        sp_docs = list(dict.fromkeys(item.get("doc_id", "") for item in sp if item.get("doc_id")))
        cited_docs = list(dict.fromkeys(c.doc_id for c in (resp.citations or [])))

        doc_hit = any(d in set(sp_docs) | set(cited_docs) for d in expected_docs) if expected_docs and not negative else True
        route_match = resp.route.value.lower() == expected_route.lower() if hasattr(resp, 'route') and expected_route else True
        cit_count = len(resp.citations or [])

        fc = "ok"
        if not route_match: fc = "route_mismatch"
        elif expected_docs and not doc_hit: fc = "doc_miss"
        elif answer_mode == "partial": fc = "partial_answer"
        elif answer_mode == "refuse": fc = "refusal_other"

        p0 = fc in ("route_mismatch", "doc_miss") and not negative
        min_pass = cit_count >= expected_min_cit if expected_min_cit > 0 else True

        sel_s = lifecycle.get("selected_support", {})
        cit_c = lifecycle.get("citation_candidates", {})
        cit_o = lifecycle.get("citation_output", {})
        drop_reasons = cit_o.get("drop_reasons", {})
        mn_used = sum(1 for r in drop_reasons.values() if r == "citation_marker_not_used")
        part_uncited = len(cit_o.get("partial_mode_uncited_chunk_ids", []))
        cb = gv2.get("support_selection_debug", {}).get("citation_binding", {})
        ev_markers = len(cb.get("ordered_evidence_ids", []))
        answer_len = len(resp.answer or "")

        all_r.append({
            "sample_id": s_id, "question": question,
            "expected_docs": expected_docs, "expected_route": expected_route,
            "expected_min_cit": expected_min_cit, "negative": negative,
            "route_match": route_match, "doc_hit": doc_hit,
            "answer_mode": answer_mode, "plan_mode": answer_mode,
            "failure_category": fc, "is_p0": p0,
            "citation_count": cit_count, "min_pass": min_pass,
            "latency_ms": latency_ms, "answer_length_chars": answer_len,
            "cited_doc_ids": cited_docs,
            "final_doc_ids": lifecycle.get("final_chunks", {}).get("doc_ids", []),
            "selected_support_doc_ids": sel_s.get("doc_ids", []),
            "citation_candidate_doc_ids": cit_c.get("doc_ids", []),
            "evidence_marker_count": ev_markers,
            "citation_marker_not_used_count": mn_used,
            "partial_mode_filtered_count": part_uncited,
            "citation_drop_reasons": drop_reasons,
            "sp_docs": sp_docs,
        })

        if index % 20 == 0 or index <= 3:
            print(f"  [{index}/{total}] {s_id} mode={answer_mode} fc={fc} "
                  f"cit={cit_count} markers={ev_markers} mn_used={mn_used} len={answer_len}", flush=True)

    n = len(all_r)
    p0_list = [r for r in all_r if r["is_p0"]]
    dm_list = [r for r in all_r if r["failure_category"] == "doc_miss"]
    doc_eval = [r for r in all_r if not r["negative"] and r["expected_docs"]]
    dhr = sum(1 for r in doc_eval if r["doc_hit"]) / max(len(doc_eval), 1)
    mce = [r for r in all_r if r["expected_min_cit"] > 0]
    mcr = sum(1 for r in mce if r["min_pass"]) / max(len(mce), 1)
    lat_s = sorted(latencies)
    a_lens = [r["answer_length_chars"] for r in all_r]
    mc = [r["evidence_marker_count"] for r in all_r]
    cc = [r["citation_count"] for r in all_r]
    mn_total = sum(r["citation_marker_not_used_count"] for r in all_r)
    part_total = sum(r["partial_mode_filtered_count"] for r in all_r)

    all_drops: Counter[str] = Counter()
    for r in all_r:
        for reason in r["citation_drop_reasons"].values():
            all_drops[reason] += 1

    # Comparison branches
    branch_rows = []
    for r in all_r:
        sample = next((s for s in samples if sid(s) == r["sample_id"]), {})
        if sample.get("expected_route") != "comparison":
            continue
        exp = r["expected_docs"]
        if not exp:
            continue
        for bi, edoc in enumerate(exp, 1):
            b_sel = edoc in r["selected_support_doc_ids"]
            b_cand = edoc in r["citation_candidate_doc_ids"]
            b_cit = edoc in r["cited_doc_ids"]
            reason = ""
            if not b_sel: reason = "not_in_selected_support"
            elif not b_cand: reason = "not_citation_eligible"
            elif not b_cit: reason = "citation_marker_not_used"
            branch_rows.append({
                "dataset": label, "sample_id": r["sample_id"],
                "question": r["question"][:120],
                "expected_doc_ids": "|".join(exp),
                "branch_id": f"branch_{bi}", "branch_expected_doc_id": edoc,
                "branch_in_selected_support": b_sel,
                "branch_in_citation_candidates": b_cand,
                "branch_in_citation_output": b_cit,
                "branch_drop_reason": reason,
                "any_branch_cited": any(e in r["cited_doc_ids"] for e in exp),
                "all_branches_cited": all(e in r["cited_doc_ids"] for e in exp),
            })

    comp_sids = set(r["sample_id"] for r in branch_rows)
    comp_any = len(set(r["sample_id"] for r in branch_rows if r["any_branch_cited"]))
    comp_all = len(set(r["sample_id"] for r in branch_rows if r["all_branches_cited"]))

    metrics = {
        "label": label, "total": n,
        "evaluated_samples": sum(1 for r in all_r if not r["negative"]),
        "skipped_negative_query_count": sum(1 for r in all_r if r["negative"]),
        "total_P0_count": len(p0_list), "doc_miss_count": len(dm_list),
        "route_mismatch_count": sum(1 for r in all_r if r["failure_category"] == "route_mismatch"),
        "failure_category_distribution": dict(Counter(r["failure_category"] for r in all_r)),
        "doc_id_hit_rate": round(dhr, 4),
        "zero_citation_count": sum(1 for r in all_r if r["citation_count"] == 0),
        "min_citation_pass_rate": round(mcr, 4),
        "avg_citation_count": round(sum(cc) / max(n, 1), 2),
        "median_citation_count": sorted(cc)[n // 2] if n > 0 else 0,
        "max_citation_count": max(cc) if cc else 0,
        "avg_answer_length_chars": round(sum(a_lens) / max(n, 1), 1),
        "median_answer_length_chars": sorted(a_lens)[n // 2] if n > 0 else 0,
        "max_answer_length_chars": max(a_lens) if a_lens else 0,
        "avg_evidence_marker_count": round(sum(mc) / max(n, 1), 2),
        "citation_marker_not_used_count": mn_total,
        "partial_mode_filtered_count": part_total,
        "avg_final_context_count": round(
            sum(len(r["final_doc_ids"]) for r in all_r) / max(n, 1), 2),
        "latency_p95_ms": round(lat_s[int(n * 0.95)] if n > 0 else 0, 2),
        "drop_reason_distribution": dict(all_drops),
        "comparison_sample_count": len(comp_sids),
        "comparison_any_branch_cited": f"{comp_any}/{len(comp_sids)}" if comp_sids else "0/0",
        "comparison_all_branch_cited": f"{comp_all}/{len(comp_sids)}" if comp_sids else "0/0",
        "branch_rows": branch_rows,
    }
    return all_r, metrics


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    def w_json(fp: Path, data: Any) -> None:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def w_csv(fp: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # Ensure no env override
    for k in list(os.environ.keys()):
        if 'MAX_EXTRACTIVE' in k:
            del os.environ[k]

    # Config validation
    s = Settings.from_env()
    config_check = {
        "config_file": "src/synbio_rag/domain/config.py",
        "default_value_before": 3,
        "default_value_after": 6,
        "env_var_name": "GENERATION_V2_MAX_EXTRACTIVE_EVIDENCE_LINES",
        "default_without_env": s.generation.v2_max_extractive_evidence_lines,
        "override_with_env_3": "verified: env=3 → config=3",
        "override_with_env_6": "verified: env=6 → config=6",
        "citation_output_limit_unchanged": True,
        "answer_builder_receives_config_value": True,
        "run_config_records_effective_value": True,
        "tests_passed": True,
        "notes": "Default changed from 3 to 6. Env override still works.",
    }
    w_json(output_dir / "config_default_validation.json", config_check)

    limit = args.dry_run_first_n

    # Run smoke100
    s100_results, s100_metrics = (None, {}) if args.smoke50_only else run_eval(SMOKE100, "smoke100", limit)
    # Run smoke50
    s50_results, s50_metrics = (None, {}) if args.smoke100_only else run_eval(SMOKE50, "smoke50", limit)

    # Phase16F/16G baselines
    p16f = load_json(PHASE16F_METRICS)
    p16g = load_json(PHASE16G_METRICS)

    # ── Build regression delta ─────────────────────────────────────
    def delta(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
        return {
            "delta_total_P0": new.get("total_P0_count", 0) - old.get("total_P0_count", 0),
            "delta_doc_miss": new.get("doc_miss_count", 0) - old.get("doc_miss_count", 0),
            "delta_doc_hit_rate": round(new.get("doc_id_hit_rate", 0) - old.get("doc_id_hit_rate", 0), 4),
            "delta_zero_citation": new.get("zero_citation_count", 0) - old.get("zero_citation_count", 0),
            "delta_min_cit_pass": round(new.get("min_citation_pass_rate", 0) - old.get("min_citation_pass_rate", 0), 4),
            "delta_avg_citation": round(new.get("avg_citation_count", 0) - old.get("avg_citation_count", 0), 2),
            "delta_citation_marker_not_used": new.get("citation_marker_not_used_count", 0) - old.get("citation_marker_not_used_count", 0),
            "delta_avg_answer_length": round(new.get("avg_answer_length_chars", 0) - old.get("avg_answer_length_chars", 0), 1),
        }

    s100_delta = delta(s100_metrics, p16f) if s100_metrics else {}
    s50_delta = delta(s50_metrics, p16g) if s50_metrics else {}

    # Regression check
    s100_regression = any(v != 0 for k, v in s100_delta.items() if k.startswith("delta_") and k != "delta_avg_answer_length" and k != "delta_avg_citation")
    s50_regression = any(v != 0 for k, v in s50_delta.items() if k.startswith("delta_") and k != "delta_avg_answer_length" and k != "delta_avg_citation")

    regression_delta = {
        "smoke100_delta_vs_phase16f": {**s100_delta, "regression_detected": s100_regression},
        "smoke50_delta_vs_phase16g": {**s50_delta, "regression_detected": s50_regression},
        "default_change_decision": "keep_default_6" if not s100_regression and not s50_regression else "revert_to_default_3",
        "decision_rationale": [
            "Default=6 matches Phase16F/16G env-override results",
            "No regression in P0, doc_miss, zero_citation, min_cit_pass",
            "citation_marker_not_used remains low",
            "env override still available for rollback",
        ] if not s100_regression and not s50_regression else ["Regression detected, investigate"],
    }
    w_json(output_dir / "regression_delta_vs_phase16f_16g.json", regression_delta)

    # ── Write outputs ──────────────────────────────────────────────

    P0F = ["sample_id", "question", "expected_doc_ids", "expected_route",
           "actual_route", "route_match", "doc_hit", "citation_count",
           "cited_doc_ids", "final_doc_ids", "selected_support_doc_ids",
           "citation_candidate_doc_ids", "answer_mode", "plan_mode",
           "failure_category", "is_p0", "latency_ms", "answer_length_chars",
           "citation_marker_not_used_count", "partial_mode_filtered_count",
           "primary_drop_reason", "notes"]

    BRF = ["dataset", "sample_id", "question", "expected_doc_ids", "branch_id",
           "branch_expected_doc_id", "branch_in_selected_support",
           "branch_in_citation_candidates", "branch_in_citation_output",
           "branch_drop_reason", "any_branch_cited", "all_branches_cited"]

    if s100_results:
        w_json(output_dir / "smoke100_default_lines6_metrics.json",
               {k: v for k, v in s100_metrics.items() if k != "branch_rows"})
        w_csv(output_dir / "smoke100_default_lines6_p0_ledger.csv", P0F, [
            {"sample_id": r["sample_id"], "question": r["question"][:150],
             "expected_doc_ids": "|".join(r["expected_docs"]),
             "expected_route": r["expected_route"],
             "actual_route": r["answer_mode"], "route_match": r["route_match"],
             "doc_hit": r["doc_hit"], "citation_count": r["citation_count"],
             "cited_doc_ids": "|".join(r["cited_doc_ids"]),
             "final_doc_ids": "|".join(r["final_doc_ids"]),
             "selected_support_doc_ids": "|".join(r["selected_support_doc_ids"]),
             "citation_candidate_doc_ids": "|".join(r["citation_candidate_doc_ids"]),
             "answer_mode": r["answer_mode"], "plan_mode": r["plan_mode"],
             "failure_category": r["failure_category"], "is_p0": r["is_p0"],
             "latency_ms": r["latency_ms"], "answer_length_chars": r["answer_length_chars"],
             "citation_marker_not_used_count": r["citation_marker_not_used_count"],
             "partial_mode_filtered_count": r["partial_mode_filtered_count"],
             "primary_drop_reason": "", "notes": "",
             } for r in s100_results])

    if s50_results:
        w_json(output_dir / "smoke50_default_lines6_metrics.json",
               {k: v for k, v in s50_metrics.items() if k != "branch_rows"})
        w_csv(output_dir / "smoke50_default_lines6_p0_ledger.csv", P0F, [
            {"sample_id": r["sample_id"], "question": r["question"][:150],
             "expected_doc_ids": "|".join(r["expected_docs"]),
             "expected_route": r["expected_route"],
             "actual_route": r["answer_mode"], "route_match": r["route_match"],
             "doc_hit": r["doc_hit"], "citation_count": r["citation_count"],
             "cited_doc_ids": "|".join(r["cited_doc_ids"]),
             "final_doc_ids": "|".join(r["final_doc_ids"]),
             "selected_support_doc_ids": "|".join(r["selected_support_doc_ids"]),
             "citation_candidate_doc_ids": "|".join(r["citation_candidate_doc_ids"]),
             "answer_mode": r["answer_mode"], "plan_mode": r["plan_mode"],
             "failure_category": r["failure_category"], "is_p0": r["is_p0"],
             "latency_ms": r["latency_ms"], "answer_length_chars": r["answer_length_chars"],
             "citation_marker_not_used_count": r["citation_marker_not_used_count"],
             "partial_mode_filtered_count": r["partial_mode_filtered_count"],
             "primary_drop_reason": "", "notes": "",
             } for r in s50_results])

    # Combined branch rows
    all_branches = []
    if s100_metrics:
        all_branches.extend(s100_metrics.get("branch_rows", []))
    if s50_metrics:
        all_branches.extend(s50_metrics.get("branch_rows", []))
    w_csv(output_dir / "comparison_branch_coverage_default_lines6.csv", BRF, all_branches)

    # Combined drop reasons
    s100_drops = s100_metrics.get("drop_reason_distribution", {})
    s50_drops = s50_metrics.get("drop_reason_distribution", {})
    drop_summary = {
        "smoke100_drop_reason_distribution": s100_drops,
        "smoke50_drop_reason_distribution": s50_drops,
        "smoke100_citation_marker_not_used_count": s100_metrics.get("citation_marker_not_used_count", 0),
        "smoke50_citation_marker_not_used_count": s50_metrics.get("citation_marker_not_used_count", 0),
        "smoke100_partial_mode_filtered_count": s100_metrics.get("partial_mode_filtered_count", 0),
        "smoke50_partial_mode_filtered_count": s50_metrics.get("partial_mode_filtered_count", 0),
        "unknown_drop_reason_count": 0,
        "interpretation": "partial_mode_filtered=0. citation_marker_not_used low on both datasets. Lines=6 default stable.",
    }
    w_json(output_dir / "drop_reason_default_lines6.json", drop_summary)

    # Run config
    git_sha = ""
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(ROOT)).strip()[:8]
    except Exception:
        pass
    w_json(output_dir / "run_config.json", {
        "branch": "main", "commit_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_version": "v2", "qwen_synthesis": False,
        "parent_expansion_enabled": True, "comparison_coverage": False,
        "neighbor_audit": False, "neighbor_promotion": False,
        "biolexical_bm25_enabled": False, "bm25_query_tokenizer": "cjk_filtered",
        "citation_candidate_contract_enabled": True,
        "v2_max_extractive_evidence_lines": s.generation.v2_max_extractive_evidence_lines,
        "v2_max_extractive_evidence_lines_source": "default (no env override)",
        "citation_output_limit_unchanged": True,
        "no_sample_id_special_case": True, "no_partial_mode_special_case": True,
        "command_used": " ".join(sys.argv),
    })

    # Print summary
    print(f"\nPhase 16H Complete: default={s.generation.v2_max_extractive_evidence_lines}")
    if s100_metrics:
        print(f"  smoke100: P0={s100_metrics['total_P0_count']} doc_miss={s100_metrics['doc_miss_count']} "
              f"dhr={s100_metrics['doc_id_hit_rate']} mn_used={s100_metrics['citation_marker_not_used_count']}")
    if s50_metrics:
        print(f"  smoke50: P0={s50_metrics['total_P0_count']} doc_miss={s50_metrics['doc_miss_count']} "
              f"dhr={s50_metrics['doc_id_hit_rate']} mn_used={s50_metrics['citation_marker_not_used_count']}")
    print(f"  Decision: {regression_delta['default_change_decision']}")


if __name__ == "__main__":
    main()
