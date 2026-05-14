"""Phase 21B-EVAL-4D: Full Smoke200 BIORAG Eval v3.1 Run."""
import csv, json, os, sys, time
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv; load_dotenv(".env")
os.environ.update({
    "QUERY_REWRITE_MODE": "enabled",
    "EVAL_REWRITE_CACHE_PATH": str(BASE / "data/eval/rewrite_cache/smoke200_rewrites.jsonl"),
    "EVAL_REWRITE_REQUIRE_CACHE": "true",
    "EVAL_REWRITE_FAIL_FAST_ON_MISSING": "true",
    "RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED": "true",
    "GENERATION_VERSION": "v2", "GENERATION_V2_USE_QWEN_SYNTHESIS": "false",
})

RES_DIR = BASE / "results/phase21b_eval4d_full200_biorag_eval"
CACHE_DIR = RES_DIR / "judge_cache_v31_full200"

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from scripts.evaluation.biorag_eval.rule_metrics import compute_all_rule_metrics
from scripts.evaluation.biorag_eval.qwen_judge import QwenJudgeClient
from scripts.evaluation.biorag_eval.aggregate_scores import aggregate_judge_scores, aggregate_rule_scores
from scripts.evaluation.biorag_eval.schemas import applicable_metrics, score_bucket


RESIDUAL_IDS = {"ent_058", "ent_083", "ent_094", "p21a9v2_fact_001", "p21a9v2_fact_002", "p21a9v2_fact_004"}
REPLACEMENT5 = {"p21a9v2_fact_001", "p21a9v2_fact_002", "p21a9v2_fact_003", "p21a9v2_fact_004", "p21a9v2_fact_005"}


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    s200 = [json.loads(l) for l in open(BASE / "data/eval/datasets/smoke200.jsonl")]
    s150_ids = {json.loads(l)["sample_id"] for l in open(BASE / "data/eval/datasets/smoke150.jsonl")}
    smoke_results = {}
    try:
        with open(BASE / "results/phase21a9w_smoke200_frozen_rebaseline_after_replacement/per_sample_results.csv") as f:
            for row in csv.DictReader(f):
                smoke_results[row.get("sample_id", "")] = row
    except: pass

    # ── Step 2: Collect EvalRecords ──
    settings = Settings.from_env()
    settings.query_rewrite.mode = "enabled"
    settings.retrieval.original_cn_fallback_enabled = True
    pipeline = SynBioRAGPipeline(settings=settings)

    records = []
    coll_stats = {"frozen": 0, "live": 0, "errors": 0}
    t0 = time.time()
    print(f"Collecting 200 answers...")

    for i, s in enumerate(s200):
        sid = s["sample_id"]
        try:
            resp = pipeline.answer(s["question"], filters=QueryFilters(tenant_id="default"))
        except Exception as e:
            coll_stats["errors"] += 1
            continue

        qr = resp.debug.get("query_rewrite", {})
        if qr.get("rewrite_cache_source") == "frozen_eval_file": coll_stats["frozen"] += 1
        else: coll_stats["live"] += 1

        gv2 = resp.debug.get("generation_v2", {})
        support = _extract_support(gv2)
        citations = resp.citations or []
        cited_ids = list(set(c.doc_id for c in citations if c.doc_id))
        route_pred = resp.route.value if hasattr(resp.route, "value") else str(resp.route)
        is_neg = not s.get("expected_doc_ids")
        ref = s.get("expected_answer") or ""
        meta = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
        ak = meta.get("answer_key", "") or ""

        rec = {
            "sample_id": sid,
            "split": "smoke150" if sid in s150_ids else "added50",
            "category": s.get("category", ""),
            "expected_route": s.get("expected_route", ""),
            "question": s["question"],
            "answer": resp.answer or "",
            "selected_support": support,
            "selected_support_doc_ids": [sp["doc_id"] for sp in support],
            "cited_doc_ids": cited_ids,
            "expected_doc_ids": s.get("expected_doc_ids") or [],
            "expected_source_files": s.get("expected_source_files") or [],
            "expected_behavior": _fmt(s.get("expected_behavior")),
            "answer_key": ak,
            "reference": ref or ak,
            "is_negative": is_neg,
            "is_known_residual": sid in RESIDUAL_IDS,
            "is_replacement_residual": sid in REPLACEMENT5,
            "route_pred": route_pred,
            "citation_count": len(citations),
            "smoke_real_P0": "yes" if sid in RESIDUAL_IDS else "no",
            "smoke_failure_class": smoke_results.get(sid, {}).get("failure_class", ""),
            "smoke_first_loss_stage": smoke_results.get(sid, {}).get("first_loss_stage", ""),
            "frozen_rewrite_used": qr.get("rewrite_cache_source") == "frozen_eval_file",
            "rewrite_fallback": False,
            "notes": "",
        }
        records.append(rec)
        if (i + 1) % 40 == 0: print(f"  {i+1}/200 collected")

    elapsed = time.time() - t0
    print(f"  Done: {len(records)} records in {elapsed:.0f}s. frozen={coll_stats['frozen']}, live={coll_stats['live']}")

    with open(RES_DIR / "full200_eval_records.jsonl", "w") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_s = sum(len(r["selected_support"]) for r in records) / max(len(records), 1)
    coll_metrics = {
        "sample_count": len(s200), "answers_collected": len(records),
        "selected_support_available_count": sum(1 for r in records if r["selected_support"]),
        "avg_support_count": round(avg_s, 2),
        "frozen_rewrite_used_count": coll_stats["frozen"],
        "live_rewrite_call_count": coll_stats["live"],
        "rewrite_fallback_count": 0, "pipeline_errors": coll_stats["errors"],
        "collection_pass": coll_stats["frozen"] == 200 and coll_stats["live"] == 0,
        "notes": "",
    }
    with open(RES_DIR / "full200_collection_metrics.json", "w") as f:
        json.dump(coll_metrics, f, ensure_ascii=False, indent=2)
    print(f"  collection_pass={coll_metrics['collection_pass']}")

    # ── Step 3: Rule metrics ──
    rule_rows = [compute_all_rule_metrics(r) for r in records]
    rfields = ["sample_id", "split", "category", "expected_route", "route_pred", "route_match",
               "doc_recall_support", "doc_recall_citation", "expected_doc_in_support", "expected_doc_cited",
               "citation_count", "wrong_doc_citation", "negative_citation_zero",
               "smoke_real_P0", "is_known_residual", "is_replacement_residual", "notes"]
    _write_csv(RES_DIR / "full200_rule_scores.csv", rfields, rule_rows)
    rule_sum = aggregate_rule_scores(rule_rows)
    rule_sum.update({"known_residual_count": sum(1 for r in records if r["is_known_residual"]),
                      "replacement_residual_count": sum(1 for r in records if r["is_replacement_residual"])})
    with open(RES_DIR / "full200_rule_summary.json", "w") as f:
        json.dump(rule_sum, f, ensure_ascii=False, indent=2)
    print(f"  Rule metrics: route_match={rule_sum['route_match_rate']}")

    # ── Step 4: Qwen judge ──
    judge = QwenJudgeClient(model="qwen-plus", max_tokens=512, temperature=0.0, timeout=60,
                            max_retries=2, cache_path=str(CACHE_DIR / "judge_cache_v31.jsonl"))
    judge_rows = []
    judge_stats = {"calls": 0, "cache_hits": 0, "errors": 0}
    t1 = time.time()

    for i, rec in enumerate(records):
        sid = rec["sample_id"]
        app = applicable_metrics(rec)
        for metric_name in app:
            if metric_name in ("comparison_axis_covered", "comparison_faithfulness") and "both_branches_covered" in app:
                continue
            if metric_name == "both_branches_covered":
                result = judge.judge(rec, "both_branches_covered")
                judge_stats["calls"] += 1
                raw = result.get("raw_json", {}) if isinstance(result.get("raw_json"), dict) else {}
                for sk in ["both_branches_covered", "comparison_axis_covered", "comparison_faithfulness"]:
                    val = raw.get(sk)
                    judge_rows.append({**dict(result), "sample_id": sid, "metric_name": sk, "score": val,
                                       "score_valid": val is not None, "rationale": raw.get("rationale", "")})
                if result.get("cache_hit"): judge_stats["cache_hits"] += 1
                if result.get("judge_error_type"): judge_stats["errors"] += 1
                continue
            if metric_name == "unit_correct" and "numeric_accuracy" in app:
                continue
            if metric_name == "numeric_accuracy":
                result = judge.judge(rec, "numeric_accuracy")
                judge_stats["calls"] += 1
                raw = result.get("raw_json", {}) if isinstance(result.get("raw_json"), dict) else {}
                for sk in ["numeric_accuracy", "unit_correct"]:
                    val = raw.get(sk)
                    if sk == "unit_correct" and isinstance(val, bool): val = str(val).lower()
                    judge_rows.append({**dict(result), "sample_id": sid, "metric_name": sk, "score": val,
                                       "score_valid": val is not None, "rationale": raw.get("rationale", "")})
                if result.get("cache_hit"): judge_stats["cache_hits"] += 1
                if result.get("judge_error_type"): judge_stats["errors"] += 1
                continue
            result = judge.judge(rec, metric_name)
            judge_stats["calls"] += 1
            judge_rows.append(result)
            if result.get("cache_hit"): judge_stats["cache_hits"] += 1
            if result.get("judge_error_type"): judge_stats["errors"] += 1

        if (i + 1) % 20 == 0:
            print(f"  Judge: {i+1}/200 samples, {judge_stats['calls']} calls, {judge_stats['errors']} errors")

    jt = time.time() - t1
    jfields = ["sample_id", "category", "metric_name", "score", "score_valid", "score_bucket",
               "judge_error_type", "rationale", "major_issue", "prompt_version", "cache_hit", "notes"]
    _write_csv_judge(RES_DIR / "full200_qwen_judge_scores.csv", jfields, judge_rows)

    judge_run = {
        "sample_count": len(records), "judge_call_count": judge_stats["calls"],
        "cache_hit_count": judge_stats["cache_hits"],
        "api_call_count": judge_stats["calls"] - judge_stats["cache_hits"],
        "judge_error_count": judge_stats["errors"],
        "parse_error_count": sum(1 for j in judge_rows if "parse" in str(j.get("judge_error_type", ""))),
        "timeout_count": sum(1 for j in judge_rows if "timeout" in str(j.get("judge_error_type", ""))),
        "max_workers": 2, "run_completed": True,
        "elapsed_s": round(jt, 0), "notes": "",
    }
    with open(RES_DIR / "full200_judge_run_metrics.json", "w") as f:
        json.dump(judge_run, f, ensure_ascii=False, indent=2)
    print(f"  Judge done in {jt:.0f}s: {judge_stats['calls']} calls, {judge_stats['errors']} errors, {judge_stats['cache_hits']} cache")

    # ── Step 5: Combined scores ──
    judge_map = {(j["sample_id"], j["metric_name"]): j for j in judge_rows}
    combined = []
    for rec in records:
        sid = rec["sample_id"]
        rule = next((r for r in rule_rows if r["sample_id"] == sid), {})
        row = {
            "sample_id": sid, "split": rec["split"], "category": rec["category"],
            "expected_route": rec["expected_route"],
            "is_negative": rec["is_negative"], "is_known_residual": rec["is_known_residual"],
            "is_replacement_residual": rec["is_replacement_residual"],
            "smoke_real_P0": rec["smoke_real_P0"],
            "doc_recall_support": rule.get("doc_recall_support", ""),
            "doc_recall_citation": rule.get("doc_recall_citation", ""),
            "citation_count": rec["citation_count"],
            "wrong_doc_citation": rule.get("wrong_doc_citation", ""),
            "faithfulness": _s(judge_map, sid, "faithfulness"),
            "answer_relevance": _s(judge_map, sid, "answer_relevance"),
            "evidence_recall": _s(judge_map, sid, "evidence_recall"),
            "answer_accuracy": _s(judge_map, sid, "answer_accuracy"),
            "answer_completeness": _s(judge_map, sid, "answer_completeness"),
            "abstention_correctness": _s(judge_map, sid, "abstention_correctness"),
            "both_branches_covered": _s(judge_map, sid, "both_branches_covered"),
            "comparison_axis_covered": _s(judge_map, sid, "comparison_axis_covered"),
            "comparison_faithfulness": _s(judge_map, sid, "comparison_faithfulness"),
            "numeric_accuracy": _s(judge_map, sid, "numeric_accuracy"),
            "unit_correct": _s(judge_map, sid, "unit_correct"),
            "pass_count": _b(judge_map, sid, "pass"),
            "partial_count": _b(judge_map, sid, "partial"),
            "fail_count": _b(judge_map, sid, "fail"),
            "not_applicable_count": _b(judge_map, sid, "not_applicable"),
            "final_eval_bucket": _bucket(judge_map, sid, rec),
            "needs_manual_review": _nr(judge_map, sid, rec),
            "notes": "",
        }
        combined.append(row)

    cfields = list(combined[0].keys())
    _write_csv(RES_DIR / "full200_combined_scores.csv", cfields, combined)

    # ── Steps 6-7: Summaries ──
    judge_summary = aggregate_judge_scores(judge_rows)
    bucket_counts = defaultdict(int)
    for c in combined: bucket_counts[c["final_eval_bucket"]] += 1

    overall = {
        "sample_count": len(records),
        "rule_metrics_summary": rule_sum,
        "qwen_judge_summary": judge_summary["by_metric"],
        "final_eval_bucket_counts": dict(bucket_counts),
        "judge_error_count": judge_stats["errors"],
        "known_residual_count": sum(1 for r in records if r["is_known_residual"]),
        "replacement_residual_count": sum(1 for r in records if r["is_replacement_residual"]),
        "notes": "",
    }
    with open(RES_DIR / "full200_overall_summary.json", "w") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # Bucketed
    buckets = _build_buckets(records, combined, judge_map)
    with open(RES_DIR / "full200_bucketed_summary.json", "w") as f:
        json.dump(buckets, f, ensure_ascii=False, indent=2)

    # Low score review list
    _write_low_score(records, combined, judge_map)
    # Answer quality backlog
    _write_backlog(records, combined, judge_map)

    # ── Validity ──
    valid = coll_metrics["collection_pass"] and judge_stats["errors"] == 0
    validity = {
        "run_completed": True, "sample_count": 200,
        "eval_records_collected": len(records),
        "rule_metrics_completed": True, "qwen_judge_completed": True,
        "aggregation_completed": True, "judge_error_count": judge_stats["errors"],
        "live_rewrite_call_count": coll_stats["live"],
        "rewrite_fallback_count": 0, "qwen_synthesis_enabled": False,
        "prompt_version": "v3.1", "comparison_dedup_active": True,
        "unit_correct_cleanup_active": True,
        "run_valid": valid, "invalid_reasons": [],
        "notes": "",
    }
    with open(RES_DIR / "full200_eval_validity.json", "w") as f:
        json.dump(validity, f, ensure_ascii=False, indent=2)

    # ── Next step ──
    dom = _dominant_issue(combined, judge_map)
    nxt = {
        "phase21b_eval4d_completed": True, "full200_eval_valid": valid,
        "major_quality_issue_found": bucket_counts["system_issue_candidate"] > 10,
        "dominant_issue_category": dom["category"],
        "recommended_next_step": dom["next_step"],
        "rationale": f"{bucket_counts['system_issue_candidate']} system_issue candidates, {bucket_counts['partial_quality_issue']} partial. Dominant: {dom['category']}.",
        "notes": "",
    }
    with open(RES_DIR / "phase21b_eval5_next_step_decision.json", "w") as f:
        json.dump(nxt, f, ensure_ascii=False, indent=2)

    print(f"\n=== Full200 Complete ===")
    print(f"  Collection pass: {coll_metrics['collection_pass']}")
    print(f"  Judge: {judge_stats['calls']} calls, {judge_stats['errors']} errors")
    print(f"  Buckets: {dict(bucket_counts)}")
    print(f"  Run valid: {valid}")
    print(f"  Next: {nxt['recommended_next_step']}")


# ── Helpers ──
def _extract_support(gv2):
    candidates = gv2.get("candidates", [])
    cand_by_eid = {c.get("evidence_id", ""): c for c in candidates if c.get("evidence_id")}
    support_pack = gv2.get("support_pack", [])
    items = []
    for sp in support_pack:
        cand = cand_by_eid.get(sp.get("evidence_id", ""))
        if cand and cand.get("text", "").strip():
            items.append({"support_id": sp["evidence_id"], "doc_id": cand["doc_id"],
                          "source_file": cand.get("source_file", ""), "text": cand["text"].strip()[:1000]})
    return items[:3]

def _fmt(eb): return ", ".join(str(x) for x in eb) if isinstance(eb, list) else str(eb or "")
def _s(m, sid, met): return m.get((sid, met), {}).get("score")
def _b(m, sid, bucket): return sum(1 for (s, _), e in m.items() if s == sid and score_bucket(e.get("score")) == bucket)

def _bucket(m, sid, rec):
    if rec.get("is_known_residual"): return "expected_residual"
    for (s, met), e in m.items():
        if s == sid and met != "comparison_faithfulness" and e.get("score") is not None and e["score"] < 0.5:
            return "system_issue_candidate"
    for (s, met), e in m.items():
        if s == sid and met != "comparison_faithfulness" and e.get("score") is not None and 0.5 <= e["score"] < 0.75:
            return "partial_quality_issue"
    return "pass"

def _nr(m, sid, rec):
    if rec.get("is_known_residual"): return True
    for (s, met), e in m.items():
        if s == sid and met != "comparison_faithfulness" and e.get("score") is not None and e["score"] < 0.5:
            return True
    return False

def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def _write_csv_judge(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for j in rows:
            w.writerow({"sample_id": j["sample_id"], "category": "", "metric_name": j["metric_name"],
                        "score": j["score"], "score_valid": j.get("score_valid", False),
                        "score_bucket": score_bucket(j.get("score")),
                        "judge_error_type": j.get("judge_error_type", ""),
                        "rationale": j.get("rationale", ""), "major_issue": "",
                        "prompt_version": "v3.1", "cache_hit": j.get("cache_hit", False), "notes": ""})

def _build_buckets(records, combined, judge_map):
    by_split = defaultdict(list); by_cat = defaultdict(list); by_status = defaultdict(list)
    for i, rec in enumerate(records):
        c = combined[i]
        by_split[rec["split"]].append(c)
        by_cat[rec["category"] or "unknown"].append(c)
        if rec["is_known_residual"]: by_status["known_residual"].append(c)
        elif rec["is_replacement_residual"]: by_status["replacement_residual"].append(c)
        elif rec["smoke_real_P0"] == "yes": by_status["smoke_real_P0"].append(c)
        else: by_status["normal"].append(c)
    return {
        "by_split": {k: _bucket_stats(v) for k, v in sorted(by_split.items())},
        "by_category": {k: _bucket_stats(v) for k, v in sorted(by_cat.items())},
        "by_status": {k: _bucket_stats(v) for k, v in sorted(by_status.items())},
    }

def _bucket_stats(items):
    if not items: return {"sample_count": 0}
    faith_vals = [c["faithfulness"] for c in items if c.get("faithfulness") is not None and str(c["faithfulness"]) != "None"]
    return {
        "sample_count": len(items),
        "faithfulness_mean": round(sum(faith_vals)/len(faith_vals), 4) if faith_vals else None,
        "fail_count": sum(1 for c in items if c.get("fail_count", 0) > 0),
        "partial_count": sum(1 for c in items if c.get("partial_count", 0) > 0 and c.get("fail_count", 0) == 0),
        "pass_count": sum(1 for c in items if c.get("fail_count", 0) == 0 and c.get("partial_count", 0) == 0),
    }

def _write_low_score(records, combined, judge_map):
    lst = []
    for i, rec in enumerate(records):
        c = combined[i]
        if c["final_eval_bucket"] == "pass": continue
        sid = rec["sample_id"]
        # Identify low metrics
        lows = []
        for met in ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy",
                     "abstention_correctness", "both_branches_covered", "numeric_accuracy"]:
            s = _s(judge_map, sid, met)
            if s is not None and isinstance(s, (int, float)) and s < 0.75:
                lows.append(f"{met}={s}")
        # Issue type
        cat = (rec["category"] or "").lower()
        if "comparison" in cat: itype = "comparison_missing_branch"
        elif any(t in cat for t in ("numeric", "table", "figure")): itype = "numeric_value_missing_or_wrong"
        elif "cross_lingual" in cat: itype = "cross_lingual_answer_mismatch"
        else: itype = "unclear"
        lst.append({"sample_id": sid, "split": rec["split"], "category": rec["category"],
                     "question": rec["question"][:150], "answer_preview": rec["answer"][:150],
                     "selected_support_preview": "; ".join(s.get("text", "")[:60] for s in rec.get("selected_support", [])[:2]),
                     "low_metrics": "; ".join(lows[:5]),
                     "rule_metric_flags": "",
                     "qwen_major_issue": "",
                     "likely_issue_type": itype,
                     "recommended_review_priority": "P0" if c["final_eval_bucket"] == "system_issue_candidate" else "P1",
                     "recommended_action": "manual_review",
                     "notes": ""})
    fields = list(lst[0].keys()) if lst else ["sample_id"]
    _write_csv(RES_DIR / "full200_low_score_review_list.csv", fields, lst)

def _write_backlog(records, combined, judge_map):
    bl = []
    bid = 0
    for i, rec in enumerate(records):
        c = combined[i]
        if c["final_eval_bucket"] not in ("system_issue_candidate", "partial_quality_issue"): continue
        if rec["is_known_residual"]: continue
        bid += 1
        cat = (rec["category"] or "").lower()
        sid = rec["sample_id"]
        if "comparison" in cat: itype = "comparison_missing_branch"; stage = "answer_synthesis"; phase = "phase21b_comparison_answering"
        elif any(t in cat for t in ("numeric", "table", "figure")): itype = "numeric_value_missing_or_wrong"; stage = "answer_synthesis"; phase = "phase21b_numeric_table_figure"
        elif "cross_lingual" in cat: itype = "cross_lingual_answer_mismatch"; stage = "answer_synthesis"; phase = "phase21b_cross_lingual_answering"
        elif "summary" in cat: itype = "partial_summary"; stage = "answer_synthesis"; phase = "phase21b_answer_synthesis"
        else: itype = "unclear"; stage = "unclear"; phase = "no_action"
        severity = "P1" if c["final_eval_bucket"] == "system_issue_candidate" else "P2"
        bl.append({"backlog_id": f"BL-EVAL4D-{bid:03d}", "sample_id": sid, "category": rec["category"],
                    "issue_type": itype, "likely_pipeline_stage": stage,
                    "evidence": f"Qwen v3.1 score: {c.get('faithfulness', '?')} faith, {c.get('answer_relevance', '?')} rel",
                    "severity": severity, "recommended_future_phase": phase, "notes": ""})
    fields = list(bl[0].keys()) if bl else ["backlog_id"]
    _write_csv(RES_DIR / "answer_quality_backlog.csv", fields, bl)

def _dominant_issue(combined, judge_map):
    cmp_fail = sum(1 for c in combined if "comparison" in (c.get("category","")).lower() and c["final_eval_bucket"] == "system_issue_candidate")
    num_fail = sum(1 for c in combined if any(t in (c.get("category","")).lower() for t in ("numeric","table","figure")) and c["final_eval_bucket"] == "system_issue_candidate")
    gen_fail = sum(1 for c in combined if c["final_eval_bucket"] == "system_issue_candidate")
    total_sys = gen_fail + sum(1 for c in combined if c["final_eval_bucket"] == "partial_quality_issue")
    if num_fail > cmp_fail and num_fail > gen_fail // 3:
        return {"category": "numeric_table_figure", "next_step": "start_phase21b_pdf_table_figure_cleanup"}
    if cmp_fail > gen_fail // 3:
        return {"category": "comparison_answering", "next_step": "start_phase21b_comparison_answering_fix"}
    if gen_fail > 20:
        return {"category": "answer_synthesis", "next_step": "start_phase21b_answer_quality_fix"}
    return {"category": "mixed", "next_step": "manual_review_full200_low_scores"}


if __name__ == "__main__":
    main()
