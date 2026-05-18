"""Phase 21B-EVAL-1: BIORAG Eval v2 — recalibrated pilot on same 19 samples."""
import csv
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv
load_dotenv(".env")

os.environ.update({
    "QUERY_REWRITE_MODE": "enabled",
    "EVAL_REWRITE_CACHE_PATH": str(BASE / "data/eval/rewrite_cache/smoke200_rewrites.jsonl"),
    "EVAL_REWRITE_REQUIRE_CACHE": "true",
    "GENERATION_VERSION": "v2",
    "GENERATION_V2_USE_QWEN_SYNTHESIS": "false",
})

RES_DIR = BASE / "results/phase21b_eval1_judge_calibration"
V0_PILOT = BASE / "results/phase21b_eval0_biorag_eval_pilot"
CACHE_PATH = RES_DIR / "judge_cache_v2.jsonl"

from scripts.evaluation.biorag_eval.rule_metrics import compute_all_rule_metrics
from scripts.evaluation.biorag_eval.qwen_judge import QwenJudgeClient
from scripts.evaluation.biorag_eval.aggregate_scores import aggregate_rule_scores, aggregate_judge_scores
from scripts.evaluation.biorag_eval.schemas import applicable_metrics, score_bucket


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Load EVAL-0 pilot records (same 19 samples, no re-collection)
    records = [json.loads(l) for l in open(V0_PILOT / "pilot_eval_records.jsonl")]
    v0_combined = list(csv.DictReader(open(V0_PILOT / "pilot_combined_scores.csv")))
    print(f"Loaded {len(records)} pilot records from EVAL-0")

    # ── Step 1: Calibration sheet ──
    cal_sheet = []
    for rec in records:
        sid = rec["sample_id"]
        v0 = next((c for c in v0_combined if c["sample_id"] == sid), {})
        support_preview = "; ".join(s.get("text", "")[:80] + "..." for s in rec.get("selected_support", [])[:2])
        cal_sheet.append({
            "sample_id": sid, "category": rec["category"],
            "expected_route": rec["expected_route"],
            "question": rec["question"][:150],
            "answer_preview": rec["answer"][:200],
            "selected_support_preview": support_preview,
            "smoke_real_P0": rec["smoke_real_P0"],
            "current_faithfulness": v0.get("faithfulness", ""),
            "current_answer_relevance": v0.get("answer_relevance", ""),
            "current_evidence_recall": v0.get("evidence_recall", ""),
            "current_answer_accuracy": v0.get("answer_accuracy", ""),
            "current_abstention_correctness": v0.get("abstention_correctness", ""),
            "current_major_issue": "",
            "human_label_placeholder": "",
            "human_notes_placeholder": "",
            "needs_manual_review_reason": "",
        })
    with open(RES_DIR / "manual_calibration_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_sheet[0].keys()))
        w.writeheader(); w.writerows(cal_sheet)

    # ── Step 2-4: Write design docs (policy, scale, aggregation) ──
    _write_design_docs()

    # ── Rule metrics (unchanged) ──
    rule_rows = [compute_all_rule_metrics(r) for r in records]

    # ── Steps 5-6: Run v2 Qwen judge ──
    judge = QwenJudgeClient(model="qwen-plus", max_tokens=512, temperature=0.0, timeout=60, max_retries=2, cache_path=str(CACHE_PATH))

    judge_rows = []
    all_metrics = sorted(set(m for r in records for m in applicable_metrics(r)))

    for rec in records:
        sid = rec["sample_id"]
        app = applicable_metrics(rec)
        for metric_name in all_metrics:
            if metric_name in app:
                # Some metrics share a prompt (comparison: both_branches_covered/comparison_axis_covered/faithfulness)
                if metric_name in ("comparison_axis_covered",) and "both_branches_covered" in app:
                    continue  # Already covered by both_branches_covered call
                if metric_name == "unit_correct" and "numeric_accuracy" in app:
                    result = judge.judge(rec, "numeric_accuracy")
                    # Extract sub-scores from the shared numeric prompt response
                    raw = result.get("raw_json", {})
                    for sub_key in ["numeric_accuracy", "unit_correct"]:
                        sub_result = dict(result)
                        sub_result["sample_id"] = sid
                        sub_result["metric_name"] = sub_key
                        sub_result["score"] = raw.get(sub_key) if isinstance(raw, dict) else None
                        sub_result["score_valid"] = raw.get(sub_key) is not None if isinstance(raw, dict) else False
                        sub_result["rationale"] = raw.get("rationale", "") if isinstance(raw, dict) else ""
                        judge_rows.append(sub_result)
                    continue
                if metric_name in ("both_branches_covered",):
                    result = judge.judge(rec, "both_branches_covered")
                    raw = result.get("raw_json", {})
                    for sub_key in ["both_branches_covered", "comparison_axis_covered", "faithfulness"]:
                        val = raw.get(sub_key) if isinstance(raw, dict) else None
                        sub_result = dict(result)
                        sub_result["sample_id"] = sid
                        sub_result["metric_name"] = sub_key
                        sub_result["score"] = val
                        sub_result["score_valid"] = val is not None
                        sub_result["rationale"] = raw.get("rationale", "") if isinstance(raw, dict) else ""
                        judge_rows.append(sub_result)
                    continue
                result = judge.judge(rec, metric_name)
                judge_rows.append(result)
            else:
                judge_rows.append({
                    "sample_id": sid, "metric_name": metric_name,
                    "score": None, "score_valid": False, "cache_hit": False,
                    "judge_error_type": "", "rationale": f"not_applicable_{rec.get('category', '?')}",
                    "raw_preview": "", "prompt_version": "v2.0",
                })

    # Write v2 judge scores
    jfields = ["sample_id", "category", "metric_name", "score", "score_valid", "score_bucket",
               "judge_error_type", "rationale", "major_issue", "prompt_version", "cache_hit", "notes"]
    with open(RES_DIR / "pilot_v2_qwen_judge_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=jfields, extrasaction="ignore")
        w.writeheader()
        for j in judge_rows:
            rec = next((r for r in records if r["sample_id"] == j["sample_id"]), {})
            w.writerow({
                "sample_id": j["sample_id"], "category": rec.get("category", ""),
                "metric_name": j["metric_name"], "score": j.get("score"),
                "score_valid": j.get("score_valid", False),
                "score_bucket": score_bucket(j.get("score")),
                "judge_error_type": j.get("judge_error_type", ""),
                "rationale": j.get("rationale", ""), "major_issue": "",
                "prompt_version": "v2.0", "cache_hit": j.get("cache_hit", False), "notes": "",
            })

    # Build combined wide table
    judge_map: dict = {}
    for j in judge_rows:
        key = (j["sample_id"], j["metric_name"])
        judge_map[key] = j

    combined = []
    for rec in records:
        sid = rec["sample_id"]
        rule = next((r for r in rule_rows if r["sample_id"] == sid), {})
        row = {
            "sample_id": sid, "category": rec["category"],
            "smoke_real_P0": rec["smoke_real_P0"],
            "doc_recall_support": rule.get("doc_recall_support", ""),
            "doc_recall_citation": rule.get("doc_recall_citation", ""),
            "faithfulness": _score(judge_map, sid, "faithfulness"),
            "answer_relevance": _score(judge_map, sid, "answer_relevance"),
            "evidence_recall": _score(judge_map, sid, "evidence_recall"),
            "answer_accuracy": _score(judge_map, sid, "answer_accuracy"),
            "answer_completeness": _score(judge_map, sid, "answer_completeness"),
            "abstention_correctness": _score(judge_map, sid, "abstention_correctness"),
            "both_branches_covered": _score(judge_map, sid, "both_branches_covered"),
            "comparison_axis_covered": _score(judge_map, sid, "comparison_axis_covered"),
            "numeric_accuracy": _score(judge_map, sid, "numeric_accuracy"),
            "unit_correct": _score(judge_map, sid, "unit_correct"),
            "pass_count": _count_bucket(judge_map, sid, "pass"),
            "partial_count": _count_bucket(judge_map, sid, "partial"),
            "fail_count": _count_bucket(judge_map, sid, "fail"),
            "not_applicable_count": _count_bucket(judge_map, sid, "not_applicable"),
            "needs_manual_review": _needs_review_v2(judge_map, sid, rec),
            "notes": "",
        }
        combined.append(row)

    cfields = list(combined[0].keys())
    with open(RES_DIR / "pilot_v2_combined_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader(); w.writerows(combined)

    # ── Step 7: V1 vs V2 comparison ──
    v1_summary = json.load(open(V0_PILOT / "pilot_score_summary.json"))
    judge_summary = aggregate_judge_scores(judge_rows)
    error_count = judge_summary["judge_error_count"]

    v2_summary = {
        "sample_count": len(records),
        "metric_applicability_counts": {},
        "judge_call_count": len(judge_rows),
        "judge_error_count": error_count,
        "parse_error_count": sum(1 for j in judge_rows if j.get("judge_error_type") == "json_parse_error"),
        "timeout_count": 0,
        "cache_hit_count": judge_summary["cache_hit_count"],
    }
    for mn, ms in judge_summary["by_metric"].items():
        v2_summary[f"{mn}_mean"] = ms["mean"]
        v2_summary[f"{mn}_pass_rate"] = ms["pass_rate"]
        v2_summary[f"{mn}_partial_rate"] = ms["partial_rate"]
        v2_summary[f"{mn}_fail_rate"] = ms["fail_rate"]
        v2_summary[f"{mn}_count"] = ms["count"]

    needs_review = sum(1 for c in combined if c["needs_manual_review"])
    v2_summary["needs_manual_review_count"] = needs_review

    with open(RES_DIR / "pilot_v2_score_summary.json", "w") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)

    # V1 vs V2
    abst_n_v1 = v1_summary["qwen_judge_summary"]["by_metric"].get("abstention_correctness", {}).get("count", 0)
    abst_n_v2 = judge_summary["by_metric"].get("abstention_correctness", {}).get("count", 0)
    v1_needs = v1_summary.get("needs_manual_review_count", 0)

    comparison = {
        "same_sample_ids": True,
        "v1_summary": {"abstention_correctness_n": abst_n_v1, "needs_review": v1_needs,
                       "faithfulness_mean": v1_summary["qwen_judge_summary"]["by_metric"].get("faithfulness", {}).get("mean")},
        "v2_summary": {"abstention_correctness_n": abst_n_v2, "needs_review": needs_review,
                       "faithfulness_mean": judge_summary["by_metric"].get("faithfulness", {}).get("mean")},
        "abstention_n_fixed": abst_n_v2 < abst_n_v1,
        "answer_accuracy_n_v2": judge_summary["by_metric"].get("answer_accuracy", {}).get("count", 0),
        "manual_review_count_before": v1_needs,
        "manual_review_count_after": needs_review,
        "metric_level_changes": "v2 uses route-aware dispatch, 4-point scale, pass/partial/fail rates",
        "notes": "",
    }
    with open(RES_DIR / "v1_vs_v2_pilot_comparison.json", "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # ── Step 9: Review cards v2 ──
    _write_review_cards_v2(records, combined, judge_map)

    # ── Step 11: Calibration assessment ──
    assessment = {
        "v2_completed": True,
        "v2_engineering_stable": error_count == 0,
        "judge_json_stable": error_count == 0,
        "metric_eligibility_fixed": abst_n_v2 < 5,
        "scale_reasonable": needs_review < 19,
        "manual_review_count_reduced": needs_review < v1_needs,
        "metrics_ready_for_full_200": needs_review < 10,
        "remaining_issues": [],
        "recommended_next_step": "manual_review_v2_cards" if needs_review > 5 else "run_biorag_eval_200",
        "rationale": f"v2: {error_count} errors, {needs_review}/{len(records)} need review, abstention N={abst_n_v2}.",
        "notes": "",
    }
    if error_count > 2:
        assessment["recommended_next_step"] = "further_prompt_calibration"
    with open(RES_DIR / "judge_calibration_v2_assessment.json", "w") as f:
        json.dump(assessment, f, ensure_ascii=False, indent=2)

    # ── Run config ──
    with open(RES_DIR / "run_config.json", "w") as f:
        json.dump({"phase": "21B-EVAL-1", "title": "BIORAG Eval v1 Rubric Calibration v2", "scale": "4-point (1.0/0.75/0.5/0.0/null)", "metric_policy": "route-aware", "pilot_samples": len(records)}, f, ensure_ascii=False, indent=2)

    print(f"\n=== V2 Pilot Complete ===")
    print(f"  Metric calls: {len(judge_rows)}, errors: {error_count}")
    print(f"  Needs review: {needs_review}/{len(records)} (was {v1_needs}/19)")
    print(f"  Abstention N: {abst_n_v2} (was {abst_n_v1})")
    for mn, ms in sorted(judge_summary["by_metric"].items()):
        if ms["count"] > 0:
            print(f"  {mn}: mean={ms['mean']}, pass={ms['pass_rate']}, partial={ms['partial_rate']}, fail={ms['fail_rate']}, n={ms['count']}")
    print(f"\n  recommended_next_step: {assessment['recommended_next_step']}")


def _score(judge_map, sid, metric):
    e = judge_map.get((sid, metric), {})
    return e.get("score")


def _count_bucket(judge_map, sid, bucket):
    count = 0
    for (s, m), e in judge_map.items():
        if s == sid and score_bucket(e.get("score")) == bucket:
            count += 1
    return count


def _needs_review_v2(judge_map, sid, record):
    if record.get("smoke_real_P0") == "yes":
        return True
    for (s, m), e in judge_map.items():
        if s != sid: continue
        score = e.get("score")
        if score is not None and score < 0.5:
            return True
    return False


def _write_design_docs():
    # Applicability policy
    import json as j
    policy = {
        "version": "v2.0",
        "route_metric_mapping": {
            "factoid": ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy"],
            "summary": ["faithfulness", "answer_relevance", "evidence_recall", "answer_completeness", "answer_accuracy"],
            "comparison": ["faithfulness", "both_branches_covered", "comparison_axis_covered", "answer_relevance", "answer_accuracy"],
            "cross_lingual": ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy"],
            "method_result_numeric": ["faithfulness", "numeric_accuracy", "unit_correct", "evidence_recall", "answer_accuracy"],
            "table_figure_caption": ["faithfulness", "numeric_accuracy", "unit_correct", "evidence_recall", "answer_accuracy"],
            "negative": ["abstention_correctness"],
        },
        "skip_rules": [
            "answer_accuracy skipped if no reference/answer_key",
            "evidence_recall skipped if no expected_behavior/answer_key/reference",
            "abstention_correctness ONLY for negative samples",
            "comparison metrics ONLY for comparison",
            "numeric_accuracy ONLY for numeric/table",
            "null scores not included in means",
        ],
    }
    with open(RES_DIR / "metric_applicability_policy_v2.json", "w") as f:
        j.dump(policy, f, ensure_ascii=False, indent=2)

    scale = {
        "version": "v2.0",
        "scale_values": [1.0, 0.75, 0.5, 0.0, None],
        "pass_threshold": 0.75,
        "partial_threshold": 0.5,
        "fail_value": 0.0,
        "null_policy": "not_applicable — excluded from all aggregates",
        "aggregation_policy": {
            "mean": "valid_only_mean (null excluded)",
            "pass_rate": "fraction with score >= 0.75",
            "partial_rate": "fraction with 0.5 <= score < 0.75",
            "fail_rate": "fraction with score < 0.5",
        },
        "notes": "4-point scale provides finer differentiation than original 3-point. 0.75=acceptable prevents over-flagging.",
    }
    with open(RES_DIR / "scoring_scale_v2.json", "w") as f:
        j.dump(scale, f, ensure_ascii=False, indent=2)

    aggr = {
        "version": "v2.0",
        "metric_dispatch_rules": "route-aware; see metric_applicability_policy_v2.json",
        "null_handling": "null scores excluded from mean, pass_rate, partial_rate, fail_rate denominators",
        "negative_handling": "abstention_correctness only; excluded from non-negative means",
        "residual_handling": "known_residual bucket reported separately; not mixed into normal pass/fail counts",
        "overall_score_formula": "mean of all valid non-null scores per sample per metric",
        "pass_rate_formula": "pass_count / valid_count",
        "notes": "",
    }
    with open(RES_DIR / "aggregation_policy_v2.json", "w") as f:
        j.dump(aggr, f, ensure_ascii=False, indent=2)


def _write_review_cards_v2(records, combined, judge_map):
    lines = ["# BIORAG Eval v2 — Manual Review Cards (Calibrated)\n"]
    for rec in records:
        sid = rec["sample_id"]
        comb = next((c for c in combined if c["sample_id"] == sid), {})
        if not comb.get("needs_manual_review"):
            continue  # Only flag review-needed samples

        support_preview = "; ".join(f"[{s['doc_id']}] {s['text'][:60]}..." for s in rec.get("selected_support", [])[:2])
        lines.append(f"## Sample {sid}\n")
        lines.append(f"- **Category**: {rec['category']} | **Smoke P0**: {rec['smoke_real_P0']}")
        lines.append(f"- **Question**: {rec['question'][:200]}")
        lines.append(f"- **Answer**: {rec['answer'][:250]}")
        lines.append(f"- **Support**: {support_preview}")
        lines.append(f"- **Expected**: {rec.get('expected_behavior', '')[:120]}")
        lines.append(f"- **V2 Scores**:")
        for m in ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy", "abstention_correctness", "both_branches_covered", "numeric_accuracy"]:
            e = judge_map.get((sid, m), {})
            score = e.get("score")
            if score is not None or e.get("score_valid"):
                lines.append(f"  - {m}: {score} — {e.get('rationale', '')}")
        lines.append(f"- **Suggested human decision**: (judge_correct / judge_too_strict / judge_too_lenient / system_issue / data_issue / unclear)")
        lines.append(f"- **Human notes**: \n")

    path = BASE / "reports/phase21b_eval1_judge_calibration/manual_review_cards_v2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
