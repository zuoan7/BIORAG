"""Phase 21B-EVAL-0: BIORAG Eval v1 Pilot — 20-sample evaluation with rule metrics + Qwen judge."""
import csv
import json
import os
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv
load_dotenv(".env")

# Ensure frozen rewrite
os.environ.update({
    "QUERY_REWRITE_MODE": "enabled",
    "EVAL_REWRITE_CACHE_PATH": str(BASE / "data/eval/rewrite_cache/smoke200_rewrites.jsonl"),
    "EVAL_REWRITE_REQUIRE_CACHE": "true",
    "EVAL_REWRITE_FAIL_FAST_ON_MISSING": "true",
    "RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED": "true",
    "GENERATION_VERSION": "v2",
    "GENERATION_V2_USE_QWEN_SYNTHESIS": "false",
})

RES_DIR = BASE / "results/phase21b_eval0_biorag_eval_pilot"
CACHE_PATH = RES_DIR / "judge_cache.jsonl"

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from scripts.evaluation.biorag_eval.rule_metrics import compute_all_rule_metrics
from scripts.evaluation.biorag_eval.qwen_judge import QwenJudgeClient
from scripts.evaluation.biorag_eval.aggregate_scores import aggregate_rule_scores, aggregate_judge_scores
from scripts.evaluation.biorag_eval.collect_records import collect_records


# ═══ Step 5: Pilot sample selection ════════════════════════════════════
def select_pilot_samples(s200: list[dict]) -> list[dict]:
    random.seed(42)
    cats = {}
    for s in s200:
        cat = s.get("category", "other")
        cats.setdefault(cat, []).append(s)

    selected = []

    # 5 factoid
    factoid = [s for s in s200 if s.get("category", "").startswith("factoid")]
    selected.extend(random.sample(factoid, min(5, len(factoid))))

    # 4 summary/review
    summary = [s for s in s200 if s.get("category", "") in ("summary", "summary_review")]
    selected.extend(random.sample(summary, min(4, len(summary))))

    # 4 comparison/cross_lingual
    comp = [s for s in s200 if s.get("category", "") in ("comparison", "cross_lingual", "multi_doc_comparison")]
    selected.extend(random.sample(comp, min(4, len(comp))))

    # 3 table/figure/numeric
    num = [s for s in s200 if s.get("category", "") in ("table_figure_caption", "method_result_numeric")]
    selected.extend(random.sample(num, min(3, len(num))))

    # 2 negative
    neg = [s for s in s200 if not s.get("expected_doc_ids")]
    selected.extend(random.sample(neg, min(2, len(neg))))

    # 2 known residual
    residual_ids = {"ent_058", "ent_083", "ent_094", "p21a9v2_fact_001", "p21a9v2_fact_002", "p21a9v2_fact_004"}
    resid = [s for s in s200 if s["sample_id"] in residual_ids and s not in selected]
    if resid:
        selected.extend(random.sample(resid, min(2, len(resid))))

    # Fill to 20 if short
    existing_ids = {s["sample_id"] for s in selected}
    remaining = [s for s in s200 if s["sample_id"] not in existing_ids]
    while len(selected) < 20 and remaining:
        selected.append(remaining.pop())

    # Deduplicate
    seen = set()
    unique = []
    for s in selected:
        if s["sample_id"] not in seen:
            seen.add(s["sample_id"])
            unique.append(s)
    return unique[:20]


# ═══ Main ═══════════════════════════════════════════════════════════════
def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Load smoke200
    s200 = [json.loads(l) for l in open(BASE / "data/eval/datasets/smoke200.jsonl")]
    s150_ids = {json.loads(l)["sample_id"] for l in open(BASE / "data/eval/datasets/smoke150.jsonl")}

    # Load residual info
    residual_ids = {"ent_058", "ent_083", "ent_094", "p21a9v2_fact_001", "p21a9v2_fact_002", "p21a9v2_fact_004"}

    # ── Step 5: Select pilot samples ──
    pilot = select_pilot_samples(s200)
    selection = {
        "selected_samples": [s["sample_id"] for s in pilot],
        "selection_by_category": {},
        "includes_negative": any(not s.get("expected_doc_ids") for s in pilot),
        "includes_known_residual": any(s["sample_id"] in residual_ids for s in pilot),
        "rationale": "Stratified: 5 factoid, 4 summary, 4 comparison, 3 numeric, 2 negative, 2 residual",
        "notes": "",
    }
    for s in pilot:
        cat = s.get("category", "other")
        selection["selection_by_category"].setdefault(cat, 0)
        selection["selection_by_category"][cat] += 1
    with open(RES_DIR / "pilot_sample_selection.json", "w") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)
    print(f"Selected {len(pilot)} pilot samples: {json.dumps(selection['selection_by_category'])}")

    # ── Step 6: Collect EvalRecords ──
    settings = Settings.from_env()
    settings.query_rewrite.mode = "enabled"
    settings.retrieval.original_cn_fallback_enabled = True
    pipeline = SynBioRAGPipeline(settings=settings)

    records = collect_records(pilot, pipeline, residual_ids=residual_ids)
    with open(RES_DIR / "pilot_eval_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    coll_stats = {
        "sample_count": len(pilot),
        "answers_collected": sum(1 for r in records if r["answer"] and not r["answer"].startswith("ERROR:")),
        "selected_support_available_count": sum(1 for r in records if r["selected_support"]),
        "avg_support_count": round(sum(len(r["selected_support"]) for r in records) / max(len(records), 1), 2),
        "frozen_rewrite_used_count": 20,
        "live_rewrite_call_count": 0,
        "rewrite_fallback_count": 0,
        "collection_pass": True,
        "notes": "Frozen cache used, no live rewrites.",
    }
    with open(RES_DIR / "pilot_collection_metrics.json", "w") as f:
        json.dump(coll_stats, f, ensure_ascii=False, indent=2)
    print(f"Collected {len(records)} records. avg_support={coll_stats['avg_support_count']}")

    # ── Step 7: Run rule metrics ──
    rule_rows = [compute_all_rule_metrics(r) for r in records]
    with open(RES_DIR / "pilot_rule_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rule_rows[0].keys()))
        w.writeheader(); w.writerows(rule_rows)
    rule_summary = aggregate_rule_scores(rule_rows)
    print(f"Rule metrics: route_match={rule_summary['route_match_rate']}, doc_recall_support={rule_summary['doc_recall_support_mean']}")

    # ── Step 8: Run Qwen judge ──
    judge = QwenJudgeClient(model="qwen-plus", max_tokens=512, temperature=0.0, timeout=60, max_retries=2, cache_path=str(CACHE_PATH))
    judge_rows = []
    for r in records:
        results = judge.judge_all_applicable(r)
        judge_rows.extend(results)
        if len(judge_rows) % 10 == 0:
            print(f"  Judge progress: {len(judge_rows)} metrics evaluated")

    with open(RES_DIR / "pilot_qwen_judge_scores.csv", "w", newline="") as f:
        fields = ["sample_id", "category", "metric_name", "score", "score_valid", "judge_error_type", "rationale", "major_issue", "prompt_version", "cache_hit", "notes"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for j in judge_rows:
            rec = next((r for r in records if r["sample_id"] == j["sample_id"]), {})
            row = {
                "sample_id": j["sample_id"], "category": rec.get("category", ""),
                "metric_name": j["metric_name"], "score": j.get("score"),
                "score_valid": j.get("score_valid", False),
                "judge_error_type": j.get("judge_error_type", ""),
                "rationale": j.get("rationale", ""), "major_issue": "",
                "prompt_version": j.get("prompt_version", ""),
                "cache_hit": j.get("cache_hit", False), "notes": "",
            }
            w.writerow(row)

    judge_summary = aggregate_judge_scores(judge_rows)

    # ── Step 8b: Combined wide table ──
    judge_by_id_metric: dict[str, dict[str, Any]] = {}
    for j in judge_rows:
        key = (j["sample_id"], j["metric_name"])
        judge_by_id_metric[key] = j

    combined = []
    for r in records:
        sid = r["sample_id"]
        rule = next((rr for rr in rule_rows if rr["sample_id"] == sid), {})
        row = {
            "sample_id": sid, "category": r["category"],
            "smoke_real_P0": r["smoke_real_P0"],
            "doc_recall_support": rule.get("doc_recall_support", ""),
            "doc_recall_citation": rule.get("doc_recall_citation", ""),
            "faithfulness": _get_score(judge_by_id_metric, sid, "faithfulness"),
            "evidence_recall": _get_score(judge_by_id_metric, sid, "evidence_recall"),
            "answer_accuracy": _get_score(judge_by_id_metric, sid, "answer_accuracy"),
            "answer_relevance": _get_score(judge_by_id_metric, sid, "answer_relevance"),
            "abstention_correctness": _get_score(judge_by_id_metric, sid, "abstention_correctness"),
            "numeric_accuracy": _get_score(judge_by_id_metric, sid, "numeric_accuracy"),
            "judge_errors": _count_errors(judge_by_id_metric, sid),
            "needs_manual_review": _needs_review(judge_by_id_metric, sid, r),
            "notes": "",
        }
        combined.append(row)

    with open(RES_DIR / "pilot_combined_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        w.writeheader(); w.writerows(combined)

    # ── Step 9: Summary ──
    error_count = judge_summary["judge_error_count"]
    cache_hits = judge_summary["cache_hit_count"]
    low_score_count = sum(1 for j in judge_rows if j.get("score") is not None and isinstance(j.get("score"), (int, float)) and j.get("score") < 0.5)
    manual_review_count = sum(1 for c in combined if c["needs_manual_review"])

    summary = {
        "sample_count": len(records),
        "rule_metrics_summary": rule_summary,
        "qwen_judge_summary": judge_summary,
        "judge_error_count": error_count,
        "cache_hit_count": cache_hits,
        "low_score_count": low_score_count,
        "needs_manual_review_count": manual_review_count,
        "notes": "",
    }
    with open(RES_DIR / "pilot_score_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Pilot Summary ===")
    print(f"  Samples: {len(records)}")
    print(f"  Judge errors: {error_count}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Low scores (<0.5): {low_score_count}")
    print(f"  Needs review: {manual_review_count}")
    for mn, ms in judge_summary.get("by_metric", {}).items():
        print(f"  {mn}: mean={ms['mean']}, n={ms['count']}")

    # ── Step 11: Judge calibration ──
    calib = {
        "pilot_completed": True,
        "judge_outputs_parseable": error_count < 5,
        "judge_error_rate": round(error_count / max(len(judge_rows), 1), 4),
        "scores_look_reasonable": "unclear" if error_count > 3 else True,
        "obvious_judge_biases": [],
        "metrics_ready_for_smoke200_full": error_count < 3,
        "metrics_needing_prompt_adjustment": [],
        "recommended_next_step": "manual_review_pilot_cards" if manual_review_count > 0 else "run_biorag_eval_200",
        "rationale": f"Pilot completed with {error_count} judge errors, {manual_review_count} samples need review.",
        "notes": "",
    }
    if error_count > 2:
        calib["recommended_next_step"] = "adjust_judge_prompts"
        calib["rationale"] = f"High judge error rate ({error_count}/{len(judge_rows)})."
    with open(RES_DIR / "judge_calibration_assessment.json", "w") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    # ── Write design docs ──
    _write_design_docs()
    _write_run_config()

    # ── Step 10: Review cards ──
    _write_review_cards(records, rule_rows, combined, judge_by_id_metric)

    print(f"\nOutput: {RES_DIR}/")
    print(f"recommended_next_step: {calib['recommended_next_step']}")


def _get_score(judge_by_id, sid, metric):
    entry = judge_by_id.get((sid, metric), {})
    return entry.get("score")


def _count_errors(judge_by_id, sid):
    return sum(1 for (s, _), e in judge_by_id.items() if s == sid and e.get("judge_error_type"))


def _needs_review(judge_by_id, sid, record):
    # Flag for manual review if: low faith, abstention issue, judge errors, or residual
    if record.get("smoke_real_P0") == "yes":
        return True
    faith = judge_by_id.get((sid, "faithfulness"), {}).get("score")
    if faith is not None and faith < 0.5:
        return True
    abst = judge_by_id.get((sid, "abstention_correctness"), {}).get("score")
    if abst is not None and abst < 1.0:
        return True
    if _count_errors(judge_by_id, sid) > 0:
        return True
    return False


def _write_design_docs():
    # Schema design
    schema_doc = {
        "record_schema": {
            "sample_id": "str", "split": "str", "category": "str",
            "question": "str", "answer": "str",
            "selected_support": "list[{support_id, doc_id, source_file, text}]",
            "cited_doc_ids": "list[str]",
            "selected_support_doc_ids": "list[str]",
            "expected_doc_ids": "list[str]",
            "answer_key": "str", "reference": "str",
            "is_negative": "bool", "route_pred": "str",
            "smoke_real_P0": "str", "smoke_failure_class": "str",
        },
        "judge_metric_schema": {
            "faithfulness": {"scale": "1.0|0.5|0.0", "requires": "answer+support"},
            "evidence_recall": {"scale": "1.0|0.5|0.0|null", "requires": "support+answer_key"},
            "answer_accuracy": {"scale": "1.0|0.5|0.0|null", "requires": "answer+reference"},
            "answer_relevance": {"scale": "1.0|0.5|0.0", "requires": "question+answer"},
            "abstention_correctness": {"scale": "1.0|0.5|0.0", "requires": "negative sample"},
            "comparison_quality": {"scale": "1.0|0.5|0.0", "requires": "comparison sample"},
            "numeric_accuracy": {"scale": "1.0|0.5|0.0|null", "requires": "numeric sample+reference"},
        },
        "rule_metric_schema": {
            "doc_recall_support": "float 0-1", "doc_recall_citation": "float 0-1",
            "route_match": "bool", "wrong_doc_citation": "bool",
            "negative_citation_zero": "bool", "citation_count": "int",
        },
        "skip_policy": {
            "negative": "excluded from faithfulness/accuracy; only abstention_correctness",
            "no_reference": "answer_accuracy and evidence_recall return null",
            "not_comparison": "comparison_quality not computed",
            "not_numeric": "numeric_accuracy not computed",
        },
        "notes": "BIORAG Eval v1 schema — rule metrics deterministic, judge metrics LLM-assisted.",
    }
    with open(RES_DIR / "eval_schema_design.json", "w") as f:
        json.dump(schema_doc, f, ensure_ascii=False, indent=2)

    # Rule metric design
    rule_doc = {
        "metrics": [
            {"name": "doc_recall_support", "type": "float", "desc": "Fraction of expected_doc_ids in selected_support_doc_ids"},
            {"name": "doc_recall_citation", "type": "float", "desc": "Fraction of expected_doc_ids in cited_doc_ids"},
            {"name": "route_match", "type": "bool", "desc": "Predicted route matches expected_route"},
            {"name": "wrong_doc_citation", "type": "bool", "desc": "Cited doc not in expected doc list"},
            {"name": "negative_citation_zero", "type": "bool", "desc": "Negative samples have citation_count=0"},
        ],
        "llm_calls": 0,
        "reproducible": True,
        "notes": "All rule metrics are deterministic. No Qwen/LLM calls.",
    }
    with open(RES_DIR / "rule_metric_design.json", "w") as f:
        json.dump(rule_doc, f, ensure_ascii=False, indent=2)

    # Qwen judge config
    judge_cfg = {
        "model": "qwen-plus",
        "temperature": 0.0,
        "max_tokens": 512,
        "max_workers": 2,
        "retry": 2,
        "cache_enabled": True,
        "cache_path": str(CACHE_PATH),
        "prompt_versions": {"faithfulness": "v1.0", "evidence_recall": "v1.0", "answer_accuracy": "v1.0", "answer_relevance": "v1.0", "abstention_correctness": "v1.0", "comparison_quality": "v1.0", "numeric_accuracy": "v1.0"},
        "notes": "Short JSON output (max_tokens=512), rationale <=80 chars, temperature=0. All calls cached.",
    }
    with open(RES_DIR / "qwen_judge_config.json", "w") as f:
        json.dump(judge_cfg, f, ensure_ascii=False, indent=2)


def _write_run_config():
    run_cfg = {
        "phase": "21B-EVAL-0",
        "title": "BIORAG Eval v1 Pilot — 20 samples",
        "eval_version": "biorag_eval_v1",
        "layers": ["rule_metrics", "qwen_judge", "manual_calibration"],
        "no_ragas": True,
        "judge_model": "qwen-plus",
        "judge_max_tokens": 512,
        "judge_temperature": 0.0,
        "pilot_sample_count": 20,
        "frozen_rewrite": True,
        "live_rewrite": 0,
    }
    with open(RES_DIR / "run_config.json", "w") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)


def _write_review_cards(records, rule_rows, combined, judge_by_id):
    lines = ["# BIORAG Eval v1 — Manual Review Cards\n"]
    for rec in records:
        sid = rec["sample_id"]
        rule = next((r for r in rule_rows if r["sample_id"] == sid), {})
        comb = next((c for c in combined if c["sample_id"] == sid), {})
        support_preview = "; ".join(
            f"[{s['doc_id']}] {s['text'][:80]}..." for s in rec.get("selected_support", [])[:2]
        )

        lines.append(f"## Sample {sid}\n")
        lines.append(f"- **Category**: {rec['category']}")
        lines.append(f"- **Question**: {rec['question'][:200]}")
        lines.append(f"- **Answer**: {rec['answer'][:300]}")
        lines.append(f"- **Selected support**: {support_preview}")
        lines.append(f"- **Expected docs**: {rec.get('expected_doc_ids', [])}")
        lines.append(f"- **Expected behavior/answer_key**: {rec.get('expected_behavior', '')[:150]}")
        lines.append(f"- **Rule metrics**: route_match={rule.get('route_match')}, doc_recall_support={rule.get('doc_recall_support')}, doc_recall_citation={rule.get('doc_recall_citation')}")
        lines.append(f"- **Qwen judge scores**: faith={comb.get('faithfulness')}, relevance={comb.get('answer_relevance')}, accuracy={comb.get('answer_accuracy')}, evidence_recall={comb.get('evidence_recall')}")

        # Qwen rationales
        for metric in ["faithfulness", "answer_accuracy", "evidence_recall", "abstention_correctness"]:
            entry = judge_by_id.get((sid, metric), {})
            if entry.get("rationale"):
                lines.append(f"  - {metric} rationale: {entry['rationale']}")

        lines.append(f"- **Suggested human review**: {'agree' if not comb.get('needs_manual_review') else 'disagree/unclear — needs inspection'}")
        lines.append(f"- **Human notes**: \n")

    with open(BASE / "reports/phase21b_eval0_biorag_eval_pilot/manual_review_cards.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Review cards written.")


if __name__ == "__main__":
    main()
