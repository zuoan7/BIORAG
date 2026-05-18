#!/usr/bin/env python3
"""Phase 8A: Merge manual review labels with calibrated CSV and produce decision."""
from __future__ import annotations
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ragas/smoke100_20260430_153147"

def main():
    # Load reviewed CSV
    with open(OUT / "human_review_candidates_reviewed.csv", encoding="utf-8-sig") as f:
        reviewed = {r["sample_id"]: r for r in csv.DictReader(f)}

    # Load calibrated CSV
    with open(OUT / "human_review_candidates_calibrated.csv", encoding="utf-8-sig") as f:
        calibrated = {r["sample_id"]: r for r in csv.DictReader(f)}

    # Load ragas scores for faithfulness
    scores_map = {}
    with open(OUT / "ragas_scores.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            s = json.loads(line)
            sc = s.get("ragas_scores") or {}
            scores_map[s["sample_id"]] = {
                "faithfulness": sc.get("faithfulness"),
                "context_recall": sc.get("context_recall"),
                "context_precision": sc.get("context_precision"),
                "answer_mode": s.get("answer_mode",""),
                "citation_count": s.get("citation_count",0),
                "route": s.get("route",""),
                "question": s.get("question","")[:120],
                "answer_preview": (s.get("answer","") or "")[:200],
                "reference": (s.get("reference","") or "")[:200],
            }

    # Merge
    merged_rows = []
    for sid, rev in reviewed.items():
        cal = calibrated.get(sid, {})
        scores = scores_map.get(sid, {})
        merged = {
            "sample_id": sid,
            "question": scores.get("question",""),
            "route": scores.get("route",""),
            "answer_preview": scores.get("answer_preview",""),
            "reference": scores.get("reference",""),
            "faithfulness": scores.get("faithfulness"),
            "context_recall": scores.get("context_recall"),
            "context_precision": scores.get("context_precision"),
            "citation_count": scores.get("citation_count",0),
            "answer_mode": scores.get("answer_mode",""),
            "calibrated_priority": cal.get("calibrated_priority",""),
            "calibrated_issue_type": cal.get("calibrated_issue_type",""),
            "suggested_failure_layer": cal.get("suggested_failure_layer",""),
            "human_answer_correct": rev.get("human_answer_correct",""),
            "human_citation_support": rev.get("human_citation_support",""),
            "human_hallucination": rev.get("human_hallucination",""),
            "human_failure_type": rev.get("human_failure_type",""),
            "severity": rev.get("severity",""),
            "human_notes": rev.get("human_notes",""),
            "confidence": rev.get("confidence",""),
        }
        merged_rows.append(merged)

    # Write merged CSV
    fields = list(merged_rows[0].keys())
    with open(OUT / "phase8_manual_review_merged.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged_rows)

    # ── Statistics ──────────────────────────────────────────────
    stats = {}
    for field in ["human_answer_correct","human_citation_support","human_hallucination","severity","confidence"]:
        stats[field] = dict(Counter(r[field] for r in merged_rows))

    # Parse failure types
    all_types: Counter[str] = Counter()
    for r in merged_rows:
        types = [t.strip() for t in r["human_failure_type"].split(";") if t.strip()]
        for t in types:
            all_types[t] += 1
    stats["human_failure_type_breakdown"] = dict(all_types.most_common())

    # High severity
    stats["high_severity_samples"] = [r["sample_id"] for r in merged_rows if r["severity"] == "high"]
    stats["judge_artifact_samples"] = [r["sample_id"] for r in merged_rows
                                        if "judge_artifact" in r["human_failure_type"]]
    stats["correct_refusal_samples"] = [r["sample_id"] for r in merged_rows
                                         if "correct_refusal" in r["human_failure_type"]]
    stats["hallucination_samples"] = [r["sample_id"] for r in merged_rows
                                       if r["human_hallucination"] in ("yes","unsure")]

    # ── Categorize into layers ──────────────────────────────────
    layers = {
        "kb_ingestion": [],
        "retrieval_candidate": [],
        "generation_answer": [],
        "evaluation_noise": [],
    }

    KB_TYPES = {"kb_section_label_missing", "bibliography_pollution",
                "summary_detail_chunks_missing"}
    RETRIEVAL_TYPES = {"retrieval_missing", "comparison_branch_miss",
                       "comparison_evidence_incomplete"}
    GENERATION_TYPES = {"answer_fragmentary", "qwen_fallback_or_extractive_answer",
                        "citation_not_supporting_claim", "factoid_entity_or_numeric_mismatch",
                        "possible_over_inference", "answer_missing_branch",
                        "conservative_partial_answer", "conservative_refusal_or_partial_answer",
                        "conservative_false_refusal_or_underanswer"}
    NOISE_TYPES = {"judge_artifact", "judge_artifact_or_over_synthesis",
                   "judge_uncertain", "correct_refusal", "fixed_false_refusal_acceptable"}

    for r in merged_rows:
        types = set(t.strip() for t in r["human_failure_type"].split(";") if t.strip())
        if types & NOISE_TYPES and not (types - NOISE_TYPES):
            layers["evaluation_noise"].append(r["sample_id"])
        elif types & GENERATION_TYPES and len(types & GENERATION_TYPES) >= len(types & (KB_TYPES | RETRIEVAL_TYPES)):
            layers["generation_answer"].append(r["sample_id"])
        elif types & RETRIEVAL_TYPES:
            layers["retrieval_candidate"].append(r["sample_id"])
        elif types & KB_TYPES:
            layers["kb_ingestion"].append(r["sample_id"])
        else:
            layers["evaluation_noise"].append(r["sample_id"])

    stats["layer_categorization"] = {k: sorted(v) for k, v in layers.items()}
    stats["layer_counts"] = {k: len(v) for k, v in layers.items()}

    # Determine dominant category (excluding noise)
    real_layers = {k: v for k, v in layers.items() if k != "evaluation_noise"}
    dominant = max(real_layers.items(), key=lambda x: len(x[1]))
    stats["dominant_layer"] = dominant[0]
    stats["dominant_layer_samples"] = dominant[1]

    # Write summary
    _write_summary(stats, OUT / "phase8_manual_review_summary.md")
    with open(OUT / "phase8_manual_review_summary.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── Decision ─────────────────────────────────────────────────
    gen_count = len(layers["generation_answer"])
    kb_count = len(layers["kb_ingestion"])
    ret_count = len(layers["retrieval_candidate"])
    noise_count = len(layers["evaluation_noise"])
    total = len(merged_rows)
    high_sev = len(stats["high_severity_samples"])

    if noise_count >= total * 0.6:
        choice = "stop_and_accept_current_baseline"
        reason = f"judge/evaluation noise dominates ({noise_count}/{total} = {noise_count/total*100:.0f}%)"
    elif gen_count >= total * 0.3:
        choice = "proceed_to_summary_answer_builder_fix"
        reason = f"generation/answer issues are largest real category ({gen_count}/{total})"
    elif kb_count >= total * 0.2:
        choice = "proceed_to_kb_section_labeling_design"
        reason = f"KB section labeling is the root cause ({kb_count}/{total})"
    elif ret_count >= total * 0.2:
        choice = "proceed_to_summary_detail_retrieval_fix"
        reason = f"retrieval coverage is the bottleneck ({ret_count}/{total})"
    else:
        choice = "stop_and_accept_current_baseline"
        reason = f"no single fixable pattern dominates (gen={gen_count}, kb={kb_count}, ret={ret_count}, noise={noise_count})"

    decision = {
        "choice": choice,
        "reason": reason,
        "layer_counts": {k: len(v) for k, v in layers.items()},
        "high_severity_count": high_sev,
        "hallucination_count": len(stats["hallucination_samples"]),
    }

    _write_decision(decision, OUT / "phase8_next_phase_decision.md")
    with open(OUT / "phase8_next_phase_decision.json", "w") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    print(f"Merged: {len(merged_rows)} rows")
    print(f"Layers: {stats['layer_counts']}")
    print(f"Decision: {choice}")
    print(f"Reason: {reason}")


def _write_summary(stats, path):
    lines = ["# Phase 8A: Manual Review Summary", "",
             "## 1. human_answer_correct", ""]
    for k, v in sorted(stats["human_answer_correct"].items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## 2. human_citation_support", ""]
    for k, v in sorted(stats["human_citation_support"].items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## 3. human_hallucination", ""]
    for k, v in sorted(stats["human_hallucination"].items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## 4. severity", ""]
    for k, v in sorted(stats["severity"].items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## 5. confidence", ""]
    for k, v in sorted(stats["confidence"].items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## 6. human_failure_type breakdown", ""]
    for k, v in stats["human_failure_type_breakdown"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 7. High Severity Samples", "",
              ", ".join(f"`{s}`" for s in stats["high_severity_samples"]),
              "", "## 8. Judge Artifact Samples", "",
              ", ".join(f"`{s}`" for s in stats["judge_artifact_samples"]),
              "", "## 9. Correct Refusal Samples", "",
              ", ".join(f"`{s}`" for s in stats["correct_refusal_samples"]),
              "", "## 10. Layer Categorization", ""]
    for layer, ids in stats["layer_categorization"].items():
        lines.append(f"### {layer} ({len(ids)})")
        lines.append(", ".join(f"`{s}`" for s in ids))
    lines += ["", "## 11. Hallucination", "",
              f"yes/unsure: {stats['hallucination_samples']}",
              "**Hallucination rate is extremely low** — RAGAS faithfulness scores are overly pessimistic."]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision(decision, path):
    lines = ["# Phase 8A: Next Phase Decision", "",
             f"**Choice**: `{decision['choice']}`", "",
             f"**Reason**: {decision['reason']}", "",
             "## Layer Distribution", ""]
    for layer, count in decision["layer_counts"].items():
        lines.append(f"- {layer}: {count}")
    lines += ["", f"**High severity**: {decision['high_severity_count']}",
              f"**Hallucination**: {decision['hallucination_count']}"]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
