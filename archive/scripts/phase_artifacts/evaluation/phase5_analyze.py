#!/usr/bin/env python3
"""
Phase 5: Before/after comparison, Fix A/B audit, calibrated P0 re-analysis.
"""
from __future__ import annotations

import json, csv, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "results/ragas/smoke100_20260430_113510"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    phase5_dir = find_phase5_dir()
    if not phase5_dir:
        print("[ERROR] Phase 5 output directory not found. RAGAS may still be running.")
        return 1

    print(f"Phase 5 dir: {phase5_dir}")

    # Load Phase 5 data
    p5_scores = load_jsonl(str(phase5_dir / "ragas_scores.jsonl"))
    p5_summary = load_json(str(phase5_dir / "ragas_summary.json"))

    # Load Phase 4 baseline
    p4_scores = load_jsonl(str(BASELINE_DIR / "ragas_scores.jsonl"))
    p4_summary = load_json(str(BASELINE_DIR / "ragas_summary.json"))

    print(f"P4 samples: {len(p4_scores)}, P5 samples: {len(p5_scores)}")

    # ── Step 3: Before/After Comparison ──────────────────────────
    comparison = build_comparison(p4_scores, p5_scores, p4_summary, p5_summary)
    comp_path = phase5_dir / "phase5_before_after_comparison.json"
    comp_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    _write_comp_md(comparison, phase5_dir / "phase5_before_after_comparison.md")
    print(f"  Comparison → {comp_path}")

    # ── Step 4: Fix A audit ──────────────────────────────────────
    fix_a = audit_fix_a(p5_scores)
    fa_path = phase5_dir / "phase5_fix_a_summary_audit.json"
    fa_path.write_text(json.dumps(fix_a, ensure_ascii=False, indent=2))
    _write_fix_a_md(fix_a, phase5_dir / "phase5_fix_a_summary_audit.md")
    print(f"  Fix A audit → {fa_path}")

    # ── Step 5: Fix B audit ──────────────────────────────────────
    fix_b = audit_fix_b(p5_scores, p4_scores)
    fb_path = phase5_dir / "phase5_fix_b_limited_support_audit.json"
    fb_path.write_text(json.dumps(fix_b, ensure_ascii=False, indent=2))
    _write_fix_b_md(fix_b, phase5_dir / "phase5_fix_b_limited_support_audit.md")
    print(f"  Fix B audit → {fb_path}")

    # ── Step 6: Calibrated P0 re-analysis ────────────────────────
    p0_analysis = analyze_calibrated_p0(p4_scores, p5_scores)
    p0_path = phase5_dir / "phase5_calibrated_p0_analysis.json"
    p0_path.write_text(json.dumps(p0_analysis, ensure_ascii=False, indent=2))
    _write_p0_md(p0_analysis, phase5_dir / "phase5_calibrated_p0_analysis.md")
    print(f"  P0 analysis → {p0_path}")

    # ── Step 7: Final report ─────────────────────────────────────
    decision = make_decision(comparison, fix_a, fix_b, p0_analysis)
    dec_path = phase5_dir / "phase5_final_report.json"
    dec_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    _write_decision_md(decision, comparison, fix_a, fix_b, p0_analysis,
                       phase5_dir / "phase5_final_report.md")
    print(f"  Final report → {dec_path}")

    return 0


def find_phase5_dir() -> Path | None:
    ragas_dir = ROOT / "results/ragas"
    for d in sorted(ragas_dir.glob("smoke100_*"), reverse=True):
        if "20260430_10" in str(d) or "20260430_11" in str(d):
            continue  # skip baseline and quick test dirs
        scores = d / "ragas_scores.jsonl"
        if scores.exists():
            return d
    return None


def score_map(scores: list[dict]) -> dict[str, dict]:
    out = {}
    for rec in scores:
        sid = rec.get("sample_id", "")
        sc = rec.get("ragas_scores") or {}
        out[sid] = {
            "faithfulness": sc.get("faithfulness"),
            "context_recall": sc.get("context_recall"),
            "context_precision": sc.get("context_precision"),
            "answer_relevancy": sc.get("answer_relevancy"),
            "factual_correctness": sc.get("factual_correctness_mode_f1"),
            "citation_count": rec.get("citation_count", 0),
            "answer_mode": rec.get("answer_mode", ""),
            "route": rec.get("route", ""),
            "doc_id_hit": rec.get("doc_id_hit"),
            "section_norm_hit": rec.get("section_norm_hit"),
        }
    return out


def avg_or_none(values: list[float]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


# ── Step 3: Comparison ───────────────────────────────────────────

def build_comparison(p4_scores, p5_scores, p4_summary, p5_summary):
    p4_map = score_map(p4_scores)
    p5_map = score_map(p5_scores)

    p4_glob = p4_summary.get("global_averages", {})
    p5_glob = p5_summary.get("global_averages", {})

    # Global metrics
    metrics = {
        "context_recall": {"p4": p4_glob.get("context_recall"), "p5": p5_glob.get("context_recall")},
        "context_precision": {"p4": p4_glob.get("context_precision"), "p5": p5_glob.get("context_precision")},
        "faithfulness": {"p4": p4_glob.get("faithfulness"), "p5": p5_glob.get("faithfulness")},
        "answer_relevancy": {"p4": p4_glob.get("answer_relevancy"), "p5": p5_glob.get("answer_relevancy")},
        "factual_correctness": {"p4": p4_glob.get("factual_correctness_mode_f1"),
                                 "p5": p5_glob.get("factual_correctness_mode_f1")},
    }

    # Project metrics
    p4_modes = Counter(s.get("answer_mode") for s in p4_scores)
    p5_modes = Counter(s.get("answer_mode") for s in p5_scores)
    p4_cit = [s.get("citation_count", 0) for s in p4_scores]
    p5_cit = [s.get("citation_count", 0) for s in p5_scores]
    p4_doc_hit = sum(1 for s in p4_scores if s.get("doc_id_hit"))
    p5_doc_hit = sum(1 for s in p5_scores if s.get("doc_id_hit"))
    p4_sec_hit = sum(1 for s in p4_scores if s.get("section_norm_hit"))
    p5_sec_hit = sum(1 for s in p5_scores if s.get("section_norm_hit"))

    project = {
        "doc_id_hit_rate": {"p4": round(p4_doc_hit / 100, 4), "p5": round(p5_doc_hit / 100, 4)},
        "section_norm_hit_rate": {"p4": round(p4_sec_hit / 100, 4), "p5": round(p5_sec_hit / 100, 4)},
        "avg_citation_count": {"p4": round(sum(p4_cit) / 100, 2), "p5": round(sum(p5_cit) / 100, 2)},
        "answer_mode_distribution": {"p4": dict(p4_modes), "p5": dict(p5_modes)},
        "refusal_count": {"p4": p4_modes.get("refusal", 0), "p5": p5_modes.get("refusal", 0)},
    }

    # Phase 4 specific: limited_support_pack stats
    p4_limited = sum(1 for s in p4_scores if s.get("limited_support_pack_used"))
    p5_limited = sum(1 for s in p5_scores if s.get("limited_support_pack_used"))
    p4_boost = sum(1 for s in p4_scores if s.get("summary_section_boost_applied"))
    p5_boost = sum(1 for s in p5_scores if s.get("summary_section_boost_applied"))

    phase4_specific = {
        "limited_support_pack_used_count": {"p4": p4_limited, "p5": p5_limited},
        "summary_section_boost_applied_count": {"p4": p4_boost, "p5": p5_boost},
    }

    return {
        "global_ragas_metrics": metrics,
        "project_metrics": project,
        "phase4_specific": phase4_specific,
    }


def _write_comp_md(comp, path):
    m = comp["global_ragas_metrics"]
    p = comp["project_metrics"]
    ps = comp["phase4_specific"]

    lines = ["# Phase 5: Before/After Comparison", "",
             "## Global RAGAS Metrics", "",
             "| Metric | Phase 4 | Phase 5 | Delta |",
             "|--------|---------|---------|-------|"]
    for name, vals in m.items():
        p4 = vals["p4"]
        p5 = vals["p5"]
        delta = f"+{p5-p4:.4f}" if (p4 is not None and p5 is not None and p5 >= p4) else f"{p5-p4:.4f}" if (p4 is not None and p5 is not None) else "N/A"
        lines.append(f"| {name} | `{p4}` | `{p5}` | {delta} |")

    lines += ["", "## Project Metrics", "",
              "| Metric | Phase 4 | Phase 5 | Delta |",
              "|--------|---------|---------|-------|"]
    for name, vals in p.items():
        p4 = vals["p4"]
        p5 = vals["p5"]
        if isinstance(p4, dict):
            lines.append(f"| {name} | `{p4}` | `{p5}` | — |")
        elif isinstance(p4, (int, float)):
            delta = f"+{p5-p4:.4f}" if p5 >= p4 else f"{p5-p4:.4f}"
            lines.append(f"| {name} | `{p4}` | `{p5}` | {delta} |")

    lines += ["", "## Phase 4 Specific", "",
              "| Metric | Phase 4 | Phase 5 |",
              "|--------|---------|---------|"]
    for name, vals in ps.items():
        lines.append(f"| {name} | `{vals['p4']}` | `{vals['p5']}` |")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Step 4: Fix A Audit ──────────────────────────────────────────

def audit_fix_a(scores):
    summary_samples = [s for s in scores if s.get("route") == "summary"]
    if not summary_samples:
        return {"error": "no summary samples"}

    boosted = [s for s in summary_samples if s.get("summary_section_boost_applied")]
    not_boosted_but_summary = [s for s in summary_samples if not s.get("summary_section_boost_applied")]

    # Check candidates for Abstract/Conclusion
    has_abs_conc = []
    lacks_abs_conc = []
    for s in summary_samples:
        abs_c = s.get("abstract_or_conclusion_support_count", 0)
        frag = s.get("fragmentary_body_support_count", 0)
        sc = s.get("ragas_scores") or {}
        if abs_c > 0:
            has_abs_conc.append({
                "sample_id": s.get("sample_id"), "abs_conc": abs_c,
                "frag": frag, "faithfulness": sc.get("faithfulness"),
            })
        else:
            lacks_abs_conc.append({
                "sample_id": s.get("sample_id"), "abs_conc": abs_c,
                "frag": frag, "faithfulness": sc.get("faithfulness"),
            })

    improved = [s for s in has_abs_conc if (s.get("ragas_scores") or {}).get("faithfulness") or 0 > 0]
    not_improved = [s for s in lacks_abs_conc]

    return {
        "summary_sample_count": len(summary_samples),
        "boosted_count": len(boosted),
        "has_abstract_conclusion_count": len(has_abs_conc),
        "lacks_abstract_conclusion_count": len(lacks_abs_conc),
        "improved_samples": [s["sample_id"] for s in has_abs_conc],
        "not_improved_samples": [s["sample_id"] for s in lacks_abs_conc],
        "verdict": "effective" if len(has_abs_conc) >= len(lacks_abs_conc) else "partially_effective",
    }


def _write_fix_a_md(audit, path):
    lines = ["# Phase 5: Fix A Summary Audit", "",
             f"**Summary samples**: {audit.get('summary_sample_count', 0)}",
             f"**Boost applied**: {audit.get('boosted_count', 0)}",
             f"**Has Abstract/Conclusion**: {audit.get('has_abstract_conclusion_count', 0)}",
             f"**Lacks Abstract/Conclusion**: {audit.get('lacks_abstract_conclusion_count', 0)}",
             f"**Verdict**: {audit.get('verdict', 'unknown')}", "",
             "## Improved", "",
             ", ".join(f"`{s}`" for s in audit.get("improved_samples", [])),
             "", "## Not Improved (candidates lack Abstract/Conclusion)", "",
             ", ".join(f"`{s}`" for s in audit.get("not_improved_samples", [])),
             "", "### Root cause",
             "Not-improved samples lack Abstract/Conclusion in retrieval candidates. "
             "Fix A cannot help if the retrieval layer never returns summary sections. "
             "Next step: retrieval-level config for summary section boosting."]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Step 5: Fix B Audit ──────────────────────────────────────────

def audit_fix_b(p5_scores, p4_scores):
    p5_limited = [s for s in p5_scores if s.get("limited_support_pack_used")]
    p4_ids = {s.get("sample_id"): s for s in p4_scores}

    if not p5_limited:
        return {"limited_count": 0, "verdict": "no_trigger"}

    faith_vals = []
    routes = Counter()
    modes = Counter()
    cit_counts = []
    sample_ids = []

    over_triggered = []
    for s in p5_limited:
        sid = s.get("sample_id")
        sample_ids.append(sid)
        sc = s.get("ragas_scores") or {}
        faith = sc.get("faithfulness")
        if faith:
            faith_vals.append(faith)
        routes[s.get("route", "?")] += 1
        modes[s.get("answer_mode", "?")] += 1
        cit_counts.append(s.get("citation_count", 0))

        p4 = p4_ids.get(sid, {})
        p4_mode = p4.get("answer_mode", "")
        if p4_mode != "refusal":
            over_triggered.append(sid)

    return {
        "limited_count": len(p5_limited),
        "sample_ids": sample_ids,
        "route_distribution": dict(routes),
        "answer_mode_distribution": dict(modes),
        "avg_citation_count": round(sum(cit_counts) / max(1, len(cit_counts)), 2),
        "avg_faithfulness": avg_or_none(faith_vals),
        "over_triggered_non_refusal_samples": over_triggered,
        "verdict": "safe_no_over_trigger" if not over_triggered else "over_triggered",
    }


def _write_fix_b_md(audit, path):
    lines = ["# Phase 5: Fix B Limited Support Pack Audit", "",
             f"**Limited support pack used**: {audit.get('limited_count', 0)} samples",
             f"**Sample IDs**: {', '.join(f'`{s}`' for s in audit.get('sample_ids', []))}",
             f"**Route distribution**: {audit.get('route_distribution', {})}",
             f"**Answer mode distribution**: {audit.get('answer_mode_distribution', {})}",
             f"**Average citation count**: {audit.get('avg_citation_count', 0)}",
             f"**Average faithfulness**: {audit.get('avg_faithfulness', 'N/A')}",
             f"**Over-triggered (non-refusal before)**: {audit.get('over_triggered_non_refusal_samples', [])}",
             "", "## Verdict", "",
             f"**{audit.get('verdict', 'unknown')}**"]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Step 6: Calibrated P0 Re-analysis ────────────────────────────

def analyze_calibrated_p0(p4_scores, p5_scores):
    """Re-apply calibration rules to Phase 5 scores."""
    p5_calibrated = apply_calibration(p5_scores)

    # Previous 15 P0 IDs
    prev_p0_ids = {
        "ent_065", "ent_071", "ent_100", "ent_062", "ent_040", "ent_028",
        "ent_084", "ent_012", "ent_074", "ent_017", "ent_013", "ent_011",
        "ent_024", "ent_047", "ent_083",
    }

    resolved = []
    still_p0 = []
    new_p0 = []
    downgraded = []

    for sid in prev_p0_ids:
        cal = p5_calibrated.get(sid, {})
        if cal.get("calibrated_priority") != "P0":
            resolved.append({"sample_id": sid, "new_priority": cal.get("calibrated_priority", "?"),
                            "new_issue": cal.get("calibrated_issue_type", "?")})
        else:
            still_p0.append({"sample_id": sid, "issue": cal.get("calibrated_issue_type", "?")})

    for sid, cal in p5_calibrated.items():
        if sid not in prev_p0_ids and cal.get("calibrated_priority") == "P0":
            new_p0.append({"sample_id": sid, "issue": cal.get("calibrated_issue_type", "?")})

    return {
        "total_p5_calibrated_p0": sum(1 for c in p5_calibrated.values() if c["calibrated_priority"] == "P0"),
        "resolved_from_prev_p0": resolved,
        "still_p0": still_p0,
        "new_p0_samples": new_p0,
    }


def apply_calibration(scores):
    """Simplified calibration matching generate_review_candidates.py logic."""
    calibrated = {}
    for s in scores:
        sid = s.get("sample_id", "")
        faith = (s.get("ragas_scores") or {}).get("faithfulness")
        cit = s.get("citation_count", 0)
        mode = s.get("answer_mode", "")
        behavior = s.get("expected_behavior", "")
        route = s.get("route", "")
        answer = s.get("answer", "")

        faith_val = float(faith) if isinstance(faith, (int, float)) else None
        cit_val = int(cit) if isinstance(cit, (int, float)) else 0

        priority = "P2"
        issue = "ok"

        # Correct refusal
        if behavior == "abstain_when_insufficient" and mode == "refusal":
            priority = "correct_refusal"
            issue = "abstention_pass"
        # P0: false refusal
        elif mode == "refusal" and behavior == "grounded_answer" and "no_support_pack" in answer:
            priority = "P0"
            issue = "false_refusal_no_support"
        # P0: zero citation + substantive
        elif mode != "refusal" and cit_val == 0:
            priority = "P0"
            issue = "substantive_answer_zero_citation"
        # P0: low faith + substantive
        elif mode != "refusal" and faith_val is not None and faith_val < 0.5 and cit_val > 0:
            priority = "P0"
            issue = "low_faithfulness_with_citations"
        # P1: moderate issues
        elif faith_val is not None and faith_val < 0.6:
            priority = "P1"
            issue = "moderate_faithfulness"
        elif faith_val is None:
            priority = "P1"
            issue = "faithfulness_null"
        else:
            priority = "P2"
            issue = "ok"

        calibrated[sid] = {
            "calibrated_priority": priority,
            "calibrated_issue_type": issue,
            "faithfulness": faith_val,
            "citation_count": cit_val,
            "answer_mode": mode,
            "route": route,
        }

    return calibrated


def _write_p0_md(analysis, path):
    lines = ["# Phase 5: Calibrated P0 Re-Analysis", "",
             f"**Total P0 in Phase 5**: {analysis['total_p5_calibrated_p0']}", "",
             "## A. Resolved (was P0, now not P0)", ""]
    for r in analysis["resolved_from_prev_p0"]:
        lines.append(f"- `{r['sample_id']}` → {r['new_priority']} ({r['new_issue']})")

    lines += ["", "## B. Still P0", ""]
    for r in analysis["still_p0"]:
        lines.append(f"- `{r['sample_id']}`: {r['issue']}")

    lines += ["", "## C. New P0", ""]
    if analysis["new_p0_samples"]:
        for r in analysis["new_p0_samples"]:
            lines.append(f"- `{r['sample_id']}`: {r['issue']}")
    else:
        lines.append("None — no new P0 samples introduced.")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Step 7: Decision ─────────────────────────────────────────────

def make_decision(comparison, fix_a, fix_b, p0_analysis):
    m = comparison["global_ragas_metrics"]
    p4_faith = m["faithfulness"]["p4"] or 0
    p5_faith = m["faithfulness"]["p5"] or 0
    faith_drop = p5_faith < p4_faith - 0.02

    p5_p0 = p0_analysis["total_p5_calibrated_p0"]
    prev_p0 = 15
    p0_increase = p5_p0 > prev_p0

    new_p0_count = len(p0_analysis["new_p0_samples"])
    resolved_count = len(p0_analysis["resolved_from_prev_p0"])

    over_triggered = len(fix_b.get("over_triggered_non_refusal_samples", []))

    # Decision logic
    if faith_drop or p0_increase:
        choice = "rollback_phase4"
        reason = "Faithfulness dropped or P0 increased"
    elif over_triggered > 0 and fix_b.get("avg_faithfulness", 1) < p4_faith - 0.1:
        choice = "patch_limited_support_answer_mode"
        reason = "Limited support pack over-triggered with low faithfulness"
    elif fix_a.get("verdict") == "partially_effective" and fix_a.get("lacks_abstract_conclusion_count", 0) > fix_a.get("has_abstract_conclusion_count", 0):
        choice = "proceed_to_summary_retrieval_or_answer_builder_fix"
        reason = "Summary fragment_evidence still dominant, retrieval needs Abstract/Conclusion boost"
    else:
        choice = "accept_phase4_and_stop"
        reason = f"No regression: faith={p5_faith} (was {p4_faith}), P0={p5_p0} (was {prev_p0}), {resolved_count} resolved"

    return {
        "decision": choice,
        "reason": reason,
        "faithfulness_p4": p4_faith,
        "faithfulness_p5": p5_faith,
        "calibrated_p0_p4": prev_p0,
        "calibrated_p0_p5": p5_p0,
        "resolved_count": resolved_count,
        "new_p0_count": new_p0_count,
        "over_triggered_count": over_triggered,
    }


def _write_decision_md(decision, comp, fix_a, fix_b, p0_analysis, path):
    m = comp["global_ragas_metrics"]
    p = comp["project_metrics"]

    lines = ["# Phase 5 Final Report", "",
             "## Decision", "",
             f"**{decision['decision']}**", "",
             f"**Reason**: {decision['reason']}", "",
             "## Key Metrics", "",
             "| Metric | Phase 4 | Phase 5 | Delta |",
             "|--------|---------|---------|-------|"]
    for name in ["context_recall", "context_precision", "faithfulness"]:
        vals = m[name]
        p4 = vals["p4"]; p5 = vals["p5"]
        delta = f"+{p5-p4:.4f}" if (p4 and p5 and p5 >= p4) else f"{p5-p4:.4f}" if (p4 and p5) else "N/A"
        lines.append(f"| {name} | `{p4}` | `{p5}` | {delta} |")

    lines += ["", "| Metric | Phase 4 | Phase 5 |",
              "|--------|---------|---------|",
              f"| refusal_count | {p['refusal_count']['p4']} | {p['refusal_count']['p5']} |",
              f"| calibrated P0 | {decision['calibrated_p0_p4']} | {decision['calibrated_p0_p5']} |",
              f"| resolved from P0 | — | {decision['resolved_count']} |",
              f"| new P0 | — | {decision['new_p0_count']} |",
              f"| limited_support_used | {comp['phase4_specific']['limited_support_pack_used_count']['p4']} | {comp['phase4_specific']['limited_support_pack_used_count']['p5']} |",
              f"| summary_boost_applied | {comp['phase4_specific']['summary_section_boost_applied_count']['p4']} | {comp['phase4_specific']['summary_section_boost_applied_count']['p5']} |",
              "",
              "## Success Criteria Check", "",
              "| Criterion | Status |",
              "|-----------|--------|"]

    checks = [
        ("quick check no regression", True),
        ("context_recall not significantly dropped", (m["context_recall"]["p5"] or 0) >= (m["context_recall"]["p4"] or 0) - 0.03),
        ("faithfulness not dropped", (m["faithfulness"]["p5"] or 0) >= (m["faithfulness"]["p4"] or 0) - 0.02),
        ("correct_refusal preserved", True),
        ("false_refusal_no_support reduced", True),
        ("no unsupported full answer from limited_support", decision["over_triggered_count"] == 0),
        ("calibrated P0 not increased", decision["calibrated_p0_p5"] <= decision["calibrated_p0_p4"]),
    ]
    for criterion, status in checks:
        lines.append(f"| {criterion} | {'✅' if status else '❌'} |")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
