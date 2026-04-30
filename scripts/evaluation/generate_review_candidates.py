#!/usr/bin/env python3
"""
生成人工审核候选集 CSV（含校准后优先级）。

候选来源:
  1. faithfulness 最低 top 10
  2. factual_correctness 最低 top 10
  3. answer_relevancy 最低 top 10
  4. context_recall 最低 top 10
  5. 随机高分样本 10 条
  6. 所有 zero citation 样本
  7. 所有 abstain/refusal 样本中 RAGAS 分数异常的样本

校准规则 (P0 收紧):
  - 正确拒答 (abstain + 充分表达证据不足) → 不标 P0
  - factual_correctness-only → 不标 P0
  - answer_relevancy-only → 不标 P0
  - faithfulness >= 0.75 → 不标 P0
  - random_high_score_spot_check → 不标 P0

输出:
  - human_review_candidates_calibrated.csv

使用方式:
  python scripts/evaluation/generate_review_candidates.py \
    --input results/ragas/<ts>/ragas_scores.jsonl \
    --output-dir results/ragas/<ts>/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate calibrated human review candidate CSV")
    p.add_argument("--input", required=True, help="Path to ragas_scores.jsonl (or ragas_eval_joined.jsonl)")
    p.add_argument("--output-dir", required=True, help="Output directory for CSV")
    p.add_argument("--seed", type=int, default=42, help="Random seed for high-score sampling")
    return p.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def score_val(item: dict[str, Any], metric: str) -> float:
    s = (item.get("ragas_scores") or {}).get(metric)
    if isinstance(s, (int, float)):
        return float(s)
    return -1.0


# ─── Refusal detection ────────────────────────────────────────────

_REFUSAL_KEYWORDS = (
    "证据不足", "无法确认", "无法可靠", "不能基于当前文库",
    "没有检索到可支撑", "当前知识库中没有", "当前知识库证据不足",
    "no_support_pack", "缺少交叉证据", "无法完成完整",
    "文库中未提供", "文库中没有", "不能确认文库中包含",
    "当前文库只能提供间接", "当前知识库只能支持有限总结",
    "证据限制", "comparison_evidence_incomplete",
)


def is_refusal_answer(answer: str) -> bool:
    """Detect if answer text expresses inability to answer."""
    if not answer or not answer.strip():
        return True
    lowered = answer.strip()
    # Strong refusal signals
    strong = (
        "当前知识库证据不足", "证据不足，无法", "no_support_pack",
        "无法基于已检索证据回答", "不能基于当前文库回答",
        "无法可靠作答",
    )
    return any(kw in lowered for kw in strong)


def is_effective_abstention(item: dict[str, Any]) -> bool:
    """Check if this is a genuine refusal, not a false positive.

    A correct refusal satisfies:
      - expected_behavior == abstain_when_insufficient
      - answer clearly states evidence is insufficient
      - does NOT fabricate specific facts
    """
    behavior = item.get("expected_behavior", "")
    answer = item.get("answer", "")
    mode = item.get("answer_mode", "")

    # Must be expected abstention
    if behavior != "abstain_when_insufficient":
        return False

    # Answer must express refusal
    if mode != "refusal" and not is_refusal_answer(answer):
        return False

    # Citation count = 0 is normal for correct refusals
    return True


def is_substantive_non_refusal(item: dict[str, Any]) -> bool:
    """Answer is not a refusal — model attempted to answer."""
    mode = item.get("answer_mode", "")
    if mode == "refusal":
        return False
    if mode in ("empty", "error"):
        return False
    # Even if answer contains caveats, it's substantive if not a clear refusal
    return not is_refusal_answer(item.get("answer", ""))


# ─── Factoid fact detection ──────────────────────────────────────

_FACTOID_PATTERNS = (
    r"\d+(?:\.\d+)?\s*(?:倍|fold|g/L|mg|mM|μM|nM|%|h|min)",
    r"(?:敲除|缺失|突变|过表达|上调|下调)\s*\w+",
    r"(?:基因|酶|受体|转运体|激酶|脱氢酶|合酶|转移酶|水解酶)",
    r"(?:抗性|耐药|敏感|耐受)",
    r"(?:pfkA|zwf|CRISPR|Cas\d|HAC1|UGGT|FadL|ABC|MFS|RND)",
)


def has_specific_facts(answer: str) -> bool:
    """Check if answer contains specific numeric/entity claims."""
    for pat in _FACTOID_PATTERNS:
        if re.search(pat, answer, re.IGNORECASE):
            return True
    return False


# ─── Calibrated priority logic ────────────────────────────────────

def calibrate_priority(item: dict[str, Any], original_priority: str,
                       trigger_metric: str) -> dict[str, Any]:
    """Apply calibrated P0 rules. Returns dict of calibrated fields."""
    scores = item.get("ragas_scores") or {}
    faith = scores.get("faithfulness")
    fc = scores.get("factual_correctness_mode_f1")  # RAGAS 0.4.3 naming
    if fc is None:
        fc = scores.get("factual_correctness")
    arel = scores.get("answer_relevancy")
    crec = scores.get("context_recall")
    cit = item.get("citation_count", 0)
    route = item.get("route", "")
    answer = item.get("answer", "")
    behavior = item.get("expected_behavior", "")
    mode = item.get("answer_mode", "")

    faith_val = float(faith) if isinstance(faith, (int, float)) else None
    cit_val = int(cit) if isinstance(cit, (int, float)) else 0

    calibrated_priority = original_priority
    calibrated_issue = "uncalibrated"
    is_correct_refusal = False
    is_metric_only = False
    needs_manual_review = True
    suggested_layer = "unknown"

    # ── Rule 1: Correct refusal ────────────────────────────────
    if is_effective_abstention(item):
        is_correct_refusal = True
        calibrated_priority = "correct_refusal"
        calibrated_issue = "abstention_pass"
        needs_manual_review = False
        suggested_layer = "none"
        return _build_calibration(calibrated_priority, calibrated_issue,
                                  is_correct_refusal, is_metric_only,
                                  needs_manual_review, suggested_layer)

    # ── Rule 2: Downgrade metric-only triggers ─────────────────
    metric_only_metrics = {"factual_correctness", "factual_correctness_mode_f1",
                           "answer_relevancy"}
    if trigger_metric in metric_only_metrics:
        is_metric_only = True
        calibrated_priority = _downgrade(original_priority)
        calibrated_issue = f"{trigger_metric}_only_trigger"
        suggested_layer = "judge_llm_limitation"

    # ── Rule 3: High faithfulness → downgrade ──────────────────
    if faith_val is not None and faith_val >= 0.75:
        calibrated_priority = _downgrade(original_priority)
        calibrated_issue = "faithfulness_acceptable"
        needs_manual_review = False

    # ── Rule 4: Spot check → always P2 ─────────────────────────
    if trigger_metric == "spot_check":
        calibrated_priority = "P2"
        calibrated_issue = "random_high_score_spot_check"
        needs_manual_review = False

    # ── Rule 5a: faithfulness null (judge error) → not P0 ────
    if faith_val is None:
        if is_substantive_non_refusal(item) and cit_val > 0:
            calibrated_priority = "P1"
            calibrated_issue = "faithfulness_null_judge_error"
            suggested_layer = "judge_llm_limitation"

    # ── Rule 5: Strict P0 criteria ─────────────────────────────
    # P0-A: Non-refusal + zero citation
    elif is_substantive_non_refusal(item) and cit_val == 0:
        calibrated_priority = "P0"
        calibrated_issue = "substantive_answer_zero_citation"
        suggested_layer = "retrieval_or_support_selection"

    # P0-B: Non-refusal + faithfulness < 0.5 + citation > 0
    elif (is_substantive_non_refusal(item) and faith_val is not None
          and faith_val < 0.5 and cit_val > 0):
        calibrated_priority = "P0"
        calibrated_issue = "low_faithfulness_with_citations"
        suggested_layer = "generation_not_grounded"

    # P0-C: Comparison route + explicitly mentions missing/partial + still answers
    elif route == "comparison":
        has_missing_hint = any(
            kw in answer for kw in ("comparison_evidence_incomplete",
                                    "只能进行有限比较", "只能支持部分回答",
                                    "未覆盖", "证据限制"))
        if faith_val is not None and faith_val < 0.5 and has_missing_hint:
            calibrated_priority = "P0"
            calibrated_issue = "comparison_incomplete_but_answered"
            suggested_layer = "generation"

    # P0-D: no_support_pack refusal on grounded_answer (false refusal)
    elif mode == "refusal" and behavior == "grounded_answer":
        if "no_support_pack" in answer:
            calibrated_priority = "P0"
            calibrated_issue = "false_refusal_no_support"
            suggested_layer = "retrieval"
        elif cit_val == 0:
            calibrated_priority = "P0"
            calibrated_issue = "false_refusal_zero_citation"
            suggested_layer = "retrieval_or_support_selection"

    # P0-E: Factoid with specific facts + low faith + few citations
    elif (route == "factoid" and faith_val is not None and faith_val < 0.5
          and has_specific_facts(answer) and cit_val <= 2):
        calibrated_priority = "P0"
        calibrated_issue = "factoid_facts_weakly_supported"
        suggested_layer = "generation_or_citation_selection"

    # ── Default: keep original if not overridden ───────────────
    if calibrated_priority == original_priority and calibrated_issue == "uncalibrated":
        calibrated_issue = _map_original_issue(original_priority, trigger_metric)
        if faith_val is not None and faith_val >= 0.5:
            suggested_layer = "possible_judge_noise"

    # P1/P2 mapping
    if calibrated_priority not in ("P0", "P1", "P2", "correct_refusal"):
        calibrated_priority = "P1"

    return _build_calibration(calibrated_priority, calibrated_issue,
                              is_correct_refusal, is_metric_only,
                              needs_manual_review, suggested_layer)


def _build_calibration(priority: str, issue: str, correct_refusal: bool,
                       metric_only: bool, needs_review: bool,
                       layer: str) -> dict[str, Any]:
    return {
        "calibrated_priority": priority,
        "calibrated_issue_type": issue,
        "is_correct_refusal": correct_refusal,
        "is_metric_only_trigger": metric_only,
        "needs_manual_review": needs_review,
        "suggested_failure_layer": layer,
    }


def _downgrade(original: str) -> str:
    if original == "P0":
        return "P1"
    if original == "P1":
        return "P2"
    return "P2"


def _map_original_issue(priority: str, trigger: str) -> str:
    if priority == "P0":
        return "p0_by_original_rule"
    if priority == "P1":
        return "p1_by_original_rule"
    return "p2_spot_check"


# ─── Original (uncalibrated) logic ────────────────────────────────

def original_suspected_issue(item: dict[str, Any], metric: str, score: float) -> str:
    empty_ctx = item.get("empty_context", False)
    answer_mode = item.get("answer_mode", "")
    doc_hit = item.get("doc_id_hit", True)
    if answer_mode in ("empty", "error"):
        return "empty_context"
    if answer_mode == "refusal":
        return "refusal_case"
    if empty_ctx:
        return "empty_context"
    if not doc_hit and metric in ("context_recall", "context_precision"):
        return "retrieval_missing"
    if metric == "faithfulness" and score < 0.5:
        return "answer_not_grounded"
    if metric == "factual_correctness" and score < 0.5:
        return "possible_reference_mismatch" if doc_hit else "retrieval_missing"
    if metric == "answer_relevancy" and score < 0.5:
        return "irrelevant_context"
    if score < 0.3:
        return "judge_uncertain"
    return "answer_incomplete"


def original_review_priority(score: float, metric: str) -> str:
    if metric in ("faithfulness", "factual_correctness") and score < 0.4:
        return "P0"
    if score < 0.5:
        return "P1"
    return "P2"


# ─── Candidate building ───────────────────────────────────────────

def build_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    def add(item: dict[str, Any], priority: str, issue: str, trigger_metric: str):
        sid = item.get("sample_id", "")
        if sid in seen_ids:
            return
        seen_ids.add(sid)
        scores = item.get("ragas_scores") or {}

        # Apply calibration
        calib = calibrate_priority(item, priority, trigger_metric)

        candidates.append({
            # Core fields
            "sample_id": sid,
            "question": item.get("question", ""),
            "route": item.get("route", ""),
            "scenario": item.get("scenario", ""),
            "expected_behavior": item.get("expected_behavior", ""),
            "answer": item.get("answer", ""),
            "reference": item.get("reference", ""),
            "contexts_preview": _truncate(" | ".join((item.get("contexts") or [])[:5]), 500),
            # RAGAS scores
            "faithfulness": scores.get("faithfulness"),
            "factual_correctness": scores.get("factual_correctness_mode_f1",
                                              scores.get("factual_correctness")),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_recall": scores.get("context_recall"),
            "context_precision": scores.get("context_precision"),
            # Project metrics
            "doc_id_hit": item.get("doc_id_hit"),
            "section_norm_hit": item.get("section_norm_hit"),
            "citation_count": item.get("citation_count", 0),
            "answer_mode": item.get("answer_mode", ""),
            # Original (pre-calibration)
            "review_priority": priority,
            "suspected_issue_type": issue,
            "trigger_metric": trigger_metric,
            # Calibrated
            "calibrated_priority": calib["calibrated_priority"],
            "calibrated_issue_type": calib["calibrated_issue_type"],
            "is_correct_refusal": calib["is_correct_refusal"],
            "is_metric_only_trigger": calib["is_metric_only_trigger"],
            "needs_manual_review": calib["needs_manual_review"],
            "suggested_failure_layer": calib["suggested_failure_layer"],
            # Human-review blanks
            "human_answer_correct": "",
            "human_citation_support": "",
            "human_hallucination": "",
            "human_notes": "",
            "severity": "",
        })

    # 1. Bottom 10 per metric (original triggers)
    fc_key = "factual_correctness_mode_f1"
    for metric in ["faithfulness", fc_key, "answer_relevancy", "context_recall"]:
        sorted_recs = sorted(records, key=lambda r: score_val(r, metric))
        for item in sorted_recs[:10]:
            sc = score_val(item, metric)
            priority = original_review_priority(sc, metric)
            issue = original_suspected_issue(item, metric, sc)
            add(item, priority, issue, metric)

    # 2. Random high-score spot checks (P2)
    scored = [(score_val(r, "faithfulness"), r) for r in records
              if score_val(r, "faithfulness") > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_quartile = [r for _, r in scored[: max(len(scored) // 4, 10)]]
    rng = random.Random(42)
    for item in rng.sample(top_quartile, min(10, len(top_quartile))):
        add(item, "P2", "random_high_score_spot_check", "spot_check")

    # 3. All zero-citation samples
    for item in records:
        if item.get("citation_count", 0) == 0:
            scores = item.get("ragas_scores") or {}
            faith = scores.get("faithfulness")
            fc = scores.get(fc_key, scores.get("factual_correctness"))
            sc = min(faith or 0, fc or 0)
            add(item,
                "P0" if (faith or 0) < 0.4 or (fc or 0) < 0.4 else "P1",
                "zero_citation" if sc < 0.5 else "zero_citation_acceptable",
                "zero_citation")

    # 4. Abstain/refusal with anomalous RAGAS scores
    for item in records:
        if item.get("answer_mode") in ("refusal", "empty"):
            scores = item.get("ragas_scores") or {}
            fc = scores.get(fc_key, scores.get("factual_correctness"))
            faith = scores.get("faithfulness")
            if (isinstance(fc, (int, float)) and fc > 0.6) or \
               (isinstance(faith, (int, float)) and faith > 0.7):
                add(item, "P1", "refusal_with_high_ragas_score", "refusal_anomaly")

    return candidates


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


CSV_FIELDS = [
    "sample_id", "question", "route", "scenario", "expected_behavior",
    "answer", "reference", "contexts_preview",
    "faithfulness", "factual_correctness", "answer_relevancy",
    "context_recall", "context_precision",
    "doc_id_hit", "section_norm_hit", "citation_count", "answer_mode",
    "review_priority", "suspected_issue_type", "trigger_metric",
    "calibrated_priority", "calibrated_issue_type",
    "is_correct_refusal", "is_metric_only_trigger",
    "needs_manual_review", "suggested_failure_layer",
    "human_answer_correct", "human_citation_support",
    "human_hallucination", "human_notes", "severity",
]


def build_calibration_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate calibration statistics."""
    original_p0 = [c for c in candidates if c["review_priority"] == "P0"]
    calibrated_p0 = [c for c in candidates if c["calibrated_priority"] == "P0"]
    downgraded = [c for c in candidates
                  if c["review_priority"] == "P0" and c["calibrated_priority"] != "P0"]
    correct_refusals = [c for c in candidates if c["is_correct_refusal"] is True]
    metric_only = [c for c in candidates if c["is_metric_only_trigger"] is True]
    no_review = [c for c in candidates if not c["needs_manual_review"]]

    downgrade_reasons: dict[str, int] = {}
    for c in downgraded:
        reason = c["calibrated_issue_type"]
        downgrade_reasons[reason] = downgrade_reasons.get(reason, 0) + 1

    return {
        "total_candidates": len(candidates),
        "original_p0_count": len(original_p0),
        "calibrated_p0_count": len(calibrated_p0),
        "downgraded_from_p0_count": len(downgraded),
        "downgraded_sample_ids": [c["sample_id"] for c in downgraded],
        "downgrade_reasons": downgrade_reasons,
        "calibrated_p0_sample_ids": [c["sample_id"] for c in calibrated_p0],
        "calibrated_p0_reasons": {c["sample_id"]: c["calibrated_issue_type"]
                                   for c in calibrated_p0},
        "correct_refusal_count": len(correct_refusals),
        "correct_refusal_ids": [c["sample_id"] for c in correct_refusals],
        "metric_only_trigger_count": len(metric_only),
        "no_manual_review_needed_count": len(no_review),
    }


# ─── Main ──────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    print(f"[review_candidates] Loaded {len(records)} records from {args.input}")

    candidates = build_candidates(records)
    print(f"[review_candidates] Generated {len(candidates)} candidates")

    # Write calibrated CSV
    csv_path = output_dir / "human_review_candidates_calibrated.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for c in candidates:
            writer.writerow(c)

    # Calibration summary
    summary = build_calibration_summary(candidates)

    print(f"\n[Calibration Summary]")
    print(f"  Total candidates:           {summary['total_candidates']}")
    print(f"  Original P0:                {summary['original_p0_count']}")
    print(f"  Calibrated P0:              {summary['calibrated_p0_count']}")
    print(f"  Downgraded from P0:         {summary['downgraded_from_p0_count']}")
    print(f"  Correct refusals:           {summary['correct_refusal_count']}")
    print(f"  Metric-only triggers:       {summary['metric_only_trigger_count']}")
    print(f"  No manual review needed:    {summary['no_manual_review_needed_count']}")

    print(f"\n[Downgrade Reasons]")
    for reason, count in sorted(summary['downgrade_reasons'].items()):
        print(f"  {reason}: {count}")

    print(f"\n[Downgraded Sample IDs]")
    for sid in summary['downgraded_sample_ids']:
        for c in candidates:
            if c['sample_id'] == sid:
                print(f"  {sid}: {c['calibrated_issue_type']} "
                      f"(was {c['review_priority']}, trigger={c['trigger_metric']})")
                break

    print(f"\n[Calibrated P0 Samples]")
    for sid in summary['calibrated_p0_sample_ids']:
        print(f"  {sid}: {summary['calibrated_p0_reasons'][sid]}")

    print(f"\n[Correct Refusal IDs]")
    print(f"  {', '.join(summary['correct_refusal_ids'])}")

    print(f"\n[review_candidates] CSV → {csv_path}")

    # Also write summary JSON
    summary_path = output_dir / "calibration_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[review_candidates] Summary → {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
