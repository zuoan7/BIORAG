#!/usr/bin/env python3
"""
生成人工审核候选集 CSV，用于后续人工审核。

候选来源:
  1. faithfulness 最低 top 10
  2. factual_correctness 最低 top 10
  3. answer_relevancy 最低 top 10
  4. context_recall 最低 top 10
  5. 随机高分样本 10 条
  6. 所有 zero citation 样本
  7. 所有 abstain/refusal 样本中 RAGAS 分数异常的样本

输出: results/ragas/<timestamp>/human_review_candidates.csv

使用方式:
  python scripts/evaluation/generate_review_candidates.py \\
    --input results/ragas/<ts>/ragas_scores.jsonl \\
    --output-dir results/ragas/<ts>/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate human review candidate CSV")
    p.add_argument("--input", required=True, help="Path to ragas_scores.jsonl")
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
    return -1.0  # missing → treat as lowest


def suspected_issue(item: dict[str, Any], metric: str, score: float) -> str:
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


def review_priority(score: float, metric: str) -> str:
    if metric in ("faithfulness", "factual_correctness") and score < 0.4:
        return "P0"
    if score < 0.5:
        return "P1"
    return "P2"


def build_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    def add(item: dict[str, Any], priority: str, issue: str, trigger_metric: str):
        sid = item.get("sample_id", "")
        if sid in seen_ids:
            return
        seen_ids.add(sid)
        scores = item.get("ragas_scores") or {}
        candidates.append({
            "sample_id": sid,
            "question": item.get("question", ""),
            "route": item.get("route", ""),
            "scenario": item.get("scenario", ""),
            "expected_behavior": item.get("expected_behavior", ""),
            "answer": item.get("answer", ""),
            "reference": item.get("reference", ""),
            "contexts_preview": _truncate(" | ".join((item.get("contexts") or [])[:5]), 500),
            "faithfulness": scores.get("faithfulness"),
            "factual_correctness": scores.get("factual_correctness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_recall": scores.get("context_recall"),
            "context_precision": scores.get("context_precision"),
            "doc_id_hit": item.get("doc_id_hit"),
            "section_norm_hit": item.get("section_norm_hit"),
            "citation_count": item.get("citation_count", 0),
            "review_priority": priority,
            "suspected_issue_type": issue,
            "trigger_metric": trigger_metric,
            "human_answer_correct": "",
            "human_citation_support": "",
            "human_hallucination": "",
            "human_notes": "",
            "severity": "",
        })

    # 1. Bottom 10 per metric
    for metric in ["faithfulness", "factual_correctness", "answer_relevancy", "context_recall"]:
        sorted_recs = sorted(records, key=lambda r: score_val(r, metric))
        for item in sorted_recs[:10]:
            sc = score_val(item, metric)
            priority = review_priority(sc, metric)
            issue = suspected_issue(item, metric, sc)
            add(item, priority, issue, metric)

    # 2. Random high-score samples (top quartile by faithfulness)
    scored = [(score_val(r, "faithfulness"), r) for r in records if score_val(r, "faithfulness") > 0]
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
            fc = scores.get("factual_correctness")
            sc = min(faith or 0, fc or 0)
            add(item,
                "P0" if (faith or 0) < 0.4 or (fc or 0) < 0.4 else "P1",
                "zero_citation" if sc < 0.5 else "zero_citation_acceptable",
                "zero_citation")

    # 4. Abstain/refusal samples with anomalous RAGAS scores
    for item in records:
        if item.get("answer_mode") in ("refusal", "empty"):
            scores = item.get("ragas_scores") or {}
            fc = scores.get("factual_correctness")
            faith = scores.get("faithfulness")
            # Anomalous: high scores on refusal/abstain
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
    "doc_id_hit", "section_norm_hit", "citation_count",
    "review_priority", "suspected_issue_type", "trigger_metric",
    "human_answer_correct", "human_citation_support",
    "human_hallucination", "human_notes", "severity",
]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    print(f"[review_candidates] Loaded {len(records)} records from {args.input}")

    candidates = build_candidates(records)
    print(f"[review_candidates] Generated {len(candidates)} candidates")

    csv_path = output_dir / "human_review_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for c in candidates:
            writer.writerow(c)

    # Summary
    priorities: dict[str, int] = {}
    issues: dict[str, int] = {}
    for c in candidates:
        p = c["review_priority"]
        priorities[p] = priorities.get(p, 0) + 1
        iss = c["suspected_issue_type"]
        issues[iss] = issues.get(iss, 0) + 1

    print(f"[review_candidates] Priority: {priorities}")
    print(f"[review_candidates] Issues: {issues}")
    print(f"[review_candidates] CSV → {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
