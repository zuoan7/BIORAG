"""BIORAG Eval v2 — Aggregation with pass/partial/fail rates."""

from __future__ import annotations
from typing import Any
from .schemas import score_bucket, PASS_THRESHOLD, PARTIAL_THRESHOLD


def aggregate_rule_scores(rule_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rule_rows)
    if n == 0:
        return {"sample_count": 0}
    return {
        "sample_count": n,
        "route_match_rate": round(sum(1 for r in rule_rows if r["route_match"]) / n, 4),
        "doc_recall_support_mean": round(sum(r["doc_recall_support"] for r in rule_rows) / n, 4),
        "doc_recall_citation_mean": round(sum(r["doc_recall_citation"] for r in rule_rows) / n, 4),
        "expected_doc_in_support_rate": round(sum(1 for r in rule_rows if r["expected_doc_in_support"]) / n, 4),
        "expected_doc_cited_rate": round(sum(1 for r in rule_rows if r["expected_doc_cited"]) / n, 4),
        "wrong_doc_citation_count": sum(1 for r in rule_rows if r["wrong_doc_citation"]),
        "negative_citation_zero_pass": all(r["negative_citation_zero"] for r in rule_rows),
    }


def aggregate_judge_scores(judge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Qwen judge scores by metric, with pass/partial/fail rates."""
    by_metric: dict[str, list[float]] = {}
    buckets: dict[str, dict[str, int]] = {}
    errors = 0
    cache_hits = 0

    for r in judge_rows:
        mn = r.get("metric_name", "?")
        score = r.get("score")
        if r.get("judge_error_type"):
            errors += 1
        if r.get("cache_hit"):
            cache_hits += 1
        if score is not None and isinstance(score, (int, float)):
            by_metric.setdefault(mn, []).append(float(score))
            b = score_bucket(float(score))
            buckets.setdefault(mn, {"pass": 0, "partial": 0, "fail": 0})
            buckets[mn][b] = buckets[mn].get(b, 0) + 1

    metric_summary = {}
    for mn in sorted(by_metric.keys()):
        vals = by_metric[mn]
        buck = buckets.get(mn, {"pass": 0, "partial": 0, "fail": 0})
        total = len(vals)
        metric_summary[mn] = {
            "mean": round(sum(vals) / total, 4) if total else None,
            "count": total,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "pass_count": buck["pass"],
            "pass_rate": round(buck["pass"] / total, 4) if total else 0,
            "partial_count": buck["partial"],
            "partial_rate": round(buck["partial"] / total, 4) if total else 0,
            "fail_count": buck["fail"],
            "fail_rate": round(buck["fail"] / total, 4) if total else 0,
        }

    return {
        "judge_error_count": errors,
        "cache_hit_count": cache_hits,
        "by_metric": metric_summary,
    }
