from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_BASELINE_RESULTS = (
    RESULTS_ROOT
    / "v3_baseline_b0_b1_20260523_b0_b1_v3_fixed_metrics"
    / "b0_stable"
    / "results.jsonl"
)
DEFAULT_REWRITE_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260523_b0_rewrite_enabled"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)
DEFAULT_REWRITE_CACHE = (
    RESULTS_ROOT / "v3_b0_rewrite_enabled_20260523_b0_rewrite_enabled" / "rewrite_cache.jsonl"
)
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
REGRESSION_REVIEW_REPORT = (
    REPORTS_ROOT / "v3_b0_rewrite_child_regression_review_20260523" / "report.md"
)

METRICS = (
    "doc_hit_at10",
    "parent_chunk_hit_at10",
    "support_parent_chunk_hit",
    "support_child_evidence_hit",
    "citation_parent_chunk_hit",
    "citation_child_evidence_hit",
)
FIRST_BREAK_ORDER = (
    "doc_miss",
    "doc_hit_parent_miss",
    "parent_hit_support_parent_miss",
    "support_parent_hit_child_miss",
    "support_hit_citation_parent_miss",
    "citation_parent_hit_child_miss",
    "citation_child_hit",
)
TRANSITION_ORDER = (
    "hit_to_hit",
    "hit_to_miss",
    "miss_to_hit",
    "miss_to_miss",
    "null_handled",
)
KNOWN_CHILD_REGRESSION_IDS = (
    "v3_pc_020",
    "v3_pc_034",
    "v3_pc_035",
    "v3_pc_075",
    "v3_pc_091",
)
ACCEPTANCE_EXPECTED = {
    "parent_chunk_hit_at10": {
        "baseline_true_count": 112,
        "rewrite_true_count": 141,
        "miss_to_hit": 33,
        "hit_to_miss": 4,
        "net": 29,
    },
    "support_child_evidence_hit": {
        "baseline_true_count": 53,
        "rewrite_true_count": 91,
        "miss_to_hit": 43,
        "hit_to_miss": 5,
        "net": 38,
    },
    "citation_child_evidence_hit": {
        "baseline_true_count": 50,
        "rewrite_true_count": 85,
        "miss_to_hit": 40,
        "hit_to_miss": 5,
        "net": 35,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit v3 B0 baseline vs B0+rewrite metric transitions."
    )
    parser.add_argument("--baseline-results", default=str(DEFAULT_BASELINE_RESULTS))
    parser.add_argument("--rewrite-results", default=str(DEFAULT_REWRITE_RESULTS))
    parser.add_argument("--rewrite-cache", default=str(DEFAULT_REWRITE_CACHE))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--run-id", default="20260523")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    baseline_path = Path(args.baseline_results)
    rewrite_path = Path(args.rewrite_results)
    rewrite_cache_path = Path(args.rewrite_cache)
    dataset_path = Path(args.dataset)
    run_id = str(args.run_id)

    baseline_rows = load_jsonl_by_id(baseline_path)
    rewrite_rows = load_jsonl_by_id(rewrite_path)
    cache_rows = load_jsonl_by_id(rewrite_cache_path)
    dataset_rows = load_jsonl_by_id(dataset_path)

    sample_ids = list(dataset_rows)
    validate_required_rows(sample_ids, baseline_rows, "baseline results")
    validate_required_rows(sample_ids, rewrite_rows, "rewrite results")

    samples = [
        build_sample(
            sample_id=sample_id,
            dataset_row=dataset_rows[sample_id],
            baseline_row=baseline_rows[sample_id],
            rewrite_row=rewrite_rows[sample_id],
            cache_row=cache_rows.get(sample_id) or {},
        )
        for sample_id in sample_ids
    ]
    summary = build_summary(
        run_id=run_id,
        baseline_path=baseline_path,
        rewrite_path=rewrite_path,
        rewrite_cache_path=rewrite_cache_path,
        dataset_path=dataset_path,
        samples=samples,
        cache_rows=cache_rows,
    )

    output_dir = RESULTS_ROOT / f"v3_b0_rewrite_transition_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_b0_rewrite_transition_{run_id}"
    write_json(output_dir / "transition_summary.json", summary)
    write_jsonl(output_dir / "transition_samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "report_dir": str(report_dir),
                "sample_count": len(samples),
            },
            ensure_ascii=False,
        )
    )


def build_sample(
    *,
    sample_id: str,
    dataset_row: dict[str, Any],
    baseline_row: dict[str, Any],
    rewrite_row: dict[str, Any],
    cache_row: dict[str, Any],
) -> dict[str, Any]:
    expected_route = str(
        dataset_row.get("expected_route")
        or baseline_row.get("expected_route")
        or rewrite_row.get("expected_route")
        or "unknown"
    )
    category = str(
        dataset_row.get("category")
        or baseline_row.get("category")
        or rewrite_row.get("category")
        or "unknown"
    )
    difficulty = str(
        dataset_row.get("difficulty")
        or baseline_row.get("difficulty")
        or rewrite_row.get("difficulty")
        or "unknown"
    )
    baseline_metrics = extract_metrics(baseline_row, expected_route=expected_route)
    rewrite_metrics = extract_metrics(rewrite_row, expected_route=expected_route)
    metric_transitions = {
        metric: transition_label(baseline_metrics.get(metric), rewrite_metrics.get(metric))
        for metric in METRICS
    }
    rewrite_first_break = classify_first_break(rewrite_metrics)

    return {
        "sample_id": sample_id,
        "category": category,
        "expected_route": expected_route,
        "difficulty": difficulty,
        "question": dataset_row.get("question") or rewrite_row.get("question"),
        "rewritten_query": cache_row.get("rewritten_query"),
        "rewrite_cache_source": str(cache_row.get("source") or "missing"),
        "expected_doc_ids": as_str_list(
            dataset_row.get("expected_doc_ids")
            or baseline_row.get("expected_doc_ids")
            or rewrite_row.get("expected_doc_ids")
        ),
        "gold_chunk_ids": as_str_list(
            dataset_row.get("gold_chunk_ids")
            or baseline_row.get("gold_chunk_ids")
            or rewrite_row.get("gold_chunk_ids")
        ),
        "gold_parent_chunk_ids": gold_parent_ids(dataset_row, baseline_row, rewrite_row),
        "baseline_metrics": baseline_metrics,
        "rewrite_metrics": rewrite_metrics,
        "metric_transitions": metric_transitions,
        "rewrite_first_break": rewrite_first_break,
        "recommended_next_bucket": recommend_next_bucket(
            expected_route=expected_route,
            rewrite_metrics=rewrite_metrics,
            metric_transitions=metric_transitions,
            rewrite_first_break=rewrite_first_break,
        ),
        "baseline_top10_chunk_ids": as_str_list(baseline_row.get("retrieved_chunk_ids_top10")),
        "rewrite_top10_chunk_ids": as_str_list(rewrite_row.get("retrieved_chunk_ids_top10")),
        "baseline_top10_parent_chunk_ids": as_str_list(
            baseline_row.get("retrieved_parent_chunk_ids_top10")
        ),
        "rewrite_top10_parent_chunk_ids": as_str_list(
            rewrite_row.get("retrieved_parent_chunk_ids_top10")
        ),
        "baseline_support_chunk_ids": as_str_list(baseline_row.get("support_chunk_ids")),
        "rewrite_support_chunk_ids": as_str_list(rewrite_row.get("support_chunk_ids")),
        "baseline_support_matched_child_ids": as_str_list(
            baseline_row.get("support_matched_child_chunk_ids")
        ),
        "rewrite_support_matched_child_ids": as_str_list(
            rewrite_row.get("support_matched_child_chunk_ids")
        ),
        "baseline_citation_chunk_ids": as_str_list(baseline_row.get("citation_chunk_ids")),
        "rewrite_citation_chunk_ids": as_str_list(rewrite_row.get("citation_chunk_ids")),
        "baseline_citation_matched_child_ids": as_str_list(
            baseline_row.get("citation_matched_child_chunk_ids")
        ),
        "rewrite_citation_matched_child_ids": as_str_list(
            rewrite_row.get("citation_matched_child_chunk_ids")
        ),
    }


def extract_metrics(row: dict[str, Any], *, expected_route: str) -> dict[str, bool | None]:
    rule_metrics = row.get("rule_metrics") or {}
    return {
        metric: normalize_metric_value(
            rule_metrics.get(metric),
            metric=metric,
            expected_route=expected_route,
        )
        for metric in METRICS
    }


def normalize_metric_value(value: Any, *, metric: str, expected_route: str) -> bool | None:
    if expected_route == "negative" and (metric == "doc_hit_at10" or metric.startswith("citation_")):
        return None
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def transition_label(baseline: bool | None, rewrite: bool | None) -> str:
    if baseline is None or rewrite is None:
        return "null_handled"
    if baseline is True and rewrite is True:
        return "hit_to_hit"
    if baseline is True and rewrite is False:
        return "hit_to_miss"
    if baseline is False and rewrite is True:
        return "miss_to_hit"
    return "miss_to_miss"


def classify_first_break(metrics: dict[str, bool | None]) -> str:
    if metrics.get("citation_child_evidence_hit") is True:
        return "citation_child_hit"
    if metrics.get("doc_hit_at10") is False:
        return "doc_miss"
    if metrics.get("doc_hit_at10") is True and metrics.get("parent_chunk_hit_at10") is not True:
        return "doc_hit_parent_miss"
    if metrics.get("doc_hit_at10") is None and metrics.get("parent_chunk_hit_at10") is not True:
        return "doc_miss"
    if (
        metrics.get("parent_chunk_hit_at10") is True
        and metrics.get("support_parent_chunk_hit") is not True
    ):
        return "parent_hit_support_parent_miss"
    if (
        metrics.get("support_parent_chunk_hit") is True
        and metrics.get("support_child_evidence_hit") is not True
    ):
        return "support_parent_hit_child_miss"
    if (
        metrics.get("support_child_evidence_hit") is True
        and metrics.get("citation_parent_chunk_hit") is not True
    ):
        return "support_hit_citation_parent_miss"
    if (
        metrics.get("citation_parent_chunk_hit") is True
        and metrics.get("citation_child_evidence_hit") is not True
    ):
        return "citation_parent_hit_child_miss"
    return "doc_miss"


def recommend_next_bucket(
    *,
    expected_route: str,
    rewrite_metrics: dict[str, bool | None],
    metric_transitions: dict[str, str],
    rewrite_first_break: str,
) -> str:
    if expected_route == "negative":
        return "stable_or_rescued"
    if (
        rewrite_metrics.get("doc_hit_at10") is False
        or metric_transitions.get("doc_hit_at10") == "hit_to_miss"
    ):
        return "rewrite_guard_or_query_union"
    if rewrite_first_break == "doc_hit_parent_miss":
        return "intra_doc_retrieval_or_rerank"
    if rewrite_first_break == "parent_hit_support_parent_miss":
        return "support_selector"
    if rewrite_first_break == "support_parent_hit_child_miss":
        return "child_rematch_or_binding"
    if rewrite_first_break in {
        "support_hit_citation_parent_miss",
        "citation_parent_hit_child_miss",
    }:
        return "citation_binding"
    return "stable_or_rescued"


def build_summary(
    *,
    run_id: str,
    baseline_path: Path,
    rewrite_path: Path,
    rewrite_cache_path: Path,
    dataset_path: Path,
    samples: list[dict[str, Any]],
    cache_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_summaries = {
        metric: summarize_metric(samples, metric)
        for metric in METRICS
    }
    group_transitions = {
        "category": summarize_groups(samples, "category"),
        "expected_route": summarize_groups(samples, "expected_route"),
        "difficulty": summarize_groups(samples, "difficulty"),
        "rewrite_cache_source": summarize_groups(samples, "rewrite_cache_source"),
    }
    support_child_transitions = transition_sample_ids(samples, "support_child_evidence_hit")
    citation_child_transitions = transition_sample_ids(samples, "citation_child_evidence_hit")
    first_break = summarize_first_break(samples)
    category_hotspots = summarize_category_hotspots(samples)
    rewrite_cache_source_table = summarize_source_table(samples)
    regression_samples = summarize_known_regressions(samples)
    acceptance_checks = build_acceptance_checks(metric_summaries, support_child_transitions)

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "baseline_results": str(baseline_path),
            "rewrite_results": str(rewrite_path),
            "rewrite_cache": str(rewrite_cache_path),
            "dataset": str(dataset_path),
        },
        "sample_count": len(samples),
        "rewrite_cache_row_count": len(cache_rows),
        "metrics": metric_summaries,
        "group_transitions": group_transitions,
        "transition_sample_ids": {
            "support_child_evidence_hit": support_child_transitions,
            "citation_child_evidence_hit": citation_child_transitions,
        },
        "remaining_miss_first_break": first_break,
        "category_hotspots": category_hotspots,
        "rewrite_cache_source_table": rewrite_cache_source_table,
        "known_child_regression_samples": regression_samples,
        "acceptance_checks": acceptance_checks,
        "next_step_recommendation": recommend_from_summary(
            first_break=first_break,
            category_hotspots=category_hotspots,
            metric_summaries=metric_summaries,
        ),
    }


def summarize_metric(samples: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    transitions = Counter(
        str((sample.get("metric_transitions") or {}).get(metric) or "unknown")
        for sample in samples
    )
    baseline_values = [
        (sample.get("baseline_metrics") or {}).get(metric)
        for sample in samples
    ]
    rewrite_values = [
        (sample.get("rewrite_metrics") or {}).get(metric)
        for sample in samples
    ]
    baseline_denominator = count_non_null(baseline_values)
    rewrite_denominator = count_non_null(rewrite_values)
    baseline_true_count = count_true(baseline_values)
    rewrite_true_count = count_true(rewrite_values)
    return {
        "baseline_true_count": baseline_true_count,
        "baseline_denominator": baseline_denominator,
        "baseline_rate": safe_rate(baseline_true_count, baseline_denominator),
        "rewrite_true_count": rewrite_true_count,
        "rewrite_denominator": rewrite_denominator,
        "rewrite_rate": safe_rate(rewrite_true_count, rewrite_denominator),
        "delta_true_count": rewrite_true_count - baseline_true_count,
        "delta_rate": delta_rate(
            safe_rate(rewrite_true_count, rewrite_denominator),
            safe_rate(baseline_true_count, baseline_denominator),
        ),
        "transition_matrix": {
            "hit_to_hit": int(transitions.get("hit_to_hit", 0)),
            "hit_to_miss": int(transitions.get("hit_to_miss", 0)),
            "miss_to_hit": int(transitions.get("miss_to_hit", 0)),
            "miss_to_miss": int(transitions.get("miss_to_miss", 0)),
            "null_handled_count": int(transitions.get("null_handled", 0)),
        },
        "eligible_transition_count": int(
            transitions.get("hit_to_hit", 0)
            + transitions.get("hit_to_miss", 0)
            + transitions.get("miss_to_hit", 0)
            + transitions.get("miss_to_miss", 0)
        ),
        "net_miss_to_hit": int(transitions.get("miss_to_hit", 0) - transitions.get("hit_to_miss", 0)),
    }


def summarize_groups(samples: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get(group_key) or "unknown")].append(sample)
    return {
        key: {
            "sample_count": len(group_samples),
            "metrics": {
                metric: summarize_metric(group_samples, metric)
                for metric in METRICS
            },
        }
        for key, group_samples in sorted(grouped.items())
    }


def transition_sample_ids(samples: list[dict[str, Any]], metric: str) -> dict[str, list[str]]:
    result = {transition: [] for transition in TRANSITION_ORDER}
    for sample in samples:
        transition = str((sample.get("metric_transitions") or {}).get(metric) or "unknown")
        result.setdefault(transition, []).append(str(sample.get("sample_id") or ""))
    return result


def summarize_first_break(samples: list[dict[str, Any]]) -> dict[str, Any]:
    remaining = [
        sample
        for sample in samples
        if (sample.get("rewrite_metrics") or {}).get("citation_child_evidence_hit") is False
    ]
    all_counts = Counter(str(sample.get("rewrite_first_break") or "unknown") for sample in samples)
    remaining_counts = Counter(
        str(sample.get("rewrite_first_break") or "unknown") for sample in remaining
    )
    bucket_counts = Counter(
        str(sample.get("recommended_next_bucket") or "unknown") for sample in remaining
    )
    total = len(remaining)
    return {
        "scope": "rewrite citation_child_evidence_hit == false; null/negative citation samples excluded",
        "remaining_miss_count": total,
        "all_sample_first_break_counts": ordered_counts(all_counts, FIRST_BREAK_ORDER, len(samples)),
        "counts": ordered_counts(remaining_counts, FIRST_BREAK_ORDER, total),
        "recommended_next_bucket_counts": ordered_counts(
            bucket_counts,
            (
                "rewrite_guard_or_query_union",
                "intra_doc_retrieval_or_rerank",
                "support_selector",
                "child_rematch_or_binding",
                "citation_binding",
                "stable_or_rescued",
            ),
            total,
        ),
    }


def summarize_category_hotspots(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("category") or "unknown")].append(sample)
    rows = []
    for category, group_samples in grouped.items():
        citation_values = [
            (sample.get("rewrite_metrics") or {}).get("citation_child_evidence_hit")
            for sample in group_samples
        ]
        citation_denominator = count_non_null(citation_values)
        remaining = [
            sample
            for sample in group_samples
            if (sample.get("rewrite_metrics") or {}).get("citation_child_evidence_hit") is False
        ]
        bucket_counts = Counter(
            str(sample.get("recommended_next_bucket") or "unknown")
            for sample in remaining
        )
        top_bucket = bucket_counts.most_common(1)[0][0] if bucket_counts else ""
        rows.append(
            {
                "category": category,
                "sample_count": len(group_samples),
                "citation_denominator": citation_denominator,
                "rewrite_citation_child_remaining_miss_count": len(remaining),
                "rewrite_citation_child_remaining_miss_rate": safe_rate(
                    len(remaining), citation_denominator
                ),
                "support_child_miss_to_hit": count_transition(
                    group_samples,
                    "support_child_evidence_hit",
                    "miss_to_hit",
                ),
                "support_child_hit_to_miss": count_transition(
                    group_samples,
                    "support_child_evidence_hit",
                    "hit_to_miss",
                ),
                "citation_child_miss_to_hit": count_transition(
                    group_samples,
                    "citation_child_evidence_hit",
                    "miss_to_hit",
                ),
                "citation_child_hit_to_miss": count_transition(
                    group_samples,
                    "citation_child_evidence_hit",
                    "hit_to_miss",
                ),
                "top_remaining_bucket": top_bucket,
                "top_remaining_bucket_count": int(bucket_counts.get(top_bucket, 0))
                if top_bucket
                else 0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["rewrite_citation_child_remaining_miss_count"]),
            -int(row["citation_child_hit_to_miss"]),
            str(row["category"]),
        ),
    )


def summarize_source_table(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("rewrite_cache_source") or "unknown")].append(sample)
    rows = []
    for source, group_samples in grouped.items():
        citation_values = [
            (sample.get("rewrite_metrics") or {}).get("citation_child_evidence_hit")
            for sample in group_samples
        ]
        citation_denominator = count_non_null(citation_values)
        remaining = sum(1 for value in citation_values if value is False)
        row = {
            "rewrite_cache_source": source,
            "sample_count": len(group_samples),
            "citation_denominator": citation_denominator,
            "rewrite_citation_child_remaining_miss_count": remaining,
            "rewrite_citation_child_remaining_miss_rate": safe_rate(
                remaining,
                citation_denominator,
            ),
        }
        for metric in (
            "parent_chunk_hit_at10",
            "support_child_evidence_hit",
            "citation_child_evidence_hit",
        ):
            summary = summarize_metric(group_samples, metric)
            matrix = summary["transition_matrix"]
            row[f"{metric}_miss_to_hit"] = matrix["miss_to_hit"]
            row[f"{metric}_hit_to_miss"] = matrix["hit_to_miss"]
            row[f"{metric}_net"] = summary["net_miss_to_hit"]
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["rewrite_cache_source"]))


def summarize_known_regressions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(sample.get("sample_id") or ""): sample for sample in samples}
    rows = []
    for sample_id in KNOWN_CHILD_REGRESSION_IDS:
        sample = by_id.get(sample_id)
        if not sample:
            rows.append({"sample_id": sample_id, "found": False})
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "found": True,
                "category": sample.get("category"),
                "expected_route": sample.get("expected_route"),
                "support_child_transition": (sample.get("metric_transitions") or {}).get(
                    "support_child_evidence_hit"
                ),
                "citation_child_transition": (sample.get("metric_transitions") or {}).get(
                    "citation_child_evidence_hit"
                ),
                "rewrite_first_break": sample.get("rewrite_first_break"),
                "recommended_next_bucket": sample.get("recommended_next_bucket"),
                "rewrite_cache_source": sample.get("rewrite_cache_source"),
            }
        )
    return rows


def build_acceptance_checks(
    metric_summaries: dict[str, dict[str, Any]],
    support_child_transitions: dict[str, list[str]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for metric, expected in ACCEPTANCE_EXPECTED.items():
        actual_summary = metric_summaries[metric]
        matrix = actual_summary["transition_matrix"]
        actual = {
            "baseline_true_count": actual_summary["baseline_true_count"],
            "rewrite_true_count": actual_summary["rewrite_true_count"],
            "miss_to_hit": matrix["miss_to_hit"],
            "hit_to_miss": matrix["hit_to_miss"],
            "net": actual_summary["net_miss_to_hit"],
        }
        checks[metric] = {
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }

    lost_ids = set(support_child_transitions.get("hit_to_miss") or [])
    expected_lost = set(KNOWN_CHILD_REGRESSION_IDS)
    checks["known_support_child_regression_ids"] = {
        "expected": sorted(expected_lost),
        "actual_hit_to_miss": sorted(lost_ids),
        "missing": sorted(expected_lost - lost_ids),
        "extra": sorted(lost_ids - expected_lost),
        "passed": lost_ids == expected_lost,
    }
    return checks


def recommend_from_summary(
    *,
    first_break: dict[str, Any],
    category_hotspots: list[dict[str, Any]],
    metric_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        key: int(row["count"])
        for key, row in (first_break.get("counts") or {}).items()
    }
    primary = max(
        (
            (key, count)
            for key, count in counts.items()
            if key != "citation_child_hit" and count > 0
        ),
        key=lambda item: (item[1], item[0]),
        default=("", 0),
    )
    bucket_by_break = {
        "doc_miss": "rewrite_guard_or_query_union",
        "doc_hit_parent_miss": "intra_doc_retrieval_or_rerank",
        "parent_hit_support_parent_miss": "support_selector",
        "support_parent_hit_child_miss": "child_rematch_or_binding",
        "support_hit_citation_parent_miss": "citation_binding",
        "citation_parent_hit_child_miss": "citation_binding",
    }
    top_categories = [
        row["category"]
        for row in category_hotspots[:3]
        if int(row["rewrite_citation_child_remaining_miss_count"]) > 0
    ]
    return {
        "primary_first_break": primary[0],
        "primary_first_break_count": primary[1],
        "primary_bucket": bucket_by_break.get(primary[0], "stable_or_rescued"),
        "top_remaining_miss_categories": top_categories,
        "support_child_net": metric_summaries["support_child_evidence_hit"]["net_miss_to_hit"],
        "citation_child_net": metric_summaries["citation_child_evidence_hit"]["net_miss_to_hit"],
        "note": "Do not change main retrieval/rerank/support/citation/generation in this audit.",
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 B0 baseline vs B0+rewrite transition audit",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Baseline results: `{summary['inputs']['baseline_results']}`",
        f"- Rewrite results: `{summary['inputs']['rewrite_results']}`",
        f"- Rewrite cache: `{summary['inputs']['rewrite_cache']}`",
        f"- Dataset: `{summary['inputs']['dataset']}`",
        f"- Sample count: {summary['sample_count']}",
        "",
        "## Overall delta",
        "",
        "| Metric | Baseline hit | Rewrite hit | Delta hit | Baseline rate | Rewrite rate | Delta rate | miss->hit | hit->miss | net | null handled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = summary["metrics"][metric]
        matrix = item["transition_matrix"]
        lines.append(
            f"| `{metric}` | {item['baseline_true_count']}/{item['baseline_denominator']} | "
            f"{item['rewrite_true_count']}/{item['rewrite_denominator']} | "
            f"{signed(item['delta_true_count'])} | {pct(item['baseline_rate'])} | "
            f"{pct(item['rewrite_rate'])} | {signed_pct(item['delta_rate'])} | "
            f"{matrix['miss_to_hit']} | {matrix['hit_to_miss']} | "
            f"{signed(item['net_miss_to_hit'])} | {matrix['null_handled_count']} |"
        )

    lines.extend(["", "## Transition matrix", ""])
    for metric in METRICS:
        item = summary["metrics"][metric]
        matrix = item["transition_matrix"]
        lines.extend(
            [
                f"### `{metric}`",
                "",
                "| Baseline \\ Rewrite | hit | miss | null handled |",
                "|---|---:|---:|---:|",
                f"| hit | {matrix['hit_to_hit']} | {matrix['hit_to_miss']} |  |",
                f"| miss | {matrix['miss_to_hit']} | {matrix['miss_to_miss']} |  |",
                f"| null |  |  | {matrix['null_handled_count']} |",
                "",
            ]
        )

    append_transition_lists(lines, samples, "support_child_evidence_hit")
    append_transition_lists(lines, samples, "citation_child_evidence_hit")

    first_break = summary["remaining_miss_first_break"]
    lines.extend(
        [
            "",
            "## Remaining miss first-break funnel",
            "",
            f"Scope: {first_break['scope']}.",
            "",
            "| First break | Count | Rate among remaining misses |",
            "|---|---:|---:|",
        ]
    )
    for key, row in first_break["counts"].items():
        lines.append(f"| `{key}` | {row['count']} | {pct(row['rate'])} |")

    lines.extend(
        [
            "",
            "## Category hotspots",
            "",
            "| Category | Samples | Citation denom | Remaining citation child miss | Remaining miss rate | support child + / - | citation child + / - | Top remaining bucket |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["category_hotspots"]:
        lines.append(
            f"| `{row['category']}` | {row['sample_count']} | {row['citation_denominator']} | "
            f"{row['rewrite_citation_child_remaining_miss_count']} | "
            f"{pct(row['rewrite_citation_child_remaining_miss_rate'])} | "
            f"+{row['support_child_miss_to_hit']} / -{row['support_child_hit_to_miss']} | "
            f"+{row['citation_child_miss_to_hit']} / -{row['citation_child_hit_to_miss']} | "
            f"`{row['top_remaining_bucket'] or 'none'}` ({row['top_remaining_bucket_count']}) |"
        )

    lines.extend(
        [
            "",
            "## Rewrite cache source",
            "",
            "| Source | Samples | Citation denom | Remaining citation child miss | Remaining miss rate | parent + / - / net | support child + / - / net | citation child + / - / net |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rewrite_cache_source_table"]:
        lines.append(
            f"| `{row['rewrite_cache_source']}` | {row['sample_count']} | "
            f"{row['citation_denominator']} | "
            f"{row['rewrite_citation_child_remaining_miss_count']} | "
            f"{pct(row['rewrite_citation_child_remaining_miss_rate'])} | "
            f"+{row['parent_chunk_hit_at10_miss_to_hit']} / "
            f"-{row['parent_chunk_hit_at10_hit_to_miss']} / "
            f"{signed(row['parent_chunk_hit_at10_net'])} | "
            f"+{row['support_child_evidence_hit_miss_to_hit']} / "
            f"-{row['support_child_evidence_hit_hit_to_miss']} / "
            f"{signed(row['support_child_evidence_hit_net'])} | "
            f"+{row['citation_child_evidence_hit_miss_to_hit']} / "
            f"-{row['citation_child_evidence_hit_hit_to_miss']} / "
            f"{signed(row['citation_child_evidence_hit_net'])} |"
        )

    lines.extend(
        [
            "",
            "## Known child evidence regressions",
            "",
            f"Existing manual review: [{REGRESSION_REVIEW_REPORT}]({relative_report_link()})",
            "",
            "| sample_id | category | support child transition | citation child transition | rewrite first break | recommended bucket | cache source |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in summary["known_child_regression_samples"]:
        lines.append(
            f"| `{row['sample_id']}` | `{row.get('category', '')}` | "
            f"`{row.get('support_child_transition', '')}` | "
            f"`{row.get('citation_child_transition', '')}` | "
            f"`{row.get('rewrite_first_break', '')}` | "
            f"`{row.get('recommended_next_bucket', '')}` | "
            f"`{row.get('rewrite_cache_source', '')}` |"
        )

    recommendation = summary["next_step_recommendation"]
    first_break_counts = first_break["counts"]
    lines.extend(
        [
            "",
            "## Next step recommendation",
            "",
            f"- Primary remaining break: `{recommendation['primary_first_break']}` "
            f"({recommendation['primary_first_break_count']} samples).",
            f"- Recommended next bucket: `{recommendation['primary_bucket']}`.",
            f"- Remaining miss split: doc miss "
            f"{first_break_counts['doc_miss']['count']}, doc-hit parent miss "
            f"{first_break_counts['doc_hit_parent_miss']['count']}, support selector "
            f"{first_break_counts['parent_hit_support_parent_miss']['count']}, child rematch/binding "
            f"{first_break_counts['support_parent_hit_child_miss']['count']}, citation binding "
            f"{first_break_counts['support_hit_citation_parent_miss']['count'] + first_break_counts['citation_parent_hit_child_miss']['count']}.",
            "- Practical priority: inspect the doc-hit parent-miss bucket first; then child rematch/binding and support selector. Rewrite guard is still needed for doc misses, including the v3_pc_091 query drift case, but it is not the dominant remaining bucket.",
            f"- Hottest remaining categories: {format_code_list(recommendation['top_remaining_miss_categories'])}.",
            f"- Rewrite net gain remains positive: support child {signed(recommendation['support_child_net'])}, "
            f"citation child {signed(recommendation['citation_child_net'])}.",
            "- Neighbor expansion is not indicated by this transition audit.",
            "- Keep this audit read-only; do not change retrieval, rerank, support/citation, or generation code here.",
        ]
    )
    return "\n".join(lines)


def append_transition_lists(
    lines: list[str],
    samples: list[dict[str, Any]],
    metric: str,
) -> None:
    miss_to_hit = samples_with_transition(samples, metric, "miss_to_hit")
    hit_to_miss = samples_with_transition(samples, metric, "hit_to_miss")
    lines.extend(
        [
            "",
            f"## `{metric}` transition samples",
            "",
            f"- miss->hit ({len(miss_to_hit)}): {format_code_list(sample_ids(miss_to_hit))}",
            f"- hit->miss ({len(hit_to_miss)}): {format_code_list(sample_ids(hit_to_miss))}",
        ]
    )


def samples_with_transition(
    samples: list[dict[str, Any]],
    metric: str,
    transition: str,
) -> list[dict[str, Any]]:
    return [
        sample
        for sample in samples
        if (sample.get("metric_transitions") or {}).get(metric) == transition
    ]


def sample_ids(samples: list[dict[str, Any]]) -> list[str]:
    return [str(sample.get("sample_id") or "") for sample in samples]


def gold_parent_ids(
    dataset_row: dict[str, Any],
    baseline_row: dict[str, Any],
    rewrite_row: dict[str, Any],
) -> list[str]:
    explicit = (
        dataset_row.get("gold_parent_chunk_ids")
        or baseline_row.get("gold_parent_chunk_ids")
        or rewrite_row.get("gold_parent_chunk_ids")
    )
    if explicit:
        return as_str_list(explicit)
    return dedupe(parent_chunk_id(chunk_id) for chunk_id in as_str_list(dataset_row.get("gold_chunk_ids")))


def count_transition(samples: list[dict[str, Any]], metric: str, transition: str) -> int:
    return sum(
        1
        for sample in samples
        if (sample.get("metric_transitions") or {}).get(metric) == transition
    )


def ordered_counts(counter: Counter[str], order: tuple[str, ...], denominator: int) -> dict[str, Any]:
    keys = list(order)
    keys.extend(sorted(key for key in counter if key not in set(keys)))
    return {
        key: {
            "count": int(counter.get(key, 0)),
            "rate": safe_rate(int(counter.get(key, 0)), denominator),
        }
        for key in keys
    }


def load_jsonl_by_id(path: Path, key: str = "sample_id") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get(key) or "")
            if not row_id:
                raise ValueError(f"{path}:{line_number} missing {key}")
            rows[row_id] = row
    return rows


def validate_required_rows(
    sample_ids: list[str],
    rows: dict[str, dict[str, Any]],
    label: str,
) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows]
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"{label} missing {len(missing)} dataset sample ids: {preview}")


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def count_true(values: list[Any]) -> int:
    return sum(1 for value in values if value is True)


def count_non_null(values: list[Any]) -> int:
    return sum(1 for value in values if value is not None)


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def delta_rate(rewrite_rate: float | None, baseline_rate: float | None) -> float | None:
    if rewrite_rate is None or baseline_rate is None:
        return None
    return round(rewrite_rate - baseline_rate, 6)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def signed(value: Any) -> str:
    if value is None:
        return "N/A"
    number = int(value) if isinstance(value, int) else float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number}"


def signed_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value) * 100:.1f}%"


def format_code_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def relative_report_link() -> str:
    return "../v3_b0_rewrite_child_regression_review_20260523/report.md"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_self_test() -> None:
    samples = [
        {
            "sample_id": "s1",
            "category": "table_content",
            "expected_route": "factoid",
            "difficulty": "hard",
            "rewrite_cache_source": "metadata.original_generated_question",
            "baseline_metrics": {
                "doc_hit_at10": False,
                "parent_chunk_hit_at10": False,
                "support_parent_chunk_hit": False,
                "support_child_evidence_hit": False,
                "citation_parent_chunk_hit": False,
                "citation_child_evidence_hit": False,
            },
            "rewrite_metrics": {
                "doc_hit_at10": True,
                "parent_chunk_hit_at10": True,
                "support_parent_chunk_hit": True,
                "support_child_evidence_hit": True,
                "citation_parent_chunk_hit": True,
                "citation_child_evidence_hit": True,
            },
        },
        {
            "sample_id": "s2",
            "category": "figure_caption",
            "expected_route": "factoid",
            "difficulty": "hard",
            "rewrite_cache_source": "identity_fallback",
            "baseline_metrics": {
                "doc_hit_at10": True,
                "parent_chunk_hit_at10": True,
                "support_parent_chunk_hit": True,
                "support_child_evidence_hit": True,
                "citation_parent_chunk_hit": True,
                "citation_child_evidence_hit": True,
            },
            "rewrite_metrics": {
                "doc_hit_at10": True,
                "parent_chunk_hit_at10": True,
                "support_parent_chunk_hit": True,
                "support_child_evidence_hit": False,
                "citation_parent_chunk_hit": True,
                "citation_child_evidence_hit": False,
            },
        },
        {
            "sample_id": "s3",
            "category": "negative_near_topic",
            "expected_route": "negative",
            "difficulty": "medium",
            "rewrite_cache_source": "identity_fallback",
            "baseline_metrics": {
                "doc_hit_at10": None,
                "parent_chunk_hit_at10": True,
                "support_parent_chunk_hit": True,
                "support_child_evidence_hit": True,
                "citation_parent_chunk_hit": None,
                "citation_child_evidence_hit": None,
            },
            "rewrite_metrics": {
                "doc_hit_at10": None,
                "parent_chunk_hit_at10": True,
                "support_parent_chunk_hit": True,
                "support_child_evidence_hit": True,
                "citation_parent_chunk_hit": None,
                "citation_child_evidence_hit": None,
            },
        },
    ]
    for sample in samples:
        sample["metric_transitions"] = {
            metric: transition_label(
                sample["baseline_metrics"][metric],
                sample["rewrite_metrics"][metric],
            )
            for metric in METRICS
        }
        sample["rewrite_first_break"] = classify_first_break(sample["rewrite_metrics"])
        sample["recommended_next_bucket"] = recommend_next_bucket(
            expected_route=sample["expected_route"],
            rewrite_metrics=sample["rewrite_metrics"],
            metric_transitions=sample["metric_transitions"],
            rewrite_first_break=sample["rewrite_first_break"],
        )

    doc_summary = summarize_metric(samples, "doc_hit_at10")
    assert doc_summary["transition_matrix"]["miss_to_hit"] == 1
    assert doc_summary["transition_matrix"]["null_handled_count"] == 1
    support_summary = summarize_metric(samples, "support_child_evidence_hit")
    assert support_summary["transition_matrix"]["hit_to_miss"] == 1
    assert samples[1]["rewrite_first_break"] == "support_parent_hit_child_miss"
    assert samples[1]["recommended_next_bucket"] == "child_rematch_or_binding"
    first_break = summarize_first_break(samples)
    assert first_break["remaining_miss_count"] == 1
    assert first_break["counts"]["support_parent_hit_child_miss"]["count"] == 1
    print("self-test passed")


if __name__ == "__main__":
    main()
