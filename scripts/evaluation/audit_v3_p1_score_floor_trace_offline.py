from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_P0_SUMMARY = RESULTS_ROOT / "v3_p0_gold_remap_offline_validation_20260524" / "summary.json"
DEFAULT_P0_SAMPLES = RESULTS_ROOT / "v3_p0_gold_remap_offline_validation_20260524" / "samples.jsonl"
DEFAULT_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
EXPECTED_TARGET_COUNT = 19
REPLAY_RATIOS = (0.4, 0.3, 0.2, 0.1, 0.0)
PRIMARY_CLASSES = {
    "raw_parent_absent",
    "raw_parent_present_trace_missing",
    "score_floor_filtered",
    "final_topk_cutoff",
    "comparison_selection_or_doc_diversity",
    "same_doc_body_coverage_replaced",
    "same_doc_wrong_parent_selected",
    "unknown_trace_gap",
}
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<section>\d+)_chunk(?P<chunk>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline P1 audit for remaining v3 doc-hit parent-miss samples after "
            "P0 gold remap validation."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--p0-summary", default=str(DEFAULT_P0_SUMMARY))
    parser.add_argument("--p0-samples", default=str(DEFAULT_P0_SAMPLES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--expected-target-count", type=int, default=EXPECTED_TARGET_COUNT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    p0_summary_path = Path(args.p0_summary)
    p0_samples_path = Path(args.p0_samples)
    results_path = Path(args.results)
    dataset_path = Path(args.dataset)

    p0_summary = load_json(p0_summary_path)
    p0_samples = load_jsonl_by_id(p0_samples_path, "sample_id")
    result_rows = load_jsonl_by_id(results_path, "sample_id")
    dataset_rows = load_jsonl_by_id(dataset_path, "sample_id")
    target_sample_ids = target_ids_from_p0_summary(p0_summary)

    samples = [
        audit_sample(
            sample_id=sample_id,
            p0_row=p0_samples[sample_id],
            result_row=result_rows[sample_id],
            dataset_row=dataset_rows.get(sample_id) or {},
        )
        for sample_id in target_sample_ids
    ]
    score_floor_replay = build_score_floor_replay(samples)
    summary = build_summary(
        run_id=str(args.run_id),
        p0_summary_path=p0_summary_path,
        p0_samples_path=p0_samples_path,
        results_path=results_path,
        dataset_path=dataset_path,
        target_sample_ids=target_sample_ids,
        samples=samples,
        score_floor_replay=score_floor_replay,
        expected_target_count=args.expected_target_count,
    )

    result_dir = RESULTS_ROOT / f"v3_p1_score_floor_trace_audit_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_p1_score_floor_trace_audit_{args.run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "samples.jsonl", samples)
    write_json(result_dir / "score_floor_replay.json", score_floor_replay)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "validation_passed": summary["validation"]["passed"],
                "target_sample_count": len(samples),
                "classification_counts": summary["classification_counts"],
                "pure_score_floor_recoverable_count": len(
                    summary["pure_score_floor_recoverable_sample_ids"]
                ),
            },
            ensure_ascii=False,
        )
    )


def audit_sample(
    *,
    sample_id: str,
    p0_row: dict[str, Any],
    result_row: dict[str, Any],
    dataset_row: dict[str, Any],
) -> dict[str, Any]:
    expected_docs = as_str_list(
        p0_row.get("expected_doc_ids") or result_row.get("expected_doc_ids")
    )
    gold_parent_ids = as_str_list(p0_row.get("recomputed_gold_parent_chunk_ids"))
    gold_child_ids = as_str_list(p0_row.get("recomputed_gold_chunk_ids"))
    debug = result_row.get("debug_digest") or {}
    rerank_hits = debug.get("rerank_hits") or {}
    selection = rerank_hits.get("selection") or {}
    trace = sorted_trace(rerank_hits.get("ranking_trace") or [])
    target_traces = [item for item in trace if parent_id(item) in set(gold_parent_ids)]
    best_target = target_traces[0] if target_traces else {}
    raw_parent_ids = raw_parent_ids_from_result(result_row, debug)
    final_parent_ids = final_parent_ids_from_result(result_row, selection)
    top10_parent_ids = as_str_list(p0_row.get("retrieved_parent_chunk_ids_top10"))
    if not top10_parent_ids:
        top10_parent_ids = as_str_list(result_row.get("retrieved_parent_chunk_ids_top10"))
    raw_rank = first_rank(raw_parent_ids, set(gold_parent_ids))
    primary_class = classify_primary(
        target_traces=target_traces,
        raw_parent_rank=raw_rank,
        selection=selection,
        final_parent_ids=final_parent_ids,
        gold_parent_ids=gold_parent_ids,
        expected_docs=expected_docs,
    )
    secondary_tags = classify_secondary_tags(
        category=str(p0_row.get("category") or dataset_row.get("category") or ""),
        expected_route=str(p0_row.get("expected_route") or dataset_row.get("expected_route") or ""),
        expected_docs=expected_docs,
        gold_parent_ids=gold_parent_ids,
        final_parent_ids=final_parent_ids,
        best_target=best_target,
    )
    score_floor = score_floor_summary(best_target, selection)
    replay_by_ratio = replay_score_floor_for_sample(
        trace=trace,
        gold_parent_ids=gold_parent_ids,
        top_k=safe_int(selection.get("top_k")) or 10,
    )
    is_pure = is_pure_score_floor_candidate(
        primary_class=primary_class,
        expected_route=str(p0_row.get("expected_route") or dataset_row.get("expected_route") or ""),
        selection=selection,
    )
    recovered = recoverable_ratio(replay_by_ratio)
    return {
        "sample_id": sample_id,
        "category": str(p0_row.get("category") or dataset_row.get("category") or ""),
        "expected_route": str(
            p0_row.get("expected_route") or dataset_row.get("expected_route") or ""
        ),
        "actual_route": str(result_row.get("actual_route") or ""),
        "question": str(p0_row.get("question") or dataset_row.get("question") or ""),
        "expected_doc_ids": expected_docs,
        "gold_child_chunk_ids": gold_child_ids,
        "gold_parent_chunk_ids": gold_parent_ids,
        "old_result_gold_parent_chunk_ids": as_str_list(result_row.get("gold_parent_chunk_ids")),
        "gold_source": "p0_recomputed_gold",
        "primary_classification": primary_class,
        "secondary_tags": secondary_tags,
        "raw_parent_rank": raw_rank,
        "raw_parent_ids_preview": raw_parent_ids[:15],
        "trace": {
            "present": bool(trace),
            "candidate_count": len(trace),
            "target_trace_count": len(target_traces),
            "target_traces": [compact_trace(item) for item in target_traces[:5]],
            "best_target_trace": compact_trace(best_target),
            "top_pre_floor_traces": [compact_trace(item) for item in trace[:5]],
        },
        "score_floor": score_floor,
        "selection": compact_selection(selection),
        "retrieved_parent_chunk_ids_top10": top10_parent_ids,
        "final_parent_chunk_ids": final_parent_ids,
        "support_parent_chunk_ids": as_str_list(p0_row.get("support_parent_chunk_ids")),
        "citation_parent_chunk_ids": as_str_list(p0_row.get("citation_parent_chunk_ids")),
        "same_doc_context": same_doc_context(
            expected_docs=expected_docs,
            gold_parent_ids=gold_parent_ids,
            final_parent_ids=final_parent_ids,
        ),
        "ratio_replay": replay_by_ratio,
        "score_floor_recoverability": {
            "pure_score_floor_candidate": is_pure,
            "recoverable_by_ratio_replay": recovered is not None,
            "first_recoverable_ratio": recovered,
            "requires_disabling_score_floor": recovered == 0.0,
        },
    }


def classify_primary(
    *,
    target_traces: list[dict[str, Any]],
    raw_parent_rank: int | None,
    selection: dict[str, Any],
    final_parent_ids: list[str],
    gold_parent_ids: list[str],
    expected_docs: list[str],
) -> str:
    if not target_traces:
        if raw_parent_rank is not None:
            return "raw_parent_present_trace_missing"
        return "raw_parent_absent"

    target_parent_ids = {parent_id(item) for item in target_traces}
    score_floor_dropped = set(as_str_list(nested(selection, "score_floor", "dropped_chunk_ids")))
    if any(
        item.get("dropped_by_score_floor") is True
        or parent_id(item) in score_floor_dropped
        or item.get("final_drop_reason") == "score_floor"
        for item in target_traces
    ):
        return "score_floor_filtered"

    same_doc_coverage = selection.get("same_doc_body_coverage") or {}
    if same_doc_coverage.get("changed") is True and target_parent_ids & set(
        as_str_list(same_doc_coverage.get("dropped_chunk_ids"))
    ):
        return "same_doc_body_coverage_replaced"

    comparison = selection.get("comparison_selection") or {}
    diversity = selection.get("doc_diversity") or {}
    if (
        comparison.get("applied") is True
        or diversity.get("applied") is True
        or any(
            str(item.get("final_drop_reason") or "")
            in {"comparison_selection_or_topk", "doc_diversity"}
            for item in target_traces
        )
    ):
        return "comparison_selection_or_doc_diversity"

    if any(item.get("post_floor_rank") is not None for item in target_traces):
        return "final_topk_cutoff"

    if any(doc_id(parent) in set(expected_docs) for parent in final_parent_ids) and not (
        set(final_parent_ids) & set(gold_parent_ids)
    ):
        return "same_doc_wrong_parent_selected"
    return "unknown_trace_gap"


def classify_secondary_tags(
    *,
    category: str,
    expected_route: str,
    expected_docs: list[str],
    gold_parent_ids: list[str],
    final_parent_ids: list[str],
    best_target: dict[str, Any],
) -> list[str]:
    tags: list[str] = []
    same_doc_distances = same_doc_distances_to_gold(
        expected_docs=expected_docs,
        gold_parent_ids=gold_parent_ids,
        final_parent_ids=final_parent_ids,
    )
    if same_doc_distances:
        if min(same_doc_distances) <= 2:
            tags.append("same_doc_adjacent_parent")
        else:
            tags.append("same_doc_far_parent")
    if any(doc_id(parent) not in set(expected_docs) for parent in final_parent_ids):
        tags.append("cross_doc_competition")
    if expected_route == "comparison" or len(gold_parent_ids) > 1:
        tags.append("comparison_multi_parent_partial")
    lower_category = category.lower()
    target_rank = safe_int(best_target.get("pre_floor_rerank_rank"))
    late_target = target_rank is None or target_rank > 10
    if late_target and ("figure" in lower_category or "caption" in lower_category):
        tags.append("figure_or_caption_late_parent")
    if late_target and ("table" in lower_category or "caption" in lower_category):
        tags.append("table_or_caption_late_parent")
    return dedupe(tags)


def score_floor_summary(best_target: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    score_floor = selection.get("score_floor") or {}
    floor = safe_float(score_floor.get("floor"))
    top_score = safe_float(score_floor.get("top_score"))
    target_score = safe_float(best_target.get("score"))
    return {
        "enabled": score_floor.get("enabled"),
        "ratio": score_floor.get("ratio"),
        "top_score": round_number(top_score),
        "floor": round_number(floor),
        "target_score": round_number(target_score),
        "target_minus_floor": round_number(target_score - floor)
        if target_score is not None and floor is not None
        else None,
        "top_minus_target": round_number(top_score - target_score)
        if top_score is not None and target_score is not None
        else None,
        "target_pre_floor_rank": best_target.get("pre_floor_rerank_rank"),
        "target_post_floor_rank": best_target.get("post_floor_rank"),
        "target_final_top10_rank": best_target.get("final_top10_rank"),
        "target_dropped_by_score_floor": best_target.get("dropped_by_score_floor"),
        "target_final_drop_reason": best_target.get("final_drop_reason"),
    }


def replay_score_floor_for_sample(
    *,
    trace: list[dict[str, Any]],
    gold_parent_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if not trace:
        return []
    top_score = safe_float(trace[0].get("score"))
    target_ids = set(gold_parent_ids)
    result = []
    for ratio in REPLAY_RATIOS:
        if ratio <= 0 or top_score is None or top_score <= 0:
            floor = None
            post = trace
        else:
            floor = top_score * ratio
            post = [item for item in trace if (safe_float(item.get("score")) or 0.0) >= floor]
        target_rank = first_rank([parent_id(item) for item in post], target_ids)
        result.append(
            {
                "ratio": ratio,
                "floor": round_number(floor),
                "top_score": round_number(top_score),
                "post_floor_count": len(post),
                "target_post_floor_rank": target_rank,
                "post_floor_recovered": target_rank is not None,
                "final_topk_recovered": target_rank is not None and target_rank <= top_k,
                "top_k": top_k,
            }
        )
    return result


def build_score_floor_replay(samples: list[dict[str, Any]]) -> dict[str, Any]:
    score_floor_samples = [
        sample for sample in samples if sample["primary_classification"] == "score_floor_filtered"
    ]
    by_ratio: dict[str, dict[str, Any]] = {}
    for ratio in REPLAY_RATIOS:
        recovered_ids = []
        post_floor_recovered_ids = []
        for sample in score_floor_samples:
            item = replay_item(sample, ratio)
            if not item:
                continue
            if item.get("post_floor_recovered"):
                post_floor_recovered_ids.append(sample["sample_id"])
            if item.get("final_topk_recovered"):
                recovered_ids.append(sample["sample_id"])
        by_ratio[str(ratio)] = {
            "ratio": ratio,
            "sample_count": len(score_floor_samples),
            "post_floor_recovered_count": len(post_floor_recovered_ids),
            "final_topk_recovered_count": len(recovered_ids),
            "post_floor_recovered_sample_ids": post_floor_recovered_ids,
            "final_topk_recovered_sample_ids": recovered_ids,
        }
    pure_recoverable_ids = [
        sample["sample_id"]
        for sample in score_floor_samples
        if sample["score_floor_recoverability"]["pure_score_floor_candidate"]
        and sample["score_floor_recoverability"]["recoverable_by_ratio_replay"]
    ]
    positive_ratio_recoverable_ids = [
        sample["sample_id"]
        for sample in score_floor_samples
        if sample["score_floor_recoverability"]["pure_score_floor_candidate"]
        and sample["score_floor_recoverability"]["first_recoverable_ratio"] is not None
        and sample["score_floor_recoverability"]["first_recoverable_ratio"] > 0.0
    ]
    requires_disabling_ids = [
        sample["sample_id"]
        for sample in score_floor_samples
        if sample["score_floor_recoverability"]["pure_score_floor_candidate"]
        and sample["score_floor_recoverability"]["requires_disabling_score_floor"]
    ]
    contributing_ids = [
        sample["sample_id"]
        for sample in score_floor_samples
        if not sample["score_floor_recoverability"]["pure_score_floor_candidate"]
    ]
    return {
        "current_ratio": 0.4,
        "ratios": list(REPLAY_RATIOS),
        "score_floor_filtered_count": len(score_floor_samples),
        "by_ratio": by_ratio,
        "pure_score_floor_recoverable_sample_ids": pure_recoverable_ids,
        "pure_positive_ratio_recoverable_sample_ids": positive_ratio_recoverable_ids,
        "pure_requires_disabling_score_floor_sample_ids": requires_disabling_ids,
        "score_floor_contributing_selection_audit_sample_ids": contributing_ids,
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "expected_route": sample["expected_route"],
                "gold_parent_chunk_ids": sample["gold_parent_chunk_ids"],
                "score_floor": sample["score_floor"],
                "pure_score_floor_candidate": sample["score_floor_recoverability"][
                    "pure_score_floor_candidate"
                ],
                "first_recoverable_ratio": sample["score_floor_recoverability"][
                    "first_recoverable_ratio"
                ],
                "ratio_replay": sample["ratio_replay"],
            }
            for sample in score_floor_samples
        ],
    }


def build_summary(
    *,
    run_id: str,
    p0_summary_path: Path,
    p0_samples_path: Path,
    results_path: Path,
    dataset_path: Path,
    target_sample_ids: list[str],
    samples: list[dict[str, Any]],
    score_floor_replay: dict[str, Any],
    expected_target_count: int,
) -> dict[str, Any]:
    validation = build_validation(target_sample_ids, samples, expected_target_count)
    classification_counts = Counter(sample["primary_classification"] for sample in samples)
    secondary_tag_counts = Counter(
        tag for sample in samples for tag in sample.get("secondary_tags") or []
    )
    not_score_floor = [
        sample["sample_id"]
        for sample in samples
        if sample["primary_classification"] != "score_floor_filtered"
    ]
    pure_recoverable = score_floor_replay["pure_score_floor_recoverable_sample_ids"]
    if pure_recoverable:
        recommendation = "review_score_floor_fix_candidates_before_code_change"
    else:
        recommendation = "do_not_fix_score_floor_next_audit_raw_retrieval_or_selection"
    return {
        "run_id": run_id,
        "scope": "P1 offline score-floor trace audit after P0 gold remap validation",
        "inputs": {
            "p0_summary": str(p0_summary_path),
            "p0_samples": str(p0_samples_path),
            "results": str(results_path),
            "dataset": str(dataset_path),
        },
        "sample_count": len(samples),
        "expected_target_count": expected_target_count,
        "target_sample_ids": target_sample_ids,
        "classification_counts": dict(classification_counts),
        "secondary_tag_counts": dict(secondary_tag_counts),
        "score_floor_replay": {
            "score_floor_filtered_count": score_floor_replay["score_floor_filtered_count"],
            "pure_score_floor_recoverable_count": len(pure_recoverable),
            "pure_positive_ratio_recoverable_count": len(
                score_floor_replay["pure_positive_ratio_recoverable_sample_ids"]
            ),
            "pure_requires_disabling_score_floor_count": len(
                score_floor_replay["pure_requires_disabling_score_floor_sample_ids"]
            ),
            "score_floor_contributing_selection_audit_count": len(
                score_floor_replay["score_floor_contributing_selection_audit_sample_ids"]
            ),
        },
        "pure_score_floor_recoverable_sample_ids": pure_recoverable,
        "pure_positive_ratio_recoverable_sample_ids": score_floor_replay[
            "pure_positive_ratio_recoverable_sample_ids"
        ],
        "pure_requires_disabling_score_floor_sample_ids": score_floor_replay[
            "pure_requires_disabling_score_floor_sample_ids"
        ],
        "score_floor_contributing_selection_audit_sample_ids": score_floor_replay[
            "score_floor_contributing_selection_audit_sample_ids"
        ],
        "not_score_floor_fix_sample_ids": not_score_floor,
        "recommendation": recommendation,
        "validation": validation,
    }


def build_validation(
    target_sample_ids: list[str],
    samples: list[dict[str, Any]],
    expected_target_count: int,
) -> dict[str, Any]:
    sample_ids = [sample["sample_id"] for sample in samples]
    criteria = [
        {
            "name": "target_count_matches_expected",
            "passed": len(sample_ids) == expected_target_count,
            "actual": len(sample_ids),
            "expected": expected_target_count,
        },
        {
            "name": "target_ids_match_p0_summary",
            "passed": sample_ids == target_sample_ids,
            "actual": sample_ids,
            "expected": target_sample_ids,
        },
        {
            "name": "one_primary_classification_per_sample",
            "passed": all(
                sample.get("primary_classification") in PRIMARY_CLASSES for sample in samples
            ),
            "invalid_sample_ids": [
                sample["sample_id"]
                for sample in samples
                if sample.get("primary_classification") not in PRIMARY_CLASSES
            ],
        },
        score_floor_field_criterion(samples),
    ]
    return {
        "passed": all(bool(item.get("passed")) for item in criteria),
        "criteria": criteria,
        "failed_criteria": [item for item in criteria if not item.get("passed")],
    }


def score_floor_field_criterion(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    required = (
        "target_score",
        "top_score",
        "floor",
        "target_minus_floor",
        "target_pre_floor_rank",
    )
    for sample in samples:
        if sample.get("primary_classification") != "score_floor_filtered":
            continue
        score_floor = sample.get("score_floor") or {}
        missing = [key for key in required if score_floor.get(key) is None]
        target_trace = (sample.get("trace") or {}).get("best_target_trace") or {}
        if not target_trace:
            missing.append("trace.best_target_trace")
        if score_floor.get("target_dropped_by_score_floor") is not True:
            missing.append("target_dropped_by_score_floor_true")
        if missing:
            failures.append({"sample_id": sample["sample_id"], "missing_fields": missing})
    return {
        "name": "score_floor_filtered_samples_have_required_breakpoints",
        "passed": not failures,
        "failures": failures,
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 P1 score-floor trace 离线审计报告",
        "",
        "## 范围",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- sample_count: {summary['sample_count']}",
        "- 口径：只读 P0 remap 后 gold 和现有 no-judge rerank trace，不运行 eval/judge。",
        f"- validation_passed: `{summary['validation']['passed']}`",
        f"- recommendation: `{summary['recommendation']}`",
        "",
        "## Primary Classification",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    for key, count in sorted(summary["classification_counts"].items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Score Floor Replay",
            "",
            (
                f"- score_floor_filtered_count: "
                f"{summary['score_floor_replay']['score_floor_filtered_count']}"
            ),
            (
                "- pure_score_floor_recoverable_sample_ids: "
                f"{format_code_list(summary['pure_score_floor_recoverable_sample_ids'])}"
            ),
            (
                "- pure_positive_ratio_recoverable_sample_ids: "
                f"{format_code_list(summary['pure_positive_ratio_recoverable_sample_ids'])}"
            ),
            (
                "- pure_requires_disabling_score_floor_sample_ids: "
                f"{format_code_list(summary['pure_requires_disabling_score_floor_sample_ids'])}"
            ),
            (
                "- score_floor_contributing_selection_audit_sample_ids: "
                f"{format_code_list(summary['score_floor_contributing_selection_audit_sample_ids'])}"
            ),
            "",
            (
                "| sample_id | class | score/floor | pre/post/final rank | "
                "first recoverable ratio | tags |"
            ),
            "|---|---|---:|---|---:|---|",
        ]
    )
    for sample in samples:
        score_floor = sample["score_floor"]
        recoverability = sample["score_floor_recoverability"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{sample['sample_id']}`",
                    f"`{sample['primary_classification']}`",
                    f"{fmt(score_floor.get('target_score'))}/{fmt(score_floor.get('floor'))}",
                    (
                        f"{fmt(score_floor.get('target_pre_floor_rank'))}/"
                        f"{fmt(score_floor.get('target_post_floor_rank'))}/"
                        f"{fmt(score_floor.get('target_final_top10_rank'))}"
                    ),
                    fmt(recoverability.get("first_recoverable_ratio")),
                    format_code_list(sample["secondary_tags"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 不建议作为 score-floor 修复处理的样本",
            "",
            format_code_list(summary["not_score_floor_fix_sample_ids"]),
            "",
            "## Validation",
            "",
            "| criterion | status | details |",
            "|---|---|---|",
        ]
    )
    for item in summary["validation"]["criteria"]:
        details = {key: value for key, value in item.items() if key not in {"name", "passed"}}
        lines.append(
            f"| `{item['name']}` | {'PASS' if item['passed'] else 'FAIL'} | "
            f"`{json.dumps(details, ensure_ascii=False, sort_keys=True)}` |"
        )
    return "\n".join(lines) + "\n"


def compact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_k": selection.get("top_k"),
        "pre_floor_chunk_ids": as_str_list(selection.get("pre_floor_chunk_ids")),
        "post_floor_chunk_ids": as_str_list(selection.get("post_floor_chunk_ids")),
        "final_chunk_ids": as_str_list(selection.get("final_chunk_ids")),
        "comparison_selection": selection.get("comparison_selection") or {},
        "doc_diversity": selection.get("doc_diversity") or {},
        "same_doc_body_coverage": selection.get("same_doc_body_coverage") or {},
        "score_floor": selection.get("score_floor") or {},
    }


def compact_trace(item: dict[str, Any]) -> dict[str, Any]:
    if not item:
        return {}
    return {
        "chunk_id": item.get("chunk_id"),
        "parent_chunk_id": item.get("parent_chunk_id") or parent_chunk_id(item.get("chunk_id")),
        "doc_id": item.get("doc_id"),
        "section": item.get("section"),
        "raw_retrieval_rank": item.get("raw_retrieval_rank"),
        "pre_floor_rerank_rank": item.get("pre_floor_rerank_rank"),
        "post_floor_rank": item.get("post_floor_rank"),
        "final_top10_rank": item.get("final_top10_rank"),
        "score": round_number(safe_float(item.get("score"))),
        "rerank_score": round_number(safe_float(item.get("rerank_score"))),
        "fusion_score": round_number(safe_float(item.get("fusion_score"))),
        "vector_score": round_number(safe_float(item.get("vector_score"))),
        "bm25_score": round_number(safe_float(item.get("bm25_score"))),
        "survived_score_floor": item.get("survived_score_floor"),
        "dropped_by_score_floor": item.get("dropped_by_score_floor"),
        "in_final_top10": item.get("in_final_top10"),
        "final_drop_reason": item.get("final_drop_reason"),
    }


def same_doc_context(
    *,
    expected_docs: list[str],
    gold_parent_ids: list[str],
    final_parent_ids: list[str],
) -> dict[str, Any]:
    same_doc_ids = [
        parent
        for parent in final_parent_ids
        if doc_id(parent) in set(expected_docs) and parent not in set(gold_parent_ids)
    ]
    return {
        "same_doc_selected_parent_ids": same_doc_ids,
        "same_doc_selected_count": len(same_doc_ids),
        "distance_to_nearest_gold": same_doc_distances_to_gold(
            expected_docs=expected_docs,
            gold_parent_ids=gold_parent_ids,
            final_parent_ids=final_parent_ids,
        ),
    }


def same_doc_distances_to_gold(
    *,
    expected_docs: list[str],
    gold_parent_ids: list[str],
    final_parent_ids: list[str],
) -> list[int]:
    distances = []
    expected_doc_set = set(expected_docs)
    for final_parent in final_parent_ids:
        if doc_id(final_parent) not in expected_doc_set or final_parent in set(gold_parent_ids):
            continue
        final_pos = parent_position(final_parent)
        if not final_pos:
            continue
        for gold_parent in gold_parent_ids:
            if doc_id(gold_parent) != doc_id(final_parent):
                continue
            gold_pos = parent_position(gold_parent)
            if not gold_pos:
                continue
            distances.append(max(abs(final_pos[0] - gold_pos[0]), abs(final_pos[1] - gold_pos[1])))
    return sorted(distances)


def final_parent_ids_from_result(
    result_row: dict[str, Any], selection: dict[str, Any]
) -> list[str]:
    final_ids = as_str_list(selection.get("final_chunk_ids"))
    if final_ids:
        return dedupe(parent_chunk_id(item) for item in final_ids)
    return as_str_list(result_row.get("retrieved_parent_chunk_ids_top10"))


def raw_parent_ids_from_result(result_row: dict[str, Any], debug: dict[str, Any]) -> list[str]:
    explicit = as_str_list(result_row.get("raw_retrieved_parent_chunk_ids"))
    if explicit:
        return explicit
    retrieval_output = debug.get("retrieval_output") or {}
    parent_ids = as_str_list(retrieval_output.get("parent_chunk_ids"))
    if parent_ids:
        return parent_ids
    return dedupe(
        parent_chunk_id(item) for item in as_str_list(result_row.get("raw_retrieved_chunk_ids"))
    )


def target_ids_from_p0_summary(summary: dict[str, Any]) -> list[str]:
    return as_str_list(
        nested(
            summary,
            "recomputed_summary",
            "diagnostic_buckets",
            "doc_hit_parent_chunk_miss",
            "sample_ids",
        )
    )


def is_pure_score_floor_candidate(
    *,
    primary_class: str,
    expected_route: str,
    selection: dict[str, Any],
) -> bool:
    if primary_class != "score_floor_filtered":
        return False
    if expected_route == "comparison":
        return False
    if (selection.get("comparison_selection") or {}).get("applied") is True:
        return False
    if (selection.get("doc_diversity") or {}).get("applied") is True:
        return False
    return True


def recoverable_ratio(replay_by_ratio: list[dict[str, Any]]) -> float | None:
    for item in replay_by_ratio:
        if item.get("ratio") == 0.4:
            continue
        if item.get("final_topk_recovered"):
            return float(item["ratio"])
    return None


def replay_item(sample: dict[str, Any], ratio: float) -> dict[str, Any] | None:
    for item in sample.get("ratio_replay") or []:
        if float(item.get("ratio")) == ratio:
            return item
    return None


def sorted_trace(trace: list[Any]) -> list[dict[str, Any]]:
    items = [item for item in trace if isinstance(item, dict)]
    return sorted(items, key=lambda item: safe_int(item.get("pre_floor_rerank_rank")) or 999999)


def parent_id(trace_item: dict[str, Any]) -> str:
    return str(trace_item.get("parent_chunk_id") or parent_chunk_id(trace_item.get("chunk_id")))


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def doc_id(parent_id_value: Any) -> str:
    value = str(parent_id_value or "")
    match = PARENT_ID_RE.match(value)
    if match:
        return match.group("doc")
    if "_sec" in value:
        return value.split("_sec", 1)[0]
    return ""


def parent_position(parent_id_value: str) -> tuple[int, int] | None:
    match = PARENT_ID_RE.match(str(parent_id_value or ""))
    if not match:
        return None
    return int(match.group("section")), int(match.group("chunk"))


def first_rank(values: list[str], targets: set[str]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in targets:
            return index
    return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def round_number(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    return round(number, 6)


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if str(value):
        return [str(value)]
    return []


def dedupe(values: Any) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for line_number, row in enumerate(load_jsonl(path), start=1):
        row_id = str(row.get(key) or "")
        if not row_id:
            raise ValueError(f"{path}:{line_number} missing key {key}")
        if row_id in rows:
            raise ValueError(f"{path}:{line_number} duplicate key {key}={row_id}")
        rows[row_id] = row
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_self_test() -> None:
    assert parent_chunk_id("x::child001") == "x"
    assert parent_chunk_id("doc_a_sec01_chunk02") == "doc_a_sec01_chunk02"

    p0_row = {
        "sample_id": "s1",
        "category": "table_content",
        "expected_route": "factoid",
        "question": "q",
        "expected_doc_ids": ["doc_a"],
        "recomputed_gold_chunk_ids": ["doc_a_sec01_chunk02"],
        "recomputed_gold_parent_chunk_ids": ["doc_a_sec01_chunk02"],
        "retrieved_parent_chunk_ids_top10": ["doc_a_sec01_chunk01"],
    }
    result_row = {
        "sample_id": "s1",
        "gold_parent_chunk_ids": ["doc_a_sec99_chunk99"],
        "actual_route": "QueryIntent.FACTOID",
        "raw_retrieved_parent_chunk_ids": ["doc_a_sec01_chunk02"],
        "retrieved_parent_chunk_ids_top10": ["doc_a_sec01_chunk01"],
        "debug_digest": {
            "rerank_hits": {
                "selection": {
                    "top_k": 10,
                    "score_floor": {
                        "enabled": True,
                        "ratio": 0.4,
                        "top_score": 10.0,
                        "floor": 4.0,
                        "dropped_chunk_ids": ["doc_a_sec01_chunk02"],
                    },
                    "final_chunk_ids": ["doc_a_sec01_chunk01"],
                    "comparison_selection": {"applied": False},
                    "doc_diversity": {"applied": False},
                    "same_doc_body_coverage": {"changed": False},
                },
                "ranking_trace": [
                    {
                        "chunk_id": "doc_a_sec01_chunk01",
                        "parent_chunk_id": "doc_a_sec01_chunk01",
                        "doc_id": "doc_a",
                        "score": 10.0,
                        "pre_floor_rerank_rank": 1,
                        "post_floor_rank": 1,
                        "final_top10_rank": 1,
                    },
                    {
                        "chunk_id": "doc_a_sec01_chunk02",
                        "parent_chunk_id": "doc_a_sec01_chunk02",
                        "doc_id": "doc_a",
                        "score": -1.0,
                        "pre_floor_rerank_rank": 2,
                        "post_floor_rank": None,
                        "final_top10_rank": None,
                        "dropped_by_score_floor": True,
                        "final_drop_reason": "score_floor",
                    },
                ],
            }
        },
    }
    sample = audit_sample(sample_id="s1", p0_row=p0_row, result_row=result_row, dataset_row={})
    assert sample["gold_parent_chunk_ids"] == ["doc_a_sec01_chunk02"]
    assert sample["old_result_gold_parent_chunk_ids"] == ["doc_a_sec99_chunk99"]
    assert sample["primary_classification"] == "score_floor_filtered"
    assert replay_item(sample, 0.4)["final_topk_recovered"] is False
    assert replay_item(sample, 0.0)["final_topk_recovered"] is True

    raw_absent = classify_primary(
        target_traces=[],
        raw_parent_rank=None,
        selection={},
        final_parent_ids=[],
        gold_parent_ids=["doc_a_sec01_chunk02"],
        expected_docs=["doc_a"],
    )
    assert raw_absent == "raw_parent_absent"

    final_topk = classify_primary(
        target_traces=[
            {
                "parent_chunk_id": "doc_a_sec01_chunk02",
                "post_floor_rank": 11,
                "final_drop_reason": "final_topk_cutoff",
            }
        ],
        raw_parent_rank=2,
        selection={},
        final_parent_ids=[],
        gold_parent_ids=["doc_a_sec01_chunk02"],
        expected_docs=["doc_a"],
    )
    assert final_topk == "final_topk_cutoff"

    comparison = classify_primary(
        target_traces=[
            {
                "parent_chunk_id": "doc_a_sec01_chunk02",
                "post_floor_rank": 3,
                "final_drop_reason": "comparison_selection_or_topk",
            }
        ],
        raw_parent_rank=2,
        selection={"comparison_selection": {"applied": True}},
        final_parent_ids=[],
        gold_parent_ids=["doc_a_sec01_chunk02"],
        expected_docs=["doc_a"],
    )
    assert comparison == "comparison_selection_or_doc_diversity"
    print("self-test passed")


if __name__ == "__main__":
    main()
