from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)
DEFAULT_RUN_ID = "20260524"
P0_SAMPLE_IDS = (
    "v3_ra_009",
    "v3_ra_014",
    "v3_ra_018",
    "v3_ra_019",
    "v3_ra_021",
    "v3_ra_026",
    "v3_ra_027",
)
RULE_RATE_KEYS = (
    "doc_recall_at10",
    "doc_hit_at10",
    "strict_doc_all_hit_at10",
    "doc_mrr_at10",
    "exact_gold_chunk_recall_at10",
    "exact_gold_chunk_hit_at10",
    "exact_gold_chunk_mrr_at10",
    "chunk_recall_at10",
    "chunk_hit_at10",
    "chunk_mrr_at10",
    "parent_chunk_recall_at10",
    "parent_chunk_hit_at10",
    "parent_chunk_mrr_at10",
    "route_match",
    "citation_doc_hit",
    "citation_doc_all_hit",
    "citation_chunk_hit",
    "citation_parent_chunk_hit",
    "citation_parent_chunk_all_hit",
    "support_parent_chunk_hit",
    "support_child_evidence_hit",
    "citation_child_evidence_hit",
    "inferred_child_evidence_hit",
    "negative_no_citation",
)
BOOLEAN_RULE_KEYS = (
    "doc_hit_at10",
    "strict_doc_all_hit_at10",
    "exact_gold_chunk_hit_at10",
    "chunk_hit_at10",
    "parent_chunk_hit_at10",
    "route_match",
    "citation_doc_hit",
    "citation_doc_all_hit",
    "citation_chunk_hit",
    "citation_parent_chunk_hit",
    "citation_parent_chunk_all_hit",
    "support_parent_chunk_hit",
    "support_child_evidence_hit",
    "citation_child_evidence_hit",
    "inferred_child_evidence_hit",
    "negative_no_citation",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline validation for v3 P0 gold remap. Recomputes rule metrics and "
            "diagnostic buckets from an existing no-judge results.jsonl without "
            "rerunning retrieval, generation, eval, or judge."
        )
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--p0-sample-ids", default=",".join(P0_SAMPLE_IDS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    p0_sample_ids = parse_sample_ids(args.p0_sample_ids)
    dataset_path = Path(args.dataset)
    results_path = Path(args.results)
    dataset_rows = load_jsonl_by_id(dataset_path, "sample_id")
    result_rows = load_jsonl(results_path)
    validate_inputs(dataset_rows=dataset_rows, result_rows=result_rows, p0_sample_ids=p0_sample_ids)

    sample_rows, old_metric_rows, recomputed_metric_rows = build_offline_rows(
        dataset_rows=dataset_rows,
        result_rows=result_rows,
        p0_sample_ids=p0_sample_ids,
    )
    old_summary = summarize_metric_rows(old_metric_rows)
    recomputed_summary = summarize_metric_rows(recomputed_metric_rows)
    validation = build_validation(
        p0_sample_ids=p0_sample_ids,
        samples=sample_rows,
        old_summary=old_summary,
        recomputed_summary=recomputed_summary,
    )
    summary = {
        "run_id": str(args.run_id),
        "scope": "v3 P0 gold remap offline rule-metric validation",
        "inputs": {
            "dataset": str(dataset_path),
            "results": str(results_path),
        },
        "outputs": {
            "rule_metrics_source": (
                "recomputed from dataset current gold over existing no-judge "
                "retrieval/support/citation fields"
            ),
            "judge_used": False,
            "retrieval_rerun": False,
            "generation_rerun": False,
        },
        "sample_count": len(sample_rows),
        "p0_sample_ids": p0_sample_ids,
        "old_summary": old_summary,
        "recomputed_summary": recomputed_summary,
        "delta": build_delta(old_summary, recomputed_summary),
        "p0_comparison": build_p0_comparison(p0_sample_ids, sample_rows),
        "validation": validation,
    }

    result_dir = RESULTS_ROOT / f"v3_p0_gold_remap_offline_validation_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_p0_gold_remap_offline_validation_{args.run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "samples.jsonl", sample_rows)
    write_markdown(report_dir / "report.md", render_report(summary, sample_rows))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "validation_passed": validation["passed"],
                "doc_hit_parent_chunk_miss_after": recomputed_summary["diagnostic_buckets"][
                    "doc_hit_parent_chunk_miss"
                ]["count"],
                "parent_chunk_hit_at10_after": recomputed_summary["rule_metric_counts"][
                    "parent_chunk_hit_at10"
                ]["true"],
            },
            ensure_ascii=False,
        )
    )


def build_offline_rows(
    *,
    dataset_rows: dict[str, dict[str, Any]],
    result_rows: list[dict[str, Any]],
    p0_sample_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    p0_set = set(p0_sample_ids)
    sample_rows: list[dict[str, Any]] = []
    old_metric_rows: list[dict[str, Any]] = []
    recomputed_metric_rows: list[dict[str, Any]] = []
    for result_row in result_rows:
        sample_id = str(result_row.get("sample_id") or "")
        dataset_row = dataset_rows[sample_id]
        old_metrics = dict(result_row.get("rule_metrics") or {})
        new_gold_chunk_ids = gold_chunk_ids_from_dataset(dataset_row)
        new_gold_parent_chunk_ids = dedupe([parent_chunk_id(chunk) for chunk in new_gold_chunk_ids])
        recomputed_metrics = compute_rule_metrics(
            dataset_row=dataset_row,
            result_row=result_row,
            gold_chunk_ids=new_gold_chunk_ids,
            gold_parent_chunk_ids=new_gold_parent_chunk_ids,
        )
        old_row = build_metric_row(
            dataset_row=dataset_row, result_row=result_row, metrics=old_metrics
        )
        recomputed_row = build_metric_row(
            dataset_row=dataset_row,
            result_row=result_row,
            metrics=recomputed_metrics,
        )
        old_metric_rows.append(old_row)
        recomputed_metric_rows.append(recomputed_row)
        sample_rows.append(
            build_sample_output(
                dataset_row=dataset_row,
                result_row=result_row,
                old_metrics=old_metrics,
                recomputed_metrics=recomputed_metrics,
                old_bucket_keys=diagnostic_bucket_keys(old_row),
                recomputed_bucket_keys=diagnostic_bucket_keys(recomputed_row),
                new_gold_chunk_ids=new_gold_chunk_ids,
                new_gold_parent_chunk_ids=new_gold_parent_chunk_ids,
                is_p0=sample_id in p0_set,
            )
        )
    return sample_rows, old_metric_rows, recomputed_metric_rows


def build_sample_output(
    *,
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    old_metrics: dict[str, Any],
    recomputed_metrics: dict[str, Any],
    old_bucket_keys: list[str],
    recomputed_bucket_keys: list[str],
    new_gold_chunk_ids: list[str],
    new_gold_parent_chunk_ids: list[str],
    is_p0: bool,
) -> dict[str, Any]:
    old_gold_chunk_ids = as_str_list(result_row.get("gold_chunk_ids"))
    old_gold_parent_chunk_ids = as_str_list(result_row.get("gold_parent_chunk_ids"))
    retrieved_parent_chunk_ids = top10_parent_chunk_ids(result_row)
    support_parent_chunk_ids = dedupe(
        [parent_chunk_id(chunk_id) for chunk_id in as_str_list(result_row.get("support_chunk_ids"))]
    )
    citation_parent_chunk_ids = citation_parent_ids(result_row)
    support_child_chunk_ids = as_str_list(result_row.get("support_matched_child_chunk_ids"))
    citation_child_chunk_ids = as_str_list(result_row.get("citation_matched_child_chunk_ids"))
    breakpoints = {
        "old_gold_parent_chunk_ids": old_gold_parent_chunk_ids,
        "new_gold_parent_chunk_ids": new_gold_parent_chunk_ids,
        "retrieved_parent_hits_after": sorted(
            set(new_gold_parent_chunk_ids) & set(retrieved_parent_chunk_ids)
        ),
        "support_parent_hits_after": sorted(
            set(new_gold_parent_chunk_ids) & set(support_parent_chunk_ids)
        ),
        "citation_parent_hits_after": sorted(
            set(new_gold_parent_chunk_ids) & set(citation_parent_chunk_ids)
        ),
        "support_child_hits_after": sorted(set(new_gold_chunk_ids) & set(support_child_chunk_ids)),
        "citation_child_hits_after": sorted(
            set(new_gold_chunk_ids) & set(citation_child_chunk_ids)
        ),
        "parent_top10_rank_after": first_rank(
            retrieved_parent_chunk_ids, set(new_gold_parent_chunk_ids)
        ),
        "support_parent_rank_after": first_rank(
            support_parent_chunk_ids, set(new_gold_parent_chunk_ids)
        ),
        "citation_parent_rank_after": first_rank(
            citation_parent_chunk_ids, set(new_gold_parent_chunk_ids)
        ),
    }
    return {
        "sample_id": str(dataset_row.get("sample_id") or result_row.get("sample_id") or ""),
        "is_p0": is_p0,
        "category": str(dataset_row.get("category") or result_row.get("category") or ""),
        "difficulty": str(dataset_row.get("difficulty") or result_row.get("difficulty") or ""),
        "expected_route": str(
            dataset_row.get("expected_route") or result_row.get("expected_route") or ""
        ),
        "actual_route": str(result_row.get("actual_route") or ""),
        "question": str(dataset_row.get("question") or result_row.get("question") or ""),
        "expected_doc_ids": as_str_list(dataset_row.get("expected_doc_ids")),
        "old_gold_chunk_ids": old_gold_chunk_ids,
        "old_gold_parent_chunk_ids": old_gold_parent_chunk_ids,
        "recomputed_gold_chunk_ids": new_gold_chunk_ids,
        "recomputed_gold_parent_chunk_ids": new_gold_parent_chunk_ids,
        "gold_changed": old_gold_chunk_ids != new_gold_chunk_ids
        or old_gold_parent_chunk_ids != new_gold_parent_chunk_ids,
        "retrieved_doc_ids_top10": as_str_list(result_row.get("retrieved_doc_ids_top10")),
        "retrieved_parent_chunk_ids_top10": retrieved_parent_chunk_ids,
        "support_parent_chunk_ids": support_parent_chunk_ids,
        "support_matched_child_chunk_ids": support_child_chunk_ids,
        "citation_parent_chunk_ids": citation_parent_chunk_ids,
        "citation_matched_child_chunk_ids": citation_child_chunk_ids,
        "old_rule_metrics": old_metrics,
        "recomputed_rule_metrics": recomputed_metrics,
        "diagnostic_buckets_before": old_bucket_keys,
        "diagnostic_buckets_after": recomputed_bucket_keys,
        "breakpoints": breakpoints,
    }


def compute_rule_metrics(
    *,
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    gold_chunk_ids: list[str],
    gold_parent_chunk_ids: list[str],
) -> dict[str, Any]:
    expected_docs = as_str_list(dataset_row.get("expected_doc_ids"))
    retrieved_doc_ids = as_str_list(result_row.get("retrieved_doc_ids_top10"))
    retrieved_chunk_ids = as_str_list(result_row.get("retrieved_chunk_ids_top10"))
    retrieved_parent_chunk_ids = top10_parent_chunk_ids(result_row)
    support_chunk_ids = as_str_list(result_row.get("support_chunk_ids"))
    support_parent_chunk_ids = dedupe([parent_chunk_id(chunk) for chunk in support_chunk_ids])
    support_matched_child_chunk_ids = as_str_list(result_row.get("support_matched_child_chunk_ids"))
    citation_doc_ids = citation_doc_ids_from_result(result_row)
    citation_chunk_ids = citation_chunk_ids_from_result(result_row)
    citation_parent_chunk_ids = citation_parent_ids(result_row)
    citation_matched_child_chunk_ids = as_str_list(
        result_row.get("citation_matched_child_chunk_ids")
    )
    inferred_matched_child_chunk_ids = as_str_list(
        result_row.get("inferred_matched_child_chunk_ids")
    )
    expected_route = str(dataset_row.get("expected_route") or "")
    is_negative = expected_route == "negative"

    doc_hits = [doc for doc in expected_docs if doc in set(retrieved_doc_ids)]
    doc_recall = len(doc_hits) / len(expected_docs) if expected_docs else None
    doc_all_hit = bool(expected_docs) and len(doc_hits) == len(expected_docs)
    doc_any_hit = bool(doc_hits) if expected_docs else None
    doc_mrr = reciprocal_rank(retrieved_doc_ids, set(expected_docs)) if expected_docs else None

    chunk_hits = [chunk for chunk in gold_chunk_ids if chunk in set(retrieved_chunk_ids)]
    chunk_recall = len(chunk_hits) / len(gold_chunk_ids) if gold_chunk_ids else None
    chunk_mrr = (
        reciprocal_rank(retrieved_chunk_ids, set(gold_chunk_ids)) if gold_chunk_ids else None
    )

    parent_hits = [
        chunk for chunk in gold_parent_chunk_ids if chunk in set(retrieved_parent_chunk_ids)
    ]
    parent_recall = len(parent_hits) / len(gold_parent_chunk_ids) if gold_parent_chunk_ids else None
    parent_mrr = (
        reciprocal_rank(retrieved_parent_chunk_ids, set(gold_parent_chunk_ids))
        if gold_parent_chunk_ids
        else None
    )

    support_parent_hits = [
        chunk for chunk in gold_parent_chunk_ids if chunk in set(support_parent_chunk_ids)
    ]
    support_child_hits = [
        chunk for chunk in gold_chunk_ids if chunk in set(support_matched_child_chunk_ids)
    ]
    citation_doc_hits = [doc for doc in expected_docs if doc in set(citation_doc_ids)]
    citation_chunk_hits = [chunk for chunk in gold_chunk_ids if chunk in set(citation_chunk_ids)]
    citation_parent_hits = [
        chunk for chunk in gold_parent_chunk_ids if chunk in set(citation_parent_chunk_ids)
    ]
    citation_child_hits = [
        chunk for chunk in gold_chunk_ids if chunk in set(citation_matched_child_chunk_ids)
    ]
    inferred_child_hits = [
        chunk for chunk in gold_chunk_ids if chunk in set(inferred_matched_child_chunk_ids)
    ]

    return {
        "doc_recall_at10": round(doc_recall, 6) if doc_recall is not None else None,
        "doc_hit_at10": doc_any_hit,
        "strict_doc_all_hit_at10": doc_all_hit if expected_docs else None,
        "doc_mrr_at10": round(doc_mrr, 6) if doc_mrr is not None else None,
        "exact_gold_chunk_recall_at10": round(chunk_recall, 6)
        if chunk_recall is not None
        else None,
        "exact_gold_chunk_hit_at10": bool(chunk_hits) if gold_chunk_ids else None,
        "exact_gold_chunk_mrr_at10": round(chunk_mrr, 6) if chunk_mrr is not None else None,
        "chunk_recall_at10": round(chunk_recall, 6) if chunk_recall is not None else None,
        "chunk_hit_at10": bool(chunk_hits) if gold_chunk_ids else None,
        "chunk_mrr_at10": round(chunk_mrr, 6) if chunk_mrr is not None else None,
        "parent_chunk_recall_at10": round(parent_recall, 6) if parent_recall is not None else None,
        "parent_chunk_hit_at10": bool(parent_hits) if gold_parent_chunk_ids else None,
        "parent_chunk_mrr_at10": round(parent_mrr, 6) if parent_mrr is not None else None,
        "route_match": normalize_route(str(result_row.get("actual_route") or "")) == expected_route,
        "citation_doc_hit": bool(citation_doc_hits) if expected_docs and not is_negative else None,
        "citation_doc_all_hit": (
            len(citation_doc_hits) == len(expected_docs)
            if expected_docs and not is_negative
            else None
        ),
        "citation_chunk_hit": bool(citation_chunk_hits)
        if gold_chunk_ids and not is_negative
        else None,
        "citation_parent_chunk_hit": (
            bool(citation_parent_hits) if gold_parent_chunk_ids and not is_negative else None
        ),
        "citation_parent_chunk_all_hit": (
            len(citation_parent_hits) == len(gold_parent_chunk_ids)
            if gold_parent_chunk_ids and not is_negative
            else None
        ),
        "support_parent_chunk_hit": bool(support_parent_hits) if gold_parent_chunk_ids else None,
        "support_child_evidence_hit": bool(support_child_hits) if gold_chunk_ids else None,
        "citation_child_evidence_hit": (
            bool(citation_child_hits) if gold_chunk_ids and not is_negative else None
        ),
        "inferred_child_evidence_hit": bool(inferred_child_hits) if gold_chunk_ids else None,
        "citation_count": citation_count(result_row),
        "negative_no_citation": citation_count(result_row) == 0 if is_negative else None,
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["rule_metrics"] for row in rows]
    rule_metrics = {
        "doc_recall_at10_avg": avg([metric.get("doc_recall_at10") for metric in metrics]),
        "doc_hit_at10_rate": rate([metric.get("doc_hit_at10") for metric in metrics]),
        "strict_doc_all_hit_at10_rate": rate(
            [metric.get("strict_doc_all_hit_at10") for metric in metrics]
        ),
        "doc_mrr_at10_avg": avg([metric.get("doc_mrr_at10") for metric in metrics]),
        "exact_gold_chunk_recall_at10_avg": avg(
            [metric.get("exact_gold_chunk_recall_at10") for metric in metrics]
        ),
        "exact_gold_chunk_hit_at10_rate": rate(
            [metric.get("exact_gold_chunk_hit_at10") for metric in metrics]
        ),
        "exact_gold_chunk_mrr_at10_avg": avg(
            [metric.get("exact_gold_chunk_mrr_at10") for metric in metrics]
        ),
        "chunk_recall_at10_avg": avg([metric.get("chunk_recall_at10") for metric in metrics]),
        "chunk_hit_at10_rate": rate([metric.get("chunk_hit_at10") for metric in metrics]),
        "chunk_mrr_at10_avg": avg([metric.get("chunk_mrr_at10") for metric in metrics]),
        "parent_chunk_recall_at10_avg": avg(
            [metric.get("parent_chunk_recall_at10") for metric in metrics]
        ),
        "parent_chunk_hit_at10_rate": rate(
            [metric.get("parent_chunk_hit_at10") for metric in metrics]
        ),
        "parent_chunk_mrr_at10_avg": avg(
            [metric.get("parent_chunk_mrr_at10") for metric in metrics]
        ),
        "route_match_rate": rate([metric.get("route_match") for metric in metrics]),
        "citation_doc_hit_rate": rate([metric.get("citation_doc_hit") for metric in metrics]),
        "citation_doc_all_hit_rate": rate(
            [metric.get("citation_doc_all_hit") for metric in metrics]
        ),
        "citation_chunk_hit_rate": rate([metric.get("citation_chunk_hit") for metric in metrics]),
        "citation_parent_chunk_hit_rate": rate(
            [metric.get("citation_parent_chunk_hit") for metric in metrics]
        ),
        "citation_parent_chunk_all_hit_rate": rate(
            [metric.get("citation_parent_chunk_all_hit") for metric in metrics]
        ),
        "support_parent_chunk_hit_rate": rate(
            [metric.get("support_parent_chunk_hit") for metric in metrics]
        ),
        "support_child_evidence_hit_rate": rate(
            [metric.get("support_child_evidence_hit") for metric in metrics]
        ),
        "citation_child_evidence_hit_rate": rate(
            [metric.get("citation_child_evidence_hit") for metric in metrics]
        ),
        "inferred_child_evidence_hit_rate": rate(
            [metric.get("inferred_child_evidence_hit") for metric in metrics]
        ),
        "negative_no_citation_rate": rate(
            [metric.get("negative_no_citation") for metric in metrics]
        ),
    }
    return {
        "sample_count": len(rows),
        "rule_metrics": rule_metrics,
        "rule_metric_counts": {
            key: count_metric([metric.get(key) for metric in metrics]) for key in BOOLEAN_RULE_KEYS
        },
        "diagnostic_buckets": build_diagnostic_buckets(rows),
    }


def build_diagnostic_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {
        "doc_miss": ("doc miss", "期望文档未进入 rerank top10。"),
        "doc_hit_parent_chunk_miss": (
            "doc hit 但 parent chunk miss",
            "期望文档命中，但期望父块未进入 rerank top10。",
        ),
        "parent_hit_support_miss": (
            "parent hit 但 support miss",
            "期望父块进入 rerank top10，但未进入 generation support。",
        ),
        "parent_hit_citation_miss": (
            "parent hit 但 citation miss",
            "期望父块进入 rerank top10，但未进入最终引用。",
        ),
        "evidence_hit_answer_failed": (
            "evidence 命中但答案失败",
            "support 或 citation 命中期望父块/子证据，但 judge 判定答案正确性未通过。",
        ),
        "judge_schema_or_call_issue": (
            "judge/schema 问题",
            "judge 调用失败或输出缺少必需 schema。",
        ),
    }
    buckets = {key: {"sample_ids": []} for key in labels}
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        by_category.setdefault(category, {key: 0 for key in labels})
        sample_id = str(row.get("sample_id") or "")
        for key in diagnostic_bucket_keys(row):
            buckets[key]["sample_ids"].append(sample_id)
            by_category[category][key] += 1

    sample_count = len(rows)
    result = {}
    for key, (label, description) in labels.items():
        sample_ids = buckets[key]["sample_ids"]
        result[key] = {
            "label": label,
            "description": description,
            "count": len(sample_ids),
            "rate": round(len(sample_ids) / sample_count, 6) if sample_count else None,
            "sample_ids": sample_ids,
            "sample_ids_preview": sample_ids[:30],
        }
    result["by_category"] = by_category
    return result


def diagnostic_bucket_keys(row: dict[str, Any]) -> list[str]:
    rule = row.get("rule_metrics") or {}
    keys = []
    if rule.get("doc_hit_at10") is False:
        keys.append("doc_miss")
    if rule.get("doc_hit_at10") is True and rule.get("parent_chunk_hit_at10") is False:
        keys.append("doc_hit_parent_chunk_miss")
    if rule.get("parent_chunk_hit_at10") is True and rule.get("support_parent_chunk_hit") is False:
        keys.append("parent_hit_support_miss")
    if rule.get("parent_chunk_hit_at10") is True and rule.get("citation_parent_chunk_hit") is False:
        keys.append("parent_hit_citation_miss")
    return keys


def build_validation(
    *,
    p0_sample_ids: list[str],
    samples: list[dict[str, Any]],
    old_summary: dict[str, Any],
    recomputed_summary: dict[str, Any],
) -> dict[str, Any]:
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    p0_samples = [samples_by_id[sample_id] for sample_id in p0_sample_ids]
    p0_failures = []
    for sample in p0_samples:
        metrics = sample["recomputed_rule_metrics"]
        failure_fields = []
        for metric_key in (
            "parent_chunk_hit_at10",
            "support_parent_chunk_hit",
            "citation_parent_chunk_hit",
        ):
            if metrics.get(metric_key) is not True:
                failure_fields.append(metric_key)
        if "doc_hit_parent_chunk_miss" in sample["diagnostic_buckets_after"]:
            failure_fields.append("diagnostic_buckets_after.doc_hit_parent_chunk_miss")
        if failure_fields:
            p0_failures.append(
                {
                    "sample_id": sample["sample_id"],
                    "failed_fields": failure_fields,
                    "breakpoints": sample["breakpoints"],
                }
            )

    before_counts = old_summary["rule_metric_counts"]
    after_counts = recomputed_summary["rule_metric_counts"]
    before_buckets = old_summary["diagnostic_buckets"]
    after_buckets = recomputed_summary["diagnostic_buckets"]
    parent_before = before_counts["parent_chunk_hit_at10"]["true"]
    parent_after = after_counts["parent_chunk_hit_at10"]["true"]
    support_before = before_counts["support_parent_chunk_hit"]["true"]
    support_after = after_counts["support_parent_chunk_hit"]["true"]
    citation_before = before_counts["citation_parent_chunk_hit"]["true"]
    citation_after = after_counts["citation_parent_chunk_hit"]["true"]
    bucket_before = before_buckets["doc_hit_parent_chunk_miss"]["count"]
    bucket_after = after_buckets["doc_hit_parent_chunk_miss"]["count"]
    criteria = [
        {
            "name": "p0_parent_chunk_hit_at10_7_of_7",
            "passed": all(
                sample["recomputed_rule_metrics"].get("parent_chunk_hit_at10") is True
                for sample in p0_samples
            ),
            "actual_true": sum(
                1
                for sample in p0_samples
                if sample["recomputed_rule_metrics"].get("parent_chunk_hit_at10") is True
            ),
            "expected_true": len(p0_samples),
        },
        {
            "name": "p0_support_parent_chunk_hit_7_of_7",
            "passed": all(
                sample["recomputed_rule_metrics"].get("support_parent_chunk_hit") is True
                for sample in p0_samples
            ),
            "actual_true": sum(
                1
                for sample in p0_samples
                if sample["recomputed_rule_metrics"].get("support_parent_chunk_hit") is True
            ),
            "expected_true": len(p0_samples),
        },
        {
            "name": "p0_citation_parent_chunk_hit_7_of_7",
            "passed": all(
                sample["recomputed_rule_metrics"].get("citation_parent_chunk_hit") is True
                for sample in p0_samples
            ),
            "actual_true": sum(
                1
                for sample in p0_samples
                if sample["recomputed_rule_metrics"].get("citation_parent_chunk_hit") is True
            ),
            "expected_true": len(p0_samples),
        },
        {
            "name": "p0_not_in_doc_hit_parent_chunk_miss",
            "passed": all(
                "doc_hit_parent_chunk_miss" not in sample["diagnostic_buckets_after"]
                for sample in p0_samples
            ),
            "remaining_sample_ids": [
                sample["sample_id"]
                for sample in p0_samples
                if "doc_hit_parent_chunk_miss" in sample["diagnostic_buckets_after"]
            ],
        },
        {
            "name": "full_parent_chunk_hit_at10_at_least_160",
            "passed": parent_after >= 160,
            "before_true": parent_before,
            "after_true": parent_after,
            "minimum_after_true": 160,
        },
        {
            "name": "full_parent_chunk_hit_at10_delta_at_least_7",
            "passed": parent_after - parent_before >= 7,
            "before_true": parent_before,
            "after_true": parent_after,
            "delta": parent_after - parent_before,
            "minimum_delta": 7,
        },
        {
            "name": "doc_hit_parent_chunk_miss_26_to_at_most_19",
            "passed": bucket_before == 26 and bucket_after <= 19,
            "before_count": bucket_before,
            "after_count": bucket_after,
            "expected_before_count": 26,
            "expected_max_after_count": 19,
        },
        {
            "name": "support_parent_chunk_hit_delta_at_least_7",
            "passed": support_after - support_before >= 7,
            "before_true": support_before,
            "after_true": support_after,
            "delta": support_after - support_before,
            "minimum_delta": 7,
        },
        {
            "name": "citation_parent_chunk_hit_delta_at_least_7",
            "passed": citation_after - citation_before >= 7,
            "before_true": citation_before,
            "after_true": citation_after,
            "delta": citation_after - citation_before,
            "minimum_delta": 7,
        },
    ]
    passed = all(bool(item.get("passed")) for item in criteria)
    return {
        "passed": passed,
        "status": "passed" if passed else "blocked_do_not_enter_p1",
        "criteria": criteria,
        "failed_criteria": [item for item in criteria if not item.get("passed")],
        "failed_p0_samples": p0_failures,
    }


def build_delta(old_summary: dict[str, Any], recomputed_summary: dict[str, Any]) -> dict[str, Any]:
    metric_delta = {}
    for key in BOOLEAN_RULE_KEYS:
        old_count = old_summary["rule_metric_counts"][key]["true"]
        new_count = recomputed_summary["rule_metric_counts"][key]["true"]
        metric_delta[key] = {
            "old_true": old_count,
            "recomputed_true": new_count,
            "delta_true": new_count - old_count,
        }
    bucket_delta = {}
    for key, value in old_summary["diagnostic_buckets"].items():
        if key == "by_category":
            continue
        old_count = value["count"]
        new_count = recomputed_summary["diagnostic_buckets"][key]["count"]
        bucket_delta[key] = {
            "old_count": old_count,
            "recomputed_count": new_count,
            "delta_count": new_count - old_count,
        }
    return {
        "rule_metric_counts": metric_delta,
        "diagnostic_buckets": bucket_delta,
    }


def build_p0_comparison(p0_sample_ids: list[str], samples: list[dict[str, Any]]) -> dict[str, Any]:
    p0_set = set(p0_sample_ids)
    p0_samples = [sample for sample in samples if sample["sample_id"] in p0_set]
    metric_keys = (
        "parent_chunk_hit_at10",
        "support_parent_chunk_hit",
        "citation_parent_chunk_hit",
        "support_child_evidence_hit",
        "citation_child_evidence_hit",
    )
    return {
        "sample_count": len(p0_samples),
        "metric_counts_before": {
            key: count_metric([sample["old_rule_metrics"].get(key) for sample in p0_samples])
            for key in metric_keys
        },
        "metric_counts_after": {
            key: count_metric([sample["recomputed_rule_metrics"].get(key) for sample in p0_samples])
            for key in metric_keys
        },
        "removed_from_doc_hit_parent_chunk_miss": [
            sample["sample_id"]
            for sample in p0_samples
            if "doc_hit_parent_chunk_miss" in sample["diagnostic_buckets_before"]
            and "doc_hit_parent_chunk_miss" not in sample["diagnostic_buckets_after"]
        ],
        "still_in_doc_hit_parent_chunk_miss": [
            sample["sample_id"]
            for sample in p0_samples
            if "doc_hit_parent_chunk_miss" in sample["diagnostic_buckets_after"]
        ],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "old_gold_parent_chunk_ids": sample["old_gold_parent_chunk_ids"],
                "recomputed_gold_parent_chunk_ids": sample["recomputed_gold_parent_chunk_ids"],
                "old_parent_chunk_hit_at10": sample["old_rule_metrics"].get(
                    "parent_chunk_hit_at10"
                ),
                "recomputed_parent_chunk_hit_at10": sample["recomputed_rule_metrics"].get(
                    "parent_chunk_hit_at10"
                ),
                "old_support_parent_chunk_hit": sample["old_rule_metrics"].get(
                    "support_parent_chunk_hit"
                ),
                "recomputed_support_parent_chunk_hit": sample["recomputed_rule_metrics"].get(
                    "support_parent_chunk_hit"
                ),
                "old_citation_parent_chunk_hit": sample["old_rule_metrics"].get(
                    "citation_parent_chunk_hit"
                ),
                "recomputed_citation_parent_chunk_hit": sample["recomputed_rule_metrics"].get(
                    "citation_parent_chunk_hit"
                ),
                "diagnostic_buckets_before": sample["diagnostic_buckets_before"],
                "diagnostic_buckets_after": sample["diagnostic_buckets_after"],
                "breakpoints": sample["breakpoints"],
            }
            for sample in p0_samples
        ],
    }


def build_metric_row(
    *,
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": str(dataset_row.get("sample_id") or result_row.get("sample_id") or ""),
        "category": str(dataset_row.get("category") or result_row.get("category") or "unknown"),
        "expected_route": str(
            dataset_row.get("expected_route") or result_row.get("expected_route") or ""
        ),
        "difficulty": str(dataset_row.get("difficulty") or result_row.get("difficulty") or ""),
        "rule_metrics": metrics,
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    validation = summary["validation"]
    old_summary = summary["old_summary"]
    new_summary = summary["recomputed_summary"]
    delta = summary["delta"]
    p0_comparison = summary["p0_comparison"]
    lines = [
        "# v3 P0 gold remap 离线验证报告",
        "",
        "## 范围",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- dataset: `{summary['inputs']['dataset']}`",
        f"- results: `{summary['inputs']['results']}`",
        f"- sample_count: {summary['sample_count']}",
        (
            "- 口径：不跑 eval/judge；只用现有 no-judge retrieval/support/citation "
            "输出，并用当前 dataset gold 重算 rule metrics。"
        ),
        f"- 结论: `{validation['status']}`",
        "",
        "## 全量指标",
        "",
        "| metric | old true | recomputed true | delta | old rate | recomputed rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "parent_chunk_hit_at10",
        "support_parent_chunk_hit",
        "citation_parent_chunk_hit",
        "support_child_evidence_hit",
        "citation_child_evidence_hit",
    ):
        old_count = old_summary["rule_metric_counts"][key]["true"]
        new_count = new_summary["rule_metric_counts"][key]["true"]
        old_rate_key = rate_key_for_metric(key)
        lines.append(
            f"| `{key}` | {old_count} | {new_count} | {new_count - old_count:+d} | "
            f"{fmt_pct(old_summary['rule_metrics'].get(old_rate_key))} | "
            f"{fmt_pct(new_summary['rule_metrics'].get(old_rate_key))} |"
        )
    lines.extend(
        [
            "",
            "## 诊断桶",
            "",
            "| bucket | old count | recomputed count | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, item in delta["diagnostic_buckets"].items():
        lines.append(
            f"| `{key}` | {item['old_count']} | {item['recomputed_count']} | "
            f"{item['delta_count']:+d} |"
        )
    lines.extend(
        [
            "",
            "## P0 样本",
            "",
            (
                "| sample_id | old parent gold | recomputed parent gold | parent hit | "
                "support hit | citation hit | bucket before | bucket after |"
            ),
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for sample in p0_comparison["samples"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{sample['sample_id']}`",
                    format_code_list(sample["old_gold_parent_chunk_ids"]),
                    format_code_list(sample["recomputed_gold_parent_chunk_ids"]),
                    fmt_bool(sample["recomputed_parent_chunk_hit_at10"]),
                    fmt_bool(sample["recomputed_support_parent_chunk_hit"]),
                    fmt_bool(sample["recomputed_citation_parent_chunk_hit"]),
                    format_code_list(sample["diagnostic_buckets_before"]),
                    format_code_list(sample["diagnostic_buckets_after"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## 验收项", "", "| criterion | status | details |", "|---|---|---|"])
    for item in validation["criteria"]:
        details = {k: v for k, v in item.items() if k not in {"name", "passed"}}
        lines.append(
            f"| `{item['name']}` | {'PASS' if item['passed'] else 'FAIL'} | "
            f"`{json.dumps(details, ensure_ascii=False, sort_keys=True)}` |"
        )

    if validation["failed_p0_samples"]:
        lines.extend(["", "## 失败样本断点", ""])
        for item in validation["failed_p0_samples"]:
            lines.append(f"### `{item['sample_id']}`")
            lines.append("")
            lines.append(
                f"- failed_fields: `{json.dumps(item['failed_fields'], ensure_ascii=False)}`"
            )
            lines.append(f"- breakpoints: `{json.dumps(item['breakpoints'], ensure_ascii=False)}`")
            lines.append("")
    if not validation["passed"]:
        lines.extend(
            [
                "## 后续",
                "",
                "验证未通过：按计划不进入 P1。先检查上方失败样本和断点字段。",
            ]
        )
    else:
        lines.extend(
            [
                "## 后续",
                "",
                (
                    "验证通过：P0 7 个样本已从 `doc_hit_parent_chunk_miss` 恢复；"
                    "可进入 P1 score-floor trace 审计。"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def gold_chunk_ids_from_dataset(dataset_row: dict[str, Any]) -> list[str]:
    source_trace = (dataset_row.get("answer_rubric") or {}).get("source_trace") or {}
    return dedupe(
        [
            *as_str_list(source_trace.get("chunk_ids")),
            *as_str_list(dataset_row.get("target_chunk_id_candidate")),
        ]
    )


def top10_parent_chunk_ids(result_row: dict[str, Any]) -> list[str]:
    explicit = as_str_list(result_row.get("retrieved_parent_chunk_ids_top10"))
    if explicit:
        return explicit
    return dedupe(
        [
            parent_chunk_id(chunk_id)
            for chunk_id in as_str_list(result_row.get("retrieved_chunk_ids_top10"))
        ]
    )


def citation_doc_ids_from_result(result_row: dict[str, Any]) -> list[str]:
    explicit = as_str_list(result_row.get("citation_doc_ids"))
    if explicit:
        return explicit
    citations = result_row.get("citations") or []
    if not isinstance(citations, list):
        return []
    return dedupe([str(item.get("doc_id") or "") for item in citations if isinstance(item, dict)])


def citation_chunk_ids_from_result(result_row: dict[str, Any]) -> list[str]:
    explicit = as_str_list(result_row.get("citation_chunk_ids"))
    if explicit:
        return explicit
    citations = result_row.get("citations") or []
    if not isinstance(citations, list):
        return []
    return dedupe([str(item.get("chunk_id") or "") for item in citations if isinstance(item, dict)])


def citation_parent_ids(result_row: dict[str, Any]) -> list[str]:
    explicit = as_str_list(result_row.get("citation_parent_chunk_ids"))
    if explicit:
        return explicit
    return dedupe(
        [parent_chunk_id(chunk_id) for chunk_id in citation_chunk_ids_from_result(result_row)]
    )


def citation_count(result_row: dict[str, Any]) -> int:
    citations = result_row.get("citations")
    if isinstance(citations, list):
        return len(citations)
    return len(citation_chunk_ids_from_result(result_row))


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for index, value in enumerate(ranked[:10], start=1):
        if value in gold:
            return 1.0 / index
    return 0.0


def normalize_route(route: str) -> str:
    value = str(route).lower().strip()
    if "." in value:
        value = value.split(".")[-1]
    return value


def first_rank(values: list[str], targets: set[str]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in targets:
            return index
    return None


def count_metric(values: list[Any]) -> dict[str, int]:
    return {
        "true": sum(1 for value in values if value is True),
        "false": sum(1 for value in values if value is False),
        "none": sum(1 for value in values if value is None),
        "total": len(values),
    }


def avg(values: list[Any]) -> float | None:
    nums = [
        float(value) for value in values if isinstance(value, (int, float)) and value is not None
    ]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 6)


def rate(values: list[Any]) -> float | None:
    vals = [value for value in values if value is not None]
    if not vals:
        return None
    return round(sum(1 for value in vals if bool(value)) / len(vals), 6)


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if str(value):
        return [str(value)]
    return []


def dedupe(values: list[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def rate_key_for_metric(metric_key: str) -> str:
    if metric_key == "parent_chunk_hit_at10":
        return "parent_chunk_hit_at10_rate"
    if metric_key == "support_parent_chunk_hit":
        return "support_parent_chunk_hit_rate"
    if metric_key == "citation_parent_chunk_hit":
        return "citation_parent_chunk_hit_rate"
    if metric_key == "support_child_evidence_hit":
        return "support_child_evidence_hit_rate"
    if metric_key == "citation_child_evidence_hit":
        return "citation_child_evidence_hit_rate"
    return f"{metric_key}_rate"


def fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "N/A"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"


def format_code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def parse_sample_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def validate_inputs(
    *,
    dataset_rows: dict[str, dict[str, Any]],
    result_rows: list[dict[str, Any]],
    p0_sample_ids: list[str],
) -> None:
    result_ids = [str(row.get("sample_id") or "") for row in result_rows]
    missing_result_ids = [sample_id for sample_id in result_ids if sample_id not in dataset_rows]
    if missing_result_ids:
        raise ValueError(
            "Result rows missing from dataset: " + ", ".join(sorted(set(missing_result_ids)))
        )
    missing_p0 = [sample_id for sample_id in p0_sample_ids if sample_id not in dataset_rows]
    missing_p0.extend(sample_id for sample_id in p0_sample_ids if sample_id not in set(result_ids))
    if missing_p0:
        raise ValueError("Missing P0 rows: " + ", ".join(sorted(set(missing_p0))))
    duplicate_result_ids = sorted(
        sample_id for sample_id, count in Counter(result_ids).items() if count > 1
    )
    if duplicate_result_ids:
        raise ValueError("Duplicate result sample_id values: " + ", ".join(duplicate_result_ids))


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

    dataset_row = {
        "sample_id": "v3_ra_009",
        "question": "q",
        "expected_route": "factoid",
        "expected_doc_ids": ["doc_a"],
        "category": "table_content",
        "answer_rubric": {
            "source_trace": {
                "chunk_ids": ["doc_a_sec01_chunk02"],
            }
        },
        "target_chunk_id_candidate": "doc_a_sec01_chunk02",
    }
    result_row = {
        "sample_id": "v3_ra_009",
        "actual_route": "QueryIntent.FACTOID",
        "gold_chunk_ids": ["doc_a_sec99_chunk99"],
        "gold_parent_chunk_ids": ["doc_a_sec99_chunk99"],
        "retrieved_doc_ids_top10": ["doc_a"],
        "retrieved_chunk_ids_top10": ["doc_a_sec01_chunk02"],
        "retrieved_parent_chunk_ids_top10": ["doc_a_sec01_chunk02"],
        "support_chunk_ids": ["doc_a_sec01_chunk02"],
        "support_matched_child_chunk_ids": ["doc_a_sec01_chunk02::child001"],
        "citation_doc_ids": ["doc_a"],
        "citation_chunk_ids": ["doc_a_sec01_chunk02"],
        "citation_parent_chunk_ids": ["doc_a_sec01_chunk02"],
        "citation_matched_child_chunk_ids": ["doc_a_sec01_chunk02::child001"],
        "citations": [{"doc_id": "doc_a", "chunk_id": "doc_a_sec01_chunk02"}],
        "rule_metrics": {
            "doc_hit_at10": True,
            "parent_chunk_hit_at10": False,
            "support_parent_chunk_hit": False,
            "citation_parent_chunk_hit": False,
        },
    }
    gold_chunk_ids = gold_chunk_ids_from_dataset(dataset_row)
    assert gold_chunk_ids == ["doc_a_sec01_chunk02"]
    recomputed = compute_rule_metrics(
        dataset_row=dataset_row,
        result_row=result_row,
        gold_chunk_ids=gold_chunk_ids,
        gold_parent_chunk_ids=dedupe([parent_chunk_id(chunk) for chunk in gold_chunk_ids]),
    )
    assert recomputed["parent_chunk_hit_at10"] is True
    assert recomputed["support_parent_chunk_hit"] is True
    assert recomputed["citation_parent_chunk_hit"] is True

    old_row = build_metric_row(
        dataset_row=dataset_row,
        result_row=result_row,
        metrics=result_row["rule_metrics"],
    )
    new_row = build_metric_row(dataset_row=dataset_row, result_row=result_row, metrics=recomputed)
    assert "doc_hit_parent_chunk_miss" in diagnostic_bucket_keys(old_row)
    assert "doc_hit_parent_chunk_miss" not in diagnostic_bucket_keys(new_row)
    print("self-test passed")


if __name__ == "__main__":
    main()
