from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_JUDGED_DIR = RESULTS_ROOT / "v3_baseline_b0_b1_20260523_b0_b1_v3_fixed_metrics"

SUPPORT_BREAK_LABELS = {
    "before_support_final_parent_miss": "final context 未命中目标父块",
    "final_parent_hit_candidate_missing": "final 命中目标父块，但 generation candidates 缺失",
    "candidate_parent_hit_selector_miss": "candidate 命中目标父块，但 support selector 未选中",
    "support_parent_hit_child_metadata_miss": "support 命中目标父块，但未保留目标 child id",
    "support_child_hit_citation_candidate_miss": "support 保留 child id，但 citation candidates 未保留",
    "citation_candidate_child_hit_binding_miss": "citation candidates 保留 child id，但 citation binding 未保留",
    "binding_child_hit_citation_output_miss": "citation binding 保留 child id，但最终 citation 未保留",
    "support_chain_ok": "support/citation child 链路未发现断点",
    "no_gold_parent": "样本缺少目标父块",
    "no_gold_child_id": "样本缺少目标 child id",
}

CHILD_BREAK_LABELS = {
    "no_gold_child_id": "样本缺少目标 child id",
    "final_chunks_missing_gold_child_id": "final_chunks 未保留目标 child id",
    "generation_candidates_missing_gold_child_id": "generation candidates 未保留目标 child id",
    "support_pack_missing_gold_child_id": "support_pack 未保留目标 child id",
    "citation_candidates_missing_gold_child_id": "citation_candidates 未保留目标 child id",
    "citation_binding_missing_gold_child_id": "citation_binding 未保留目标 child id",
    "matched_child_chain_ok": "matched child 链路完整",
}

RERANK_REASON_LABELS = {
    "no_gold_parent": "样本缺少目标父块",
    "gold_parent_not_in_raw_retrieval": "目标父块未进入 raw retrieval",
    "rerank_trace_missing": "debug 缺少 rerank ranking_trace",
    "gold_parent_in_final_top10": "目标父块已进入 rerank final top10",
    "score_floor_filtered": "目标父块被 score floor 过滤",
    "doc_diversity_or_topk": "目标父块进入 doc diversity overflow 或被 topK 截断",
    "comparison_selection_or_topk": "目标父块被 comparison selection/topK 挤出",
    "top10_cutoff": "目标父块未过 final top10 截断",
    "same_doc_wrong_parent": "同文档其它父块进入 top10，目标父块未进入",
    "cross_doc_competition_or_low_score": "跨文档竞争或目标父块分数不足",
    "unknown_rerank_miss": "rerank miss 原因未知",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v3 support selector and rerank traces.")
    parser.add_argument("--judged-result-dir", default=str(DEFAULT_JUDGED_DIR))
    parser.add_argument("--debug-result-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.debug_result_dir:
        raise SystemExit("--debug-result-dir is required unless --self-test is used")

    judged_dir = Path(args.judged_result_dir)
    debug_dir = Path(args.debug_result_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"v3_support_rerank_audit_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_support_rerank_audit_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    judged_rows = {variant: load_variant(judged_dir, variant) for variant in VARIANTS}
    debug_rows = {variant: load_variant(debug_dir, variant) for variant in VARIANTS}

    samples_by_variant: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        samples = audit_variant(
            variant=variant,
            judged_rows=judged_rows[variant],
            debug_rows=debug_rows[variant],
        )
        samples_by_variant[variant] = samples
        summaries[variant] = summarize_samples(samples)
        write_jsonl(result_dir / f"{variant}_samples.jsonl", samples)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judged_result_dir": str(judged_dir),
        "debug_result_dir": str(debug_dir),
        "variants": summaries,
        "comparison": compare_variants(samples_by_variant),
    }
    write_json(result_dir / "audit_summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary))


def load_variant(root: Path, variant: str) -> dict[str, dict[str, Any]]:
    path = root / variant / "results.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("sample_id") or "")] = row
    return rows


def audit_variant(
    *,
    variant: str,
    judged_rows: dict[str, dict[str, Any]],
    debug_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sample_ids = sorted(set(judged_rows) | set(debug_rows))
    audited = []
    for sample_id in sample_ids:
        judged = judged_rows.get(sample_id) or {}
        debug = debug_rows.get(sample_id) or judged
        row = audit_sample(judged=judged, debug=debug)
        row["variant_key"] = variant
        audited.append(row)
    return audited


def audit_sample(*, judged: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    base = debug or judged
    expected_docs = [str(value) for value in base.get("expected_doc_ids") or []]
    gold_chunks = [str(value) for value in base.get("gold_chunk_ids") or []]
    gold_children = [chunk_id for chunk_id in gold_chunks if "::child" in chunk_id]
    gold_parents = dedupe(parent_chunk_id(chunk_id) for chunk_id in gold_chunks)
    judge = judged.get("judge") or {}

    debug_digest = debug.get("debug_digest") or {}
    generation = debug_digest.get("generation_v2") or {}
    rerank_hits = debug_digest.get("rerank_hits") or {}

    raw_doc_ids = list(debug.get("raw_retrieved_doc_ids") or stage_values(debug_digest, "retrieval_output", "doc_ids"))
    raw_chunk_ids = list(debug.get("raw_retrieved_chunk_ids") or stage_values(debug_digest, "retrieval_output", "chunk_ids"))
    rerank_doc_ids = list(debug.get("retrieved_doc_ids_top10") or stage_values(debug_digest, "rerank_output", "doc_ids"))
    rerank_chunk_ids = list(debug.get("retrieved_chunk_ids_top10") or stage_values(debug_digest, "rerank_output", "chunk_ids"))
    final_chunk_ids = list(debug.get("final_chunk_ids") or stage_values(debug_digest, "final_chunks", "kept_chunk_ids") or stage_values(debug_digest, "final_chunks", "chunk_ids"))

    raw_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in raw_chunk_ids)
    rerank_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in rerank_chunk_ids)
    final_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in final_chunk_ids)

    candidates = list(generation.get("candidates") or [])
    support_pack = list(generation.get("support_pack") or [])
    citation_candidates = list(generation.get("citation_candidates") or [])
    citation_binding = generation.get("citation_binding") or {}
    support_selector = generation.get("support_selector") or {}
    selection_debug = support_selector.get("selection_debug") if isinstance(support_selector, dict) else {}
    if not isinstance(selection_debug, dict):
        selection_debug = {}

    candidate_parent_ids = parents_from_items(candidates)
    support_parent_ids = parents_from_items(support_pack)
    citation_candidate_parent_ids = parents_from_items(citation_candidates)

    final_child_ids = stage_child_ids(debug_digest.get("final_chunks") or {})
    candidate_child_ids = child_ids_from_items(candidates)
    support_child_ids = child_ids_from_items(support_pack)
    citation_candidate_child_ids = child_ids_from_items(citation_candidates)
    binding_child_ids_all = child_ids_from_binding(citation_binding, ordered_only=False)
    binding_child_ids_ordered = child_ids_from_binding(citation_binding, ordered_only=True)
    citation_output_child_ids = list(debug.get("citation_matched_child_chunk_ids") or binding_child_ids_ordered)

    target_candidates = filter_items_by_parent(candidates, gold_parents)
    target_support_items = filter_items_by_parent(support_pack, gold_parents)
    target_citation_candidates = filter_items_by_parent(citation_candidates, gold_parents)
    target_evidence_ids = [str(item.get("evidence_id") or "") for item in target_candidates]
    drop_reasons_by_eid = selection_debug.get("drop_reasons_by_evidence_id") or {}

    hits = {
        "raw_doc_hit": any_in(expected_docs, raw_doc_ids) if expected_docs else None,
        "raw_parent_hit": any_in(gold_parents, raw_parent_ids) if gold_parents else None,
        "rerank_doc_hit": any_in(expected_docs, rerank_doc_ids) if expected_docs else None,
        "rerank_parent_hit": any_in(gold_parents, rerank_parent_ids) if gold_parents else None,
        "final_parent_hit": any_in(gold_parents, final_parent_ids) if gold_parents else None,
        "candidate_parent_hit": any_in(gold_parents, candidate_parent_ids) if gold_parents else None,
        "support_parent_hit": any_in(gold_parents, support_parent_ids) if gold_parents else None,
        "citation_candidate_parent_hit": any_in(gold_parents, citation_candidate_parent_ids) if gold_parents else None,
        "final_child_hit": any_in(gold_children, final_child_ids) if gold_children else None,
        "candidate_child_hit": any_in(gold_children, candidate_child_ids) if gold_children else None,
        "support_child_hit": any_in(gold_children, support_child_ids) if gold_children else None,
        "citation_candidate_child_hit": any_in(gold_children, citation_candidate_child_ids) if gold_children else None,
        "citation_binding_child_hit": any_in(gold_children, binding_child_ids_ordered) if gold_children else None,
        "citation_output_child_hit": any_in(gold_children, citation_output_child_ids) if gold_children else None,
    }

    matched_child_break = classify_matched_child_break(hits, gold_children)
    support_break = classify_support_break(hits, gold_parents, gold_children)
    rerank_audit = audit_rerank(
        expected_docs=expected_docs,
        gold_parents=gold_parents,
        raw_parent_ids=raw_parent_ids,
        rerank_doc_ids=rerank_doc_ids,
        rerank_parent_ids=rerank_parent_ids,
        rerank_hits=rerank_hits,
    )

    buckets = []
    if support_break not in {"support_chain_ok", "no_gold_parent", "no_gold_child_id", "before_support_final_parent_miss"}:
        buckets.append(support_break)
    if matched_child_break not in {"matched_child_chain_ok", "no_gold_child_id"}:
        buckets.append(matched_child_break)
    if hits["rerank_doc_hit"] is True and hits["rerank_parent_hit"] is False:
        buckets.append("rerank_doc_hit_parent_miss")
    if rerank_audit["reason"] not in {"gold_parent_in_final_top10", "no_gold_parent"}:
        buckets.append(rerank_audit["reason"])
    if nested(judge, "answer_correctness", "correctness_pass") is False and (
        hits["support_child_hit"] is True or hits["citation_output_child_hit"] is True
    ):
        buckets.append("child_evidence_hit_answer_failed")

    return {
        "sample_id": str(base.get("sample_id") or ""),
        "question": base.get("question"),
        "category": base.get("category"),
        "expected_route": base.get("expected_route"),
        "expected_doc_ids": expected_docs,
        "gold_chunk_ids": gold_chunks,
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "answer_correctness_pass": nested(judge, "answer_correctness", "correctness_pass"),
        "hits": hits,
        "support_first_break": support_break,
        "matched_child_first_break": matched_child_break,
        "selector_target_evidence_ids": target_evidence_ids,
        "selector_drop_reasons_for_target": {
            evidence_id: drop_reasons_by_eid.get(evidence_id, "")
            for evidence_id in target_evidence_ids
        },
        "support_score_ranking_for_target": filter_items_by_parent(
            selection_debug.get("support_score_ranking") or [],
            gold_parents,
        ),
        "protected_seed_inserted_evidence_ids": selection_debug.get(
            "protected_seed_inserted_evidence_ids"
        )
        or [],
        "target_support_items": target_support_items,
        "target_citation_candidates": target_citation_candidates,
        "matched_child_trace": {
            "final_chunks": {
                "matched_child_chunk_ids": final_child_ids,
                "matched_child_chunk_ids_by_chunk_id": (debug_digest.get("final_chunks") or {}).get(
                    "matched_child_chunk_ids_by_chunk_id", {}
                ),
            },
            "generation_candidates": child_map_from_items(candidates),
            "support_pack": child_map_from_items(support_pack),
            "citation_candidates": child_map_from_items(citation_candidates),
            "citation_binding_all_child_ids": binding_child_ids_all,
            "citation_binding_ordered_child_ids": binding_child_ids_ordered,
            "citation_output_child_ids": citation_output_child_ids,
        },
        "rerank_audit": rerank_audit,
        "raw_retrieved_parent_chunk_ids": raw_parent_ids,
        "rerank_parent_chunk_ids_top10": rerank_parent_ids,
        "final_parent_chunk_ids": final_parent_ids,
        "candidate_parent_chunk_ids": candidate_parent_ids,
        "support_parent_chunk_ids": support_parent_ids,
        "citation_candidate_parent_chunk_ids": citation_candidate_parent_ids,
        "audit_buckets": dedupe(buckets),
    }


def classify_support_break(
    hits: dict[str, Any],
    gold_parents: list[str],
    gold_children: list[str],
) -> str:
    if not gold_parents:
        return "no_gold_parent"
    if hits["final_parent_hit"] is not True:
        return "before_support_final_parent_miss"
    if hits["candidate_parent_hit"] is not True:
        return "final_parent_hit_candidate_missing"
    if hits["support_parent_hit"] is not True:
        return "candidate_parent_hit_selector_miss"
    if not gold_children:
        return "no_gold_child_id"
    if hits["support_child_hit"] is not True:
        return "support_parent_hit_child_metadata_miss"
    if hits["citation_candidate_child_hit"] is not True:
        return "support_child_hit_citation_candidate_miss"
    if hits["citation_binding_child_hit"] is not True:
        return "citation_candidate_child_hit_binding_miss"
    if hits["citation_output_child_hit"] is not True:
        return "binding_child_hit_citation_output_miss"
    return "support_chain_ok"


def classify_matched_child_break(hits: dict[str, Any], gold_children: list[str]) -> str:
    if not gold_children:
        return "no_gold_child_id"
    if hits["final_child_hit"] is not True:
        return "final_chunks_missing_gold_child_id"
    if hits["candidate_child_hit"] is not True:
        return "generation_candidates_missing_gold_child_id"
    if hits["support_child_hit"] is not True:
        return "support_pack_missing_gold_child_id"
    if hits["citation_candidate_child_hit"] is not True:
        return "citation_candidates_missing_gold_child_id"
    if hits["citation_binding_child_hit"] is not True:
        return "citation_binding_missing_gold_child_id"
    return "matched_child_chain_ok"


def audit_rerank(
    *,
    expected_docs: list[str],
    gold_parents: list[str],
    raw_parent_ids: list[str],
    rerank_doc_ids: list[str],
    rerank_parent_ids: list[str],
    rerank_hits: dict[str, Any],
) -> dict[str, Any]:
    if not gold_parents:
        return {"reason": "no_gold_parent", "target_traces": []}
    if not any_in(gold_parents, raw_parent_ids):
        return {
            "reason": "gold_parent_not_in_raw_retrieval",
            "target_raw_parent_rank": first_rank(raw_parent_ids, set(gold_parents)),
            "target_traces": [],
        }

    trace = rerank_hits.get("ranking_trace") or []
    if not isinstance(trace, list) or not trace:
        return {
            "reason": "rerank_trace_missing",
            "target_raw_parent_rank": first_rank(raw_parent_ids, set(gold_parents)),
            "target_traces": [],
        }

    target_traces = [
        item
        for item in trace
        if parent_chunk_id(item.get("chunk_id")) in set(gold_parents)
        or str(item.get("parent_chunk_id") or "") in set(gold_parents)
    ]
    target_traces = sorted(
        target_traces,
        key=lambda item: (
            int(item.get("pre_floor_rerank_rank") or 999999),
            int(item.get("raw_retrieval_rank") or 999999),
        ),
    )
    top_k = int(nested(rerank_hits, "selection", "top_k") or 10)
    cutoff_score = topk_cutoff_score(trace, top_k)
    best_target = target_traces[0] if target_traces else {}
    best_score = to_float(best_target.get("score"))
    same_doc_wrong_parent = any_in(expected_docs, rerank_doc_ids) and not any_in(
        gold_parents, rerank_parent_ids
    )
    score_gap = (
        round(best_score - cutoff_score, 6)
        if best_score is not None and cutoff_score is not None
        else None
    )

    if any(item.get("final_top10_rank") for item in target_traces):
        reason = "gold_parent_in_final_top10"
    elif any(item.get("dropped_by_score_floor") or (item.get("pre_floor_rerank_rank") and not item.get("post_floor_rank")) for item in target_traces):
        reason = "score_floor_filtered"
    elif any(item.get("doc_diversity_overflow") for item in target_traces):
        reason = "doc_diversity_or_topk"
    elif bool(nested(rerank_hits, "selection", "comparison_selection", "applied")):
        reason = "comparison_selection_or_topk"
    elif target_traces and all(
        item.get("post_floor_rank") and int(item.get("post_floor_rank")) > top_k
        for item in target_traces
    ):
        reason = "top10_cutoff"
    elif same_doc_wrong_parent:
        reason = "same_doc_wrong_parent"
    elif target_traces:
        reason = "cross_doc_competition_or_low_score"
    else:
        reason = "unknown_rerank_miss"

    return {
        "reason": reason,
        "target_raw_parent_rank": first_rank(raw_parent_ids, set(gold_parents)),
        "top_k": top_k,
        "top10_cutoff_score": cutoff_score,
        "best_target_score": best_score,
        "score_gap_to_top10_cutoff": score_gap,
        "same_doc_wrong_parent": bool(same_doc_wrong_parent),
        "comparison_selection_applied": bool(
            nested(rerank_hits, "selection", "comparison_selection", "applied")
        ),
        "score_floor": nested(rerank_hits, "selection", "score_floor") or {},
        "target_traces": target_traces,
        "top10_trace": sorted(
            [item for item in trace if item.get("final_top10_rank")],
            key=lambda item: int(item.get("final_top10_rank") or 999999),
        )[:10],
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(samples)
    support_counts = Counter(sample["support_first_break"] for sample in samples)
    child_counts = Counter(sample["matched_child_first_break"] for sample in samples)
    rerank_counts = Counter(sample["rerank_audit"]["reason"] for sample in samples)
    bucket_counts: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        category = str(sample.get("category") or "unknown")
        for bucket in sample["audit_buckets"]:
            bucket_counts[bucket] += 1
            by_category[category][bucket] += 1

    return {
        "sample_count": sample_count,
        "rates": {
            "raw_parent_hit_rate": rate(nested(sample, "hits", "raw_parent_hit") for sample in samples),
            "rerank_doc_hit_rate": rate(nested(sample, "hits", "rerank_doc_hit") for sample in samples),
            "rerank_parent_hit_rate": rate(nested(sample, "hits", "rerank_parent_hit") for sample in samples),
            "final_parent_hit_rate": rate(nested(sample, "hits", "final_parent_hit") for sample in samples),
            "candidate_parent_hit_rate": rate(nested(sample, "hits", "candidate_parent_hit") for sample in samples),
            "support_parent_hit_rate": rate(nested(sample, "hits", "support_parent_hit") for sample in samples),
            "support_child_hit_rate": rate(nested(sample, "hits", "support_child_hit") for sample in samples),
            "citation_candidate_child_hit_rate": rate(nested(sample, "hits", "citation_candidate_child_hit") for sample in samples),
            "citation_output_child_hit_rate": rate(nested(sample, "hits", "citation_output_child_hit") for sample in samples),
            "answer_correctness_pass_rate": rate(sample.get("answer_correctness_pass") for sample in samples),
        },
        "support_first_break_counts": count_table(support_counts, SUPPORT_BREAK_LABELS, sample_count),
        "matched_child_first_break_counts": count_table(child_counts, CHILD_BREAK_LABELS, sample_count),
        "rerank_reason_counts": count_table(rerank_counts, RERANK_REASON_LABELS, sample_count),
        "audit_bucket_counts": {
            key: {
                "count": count,
                "rate": round(count / sample_count, 6) if sample_count else None,
            }
            for key, count in sorted(bucket_counts.items())
        },
        "rerank_doc_hit_parent_miss_count": sum(
            1
            for sample in samples
            if nested(sample, "hits", "rerank_doc_hit") is True
            and nested(sample, "hits", "rerank_parent_hit") is False
        ),
        "by_category": {category: dict(counts) for category, counts in sorted(by_category.items())},
        "top_examples": build_top_examples(samples),
    }


def compare_variants(samples_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    b0 = {sample["sample_id"]: sample for sample in samples_by_variant.get("b0_stable", [])}
    b1 = {sample["sample_id"]: sample for sample in samples_by_variant.get("b1_parent_expansion", [])}
    shared = sorted(set(b0) & set(b1))
    support_improved = [
        sample_id
        for sample_id in shared
        if b0[sample_id].get("support_first_break") != "support_chain_ok"
        and b1[sample_id].get("support_first_break") == "support_chain_ok"
    ]
    support_regressed = [
        sample_id
        for sample_id in shared
        if b0[sample_id].get("support_first_break") == "support_chain_ok"
        and b1[sample_id].get("support_first_break") != "support_chain_ok"
    ]
    rerank_parent_improved = [
        sample_id
        for sample_id in shared
        if nested(b0[sample_id], "hits", "rerank_parent_hit") is not True
        and nested(b1[sample_id], "hits", "rerank_parent_hit") is True
    ]
    rerank_parent_regressed = [
        sample_id
        for sample_id in shared
        if nested(b0[sample_id], "hits", "rerank_parent_hit") is True
        and nested(b1[sample_id], "hits", "rerank_parent_hit") is not True
    ]
    return {
        "shared_sample_count": len(shared),
        "support_chain_improved_count": len(support_improved),
        "support_chain_improved_preview": support_improved[:30],
        "support_chain_regressed_count": len(support_regressed),
        "support_chain_regressed_preview": support_regressed[:30],
        "rerank_parent_improved_count": len(rerank_parent_improved),
        "rerank_parent_improved_preview": rerank_parent_improved[:30],
        "rerank_parent_regressed_count": len(rerank_parent_regressed),
        "rerank_parent_regressed_preview": rerank_parent_regressed[:30],
    }


def build_top_examples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for sample in samples:
        if not sample["audit_buckets"]:
            continue
        rerank = sample["rerank_audit"]
        examples.append(
            {
                "sample_id": sample["sample_id"],
                "category": sample.get("category"),
                "support_first_break": sample["support_first_break"],
                "matched_child_first_break": sample["matched_child_first_break"],
                "rerank_reason": rerank["reason"],
                "gold_parent_chunk_ids": sample["gold_parent_chunk_ids"],
                "selector_drop_reasons_for_target": sample["selector_drop_reasons_for_target"],
                "target_pre_floor_ranks": [
                    item.get("pre_floor_rerank_rank") for item in rerank.get("target_traces", [])
                ],
                "target_final_ranks": [
                    item.get("final_top10_rank") for item in rerank.get("target_traces", [])
                ],
                "score_gap_to_top10_cutoff": rerank.get("score_gap_to_top10_cutoff"),
            }
        )
    return examples[:40]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# v3 support/rerank trace 审计报告",
        "",
        f"- 运行 ID：`{summary['run_id']}`",
        f"- judged 结果目录：`{summary['judged_result_dir']}`",
        f"- debug 结果目录：`{summary['debug_result_dir']}`",
        "",
        "## 阶段命中率",
        "",
        "| 变体 | raw parent hit | rerank doc hit | rerank parent hit | final parent hit | candidate parent hit | support parent hit | support child hit | citation candidate child hit | citation output child hit | 答案通过率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, item in summary["variants"].items():
        rates = item["rates"]
        lines.append(
            f"| {variant} | {pct(rates['raw_parent_hit_rate'])} | {pct(rates['rerank_doc_hit_rate'])} | "
            f"{pct(rates['rerank_parent_hit_rate'])} | {pct(rates['final_parent_hit_rate'])} | "
            f"{pct(rates['candidate_parent_hit_rate'])} | {pct(rates['support_parent_hit_rate'])} | "
            f"{pct(rates['support_child_hit_rate'])} | {pct(rates['citation_candidate_child_hit_rate'])} | "
            f"{pct(rates['citation_output_child_hit_rate'])} | {pct(rates['answer_correctness_pass_rate'])} |"
        )

    for title, key in (
        ("## Support Selector 首个断点", "support_first_break_counts"),
        ("## Matched Child 保留链路首个断点", "matched_child_first_break_counts"),
        ("## Rerank 目标父块原因", "rerank_reason_counts"),
    ):
        lines.extend(["", title, ""])
        for variant, item in summary["variants"].items():
            lines.append(f"### {variant}")
            lines.append("")
            lines.append("| 分类 | 样本数 | 占比 |")
            lines.append("|---|---:|---:|")
            for bucket, row in item[key].items():
                lines.append(f"| {row['label']} (`{bucket}`) | {row['count']} | {pct(row['rate'])} |")
            lines.append("")

    lines.extend(["## Rerank doc hit 但 parent miss", ""])
    for variant, item in summary["variants"].items():
        lines.append(f"- `{variant}`：{item['rerank_doc_hit_parent_miss_count']} 条")

    comparison = summary["comparison"]
    lines.extend(
        [
            "",
            "## B1 相对 B0",
            "",
            f"- 共同样本数：{comparison['shared_sample_count']}",
            f"- support 链路改善：{comparison['support_chain_improved_count']} 条",
            f"- support 链路回退：{comparison['support_chain_regressed_count']} 条",
            f"- rerank parent hit 改善：{comparison['rerank_parent_improved_count']} 条",
            f"- rerank parent hit 回退：{comparison['rerank_parent_regressed_count']} 条",
            "",
            "## 典型样本",
            "",
        ]
    )
    for variant, item in summary["variants"].items():
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| sample_id | category | support break | child break | rerank reason | gold parent | selector drop reason | target pre-floor rank | target final rank | score gap |")
        lines.append("|---|---|---|---|---|---|---|---:|---:|---:|")
        for example in item["top_examples"][:20]:
            drop_reason = json.dumps(
                example["selector_drop_reasons_for_target"],
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(
                f"| {example['sample_id']} | {example['category']} | "
                f"`{example['support_first_break']}` | `{example['matched_child_first_break']}` | "
                f"`{example['rerank_reason']}` | {', '.join(example['gold_parent_chunk_ids'])} | "
                f"{drop_reason} | {', '.join(str(v) for v in example['target_pre_floor_ranks'])} | "
                f"{', '.join(str(v) for v in example['target_final_ranks'])} | "
                f"{fmt(example['score_gap_to_top10_cutoff'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def count_table(counter: Counter[str], labels: dict[str, str], sample_count: int) -> dict[str, dict[str, Any]]:
    keys = list(labels)
    for key in counter:
        if key not in labels:
            keys.append(key)
    return {
        key: {
            "label": labels.get(key, key),
            "count": int(counter.get(key, 0)),
            "rate": round(counter.get(key, 0) / sample_count, 6) if sample_count else None,
        }
        for key in keys
    }


def stage_values(debug_digest: dict[str, Any], stage: str, key: str) -> list[str]:
    value = (debug_digest.get(stage) or {}).get(key) or []
    return [str(item) for item in value]


def stage_child_ids(stage: dict[str, Any]) -> list[str]:
    return [str(item) for item in stage.get("matched_child_chunk_ids") or []]


def parents_from_items(items: list[dict[str, Any]]) -> list[str]:
    return dedupe(parent_chunk_id(item.get("chunk_id")) for item in items if isinstance(item, dict))


def child_ids_from_items(items: list[dict[str, Any]]) -> list[str]:
    child_ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        child_ids.extend(str(value) for value in item.get("matched_child_chunk_ids") or [])
    return dedupe(child_ids)


def child_map_from_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or item.get("chunk_id") or "")
        if not evidence_id:
            continue
        result[evidence_id] = {
            "chunk_id": item.get("chunk_id"),
            "parent_chunk_id": parent_chunk_id(item.get("chunk_id")),
            "doc_id": item.get("doc_id"),
            "matched_child_chunk_ids": [
                str(value) for value in item.get("matched_child_chunk_ids") or []
            ],
        }
    return result


def child_ids_from_binding(binding: dict[str, Any], *, ordered_only: bool) -> list[str]:
    if not isinstance(binding, dict):
        return []
    child_map = binding.get("matched_child_chunk_ids_by_evidence_id") or {}
    if not isinstance(child_map, dict):
        return []
    evidence_ids = binding.get("ordered_evidence_ids") if ordered_only else child_map.keys()
    child_ids: list[str] = []
    for evidence_id in evidence_ids or []:
        child_ids.extend(str(value) for value in child_map.get(str(evidence_id), []) or [])
    return dedupe(child_ids)


def filter_items_by_parent(items: Any, gold_parents: list[str]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    parent_set = set(gold_parents)
    return [
        item
        for item in items
        if isinstance(item, dict)
        and (
            parent_chunk_id(item.get("chunk_id")) in parent_set
            or str(item.get("parent_chunk_id") or "") in parent_set
        )
    ]


def topk_cutoff_score(trace: list[dict[str, Any]], top_k: int) -> float | None:
    post_floor = [
        item
        for item in trace
        if isinstance(item.get("post_floor_rank"), (int, float))
        and int(item.get("post_floor_rank")) == top_k
    ]
    if post_floor:
        return to_float(post_floor[0].get("score"))
    final_ranked = [
        item
        for item in trace
        if isinstance(item.get("final_top10_rank"), (int, float))
    ]
    if not final_ranked:
        return None
    final_ranked.sort(key=lambda item: int(item.get("final_top10_rank") or 999999))
    return to_float(final_ranked[-1].get("score"))


def first_rank(values: list[str], targets: set[str]) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value in targets:
            return idx
    return None


def any_in(targets: list[str], values: list[str]) -> bool:
    value_set = {str(value) for value in values}
    return any(str(target) in value_set for target in targets)


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


def nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rate(values: Any) -> float | None:
    vals = [value for value in values if value is not None]
    if not vals:
        return None
    return round(sum(1 for value in vals if bool(value)) / len(vals), 6)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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
    judged = {
        "sample_id": "s1",
        "judge": {
            "answer_correctness": {"correctness_pass": False},
            "faithfulness": {},
            "citation_accuracy": {},
        },
    }
    debug = {
        "sample_id": "s1",
        "question": "q",
        "category": "table_content",
        "expected_doc_ids": ["doc_a"],
        "gold_chunk_ids": ["doc_a_sec01_chunk01::child001"],
        "raw_retrieved_doc_ids": ["doc_a"],
        "raw_retrieved_chunk_ids": ["doc_a_sec01_chunk01"],
        "retrieved_doc_ids_top10": ["doc_a"],
        "retrieved_chunk_ids_top10": ["doc_a_sec02_chunk01"],
        "final_chunk_ids": ["doc_a_sec01_chunk01"],
        "debug_digest": {
            "final_chunks": {
                "matched_child_chunk_ids": ["doc_a_sec01_chunk01::child001"],
                "matched_child_chunk_ids_by_chunk_id": {
                    "doc_a_sec01_chunk01": ["doc_a_sec01_chunk01::child001"]
                },
            },
            "rerank_hits": {
                "selection": {
                    "top_k": 10,
                    "score_floor": {"dropped_chunk_ids": ["doc_a_sec01_chunk01"]},
                    "comparison_selection": {"applied": False},
                },
                "ranking_trace": [
                    {
                        "chunk_id": "doc_a_sec01_chunk01",
                        "parent_chunk_id": "doc_a_sec01_chunk01",
                        "doc_id": "doc_a",
                        "raw_retrieval_rank": 1,
                        "pre_floor_rerank_rank": 11,
                        "post_floor_rank": None,
                        "final_top10_rank": None,
                        "score": 0.3,
                        "dropped_by_score_floor": True,
                    }
                ],
            },
            "generation_v2": {
                "candidates": [
                    {
                        "evidence_id": "E1",
                        "chunk_id": "doc_a_sec01_chunk01",
                        "parent_chunk_id": "doc_a_sec01_chunk01",
                        "doc_id": "doc_a",
                        "matched_child_chunk_ids": ["doc_a_sec01_chunk01::child001"],
                    }
                ],
                "support_selector": {
                    "selection_debug": {
                        "drop_reasons_by_evidence_id": {"E1": "score_too_low"},
                        "support_score_ranking": [
                            {
                                "evidence_id": "E1",
                                "chunk_id": "doc_a_sec01_chunk01",
                                "parent_chunk_id": "doc_a_sec01_chunk01",
                                "support_score": 0.2,
                            }
                        ],
                    }
                },
                "support_pack": [],
                "citation_candidates": [],
                "citation_binding": {
                    "ordered_evidence_ids": [],
                    "matched_child_chunk_ids_by_evidence_id": {},
                },
            },
        },
    }
    row = audit_sample(judged=judged, debug=debug)
    assert row["support_first_break"] == "candidate_parent_hit_selector_miss"
    assert row["matched_child_first_break"] == "support_pack_missing_gold_child_id"
    assert row["rerank_audit"]["reason"] == "score_floor_filtered"
    assert row["selector_drop_reasons_for_target"] == {"E1": "score_too_low"}
    summary = summarize_samples([row])
    assert summary["sample_count"] == 1
    assert summary["support_first_break_counts"]["candidate_parent_hit_selector_miss"]["count"] == 1
    print("self-test passed")


if __name__ == "__main__":
    main()
