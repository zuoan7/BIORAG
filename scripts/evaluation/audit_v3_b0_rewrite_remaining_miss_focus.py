from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_TRANSITION_DIR = RESULTS_ROOT / "v3_b0_rewrite_transition_20260523"
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
DEFAULT_CHILD_CHUNKS = Path("data/paper_round1/chunks/child_chunks.jsonl")

TARGET_FIRST_BREAKS = (
    "doc_hit_parent_miss",
    "parent_hit_support_parent_miss",
    "support_parent_hit_child_miss",
)
DOC_PARENT_CLASSES = (
    "comparison_or_summary_multi_parent",
    "gold_parent_not_in_raw_retrieval",
    "gold_child_raw_recalled_but_parent_lost",
    "gold_parent_raw_but_rerank_dropped",
    "same_doc_adjacent_parent_selected",
    "same_doc_far_parent_selected",
    "cross_doc_competition",
)
SUPPORT_SELECTOR_CLASSES = (
    "summary_or_comparison_support_selection_scope",
    "target_parent_missing_from_generation_candidates",
    "support_selector_score_too_low",
    "support_selector_dropped_target_parent",
    "support_selector_selected_other_candidates",
    "support_parent_hit_rule_metric_inconsistency",
    "support_selector_trace_missing",
)
CHILD_MISS_CLASSES = (
    "multi_child_or_summary_label_scope",
    "no_gold_child_id",
    "gold_child_missing_from_child_chunks",
    "gold_child_lost_between_candidate_and_support",
    "target_support_parent_no_matched_child",
    "target_support_parent_wrong_child_matched",
    "wrong_parent_child_only",
    "child_match_trace_missing",
)
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<section>\d+)_chunk(?P<chunk>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused offline audit for v3 B0+rewrite remaining misses."
    )
    parser.add_argument(
        "--transition-samples",
        default=str(DEFAULT_TRANSITION_DIR / "transition_samples.jsonl"),
    )
    parser.add_argument(
        "--transition-summary",
        default=str(DEFAULT_TRANSITION_DIR / "transition_summary.json"),
    )
    parser.add_argument("--rewrite-results", default=str(DEFAULT_REWRITE_RESULTS))
    parser.add_argument("--rewrite-cache", default=str(DEFAULT_REWRITE_CACHE))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--run-id", default="20260523")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    transition_samples_path = Path(args.transition_samples)
    transition_summary_path = Path(args.transition_summary)
    rewrite_results_path = Path(args.rewrite_results)
    rewrite_cache_path = Path(args.rewrite_cache)
    dataset_path = Path(args.dataset)
    child_chunks_path = Path(args.child_chunks)
    run_id = str(args.run_id)

    transition_rows = load_jsonl(transition_samples_path)
    transition_summary = load_json(transition_summary_path)
    rewrite_rows = load_jsonl_by_id(rewrite_results_path)
    cache_rows = load_jsonl_by_id(rewrite_cache_path)
    dataset_rows = load_jsonl_by_id(dataset_path)
    child_records = load_jsonl_by_id(child_chunks_path, key="chunk_id")

    target_rows = [
        row
        for row in transition_rows
        if str(row.get("rewrite_first_break") or "") in TARGET_FIRST_BREAKS
        and (row.get("rewrite_metrics") or {}).get("citation_child_evidence_hit") is False
    ]
    samples = [
        audit_sample(
            transition_row=row,
            rewrite_row=rewrite_rows.get(str(row.get("sample_id") or "")) or {},
            cache_row=cache_rows.get(str(row.get("sample_id") or "")) or {},
            dataset_row=dataset_rows.get(str(row.get("sample_id") or "")) or {},
            child_records=child_records,
        )
        for row in target_rows
    ]
    summary = build_summary(
        run_id=run_id,
        transition_samples_path=transition_samples_path,
        transition_summary_path=transition_summary_path,
        rewrite_results_path=rewrite_results_path,
        rewrite_cache_path=rewrite_cache_path,
        dataset_path=dataset_path,
        child_chunks_path=child_chunks_path,
        transition_summary=transition_summary,
        samples=samples,
    )

    result_dir = RESULTS_ROOT / f"v3_b0_rewrite_remaining_miss_focus_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_b0_rewrite_remaining_miss_focus_{run_id}"
    write_json(result_dir / "focus_summary.json", summary)
    write_jsonl(result_dir / "focus_samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "target_sample_count": len(samples),
            },
            ensure_ascii=False,
        )
    )


def audit_sample(
    *,
    transition_row: dict[str, Any],
    rewrite_row: dict[str, Any],
    cache_row: dict[str, Any],
    dataset_row: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(transition_row.get("sample_id") or rewrite_row.get("sample_id") or "")
    expected_route = str(
        transition_row.get("expected_route")
        or dataset_row.get("expected_route")
        or rewrite_row.get("expected_route")
        or "unknown"
    )
    category = str(
        transition_row.get("category")
        or dataset_row.get("category")
        or rewrite_row.get("category")
        or "unknown"
    )
    gold_children = as_str_list(
        transition_row.get("gold_chunk_ids")
        or dataset_row.get("gold_chunk_ids")
        or rewrite_row.get("gold_chunk_ids")
    )
    gold_children = [chunk_id for chunk_id in gold_children if "::child" in chunk_id]
    gold_parents = as_str_list(
        transition_row.get("gold_parent_chunk_ids")
        or rewrite_row.get("gold_parent_chunk_ids")
        or [parent_chunk_id(chunk_id) for chunk_id in gold_children]
    )
    expected_docs = as_str_list(
        transition_row.get("expected_doc_ids")
        or dataset_row.get("expected_doc_ids")
        or rewrite_row.get("expected_doc_ids")
    )

    debug_digest = rewrite_row.get("debug_digest") or {}
    generation = debug_digest.get("generation_v2") or {}
    rerank_hits = debug_digest.get("rerank_hits") or {}
    trace = list(rerank_hits.get("ranking_trace") or [])
    raw_child_trace = list((debug_digest.get("raw_child_trace") or {}).get("raw_child_trace") or [])
    raw_parent_ids = as_str_list(
        rewrite_row.get("raw_retrieved_parent_chunk_ids")
        or stage_values(debug_digest, "retrieval_output", "parent_chunk_ids")
        or rewrite_row.get("raw_retrieved_chunk_ids")
    )
    top10_parent_ids = as_str_list(
        transition_row.get("rewrite_top10_parent_chunk_ids")
        or rewrite_row.get("retrieved_parent_chunk_ids_top10")
    )
    support_parent_ids = dedupe(
        parent_chunk_id(chunk_id)
        for chunk_id in as_str_list(
            transition_row.get("rewrite_support_chunk_ids") or rewrite_row.get("support_chunk_ids")
        )
    )
    support_child_ids = as_str_list(
        transition_row.get("rewrite_support_matched_child_ids")
        or rewrite_row.get("support_matched_child_chunk_ids")
    )
    citation_child_ids = as_str_list(
        transition_row.get("rewrite_citation_matched_child_ids")
        or rewrite_row.get("citation_matched_child_chunk_ids")
    )

    common = {
        "sample_id": sample_id,
        "focus_bucket": str(transition_row.get("rewrite_first_break") or ""),
        "category": category,
        "expected_route": expected_route,
        "difficulty": transition_row.get("difficulty") or dataset_row.get("difficulty"),
        "question": transition_row.get("question") or dataset_row.get("question"),
        "rewritten_query": transition_row.get("rewritten_query") or cache_row.get("rewritten_query"),
        "rewrite_cache_source": transition_row.get("rewrite_cache_source") or cache_row.get("source"),
        "expected_doc_ids": expected_docs,
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "gold_children_exist_in_child_chunks": {
            child_id: child_id in child_records for child_id in gold_children
        },
        "rewrite_metrics": transition_row.get("rewrite_metrics") or {},
        "rewrite_top10_parent_chunk_ids": top10_parent_ids,
        "rewrite_support_parent_chunk_ids": support_parent_ids,
        "rewrite_support_matched_child_ids": support_child_ids,
        "rewrite_citation_matched_child_ids": citation_child_ids,
        "raw_parent_rank": first_rank(raw_parent_ids, set(gold_parents)),
        "rerank_target_trace": summarize_target_trace(trace, gold_parents),
        "same_doc_context": summarize_same_doc_context(top10_parent_ids, expected_docs, gold_parents),
        "raw_child_context": summarize_raw_child_context(
            raw_child_trace=raw_child_trace,
            expected_docs=expected_docs,
            gold_parents=gold_parents,
            gold_children=gold_children,
        ),
    }

    focus_bucket = common["focus_bucket"]
    if focus_bucket == "doc_hit_parent_miss":
        classification = classify_doc_parent_miss(common)
        common["secondary_break_classification"] = classification
        common["recommended_next_bucket"] = recommend_doc_parent_bucket(classification)
        common["fix_hint"] = doc_parent_fix_hint(classification, common)
    elif focus_bucket == "parent_hit_support_parent_miss":
        support_audit = audit_support_selector(
            generation=generation,
            gold_parents=gold_parents,
            expected_route=expected_route,
            category=category,
        )
        common["support_selector_audit"] = support_audit
        common["secondary_break_classification"] = support_audit["classification"]
        common["recommended_next_bucket"] = support_audit["recommended_next_bucket"]
        common["fix_hint"] = support_audit["fix_hint"]
    elif focus_bucket == "support_parent_hit_child_miss":
        child_audit = audit_child_miss(
            generation=generation,
            gold_parents=gold_parents,
            gold_children=gold_children,
            child_records=child_records,
            support_child_ids=support_child_ids,
            expected_route=expected_route,
            category=category,
        )
        common["child_match_audit"] = child_audit
        common["secondary_break_classification"] = child_audit["classification"]
        common["recommended_next_bucket"] = child_audit["recommended_next_bucket"]
        common["fix_hint"] = child_audit["fix_hint"]
    else:
        common["secondary_break_classification"] = "out_of_scope"
        common["recommended_next_bucket"] = "out_of_scope"
        common["fix_hint"] = "Not part of this focused audit."

    return common


def summarize_target_trace(trace: list[Any], gold_parents: list[str]) -> dict[str, Any]:
    target_set = set(gold_parents)
    target_items = [
        item
        for item in trace
        if isinstance(item, dict)
        and (
            parent_chunk_id(item.get("chunk_id")) in target_set
            or str(item.get("parent_chunk_id") or "") in target_set
        )
    ]
    target_items = sorted(
        target_items,
        key=lambda item: (
            int(item.get("pre_floor_rerank_rank") or 999999),
            int(item.get("raw_retrieval_rank") or 999999),
        ),
    )
    top_k = 10
    cutoff_score = topk_cutoff_score(trace, top_k)
    best = target_items[0] if target_items else {}
    best_score = to_float(best.get("score"))
    return {
        "trace_present": bool(trace),
        "target_trace_count": len(target_items),
        "best_target": compact_trace_item(best),
        "target_final_top10_rank": first_non_null(item.get("final_top10_rank") for item in target_items),
        "target_pre_floor_rank": first_non_null(item.get("pre_floor_rerank_rank") for item in target_items),
        "target_post_floor_rank": first_non_null(item.get("post_floor_rank") for item in target_items),
        "target_raw_retrieval_rank": first_non_null(item.get("raw_retrieval_rank") for item in target_items),
        "target_drop_reasons": dedupe(str(item.get("final_drop_reason") or "") for item in target_items),
        "target_dropped_by_score_floor": any(bool(item.get("dropped_by_score_floor")) for item in target_items),
        "target_doc_diversity_overflow": any(bool(item.get("doc_diversity_overflow")) for item in target_items),
        "top10_cutoff_score": cutoff_score,
        "best_target_score": best_score,
        "score_gap_to_top10_cutoff": round(best_score - cutoff_score, 6)
        if best_score is not None and cutoff_score is not None
        else None,
    }


def summarize_same_doc_context(
    top10_parent_ids: list[str],
    expected_docs: list[str],
    gold_parents: list[str],
) -> dict[str, Any]:
    docs = set(expected_docs) | {doc_id_from_parent(parent_id) for parent_id in gold_parents}
    docs.discard("")
    same_doc_parents = [
        parent_id for parent_id in top10_parent_ids if doc_id_from_parent(parent_id) in docs
    ]
    nearest = nearest_parent_distance(same_doc_parents, gold_parents)
    if nearest is None:
        relation = "no_same_doc_parent_selected"
    elif nearest <= 1:
        relation = "adjacent"
    else:
        relation = "far"
    return {
        "same_doc_selected_parent_ids": same_doc_parents,
        "nearest_parent_distance": nearest,
        "same_doc_relation": relation,
    }


def summarize_raw_child_context(
    *,
    raw_child_trace: list[Any],
    expected_docs: list[str],
    gold_parents: list[str],
    gold_children: list[str],
) -> dict[str, Any]:
    expected_doc_set = set(expected_docs)
    gold_parent_set = set(gold_parents)
    gold_child_set = set(gold_children)
    child_ids = [
        str(item.get("child_chunk_id") or "")
        for item in raw_child_trace
        if isinstance(item, dict)
    ]
    gold_child_ranks = [
        int(item.get("rank") or 0)
        for item in raw_child_trace
        if isinstance(item, dict) and str(item.get("child_chunk_id") or "") in gold_child_set
    ]
    gold_parent_child_ranks = [
        int(item.get("rank") or 0)
        for item in raw_child_trace
        if isinstance(item, dict) and str(item.get("parent_chunk_id") or "") in gold_parent_set
    ]
    expected_doc_ranks = [
        int(item.get("rank") or 0)
        for item in raw_child_trace
        if isinstance(item, dict) and str(item.get("doc_id") or "") in expected_doc_set
    ]
    return {
        "trace_present": bool(raw_child_trace),
        "raw_child_count": len(raw_child_trace),
        "gold_child_raw_hit": bool(gold_child_ranks) if gold_children else None,
        "gold_child_raw_ranks": sorted(rank for rank in gold_child_ranks if rank),
        "gold_parent_any_child_raw_hit": bool(gold_parent_child_ranks),
        "gold_parent_child_raw_ranks": sorted(rank for rank in gold_parent_child_ranks if rank),
        "expected_doc_child_raw_hit": bool(expected_doc_ranks),
        "expected_doc_first_child_rank": min(
            [rank for rank in expected_doc_ranks if rank],
            default=None,
        ),
        "raw_child_ids_contains_gold": any(child_id in set(child_ids) for child_id in gold_children),
    }


def classify_doc_parent_miss(sample: dict[str, Any]) -> str:
    expected_route = str(sample.get("expected_route") or "")
    category = str(sample.get("category") or "")
    raw_child = sample.get("raw_child_context") or {}
    trace = sample.get("rerank_target_trace") or {}
    same_doc = sample.get("same_doc_context") or {}

    if expected_route in {"comparison", "summary"} or category in {"comparison", "summary_review"}:
        return "comparison_or_summary_multi_parent"
    if sample.get("raw_parent_rank") is None:
        if raw_child.get("gold_child_raw_hit") is True or raw_child.get("gold_parent_any_child_raw_hit") is True:
            return "gold_child_raw_recalled_but_parent_lost"
        return "gold_parent_not_in_raw_retrieval"
    if trace.get("target_trace_count"):
        if trace.get("target_final_top10_rank") is None:
            return "gold_parent_raw_but_rerank_dropped"
    relation = same_doc.get("same_doc_relation")
    if relation == "adjacent":
        return "same_doc_adjacent_parent_selected"
    if relation == "far":
        return "same_doc_far_parent_selected"
    return "cross_doc_competition"


def recommend_doc_parent_bucket(classification: str) -> str:
    if classification in {
        "gold_parent_raw_but_rerank_dropped",
        "same_doc_far_parent_selected",
        "cross_doc_competition",
        "gold_child_raw_recalled_but_parent_lost",
    }:
        return "intra_doc_retrieval_or_rerank"
    if classification == "same_doc_adjacent_parent_selected":
        return "data_or_chunk_boundary_or_child_rematch"
    if classification == "gold_parent_not_in_raw_retrieval":
        return "rewrite_guard_or_query_union"
    if classification == "comparison_or_summary_multi_parent":
        return "metric_scope_or_multi_parent_support"
    return "intra_doc_retrieval_or_rerank"


def doc_parent_fix_hint(classification: str, sample: dict[str, Any]) -> str:
    trace = sample.get("rerank_target_trace") or {}
    relation = (sample.get("same_doc_context") or {}).get("same_doc_relation")
    if classification == "gold_parent_raw_but_rerank_dropped":
        reasons = ", ".join(trace.get("target_drop_reasons") or []) or "topK/rerank"
        return f"Gold parent entered raw retrieval but was dropped before final top10 ({reasons}); inspect rerank score floor/topK and same-doc ranking."
    if classification == "gold_parent_not_in_raw_retrieval":
        return "Gold parent is absent from raw parent retrieval; inspect rewrite query anchors or query union before selector changes."
    if classification == "gold_child_raw_recalled_but_parent_lost":
        return "Gold child or gold-parent child signal appears in raw child trace but is not materialized as final parent; inspect child-to-parent aggregation."
    if classification == "same_doc_adjacent_parent_selected":
        return "A neighboring parent from the same doc was selected; inspect chunk boundary and child rematch before broad retrieval changes."
    if classification == "same_doc_far_parent_selected":
        return "Same doc was retrieved but a far parent won; inspect intra-doc rerank features and query anchor matching."
    if classification == "comparison_or_summary_multi_parent":
        return "Comparison/summary sample may need multi-parent support or a separate metric interpretation before code changes."
    return f"Expected doc hit without gold parent; same-doc relation={relation}. Inspect competing top10 parents."


def audit_support_selector(
    *,
    generation: dict[str, Any],
    gold_parents: list[str],
    expected_route: str,
    category: str,
) -> dict[str, Any]:
    candidates = [item for item in generation.get("candidates") or [] if isinstance(item, dict)]
    support_pack = [item for item in generation.get("support_pack") or [] if isinstance(item, dict)]
    selector = generation.get("support_selector") or {}
    selection_debug = selector.get("selection_debug") if isinstance(selector, dict) else {}
    if not isinstance(selection_debug, dict):
        selection_debug = {}
    ranking = [
        item for item in selection_debug.get("support_score_ranking") or [] if isinstance(item, dict)
    ]
    target_candidates = filter_items_by_parent(candidates, gold_parents)
    target_support = filter_items_by_parent(support_pack, gold_parents)
    target_ranking = filter_items_by_parent(ranking, gold_parents)
    target_evidence_ids = dedupe(str(item.get("evidence_id") or "") for item in target_candidates)
    drop_reasons_by_evidence = selection_debug.get("drop_reasons_by_evidence_id") or {}
    target_drop_reasons = dedupe(
        str(item.get("drop_reason") or drop_reasons_by_evidence.get(str(item.get("evidence_id") or "")) or "")
        for item in target_ranking or target_candidates
    )

    if expected_route in {"comparison", "summary"} or category in {"comparison", "summary_review"}:
        classification = "summary_or_comparison_support_selection_scope"
    elif target_support:
        classification = "support_parent_hit_rule_metric_inconsistency"
    elif not target_candidates:
        classification = "target_parent_missing_from_generation_candidates"
    elif target_drop_reasons and all(reason == "score_too_low" for reason in target_drop_reasons):
        classification = "support_selector_score_too_low"
    elif target_drop_reasons:
        classification = "support_selector_dropped_target_parent"
    elif target_candidates:
        classification = "support_selector_selected_other_candidates"
    else:
        classification = "support_selector_trace_missing"

    return {
        "classification": classification,
        "recommended_next_bucket": support_selector_bucket(classification),
        "fix_hint": support_selector_fix_hint(classification),
        "target_candidate_count": len(target_candidates),
        "target_support_count": len(target_support),
        "target_support_score_ranking": [
            compact_support_rank_item(item) for item in target_ranking
        ],
        "target_evidence_ids": target_evidence_ids,
        "target_drop_reasons": target_drop_reasons,
        "selected_evidence_ids": as_str_list(selector.get("selected_evidence_ids"))
        if isinstance(selector, dict)
        else [],
        "below_min_score_count": selection_debug.get("below_min_score_count"),
        "eligible_count": selection_debug.get("eligible_count"),
        "candidate_count": selection_debug.get("candidate_count"),
    }


def support_selector_bucket(classification: str) -> str:
    if classification == "target_parent_missing_from_generation_candidates":
        return "generation_candidate_builder"
    if classification == "summary_or_comparison_support_selection_scope":
        return "metric_scope_or_multi_parent_support"
    if classification == "support_parent_hit_rule_metric_inconsistency":
        return "audit_metric_contract"
    return "support_selector"


def support_selector_fix_hint(classification: str) -> str:
    if classification == "support_selector_score_too_low":
        return "Target parent is a candidate but support score floor drops it; inspect support score features/protection for gold parent seeds."
    if classification == "target_parent_missing_from_generation_candidates":
        return "Gold parent reaches retrieval but not generation candidates; inspect final context/candidate construction."
    if classification == "summary_or_comparison_support_selection_scope":
        return "Multi-parent question likely needs route-specific support selection before tuning factoid thresholds."
    if classification == "support_selector_selected_other_candidates":
        return "Target parent candidate exists but selection prefers other evidence; inspect ranking features and protected seed rules."
    return "Inspect support selector debug for target evidence drop reasons."


def audit_child_miss(
    *,
    generation: dict[str, Any],
    gold_parents: list[str],
    gold_children: list[str],
    child_records: dict[str, dict[str, Any]],
    support_child_ids: list[str],
    expected_route: str,
    category: str,
) -> dict[str, Any]:
    candidates = [item for item in generation.get("candidates") or [] if isinstance(item, dict)]
    support_pack = [item for item in generation.get("support_pack") or [] if isinstance(item, dict)]
    target_candidates = filter_items_by_parent(candidates, gold_parents)
    target_support = filter_items_by_parent(support_pack, gold_parents)
    target_candidate_child_ids = child_ids_from_items(target_candidates)
    target_support_child_ids = child_ids_from_items(target_support)
    missing_gold_children = [child_id for child_id in gold_children if child_id not in child_records]

    if expected_route in {"comparison", "summary"} or category in {"comparison", "summary_review"}:
        classification = "multi_child_or_summary_label_scope"
    elif not gold_children:
        classification = "no_gold_child_id"
    elif missing_gold_children:
        classification = "gold_child_missing_from_child_chunks"
    elif any_in(gold_children, target_candidate_child_ids) and not any_in(
        gold_children,
        target_support_child_ids,
    ):
        classification = "gold_child_lost_between_candidate_and_support"
    elif target_support and not target_support_child_ids:
        classification = "target_support_parent_no_matched_child"
    elif target_support_child_ids and not any_in(gold_children, target_support_child_ids):
        classification = "target_support_parent_wrong_child_matched"
    elif support_child_ids and not any_in(gold_children, support_child_ids):
        classification = "wrong_parent_child_only"
    else:
        classification = "child_match_trace_missing"

    return {
        "classification": classification,
        "recommended_next_bucket": child_miss_bucket(classification),
        "fix_hint": child_miss_fix_hint(classification),
        "target_candidate_count": len(target_candidates),
        "target_support_count": len(target_support),
        "target_candidate_child_ids": target_candidate_child_ids,
        "target_support_child_ids": target_support_child_ids,
        "global_support_child_ids": support_child_ids,
        "missing_gold_children": missing_gold_children,
        "target_support_items": [compact_support_item(item) for item in target_support],
    }


def child_miss_bucket(classification: str) -> str:
    if classification in {"gold_child_missing_from_child_chunks", "multi_child_or_summary_label_scope"}:
        return "data_or_metric_scope"
    return "child_rematch_or_binding"


def child_miss_fix_hint(classification: str) -> str:
    if classification == "target_support_parent_no_matched_child":
        return "Gold parent is selected but no child evidence is bound; inspect child rematch and parent-child generation view."
    if classification == "target_support_parent_wrong_child_matched":
        return "Gold parent is selected but wrong child is bound; inspect child rematch scoring within the parent."
    if classification == "gold_child_lost_between_candidate_and_support":
        return "Gold child appears before support but is lost by support packing; inspect support child metadata propagation."
    if classification == "gold_child_missing_from_child_chunks":
        return "Gold child id is not present in child_chunks; inspect dataset/chunk id consistency."
    if classification == "multi_child_or_summary_label_scope":
        return "Summary/comparison labels may need multi-child evidence handling before tuning child rematch."
    return "Inspect child rematch trace and matched_child propagation for the selected parent."


def build_summary(
    *,
    run_id: str,
    transition_samples_path: Path,
    transition_summary_path: Path,
    rewrite_results_path: Path,
    rewrite_cache_path: Path,
    dataset_path: Path,
    child_chunks_path: Path,
    transition_summary: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    by_focus = group_counts(samples, "focus_bucket")
    by_secondary = nested_group_counts(samples, "focus_bucket", "secondary_break_classification")
    by_recommendation = group_counts(samples, "recommended_next_bucket")
    doc_samples = [sample for sample in samples if sample.get("focus_bucket") == "doc_hit_parent_miss"]
    support_samples = [
        sample for sample in samples if sample.get("focus_bucket") == "parent_hit_support_parent_miss"
    ]
    child_samples = [
        sample for sample in samples if sample.get("focus_bucket") == "support_parent_hit_child_miss"
    ]

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "transition_samples": str(transition_samples_path),
            "transition_summary": str(transition_summary_path),
            "rewrite_results": str(rewrite_results_path),
            "rewrite_cache": str(rewrite_cache_path),
            "dataset": str(dataset_path),
            "child_chunks": str(child_chunks_path),
        },
        "source_remaining_miss_first_break": transition_summary.get("remaining_miss_first_break"),
        "target_first_breaks": list(TARGET_FIRST_BREAKS),
        "target_sample_count": len(samples),
        "target_counts_by_focus_bucket": by_focus,
        "secondary_break_counts_by_focus_bucket": by_secondary,
        "recommended_next_bucket_counts": by_recommendation,
        "doc_hit_parent_miss": summarize_doc_parent_samples(doc_samples),
        "support_selector_miss": summarize_support_selector_samples(support_samples),
        "child_rematch_miss": summarize_child_miss_samples(child_samples),
        "category_hotspots": summarize_category_hotspots(samples),
        "top_examples": build_top_examples(samples),
        "next_step_recommendation": recommend_from_focus(samples),
        "acceptance_checks": build_acceptance_checks(samples),
    }


def summarize_doc_parent_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    relation_counts = Counter(
        str((sample.get("same_doc_context") or {}).get("same_doc_relation") or "unknown")
        for sample in samples
    )
    drop_reason_counts: Counter[str] = Counter()
    for sample in samples:
        reasons = (sample.get("rerank_target_trace") or {}).get("target_drop_reasons") or []
        if not reasons:
            drop_reason_counts["no_target_trace_or_no_drop_reason"] += 1
        for reason in reasons:
            drop_reason_counts[str(reason or "empty")] += 1
    return {
        "sample_count": len(samples),
        "secondary_break_counts": ordered_count_table(
            Counter(str(sample.get("secondary_break_classification") or "") for sample in samples),
            DOC_PARENT_CLASSES,
            len(samples),
        ),
        "same_doc_relation_counts": count_table(relation_counts, len(samples)),
        "target_drop_reason_counts": count_table(drop_reason_counts, len(samples)),
        "raw_parent_rank_present_count": sum(
            1 for sample in samples if sample.get("raw_parent_rank") is not None
        ),
        "gold_child_raw_hit_count": sum(
            1
            for sample in samples
            if (sample.get("raw_child_context") or {}).get("gold_child_raw_hit") is True
        ),
    }


def summarize_support_selector_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "secondary_break_counts": ordered_count_table(
            Counter(str(sample.get("secondary_break_classification") or "") for sample in samples),
            SUPPORT_SELECTOR_CLASSES,
            len(samples),
        ),
        "target_candidate_present_count": sum(
            1
            for sample in samples
            if (sample.get("support_selector_audit") or {}).get("target_candidate_count", 0) > 0
        ),
        "score_too_low_count": sum(
            1
            for sample in samples
            if sample.get("secondary_break_classification") == "support_selector_score_too_low"
        ),
    }


def summarize_child_miss_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "secondary_break_counts": ordered_count_table(
            Counter(str(sample.get("secondary_break_classification") or "") for sample in samples),
            CHILD_MISS_CLASSES,
            len(samples),
        ),
        "target_support_parent_without_child_count": sum(
            1
            for sample in samples
            if sample.get("secondary_break_classification")
            == "target_support_parent_no_matched_child"
        ),
        "wrong_child_count": sum(
            1
            for sample in samples
            if sample.get("secondary_break_classification")
            == "target_support_parent_wrong_child_matched"
        ),
    }


def summarize_category_hotspots(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("category") or "unknown")].append(sample)
    rows = []
    for category, group_samples in grouped.items():
        rec_counts = Counter(str(sample.get("recommended_next_bucket") or "") for sample in group_samples)
        focus_counts = Counter(str(sample.get("focus_bucket") or "") for sample in group_samples)
        top_rec = rec_counts.most_common(1)[0][0] if rec_counts else ""
        rows.append(
            {
                "category": category,
                "sample_count": len(group_samples),
                "focus_bucket_counts": dict(sorted(focus_counts.items())),
                "recommended_next_bucket_counts": dict(sorted(rec_counts.items())),
                "top_recommended_next_bucket": top_rec,
                "top_recommended_next_bucket_count": int(rec_counts.get(top_rec, 0)) if top_rec else 0,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["sample_count"]), str(row["category"])))


def build_top_examples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "doc_hit_parent_miss": 0,
        "support_parent_hit_child_miss": 1,
        "parent_hit_support_parent_miss": 2,
    }
    rows = sorted(
        samples,
        key=lambda sample: (
            priority.get(str(sample.get("focus_bucket") or ""), 99),
            str(sample.get("category") or ""),
            str(sample.get("sample_id") or ""),
        ),
    )
    return [
        {
            "sample_id": sample.get("sample_id"),
            "focus_bucket": sample.get("focus_bucket"),
            "category": sample.get("category"),
            "secondary_break_classification": sample.get("secondary_break_classification"),
            "recommended_next_bucket": sample.get("recommended_next_bucket"),
            "raw_parent_rank": sample.get("raw_parent_rank"),
            "target_pre_floor_rank": nested(sample, "rerank_target_trace", "target_pre_floor_rank"),
            "target_drop_reasons": nested(sample, "rerank_target_trace", "target_drop_reasons") or [],
            "same_doc_relation": nested(sample, "same_doc_context", "same_doc_relation"),
            "nearest_parent_distance": nested(sample, "same_doc_context", "nearest_parent_distance"),
            "fix_hint": sample.get("fix_hint"),
        }
        for sample in rows[:30]
    ]


def recommend_from_focus(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rec_counts = Counter(str(sample.get("recommended_next_bucket") or "") for sample in samples)
    actionable_counts = {
        key: count
        for key, count in rec_counts.items()
        if key not in {"metric_scope_or_multi_parent_support", "data_or_metric_scope"}
    }
    primary_bucket = max(actionable_counts.items(), key=lambda item: (item[1], item[0]))[0] if actionable_counts else ""
    doc_summary = summarize_doc_parent_samples(
        [sample for sample in samples if sample.get("focus_bucket") == "doc_hit_parent_miss"]
    )
    return {
        "primary_bucket": primary_bucket,
        "primary_bucket_count": int(rec_counts.get(primary_bucket, 0)) if primary_bucket else 0,
        "recommended_order": [
            bucket
            for bucket, _ in sorted(rec_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "doc_parent_secondary_top": first_count_key(doc_summary["secondary_break_counts"]),
        "note": "This is an offline audit recommendation; no main-chain changes were made.",
    }


def build_acceptance_checks(samples: list[dict[str, Any]]) -> dict[str, Any]:
    focus_counts = Counter(str(sample.get("focus_bucket") or "") for sample in samples)
    doc_unknown = [
        sample.get("sample_id")
        for sample in samples
        if sample.get("focus_bucket") == "doc_hit_parent_miss"
        and str(sample.get("secondary_break_classification") or "").endswith("unknown")
    ]
    missing_recommendations = [
        sample.get("sample_id")
        for sample in samples
        if not sample.get("recommended_next_bucket") or not sample.get("fix_hint")
    ]
    return {
        "target_sample_count_at_least_83": {
            "actual": len(samples),
            "passed": len(samples) >= 83,
        },
        "doc_hit_parent_miss_count_equals_35": {
            "actual": int(focus_counts.get("doc_hit_parent_miss", 0)),
            "passed": int(focus_counts.get("doc_hit_parent_miss", 0)) == 35,
        },
        "support_parent_hit_child_miss_count_equals_26": {
            "actual": int(focus_counts.get("support_parent_hit_child_miss", 0)),
            "passed": int(focus_counts.get("support_parent_hit_child_miss", 0)) == 26,
        },
        "parent_hit_support_parent_miss_count_equals_22": {
            "actual": int(focus_counts.get("parent_hit_support_parent_miss", 0)),
            "passed": int(focus_counts.get("parent_hit_support_parent_miss", 0)) == 22,
        },
        "doc_hit_parent_miss_no_unknown": {
            "unknown_sample_ids": doc_unknown,
            "passed": not doc_unknown,
        },
        "all_samples_have_recommendation_and_fix_hint": {
            "missing_sample_ids": missing_recommendations,
            "passed": not missing_recommendations,
        },
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 B0+rewrite remaining miss focused audit",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Transition samples: `{summary['inputs']['transition_samples']}`",
        f"- Rewrite results: `{summary['inputs']['rewrite_results']}`",
        f"- Target sample count: {summary['target_sample_count']}",
        "",
        "## Target overview",
        "",
        "| Focus bucket | Count |",
        "|---|---:|",
    ]
    for key, row in summary["target_counts_by_focus_bucket"].items():
        lines.append(f"| `{key}` | {row['count']} |")

    lines.extend(
        [
            "",
            "## Doc-hit parent-miss root causes",
            "",
            "| Classification | Count | Rate |",
            "|---|---:|---:|",
        ]
    )
    for key, row in summary["doc_hit_parent_miss"]["secondary_break_counts"].items():
        lines.append(f"| `{key}` | {row['count']} | {pct(row['rate'])} |")

    lines.extend(["", "Same-doc relation:", "", "| Relation | Count | Rate |", "|---|---:|---:|"])
    for key, row in summary["doc_hit_parent_miss"]["same_doc_relation_counts"].items():
        lines.append(f"| `{key}` | {row['count']} | {pct(row['rate'])} |")

    lines.extend(["", "Target rerank drop reasons:", "", "| Drop reason | Count | Rate |", "|---|---:|---:|"])
    for key, row in summary["doc_hit_parent_miss"]["target_drop_reason_counts"].items():
        lines.append(f"| `{key}` | {row['count']} | {pct(row['rate'])} |")

    lines.extend(
        [
            "",
            "## Support selector misses",
            "",
            "| Classification | Count | Rate |",
            "|---|---:|---:|",
        ]
    )
    for key, row in summary["support_selector_miss"]["secondary_break_counts"].items():
        lines.append(f"| `{key}` | {row['count']} | {pct(row['rate'])} |")

    lines.extend(
        [
            "",
            "## Child rematch/binding misses",
            "",
            "| Classification | Count | Rate |",
            "|---|---:|---:|",
        ]
    )
    for key, row in summary["child_rematch_miss"]["secondary_break_counts"].items():
        lines.append(f"| `{key}` | {row['count']} | {pct(row['rate'])} |")

    lines.extend(
        [
            "",
            "## Category hotspots",
            "",
            "| Category | Samples | Top recommended bucket | Focus buckets |",
            "|---|---:|---|---|",
        ]
    )
    for row in summary["category_hotspots"]:
        lines.append(
            f"| `{row['category']}` | {row['sample_count']} | "
            f"`{row['top_recommended_next_bucket']}` ({row['top_recommended_next_bucket_count']}) | "
            f"{json.dumps(row['focus_bucket_counts'], ensure_ascii=False, sort_keys=True)} |"
        )

    lines.extend(
        [
            "",
            "## High-value examples",
            "",
            "| sample_id | focus | category | classification | next bucket | raw rank | pre-rerank rank | same-doc | drop reasons |",
            "|---|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for example in summary["top_examples"][:20]:
        lines.append(
            f"| `{example['sample_id']}` | `{example['focus_bucket']}` | `{example['category']}` | "
            f"`{example['secondary_break_classification']}` | `{example['recommended_next_bucket']}` | "
            f"{fmt(example['raw_parent_rank'])} | {fmt(example['target_pre_floor_rank'])} | "
            f"`{example['same_doc_relation']}`/{fmt(example['nearest_parent_distance'])} | "
            f"{format_code_list(example['target_drop_reasons'])} |"
        )

    doc_parent = summary["doc_hit_parent_miss"]
    child_miss = summary["child_rematch_miss"]
    adjacent_count = nested(doc_parent, "same_doc_relation_counts", "adjacent", "count") or 0
    missing_child_count = nested(
        child_miss,
        "secondary_break_counts",
        "gold_child_missing_from_child_chunks",
        "count",
    ) or 0
    no_gold_child_count = nested(
        child_miss,
        "secondary_break_counts",
        "no_gold_child_id",
        "count",
    ) or 0
    multi_scope_count = nested(
        child_miss,
        "secondary_break_counts",
        "multi_child_or_summary_label_scope",
        "count",
    ) or 0
    lines.extend(
        [
            "",
            "## Data and chunk-boundary signals",
            "",
            f"- Same-doc adjacent parent selection appears in {adjacent_count} doc-hit parent-miss samples, but it is not the dominant primary root cause.",
            f"- Gold child ids missing from `child_chunks.jsonl`: {missing_child_count}.",
            f"- Child-miss samples with no gold child id or route/metric scope issues: {no_gold_child_count + multi_scope_count}.",
            "- This points to targeted guard/support/selector work before broad neighbor expansion.",
        ]
    )

    rec = summary["next_step_recommendation"]
    lines.extend(
        [
            "",
            "## Next step recommendation",
            "",
            f"- Primary next bucket: `{rec['primary_bucket']}` ({rec['primary_bucket_count']} samples in this focused audit).",
            f"- Recommended order: {format_code_list(rec['recommended_order'])}.",
            f"- Doc-parent top cause: `{rec['doc_parent_secondary_top']}`.",
            "- Do not start neighbor expansion unless adjacent same-doc parent selection dominates; this audit keeps it as a data/chunk-boundary signal, not the default fix.",
            "- No judge calls, no RAG rerun, and no main-chain changes are part of this audit.",
        ]
    )
    return "\n".join(lines)


def group_counts(samples: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    counter = Counter(str(sample.get(key) or "unknown") for sample in samples)
    return count_table(counter, len(samples))


def nested_group_counts(
    samples: list[dict[str, Any]],
    outer_key: str,
    inner_key: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for sample in samples:
        outer = str(sample.get(outer_key) or "unknown")
        inner = str(sample.get(inner_key) or "unknown")
        grouped[outer][inner] += 1
        totals[outer] += 1
    return {
        outer: count_table(counter, totals[outer])
        for outer, counter in sorted(grouped.items())
    }


def count_table(counter: Counter[str], denominator: int) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "count": int(count),
            "rate": safe_rate(int(count), denominator),
        }
        for key, count in sorted(counter.items())
    }


def ordered_count_table(
    counter: Counter[str],
    order: tuple[str, ...],
    denominator: int,
) -> dict[str, dict[str, Any]]:
    keys = list(order)
    keys.extend(sorted(key for key in counter if key not in set(keys)))
    return {
        key: {
            "count": int(counter.get(key, 0)),
            "rate": safe_rate(int(counter.get(key, 0)), denominator),
        }
        for key in keys
    }


def first_count_key(table: dict[str, dict[str, Any]]) -> str:
    nonzero = [(key, row["count"]) for key, row in table.items() if row["count"]]
    if not nonzero:
        return ""
    return max(nonzero, key=lambda item: (item[1], item[0]))[0]


def filter_items_by_parent(items: list[dict[str, Any]], gold_parents: list[str]) -> list[dict[str, Any]]:
    parent_set = set(gold_parents)
    return [
        item
        for item in items
        if parent_chunk_id(item.get("chunk_id")) in parent_set
        or str(item.get("parent_chunk_id") or "") in parent_set
    ]


def child_ids_from_items(items: list[dict[str, Any]]) -> list[str]:
    child_ids: list[str] = []
    for item in items:
        child_ids.extend(str(value) for value in item.get("matched_child_chunk_ids") or [])
    return dedupe(child_ids)


def compact_trace_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or not item:
        return {}
    return {
        "chunk_id": item.get("chunk_id"),
        "parent_chunk_id": item.get("parent_chunk_id"),
        "doc_id": item.get("doc_id"),
        "raw_retrieval_rank": item.get("raw_retrieval_rank"),
        "pre_floor_rerank_rank": item.get("pre_floor_rerank_rank"),
        "post_floor_rank": item.get("post_floor_rank"),
        "final_top10_rank": item.get("final_top10_rank"),
        "score": item.get("score"),
        "final_drop_reason": item.get("final_drop_reason"),
        "dropped_by_score_floor": item.get("dropped_by_score_floor"),
        "doc_diversity_overflow": item.get("doc_diversity_overflow"),
        "section": item.get("section"),
    }


def compact_support_rank_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "chunk_id": item.get("chunk_id"),
        "parent_chunk_id": item.get("parent_chunk_id"),
        "selected": item.get("selected"),
        "drop_reason": item.get("drop_reason"),
        "support_rank": item.get("support_rank"),
        "support_score": item.get("support_score"),
        "rerank_rank": item.get("rerank_rank"),
        "matched_child_chunk_ids": item.get("matched_child_chunk_ids") or [],
    }


def compact_support_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "chunk_id": item.get("chunk_id"),
        "parent_chunk_id": item.get("parent_chunk_id"),
        "generation_evidence_role": item.get("generation_evidence_role"),
        "matched_child_chunk_ids": item.get("matched_child_chunk_ids") or [],
        "parent_child_generation_view_used": item.get("parent_child_generation_view_used"),
        "support_score": item.get("support_score"),
    }


def topk_cutoff_score(trace: list[Any], top_k: int) -> float | None:
    ranked = [
        item
        for item in trace
        if isinstance(item, dict) and isinstance(item.get("final_top10_rank"), (int, float))
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda item: int(item.get("final_top10_rank") or 999999))
    if len(ranked) >= top_k:
        return to_float(ranked[top_k - 1].get("score"))
    return to_float(ranked[-1].get("score"))


def nearest_parent_distance(parent_ids: list[str], gold_parent_ids: list[str]) -> int | None:
    distances: list[int] = []
    for parent_id in parent_ids:
        parsed_parent = parse_parent_id(parent_id)
        if not parsed_parent:
            continue
        for gold_parent_id in gold_parent_ids:
            parsed_gold = parse_parent_id(gold_parent_id)
            if not parsed_gold or parsed_parent["doc"] != parsed_gold["doc"]:
                continue
            distances.append(abs(parsed_parent["chunk"] - parsed_gold["chunk"]))
    return min(distances) if distances else None


def parse_parent_id(parent_id: str) -> dict[str, Any] | None:
    match = PARENT_ID_RE.match(str(parent_id or ""))
    if not match:
        return None
    return {
        "doc": match.group("doc"),
        "section": int(match.group("section")),
        "chunk": int(match.group("chunk")),
    }


def doc_id_from_parent(parent_id: str) -> str:
    parsed = parse_parent_id(parent_id)
    return str(parsed["doc"]) if parsed else ""


def stage_values(debug_digest: dict[str, Any], stage: str, key: str) -> list[str]:
    value = (debug_digest.get(stage) or {}).get(key) or []
    return as_str_list(value)


def first_rank(values: list[str], targets: set[str]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in targets:
            return index
    return None


def first_non_null(values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def any_in(targets: list[str], values: list[str]) -> bool:
    value_set = set(values)
    return any(target in value_set for target in targets)


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


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


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def format_code_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    transition_rows = [
        {
            "sample_id": "s_doc",
            "rewrite_first_break": "doc_hit_parent_miss",
            "category": "normal_factoid",
            "expected_route": "factoid",
            "expected_doc_ids": ["doc_a"],
            "gold_chunk_ids": ["doc_a_sec02_chunk03::child001"],
            "gold_parent_chunk_ids": ["doc_a_sec02_chunk03"],
            "rewrite_top10_parent_chunk_ids": ["doc_a_sec01_chunk02"],
        },
        {
            "sample_id": "s_support",
            "rewrite_first_break": "parent_hit_support_parent_miss",
            "category": "table_content",
            "expected_route": "factoid",
            "expected_doc_ids": ["doc_b"],
            "gold_chunk_ids": ["doc_b_sec01_chunk02::child001"],
            "gold_parent_chunk_ids": ["doc_b_sec01_chunk02"],
        },
        {
            "sample_id": "s_child",
            "rewrite_first_break": "support_parent_hit_child_miss",
            "category": "table_content",
            "expected_route": "factoid",
            "expected_doc_ids": ["doc_c"],
            "gold_chunk_ids": ["doc_c_sec01_chunk02::child001"],
            "gold_parent_chunk_ids": ["doc_c_sec01_chunk02"],
            "rewrite_support_matched_child_ids": [],
        },
    ]
    rewrite_rows = {
        "s_doc": {
            "sample_id": "s_doc",
            "raw_retrieved_parent_chunk_ids": ["doc_a_sec02_chunk03"],
            "debug_digest": {
                "rerank_hits": {
                    "ranking_trace": [
                        {
                            "chunk_id": "doc_a_sec02_chunk03",
                            "parent_chunk_id": "doc_a_sec02_chunk03",
                            "raw_retrieval_rank": 1,
                            "pre_floor_rerank_rank": 2,
                            "post_floor_rank": None,
                            "final_top10_rank": None,
                            "score": 0.1,
                            "final_drop_reason": "score_floor",
                            "dropped_by_score_floor": True,
                        },
                        {
                            "chunk_id": "doc_a_sec01_chunk02",
                            "parent_chunk_id": "doc_a_sec01_chunk02",
                            "final_top10_rank": 1,
                            "score": 1.0,
                        },
                    ]
                }
            },
        },
        "s_support": {
            "sample_id": "s_support",
            "debug_digest": {
                "generation_v2": {
                    "candidates": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "doc_b_sec01_chunk02",
                            "parent_chunk_id": "doc_b_sec01_chunk02",
                        }
                    ],
                    "support_pack": [],
                    "support_selector": {
                        "selected_evidence_ids": [],
                        "selection_debug": {
                            "drop_reasons_by_evidence_id": {"E1": "score_too_low"},
                            "support_score_ranking": [
                                {
                                    "evidence_id": "E1",
                                    "chunk_id": "doc_b_sec01_chunk02",
                                    "parent_chunk_id": "doc_b_sec01_chunk02",
                                    "drop_reason": "score_too_low",
                                }
                            ],
                        },
                    },
                }
            },
        },
        "s_child": {
            "sample_id": "s_child",
            "debug_digest": {
                "generation_v2": {
                    "support_pack": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "doc_c_sec01_chunk02",
                            "parent_chunk_id": "doc_c_sec01_chunk02",
                            "matched_child_chunk_ids": [],
                        }
                    ],
                    "candidates": [],
                }
            },
        },
    }
    child_records = {"doc_c_sec01_chunk02::child001": {}}
    samples = [
        audit_sample(
            transition_row=row,
            rewrite_row=rewrite_rows[row["sample_id"]],
            cache_row={},
            dataset_row={},
            child_records=child_records,
        )
        for row in transition_rows
    ]
    assert samples[0]["secondary_break_classification"] == "gold_parent_raw_but_rerank_dropped"
    assert samples[1]["secondary_break_classification"] == "support_selector_score_too_low"
    assert samples[2]["secondary_break_classification"] == "target_support_parent_no_matched_child"
    summary = build_summary(
        run_id="self_test",
        transition_samples_path=Path("transition_samples.jsonl"),
        transition_summary_path=Path("transition_summary.json"),
        rewrite_results_path=Path("results.jsonl"),
        rewrite_cache_path=Path("rewrite_cache.jsonl"),
        dataset_path=Path("dataset.jsonl"),
        child_chunks_path=Path("child_chunks.jsonl"),
        transition_summary={},
        samples=samples,
    )
    assert summary["target_sample_count"] == 3
    assert summary["doc_hit_parent_miss"]["secondary_break_counts"][
        "gold_parent_raw_but_rerank_dropped"
    ]["count"] == 1
    print("self-test passed")


if __name__ == "__main__":
    main()
