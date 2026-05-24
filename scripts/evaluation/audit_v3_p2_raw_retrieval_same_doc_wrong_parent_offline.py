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
DEFAULT_P1_SAMPLES = RESULTS_ROOT / "v3_p1_score_floor_trace_audit_20260524" / "samples.jsonl"
DEFAULT_P0_SAMPLES = RESULTS_ROOT / "v3_p0_gold_remap_offline_validation_20260524" / "samples.jsonl"
DEFAULT_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_CHILD_CHUNKS = Path("data/paper_round1/chunks/child_chunks.jsonl")
EXPECTED_TARGET_COUNT = 9
PRIMARY_CLASSES = {
    "gold_stable_block_mismatch_same_doc_candidate",
    "current_gold_valid_raw_child_miss",
    "current_gold_valid_parent_only_raw_miss",
    "comparison_partial_gold_scope_gap",
    "gold_parent_missing_from_parent_chunks",
    "gold_child_missing_from_child_chunks",
    "raw_child_trace_missing",
    "unknown_raw_retrieval_gap",
}
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<section>\d+)_chunk(?P<chunk>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline P2 audit for raw retrieval / same-doc wrong-parent misses."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--p1-samples", default=str(DEFAULT_P1_SAMPLES))
    parser.add_argument("--p0-samples", default=str(DEFAULT_P0_SAMPLES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--expected-target-count", type=int, default=EXPECTED_TARGET_COUNT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    p1_samples_path = Path(args.p1_samples)
    p0_samples_path = Path(args.p0_samples)
    results_path = Path(args.results)
    dataset_path = Path(args.dataset)
    parent_chunks_path = Path(args.parent_chunks)
    child_chunks_path = Path(args.child_chunks)

    p1_rows = load_jsonl(p1_samples_path)
    p0_rows = load_jsonl_by_id(p0_samples_path, "sample_id")
    result_rows = load_jsonl_by_id(results_path, "sample_id")
    dataset_rows = load_jsonl_by_id(dataset_path, "sample_id")
    target_ids = [
        str(row.get("sample_id") or "")
        for row in p1_rows
        if row.get("primary_classification") == "raw_parent_absent"
    ]
    target_context = build_target_context(target_ids, p0_rows, dataset_rows)
    chunk_scan = scan_chunk_indexes(
        parent_chunks_path=parent_chunks_path,
        child_chunks_path=child_chunks_path,
        targets=target_context,
    )
    samples = [
        audit_sample(
            sample_id=sample_id,
            p0_row=p0_rows[sample_id],
            p1_row=next(row for row in p1_rows if row.get("sample_id") == sample_id),
            result_row=result_rows[sample_id],
            dataset_row=dataset_rows[sample_id],
            chunk_scan=chunk_scan,
        )
        for sample_id in target_ids
    ]
    summary = build_summary(
        run_id=str(args.run_id),
        input_paths={
            "p1_samples": str(p1_samples_path),
            "p0_samples": str(p0_samples_path),
            "results": str(results_path),
            "dataset": str(dataset_path),
            "parent_chunks": str(parent_chunks_path),
            "child_chunks": str(child_chunks_path),
        },
        target_ids=target_ids,
        samples=samples,
        expected_target_count=args.expected_target_count,
    )

    result_dir = RESULTS_ROOT / f"v3_p2_raw_retrieval_same_doc_wrong_parent_audit_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_p2_raw_retrieval_same_doc_wrong_parent_audit_{args.run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "validation_passed": summary["validation"]["passed"],
                "target_sample_count": len(samples),
                "classification_counts": summary["classification_counts"],
                "what_if_parent_hit_count": summary["stable_block_full_cover_what_if"][
                    "parent_hit_count"
                ],
            },
            ensure_ascii=False,
        )
    )


def audit_sample(
    *,
    sample_id: str,
    p0_row: dict[str, Any],
    p1_row: dict[str, Any],
    result_row: dict[str, Any],
    dataset_row: dict[str, Any],
    chunk_scan: dict[str, Any],
) -> dict[str, Any]:
    expected_docs = as_str_list(
        p0_row.get("expected_doc_ids") or dataset_row.get("expected_doc_ids")
    )
    gold_parent_ids = as_str_list(p0_row.get("recomputed_gold_parent_chunk_ids"))
    gold_chunk_ids = as_str_list(p0_row.get("recomputed_gold_chunk_ids"))
    gold_child_ids = [chunk_id for chunk_id in gold_chunk_ids if "::child" in chunk_id]
    stable_blocks = as_str_list(dataset_row.get("stable_target_block_ids"))
    parent_records = chunk_scan["parent_records"]
    child_records = chunk_scan["child_records"]
    parent_candidates = chunk_scan["stable_parent_candidates_by_sample"].get(sample_id) or []
    full_cover_candidates = [item for item in parent_candidates if item["covers_all_stable_blocks"]]
    full_cover_parent_ids = [item["chunk_id"] for item in full_cover_candidates]
    gold_block_status = build_gold_block_status(
        gold_parent_ids=gold_parent_ids,
        stable_blocks=stable_blocks,
        parent_records=parent_records,
    )
    debug = result_row.get("debug_digest") or {}
    raw_trace = [
        item
        for item in ((debug.get("raw_child_trace") or {}).get("raw_child_trace") or [])
        if isinstance(item, dict)
    ]
    retrieval_output = debug.get("retrieval_output") or {}
    raw_summary = summarize_raw_trace(
        raw_trace=raw_trace,
        expected_docs=expected_docs,
        gold_parent_ids=gold_parent_ids,
        gold_child_ids=gold_child_ids,
        full_cover_parent_ids=full_cover_parent_ids,
    )
    stage_summary = build_stage_summary(
        p0_row=p0_row,
        result_row=result_row,
        retrieval_output=retrieval_output,
        full_cover_parent_ids=full_cover_parent_ids,
    )
    primary_class = classify_primary(
        expected_route=str(p0_row.get("expected_route") or dataset_row.get("expected_route") or ""),
        gold_parent_ids=gold_parent_ids,
        gold_child_ids=gold_child_ids,
        stable_blocks=stable_blocks,
        raw_trace=raw_trace,
        parent_records=parent_records,
        child_records=child_records,
        gold_block_status=gold_block_status,
        full_cover_candidates=full_cover_candidates,
    )
    temp_gold_parent_ids = full_cover_parent_ids or gold_parent_ids
    what_if = what_if_stage_hits(
        temp_gold_parent_ids=temp_gold_parent_ids,
        p0_row=p0_row,
        result_row=result_row,
    )
    return {
        "sample_id": sample_id,
        "category": str(p0_row.get("category") or dataset_row.get("category") or ""),
        "expected_route": str(
            p0_row.get("expected_route") or dataset_row.get("expected_route") or ""
        ),
        "question": str(p0_row.get("question") or dataset_row.get("question") or ""),
        "expected_doc_ids": expected_docs,
        "gold_chunk_ids": gold_chunk_ids,
        "gold_parent_chunk_ids": gold_parent_ids,
        "gold_child_chunk_ids": gold_child_ids,
        "stable_target_block_ids": stable_blocks,
        "primary_classification": primary_class,
        "p1_secondary_tags": as_str_list(p1_row.get("secondary_tags")),
        "gold_existence": {
            "missing_parent_chunk_ids": [
                parent_id for parent_id in gold_parent_ids if parent_id not in parent_records
            ],
            "missing_child_chunk_ids": [
                child_id for child_id in gold_child_ids if child_id not in child_records
            ],
            "has_child_level_gold": bool(gold_child_ids),
        },
        "gold_stable_block_status": gold_block_status,
        "stable_block_parent_candidates": parent_candidates,
        "raw_child_trace": raw_summary,
        "retrieval_stage": stage_summary,
        "same_doc_context": same_doc_context(
            expected_docs=expected_docs,
            gold_parent_ids=gold_parent_ids,
            selected_parent_ids=as_str_list(p0_row.get("retrieved_parent_chunk_ids_top10")),
        ),
        "stable_block_full_cover_what_if": what_if,
    }


def classify_primary(
    *,
    expected_route: str,
    gold_parent_ids: list[str],
    gold_child_ids: list[str],
    stable_blocks: list[str],
    raw_trace: list[dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    child_records: dict[str, dict[str, Any]],
    gold_block_status: dict[str, Any],
    full_cover_candidates: list[dict[str, Any]],
) -> str:
    if not raw_trace:
        return "raw_child_trace_missing"
    if any(parent_id not in parent_records for parent_id in gold_parent_ids):
        return "gold_parent_missing_from_parent_chunks"
    if any(child_id not in child_records for child_id in gold_child_ids):
        return "gold_child_missing_from_child_chunks"
    if expected_route == "comparison" or len(gold_parent_ids) > 1:
        return "comparison_partial_gold_scope_gap"
    current_gold_covers = bool(gold_block_status.get("gold_parent_covers_all_stable_blocks"))
    same_doc_non_gold_full_cover = [
        item
        for item in full_cover_candidates
        if item["chunk_id"] not in set(gold_parent_ids)
        and item["doc_id"] in {doc_id(parent_id) for parent_id in gold_parent_ids}
    ]
    if stable_blocks and not current_gold_covers and same_doc_non_gold_full_cover:
        return "gold_stable_block_mismatch_same_doc_candidate"
    if gold_child_ids:
        return "current_gold_valid_raw_child_miss"
    if gold_parent_ids:
        return "current_gold_valid_parent_only_raw_miss"
    return "unknown_raw_retrieval_gap"


def build_target_context(
    target_ids: list[str],
    p0_rows: dict[str, dict[str, Any]],
    dataset_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    context = {}
    for sample_id in target_ids:
        p0_row = p0_rows[sample_id]
        dataset_row = dataset_rows[sample_id]
        context[sample_id] = {
            "sample_id": sample_id,
            "expected_doc_ids": as_str_list(
                p0_row.get("expected_doc_ids") or dataset_row.get("expected_doc_ids")
            ),
            "gold_parent_ids": as_str_list(p0_row.get("recomputed_gold_parent_chunk_ids")),
            "gold_child_ids": [
                chunk_id
                for chunk_id in as_str_list(p0_row.get("recomputed_gold_chunk_ids"))
                if "::child" in chunk_id
            ],
            "stable_blocks": as_str_list(dataset_row.get("stable_target_block_ids")),
        }
    return context


def scan_chunk_indexes(
    *,
    parent_chunks_path: Path,
    child_chunks_path: Path,
    targets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_parent_ids = {
        parent_id for target in targets.values() for parent_id in target["gold_parent_ids"]
    }
    target_child_ids = {
        child_id for target in targets.values() for child_id in target["gold_child_ids"]
    }
    samples_by_doc: dict[str, list[str]] = {}
    for sample_id, target in targets.items():
        for expected_doc in target["expected_doc_ids"]:
            samples_by_doc.setdefault(expected_doc, []).append(sample_id)

    parent_records: dict[str, dict[str, Any]] = {}
    stable_candidates: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in targets}
    with parent_chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id") or "")
            doc = str(record.get("doc_id") or "")
            if chunk_id in target_parent_ids:
                parent_records[chunk_id] = compact_parent_record(record)
            if doc not in samples_by_doc:
                continue
            record_blocks = record_block_ids(record)
            for sample_id in samples_by_doc[doc]:
                stable_blocks = set(targets[sample_id]["stable_blocks"])
                if not stable_blocks:
                    continue
                hit_blocks = sorted(record_blocks & stable_blocks)
                if not hit_blocks:
                    continue
                stable_candidates[sample_id].append(
                    {
                        **compact_parent_record(record),
                        "hit_block_ids": hit_blocks,
                        "hit_block_count": len(hit_blocks),
                        "covers_all_stable_blocks": set(hit_blocks) == stable_blocks,
                        "is_current_gold_parent": chunk_id
                        in set(targets[sample_id]["gold_parent_ids"]),
                    }
                )

    child_records = {}
    with child_chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id") or "")
            if chunk_id in target_child_ids:
                child_records[chunk_id] = compact_child_record(record)

    for sample_id, candidates in stable_candidates.items():
        candidates.sort(
            key=lambda item: (
                not bool(item["covers_all_stable_blocks"]),
                not bool(item["is_current_gold_parent"]),
                -int(item["hit_block_count"]),
                str(item["chunk_id"]),
            )
        )
    return {
        "parent_records": parent_records,
        "child_records": child_records,
        "stable_parent_candidates_by_sample": stable_candidates,
    }


def build_gold_block_status(
    *,
    gold_parent_ids: list[str],
    stable_blocks: list[str],
    parent_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stable_set = set(stable_blocks)
    per_parent = []
    covered = set()
    for parent_id in gold_parent_ids:
        record = parent_records.get(parent_id) or {}
        block_ids = set(as_str_list(record.get("block_ids")))
        hit_blocks = sorted(block_ids & stable_set)
        covered.update(hit_blocks)
        per_parent.append(
            {
                "chunk_id": parent_id,
                "exists": bool(record),
                "section": record.get("section"),
                "hit_block_ids": hit_blocks,
                "hit_block_count": len(hit_blocks),
                "covers_all_stable_blocks": bool(stable_set) and set(hit_blocks) == stable_set,
            }
        )
    return {
        "stable_block_count": len(stable_blocks),
        "gold_parent_stable_hit_block_ids": sorted(covered),
        "gold_parent_overlaps_stable_blocks": bool(covered) if stable_blocks else None,
        "gold_parent_covers_all_stable_blocks": bool(stable_set) and covered == stable_set,
        "missing_stable_block_ids": sorted(stable_set - covered),
        "per_gold_parent": per_parent,
    }


def summarize_raw_trace(
    *,
    raw_trace: list[dict[str, Any]],
    expected_docs: list[str],
    gold_parent_ids: list[str],
    gold_child_ids: list[str],
    full_cover_parent_ids: list[str],
) -> dict[str, Any]:
    expected_doc_set = set(expected_docs)
    gold_parent_set = set(gold_parent_ids)
    gold_child_set = set(gold_child_ids)
    full_cover_set = set(full_cover_parent_ids)
    gold_child_ranks = []
    gold_parent_child_ranks = []
    full_cover_parent_ranks = []
    expected_doc_wrong_parent_items = []
    for item in raw_trace:
        rank = safe_int(item.get("rank"))
        child_id = str(item.get("child_chunk_id") or "")
        parent_id = str(item.get("parent_chunk_id") or parent_chunk_id(child_id))
        item_doc = str(item.get("doc_id") or "")
        if child_id in gold_child_set:
            gold_child_ranks.append(rank)
        if parent_id in gold_parent_set:
            gold_parent_child_ranks.append(rank)
        if parent_id in full_cover_set:
            full_cover_parent_ranks.append(rank)
        if item_doc in expected_doc_set and parent_id not in gold_parent_set:
            expected_doc_wrong_parent_items.append(compact_raw_trace_item(item, gold_parent_ids))
    return {
        "count": len(raw_trace),
        "gold_child_exact_ranks": sorted(rank for rank in gold_child_ranks if rank is not None),
        "gold_parent_child_ranks": sorted(
            rank for rank in gold_parent_child_ranks if rank is not None
        ),
        "full_cover_parent_ranks": sorted(
            rank for rank in full_cover_parent_ranks if rank is not None
        ),
        "expected_doc_wrong_parent_count": len(expected_doc_wrong_parent_items),
        "expected_doc_wrong_parent_preview": expected_doc_wrong_parent_items[:10],
        "raw_child_trace_preview": [
            compact_raw_trace_item(item, gold_parent_ids) for item in raw_trace[:10]
        ],
    }


def build_stage_summary(
    *,
    p0_row: dict[str, Any],
    result_row: dict[str, Any],
    retrieval_output: dict[str, Any],
    full_cover_parent_ids: list[str],
) -> dict[str, Any]:
    full_cover_set = set(full_cover_parent_ids)
    retrieval_parent_ids = as_str_list(
        retrieval_output.get("parent_chunk_ids") or result_row.get("raw_retrieved_parent_chunk_ids")
    )
    top10_parent_ids = as_str_list(p0_row.get("retrieved_parent_chunk_ids_top10"))
    support_parent_ids = as_str_list(p0_row.get("support_parent_chunk_ids"))
    citation_parent_ids = as_str_list(p0_row.get("citation_parent_chunk_ids"))
    return {
        "raw_retrieval_full_cover_rank": first_rank(retrieval_parent_ids, full_cover_set),
        "top10_full_cover_rank": first_rank(top10_parent_ids, full_cover_set),
        "support_full_cover_rank": first_rank(support_parent_ids, full_cover_set),
        "citation_full_cover_rank": first_rank(citation_parent_ids, full_cover_set),
        "raw_retrieval_parent_ids_preview": retrieval_parent_ids[:15],
        "top10_parent_ids": top10_parent_ids,
        "support_parent_ids": support_parent_ids,
        "citation_parent_ids": citation_parent_ids,
        "matched_child_map_full_cover_hits": matched_child_hits(
            retrieval_output.get("matched_child_chunk_ids_by_chunk_id") or {},
            full_cover_set,
        ),
    }


def what_if_stage_hits(
    *,
    temp_gold_parent_ids: list[str],
    p0_row: dict[str, Any],
    result_row: dict[str, Any],
) -> dict[str, Any]:
    temp_set = set(temp_gold_parent_ids)
    raw_parent_ids = as_str_list(result_row.get("raw_retrieved_parent_chunk_ids"))
    top10_parent_ids = as_str_list(p0_row.get("retrieved_parent_chunk_ids_top10"))
    support_parent_ids = as_str_list(p0_row.get("support_parent_chunk_ids"))
    citation_parent_ids = as_str_list(p0_row.get("citation_parent_chunk_ids"))
    return {
        "temp_gold_parent_ids": temp_gold_parent_ids,
        "raw_parent_hit": bool(set(raw_parent_ids) & temp_set) if temp_set else None,
        "parent_hit_at10": bool(set(top10_parent_ids) & temp_set) if temp_set else None,
        "support_parent_chunk_hit": bool(set(support_parent_ids) & temp_set) if temp_set else None,
        "citation_parent_chunk_hit": bool(set(citation_parent_ids) & temp_set)
        if temp_set
        else None,
        "raw_parent_rank": first_rank(raw_parent_ids, temp_set),
        "top10_rank": first_rank(top10_parent_ids, temp_set),
        "support_rank": first_rank(support_parent_ids, temp_set),
        "citation_rank": first_rank(citation_parent_ids, temp_set),
    }


def build_summary(
    *,
    run_id: str,
    input_paths: dict[str, str],
    target_ids: list[str],
    samples: list[dict[str, Any]],
    expected_target_count: int,
) -> dict[str, Any]:
    classification_counts = Counter(sample["primary_classification"] for sample in samples)
    what_if = {
        "sample_count": len(samples),
        "raw_parent_hit_count": sum(
            1 for sample in samples if sample["stable_block_full_cover_what_if"]["raw_parent_hit"]
        ),
        "parent_hit_count": sum(
            1 for sample in samples if sample["stable_block_full_cover_what_if"]["parent_hit_at10"]
        ),
        "support_hit_count": sum(
            1
            for sample in samples
            if sample["stable_block_full_cover_what_if"]["support_parent_chunk_hit"]
        ),
        "citation_hit_count": sum(
            1
            for sample in samples
            if sample["stable_block_full_cover_what_if"]["citation_parent_chunk_hit"]
        ),
    }
    validation = build_validation(target_ids, samples, expected_target_count)
    return {
        "run_id": run_id,
        "scope": "P2 raw retrieval / same-doc wrong-parent offline audit",
        "inputs": input_paths,
        "sample_count": len(samples),
        "expected_target_count": expected_target_count,
        "target_sample_ids": target_ids,
        "classification_counts": dict(classification_counts),
        "gold_remap_review_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "gold_stable_block_mismatch_same_doc_candidate"
        ],
        "true_raw_retrieval_miss_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"]
            in {
                "current_gold_valid_raw_child_miss",
                "current_gold_valid_parent_only_raw_miss",
            }
        ],
        "comparison_scope_gap_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "comparison_partial_gold_scope_gap"
        ],
        "stable_block_full_cover_what_if": what_if,
        "validation": validation,
    }


def build_validation(
    target_ids: list[str], samples: list[dict[str, Any]], expected_target_count: int
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
            "name": "target_ids_preserved_from_p1_raw_parent_absent",
            "passed": sample_ids == target_ids,
            "actual": sample_ids,
            "expected": target_ids,
        },
        {
            "name": "one_primary_classification_per_sample",
            "passed": all(
                sample["primary_classification"] in PRIMARY_CLASSES for sample in samples
            ),
            "invalid_sample_ids": [
                sample["sample_id"]
                for sample in samples
                if sample["primary_classification"] not in PRIMARY_CLASSES
            ],
        },
        {
            "name": "all_samples_have_gold_block_audit",
            "passed": all(bool(sample.get("gold_stable_block_status")) for sample in samples),
            "missing_sample_ids": [
                sample["sample_id"]
                for sample in samples
                if not sample.get("gold_stable_block_status")
            ],
        },
    ]
    return {
        "passed": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "failed_criteria": [item for item in criteria if not item["passed"]],
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 P2 raw retrieval / same-doc wrong-parent 离线审计报告",
        "",
        "## 范围",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- sample_count: {summary['sample_count']}",
        (
            "- 口径：只读 P1 raw_parent_absent、P0 remap gold、"
            "现有 no-judge raw trace 和 chunk JSONL。"
        ),
        f"- validation_passed: `{summary['validation']['passed']}`",
        "",
        "## 分类汇总",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["classification_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    what_if = summary["stable_block_full_cover_what_if"]
    lines.extend(
        [
            "",
            "## Stable-block full-cover what-if",
            "",
            f"- raw_parent_hit_count: {what_if['raw_parent_hit_count']}/{what_if['sample_count']}",
            f"- parent_hit_count: {what_if['parent_hit_count']}/{what_if['sample_count']}",
            f"- support_hit_count: {what_if['support_hit_count']}/{what_if['sample_count']}",
            f"- citation_hit_count: {what_if['citation_hit_count']}/{what_if['sample_count']}",
            "",
            "## 样本明细",
            "",
            (
                "| sample_id | class | gold parent | full-cover candidates | "
                "top10 hit what-if | raw child signal |"
            ),
            "|---|---|---|---|---:|---|",
        ]
    )
    for sample in samples:
        raw = sample["raw_child_trace"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{sample['sample_id']}`",
                    f"`{sample['primary_classification']}`",
                    format_code_list(sample["gold_parent_chunk_ids"]),
                    format_code_list(
                        [
                            item["chunk_id"]
                            for item in sample["stable_block_parent_candidates"]
                            if item["covers_all_stable_blocks"]
                        ]
                    ),
                    fmt_bool(sample["stable_block_full_cover_what_if"]["parent_hit_at10"]),
                    (
                        "gold_child="
                        f"{format_rank_list(raw['gold_child_exact_ranks'])}; "
                        "gold_parent_child="
                        f"{format_rank_list(raw['gold_parent_child_ranks'])}; "
                        "full_cover="
                        f"{format_rank_list(raw['full_cover_parent_ranks'])}"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 建议分流",
            "",
            (
                "- 先做 gold/stable-block remap 复核："
                f"{format_code_list(summary['gold_remap_review_sample_ids'])}"
            ),
            (
                "- 当前 gold 合法但 raw retrieval 真漏召回："
                f"{format_code_list(summary['true_raw_retrieval_miss_sample_ids'])}"
            ),
            (
                "- comparison scope 单独处理："
                f"{format_code_list(summary['comparison_scope_gap_sample_ids'])}"
            ),
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


def compact_parent_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(record.get("chunk_id") or ""),
        "doc_id": str(record.get("doc_id") or ""),
        "section": str(record.get("section") or ""),
        "source_file": str(record.get("source_file") or ""),
        "block_ids": as_str_list(record.get("source_block_ids") or record.get("block_ids")),
        "contains_table_caption": bool(record.get("contains_table_caption")),
        "contains_table_text": bool(record.get("contains_table_text")),
        "contains_figure_caption": bool(record.get("contains_figure_caption")),
        "text_length": len(str(record.get("retrieval_text") or record.get("text") or "")),
    }


def compact_child_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(record.get("chunk_id") or ""),
        "parent_chunk_id": str(
            record.get("parent_chunk_id") or parent_chunk_id(record.get("chunk_id"))
        ),
        "doc_id": str(record.get("doc_id") or ""),
        "section": str(record.get("section") or ""),
        "block_ids": as_str_list(record.get("source_block_ids") or record.get("block_ids")),
        "contains_table_caption": bool(record.get("contains_table_caption")),
        "contains_table_text": bool(record.get("contains_table_text")),
        "contains_figure_caption": bool(record.get("contains_figure_caption")),
        "text_length": len(str(record.get("retrieval_text") or record.get("text") or "")),
    }


def compact_raw_trace_item(item: dict[str, Any], gold_parent_ids: list[str]) -> dict[str, Any]:
    parent_id = str(item.get("parent_chunk_id") or parent_chunk_id(item.get("child_chunk_id")))
    return {
        "rank": item.get("rank"),
        "child_chunk_id": item.get("child_chunk_id"),
        "parent_chunk_id": parent_id,
        "doc_id": item.get("doc_id"),
        "section": item.get("section"),
        "source": item.get("source"),
        "fusion_score": round_number(item.get("fusion_score")),
        "vector_score": round_number(item.get("vector_score")),
        "bm25_score": round_number(item.get("bm25_score")),
        "nearest_gold_parent_distance": nearest_parent_distance([parent_id], gold_parent_ids),
    }


def matched_child_hits(
    matched_by_chunk: dict[str, Any],
    target_parent_ids: set[str],
) -> dict[str, list[str]]:
    return {
        str(parent_id): as_str_list(matched_by_chunk.get(parent_id))
        for parent_id in target_parent_ids
        if matched_by_chunk.get(parent_id)
    }


def same_doc_context(
    *,
    expected_docs: list[str],
    gold_parent_ids: list[str],
    selected_parent_ids: list[str],
) -> dict[str, Any]:
    same_doc = [
        parent_id
        for parent_id in selected_parent_ids
        if doc_id(parent_id) in set(expected_docs) and parent_id not in set(gold_parent_ids)
    ]
    return {
        "same_doc_selected_parent_ids": same_doc,
        "nearest_distance": nearest_parent_distance(same_doc, gold_parent_ids),
    }


def record_block_ids(record: dict[str, Any]) -> set[str]:
    return set(as_str_list(record.get("source_block_ids") or record.get("block_ids")))


def nearest_parent_distance(parent_ids: list[str], gold_parent_ids: list[str]) -> int | None:
    distances = []
    for parent_id_value in parent_ids:
        parent_pos = parent_position(parent_id_value)
        if parent_pos is None:
            continue
        for gold_parent_id in gold_parent_ids:
            if doc_id(parent_id_value) != doc_id(gold_parent_id):
                continue
            gold_pos = parent_position(gold_parent_id)
            if gold_pos is None:
                continue
            distances.append(
                max(abs(parent_pos[0] - gold_pos[0]), abs(parent_pos[1] - gold_pos[1]))
            )
    return min(distances) if distances else None


def parent_position(parent_id_value: str) -> tuple[int, int] | None:
    match = PARENT_ID_RE.match(str(parent_id_value or ""))
    if not match:
        return None
    return int(match.group("section")), int(match.group("chunk"))


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def doc_id(parent_id_value: Any) -> str:
    value = str(parent_id_value or "")
    if "_sec" in value:
        return value.split("_sec", 1)[0]
    return ""


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


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if str(value):
        return [str(value)]
    return []


def format_code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def format_rank_list(values: list[int]) -> str:
    if not values:
        return "-"
    return ",".join(str(value) for value in values)


def fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "N/A"


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
    parent_records = {
        "doc_a_sec01_chunk01": {
            "chunk_id": "doc_a_sec01_chunk01",
            "doc_id": "doc_a",
            "block_ids": ["b1", "b2"],
        },
        "doc_a_sec02_chunk02": {
            "chunk_id": "doc_a_sec02_chunk02",
            "doc_id": "doc_a",
            "block_ids": ["b3"],
        },
    }
    status = build_gold_block_status(
        gold_parent_ids=["doc_a_sec01_chunk01"],
        stable_blocks=["b1", "b2"],
        parent_records=parent_records,
    )
    assert status["gold_parent_covers_all_stable_blocks"] is True
    candidates = [
        {
            "chunk_id": "doc_a_sec02_chunk02",
            "doc_id": "doc_a",
            "covers_all_stable_blocks": True,
        }
    ]
    mismatch = classify_primary(
        expected_route="factoid",
        gold_parent_ids=["doc_a_sec01_chunk01"],
        gold_child_ids=[],
        stable_blocks=["b3"],
        raw_trace=[{"rank": 1}],
        parent_records=parent_records,
        child_records={},
        gold_block_status={
            "gold_parent_covers_all_stable_blocks": False,
        },
        full_cover_candidates=candidates,
    )
    assert mismatch == "gold_stable_block_mismatch_same_doc_candidate"
    raw_child_miss = classify_primary(
        expected_route="factoid",
        gold_parent_ids=["doc_a_sec01_chunk01"],
        gold_child_ids=["doc_a_sec01_chunk01::child001"],
        stable_blocks=["b1"],
        raw_trace=[{"rank": 1}],
        parent_records=parent_records,
        child_records={
            "doc_a_sec01_chunk01::child001": {"chunk_id": "doc_a_sec01_chunk01::child001"}
        },
        gold_block_status={"gold_parent_covers_all_stable_blocks": True},
        full_cover_candidates=[],
    )
    assert raw_child_miss == "current_gold_valid_raw_child_miss"
    parent_only = classify_primary(
        expected_route="factoid",
        gold_parent_ids=["doc_a_sec01_chunk01"],
        gold_child_ids=[],
        stable_blocks=["b1"],
        raw_trace=[{"rank": 1}],
        parent_records=parent_records,
        child_records={},
        gold_block_status={"gold_parent_covers_all_stable_blocks": True},
        full_cover_candidates=[],
    )
    assert parent_only == "current_gold_valid_parent_only_raw_miss"
    comparison = classify_primary(
        expected_route="comparison",
        gold_parent_ids=["doc_a_sec01_chunk01", "doc_b_sec01_chunk01"],
        gold_child_ids=[],
        stable_blocks=["b1"],
        raw_trace=[{"rank": 1}],
        parent_records=parent_records | {"doc_b_sec01_chunk01": {"block_ids": ["b4"]}},
        child_records={},
        gold_block_status={"gold_parent_covers_all_stable_blocks": True},
        full_cover_candidates=[],
    )
    assert comparison == "comparison_partial_gold_scope_gap"
    print("self-test passed")


if __name__ == "__main__":
    main()
