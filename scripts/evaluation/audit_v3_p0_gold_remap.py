from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_CHILD_CHUNKS = Path("data/paper_round1/chunks/child_chunks.jsonl")
DEFAULT_REWRITE_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)
DEFAULT_RUN_ID = "20260524"
TARGET_SAMPLE_IDS = (
    "v3_ra_009",
    "v3_ra_014",
    "v3_ra_018",
    "v3_ra_019",
    "v3_ra_021",
    "v3_ra_026",
    "v3_ra_027",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline P0 audit for v3 gold ids that drifted from current chunk indexes."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--rewrite-results", default=str(DEFAULT_REWRITE_RESULTS))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--sample-ids", default=",".join(TARGET_SAMPLE_IDS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    sample_ids = [item.strip() for item in str(args.sample_ids).split(",") if item.strip()]
    dataset_rows = load_jsonl_by_id(Path(args.dataset), "sample_id")
    rewrite_rows = load_jsonl_by_id(Path(args.rewrite_results), "sample_id")
    validate_required_rows(sample_ids, dataset_rows, "dataset")
    validate_required_rows(sample_ids, rewrite_rows, "rewrite results")

    targets = [build_target(dataset_rows[sample_id]) for sample_id in sample_ids]
    parent_scan = scan_chunk_index(Path(args.parent_chunks), targets, index_kind="parent")
    child_scan = scan_chunk_index(Path(args.child_chunks), targets, index_kind="child")
    samples = [
        audit_sample(
            target=target,
            rewrite_row=rewrite_rows[target["sample_id"]],
            parent_scan=parent_scan,
            child_scan=child_scan,
        )
        for target in targets
    ]
    summary = build_summary(
        run_id=str(args.run_id),
        dataset_path=Path(args.dataset),
        parent_chunks_path=Path(args.parent_chunks),
        child_chunks_path=Path(args.child_chunks),
        rewrite_results_path=Path(args.rewrite_results),
        sample_ids=sample_ids,
        samples=samples,
    )

    result_dir = RESULTS_ROOT / f"v3_p0_gold_remap_audit_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_p0_gold_remap_audit_{args.run_id}"
    write_json(result_dir / "p0_gold_remap_summary.json", summary)
    write_jsonl(result_dir / "p0_gold_remap_samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "target_sample_count": len(samples),
                "classification_counts": summary["classification_counts"],
            },
            ensure_ascii=False,
        )
    )


def build_target(dataset_row: dict[str, Any]) -> dict[str, Any]:
    source_trace = (dataset_row.get("answer_rubric") or {}).get("source_trace") or {}
    old_source_trace_chunk_ids = as_str_list(source_trace.get("chunk_ids"))
    old_target = str(dataset_row.get("target_chunk_id_candidate") or "")
    old_ids = dedupe([*old_source_trace_chunk_ids, old_target])
    return {
        "sample_id": str(dataset_row.get("sample_id") or ""),
        "question": str(dataset_row.get("question") or ""),
        "category": str(dataset_row.get("category") or ""),
        "expected_route": str(dataset_row.get("expected_route") or ""),
        "expected_doc_ids": as_str_list(dataset_row.get("expected_doc_ids")),
        "old_target_chunk_id_candidate": old_target,
        "old_source_trace_chunk_ids": old_source_trace_chunk_ids,
        "stable_target_block_ids": as_str_list(dataset_row.get("stable_target_block_ids")),
        "old_gold_ids": old_ids,
        "source_dataset": str(source_trace.get("source_dataset") or ""),
        "source_sample_id": str(source_trace.get("source_sample_id") or ""),
    }


def scan_chunk_index(
    path: Path,
    targets: list[dict[str, Any]],
    *,
    index_kind: str,
) -> dict[str, Any]:
    samples_by_doc: dict[str, list[dict[str, Any]]] = {}
    old_ids = set()
    for target in targets:
        old_ids.update(target["old_gold_ids"])
        for doc_id in target["expected_doc_ids"]:
            samples_by_doc.setdefault(doc_id, []).append(target)

    old_id_hits: dict[str, dict[str, Any]] = {}
    candidates_by_sample: dict[str, dict[str, dict[str, Any]]] = {
        target["sample_id"]: {} for target in targets
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id") or "")
            if chunk_id in old_ids:
                old_id_hits[chunk_id] = compact_chunk_record(record, hit_blocks=[])

            doc_id = str(record.get("doc_id") or "")
            possible_targets = samples_by_doc.get(doc_id) or []
            if not possible_targets:
                continue
            record_blocks = record_block_ids(record)
            if not record_blocks:
                continue
            for target in possible_targets:
                stable_blocks = set(target["stable_target_block_ids"])
                hit_blocks = sorted(record_blocks & stable_blocks)
                if not hit_blocks:
                    continue
                sample_id = target["sample_id"]
                candidates_by_sample[sample_id][chunk_id] = compact_chunk_record(
                    record,
                    hit_blocks=hit_blocks,
                    stable_blocks=target["stable_target_block_ids"],
                    index_kind=index_kind,
                )
    return {
        "old_id_hits": old_id_hits,
        "candidates_by_sample": {
            sample_id: list(candidates.values())
            for sample_id, candidates in candidates_by_sample.items()
        },
    }


def compact_chunk_record(
    record: dict[str, Any],
    *,
    hit_blocks: list[str],
    stable_blocks: list[str] | None = None,
    index_kind: str = "",
) -> dict[str, Any]:
    stable_block_set = set(stable_blocks or [])
    hit_block_set = set(hit_blocks)
    return {
        "index_kind": index_kind,
        "chunk_id": str(record.get("chunk_id") or ""),
        "parent_chunk_id": str(record.get("parent_chunk_id") or record.get("chunk_id") or ""),
        "doc_id": str(record.get("doc_id") or ""),
        "source_file": str(record.get("source_file") or ""),
        "section": str(record.get("section") or ""),
        "page_numbers": record.get("page_numbers") or [],
        "hit_block_ids": hit_blocks,
        "hit_block_count": len(hit_blocks),
        "covers_all_stable_blocks": bool(stable_block_set) and hit_block_set == stable_block_set,
        "stable_block_coverage_rate": round(len(hit_block_set) / len(stable_block_set), 6)
        if stable_block_set
        else None,
        "contains_table_caption": bool(record.get("contains_table_caption")),
        "contains_table_text": bool(record.get("contains_table_text")),
        "contains_figure_caption": bool(record.get("contains_figure_caption")),
        "block_types": record.get("block_types") or [],
        "evidence_types": record.get("evidence_types") or [],
        "text_preview": compact_text(record.get("retrieval_text") or record.get("text")),
    }


def audit_sample(
    *,
    target: dict[str, Any],
    rewrite_row: dict[str, Any],
    parent_scan: dict[str, Any],
    child_scan: dict[str, Any],
) -> dict[str, Any]:
    sample_id = target["sample_id"]
    parent_candidates = sorted_candidates(parent_scan["candidates_by_sample"].get(sample_id) or [])
    child_candidates = sorted_candidates(child_scan["candidates_by_sample"].get(sample_id) or [])
    old_parent_hits = [
        parent_scan["old_id_hits"][old_id]
        for old_id in target["old_gold_ids"]
        if old_id in parent_scan["old_id_hits"]
    ]
    old_child_hits = [
        child_scan["old_id_hits"][old_id]
        for old_id in target["old_gold_ids"]
        if old_id in child_scan["old_id_hits"]
    ]

    recommended_parent_ids = [
        item["chunk_id"] for item in parent_candidates if item["covers_all_stable_blocks"]
    ]
    recommended_child_ids = [
        item["chunk_id"]
        for item in child_candidates
        if item["parent_chunk_id"] in set(recommended_parent_ids)
    ]
    classification, confidence, reason = classify_remap(
        old_parent_hits=old_parent_hits,
        old_child_hits=old_child_hits,
        parent_candidates=parent_candidates,
        child_candidates=child_candidates,
        recommended_parent_ids=recommended_parent_ids,
        recommended_child_ids=recommended_child_ids,
    )
    hit_status = current_hit_status(rewrite_row, recommended_parent_ids, recommended_child_ids)
    return {
        "sample_id": sample_id,
        "category": target["category"],
        "expected_route": target["expected_route"],
        "question": target["question"],
        "source_dataset": target["source_dataset"],
        "source_sample_id": target["source_sample_id"],
        "expected_doc_ids": target["expected_doc_ids"],
        "old_target_chunk_id_candidate": target["old_target_chunk_id_candidate"],
        "old_source_trace_chunk_ids": target["old_source_trace_chunk_ids"],
        "stable_target_block_ids": target["stable_target_block_ids"],
        "old_ids_exist_in_parent_chunks": bool(old_parent_hits),
        "old_ids_exist_in_child_chunks": bool(old_child_hits),
        "old_parent_index_hits": old_parent_hits,
        "old_child_index_hits": old_child_hits,
        "current_parent_candidates": parent_candidates,
        "current_child_candidates": child_candidates,
        "recommended_parent_chunk_ids": recommended_parent_ids,
        "recommended_child_chunk_ids": recommended_child_ids,
        "remap_classification": classification,
        "confidence": confidence,
        "reason": reason,
        **hit_status,
    }


def classify_remap(
    *,
    old_parent_hits: list[dict[str, Any]],
    old_child_hits: list[dict[str, Any]],
    parent_candidates: list[dict[str, Any]],
    child_candidates: list[dict[str, Any]],
    recommended_parent_ids: list[str],
    recommended_child_ids: list[str],
) -> tuple[str, str, str]:
    if old_parent_hits or old_child_hits:
        return (
            "old_id_still_indexed",
            "low",
            "旧 gold id 仍能在当前 index 中找到，需要人工判断是否真的漂移。",
        )
    if not parent_candidates:
        return (
            "unresolved_no_current_parent",
            "low",
            "expected_doc_ids 内找不到覆盖 stable blocks 的当前 parent。",
        )
    if len(recommended_parent_ids) > 1:
        return (
            "ambiguous_multiple_parent_candidates",
            "low",
            "多个当前 parent 覆盖全部 stable blocks，需要人工复核。",
        )
    if not recommended_parent_ids:
        return (
            "unresolved_no_full_block_parent",
            "low",
            "有同文档 block overlap，但没有 parent 覆盖全部 stable blocks。",
        )
    if not recommended_child_ids:
        return (
            "parent_only_current_remap",
            "medium",
            "唯一当前 parent 覆盖全部 stable blocks，但没有 child 候选直接覆盖。",
        )
    if len(recommended_child_ids) == 1:
        child = next(
            item
            for item in child_candidates
            if item["chunk_id"] == recommended_child_ids[0]
        )
        if child["covers_all_stable_blocks"]:
            return (
                "safe_parent_remap",
                "high",
                "旧 gold id 缺失；同一 expected doc 下唯一 parent 和唯一 child "
                "覆盖全部 stable blocks。",
            )
    return (
        "safe_parent_remap_child_split",
        "high",
        "旧 gold id 缺失；同一 expected doc 下唯一 parent 覆盖全部 stable blocks，"
        "child 证据分散或仅部分覆盖。",
    )


def current_hit_status(
    rewrite_row: dict[str, Any],
    recommended_parent_ids: list[str],
    recommended_child_ids: list[str],
) -> dict[str, Any]:
    parent_set = set(recommended_parent_ids)
    child_set = set(recommended_child_ids)
    raw_parent_ids = as_str_list(rewrite_row.get("raw_retrieved_parent_chunk_ids"))
    top10_parent_ids = as_str_list(rewrite_row.get("retrieved_parent_chunk_ids_top10"))
    support_parent_ids = as_str_list(rewrite_row.get("support_chunk_ids"))
    support_parent_ids = dedupe(parent_chunk_id(item) for item in support_parent_ids)
    citation_parent_ids = as_str_list(rewrite_row.get("citation_parent_chunk_ids"))
    support_child_ids = as_str_list(rewrite_row.get("support_matched_child_chunk_ids"))
    citation_child_ids = as_str_list(rewrite_row.get("citation_matched_child_chunk_ids"))
    return {
        "current_parent_hit_in_raw": any_in(parent_set, raw_parent_ids),
        "current_parent_hit_in_top10": any_in(parent_set, top10_parent_ids),
        "current_parent_hit_in_support": any_in(parent_set, support_parent_ids),
        "current_parent_hit_in_citation": any_in(parent_set, citation_parent_ids),
        "current_child_hit_in_support": any_in(child_set, support_child_ids) if child_set else None,
        "current_child_hit_in_citation": (
            any_in(child_set, citation_child_ids) if child_set else None
        ),
        "current_parent_raw_rank": first_rank(raw_parent_ids, parent_set),
        "current_parent_top10_rank": first_rank(top10_parent_ids, parent_set),
    }


def build_summary(
    *,
    run_id: str,
    dataset_path: Path,
    parent_chunks_path: Path,
    child_chunks_path: Path,
    rewrite_results_path: Path,
    sample_ids: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    classification_counts = Counter(sample["remap_classification"] for sample in samples)
    confidence_counts = Counter(sample["confidence"] for sample in samples)
    return {
        "run_id": run_id,
        "scope": "P0 gold remap audit for doc_hit_parent_chunk_miss drift samples",
        "inputs": {
            "dataset": str(dataset_path),
            "parent_chunks": str(parent_chunks_path),
            "child_chunks": str(child_chunks_path),
            "rewrite_results": str(rewrite_results_path),
        },
        "sample_ids": sample_ids,
        "target_sample_count": len(samples),
        "classification_counts": dict(classification_counts),
        "confidence_counts": dict(confidence_counts),
        "old_ids_missing_from_both_indexes_count": sum(
            1
            for sample in samples
            if not sample["old_ids_exist_in_parent_chunks"]
            and not sample["old_ids_exist_in_child_chunks"]
        ),
        "unique_recommended_parent_count": sum(
            1 for sample in samples if len(sample["recommended_parent_chunk_ids"]) == 1
        ),
        "blocking_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["remap_classification"]
            in {
                "ambiguous_multiple_parent_candidates",
                "unresolved_no_current_parent",
                "unresolved_no_full_block_parent",
            }
        ],
        "remap_parent_hit_recomputed": {
            "raw_count": sum(1 for sample in samples if sample["current_parent_hit_in_raw"]),
            "top10_count": sum(1 for sample in samples if sample["current_parent_hit_in_top10"]),
            "support_count": sum(
                1 for sample in samples if sample["current_parent_hit_in_support"]
            ),
            "citation_count": sum(
                1 for sample in samples if sample["current_parent_hit_in_citation"]
            ),
        },
        "samples_preview": [
            {
                "sample_id": sample["sample_id"],
                "old_target_chunk_id_candidate": sample["old_target_chunk_id_candidate"],
                "recommended_parent_chunk_ids": sample["recommended_parent_chunk_ids"],
                "recommended_child_chunk_ids": sample["recommended_child_chunk_ids"],
                "remap_classification": sample["remap_classification"],
                "current_parent_hit_in_top10": sample["current_parent_hit_in_top10"],
            }
            for sample in samples
        ],
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 P0 gold remap 审计报告",
        "",
        "## 范围",
        "",
        "- 目标：确认 7 个 P0 gold id 漂移样本在当前 chunk index 中的 remap 候选。",
        "- 本报告只做离线审计，不修改 dataset，不运行 eval，不调整 retrieval/rerank/score floor。",
        "",
        "## 汇总",
        "",
        f"- target_sample_count: {summary['target_sample_count']}",
        (
            "- classification_counts: `"
            f"{json.dumps(summary['classification_counts'], ensure_ascii=False)}`"
        ),
        f"- confidence_counts: `{json.dumps(summary['confidence_counts'], ensure_ascii=False)}`",
        (
            "- old ids missing from both indexes: "
            f"{summary['old_ids_missing_from_both_indexes_count']}"
        ),
        f"- unique recommended parent count: {summary['unique_recommended_parent_count']}",
        f"- blocking_sample_ids: {format_code_list(summary['blocking_sample_ids'])}",
        "",
        "按 remap 后 parent hit 离线重算：",
        "",
        "| stage | count |",
        "|---|---:|",
        f"| raw | {summary['remap_parent_hit_recomputed']['raw_count']} |",
        f"| top10 | {summary['remap_parent_hit_recomputed']['top10_count']} |",
        f"| support | {summary['remap_parent_hit_recomputed']['support_count']} |",
        f"| citation | {summary['remap_parent_hit_recomputed']['citation_count']} |",
        "",
        "## 样本明细",
        "",
        (
            "| sample_id | old target | recommended parent | recommended child | "
            "classification | top10 hit | reason |"
        ),
        "|---|---|---|---|---|---:|---|",
    ]
    for sample in samples:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{sample['sample_id']}`",
                    f"`{sample['old_target_chunk_id_candidate']}`",
                    format_code_list(sample["recommended_parent_chunk_ids"]),
                    format_code_list(sample["recommended_child_chunk_ids"]),
                    f"`{sample['remap_classification']}`",
                    fmt_bool(sample["current_parent_hit_in_top10"]),
                    sample["reason"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 后续建议",
            "",
            "1. 若本报告无 blocking 样本，再制定 dataset remap patch 计划。",
            "2. dataset patch 应只更新 gold/source trace，不混入 retrieval/rerank 策略改动。",
            "3. patch 后用离线 rule metric 重算先验证 parent hit delta，再决定是否跑正式 eval。",
        ]
    )
    return "\n".join(lines) + "\n"


def sorted_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("covers_all_stable_blocks")),
            -int(item.get("hit_block_count") or 0),
            str(item.get("chunk_id") or ""),
        ),
    )


def record_block_ids(record: dict[str, Any]) -> set[str]:
    return {str(item) for item in (record.get("source_block_ids") or record.get("block_ids") or [])}


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


def any_in(targets: set[str], values: list[str]) -> bool:
    return any(target in set(values) for target in targets)


def first_rank(values: list[str], targets: set[str]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in targets:
            return index
    return None


def compact_text(value: Any, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def format_code_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def fmt_bool(value: Any) -> str:
    if value is None:
        return "N/A"
    return "true" if bool(value) else "false"


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get(key) or "")
            if not row_id:
                raise ValueError(f"{path}:{line_number} missing key {key}")
            rows[row_id] = row
    return rows


def validate_required_rows(
    sample_ids: list[str],
    rows: dict[str, dict[str, Any]],
    label: str,
) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"Missing {label} rows: {', '.join(missing)}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_self_test() -> None:
    assert parent_chunk_id("doc_a_sec01_chunk02::child003") == "doc_a_sec01_chunk02"
    assert parent_chunk_id("doc_a_sec01_chunk02") == "doc_a_sec01_chunk02"
    target = {
        "sample_id": "s1",
        "expected_doc_ids": ["doc_a"],
        "old_gold_ids": ["doc_a_sec99_chunk99"],
        "stable_target_block_ids": ["b1", "b2"],
    }
    same_doc = {
        "chunk_id": "doc_a_sec01_chunk02",
        "parent_chunk_id": "doc_a_sec01_chunk02",
        "doc_id": "doc_a",
        "source_block_ids": ["b1", "b2"],
    }
    other_doc = {
        "chunk_id": "doc_b_sec01_chunk02",
        "parent_chunk_id": "doc_b_sec01_chunk02",
        "doc_id": "doc_b",
        "source_block_ids": ["b1", "b2"],
    }
    assert record_matches_target(same_doc, target) == ["b1", "b2"]
    assert record_matches_target(other_doc, target) == []
    classification = classify_remap(
        old_parent_hits=[],
        old_child_hits=[],
        parent_candidates=[
            {
                "chunk_id": "doc_a_sec01_chunk02",
                "covers_all_stable_blocks": True,
            }
        ],
        child_candidates=[],
        recommended_parent_ids=["doc_a_sec01_chunk02"],
        recommended_child_ids=[],
    )
    assert classification[0] == "parent_only_current_remap"
    print("self-test passed")


def record_matches_target(record: dict[str, Any], target: dict[str, Any]) -> list[str]:
    if str(record.get("doc_id") or "") not in set(target["expected_doc_ids"]):
        return []
    return sorted(record_block_ids(record) & set(target["stable_target_block_ids"]))


if __name__ == "__main__":
    main()
