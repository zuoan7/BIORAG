from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<sec>\d+)_chunk(?P<chunk>\d+)$")

CLASSIFICATION_LABELS = {
    "gold_child_recalled_parent_lift_lost": "gold child 已召回，但未 lift 到 gold parent",
    "gold_parent_child_recalled_parent_lift_lost": "gold parent 下其它 child 已召回，但 parent 聚合丢失",
    "same_doc_child_wrong_parent": "expected doc 下召回了 child，但都属于错误 parent",
    "expected_doc_child_missing": "expected doc 下没有 raw child 命中",
    "no_gold_child_id": "样本没有 gold child id，无法判断 child recall",
    "raw_child_trace_missing": "debug 缺少 raw child trace",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw child retrieval trace for v3 parent raw misses.")
    parser.add_argument("--debug-result-dir", default="")
    parser.add_argument("--support-audit-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.debug_result_dir:
        raise SystemExit("--debug-result-dir is required unless --self-test is used")
    if not args.support_audit_dir:
        raise SystemExit("--support-audit-dir is required unless --self-test is used")

    debug_dir = Path(args.debug_result_dir)
    support_audit_dir = Path(args.support_audit_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"v3_raw_child_trace_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_raw_child_trace_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples_by_variant: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        support_rows = load_jsonl(support_audit_dir / f"{variant}_samples.jsonl")
        debug_rows = {
            str(row.get("sample_id") or ""): row
            for row in load_jsonl(debug_dir / variant / "results.jsonl")
        }
        samples = audit_variant(variant=variant, support_rows=support_rows, debug_rows=debug_rows)
        samples_by_variant[variant] = samples
        summaries[variant] = summarize_samples(samples)
        write_jsonl(output_dir / f"{variant}_samples.jsonl", samples)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "debug_result_dir": str(debug_dir),
        "support_audit_dir": str(support_audit_dir),
        "scope": "rerank_doc_hit_parent_miss/gold_parent_not_in_raw_retrieval",
        "variants": summaries,
        "comparison": compare_variants(samples_by_variant),
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary, samples_by_variant))
    print(json.dumps({"output_dir": str(output_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def audit_variant(
    *,
    variant: str,
    support_rows: list[dict[str, Any]],
    debug_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    target_rows = [
        row
        for row in support_rows
        if "rerank_doc_hit_parent_miss" in (row.get("audit_buckets") or [])
        and (row.get("rerank_audit") or {}).get("reason") == "gold_parent_not_in_raw_retrieval"
    ]
    samples = []
    for support_row in target_rows:
        sample_id = str(support_row.get("sample_id") or "")
        samples.append(
            audit_sample(
                variant=variant,
                support_row=support_row,
                debug_row=debug_rows.get(sample_id) or {},
            )
        )
    return samples


def audit_sample(
    *,
    variant: str,
    support_row: dict[str, Any],
    debug_row: dict[str, Any],
) -> dict[str, Any]:
    debug_digest = debug_row.get("debug_digest") or {}
    raw_child_debug = debug_digest.get("raw_child_trace") or {}
    trace = [item for item in raw_child_debug.get("raw_child_trace") or [] if isinstance(item, dict)]
    aggregation = raw_child_debug.get("child_to_parent_aggregation") or {}
    expected_docs = [str(item) for item in support_row.get("expected_doc_ids") or []]
    gold_children = [str(item) for item in support_row.get("gold_child_chunk_ids") or []]
    gold_parents = [str(item) for item in support_row.get("gold_parent_chunk_ids") or []]
    raw_parent_ids = [str(item) for item in debug_row.get("raw_retrieved_parent_chunk_ids") or []]

    gold_child_ranks = [
        int(item.get("rank") or 0)
        for item in trace
        if str(item.get("child_chunk_id") or "") in set(gold_children)
    ]
    gold_parent_child_ranks = [
        int(item.get("rank") or 0)
        for item in trace
        if str(item.get("parent_chunk_id") or "") in set(gold_parents)
    ]
    expected_doc_items = [
        item for item in trace if str(item.get("doc_id") or "") in set(expected_docs)
    ]
    expected_doc_parent_ids = dedupe(
        str(item.get("parent_chunk_id") or "") for item in expected_doc_items
    )
    classification = classify_sample(
        trace=trace,
        gold_children=gold_children,
        gold_child_ranks=gold_child_ranks,
        gold_parent_child_ranks=gold_parent_child_ranks,
        expected_doc_items=expected_doc_items,
        gold_parents=gold_parents,
        raw_parent_ids=raw_parent_ids,
    )
    return {
        "variant_key": variant,
        "sample_id": support_row.get("sample_id"),
        "question": support_row.get("question"),
        "category": support_row.get("category"),
        "expected_doc_ids": expected_docs,
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "raw_child_trace_count": len(trace),
        "raw_child_gold_hit": bool(gold_child_ranks) if gold_children else None,
        "raw_gold_parent_any_child_hit": bool(gold_parent_child_ranks),
        "raw_expected_doc_child_hit": bool(expected_doc_items),
        "raw_expected_doc_first_child_rank": min(
            [int(item.get("rank") or 999999) for item in expected_doc_items],
            default=None,
        ),
        "raw_expected_doc_parent_ids": expected_doc_parent_ids,
        "nearest_same_doc_parent_distance": nearest_parent_distance(expected_doc_parent_ids, gold_parents),
        "gold_child_raw_ranks": sorted(rank for rank in gold_child_ranks if rank),
        "gold_parent_child_raw_ranks": sorted(rank for rank in gold_parent_child_ranks if rank),
        "raw_parent_ids_contains_gold_parent": any_in(gold_parents, raw_parent_ids),
        "child_to_parent_aggregation": compact_aggregation(aggregation),
        "classification": classification,
        "raw_child_trace_preview": compact_trace_preview(trace, expected_docs, gold_children, gold_parents),
    }


def classify_sample(
    *,
    trace: list[dict[str, Any]],
    gold_children: list[str],
    gold_child_ranks: list[int],
    gold_parent_child_ranks: list[int],
    expected_doc_items: list[dict[str, Any]],
    gold_parents: list[str],
    raw_parent_ids: list[str],
) -> str:
    if not trace:
        return "raw_child_trace_missing"
    if not gold_children:
        return "no_gold_child_id"
    raw_parent_hit = any_in(gold_parents, raw_parent_ids)
    if gold_child_ranks and not raw_parent_hit:
        return "gold_child_recalled_parent_lift_lost"
    if gold_parent_child_ranks and not raw_parent_hit:
        return "gold_parent_child_recalled_parent_lift_lost"
    if expected_doc_items:
        return "same_doc_child_wrong_parent"
    return "expected_doc_child_missing"


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(sample.get("classification") or "unknown") for sample in samples)
    gold_child_scored = [sample for sample in samples if sample.get("raw_child_gold_hit") is not None]
    gold_child_hits = [sample for sample in gold_child_scored if sample.get("raw_child_gold_hit") is True]
    expected_doc_hits = [sample for sample in samples if sample.get("raw_expected_doc_child_hit") is True]
    return {
        "target_case_count": len(samples),
        "classification_counts": dict(counts),
        "gold_child_judgable_count": len(gold_child_scored),
        "gold_child_raw_hit_count": len(gold_child_hits),
        "gold_child_raw_hit_rate": round(len(gold_child_hits) / len(gold_child_scored), 6)
        if gold_child_scored
        else None,
        "expected_doc_child_hit_count": len(expected_doc_hits),
        "expected_doc_child_hit_rate": round(len(expected_doc_hits) / len(samples), 6) if samples else None,
        "raw_child_trace_missing_count": counts.get("raw_child_trace_missing", 0),
    }


def compare_variants(samples_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    b0_ids = {str(sample.get("sample_id") or "") for sample in samples_by_variant.get("b0_stable", [])}
    b1_ids = {str(sample.get("sample_id") or "") for sample in samples_by_variant.get("b1_parent_expansion", [])}
    return {
        "same_sample_ids": b0_ids == b1_ids,
        "shared_sample_count": len(b0_ids & b1_ids),
        "b0_only": sorted(b0_ids - b1_ids),
        "b1_only": sorted(b1_ids - b0_ids),
    }


def render_report(summary: dict[str, Any], samples_by_variant: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Raw Child Trace 审计报告",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- debug_result_dir: `{summary['debug_result_dir']}`",
        f"- support_audit_dir: `{summary['support_audit_dir']}`",
        f"- scope: `{summary['scope']}`",
        "",
        "## 总览",
        "",
        "| 变体 | 目标样本 | 可判断 gold child | gold child raw hit | expected doc child hit | trace 缺失 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, item in summary["variants"].items():
        lines.append(
            f"| {variant} | {item['target_case_count']} | {item['gold_child_judgable_count']} | "
            f"{item['gold_child_raw_hit_count']} ({pct(item['gold_child_raw_hit_rate'])}) | "
            f"{item['expected_doc_child_hit_count']} ({pct(item['expected_doc_child_hit_rate'])}) | "
            f"{item['raw_child_trace_missing_count']} |"
        )
    lines.extend(["", "## 分类", ""])
    lines.append("| 分类 | 含义 | b0_stable | b1_parent_expansion |")
    lines.append("|---|---|---:|---:|")
    all_classes = sorted(
        set(summary["variants"].get("b0_stable", {}).get("classification_counts", {}))
        | set(summary["variants"].get("b1_parent_expansion", {}).get("classification_counts", {}))
    )
    for name in all_classes:
        b0 = summary["variants"].get("b0_stable", {}).get("classification_counts", {}).get(name, 0)
        b1 = summary["variants"].get("b1_parent_expansion", {}).get("classification_counts", {}).get(name, 0)
        lines.append(f"| `{name}` | {CLASSIFICATION_LABELS.get(name, name)} | {b0} | {b1} |")
    lines.extend(["", "## 样例", ""])
    lines.append("| 变体 | sample_id | 分类 | gold child hit | expected doc child hit | nearest distance | expected doc parents |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for variant, samples in samples_by_variant.items():
        for sample in samples[:12]:
            lines.append(
                f"| {variant} | `{sample['sample_id']}` | `{sample['classification']}` | "
                f"{fmt_bool(sample.get('raw_child_gold_hit'))} | "
                f"{fmt_bool(sample.get('raw_expected_doc_child_hit'))} | "
                f"{fmt(sample.get('nearest_same_doc_parent_distance'))} | "
                f"{', '.join(sample.get('raw_expected_doc_parent_ids') or [])[:160]} |"
            )
    lines.extend(["", "## 解释", ""])
    lines.append("- `no_gold_child_id` 表示数据集中该样本没有 child 级 gold 标注，不能判断 gold child 是否被 raw 检索召回。")
    lines.append("- `same_doc_child_wrong_parent` 表示 child 检索已进入 expected doc，但落在错误 parent；这类才适合评估受控 parent 邻居扩展。")
    lines.append("- `gold_child_recalled_parent_lift_lost` 才指向 child -> parent 聚合或 metadata 映射问题。")
    return "\n".join(lines)


def compact_aggregation(aggregation: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_child_count": aggregation.get("raw_child_count"),
        "raw_parent_count": aggregation.get("raw_parent_count"),
        "materialized_parent_count": aggregation.get("materialized_parent_count"),
        "parent_ids_from_children": list(aggregation.get("parent_ids_from_children") or [])[:80],
    }


def compact_trace_preview(
    trace: list[dict[str, Any]],
    expected_docs: list[str],
    gold_children: list[str],
    gold_parents: list[str],
) -> list[dict[str, Any]]:
    expected_doc_set = set(expected_docs)
    gold_child_set = set(gold_children)
    gold_parent_set = set(gold_parents)
    selected = [
        item
        for item in trace
        if str(item.get("doc_id") or "") in expected_doc_set
        or str(item.get("child_chunk_id") or "") in gold_child_set
        or str(item.get("parent_chunk_id") or "") in gold_parent_set
    ]
    if not selected:
        selected = trace[:8]
    return [
        {
            "rank": item.get("rank"),
            "child_chunk_id": item.get("child_chunk_id"),
            "parent_chunk_id": item.get("parent_chunk_id"),
            "doc_id": item.get("doc_id"),
            "source": item.get("source"),
            "vector_score": item.get("vector_score"),
            "bm25_score": item.get("bm25_score"),
            "fusion_score": item.get("fusion_score"),
        }
        for item in selected[:20]
    ]


def nearest_parent_distance(parent_ids: list[str], gold_parents: list[str]) -> int | None:
    distances: list[int] = []
    for parent_id in parent_ids:
        parsed_parent = parse_parent_id(parent_id)
        if parsed_parent is None:
            continue
        for gold_parent in gold_parents:
            parsed_gold = parse_parent_id(gold_parent)
            if parsed_gold is None or parsed_parent[0] != parsed_gold[0]:
                continue
            distances.append(abs(parsed_parent[2] - parsed_gold[2]))
    return min(distances) if distances else None


def parse_parent_id(value: str) -> tuple[str, int, int] | None:
    match = PARENT_ID_RE.match(str(value or ""))
    if not match:
        return None
    return match.group("doc"), int(match.group("sec")), int(match.group("chunk"))


def any_in(targets: list[str], values: list[str]) -> bool:
    return bool(set(targets) & set(values))


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


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def fmt_bool(value: Any) -> str:
    if value is None:
        return "N/A"
    return "1" if bool(value) else "0"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    support = {
        "sample_id": "s1",
        "question": "q",
        "expected_doc_ids": ["doc_a"],
        "gold_child_chunk_ids": ["doc_a_sec01_chunk01::child001"],
        "gold_parent_chunk_ids": ["doc_a_sec01_chunk01"],
        "audit_buckets": ["rerank_doc_hit_parent_miss"],
        "rerank_audit": {"reason": "gold_parent_not_in_raw_retrieval"},
    }
    debug = {
        "raw_retrieved_parent_chunk_ids": ["doc_a_sec02_chunk02"],
        "debug_digest": {
            "raw_child_trace": {
                "raw_child_trace": [
                    {
                        "rank": 1,
                        "child_chunk_id": "doc_a_sec01_chunk01::child001",
                        "parent_chunk_id": "doc_a_sec01_chunk01",
                        "doc_id": "doc_a",
                    }
                ],
                "child_to_parent_aggregation": {
                    "raw_child_count": 1,
                    "raw_parent_count": 1,
                    "parent_ids_from_children": ["doc_a_sec01_chunk01"],
                },
            }
        },
    }
    row = audit_sample(variant="b0_stable", support_row=support, debug_row=debug)
    assert row["classification"] == "gold_child_recalled_parent_lift_lost", row

    debug["raw_retrieved_parent_chunk_ids"] = ["doc_a_sec02_chunk02"]
    debug["debug_digest"]["raw_child_trace"]["raw_child_trace"][0] = {
        "rank": 1,
        "child_chunk_id": "doc_a_sec02_chunk02::child001",
        "parent_chunk_id": "doc_a_sec02_chunk02",
        "doc_id": "doc_a",
    }
    row = audit_sample(variant="b0_stable", support_row=support, debug_row=debug)
    assert row["classification"] == "same_doc_child_wrong_parent", row

    support["gold_child_chunk_ids"] = []
    row = audit_sample(variant="b0_stable", support_row=support, debug_row=debug)
    assert row["classification"] == "no_gold_child_id", row
    print("self-test passed")


if __name__ == "__main__":
    main()
