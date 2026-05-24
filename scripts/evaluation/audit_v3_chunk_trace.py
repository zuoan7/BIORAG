from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")


BUCKETS = {
    "raw_doc_miss": "raw retrieval 未命中期望文档",
    "raw_doc_hit_rerank_doc_miss": "raw retrieval 命中文档，但 rerank top10 未命中文档",
    "rerank_doc_hit_parent_miss": "rerank top10 命中文档，但未命中目标父块",
    "rerank_parent_hit_seed_miss": "rerank top10 命中父块，但 final_top_k seed 未保留",
    "seed_parent_hit_final_miss": "seed 命中父块，但 final context 未保留",
    "final_parent_hit_support_parent_miss": "final context 命中父块，但 support 未保留",
    "support_parent_hit_child_metadata_miss": "support 命中父块，但未携带目标子证据 id",
    "support_child_hit_citation_child_miss": "support 命中目标子证据，但 citation 未保留目标子证据",
    "child_evidence_hit_answer_failed": "目标子证据已进入 support/citation，但答案正确性失败",
    "judge_schema_or_call_issue": "judge 调用或 schema 问题",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v3 B0/B1 chunk trace failures.")
    parser.add_argument("--judged-result-dir", required=False)
    parser.add_argument("--debug-result-dir", required=False)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.judged_result_dir or not args.debug_result_dir:
        raise SystemExit("--judged-result-dir and --debug-result-dir are required unless --self-test is used")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"v3_chunk_trace_audit_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_chunk_trace_audit_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    judged_dir = Path(args.judged_result_dir)
    debug_dir = Path(args.debug_result_dir)
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
        write_jsonl(result_dir / f"{variant}_audit_samples.jsonl", samples)

    comparison = compare_variants(samples_by_variant)
    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judged_result_dir": str(judged_dir),
        "debug_result_dir": str(debug_dir),
        "variants": summaries,
        "comparison": comparison,
    }
    write_json(result_dir / "audit_summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary))


def load_variant(root: Path, variant: str) -> dict[str, dict[str, Any]]:
    path = root / variant / "results.jsonl"
    rows = {}
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
        row = merge_for_audit(judged, debug)
        row["variant_key"] = variant
        row["audit_buckets"] = classify_buckets(row)
        audited.append(row)
    return audited


def merge_for_audit(judged: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    sample_id = str((debug or judged).get("sample_id") or "")
    expected_docs = [str(v) for v in (debug or judged).get("expected_doc_ids") or []]
    gold_chunks = [str(v) for v in (debug or judged).get("gold_chunk_ids") or []]
    gold_parents = dedupe(parent_chunk_id(chunk_id) for chunk_id in gold_chunks)
    expected_route = str((debug or judged).get("expected_route") or "")
    judge = judged.get("judge") or {}

    raw_chunk_ids = list(debug.get("raw_retrieved_chunk_ids") or [])
    raw_doc_ids = list(debug.get("raw_retrieved_doc_ids") or [])
    rerank_chunk_ids = list(debug.get("retrieved_chunk_ids_top10") or [])
    rerank_doc_ids = list(debug.get("retrieved_doc_ids_top10") or [])
    seed_chunk_ids = list(debug.get("seed_chunk_ids") or stage_ids(debug, "seed_chunks"))
    final_chunk_ids = list(debug.get("final_chunk_ids") or stage_ids(debug, "final_chunks"))
    support_chunk_ids = list(debug.get("support_chunk_ids") or [])
    citation_chunk_ids = list(debug.get("citation_chunk_ids") or [])
    support_child_ids = list(debug.get("support_matched_child_chunk_ids") or [])
    citation_child_ids = list(debug.get("citation_matched_child_chunk_ids") or [])

    raw_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in raw_chunk_ids)
    rerank_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in rerank_chunk_ids)
    seed_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in seed_chunk_ids)
    final_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in final_chunk_ids)
    support_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in support_chunk_ids)
    citation_parent_ids = dedupe(parent_chunk_id(chunk_id) for chunk_id in citation_chunk_ids)

    raw_doc_hit = any(doc_id in set(raw_doc_ids) for doc_id in expected_docs) if expected_docs else None
    rerank_doc_hit = any(doc_id in set(rerank_doc_ids) for doc_id in expected_docs) if expected_docs else None
    rerank_parent_hit = any(parent_id in set(rerank_parent_ids) for parent_id in gold_parents) if gold_parents else None
    seed_parent_hit = any(parent_id in set(seed_parent_ids) for parent_id in gold_parents) if gold_parents else None
    final_parent_hit = any(parent_id in set(final_parent_ids) for parent_id in gold_parents) if gold_parents else None
    support_parent_hit = any(parent_id in set(support_parent_ids) for parent_id in gold_parents) if gold_parents else None
    citation_parent_hit = any(parent_id in set(citation_parent_ids) for parent_id in gold_parents) if gold_parents else None
    support_child_hit = any(chunk_id in set(support_child_ids) for chunk_id in gold_chunks) if gold_chunks else None
    citation_child_hit = (
        any(chunk_id in set(citation_child_ids) for chunk_id in gold_chunks)
        if gold_chunks and expected_route != "negative"
        else None
    )
    correctness_pass = nested(judge, "answer_correctness", "correctness_pass")
    parent_expansion = (debug.get("debug_digest") or {}).get("parent_expansion") or {}

    return {
        "sample_id": sample_id,
        "question": (debug or judged).get("question"),
        "category": (debug or judged).get("category"),
        "expected_route": expected_route,
        "expected_doc_ids": expected_docs,
        "gold_chunk_ids": gold_chunks,
        "gold_parent_chunk_ids": gold_parents,
        "raw_doc_hit": raw_doc_hit,
        "rerank_doc_hit": rerank_doc_hit,
        "rerank_parent_hit": rerank_parent_hit,
        "seed_parent_hit": seed_parent_hit,
        "final_parent_hit": final_parent_hit,
        "support_parent_hit": support_parent_hit,
        "citation_parent_hit": citation_parent_hit,
        "support_child_hit": support_child_hit,
        "citation_child_hit": citation_child_hit,
        "answer_correctness_pass": correctness_pass,
        "judge_error": bool(judge.get("judge_error")) if judge else False,
        "judge_schema_ok": judge_schema_ok(judge),
        "raw_retrieved_doc_ids": raw_doc_ids,
        "raw_retrieved_parent_chunk_ids": raw_parent_ids,
        "rerank_parent_chunk_ids_top10": rerank_parent_ids,
        "seed_parent_chunk_ids": seed_parent_ids,
        "final_parent_chunk_ids": final_parent_ids,
        "support_parent_chunk_ids": support_parent_ids,
        "citation_parent_chunk_ids": citation_parent_ids,
        "support_matched_child_chunk_ids": support_child_ids,
        "citation_matched_child_chunk_ids": citation_child_ids,
        "parent_expansion": parent_expansion,
        "stage_hash": stable_stage_hash(
            {
                "raw": raw_parent_ids,
                "rerank": rerank_parent_ids,
                "seed": seed_parent_ids,
                "final": final_parent_ids,
                "support": support_parent_ids,
                "citation": citation_parent_ids,
            }
        ),
    }


def classify_buckets(row: dict[str, Any]) -> list[str]:
    buckets = []
    if row["raw_doc_hit"] is False:
        buckets.append("raw_doc_miss")
    if row["raw_doc_hit"] is True and row["rerank_doc_hit"] is False:
        buckets.append("raw_doc_hit_rerank_doc_miss")
    if row["rerank_doc_hit"] is True and row["rerank_parent_hit"] is False:
        buckets.append("rerank_doc_hit_parent_miss")
    if row["rerank_parent_hit"] is True and row["seed_parent_hit"] is False:
        buckets.append("rerank_parent_hit_seed_miss")
    if row["seed_parent_hit"] is True and row["final_parent_hit"] is False:
        buckets.append("seed_parent_hit_final_miss")
    if row["final_parent_hit"] is True and row["support_parent_hit"] is False:
        buckets.append("final_parent_hit_support_parent_miss")
    if row["support_parent_hit"] is True and row["support_child_hit"] is False:
        buckets.append("support_parent_hit_child_metadata_miss")
    if row["support_child_hit"] is True and row["citation_child_hit"] is False:
        buckets.append("support_child_hit_citation_child_miss")
    if (row["support_child_hit"] is True or row["citation_child_hit"] is True) and row["answer_correctness_pass"] is False:
        buckets.append("child_evidence_hit_answer_failed")
    if row["judge_error"] or row["judge_schema_ok"] is False:
        buckets.append("judge_schema_or_call_issue")
    return buckets


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(samples)
    bucket_counts = {key: [] for key in BUCKETS}
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        category = str(sample.get("category") or "unknown")
        for bucket in sample["audit_buckets"]:
            bucket_counts[bucket].append(sample["sample_id"])
            by_category[category][bucket] += 1

    parent_expansion_reasons = Counter(
        str((sample.get("parent_expansion") or {}).get("reason") or "")
        for sample in samples
        if sample.get("parent_expansion")
    )
    added_target_parent = sum(
        1
        for sample in samples
        if sample.get("seed_parent_hit") is False and sample.get("final_parent_hit") is True
    )
    return {
        "sample_count": sample_count,
        "rates": {
            "raw_doc_hit_rate": rate(sample.get("raw_doc_hit") for sample in samples),
            "rerank_doc_hit_rate": rate(sample.get("rerank_doc_hit") for sample in samples),
            "rerank_parent_hit_rate": rate(sample.get("rerank_parent_hit") for sample in samples),
            "seed_parent_hit_rate": rate(sample.get("seed_parent_hit") for sample in samples),
            "final_parent_hit_rate": rate(sample.get("final_parent_hit") for sample in samples),
            "support_parent_hit_rate": rate(sample.get("support_parent_hit") for sample in samples),
            "support_child_hit_rate": rate(sample.get("support_child_hit") for sample in samples),
            "citation_child_hit_rate": rate(sample.get("citation_child_hit") for sample in samples),
            "answer_correctness_pass_rate": rate(sample.get("answer_correctness_pass") for sample in samples),
        },
        "buckets": {
            key: {
                "label": label,
                "count": len(sample_ids),
                "rate": round(len(sample_ids) / sample_count, 6) if sample_count else None,
                "sample_ids_preview": sample_ids[:30],
            }
            for key, label in BUCKETS.items()
            for sample_ids in [bucket_counts[key]]
        },
        "by_category": {
            category: dict(counts)
            for category, counts in sorted(by_category.items())
        },
        "parent_expansion_reason_counts": dict(parent_expansion_reasons),
        "seed_miss_final_hit_count": added_target_parent,
        "top_examples": build_top_examples(samples),
    }


def compare_variants(samples_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    b0 = {sample["sample_id"]: sample for sample in samples_by_variant.get("b0_stable", [])}
    b1 = {sample["sample_id"]: sample for sample in samples_by_variant.get("b1_parent_expansion", [])}
    sample_ids = sorted(set(b0) & set(b1))
    rerank_same = [
        sample_id
        for sample_id in sample_ids
        if b0[sample_id].get("rerank_parent_chunk_ids_top10") == b1[sample_id].get("rerank_parent_chunk_ids_top10")
    ]
    final_parent_improved = [
        sample_id
        for sample_id in sample_ids
        if b0[sample_id].get("final_parent_hit") is not True and b1[sample_id].get("final_parent_hit") is True
    ]
    final_parent_regressed = [
        sample_id
        for sample_id in sample_ids
        if b0[sample_id].get("final_parent_hit") is True and b1[sample_id].get("final_parent_hit") is not True
    ]
    support_child_improved = [
        sample_id
        for sample_id in sample_ids
        if b0[sample_id].get("support_child_hit") is not True and b1[sample_id].get("support_child_hit") is True
    ]
    support_child_regressed = [
        sample_id
        for sample_id in sample_ids
        if b0[sample_id].get("support_child_hit") is True and b1[sample_id].get("support_child_hit") is not True
    ]
    return {
        "shared_sample_count": len(sample_ids),
        "rerank_parent_top10_same_count": len(rerank_same),
        "rerank_parent_top10_same_rate": round(len(rerank_same) / len(sample_ids), 6) if sample_ids else None,
        "final_parent_improved_count": len(final_parent_improved),
        "final_parent_improved_preview": final_parent_improved[:30],
        "final_parent_regressed_count": len(final_parent_regressed),
        "final_parent_regressed_preview": final_parent_regressed[:30],
        "support_child_improved_count": len(support_child_improved),
        "support_child_improved_preview": support_child_improved[:30],
        "support_child_regressed_count": len(support_child_regressed),
        "support_child_regressed_preview": support_child_regressed[:30],
    }


def build_top_examples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for sample in samples:
        if not sample["audit_buckets"]:
            continue
        examples.append(
            {
                "sample_id": sample["sample_id"],
                "category": sample.get("category"),
                "buckets": sample["audit_buckets"],
                "gold_parent_chunk_ids": sample.get("gold_parent_chunk_ids"),
                "rerank_parent_chunk_ids_top10": sample.get("rerank_parent_chunk_ids_top10")[:5],
                "final_parent_chunk_ids": sample.get("final_parent_chunk_ids")[:5],
                "support_parent_chunk_ids": sample.get("support_parent_chunk_ids")[:5],
                "support_matched_child_chunk_ids": sample.get("support_matched_child_chunk_ids")[:5],
            }
        )
    return examples[:30]


def stage_ids(row: dict[str, Any], key: str) -> list[str]:
    stage = ((row.get("debug_digest") or {}).get(key) or {})
    return list(stage.get("kept_chunk_ids") or stage.get("chunk_ids") or [])


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


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


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def judge_schema_ok(judge: dict[str, Any]) -> bool | None:
    if not judge:
        return None
    return all(
        isinstance(judge.get(section), dict)
        for section in ("answer_correctness", "faithfulness", "citation_accuracy")
    )


def rate(values: Any) -> float | None:
    vals = [value for value in values if value is not None]
    if not vals:
        return None
    return round(sum(1 for value in vals if bool(value)) / len(vals), 6)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def stable_stage_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# v3 chunk trace 审计报告",
        "",
        f"- 运行 ID：`{summary['run_id']}`",
        f"- judged 结果目录：`{summary['judged_result_dir']}`",
        f"- debug 结果目录：`{summary['debug_result_dir']}`",
        "",
        "## 总体阶段命中率",
        "",
        "| 变体 | raw doc hit | rerank doc hit | rerank parent hit | seed parent hit | final parent hit | support parent hit | support child hit | citation child hit | 答案通过率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, item in summary["variants"].items():
        rates = item["rates"]
        lines.append(
            f"| {variant} | {pct(rates['raw_doc_hit_rate'])} | {pct(rates['rerank_doc_hit_rate'])} | "
            f"{pct(rates['rerank_parent_hit_rate'])} | {pct(rates['seed_parent_hit_rate'])} | "
            f"{pct(rates['final_parent_hit_rate'])} | {pct(rates['support_parent_hit_rate'])} | "
            f"{pct(rates['support_child_hit_rate'])} | {pct(rates['citation_child_hit_rate'])} | "
            f"{pct(rates['answer_correctness_pass_rate'])} |"
        )

    lines.extend(["", "## 诊断桶", ""])
    for variant, item in summary["variants"].items():
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| 问题类型 | 样本数 | 占比 | 样本预览 |")
        lines.append("|---|---:|---:|---|")
        for bucket_key, bucket in item["buckets"].items():
            preview = ", ".join(bucket["sample_ids_preview"])
            lines.append(
                f"| {bucket['label']} | {bucket['count']} | {pct(bucket['rate'])} | {preview} |"
            )
        lines.append("")
        lines.append("parent expansion reason 分布：")
        lines.append("")
        for reason, count in sorted(item["parent_expansion_reason_counts"].items()):
            lines.append(f"- `{reason or 'empty'}`：{count}")
        lines.append("")
        lines.append(f"seed 未命中但 final 命中目标父块：{item['seed_miss_final_hit_count']} 条")
        lines.append("")

    comparison = summary["comparison"]
    lines.extend(
        [
            "## B1 相对 B0",
            "",
            f"- 共同样本数：{comparison['shared_sample_count']}",
            f"- rerank parent top10 完全一致：{comparison['rerank_parent_top10_same_count']} 条，{pct(comparison['rerank_parent_top10_same_rate'])}",
            f"- final parent hit 改善：{comparison['final_parent_improved_count']} 条",
            f"- final parent hit 回退：{comparison['final_parent_regressed_count']} 条",
            f"- support child hit 改善：{comparison['support_child_improved_count']} 条",
            f"- support child hit 回退：{comparison['support_child_regressed_count']} 条",
            "",
            "## 典型样本",
            "",
        ]
    )
    for variant, item in summary["variants"].items():
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| sample_id | category | buckets | gold parent | rerank top5 parent | final top5 parent | support top5 parent | support child top5 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for example in item["top_examples"][:15]:
            lines.append(
                f"| {example['sample_id']} | {example['category']} | {', '.join(example['buckets'])} | "
                f"{', '.join(example['gold_parent_chunk_ids'])} | "
                f"{', '.join(example['rerank_parent_chunk_ids_top10'])} | "
                f"{', '.join(example['final_parent_chunk_ids'])} | "
                f"{', '.join(example['support_parent_chunk_ids'])} | "
                f"{', '.join(example['support_matched_child_chunk_ids'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_self_test() -> None:
    row = {
        "raw_doc_hit": True,
        "rerank_doc_hit": True,
        "rerank_parent_hit": True,
        "seed_parent_hit": True,
        "final_parent_hit": True,
        "support_parent_hit": True,
        "support_child_hit": False,
        "citation_child_hit": False,
        "answer_correctness_pass": False,
        "judge_error": False,
        "judge_schema_ok": True,
    }
    assert classify_buckets(row) == ["support_parent_hit_child_metadata_miss"]
    row["support_child_hit"] = True
    assert classify_buckets(row) == [
        "support_child_hit_citation_child_miss",
        "child_evidence_hit_answer_failed",
    ]
    assert parent_chunk_id("doc_1_sec01_chunk02::child003") == "doc_1_sec01_chunk02"
    print("self-test passed")


if __name__ == "__main__":
    main()
