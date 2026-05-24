from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_SOURCE_RUN_ID = "20260523_support_rerank_debug"
VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<sec>\d+)_chunk(?P<chunk>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline replay for priority1 rerank doc-hit parent-miss cases."
    )
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    source_run_id = args.source_run_id
    run_id = args.run_id or f"{source_run_id}_priority1_replay"
    audit_dir = RESULTS_ROOT / f"v3_support_rerank_audit_{source_run_id}"
    debug_dir = RESULTS_ROOT / f"v3_baseline_b0_b1_{source_run_id}"
    output_dir = RESULTS_ROOT / f"v3_priority1_replay_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_priority1_replay_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = {variant: load_jsonl(audit_dir / f"{variant}_samples.jsonl") for variant in VARIANTS}
    debug_rows = {
        variant: {
            str(row.get("sample_id") or ""): row
            for row in load_jsonl(debug_dir / variant / "results.jsonl")
        }
        for variant in VARIANTS
    }
    miss_rows = {
        variant: [
            row
            for row in rows
            if "rerank_doc_hit_parent_miss" in (row.get("audit_buckets") or [])
        ]
        for variant, rows in audit_rows.items()
    }

    baseline = build_baseline(miss_rows)
    raw_replay = replay_raw_same_doc(miss_rows)
    score_floor_replay = replay_score_floor(miss_rows, debug_rows)
    comparison_replay = replay_comparison_selection(miss_rows, debug_rows)
    topk_replay = replay_topk_cutoff(miss_rows, debug_rows)

    summary = {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audit_dir": str(audit_dir),
        "debug_dir": str(debug_dir),
        "scope": "priority1_rerank_doc_hit_parent_miss",
        "baseline": baseline,
        "raw_same_doc_replay": raw_replay,
        "score_floor_replay": score_floor_replay,
        "comparison_selection_replay": comparison_replay,
        "topk_cutoff_replay": topk_replay,
        "recommendation": build_recommendation(
            raw_replay=raw_replay,
            score_floor_replay=score_floor_replay,
            comparison_replay=comparison_replay,
            topk_replay=topk_replay,
        ),
    }

    write_json(output_dir / "replay_summary.json", summary)
    write_json(output_dir / "raw_same_doc_replay.json", raw_replay)
    write_json(output_dir / "score_floor_replay.json", score_floor_replay)
    write_json(output_dir / "comparison_selection_replay.json", comparison_replay)
    write_json(output_dir / "topk_cutoff_replay.json", topk_replay)
    write_jsonl(output_dir / "raw_same_doc_replay_samples.jsonl", flatten_samples(raw_replay))
    write_jsonl(output_dir / "score_floor_replay_samples.jsonl", flatten_samples(score_floor_replay))
    write_jsonl(output_dir / "comparison_selection_replay_samples.jsonl", flatten_samples(comparison_replay))
    write_jsonl(output_dir / "topk_cutoff_replay_samples.jsonl", flatten_samples(topk_replay))
    write_markdown(report_dir / "report.md", render_report(summary))
    print(json.dumps({"output_dir": str(output_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def build_baseline(miss_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant, rows in miss_rows.items():
        reasons = Counter(row.get("rerank_audit", {}).get("reason") for row in rows)
        buckets = Counter(classify_bucket(row) for row in rows)
        variants[variant] = {
            "case_count": len(rows),
            "reason_counts": dict(reasons),
            "diagnosis_bucket_counts": dict(buckets),
            "support_first_break_counts": dict(Counter(row.get("support_first_break") for row in rows)),
        }
    b0_ids = {str(row.get("sample_id") or "") for row in miss_rows.get("b0_stable", [])}
    b1_ids = {str(row.get("sample_id") or "") for row in miss_rows.get("b1_parent_expansion", [])}
    return {
        "variants": variants,
        "cross_variant": {
            "same_sample_ids": b0_ids == b1_ids,
            "shared_sample_count": len(b0_ids & b1_ids),
            "b0_only": sorted(b0_ids - b1_ids),
            "b1_only": sorted(b1_ids - b0_ids),
        },
    }


def replay_raw_same_doc(miss_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    windows = (1, 2)
    result: dict[str, Any] = {"variants": {}}
    for variant, rows in miss_rows.items():
        target_rows = [row for row in rows if classify_bucket(row) == "raw_same_doc_wrong_parent"]
        by_window: dict[str, Any] = {}
        for window in windows:
            samples = [raw_window_sample(row, window) for row in target_rows]
            by_window[f"same_doc_window_pm{window}"] = summarize_replay_samples(samples)
        oracle_samples = [raw_oracle_sample(row) for row in target_rows]
        by_window["oracle_add_gold_parent_from_expected_doc"] = summarize_replay_samples(oracle_samples)
        result["variants"][variant] = {
            "case_count": len(target_rows),
            "method_note": (
                "Window replay is an id-pattern estimate: for each raw expected-doc parent "
                "doc_x_secNN_chunkMM, it synthesizes neighboring parent ids by +/- window."
            ),
            "replays": by_window,
        }
    return result


def raw_window_sample(row: dict[str, Any], window: int) -> dict[str, Any]:
    raw_ids = [str(item) for item in row.get("raw_retrieved_parent_chunk_ids") or []]
    raw_set = set(raw_ids)
    gold = [str(item) for item in row.get("gold_parent_chunk_ids") or []]
    expected = {str(item) for item in row.get("expected_doc_ids") or []}
    sibling_ids = [item for item in raw_ids if parent_doc_id(item) in expected]
    added: list[str] = []
    for sibling_id in sibling_ids:
        added.extend(neighbor_parent_ids(sibling_id, window))
    added = dedupe(item for item in added if item not in raw_set)
    recovered = any(item in set(added) or item in raw_set for item in gold)
    return {
        "sample_id": row.get("sample_id"),
        "variant_key": row.get("variant_key"),
        "question": row.get("question"),
        "expected_doc_ids": row.get("expected_doc_ids") or [],
        "gold_parent_chunk_ids": gold,
        "window": window,
        "recovered": recovered,
        "added_candidate_count": len(added),
        "raw_expected_doc_sibling_count": len(sibling_ids),
        "raw_expected_doc_first_sibling_rank": first_rank(raw_ids, set(sibling_ids)),
        "raw_expected_doc_siblings_preview": [
            {"rank": idx, "parent_chunk_id": parent_id}
            for idx, parent_id in enumerate(raw_ids, start=1)
            if parent_id in set(sibling_ids)
        ][:12],
        "added_parent_ids_preview": added[:20],
    }


def raw_oracle_sample(row: dict[str, Any]) -> dict[str, Any]:
    raw_ids = [str(item) for item in row.get("raw_retrieved_parent_chunk_ids") or []]
    gold = [str(item) for item in row.get("gold_parent_chunk_ids") or []]
    expected = {str(item) for item in row.get("expected_doc_ids") or []}
    sibling_ids = [item for item in raw_ids if parent_doc_id(item) in expected]
    added = [item for item in gold if item not in set(raw_ids) and parent_doc_id(item) in expected]
    return {
        "sample_id": row.get("sample_id"),
        "variant_key": row.get("variant_key"),
        "question": row.get("question"),
        "expected_doc_ids": row.get("expected_doc_ids") or [],
        "gold_parent_chunk_ids": gold,
        "recovered": bool(added),
        "added_candidate_count": len(added),
        "raw_expected_doc_sibling_count": len(sibling_ids),
        "raw_expected_doc_first_sibling_rank": first_rank(raw_ids, set(sibling_ids)),
        "added_parent_ids_preview": added,
    }


def replay_score_floor(
    miss_rows: dict[str, list[dict[str, Any]]],
    debug_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    ratios = (0.3, 0.2, 0.1, 0.0)
    result: dict[str, Any] = {"variants": {}}
    for variant, rows in miss_rows.items():
        target_rows = [row for row in rows if classify_bucket(row) == "rerank_score_floor_filtered"]
        by_ratio: dict[str, Any] = {}
        for ratio in ratios:
            samples = [
                score_floor_sample(row, debug_rows[variant].get(str(row.get("sample_id") or ""), {}), ratio)
                for row in target_rows
            ]
            by_ratio[str(ratio)] = summarize_replay_samples(samples)
        result["variants"][variant] = {
            "case_count": len(target_rows),
            "current_ratio": 0.4,
            "replays": by_ratio,
        }
    return result


def score_floor_sample(row: dict[str, Any], debug_row: dict[str, Any], ratio: float) -> dict[str, Any]:
    trace = pre_floor_trace(debug_row)
    original_post_count = sum(1 for item in trace if item.get("post_floor_rank"))
    target_ids = set(row.get("gold_parent_chunk_ids") or [])
    top_score = to_float(trace[0].get("score")) if trace else None
    if not trace or ratio <= 0 or top_score is None or top_score <= 0:
        post = trace
        floor = None
    else:
        floor = top_score * ratio
        post = [item for item in trace if (to_float(item.get("score")) or 0.0) >= floor]
    post_rank = rank_first_target(post, target_ids)
    top_k = int(nested(row, "rerank_audit", "top_k") or 10)
    final_recovered = post_rank is not None and post_rank <= top_k
    return {
        "sample_id": row.get("sample_id"),
        "variant_key": row.get("variant_key"),
        "question": row.get("question"),
        "gold_parent_chunk_ids": row.get("gold_parent_chunk_ids") or [],
        "ratio": ratio,
        "floor": round(floor, 6) if floor is not None else None,
        "top_score": round(top_score, 6) if top_score is not None else None,
        "post_floor_recovered": post_rank is not None,
        "final_topk_recovered": final_recovered,
        "recovered": final_recovered,
        "target_post_floor_rank": post_rank,
        "top_k": top_k,
        "post_floor_count": len(post),
        "added_candidate_count": max(0, len(post) - original_post_count),
        "target_trace": compact_target_trace(row),
    }


def replay_comparison_selection(
    miss_rows: dict[str, list[dict[str, Any]]],
    debug_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"variants": {}}
    for variant, rows in miss_rows.items():
        target_rows = [row for row in rows if classify_bucket(row) == "comparison_selection_or_topk"]
        replays = {
            "disable_comparison_selection": [],
            "protect_pre_floor_top3": [],
            "increase_final_top_k_to_12": [],
        }
        for row in target_rows:
            debug_row = debug_rows[variant].get(str(row.get("sample_id") or ""), {})
            replays["disable_comparison_selection"].append(
                comparison_sample(row, debug_row, mode="disable_comparison_selection", top_k=10)
            )
            replays["protect_pre_floor_top3"].append(
                comparison_sample(row, debug_row, mode="protect_pre_floor_top3", top_k=10)
            )
            replays["increase_final_top_k_to_12"].append(
                comparison_sample(row, debug_row, mode="current_comparison", top_k=12)
            )
        result["variants"][variant] = {
            "case_count": len(target_rows),
            "replays": {
                key: summarize_replay_samples(samples)
                for key, samples in replays.items()
            },
        }
    return result


def comparison_sample(row: dict[str, Any], debug_row: dict[str, Any], *, mode: str, top_k: int) -> dict[str, Any]:
    trace = pre_floor_trace(debug_row)
    post = [item for item in trace if item.get("post_floor_rank")]
    post.sort(key=lambda item: int(item.get("post_floor_rank") or 999999))
    target_ids = set(row.get("gold_parent_chunk_ids") or [])
    selection = nested(debug_row, "debug_digest", "rerank_hits", "selection") or {}
    if mode == "disable_comparison_selection":
        selected = apply_diversity(post, selection)[:top_k]
        selected_ids = [parent_id(item) for item in selected]
        recovered = any(item in target_ids for item in selected_ids)
        displaced = []
    elif mode == "protect_pre_floor_top3":
        current_ids = list(nested(selection, "comparison_selection", "selected_chunk_ids") or [])
        target_trace = first_target_trace(row)
        target_id = str(target_trace.get("parent_chunk_id") or target_trace.get("chunk_id") or "")
        protected = (
            target_id
            and target_id in target_ids
            and target_trace.get("post_floor_rank") is not None
            and int(target_trace.get("pre_floor_rerank_rank") or 999999) <= 3
        )
        selected_ids = current_ids[:top_k]
        displaced = []
        if protected and target_id not in selected_ids:
            if len(selected_ids) >= top_k and selected_ids:
                displaced = [selected_ids[-1]]
                selected_ids = selected_ids[:-1]
            selected_ids.append(target_id)
        recovered = any(item in target_ids for item in selected_ids)
    else:
        selected = apply_current_comparison(post, debug_row, top_k=top_k)
        selected_ids = [parent_id(item) for item in selected]
        recovered = any(item in target_ids for item in selected_ids)
        displaced = []
    return {
        "sample_id": row.get("sample_id"),
        "variant_key": row.get("variant_key"),
        "question": row.get("question"),
        "gold_parent_chunk_ids": row.get("gold_parent_chunk_ids") or [],
        "mode": mode,
        "top_k": top_k,
        "recovered": recovered,
        "added_candidate_count": max(0, top_k - 10),
        "selected_parent_ids_preview": selected_ids[:15],
        "displaced_parent_ids": displaced,
        "target_trace": compact_target_trace(row),
        "branch_steps": list(nested(selection, "comparison_selection", "branch_steps") or []),
    }


def replay_topk_cutoff(
    miss_rows: dict[str, list[dict[str, Any]]],
    debug_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    top_ks = (10, 12, 15, 20)
    result: dict[str, Any] = {"variants": {}}
    for variant, rows in miss_rows.items():
        target_rows = [row for row in rows if classify_bucket(row) == "rerank_top10_cutoff"]
        by_topk: dict[str, Any] = {}
        for top_k in top_ks:
            samples = [
                topk_sample(row, debug_rows[variant].get(str(row.get("sample_id") or ""), {}), top_k)
                for row in target_rows
            ]
            by_topk[str(top_k)] = summarize_replay_samples(samples)
        result["variants"][variant] = {
            "case_count": len(target_rows),
            "replays": by_topk,
        }
    return result


def topk_sample(row: dict[str, Any], debug_row: dict[str, Any], top_k: int) -> dict[str, Any]:
    post = [item for item in pre_floor_trace(debug_row) if item.get("post_floor_rank")]
    post.sort(key=lambda item: int(item.get("post_floor_rank") or 999999))
    target_rank = rank_first_target(post, set(row.get("gold_parent_chunk_ids") or []))
    recovered = target_rank is not None and target_rank <= top_k
    return {
        "sample_id": row.get("sample_id"),
        "variant_key": row.get("variant_key"),
        "question": row.get("question"),
        "gold_parent_chunk_ids": row.get("gold_parent_chunk_ids") or [],
        "top_k": top_k,
        "recovered": recovered,
        "target_post_floor_rank": target_rank,
        "added_candidate_count": max(0, top_k - 10),
        "target_trace": compact_target_trace(row),
    }


def apply_current_comparison(post: list[dict[str, Any]], debug_row: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    selection = nested(debug_row, "debug_digest", "rerank_hits", "selection") or {}
    diversified = apply_diversity(post, selection)
    query_count = int(nested(selection, "comparison_selection", "query_count") or 1)
    if query_count <= 1:
        return diversified[:top_k]
    selected: list[dict[str, Any]] = []
    selected_doc_ids: set[str] = set()
    for query_idx in range(1, query_count):
        best = None
        best_score = None
        for item in diversified:
            doc_id = str(item.get("doc_id") or "")
            if doc_id in selected_doc_ids:
                continue
            scores = item.get("query_scores") or []
            if query_idx >= len(scores):
                continue
            score = to_float(scores[query_idx])
            if score is None:
                continue
            if best_score is None or score > best_score:
                best = item
                best_score = score
        if best is None or best_score is None or best_score < 1.0:
            continue
        selected.append(best)
        selected_doc_ids.add(str(best.get("doc_id") or ""))
        if len(selected) >= top_k:
            return selected[:top_k]
    for item in diversified:
        doc_id = str(item.get("doc_id") or "")
        if doc_id in selected_doc_ids:
            continue
        selected.append(item)
        selected_doc_ids.add(doc_id)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def apply_diversity(post: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    diversity = selection.get("doc_diversity") if isinstance(selection, dict) else {}
    if not isinstance(diversity, dict) or not diversity.get("applied"):
        return list(post)
    max_per_doc = int(diversity.get("max_per_doc") or 1)
    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in post:
        doc_id = str(item.get("doc_id") or "")
        if counts[doc_id] < max_per_doc:
            selected.append(item)
            counts[doc_id] += 1
        else:
            overflow.append(item)
    return selected + overflow


def summarize_replay_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [sample for sample in samples if sample.get("recovered")]
    added_counts = [int(sample.get("added_candidate_count") or 0) for sample in samples]
    return {
        "case_count": len(samples),
        "recovered_count": len(recovered),
        "recovered_rate": round(len(recovered) / len(samples), 6) if samples else None,
        "recovered_sample_ids": [str(sample.get("sample_id") or "") for sample in recovered],
        "added_candidate_count_avg": round(sum(added_counts) / len(added_counts), 3) if added_counts else None,
        "added_candidate_count_median": median(added_counts) if added_counts else None,
        "added_candidate_count_max": max(added_counts) if added_counts else None,
        "samples": samples,
    }


def build_recommendation(
    *,
    raw_replay: dict[str, Any],
    score_floor_replay: dict[str, Any],
    comparison_replay: dict[str, Any],
    topk_replay: dict[str, Any],
) -> list[str]:
    b0_score = nested(score_floor_replay, "variants", "b0_stable", "replays", "0.0", "recovered_count") or 0
    b0_raw_pm2 = nested(raw_replay, "variants", "b0_stable", "replays", "same_doc_window_pm2", "recovered_count") or 0
    b0_comp_disable = (
        nested(comparison_replay, "variants", "b0_stable", "replays", "disable_comparison_selection", "recovered_count")
        or 0
    )
    b0_topk12 = nested(topk_replay, "variants", "b0_stable", "replays", "12", "recovered_count") or 0
    recommendations = []
    if b0_score:
        recommendations.append(
            f"Score-floor replay recovers {b0_score} B0 cases at ratio=0.0; validate a less blunt floor policy first."
        )
    if b0_raw_pm2:
        recommendations.append(
            f"Same-doc +/-2 window recovers {b0_raw_pm2} B0 raw-miss cases; inspect added candidate volume before policy work."
        )
    if b0_comp_disable:
        recommendations.append(
            f"Disabling comparison selection would recover {b0_comp_disable} B0 cases; inspect intent/selection guard for top-ranked targets."
        )
    if b0_topk12:
        recommendations.append(f"Raising topK to 12 recovers {b0_topk12} B0 cutoff cases; low priority due to small count.")
    return recommendations


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Priority 1 replay 报告",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- source_run_id: `{summary['source_run_id']}`",
        f"- audit_dir: `{summary['audit_dir']}`",
        f"- debug_dir: `{summary['debug_dir']}`",
        "",
        "## Baseline",
        "",
        "| 变体 | 总数 | raw 同文档错父块 | score floor | comparison/topK | topK cutoff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, item in summary["baseline"]["variants"].items():
        counts = item["diagnosis_bucket_counts"]
        lines.append(
            f"| {variant} | {item['case_count']} | {counts.get('raw_same_doc_wrong_parent', 0)} | "
            f"{counts.get('rerank_score_floor_filtered', 0)} | {counts.get('comparison_selection_or_topk', 0)} | "
            f"{counts.get('rerank_top10_cutoff', 0)} |"
        )

    lines.extend(["", "## Raw Same-Doc Replay", ""])
    lines.append("| 变体 | 方法 | 恢复数 | 恢复率 | 平均新增候选 | 最大新增候选 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant, item in summary["raw_same_doc_replay"]["variants"].items():
        for name, replay in item["replays"].items():
            lines.append(metric_row(variant, name, replay))

    lines.extend(["", "## Score Floor Replay", ""])
    lines.append("| 变体 | ratio | 恢复数 | 恢复率 | 平均新增候选 | 最大新增候选 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant, item in summary["score_floor_replay"]["variants"].items():
        for name, replay in item["replays"].items():
            lines.append(metric_row(variant, name, replay))

    lines.extend(["", "## Comparison Selection Replay", ""])
    lines.append("| 变体 | 方法 | 恢复数 | 恢复率 | 平均新增候选 | 最大新增候选 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant, item in summary["comparison_selection_replay"]["variants"].items():
        for name, replay in item["replays"].items():
            lines.append(metric_row(variant, name, replay))

    lines.extend(["", "## TopK Cutoff Replay", ""])
    lines.append("| 变体 | topK | 恢复数 | 恢复率 | 平均新增候选 | 最大新增候选 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant, item in summary["topk_cutoff_replay"]["variants"].items():
        for name, replay in item["replays"].items():
            lines.append(metric_row(variant, name, replay))

    lines.extend(["", "## 建议", ""])
    for item in summary["recommendation"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 注意", ""])
    lines.append("- raw same-doc window 是基于 parent id 编号的离线估算，不验证该邻居块是否真实存在于索引。")
    lines.append("- score floor、comparison 和 topK replay 基于现有 rerank trace，不重新调用 reranker。")
    lines.append("- 本报告只验证恢复上限和候选膨胀规模，不代表最终策略改动。")
    return "\n".join(lines)


def metric_row(variant: str, name: str, replay: dict[str, Any]) -> str:
    return (
        f"| {variant} | `{name}` | {replay['recovered_count']} | {pct(replay['recovered_rate'])} | "
        f"{fmt(replay['added_candidate_count_avg'])} | {fmt(replay['added_candidate_count_max'])} |"
    )


def flatten_samples(section: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, variant_data in (section.get("variants") or {}).items():
        for replay_name, replay_data in (variant_data.get("replays") or {}).items():
            for sample in replay_data.get("samples") or []:
                row = dict(sample)
                row["variant_key"] = variant
                row["replay_name"] = replay_name
                rows.append(row)
    return rows


def classify_bucket(row: dict[str, Any]) -> str:
    reason = row.get("rerank_audit", {}).get("reason")
    if reason == "gold_parent_not_in_raw_retrieval":
        return "raw_same_doc_wrong_parent"
    if reason == "score_floor_filtered":
        return "rerank_score_floor_filtered"
    if reason == "comparison_selection_or_topk":
        return "comparison_selection_or_topk"
    if reason == "top10_cutoff":
        return "rerank_top10_cutoff"
    return str(reason or "unknown")


def pre_floor_trace(debug_row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(nested(debug_row, "debug_digest", "rerank_hits", "ranking_trace") or [])
    trace = [item for item in trace if isinstance(item, dict)]
    trace.sort(key=lambda item: int(item.get("pre_floor_rerank_rank") or 999999))
    return trace


def first_target_trace(row: dict[str, Any]) -> dict[str, Any]:
    traces = row.get("rerank_audit", {}).get("target_traces") or []
    return traces[0] if traces and isinstance(traces[0], dict) else {}


def compact_target_trace(row: dict[str, Any]) -> dict[str, Any]:
    trace = first_target_trace(row)
    return {
        "raw_rank": trace.get("raw_retrieval_rank"),
        "pre_floor_rank": trace.get("pre_floor_rerank_rank"),
        "post_floor_rank": trace.get("post_floor_rank"),
        "final_rank": trace.get("final_top10_rank"),
        "score": trace.get("score"),
        "score_gap_to_top10_cutoff": row.get("rerank_audit", {}).get("score_gap_to_top10_cutoff"),
        "final_drop_reason": trace.get("final_drop_reason"),
    }


def rank_first_target(trace: list[dict[str, Any]], target_parent_ids: set[str]) -> int | None:
    for idx, item in enumerate(trace, start=1):
        if parent_id(item) in target_parent_ids:
            return idx
    return None


def parent_id(item: dict[str, Any]) -> str:
    return str(item.get("parent_chunk_id") or item.get("chunk_id") or "")


def neighbor_parent_ids(parent_id_value: str, window: int) -> list[str]:
    parsed = parse_parent_id(parent_id_value)
    if parsed is None:
        return []
    doc_id, sec_num, chunk_num, sec_width, chunk_width = parsed
    neighbors = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        sec = sec_num + offset
        chunk = chunk_num + offset
        if sec < 0 or chunk < 0:
            continue
        neighbors.append(f"{doc_id}_sec{sec:0{sec_width}d}_chunk{chunk:0{chunk_width}d}")
    return neighbors


def parse_parent_id(value: str) -> tuple[str, int, int, int, int] | None:
    match = PARENT_ID_RE.match(str(value or ""))
    if not match:
        return None
    sec_text = match.group("sec")
    chunk_text = match.group("chunk")
    return (
        match.group("doc"),
        int(sec_text),
        int(chunk_text),
        len(sec_text),
        len(chunk_text),
    )


def parent_doc_id(value: str) -> str:
    parsed = parse_parent_id(value)
    if parsed is not None:
        return parsed[0]
    parts = str(value or "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(value or "")


def first_rank(values: list[str], targets: set[str]) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value in targets:
            return idx
    return None


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


if __name__ == "__main__":
    main()
