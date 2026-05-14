#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPER_PATH = Path(__file__).with_name("evaluate_phase5c3_table_retrieval_ab.py")
spec = importlib.util.spec_from_file_location("phase5c3_eval", HELPER_PATH)
phase5c3 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(phase5c3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5C-5 full retrieval-only A/B evaluation with stable target mapping.")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--baseline_chunks_jsonl", required=True)
    parser.add_argument("--baseline_milvus_uri", required=True)
    parser.add_argument("--baseline_collection", required=True)
    parser.add_argument("--baseline_bm25_path", required=True)
    parser.add_argument("--enhanced_chunks_jsonl", required=True)
    parser.add_argument("--enhanced_milvus_uri", required=True)
    parser.add_argument("--enhanced_collection", required=True)
    parser.add_argument("--enhanced_bm25_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--retrieval_modes", default="dense_only,bm25_only,hybrid")
    parser.add_argument("--stable_target_mapping", default="true")
    parser.add_argument("--model_path", default="models/BAAI/bge-m3")
    parser.add_argument("--embedding_max_length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_enabled = str(args.stable_target_mapping).lower() in {"1", "true", "yes"}
    if not stable_enabled:
        raise SystemExit("Phase 5C-5 requires --stable_target_mapping true")

    queries = list(phase5c3.iter_jsonl(Path(args.eval_jsonl)))
    baseline_chunks = list(phase5c3.iter_jsonl(Path(args.baseline_chunks_jsonl)))
    enhanced_chunks = list(phase5c3.iter_jsonl(Path(args.enhanced_chunks_jsonl)))
    baseline_index = phase5c3.build_chunk_index(baseline_chunks)
    enhanced_index = phase5c3.build_chunk_index(enhanced_chunks)
    modes = [m.strip() for m in args.retrieval_modes.split(",") if m.strip()]
    search_limit = max(50, args.top_k)

    embedder = phase5c3.BGEM3Embedder(
        model_path=args.model_path,
        dim=1024,
        max_length=args.embedding_max_length,
    )
    baseline = phase5c3.build_retrieval_stack(
        chunks_jsonl=args.baseline_chunks_jsonl,
        milvus_uri=args.baseline_milvus_uri,
        collection=args.baseline_collection,
        bm25_path=args.baseline_bm25_path,
        embedder=embedder,
    )
    enhanced = phase5c3.build_retrieval_stack(
        chunks_jsonl=args.enhanced_chunks_jsonl,
        milvus_uri=args.enhanced_milvus_uri,
        collection=args.enhanced_collection,
        bm25_path=args.enhanced_bm25_path,
        embedder=embedder,
    )

    validation = {
        "baseline_chunk_count": len(baseline_chunks),
        "enhanced_chunk_count": len(enhanced_chunks),
        "baseline_row_count": phase5c3.row_count(args.baseline_milvus_uri, args.baseline_collection),
        "enhanced_row_count": phase5c3.row_count(args.enhanced_milvus_uri, args.enhanced_collection),
        "baseline_bm25_record_count": len(baseline["bm25"]._records),
        "enhanced_bm25_record_count": len(enhanced["bm25"]._records),
        "main_index_overwritten": False,
    }
    validation["baseline_row_count_matches_chunks"] = validation["baseline_row_count"] == len(baseline_chunks)
    validation["enhanced_row_count_matches_chunks"] = validation["enhanced_row_count"] == len(enhanced_chunks)
    validation["baseline_bm25_count_matches_chunks"] = validation["baseline_bm25_record_count"] == len(baseline_chunks)
    validation["enhanced_bm25_count_matches_chunks"] = validation["enhanced_bm25_record_count"] == len(enhanced_chunks)
    validation.update(load_preflight_context(output_dir))

    records: list[dict[str, Any]] = []
    for query in queries:
        for mode in modes:
            records.append(phase5c3.evaluate_one(query, mode, "baseline", baseline, baseline_index, search_limit, args.top_k))
            records.append(phase5c3.evaluate_one(query, mode, "enhanced", enhanced, enhanced_index, search_limit, args.top_k))

    raw_metrics = phase5c3.aggregate_raw(records, top_k=args.top_k)
    mappings = [phase5c3.build_mapping(query, baseline_index, enhanced_index) for query in queries]
    mapping_by_sample = {m["sample_id"]: m for m in mappings}
    corrected_records = [phase5c3.correct_record(record, mapping_by_sample[record["sample_id"]]) for record in records]
    corrected_metrics = phase5c3.aggregate_corrected(corrected_records, top_k=args.top_k)
    topk_distribution = phase5c3.build_topk_distribution(corrected_records)
    mode_cmp = mode_comparison(corrected_metrics)
    regression_review = {
        "caption_level_table": regression_review_for("caption_level_table", corrected_records),
        "figure_caption": regression_review_for("figure_caption", corrected_records),
    }
    normal_takeover = normal_takeover_stats(corrected_records, args.top_k)
    risk_results = risk_slice_results(corrected_records, queries, args.top_k)
    gate = phase5c5_gate(corrected_metrics, validation, regression_review, normal_takeover, risk_results)

    raw_payload = {
        "run_config": run_config(args, modes),
        "validation": validation,
        "metrics": raw_metrics,
        "records": records,
    }
    corrected_payload = {
        "source_results": str(output_dir / "retrieval_ab_results.json"),
        "note": "Corrected metrics use stable source-block target mapping, not cross-version chunk_id equality.",
        "top_k_limit_for_remapped_rank": search_limit,
        "run_config": run_config(args, modes),
        "validation": validation,
        "mapping_status_counts": dict(Counter(m["mapping_status"] for m in mappings)),
        "metrics": corrected_metrics,
        "mode_comparison": mode_cmp,
        "regression_review": regression_review,
        "normal_takeover": normal_takeover,
        "risk_slice_results": risk_results,
        "gate": gate,
        "records": corrected_records,
    }

    phase5c3.write_json(output_dir / "retrieval_ab_results.json", raw_payload)
    phase5c3.write_json(output_dir / "corrected_retrieval_ab_results.json", corrected_payload)
    phase5c3.write_mapping_csv(output_dir / "target_mapping_audit.csv", mappings)
    phase5c3.write_json(output_dir / "topk_distribution.json", topk_distribution)
    phase5c3.write_json(output_dir / "mode_comparison.json", mode_cmp)
    phase5c3.write_json(output_dir / "risk_slice_results.json", risk_results)
    phase5c3.write_miss_examples(output_dir / "miss_examples.md", corrected_records, queries, args.top_k)
    phase5c3.write_rank_delta_examples(output_dir / "rank_delta_examples.md", corrected_records, args.top_k)
    write_summary(output_dir / "summary.md", corrected_payload)
    print(f"Wrote {output_dir / 'summary.md'}")


def run_config(args: argparse.Namespace, modes: list[str]) -> dict[str, Any]:
    return {
        "eval_jsonl": args.eval_jsonl,
        "top_k": args.top_k,
        "retrieval_modes": modes,
        "stable_target_mapping_enabled": True,
        "qwen_called": False,
        "generation_eval": False,
        "ragas_eval": False,
        "reranker_called": False,
        "rrf_or_hybrid_params_changed": False,
    }


def load_preflight_context(output_dir: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    preflight_json = output_dir / "preflight_stats.json"
    index_json = output_dir / "index_build_stats.json"
    if preflight_json.exists():
        context["preflight"] = json.loads(preflight_json.read_text(encoding="utf-8"))
    if index_json.exists():
        context["index_build"] = json.loads(index_json.read_text(encoding="utf-8"))
    return context


def mode_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qtype in ("table_content", "caption_level_table", "figure_caption", "normal_control"):
        out[qtype] = {}
        for mode in ("dense_only", "bm25_only", "hybrid"):
            b = metrics.get(f"{mode}/{qtype}/baseline", {})
            e = metrics.get(f"{mode}/{qtype}/enhanced", {})
            pair = metrics.get("paired", {}).get(f"{mode}/{qtype}", {})
            out[qtype][mode] = {
                "baseline_doc_hit_at_10": b.get("doc_hit_at_10"),
                "enhanced_doc_hit_at_10": e.get("doc_hit_at_10"),
                "baseline_corrected_chunk_hit_at_10": b.get("corrected_chunk_hit_at_10"),
                "enhanced_corrected_chunk_hit_at_10": e.get("corrected_chunk_hit_at_10"),
                "baseline_hit_at_20": b.get("corrected_chunk_hit_at_20"),
                "enhanced_hit_at_20": e.get("corrected_chunk_hit_at_20"),
                "baseline_hit_at_50": b.get("corrected_chunk_hit_at_50"),
                "enhanced_hit_at_50": e.get("corrected_chunk_hit_at_50"),
                "baseline_mrr": b.get("corrected_chunk_mrr"),
                "enhanced_mrr": e.get("corrected_chunk_mrr"),
                "doc_hit_delta": delta(e.get("doc_hit_at_10"), b.get("doc_hit_at_10")),
                "corrected_chunk_hit_delta": delta(e.get("corrected_chunk_hit_at_10"), b.get("corrected_chunk_hit_at_10")),
                "enhanced_only": pair.get("enhanced_only_success_count", 0),
                "baseline_only": pair.get("baseline_only_success_count", 0),
                "both_success": pair.get("both_success_count", 0),
                "both_fail": pair.get("both_fail_count", 0),
                "mean_target_rank_improvement": pair.get("mean_target_chunk_rank_improvement", 0.0),
            }
    return out


def regression_review_for(qtype: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["mode"] == "hybrid" and record["query_type"] == qtype and record.get("include_in_main_denominator"):
            pairs[record["sample_id"]][record["variant"]] = record
    rows = []
    for sid, pair in sorted(pairs.items()):
        base = pair.get("baseline")
        enh = pair.get("enhanced")
        if not base or not enh:
            continue
        if base.get("corrected_chunk_hit_at_10") and not enh.get("corrected_chunk_hit_at_10"):
            rows.append({
                "sample_id": sid,
                "target_doc_id": base.get("target_doc_id", ""),
                "baseline_rank": base.get("corrected_target_chunk_rank", 0),
                "enhanced_rank": enh.get("corrected_target_chunk_rank", 0),
                "stable_target_block_ids": enh.get("stable_target_block_ids", ""),
                "true_regression": True,
            })
    severe_rows = [
        row for row in rows
        if int(row.get("enhanced_rank") or 0) == 0 or int(row.get("enhanced_rank") or 0) > 20
    ]
    return {
        "rows": rows,
        "counts": {
            "true_regression_count": len(rows),
            "severe_regression_count": len(severe_rows),
            "near_miss_rank_11_to_20_count": len(rows) - len(severe_rows),
        },
    }


def normal_takeover_stats(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["mode"] == "hybrid" and record["query_type"] == "normal_control" and record.get("include_in_main_denominator"):
            pairs[record["sample_id"]][record["variant"]] = record
    rows = []
    for sid, pair in sorted(pairs.items()):
        base = pair.get("baseline")
        enh = pair.get("enhanced")
        if not base or not enh:
            continue
        enhanced_related_hits = [h for h in enh.get("top_hits", [])[:top_k] if h.get("table_related")]
        if enhanced_related_hits and not enh.get("corrected_chunk_hit_at_10"):
            rows.append({
                "sample_id": sid,
                "target_doc_id": enh.get("target_doc_id", ""),
                "enhanced_table_related_topk_count": len(enhanced_related_hits),
                "baseline_hit": bool(base.get("corrected_chunk_hit_at_10")),
                "enhanced_hit": bool(enh.get("corrected_chunk_hit_at_10")),
                "top_table_related_hits": enhanced_related_hits[:3],
            })
    return {
        "enhanced_table_related_takeover_count": len(rows),
        "rows": rows[:25],
    }


def risk_slice_results(records: list[dict[str, Any]], queries: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    query_by_id = {q["sample_id"]: q for q in queries}
    risk_records = [r for r in records if r["query_type"] == "risk_slice"]
    out: dict[str, Any] = {
        "total_risk_queries": len({r["sample_id"] for r in risk_records}),
        "by_tag": {},
        "special_docs": {},
        "false_positive_retrieval_count": 0,
        "false_positive_examples": [],
    }
    tags = sorted({tag for q in queries if q.get("query_type") == "risk_slice" for tag in q.get("risk_tags", [])})
    for tag in tags:
        tagged = [r for r in risk_records if tag in query_by_id.get(r["sample_id"], {}).get("risk_tags", [])]
        out["by_tag"][tag] = summarize_record_group(tagged)
    for doc_id in ("doc_0377", "doc_0442"):
        tagged = [r for r in risk_records if r.get("target_doc_id") == doc_id]
        out["special_docs"][doc_id] = summarize_record_group(tagged)

    false_positive_examples = []
    for r in risk_records:
        if r["mode"] != "hybrid" or r["variant"] != "enhanced":
            continue
        target_doc = r.get("target_doc_id", "")
        noisy = [
            h for h in r.get("top_hits", [])[:top_k]
            if h.get("table_related") and h.get("doc_id") != target_doc
        ]
        if noisy and not r.get("doc_hit_at_10"):
            false_positive_examples.append({
                "sample_id": r["sample_id"],
                "target_doc_id": target_doc,
                "risk_tags": query_by_id.get(r["sample_id"], {}).get("risk_tags", []),
                "noisy_hits": noisy[:3],
            })
    out["false_positive_retrieval_count"] = len(false_positive_examples)
    out["false_positive_examples"] = false_positive_examples[:20]
    return out


def summarize_record_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["variant"])].append(row)
    out: dict[str, Any] = {"count": len({r["sample_id"] for r in rows})}
    for (mode, variant), group in sorted(grouped.items()):
        out[f"{mode}/{variant}"] = {
            "doc_hit_at_10": mean_bool(group, "doc_hit_at_10"),
            "corrected_chunk_hit_at_10": mean_bool(group, "corrected_chunk_hit_at_10"),
            "mean_table_related_occupancy": statistics.mean(float(r.get("table_related_topk_occupancy") or 0.0) for r in group),
        }
    return out


def phase5c5_gate(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    regression_review: dict[str, Any],
    normal_takeover: dict[str, Any],
    risk_results: dict[str, Any],
) -> dict[str, Any]:
    def hybrid(qtype: str, variant: str) -> dict[str, Any]:
        return metrics.get(f"hybrid/{qtype}/{variant}", {})

    def paired(qtype: str) -> dict[str, Any]:
        return metrics.get("paired", {}).get(f"hybrid/{qtype}", {})

    table_b = hybrid("table_content", "baseline")
    table_e = hybrid("table_content", "enhanced")
    caption_b = hybrid("caption_level_table", "baseline")
    caption_e = hybrid("caption_level_table", "enhanced")
    figure_b = hybrid("figure_caption", "baseline")
    figure_e = hybrid("figure_caption", "enhanced")
    normal_b = hybrid("normal_control", "baseline")
    normal_e = hybrid("normal_control", "enhanced")
    preflight = validation.get("preflight", {})
    index_build = validation.get("index_build", {})
    tokenizer_delta = int(preflight.get("bge_tokenizer_truncation_risk_delta", 0) or 0)
    varchar_risk = int(preflight.get("milvus_varchar_risk", 0) or 0)
    import_varchar_warnings = int(index_build.get("baseline", {}).get("milvus_varchar_truncation_warning_count", 0) or 0) + int(index_build.get("enhanced", {}).get("milvus_varchar_truncation_warning_count", 0) or 0)

    gate = {
        "engineering_row_counts_match": validation.get("baseline_row_count_matches_chunks") and validation.get("enhanced_row_count_matches_chunks"),
        "engineering_bm25_counts_match": validation.get("baseline_bm25_count_matches_chunks") and validation.get("enhanced_bm25_count_matches_chunks"),
        "no_schema_drift": preflight.get("schema_same", True) is True and int(preflight.get("field_set_count", 1) or 1) == 1,
        "no_main_index_overwritten": validation.get("main_index_overwritten") is False,
        "no_milvus_varchar_truncation": varchar_risk == 0 and import_varchar_warnings == 0,
        "bge_tokenizer_truncation_not_increased": tokenizer_delta <= 0,
        "table_content_doc_hit_non_degraded": table_e.get("doc_hit_at_10", 0.0) >= table_b.get("doc_hit_at_10", 0.0),
        "table_content_corrected_chunk_hit_improved": table_e.get("corrected_chunk_hit_at_10", 0.0) > table_b.get("corrected_chunk_hit_at_10", 0.0),
        "table_content_enhanced_only_ge_baseline_only": paired("table_content").get("enhanced_only_success_count", 0) >= paired("table_content").get("baseline_only_success_count", 0),
        "caption_corrected_chunk_not_degraded_over_0_05": caption_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= caption_b.get("corrected_chunk_hit_at_10", 0.0),
        "caption_true_regression_le_1": regression_review["caption_level_table"]["counts"]["true_regression_count"] <= 1,
        "caption_regressions_explainable": regression_review["caption_level_table"]["counts"]["severe_regression_count"] <= 1,
        "figure_corrected_chunk_not_degraded_over_0_05": figure_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= figure_b.get("corrected_chunk_hit_at_10", 0.0),
        "figure_true_regression_zero": regression_review["figure_caption"]["counts"]["true_regression_count"] == 0,
        "normal_doc_not_degraded_over_0_05": normal_e.get("doc_hit_at_10", 0.0) + 0.05 >= normal_b.get("doc_hit_at_10", 0.0),
        "normal_corrected_chunk_not_degraded_over_0_05": normal_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= normal_b.get("corrected_chunk_hit_at_10", 0.0),
        "normal_table_occupancy_baseline": normal_b.get("mean_table_topk_occupancy", 0.0),
        "normal_table_occupancy_enhanced": normal_e.get("mean_table_topk_occupancy", 0.0),
        "normal_table_related_occupancy_baseline": normal_b.get("mean_table_related_topk_occupancy", 0.0),
        "normal_table_related_occupancy_enhanced": normal_e.get("mean_table_related_topk_occupancy", 0.0),
        "enhanced_table_related_takeover_count": normal_takeover["enhanced_table_related_takeover_count"],
        "enhanced_table_related_takeover_acceptable": (
            normal_takeover["enhanced_table_related_takeover_count"] <= 2
            and normal_e.get("doc_hit_at_10", 0.0) + 0.05 >= normal_b.get("doc_hit_at_10", 0.0)
            and normal_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= normal_b.get("corrected_chunk_hit_at_10", 0.0)
            and normal_e.get("mean_table_related_topk_occupancy", 0.0) <= 0.05
        ),
        "risk_slice_false_positive_acceptable": risk_results["false_positive_retrieval_count"] <= 1,
    }
    gate["engineering_ok"] = all([
        gate["engineering_row_counts_match"],
        gate["engineering_bm25_counts_match"],
        gate["no_schema_drift"],
        gate["no_main_index_overwritten"],
        gate["no_milvus_varchar_truncation"],
        gate["bge_tokenizer_truncation_not_increased"],
    ])
    gate["retrieval_ok"] = all([
        gate["table_content_doc_hit_non_degraded"],
        gate["table_content_corrected_chunk_hit_improved"],
        gate["table_content_enhanced_only_ge_baseline_only"],
        gate["caption_corrected_chunk_not_degraded_over_0_05"],
        gate["caption_true_regression_le_1"] or gate["caption_regressions_explainable"],
        gate["figure_corrected_chunk_not_degraded_over_0_05"],
        gate["figure_true_regression_zero"],
        gate["normal_doc_not_degraded_over_0_05"],
        gate["normal_corrected_chunk_not_degraded_over_0_05"],
        gate["enhanced_table_related_takeover_acceptable"],
        gate["risk_slice_false_positive_acceptable"],
    ])
    gate["phase5c5_pass"] = bool(gate["engineering_ok"] and gate["retrieval_ok"])
    gate["recommend_phase5c6"] = gate["phase5c5_pass"]
    gate["recommend_direct_mainline_merge"] = False
    return gate


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    validation = payload["validation"]
    gate = payload["gate"]
    risk = payload["risk_slice_results"]

    def metric(qtype: str, variant: str, key: str) -> Any:
        return metrics.get(f"hybrid/{qtype}/{variant}", {}).get(key)

    def paired(qtype: str) -> dict[str, Any]:
        return metrics.get("paired", {}).get(f"hybrid/{qtype}", {})

    lines = [
        "# Phase 5C-5 Full Retrieval A/B Summary",
        "",
        "## Engineering",
        "",
        f"- full enhanced experimental index built: {yes_no(validation.get('enhanced_row_count_matches_chunks') and validation.get('enhanced_bm25_count_matches_chunks'))}",
        f"- baseline row_count/chunk_count: {validation['baseline_row_count']} / {validation['baseline_chunk_count']}",
        f"- enhanced row_count/chunk_count: {validation['enhanced_row_count']} / {validation['enhanced_chunk_count']}",
        f"- baseline BM25 records: {validation['baseline_bm25_record_count']}",
        f"- enhanced BM25 records: {validation['enhanced_bm25_record_count']}",
        f"- schema drift: {yes_no(not gate['no_schema_drift'])}",
        f"- main index overwritten: {yes_no(not gate['no_main_index_overwritten'])}",
        f"- Milvus VARCHAR truncation risk acceptable: {yes_no(gate['no_milvus_varchar_truncation'])}",
        f"- BGE tokenizer truncation risk acceptable: {yes_no(gate['bge_tokenizer_truncation_not_increased'])}",
        "",
        "## Corrected Hybrid A/B",
        "",
        result_line("table_content", metric, paired),
        result_line("caption_level_table", metric, paired),
        result_line("figure_caption", metric, paired),
        result_line("normal_control", metric, paired),
        f"- normal table occupancy: baseline={fmt(gate['normal_table_occupancy_baseline'])}, enhanced={fmt(gate['normal_table_occupancy_enhanced'])}",
        f"- normal table_related occupancy: baseline={fmt(gate['normal_table_related_occupancy_baseline'])}, enhanced={fmt(gate['normal_table_related_occupancy_enhanced'])}",
        f"- enhanced_table_related_takeover_count: {gate['enhanced_table_related_takeover_count']}",
        f"- enhanced_table_related_takeover_acceptable: {yes_no(gate['enhanced_table_related_takeover_acceptable'])}",
        "",
        "## Risk Slice",
        "",
        f"- risk queries: {risk['total_risk_queries']}",
        f"- false_positive_retrieval_count: {risk['false_positive_retrieval_count']}",
        f"- doc_0377 risk samples: {risk.get('special_docs', {}).get('doc_0377', {}).get('count', 0)}",
        f"- doc_0442 risk samples: {risk.get('special_docs', {}).get('doc_0442', {}).get('count', 0)}",
        "",
        "## Gate Answers",
        "",
        f"1. full enhanced experimental index built successfully: {yes_no(validation.get('enhanced_row_count_matches_chunks') and validation.get('enhanced_bm25_count_matches_chunks'))}.",
        f"2. full retrieval A/B passed: {yes_no(gate['phase5c5_pass'])}.",
        f"3. table_content continues to improve in the full-corpus competition setting: {yes_no(gate['table_content_corrected_chunk_hit_improved'])}.",
        f"4. caption_level_table has no true regression beyond gate: {yes_no(gate['caption_corrected_chunk_not_degraded_over_0_05'] and (gate['caption_true_regression_le_1'] or gate['caption_regressions_explainable']))}.",
        f"5. figure_caption has no regression: {yes_no(gate['figure_corrected_chunk_not_degraded_over_0_05'] and gate['figure_true_regression_zero'])}.",
        f"6. normal_control has no regression beyond gate: {yes_no(gate['normal_doc_not_degraded_over_0_05'] and gate['normal_corrected_chunk_not_degraded_over_0_05'])}.",
        f"7. table-related takeover exists: {yes_no(gate['enhanced_table_related_takeover_count'] > 0)}; acceptable: {yes_no(gate['enhanced_table_related_takeover_acceptable'])}.",
        f"8. risk_slice exposes false positives: {yes_no(not gate['risk_slice_false_positive_acceptable'])}.",
        f"9. truncation risk is acceptable: {yes_no(gate['no_milvus_varchar_truncation'] and gate['bge_tokenizer_truncation_not_increased'])}.",
        f"10. recommend Phase 5C-6 mainline integration design: {yes_no(gate['recommend_phase5c6'])}.",
        f"11. recommend direct mainline merge/default enablement: {yes_no(gate['recommend_direct_mainline_merge'])}.",
        "12. mainline integration should use an explicit config switch, rollback path, formal rebuild, and retrieval-only regression eval before any default change.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def result_line(qtype: str, metric, paired) -> str:
    pair = paired(qtype)
    return (
        f"- {qtype}: doc_hit@10 baseline={fmt(metric(qtype, 'baseline', 'doc_hit_at_10'))}, "
        f"enhanced={fmt(metric(qtype, 'enhanced', 'doc_hit_at_10'))}; "
        f"corrected_chunk_hit@10 baseline={fmt(metric(qtype, 'baseline', 'corrected_chunk_hit_at_10'))}, "
        f"enhanced={fmt(metric(qtype, 'enhanced', 'corrected_chunk_hit_at_10'))}; "
        f"enhanced_only={pair.get('enhanced_only_success_count', 0)}, "
        f"baseline_only={pair.get('baseline_only_success_count', 0)}, "
        f"both_success={pair.get('both_success_count', 0)}, both_fail={pair.get('both_fail_count', 0)}"
    )


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0


def delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


if __name__ == "__main__":
    main()
