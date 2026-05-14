#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from pymilvus import MilvusClient

from src.synbio_rag.domain.config import KnowledgeBaseConfig, RetrievalConfig
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5C-2 retrieval-only A/B evaluation.")
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
    parser.add_argument("--model_path", default="models/BAAI/bge-m3")
    parser.add_argument("--embedding_max_length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = list(iter_jsonl(Path(args.eval_jsonl)))
    baseline_chunks = list(iter_jsonl(Path(args.baseline_chunks_jsonl)))
    enhanced_chunks = list(iter_jsonl(Path(args.enhanced_chunks_jsonl)))
    modes = [m.strip() for m in args.retrieval_modes.split(",") if m.strip()]
    search_limit = max(50, args.top_k)

    embedder = BGEM3Embedder(
        model_path=args.model_path,
        dim=1024,
        max_length=args.embedding_max_length,
    )
    baseline = build_retrieval_stack(
        chunks_jsonl=args.baseline_chunks_jsonl,
        milvus_uri=args.baseline_milvus_uri,
        collection=args.baseline_collection,
        bm25_path=args.baseline_bm25_path,
        embedder=embedder,
    )
    enhanced = build_retrieval_stack(
        chunks_jsonl=args.enhanced_chunks_jsonl,
        milvus_uri=args.enhanced_milvus_uri,
        collection=args.enhanced_collection,
        bm25_path=args.enhanced_bm25_path,
        embedder=embedder,
    )

    validation = {
        "baseline_chunk_count": len(baseline_chunks),
        "enhanced_chunk_count": len(enhanced_chunks),
        "baseline_row_count": row_count(args.baseline_milvus_uri, args.baseline_collection),
        "enhanced_row_count": row_count(args.enhanced_milvus_uri, args.enhanced_collection),
        "baseline_bm25_record_count": len(baseline["bm25"]._records),
        "enhanced_bm25_record_count": len(enhanced["bm25"]._records),
        "baseline_row_count_matches_chunks": False,
        "enhanced_row_count_matches_chunks": False,
        "baseline_bm25_count_matches_chunks": len(baseline["bm25"]._records) == len(baseline_chunks),
        "enhanced_bm25_count_matches_chunks": len(enhanced["bm25"]._records) == len(enhanced_chunks),
    }
    validation["baseline_row_count_matches_chunks"] = validation["baseline_row_count"] == len(baseline_chunks)
    validation["enhanced_row_count_matches_chunks"] = validation["enhanced_row_count"] == len(enhanced_chunks)

    records: list[dict[str, Any]] = []
    for query in queries:
        for mode in modes:
            records.append(evaluate_one(query, mode, "baseline", baseline, search_limit, args.top_k))
            records.append(evaluate_one(query, mode, "enhanced", enhanced, search_limit, args.top_k))

    metrics = aggregate(records, top_k=args.top_k)
    topk_distribution = build_topk_distribution(records, top_k=args.top_k)
    mode_comparison = build_mode_comparison(metrics)
    gate = evaluate_gate(metrics, validation)

    payload = {
        "run_config": {
            "eval_jsonl": args.eval_jsonl,
            "top_k": args.top_k,
            "retrieval_modes": modes,
            "qwen_called": False,
            "generation_eval": False,
            "reranker_called": False,
        },
        "validation": validation,
        "metrics": metrics,
        "mode_comparison": mode_comparison,
        "gate": gate,
        "records": records,
    }
    write_json(output_dir / "retrieval_ab_results.json", payload)
    write_json(output_dir / "topk_distribution.json", topk_distribution)
    write_json(output_dir / "mode_comparison.json", mode_comparison)
    write_miss_examples(output_dir / "miss_examples.md", records, queries, args.top_k)
    write_rank_delta_examples(output_dir / "rank_delta_examples.md", records, args.top_k)
    write_summary(output_dir / "summary.md", payload)
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'retrieval_ab_results.json'}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_retrieval_stack(
    chunks_jsonl: str,
    milvus_uri: str,
    collection: str,
    bm25_path: str,
    embedder: BGEM3Embedder,
) -> dict[str, Any]:
    retrieval = RetrievalConfig(
        milvus_uri=milvus_uri,
        collection_name=collection,
        bm25_cache_path=bm25_path,
        search_limit=50,
        dense_limit=50,
        bm25_limit=50,
    )
    kb = KnowledgeBaseConfig(chunk_jsonl=chunks_jsonl)
    dense = MilvusRetriever(retrieval, embedder)
    bm25 = BM25Retriever(retrieval, kb, milvus_client=dense.client)
    start = time.time()
    bm25._ensure_index()
    return {
        "config": retrieval,
        "kb": kb,
        "dense": dense,
        "bm25": bm25,
        "hybrid": HybridRetriever(retrieval, dense, bm25),
        "bm25_build_elapsed_sec": time.time() - start,
    }


def row_count(uri: str, collection: str) -> int:
    client = MilvusClient(uri)
    stats = client.get_collection_stats(collection)
    return int(stats.get("row_count", 0))


def evaluate_one(
    query: dict[str, Any],
    mode: str,
    variant: str,
    stack: dict[str, Any],
    search_limit: int,
    top_k: int,
) -> dict[str, Any]:
    question = str(query.get("query") or "")
    if mode == "dense_only":
        hits = stack["dense"].search(question, limit=search_limit, filters=None)
    elif mode == "bm25_only":
        hits = stack["bm25"].search(question, limit=search_limit, filters=None)
    elif mode == "hybrid":
        hits = stack["hybrid"].search(question, limit=search_limit, filters=None, analysis=None)
    else:
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    target_chunk_id = str(query.get(f"target_chunk_id_{variant}") or "")
    target_doc_id = str(query.get("target_doc_id") or "")
    target_rank = best_chunk_rank(hits, target_chunk_id)
    doc_rank = best_doc_rank(hits, target_doc_id)
    top_hits = [serialize_hit(hit, rank) for rank, hit in enumerate(hits[:top_k], start=1)]
    return {
        "sample_id": query.get("sample_id", ""),
        "query_type": query.get("query_type", ""),
        "include_in_main_denominator": bool(query.get("include_in_main_denominator", True)),
        "query": question,
        "mode": mode,
        "variant": variant,
        "target_doc_id": target_doc_id,
        "target_chunk_id": target_chunk_id,
        "target_doc_rank": doc_rank,
        "target_chunk_rank": target_rank,
        "doc_hit_at_10": 0 < doc_rank <= 10,
        "doc_hit_at_20": 0 < doc_rank <= 20,
        "doc_hit_at_50": 0 < doc_rank <= 50,
        "chunk_hit_at_10": bool(target_chunk_id) and 0 < target_rank <= 10,
        "chunk_hit_at_20": bool(target_chunk_id) and 0 < target_rank <= 20,
        "chunk_hit_at_50": bool(target_chunk_id) and 0 < target_rank <= 50,
        "table_topk_occupancy": table_occupancy(hits[:top_k]),
        "top_hits": top_hits,
    }


def best_chunk_rank(hits: list[Any], target_chunk_id: str) -> int:
    if not target_chunk_id:
        return 0
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk_id == target_chunk_id:
            return rank
    return 0


def best_doc_rank(hits: list[Any], target_doc_id: str) -> int:
    if not target_doc_id:
        return 0
    for rank, hit in enumerate(hits, start=1):
        if hit.doc_id == target_doc_id:
            return rank
    return 0


def serialize_hit(hit: Any, rank: int) -> dict[str, Any]:
    metadata = dict(hit.metadata or {})
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section": hit.section,
        "content_kind": metadata.get("content_kind", ""),
        "contains_table_text": bool(metadata.get("contains_table_text")),
        "contains_table_caption": bool(metadata.get("contains_table_caption")),
        "block_types": metadata.get("block_types") or [],
        "evidence_types": metadata.get("evidence_types") or [],
        "score": hit.fusion_score or hit.vector_score or hit.bm25_score or 0.0,
    }


def table_occupancy(hits: list[Any]) -> float:
    if not hits:
        return 0.0
    table_hits = 0
    for hit in hits:
        meta = dict(hit.metadata or {})
        if meta.get("contains_table_text") or meta.get("contains_table_caption"):
            table_hits += 1
            continue
        evidence_types = set(meta.get("evidence_types") or meta.get("block_types") or [])
        if "table_caption" in evidence_types or "table_text" in evidence_types:
            table_hits += 1
    return table_hits / len(hits)


def aggregate(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("include_in_main_denominator"):
            by_key[(record["mode"], record["query_type"], record["variant"])].append(record)

    metrics: dict[str, Any] = {}
    for (mode, query_type, variant), rows in sorted(by_key.items()):
        key = f"{mode}/{query_type}/{variant}"
        metrics[key] = basic_metrics(rows)

    paired: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in records}):
        for query_type in sorted({r["query_type"] for r in records if r.get("include_in_main_denominator")}):
            base = [r for r in records if r["mode"] == mode and r["query_type"] == query_type and r["variant"] == "baseline" and r.get("include_in_main_denominator")]
            enh = [r for r in records if r["mode"] == mode and r["query_type"] == query_type and r["variant"] == "enhanced" and r.get("include_in_main_denominator")]
            paired[f"{mode}/{query_type}"] = paired_metrics(base, enh, top_k)
    metrics["paired"] = paired
    return metrics


def basic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"count": 0}
    return {
        "count": n,
        "doc_hit_at_10": mean_bool(rows, "doc_hit_at_10"),
        "doc_hit_at_20": mean_bool(rows, "doc_hit_at_20"),
        "doc_hit_at_50": mean_bool(rows, "doc_hit_at_50"),
        "chunk_hit_at_10": mean_bool(rows, "chunk_hit_at_10"),
        "chunk_hit_at_20": mean_bool(rows, "chunk_hit_at_20"),
        "chunk_hit_at_50": mean_bool(rows, "chunk_hit_at_50"),
        "mean_doc_rank_found": mean_rank(rows, "target_doc_rank"),
        "mean_chunk_rank_found": mean_rank(rows, "target_chunk_rank"),
        "mean_table_topk_occupancy": statistics.mean(float(r["table_topk_occupancy"]) for r in rows),
    }


def paired_metrics(base: list[dict[str, Any]], enh: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    b_by_id = {r["sample_id"]: r for r in base}
    e_by_id = {r["sample_id"]: r for r in enh}
    ids = sorted(set(b_by_id) & set(e_by_id))
    enhanced_only = baseline_only = both_success = both_fail = 0
    rank_deltas: list[int] = []
    doc_rank_deltas: list[int] = []
    for sid in ids:
        b = b_by_id[sid]
        e = e_by_id[sid]
        b_hit = bool(b["chunk_hit_at_10"])
        e_hit = bool(e["chunk_hit_at_10"])
        if e_hit and not b_hit:
            enhanced_only += 1
        elif b_hit and not e_hit:
            baseline_only += 1
        elif b_hit and e_hit:
            both_success += 1
        else:
            both_fail += 1
        rank_deltas.append(rank_for_delta(b["target_chunk_rank"], top_k) - rank_for_delta(e["target_chunk_rank"], top_k))
        doc_rank_deltas.append(rank_for_delta(b["target_doc_rank"], top_k) - rank_for_delta(e["target_doc_rank"], top_k))
    return {
        "count": len(ids),
        "enhanced_only_success_count": enhanced_only,
        "baseline_only_success_count": baseline_only,
        "both_success_count": both_success,
        "both_fail_count": both_fail,
        "mean_target_chunk_rank_improvement": statistics.mean(rank_deltas) if rank_deltas else 0.0,
        "mean_doc_rank_improvement": statistics.mean(doc_rank_deltas) if doc_rank_deltas else 0.0,
    }


def rank_for_delta(rank: int, top_k: int) -> int:
    return rank if rank > 0 else top_k + 1


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def mean_rank(rows: list[dict[str, Any]], key: str) -> float | None:
    ranks = [int(r[key]) for r in rows if int(r.get(key) or 0) > 0]
    return statistics.mean(ranks) if ranks else None


def build_topk_distribution(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    dist: dict[str, Any] = {}
    for record in records:
        if record["variant"] != "enhanced":
            continue
        key = f"{record['mode']}/{record['query_type']}"
        item = dist.setdefault(key, {"count": 0, "table_topk_occupancy_values": [], "top_doc_ids": Counter()})
        item["count"] += 1
        item["table_topk_occupancy_values"].append(record["table_topk_occupancy"])
        for hit in record["top_hits"]:
            item["top_doc_ids"][hit["doc_id"]] += 1
    for item in dist.values():
        values = item.pop("table_topk_occupancy_values")
        counter = item["top_doc_ids"]
        item["mean_table_topk_occupancy"] = statistics.mean(values) if values else 0.0
        item["top_doc_ids"] = counter.most_common(10)
    return dist


def build_mode_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for query_type in ("table_content", "caption_level_table", "normal_control"):
        comparison[query_type] = {}
        for mode in ("dense_only", "bm25_only", "hybrid"):
            b = metrics.get(f"{mode}/{query_type}/baseline", {})
            e = metrics.get(f"{mode}/{query_type}/enhanced", {})
            comparison[query_type][mode] = {
                "baseline_doc_hit_at_10": b.get("doc_hit_at_10"),
                "enhanced_doc_hit_at_10": e.get("doc_hit_at_10"),
                "baseline_chunk_hit_at_10": b.get("chunk_hit_at_10"),
                "enhanced_chunk_hit_at_10": e.get("chunk_hit_at_10"),
                "doc_hit_delta": none_delta(e.get("doc_hit_at_10"), b.get("doc_hit_at_10")),
                "chunk_hit_delta": none_delta(e.get("chunk_hit_at_10"), b.get("chunk_hit_at_10")),
            }
    return comparison


def none_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def evaluate_gate(metrics: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    hybrid = lambda q, v: metrics.get(f"hybrid/{q}/{v}", {})
    paired = metrics.get("paired", {})
    table_b = hybrid("table_content", "baseline")
    table_e = hybrid("table_content", "enhanced")
    caption_b = hybrid("caption_level_table", "baseline")
    caption_e = hybrid("caption_level_table", "enhanced")
    normal_b = hybrid("normal_control", "baseline")
    normal_e = hybrid("normal_control", "enhanced")
    table_pair = paired.get("hybrid/table_content", {})
    normal_occupancy_baseline = normal_b.get("mean_table_topk_occupancy", 1.0)
    normal_occupancy_enhanced = normal_e.get("mean_table_topk_occupancy", 1.0)

    engineering_ok = all([
        validation["baseline_row_count_matches_chunks"],
        validation["enhanced_row_count_matches_chunks"],
        validation["baseline_bm25_count_matches_chunks"],
        validation["enhanced_bm25_count_matches_chunks"],
    ])
    return {
        "engineering_ok": engineering_ok,
        "table_content_doc_hit_non_degraded": table_e.get("doc_hit_at_10", 0.0) >= table_b.get("doc_hit_at_10", 0.0),
        "table_content_chunk_hit_improved": table_e.get("chunk_hit_at_10", 0.0) > table_b.get("chunk_hit_at_10", 0.0),
        "table_content_enhanced_only_ge_baseline_only": table_pair.get("enhanced_only_success_count", 0) >= table_pair.get("baseline_only_success_count", 0),
        "caption_doc_not_degraded_over_0_05": caption_e.get("doc_hit_at_10", 0.0) + 0.05 >= caption_b.get("doc_hit_at_10", 0.0),
        "caption_chunk_not_degraded_over_0_05": caption_e.get("chunk_hit_at_10", 0.0) + 0.05 >= caption_b.get("chunk_hit_at_10", 0.0),
        "normal_doc_not_degraded_over_0_05": normal_e.get("doc_hit_at_10", 0.0) + 0.05 >= normal_b.get("doc_hit_at_10", 0.0),
        "normal_table_occupancy_threshold_ok": normal_occupancy_enhanced <= 0.15,
        "normal_table_occupancy_non_increased": normal_occupancy_enhanced <= normal_occupancy_baseline,
        "normal_table_occupancy_ok": normal_occupancy_enhanced <= 0.15,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_miss_examples(path: Path, records: list[dict[str, Any]], queries: list[dict[str, Any]], top_k: int) -> None:
    query_by_id = {q["sample_id"]: q for q in queries}
    misses = [
        r for r in records
        if r["variant"] == "enhanced" and r["mode"] == "hybrid" and r.get("include_in_main_denominator") and not r["chunk_hit_at_10"]
    ][:20]
    lines = ["# Phase 5C-2 Miss Examples", ""]
    for r in misses:
        q = query_by_id.get(r["sample_id"], {})
        lines.extend([
            f"## {r['sample_id']} ({r['query_type']})",
            "",
            f"- query: {r['query']}",
            f"- target_doc_id: {r['target_doc_id']}",
            f"- target_chunk_id: {r['target_chunk_id']}",
            f"- target_doc_rank: {r['target_doc_rank']}",
            f"- target_chunk_rank: {r['target_chunk_rank']}",
            f"- anchor_terms: {q.get('anchor_terms', [])}",
            f"- top{top_k}: {[(h['rank'], h['doc_id'], h['chunk_id']) for h in r['top_hits']]}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rank_delta_examples(path: Path, records: list[dict[str, Any]], top_k: int) -> None:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        if r["mode"] == "hybrid" and r.get("include_in_main_denominator"):
            grouped[(r["sample_id"], r["query_type"])][r["variant"]] = r
    deltas = []
    for (sid, qtype), pair in grouped.items():
        if "baseline" in pair and "enhanced" in pair:
            b = pair["baseline"]
            e = pair["enhanced"]
            delta = rank_for_delta(b["target_chunk_rank"], top_k) - rank_for_delta(e["target_chunk_rank"], top_k)
            deltas.append((delta, sid, qtype, b, e))
    deltas.sort(key=lambda item: item[0], reverse=True)
    lines = ["# Phase 5C-2 Rank Delta Examples", "", "## Largest Improvements", ""]
    for delta, sid, qtype, b, e in deltas[:12]:
        lines.append(f"- {sid} ({qtype}): chunk rank baseline={b['target_chunk_rank'] or 'miss'} enhanced={e['target_chunk_rank'] or 'miss'} delta={delta}")
    lines.extend(["", "## Largest Regressions", ""])
    for delta, sid, qtype, b, e in reversed(deltas[-12:]):
        lines.append(f"- {sid} ({qtype}): chunk rank baseline={b['target_chunk_rank'] or 'miss'} enhanced={e['target_chunk_rank'] or 'miss'} delta={delta}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    validation = payload["validation"]
    gate = payload["gate"]

    def row(query_type: str, metric: str) -> str:
        b = metrics.get(f"hybrid/{query_type}/baseline", {}).get(metric, 0.0)
        e = metrics.get(f"hybrid/{query_type}/enhanced", {}).get(metric, 0.0)
        return f"baseline={fmt(b)}, enhanced={fmt(e)}, delta={fmt(float(e or 0) - float(b or 0))}"

    def paired_line(query_type: str) -> str:
        p = metrics.get("paired", {}).get(f"hybrid/{query_type}", {})
        return (
            f"enhanced_only={p.get('enhanced_only_success_count', 0)}, "
            f"baseline_only={p.get('baseline_only_success_count', 0)}, "
            f"both_success={p.get('both_success_count', 0)}, both_fail={p.get('both_fail_count', 0)}"
        )

    lines = [
        "# Phase 5C-2 Table Retrieval A/B Summary",
        "",
        "## Engineering Checks",
        "",
        f"- baseline row_count/chunk_count: {validation['baseline_row_count']} / {validation['baseline_chunk_count']}",
        f"- enhanced row_count/chunk_count: {validation['enhanced_row_count']} / {validation['enhanced_chunk_count']}",
        f"- baseline BM25 records: {validation['baseline_bm25_record_count']}",
        f"- enhanced BM25 records: {validation['enhanced_bm25_record_count']}",
        f"- engineering_ok: {str(gate['engineering_ok']).lower()}",
        f"- qwen_called: {str(payload['run_config']['qwen_called']).lower()}",
        f"- generation_eval: {str(payload['run_config']['generation_eval']).lower()}",
        "",
        "## Hybrid Main Results",
        "",
        f"- table_content doc_hit@10: {row('table_content', 'doc_hit_at_10')}",
        f"- table_content target_chunk_hit@10: {row('table_content', 'chunk_hit_at_10')}",
        f"- table_content hit split: {paired_line('table_content')}",
        f"- caption_level_table doc_hit@10: {row('caption_level_table', 'doc_hit_at_10')}",
        f"- caption_level_table chunk_hit@10: {row('caption_level_table', 'chunk_hit_at_10')}",
        f"- normal_control doc_hit@10: {row('normal_control', 'doc_hit_at_10')}",
        f"- normal_control chunk_hit@10: {row('normal_control', 'chunk_hit_at_10')}",
        f"- normal_control enhanced table/top-k occupancy: {fmt(metrics.get('hybrid/normal_control/enhanced', {}).get('mean_table_topk_occupancy', 0.0))}",
        "",
        "## Required Answers",
        "",
        f"1. Phase 5C-1 enhanced table_related retrieval benefit: {yes_no(gate['table_content_chunk_hit_improved'])}.",
        f"2. Caption-level table retrieval harmed: {yes_no(not (gate['caption_doc_not_degraded_over_0_05'] and gate['caption_chunk_not_degraded_over_0_05']))}.",
        f"3. Normal-control retrieval harmed: {yes_no(not gate['normal_doc_not_degraded_over_0_05'])}.",
        f"4. Table-related takeover on normal queries: {takeover_answer(metrics, gate)}.",
        f"5. Main source of benefit: {benefit_source(payload['mode_comparison'] if 'mode_comparison' in payload else {})}.",
        f"6. Recommend Phase 5C-3: {yes_no(recommend_5c3(gate))}.",
        f"7. Recommend expanding to more docs: {yes_no(recommend_5c3(gate))}.",
        f"8. Need heuristic adjustment: {yes_no(not gate['normal_table_occupancy_ok'] or not gate['caption_chunk_not_degraded_over_0_05'])}.",
        "9. Need OCR: no.",
        f"10. Phase 5C can serve as stronger table-content pilot baseline than Phase 4: {yes_no(gate['table_content_chunk_hit_improved'] and gate['engineering_ok'])}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def benefit_source(mode_comparison: dict[str, Any]) -> str:
    table = mode_comparison.get("table_content", {})
    deltas = {
        mode: (vals.get("chunk_hit_delta") or 0.0)
        for mode, vals in table.items()
    }
    if not deltas:
        return "n/a"
    best_mode, best_delta = max(deltas.items(), key=lambda item: item[1])
    if best_delta <= 0:
        return "no positive target-chunk delta observed"
    return best_mode


def takeover_answer(metrics: dict[str, Any], gate: dict[str, Any]) -> str:
    baseline = metrics.get("hybrid/normal_control/baseline", {}).get("mean_table_topk_occupancy", 0.0)
    enhanced = metrics.get("hybrid/normal_control/enhanced", {}).get("mean_table_topk_occupancy", 0.0)
    if gate.get("normal_table_occupancy_non_increased") and not gate.get("normal_table_occupancy_threshold_ok"):
        return f"no obvious relative takeover, but threshold failed ({fmt(enhanced)} > 0.150; baseline {fmt(baseline)})"
    if gate.get("normal_table_occupancy_ok"):
        return "no"
    return f"yes ({fmt(enhanced)} > 0.150)"


def recommend_5c3(gate: dict[str, Any]) -> bool:
    return all([
        gate["engineering_ok"],
        gate["table_content_doc_hit_non_degraded"],
        gate["table_content_enhanced_only_ge_baseline_only"],
        gate["caption_doc_not_degraded_over_0_05"],
        gate["caption_chunk_not_degraded_over_0_05"],
        gate["normal_doc_not_degraded_over_0_05"],
        gate["normal_table_occupancy_ok"],
    ])


if __name__ == "__main__":
    main()
