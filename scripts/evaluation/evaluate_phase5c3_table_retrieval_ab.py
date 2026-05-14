#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Phase 5C-3 retrieval-only A/B evaluation with stable target mapping.")
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
    baseline_index = build_chunk_index(baseline_chunks)
    enhanced_index = build_chunk_index(enhanced_chunks)
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
    }
    validation["baseline_row_count_matches_chunks"] = validation["baseline_row_count"] == len(baseline_chunks)
    validation["enhanced_row_count_matches_chunks"] = validation["enhanced_row_count"] == len(enhanced_chunks)
    validation["baseline_bm25_count_matches_chunks"] = validation["baseline_bm25_record_count"] == len(baseline_chunks)
    validation["enhanced_bm25_count_matches_chunks"] = validation["enhanced_bm25_record_count"] == len(enhanced_chunks)

    records: list[dict[str, Any]] = []
    for query in queries:
        for mode in modes:
            records.append(evaluate_one(query, mode, "baseline", baseline, baseline_index, search_limit, args.top_k))
            records.append(evaluate_one(query, mode, "enhanced", enhanced, enhanced_index, search_limit, args.top_k))

    raw_metrics = aggregate_raw(records, top_k=args.top_k)
    mappings = [build_mapping(query, baseline_index, enhanced_index) for query in queries]
    mapping_by_sample = {m["sample_id"]: m for m in mappings}
    corrected_records = [correct_record(record, mapping_by_sample[record["sample_id"]]) for record in records]
    corrected_metrics = aggregate_corrected(corrected_records, top_k=args.top_k)
    topk_distribution = build_topk_distribution(corrected_records)
    mode_cmp = mode_comparison(corrected_metrics)
    caption_review = caption_regression_review(records, corrected_records, mapping_by_sample)
    gate = corrected_gate(corrected_metrics, validation, caption_review)

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
        "caption_regression_review": caption_review,
        "gate": gate,
        "records": corrected_records,
    }

    write_json(output_dir / "retrieval_ab_results.json", raw_payload)
    write_json(output_dir / "corrected_retrieval_ab_results.json", corrected_payload)
    write_mapping_csv(output_dir / "target_mapping_audit.csv", mappings)
    write_json(output_dir / "topk_distribution.json", topk_distribution)
    write_json(output_dir / "mode_comparison.json", mode_cmp)
    write_miss_examples(output_dir / "miss_examples.md", corrected_records, queries, args.top_k)
    write_rank_delta_examples(output_dir / "rank_delta_examples.md", corrected_records, args.top_k)
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
        "reranker_called": False,
    }


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


def build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_doc_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        by_doc[doc_id].append(chunk)
        for block_id in chunk_block_ids(chunk):
            by_doc_block[(doc_id, block_id)].append(chunk)
    return {"by_id": by_id, "by_doc": by_doc, "by_doc_block": by_doc_block}


def evaluate_one(
    query: dict[str, Any],
    mode: str,
    variant: str,
    stack: dict[str, Any],
    chunk_index: dict[str, Any],
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
    top_hits = [serialize_hit(hit, rank, chunk_index) for rank, hit in enumerate(hits[:search_limit], start=1)]
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
        "table_topk_occupancy": table_occupancy(top_hits[:top_k]),
        "table_related_topk_occupancy": table_related_occupancy(top_hits[:top_k]),
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


def serialize_hit(hit: Any, rank: int, chunk_index: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(hit.metadata or {})
    chunk = chunk_index["by_id"].get(hit.chunk_id, {})
    source_related = chunk_has_table_related(chunk)
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section": hit.section,
        "content_kind": metadata.get("content_kind", ""),
        "contains_table_text": bool(metadata.get("contains_table_text")),
        "contains_table_caption": bool(metadata.get("contains_table_caption")),
        "table_related": source_related,
        "block_types": metadata.get("block_types") or chunk.get("block_types") or [],
        "evidence_types": metadata.get("evidence_types") or chunk.get("evidence_types") or [],
        "score": hit.fusion_score or hit.vector_score or hit.bm25_score or 0.0,
    }


def chunk_has_table_related(chunk: dict[str, Any]) -> bool:
    return any(
        isinstance(meta, dict) and meta.get("table_related") is True
        for meta in chunk.get("source_block_metadata", []) or []
    )


def table_occupancy(hits: list[dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    table_hits = 0
    for hit in hits:
        if hit.get("contains_table_text") or hit.get("contains_table_caption"):
            table_hits += 1
            continue
        evidence_types = set(hit.get("evidence_types") or hit.get("block_types") or [])
        if "table_caption" in evidence_types or "table_text" in evidence_types:
            table_hits += 1
    return table_hits / len(hits)


def table_related_occupancy(hits: list[dict[str, Any]]) -> float:
    return sum(1 for hit in hits if hit.get("table_related")) / len(hits) if hits else 0.0


def build_mapping(query: dict[str, Any], baseline_index: dict[str, Any], enhanced_index: dict[str, Any]) -> dict[str, Any]:
    qtype = str(query.get("query_type", ""))
    doc_id = str(query.get("target_doc_id", ""))
    original_baseline = str(query.get("target_chunk_id_baseline") or "")
    original_enhanced = str(query.get("target_chunk_id_enhanced") or "")
    baseline_chunk = baseline_index["by_id"].get(original_baseline)
    stable_ids = normalize_stable_ids(query.get("stable_target_block_ids"))
    notes: list[str] = []

    if not stable_ids:
        if qtype == "caption_level_table" and baseline_chunk:
            stable_ids = table_caption_block_ids(baseline_chunk) or chunk_block_ids(baseline_chunk)
            notes.append("derived from baseline table-caption chunk")
        elif qtype == "table_content":
            stable_ids = [x for x in [query.get("target_associated_block_id"), query.get("target_caption_block_id")] if x]
            notes.append("derived from table_content associated/caption block ids")
        elif qtype == "normal_control" and baseline_chunk:
            stable_ids = non_table_block_ids(baseline_chunk) or chunk_block_ids(baseline_chunk)
            notes.append("derived from baseline normal source blocks")
    else:
        notes.append("used explicit stable_target_block_ids")

    corrected_baseline, baseline_candidates = corrected_target(qtype, "baseline", doc_id, original_baseline, stable_ids, query, baseline_index)
    corrected_enhanced, enhanced_candidates = corrected_target(qtype, "enhanced", doc_id, original_enhanced, stable_ids, query, enhanced_index)
    if len(baseline_candidates) > 1:
        notes.append(f"baseline candidates={','.join(c['chunk_id'] for c in baseline_candidates)}")
    if len(enhanced_candidates) > 1:
        notes.append(f"enhanced candidates={','.join(c['chunk_id'] for c in enhanced_candidates)}")

    return {
        "sample_id": query.get("sample_id", ""),
        "query_type": qtype,
        "target_doc_id": doc_id,
        "original_baseline_target_chunk_id": original_baseline,
        "original_enhanced_target_chunk_id": original_enhanced,
        "stable_target_block_ids": ";".join(stable_ids),
        "corrected_baseline_target_chunk_id": corrected_baseline,
        "corrected_enhanced_target_chunk_id": corrected_enhanced,
        "mapping_status": mapping_status_for(stable_ids, corrected_baseline, corrected_enhanced, baseline_candidates, enhanced_candidates, original_baseline, original_enhanced),
        "mapping_notes": "; ".join(notes),
    }


def normalize_stable_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [item for item in value.split(";") if item]
    return []


def corrected_target(
    qtype: str,
    variant: str,
    doc_id: str,
    original_chunk_id: str,
    stable_ids: list[str],
    query: dict[str, Any],
    chunk_index: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if not stable_ids:
        return original_chunk_id, []
    preferred_ids = list(stable_ids)
    if qtype == "table_content" and variant == "enhanced":
        assoc = str(query.get("target_associated_block_id") or "")
        preferred_ids = [assoc] if assoc else stable_ids
    elif qtype == "table_content" and variant == "baseline":
        caption = str(query.get("target_caption_block_id") or "")
        preferred_ids = [caption] if caption else stable_ids

    candidates = candidate_chunks(chunk_index, doc_id, preferred_ids)
    if not candidates and preferred_ids != stable_ids:
        candidates = candidate_chunks(chunk_index, doc_id, stable_ids)
    if not candidates:
        return original_chunk_id, []
    return choose_candidate(candidates, qtype, variant, preferred_ids)["chunk_id"], candidates


def candidate_chunks(chunk_index: dict[str, Any], doc_id: str, block_ids: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for block_id in block_ids:
        for chunk in chunk_index["by_doc_block"].get((doc_id, block_id), []):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                candidates.append(chunk)
    return candidates


def choose_candidate(candidates: list[dict[str, Any]], qtype: str, variant: str, preferred_block_ids: list[str]) -> dict[str, Any]:
    preferred = set(preferred_block_ids)

    def score(chunk: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        blocks = set(chunk_block_ids(chunk))
        overlap = len(blocks & preferred)
        is_caption = int(bool(chunk.get("contains_table_caption")))
        is_table = int(bool(chunk.get("contains_table_caption") or chunk.get("contains_table_text")))
        is_related = int(chunk_has_table_related(chunk))
        is_normal = int("paragraph" in set(chunk.get("evidence_types") or chunk.get("block_types") or []))
        if qtype == "caption_level_table":
            primary = is_caption
        elif qtype == "table_content":
            primary = is_related + is_table if variant == "enhanced" else is_caption
        elif qtype == "normal_control":
            primary = is_normal - is_table
        else:
            primary = 0
        return (overlap, primary, is_caption, is_table, -len(chunk.get("text", "")), chunk["chunk_id"])

    return sorted(candidates, key=score, reverse=True)[0]


def mapping_status_for(
    stable_ids: list[str],
    corrected_baseline: str,
    corrected_enhanced: str,
    baseline_candidates: list[dict[str, Any]],
    enhanced_candidates: list[dict[str, Any]],
    original_baseline: str,
    original_enhanced: str,
) -> str:
    if not stable_ids:
        return "not_applicable"
    if not corrected_baseline or not corrected_enhanced:
        return "mapping_missing"
    if len(baseline_candidates) > 1 or len(enhanced_candidates) > 1:
        return "multiple_candidate_chunks"
    if corrected_baseline != original_baseline or corrected_enhanced != original_enhanced:
        return "remapped_to_new_chunk"
    return "mapped_same_chunk"


def chunk_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    ids = chunk.get("source_block_ids") or chunk.get("block_ids") or []
    return [str(item) for item in ids if item]


def table_caption_block_ids(chunk: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if isinstance(meta, dict) and meta.get("type") == "table_caption":
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    return ids


def non_table_block_ids(chunk: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if not isinstance(meta, dict):
            continue
        if meta.get("type") not in {"table_caption", "table_text", "figure_caption"}:
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    return ids


def correct_record(record: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    variant = record["variant"]
    corrected_target = mapping[f"corrected_{variant}_target_chunk_id"]
    original_target = record.get("target_chunk_id", "")
    corrected_rank, rank_source = corrected_rank_for(record, corrected_target, original_target)
    out = dict(record)
    out["original_target_chunk_id"] = original_target
    out["corrected_target_chunk_id"] = corrected_target
    out["stable_target_block_ids"] = mapping["stable_target_block_ids"]
    out["mapping_status"] = mapping["mapping_status"]
    out["corrected_target_chunk_rank"] = corrected_rank
    out["corrected_rank_source"] = rank_source
    out["original_chunk_hit_at_10"] = bool(record.get("chunk_hit_at_10"))
    out["corrected_chunk_hit_at_10"] = bool(corrected_target and 0 < corrected_rank <= 10)
    out["corrected_chunk_hit_at_20"] = bool(corrected_target and 0 < corrected_rank <= 20)
    out["corrected_chunk_hit_at_50"] = bool(corrected_target and 0 < corrected_rank <= 50)
    return out


def corrected_rank_for(record: dict[str, Any], corrected_target: str, original_target: str) -> tuple[int, str]:
    if not corrected_target:
        return 0, "missing_target"
    if corrected_target == original_target:
        return int(record.get("target_chunk_rank") or 0), "original_full_rank"
    for hit in record.get("top_hits") or []:
        if hit.get("chunk_id") == corrected_target:
            return int(hit.get("rank") or 0), "stored_top50_rank"
    return 0, "not_in_stored_top50"


def aggregate_raw(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    return aggregate(records, hit_prefix="", rank_key="target_chunk_rank", chunk_hit_key="chunk_hit_at_10", top_k=top_k)


def aggregate_corrected(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    return aggregate(records, hit_prefix="corrected_", rank_key="corrected_target_chunk_rank", chunk_hit_key="corrected_chunk_hit_at_10", top_k=top_k)


def aggregate(records: list[dict[str, Any]], hit_prefix: str, rank_key: str, chunk_hit_key: str, top_k: int) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("include_in_main_denominator"):
            by_key[(record["mode"], record["query_type"], record["variant"])].append(record)

    metrics: dict[str, Any] = {}
    for (mode, qtype, variant), rows in sorted(by_key.items()):
        metrics[f"{mode}/{qtype}/{variant}"] = basic_metrics(rows, hit_prefix, rank_key)

    paired: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in records}):
        for qtype in sorted({r["query_type"] for r in records if r.get("include_in_main_denominator")}):
            base = [r for r in records if r["mode"] == mode and r["query_type"] == qtype and r["variant"] == "baseline" and r.get("include_in_main_denominator")]
            enh = [r for r in records if r["mode"] == mode and r["query_type"] == qtype and r["variant"] == "enhanced" and r.get("include_in_main_denominator")]
            paired[f"{mode}/{qtype}"] = paired_metrics(base, enh, chunk_hit_key, rank_key, top_k)
    metrics["paired"] = paired
    return metrics


def basic_metrics(rows: list[dict[str, Any]], hit_prefix: str, rank_key: str) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "doc_hit_at_10": mean_bool(rows, "doc_hit_at_10"),
        "doc_hit_at_20": mean_bool(rows, "doc_hit_at_20"),
        "doc_hit_at_50": mean_bool(rows, "doc_hit_at_50"),
        f"{hit_prefix}chunk_hit_at_10": mean_bool(rows, f"{hit_prefix}chunk_hit_at_10" if hit_prefix else "chunk_hit_at_10"),
        f"{hit_prefix}chunk_hit_at_20": mean_bool(rows, f"{hit_prefix}chunk_hit_at_20" if hit_prefix else "chunk_hit_at_20"),
        f"{hit_prefix}chunk_hit_at_50": mean_bool(rows, f"{hit_prefix}chunk_hit_at_50" if hit_prefix else "chunk_hit_at_50"),
        "doc_mrr": mean_reciprocal(rows, "target_doc_rank"),
        f"{hit_prefix}chunk_mrr": mean_reciprocal(rows, rank_key),
        "mean_doc_rank_found": mean_rank(rows, "target_doc_rank"),
        f"mean_{hit_prefix}chunk_rank_found": mean_rank(rows, rank_key),
        "mean_table_topk_occupancy": statistics.mean(float(r["table_topk_occupancy"]) for r in rows),
        "mean_table_related_topk_occupancy": statistics.mean(float(r["table_related_topk_occupancy"]) for r in rows),
        "rank_source_counts": dict(Counter(r.get("corrected_rank_source", "original") for r in rows)),
    }


def paired_metrics(base: list[dict[str, Any]], enh: list[dict[str, Any]], chunk_hit_key: str, rank_key: str, top_k: int) -> dict[str, Any]:
    b_by_id = {r["sample_id"]: r for r in base}
    e_by_id = {r["sample_id"]: r for r in enh}
    ids = sorted(set(b_by_id) & set(e_by_id))
    enhanced_only = baseline_only = both_success = both_fail = 0
    rank_deltas: list[int] = []
    doc_rank_deltas: list[int] = []
    for sid in ids:
        b = b_by_id[sid]
        e = e_by_id[sid]
        b_hit = bool(b.get(chunk_hit_key))
        e_hit = bool(e.get(chunk_hit_key))
        if e_hit and not b_hit:
            enhanced_only += 1
        elif b_hit and not e_hit:
            baseline_only += 1
        elif b_hit and e_hit:
            both_success += 1
        else:
            both_fail += 1
        rank_deltas.append(rank_for_delta(b.get(rank_key, 0), top_k) - rank_for_delta(e.get(rank_key, 0), top_k))
        doc_rank_deltas.append(rank_for_delta(b.get("target_doc_rank", 0), top_k) - rank_for_delta(e.get("target_doc_rank", 0), top_k))
    return {
        "count": len(ids),
        "enhanced_only_success_count": enhanced_only,
        "baseline_only_success_count": baseline_only,
        "both_success_count": both_success,
        "both_fail_count": both_fail,
        "mean_target_chunk_rank_improvement": statistics.mean(rank_deltas) if rank_deltas else 0.0,
        "mean_doc_rank_improvement": statistics.mean(doc_rank_deltas) if doc_rank_deltas else 0.0,
    }


def rank_for_delta(rank: Any, top_k: int) -> int:
    rank_i = int(rank or 0)
    return rank_i if rank_i > 0 else top_k + 1


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def mean_rank(rows: list[dict[str, Any]], key: str) -> float | None:
    ranks = [int(r[key]) for r in rows if int(r.get(key) or 0) > 0]
    return statistics.mean(ranks) if ranks else None


def mean_reciprocal(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean((1.0 / int(r.get(key) or 0)) if int(r.get(key) or 0) > 0 else 0.0 for r in rows) if rows else 0.0


def build_topk_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    dist: dict[str, Any] = {}
    for record in records:
        if record["variant"] != "enhanced":
            continue
        key = f"{record['mode']}/{record['query_type']}"
        item = dist.setdefault(key, {"count": 0, "table_topk_occupancy_values": [], "table_related_topk_occupancy_values": [], "top_doc_ids": Counter()})
        item["count"] += 1
        item["table_topk_occupancy_values"].append(record["table_topk_occupancy"])
        item["table_related_topk_occupancy_values"].append(record["table_related_topk_occupancy"])
        for hit in record["top_hits"][:10]:
            item["top_doc_ids"][hit["doc_id"]] += 1
    for item in dist.values():
        table_values = item.pop("table_topk_occupancy_values")
        related_values = item.pop("table_related_topk_occupancy_values")
        counter = item["top_doc_ids"]
        item["mean_table_topk_occupancy"] = statistics.mean(table_values) if table_values else 0.0
        item["mean_table_related_topk_occupancy"] = statistics.mean(related_values) if related_values else 0.0
        item["top_doc_ids"] = counter.most_common(10)
    return dist


def mode_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qtype in ("table_content", "caption_level_table", "normal_control"):
        out[qtype] = {}
        for mode in ("dense_only", "bm25_only", "hybrid"):
            b = metrics.get(f"{mode}/{qtype}/baseline", {})
            e = metrics.get(f"{mode}/{qtype}/enhanced", {})
            out[qtype][mode] = {
                "baseline_doc_hit_at_10": b.get("doc_hit_at_10"),
                "enhanced_doc_hit_at_10": e.get("doc_hit_at_10"),
                "baseline_corrected_chunk_hit_at_10": b.get("corrected_chunk_hit_at_10"),
                "enhanced_corrected_chunk_hit_at_10": e.get("corrected_chunk_hit_at_10"),
                "corrected_doc_hit_delta": delta(e.get("doc_hit_at_10"), b.get("doc_hit_at_10")),
                "corrected_chunk_hit_delta": delta(e.get("corrected_chunk_hit_at_10"), b.get("corrected_chunk_hit_at_10")),
            }
    return out


def caption_regression_review(
    raw_records: list[dict[str, Any]],
    corrected_records: list[dict[str, Any]],
    mapping_by_sample: dict[str, dict[str, str]],
) -> dict[str, Any]:
    raw_pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    corr_pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in raw_records:
        if record["mode"] == "hybrid" and record["query_type"] == "caption_level_table" and record.get("include_in_main_denominator"):
            raw_pairs[record["sample_id"]][record["variant"]] = record
    for record in corrected_records:
        if record["mode"] == "hybrid" and record["query_type"] == "caption_level_table" and record.get("include_in_main_denominator"):
            corr_pairs[record["sample_id"]][record["variant"]] = record
    rows = []
    for sid, pair in sorted(raw_pairs.items()):
        base = pair.get("baseline")
        enh = pair.get("enhanced")
        corr = corr_pairs.get(sid, {})
        if not base or not enh or not corr.get("enhanced"):
            continue
        if not base.get("chunk_hit_at_10") or enh.get("chunk_hit_at_10"):
            continue
        corrected_enh = corr["enhanced"]
        artifact = bool(corrected_enh.get("corrected_chunk_hit_at_10") and corrected_enh.get("corrected_target_chunk_id") != enh.get("target_chunk_id"))
        true_regression = not corrected_enh.get("corrected_chunk_hit_at_10")
        rows.append({
            "sample_id": sid,
            "original_enhanced_hit": bool(enh.get("chunk_hit_at_10")),
            "corrected_enhanced_hit": bool(corrected_enh.get("corrected_chunk_hit_at_10")),
            "mapping_artifact": artifact,
            "true_caption_regression": true_regression,
            "stable_target_block_ids": mapping_by_sample[sid]["stable_target_block_ids"],
        })
    return {
        "rows": rows,
        "counts": {
            "regression_count_original": len(rows),
            "mapping_artifact_count": sum(1 for row in rows if row["mapping_artifact"]),
            "true_caption_regression_count": sum(1 for row in rows if row["true_caption_regression"]),
        },
    }


def corrected_gate(metrics: dict[str, Any], validation: dict[str, Any], caption_review: dict[str, Any]) -> dict[str, Any]:
    hybrid = lambda q, v: metrics.get(f"hybrid/{q}/{v}", {})
    table_b = hybrid("table_content", "baseline")
    table_e = hybrid("table_content", "enhanced")
    caption_b = hybrid("caption_level_table", "baseline")
    caption_e = hybrid("caption_level_table", "enhanced")
    normal_b = hybrid("normal_control", "baseline")
    normal_e = hybrid("normal_control", "enhanced")
    table_pair = metrics.get("paired", {}).get("hybrid/table_content", {})
    normal_occ_b = normal_b.get("mean_table_topk_occupancy", 0.0)
    normal_occ_e = normal_e.get("mean_table_topk_occupancy", 0.0)
    normal_related_b = normal_b.get("mean_table_related_topk_occupancy", 0.0)
    normal_related_e = normal_e.get("mean_table_related_topk_occupancy", 0.0)
    return {
        "engineering_ok": all([
            validation.get("baseline_row_count_matches_chunks"),
            validation.get("enhanced_row_count_matches_chunks"),
            validation.get("baseline_bm25_count_matches_chunks"),
            validation.get("enhanced_bm25_count_matches_chunks"),
        ]),
        "table_content_doc_hit_non_degraded": table_e.get("doc_hit_at_10", 0.0) >= table_b.get("doc_hit_at_10", 0.0),
        "table_content_corrected_chunk_hit_improved": table_e.get("corrected_chunk_hit_at_10", 0.0) > table_b.get("corrected_chunk_hit_at_10", 0.0),
        "table_content_enhanced_only_ge_baseline_only": table_pair.get("enhanced_only_success_count", 0) >= table_pair.get("baseline_only_success_count", 0),
        "caption_corrected_chunk_not_degraded_over_0_05": caption_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= caption_b.get("corrected_chunk_hit_at_10", 0.0),
        "caption_true_regression_le_1": caption_review["counts"]["true_caption_regression_count"] <= 1,
        "normal_doc_not_degraded_over_0_05": normal_e.get("doc_hit_at_10", 0.0) + 0.05 >= normal_b.get("doc_hit_at_10", 0.0),
        "normal_corrected_chunk_not_degraded_over_0_05": normal_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= normal_b.get("corrected_chunk_hit_at_10", 0.0),
        "normal_table_occupancy_baseline": normal_occ_b,
        "normal_table_occupancy_enhanced": normal_occ_e,
        "normal_table_related_occupancy_baseline": normal_related_b,
        "normal_table_related_occupancy_enhanced": normal_related_e,
        "normal_table_occupancy_non_increased_or_hits_ok": normal_occ_e <= normal_occ_b or (
            normal_e.get("doc_hit_at_10", 0.0) + 0.05 >= normal_b.get("doc_hit_at_10", 0.0)
            and normal_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= normal_b.get("corrected_chunk_hit_at_10", 0.0)
        ),
        "no_table_related_takeover": normal_related_e <= max(0.05, normal_related_b + 0.05),
    }


def write_mapping_csv(path: Path, mappings: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "query_type",
        "target_doc_id",
        "original_baseline_target_chunk_id",
        "original_enhanced_target_chunk_id",
        "stable_target_block_ids",
        "corrected_baseline_target_chunk_id",
        "corrected_enhanced_target_chunk_id",
        "mapping_status",
        "mapping_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mappings)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_miss_examples(path: Path, records: list[dict[str, Any]], queries: list[dict[str, Any]], top_k: int) -> None:
    query_by_id = {q["sample_id"]: q for q in queries}
    misses = [
        r for r in records
        if r["variant"] == "enhanced" and r["mode"] == "hybrid" and r.get("include_in_main_denominator") and not r["corrected_chunk_hit_at_10"]
    ][:25]
    lines = ["# Phase 5C-3 Corrected Miss Examples", ""]
    for r in misses:
        q = query_by_id.get(r["sample_id"], {})
        lines.extend([
            f"## {r['sample_id']} ({r['query_type']})",
            "",
            f"- query: {r['query']}",
            f"- target_doc_id: {r['target_doc_id']}",
            f"- corrected target chunk: {r['corrected_target_chunk_id']}",
            f"- stable target block ids: {r['stable_target_block_ids']}",
            f"- target_doc_rank: {r['target_doc_rank']}",
            f"- corrected target_chunk_rank: {r['corrected_target_chunk_rank']}",
            f"- anchor_terms: {q.get('anchor_terms', [])}",
            f"- top{top_k}: {[(h['rank'], h['doc_id'], h['chunk_id']) for h in r['top_hits'][:top_k]]}",
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
            delta_value = rank_for_delta(b["corrected_target_chunk_rank"], top_k) - rank_for_delta(e["corrected_target_chunk_rank"], top_k)
            deltas.append((delta_value, sid, qtype, b, e))
    deltas.sort(key=lambda item: item[0], reverse=True)
    lines = ["# Phase 5C-3 Corrected Rank Delta Examples", "", "## Largest Improvements", ""]
    for delta_value, sid, qtype, b, e in deltas[:15]:
        lines.append(f"- {sid} ({qtype}): corrected chunk rank baseline={b['corrected_target_chunk_rank'] or 'miss'} enhanced={e['corrected_target_chunk_rank'] or 'miss'} delta={delta_value}")
    lines.extend(["", "## Largest Regressions", ""])
    for delta_value, sid, qtype, b, e in reversed(deltas[-15:]):
        lines.append(f"- {sid} ({qtype}): corrected chunk rank baseline={b['corrected_target_chunk_rank'] or 'miss'} enhanced={e['corrected_target_chunk_rank'] or 'miss'} delta={delta_value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    validation = payload["validation"]
    gate = payload["gate"]
    caption_counts = payload["caption_regression_review"]["counts"]
    recommend_5c4 = all([
        gate["engineering_ok"],
        gate["table_content_doc_hit_non_degraded"],
        gate["table_content_corrected_chunk_hit_improved"],
        gate["table_content_enhanced_only_ge_baseline_only"],
        gate["caption_corrected_chunk_not_degraded_over_0_05"],
        gate["caption_true_regression_le_1"],
        gate["normal_doc_not_degraded_over_0_05"],
        gate["normal_corrected_chunk_not_degraded_over_0_05"],
        gate["normal_table_occupancy_non_increased_or_hits_ok"],
        gate["no_table_related_takeover"],
    ])

    def metric(qtype: str, variant: str, key: str) -> Any:
        return metrics.get(f"hybrid/{qtype}/{variant}", {}).get(key)

    def paired(qtype: str) -> dict[str, Any]:
        return metrics.get("paired", {}).get(f"hybrid/{qtype}", {})

    lines = [
        "# Phase 5C-3 Table Retrieval A/B Summary",
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
        "## Corrected Hybrid Results",
        "",
        f"- table_content doc_hit@10: baseline={fmt(metric('table_content', 'baseline', 'doc_hit_at_10'))}, enhanced={fmt(metric('table_content', 'enhanced', 'doc_hit_at_10'))}",
        f"- table_content corrected_chunk_hit@10: baseline={fmt(metric('table_content', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('table_content', 'enhanced', 'corrected_chunk_hit_at_10'))}",
        f"- table_content hit split: {split_line(paired('table_content'))}",
        f"- caption_level_table corrected_chunk_hit@10: baseline={fmt(metric('caption_level_table', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('caption_level_table', 'enhanced', 'corrected_chunk_hit_at_10'))}",
        f"- caption true regression count: {caption_counts['true_caption_regression_count']}",
        f"- normal_control doc_hit@10: baseline={fmt(metric('normal_control', 'baseline', 'doc_hit_at_10'))}, enhanced={fmt(metric('normal_control', 'enhanced', 'doc_hit_at_10'))}",
        f"- normal_control corrected_chunk_hit@10: baseline={fmt(metric('normal_control', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('normal_control', 'enhanced', 'corrected_chunk_hit_at_10'))}",
        f"- normal table occupancy: baseline={fmt(gate['normal_table_occupancy_baseline'])}, enhanced={fmt(gate['normal_table_occupancy_enhanced'])}",
        f"- normal table_related occupancy: baseline={fmt(gate['normal_table_related_occupancy_baseline'])}, enhanced={fmt(gate['normal_table_related_occupancy_enhanced'])}",
        "",
        "## Gate Answers",
        "",
        f"1. Phase 5C table enhancement remains effective on representative docs: {yes_no(gate['table_content_corrected_chunk_hit_improved'])}.",
        f"2. table_content stable improvement: {yes_no(gate['table_content_doc_hit_non_degraded'] and gate['table_content_corrected_chunk_hit_improved'])}.",
        f"3. caption_level_table has no true regression beyond gate: {yes_no(gate['caption_corrected_chunk_not_degraded_over_0_05'] and gate['caption_true_regression_le_1'])}.",
        f"4. normal_control has no degradation beyond gate: {yes_no(gate['normal_doc_not_degraded_over_0_05'] and gate['normal_corrected_chunk_not_degraded_over_0_05'])}.",
        f"5. table-related takeover exists: {yes_no(not gate['no_table_related_takeover'])}.",
        "6. false positive risk is controlled if association audit gate is green; see association_audit/summary.md.",
        "7. chunk/token growth is controlled if chunk_ab_summary gate is green.",
        f"8. recommend full-corpus pre-planning rather than direct merge: {yes_no(recommend_5c4)}.",
        "9. OCR is still not needed.",
        "10. heuristic adjustment needed before next phase: no if association audit remains green.",
        "11. schema/table_object needed: no.",
        f"12. recommend Phase 5C-4: {yes_no(recommend_5c4)}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def split_line(pair: dict[str, Any]) -> str:
    return (
        f"enhanced_only={pair.get('enhanced_only_success_count', 0)}, "
        f"baseline_only={pair.get('baseline_only_success_count', 0)}, "
        f"both_success={pair.get('both_success_count', 0)}, both_fail={pair.get('both_fail_count', 0)}"
    )


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


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
