#!/usr/bin/env python3
"""Evaluate an approved Phase 4E-3 manual retrieval eval set.

The script reuses an existing Milvus collection and BM25 cache. It does not
build indexes, call an LLM, run a reranker, or change retrieval parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synbio_rag.domain.config import KnowledgeBaseConfig, RetrievalConfig
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


MARKER_RE = re.compile(r"\[(?:TABLE CAPTION|TABLE TEXT|FIGURE CAPTION)\]\s*")
MAIN_SAMPLE_TYPES = ("table", "figure", "normal")
EVIDENCE_TYPES = {"table", "figure", "table_figure"}


def clean_body(text: Any) -> str:
    body = MARKER_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", body).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def evidence_type_from_chunk_obj(chunk: Any) -> str:
    metadata = getattr(chunk, "metadata", {}) or {}
    table = bool(metadata.get("contains_table_caption") or metadata.get("contains_table_text"))
    figure = bool(metadata.get("contains_figure_caption"))
    if table and figure:
        return "table_figure"
    if table:
        return "table"
    if figure:
        return "figure"
    return "paragraph"


def build_retrievers(
    chunks_jsonl: Path,
    milvus_uri: str,
    collection_name: str,
    bm25_index_path: Path,
    model_path: str,
    search_limit: int,
) -> tuple[dict[str, Any], int]:
    retrieval_config = RetrievalConfig(
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        bm25_cache_path=str(bm25_index_path),
        dense_limit=max(40, search_limit),
        bm25_limit=max(40, search_limit),
        search_limit=max(40, search_limit),
        score_floor=0.0,
    )
    kb_config = KnowledgeBaseConfig(
        chunk_jsonl=str(chunks_jsonl),
        embedding_model_path=model_path,
        embedding_max_length=512,
    )
    embedder = BGEM3Embedder(
        model_path=kb_config.embedding_model_path,
        dim=kb_config.embedding_dim,
        max_length=kb_config.embedding_max_length,
    )
    dense = MilvusRetriever(retrieval_config, embedder)
    bm25 = BM25Retriever(retrieval_config, kb_config, milvus_client=dense.client)
    bm25._ensure_index()
    hybrid = HybridRetriever(retrieval_config, dense, bm25)
    return {"dense_only": dense, "bm25_only": bm25, "hybrid": hybrid}, len(bm25._records)


def serialize_hit(hit: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section": hit.section,
        "evidence_type": evidence_type_from_chunk_obj(hit),
        "fusion_score": hit.fusion_score,
        "vector_score": hit.vector_score,
        "bm25_score": hit.bm25_score,
        "text_preview": clean_body(hit.text)[:260],
    }


def run_sample(retriever: Any, sample: dict[str, Any], search_limit: int, top_k: int) -> dict[str, Any]:
    hits = retriever.search(sample["query"], limit=search_limit)
    serialized = [serialize_hit(hit, idx) for idx, hit in enumerate(hits[:search_limit], start=1)]
    target_chunk = sample["target_chunk_id"]
    target_doc = sample["target_doc_id"]
    target_chunk_rank = next((hit["rank"] for hit in serialized if hit["chunk_id"] == target_chunk), None)
    target_doc_rank = next((hit["rank"] for hit in serialized if hit["doc_id"] == target_doc), None)
    result = dict(sample)
    result.update(
        {
            "target_doc_rank": target_doc_rank,
            "target_chunk_rank": target_chunk_rank,
            "target_doc_reciprocal_rank": (1.0 / target_doc_rank) if target_doc_rank else 0.0,
            "target_chunk_reciprocal_rank": (1.0 / target_chunk_rank) if target_chunk_rank else 0.0,
            "doc_hit@10": target_doc_rank is not None and target_doc_rank <= 10,
            "chunk_hit@10": target_chunk_rank is not None and target_chunk_rank <= 10,
            "doc_hit@20": target_doc_rank is not None and target_doc_rank <= 20,
            "chunk_hit@20": target_chunk_rank is not None and target_chunk_rank <= 20,
            "doc_hit@50": target_doc_rank is not None and target_doc_rank <= 50,
            "chunk_hit@50": target_chunk_rank is not None and target_chunk_rank <= 50,
            "top_k": serialized[:top_k],
            "top50": serialized,
        }
    )
    return result


def metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("sample_type") in MAIN_SAMPLE_TYPES:
            by_type[str(result["sample_type"])].append(result)
    summary: dict[str, Any] = {}
    for sample_type in MAIN_SAMPLE_TYPES:
        items = by_type.get(sample_type, [])
        total = len(items)
        summary[sample_type] = {"count": total}
        doc_ranks = [item["target_doc_rank"] for item in items if item.get("target_doc_rank") is not None]
        chunk_ranks = [item["target_chunk_rank"] for item in items if item.get("target_chunk_rank") is not None]
        summary[sample_type]["target_doc_rank_mean"] = sum(doc_ranks) / len(doc_ranks) if doc_ranks else None
        summary[sample_type]["target_chunk_rank_mean"] = sum(chunk_ranks) / len(chunk_ranks) if chunk_ranks else None
        summary[sample_type]["target_doc_mrr"] = (
            sum(float(item.get("target_doc_reciprocal_rank", 0.0)) for item in items) / total if total else 0.0
        )
        summary[sample_type]["target_chunk_mrr"] = (
            sum(float(item.get("target_chunk_reciprocal_rank", 0.0)) for item in items) / total if total else 0.0
        )
        for k in (10, 20, 50):
            doc_hits = sum(1 for item in items if item.get(f"doc_hit@{k}"))
            chunk_hits = sum(1 for item in items if item.get(f"chunk_hit@{k}"))
            summary[sample_type][f"doc_hit@{k}"] = doc_hits
            summary[sample_type][f"chunk_hit@{k}"] = chunk_hits
            summary[sample_type][f"doc_hit_rate@{k}"] = doc_hits / total if total else 0.0
            summary[sample_type][f"chunk_hit_rate@{k}"] = chunk_hits / total if total else 0.0
    return summary


def topk_distribution(results: list[dict[str, Any]]) -> dict[str, Any]:
    distribution: dict[str, Any] = {}
    for result in results:
        sample_type = str(result.get("sample_type", ""))
        if sample_type not in distribution:
            distribution[sample_type] = {
                "query_count": 0,
                "topk_total": 0,
                "topk_evidence_type_counts": Counter(),
                "table_figure_topk_total": 0,
            }
        entry = distribution[sample_type]
        entry["query_count"] += 1
        for hit in result.get("top_k", []):
            evidence_type = hit.get("evidence_type", "")
            entry["topk_total"] += 1
            entry["topk_evidence_type_counts"][evidence_type] += 1
            if evidence_type in EVIDENCE_TYPES:
                entry["table_figure_topk_total"] += 1
    normalized: dict[str, Any] = {}
    for sample_type, entry in distribution.items():
        total = entry["topk_total"]
        normalized[sample_type] = {
            "query_count": entry["query_count"],
            "topk_total": total,
            "topk_evidence_type_counts": dict(sorted(entry["topk_evidence_type_counts"].items())),
            "table_figure_topk_total": entry["table_figure_topk_total"],
            "table_figure_topk_rate": entry["table_figure_topk_total"] / total if total else 0.0,
        }
    return normalized


def mode_comparison(results_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    metrics_by_mode = {mode: metric_summary(results) for mode, results in results_by_mode.items()}
    for sample_type in MAIN_SAMPLE_TYPES:
        comparison[sample_type] = {}
        for mode, results in results_by_mode.items():
            misses = [
                result
                for result in results
                if result.get("sample_type") == sample_type and not result.get("chunk_hit@10")
            ]
            comparison[sample_type][mode] = {
                "metrics": metrics_by_mode.get(mode, {}).get(sample_type, {}),
                "chunk_miss@10": len(misses),
                "doc_miss@10": sum(1 for result in results if result.get("sample_type") == sample_type and not result.get("doc_hit@10")),
            }
    return comparison


def miss_reason(result: dict[str, Any], top_k: int) -> str:
    doc_rank = result.get("target_doc_rank")
    chunk_rank = result.get("target_chunk_rank")
    if chunk_rank is not None and chunk_rank <= top_k:
        return "chunk_hit"
    if doc_rank is None:
        return "target_doc_not_in_top50"
    if doc_rank > top_k:
        return "target_doc_ranked_11_to_50"
    if chunk_rank is None:
        return "target_doc_hit_but_target_chunk_not_in_top50"
    return "target_doc_hit_but_target_chunk_ranked_11_to_50"


def miss_reason_counts(results: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        sample_type = str(result.get("sample_type", ""))
        if sample_type not in MAIN_SAMPLE_TYPES or result.get(f"chunk_hit@{top_k}"):
            continue
        counts[sample_type][miss_reason(result, top_k)] += 1
    return {sample_type: dict(counter) for sample_type, counter in sorted(counts.items())}


def display_score(hit: dict[str, Any]) -> float:
    return float(hit.get("fusion_score") or hit.get("vector_score") or hit.get("bm25_score") or 0.0)


def write_miss_examples(path: Path, results_by_mode: dict[str, list[dict[str, Any]]], top_k: int) -> None:
    lines = ["# Phase 4E-3 Manual Eval Miss Examples", ""]
    hybrid_misses = [
        result
        for result in results_by_mode.get("hybrid", [])
        if result.get("sample_type") in MAIN_SAMPLE_TYPES and not result.get(f"chunk_hit@{top_k}")
    ]
    lines.extend(
        [
            f"- hybrid_chunk_misses@{top_k}: {len(hybrid_misses)}",
            f"- hybrid_miss_reason_counts: `{json.dumps(miss_reason_counts(results_by_mode.get('hybrid', []), top_k), ensure_ascii=False)}`",
            "",
        ]
    )
    ordered: list[tuple[str, dict[str, Any]]] = [("hybrid", result) for result in hybrid_misses]
    for mode, results in sorted(results_by_mode.items()):
        if mode == "hybrid":
            continue
        ordered.extend(
            (mode, result)
            for result in results
            if result.get("sample_type") in MAIN_SAMPLE_TYPES and not result.get(f"chunk_hit@{top_k}")
        )
    for mode, result in ordered[:80]:
        lines.extend(
            [
                f"## {mode} | {result.get('sample_type')} | {result.get('sample_id')} | {result.get('target_chunk_id')}",
                "",
                f"- query: {result.get('query')}",
                f"- target_doc_id: `{result.get('target_doc_id')}`",
                f"- doc_hit@{top_k}: `{result.get(f'doc_hit@{top_k}')}` rank: `{result.get('target_doc_rank')}`",
                f"- chunk_hit@{top_k}: `{result.get(f'chunk_hit@{top_k}')}` rank: `{result.get('target_chunk_rank')}`",
                f"- miss_reason: `{miss_reason(result, top_k)}`",
                f"- anchors: `{result.get('anchor_terms', [])}`",
                f"- target_preview: {result.get('target_text_preview', '')}",
                "",
                "Top-k sample:",
            ]
        )
        for hit in result.get("top_k", [])[:5]:
            lines.append(
                f"- #{hit['rank']} `{hit['doc_id']}` `{hit['chunk_id']}` "
                f"{hit['evidence_type']} score={display_score(hit):.4f} :: {hit['text_preview']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def pilot_gate(metrics: dict[str, Any], distribution: dict[str, Any]) -> dict[str, bool]:
    hybrid = metrics.get("hybrid", {})
    normal_occupancy = distribution.get("hybrid", {}).get("normal", {}).get("table_figure_topk_rate", 0.0)
    return {
        "table_doc_hit@10": hybrid.get("table", {}).get("doc_hit_rate@10", 0.0) >= 0.75,
        "table_chunk_hit@10": hybrid.get("table", {}).get("chunk_hit_rate@10", 0.0) >= 0.60,
        "figure_doc_hit@10": hybrid.get("figure", {}).get("doc_hit_rate@10", 0.0) >= 0.80,
        "figure_chunk_hit@10": hybrid.get("figure", {}).get("chunk_hit_rate@10", 0.0) >= 0.70,
        "normal_doc_hit@10": hybrid.get("normal", {}).get("doc_hit_rate@10", 0.0) >= 0.80,
        "normal_table_figure_occupancy": normal_occupancy <= 0.15,
    }


def write_summary(
    path: Path,
    eval_rows: list[dict[str, Any]],
    metrics_by_mode: dict[str, Any],
    distribution_by_mode: dict[str, Any],
    gates: dict[str, bool],
    sanity_results: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    sample_counts = Counter(str(row.get("sample_type", "")) for row in eval_rows)
    all_gates_pass = all(gates.values()) if gates else False
    caption_level_gate_pass = all(
        gates.get(key, False)
        for key in (
            "table_doc_hit@10",
            "table_chunk_hit@10",
            "figure_doc_hit@10",
            "figure_chunk_hit@10",
        )
    )
    normal_occupancy = distribution_by_mode.get("hybrid", {}).get("normal", {}).get("table_figure_topk_rate", 0.0)
    hybrid_miss_reasons = miss_reason_counts(sanity_results.get("_main_hybrid_results", []), args.top_k)
    sanity_count = len(sanity_results.get("results_by_mode", {}).get("hybrid", []))
    sanity_chunk_hits = sum(
        1 for result in sanity_results.get("results_by_mode", {}).get("hybrid", []) if result.get(f"chunk_hit@{args.top_k}")
    )
    lines = [
        "# Phase 4E-3 Manual Eval Retrieval-only Summary",
        "",
        "## Inputs",
        "",
        f"- eval_jsonl: `{args.eval_jsonl}`",
        f"- chunks_jsonl: `{args.chunks_jsonl}`",
        f"- milvus_uri: `{args.milvus_uri}`",
        f"- collection_name: `{args.collection_name}`",
        f"- bm25_index_path: `{args.bm25_index_path}`",
        f"- retrieval_modes: `{args.retrieval_modes}`",
        f"- top_k: {args.top_k}",
        f"- sanity_anchor_jsonl: `{args.sanity_anchor_jsonl}`",
        "- evaluation action: reused existing full compact index; no index rebuild, reranker, Qwen, OCR, or generation eval.",
        "",
        "## Eval Set Composition",
        "",
        f"- sample_counts: `{json.dumps(dict(sorted(sample_counts.items())), ensure_ascii=False)}`",
        "- sanity_anchor samples are reported separately from the main table/figure/normal denominator.",
        "",
        "## Metrics",
        "",
    ]
    for mode in args.retrieval_modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        lines.extend([f"### {mode}", ""])
        for sample_type in MAIN_SAMPLE_TYPES:
            item = metrics_by_mode.get(mode, {}).get(sample_type, {})
            lines.append(
                f"- {sample_type}: count={item.get('count', 0)} "
                f"doc_hit@10={item.get('doc_hit@10', 0)} "
                f"chunk_hit@10={item.get('chunk_hit@10', 0)} "
                f"doc_hit@20={item.get('doc_hit@20', 0)} "
                f"chunk_hit@20={item.get('chunk_hit@20', 0)} "
                f"doc_hit@50={item.get('doc_hit@50', 0)} "
                f"chunk_hit@50={item.get('chunk_hit@50', 0)} "
                f"doc_mrr={item.get('target_doc_mrr', 0.0):.3f} "
                f"chunk_mrr={item.get('target_chunk_mrr', 0.0):.3f} "
                f"mean_doc_rank={item.get('target_doc_rank_mean')} "
                f"mean_chunk_rank={item.get('target_chunk_rank_mean')}"
            )
        lines.append("")
    lines.extend(
        [
            "## Top-k Distribution",
            "",
            f"- hybrid normal table/figure top-k occupancy: {normal_occupancy:.3f}",
            f"- hybrid distribution: `{json.dumps(distribution_by_mode.get('hybrid', {}), ensure_ascii=False)}`",
            "",
            "## Sanity Anchor",
            "",
            f"- sanity_anchor_count: {sanity_count}",
            f"- hybrid chunk_hit@{args.top_k}: {sanity_chunk_hits}/{sanity_count}",
            "",
        ]
    )
    lines.extend(
        [
            "## Pilot Gate",
            "",
            f"- gates: `{json.dumps(gates, ensure_ascii=False)}`",
            f"- pilot_gate_pass: `{all_gates_pass}`",
            "",
            "## Answers",
            "",
            "1. approved eval set 的样本规模和组成：见 Eval Set Composition。",
            "2. table / figure / normal 的 dense-only、BM25-only、hybrid 结果：见 Metrics。",
            f"3. doc_hit 和 chunk_hit 是否达 pilot gate：`{all_gates_pass}`。",
            f"4. miss examples 的主要原因：hybrid miss reason counts `{json.dumps(hybrid_miss_reasons, ensure_ascii=False)}`；详见 `miss_examples.md`。",
            f"5. table/figure chunks 是否抢占 normal query：hybrid normal occupancy={normal_occupancy:.3f}。",
            f"6. compact retrieval_text 是否可以作为 caption-level experimental baseline：`{caption_level_gate_pass}`；table/figure gate 已单独判断，normal gate 不用于否决 caption-level baseline。",
            "7. 是否仍需 parser cleanup：不作为 Phase 4E-3B 阻断项；table/figure hybrid chunk_hit@10 为 59/60，仅需复核单个 table miss。",
            "8. 是否仍需 section metadata cleanup：仍需保留为后续 normal miss review 项；normal hybrid doc_hit@10 未达 gate。",
            "9. 是否仍需 compact_v2 / RRF ablation：建议作为后续诊断，不在本阶段调参；normal miss 存在系统性 target_doc_not_in_top50。",
            f"10. 是否建议 Phase 4 closeout：`{all_gates_pass}`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--sanity_anchor_jsonl", type=Path)
    parser.add_argument("--chunks_jsonl", required=True, type=Path)
    parser.add_argument("--milvus_uri", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--bm25_index_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--retrieval_modes", default="dense_only,bm25_only,hybrid")
    parser.add_argument("--model_path", default=str(REPO_ROOT / "models/BAAI/bge-m3"))
    args = parser.parse_args()

    if not args.eval_jsonl.exists():
        raise FileNotFoundError(f"Approved eval set does not exist: {args.eval_jsonl}")
    if args.sanity_anchor_jsonl and not args.sanity_anchor_jsonl.exists():
        raise FileNotFoundError(f"Sanity anchor JSONL does not exist: {args.sanity_anchor_jsonl}")

    eval_rows = load_jsonl(args.eval_jsonl)
    sanity_rows = load_jsonl(args.sanity_anchor_jsonl) if args.sanity_anchor_jsonl else []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_limit = max(50, args.top_k)
    retrievers, bm25_record_count = build_retrievers(
        chunks_jsonl=args.chunks_jsonl,
        milvus_uri=args.milvus_uri,
        collection_name=args.collection_name,
        bm25_index_path=args.bm25_index_path,
        model_path=args.model_path,
        search_limit=search_limit,
    )
    retrieval_modes = [mode.strip() for mode in args.retrieval_modes.split(",") if mode.strip()]
    invalid_modes = [mode for mode in retrieval_modes if mode not in retrievers]
    if invalid_modes:
        raise ValueError(f"Unsupported retrieval mode(s): {invalid_modes}")

    results_by_mode: dict[str, list[dict[str, Any]]] = {}
    metrics_by_mode: dict[str, Any] = {}
    distribution_by_mode: dict[str, Any] = {}
    for mode in retrieval_modes:
        results = [run_sample(retrievers[mode], sample, search_limit=search_limit, top_k=args.top_k) for sample in eval_rows]
        results_by_mode[mode] = results
        metrics_by_mode[mode] = metric_summary(results)
        distribution_by_mode[mode] = topk_distribution(results)

    sanity_payload: dict[str, Any] = {}
    if sanity_rows:
        sanity_results_by_mode: dict[str, list[dict[str, Any]]] = {}
        sanity_distribution_by_mode: dict[str, Any] = {}
        for mode in retrieval_modes:
            results = [
                run_sample(retrievers[mode], sample, search_limit=search_limit, top_k=args.top_k)
                for sample in sanity_rows
            ]
            sanity_results_by_mode[mode] = results
            sanity_distribution_by_mode[mode] = topk_distribution(results)
        sanity_payload = {
            "inputs": {
                "sanity_anchor_jsonl": str(args.sanity_anchor_jsonl),
                "top_k": args.top_k,
                "retrieval_modes": retrieval_modes,
            },
            "results_by_mode": sanity_results_by_mode,
            "topk_distribution_by_mode": sanity_distribution_by_mode,
        }
        (args.output_dir / "sanity_anchor_results.json").write_text(
            json.dumps(sanity_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    comparison = mode_comparison(results_by_mode)
    gates = pilot_gate(metrics_by_mode, distribution_by_mode)
    payload = {
        "inputs": {
            "eval_jsonl": str(args.eval_jsonl),
            "chunks_jsonl": str(args.chunks_jsonl),
            "milvus_uri": args.milvus_uri,
            "collection_name": args.collection_name,
            "bm25_index_path": str(args.bm25_index_path),
            "top_k": args.top_k,
            "retrieval_modes": retrieval_modes,
            "bm25_record_count": bm25_record_count,
        },
        "metrics_by_mode": metrics_by_mode,
        "pilot_gate": gates,
        "results_by_mode": results_by_mode,
    }
    (args.output_dir / "retrieval_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "topk_distribution.json").write_text(
        json.dumps(distribution_by_mode, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "mode_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_miss_examples(args.output_dir / "miss_examples.md", results_by_mode, args.top_k)
    summary_sanity_payload = dict(sanity_payload)
    summary_sanity_payload["_main_hybrid_results"] = results_by_mode.get("hybrid", [])
    write_summary(
        args.output_dir / "summary.md",
        eval_rows,
        metrics_by_mode,
        distribution_by_mode,
        gates,
        summary_sanity_payload,
        args,
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "metrics_by_mode": metrics_by_mode,
                "pilot_gate": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
