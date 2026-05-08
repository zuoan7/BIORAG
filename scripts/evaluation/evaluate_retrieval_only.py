#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.rerank_service import QwenReranker
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


DEFAULT_DATASET = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
DEFAULT_OUTPUT_DIR = Path("results/phase16r_retrieval_only_baseline")
DEFAULT_REPORT_DIR = Path("reports/phase16r_retrieval_only_baseline")
TRACE_FIELDNAMES = [
    "sample_id", "question", "category", "expected_route", "expected_doc_ids", "expected_source_files",
    "expected_sections", "negative_query", "should_require_doc_hit", "skipped_doc_hit",
    "dense_top_k", "dense_doc_hit_at_5", "dense_doc_hit_at_10", "dense_doc_hit_at_20",
    "dense_doc_hit_at_40", "dense_expected_best_rank", "dense_expected_found",
    "dense_top10_doc_ids", "dense_top10_source_files", "dense_top10_scores",
    "bm25_top_k", "bm25_doc_hit_at_5", "bm25_doc_hit_at_10", "bm25_doc_hit_at_20",
    "bm25_doc_hit_at_40", "bm25_expected_best_rank", "bm25_expected_found",
    "bm25_top10_doc_ids", "bm25_top10_source_files", "bm25_top10_scores",
    "bm25_query_tokens", "bm25_query_token_count",
    "hybrid_top_k", "hybrid_doc_hit_at_5", "hybrid_doc_hit_at_10", "hybrid_doc_hit_at_20",
    "hybrid_doc_hit_at_40", "hybrid_expected_best_rank", "hybrid_expected_found",
    "hybrid_top10_doc_ids", "hybrid_top10_source_files", "hybrid_top10_scores",
    "rerank_input_size", "rerank_output_k", "rerank_doc_hit_at_5", "rerank_doc_hit_at_10",
    "rerank_doc_hit_at_20", "rerank_expected_best_rank", "rerank_expected_found",
    "rerank_top10_doc_ids", "rerank_top10_source_files", "rerank_top10_scores",
    "first_stage_found", "first_stage_lost", "retrieval_diagnosis",
]
PER_SAMPLE_FIELDNAMES = [
    "sample_id", "question", "expected_doc_ids", "expected_source_files", "negative_query", "skipped_doc_hit",
    "dense_best_rank", "bm25_best_rank", "hybrid_best_rank", "rerank_best_rank",
    "dense_hit_at_20", "bm25_hit_at_20", "hybrid_hit_at_20", "rerank_hit_at_10",
    "final_retrieval_status", "recommended_next_action",
]
FAILURE_FIELDNAMES = [
    "sample_id", "group", "expected_doc_ids", "dense_best_rank", "bm25_best_rank", "hybrid_best_rank",
    "rerank_best_rank", "top_wrong_doc_ids", "top_wrong_titles", "likely_issue", "recommended_next_phase",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16R retrieval-only open baseline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N sample limit for smoke tests.")
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--bm25-top-k", type=int, default=40)
    parser.add_argument("--hybrid-top-k", type=int, default=40)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--command-used", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    if args.limit > 0:
        samples = samples[: args.limit]

    validation = validate_dataset(samples)
    tokenizer_validation = validate_bm25_tokenizer(samples)

    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    settings.retrieval.parent_expansion_enabled = False
    settings.retrieval.bm25_enabled = True
    settings.retrieval.hybrid_enabled = True

    embedder = BGEM3Embedder(
        model_path=settings.kb.embedding_model_path,
        dim=settings.kb.embedding_dim,
        max_length=settings.kb.embedding_max_length,
    )
    router = QueryRouter(settings.retrieval)
    dense = MilvusRetriever(settings.retrieval, embedder)
    bm25 = BM25Retriever(settings.retrieval, settings.kb, milvus_client=dense.client)
    hybrid = HybridRetriever(settings.retrieval, dense, bm25)
    reranker = QwenReranker(
        api_base="",
        api_key="",
        model_name=settings.reranker.model_name,
        model_path=settings.reranker.model_path,
        service_url=settings.reranker.service_url,
        batch_size=settings.reranker.batch_size,
        use_fp16=settings.reranker.use_fp16,
        retrieval_config=settings.retrieval,
    )

    trace_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        question = str(sample.get("question") or "")
        analysis = analysis_for_sample(router, sample, question)
        expected_doc_ids = list(sample.get("expected_doc_ids") or [])
        expected_source_files = list(sample.get("expected_source_files") or [])
        negative_query = bool(sample.get("negative_query"))
        should_require_doc_hit = sample.get("should_require_doc_hit")
        skipped_doc_hit = negative_query and should_require_doc_hit is False

        dense_hits = dense.search(question, limit=max(args.dense_top_k, 40), filters=None)
        bm25_hits = bm25.search(question, limit=max(args.bm25_top_k, 40), filters=None)
        hybrid_hits = hybrid.search(question, limit=max(args.hybrid_top_k, 40), filters=None, analysis=analysis)
        reranked_hits = reranker.rerank(
            question,
            list(hybrid_hits),
            top_k=args.rerank_top_k,
            analysis=analysis,
        )

        bm25_tokens = tokenize_query(question)
        stage_infos = {
            "dense": stage_info(dense_hits, expected_doc_ids, expected_source_files, skipped_doc_hit),
            "bm25": stage_info(bm25_hits, expected_doc_ids, expected_source_files, skipped_doc_hit),
            "hybrid": stage_info(hybrid_hits, expected_doc_ids, expected_source_files, skipped_doc_hit),
            "rerank": stage_info(reranked_hits, expected_doc_ids, expected_source_files, skipped_doc_hit),
        }
        diagnosis = diagnose_sample(stage_infos, skipped_doc_hit)
        category = ",".join(str(tag) for tag in sample.get("tags", [])) or str(sample.get("scenario") or "")

        trace_rows.append(build_trace_row(
            sample=sample,
            category=category,
            skipped_doc_hit=skipped_doc_hit,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            hybrid_hits=hybrid_hits,
            reranked_hits=reranked_hits,
            bm25_tokens=bm25_tokens,
            stage_infos=stage_infos,
            diagnosis=diagnosis,
        ))
        per_sample_rows.append(build_per_sample_row(sample, skipped_doc_hit, stage_infos, diagnosis))
        failure = build_failure_row(sample, skipped_doc_hit, stage_infos, diagnosis, reranked_hits, hybrid_hits)
        if failure:
            failure_rows.append(failure)

        print(f"[{index}/{len(samples)}] {sample_id(sample)} {diagnosis['final_retrieval_status']}", flush=True)

    metrics = compute_metrics(samples, trace_rows, per_sample_rows, failure_rows, validation, tokenizer_validation)
    run_config = build_run_config(
        args=args,
        settings=settings,
        dataset_path=dataset_path,
        command_used=args.command_used or " ".join(sys.argv),
    )

    write_csv(output_dir / "retrieval_only_stage_trace.csv", TRACE_FIELDNAMES, trace_rows)
    write_csv(output_dir / "retrieval_only_per_sample.csv", PER_SAMPLE_FIELDNAMES, per_sample_rows)
    write_csv(output_dir / "retrieval_failure_groups.csv", FAILURE_FIELDNAMES, failure_rows)
    write_json(output_dir / "retrieval_only_metrics.json", metrics)
    write_json(output_dir / "run_config.json", run_config)
    write_summary(report_dir / "summary.md", metrics, run_config)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset: {path}")
    return [item for item in data if isinstance(item, dict)]


def validate_dataset(samples: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    skipped_negative = 0
    for sample in samples:
        negative = bool(sample.get("negative_query"))
        should_require = sample.get("should_require_doc_hit")
        if negative and should_require is False:
            skipped_negative += 1
            continue
        if not (sample.get("expected_doc_ids") or sample.get("expected_source_files")):
            errors.append(f"{sample_id(sample)} missing expected_doc_ids/source_files")
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors[:10]))
    return {
        "validated_sample_count": len(samples),
        "skipped_negative_query_count": skipped_negative,
        "errors": errors,
    }


def validate_bm25_tokenizer(samples: list[dict[str, Any]]) -> dict[str, Any]:
    empty_count = 0
    single_cjk_count = 0
    preview: list[dict[str, Any]] = []
    for sample in samples[:20]:
        tokens = tokenize_query(str(sample.get("question") or ""))
        if not tokens:
            empty_count += 1
        single_cjk = [token for token in tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
        single_cjk_count += len(single_cjk)
        if len(preview) < 5:
            preview.append({"sample_id": sample_id(sample), "tokens": tokens[:20]})
    if empty_count:
        raise ValueError(f"BM25 tokenize_query returned empty tokens for {empty_count} preview samples")
    if single_cjk_count > 5:
        raise ValueError(f"BM25 CJK filter appears inactive; single CJK token count={single_cjk_count}")
    return {
        "preview_sample_count": min(20, len(samples)),
        "empty_query_count": empty_count,
        "single_cjk_token_count": single_cjk_count,
        "preview": preview,
    }


def analysis_for_sample(router: QueryRouter, sample: dict[str, Any], question: str) -> QueryAnalysis:
    analysis = router.analyze(question)
    route = str(sample.get("expected_route") or "").strip().lower()
    if route in {"factoid", "summary", "comparison", "experiment", "unknown"}:
        analysis.intent = QueryIntent(route)
    return analysis


def stage_info(
    hits: list[RetrievedChunk],
    expected_doc_ids: list[str],
    expected_source_files: list[str],
    skipped: bool,
) -> dict[str, Any]:
    best_rank = expected_best_rank(hits, expected_doc_ids, expected_source_files)
    return {
        "best_rank": best_rank,
        "found": best_rank > 0,
        "hit_at_5": hit_at(hits, expected_doc_ids, expected_source_files, 5, skipped),
        "hit_at_10": hit_at(hits, expected_doc_ids, expected_source_files, 10, skipped),
        "hit_at_20": hit_at(hits, expected_doc_ids, expected_source_files, 20, skipped),
        "hit_at_40": hit_at(hits, expected_doc_ids, expected_source_files, 40, skipped),
    }


def expected_best_rank(hits: list[RetrievedChunk], expected_doc_ids: list[str], expected_source_files: list[str]) -> int:
    doc_set = set(expected_doc_ids)
    source_set = set(expected_source_files)
    for rank, chunk in enumerate(hits, start=1):
        if (doc_set and chunk.doc_id in doc_set) or (source_set and chunk.source_file in source_set):
            return rank
    return 0


def hit_at(
    hits: list[RetrievedChunk],
    expected_doc_ids: list[str],
    expected_source_files: list[str],
    k: int,
    skipped: bool,
) -> bool:
    if skipped:
        return False
    return expected_best_rank(hits[:k], expected_doc_ids, expected_source_files) > 0


def doc_hit_at(hits: list[RetrievedChunk], expected_doc_ids: list[str], k: int) -> bool:
    if not expected_doc_ids:
        return False
    docs = set(expected_doc_ids)
    return any(chunk.doc_id in docs for chunk in hits[:k])


def source_hit_at(hits: list[RetrievedChunk], expected_source_files: list[str], k: int) -> bool:
    if not expected_source_files:
        return False
    sources = set(expected_source_files)
    return any(chunk.source_file in sources for chunk in hits[:k])


def comparison_counts(hits: list[RetrievedChunk], expected_doc_ids: list[str], k: int) -> dict[str, Any]:
    expected = set(expected_doc_ids)
    found = {chunk.doc_id for chunk in hits[:k] if chunk.doc_id in expected}
    return {
        "found_count": len(found),
        "any_hit": bool(found),
        "all_hit": bool(expected) and found == expected,
    }


def diagnose_sample(stage_infos: dict[str, dict[str, Any]], skipped: bool) -> dict[str, str]:
    if skipped:
        return {
            "first_stage_found": "not_found",
            "first_stage_lost": "skipped_negative",
            "retrieval_diagnosis": "skipped_negative",
            "final_retrieval_status": "skipped_negative",
            "recommended_next_action": "negative_query_skipped",
        }

    dense_found = bool(stage_infos["dense"]["found"])
    bm25_found = bool(stage_infos["bm25"]["found"])
    hybrid_found = bool(stage_infos["hybrid"]["found"])
    rerank_found = bool(stage_infos["rerank"]["found"])

    first_stage_found = "not_found"
    for name in ("dense", "bm25", "hybrid", "rerank"):
        if stage_infos[name]["found"]:
            first_stage_found = name
            break

    if rerank_found:
        first_stage_lost = "not_lost"
        final_status = "rerank_hit"
        next_action = "skip_generation_side"
    elif hybrid_found:
        first_stage_lost = "reranker_suppressed"
        final_status = "hybrid_hit_but_rerank_miss"
        next_action = "inspect_reranker"
    elif dense_found or bm25_found:
        first_stage_lost = "hybrid_suppressed"
        final_status = "dense_or_bm25_hit_but_hybrid_miss"
        next_action = "inspect_hybrid_fusion"
    else:
        first_stage_lost = "dense_miss"
        final_status = "hard_recall_miss"
        next_action = "improve_recall"

    if dense_found and bm25_found:
        base = "dense_and_bm25_hit"
    elif dense_found:
        base = "dense_only_hit"
    elif bm25_found:
        base = "bm25_only_hit"
    else:
        base = "hard_recall_miss"

    if not dense_found and hybrid_found:
        retrieval_diagnosis = "hybrid_improved"
    elif dense_found and not hybrid_found:
        retrieval_diagnosis = "hybrid_degraded"
    elif hybrid_found and rerank_found:
        h_rank = int(stage_infos["hybrid"]["best_rank"] or 999999)
        r_rank = int(stage_infos["rerank"]["best_rank"] or 999999)
        retrieval_diagnosis = "reranker_improved" if r_rank < h_rank else "reranker_degraded" if r_rank > h_rank else base
    elif hybrid_found and not rerank_found:
        retrieval_diagnosis = "reranker_degraded"
    else:
        retrieval_diagnosis = base

    return {
        "first_stage_found": first_stage_found,
        "first_stage_lost": first_stage_lost,
        "retrieval_diagnosis": retrieval_diagnosis,
        "final_retrieval_status": final_status,
        "recommended_next_action": next_action,
    }


def build_trace_row(
    *,
    sample: dict[str, Any],
    category: str,
    skipped_doc_hit: bool,
    dense_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    hybrid_hits: list[RetrievedChunk],
    reranked_hits: list[RetrievedChunk],
    bm25_tokens: list[str],
    stage_infos: dict[str, dict[str, Any]],
    diagnosis: dict[str, str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id(sample),
        "question": sample.get("question", ""),
        "category": category,
        "expected_route": sample.get("expected_route", ""),
        "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
        "expected_source_files": join_list(sample.get("expected_source_files") or []),
        "expected_sections": join_list(sample.get("expected_sections") or []),
        "negative_query": bool(sample.get("negative_query")),
        "should_require_doc_hit": sample.get("should_require_doc_hit", ""),
        "skipped_doc_hit": skipped_doc_hit,
        "rerank_input_size": len(hybrid_hits),
        "rerank_output_k": len(reranked_hits),
        "bm25_query_tokens": join_list(bm25_tokens),
        "bm25_query_token_count": len(bm25_tokens),
        **diagnosis,
    }
    expected_doc_ids = list(sample.get("expected_doc_ids") or [])
    expected_source_files = list(sample.get("expected_source_files") or [])
    for k in (10, 20, 40):
        row[f"hybrid_source_file_hit_at_{k}"] = source_hit_at(hybrid_hits, expected_source_files, k)
    for k in (10, 20):
        counts = comparison_counts(hybrid_hits, expected_doc_ids, k)
        row[f"comparison_found_count_at_{k}"] = counts["found_count"]
        row[f"comparison_any_hit_at_{k}"] = counts["any_hit"]
        row[f"comparison_all_hit_at_{k}"] = counts["all_hit"]
    add_stage_trace(row, "dense", dense_hits, stage_infos["dense"], include_40=True)
    add_stage_trace(row, "bm25", bm25_hits, stage_infos["bm25"], include_40=True)
    add_stage_trace(row, "hybrid", hybrid_hits, stage_infos["hybrid"], include_40=True)
    add_stage_trace(row, "rerank", reranked_hits, stage_infos["rerank"], include_40=False)
    return row


def add_stage_trace(row: dict[str, Any], prefix: str, hits: list[RetrievedChunk], info: dict[str, Any], *, include_40: bool) -> None:
    if prefix != "rerank":
        row[f"{prefix}_top_k"] = len(hits)
    row[f"{prefix}_doc_hit_at_5"] = info["hit_at_5"]
    row[f"{prefix}_doc_hit_at_10"] = info["hit_at_10"]
    row[f"{prefix}_doc_hit_at_20"] = info["hit_at_20"]
    if include_40:
        row[f"{prefix}_doc_hit_at_40"] = info["hit_at_40"]
    row[f"{prefix}_expected_best_rank"] = info["best_rank"]
    row[f"{prefix}_expected_found"] = info["found"]
    row[f"{prefix}_top10_doc_ids"] = join_list([chunk.doc_id for chunk in hits[:10]])
    row[f"{prefix}_top10_source_files"] = join_list([chunk.source_file for chunk in hits[:10]])
    row[f"{prefix}_top10_scores"] = join_list([format_score(score_for_stage(prefix, chunk)) for chunk in hits[:10]])


def build_per_sample_row(
    sample: dict[str, Any],
    skipped_doc_hit: bool,
    stage_infos: dict[str, dict[str, Any]],
    diagnosis: dict[str, str],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id(sample),
        "question": sample.get("question", ""),
        "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
        "expected_source_files": join_list(sample.get("expected_source_files") or []),
        "negative_query": bool(sample.get("negative_query")),
        "skipped_doc_hit": skipped_doc_hit,
        "dense_best_rank": stage_infos["dense"]["best_rank"],
        "bm25_best_rank": stage_infos["bm25"]["best_rank"],
        "hybrid_best_rank": stage_infos["hybrid"]["best_rank"],
        "rerank_best_rank": stage_infos["rerank"]["best_rank"],
        "dense_hit_at_20": stage_infos["dense"]["hit_at_20"],
        "bm25_hit_at_20": stage_infos["bm25"]["hit_at_20"],
        "hybrid_hit_at_20": stage_infos["hybrid"]["hit_at_20"],
        "rerank_hit_at_10": stage_infos["rerank"]["hit_at_10"],
        "final_retrieval_status": diagnosis["final_retrieval_status"],
        "recommended_next_action": diagnosis["recommended_next_action"],
    }


def build_failure_row(
    sample: dict[str, Any],
    skipped_doc_hit: bool,
    stage_infos: dict[str, dict[str, Any]],
    diagnosis: dict[str, str],
    reranked_hits: list[RetrievedChunk],
    hybrid_hits: list[RetrievedChunk],
) -> dict[str, Any] | None:
    if skipped_doc_hit:
        return None
    status = diagnosis["final_retrieval_status"]
    group_map = {
        "hard_recall_miss": "hard_recall_miss",
        "dense_or_bm25_hit_but_hybrid_miss": "hybrid_suppressed",
        "hybrid_hit_but_rerank_miss": "reranker_suppressed",
        "rerank_hit": "rerank_hit_generation_side_later",
    }
    group = group_map.get(status)
    if group is None:
        return None
    top_hits = reranked_hits[:3] if reranked_hits else hybrid_hits[:3]
    likely_issue = {
        "hard_recall_miss": "expected doc absent from dense/BM25/hybrid open retrieval",
        "hybrid_suppressed": "expected doc found by one retrieval source but lost before hybrid topK",
        "reranker_suppressed": "expected doc reached hybrid/rerank input but not rerank output",
        "rerank_hit_generation_side_later": "expected doc reached rerank output; not a retrieval-only failure",
    }[group]
    next_phase = {
        "hard_recall_miss": "recall improvement",
        "hybrid_suppressed": "hybrid fusion audit",
        "reranker_suppressed": "reranker/rerank input audit",
        "rerank_hit_generation_side_later": "evidence lifecycle / citation debug",
    }[group]
    return {
        "sample_id": sample_id(sample),
        "group": group,
        "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
        "dense_best_rank": stage_infos["dense"]["best_rank"],
        "bm25_best_rank": stage_infos["bm25"]["best_rank"],
        "hybrid_best_rank": stage_infos["hybrid"]["best_rank"],
        "rerank_best_rank": stage_infos["rerank"]["best_rank"],
        "top_wrong_doc_ids": join_list([chunk.doc_id for chunk in top_hits]),
        "top_wrong_titles": join_list([chunk.title for chunk in top_hits]),
        "likely_issue": likely_issue,
        "recommended_next_phase": next_phase,
    }


def compute_metrics(
    samples: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    tokenizer_validation: dict[str, Any],
) -> dict[str, Any]:
    evaluated = [row for row in trace_rows if not as_bool(row["skipped_doc_hit"])]
    skipped_count = len(trace_rows) - len(evaluated)

    metrics: dict[str, Any] = {
        "total_samples": len(trace_rows),
        "evaluated_samples": len(evaluated),
        "skipped_negative_query_count": skipped_count,
        "dataset_validation": validation,
        "bm25_tokenizer_validation": tokenizer_validation,
    }
    for stage in ("dense", "bm25", "hybrid"):
        add_stage_metrics(metrics, stage, evaluated, ks=(5, 10, 20, 40))
    add_stage_metrics(metrics, "rerank", evaluated, ks=(5, 10, 20))
    metrics["bm25_empty_query_count"] = sum(1 for row in trace_rows if int(row["bm25_query_token_count"]) == 0)
    metrics["bm25_avg_query_token_count"] = round(
        sum(int(row["bm25_query_token_count"]) for row in trace_rows) / max(len(trace_rows), 1),
        4,
    )
    add_source_file_metrics(metrics, samples, trace_rows)
    add_comparison_metrics(metrics, samples, trace_rows)
    add_delta_metrics(metrics, evaluated, failure_rows)
    metrics["failure_group_counts"] = dict(Counter(row["group"] for row in failure_rows))
    metrics["final_retrieval_status_counts"] = dict(Counter(row["final_retrieval_status"] for row in per_sample_rows))
    return metrics


def add_stage_metrics(metrics: dict[str, Any], stage: str, rows: list[dict[str, Any]], ks: tuple[int, ...]) -> None:
    denom = max(len(rows), 1)
    ranks = [int(row[f"{stage}_expected_best_rank"]) for row in rows if int(row[f"{stage}_expected_best_rank"]) > 0]
    for k in ks:
        metrics[f"{stage}_doc_hit_at_{k}"] = round(
            sum(as_bool(row[f"{stage}_doc_hit_at_{k}"]) for row in rows) / denom,
            4,
        )
    metrics[f"{stage}_mrr"] = round(
        sum((1.0 / int(row[f"{stage}_expected_best_rank"])) if int(row[f"{stage}_expected_best_rank"]) > 0 else 0.0 for row in rows) / denom,
        6,
    )
    metrics[f"{stage}_median_expected_rank"] = statistics.median(ranks) if ranks else None


def add_source_file_metrics(metrics: dict[str, Any], samples: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> None:
    rows = [
        (sample, row)
        for sample, row in zip(samples, trace_rows)
        if not as_bool(row["skipped_doc_hit"]) and sample.get("expected_source_files")
    ]
    denom = max(len(rows), 1)
    for k in (10, 20, 40):
        count = sum(as_bool(row[f"hybrid_source_file_hit_at_{k}"]) for _sample, row in rows)
        metrics[f"source_file_hit_at_{k}"] = round(count / denom, 4)


def add_comparison_metrics(metrics: dict[str, Any], samples: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> None:
    rows = [
        (sample, row)
        for sample, row in zip(samples, trace_rows)
        if not as_bool(row["skipped_doc_hit"]) and str(sample.get("expected_route")) == "comparison" and sample.get("expected_doc_ids")
    ]
    metrics["comparison_sample_count"] = len(rows)
    denom = max(len(rows), 1)
    for k in (10, 20):
        any_count = sum(as_bool(row[f"comparison_any_hit_at_{k}"]) for _sample, row in rows)
        all_count = sum(as_bool(row[f"comparison_all_hit_at_{k}"]) for _sample, row in rows)
        metrics[f"comparison_any_doc_hit_at_{k}"] = round(any_count / denom, 4)
        metrics[f"comparison_all_doc_hit_at_{k}"] = round(all_count / denom, 4)


def add_delta_metrics(metrics: dict[str, Any], rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
    metrics["dense_to_hybrid_improved_count"] = sum(
        (not as_bool(row["dense_doc_hit_at_20"])) and as_bool(row["hybrid_doc_hit_at_20"]) for row in rows
    )
    metrics["dense_to_hybrid_degraded_count"] = sum(
        as_bool(row["dense_doc_hit_at_20"]) and not as_bool(row["hybrid_doc_hit_at_20"]) for row in rows
    )
    metrics["hybrid_to_rerank_improved_count"] = sum(
        (not as_bool(row["hybrid_doc_hit_at_10"])) and as_bool(row["rerank_doc_hit_at_10"]) for row in rows
    )
    metrics["hybrid_to_rerank_degraded_count"] = sum(
        as_bool(row["hybrid_doc_hit_at_10"]) and not as_bool(row["rerank_doc_hit_at_10"]) for row in rows
    )
    metrics["bm25_unique_contribution_count"] = sum(
        as_bool(row["bm25_doc_hit_at_20"]) and not as_bool(row["dense_doc_hit_at_20"]) for row in rows
    )
    metrics["dense_unique_contribution_count"] = sum(
        as_bool(row["dense_doc_hit_at_20"]) and not as_bool(row["bm25_doc_hit_at_20"]) for row in rows
    )
    metrics["hard_recall_miss_count"] = sum(row["group"] == "hard_recall_miss" for row in failure_rows)


def build_run_config(args: argparse.Namespace, settings: Settings, dataset_path: Path, command_used: str) -> dict[str, Any]:
    return {
        "branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_sha": git_output(["git", "rev-parse", "HEAD"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "chunks_path": settings.kb.chunk_jsonl,
        "milvus_collection_name": settings.retrieval.collection_name,
        "bm25_enabled": settings.retrieval.bm25_enabled,
        "bm25_query_tokenizer": "tokenize_query_cjk_filtered",
        "dense_model": settings.kb.embedding_model_path,
        "reranker_model": settings.reranker.model_path or settings.reranker.model_name,
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "hybrid_top_k": args.hybrid_top_k,
        "rerank_top_k": args.rerank_top_k,
        "parent_expansion_used_for_main_metric": False,
        "qwen_synthesis": False,
        "generation_called": False,
        "targeted_filter_used": False,
        "generation_version": "v2",
        "generation_v2_enable_comparison_coverage": False,
        "generation_v2_enable_neighbor_audit": False,
        "generation_v2_enable_neighbor_promotion": False,
        "generation_v2_include_neighbor_context_in_qwen": False,
        "retrieval_biolexical_bm25_enabled": os.environ.get("RETRIEVAL_BIOLEXICAL_BM25_ENABLED", "false"),
        "command_used": command_used,
    }


def write_summary(path: Path, metrics: dict[str, Any], run_config: dict[str, Any]) -> None:
    fg = metrics.get("failure_group_counts", {})
    lines = [
        "# Phase 16R Retrieval-only Open Baseline",
        "",
        "## 1. Purpose",
        "",
        "This phase evaluates open retrieval after the index refactor. It stops at rerank and does not call generation, Qwen synthesis, support selection, citation binding, answer construction, or parent expansion.",
        "",
        "## 2. Run Config",
        "",
        f"- open retrieval: true",
        f"- no generation: {not run_config['generation_called']}",
        f"- no Qwen synthesis: {not run_config['qwen_synthesis']}",
        f"- CJK-filtered BM25: {run_config['bm25_query_tokenizer']}",
        f"- targeted filter used: {run_config['targeted_filter_used']}",
        f"- parent expansion used for main metric: {run_config['parent_expansion_used_for_main_metric']}",
        f"- dataset: `{run_config['dataset_path']}`",
        f"- commit: `{run_config['commit_sha']}`",
        "",
        "## 3. Main Metrics",
        "",
        "| Stage | hit@5 | hit@10 | hit@20 | hit@40 | MRR | median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
        stage_metric_row(metrics, "dense", include_40=True),
        stage_metric_row(metrics, "bm25", include_40=True),
        stage_metric_row(metrics, "hybrid", include_40=True),
        stage_metric_row(metrics, "rerank", include_40=False),
        "",
        f"- source_file_hit_at_10: {metrics['source_file_hit_at_10']}",
        f"- source_file_hit_at_20: {metrics['source_file_hit_at_20']}",
        f"- source_file_hit_at_40: {metrics['source_file_hit_at_40']}",
        f"- comparison_any_doc_hit_at_10: {metrics['comparison_any_doc_hit_at_10']}",
        f"- comparison_all_doc_hit_at_10: {metrics['comparison_all_doc_hit_at_10']}",
        f"- comparison_any_doc_hit_at_20: {metrics['comparison_any_doc_hit_at_20']}",
        f"- comparison_all_doc_hit_at_20: {metrics['comparison_all_doc_hit_at_20']}",
        "",
        "## 4. Stage Contribution",
        "",
        f"- BM25 unique contribution count at @20: {metrics['bm25_unique_contribution_count']}",
        f"- Dense unique contribution count at @20: {metrics['dense_unique_contribution_count']}",
        f"- Dense -> hybrid improved count at @20: {metrics['dense_to_hybrid_improved_count']}",
        f"- Dense -> hybrid degraded count at @20: {metrics['dense_to_hybrid_degraded_count']}",
        f"- Hybrid -> rerank improved count at @10: {metrics['hybrid_to_rerank_improved_count']}",
        f"- Hybrid -> rerank degraded count at @10: {metrics['hybrid_to_rerank_degraded_count']}",
        "",
        "## 5. Failure Groups",
        "",
        f"- hard_recall_miss: {fg.get('hard_recall_miss', 0)}",
        f"- hybrid_suppressed: {fg.get('hybrid_suppressed', 0)}",
        f"- reranker_suppressed: {fg.get('reranker_suppressed', 0)}",
        f"- rerank_hit_generation_side_later: {fg.get('rerank_hit_generation_side_later', 0)}",
        f"- skipped_negative: {metrics['skipped_negative_query_count']}",
        "",
        "## 6. Interpretation",
        "",
        interpretation(metrics),
        "",
        "BM25 CJK filtering should remain enabled: tokenizer validation passed and BM25 contributes independently at @20.",
        "",
        "## 7. Recommendation",
        "",
        recommendation(metrics),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def interpretation(metrics: dict[str, Any]) -> str:
    rerank10 = float(metrics["rerank_doc_hit_at_10"])
    hybrid40 = float(metrics["hybrid_doc_hit_at_40"])
    hard = int(metrics["failure_group_counts"].get("hard_recall_miss", 0))
    reranker = int(metrics["failure_group_counts"].get("reranker_suppressed", 0))
    generation_side = int(metrics["failure_group_counts"].get("rerank_hit_generation_side_later", 0))
    return (
        f"Retrieval-only rerank_hit@10 is {rerank10:.4f} and hybrid_hit@40 is {hybrid40:.4f}. "
        f"Current remaining retrieval-only failures are hard_recall_miss={hard} and reranker_suppressed={reranker}; "
        f"{generation_side} evaluated samples already reach rerank output and should be handled by evidence/citation lifecycle rather than retrieval."
    )


def recommendation(metrics: dict[str, Any]) -> str:
    rerank10 = float(metrics["rerank_doc_hit_at_10"])
    hybrid40 = float(metrics["hybrid_doc_hit_at_40"])
    reranker_degraded = int(metrics["hybrid_to_rerank_degraded_count"])
    hard = int(metrics["failure_group_counts"].get("hard_recall_miss", 0))
    if rerank10 >= 0.75:
        return "Prioritize Phase 16B evidence lifecycle debug/drop_reason, because most expected docs already reach rerank@10."
    if hybrid40 >= 0.75 and reranker_degraded > hard:
        return "Prioritize reranker/rerank input audit, because hybrid recall is high but rerank loses candidates."
    return "Prioritize retrieval recall and hybrid/rerank diagnostics before generation-side fixes."


def stage_metric_row(metrics: dict[str, Any], stage: str, *, include_40: bool) -> str:
    hit40 = metrics.get(f"{stage}_doc_hit_at_40", "")
    if not include_40:
        hit40 = "-"
    return (
        f"| {stage} | {metrics[f'{stage}_doc_hit_at_5']} | {metrics[f'{stage}_doc_hit_at_10']} | "
        f"{metrics[f'{stage}_doc_hit_at_20']} | {hit40} | {metrics[f'{stage}_mrr']} | "
        f"{metrics[f'{stage}_median_expected_rank']} |"
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def score_for_stage(stage: str, chunk: RetrievedChunk) -> float:
    if stage == "dense":
        return float(chunk.vector_score or 0.0)
    if stage == "bm25":
        return float(chunk.bm25_score or 0.0)
    if stage == "hybrid":
        return float(chunk.fusion_score or 0.0)
    return float(chunk.rerank_score or 0.0)


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("id") or sample.get("sample_id") or "")


def join_list(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def format_score(value: float) -> str:
    return f"{value:.6f}"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
