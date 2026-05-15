#!/usr/bin/env python3
"""Promote Phase 5C-5 baseline_full assets into a durable clean baseline.

This script intentionally avoids cleaning, chunking, BM25 rebuilding, retrieval
logic changes, generation evaluation, Qwen, RAGAS, OCR, and production indexes.
It copies verified chunks/BM25 assets, imports Milvus from the copied chunks,
and writes validation reports for Phase 5F-4C.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymilvus import MilvusClient

from scripts.ingestion import import_to_milvus as milvus_import
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import QueryFilters, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever


EXPECTED_DATASET_SHA256 = (
    "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
)
EXPECTED_DATASET_COUNT = 90
EXPECTED_QUERY_TYPE_DISTRIBUTION = {
    "table_content": 31,
    "caption_level_table": 9,
    "figure_caption": 20,
    "normal_control": 30,
}
EXPECTED_CHUNK_COUNT = 15802
EXPECTED_BM25_RECORDS = 15802
EXPECTED_VECTOR_DIM = 1024
REFERENCE_PHASE5F4_STABLE_AT_10 = 0.9555555555555556
REFERENCE_PHASE5F4_DOC_AT_10 = 0.9555555555555556

DATASET_PATH = (
    ROOT
    / "reports"
    / "phase5f_eval_semantic_enhancement_v2"
    / "strict_main_eval_set_v2.jsonl"
)
SOURCE_CHUNKS = Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
SOURCE_BM25 = Path("/tmp/biorag_phase5c5_baseline_full/bm25_index.json")

DURABLE_ROOT = ROOT / "data" / "baselines" / "phase5f_official_clean_baseline"
DURABLE_DATASET_DIR = DURABLE_ROOT / "dataset"
DURABLE_CHUNKS_DIR = DURABLE_ROOT / "chunks"
DURABLE_BM25_DIR = DURABLE_ROOT / "bm25"
DURABLE_MILVUS_DIR = DURABLE_ROOT / "milvus"
DURABLE_REPORTS_VALIDATION_DIR = DURABLE_ROOT / "reports" / "validation"
DURABLE_CONFIG_DIR = DURABLE_ROOT / "config_snapshot"
DURABLE_CHUNKS = DURABLE_CHUNKS_DIR / "chunks.jsonl"
DURABLE_BM25 = DURABLE_BM25_DIR / "bm25_index.json"
DURABLE_MILVUS_URI = DURABLE_MILVUS_DIR / "milvus_lite.db"

REPORT_DIR = ROOT / "reports" / "phase5f4c_official_baseline_promotion"
COLLECTION_NAME = "synbio_phase5f_official_clean_baseline"
BASELINE_NAME = "phase5f_official_clean_baseline"
QUERY_TYPES = ("table_content", "caption_level_table", "figure_caption", "normal_control")

FORBIDDEN_COLLECTIONS = {
    "synbio_papers",
    "current_default",
    "synbio_phase5c5_baseline_full",
}


@dataclass
class ChunkIndex:
    chunks: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    by_doc: dict[str, list[str]]
    by_doc_block: dict[tuple[str, str], list[str]]
    retrieval_nonempty: dict[str, bool]


class Blocker(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5F-4C official clean baseline promotion."
    )
    parser.add_argument("--source-chunks", default=str(SOURCE_CHUNKS))
    parser.add_argument("--source-bm25", default=str(SOURCE_BM25))
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--durable-root", default=str(DURABLE_ROOT))
    parser.add_argument("--milvus-uri", default=str(DURABLE_MILVUS_URI))
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--model-path", default=str(ROOT / "models" / "BAAI" / "bge-m3"))
    parser.add_argument("--import-embed-max-length", type=int, default=4096)
    parser.add_argument("--retrieval-embed-max-length", type=int, default=512)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.collection_name in FORBIDDEN_COLLECTIONS:
        raise Blocker(f"refusing forbidden collection name: {args.collection_name}")

    global DURABLE_ROOT
    global DURABLE_DATASET_DIR
    global DURABLE_CHUNKS_DIR
    global DURABLE_BM25_DIR
    global DURABLE_MILVUS_DIR
    global DURABLE_REPORTS_VALIDATION_DIR
    global DURABLE_CONFIG_DIR
    global DURABLE_CHUNKS
    global DURABLE_BM25

    DURABLE_ROOT = Path(args.durable_root)
    if not DURABLE_ROOT.is_absolute():
        DURABLE_ROOT = ROOT / DURABLE_ROOT
    DURABLE_DATASET_DIR = DURABLE_ROOT / "dataset"
    DURABLE_CHUNKS_DIR = DURABLE_ROOT / "chunks"
    DURABLE_BM25_DIR = DURABLE_ROOT / "bm25"
    DURABLE_MILVUS_DIR = DURABLE_ROOT / "milvus"
    DURABLE_REPORTS_VALIDATION_DIR = DURABLE_ROOT / "reports" / "validation"
    DURABLE_CONFIG_DIR = DURABLE_ROOT / "config_snapshot"
    DURABLE_CHUNKS = DURABLE_CHUNKS_DIR / "chunks.jsonl"
    DURABLE_BM25 = DURABLE_BM25_DIR / "bm25_index.json"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_durable_dirs()

    source_chunks = Path(args.source_chunks)
    source_bm25 = Path(args.source_bm25)
    dataset_path = Path(args.dataset)
    milvus_uri = Path(args.milvus_uri)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not milvus_uri.is_absolute():
        milvus_uri = ROOT / milvus_uri

    print("[1/8] Source validation")
    source_validation = validate_sources(dataset_path, source_chunks, source_bm25)
    write_source_validation(source_validation)
    ensure_no_blockers(source_validation, "source validation")

    print("[2/8] Copy chunks and BM25 to durable path")
    copy_summary = copy_assets(source_validation, source_chunks, source_bm25)
    write_copy_summary(copy_summary)
    ensure_no_blockers(copy_summary, "copy")

    chunks = load_jsonl(DURABLE_CHUNKS)
    chunk_index = build_chunk_index(chunks)
    samples = load_jsonl(dataset_path)
    bm25_payload = load_json(DURABLE_BM25)
    bm25_ids = bm25_chunk_ids(bm25_payload)

    print("[3/8] Milvus-only durable import")
    import_summary = ensure_milvus_collection(
        chunks=chunks,
        chunk_ids=set(chunk_index.by_id),
        milvus_uri=milvus_uri,
        collection_name=args.collection_name,
        model_path=Path(args.model_path),
        embed_max_length=args.import_embed_max_length,
    )
    write_milvus_import_summary(import_summary)
    ensure_no_blockers(import_summary, "milvus import")

    print("[4/8] Official manifest")
    manifest = build_manifest(
        dataset_path=dataset_path,
        dataset_info=source_validation["dataset"],
        chunk_info=copy_summary["chunks"],
        bm25_info=copy_summary["bm25"],
        import_summary=import_summary,
        milvus_uri=milvus_uri,
        collection_name=args.collection_name,
    )
    write_manifest_and_readme(manifest)

    print("[5/8] Target coverage validation")
    milvus_rows = load_milvus_rows(milvus_uri, args.collection_name)
    coverage_rows, coverage_summary = build_coverage(
        samples=samples,
        chunk_index=chunk_index,
        bm25_ids=bm25_ids,
        milvus_rows=milvus_rows,
    )
    write_coverage_reports(coverage_rows, coverage_summary)

    print("[6/8] Retrieval-only official baseline validation")
    retrieval = run_retrieval_validation(
        samples=samples,
        chunk_index=chunk_index,
        bm25_payload=bm25_payload,
        milvus_uri=milvus_uri,
        collection_name=args.collection_name,
        model_path=Path(args.model_path),
        retrieval_embed_max_length=args.retrieval_embed_max_length,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
    )
    write_retrieval_reports(retrieval)

    print("[7/8] Baseline registry")
    registry_path = write_baseline_registry(manifest)
    registry_summary = {"path": rel(registry_path), "written": registry_path.exists()}

    print("[8/8] Final validation and closeout decision")
    final_validation = build_final_validation(
        source_validation=source_validation,
        copy_summary=copy_summary,
        import_summary=import_summary,
        manifest=manifest,
        coverage_summary=coverage_summary,
        retrieval=retrieval,
        registry_summary=registry_summary,
    )
    write_final_reports(final_validation)

    if not final_validation["can_enter_phase5f5_closeout"]:
        raise Blocker("Phase 5F-5 closeout is blocked; see closeout_decision.md")

    print("Phase 5F-4C promotion completed.")


def prepare_durable_dirs() -> None:
    for path in (
        DURABLE_DATASET_DIR,
        DURABLE_CHUNKS_DIR,
        DURABLE_BM25_DIR,
        DURABLE_MILVUS_DIR,
        DURABLE_REPORTS_VALIDATION_DIR,
        DURABLE_CONFIG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_sources(dataset_path: Path, chunks_path: Path, bm25_path: Path) -> dict[str, Any]:
    dataset = scan_dataset(dataset_path)
    chunks = scan_chunks(chunks_path)
    bm25 = scan_bm25(bm25_path, set(chunks.get("chunk_ids", [])))
    blockers: list[str] = []

    if not dataset.get("exists"):
        blockers.append("dataset missing")
    if dataset.get("sha256") != EXPECTED_DATASET_SHA256:
        blockers.append("dataset SHA mismatch")
    if dataset.get("count") != EXPECTED_DATASET_COUNT:
        blockers.append("dataset count mismatch")
    if dataset.get("query_type_distribution") != EXPECTED_QUERY_TYPE_DISTRIBUTION:
        blockers.append("dataset query_type distribution mismatch")
    if not chunks.get("exists"):
        blockers.append("chunks file missing")
    if chunks.get("line_count") != EXPECTED_CHUNK_COUNT:
        blockers.append("chunks count mismatch")
    if chunks.get("duplicate_chunk_id_count"):
        blockers.append("chunks chunk_id not unique")
    if chunks.get("empty_doc_id_count"):
        blockers.append("chunks doc_id empty")
    if chunks.get("table_enhancement_metadata_chunk_count"):
        blockers.append("chunks contain table enhancement metadata")
    if chunks.get("caption_cleanup_metadata_chunk_count"):
        blockers.append("chunks contain caption cleanup metadata")
    if chunks.get("parse_errors"):
        blockers.append("chunks JSONL parse errors")
    if not bm25.get("exists"):
        blockers.append("BM25 file missing")
    if bm25.get("record_count") != EXPECTED_BM25_RECORDS:
        blockers.append("BM25 record count mismatch")
    if not bm25.get("chunk_id_set_matches_chunks"):
        blockers.append("BM25 chunk_id set does not match chunks")
    if bm25.get("parse_error"):
        blockers.append("BM25 JSON parse error")

    return {
        "generated_at": utc_now(),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "dataset": dataset,
        "chunks": strip_large_sets(chunks),
        "bm25": strip_large_sets(bm25),
        "source_files_readable": bool(
            dataset.get("readable") and chunks.get("readable") and bm25.get("readable")
        ),
        "migration_should_stop": bool(blockers),
    }


def scan_dataset(path: Path) -> dict[str, Any]:
    info = file_info(path)
    info["expected_sha256"] = EXPECTED_DATASET_SHA256
    if not path.exists():
        return info
    samples: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_no}: {exc}")
                    continue
                samples.append(sample)
    except Exception as exc:
        info["read_error"] = f"{type(exc).__name__}: {exc}"
        return info

    query_types = Counter(str(sample.get("query_type", "")) for sample in samples)
    info.update(
        {
            "readable": True,
            "sha256": sha256_file(path),
            "sha256_matches_expected": sha256_file(path) == EXPECTED_DATASET_SHA256,
            "count": len(samples),
            "query_type_distribution": dict(query_types),
            "stable_target_block_ids_nonempty": sum(1 for item in samples if stable_ids(item)),
            "target_doc_id_nonempty": sum(
                1 for item in samples if str(item.get("target_doc_id") or "").strip()
            ),
            "include_in_main_denominator_true": sum(
                1 for item in samples if item.get("include_in_main_denominator") is True
            ),
            "diagnostic_or_lexical_stress_count": sum(
                1
                for item in samples
                if str(item.get("ability_scope") or "").lower()
                in {"diagnostic", "lexical_stress", "lexical-stress"}
            ),
            "parse_errors": parse_errors,
        }
    )
    return info


def scan_chunks(path: Path) -> dict[str, Any]:
    info = file_info(path)
    if not path.exists():
        return info
    chunk_ids: set[str] = set()
    duplicate_ids: list[str] = []
    doc_ids: set[str] = set()
    empty_doc_id_count = 0
    parse_errors: list[str] = []
    content_shape_counts: Counter[str] = Counter()
    parser_stage_counts: Counter[str] = Counter()
    field_sets: Counter[str] = Counter()
    retrieval_text_empty_count = 0
    table_enhancement_count = 0
    caption_cleanup_count = 0
    max_retrieval_text_chars = 0
    max_text_chars = 0
    source_block_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_no}: {exc}")
                    continue

                chunk_id = str(item.get("chunk_id") or "")
                doc_id = str(item.get("doc_id") or "")
                if chunk_id in chunk_ids:
                    duplicate_ids.append(chunk_id)
                if chunk_id:
                    chunk_ids.add(chunk_id)
                if doc_id:
                    doc_ids.add(doc_id)
                else:
                    empty_doc_id_count += 1

                retrieval_text = str(item.get("retrieval_text") or "")
                text = str(item.get("text") or "")
                if not retrieval_text.strip():
                    retrieval_text_empty_count += 1
                max_retrieval_text_chars = max(max_retrieval_text_chars, len(retrieval_text))
                max_text_chars = max(max_text_chars, len(text))
                if item.get("contains_table_text"):
                    content_shape_counts["table_text"] += 1
                elif item.get("contains_table_caption"):
                    content_shape_counts["table_caption"] += 1
                elif item.get("contains_figure_caption"):
                    content_shape_counts["figure_caption"] += 1
                elif item.get("contains_image"):
                    content_shape_counts["image_related"] += 1
                elif item.get("contains_references"):
                    content_shape_counts["references"] += 1
                else:
                    content_shape_counts["body_or_other"] += 1
                if item.get("parser_stage"):
                    parser_stage_counts[str(item.get("parser_stage"))] += 1
                field_sets["|".join(sorted(item.keys()))] += 1
                if has_table_enhancement_metadata(item):
                    table_enhancement_count += 1
                if has_caption_cleanup_metadata(item):
                    caption_cleanup_count += 1
                for block_id in chunk_block_ids(item):
                    if doc_id and chunk_id:
                        source_block_index[(doc_id, block_id)].append(chunk_id)
    except Exception as exc:
        info["read_error"] = f"{type(exc).__name__}: {exc}"
        return info

    info.update(
        {
            "readable": True,
            "sha256": sha256_file(path),
            "line_count": len(chunk_ids) + len(duplicate_ids),
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "duplicate_chunk_id_count": len(duplicate_ids),
            "duplicate_chunk_ids_sample": duplicate_ids[:20],
            "doc_id_count": len(doc_ids),
            "empty_doc_id_count": empty_doc_id_count,
            "retrieval_text_empty_count": retrieval_text_empty_count,
            "max_retrieval_text_chars": max_retrieval_text_chars,
            "max_text_chars": max_text_chars,
            "content_shape_counts": dict(content_shape_counts),
            "parser_stage_counts": dict(parser_stage_counts),
            "field_set_count": len(field_sets),
            "field_sets_top10": [
                {"field_set": field_set, "count": count}
                for field_set, count in field_sets.most_common(10)
            ],
            "table_enhancement_metadata_chunk_count": table_enhancement_count,
            "caption_cleanup_metadata_chunk_count": caption_cleanup_count,
            "is_non_enhanced_baseline": table_enhancement_count == 0 and caption_cleanup_count == 0,
            "parse_errors": parse_errors,
        }
    )
    return info


def scan_bm25(path: Path, expected_chunk_ids: set[str]) -> dict[str, Any]:
    info = file_info(path)
    if not path.exists():
        return info
    try:
        payload = load_json(path)
    except Exception as exc:
        info.update({"readable": False, "parse_error": f"{type(exc).__name__}: {exc}"})
        return info

    records = payload.get("records", []) if isinstance(payload, dict) else []
    record_ids: set[str] = set()
    duplicate_ids: list[str] = []
    missing_chunk_id_records = 0
    for record in records:
        chunk_id = str(record.get("chunk_id") or "") if isinstance(record, dict) else ""
        if not chunk_id:
            missing_chunk_id_records += 1
            continue
        if chunk_id in record_ids:
            duplicate_ids.append(chunk_id)
        record_ids.add(chunk_id)

    info.update(
        {
            "readable": True,
            "sha256": sha256_file(path),
            "record_count": len(records),
            "chunk_ids": record_ids,
            "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "doc_len_count": len(payload.get("doc_len", [])) if isinstance(payload, dict) else 0,
            "term_freqs_count": len(payload.get("term_freqs", [])) if isinstance(payload, dict) else 0,
            "doc_freq_count": len(payload.get("doc_freq", {})) if isinstance(payload, dict) else 0,
            "avgdl": payload.get("avgdl") if isinstance(payload, dict) else None,
            "duplicate_chunk_id_count": len(duplicate_ids),
            "duplicate_chunk_ids_sample": duplicate_ids[:20],
            "missing_chunk_id_records": missing_chunk_id_records,
            "chunk_id_set_matches_chunks": bool(expected_chunk_ids) and record_ids == expected_chunk_ids,
            "records_missing_from_chunks_count": len(record_ids - expected_chunk_ids),
            "chunks_missing_from_records_count": len(expected_chunk_ids - record_ids),
            "records_missing_from_chunks_sample": sorted(record_ids - expected_chunk_ids)[:20],
            "chunks_missing_from_records_sample": sorted(expected_chunk_ids - record_ids)[:20],
        }
    )
    return info


def copy_assets(
    source_validation: dict[str, Any],
    source_chunks: Path,
    source_bm25: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    copied: list[str] = []
    reused: list[str] = []

    chunks_result = copy_one_asset(source_chunks, DURABLE_CHUNKS, "chunks")
    bm25_result = copy_one_asset(source_bm25, DURABLE_BM25, "bm25")
    for result in (chunks_result, bm25_result):
        if result.get("status") == "conflict":
            blockers.append(result["blocker"])
        if result.get("copied"):
            copied.append(result["kind"])
        if result.get("reused"):
            reused.append(result["kind"])

    if not blockers:
        chunk_stats = scan_chunks(DURABLE_CHUNKS)
        bm25_stats = scan_bm25(DURABLE_BM25, set(chunk_stats.get("chunk_ids", [])))
        if chunk_stats.get("line_count") != EXPECTED_CHUNK_COUNT:
            blockers.append("durable chunks count mismatch after copy")
        if bm25_stats.get("record_count") != EXPECTED_BM25_RECORDS:
            blockers.append("durable BM25 record count mismatch after copy")
        if not bm25_stats.get("chunk_id_set_matches_chunks"):
            blockers.append("durable BM25 chunk ids do not match durable chunks")
        write_text(DURABLE_CHUNKS_DIR / "chunks.sha256.txt", f"{chunk_stats['sha256']}  chunks.jsonl\n")
        write_text(DURABLE_BM25_DIR / "bm25.sha256.txt", f"{bm25_stats['sha256']}  bm25_index.json\n")
        write_json(DURABLE_CHUNKS_DIR / "chunk_stats.json", strip_large_sets(chunk_stats))
        write_json(
            DURABLE_BM25_DIR / "bm25_manifest.json",
            {
                "path": rel(DURABLE_BM25),
                "source_path": str(source_bm25),
                "sha256": bm25_stats.get("sha256"),
                "record_count": bm25_stats.get("record_count"),
                "chunk_id_set_matches_chunks": bm25_stats.get("chunk_id_set_matches_chunks"),
                "top_level_keys": bm25_stats.get("top_level_keys"),
                "doc_len_count": bm25_stats.get("doc_len_count"),
                "term_freqs_count": bm25_stats.get("term_freqs_count"),
                "doc_freq_count": bm25_stats.get("doc_freq_count"),
                "avgdl": bm25_stats.get("avgdl"),
                "bm25_rebuilt": False,
                "created_at": utc_now(),
            },
        )
    else:
        chunk_stats = {}
        bm25_stats = {}

    return {
        "generated_at": utc_now(),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "copied": copied,
        "reused": reused,
        "source_validation_status": source_validation.get("status"),
        "chunks": {
            **chunks_result,
            "sha256": chunk_stats.get("sha256"),
            "line_count": chunk_stats.get("line_count"),
            "chunk_count": chunk_stats.get("chunk_count"),
            "doc_id_count": chunk_stats.get("doc_id_count"),
            "table_enhancement_metadata_chunk_count": chunk_stats.get(
                "table_enhancement_metadata_chunk_count"
            ),
            "caption_cleanup_metadata_chunk_count": chunk_stats.get(
                "caption_cleanup_metadata_chunk_count"
            ),
        },
        "bm25": {
            **bm25_result,
            "sha256": bm25_stats.get("sha256"),
            "record_count": bm25_stats.get("record_count"),
            "chunk_id_set_matches_chunks": bm25_stats.get("chunk_id_set_matches_chunks"),
        },
    }


def copy_one_asset(source: Path, destination: Path, kind: str) -> dict[str, Any]:
    source_sha = sha256_file(source)
    result: dict[str, Any] = {
        "kind": kind,
        "source_path": str(source),
        "destination_path": rel(destination),
        "source_sha256": source_sha,
        "destination_exists_before": destination.exists(),
        "copied": False,
        "reused": False,
        "status": "pending",
    }
    if destination.exists():
        dest_sha = sha256_file(destination)
        result["destination_sha256_before"] = dest_sha
        if dest_sha == source_sha:
            result.update({"status": "reused", "reused": True})
            return result
        result.update(
            {
                "status": "conflict",
                "blocker": f"{kind} destination exists with different hash: {destination}",
            }
        )
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    result.update(
        {
            "copied": True,
            "status": "copied",
            "destination_sha256_after": sha256_file(destination),
        }
    )
    return result


def ensure_milvus_collection(
    chunks: list[dict[str, Any]],
    chunk_ids: set[str],
    milvus_uri: Path,
    collection_name: str,
    model_path: Path,
    embed_max_length: int,
) -> dict[str, Any]:
    start = time.time()
    milvus_uri.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "python scripts/ingestion/import_to_milvus.py "
        f"--jsonl {rel(DURABLE_CHUNKS)} "
        f"--collection_name {collection_name} "
        f"--milvus_uri {rel(milvus_uri)} "
        f"--embedding bge-m3 --model_path {rel(model_path)} "
        f"--dim {EXPECTED_VECTOR_DIM} --embed-max-length {embed_max_length}"
    )
    write_text(DURABLE_CONFIG_DIR / "import_command.txt", command + "\n")
    write_text(
        DURABLE_CONFIG_DIR / "environment_notes.md",
        "\n".join(
            [
                "# Environment notes",
                "",
                f"- generated_at: `{utc_now()}`",
                f"- model_path: `{rel(model_path)}`",
                f"- import_embedding_max_length: `{embed_max_length}`",
                f"- milvus_uri: `{rel(milvus_uri)}`",
                f"- collection: `{collection_name}`",
                "- chunks_regenerated: no",
                "- bm25_rebuilt: no",
                "- qwen_called: no",
                "- ragas_run: no",
                "- production_index_overwritten: no",
            ]
        )
        + "\n",
    )

    client = MilvusClient(str(milvus_uri.resolve()))
    collections = client.list_collections()
    reused = False
    imported = False

    if collection_name in collections:
        existing = inspect_milvus_collection(client, collection_name, chunk_ids)
        blockers = []
        if existing.get("row_count") != EXPECTED_CHUNK_COUNT:
            blockers.append("existing durable collection row count mismatch")
        if not existing.get("schema_compatible"):
            blockers.append("existing durable collection schema mismatch")
        if not existing.get("chunk_id_set_matches_chunks"):
            blockers.append("existing durable collection chunk_id set mismatch")
        summary = {
            "generated_at": utc_now(),
            "status": "pass" if not blockers else "blocked",
            "blockers": blockers,
            "action": "reuse_existing_collection" if not blockers else "blocked_existing_collection_conflict",
            "imported": False,
            "reused": not blockers,
            "milvus_uri": rel(milvus_uri),
            "collection": collection_name,
            "import_command": command,
            "elapsed_sec": round(time.time() - start, 3),
            **existing,
        }
        write_milvus_artifacts(summary)
        return summary

    schema = milvus_import.build_collection_schema(dim=EXPECTED_VECTOR_DIM)
    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"  created collection: {collection_name}")

    embedder = milvus_import.create_embedder(
        embedding_type="bge-m3",
        model_path=str(model_path),
        dim=EXPECTED_VECTOR_DIM,
        embed_max_length=embed_max_length,
    )
    encode_batch_size = 64
    upsert_batch_size = 500
    inserted = 0
    for start_idx in range(0, len(chunks), upsert_batch_size):
        batch = chunks[start_idx : start_idx + upsert_batch_size]
        embeddings: list[list[float]] = []
        texts = [str(item.get("retrieval_text") or item.get("text") or "") for item in batch]
        for encode_start in range(0, len(texts), encode_batch_size):
            embeddings.extend(embedder.encode(texts[encode_start : encode_start + encode_batch_size]))
        inserted += milvus_import.upsert_chunks(
            client=client,
            collection_name=collection_name,
            chunks=batch,
            embeddings=embeddings,
            batch_size=upsert_batch_size,
        )
        print(f"  import progress: {inserted}/{len(chunks)}")

    milvus_import.log_truncation_summary()
    print("  creating index")
    milvus_import.create_index(client, collection_name)
    client.flush(collection_name)
    imported = True

    del embedder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    imported_info = inspect_milvus_collection(client, collection_name, chunk_ids)
    truncation_stats = {
        key: len(value)
        for key, value in getattr(milvus_import, "_truncation_stats", {}).items()
    }
    blockers = []
    if imported_info.get("row_count") != EXPECTED_CHUNK_COUNT:
        blockers.append("imported collection row count mismatch")
    if not imported_info.get("schema_compatible"):
        blockers.append("imported collection schema mismatch")
    if not imported_info.get("chunk_id_set_matches_chunks"):
        blockers.append("imported collection chunk_id set mismatch")

    summary = {
        "generated_at": utc_now(),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "action": "import_new_collection",
        "imported": imported,
        "reused": reused,
        "milvus_uri": rel(milvus_uri),
        "collection": collection_name,
        "import_command": command,
        "embedding_model": "BAAI/bge-m3",
        "embedding_model_path": rel(model_path),
        "import_embedding_max_length": embed_max_length,
        "source_chunks": rel(DURABLE_CHUNKS),
        "source_chunk_count": len(chunks),
        "inserted": inserted,
        "varchar_truncation_counts": truncation_stats,
        "elapsed_sec": round(time.time() - start, 3),
        **imported_info,
    }
    write_milvus_artifacts(summary)
    return summary


def inspect_milvus_collection(
    client: MilvusClient,
    collection_name: str,
    chunk_ids: set[str],
) -> dict[str, Any]:
    stats = client.get_collection_stats(collection_name)
    row_count = int(stats.get("row_count", 0))
    desc = client.describe_collection(collection_name)
    fields = []
    field_names: set[str] = set()
    primary_key = ""
    vector_dimension = None
    for field in desc.get("fields", []):
        name = str(field.get("name") or "")
        field_names.add(name)
        params = field.get("params") or {}
        field_type = stringify_milvus_type(field.get("type"))
        if field.get("is_primary"):
            primary_key = name
        if "FLOAT_VECTOR" in field_type:
            vector_dimension = int(params.get("dim", 0))
        fields.append(
            {
                "name": name,
                "type": field_type,
                "params": params,
                "is_primary": bool(field.get("is_primary")),
                "description": field.get("description", ""),
            }
        )

    required = {
        "chunk_id",
        "doc_id",
        "text",
        "retrieval_text",
        "metadata_json",
        "embedding",
    }
    schema_compatible = (
        required.issubset(field_names)
        and primary_key == "chunk_id"
        and vector_dimension == EXPECTED_VECTOR_DIM
    )
    ids: set[str] = set()
    id_error = ""
    try:
        rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["chunk_id"],
            limit=max(row_count, 1),
        )
        ids = {str(row.get("chunk_id") or "") for row in rows if row.get("chunk_id")}
    except Exception as exc:
        id_error = f"{type(exc).__name__}: {exc}"

    return {
        "row_count": row_count,
        "stats": stats,
        "schema": {
            "collection_name": desc.get("collection_name"),
            "description": desc.get("description"),
            "auto_id": desc.get("auto_id"),
            "enable_dynamic_field": desc.get("enable_dynamic_field"),
            "fields": fields,
        },
        "field_names": sorted(field_names),
        "primary_key_field": primary_key,
        "vector_dimension": vector_dimension,
        "schema_compatible": schema_compatible,
        "required_fields_present": sorted(required & field_names),
        "required_fields_missing": sorted(required - field_names),
        "chunk_id_set_query_count": len(ids),
        "chunk_id_set_matches_chunks": bool(ids) and ids == chunk_ids,
        "chunks_missing_from_milvus_count": len(chunk_ids - ids) if ids else None,
        "chunks_missing_from_milvus_sample": sorted(chunk_ids - ids)[:20] if ids else [],
        "milvus_ids_missing_from_chunks_count": len(ids - chunk_ids) if ids else None,
        "milvus_ids_missing_from_chunks_sample": sorted(ids - chunk_ids)[:20] if ids else [],
        "id_query_error": id_error,
    }


def write_milvus_artifacts(summary: dict[str, Any]) -> None:
    schema = summary.get("schema", {})
    write_json(DURABLE_MILVUS_DIR / "collection_schema.json", schema)
    write_json(
        DURABLE_MILVUS_DIR / "collection_manifest.json",
        {
            "baseline_name": BASELINE_NAME,
            "milvus_uri": summary.get("milvus_uri"),
            "collection": summary.get("collection"),
            "row_count": summary.get("row_count"),
            "schema_compatible": summary.get("schema_compatible"),
            "vector_dimension": summary.get("vector_dimension"),
            "chunk_id_set_matches_chunks": summary.get("chunk_id_set_matches_chunks"),
            "created_at": summary.get("generated_at"),
            "is_production_default": False,
        },
    )
    write_json(DURABLE_MILVUS_DIR / "import_summary.json", summary)


def build_manifest(
    dataset_path: Path,
    dataset_info: dict[str, Any],
    chunk_info: dict[str, Any],
    bm25_info: dict[str, Any],
    import_summary: dict[str, Any],
    milvus_uri: Path,
    collection_name: str,
) -> dict[str, Any]:
    return {
        "baseline_name": BASELINE_NAME,
        "status": "official_clean_baseline",
        "source_asset": "phase5c5_baseline_full",
        "dataset_path": rel(dataset_path),
        "dataset_sha256": dataset_info.get("sha256"),
        "dataset_count": dataset_info.get("count"),
        "query_type_distribution": dataset_info.get("query_type_distribution"),
        "chunks_path": rel(DURABLE_CHUNKS),
        "chunks_sha256": chunk_info.get("sha256"),
        "chunk_count": chunk_info.get("line_count"),
        "bm25_path": rel(DURABLE_BM25),
        "bm25_sha256": bm25_info.get("sha256"),
        "bm25_records": bm25_info.get("record_count"),
        "milvus_uri": rel(milvus_uri),
        "milvus_collection": collection_name,
        "milvus_row_count": import_summary.get("row_count"),
        "embedding_model": "BAAI/bge-m3",
        "vector_dimension": import_summary.get("vector_dimension"),
        "retrieval_schema_version": RetrievalConfig().index_schema_version,
        "table_enhancement_enabled": False,
        "caption_cleanup_enabled": False,
        "is_enhanced_variant": False,
        "is_production_default": False,
        "production_current_reference": "synbio_papers / current_default",
        "created_from_tmp_assets": True,
        "tmp_source_paths": {
            "chunks": str(SOURCE_CHUNKS),
            "bm25": str(SOURCE_BM25),
            "source_milvus_reference": "/tmp/phase5c5_baseline_full.db",
            "source_collection_reference": "synbio_phase5c5_baseline_full",
        },
        "created_at": utc_now(),
        "validation_status": "pending_final_validation",
        "known_caveats": [
            "strict all_ids_present 89/90 if still true",
            "caption_level_table sample size small",
            "clean-main baseline v0, not final formal benchmark",
        ],
    }


def write_manifest_and_readme(manifest: dict[str, Any]) -> None:
    write_json(DURABLE_ROOT / "manifest.json", manifest)
    write_text(DURABLE_DATASET_DIR / "strict_main_eval_set_v2.path.txt", manifest["dataset_path"] + "\n")
    write_text(
        DURABLE_DATASET_DIR / "strict_main_eval_set_v2.sha256.txt",
        manifest["dataset_sha256"] + "\n",
    )
    readme = [
        "# Phase 5F Official Clean Baseline",
        "",
        "This directory freezes the official clean baseline promoted from Phase 5C-5 baseline_full assets.",
        "",
        "- This is the official clean baseline.",
        "- This is not the table-enhancement-ON variant.",
        "- This is not the production current default.",
        "- This should not be compared directly with old Phase 5C/5D denominators.",
        "- Future retrieval fixes should use this baseline and the frozen strict_main_eval_set_v2 first.",
        "",
        "Core assets:",
        f"- dataset: `{manifest['dataset_path']}`",
        f"- dataset_sha256: `{manifest['dataset_sha256']}`",
        f"- chunks: `{manifest['chunks_path']}`",
        f"- chunks_sha256: `{manifest['chunks_sha256']}`",
        f"- BM25: `{manifest['bm25_path']}`",
        f"- BM25_sha256: `{manifest['bm25_sha256']}`",
        f"- Milvus URI: `{manifest['milvus_uri']}`",
        f"- Milvus collection: `{manifest['milvus_collection']}`",
        "",
        "Scope guardrails:",
        "- chunks_regenerated: no",
        "- BM25_rebuilt: no",
        "- retrieval_code_modified: no",
        "- cleaning_code_modified: no",
        "- chunk_logic_modified: no",
        "- dataset_modified: no",
        "- Qwen/RAGAS/OCR: no",
    ]
    write_text(DURABLE_ROOT / "README.md", "\n".join(readme) + "\n")
    write_text(
        REPORT_DIR / "official_manifest_summary.md",
        "\n".join(
            [
                "# Official manifest summary",
                "",
                f"- baseline_name: `{manifest['baseline_name']}`",
                f"- status: `{manifest['status']}`",
                f"- dataset_sha256: `{manifest['dataset_sha256']}`",
                f"- chunk_count: `{manifest['chunk_count']}`",
                f"- bm25_records: `{manifest['bm25_records']}`",
                f"- milvus_collection: `{manifest['milvus_collection']}`",
                f"- milvus_row_count: `{manifest['milvus_row_count']}`",
                "- table_enhancement_enabled: `false`",
                "- caption_cleanup_enabled: `false`",
                "- is_production_default: `false`",
            ]
        )
        + "\n",
    )


def load_milvus_rows(milvus_uri: Path, collection_name: str) -> list[dict[str, Any]]:
    client = MilvusClient(str(milvus_uri.resolve()))
    stats = client.get_collection_stats(collection_name)
    row_count = int(stats.get("row_count", 0))
    return client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["chunk_id", "doc_id", "retrieval_text", "metadata_json"],
        limit=max(row_count, 1),
    )


def build_coverage(
    samples: list[dict[str, Any]],
    chunk_index: ChunkIndex,
    bm25_ids: set[str],
    milvus_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    milvus_by_doc_block: dict[tuple[str, str], list[str]] = defaultdict(list)
    milvus_ids: set[str] = set()
    milvus_retrieval_nonempty: dict[str, bool] = {}
    for row in milvus_rows:
        chunk_id = str(row.get("chunk_id") or "")
        doc_id = str(row.get("doc_id") or "")
        milvus_ids.add(chunk_id)
        milvus_retrieval_nonempty[chunk_id] = bool(str(row.get("retrieval_text") or "").strip())
        metadata = safe_parse_json(row.get("metadata_json"))
        for block_id in parse_block_ids(metadata.get("source_block_ids") or metadata.get("block_ids")):
            if doc_id and chunk_id:
                milvus_by_doc_block[(doc_id, block_id)].append(chunk_id)

    rows: list[dict[str, Any]] = []
    qtype_summary: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        sid = sample_id(sample)
        qtype = str(sample.get("query_type") or "")
        target_doc = str(sample.get("target_doc_id") or "")
        stable = stable_ids(sample)
        target_doc_present = target_doc in chunk_index.by_doc
        present_blocks: list[str] = []
        missing_blocks: list[str] = []
        target_chunk_ids: list[str] = []
        milvus_missing_blocks: list[str] = []

        for block_id in stable:
            chunk_ids = chunk_index.by_doc_block.get((target_doc, block_id), [])
            if chunk_ids:
                present_blocks.append(block_id)
                target_chunk_ids.extend(chunk_ids)
            else:
                missing_blocks.append(block_id)
            if not milvus_by_doc_block.get((target_doc, block_id)):
                milvus_missing_blocks.append(block_id)

        target_chunk_ids = sorted(set(target_chunk_ids))
        missing_milvus_chunks = sorted(set(target_chunk_ids) - milvus_ids)
        missing_bm25_chunks = sorted(set(target_chunk_ids) - bm25_ids)
        stable_any = bool(present_blocks)
        stable_all = bool(stable) and not missing_blocks
        target_retrievable = stable_any and any(
            chunk_index.retrieval_nonempty.get(chunk_id, False) for chunk_id in target_chunk_ids
        )
        row = {
            "sample_id": sid,
            "query_type": qtype,
            "target_doc_id": target_doc,
            "target_doc_present": target_doc_present,
            "stable_target_block_ids": ";".join(stable),
            "present_stable_block_ids": ";".join(sorted(set(present_blocks))),
            "missing_stable_block_ids": ";".join(sorted(set(missing_blocks))),
            "stable_block_any_present": stable_any,
            "stable_block_all_present": stable_all,
            "target_block_chunk_ids": ";".join(target_chunk_ids),
            "target_block_retrieval_text_nonempty": target_retrievable,
            "target_block_missing_in_copied_chunks": ";".join(sorted(set(missing_blocks))),
            "target_block_missing_in_milvus": ";".join(sorted(set(milvus_missing_blocks))),
            "target_chunk_missing_in_milvus": ";".join(missing_milvus_chunks),
            "target_chunk_missing_in_bm25": ";".join(missing_bm25_chunks),
            "target_doc_present_but_stable_block_missing": target_doc_present and not stable_any,
            "stable_block_present_but_unretrievable": stable_any and not target_retrievable,
        }
        rows.append(row)
        qtype_summary[qtype]["total"] += 1
        qtype_summary[qtype]["target_doc_present"] += int(target_doc_present)
        qtype_summary[qtype]["stable_any_present"] += int(stable_any)
        qtype_summary[qtype]["stable_all_present"] += int(stable_all)
        qtype_summary[qtype]["milvus_target_chunks_present"] += int(not missing_milvus_chunks)
        qtype_summary[qtype]["bm25_target_chunks_present"] += int(not missing_bm25_chunks)

    total = len(rows)
    summary = {
        "generated_at": utc_now(),
        "total_samples": total,
        "target_doc_coverage_count": sum(1 for row in rows if row["target_doc_present"]),
        "stable_block_any_coverage_count": sum(1 for row in rows if row["stable_block_any_present"]),
        "stable_block_all_ids_present_count": sum(1 for row in rows if row["stable_block_all_present"]),
        "stable_block_retrievable_count": sum(
            1 for row in rows if row["target_block_retrieval_text_nonempty"]
        ),
        "target_doc_coverage_pass": sum(1 for row in rows if row["target_doc_present"]) == total,
        "stable_block_coverage_pass": sum(1 for row in rows if row["stable_block_any_present"]) == total,
        "stable_block_all_ids_present_pass": sum(1 for row in rows if row["stable_block_all_present"]) == total,
        "target_block_missing_in_copied_chunks_count": sum(
            1 for row in rows if row["target_block_missing_in_copied_chunks"]
        ),
        "target_block_missing_in_milvus_count": sum(
            1 for row in rows if row["target_block_missing_in_milvus"]
        ),
        "target_chunk_missing_in_bm25_count": sum(
            1 for row in rows if row["target_chunk_missing_in_bm25"]
        ),
        "query_type_summary": {key: dict(value) for key, value in sorted(qtype_summary.items())},
        "coverage_matches_phase5f4b": (
            sum(1 for row in rows if row["target_doc_present"]) == 90
            and sum(1 for row in rows if row["stable_block_any_present"]) == 90
            and sum(1 for row in rows if row["stable_block_all_present"]) == 89
        ),
        "warning": "strict all_ids_present 89/90 is non-blocking"
        if sum(1 for row in rows if row["stable_block_all_present"]) == 89
        else "",
    }
    return rows, summary


def run_retrieval_validation(
    samples: list[dict[str, Any]],
    chunk_index: ChunkIndex,
    bm25_payload: dict[str, Any],
    milvus_uri: Path,
    collection_name: str,
    model_path: Path,
    retrieval_embed_max_length: int,
    candidate_limit: int,
    top_k: int,
) -> dict[str, Any]:
    start = time.time()
    config = RetrievalConfig(
        milvus_uri=str(milvus_uri.resolve()),
        collection_name=collection_name,
        search_limit=candidate_limit,
        dense_limit=candidate_limit,
        bm25_limit=candidate_limit,
    )
    bm25 = CachedBM25(bm25_payload, config)
    embedder = BGEM3Embedder(
        str(model_path.resolve()),
        dim=EXPECTED_VECTOR_DIM,
        max_length=retrieval_embed_max_length,
    )
    dense = DenseMilvusAdapter(milvus_uri, collection_name, config, embedder)
    hybrid = HybridRetriever(config, dense, bm25)

    rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        query = str(sample.get("query") or "")
        hits = hybrid.search(query, limit=candidate_limit, filters=None, analysis=None)
        row, topk = evaluate_sample(
            sample=sample,
            hits=hits,
            chunk_index=chunk_index,
            top_k=top_k,
        )
        rows.append(row)
        topk_rows.append(topk)
        if idx % 10 == 0 or idx == len(samples):
            print(f"  retrieval progress: {idx}/{len(samples)}")

    metrics = build_metrics(rows)
    prior = load_prior_phase5f4_baseline_results()
    differences = compare_prior(rows, prior)
    overall = metrics["overall"]
    stable_delta = abs(overall["stable_block_hit_at_10"] - REFERENCE_PHASE5F4_STABLE_AT_10)
    doc_delta = abs(overall["doc_hit_at_10"] - REFERENCE_PHASE5F4_DOC_AT_10)
    close_to_reference = stable_delta <= 0.0223 and doc_delta <= 0.0223

    return {
        "generated_at": utc_now(),
        "status": "pass" if close_to_reference else "validation_review_required",
        "run_config": {
            "dataset": rel(DATASET_PATH),
            "chunks": rel(DURABLE_CHUNKS),
            "bm25": rel(DURABLE_BM25),
            "milvus_uri": rel(milvus_uri),
            "milvus_collection": collection_name,
            "retrieval_only": True,
            "generation_eval": False,
            "qwen_called": False,
            "ragas_run": False,
            "ocr": False,
            "candidate_limit": candidate_limit,
            "top_k": top_k,
            "target_matching": "stable_target_block_ids corrected matching",
            "diagnostic_or_lexical_stress_in_denominator": False,
            "retrieval_parameters_tuned": False,
        },
        "reference_phase5f4_phase5c5_baseline_full": {
            "doc_hit_at_10": REFERENCE_PHASE5F4_DOC_AT_10,
            "stable_block_hit_at_10": REFERENCE_PHASE5F4_STABLE_AT_10,
        },
        "close_to_reference": close_to_reference,
        "doc_hit_at_10_delta_abs": doc_delta,
        "stable_block_hit_at_10_delta_abs": stable_delta,
        "metrics": metrics,
        "per_sample_results": rows,
        "topk_examples": topk_rows,
        "prior_difference_count": len(differences),
        "prior_differences": differences,
        "elapsed_sec": round(time.time() - start, 3),
    }


class CachedBM25:
    def __init__(self, payload: dict[str, Any], config: RetrievalConfig):
        self.config = config
        self.records = [record_to_retrieved(item) for item in payload.get("records", [])]
        self.doc_len = [int(value) for value in payload.get("doc_len", [])]
        self.doc_freq = defaultdict(int, {k: int(v) for k, v in payload.get("doc_freq", {}).items()})
        self.term_freqs = [
            Counter({k: int(v) for k, v in item.items()})
            for item in payload.get("term_freqs", [])
        ]
        self.avgdl = float(payload.get("avgdl", 0.0))

    def search(
        self,
        question: str,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        query_terms = tokenize_query(question)
        if not query_terms:
            return []
        scored: list[RetrievedChunk] = []
        for idx, chunk in self._filter_records(filters):
            score = self._score(query_terms, idx)
            if score <= 0:
                continue
            item = clone_chunk(chunk)
            item.bm25_score = score
            scored.append(item)
        scored.sort(key=lambda item: item.bm25_score, reverse=True)
        return scored[:limit]

    def _filter_records(self, filters: QueryFilters | None) -> list[tuple[int, RetrievedChunk]]:
        if not filters:
            return list(enumerate(self.records))
        doc_ids = set(filters.doc_ids)
        sections = set(filters.sections)
        source_files = set(filters.source_files)
        items: list[tuple[int, RetrievedChunk]] = []
        for idx, chunk in enumerate(self.records):
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            if sections and chunk.section not in sections:
                continue
            if source_files and chunk.source_file not in source_files:
                continue
            items.append((idx, chunk))
        return items

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        score = 0.0
        tf = self.term_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        n_docs = len(self.records)
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.config.bm25_k1 * (
                1 - self.config.bm25_b + self.config.bm25_b * dl / max(self.avgdl, 1.0)
            )
            score += idf * (freq * (self.config.bm25_k1 + 1)) / max(denom, 1e-9)
        return score


class DenseMilvusAdapter:
    def __init__(
        self,
        uri: Path,
        collection: str,
        config: RetrievalConfig,
        embedder: BGEM3Embedder,
    ):
        self.uri = str(uri.resolve())
        self.collection = collection
        self.config = config
        self.embedder = embedder
        self.client = MilvusClient(self.uri)
        self._embedding_cache: dict[str, list[float]] = {}

    def search(
        self,
        question: str,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        query_vec = self._encode(question)
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vec],
            anns_field=self.config.vector_field,
            limit=limit,
            search_params=self._search_params(),
            output_fields=[
                "chunk_id",
                "doc_id",
                "source_file",
                "title",
                "section",
                "page_start",
                "page_end",
                "chunk_index",
                "text",
                "retrieval_text",
                "content_kind",
                "quality_score",
                "contains_table_text",
                "contains_table_caption",
                "contains_figure_caption",
                "contains_image",
                "object_type",
                "object_id",
                "metadata_json",
            ],
            filter="",
        )
        if not results:
            return []
        chunks: list[RetrievedChunk] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            score = float(hit.get("distance", 0.0))
            if score < self.config.score_floor:
                continue
            metadata = {
                "chunk_index": entity.get("chunk_index"),
                "retrieval_text": entity.get("retrieval_text", ""),
                "content_kind": entity.get("content_kind", "body"),
                "quality_score": entity.get("quality_score", 0.0),
                "contains_table_text": entity.get("contains_table_text", False),
                "contains_table_caption": entity.get("contains_table_caption", False),
                "contains_figure_caption": entity.get("contains_figure_caption", False),
                "contains_image": entity.get("contains_image", False),
                "object_type": entity.get("object_type", "body"),
                "object_id": entity.get("object_id", ""),
            }
            metadata.update(safe_parse_json(entity.get("metadata_json", "")))
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(entity.get("chunk_id", "")),
                    doc_id=str(entity.get("doc_id", "")),
                    source_file=str(entity.get("source_file", "")),
                    title=str(entity.get("title", "")),
                    section=str(entity.get("section", "")),
                    text=str(entity.get("text", "")),
                    page_start=normalize_page(entity.get("page_start")),
                    page_end=normalize_page(entity.get("page_end")),
                    vector_score=score,
                    metadata=metadata,
                )
            )
        return chunks

    def _encode(self, question: str) -> list[float]:
        cached = self._embedding_cache.get(question)
        if cached is not None:
            return cached
        vec = self.embedder.encode([question])[0]
        self._embedding_cache[question] = vec
        return vec

    def _search_params(self) -> dict[str, Any]:
        if self.config.index_type == "HNSW":
            return {"metric_type": self.config.metric_type, "params": {"ef": self.config.hnsw_ef}}
        return {"metric_type": self.config.metric_type, "params": {"nprobe": self.config.nprobe}}


def evaluate_sample(
    sample: dict[str, Any],
    hits: list[RetrievedChunk],
    chunk_index: ChunkIndex,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_doc_id = str(sample.get("target_doc_id") or "")
    stable = stable_ids(sample)
    doc_rank = first_rank([hit.doc_id == target_doc_id for hit in hits])
    stable_rank = 0
    for rank, hit in enumerate(hits, start=1):
        if hit.doc_id != target_doc_id:
            continue
        chunk = chunk_index.by_id.get(hit.chunk_id, {})
        if set(chunk_block_ids(chunk)) & set(stable):
            stable_rank = rank
            break
    target_chunk_ids = sorted(
        {
            chunk_id
            for block_id in stable
            for chunk_id in chunk_index.by_doc_block.get((target_doc_id, block_id), [])
        }
    )
    top_hits = [serialize_hit(hit, rank, chunk_index) for rank, hit in enumerate(hits[:top_k], start=1)]
    top10 = top_hits[:10]
    row = {
        "sample_id": sample_id(sample),
        "query_type": sample.get("query_type", ""),
        "ability_scope": sample.get("ability_scope", ""),
        "query": sample.get("query", ""),
        "target_doc_id": target_doc_id,
        "stable_target_block_ids": stable,
        "stable_target_chunk_ids": target_chunk_ids,
        "stable_target_mapping_found": bool(target_chunk_ids),
        "retrieved_doc_ids_top10": [hit["doc_id"] for hit in top10],
        "retrieved_chunk_ids_top10": [hit["chunk_id"] for hit in top10],
        "retrieved_block_ids_top10": [hit["block_ids"] for hit in top10],
        "doc_hit_at_1": 0 < doc_rank <= 1,
        "doc_hit_at_5": 0 < doc_rank <= 5,
        "doc_hit_at_10": 0 < doc_rank <= 10,
        "doc_hit_at_20": 0 < doc_rank <= 20,
        "stable_block_hit_at_1": 0 < stable_rank <= 1,
        "stable_block_hit_at_5": 0 < stable_rank <= 5,
        "stable_block_hit_at_10": 0 < stable_rank <= 10,
        "stable_block_hit_at_20": 0 < stable_rank <= 20,
        "first_doc_hit_rank": doc_rank,
        "first_stable_block_hit_rank": stable_rank,
        "retrieval_mode": "hybrid_rrf_durable_milvus_copied_bm25",
        "index_variant": BASELINE_NAME,
    }
    topk = {
        "sample_id": row["sample_id"],
        "query_type": row["query_type"],
        "query": row["query"],
        "target_doc_id": target_doc_id,
        "stable_target_block_ids": stable,
        "first_doc_hit_rank": doc_rank,
        "first_stable_block_hit_rank": stable_rank,
        "top_hits": top_hits,
    }
    return row, topk


def serialize_hit(hit: RetrievedChunk, rank: int, chunk_index: ChunkIndex) -> dict[str, Any]:
    chunk = chunk_index.by_id.get(hit.chunk_id, {})
    metadata = dict(hit.metadata or {})
    block_ids = chunk_block_ids(chunk) or parse_block_ids(metadata.get("source_block_ids"))
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section": hit.section,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "block_ids": block_ids,
        "block_types": parse_block_ids(chunk.get("block_types") or metadata.get("block_types")),
        "evidence_types": parse_block_ids(chunk.get("evidence_types") or metadata.get("evidence_types")),
        "score": hit.fusion_score or hit.vector_score or hit.bm25_score or 0.0,
        "text_preview": truncate(hit.text, 220),
    }


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {
        qtype: aggregate_metric_rows([row for row in rows if row["query_type"] == qtype])
        for qtype in sorted({str(row["query_type"]) for row in rows})
    }
    return {
        "overall": aggregate_metric_rows(rows),
        "by_query_type": by_type,
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "doc_hit_at_1": mean_bool(rows, "doc_hit_at_1"),
        "doc_hit_at_5": mean_bool(rows, "doc_hit_at_5"),
        "doc_hit_at_10": mean_bool(rows, "doc_hit_at_10"),
        "doc_hit_at_20": mean_bool(rows, "doc_hit_at_20"),
        "stable_block_hit_at_1": mean_bool(rows, "stable_block_hit_at_1"),
        "stable_block_hit_at_5": mean_bool(rows, "stable_block_hit_at_5"),
        "stable_block_hit_at_10": mean_bool(rows, "stable_block_hit_at_10"),
        "stable_block_hit_at_20": mean_bool(rows, "stable_block_hit_at_20"),
        "doc_mrr": mean_reciprocal(rows, "first_doc_hit_rank"),
        "stable_block_mrr": mean_reciprocal(rows, "first_stable_block_hit_rank"),
        "mean_first_doc_hit_rank": mean_rank(rows, "first_doc_hit_rank"),
        "mean_first_stable_block_hit_rank": mean_rank(rows, "first_stable_block_hit_rank"),
        "stable_target_mapping_found_rate": mean_bool(rows, "stable_target_mapping_found"),
    }


def load_prior_phase5f4_baseline_results() -> dict[str, dict[str, Any]]:
    path = ROOT / "reports" / "phase5f4_clean_main_baseline" / "per_sample_results.jsonl"
    prior: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return prior
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("index_variant") == "phase5c5_baseline_full":
                prior[str(row.get("sample_id"))] = row
    return prior


def compare_prior(
    rows: list[dict[str, Any]],
    prior: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for row in rows:
        old = prior.get(str(row.get("sample_id")))
        if not old:
            continue
        fields = [
            "doc_hit_at_10",
            "stable_block_hit_at_10",
            "stable_block_hit_at_20",
            "first_doc_hit_rank",
            "first_stable_block_hit_rank",
        ]
        changed = {field: {"official": row.get(field), "phase5f4": old.get(field)} for field in fields if row.get(field) != old.get(field)}
        if changed:
            differences.append(
                {
                    "sample_id": row.get("sample_id"),
                    "query_type": row.get("query_type"),
                    "changes": changed,
                }
            )
    return differences


def write_source_validation(summary: dict[str, Any]) -> None:
    write_json(REPORT_DIR / "source_validation.json", summary)
    lines = [
        "# Source validation",
        "",
        f"- status: `{summary['status']}`",
        f"- blockers: {', '.join(summary['blockers']) if summary['blockers'] else 'none'}",
        f"- dataset_sha256_matches: `{str(summary['dataset'].get('sha256_matches_expected')).lower()}`",
        f"- dataset_count: `{summary['dataset'].get('count')}`",
        f"- chunks_exists: `{str(summary['chunks'].get('exists')).lower()}`",
        f"- chunks_count: `{summary['chunks'].get('line_count')}`",
        f"- chunk_id_unique: `{str(summary['chunks'].get('duplicate_chunk_id_count') == 0).lower()}`",
        f"- doc_id_nonempty: `{str(summary['chunks'].get('empty_doc_id_count') == 0).lower()}`",
        f"- table_enhancement_metadata_chunk_count: `{summary['chunks'].get('table_enhancement_metadata_chunk_count')}`",
        f"- caption_cleanup_metadata_chunk_count: `{summary['chunks'].get('caption_cleanup_metadata_chunk_count')}`",
        f"- bm25_exists: `{str(summary['bm25'].get('exists')).lower()}`",
        f"- bm25_records: `{summary['bm25'].get('record_count')}`",
        f"- bm25_chunk_ids_match_chunks: `{str(summary['bm25'].get('chunk_id_set_matches_chunks')).lower()}`",
    ]
    write_text(REPORT_DIR / "source_validation.md", "\n".join(lines) + "\n")


def write_copy_summary(summary: dict[str, Any]) -> None:
    write_json(REPORT_DIR / "copy_summary.json", summary)
    lines = [
        "# Copy summary",
        "",
        f"- status: `{summary['status']}`",
        f"- blockers: {', '.join(summary['blockers']) if summary['blockers'] else 'none'}",
        f"- copied: `{summary['copied']}`",
        f"- reused: `{summary['reused']}`",
        f"- chunks_path: `{summary['chunks'].get('destination_path')}`",
        f"- chunks_count: `{summary['chunks'].get('line_count')}`",
        f"- chunks_sha256: `{summary['chunks'].get('sha256')}`",
        f"- bm25_path: `{summary['bm25'].get('destination_path')}`",
        f"- bm25_records: `{summary['bm25'].get('record_count')}`",
        f"- bm25_sha256: `{summary['bm25'].get('sha256')}`",
    ]
    write_text(REPORT_DIR / "copy_summary.md", "\n".join(lines) + "\n")


def write_milvus_import_summary(summary: dict[str, Any]) -> None:
    write_json(REPORT_DIR / "milvus_import_summary.json", summary)
    lines = [
        "# Milvus import summary",
        "",
        f"- status: `{summary['status']}`",
        f"- action: `{summary.get('action')}`",
        f"- blockers: {', '.join(summary['blockers']) if summary['blockers'] else 'none'}",
        f"- milvus_uri: `{summary.get('milvus_uri')}`",
        f"- collection: `{summary.get('collection')}`",
        f"- row_count: `{summary.get('row_count')}`",
        f"- vector_dimension: `{summary.get('vector_dimension')}`",
        f"- schema_compatible: `{str(summary.get('schema_compatible')).lower()}`",
        f"- chunk_id_set_matches_chunks: `{str(summary.get('chunk_id_set_matches_chunks')).lower()}`",
        f"- imported: `{str(summary.get('imported')).lower()}`",
        f"- reused: `{str(summary.get('reused')).lower()}`",
        f"- elapsed_sec: `{summary.get('elapsed_sec')}`",
    ]
    write_text(REPORT_DIR / "milvus_import_summary.md", "\n".join(lines) + "\n")


def write_coverage_reports(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fieldnames = [
        "sample_id",
        "query_type",
        "target_doc_id",
        "target_doc_present",
        "stable_target_block_ids",
        "present_stable_block_ids",
        "missing_stable_block_ids",
        "stable_block_any_present",
        "stable_block_all_present",
        "target_block_chunk_ids",
        "target_block_retrieval_text_nonempty",
        "target_block_missing_in_copied_chunks",
        "target_block_missing_in_milvus",
        "target_chunk_missing_in_milvus",
        "target_chunk_missing_in_bm25",
        "target_doc_present_but_stable_block_missing",
        "stable_block_present_but_unretrievable",
    ]
    write_csv(REPORT_DIR / "target_coverage_after_promotion.csv", rows, fieldnames)
    write_json(DURABLE_REPORTS_VALIDATION_DIR / "target_coverage_after_promotion.json", summary)
    lines = [
        "# Target coverage after promotion",
        "",
        f"- target_doc coverage: `{summary['target_doc_coverage_count']}/{summary['total_samples']}`",
        f"- stable_block coverage: `{summary['stable_block_any_coverage_count']}/{summary['total_samples']}`",
        f"- stricter all_ids_present: `{summary['stable_block_all_ids_present_count']}/{summary['total_samples']}`",
        f"- stable_block retrievable: `{summary['stable_block_retrievable_count']}/{summary['total_samples']}`",
        f"- copied chunks missing target block rows: `{summary['target_block_missing_in_copied_chunks_count']}`",
        f"- Milvus missing target block rows: `{summary['target_block_missing_in_milvus_count']}`",
        f"- BM25 missing target chunk rows: `{summary['target_chunk_missing_in_bm25_count']}`",
        f"- coverage_matches_phase5f4b: `{str(summary['coverage_matches_phase5f4b']).lower()}`",
    ]
    if summary.get("warning"):
        lines.append(f"- warning: {summary['warning']}")
    lines.extend(["", "## By query type", ""])
    for qtype, values in summary["query_type_summary"].items():
        lines.append(f"- {qtype}: {json.dumps(values, ensure_ascii=False)}")
    write_text(REPORT_DIR / "target_coverage_after_promotion.md", "\n".join(lines) + "\n")


def write_retrieval_reports(retrieval: dict[str, Any]) -> None:
    public_payload = dict(retrieval)
    per_sample = public_payload.pop("per_sample_results")
    topk = public_payload.pop("topk_examples")
    write_json(REPORT_DIR / "official_baseline_results.json", public_payload)
    write_json(DURABLE_REPORTS_VALIDATION_DIR / "official_baseline_results.json", public_payload)
    write_jsonl(REPORT_DIR / "official_per_sample_results.jsonl", per_sample)
    write_jsonl(REPORT_DIR / "official_topk_examples.jsonl", topk)
    rows = []
    rows.append({"query_type": "overall", **retrieval["metrics"]["overall"]})
    for qtype, values in retrieval["metrics"]["by_query_type"].items():
        rows.append({"query_type": qtype, **values})
    fieldnames = [
        "query_type",
        "count",
        "doc_hit_at_1",
        "doc_hit_at_5",
        "doc_hit_at_10",
        "doc_hit_at_20",
        "stable_block_hit_at_1",
        "stable_block_hit_at_5",
        "stable_block_hit_at_10",
        "stable_block_hit_at_20",
        "doc_mrr",
        "stable_block_mrr",
        "mean_first_doc_hit_rank",
        "mean_first_stable_block_hit_rank",
        "stable_target_mapping_found_rate",
    ]
    write_csv(REPORT_DIR / "official_baseline_by_query_type.csv", rows, fieldnames)
    lines = [
        "# Official baseline by query type",
        "",
        "| query_type | n | doc@10 | stable@10 | stable@20 | doc@20 | mapping_found |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['query_type']} | {row.get('count', 0)} | "
            f"{fmt_pct(row.get('doc_hit_at_10'))} | {fmt_pct(row.get('stable_block_hit_at_10'))} | "
            f"{fmt_pct(row.get('stable_block_hit_at_20'))} | {fmt_pct(row.get('doc_hit_at_20'))} | "
            f"{fmt_pct(row.get('stable_target_mapping_found_rate'))} |"
        )
    lines.extend(
        [
            "",
            f"- close_to_phase5f4_phase5c5_baseline_full: `{str(retrieval['close_to_reference']).lower()}`",
            f"- prior_difference_count: `{retrieval['prior_difference_count']}`",
        ]
    )
    write_text(REPORT_DIR / "official_baseline_by_query_type.md", "\n".join(lines) + "\n")
    write_miss_examples(per_sample, retrieval)


def write_miss_examples(rows: list[dict[str, Any]], retrieval: dict[str, Any]) -> None:
    misses = [row for row in rows if not row["stable_block_hit_at_10"]]
    lines = [
        "# Official baseline miss examples",
        "",
        f"- stable@10: `{fmt_pct(retrieval['metrics']['overall'].get('stable_block_hit_at_10'))}`",
        f"- miss_count: `{len(misses)}`",
        "",
    ]
    for qtype in QUERY_TYPES:
        qrows = [row for row in misses if row["query_type"] == qtype]
        lines.extend([f"## {qtype}", ""])
        if not qrows:
            lines.append("- none")
        for row in qrows[:10]:
            lines.append(
                f"- `{row['sample_id']}`: doc_rank={row['first_doc_hit_rank']}, "
                f"stable_rank={row['first_stable_block_hit_rank']}, target_doc={row['target_doc_id']}"
            )
        lines.append("")
    if retrieval["prior_differences"]:
        lines.extend(["## Differences vs Phase 5F-4 phase5c5_baseline_full", ""])
        for diff in retrieval["prior_differences"][:20]:
            lines.append(f"- `{diff['sample_id']}` ({diff['query_type']}): {json.dumps(diff['changes'])}")
    write_text(REPORT_DIR / "official_miss_examples.md", "\n".join(lines) + "\n")


def write_baseline_registry(manifest: dict[str, Any]) -> Path:
    path = ROOT / "configs" / "baseline_registry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""official_clean_baseline:
  name: {BASELINE_NAME}
  dataset_path: {manifest['dataset_path']}
  dataset_sha256: {manifest['dataset_sha256']}
  chunks_path: {manifest['chunks_path']}
  chunk_count: {manifest['chunk_count']}
  bm25_path: {manifest['bm25_path']}
  bm25_records: {manifest['bm25_records']}
  milvus_uri: {manifest['milvus_uri']}
  milvus_collection: {manifest['milvus_collection']}
  table_enhancement_enabled: false
  caption_cleanup_enabled: false
  status: official

experimental_variants:
  table_enhancement_on:
    status: experimental
    default_enabled: false

legacy_production_reference:
  name: current_default
  milvus_collection: synbio_papers
  chunk_count: 10610
  status: legacy_production_reference
"""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            backup = path.with_suffix(path.suffix + ".phase5f4c.bak")
            if not backup.exists():
                backup.write_text(existing, encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    return path


def build_final_validation(
    source_validation: dict[str, Any],
    copy_summary: dict[str, Any],
    import_summary: dict[str, Any],
    manifest: dict[str, Any],
    coverage_summary: dict[str, Any],
    retrieval: dict[str, Any],
    registry_summary: dict[str, Any],
) -> dict[str, Any]:
    overall = retrieval["metrics"]["overall"]
    checks = {
        "chunks_copied_to_durable_path": copy_summary["chunks"].get("status") in {"copied", "reused"},
        "bm25_copied_to_durable_path": copy_summary["bm25"].get("status") in {"copied", "reused"},
        "milvus_official_collection_established": import_summary.get("row_count") == EXPECTED_CHUNK_COUNT,
        "manifest_generated": (DURABLE_ROOT / "manifest.json").exists(),
        "dataset_sha_matches": source_validation["dataset"].get("sha256") == EXPECTED_DATASET_SHA256,
        "target_doc_coverage_90_90": coverage_summary.get("target_doc_coverage_count") == 90,
        "stable_block_coverage_90_90": coverage_summary.get("stable_block_any_coverage_count") == 90,
        "official_baseline_retrieval_close_to_phase5c5": retrieval.get("close_to_reference") is True,
        "rebuild_chunks_needed": False,
        "rebuild_bm25_needed": False,
        "retrieval_cleaning_chunk_eval_dataset_modified": False,
        "qwen_or_ragas_called": False,
        "strict_main_eval_set_v2_hash_remains_denominator": True,
        "asset_blocker": bool(
            source_validation.get("blockers")
            or copy_summary.get("blockers")
            or import_summary.get("blockers")
        ),
    }
    blockers = []
    if not checks["chunks_copied_to_durable_path"]:
        blockers.append("chunks durable copy missing")
    if not checks["bm25_copied_to_durable_path"]:
        blockers.append("BM25 durable copy missing")
    if not checks["milvus_official_collection_established"]:
        blockers.append("Milvus official collection row count mismatch")
    if not checks["manifest_generated"]:
        blockers.append("manifest missing")
    if not checks["dataset_sha_matches"]:
        blockers.append("dataset SHA mismatch")
    if not checks["target_doc_coverage_90_90"]:
        blockers.append("target_doc coverage not 90/90")
    if not checks["stable_block_coverage_90_90"]:
        blockers.append("stable_block coverage not 90/90")
    if not checks["official_baseline_retrieval_close_to_phase5c5"]:
        blockers.append("official baseline retrieval not close to phase5c5_baseline_full")
    if checks["asset_blocker"]:
        blockers.append("asset blocker present")

    return {
        "generated_at": utc_now(),
        "checks": checks,
        "blockers": blockers,
        "can_enter_phase5f5_closeout": not blockers,
        "registry": registry_summary,
        "dataset_sha256": source_validation["dataset"].get("sha256"),
        "chunks": copy_summary["chunks"],
        "bm25": copy_summary["bm25"],
        "milvus": {
            "collection": import_summary.get("collection"),
            "row_count": import_summary.get("row_count"),
            "vector_dimension": import_summary.get("vector_dimension"),
        },
        "coverage": coverage_summary,
        "retrieval_overall": overall,
        "retrieval_by_query_type": retrieval["metrics"]["by_query_type"],
    }


def write_final_reports(final_validation: dict[str, Any]) -> None:
    write_json(DURABLE_REPORTS_VALIDATION_DIR / "final_validation.json", final_validation)
    checks = final_validation["checks"]
    lines = [
        "# Final validation",
        "",
        f"1. chunks copied to durable path: `{yes_no(checks['chunks_copied_to_durable_path'])}`.",
        f"2. BM25 copied to durable path: `{yes_no(checks['bm25_copied_to_durable_path'])}`.",
        f"3. Milvus official collection established: `{yes_no(checks['milvus_official_collection_established'])}`.",
        f"4. manifest generated: `{yes_no(checks['manifest_generated'])}`.",
        f"5. dataset SHA still matches: `{yes_no(checks['dataset_sha_matches'])}`.",
        f"6. target coverage 90/90: `{yes_no(checks['target_doc_coverage_90_90'])}`.",
        f"7. stable_block coverage 90/90: `{yes_no(checks['stable_block_coverage_90_90'])}`.",
        f"8. official baseline retrieval close to phase5c5_baseline_full: `{yes_no(checks['official_baseline_retrieval_close_to_phase5c5'])}`.",
        "9. rebuild chunks needed: `no`.",
        "10. rebuild BM25 needed: `no`.",
        "11. retrieval / cleaning / chunk / eval / dataset modified: `no`.",
        "12. Qwen / RAGAS called: `no`.",
        f"13. can enter Phase 5F-5 closeout: `{yes_no(final_validation['can_enter_phase5f5_closeout'])}`.",
        f"14. blocker reason: `{', '.join(final_validation['blockers']) if final_validation['blockers'] else 'none'}`.",
        "15. continue fixing strict_main_eval_set_v2 hash as denominator: `yes`.",
    ]
    write_text(REPORT_DIR / "final_validation.md", "\n".join(lines) + "\n")
    decision = [
        "# Closeout decision",
        "",
        f"- decision: `{'enter_phase5f5_closeout' if final_validation['can_enter_phase5f5_closeout'] else 'blocked'}`",
        f"- blockers: `{', '.join(final_validation['blockers']) if final_validation['blockers'] else 'none'}`",
        f"- official_collection: `{final_validation['milvus']['collection']}`",
        f"- row_count: `{final_validation['milvus']['row_count']}`",
        f"- doc_hit@10: `{fmt_pct(final_validation['retrieval_overall'].get('doc_hit_at_10'))}`",
        f"- stable_block_hit@10: `{fmt_pct(final_validation['retrieval_overall'].get('stable_block_hit_at_10'))}`",
        f"- stable_block_hit@20: `{fmt_pct(final_validation['retrieval_overall'].get('stable_block_hit_at_20'))}`",
    ]
    write_text(REPORT_DIR / "closeout_decision.md", "\n".join(decision) + "\n")


def ensure_no_blockers(payload: dict[str, Any], label: str) -> None:
    blockers = payload.get("blockers") or []
    if blockers:
        raise Blocker(f"{label} blocked: {', '.join(str(item) for item in blockers)}")


def build_chunk_index(chunks: list[dict[str, Any]]) -> ChunkIndex:
    by_id: dict[str, dict[str, Any]] = {}
    by_doc: dict[str, list[str]] = defaultdict(list)
    by_doc_block: dict[tuple[str, str], list[str]] = defaultdict(list)
    retrieval_nonempty: dict[str, bool] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        doc_id = str(chunk.get("doc_id") or "")
        if chunk_id:
            by_id[chunk_id] = chunk
            retrieval_nonempty[chunk_id] = bool(str(chunk.get("retrieval_text") or "").strip())
        if doc_id and chunk_id:
            by_doc[doc_id].append(chunk_id)
        for block_id in chunk_block_ids(chunk):
            if doc_id and chunk_id:
                by_doc_block[(doc_id, block_id)].append(chunk_id)
    return ChunkIndex(
        chunks=chunks,
        by_id=by_id,
        by_doc=dict(by_doc),
        by_doc_block=by_doc_block,
        retrieval_nonempty=retrieval_nonempty,
    )


def has_table_enhancement_metadata(chunk: dict[str, Any]) -> bool:
    top_level_keys = {
        "table_related",
        "table_related_type",
        "table_enhancement_enabled",
        "phase5c1_pilot",
        "associated_table_caption_block_id",
        "associated_table_caption_text",
        "associated_table_id",
        "table_object",
    }
    for key in top_level_keys:
        if key in chunk and chunk.get(key):
            return True
    for item in chunk.get("source_block_metadata") or []:
        if not isinstance(item, dict):
            continue
        if item.get("table_related") is True:
            return True
        if item.get("table_enhancement_enabled") is True:
            return True
        if item.get("phase5c1_pilot") is True:
            return True
        for key in (
            "table_related_type",
            "associated_table_caption_block_id",
            "associated_table_caption_text",
            "associated_table_id",
        ):
            if item.get(key):
                return True
    return False


def has_caption_cleanup_metadata(chunk: dict[str, Any]) -> bool:
    keys = set(chunk.keys())
    for item in chunk.get("source_block_metadata") or []:
        if isinstance(item, dict):
            keys.update(item.keys())
            for value in item.values():
                if isinstance(value, str) and (
                    "caption_cleanup" in value.lower() or "phase5d" in value.lower()
                ):
                    return True
    for key in keys:
        lowered = str(key).lower()
        if "caption_cleanup" in lowered or "phase5d" in lowered:
            return True
    return False


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
    }
    if path.exists():
        info["size_bytes"] = path.stat().st_size
        info["mtime_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()
    return info


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def bm25_chunk_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str(record.get("chunk_id") or "")
        for record in payload.get("records", [])
        if isinstance(record, dict) and record.get("chunk_id")
    }


def record_to_retrieved(item: dict[str, Any]) -> RetrievedChunk:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return RetrievedChunk(
        chunk_id=str(item.get("chunk_id", "")),
        doc_id=str(item.get("doc_id", "")),
        source_file=str(item.get("source_file", "")),
        title=str(item.get("title", "")),
        section=str(item.get("section", "")),
        text=str(item.get("text", "")),
        page_start=normalize_page(item.get("page_start")),
        page_end=normalize_page(item.get("page_end")),
        metadata=dict(metadata),
    )


def clone_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        title=chunk.title,
        section=chunk.section,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score,
        rerank_score=chunk.rerank_score,
        fusion_score=chunk.fusion_score,
        metadata=dict(chunk.metadata),
    )


def chunk_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    block_ids = parse_block_ids(chunk.get("source_block_ids"))
    block_ids.extend(parse_block_ids(chunk.get("block_ids")))
    for item in chunk.get("source_block_metadata") or []:
        if isinstance(item, dict):
            block_ids.extend(parse_block_ids(item.get("block_id")))
    return sorted(set(block_ids))


def stable_ids(sample: dict[str, Any]) -> list[str]:
    return parse_block_ids(sample.get("stable_target_block_ids"))


def parse_block_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(parse_block_ids(item))
        return out
    return [str(value)]


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or "")


def first_rank(flags: list[bool]) -> int:
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return idx
    return 0


def normalize_page(value: Any) -> int | None:
    if value in (None, "", -1):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0


def mean_rank(rows: list[dict[str, Any]], key: str) -> float | None:
    ranks = [int(row.get(key) or 0) for row in rows if int(row.get(key) or 0) > 0]
    return statistics.mean(ranks) if ranks else None


def mean_reciprocal(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return statistics.mean(
        (1 / int(row.get(key) or 0)) if int(row.get(key) or 0) > 0 else 0.0
        for row in rows
    )


def truncate(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def safe_parse_json(raw: Any) -> dict[str, Any]:
    if not raw or not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"metadata_json_parse_error": True}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name"):
        return value.name
    return str(value)


def strip_large_sets(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    for key in ("chunk_ids",):
        if key in out:
            out[f"{key}_count"] = len(out[key])
            out[f"{key}_sample"] = sorted(out[key])[:20]
            out.pop(key)
    return out


def stringify_milvus_type(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


if __name__ == "__main__":
    try:
        main()
    except Blocker as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        sys.exit(2)
