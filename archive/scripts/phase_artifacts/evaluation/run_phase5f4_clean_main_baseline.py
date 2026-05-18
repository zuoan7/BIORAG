#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pymilvus import MilvusClient
except Exception:  # pragma: no cover - inventory records the import failure.
    MilvusClient = None  # type: ignore[assignment]

from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import QueryFilters, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import _tokenize, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever


DATASET = Path("reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl")
OUT_DIR = Path("reports/phase5f4_clean_main_baseline")
TOP_K = 20
CANDIDATE_LIMIT = 40
PRIMARY_VARIANT = "current_default"
QUERY_TYPES = ("table_content", "caption_level_table", "figure_caption", "normal_control")
FAILURE_CATEGORIES = (
    "eval_sample_issue",
    "target_mapping_issue",
    "doc_recall_issue",
    "chunk_ranking_issue",
    "stable_block_not_retrieved",
    "table_related_text_gap",
    "caption_retrieval_gap",
    "figure_caption_gap",
    "normal_control_gap",
    "possible_table_takeover",
    "possible_metadata_stale",
    "asset_or_index_issue",
    "needs_manual_review",
)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    label: str
    chunks_path: Path
    milvus_uri: Path | None
    collection: str
    bm25_cache_path: Path | None
    role: str


@dataclass
class ChunkIndex:
    chunks: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    by_doc_block: dict[tuple[str, str], list[dict[str, Any]]]


class InMemoryBM25:
    def __init__(self, chunks: list[dict[str, Any]], config: RetrievalConfig):
        self.config = config
        self.records: list[RetrievedChunk] = [chunk_to_retrieved(item) for item in chunks]
        self.doc_len: list[int] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freqs: list[Counter[str]] = []
        self.avgdl = 0.0
        self._build_index()

    def _build_index(self) -> None:
        for record in self.records:
            terms = _tokenize(retrieval_text(record))
            tf = Counter(terms)
            self.term_freqs.append(tf)
            self.doc_len.append(len(terms))
            for term in tf:
                self.doc_freq[term] += 1
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    def search(self, question: str, limit: int, filters: QueryFilters | None = None) -> list[RetrievedChunk]:
        query_terms = tokenize_query(question) or _tokenize(question)
        if not query_terms:
            return []
        scored: list[RetrievedChunk] = []
        for idx, record in self._filter_records(filters):
            score = self._score(query_terms, idx)
            if score <= 0:
                continue
            item = clone_chunk(record)
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
    def __init__(self, uri: Path, collection: str, config: RetrievalConfig, embedder: BGEM3Embedder):
        if MilvusClient is None:
            raise RuntimeError("pymilvus is not importable")
        self.uri = str(uri.resolve())
        self.collection = collection
        self.config = config
        self.embedder = embedder
        self.client = MilvusClient(self.uri)
        self._embedding_cache: dict[str, list[float]] = {}

    def search(self, question: str, limit: int, filters: QueryFilters | None = None) -> list[RetrievedChunk]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5F-4 clean-main retrieval-only baseline.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-limit", type=int, default=CANDIDATE_LIMIT)
    parser.add_argument("--model-path", default="models/BAAI/bge-m3")
    parser.add_argument("--embedding-max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, freeze = freeze_dataset(dataset_path, output_dir)
    specs = candidate_asset_specs()
    inventory = inventory_assets(specs)
    write_json(output_dir / "retrieval_asset_inventory.json", inventory)
    write_asset_inventory_md(output_dir / "retrieval_asset_inventory.md", inventory)
    write_eval_protocol(output_dir / "eval_protocol.md", dataset_path, args.top_k, args.candidate_limit, inventory)

    if freeze["structural_blockers"]:
        write_blocked_outputs(output_dir, freeze, inventory)
        return

    runnable = choose_runnable_variants(specs, inventory)
    if not runnable:
        bm25_spec = first_bm25_sanity_spec(specs, inventory)
        if bm25_spec is None:
            write_missing_asset_outputs(output_dir, freeze, inventory)
            return
        runnable = [bm25_spec]

    results, topk_rows = run_retrieval(samples, runnable, inventory, args)
    write_retrieval_outputs(output_dir, samples, freeze, inventory, results, topk_rows, args)
    write_failure_outputs(output_dir, samples, results, primary_variant_name(results))
    write_reviews(output_dir, results, primary_variant_name(results))
    write_variant_comparison(output_dir, results)
    write_closeout_reports(output_dir, freeze, inventory, results, primary_variant_name(results))


def freeze_dataset(path: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exists = path.exists()
    sha = sha256_file(path) if exists else ""
    parse_errors: list[str] = []
    samples: list[dict[str, Any]] = []
    if exists:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_no}: {exc}")
                    continue
                if not isinstance(item, dict):
                    parse_errors.append(f"line {line_no}: expected object")
                    continue
                item["_line_no"] = line_no
                samples.append(item)

    query_type_counts = Counter(str(s.get("query_type", "")) for s in samples)
    ability_scope_counts = Counter(str(s.get("ability_scope", "")) for s in samples)
    sample_id_counts = Counter(str(s.get("sample_id", "")) for s in samples)
    query_counts = Counter(normalize_space(str(s.get("query", ""))) for s in samples)
    triplet_counts = Counter(target_triplet(s) for s in samples)
    stable_nonempty = sum(1 for s in samples if stable_ids(s))
    target_doc_empty = [sample_id(s) for s in samples if not str(s.get("target_doc_id") or "").strip()]
    include_not_true = [sample_id(s) for s in samples if s.get("include_in_main_denominator") is not True]
    hard_not_true = [sample_id(s) for s in samples if s.get("hard_rule_passed") is not True]
    empty_stable = [sample_id(s) for s in samples if not stable_ids(s)]
    duplicate_sample_ids = sorted(k for k, v in sample_id_counts.items() if k and v > 1)
    duplicate_queries = sorted(k for k, v in query_counts.items() if k and v > 1)
    duplicate_triplets = sorted("|".join(k) for k, v in triplet_counts.items() if all(k) and v > 1)
    schema_rows = [schema_check_row(s) for s in samples]
    query_lengths = [len(str(s.get("query", ""))) for s in samples]
    query_token_lengths = [len(str(s.get("query", "")).split()) for s in samples]
    target_doc_counts = Counter(str(s.get("target_doc_id", "")) for s in samples)
    residuals = residual_summary(samples)
    target_chunk_id_only = [
        sample_id(s)
        for s in samples
        if bool(s.get("target_chunk_id_only")) or (bool(s.get("target_chunk_id")) and not stable_ids(s))
    ]
    row_cell_queries = [sample_id(s) for s in samples if is_row_cell_query(str(s.get("query", "")))]
    ocr_image_queries = [sample_id(s) for s in samples if is_ocr_image_query(str(s.get("query", "")))]
    metadata_caveats = metadata_stale_caveats(samples)
    blockers: list[str] = []
    if not exists:
        blockers.append("dataset_file_missing")
    if parse_errors:
        blockers.append("jsonl_parse_failed")
    if empty_stable:
        blockers.append("stable_target_block_ids_empty")
    if include_not_true:
        blockers.append("include_in_main_denominator_not_true")
    if duplicate_sample_ids:
        blockers.append("sample_id_duplicate")
    if target_doc_empty:
        blockers.append("target_doc_id_empty")

    freeze = {
        "dataset_path": str(path),
        "exists": exists,
        "sha256": sha,
        "total_samples": len(samples),
        "query_type_distribution": dict(query_type_counts),
        "ability_scope_distribution": dict(ability_scope_counts),
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_queries": duplicate_queries,
        "duplicate_query_target_doc_stable_ids": duplicate_triplets,
        "stable_target_block_ids_coverage": stable_nonempty / len(samples) if samples else 0.0,
        "stable_target_block_ids_nonempty_count": stable_nonempty,
        "target_doc_id_empty_sample_ids": target_doc_empty,
        "include_in_main_denominator_all_true": not include_not_true and bool(samples),
        "include_in_main_denominator_not_true_sample_ids": include_not_true,
        "hard_rule_passed_all_true": not hard_not_true and bool(samples),
        "hard_rule_passed_not_true_sample_ids": hard_not_true,
        "residuals": residuals,
        "target_chunk_id_only_sample_ids": target_chunk_id_only,
        "row_cell_structured_table_query_sample_ids": row_cell_queries,
        "ocr_image_query_sample_ids": ocr_image_queries,
        "query_length_chars": describe_numbers(query_lengths),
        "query_length_tokens": describe_numbers(query_token_lengths),
        "target_doc_id_distribution": dict(target_doc_counts),
        "target_doc_id_top20": target_doc_counts.most_common(20),
        "metadata_caveats": metadata_caveats,
        "known_caveat": "target_semantic_type and other semantic metadata are treated as descriptive; stale metadata is not a blocker unless target matching is affected.",
        "jsonl_parse_errors": parse_errors,
        "structural_blockers": blockers,
    }
    write_json(output_dir / "dataset_manifest.json", freeze)
    (output_dir / "dataset_hash.txt").write_text(f"{sha}  {path}\n", encoding="utf-8")
    (output_dir / "dataset_sample_ids.txt").write_text(
        "\n".join(sample_id(s) for s in samples) + ("\n" if samples else ""),
        encoding="utf-8",
    )
    write_csv(output_dir / "dataset_schema_check.csv", schema_rows)
    write_dataset_freeze_md(output_dir / "dataset_freeze.md", freeze)
    return samples, freeze


def schema_check_row(sample: dict[str, Any]) -> dict[str, Any]:
    query = str(sample.get("query", ""))
    stable = stable_ids(sample)
    return {
        "line_no": sample.get("_line_no", ""),
        "sample_id": sample_id(sample),
        "query_type": sample.get("query_type", ""),
        "ability_scope": sample.get("ability_scope", ""),
        "target_doc_id": sample.get("target_doc_id", ""),
        "query_len_chars": len(query),
        "query_len_tokens": len(query.split()),
        "stable_target_block_ids_count": len(stable),
        "include_in_main_denominator_is_true": sample.get("include_in_main_denominator") is True,
        "hard_rule_passed_is_true": sample.get("hard_rule_passed") is True,
        "target_doc_id_empty": not bool(str(sample.get("target_doc_id") or "").strip()),
        "caption_residual_query": bool(re.search(r"\bCAPTION\b", query)),
        "which_table_residual_query": bool(re.search(r"\bWhich\s+table\b", query, re.I)),
        "where_residual_query": bool(re.search(r"\bWhere\b", query, re.I)),
        "table_caption_residual_query": bool(re.search(r"\bTable\s+caption\b", query, re.I)),
        "target_chunk_id_only": bool(sample.get("target_chunk_id_only")) or (bool(sample.get("target_chunk_id")) and not stable),
        "row_cell_structured_table_query": is_row_cell_query(query),
        "ocr_image_query": is_ocr_image_query(query),
    }


def residual_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = {
        "CAPTION": re.compile(r"\bCAPTION\b"),
        "Which table": re.compile(r"\bWhich\s+table\b", re.I),
        "Where": re.compile(r"\bWhere\b", re.I),
        "Table caption": re.compile(r"\bTable\s+caption\b", re.I),
    }
    out: dict[str, Any] = {}
    for name, pattern in patterns.items():
        sample_ids = [sample_id(s) for s in samples if pattern.search(str(s.get("query", "")))]
        out[name] = {"query_count": len(sample_ids), "sample_ids": sample_ids}
    return out


def metadata_stale_caveats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for sample in samples:
        qtype = str(sample.get("query_type", ""))
        semantic_type = str(sample.get("target_semantic_type") or "")
        query = str(sample.get("query") or "").lower()
        stale_hint = False
        if qtype == "normal_control" and any(word in semantic_type for word in ("table", "caption", "figure")):
            stale_hint = True
        if qtype in {"table_content", "caption_level_table"} and "figure" in semantic_type and "figure" not in query:
            stale_hint = True
        if stale_hint:
            rows.append(
                {
                    "sample_id": sample_id(sample),
                    "query_type": qtype,
                    "target_semantic_type": semantic_type,
                }
            )
    return {
        "possible_metadata_stale_count": len(rows),
        "examples": rows[:20],
        "blocking": False,
    }


def candidate_asset_specs() -> list[AssetSpec]:
    env = read_env_file(ROOT / ".env")
    milvus_uri = Path(os.getenv("SYNBIO_MILVUS_URI") or os.getenv("MILVUS_URI") or env.get("SYNBIO_MILVUS_URI") or env.get("MILVUS_URI") or "runtime/vectorstores/milvus/papers.db")
    collection = os.getenv("MILVUS_COLLECTION") or env.get("MILVUS_COLLECTION") or "synbio_papers"
    if not milvus_uri.is_absolute():
        milvus_uri = ROOT / milvus_uri
    return [
        AssetSpec(
            name="current_default",
            label="Current default production/baseline index",
            chunks_path=ROOT / "data/paper_round1/chunks/chunks.jsonl",
            milvus_uri=milvus_uri,
            collection=collection,
            bm25_cache_path=ROOT / "data/paper_round1/chunks/bm25_index.json",
            role="baseline/default",
        ),
        AssetSpec(
            name="phase5c5_baseline_full",
            label="Phase 5C-5 baseline full compact index",
            chunks_path=Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
            milvus_uri=Path("/tmp/phase5c5_baseline_full.db"),
            collection="synbio_phase5c5_baseline_full",
            bm25_cache_path=Path("/tmp/biorag_phase5c5_baseline_full/bm25_index.json"),
            role="baseline/default",
        ),
        AssetSpec(
            name="phase5c5_enhanced_full",
            label="Phase 5C full enhanced table index",
            chunks_path=Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
            milvus_uri=Path("/tmp/phase5c5_enhanced_full.db"),
            collection="synbio_phase5c5_enhanced_full",
            bm25_cache_path=Path("/tmp/biorag_phase5c5_enhanced_full/bm25_index.json"),
            role="table_enhanced",
        ),
        AssetSpec(
            name="phase5d_caption_cleanup_chunks",
            label="Phase 5D cleanup experimental chunks",
            chunks_path=Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
            milvus_uri=None,
            collection="",
            bm25_cache_path=None,
            role="caption_cleanup_optional",
        ),
    ]


def inventory_assets(specs: list[AssetSpec]) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for spec in specs:
        chunk_count = count_lines(spec.chunks_path) if spec.chunks_path.exists() else 0
        milvus_info = inspect_milvus(spec.milvus_uri, spec.collection)
        bm25_info = inspect_bm25_cache(spec.bm25_cache_path)
        row_count = milvus_info.get("row_count")
        assets[spec.name] = {
            "label": spec.label,
            "role": spec.role,
            "chunks_path": str(spec.chunks_path),
            "chunks_exists": spec.chunks_path.exists(),
            "chunk_count": chunk_count,
            "milvus_uri": str(spec.milvus_uri) if spec.milvus_uri else "",
            "collection": spec.collection,
            "milvus": milvus_info,
            "bm25_cache_path": str(spec.bm25_cache_path) if spec.bm25_cache_path else "",
            "bm25_cache": bm25_info,
            "index_row_count_matches_chunks": bool(row_count == chunk_count) if row_count is not None and chunk_count else False,
            "can_in_memory_bm25_sanity": spec.chunks_path.exists(),
            "runnable_full_hybrid": bool(
                spec.chunks_path.exists()
                and spec.milvus_uri
                and milvus_info.get("collection_available")
                and chunk_count > 0
            ),
        }
    full_available = any(item["runnable_full_hybrid"] for item in assets.values())
    return {
        "generated_at": utc_now(),
        "index_rebuilt": False,
        "qwen_called": False,
        "ragas_run": False,
        "assets": assets,
        "full_index_available": full_available,
        "all_full_index_unavailable": not full_available,
        "obvious_missing_paths": [
            {"asset": name, "path": item["chunks_path"], "kind": "chunks"}
            for name, item in assets.items()
            if not item["chunks_exists"]
        ]
        + [
            {"asset": name, "path": item["milvus_uri"], "kind": "milvus"}
            for name, item in assets.items()
            if item["milvus_uri"] and not item["milvus"].get("db_exists")
        ],
        "rebuild_need": "No rebuild needed for this run because at least one full hybrid-capable index is available."
        if full_available
        else "A separate rebuild plan is needed before full hybrid baseline; this script will not rebuild.",
    }


def inspect_milvus(uri: Path | None, collection: str) -> dict[str, Any]:
    if uri is None:
        return {"db_exists": False, "collection_available": False, "row_count": None, "notes": "no milvus index configured"}
    info: dict[str, Any] = {
        "db_exists": uri.exists(),
        "collection_available": False,
        "row_count": None,
        "collections": [],
        "error": "",
    }
    if not uri.exists():
        return info
    if MilvusClient is None:
        info["error"] = "pymilvus import failed"
        return info
    try:
        client = MilvusClient(str(uri.resolve()))
        collections = client.list_collections()
        info["collections"] = collections
        info["collection_available"] = collection in collections
        if info["collection_available"]:
            stats = client.get_collection_stats(collection)
            info["row_count"] = int(stats.get("row_count", 0))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def inspect_bm25_cache(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "record_count": None, "error": "", "notes": "no cache configured"}
    info: dict[str, Any] = {"exists": path.exists(), "record_count": None, "error": ""}
    if not path.exists():
        return info
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        info["record_count"] = len(records) if isinstance(records, list) else None
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def choose_runnable_variants(specs: list[AssetSpec], inventory: dict[str, Any]) -> list[AssetSpec]:
    runnable = []
    assets = inventory["assets"]
    for spec in specs:
        if assets.get(spec.name, {}).get("runnable_full_hybrid"):
            runnable.append(spec)
    preferred = ["current_default", "phase5c5_baseline_full", "phase5c5_enhanced_full"]
    return sorted(runnable, key=lambda spec: preferred.index(spec.name) if spec.name in preferred else 99)


def first_bm25_sanity_spec(specs: list[AssetSpec], inventory: dict[str, Any]) -> AssetSpec | None:
    for spec in specs:
        if inventory["assets"].get(spec.name, {}).get("can_in_memory_bm25_sanity"):
            return AssetSpec(
                name=f"{spec.name}_bm25_sanity",
                label=f"{spec.label} BM25-only preliminary sanity",
                chunks_path=spec.chunks_path,
                milvus_uri=None,
                collection="",
                bm25_cache_path=spec.bm25_cache_path,
                role="bm25_only_preliminary",
            )
    return None


def run_retrieval(
    samples: list[dict[str, Any]],
    specs: list[AssetSpec],
    inventory: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    embedder: BGEM3Embedder | None = None
    if any(spec.milvus_uri for spec in specs):
        embedder = BGEM3Embedder(str(model_path), dim=1024, max_length=args.embedding_max_length)

    all_results: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for spec in specs:
        start = time.time()
        chunk_index = build_chunk_index(load_chunks(spec.chunks_path))
        config = RetrievalConfig(
            milvus_uri=str(spec.milvus_uri or ""),
            collection_name=spec.collection,
            search_limit=args.candidate_limit,
            dense_limit=args.candidate_limit,
            bm25_limit=args.candidate_limit,
        )
        bm25 = InMemoryBM25(chunk_index.chunks, config)
        if spec.milvus_uri and embedder is not None:
            dense = DenseMilvusAdapter(spec.milvus_uri, spec.collection, config, embedder)
            hybrid = HybridRetriever(config, dense, bm25)
            retrieval_mode = "hybrid_rrf_existing_milvus_inmemory_bm25"
        else:
            dense = None
            hybrid = None
            retrieval_mode = "bm25_only_preliminary_sanity"

        for sample in samples:
            query = str(sample.get("query") or "")
            if hybrid is not None:
                hits = hybrid.search(query, limit=args.candidate_limit, filters=None, analysis=None)
            else:
                hits = bm25.search(query, limit=args.candidate_limit, filters=None)
            row, example = evaluate_sample(sample, hits, chunk_index, spec.name, retrieval_mode, args.top_k)
            all_results.append(row)
            topk_rows.append(example)
        elapsed = time.time() - start
        inventory["assets"][spec.name]["retrieval_elapsed_sec"] = round(elapsed, 3)
    return all_results, topk_rows


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_chunk_index(chunks: list[dict[str, Any]]) -> ChunkIndex:
    by_id: dict[str, dict[str, Any]] = {}
    by_doc_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        doc_id = str(chunk.get("doc_id") or "")
        by_id[chunk_id] = chunk
        for block_id in chunk_block_ids(chunk):
            by_doc_block[(doc_id, block_id)].append(chunk)
    return ChunkIndex(chunks=chunks, by_id=by_id, by_doc_block=by_doc_block)


def evaluate_sample(
    sample: dict[str, Any],
    hits: list[RetrievedChunk],
    chunk_index: ChunkIndex,
    variant: str,
    retrieval_mode: str,
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
    target_candidates = candidate_chunks_for_stable(chunk_index, target_doc_id, stable)
    top_hits = [serialize_hit(hit, rank, chunk_index) for rank, hit in enumerate(hits[:top_k], start=1)]
    top10 = top_hits[:10]
    table_related_top10_count = sum(1 for hit in top10 if hit["table_related"])
    table_or_caption_top10_count = sum(1 for hit in top10 if hit["table_or_caption_related"])
    notes = []
    if not target_candidates:
        notes.append("stable target block ids did not map to this chunk set")
    if retrieval_mode.startswith("bm25_only"):
        notes.append("BM25-only preliminary baseline, not full hybrid baseline")
    row = {
        "sample_id": sample_id(sample),
        "query_type": sample.get("query_type", ""),
        "ability_scope": sample.get("ability_scope", ""),
        "query": sample.get("query", ""),
        "target_doc_id": target_doc_id,
        "stable_target_block_ids": stable,
        "stable_target_chunk_ids": [c.get("chunk_id", "") for c in target_candidates],
        "stable_target_mapping_found": bool(target_candidates),
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
        "retrieval_mode": retrieval_mode,
        "index_variant": variant,
        "table_related_top10_count": table_related_top10_count,
        "table_or_caption_top10_count": table_or_caption_top10_count,
        "table_related_top10_occupancy": table_related_top10_count / len(top10) if top10 else 0.0,
        "table_or_caption_top10_occupancy": table_or_caption_top10_count / len(top10) if top10 else 0.0,
        "notes": "; ".join(notes),
    }
    example = {
        "sample_id": row["sample_id"],
        "query_type": row["query_type"],
        "index_variant": variant,
        "retrieval_mode": retrieval_mode,
        "query": row["query"],
        "target_doc_id": target_doc_id,
        "stable_target_block_ids": stable,
        "first_doc_hit_rank": doc_rank,
        "first_stable_block_hit_rank": stable_rank,
        "top_hits": top_hits,
    }
    return row, example


def candidate_chunks_for_stable(index: ChunkIndex, doc_id: str, stable: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for block_id in stable:
        for chunk in index.by_doc_block.get((doc_id, block_id), []):
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            candidates.append(chunk)
    return candidates


def serialize_hit(hit: RetrievedChunk, rank: int, chunk_index: ChunkIndex) -> dict[str, Any]:
    chunk = chunk_index.by_id.get(hit.chunk_id, {})
    metadata = dict(hit.metadata or {})
    block_ids = chunk_block_ids(chunk) or listify(metadata.get("source_block_ids"))
    block_types = listify(metadata.get("block_types")) or listify(chunk.get("block_types"))
    evidence_types = listify(metadata.get("evidence_types")) or listify(chunk.get("evidence_types"))
    table_related = chunk_has_table_related(chunk)
    contains_table_text = bool(metadata.get("contains_table_text")) or bool(chunk.get("contains_table_text"))
    contains_table_caption = bool(metadata.get("contains_table_caption")) or bool(chunk.get("contains_table_caption"))
    contains_figure_caption = bool(metadata.get("contains_figure_caption")) or bool(chunk.get("contains_figure_caption"))
    table_or_caption_related = bool(
        table_related
        or contains_table_text
        or contains_table_caption
        or contains_figure_caption
        or {"table_text", "table_caption", "figure_caption"} & set(block_types + evidence_types)
    )
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section": hit.section,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "block_ids": block_ids,
        "block_types": block_types,
        "evidence_types": evidence_types,
        "contains_table_text": contains_table_text,
        "contains_table_caption": contains_table_caption,
        "contains_figure_caption": contains_figure_caption,
        "table_related": table_related,
        "table_or_caption_related": table_or_caption_related,
        "score": hit.fusion_score or hit.vector_score or hit.bm25_score or 0.0,
        "text_preview": truncate(hit.text, 220),
    }


def write_retrieval_outputs(
    output_dir: Path,
    samples: list[dict[str, Any]],
    freeze: dict[str, Any],
    inventory: dict[str, Any],
    results: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    metrics = build_metrics(results)
    payload = {
        "run_config": {
            "phase": "Phase 5F-4 Clean Main Baseline Establishment",
            "dataset": freeze["dataset_path"],
            "top_k": args.top_k,
            "candidate_limit": args.candidate_limit,
            "retrieval_only": True,
            "qwen_called": False,
            "generation_eval": False,
            "ragas_eval": False,
            "ocr": False,
            "index_rebuilt": False,
            "target_matching": "stable_target_block_ids corrected matching",
        },
        "dataset": {
            "sha256": freeze["sha256"],
            "total_samples": freeze["total_samples"],
            "query_type_distribution": freeze["query_type_distribution"],
        },
        "asset_inventory": {
            "used_variants": sorted({r["index_variant"] for r in results}),
            "full_index_available": inventory["full_index_available"],
        },
        "primary_variant": primary_variant_name(results),
        "metrics": metrics,
    }
    write_json(output_dir / "main_results.json", payload)
    write_jsonl(output_dir / "per_sample_results.jsonl", results)
    write_jsonl(output_dir / "topk_examples.jsonl", topk_rows)
    by_type_rows = metrics_rows(metrics)
    write_csv(output_dir / "main_results_by_query_type.csv", by_type_rows)
    write_results_by_type_md(output_dir / "main_results_by_query_type.md", by_type_rows)


def build_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_variant[str(row["index_variant"])].append(row)
    for variant, rows in sorted(by_variant.items()):
        metrics[variant] = {
            "overall": aggregate_metric_rows(rows),
            "by_query_type": {
                qtype: aggregate_metric_rows([r for r in rows if r["query_type"] == qtype])
                for qtype in sorted({str(r["query_type"]) for r in rows})
            },
            "target_doc_id_recall": target_doc_recall(rows),
        }
    return metrics


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
        "mean_table_related_top10_occupancy": statistics.mean(float(r["table_related_top10_occupancy"]) for r in rows),
        "mean_table_or_caption_top10_occupancy": statistics.mean(float(r["table_or_caption_top10_occupancy"]) for r in rows),
        "normal_table_caption_takeover_count": sum(
            1
            for r in rows
            if r["query_type"] == "normal_control"
            and not r["stable_block_hit_at_10"]
            and int(r["table_or_caption_top10_count"]) > 0
        ),
    }


def target_doc_recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_doc_id"])].append(row)
    return {
        "target_doc_count": len(grouped),
        "doc_hit_at_10_by_target_doc": {
            doc_id: mean_bool(doc_rows, "doc_hit_at_10")
            for doc_id, doc_rows in sorted(grouped.items())
        },
    }


def metrics_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, payload in metrics.items():
        for qtype, values in {"overall": payload["overall"], **payload["by_query_type"]}.items():
            row = {"index_variant": variant, "query_type": qtype}
            row.update(values)
            rows.append(row)
    return rows


def write_failure_outputs(
    output_dir: Path,
    samples: list[dict[str, Any]],
    results: list[dict[str, Any]],
    primary_variant: str,
) -> None:
    ledger = []
    for row in results:
        is_near_miss = bool(not row["stable_block_hit_at_10"] and row["stable_block_hit_at_20"])
        is_miss = not row["stable_block_hit_at_10"]
        if not is_miss and not is_near_miss:
            continue
        category = classify_failure(row)
        ledger.append(
            {
                "index_variant": row["index_variant"],
                "sample_id": row["sample_id"],
                "query_type": row["query_type"],
                "target_doc_id": row["target_doc_id"],
                "first_doc_hit_rank": row["first_doc_hit_rank"],
                "first_stable_block_hit_rank": row["first_stable_block_hit_rank"],
                "doc_hit_at_10": row["doc_hit_at_10"],
                "stable_block_hit_at_10": row["stable_block_hit_at_10"],
                "stable_block_hit_at_20": row["stable_block_hit_at_20"],
                "near_miss": is_near_miss,
                "failure_category": category,
                "notes": row.get("notes", ""),
                "query": row["query"],
            }
        )
    write_csv(output_dir / "failure_ledger.csv", ledger)
    primary_rows = [r for r in results if r["index_variant"] == primary_variant]
    write_examples_md(output_dir / "miss_examples.md", primary_rows, want_near=False)
    write_examples_md(output_dir / "near_miss_examples.md", primary_rows, want_near=True)
    write_failure_taxonomy_summary(output_dir / "failure_taxonomy_summary.md", ledger, primary_rows)


def classify_failure(row: dict[str, Any]) -> str:
    qtype = str(row["query_type"])
    if not row["stable_target_mapping_found"]:
        return "target_mapping_issue"
    if not row["doc_hit_at_10"]:
        return "doc_recall_issue"
    if qtype == "normal_control" and int(row["table_or_caption_top10_count"]) > 0:
        return "possible_table_takeover"
    if row["stable_block_hit_at_20"] and not row["stable_block_hit_at_10"]:
        return "chunk_ranking_issue"
    if qtype == "table_content":
        return "table_related_text_gap"
    if qtype == "caption_level_table":
        return "caption_retrieval_gap"
    if qtype == "figure_caption":
        return "figure_caption_gap"
    if qtype == "normal_control":
        return "normal_control_gap"
    return "stable_block_not_retrieved"


def write_reviews(output_dir: Path, results: list[dict[str, Any]], primary_variant: str) -> None:
    filenames = {
        "table_content": "table_content_review.md",
        "caption_level_table": "caption_level_table_review.md",
        "figure_caption": "figure_caption_review.md",
        "normal_control": "normal_control_review.md",
    }
    for qtype, filename in filenames.items():
        write_review_md(output_dir / filename, qtype, results, primary_variant)


def write_review_md(path: Path, qtype: str, results: list[dict[str, Any]], primary_variant: str) -> None:
    primary_rows = [r for r in results if r["index_variant"] == primary_variant and r["query_type"] == qtype]
    metrics = aggregate_metric_rows(primary_rows)
    successes = [r for r in primary_rows if r["stable_block_hit_at_10"]][:5]
    misses = [r for r in primary_rows if not r["stable_block_hit_at_10"]][:5]
    near = [r for r in primary_rows if not r["stable_block_hit_at_10"] and r["stable_block_hit_at_20"]][:5]
    mapping_issue = sum(1 for r in primary_rows if not r["stable_target_mapping_found"])
    query_residual = "No structural query-template residual was detected in dataset freeze."
    if qtype == "caption_level_table":
        caution = "This slice is small; conclusions should be treated as a clean baseline only."
    elif qtype == "table_content":
        caution = "This slice is a clean baseline, not comprehensive acceptance."
    elif qtype == "figure_caption":
        caution = "This slice tests caption retrieval only, not image understanding."
    else:
        caution = "This is a protection set; normal failures remain important even if table slices perform well."
    lines = [
        f"# {qtype} review",
        "",
        f"Primary variant: `{primary_variant}`.",
        "",
        f"1. sample count: {metrics.get('count', 0)}.",
        f"2. doc_hit@10: {fmt_pct(metrics.get('doc_hit_at_10', 0.0))}.",
        f"3. stable_block_hit@10: {fmt_pct(metrics.get('stable_block_hit_at_10', 0.0))}.",
        f"4. stable_block_hit@20: {fmt_pct(metrics.get('stable_block_hit_at_20', 0.0))}.",
        f"5. main success pattern: {success_pattern(successes)}.",
        f"6. main failure pattern: {failure_pattern(misses)}.",
        f"7. target mapping issue: {mapping_issue} sample(s).",
        f"8. query quality residual issue: {query_residual}",
        f"9. follow-up dataset correction needed: {'yes' if mapping_issue else 'not immediately'}; review failures before editing data.",
        f"10. follow-up retrieval repair needed: {'yes' if misses else 'no immediate repair from this slice'}.",
        "11. parser/chunking repair needed: only if target mapping or repeated text gaps are confirmed manually.",
        f"12. recommend next phase: {'yes' if not mapping_issue else 'blocked until mapping review'}; {caution}",
        "",
        "Representative successes:",
    ]
    lines.extend(example_bullets(successes))
    lines.extend(["", "Top failures:"])
    lines.extend(example_bullets(misses))
    lines.extend(["", "Near misses:"])
    lines.extend(example_bullets(near))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_variant_comparison(output_dir: Path, results: list[dict[str, Any]]) -> None:
    variants = sorted({str(r["index_variant"]) for r in results})
    if len(variants) < 2:
        (output_dir / "variant_comparison.md").write_text(
            "# Variant comparison\n\nSingle-variant baseline only; comparison skipped.\n",
            encoding="utf-8",
        )
        return
    baseline = "phase5c5_baseline_full" if "phase5c5_baseline_full" in variants else variants[0]
    enhanced = "phase5c5_enhanced_full" if "phase5c5_enhanced_full" in variants else variants[1]
    comparison = compare_variants(results, baseline, enhanced)
    write_json(output_dir / "variant_comparison.json", comparison)
    write_variant_comparison_md(output_dir / "variant_comparison.md", comparison)
    write_rank_delta_examples(output_dir / "rank_delta_examples.md", results, baseline, enhanced)


def compare_variants(results: list[dict[str, Any]], baseline: str, enhanced: str) -> dict[str, Any]:
    b_rows = [r for r in results if r["index_variant"] == baseline]
    e_rows = [r for r in results if r["index_variant"] == enhanced]
    e_by_id = {r["sample_id"]: r for r in e_rows}
    pairs = [(b, e_by_id[b["sample_id"]]) for b in b_rows if b["sample_id"] in e_by_id]

    def paired_counts(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, int]:
        out = Counter()
        for b, e in rows:
            b_hit = bool(b["stable_block_hit_at_10"])
            e_hit = bool(e["stable_block_hit_at_10"])
            if b_hit and e_hit:
                out["both_success"] += 1
            elif b_hit and not e_hit:
                out["baseline_only"] += 1
            elif e_hit and not b_hit:
                out["enhanced_only"] += 1
            else:
                out["both_fail"] += 1
        return dict(out)

    by_type = {}
    for qtype in sorted({str(r["query_type"]) for r in b_rows}):
        q_pairs = [(b, e) for b, e in pairs if b["query_type"] == qtype]
        by_type[qtype] = {
            "baseline": aggregate_metric_rows([b for b, _ in q_pairs]),
            "enhanced": aggregate_metric_rows([e for _, e in q_pairs]),
            "paired": paired_counts(q_pairs),
        }
    return {
        "wording_guardrail": "Current performance comparison on strict_main_eval_set_v2 only; not Phase 5C/5D effect validation.",
        "baseline_variant": baseline,
        "enhanced_variant": enhanced,
        "overall": {
            "baseline": aggregate_metric_rows([b for b, _ in pairs]),
            "enhanced": aggregate_metric_rows([e for _, e in pairs]),
            "paired": paired_counts(pairs),
        },
        "by_query_type": by_type,
    }


def write_closeout_reports(
    output_dir: Path,
    freeze: dict[str, Any],
    inventory: dict[str, Any],
    results: list[dict[str, Any]],
    primary_variant: str,
) -> None:
    metrics = build_metrics(results)
    primary_metrics = metrics.get(primary_variant, {})
    failure_counts = Counter(classify_failure(r) for r in results if r["index_variant"] == primary_variant and not r["stable_block_hit_at_10"])
    max_failure = failure_counts.most_common(1)[0][0] if failure_counts else "none"
    summary_lines = [
        "# Phase 5F-4 summary",
        "",
        f"1. strict_main_eval_set_v2 frozen: {'yes' if not freeze['structural_blockers'] else 'no'}.",
        f"2. dataset hash: `{freeze['sha256']}`.",
        f"3. dataset total: {freeze['total_samples']}.",
        f"4. query_type distribution: {json.dumps(freeze['query_type_distribution'], ensure_ascii=False)}.",
        f"5. structural blocker: {'none' if not freeze['structural_blockers'] else ', '.join(freeze['structural_blockers'])}.",
        f"6. retrieval assets used: {', '.join(sorted({r['index_variant'] for r in results}))}.",
        "7. index rebuilt: no.",
        "8. Qwen called: no.",
        "9. RAGAS run: no.",
        f"10. primary overall doc_hit@10: {fmt_pct(primary_metrics.get('overall', {}).get('doc_hit_at_10', 0.0))}.",
        f"11. primary overall stable_block_hit@10: {fmt_pct(primary_metrics.get('overall', {}).get('stable_block_hit_at_10', 0.0))}.",
        "12. query_type results:",
    ]
    for qtype, values in primary_metrics.get("by_query_type", {}).items():
        summary_lines.append(
            f"   - {qtype}: doc_hit@10={fmt_pct(values.get('doc_hit_at_10', 0.0))}, "
            f"stable_block_hit@10={fmt_pct(values.get('stable_block_hit_at_10', 0.0))}, "
            f"stable_block_hit@20={fmt_pct(values.get('stable_block_hit_at_20', 0.0))}"
        )
    summary_lines.extend(
        [
            "13. diagnostic/lexical mixed into main: no.",
            "14. old-change effect validation: no.",
            "15. clean baseline establishment: yes.",
            "16. recommend using this baseline as future comparison start: yes, with the fixed hash above.",
            f"17. largest failure type: {max_failure}.",
            "18. recommend immediate code changes: no; this phase only establishes baseline and backlog.",
            f"19. recommend next phase: {'yes' if not freeze['structural_blockers'] else 'no'}.",
            "",
            f"Primary variant: `{primary_variant}`.",
            f"Rebuild need statement: {inventory['rebuild_need']}",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "clean_baseline_closeout.md").write_text(
        "\n".join(
            [
                "# Clean baseline closeout",
                "",
                "Phase 5F-4 established a retrieval-only baseline on the cleaned strict main eval set.",
                "This is not Phase 5C/5D effect validation and should not be compared directly to old-denominator metrics.",
                "",
                f"- dataset_sha256: `{freeze['sha256']}`",
                f"- primary_variant: `{primary_variant}`",
                "- index_rebuilt: no",
                "- qwen_called: no",
                "- ragas_run: no",
                f"- largest_failure_type: {max_failure}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_backlog(output_dir / "next_repair_backlog.md", results, primary_variant)
    write_next_phase_plan(output_dir / "next_phase_plan.md", freeze, max_failure)


def write_backlog(path: Path, results: list[dict[str, Any]], primary_variant: str) -> None:
    rows = [r for r in results if r["index_variant"] == primary_variant and not r["stable_block_hit_at_10"]]
    buckets = {
        "P0": [r for r in rows if classify_failure(r) == "target_mapping_issue"],
        "P1": [r for r in rows if classify_failure(r) in {"doc_recall_issue", "chunk_ranking_issue", "stable_block_not_retrieved", "table_related_text_gap", "caption_retrieval_gap", "figure_caption_gap", "normal_control_gap", "possible_table_takeover"}],
        "P2": [r for r in rows if classify_failure(r) in {"asset_or_index_issue"}],
        "P3": [],
    }
    lines = ["# Next repair backlog", ""]
    lines.append("P0: blocking dataset or target mapping issues")
    lines.extend(example_bullets(buckets["P0"][:20]))
    lines.extend(["", "P1: true retrieval issues"])
    lines.extend(example_bullets(buckets["P1"][:20]))
    lines.extend(["", "P2: parser/chunking follow-up issues"])
    lines.extend(example_bullets(buckets["P2"][:20]) or ["- none identified from automatic classification."])
    lines.extend(["", "P3: future structured table / OCR / image capabilities"])
    lines.append("- No P3 item is implemented in this phase. Structured table objects, OCR, and image understanding remain future scope.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_phase_plan(path: Path, freeze: dict[str, Any], max_failure: str) -> None:
    lines = [
        "# Next phase plan",
        "",
        "1. Keep strict_main_eval_set_v2 fixed by SHA256 for future comparisons.",
        "2. Manually review P0 target mapping rows before any dataset edit.",
        f"3. Prioritize P1 retrieval investigation around `{max_failure}` if no P0 blocker remains.",
        "4. Only after review, decide whether a separate rebuild plan is needed for missing optional assets.",
        "5. Do not mix diagnostic or lexical-stress sets into the main denominator.",
    ]
    if freeze["structural_blockers"]:
        lines.append(f"6. Current blocker(s): {', '.join(freeze['structural_blockers'])}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_outputs(output_dir: Path, freeze: dict[str, Any], inventory: dict[str, Any]) -> None:
    payload = {
        "blocked": True,
        "reason": "structural dataset blocker",
        "structural_blockers": freeze["structural_blockers"],
        "index_rebuilt": False,
        "qwen_called": False,
        "ragas_run": False,
    }
    write_json(output_dir / "main_results.json", payload)
    (output_dir / "summary.md").write_text(
        "# Phase 5F-4 summary\n\nRetrieval testing stopped because structural dataset blocker(s) were found: "
        + ", ".join(freeze["structural_blockers"])
        + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "per_sample_results.jsonl", [])
    write_json(output_dir / "topk_examples.jsonl", [])
    write_json(output_dir / "variant_comparison.json", {"skipped": True, "reason": "dataset_blocked"})
    write_next_phase_plan(output_dir / "next_phase_plan.md", freeze, "dataset_blocker")


def write_missing_asset_outputs(output_dir: Path, freeze: dict[str, Any], inventory: dict[str, Any]) -> None:
    payload = {
        "blocked": True,
        "reason": "missing retrieval assets",
        "index_rebuilt": False,
        "qwen_called": False,
        "ragas_run": False,
    }
    write_json(output_dir / "main_results.json", payload)
    (output_dir / "summary.md").write_text(
        "# Phase 5F-4 summary\n\nRetrieval testing stopped because no full index or chunks were available. No rebuild was attempted.\n",
        encoding="utf-8",
    )


def primary_variant_name(results: list[dict[str, Any]]) -> str:
    variants = {str(r["index_variant"]) for r in results}
    if PRIMARY_VARIANT in variants:
        return PRIMARY_VARIANT
    return sorted(variants)[0] if variants else PRIMARY_VARIANT


def write_dataset_freeze_md(path: Path, freeze: dict[str, Any]) -> None:
    lines = [
        "# Dataset freeze",
        "",
        f"- exists: {yes_no(freeze['exists'])}",
        f"- sha256: `{freeze['sha256']}`",
        f"- total samples: {freeze['total_samples']}",
        f"- query_type distribution: {json.dumps(freeze['query_type_distribution'], ensure_ascii=False)}",
        f"- ability_scope distribution: {json.dumps(freeze['ability_scope_distribution'], ensure_ascii=False)}",
        f"- duplicate sample_id count: {len(freeze['duplicate_sample_ids'])}",
        f"- duplicate query count: {len(freeze['duplicate_queries'])}",
        f"- duplicate query+target_doc_id+stable_target_block_ids count: {len(freeze['duplicate_query_target_doc_stable_ids'])}",
        f"- stable_target_block_ids coverage: {fmt_pct(freeze['stable_target_block_ids_coverage'])}",
        f"- target_doc_id empty count: {len(freeze['target_doc_id_empty_sample_ids'])}",
        f"- include_in_main_denominator all true: {yes_no(freeze['include_in_main_denominator_all_true'])}",
        f"- hard_rule_passed all true: {yes_no(freeze['hard_rule_passed_all_true'])}",
        f"- target_chunk_id_only count: {len(freeze['target_chunk_id_only_sample_ids'])}",
        f"- row/cell structured table query count: {len(freeze['row_cell_structured_table_query_sample_ids'])}",
        f"- OCR/image query count: {len(freeze['ocr_image_query_sample_ids'])}",
        f"- query length chars: {json.dumps(freeze['query_length_chars'])}",
        f"- query length tokens: {json.dumps(freeze['query_length_tokens'])}",
        f"- structural blockers: {'none' if not freeze['structural_blockers'] else ', '.join(freeze['structural_blockers'])}",
        "",
        "Residual query templates:",
    ]
    for name, item in freeze["residuals"].items():
        lines.append(f"- {name}: {item['query_count']}")
    lines.extend(["", "Top 20 target_doc_id counts:"])
    for doc_id, count in freeze["target_doc_id_top20"]:
        lines.append(f"- {doc_id}: {count}")
    lines.extend(
        [
            "",
            "Known caveat:",
            f"- {freeze['known_caveat']}",
            f"- possible metadata stale count: {freeze['metadata_caveats']['possible_metadata_stale_count']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_asset_inventory_md(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Retrieval asset inventory",
        "",
        f"- index rebuilt: {yes_no(inventory['index_rebuilt'])}",
        f"- full index available: {yes_no(inventory['full_index_available'])}",
        f"- rebuild need: {inventory['rebuild_need']}",
        "",
        "| asset | role | chunks | chunk_count | milvus_collection | row_count | row_count_matches_chunks | bm25_cache | bm25_records | runnable_full_hybrid |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---|",
    ]
    for name, item in inventory["assets"].items():
        lines.append(
            "| {name} | {role} | {chunks} | {chunk_count} | {collection} | {row_count} | {match} | {bm25} | {bm25_count} | {runnable} |".format(
                name=name,
                role=item["role"],
                chunks=yes_no(item["chunks_exists"]),
                chunk_count=item["chunk_count"],
                collection=item["collection"] or "n/a",
                row_count=item["milvus"].get("row_count"),
                match=yes_no(item["index_row_count_matches_chunks"]),
                bm25=yes_no(item["bm25_cache"].get("exists")),
                bm25_count=item["bm25_cache"].get("record_count"),
                runnable=yes_no(item["runnable_full_hybrid"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_protocol(path: Path, dataset_path: Path, top_k: int, candidate_limit: int, inventory: dict[str, Any]) -> None:
    lines = [
        "# Eval protocol",
        "",
        "This run is Clean Main Baseline Establishment.",
        "",
        "This run is not Phase 5C/5D effect validation.",
        "",
        f"Main denominator: `{dataset_path}`.",
        "",
        "Primary metrics:",
        "- doc_hit@10",
        "- stable_block_hit@10",
        "- stable_block_hit@20",
        "- query_type breakdown",
        "- target_doc_id recall",
        "- corrected stable target matching via stable_target_block_ids",
        "",
        "Auxiliary metrics:",
        "- rank of first doc hit",
        "- rank of first stable block hit",
        "- hit@1 / hit@5 / hit@10 / hit@20",
        "- top-k table_related occupancy",
        "- normal table/caption takeover",
        "- baseline_only / enhanced_only / both_success / both_fail when multiple variants are available",
        "",
        "Forbidden for this run:",
        "- target_chunk_id as the only cross-version target",
        "- diagnostic or lexical stress samples in the main denominator",
        "- Qwen, generation eval, RAGAS, OCR, parser changes, chunk changes, retrieval parameter tuning",
        "",
        f"Retrieval top_k for reported hits: {top_k}. Candidate limit follows current default search_limit: {candidate_limit}.",
        "Available variants are compared only as current performance on the new clean eval set.",
        f"Index rebuilt: {yes_no(inventory['index_rebuilt'])}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_by_type_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Main results by query type",
        "",
        "| variant | query_type | n | doc@10 | stable@10 | stable@20 | doc@20 | mapping_found | table_related_occ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index_variant']} | {row['query_type']} | {row.get('count', 0)} | "
            f"{fmt_pct(row.get('doc_hit_at_10', 0.0))} | {fmt_pct(row.get('stable_block_hit_at_10', 0.0))} | "
            f"{fmt_pct(row.get('stable_block_hit_at_20', 0.0))} | {fmt_pct(row.get('doc_hit_at_20', 0.0))} | "
            f"{fmt_pct(row.get('stable_target_mapping_found_rate', 0.0))} | {fmt_pct(row.get('mean_table_related_top10_occupancy', 0.0))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_examples_md(path: Path, rows: list[dict[str, Any]], want_near: bool) -> None:
    title = "Near miss examples" if want_near else "Miss examples"
    selected = []
    for row in rows:
        near = bool(not row["stable_block_hit_at_10"] and row["stable_block_hit_at_20"])
        if want_near and near:
            selected.append(row)
        elif not want_near and not row["stable_block_hit_at_10"] and not near:
            selected.append(row)
    lines = [f"# {title}", ""]
    for qtype in QUERY_TYPES:
        qrows = [r for r in selected if r["query_type"] == qtype][:8]
        lines.extend([f"## {qtype}", ""])
        lines.extend(example_bullets(qrows) or ["- none"])
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_taxonomy_summary(path: Path, ledger: list[dict[str, Any]], primary_rows: list[dict[str, Any]]) -> None:
    primary_miss = [r for r in primary_rows if not r["stable_block_hit_at_10"]]
    counts = Counter(classify_failure(r) for r in primary_miss)
    lines = ["# Failure taxonomy summary", "", "Primary variant failure counts:"]
    for category in FAILURE_CATEGORIES:
        lines.append(f"- {category}: {counts.get(category, 0)}")
    lines.append("")
    for qtype in QUERY_TYPES:
        qrows = [r for r in primary_rows if r["query_type"] == qtype]
        failures = [r for r in qrows if not r["stable_block_hit_at_10"]]
        successes = [r for r in qrows if r["stable_block_hit_at_10"]]
        near = [r for r in failures if r["stable_block_hit_at_20"]]
        qcounts = Counter(classify_failure(r) for r in failures)
        lines.extend(
            [
                f"## {qtype}",
                "",
                f"- top failures: {', '.join(f'{k}={v}' for k, v in qcounts.most_common(5)) or 'none'}",
                f"- representative successes: {', '.join(r['sample_id'] for r in successes[:5]) or 'none'}",
                f"- near misses: {', '.join(r['sample_id'] for r in near[:5]) or 'none'}",
                f"- concentrated issue: {qcounts.most_common(1)[0][0] if qcounts else 'none'}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_variant_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Variant comparison",
        "",
        "This is a current-performance comparison on the new clean main eval set. It is not Phase 5C/5D effect validation.",
        "",
        f"- baseline variant: `{comparison['baseline_variant']}`",
        f"- enhanced variant: `{comparison['enhanced_variant']}`",
        "",
        "| slice | baseline doc@10 | enhanced doc@10 | baseline stable@10 | enhanced stable@10 | baseline_only | enhanced_only | both_success | both_fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = comparison["overall"]
    lines.append(comparison_table_row("overall", overall))
    for qtype, payload in comparison["by_query_type"].items():
        lines.append(comparison_table_row(qtype, payload))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_table_row(label: str, payload: dict[str, Any]) -> str:
    b = payload["baseline"]
    e = payload["enhanced"]
    p = payload["paired"]
    return (
        f"| {label} | {fmt_pct(b.get('doc_hit_at_10', 0.0))} | {fmt_pct(e.get('doc_hit_at_10', 0.0))} | "
        f"{fmt_pct(b.get('stable_block_hit_at_10', 0.0))} | {fmt_pct(e.get('stable_block_hit_at_10', 0.0))} | "
        f"{p.get('baseline_only', 0)} | {p.get('enhanced_only', 0)} | {p.get('both_success', 0)} | {p.get('both_fail', 0)} |"
    )


def write_rank_delta_examples(path: Path, results: list[dict[str, Any]], baseline: str, enhanced: str) -> None:
    b_rows = {r["sample_id"]: r for r in results if r["index_variant"] == baseline}
    e_rows = {r["sample_id"]: r for r in results if r["index_variant"] == enhanced}
    deltas = []
    for sid, b in b_rows.items():
        e = e_rows.get(sid)
        if not e:
            continue
        b_rank = rank_for_delta(b["first_stable_block_hit_rank"])
        e_rank = rank_for_delta(e["first_stable_block_hit_rank"])
        deltas.append((b_rank - e_rank, b, e))
    deltas.sort(key=lambda item: item[0], reverse=True)
    lines = ["# Rank delta examples", "", "Positive delta means the enhanced variant ranked the stable target earlier.", ""]
    lines.append("## Improved")
    lines.extend(delta_bullets(deltas[:10]))
    lines.extend(["", "## Regressed"])
    lines.extend(delta_bullets(list(reversed(deltas[-10:]))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delta_bullets(items: list[tuple[int, dict[str, Any], dict[str, Any]]]) -> list[str]:
    if not items:
        return ["- none"]
    return [
        f"- {b['sample_id']} ({b['query_type']}): delta={delta}, baseline_rank={b['first_stable_block_hit_rank']}, enhanced_rank={e['first_stable_block_hit_rank']}"
        for delta, b, e in items
    ]


def chunk_to_retrieved(item: dict[str, Any]) -> RetrievedChunk:
    metadata = {
        "chunk_index": item.get("chunk_index"),
        "retrieval_text": item.get("retrieval_text", ""),
        "content_kind": item.get("content_kind", "body"),
        "quality_score": item.get("quality_score", 0.0),
        "contains_table_text": item.get("contains_table_text", False),
        "contains_table_caption": item.get("contains_table_caption", False),
        "contains_figure_caption": item.get("contains_figure_caption", False),
        "contains_image": item.get("contains_image", False),
        "object_type": item.get("object_type", "body"),
        "object_id": item.get("object_id", ""),
        "block_types": item.get("block_types", []),
        "evidence_types": item.get("evidence_types", []),
        "source_block_ids": item.get("source_block_ids") or item.get("block_ids") or [],
    }
    return RetrievedChunk(
        chunk_id=str(item.get("chunk_id", "")),
        doc_id=str(item.get("doc_id", "")),
        source_file=str(item.get("source_file", "")),
        title=str(item.get("title", "")),
        section=str(item.get("section", "")),
        text=str(item.get("text", "")),
        page_start=normalize_page(item.get("page_start")),
        page_end=normalize_page(item.get("page_end")),
        metadata=metadata,
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


def retrieval_text(chunk: RetrievedChunk) -> str:
    rt = str(chunk.metadata.get("retrieval_text") or "")
    if rt:
        return rt
    parts = []
    if chunk.title:
        parts.append(f"title {chunk.title}")
    if chunk.section:
        parts.append(f"section {chunk.section}")
    if chunk.source_file:
        parts.append(f"source_file {chunk.source_file}")
    if chunk.doc_id:
        parts.append(f"doc_id {chunk.doc_id}")
    parts.append(chunk.text or "")
    return "\n".join(parts)


def chunk_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    return [str(item) for item in (chunk.get("source_block_ids") or chunk.get("block_ids") or []) if item]


def chunk_has_table_related(chunk: dict[str, Any]) -> bool:
    for meta in chunk.get("source_block_metadata") or []:
        if isinstance(meta, dict) and meta.get("table_related") is True:
            return True
    return False


def stable_ids(sample: dict[str, Any]) -> list[str]:
    value = sample.get("stable_target_block_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or "")


def target_triplet(sample: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_space(str(sample.get("query", ""))),
        str(sample.get("target_doc_id", "")),
        ";".join(stable_ids(sample)),
    )


def is_row_cell_query(query: str) -> bool:
    patterns = (
        r"\bwhich\s+(table\s+)?(row|cell|column)\b",
        r"\b(table\s+)?(row|cell|column)\s+(number|value|index|header)\b",
        r"\b(row|cell|column)\s+in\s+(the\s+)?table\b",
        r"\btable\s+(row|cell|column)\b",
        r"\brow\s*/\s*column\b",
    )
    return any(re.search(pattern, query, re.I) for pattern in patterns)


def is_ocr_image_query(query: str) -> bool:
    return bool(re.search(r"\b(ocr|image|pixel|photograph|microscopy|visual|shown in the image)\b", query, re.I))


def first_rank(flags: list[bool]) -> int:
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return idx
    return 0


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0


def mean_rank(rows: list[dict[str, Any]], key: str) -> float | None:
    ranks = [int(row.get(key) or 0) for row in rows if int(row.get(key) or 0) > 0]
    return statistics.mean(ranks) if ranks else None


def mean_reciprocal(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean((1 / int(row.get(key) or 0)) if int(row.get(key) or 0) > 0 else 0.0 for row in rows) if rows else 0.0


def rank_for_delta(value: Any) -> int:
    rank = int(value or 0)
    return rank if rank > 0 else TOP_K + 1


def describe_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "min": values_sorted[0],
        "p25": percentile(values_sorted, 0.25),
        "median": statistics.median(values_sorted),
        "p75": percentile(values_sorted, 0.75),
        "max": values_sorted[-1],
        "mean": statistics.mean(values_sorted),
    }


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    idx = (len(values) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(values[int(idx)])
    return values[lower] * (upper - idx) + values[upper] * (idx - lower)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_page(value: Any) -> int | None:
    if value in (None, "", -1):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def safe_parse_json(raw: Any) -> dict[str, Any]:
    if not raw or not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"metadata_json_parse_error": True}
    return parsed if isinstance(parsed, dict) else {}


def truncate(text: str, limit: int) -> str:
    text = normalize_space(str(text or ""))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def success_pattern(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no stable-block top10 successes in primary variant"
    return "stable target reached in top10 for examples " + ", ".join(row["sample_id"] for row in rows[:5])


def failure_pattern(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no stable-block top10 misses in primary variant"
    counts = Counter(classify_failure(row) for row in rows)
    return ", ".join(f"{k}={v}" for k, v in counts.most_common())


def example_bullets(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return [
        f"- {row['sample_id']} ({row['query_type']}): doc_rank={row['first_doc_hit_rank']}, stable_rank={row['first_stable_block_hit_rank']}, category={classify_failure(row) if not row['stable_block_hit_at_10'] else 'success'}"
        for row in rows
    ]


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
