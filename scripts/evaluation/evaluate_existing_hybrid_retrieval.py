#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.rerank_service import QwenReranker
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryFilters, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


FIGURE_HINTS = ("figure", "fig.", "fig ", "图")
TABLE_HINTS = (
    "table",
    "primer",
    "sequence",
    "strain",
    "vmax",
    "km",
    "glycan",
    "relative peak area",
    "parameter",
)
DIAGNOSTIC_REFERENCE = {"dense_top5": "4/9", "hybrid_top5": "8/9"}


@dataclass
class QuerySpec:
    id: str
    query: str
    expected_doc_id: str | None
    expected_keywords: list[str]
    expected_block_types: list[str]
    category: str
    exclude_from_primary_metrics: bool = False
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the existing formal retrieval pipeline on small-sample queries.")
    parser.add_argument("--query_spec", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--parsed_clean_dir", default="data/small_exp/parsed_clean_table_min_verify")
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--milvus_uri", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reranker_model_path", required=True)
    parser.add_argument("--candidate_k", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rerank_candidate_k", type=int, default=10)
    parser.add_argument("--embedding_max_length", type=int, default=2048)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_query_specs(path: str) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            specs.append(
                QuerySpec(
                    id=item["id"],
                    query=item["query"],
                    expected_doc_id=item.get("expected_doc_id"),
                    expected_keywords=item.get("expected_keywords", []),
                    expected_block_types=item.get("expected_block_types", []),
                    category=item.get("category", "body"),
                    exclude_from_primary_metrics=bool(item.get("exclude_from_primary_metrics", False)),
                    notes=str(item.get("notes", "")),
                )
            )
    return specs


def load_chunks(path: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            chunks.append(item)
            by_id[item["chunk_id"]] = item
    return chunks, by_id


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("′", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("_", " ")
    text = text.replace("fig.", "figure ")
    text = re.sub(r"(?<=\d)\s*-\s*(?=[a-z])", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_variants(keyword: str) -> set[str]:
    base = normalize_text(keyword)
    variants = {base}
    compact = re.sub(r"[\s\\-_/]+", "", base)
    if compact:
        variants.add(compact)
    stripped = base.replace("'", "")
    if stripped:
        variants.add(stripped)
    if "2'-fl" in base or "2-fl" in base:
        variants.update({"2'-fl", "2′-fl", "2-fl", "2 fl", "2fl"})
    if "3'-fl" in base or "3-fl" in base:
        variants.update({"3'-fl", "3′-fl", "3-fl", "3 fl", "3fl"})
    if "km abts" in base or "km_abts" in keyword.lower():
        variants.update({"km abts", "km_abts", "km", "abts", "kmabts"})
    if "vmax" in base:
        variants.update({"vmax", "v max", "v_max"})
    if "ppfwk3hrp" in compact:
        variants.update({"ppfwk3hrp", "ppfwk3 hrp"})
    if "ppmutshrp" in compact:
        variants.update({"ppmutshrp", "ppmuts hrp"})
    if "figure 4" in base or "fig 4" in base:
        variants.update({"figure 4", "fig 4", "fig. 4"})
    if "figure 1" in base or "fig 1" in base:
        variants.update({"figure 1", "fig 1", "fig. 1"})
    return {item for item in variants if item}


def text_contains_keyword(text: str, keyword: str) -> bool:
    norm = normalize_text(text)
    compact = re.sub(r"[\s\\-_/]+", "", norm)
    for variant in keyword_variants(keyword):
        if variant in norm or variant in compact:
            return True
    return False


def compute_keyword_hits(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    matched = [kw for kw in keywords if text_contains_keyword(text, kw)]
    return len(matched), matched


def expected_block_type_hit(block_types: list[str], expected_block_types: list[str]) -> bool:
    if not expected_block_types:
        return False
    return any(item in block_types for item in expected_block_types)


def evaluate_chunk_against_expectation(chunk: dict[str, Any], spec: QuerySpec) -> dict[str, Any]:
    text = chunk.get("text", "")
    keyword_hits, matched_keywords = compute_keyword_hits(text, spec.expected_keywords)
    expected_doc_hit = spec.expected_doc_id is None or chunk.get("doc_id") == spec.expected_doc_id
    expected_block_hit = expected_block_type_hit(chunk.get("block_types", []), spec.expected_block_types)
    required_keyword_hits = math.ceil(len(spec.expected_keywords) * 0.75) if spec.expected_keywords else 0
    full_keyword_hit = keyword_hits >= required_keyword_hits if spec.expected_keywords else expected_doc_hit
    target_match = expected_doc_hit and full_keyword_hit and (expected_block_hit if spec.expected_block_types else True)
    return {
        "expected_doc_hit": expected_doc_hit,
        "expected_block_hit": expected_block_hit,
        "keyword_hits": keyword_hits,
        "matched_keywords": matched_keywords,
        "keywords_total": len(spec.expected_keywords),
        "full_keyword_hit": full_keyword_hit,
        "target_match": target_match,
    }


def find_target_chunks(spec: QuerySpec, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for chunk in chunks:
        evaluation = evaluate_chunk_against_expectation(chunk, spec)
        if evaluation["keyword_hits"] <= 0:
            continue
        matches.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "source_file": chunk.get("source_file"),
                "section": chunk.get("section"),
                "section_path": chunk.get("section_path", []),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "block_types": chunk.get("block_types", []),
                "text_preview": truncate_text(chunk.get("text", ""), 240),
                **evaluation,
            }
        )
    matches.sort(
        key=lambda item: (
            int(item["target_match"]),
            int(item["expected_doc_hit"]),
            item["keyword_hits"],
            int(item["expected_block_hit"]),
        ),
        reverse=True,
    )
    return {"target_exists_in_chunks": any(item["target_match"] for item in matches), "top_matches": matches[:5]}


def find_target_in_parsed_clean(spec: QuerySpec, parsed_clean_dir: str) -> dict[str, Any]:
    if not parsed_clean_dir or not spec.expected_doc_id:
        return {"target_exists_in_parsed_clean": None, "matches": []}
    path = Path(parsed_clean_dir) / f"{spec.expected_doc_id}.json"
    if not path.exists():
        return {"target_exists_in_parsed_clean": None, "matches": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    matches: list[dict[str, Any]] = []
    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            text = block.get("text", "")
            keyword_hits, matched = compute_keyword_hits(text, spec.expected_keywords)
            if keyword_hits <= 0:
                continue
            matches.append(
                {
                    "page": page.get("page"),
                    "type": block.get("type"),
                    "section_path": block.get("section_path", []),
                    "keyword_hits": keyword_hits,
                    "matched_keywords": matched,
                    "text_preview": truncate_text(text, 240),
                }
            )
    matches.sort(key=lambda item: item["keyword_hits"], reverse=True)
    return {"target_exists_in_parsed_clean": bool(matches), "matches": matches[:5]}


def query_mentions_expected_doc(spec: QuerySpec) -> bool:
    if not spec.expected_doc_id:
        return False
    return spec.expected_doc_id.lower() in spec.query.lower()


def build_runtime(settings: Settings) -> dict[str, Any]:
    embedder = BGEM3Embedder(
        model_path=settings.kb.embedding_model_path,
        dim=settings.kb.embedding_dim,
        max_length=settings.kb.embedding_max_length,
    )
    router = QueryRouter(settings.retrieval)
    dense = MilvusRetriever(settings.retrieval, embedder)
    bm25 = BM25Retriever(
        retrieval_config=settings.retrieval,
        kb_config=settings.kb,
        milvus_client=dense.client,
    )
    hybrid = HybridRetriever(
        config=settings.retrieval,
        dense_retriever=dense,
        bm25_retriever=bm25,
    )
    reranker = QwenReranker(
        api_base=settings.reranker.api_base,
        api_key=settings.reranker.api_key,
        model_name=settings.reranker.model_name,
        model_path=settings.reranker.model_path,
        service_url=settings.reranker.service_url,
        batch_size=settings.reranker.batch_size,
        use_fp16=settings.reranker.use_fp16,
        retrieval_config=settings.retrieval,
    )
    return {"embedder": embedder, "router": router, "dense": dense, "bm25": bm25, "hybrid": hybrid, "reranker": reranker}


def augment_chunk(chunk: RetrievedChunk, chunk_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sidecar = chunk_map.get(chunk.chunk_id, {})
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_file": chunk.source_file or sidecar.get("source_file"),
        "title": chunk.title or sidecar.get("title"),
        "section": chunk.section or sidecar.get("section"),
        "section_path": sidecar.get("section_path", []),
        "page_start": chunk.page_start if chunk.page_start is not None else sidecar.get("page_start"),
        "page_end": chunk.page_end if chunk.page_end is not None else sidecar.get("page_end"),
        "block_types": sidecar.get("block_types", []),
        "text": sidecar.get("text", chunk.text),
        "vector_score": float(chunk.vector_score or 0.0),
        "bm25_score": float(chunk.bm25_score or 0.0),
        "fusion_score": float(chunk.fusion_score or 0.0),
        "rerank_score": float(chunk.rerank_score or 0.0),
        "guarded_score": float(chunk.metadata.get("guarded_score", 0.0) or 0.0),
        "guarded_keyword_completeness": float(chunk.metadata.get("guarded_keyword_completeness", 0.0) or 0.0),
        "guarded_marker_score": float(chunk.metadata.get("guarded_marker_score", 0.0) or 0.0),
        "guarded_reference_bonus": float(chunk.metadata.get("guarded_reference_bonus", 0.0) or 0.0),
        "guarded_doc_score": float(chunk.metadata.get("guarded_doc_score", 0.0) or 0.0),
        "guarded_penalty": float(chunk.metadata.get("guarded_penalty", 0.0) or 0.0),
        "guarded_matched_anchors": list(chunk.metadata.get("guarded_matched_anchors", []) or []),
        "guarded_completeness_score": float(chunk.metadata.get("guarded_completeness_score", 0.0) or 0.0),
        "guarded_rank1_guard_triggered": bool(chunk.metadata.get("guarded_rank1_guard_triggered", False)),
        "guarded_rank1_guard_reason": str(chunk.metadata.get("guarded_rank1_guard_reason", "")),
        "guarded_rank1_promoted_from": chunk.metadata.get("guarded_rank1_promoted_from"),
    }


def annotate_results(
    results: list[RetrievedChunk],
    spec: QuerySpec,
    chunk_map: dict[str, dict[str, Any]],
    dense_rank_map: dict[str, int],
    bm25_rank_map: dict[str, int],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for rank, chunk in enumerate(results, start=1):
        row = augment_chunk(chunk, chunk_map)
        evaluation = evaluate_chunk_against_expectation(row, spec)
        row.update(evaluation)
        row["final_rank"] = rank
        row["dense_rank"] = dense_rank_map.get(chunk.chunk_id)
        row["sparse_rank"] = bm25_rank_map.get(chunk.chunk_id)
        annotated.append(row)
    return annotated


def locate_best_rank(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row.get("target_match"):
            return int(row["final_rank"])
    return None


def grade_rank(rank: int | None, top_k: int) -> str:
    if rank is None:
        return "fail"
    if rank <= top_k:
        return "pass"
    return "partial"


def determine_failure_reason(
    spec: QuerySpec,
    target_exists_in_chunks: bool,
    parsed_clean_exists: bool | None,
    dense_rank: int | None,
    hybrid_rank: int | None,
    plain_rerank_rank: int | None,
    guarded_rank: int | None,
    doc_rank: int | None,
    top_k: int,
) -> tuple[str, str]:
    if spec.exclude_from_primary_metrics and not target_exists_in_chunks:
        return "table2_extraction_gap", "Keep excluded from retrieval metrics and fix Table 2 extraction separately."
    if not target_exists_in_chunks:
        if parsed_clean_exists:
            return "extraction_gap_or_invalid_spec", "Target-like evidence exists in parsed_clean but not in chunks; inspect chunking or spec strictness."
        return "extraction_missing", "Fix extraction or query spec before tuning retrieval."
    best_rank = min([rank for rank in (hybrid_rank, guarded_rank, doc_rank) if rank is not None], default=None)
    if dense_rank is not None and dense_rank <= top_k and best_rank is not None and best_rank <= top_k:
        return "ok", "No retrieval change required for this query."
    if hybrid_rank is not None and hybrid_rank <= top_k and plain_rerank_rank is not None and plain_rerank_rank > top_k:
        return "plain_reranker_overrides_complete_evidence", "Keep reranker guarded for table/figure evidence instead of letting plain rerank fully override hybrid order."
    if guarded_rank is not None and guarded_rank <= top_k:
        return "guarded_reranker_recovers_hybrid", "Use guarded reranking; plain reranker was over-promoting incomplete evidence."
    if best_rank is not None and best_rank <= top_k:
        return "needs_bm25_hybrid", "Dense-only is weak; hybrid sparse recall or guarded reranking is carrying the result."
    if doc_rank is not None and doc_rank <= top_k:
        return "cross_doc_competition", "Cross-document competition is suppressing the right chunk; consider doc routing when doc is explicit."
    if dense_rank is not None:
        return "dense_ranking_weak", "Increase candidate pool or improve fusion/rerank logic."
    return "unresolved", "Inspect query intent, sparse recall, and evidence matching."


def truncate_text(text: str, limit: int = 500) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def to_rank_map(results: list[RetrievedChunk]) -> dict[str, int]:
    return {chunk.chunk_id: idx for idx, chunk in enumerate(results, start=1)}


def analyze_table2(parsed_clean_dir: str, chunk_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parsed_path = Path(parsed_clean_dir) / "doc_0045.json"
    if not parsed_path.exists():
        return {"available": False}
    doc = json.loads(parsed_path.read_text(encoding="utf-8"))
    table2_blocks: list[dict[str, Any]] = []
    context_blocks: list[dict[str, Any]] = []
    for page in doc.get("pages", []):
        blocks = page.get("blocks", [])
        for idx, block in enumerate(blocks):
            text = block.get("text", "")
            if "table 2" not in normalize_text(text):
                continue
            record = {
                "page": page.get("page"),
                "block_index": idx,
                "type": block.get("type"),
                "section_path": block.get("section_path", []),
                "text_preview": truncate_text(text, 220),
            }
            table2_blocks.append(record)
            for follow in blocks[idx : min(len(blocks), idx + 6)]:
                context_blocks.append(
                    {
                        "page": page.get("page"),
                        "type": follow.get("type"),
                        "section_path": follow.get("section_path", []),
                        "text_preview": truncate_text(follow.get("text", ""), 220),
                    }
                )
    suspicious_chunks = []
    for chunk in chunk_map.values():
        text = normalize_text(chunk.get("text", ""))
        if "ppmuts" in text and "ppfwk3" in text and "figure 4" in text:
            suspicious_chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "section": chunk.get("section"),
                    "block_types": chunk.get("block_types", []),
                    "text_preview": truncate_text(chunk.get("text", ""), 260),
                }
            )
    return {
        "available": True,
        "table2_blocks": table2_blocks[:5],
        "context_blocks": context_blocks[:8],
        "suspicious_chunks": suspicious_chunks[:3],
    }


def mode_run(
    mode: str,
    spec: QuerySpec,
    runtime: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
    candidate_k: int,
    top_k: int,
    doc_routed: bool = False,
) -> dict[str, Any]:
    router: QueryRouter = runtime["router"]
    dense: MilvusRetriever = runtime["dense"]
    bm25: BM25Retriever = runtime["bm25"]
    hybrid: HybridRetriever = runtime["hybrid"]
    reranker: QwenReranker = runtime["reranker"]

    analysis: QueryAnalysis = router.analyze(spec.query)
    filters = QueryFilters(doc_ids=[spec.expected_doc_id]) if doc_routed and spec.expected_doc_id else None

    dense_hits = dense.search(spec.query, limit=analysis.search_limit, filters=filters)
    bm25_hits = bm25.search(spec.query, limit=analysis.search_limit, filters=filters)
    dense_rank_map = to_rank_map(dense_hits)
    bm25_rank_map = to_rank_map(bm25_hits)

    if mode == "formal_dense":
        candidates = dense_hits
    elif mode in {
        "formal_hybrid_no_reranker",
        "formal_hybrid_plain_reranker",
        "formal_hybrid_guarded_reranker",
        "formal_hybrid_guarded_rank1_protected",
        "formal_hybrid_doc_routed",
    }:
        candidates = hybrid.search(spec.query, limit=analysis.search_limit, filters=filters, analysis=analysis)
    else:
        raise ValueError(f"unknown mode {mode}")

    if mode == "formal_hybrid_plain_reranker":
        final_hits = reranker.rerank(
            spec.query,
            candidates,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
            mode="plain",
        )
    elif mode == "formal_hybrid_guarded_reranker":
        final_hits = reranker.rerank(
            spec.query,
            candidates,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
            mode="guarded",
        )
    elif mode == "formal_hybrid_guarded_rank1_protected":
        final_hits = reranker.rerank(
            spec.query,
            candidates,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
            mode="guarded_rank1",
        )
    elif mode == "formal_hybrid_doc_routed":
        final_hits = reranker.rerank(
            spec.query,
            candidates,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
            mode="guarded_rank1",
        )
    else:
        final_hits = candidates

    annotated = annotate_results(final_hits[:top_k], spec, chunk_map, dense_rank_map, bm25_rank_map)
    return {
        "mode": mode,
        "analysis": analysis,
        "filters": filters,
        "dense_rank_map": dense_rank_map,
        "bm25_rank_map": bm25_rank_map,
        "hybrid_debug": getattr(hybrid, "last_debug", {}),
        "rerank_debug": getattr(reranker, "last_debug", {}),
        "results": annotated,
        "target_rank": locate_best_rank(annotated),
        "grade": grade_rank(locate_best_rank(annotated), top_k=top_k),
    }


def build_report(
    args: argparse.Namespace,
    settings: Settings,
    default_config: dict[str, Any],
    query_results: list[dict[str, Any]],
    table2_info: dict[str, Any],
) -> str:
    branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_text(["git", "rev-parse", "HEAD"])
    formal_dense_summary = Counter(item["modes"]["formal_dense"]["grade"] for item in query_results if not item["exclude_from_primary_metrics"])
    formal_hybrid_summary = Counter(item["modes"]["formal_hybrid_no_reranker"]["grade"] for item in query_results if not item["exclude_from_primary_metrics"])
    formal_hybrid_plain_summary = Counter(item["modes"]["formal_hybrid_plain_reranker"]["grade"] for item in query_results if not item["exclude_from_primary_metrics"])
    formal_hybrid_guarded_summary = Counter(item["modes"]["formal_hybrid_guarded_reranker"]["grade"] for item in query_results if not item["exclude_from_primary_metrics"])
    formal_hybrid_rank1_guard_summary = Counter(item["modes"]["formal_hybrid_guarded_rank1_protected"]["grade"] for item in query_results if not item["exclude_from_primary_metrics"])
    formal_hybrid_doc_summary = Counter(item["modes"]["formal_hybrid_doc_routed"]["grade"] for item in query_results if not item["exclude_from_primary_metrics"])

    primary_results = [
        item
        for item in query_results
        if not item["exclude_from_primary_metrics"]
        and item["likely_failure_reason"] not in {"table2_extraction_gap", "extraction_gap_or_invalid_spec", "extraction_missing"}
    ]
    formal_dense_pass = sum(1 for item in primary_results if item["modes"]["formal_dense"]["target_rank"] and item["modes"]["formal_dense"]["target_rank"] <= args.top_k)
    formal_hybrid_pass = sum(1 for item in primary_results if item["modes"]["formal_hybrid_no_reranker"]["target_rank"] and item["modes"]["formal_hybrid_no_reranker"]["target_rank"] <= args.top_k)
    formal_hybrid_plain_pass = sum(1 for item in primary_results if item["modes"]["formal_hybrid_plain_reranker"]["target_rank"] and item["modes"]["formal_hybrid_plain_reranker"]["target_rank"] <= args.top_k)
    formal_hybrid_guarded_pass = sum(1 for item in primary_results if item["modes"]["formal_hybrid_guarded_reranker"]["target_rank"] and item["modes"]["formal_hybrid_guarded_reranker"]["target_rank"] <= args.top_k)
    formal_hybrid_rank1_guard_pass = sum(1 for item in primary_results if item["modes"]["formal_hybrid_guarded_rank1_protected"]["target_rank"] and item["modes"]["formal_hybrid_guarded_rank1_protected"]["target_rank"] <= args.top_k)
    formal_hybrid_doc_pass = sum(1 for item in primary_results if item["modes"]["formal_hybrid_doc_routed"]["target_rank"] and item["modes"]["formal_hybrid_doc_routed"]["target_rank"] <= args.top_k)
    denominator = len(primary_results)
    plain_fail_cases = [
        item["spec"].id
        for item in query_results
        if not item["exclude_from_primary_metrics"]
        and item["modes"]["formal_hybrid_no_reranker"]["target_rank"]
        and item["modes"]["formal_hybrid_no_reranker"]["target_rank"] <= args.top_k
        and not (
            item["modes"]["formal_hybrid_plain_reranker"]["target_rank"]
            and item["modes"]["formal_hybrid_plain_reranker"]["target_rank"] <= args.top_k
        )
    ]
    guarded_incomplete_rank1_before = []
    for item in query_results:
        top_row = item["modes"]["formal_hybrid_guarded_reranker"]["results"][:1]
        if not top_row:
            continue
        row = top_row[0]
        if not row["expected_block_hit"] or not row["full_keyword_hit"]:
            guarded_incomplete_rank1_before.append(
                {
                    "query_id": item["spec"].id,
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "block_types": row["block_types"],
                    "matched_keywords": row["matched_keywords"],
                    "expected_block_hit": row["expected_block_hit"],
                    "full_keyword_hit": row["full_keyword_hit"],
                }
            )
    guarded_incomplete_rank1_after = []
    rank1_guard_triggered = []
    for item in query_results:
        top_row = item["modes"]["formal_hybrid_guarded_rank1_protected"]["results"][:1]
        if top_row:
            row = top_row[0]
            if row.get("guarded_rank1_guard_triggered"):
                rank1_guard_triggered.append(
                    {
                        "query_id": item["spec"].id,
                        "chunk_id": row["chunk_id"],
                        "promoted_from": row.get("guarded_rank1_promoted_from"),
                        "reason": row.get("guarded_rank1_guard_reason"),
                    }
                )
            if not row["expected_block_hit"] or not row["full_keyword_hit"]:
                guarded_incomplete_rank1_after.append(
                    {
                        "query_id": item["spec"].id,
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "block_types": row["block_types"],
                        "matched_keywords": row["matched_keywords"],
                        "expected_block_hit": row["expected_block_hit"],
                        "full_keyword_hit": row["full_keyword_hit"],
                    }
                )

    lines: list[str] = []
    lines.extend(
        [
            "# Rank1 Evidence Guard Eval",
            "",
            "## Run Metadata",
            f"- Branch: `{branch}`",
            f"- Commit: `{commit}`",
            "- Modified files: `src/synbio_rag/application/rerank_service.py`, `src/synbio_rag/domain/config.py`, `scripts/evaluation/evaluate_existing_hybrid_retrieval.py`, `scripts/evaluation/evaluate_guarded_reranker.py`, `tests/test_guarded_reranker.py`.",
            f"- Formal hybrid code entry: `src/synbio_rag/application/pipeline.py` -> `SynBioRAGPipeline.retriever`; retrieval implementation in `src/synbio_rag/infrastructure/vectorstores/hybrid.py`.",
            f"- Formal dense code entry: `src/synbio_rag/infrastructure/vectorstores/milvus.py` -> `MilvusRetriever.search`.",
            f"- Reranker entry: `src/synbio_rag/application/rerank_service.py` -> `QwenReranker.rerank`.",
            f"- Rank1 guard entry: `src/synbio_rag/application/rerank_service.py` -> `_apply_rank1_evidence_guard` under mode `guarded_rank1`.",
            f"- Rerank mode config: `BIORAG_RERANK_MODE` / `RETRIEVAL_RERANK_MODE`, current eval default=`{settings.retrieval.rerank_mode}`.",
            f"- Added rank1 guard config: `RETRIEVAL_GUARDED_RANK1_MIN_COMPLETENESS_GAIN={settings.retrieval.guarded_rank1_min_completeness_gain}`, `RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP={settings.retrieval.guarded_rank1_max_score_gap}`.",
            f"- Collection: `{args.collection_name}`",
            f"- Milvus URI: `{Path(args.milvus_uri).resolve()}`",
            f"- Chunks JSONL: `{Path(args.chunks_jsonl).resolve()}`",
            f"- Query spec: `{Path(args.query_spec).resolve()}`",
            f"- Query count: `{len(query_results)}`",
            f"- Diagnostic reference baseline: dense top5 `{DIAGNOSTIC_REFERENCE['dense_top5']}`, hybrid top5 `{DIAGNOSTIC_REFERENCE['hybrid_top5']}`.",
            "",
            "## Formal Retrieval Audit",
            f"- Default retrieval config: `hybrid_enabled={default_config['hybrid_enabled']}`, `bm25_enabled={default_config['bm25_enabled']}`, `search_limit={default_config['search_limit']}`, `dense_limit={default_config['dense_limit']}`, `bm25_limit={default_config['bm25_limit']}`, `rerank_top_k={default_config['rerank_top_k']}`, `final_top_k={default_config['final_top_k']}`.",
            f"- Evaluation config: `hybrid_enabled={settings.retrieval.hybrid_enabled}`, `bm25_enabled={settings.retrieval.bm25_enabled}`, `search_limit={settings.retrieval.search_limit}`, `dense_limit={settings.retrieval.dense_limit}`, `bm25_limit={settings.retrieval.bm25_limit}`, `rerank_top_k={settings.retrieval.rerank_top_k}`, `final_top_k={settings.retrieval.final_top_k}`.",
            "- Formal hybrid is enabled by default in `RetrievalConfig` and used through `HybridRetriever.search` before reranking.",
            "- Hybrid can now be explicitly toggled with `RETRIEVAL_HYBRID_ENABLED` and `RETRIEVAL_BM25_ENABLED`; before this patch there was no env-level toggle wiring.",
            "- Sparse recall source: `BM25Retriever` reads `kb.chunk_jsonl` when it exists; this evaluation points it to the same small-sample `chunks.jsonl`, so BM25 text is aligned with the diagnostic data source.",
            "- Fusion: reciprocal-rank fusion in `hybrid.py`, weighted by `dense_rrf_weight`, `bm25_rrf_weight`, `rrf_k`, and optional CJK BM25 down-weighting.",
            "- Query routing: `QueryRouter` adjusts `search_limit` and `rerank_top_k` by intent, but there is no dedicated table/figure intent path.",
            "- Structured evidence boost: formal hybrid now applies a small marker-based boost from chunk text markers such as `[TABLE TEXT]`, `[TABLE CAPTION]`, and `[FIGURE CAPTION]`. This aligns the existing retriever with the diagnostic finding without changing Milvus schema.",
            "- Reranker placement: reranker runs after hybrid candidate generation, not before. Candidate count is capped by `rerank_top_k`.",
            "- Rerank modes: `off` keeps hybrid order, `plain` lets reranker directly reorder candidates, `guarded` blends reranker with hybrid score, keyword completeness, structure markers, and explicit doc-route hints.",
            "",
            "## Summary Metrics",
            f"- formal_dense summary: `{dict(formal_dense_summary)}`",
            f"- formal_hybrid_no_reranker summary: `{dict(formal_hybrid_summary)}`",
            f"- formal_hybrid_plain_reranker summary: `{dict(formal_hybrid_plain_summary)}`",
            f"- guarded_before_rank1_guard summary: `{dict(formal_hybrid_guarded_summary)}`",
            f"- guarded_after_rank1_guard summary: `{dict(formal_hybrid_rank1_guard_summary)}`",
            f"- formal_hybrid_doc_routed summary: `{dict(formal_hybrid_doc_summary)}`",
            f"- Primary metric denominator after excluding extraction gaps: `{denominator}`",
            f"- formal_dense top{args.top_k}: `{formal_dense_pass}/{denominator}`",
            f"- formal_hybrid_no_reranker top{args.top_k}: `{formal_hybrid_pass}/{denominator}`",
            f"- formal_hybrid_plain_reranker top{args.top_k}: `{formal_hybrid_plain_pass}/{denominator}`",
            f"- guarded_before_rank1_guard top{args.top_k}: `{formal_hybrid_guarded_pass}/{denominator}`",
            f"- guarded_after_rank1_guard top{args.top_k}: `{formal_hybrid_rank1_guard_pass}/{denominator}`",
            f"- formal_hybrid_doc_routed top{args.top_k}: `{formal_hybrid_doc_pass}/{denominator}`",
            "",
            "## Diagnostic Comparison",
            f"- Target reproduction check: formal_hybrid_no_reranker top{args.top_k} `{formal_hybrid_pass}/{denominator}` vs diagnostic hybrid top{args.top_k} `{DIAGNOSTIC_REFERENCE['hybrid_top5']}`.",
            "- Remaining gaps should now be judged by comparing `plain` against `guarded`, not by changing BM25 or dense retrieval.",
            "",
        ]
    )

    lines.extend(
        [
            "## Reranker Failure Cases",
            f"- Plain reranker regressions relative to no-reranker: `{plain_fail_cases}`",
            f"- Guarded before rank1 guard still has incomplete-or-wrong-type rank1 cases: `{bool(guarded_incomplete_rank1_before)}`",
            f"- Guarded after rank1 guard still has incomplete-or-wrong-type rank1 cases: `{bool(guarded_incomplete_rank1_after)}`",
        ]
    )
    for item in guarded_incomplete_rank1_before:
        lines.append(
            f"  - before query=`{item['query_id']}` rank1_chunk=`{item['chunk_id']}` doc_id=`{item['doc_id']}` "
            f"block_types=`{item['block_types']}` expected_block_hit=`{item['expected_block_hit']}` "
            f"expected_keywords_hit=`{item['full_keyword_hit']}` matched_keywords=`{item['matched_keywords']}`"
        )
    for item in guarded_incomplete_rank1_after:
        lines.append(
            f"  - after query=`{item['query_id']}` rank1_chunk=`{item['chunk_id']}` doc_id=`{item['doc_id']}` "
            f"block_types=`{item['block_types']}` expected_block_hit=`{item['expected_block_hit']}` "
            f"expected_keywords_hit=`{item['full_keyword_hit']}` matched_keywords=`{item['matched_keywords']}`"
        )
    lines.append(f"- Rank1 guard triggered queries: `{rank1_guard_triggered}`")

    lines.append("## Per-Query Results")
    for item in query_results:
        spec = item["spec"]
        lines.extend(
            [
                "",
                f"### {spec.id}",
                f"- Query: `{spec.query}`",
                f"- Category: `{spec.category}`",
                f"- expected_doc_id: `{spec.expected_doc_id}`",
                f"- expected_keywords: `{spec.expected_keywords}`",
                f"- expected_block_types: `{spec.expected_block_types}`",
                f"- Excluded from primary metrics: `{spec.exclude_from_primary_metrics}`",
                f"- target_exists_in_chunks: `{item['target_exists_in_chunks']}`",
                f"- target_exists_in_parsed_clean: `{item['target_exists_in_parsed_clean']}`",
                f"- best_mode: `{item['best_mode']}`",
                f"- likely_failure_reason: `{item['likely_failure_reason']}`",
                f"- recommended_next_action: `{item['recommended_next_action']}`",
            ]
        )
        if spec.notes:
            lines.append(f"- Notes: `{spec.notes}`")
        if item["chunk_targets"]:
            lines.append("- Target evidence in chunks:")
            for evidence in item["chunk_targets"][:3]:
                lines.append(
                    f"  - chunk_id=`{evidence['chunk_id']}` doc_id=`{evidence['doc_id']}` block_types=`{evidence['block_types']}` keyword_hits=`{evidence['keyword_hits']}/{evidence['keywords_total']}` target_match=`{evidence['target_match']}` text=`{evidence['text_preview']}`"
                )
        for mode_name in (
            "formal_dense",
            "formal_hybrid_no_reranker",
            "formal_hybrid_plain_reranker",
            "formal_hybrid_guarded_reranker",
            "formal_hybrid_guarded_rank1_protected",
            "formal_hybrid_doc_routed",
        ):
            mode = item["modes"][mode_name]
            lines.append(
                f"- {mode_name}: grade=`{mode['grade']}` target_rank=`{mode['target_rank']}`"
            )
            for row in mode["results"]:
                lines.append(
                    "  - "
                    f"rank=`{row['final_rank']}` dense_rank=`{row.get('dense_rank')}` sparse_rank=`{row.get('sparse_rank')}` "
                    f"vector=`{row['vector_score']:.4f}` bm25=`{row['bm25_score']:.4f}` fusion=`{row['fusion_score']:.4f}` rerank=`{row['rerank_score']:.4f}` guarded=`{row['guarded_score']:.4f}` "
                    f"chunk_id=`{row['chunk_id']}` doc_id=`{row['doc_id']}` section=`{row['section']}` section_path=`{row['section_path']}` "
                    f"pages=`{row['page_start']}-{row['page_end']}` block_types=`{row['block_types']}` "
                    f"expected_doc_hit=`{row['expected_doc_hit']}` expected_keywords_hit=`{row['full_keyword_hit']}` expected_block_types_hit=`{row['expected_block_hit']}` "
                    f"matched_keywords=`{row['matched_keywords']}` guarded_keyword=`{row['guarded_keyword_completeness']:.3f}` guarded_marker=`{row['guarded_marker_score']:.3f}` "
                    f"guarded_ref=`{row['guarded_reference_bonus']:.3f}` guarded_doc=`{row['guarded_doc_score']:.3f}` guarded_penalty=`{row['guarded_penalty']:.3f}` "
                    f"guarded_completeness=`{row['guarded_completeness_score']:.3f}` rank1_complete=`{row['expected_block_hit'] and row['full_keyword_hit']}` "
                    f"rank1_guard_triggered=`{row['guarded_rank1_guard_triggered']}` rank1_guard_reason=`{row['guarded_rank1_guard_reason']}` "
                    f"rank1_promoted_from=`{row['guarded_rank1_promoted_from']}` guarded_matched_anchors=`{row['guarded_matched_anchors']}` text=`{truncate_text(row['text'])}`"
                )

    lines.extend(
        [
            "",
            "## Table 2 Localization",
        ]
    )
    if not table2_info.get("available"):
        lines.append("- Parsed-clean Table 2 diagnostics unavailable.")
    else:
        lines.append(f"- Parsed-clean contains direct `Table 2` blocks: `{bool(table2_info['table2_blocks'])}`")
        for row in table2_info.get("table2_blocks", []):
            lines.append(
                f"  - page=`{row['page']}` block_index=`{row['block_index']}` type=`{row['type']}` section_path=`{row['section_path']}` text=`{row['text_preview']}`"
            )
        lines.append("- Following parsed-clean context:")
        for row in table2_info.get("context_blocks", []):
            lines.append(
                f"  - page=`{row['page']}` type=`{row['type']}` section_path=`{row['section_path']}` text=`{row['text_preview']}`"
            )
        lines.append("- Suspicious chunk-level evidence:")
        for row in table2_info.get("suspicious_chunks", []):
            lines.append(
                f"  - chunk_id=`{row['chunk_id']}` doc_id=`{row['doc_id']}` section=`{row['section']}` block_types=`{row['block_types']}` text=`{row['text_preview']}`"
            )
        lines.append(
            "- Conclusion: Table 2 is still an extraction gap. Parsed clean only retains paragraph-style mentions, and a structured parameter row is fused with `Figure 4` context instead of forming stable `table_caption/table_text` evidence."
        )

    lines.extend(
        [
            "",
            "## Conclusions",
            f"- q_doc0001_materials misclassification fixed: `{next(item for item in query_results if item['spec'].id == 'q_doc0001_materials')['likely_failure_reason'] != 'extraction_missing'}`",
            f"- q_doc0001_fig4 status: `{next(item for item in query_results if item['spec'].id == 'q_doc0001_fig4')['likely_failure_reason']}`",
            f"- plain reranker underperforms no-reranker: `{formal_hybrid_plain_pass < formal_hybrid_pass}`",
            f"- guarded reranker recovers no-reranker or better: `{formal_hybrid_guarded_pass >= formal_hybrid_pass}`",
            f"- rank1-protected guarded reranker recovers or exceeds guarded-before: `{formal_hybrid_rank1_guard_pass >= formal_hybrid_guarded_pass}`",
            f"- Guarded after rank1 guard still has rank1 incomplete-evidence cases: `{bool(guarded_incomplete_rank1_after)}`",
            f"- Table-query failure reasons: `{dict(Counter(item['likely_failure_reason'] for item in query_results if item['spec'].category == 'table'))}`",
            f"- Figure-query failure reasons: `{dict(Counter(item['likely_failure_reason'] for item in query_results if item['spec'].category == 'figure'))}`",
            f"- Body-query failure reasons: `{dict(Counter(item['likely_failure_reason'] for item in query_results if item['spec'].category == 'body'))}`",
            "- Recommendation on end-to-end generation eval: only proceed if guarded reranker preserves hybrid evidence quality; plain reranker alone is not a safe production default for table/figure-heavy queries.",
            "- Recommendation on Table 2: keep it as a dedicated parsing/chunking follow-up.",
            "- Recommendation on Milvus built-in BM25 A/B: defer it. Current gap is reranker behavior, not BM25 implementation parity.",
            "- Recommendation on false-heading cleanup: defer it unless new retrieval misses can be traced back to heading noise rather than retrieval configuration.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, cwd=ROOT).strip()


def main() -> None:
    args = parse_args()
    specs = load_query_specs(args.query_spec)
    chunks, chunk_map = load_chunks(args.chunks_jsonl)

    settings = Settings.from_env()
    default_config = {
        "hybrid_enabled": settings.retrieval.hybrid_enabled,
        "bm25_enabled": settings.retrieval.bm25_enabled,
        "search_limit": settings.retrieval.search_limit,
        "dense_limit": settings.retrieval.dense_limit,
        "bm25_limit": settings.retrieval.bm25_limit,
        "rerank_top_k": settings.retrieval.rerank_top_k,
        "final_top_k": settings.retrieval.final_top_k,
    }

    settings.retrieval.milvus_uri = str(Path(args.milvus_uri).resolve())
    settings.retrieval.collection_name = args.collection_name
    settings.kb.chunk_jsonl = str(Path(args.chunks_jsonl).resolve())
    settings.kb.chunk_dir = str(Path(args.chunks_jsonl).resolve().parent)
    settings.kb.embedding_model_path = str(Path(args.model_path).resolve())
    settings.kb.embedding_max_length = args.embedding_max_length
    settings.reranker.model_path = str(Path(args.reranker_model_path).resolve())
    settings.retrieval.search_limit = args.candidate_k
    settings.retrieval.dense_limit = args.candidate_k
    settings.retrieval.bm25_limit = args.candidate_k
    settings.retrieval.rerank_top_k = max(args.top_k, args.rerank_candidate_k)
    settings.retrieval.final_top_k = args.top_k
    settings.retrieval.hybrid_enabled = True
    settings.retrieval.bm25_enabled = True

    runtime = build_runtime(settings)
    query_results: list[dict[str, Any]] = []
    for spec in specs:
        existence = find_target_chunks(spec, chunks)
        parsed_clean = find_target_in_parsed_clean(spec, args.parsed_clean_dir)

        modes = {
            "formal_dense": mode_run("formal_dense", spec, runtime, chunk_map, args.candidate_k, args.top_k),
            "formal_hybrid_no_reranker": mode_run("formal_hybrid_no_reranker", spec, runtime, chunk_map, args.candidate_k, args.top_k),
            "formal_hybrid_plain_reranker": mode_run(
                "formal_hybrid_plain_reranker",
                spec,
                runtime,
                chunk_map,
                args.candidate_k,
                args.top_k,
            ),
            "formal_hybrid_guarded_reranker": mode_run(
                "formal_hybrid_guarded_reranker",
                spec,
                runtime,
                chunk_map,
                args.candidate_k,
                args.top_k,
            ),
            "formal_hybrid_guarded_rank1_protected": mode_run(
                "formal_hybrid_guarded_rank1_protected",
                spec,
                runtime,
                chunk_map,
                args.candidate_k,
                args.top_k,
            ),
            "formal_hybrid_doc_routed": mode_run(
                "formal_hybrid_doc_routed",
                spec,
                runtime,
                chunk_map,
                args.candidate_k,
                args.top_k,
                doc_routed=query_mentions_expected_doc(spec),
            ),
        }
        mode_priority = {
            "formal_hybrid_guarded_rank1_protected": 5,
            "formal_hybrid_guarded_reranker": 4,
            "formal_hybrid_doc_routed": 3,
            "formal_hybrid_no_reranker": 2,
            "formal_hybrid_plain_reranker": 1,
            "formal_dense": 0,
        }
        best_mode = max(
            modes.items(),
            key=lambda item: (
                item[1]["target_rank"] is not None and item[1]["target_rank"] <= args.top_k,
                -(item[1]["target_rank"] or 10**9),
                mode_priority.get(item[0], -1),
            ),
        )[0]
        likely_failure_reason, recommended_next_action = determine_failure_reason(
            spec=spec,
            target_exists_in_chunks=existence["target_exists_in_chunks"],
            parsed_clean_exists=parsed_clean["target_exists_in_parsed_clean"],
            dense_rank=modes["formal_dense"]["target_rank"],
            hybrid_rank=modes["formal_hybrid_no_reranker"]["target_rank"],
            plain_rerank_rank=modes["formal_hybrid_plain_reranker"]["target_rank"],
            guarded_rank=modes["formal_hybrid_guarded_rank1_protected"]["target_rank"],
            doc_rank=modes["formal_hybrid_doc_routed"]["target_rank"],
            top_k=args.top_k,
        )
        query_results.append(
            {
                "spec": spec,
                "exclude_from_primary_metrics": spec.exclude_from_primary_metrics,
                "target_exists_in_chunks": existence["target_exists_in_chunks"],
                "target_exists_in_parsed_clean": parsed_clean["target_exists_in_parsed_clean"],
                "chunk_targets": existence["top_matches"],
                "parsed_clean_targets": parsed_clean["matches"],
                "modes": modes,
                "best_mode": best_mode,
                "likely_failure_reason": likely_failure_reason,
                "recommended_next_action": recommended_next_action,
            }
        )

    table2_info = analyze_table2(args.parsed_clean_dir, chunk_map)
    report = build_report(args, settings, default_config, query_results, table2_info)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
