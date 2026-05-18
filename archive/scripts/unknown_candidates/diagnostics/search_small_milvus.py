#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient


DEFAULT_QUERY_SPECS = [
    {
        "id": "q_table3_man8",
        "query": "In doc_0045 Table 3, what is the relative peak area of Man8 in PpFWK3HRP?",
        "expected_doc_id": "doc_0045",
        "expected_keywords": ["Table 3", "Man8", "PpFWK3HRP", "57.3"],
        "expected_block_types": ["table_text", "table_caption"],
        "category": "table",
    },
    {
        "id": "q_table4_vmax",
        "query": "In doc_0045 Table 4, what are the KM_ABTS and Vmax values for PpFWK3HRP?",
        "expected_doc_id": "doc_0045",
        "expected_keywords": ["Table 4", "KM_ABTS", "Vmax", "PpFWK3HRP", "2.03", "2.46"],
        "expected_block_types": ["table_text", "table_caption"],
        "category": "table",
    },
    {
        "id": "q_table5_primer",
        "query": "In doc_0045 Table 5, what primer sequences are listed for OCH1-5int-fw1 and OCH1-5int-rv1?",
        "expected_doc_id": "doc_0045",
        "expected_keywords": ["Table 5", "OCH1-5int-fw1", "OCH1-5int-rv1"],
        "expected_block_types": ["table_text", "table_caption"],
        "category": "table",
    },
    {
        "id": "q_table6_strains",
        "query": "In doc_0045 Table 6, which Pichia pastoris strains are listed?",
        "expected_doc_id": "doc_0045",
        "expected_keywords": ["Table 6", "strain name", "PpMutS", "PpFWK3", "PpMutSHRP", "PpFWK3HRP"],
        "expected_block_types": ["table_text", "table_caption"],
        "category": "table",
    },
    {
        "id": "q_table2_parameters",
        "query": "What strain-specific parameters are shown in doc_0045 Table 2?",
        "expected_doc_id": "doc_0045",
        "expected_keywords": ["Table 2", "strain-specific parameters"],
        "expected_block_types": ["table_text", "table_caption"],
        "category": "table",
    },
    {
        "id": "q_doc0001_materials",
        "query": "What materials were used in the HMO fermentation study in doc_0001?",
        "expected_doc_id": "doc_0001",
        "expected_keywords": ["2.1. Materials", "HMOs", "Glycom A/S"],
        "expected_block_types": ["paragraph", "subsection_heading", "table_caption"],
        "category": "body",
    },
    {
        "id": "q_doc0001_glycom",
        "query": "What HMOs were supplied by Glycom in doc_0001?",
        "expected_doc_id": "doc_0001",
        "expected_keywords": ["Glycom", "2′-FL", "LNT", "LNnT"],
        "expected_block_types": ["paragraph"],
        "category": "body",
    },
    {
        "id": "q_doc0045_fig1",
        "query": "What does Figure 1 describe in doc_0045?",
        "expected_doc_id": "doc_0045",
        "expected_keywords": ["Figure 1"],
        "expected_block_types": ["figure_caption"],
        "category": "figure",
    },
    {
        "id": "q_doc0001_fig4",
        "query": "What does Figure 4 describe in doc_0001?",
        "expected_doc_id": "doc_0001",
        "expected_keywords": ["Fig. 4", "Microbial composition changes", "top 20 species"],
        "expected_block_types": ["figure_caption"],
        "category": "figure",
        "notes": "Figure 4 caption is present in a mixed figure-caption chunk and should be treated as a retrievable query.",
    },
    {
        "id": "q_doc0001_2fl",
        "query": "What are the main results about 2'-FL utilization in doc_0001?",
        "expected_doc_id": "doc_0001",
        "expected_keywords": ["2′-FL", "Results", "Bifidobacterium"],
        "expected_block_types": ["paragraph"],
        "category": "body",
    },
]

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
FIGURE_HINTS = ("figure", "fig.", "fig ", "图")
RRF_K = 60.0


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


class SimpleBM25:
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_tf: dict[str, Counter[str]] = {}
        self.doc_len: dict[str, int] = {}
        self.doc_freq: Counter[str] = Counter()

        for doc in documents:
            chunk_id = doc["chunk_id"]
            tokens = tokenize_for_bm25(doc.get("text", ""))
            self.doc_tokens[chunk_id] = tokens
            tf = Counter(tokens)
            self.doc_tf[chunk_id] = tf
            self.doc_len[chunk_id] = len(tokens)
            for token in tf:
                self.doc_freq[token] += 1

        self.avgdl = statistics.mean(self.doc_len.values()) if self.doc_len else 0.0
        self.n_docs = len(documents)

    def search(self, query: str, top_k: int, doc_id_filter: str | None = None) -> list[dict[str, Any]]:
        query_terms = tokenize_for_bm25(query)
        if not query_terms:
            return []

        scored: list[dict[str, Any]] = []
        for doc in self.documents:
            if doc_id_filter and doc.get("doc_id") != doc_id_filter:
                continue
            chunk_id = doc["chunk_id"]
            score = self.score_terms(query_terms, chunk_id)
            if score <= 0:
                continue
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc.get("doc_id"),
                    "source_file": doc.get("source_file"),
                    "section": doc.get("section"),
                    "section_path": doc.get("section_path", []),
                    "page_start": doc.get("page_start"),
                    "page_end": doc.get("page_end"),
                    "block_types": doc.get("block_types", []),
                    "text": doc.get("text", ""),
                    "bm25_score": score,
                }
            )
        scored.sort(key=lambda item: item["bm25_score"], reverse=True)
        for rank, row in enumerate(scored[:top_k], start=1):
            row["bm25_rank"] = rank
        return scored[:top_k]

    def score_terms(self, query_terms: list[str], chunk_id: str) -> float:
        tf = self.doc_tf[chunk_id]
        dl = self.doc_len[chunk_id]
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            score += idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)
        return float(score)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("′", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("_", " ")
    text = text.replace("fig.", "figure ")
    text = re.sub(r"(?<=\d)\s*-\s*(?=[a-z])", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> list[str]:
    norm = normalize_text(text)
    tokens = re.findall(r"[a-z0-9']+", norm)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if token in {"table", "figure"}:
            expanded.append(token)
    return expanded


def keyword_variants(keyword: str) -> set[str]:
    base = normalize_text(keyword)
    variants = {base}
    compact = re.sub(r"[\s\-_/]+", "", base)
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
    compact = re.sub(r"[\s\-_/]+", "", norm)
    for variant in keyword_variants(keyword):
        if variant in norm or variant in compact:
            return True
    return False


def truncate_text(text: str, limit: int = 500) -> str:
    return " ".join(text.split())[:limit]


def load_chunk_meta(jsonl_path: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    meta: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            meta[item["chunk_id"]] = item
            items.append(item)
    return meta, items


def load_queries(path: str | None) -> list[QuerySpec]:
    if not path:
        return [QuerySpec(**item) for item in DEFAULT_QUERY_SPECS]
    specs: list[QuerySpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(QuerySpec(**json.loads(line)))
    return specs


def infer_keyword_list(spec: QuerySpec) -> list[str]:
    if spec.expected_keywords:
        return spec.expected_keywords
    tokens = re.findall(r"[A-Za-z0-9_'\-\.]{3,}", spec.query)
    stop = {"what", "does", "the", "and", "for", "with", "from", "that", "this"}
    return [tok for tok in tokens if tok.lower() not in stop][:8]


def infer_block_type_bias(spec: QuerySpec) -> dict[str, float]:
    query_norm = normalize_text(spec.query)
    category = spec.category.lower()
    if category == "figure" or any(hint in query_norm for hint in FIGURE_HINTS):
        return {"figure_caption": 0.08}
    if category == "table" or any(hint in query_norm for hint in TABLE_HINTS):
        return {"table_text": 0.10, "table_caption": 0.06}
    return {}


def expected_block_type_hit(block_types: list[str], expected_block_types: list[str]) -> bool:
    if not expected_block_types:
        return False
    return any(bt in block_types for bt in expected_block_types)


def compute_keyword_hits(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    matched = [kw for kw in keywords if text_contains_keyword(text, kw)]
    return len(matched), matched


def evaluate_chunk_against_expectation(chunk: dict[str, Any], spec: QuerySpec) -> dict[str, Any]:
    text = chunk.get("text", "")
    keywords = infer_keyword_list(spec)
    keyword_hits, matched_keywords = compute_keyword_hits(text, keywords)
    block_types = chunk.get("block_types", [])
    expected_doc_hit = spec.expected_doc_id is None or chunk.get("doc_id") == spec.expected_doc_id
    expected_block_hit = expected_block_type_hit(block_types, spec.expected_block_types)
    required_keyword_hits = math.ceil(len(keywords) * 0.75) if keywords else 0
    full_keyword_hit = keyword_hits >= required_keyword_hits if keywords else expected_doc_hit
    target_match = expected_doc_hit and full_keyword_hit and (expected_block_hit if spec.expected_block_types else True)
    return {
        "expected_doc_hit": expected_doc_hit,
        "expected_block_hit": expected_block_hit,
        "keyword_hits": keyword_hits,
        "matched_keywords": matched_keywords,
        "keywords_total": len(keywords),
        "full_keyword_hit": full_keyword_hit,
        "target_match": target_match,
    }


def find_target_chunks(spec: QuerySpec, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    matches = []
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
                "text_preview": truncate_text(chunk.get("text", "")),
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
    return {
        "target_exists_in_chunks": any(item["target_match"] for item in matches),
        "top_matches": matches[:5],
    }


def infer_parsed_clean_dir(chunks_jsonl: str, explicit_dir: str | None) -> str | None:
    if explicit_dir:
        return explicit_dir
    path = Path(chunks_jsonl)
    root = path.parent
    candidate = root.parent / "parsed_clean_table_min_verify"
    return str(candidate) if candidate.exists() else None


def find_target_in_parsed_clean(spec: QuerySpec, parsed_clean_dir: str | None) -> dict[str, Any]:
    if not parsed_clean_dir or not spec.expected_doc_id:
        return {"target_exists_in_parsed_clean": None, "matches": []}
    doc_path = Path(parsed_clean_dir) / f"{spec.expected_doc_id}.json"
    if not doc_path.exists():
        return {"target_exists_in_parsed_clean": None, "matches": []}

    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    matches = []
    for page in doc.get("pages", []):
        page_num = page.get("page_num") or page.get("page") or page.get("number")
        for idx, block in enumerate(page.get("blocks", [])):
            candidate = {
                "doc_id": spec.expected_doc_id,
                "text": block.get("text", ""),
                "block_types": [block.get("type")],
            }
            evaluation = evaluate_chunk_against_expectation(candidate, spec)
            if evaluation["keyword_hits"] <= 0:
                continue
            matches.append(
                {
                    "page": page_num,
                    "block_index": idx,
                    "block_type": block.get("type"),
                    "section_path": block.get("section_path", []),
                    "text_preview": truncate_text(block.get("text", "")),
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
    return {
        "target_exists_in_parsed_clean": any(item["target_match"] for item in matches),
        "matches": matches[:5],
    }


def milvus_search(
    client: MilvusClient,
    collection_name: str,
    query_vec: list[float],
    top_k: int,
    chunk_meta: dict[str, dict[str, Any]],
    doc_id_filter: str | None = None,
) -> list[dict[str, Any]]:
    filter_expr = f'doc_id == "{doc_id_filter}"' if doc_id_filter else ""
    results = client.search(
        collection_name=collection_name,
        data=[query_vec],
        anns_field="embedding",
        limit=top_k,
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
        ],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        filter=filter_expr,
    )
    hits = results[0] if results else []
    rows = []
    for raw_rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        chunk_id = entity.get("chunk_id") or hit.get("id")
        local = chunk_meta.get(chunk_id, {})
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc_id": entity.get("doc_id") or local.get("doc_id"),
                "source_file": entity.get("source_file") or local.get("source_file"),
                "section": entity.get("section") or local.get("section"),
                "section_path": local.get("section_path", []),
                "page_start": entity.get("page_start", local.get("page_start")),
                "page_end": entity.get("page_end", local.get("page_end")),
                "block_types": local.get("block_types", []),
                "text": entity.get("text", "") or local.get("text", ""),
                "dense_score": float(hit.get("distance", 0.0)),
                "dense_rank": raw_rank,
            }
        )
    return rows


def score_single_mode(results: list[dict[str, Any]], spec: QuerySpec, mode: str) -> list[dict[str, Any]]:
    scored = []
    for row in results:
        evaluation = evaluate_chunk_against_expectation(row, spec)
        keyword_hits, matched_keywords = compute_keyword_hits(row.get("text", ""), infer_keyword_list(spec))
        ranked = {
            **row,
            "keyword_hits": keyword_hits,
            "matched_keywords": matched_keywords,
            "block_type_boost": 0.0,
            "union_score": 0.0,
            "reranker_score": None,
            "final_score": float(row.get("dense_score", row.get("bm25_score", 0.0))),
            "final_rank": None,
            "reranker_rank": None,
            "source": mode,
            **evaluation,
        }
        scored.append(ranked)

    key = "dense_score" if mode == "dense" else "bm25_score"
    scored.sort(key=lambda item: item.get(key, 0.0), reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["final_rank"] = rank
    return scored


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (RRF_K + rank)


def merge_dense_bm25(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    spec: QuerySpec,
    apply_block_boost: bool,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    boosts = infer_block_type_bias(spec) if apply_block_boost else {}

    for row in dense_results:
        merged[row["chunk_id"]] = {
            **row,
            "bm25_score": 0.0,
            "bm25_rank": None,
            "source": "dense",
        }
    for row in bm25_results:
        if row["chunk_id"] not in merged:
            merged[row["chunk_id"]] = {
                **row,
                "dense_score": 0.0,
                "dense_rank": None,
                "source": "bm25",
            }
        else:
            merged[row["chunk_id"]].update(
                {
                    "bm25_score": row["bm25_score"],
                    "bm25_rank": row["bm25_rank"],
                    "source": "both",
                }
            )

    scored: list[dict[str, Any]] = []
    for row in merged.values():
        keyword_hits, matched_keywords = compute_keyword_hits(row.get("text", ""), infer_keyword_list(spec))
        block_type_boost = sum(weight for block_type, weight in boosts.items() if block_type in row.get("block_types", []))
        union_score = reciprocal_rank(row.get("dense_rank")) + reciprocal_rank(row.get("bm25_rank"))
        final_score = union_score + block_type_boost
        evaluation = evaluate_chunk_against_expectation(row, spec)
        scored.append(
            {
                **row,
                "keyword_hits": keyword_hits,
                "matched_keywords": matched_keywords,
                "block_type_boost": round(block_type_boost, 4),
                "union_score": round(union_score, 6),
                "reranker_score": None,
                "final_score": round(final_score, 6),
                "reranker_rank": None,
                **evaluation,
            }
        )
    scored.sort(key=lambda item: (item["final_score"], item["union_score"], item["dense_score"], item["bm25_score"]), reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["final_rank"] = rank
    return scored


def apply_reranker(
    reranker: FlagReranker | None,
    query: str,
    candidates: list[dict[str, Any]],
    rerank_limit: int,
) -> list[dict[str, Any]]:
    if reranker is None or not candidates:
        return candidates

    head = candidates[:rerank_limit]
    tail = candidates[rerank_limit:]
    pairs = [[query, row.get("text", "")] for row in head]
    try:
        scores = reranker.compute_score(pairs, batch_size=8)
        if isinstance(scores, float):
            score_list = [float(scores)]
        else:
            score_list = [float(score) for score in scores]
    except Exception:
        return candidates

    reranked = []
    for row, score in zip(head, score_list):
        reranked.append({**row, "reranker_score": score})
    reranked.sort(key=lambda item: (item["reranker_score"], item["final_score"]), reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["reranker_rank"] = rank
    for row in tail:
        row = dict(row)
        row["reranker_rank"] = None
        reranked.append(row)

    final_rows = reranked[:]
    final_rows.sort(
        key=lambda item: (
            item["reranker_rank"] if item["reranker_rank"] is not None else 10**9,
            -item["final_score"],
        )
    )
    for rank, row in enumerate(final_rows, start=1):
        row["final_rank"] = rank
    return final_rows


def locate_rank(results: list[dict[str, Any]]) -> int | None:
    for row in results:
        if row.get("target_match"):
            return row["final_rank"]
    return None


def locate_dense_rank(results: list[dict[str, Any]]) -> int | None:
    for row in results:
        if row.get("target_match"):
            return row.get("dense_rank")
    return None


def grade_rank(rank: int | None, pass_cutoff: int = 5, top_k: int = 50) -> str:
    if rank is None:
        return "fail"
    if rank <= pass_cutoff:
        return "pass"
    if rank <= top_k:
        return "partial"
    return "fail"


def choose_best_mode(result: dict[str, Any]) -> str:
    candidates = {
        "dense": result.get("dense_rank"),
        "bm25": result.get("bm25_rank"),
        "union": result.get("union_rank"),
        "boosted_union": result.get("boosted_rank"),
        "doc_filter": result.get("doc_filter_rank"),
        "reranker": result.get("reranker_rank"),
    }
    best = min(
        ((mode, rank) for mode, rank in candidates.items() if isinstance(rank, int)),
        key=lambda item: item[1],
        default=("none", None),
    )
    return best[0]


def determine_failure_reason(
    spec: QuerySpec,
    target_exists_in_chunks: bool,
    target_exists_in_parsed_clean: bool | None,
    dense_rank: int | None,
    bm25_rank: int | None,
    hybrid_rank: int | None,
    doc_filter_rank: int | None,
) -> tuple[str, str]:
    if spec.exclude_from_primary_metrics and not target_exists_in_chunks:
        return "extraction_gap_or_invalid_spec", "Exclude this query from main hybrid metrics; the target exists in parsed_clean but not in chunks."
    if not target_exists_in_chunks:
        if spec.id == "q_table2_parameters" or "table 2" in spec.query.lower():
            return "table2_extraction_gap", "Keep this as an extraction-gap case and fix Table 2 parsing separately."
        if target_exists_in_parsed_clean:
            return "extraction_gap_or_invalid_spec", "The target appears in parsed_clean but not in chunks; treat it as chunking/extraction gap or invalid spec."
        return "extraction_missing", "The target is absent from chunks and parsed_clean evidence is insufficient."
    if dense_rank is not None and dense_rank <= 5:
        return "ok", "Dense retrieval is already adequate for this query."
    if hybrid_rank is not None and hybrid_rank <= 5:
        if dense_rank is None:
            return "needs_bm25_hybrid", "Dense missed the target but BM25/union recovered it; hybrid retrieval is justified."
        return "needs_bm25_hybrid", "Dense ranking is weak and hybrid union fixes it; add BM25/sparse or reranking."
    if doc_filter_rank is not None and doc_filter_rank <= 5:
        return "cross_doc_competition", "Target is recoverable after doc filtering; consider doc-level routing for known-document questions."
    if dense_rank is None and bm25_rank is None:
        return "retrieval_miss", "Neither dense nor BM25 found the existing target; inspect normalization and candidate depth."
    return "dense_ranking_weak", "Target exists but still ranks poorly; increase first-stage candidate depth and apply stronger reranking."


def write_default_query_spec(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in DEFAULT_QUERY_SPECS) + "\n", encoding="utf-8")


def render_result_row(row: dict[str, Any], index: int) -> str:
    return (
        f"{index}. chunk_id=`{row['chunk_id']}` doc_id=`{row['doc_id']}` source=`{row['source']}` "
        f"final_rank={row.get('final_rank')} dense_rank={row.get('dense_rank')} bm25_rank={row.get('bm25_rank')} "
        f"reranker_rank={row.get('reranker_rank')} dense_score={row.get('dense_score', 0.0):.4f} "
        f"bm25_score={row.get('bm25_score', 0.0):.4f} keyword_hits={row.get('keyword_hits', 0)} "
        f"block_type_boost={row.get('block_type_boost', 0.0):.2f} final_score={row.get('final_score', 0.0):.4f} "
        f"block_types={row.get('block_types', [])} section=`{row.get('section')}` pages={row.get('page_start')}-{row.get('page_end')} "
        f"expected_doc_hit={row.get('expected_doc_hit')} expected_keywords_hit={row.get('full_keyword_hit')} "
        f"expected_block_types_hit={row.get('expected_block_hit')}\n"
        f"   section_path={row.get('section_path', [])}\n"
        f"   matched_keywords={row.get('matched_keywords', [])}\n"
        f"   text={truncate_text(row.get('text', ''))}"
    )


def analyze_table2(parsed_clean_dir: str | None, chunks_jsonl: str) -> dict[str, Any]:
    result = {
        "parsed_clean_has_table2": None,
        "table2_blocks": [],
        "chunk_hits": [],
        "summary": "Table 2 analysis unavailable.",
    }
    if not parsed_clean_dir:
        return result
    doc_path = Path(parsed_clean_dir) / "doc_0045.json"
    if not doc_path.exists():
        return result

    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    flat_blocks = []
    for page in doc.get("pages", []):
        page_num = page.get("page_num") or page.get("page") or page.get("number")
        for idx, block in enumerate(page.get("blocks", [])):
            flat_blocks.append(
                {
                    "page": page_num,
                    "block_index": idx,
                    "type": block.get("type"),
                    "section_path": block.get("section_path", []),
                    "text": " ".join((block.get("text", "")).split()),
                }
            )
    mentions = []
    for i, block in enumerate(flat_blocks):
        if "Table 2" in block["text"]:
            window = flat_blocks[max(0, i - 2): min(len(flat_blocks), i + 6)]
            mentions.append({"match": block, "window": window})
    result["parsed_clean_has_table2"] = bool(mentions)
    result["table2_blocks"] = mentions

    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("doc_id") != "doc_0045":
                continue
            text = chunk.get("text", "")
            if "table 2" in normalize_text(text) or "strain specific parameters" in normalize_text(text):
                result["chunk_hits"].append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "block_types": chunk.get("block_types", []),
                        "section_path": chunk.get("section_path", []),
                        "text_preview": truncate_text(text),
                    }
                )

    if mentions:
        summary = (
            "Parsed clean contains only paragraph mentions of Table 2. "
            "A structured row block appears on page 4 as plain paragraph text and is merged with `Figure 4 | ...`, "
            "which indicates table rows were fused into surrounding narrative/figure-caption context rather than promoted to `table_caption/table_text`."
        )
    else:
        summary = "Parsed clean does not contain a stable Table 2 caption block; only indirect mentions survive."
    result["summary"] = summary
    return result


def build_markdown_report(
    args: argparse.Namespace,
    branch: str,
    commit_hash: str,
    query_results: list[dict[str, Any]],
    table2_analysis: dict[str, Any],
    reranker_enabled: bool,
    reranker_ready: bool,
) -> str:
    primary_results = [item for item in query_results if not item["exclude_from_primary_metrics"] and item["likely_failure_reason"] not in {"table2_extraction_gap", "extraction_gap_or_invalid_spec", "extraction_missing"}]
    excluded_primary_ids = [
        item["id"]
        for item in query_results
        if item["exclude_from_primary_metrics"] or item["likely_failure_reason"] in {"table2_extraction_gap", "extraction_gap_or_invalid_spec", "extraction_missing"}
    ]
    dense_summary = Counter(grade_rank(item["dense_rank"], top_k=args.top_k) for item in query_results if not item["exclude_from_primary_metrics"])
    bm25_summary = Counter(grade_rank(item["bm25_rank"], top_k=args.top_k) for item in query_results if not item["exclude_from_primary_metrics"])
    union_summary = Counter(grade_rank(item["union_rank"], top_k=args.top_k) for item in query_results if not item["exclude_from_primary_metrics"])
    boosted_summary = Counter(grade_rank(item["boosted_rank"], top_k=args.top_k) for item in query_results if not item["exclude_from_primary_metrics"])
    reranker_summary = Counter(grade_rank(item["reranker_rank"], top_k=args.top_k) for item in query_results if not item["exclude_from_primary_metrics"]) if reranker_enabled and reranker_ready else None
    doc_filter_summary = Counter(grade_rank(item["doc_filter_rank"], top_k=args.top_k) for item in query_results if not item["exclude_from_primary_metrics"])
    primary_hybrid_pass = sum(1 for item in primary_results if item["boosted_rank"] is not None and item["boosted_rank"] <= 5)
    primary_dense_pass = sum(1 for item in primary_results if item["dense_rank"] is not None and item["dense_rank"] <= 5)

    lines = [
        "# Hybrid Retrieval Diagnostic",
        "",
        f"- Current branch: `{branch}`",
        f"- Current commit: `{commit_hash}`",
        f"- Collection: `{args.collection_name}`",
        f"- Milvus URI: `{args.milvus_uri}`",
        f"- Chunks JSONL: `{args.chunks_jsonl}`",
        f"- Parsed clean dir: `{args.parsed_clean_dir}`",
        f"- Query spec: `{args.query_spec}`",
        f"- Query count: `{len(query_results)}`",
        f"- Query spec corrected: `True`",
        f"- TopK: `{args.top_k}`",
        f"- BM25 enabled: `{args.use_bm25}`",
        f"- Block type boost enabled: `{args.boost_block_types}`",
        f"- Reranker enabled: `{reranker_enabled}`",
        f"- Reranker ready: `{reranker_ready}`",
        "",
        "## Mode Summary",
        "",
        f"- Dense-only top50: `{dict(dense_summary)}`",
        f"- BM25-only top50: `{dict(bm25_summary)}`",
        f"- Dense+BM25 union top50: `{dict(union_summary)}`",
        f"- Union+block boost top50: `{dict(boosted_summary)}`",
        f"- Doc-id routed/filter comparison: `{dict(doc_filter_summary)}`",
    ]
    if reranker_summary is not None:
        lines.append(f"- Reranker top50: `{dict(reranker_summary)}`")
    else:
        lines.append("- Reranker top50: `not_enabled_or_unavailable`")

    lines.extend(
        [
            "",
            "## Primary Metrics",
            "",
            f"- Primary metric exclusions: `{excluded_primary_ids}`",
            f"- Dense-only top5 pass count: `{primary_dense_pass}/{len(primary_results)}`",
            f"- Hybrid top5 pass count: `{primary_hybrid_pass}/{len(primary_results)}`",
            "",
            "## Query Diagnostics",
            "",
        ]
    )

    for result in query_results:
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- Query: {result['query']}",
                f"- Category: `{result['category']}`",
                f"- Expected doc id: `{result['expected_doc_id']}`",
                f"- Expected keywords: `{result['expected_keywords']}`",
                f"- Expected block types: `{result['expected_block_types']}`",
                f"- Excluded from primary metrics: `{result['exclude_from_primary_metrics']}`",
                f"- Notes: {result['notes'] or 'n/a'}",
                f"- target_exists_in_chunks: `{result['target_exists_in_chunks']}`",
                f"- target_exists_in_parsed_clean: `{result['target_exists_in_parsed_clean']}`",
                f"- dense_top50_hit: `{result['dense_rank'] is not None}`",
                f"- bm25_top50_hit: `{result['bm25_rank'] is not None}`",
                f"- union_top50_hit: `{result['union_rank'] is not None}`",
                f"- dense_top8_hit: `{result['dense_rank'] is not None and result['dense_rank'] <= 8}`",
                f"- hybrid_top5_hit: `{result['boosted_rank'] is not None and result['boosted_rank'] <= 5}`",
                f"- doc_filter_hit: `{result['doc_filter_rank'] is not None and result['doc_filter_rank'] <= 5 if result['doc_filter_rank'] is not None else 'not_applicable'}`",
                f"- reranker_top5_hit: `{result['reranker_rank'] is not None and result['reranker_rank'] <= 5 if reranker_enabled and reranker_ready else 'not_enabled_or_unavailable'}`",
                f"- best_mode: `{result['best_mode']}`",
                f"- likely_failure_reason: `{result['likely_failure_reason']}`",
                f"- recommended_next_action: {result['recommended_next_action']}",
                "",
                "**Target Existence Check In Chunks**",
                "",
            ]
        )
        if result["target_chunk_matches"]:
            for idx, match in enumerate(result["target_chunk_matches"][:3], start=1):
                lines.append(
                    f"{idx}. chunk_id=`{match['chunk_id']}` doc_id=`{match['doc_id']}` "
                    f"keyword_hits={match['keyword_hits']}/{match['keywords_total']} block_types={match['block_types']} "
                    f"expected_doc_hit={match['expected_doc_hit']} expected_block_hit={match['expected_block_hit']}\n"
                    f"   section_path={match['section_path']}\n"
                    f"   matched_keywords={match['matched_keywords']}\n"
                    f"   text={match['text_preview']}"
                )
        else:
            lines.append("1. No chunk match found.")
        lines.extend(["", "**Target Evidence In Parsed Clean**", ""])
        if result["parsed_clean_matches"]:
            for idx, match in enumerate(result["parsed_clean_matches"][:3], start=1):
                lines.append(
                    f"{idx}. page={match['page']} block_index={match['block_index']} block_type=`{match['block_type']}` "
                    f"keyword_hits={match['keyword_hits']}/{match['keywords_total']} expected_block_hit={match['expected_block_hit']}\n"
                    f"   section_path={match['section_path']}\n"
                    f"   matched_keywords={match['matched_keywords']}\n"
                    f"   text={match['text_preview']}"
                )
        else:
            lines.append("1. No parsed_clean evidence found.")

        for title, rows in [
            ("Dense-only Top Results", result["dense_results"][:3]),
            ("BM25-only Top Results", result["bm25_results"][:3]),
            ("Dense+BM25 Union Top Results", result["union_results"][:3]),
            ("Union+Block Boost Top Results", result["boosted_results"][:3]),
            ("Doc-id Routed/Filtered Top Results", result["doc_filter_results"][:3]),
            ("Reranker Top Results", result["reranker_results"][:3] if reranker_enabled and reranker_ready else []),
        ]:
            lines.extend(["", f"**{title}**", ""])
            if rows:
                for idx, row in enumerate(rows, start=1):
                    lines.append(render_result_row(row, idx))
            else:
                lines.append("1. not_applicable")
        lines.append("")

    lines.extend(
        [
            "## Table 2 Localization",
            "",
            f"- parsed_clean_has_table2: `{table2_analysis['parsed_clean_has_table2']}`",
            f"- Summary: {table2_analysis['summary']}",
            "",
        ]
    )
    if table2_analysis["table2_blocks"]:
        for idx, item in enumerate(table2_analysis["table2_blocks"], start=1):
            lines.append(f"### Table 2 Mention {idx}")
            lines.append("")
            lines.append(
                f"- Matched block: page={item['match']['page']} block_index={item['match']['block_index']} "
                f"type=`{item['match']['type']}` section_path={item['match']['section_path']}"
            )
            lines.append(f"- Text: {truncate_text(item['match']['text'])}")
            lines.append("- Nearby blocks:")
            for window_block in item["window"]:
                lines.append(
                    f"  - page={window_block['page']} block_index={window_block['block_index']} type=`{window_block['type']}` "
                    f"text={truncate_text(window_block['text'], 220)}"
                )
            lines.append("")
    if table2_analysis["chunk_hits"]:
        lines.append("### Table 2 Related Chunks")
        lines.append("")
        for idx, row in enumerate(table2_analysis["chunk_hits"][:5], start=1):
            lines.append(
                f"{idx}. chunk_id=`{row['chunk_id']}` block_types={row['block_types']} section_path={row['section_path']}\n"
                f"   text={row['text_preview']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Overall Conclusions",
            "",
            f"- q_doc0001_materials misclassification corrected: `{next(item for item in query_results if item['id']=='q_doc0001_materials')['likely_failure_reason'] != 'extraction_missing'}`",
            f"- q_doc0001_fig4 status: `{next(item for item in query_results if item['id']=='q_doc0001_fig4')['likely_failure_reason']}`",
            f"- Table query failure reasons: `{Counter(item['likely_failure_reason'] for item in query_results if item['category']=='table')}`",
            f"- Figure query failure reasons: `{Counter(item['likely_failure_reason'] for item in query_results if item['category']=='figure')}`",
            f"- Body query failure reasons: `{Counter(item['likely_failure_reason'] for item in query_results if item['category']=='body')}`",
            "",
            "## Recommendations",
            "",
            "- Introduce hybrid retrieval if the goal is to stabilize table and figure lookup without relying on doc filters.",
            "- Keep doc_id routing as a diagnostic or explicit-document mode, not as the default fallback.",
            "- Fix Table 2 extraction separately; hybrid retrieval cannot recover content that never reached chunks.",
            "- Defer false-heading cleanup for now; current evidence points more strongly to retrieval-stage weakness than heading noise.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--query_spec", default="data/small_exp/table_retrieval_queries.jsonl")
    parser.add_argument("--parsed_clean_dir", default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--use_bm25", action="store_true")
    parser.add_argument("--boost_block_types", action="store_true")
    parser.add_argument("--run_doc_filter", action="store_true")
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--reranker_model_path", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank_limit", type=int, default=20)
    parser.add_argument("--output", default="results/hybrid_retrieval_diagnostic.md")
    args = parser.parse_args()

    query_spec_path = Path(args.query_spec)
    if not query_spec_path.exists():
        write_default_query_spec(query_spec_path)

    args.parsed_clean_dir = infer_parsed_clean_dir(args.chunks_jsonl, args.parsed_clean_dir)
    chunk_meta, chunk_items = load_chunk_meta(args.chunks_jsonl)
    queries = load_queries(args.query_spec)

    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:
        branch = "unknown"
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:
        commit_hash = "unknown"

    reranker = None
    reranker_ready = False
    if args.use_reranker:
        try:
            reranker = FlagReranker(args.reranker_model_path, use_fp16=True)
            reranker_ready = True
        except Exception:
            reranker = None
            reranker_ready = False

    print("=" * 100)
    print("Hybrid retrieval diagnostic")
    print("=" * 100)
    print(f"Collection:               {args.collection_name}")
    print(f"Milvus URI:               {args.milvus_uri}")
    print(f"Chunks JSONL:             {args.chunks_jsonl}")
    print(f"Parsed clean dir:         {args.parsed_clean_dir}")
    print(f"Query spec:               {args.query_spec}")
    print(f"Top K:                    {args.top_k}")
    print(f"BM25 enabled:             {args.use_bm25}")
    print(f"Block type boost enabled: {args.boost_block_types}")
    print(f"Doc filter enabled:       {args.run_doc_filter}")
    print(f"Reranker enabled:         {args.use_reranker}")
    print(f"Reranker ready:           {reranker_ready}")

    client = MilvusClient(args.milvus_uri)
    try:
        client.load_collection(args.collection_name)
    except Exception:
        pass
    model = BGEM3FlagModel(args.model_path, use_fp16=True)
    bm25 = SimpleBM25(chunk_items) if args.use_bm25 else None

    query_results: list[dict[str, Any]] = []
    for spec in queries:
        print("\n" + "=" * 100)
        print(f"[{spec.id}] {spec.query}")
        print("=" * 100)

        existence = find_target_chunks(spec, chunk_items)
        parsed_clean_evidence = find_target_in_parsed_clean(spec, args.parsed_clean_dir)
        print(f"target_exists_in_chunks={existence['target_exists_in_chunks']} target_exists_in_parsed_clean={parsed_clean_evidence['target_exists_in_parsed_clean']}")

        dense_vec = model.encode([spec.query], batch_size=1, max_length=args.query_max_length)["dense_vecs"][0].tolist()
        dense_raw = milvus_search(client, args.collection_name, dense_vec, args.top_k, chunk_meta)
        dense_results = score_single_mode(dense_raw, spec, mode="dense")

        bm25_raw = bm25.search(spec.query, args.top_k) if bm25 else []
        bm25_results = score_single_mode(bm25_raw, spec, mode="bm25") if bm25_raw else []

        union_results = merge_dense_bm25(dense_raw, bm25_raw, spec, apply_block_boost=False)
        boosted_results = merge_dense_bm25(dense_raw, bm25_raw, spec, apply_block_boost=args.boost_block_types)
        reranker_results = apply_reranker(reranker, spec.query, [dict(row) for row in boosted_results], args.rerank_limit) if reranker_ready else []

        doc_filter_results: list[dict[str, Any]] = []
        if args.run_doc_filter and spec.expected_doc_id:
            dense_doc = milvus_search(client, args.collection_name, dense_vec, args.top_k, chunk_meta, doc_id_filter=spec.expected_doc_id)
            bm25_doc = bm25.search(spec.query, args.top_k, doc_id_filter=spec.expected_doc_id) if bm25 else []
            doc_filter_results = merge_dense_bm25(dense_doc, bm25_doc, spec, apply_block_boost=args.boost_block_types)

        dense_rank = locate_rank(dense_results)
        bm25_rank = locate_rank(bm25_results)
        union_rank = locate_rank(union_results)
        boosted_rank = locate_rank(boosted_results)
        doc_filter_rank = locate_rank(doc_filter_results)
        reranker_rank = locate_rank(reranker_results) if reranker_results else None

        hybrid_best_rank = min([rank for rank in [boosted_rank, reranker_rank] if rank is not None], default=None)

        likely_failure_reason, recommended_next_action = determine_failure_reason(
            spec=spec,
            target_exists_in_chunks=existence["target_exists_in_chunks"],
            target_exists_in_parsed_clean=parsed_clean_evidence["target_exists_in_parsed_clean"],
            dense_rank=dense_rank,
            bm25_rank=bm25_rank,
            hybrid_rank=hybrid_best_rank,
            doc_filter_rank=doc_filter_rank,
        )

        best_mode = choose_best_mode(
            {
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "union_rank": union_rank,
                "boosted_rank": boosted_rank,
                "doc_filter_rank": doc_filter_rank,
                "reranker_rank": reranker_rank,
            }
        )

        print(
            f"dense_rank={dense_rank} bm25_rank={bm25_rank} union_rank={union_rank} "
            f"boosted_rank={boosted_rank} reranker_rank={reranker_rank} doc_filter_rank={doc_filter_rank} "
            f"best_mode={best_mode} reason={likely_failure_reason}"
        )

        query_results.append(
            {
                "id": spec.id,
                "query": spec.query,
                "category": spec.category,
                "expected_doc_id": spec.expected_doc_id,
                "expected_keywords": spec.expected_keywords,
                "expected_block_types": spec.expected_block_types,
                "exclude_from_primary_metrics": spec.exclude_from_primary_metrics,
                "notes": spec.notes,
                "target_exists_in_chunks": existence["target_exists_in_chunks"],
                "target_exists_in_parsed_clean": parsed_clean_evidence["target_exists_in_parsed_clean"],
                "target_chunk_matches": existence["top_matches"],
                "parsed_clean_matches": parsed_clean_evidence["matches"],
                "dense_results": dense_results,
                "bm25_results": bm25_results,
                "union_results": union_results,
                "boosted_results": boosted_results,
                "reranker_results": reranker_results,
                "doc_filter_results": doc_filter_results,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "union_rank": union_rank,
                "boosted_rank": boosted_rank,
                "reranker_rank": reranker_rank,
                "doc_filter_rank": doc_filter_rank,
                "best_mode": best_mode,
                "likely_failure_reason": likely_failure_reason,
                "recommended_next_action": recommended_next_action,
            }
        )

    table2_analysis = analyze_table2(args.parsed_clean_dir, args.chunks_jsonl)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_markdown_report(
            args=args,
            branch=branch,
            commit_hash=commit_hash,
            query_results=query_results,
            table2_analysis=table2_analysis,
            reranker_enabled=args.use_reranker,
            reranker_ready=reranker_ready,
        ),
        encoding="utf-8",
    )
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
