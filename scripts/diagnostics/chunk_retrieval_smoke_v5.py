#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 Phase3b Chunk Retrieval Smoke — diagnostic-only, no Milvus rebuild.

Reads chunks.jsonl, runs BM25 / vector / hybrid / hybrid_rerank retrieval on
smoke cases, and generates pass/warning/fail reports.

Does NOT modify production retrieval, Milvus, generation, or reranker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ── BM25 diagnostic retriever ──────────────────────────────────────────────

class BM25DiagnosticRetriever:
    """Self-contained BM25 for diagnostic smoke. Not production BM25."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[dict[str, Any]] = []
        self.doc_freqs: dict[str, int] = {}
        self.avgdl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        tokens: list[str] = []
        for tok in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff\u3400-\u4dbf_]+", text):
            if len(tok) >= 2 or tok.isdigit() or re.search(r"[\u4e00-\u9fff]", tok):
                tokens.append(tok)
        return tokens

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        doc_lengths: list[int] = []
        for ch in chunks:
            text = ch.get("retrieval_text", "") or ch.get("text", "")
            tokens = self._tokenize(text)
            doc_lengths.append(len(tokens))
            for tok in set(tokens):
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1
        self.avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
        self._doc_lengths = doc_lengths
        self._doc_tokens: list[list[str]] = []
        for ch in chunks:
            text = ch.get("retrieval_text", "") or ch.get("text", "")
            self._doc_tokens.append(self._tokenize(text))

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        N = len(self.chunks)
        query_tokens = self._tokenize(query)
        scores: list[float] = [0.0] * N

        for qt in query_tokens:
            df = self.doc_freqs.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for i, doc_tokens in enumerate(self._doc_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = self._doc_lengths[i]
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[i] += score

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[dict[str, Any]] = []
        for idx, score in ranked[:limit]:
            if score <= 0:
                continue
            ch = self.chunks[idx]
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "doc_id": ch.get("doc_id", ""),
                "chunk_id": ch.get("chunk_id", ""),
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
                "evidence_types": ch.get("evidence_types", []),
                "block_types": ch.get("block_types", []),
                "source_block_ids": (ch.get("source_block_ids", []) or [])[:10],
                "text": (ch.get("text", "") or "")[:2000],
                "retrieval_text": (ch.get("retrieval_text", "") or "")[:2000],
            })
        return results


# ── Vector diagnostic retriever ─────────────────────────────────────────────

class VectorDiagnosticRetriever:
    """Diagnostic vector retriever using bge-m3. Embeddings cached to disk."""

    def __init__(self, model_path: str, cache_dir: str):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model: Any = None
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(self.model_path, use_fp16=True)

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self._load_model()
        keys = [
            ch.get("chunk_id", str(i))
            for i, ch in enumerate(chunks)
        ]
        cache_key = "emb_" + str(hash(tuple(keys))) + ".json"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            loaded = json.loads(cache_path.read_text())
            self.embeddings = loaded["embeddings"]
            print(f"  [vector] loaded {len(self.embeddings)} cached embeddings")
            return

        texts = [ch.get("retrieval_text", "") or ch.get("text", "") for ch in chunks]
        batch_size = 32
        self.embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            output = self.model.encode(batch, batch_size=batch_size, max_length=512)
            for vec in output["dense_vecs"]:
                self.embeddings.append(vec.tolist() if hasattr(vec, "tolist") else vec)

        cache_path.write_text(json.dumps({"embeddings": self.embeddings}))
        print(f"  [vector] computed and cached {len(self.embeddings)} embeddings")

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        self._load_model()
        output = self.model.encode([query], batch_size=1, max_length=512)
        q_vec = output["dense_vecs"][0]
        if hasattr(q_vec, "tolist"):
            q_vec = q_vec.tolist()

        scores = [self._cosine(q_vec, e) for e in self.embeddings]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        for idx, score in ranked[:limit]:
            if score <= 0:
                continue
            ch = self.chunks[idx]
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "doc_id": ch.get("doc_id", ""),
                "chunk_id": ch.get("chunk_id", ""),
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
                "evidence_types": ch.get("evidence_types", []),
                "block_types": ch.get("block_types", []),
                "source_block_ids": (ch.get("source_block_ids", []) or [])[:10],
                "text": (ch.get("text", "") or "")[:2000],
                "retrieval_text": (ch.get("retrieval_text", "") or "")[:2000],
            })
        return results


# ── Reranker diagnostic ─────────────────────────────────────────────────────

class RerankerDiagnostic:
    """Diagnostic reranker using bge-reranker-v2-m3."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Any = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(self.model_path, use_fp16=True)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []
        self._load_model()
        pairs = [[query, c.get("retrieval_text", "") or c.get("text", "")] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[dict[str, Any]] = []
        for c, s in scored[:top_k]:
            r = dict(c)
            r["rerank_score"] = float(s)
            r["rank"] = len(results) + 1
            results.append(r)
        return results


# ── Hybrid RRF ──────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = 60,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    info: dict[str, dict[str, Any]] = {}

    for r in bm25_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + r["rank"])
        info[cid] = r

    for r in vector_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + vector_weight / (k + r["rank"])
        if cid not in info:
            info[cid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results: list[dict[str, Any]] = []
    for cid, score in ranked:
        r = dict(info[cid])
        r["rrf_score"] = float(score)
        r["rank"] = len(results) + 1
        results.append(r)
    return results


# ── Evaluation ──────────────────────────────────────────────────────────────

def load_chunks(path: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_cases(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def check_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t.lower() in text.lower()]


def evaluate_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    top = results[:top_k]
    top_doc_ids = [r["doc_id"] for r in top]
    top_texts = " ".join(r.get("text", "") + " " + r.get("retrieval_text", "") for r in top)

    is_negative = case.get("negative_only", False)

    if is_negative:
        forbidden = case.get("forbidden_terms", [])
        hits_top1 = check_terms(
            top[0].get("text", "") + " " + top[0].get("retrieval_text", ""),
            forbidden,
        ) if top else []
        hits_top5 = check_terms(top_texts, forbidden)
        max_allowed = case.get("max_forbidden_topk_hits", 0)
        passed = len(hits_top5) <= max_allowed
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "expected_doc_ids": [],
            "mode": "N/A",
            "pass": passed,
            "verdict": "PASS" if passed else "FAIL",
            "top_k_results": top,
            "doc_hit_at_1": False,
            "doc_hit_at_3": False,
            "doc_hit_at_5": False,
            "expected_terms_hit": [],
            "expected_terms_missed": [],
            "evidence_type_hit": False,
            "forbidden_terms_top1": hits_top1,
            "forbidden_terms_top5": hits_top5,
            "mrr_doc": 0.0,
        }

    expected_docs = case.get("expected_doc_ids", [])
    expected_terms = case.get("expected_terms_any", [])
    expected_types = case.get("expected_evidence_types_any", [])
    negative_terms = case.get("negative_terms_absent", [])

    doc_hit_at_1 = top[0]["doc_id"] in expected_docs if top else False
    doc_hit_at_3 = any(r["doc_id"] in expected_docs for r in top[:3])
    doc_hit_at_5 = any(r["doc_id"] in expected_docs for r in top[:5])

    matched_terms = check_terms(top_texts, expected_terms)
    missed_terms = [t for t in expected_terms if t not in matched_terms]
    terms_hit = len(matched_terms) > 0

    types_in_top = set()
    for r in top:
        for et in (r.get("evidence_types") or []):
            types_in_top.add(et)
    type_hit = bool(set(expected_types) & types_in_top) if expected_types else True

    negative_term_hits = check_terms(top_texts, negative_terms)

    mrr = 0.0
    for rank, r in enumerate(top, 1):
        if r["doc_id"] in expected_docs:
            mrr = 1.0 / rank
            break

    passed = doc_hit_at_5 and terms_hit
    if negative_term_hits:
        passed = False

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "query": case["query"],
        "expected_doc_ids": expected_docs,
        "mode": "N/A",
        "pass": passed,
        "verdict": "PASS" if passed else ("WARNING" if doc_hit_at_5 else "FAIL"),
        "top_k_results": top,
        "doc_hit_at_1": doc_hit_at_1,
        "doc_hit_at_3": doc_hit_at_3,
        "doc_hit_at_5": doc_hit_at_5,
        "expected_terms_hit": matched_terms,
        "expected_terms_missed": missed_terms,
        "evidence_type_hit": type_hit,
        "forbidden_terms_top1": check_terms(top[0].get("text", "") + " " + top[0].get("retrieval_text", ""), negative_terms) if top else [],
        "forbidden_terms_top5": negative_term_hits,
        "mrr_doc": mrr,
    }


def compute_mode_metrics(
    mode_name: str,
    status: str,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [c for c in case_results if not c.get("category") == "negative" or not c.get("forbidden_terms_top5")]
    negative = [c for c in case_results if c.get("category") == "negative"]

    # Re-filter: positive cases are non-negative, negative are negative_only
    pos_cases = []
    neg_cases = []
    for c in case_results:
        cat = c.get("category", "")
        if cat == "negative":
            neg_cases.append(c)
        else:
            pos_cases.append(c)

    n_pos = len(pos_cases)
    n_neg = len(neg_cases)

    doc_hit_1 = sum(1 for c in pos_cases if c.get("doc_hit_at_1")) / max(n_pos, 1)
    doc_hit_3 = sum(1 for c in pos_cases if c.get("doc_hit_at_3")) / max(n_pos, 1)
    doc_hit_5 = sum(1 for c in pos_cases if c.get("doc_hit_at_5")) / max(n_pos, 1)
    terms_hit_5 = sum(1 for c in pos_cases if c.get("expected_terms_hit")) / max(n_pos, 1)
    type_hit_5 = sum(1 for c in pos_cases if c.get("evidence_type_hit")) / max(n_pos, 1)
    mrr = sum(c.get("mrr_doc", 0.0) for c in pos_cases) / max(n_pos, 1)

    forbidden_top1 = sum(len(c.get("forbidden_terms_top1", [])) for c in neg_cases)
    forbidden_top5 = sum(len(c.get("forbidden_terms_top5", [])) for c in neg_cases)

    neg_pass = sum(1 for c in neg_cases if c.get("pass")) / max(n_neg, 1) if n_neg else 1.0

    by_category: dict[str, dict[str, Any]] = {}
    for cat in ["table", "figure", "paragraph", "negative"]:
        cat_cases = [c for c in case_results if c.get("category") == cat]
        n_cat = len(cat_cases)
        if n_cat == 0:
            continue
        if cat == "negative":
            cat_pass_rate = sum(1 for c in cat_cases if c.get("pass")) / n_cat
            by_category[cat] = {"case_count": n_cat, "pass_count": sum(1 for c in cat_cases if c.get("pass")), "pass_rate": cat_pass_rate}
        else:
            cat_doc_hit_5 = sum(1 for c in cat_cases if c.get("doc_hit_at_5")) / n_cat
            cat_type_hit = sum(1 for c in cat_cases if c.get("evidence_type_hit")) / n_cat
            by_category[cat] = {
                "case_count": n_cat,
                "pass_count": sum(1 for c in cat_cases if c.get("pass")),
                "pass_rate": sum(1 for c in cat_cases if c.get("pass")) / n_cat,
                "doc_hit_at_5": cat_doc_hit_5,
                "evidence_type_hit_at_5": cat_type_hit,
            }

    evidence_type_dist: Counter = Counter()
    for c in pos_cases:
        top = c.get("top_k_results", [])
        if top:
            for et in top[0].get("evidence_types", []):
                evidence_type_dist[et] += 1

    return {
        "mode": mode_name,
        "status": status,
        "case_count": len(case_results),
        "positive_case_count": n_pos,
        "negative_case_count": n_neg,
        "doc_hit_at_1": round(doc_hit_1, 4),
        "doc_hit_at_3": round(doc_hit_3, 4),
        "doc_hit_at_5": round(doc_hit_5, 4),
        "expected_terms_hit_at_5": round(terms_hit_5, 4),
        "evidence_type_hit_at_5": round(type_hit_5, 4),
        "mrr_doc": round(mrr, 4),
        "forbidden_term_top1_count": forbidden_top1,
        "forbidden_term_top5_count": forbidden_top5,
        "negative_case_pass_rate": round(neg_pass, 4),
        "top1_evidence_type_distribution": dict(evidence_type_dist.most_common(10)),
        "by_category": by_category,
        "pass_count": sum(1 for c in case_results if c.get("pass")),
        "fail_count": sum(1 for c in case_results if not c.get("pass")),
    }


# ── Reports ─────────────────────────────────────────────────────────────────

def render_markdown(
    all_metrics: list[dict[str, Any]],
    all_case_results: dict[str, list[dict[str, Any]]],
    case_summary: dict[str, int],
) -> str:
    lines: list[str] = []
    lines.append("# v5 Phase3b Chunk Retrieval Smoke Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append("Diagnostic retrieval smoke on Phase B + Hotfix4 final chunks.")
    lines.append("No Milvus rebuild, no retrieval/generation/reranker modification.")
    lines.append("")
    lines.append("## 2. Input")
    lines.append("")
    lines.append("- chunks: `tmp/v5_phaseB_full_ingestion/chunks/chunks.jsonl` (Phase B + Hotfix4)")
    lines.append("- cases: `data/evaluation/v5_phase3b_chunk_retrieval_smoke_cases.jsonl`")
    lines.append("")
    lines.append("## 3. Case Set Summary")
    lines.append("")
    for cat, count in sorted(case_summary.items()):
        lines.append(f"- {cat}: {count} cases")
    lines.append(f"- **total**: {sum(case_summary.values())}")
    lines.append("")

    lines.append("## 4. Retrieval Mode Status")
    lines.append("")
    for m in all_metrics:
        lines.append(f"- **{m['mode']}**: {m['status']} ({m['case_count']} cases, {m['pass_count']} pass, {m['fail_count']} fail)")
    lines.append("")

    lines.append("## 5. Metrics")
    lines.append("")
    for m in all_metrics:
        lines.append(f"### {m['mode']} ({m['status']})")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        for key in ["doc_hit_at_1", "doc_hit_at_3", "doc_hit_at_5", "expected_terms_hit_at_5",
                     "evidence_type_hit_at_5", "mrr_doc", "forbidden_term_top1_count",
                     "forbidden_term_top5_count", "negative_case_pass_rate"]:
            lines.append(f"| {key} | {m.get(key, 'N/A')} |")
        lines.append("")
        bc = m.get("by_category", {})
        if bc:
            lines.append("#### By Category")
            lines.append("")
            lines.append("| Category | Cases | Pass | Pass Rate | Doc Hit@5 | EvType Hit@5 |")
            lines.append("|---|---|---|---|---|---|")
            for cat, info in sorted(bc.items()):
                lines.append(f"| {cat} | {info.get('case_count','')} | {info.get('pass_count','')} | "
                            f"{info.get('pass_rate','?')} | {info.get('doc_hit_at_5','N/A')} | "
                            f"{info.get('evidence_type_hit_at_5','N/A')} |")
            lines.append("")

    lines.append("## 6. Gate Cases")
    lines.append("")
    for mode_name, case_list in all_case_results.items():
        lines.append(f"### {mode_name}")
        for c in case_list:
            cid = c["case_id"]
            if cid in ("tbl_doc0005_gm_orf2729", "fig_doc0005_pathway") or c.get("category") == "negative":
                lines.append(f"- **{cid}**: {c['verdict']} (doc_hit@5={c.get('doc_hit_at_5', 'N/A')})")
        lines.append("")

    lines.append("## 7. Failed / Warning Cases")
    lines.append("")
    for mode_name, case_list in all_case_results.items():
        fw = [c for c in case_list if c.get("verdict") in ("FAIL", "WARNING")]
        if not fw:
            lines.append(f"### {mode_name}: none")
            lines.append("")
            continue
        lines.append(f"### {mode_name}")
        for c in fw:
            lines.append(f"- **{c['case_id']}** [{c['verdict']}]: query=`{c['query'][:80]}`")
            lines.append(f"  - doc_hit@5={c.get('doc_hit_at_5')} terms_missed={c.get('expected_terms_missed')}")
            top = c.get("top_k_results", [])
            if top:
                lines.append(f"  - top1: {top[0].get('chunk_id','?')} ({top[0].get('doc_id','?')})")
                lines.append(f"  - top1 evidence: {top[0].get('evidence_types',[])}")
                lines.append(f"  - top1 preview: {top[0].get('text','')[:120]}")
        lines.append("")

    lines.append("## 8. Interpretation")
    lines.append("")
    for m in all_metrics:
        parts = [f"**{m['mode']}**: "]
        dh5 = m.get("doc_hit_at_5", 0)
        th5 = m.get("expected_terms_hit_at_5", 0)
        if dh5 >= 0.80 and th5 >= 0.70:
            parts.append("Solid retrieval. ")
        elif dh5 >= 0.65:
            parts.append("Adequate but some misses. ")
        else:
            parts.append("Below smoke threshold. ")
        f5 = m.get("forbidden_term_top5_count", 0)
        if f5 > 0:
            parts.append(f"Contamination detected (top5={f5}). ")
        else:
            parts.append("No contamination. ")
        lines.append("".join(parts))

    return "\n".join(lines)


def render_manual_review(
    all_case_results: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []
    lines.append("# v5 Phase3b Manual Review Suggestions")
    lines.append("")

    all_fw: list[dict[str, Any]] = []
    for mode_name, case_list in all_case_results.items():
        for c in case_list:
            c = dict(c)
            c["_mode"] = mode_name
            all_fw.append(c)

    failed = [c for c in all_fw if c.get("verdict") == "FAIL"]
    warned = [c for c in all_fw if c.get("verdict") == "WARNING"]

    lines.append(f"Failed cases: {len(failed)}, Warning cases: {len(warned)}")
    lines.append("")

    review_items: list[dict[str, Any]] = []
    for c in failed + warned:
        review_items.append(c)

    if not review_items:
        lines.append("No failed or warning cases. No manual review required.")
        return "\n".join(lines)

    lines.append("## Review Items")
    lines.append("")
    lines.append("| # | case_id | verdict | query | top_doc | top_chunk | evidence_types |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, c in enumerate(review_items[:15], 1):
        top = c.get("top_k_results", [])
        top_doc = top[0].get("doc_id", "?") if top else "none"
        top_chunk = top[0].get("chunk_id", "?")[:40] if top else "none"
        top_ev = ",".join(top[0].get("evidence_types", [])[:3]) if top else ""
        lines.append(f"| {i} | {c['case_id']} | {c['verdict']} | {c['query'][:60]} | {top_doc} | {top_chunk} | {top_ev} |")

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    for i, c in enumerate(review_items[:15], 1):
        lines.append(f"### {i}. {c['case_id']} [{c['verdict']}] (mode={c.get('_mode','?')})")
        lines.append(f"- query: {c['query']}")
        lines.append(f"- expected_doc_ids: {c.get('expected_doc_ids', [])}")
        lines.append(f"- doc_hit@5: {c.get('doc_hit_at_5')}")
        lines.append(f"- terms_missed: {c.get('expected_terms_missed', [])}")
        lines.append(f"- forbidden_top5: {c.get('forbidden_terms_top5', [])}")
        lines.append("")
        for r in (c.get("top_k_results", []) or [])[:5]:
            lines.append(f"  - rank={r['rank']} {r['chunk_id']} doc={r['doc_id']} pages={r.get('page_start','?')}-{r.get('page_end','?')}")
            lines.append(f"    evidence_types={r.get('evidence_types',[])}")
            lines.append(f"    preview: {(r.get('text','') or r.get('retrieval_text',''))[:150]}")
        lines.append("")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v5 Phase3b Chunk Retrieval Smoke")
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output_json", default="reports/v5_phase3b_chunk_retrieval_smoke.json")
    parser.add_argument("--report", default="docs/reports/v5_phase3b_chunk_retrieval_smoke.md")
    parser.add_argument("--manual_review", default="docs/reports/v5_phase3b_manual_review.md")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--cache_dir", default="tmp/v5_phase3b_retrieval_smoke/cache")
    parser.add_argument("--embedding_model_path", default="./models/BAAI/bge-m3")
    parser.add_argument("--reranker_model_path", default="./models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--skip_vector", action="store_true")
    parser.add_argument("--skip_reranker", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("v5 Phase3b Chunk Retrieval Smoke")
    print("=" * 60)

    # Load
    chunks = load_chunks(args.chunks_jsonl)
    print(f"\nLoaded {len(chunks)} chunks")

    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} smoke cases")

    case_summary = Counter(c["category"] for c in cases)
    print(f"Case categories: {dict(case_summary)}")

    # ── BM25 ────────────────────────────────────────────────────────────────
    print("\n── BM25 Retrieval ──")
    bm25 = BM25DiagnosticRetriever()
    bm25.index(chunks)
    bm25_results: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        results = bm25.search(case["query"], args.candidate_k)[:args.top_k]
        bm25_results[case["case_id"]] = results

    bm25_case_results = [
        dict(evaluate_case(case, bm25_results[case["case_id"]], args.top_k), mode="bm25")
        for case in cases
    ]
    bm25_metrics = compute_mode_metrics("bm25", "RUN", bm25_case_results)
    print(f"  doc_hit@5={bm25_metrics['doc_hit_at_5']} terms@5={bm25_metrics['expected_terms_hit_at_5']} "
          f"forbidden_top5={bm25_metrics['forbidden_term_top5_count']}")

    all_metrics = [bm25_metrics]
    all_case_results: dict[str, list[dict[str, Any]]] = {"bm25": bm25_case_results}

    # ── Vector ──────────────────────────────────────────────────────────────
    vector_metrics = None
    vector_results: dict[str, list[dict[str, Any]]] = {}
    if args.skip_vector:
        print("\n── Vector Retrieval: SKIPPED (--skip_vector) ──")
        all_metrics.append({"mode": "vector", "status": "SKIPPED", "reason": "--skip_vector"})
    elif not Path(args.embedding_model_path).exists():
        print(f"\n── Vector Retrieval: SKIPPED (model not found: {args.embedding_model_path}) ──")
        all_metrics.append({"mode": "vector", "status": "SKIPPED", "reason": "model not found"})
    else:
        print("\n── Vector Retrieval ──")
        try:
            vector = VectorDiagnosticRetriever(args.embedding_model_path, args.cache_dir)
            vector.index(chunks)
            for case in cases:
                results = vector.search(case["query"], args.candidate_k)[:args.top_k]
                vector_results[case["case_id"]] = results

            vec_case_results = [
                dict(evaluate_case(case, vector_results[case["case_id"]], args.top_k), mode="vector")
                for case in cases
            ]
            vector_metrics = compute_mode_metrics("vector", "RUN", vec_case_results)
            all_metrics.append(vector_metrics)
            all_case_results["vector"] = vec_case_results
            print(f"  doc_hit@5={vector_metrics['doc_hit_at_5']} terms@5={vector_metrics['expected_terms_hit_at_5']} "
                  f"forbidden_top5={vector_metrics['forbidden_term_top5_count']}")
        except Exception as e:
            print(f"  Vector retrieval FAILED: {e}")
            all_metrics.append({"mode": "vector", "status": "FAIL", "reason": str(e)})

    # ── Hybrid ─────────────────────────────────────────────────────────────
    hybrid_metrics = None
    hybrid_results: dict[str, list[dict[str, Any]]] = {}
    if bm25_results and vector_results:
        print("\n── Hybrid RRF Retrieval ──")
        for case in cases:
            vec = vector_results.get(case["case_id"], [])
            bm = bm25_results.get(case["case_id"], [])
            fused = reciprocal_rank_fusion(bm, vec, k=60)
            hybrid_results[case["case_id"]] = fused[:args.top_k]

        hy_case_results = [
            dict(evaluate_case(case, hybrid_results[case["case_id"]], args.top_k), mode="hybrid")
            for case in cases
        ]
        hybrid_metrics = compute_mode_metrics("hybrid", "RUN", hy_case_results)
        all_metrics.append(hybrid_metrics)
        all_case_results["hybrid"] = hy_case_results
        print(f"  doc_hit@5={hybrid_metrics['doc_hit_at_5']} terms@5={hybrid_metrics['expected_terms_hit_at_5']} "
              f"forbidden_top5={hybrid_metrics['forbidden_term_top5_count']}")
    else:
        print("\n── Hybrid RRF Retrieval: SKIPPED (need both BM25 and vector) ──")
        all_metrics.append({"mode": "hybrid", "status": "SKIPPED", "reason": "need both BM25 and vector"})

    # ── Hybrid + Rerank ────────────────────────────────────────────────────
    rerank_metrics = None
    if args.skip_reranker:
        print("\n── Hybrid+Rerank: SKIPPED (--skip_reranker) ──")
        all_metrics.append({"mode": "hybrid_rerank", "status": "SKIPPED", "reason": "--skip_reranker"})
    elif not Path(args.reranker_model_path).exists():
        print(f"\n── Hybrid+Rerank: SKIPPED (model not found: {args.reranker_model_path}) ──")
        all_metrics.append({"mode": "hybrid_rerank", "status": "SKIPPED", "reason": "model not found"})
    elif not hybrid_results:
        print("\n── Hybrid+Rerank: SKIPPED (no hybrid results) ──")
        all_metrics.append({"mode": "hybrid_rerank", "status": "SKIPPED", "reason": "no hybrid results"})
    else:
        print("\n── Hybrid+Rerank (bge-reranker-v2-m3) ──")
        try:
            reranker = RerankerDiagnostic(args.reranker_model_path)
            rerank_results: dict[str, list[dict[str, Any]]] = {}
            for case in cases:
                candidates = hybrid_results.get(case["case_id"], [])[:args.candidate_k]
                reranked = reranker.rerank(case["query"], candidates, args.top_k)
                rerank_results[case["case_id"]] = reranked

            rr_case_results = [
                dict(evaluate_case(case, rerank_results[case["case_id"]], args.top_k), mode="hybrid_rerank")
                for case in cases
            ]
            rerank_metrics = compute_mode_metrics("hybrid_rerank", "RUN", rr_case_results)
            all_metrics.append(rerank_metrics)
            all_case_results["hybrid_rerank"] = rr_case_results
            print(f"  doc_hit@5={rerank_metrics['doc_hit_at_5']} terms@5={rerank_metrics['expected_terms_hit_at_5']} "
                  f"forbidden_top5={rerank_metrics['forbidden_term_top5_count']}")
        except Exception as e:
            print(f"  Hybrid+Rerank FAILED: {e}")
            all_metrics.append({"mode": "hybrid_rerank", "status": "FAIL", "reason": str(e)})

    # ── Overall verdict ────────────────────────────────────────────────────
    print("\n── Verdict ──")

    # Check gate cases
    bm25_cases = {c["case_id"]: c for c in bm25_case_results}
    gate_results: dict[str, bool] = {}
    for gate_id in ["tbl_doc0005_gm_orf2729", "fig_doc0005_pathway"]:
        gc = bm25_cases.get(gate_id)
        gate_results[gate_id] = gc["pass"] if gc else False

    # Check negative contamination in bm25
    neg_cases_bm25 = [c for c in bm25_case_results if c.get("category") == "negative"]
    neg_contaminated = sum(len(c.get("forbidden_terms_top5", [])) for c in neg_cases_bm25)

    # Determine overall
    bm25_dh5 = bm25_metrics["doc_hit_at_5"]
    bm25_th5 = bm25_metrics["expected_terms_hit_at_5"]

    if not gate_results.get("tbl_doc0005_gm_orf2729", False):
        overall = "FAIL"
        reason = "doc_0005 gm_orf2729 table case failed"
    elif not gate_results.get("fig_doc0005_pathway", False):
        overall = "FAIL"
        reason = "doc_0005 Fig. 6 figure case failed"
    elif neg_contaminated > 0:
        overall = "FAIL"
        reason = f"negative contamination in top5: {neg_contaminated}"
    elif bm25_dh5 >= 0.80 and bm25_th5 >= 0.70:
        overall = "PASS"
        reason = "all gate cases pass, metrics above threshold"
    elif bm25_dh5 >= 0.65:
        overall = "WARNING"
        reason = f"doc_hit@5={bm25_dh5:.2f} below 0.80 threshold"
    else:
        overall = "FAIL"
        reason = f"doc_hit@5={bm25_dh5:.2f} below 0.65 threshold"

    print(f"  Overall: {overall} — {reason}")
    print(f"  Gate tbl_doc0005_gm_orf2729: {'PASS' if gate_results.get('tbl_doc0005_gm_orf2729') else 'FAIL'}")
    print(f"  Gate fig_doc0005_pathway: {'PASS' if gate_results.get('fig_doc0005_pathway') else 'FAIL'}")
    print(f"  Negative contamination top5: {neg_contaminated}")

    # ── Output ─────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "chunks_jsonl": args.chunks_jsonl,
            "cases_file": args.cases,
            "chunk_count": len(chunks),
            "case_count": len(cases),
            "case_summary": dict(case_summary),
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "overall_verdict": overall,
            "overall_reason": reason,
            "gate_results": gate_results,
            "negative_contamination_top5": neg_contaminated,
        },
        "metrics": all_metrics,
        "case_results": {mode: [dict(c) for c in cases] for mode, cases in all_case_results.items()},
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nJSON: {args.output_json}")

    report_md = render_markdown(all_metrics, all_case_results, dict(case_summary))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report_md)
    print(f"Report: {args.report}")

    review_md = render_manual_review(all_case_results)
    Path(args.manual_review).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_review).write_text(review_md)
    print(f"Manual review: {args.manual_review}")


if __name__ == "__main__":
    main()
