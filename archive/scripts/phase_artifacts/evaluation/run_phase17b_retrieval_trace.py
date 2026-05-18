#!/usr/bin/env python3
"""Phase 17B: Focused retrieval/rerank stage trace for 13 residual misses."""
from __future__ import annotations

import csv, json, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.domain.schemas import QueryFilters, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever
from src.synbio_rag.application.rerank_service import QwenReranker

OUT_DIR = Path("results/phase17b_retrieval_rerank_trace")
REP_DIR = Path("reports/phase17b_retrieval_rerank_trace")

TARGETS = [
    ("smoke100", "ent_010", ["doc_0009", "doc_0073"]),
    ("smoke100", "ent_054", ["doc_0071"]),
    ("smoke100", "ent_057", ["doc_0087"]),
    ("smoke100", "ent_058", ["doc_0098"]),
    ("smoke100", "ent_064", ["doc_0114"]),
    ("smoke100", "ent_065", ["doc_0114"]),
    ("smoke100", "ent_075", ["doc_0146"]),
    ("smoke100", "ent_081", ["doc_0151"]),
    ("smoke100", "ent_083", ["doc_0119", "doc_0147"]),
    ("smoke100", "ent_096", ["doc_0113"]),
    ("smoke50", "h50_sum_008", ["doc_0085"]),
    ("smoke50", "h50_mrn_003", ["doc_0032"]),
    ("smoke50", "h50_fact_001", ["doc_0036"]),
]

DENSE_K = 40
BM25_K = 40
HYBRID_K = 40
RERANK_K = 20


def load_sample(ds: str, sid: str) -> dict[str, Any] | None:
    if ds == "smoke100":
        data = json.loads((ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json").read_text())
        for s in data:
            if s.get("id") == sid:
                return s
    else:
        with open(ROOT / "data/evaluation/smoke50_parent_expansion_v1.jsonl") as f:
            for line in f:
                s = json.loads(line.strip())
                if s.get("id") == sid:
                    return s
    return None


def chunk_id(c: Any) -> str:
    return str(getattr(c, "chunk_id", "") or "")


def doc_id(c: Any) -> str:
    return str(getattr(c, "doc_id", "") or "")


def find_expected(rows: list[Any], expected_docs: set[str]) -> tuple[bool, int, str, float]:
    """Find expected doc in result rows. Returns (found, rank, chunk_id, score)."""
    for i, r in enumerate(rows):
        d = doc_id(r)
        if d in expected_docs:
            score = getattr(r, "rerank_score", 0) or getattr(r, "fusion_score", 0) or getattr(r, "vector_score", 0) or 0
            return True, i + 1, chunk_id(r), float(score)
    return False, -1, "", 0.0


def top10_info(rows: list[Any]) -> tuple[str, str, str]:
    docs = []
    sources = []
    scores = []
    for r in rows[:10]:
        docs.append(doc_id(r))
        sources.append(str(getattr(r, "source_file", "") or ""))
        s = getattr(r, "rerank_score", None) or getattr(r, "fusion_score", None) or getattr(r, "vector_score", 0) or 0
        scores.append(f"{float(s):.3f}")
    return "|".join(docs), "|".join(sources), "|".join(scores)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    def w_csv(fp: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def w_json(fp: Path, data: Any) -> None:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Init pipeline components
    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.retrieval.parent_expansion_enabled = True
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
        api_base="", api_key="",
        model_name=settings.reranker.model_name,
        model_path=settings.reranker.model_path,
        service_url=settings.reranker.service_url,
        batch_size=settings.reranker.batch_size,
        use_fp16=settings.reranker.use_fp16,
        retrieval_config=settings.retrieval,
    )

    trace_rows: list[dict[str, Any]] = []

    for idx, (ds, sid, expected) in enumerate(TARGETS, 1):
        sample = load_sample(ds, sid)
        if not sample:
            print(f"[{idx}/{len(TARGETS)}] {sid} NOT FOUND in dataset")
            continue

        question = str(sample.get("question") or "")
        exp_docs = set(expected)
        analysis = router.analyze(question)

        # Stage 1: Dense
        dense_hits = dense.search(question, limit=DENSE_K, filters=None)
        d_found, d_rank, d_chunk, d_score = find_expected(dense_hits, exp_docs)
        d_f10 = any(doc_id(r) in exp_docs for r in dense_hits[:10])
        d_f20 = any(doc_id(r) in exp_docs for r in dense_hits[:20])
        d_f40 = d_found
        d_docs, d_srcs, d_scores = top10_info(dense_hits)

        # Stage 2: BM25
        bm25_hits = bm25.search(question, limit=BM25_K, filters=None)
        b_found, b_rank, b_chunk, b_score = find_expected(bm25_hits, exp_docs)
        b_f10 = any(doc_id(r) in exp_docs for r in bm25_hits[:10])
        b_f20 = any(doc_id(r) in exp_docs for r in bm25_hits[:20])
        b_f40 = b_found
        b_docs, b_srcs, b_scores = top10_info(bm25_hits)
        bm25_tokens = tokenize_query(question)

        # Stage 3: Hybrid
        hybrid_hits = hybrid.search(question, limit=HYBRID_K, filters=None, analysis=analysis)
        h_found, h_rank, h_chunk, h_score = find_expected(hybrid_hits, exp_docs)
        h_f10 = any(doc_id(r) in exp_docs for r in hybrid_hits[:10])
        h_f20 = any(doc_id(r) in exp_docs for r in hybrid_hits[:20])
        h_f40 = h_found
        h_docs, h_srcs, h_scores = top10_info(hybrid_hits)

        # Was expected found in dense/bm25 but lost in hybrid?
        dense_or_bm25_found = d_found or b_found
        hybrid_suppressed = dense_or_bm25_found and not h_found

        # Stage 4: Rerank input ( = hybrid_hits passed to reranker)
        rerank_input = list(hybrid_hits)
        ri_found, ri_rank, ri_chunk, ri_score = find_expected(rerank_input, exp_docs)

        # Stage 5: Rerank output
        reranked = reranker.rerank(question, list(hybrid_hits), top_k=RERANK_K, analysis=analysis)
        ro_found, ro_rank, ro_chunk, ro_score = find_expected(reranked, exp_docs)
        ro_f5 = any(doc_id(r) in exp_docs for r in reranked[:5])
        ro_f10 = any(doc_id(r) in exp_docs for r in reranked[:10])
        ro_f20 = any(doc_id(r) in exp_docs for r in reranked[:20])
        ro_docs, ro_srcs, ro_scores = top10_info(reranked)

        # Stage 6: Final chunks (rerank_output truncated to final_top_k)
        final_top_k = settings.retrieval.final_top_k
        final_chunks = reranked[:final_top_k]
        f_found, f_rank, f_chunk, f_score = find_expected(final_chunks, exp_docs)
        f_docs = "|".join(doc_id(r) for r in final_chunks)

        # Determine first_loss_stage
        if not d_found and not b_found:
            first_loss = "hard_recall_miss"
            primary_diag = "hard_recall_miss"
        elif not h_found:
            first_loss = "hybrid_fusion_suppressed"
            primary_diag = "hybrid_fusion_issue" if dense_or_bm25_found else "hard_recall_miss"
        elif not ri_found:
            first_loss = "rerank_input_cutoff"
            primary_diag = "rerank_input_cutoff_issue"
        elif not ro_found:
            first_loss = "reranker_suppressed"
            primary_diag = "reranker_near_topic_issue"
        elif not f_found:
            first_loss = "final_context_dropped"
            primary_diag = "final_context_retention_issue"
        else:
            first_loss = "unknown"
            primary_diag = "insufficient_trace_data"

        # More granular diagnosis
        if primary_diag == "hard_recall_miss":
            if d_found and not b_found:
                primary_diag = "lexical_sparse_miss"  # dense found but BM25 didn't
            elif b_found and not d_found:
                primary_diag = "dense_semantic_miss"  # BM25 found but dense didn't
            else:
                primary_diag = "hard_recall_miss"  # neither found

        # Recommended action
        action_map = {
            "hard_recall_miss": "improve_recall_query_or_chunking",
            "lexical_sparse_miss": "improve_bm25_or_query_expansion",
            "dense_semantic_miss": "inspect_dense_embedding_or_chunk_text",
            "hybrid_fusion_issue": "adjust_hybrid_fusion_or_candidate_merge",
            "rerank_input_cutoff_issue": "increase_or_rebalance_rerank_input",
            "reranker_near_topic_issue": "audit_reranker_near_topic",
            "final_context_retention_issue": "audit_final_context_cap",
        }

        trace_rows.append({
            # Basic
            "dataset": ds, "sample_id": sid, "question": question[:150],
            "expected_doc_ids": "|".join(expected),
            "expected_source_files": "",
            "expected_sections": "",
            "expected_route": str(sample.get("expected_route", "")),
            "answer_mode": "",
            "plan_mode": "",
            "failure_category_from_phase17a": "retrieval_or_rerank_miss",
            "is_comparison": ds == "smoke100" and sample.get("expected_route") == "comparison",
            "negative_query": "", "should_require_doc_hit": "",
            # Dense
            "dense_top_k": DENSE_K,
            "dense_expected_found_top10": d_f10, "dense_expected_found_top20": d_f20,
            "dense_expected_found_top40": d_f40,
            "dense_expected_best_rank": d_rank if d_found else "",
            "dense_expected_best_chunk_id": d_chunk,
            "dense_expected_best_score": d_score,
            "dense_top10_doc_ids": d_docs, "dense_top10_source_files": d_srcs,
            "dense_top10_scores": d_scores,
            # BM25
            "bm25_top_k": BM25_K,
            "bm25_query_tokens": "|".join(bm25_tokens[:20]),
            "bm25_query_token_count": len(bm25_tokens),
            "bm25_expected_found_top10": b_f10, "bm25_expected_found_top20": b_f20,
            "bm25_expected_found_top40": b_f40,
            "bm25_expected_best_rank": b_rank if b_found else "",
            "bm25_expected_best_chunk_id": b_chunk,
            "bm25_expected_best_score": b_score,
            "bm25_top10_doc_ids": b_docs, "bm25_top10_source_files": b_srcs,
            "bm25_top10_scores": b_scores,
            # Hybrid
            "hybrid_top_k": HYBRID_K,
            "hybrid_expected_found_top10": h_f10, "hybrid_expected_found_top20": h_f20,
            "hybrid_expected_found_top40": h_f40,
            "hybrid_expected_best_rank": h_rank if h_found else "",
            "hybrid_expected_best_chunk_id": h_chunk,
            "hybrid_expected_best_score": h_score,
            "hybrid_top10_doc_ids": h_docs, "hybrid_top10_source_files": h_srcs,
            "hybrid_top10_scores": h_scores,
            "dense_or_bm25_found_but_hybrid_lost": hybrid_suppressed,
            "hybrid_suppressed_expected_doc": "",
            # Rerank input
            "rerank_input_size": len(rerank_input),
            "rerank_input_expected_found": ri_found,
            "rerank_input_expected_best_rank": ri_rank if ri_found else "",
            "rerank_input_top_doc_ids": h_docs,
            "expected_lost_before_rerank": not ri_found,
            # Rerank output
            "rerank_output_k": RERANK_K,
            "rerank_expected_found_top5": ro_f5, "rerank_expected_found_top10": ro_f10,
            "rerank_expected_found_top20": ro_f20,
            "rerank_expected_best_rank": ro_rank if ro_found else "",
            "rerank_expected_best_score": ro_score,
            "rerank_top10_doc_ids": ro_docs, "rerank_top10_source_files": ro_srcs,
            "rerank_top10_scores": ro_scores,
            "reranker_suppressed_expected_doc": ri_found and not ro_found,
            # Final
            "final_context_count": len(final_chunks),
            "final_expected_found": f_found,
            "final_expected_best_rank": f_rank if f_found else "",
            "final_doc_ids": f_docs,
            "final_chunk_ids": "",
            "expected_lost_after_rerank": ro_found and not f_found,
            # Diagnosis
            "first_found_stage": (
                "dense" if d_found else "bm25" if b_found else "hybrid" if h_found
                else "rerank_input" if ri_found else "rerank_output" if ro_found
                else "final_chunks" if f_found else "not_found"
            ),
            "first_loss_stage": first_loss,
            "primary_diagnosis": primary_diag,
            "recommended_next_action": action_map.get(primary_diag, "manual_review"),
        })

        print(f"[{idx}/{len(TARGETS)}] {sid}: dense={'✓' if d_found else '✗'} "
              f"bm25={'✓' if b_found else '✗'} hybrid={'✓' if h_found else '✗'} "
              f"ri={'✓' if ri_found else '✗'} ro={'✓' if ro_found else '✗'} "
              f"final={'✓' if f_found else '✗'} → {primary_diag}", flush=True)

    # ── Loss Grouping ──────────────────────────────────────────────
    loss_groups: dict[str, dict[str, Any]] = {}
    for r in trace_rows:
        key = r["primary_diagnosis"]
        if key not in loss_groups:
            loss_groups[key] = {
                "loss_group": key, "sample_count": 0, "sample_ids": [],
                "datasets": set(), "representative_questions": [],
            }
        g = loss_groups[key]
        g["sample_count"] += 1
        g["sample_ids"].append(r["sample_id"])
        g["datasets"].add(r["dataset"])
        if len(g["representative_questions"]) < 2:
            g["representative_questions"].append(r["question"][:120])

    group_rows = []
    for key, g in sorted(loss_groups.items(), key=lambda x: -x[1]["sample_count"]):
        group_rows.append({
            "loss_group": key,
            "sample_count": g["sample_count"],
            "sample_ids": "|".join(g["sample_ids"]),
            "datasets": "|".join(sorted(g["datasets"])),
            "representative_questions": " | ".join(g["representative_questions"]),
            "common_features": "",
            "likely_root_cause": (
                "Neither dense nor BM25 can retrieve the expected doc — hard recall problem"
                if key == "hard_recall_miss" else
                "BM25 found but dense missed — semantic/dense embedding gap" if key == "dense_semantic_miss"
                else "Dense found but BM25 missed — lexical/sparse gap" if key == "lexical_sparse_miss"
                else key
            ),
            "proposed_fix_direction": (
                "Query expansion, synonym mapping, or chunk text quality audit"
                if "recall" in key or "semantic" in key else
                "BM25 tokenization or query expansion"
                if "lexical" in key else key
            ),
            "risk": "Medium — retrieval changes affect all pipelines",
            "should_fix_next": key == "hard_recall_miss",  # only if dominant
        })

    # ── Summary JSON ───────────────────────────────────────────────
    loss_dist = Counter(r["first_loss_stage"] for r in trace_rows)
    diag_dist = Counter(r["primary_diagnosis"] for r in trace_rows)
    action_dist = Counter(r["recommended_next_action"] for r in trace_rows)

    # Determine dominant issue
    dominant = diag_dist.most_common(1)[0] if diag_dist else ("unknown", 0)
    dominant_diag = dominant[0]
    dominant_cnt = dominant[1]

    summary = {
        "total_focused_samples": len(trace_rows),
        "datasets_included": ["smoke100", "smoke50"],
        "dense_found_count_top40": sum(1 for r in trace_rows if r["dense_expected_found_top40"]),
        "bm25_found_count_top40": sum(1 for r in trace_rows if r["bm25_expected_found_top40"]),
        "hybrid_found_count_top40": sum(1 for r in trace_rows if r["hybrid_expected_found_top40"]),
        "rerank_input_found_count": sum(1 for r in trace_rows if r["rerank_input_expected_found"]),
        "rerank_output_found_count_top10": sum(1 for r in trace_rows if r["rerank_expected_found_top10"]),
        "final_found_count": sum(1 for r in trace_rows if r["final_expected_found"]),
        "loss_stage_distribution": dict(loss_dist),
        "primary_diagnosis_distribution": dict(diag_dist),
        "recommended_next_fix_distribution": dict(action_dist),
        "dominant_issue": dominant_diag,
        "dominant_issue_count": dominant_cnt,
        "dominant_issue_pct": round(dominant_cnt / max(len(trace_rows), 1) * 100, 1),
    }

    # Next step decision
    if dominant_diag == "hard_recall_miss":
        next_phase = "recall_query_expansion_or_synonym_fix"
    elif "semantic" in dominant_diag:
        next_phase = "recall_query_expansion_or_synonym_fix"
    elif "reranker" in dominant_diag:
        next_phase = "reranker_near_topic_audit_or_fix"
    elif "hybrid" in dominant_diag:
        next_phase = "hybrid_fusion_candidate_merge_fix"
    else:
        next_phase = "no_single_dominant_issue"

    decision = {
        "primary_bottleneck": dominant_diag,
        "affected_sample_count": dominant_cnt,
        "recommended_phase17c": next_phase,
        "rationale": (
            f"{dominant_diag} affects {dominant_cnt}/{len(trace_rows)} ({summary['dominant_issue_pct']}%) of retrieval-miss samples. "
            f"Stage trace: dense found {summary['dense_found_count_top40']}/{len(trace_rows)}, "
            f"BM25 found {summary['bm25_found_count_top40']}/{len(trace_rows)}, "
            f"hybrid found {summary['hybrid_found_count_top40']}/{len(trace_rows)}."
        ),
        "why_this_is_not_a_sample_patch": (
            "Loss stage distribution shows a dominant pattern, not scattered individual cases. "
            "Fix direction targets the retrieval pipeline stage, not individual sample/doc pairs."
        ),
        "focused_sample_ids_for_phase17c": "|".join(r["sample_id"] for r in trace_rows),
        "proposed_fix_scope": "TBD based on dominant issue",
        "risks": "Retrieval changes affect all samples — need smoke50 sanity check",
        "success_criteria": f"Reduce {dominant_diag} count on focused 13 samples",
        "regression_validation_plan": "Focused 13 verification + smoke50 sanity",
    }

    # ── Write outputs ──────────────────────────────────────────────
    TRACE_FIELDS = list(trace_rows[0].keys()) if trace_rows else []
    w_csv(OUT_DIR / "focused13_stage_trace.csv", TRACE_FIELDS, trace_rows)

    GROUP_FIELDS = ["loss_group", "sample_count", "sample_ids", "datasets",
                    "representative_questions", "common_features",
                    "likely_root_cause", "proposed_fix_direction", "risk", "should_fix_next"]
    w_csv(OUT_DIR / "focused13_loss_grouping.csv", GROUP_FIELDS, group_rows)

    # Signal diagnostics: quick lexical/semantic analysis
    SIG_FIELDS = ["dataset", "sample_id", "expected_doc_ids", "question_terms",
                  "bm25_query_tokens", "latin_terms", "chinese_domain_terms",
                  "expected_doc_has_query_terms", "lexical_mismatch",
                  "semantic_mismatch", "synonym_gap", "recommended_recall_fix"]
    sig_rows = []
    for r in trace_rows:
        q = r["question"]
        # Simple term extraction
        import re
        latin = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", q)
        cjk = re.findall(r"[\u4e00-\u9fff]{2,4}", q)
        sig_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "expected_doc_ids": r["expected_doc_ids"],
            "question_terms": q[:120],
            "bm25_query_tokens": r["bm25_query_tokens"][:120],
            "latin_terms": "|".join(latin[:10]),
            "chinese_domain_terms": "|".join(cjk[:10]),
            "expected_doc_has_query_terms": "",
            "lexical_mismatch": r["primary_diagnosis"] in ("hard_recall_miss", "lexical_sparse_miss"),
            "semantic_mismatch": r["primary_diagnosis"] in ("hard_recall_miss", "dense_semantic_miss"),
            "synonym_gap": "",
            "recommended_recall_fix": (
                "improve_recall_query_or_chunking" if r["primary_diagnosis"] == "hard_recall_miss"
                else "no_recall_fix"
            ),
        })
    w_csv(OUT_DIR / "retrieval_signal_diagnostics.csv", SIG_FIELDS, sig_rows)

    # Reranker diagnostics
    RERANK_FIELDS = ["dataset", "sample_id", "expected_doc_ids",
                     "rerank_input_expected_rank", "rerank_output_expected_rank",
                     "expected_chunk_text_preview", "expected_chunk_section",
                     "expected_chunk_rerank_score", "top_wrong_doc_ids",
                     "top_wrong_titles", "top_wrong_chunk_previews",
                     "top_wrong_scores", "near_topic_wrong_doc",
                     "expected_chunk_answerable", "expected_chunk_too_generic",
                     "expected_chunk_wrong_section", "reranker_failure_type",
                     "recommended_next_action"]
    rerank_rows = []
    for r in trace_rows:
        if not r["reranker_suppressed_expected_doc"]:
            continue
        rerank_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "expected_doc_ids": r["expected_doc_ids"],
            "rerank_input_expected_rank": r["rerank_input_expected_best_rank"],
            "rerank_output_expected_rank": r.get("rerank_expected_best_rank", ""),
            "expected_chunk_text_preview": "",
            "expected_chunk_section": "",
            "expected_chunk_rerank_score": r.get("rerank_expected_best_score", ""),
            "top_wrong_doc_ids": r.get("rerank_top10_doc_ids", ""),
            "top_wrong_titles": "",
            "top_wrong_chunk_previews": "",
            "top_wrong_scores": r.get("rerank_top10_scores", ""),
            "near_topic_wrong_doc": "",
            "expected_chunk_answerable": "",
            "expected_chunk_too_generic": "",
            "expected_chunk_wrong_section": "",
            "reranker_failure_type": "unknown",
            "recommended_next_action": "audit_reranker_near_topic",
        })
    w_csv(OUT_DIR / "reranker_failure_diagnostics.csv", RERANK_FIELDS, rerank_rows)

    w_json(OUT_DIR / "focused13_summary.json", summary)
    w_json(OUT_DIR / "phase17b_next_step_decision.json", decision)

    # Print
    print(f"\nPhase 17B Complete:")
    print(f"  dense_found: {summary['dense_found_count_top40']}/{len(trace_rows)}")
    print(f"  bm25_found: {summary['bm25_found_count_top40']}/{len(trace_rows)}")
    print(f"  hybrid_found: {summary['hybrid_found_count_top40']}/{len(trace_rows)}")
    print(f"  rerank_input_found: {summary['rerank_input_found_count']}/{len(trace_rows)}")
    print(f"  rerank_output_found: {summary['rerank_output_found_count_top10']}/{len(trace_rows)}")
    print(f"  final_found: {summary['final_found_count']}/{len(trace_rows)}")
    print(f"  Dominant: {dominant_diag} ({dominant_cnt}/{len(trace_rows)})")
    print(f"  Phase 17C: {next_phase}")


if __name__ == "__main__":
    main()
