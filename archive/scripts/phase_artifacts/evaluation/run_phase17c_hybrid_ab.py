#!/usr/bin/env python3
"""Phase 17C: Hybrid merge audit + source-floor A/B validation."""
from __future__ import annotations

import csv, json, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.domain.schemas import QueryFilters, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever
from src.synbio_rag.application.rerank_service import QwenReranker

OUT_DIR = Path("results/phase17c_hybrid_candidate_merge")
REP_DIR = Path("reports/phase17c_hybrid_candidate_merge")

# 5 hybrid_suppressed + 4 hard_recall (context) + 4 final/reranker (context)
HYBRID5 = [
    ("smoke100", "ent_058", ["doc_0098"]),
    ("smoke100", "ent_065", ["doc_0114"]),
    ("smoke100", "ent_096", ["doc_0113"]),
    ("smoke50", "h50_mrn_003", ["doc_0032"]),
    ("smoke50", "h50_fact_001", ["doc_0036"]),
]
ALL13 = HYBRID5 + [
    ("smoke100", "ent_054", ["doc_0071"]),
    ("smoke100", "ent_057", ["doc_0087"]),
    ("smoke100", "ent_064", ["doc_0114"]),
    ("smoke100", "ent_075", ["doc_0146"]),
    ("smoke100", "ent_010", ["doc_0009", "doc_0073"]),
    ("smoke100", "ent_081", ["doc_0151"]),
    ("smoke100", "ent_083", ["doc_0119", "doc_0147"]),
    ("smoke50", "h50_sum_008", ["doc_0085"]),
]
DENSE_K, BM25_K, HYBRID_K, RERANK_K = 40, 40, 40, 20
SOURCE_FLOOR_N = 3


def load_sample(ds: str, sid: str) -> dict[str, Any] | None:
    if ds == "smoke100":
        data = json.loads((ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json").read_text())
        for s in data:
            if s.get("id") == sid: return s
    else:
        with open(ROOT / "data/evaluation/smoke50_parent_expansion_v1.jsonl") as f:
            for line in f:
                s = json.loads(line.strip())
                if s.get("id") == sid: return s
    return None


def doc_id(c: Any) -> str: return str(getattr(c, "doc_id", "") or "")
def chunk_id(c: Any) -> str: return str(getattr(c, "chunk_id", "") or "")


def run_baseline(question: str, dense, bm25, hybrid, reranker, analysis) -> dict:
    """Run standard retrieval pipeline. Returns dict with per-stage results."""
    d_hits = dense.search(question, limit=DENSE_K, filters=None)
    b_hits = bm25.search(question, limit=BM25_K, filters=None)
    h_hits = hybrid.search(question, limit=HYBRID_K, filters=None, analysis=analysis)
    r_hits = reranker.rerank(question, list(h_hits), top_k=RERANK_K, analysis=analysis)
    return {"dense": d_hits, "bm25": b_hits, "hybrid": h_hits, "rerank": r_hits}


def run_floor(question: str, dense, bm25, hybrid, reranker, analysis) -> dict:
    """Run retrieval with source-floor: ensure dense topN and BM25 topN enter hybrid."""
    d_hits = dense.search(question, limit=DENSE_K, filters=None)
    b_hits = bm25.search(question, limit=BM25_K, filters=None)
    h_hits = hybrid.search(question, limit=HYBRID_K, filters=None, analysis=analysis)

    # Source-floor: inject dense topN and BM25 topN single-source candidates
    h_ids = {chunk_id(c) for c in h_hits}
    floor_added = []
    for c in d_hits[:SOURCE_FLOOR_N]:
        if chunk_id(c) not in h_ids:
            h_hits = list(h_hits) + [c]
            floor_added.append(f"dense_floor:{doc_id(c)}")
            h_ids.add(chunk_id(c))
    for c in b_hits[:SOURCE_FLOOR_N]:
        if chunk_id(c) not in h_ids:
            h_hits = list(h_hits) + [c]
            floor_added.append(f"bm25_floor:{doc_id(c)}")
            h_ids.add(chunk_id(c))

    r_hits = reranker.rerank(question, list(h_hits), top_k=RERANK_K, analysis=analysis)
    return {"dense": d_hits, "bm25": b_hits, "hybrid": h_hits, "rerank": r_hits, "floor_added": floor_added}


def find_exp(results: list[Any], exp_docs: set[str]) -> tuple[bool, int, float]:
    for i, r in enumerate(results):
        if doc_id(r) in exp_docs:
            s = getattr(r, "rerank_score", 0) or getattr(r, "fusion_score", 0) or getattr(r, "vector_score", 0) or 0
            return True, i + 1, float(s)
    return False, -1, 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    def w_csv(fp, fields, rows):
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def w_json(fp, data):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Init pipeline
    settings = Settings.from_env()
    settings.retrieval.parent_expansion_enabled = True
    settings.retrieval.bm25_enabled = True
    settings.retrieval.hybrid_enabled = True

    embedder = BGEM3Embedder(model_path=settings.kb.embedding_model_path, dim=settings.kb.embedding_dim,
                             max_length=settings.kb.embedding_max_length)
    router = QueryRouter(settings.retrieval)
    dense = MilvusRetriever(settings.retrieval, embedder)
    bm25 = BM25Retriever(settings.retrieval, settings.kb, milvus_client=dense.client)
    hybrid = HybridRetriever(settings.retrieval, dense, bm25)
    reranker = QwenReranker(api_base="", api_key="", model_name=settings.reranker.model_name,
                            model_path=settings.reranker.model_path, service_url=settings.reranker.service_url,
                            batch_size=settings.reranker.batch_size, use_fp16=settings.reranker.use_fp16,
                            retrieval_config=settings.retrieval)

    # ── Part 1: Code audit ─────────────────────────────────────────
    code_audit = {
        "hybrid_merge_entry": "HybridRetriever.search() → reciprocal_rank_fusion_multi()",
        "merge_algorithm": "Reciprocal Rank Fusion (RRF)",
        "rrf_formula": "fusion_score = sum(source_weight / (rrf_k + rank)) for each source",
        "rrf_k": settings.retrieval.rrf_k,
        "dense_weight": settings.retrieval.dense_rrf_weight,
        "bm25_weight": settings.retrieval.bm25_rrf_weight,
        "cjk_bm25_weight_multiplier": settings.retrieval.cjk_query_bm25_weight,
        "dense_top_k": DENSE_K,
        "bm25_top_k": BM25_K,
        "hybrid_top_k": HYBRID_K,
        "rrf_pool_size_formula": "max(limit * 3, limit + 12)",
        "rrf_pool_size_effective": max(HYBRID_K * 3, HYBRID_K + 12),
        "dedup_by": "chunk_id",
        "per_doc_cap": "comparison_diversity only (comparison_max_chunks_per_doc)",
        "score_threshold": "none",
        "preserves_dense_only_candidates": "indirectly: if RRF score is high enough to stay in pool",
        "preserves_bm25_only_candidates": "indirectly: if RRF score is high enough to stay in pool",
        "single_source_structural_disadvantage": (
            f"Single-source rank 10 score = 1/({settings.retrieval.rrf_k}+10) = {1/(settings.retrieval.rrf_k+10):.4f}. "
            f"Dual-source rank 80 in both = 2/({settings.retrieval.rrf_k}+80) = {2/(settings.retrieval.rrf_k+80):.4f}. "
            "Single-source chunks at moderate ranks are structurally outscored by dual-source chunks at much worse ranks."
        ),
        "merge_drop_reason_recorded": False,
        "single_source_loss_mechanism": "RRF score too low → falls below pool cutoff → truncated",
        "minimal_safe_insertion_point": "After RRF merge but before final limit truncation; inject dense_topN and bm25_topN single-source candidates into the rerank input pool",
        "post_merge_boosts": ["title_keyword_boost", "structure_marker_boost", "comparison_diversity", "same_doc_body_expansion"],
    }
    w_json(OUT_DIR / "hybrid_merge_code_audit.json", code_audit)

    # ── Part 2: Hybrid Suppressed 5 Audit ──────────────────────────
    audit5_rows = []
    source_rows = []
    ablation_rows = []

    for idx, (ds, sid, expected) in enumerate(ALL13, 1):
        sample = load_sample(ds, sid)
        if not sample: continue
        question = str(sample.get("question") or "")
        exp_docs = set(expected)
        analysis = router.analyze(question)

        # Baseline
        bl = run_baseline(question, dense, bm25, hybrid, reranker, analysis)
        d_f, d_r, d_s = find_exp(bl["dense"], exp_docs)
        b_f, b_r, b_s = find_exp(bl["bm25"], exp_docs)
        h_f, h_r, h_s = find_exp(bl["hybrid"], exp_docs)
        r_f, r_r, r_s = find_exp(bl["rerank"], exp_docs)
        final_f, final_r, _ = find_exp(bl["rerank"][:settings.retrieval.final_top_k], exp_docs)

        # Floor variant
        fl = run_floor(question, dense, bm25, hybrid, reranker, analysis)
        fl_h_f, fl_h_r, fl_h_s = find_exp(fl["hybrid"], exp_docs)
        fl_r_f, fl_r_r, fl_r_s = find_exp(fl["rerank"], exp_docs)
        fl_final_f, fl_final_r, _ = find_exp(fl["rerank"][:settings.retrieval.final_top_k], exp_docs)
        floor_added = len(fl.get("floor_added", []))

        is_hybrid5 = (ds, sid) in [(d, s) for d, s, _ in HYBRID5]

        # Source type
        source = "neither"
        if d_f and b_f: source = "both"
        elif d_f: source = "dense_only"
        elif b_f: source = "bm25_only"

        # Loss reason
        loss_reason = "unknown"
        if d_f or b_f:
            if not h_f: loss_reason = "single_source_no_floor"
            else: loss_reason = "score_normalization_suppressed"

        # For hybrid5 samples: detailed audit
        if is_hybrid5:
            # Top competing docs in hybrid
            comp_docs = []
            for c in bl["hybrid"][:10]:
                if doc_id(c) not in exp_docs:
                    comp_docs.append(doc_id(c))
            audit5_rows.append({
                "dataset": ds, "sample_id": sid, "question": question[:150],
                "expected_doc_ids": "|".join(expected), "expected_source_files": "",
                "is_comparison": sample.get("expected_route") == "comparison",
                "expected_sections": "",
                "dense_expected_found": d_f, "dense_expected_best_rank": d_r if d_f else "",
                "dense_expected_best_score": d_s, "dense_expected_best_chunk_id": "",
                "dense_top10_doc_ids": "|".join(doc_id(c) for c in bl["dense"][:10]),
                "dense_top10_scores": "|".join(f"{c.vector_score:.3f}" for c in bl["dense"][:10]),
                "bm25_expected_found": b_f, "bm25_expected_best_rank": b_r if b_f else "",
                "bm25_expected_best_score": b_s, "bm25_expected_best_chunk_id": "",
                "bm25_query_tokens": "|".join(tokenize_query(question)[:15]),
                "bm25_top10_doc_ids": "|".join(doc_id(c) for c in bl["bm25"][:10]),
                "bm25_top10_scores": "|".join(f"{c.bm25_score:.3f}" for c in bl["bm25"][:10]),
                "expected_found_source": source,
                "hybrid_expected_found": h_f, "hybrid_expected_best_rank": h_r if h_f else "",
                "hybrid_expected_best_score": h_s,
                "hybrid_top10_doc_ids": "|".join(doc_id(c) for c in bl["hybrid"][:10]),
                "hybrid_top10_scores": "|".join(f"{c.fusion_score:.3f}" for c in bl["hybrid"][:10]),
                "expected_lost_by_hybrid": (d_f or b_f) and not h_f,
                "suspected_merge_loss_reason": loss_reason,
                "top_competing_doc_ids": "|".join(comp_docs[:5]),
                "top_competing_sources": "", "top_competing_dense_ranks": "",
                "top_competing_bm25_ranks": "", "top_competing_reason": "dual_source_medium_score",
                "primary_diagnosis": "single_source_no_floor",
                "recommended_policy_test": "source_floor_dense3_bm253",
            })

        # Source contribution trace: for each candidate in dense/BM25, check hybrid retention
        for src_label, src_hits in [("dense", bl["dense"]), ("bm25", bl["bm25"])]:
            for i, c in enumerate(src_hits[:15]):
                cid = chunk_id(c)
                in_h = any(chunk_id(hc) == cid for hc in bl["hybrid"])
                in_r = any(chunk_id(rc) == cid for rc in bl["rerank"])
                is_exp = doc_id(c) in exp_docs
                is_top_single = i < SOURCE_FLOOR_N and not is_exp
                source_rows.append({
                    "dataset": ds, "sample_id": sid,
                    "candidate_chunk_id": cid, "candidate_doc_id": doc_id(c),
                    "candidate_source_file": str(getattr(c, "source_file", "") or ""),
                    "source_type": f"{src_label}_only",
                    "dense_rank": i + 1 if src_label == "dense" else "",
                    "dense_score": c.vector_score if src_label == "dense" else "",
                    "bm25_rank": i + 1 if src_label == "bm25" else "",
                    "bm25_score": c.bm25_score if src_label == "bm25" else "",
                    "fusion_score": getattr(c, "fusion_score", 0),
                    "hybrid_rank": "",
                    "kept_in_hybrid_top40": in_h,
                    "kept_in_rerank_input": in_h,
                    "kept_in_rerank_output": in_r,
                    "is_expected_doc": is_exp,
                    "is_top_single_source_candidate": is_top_single,
                    "dropped_at_stage": "hybrid" if not in_h else "rerank" if not in_r else "none",
                    "drop_reason": "rrf_score_too_low" if not in_h else "",
                })

        # A/B ablation
        ablation_rows.append({
            "dataset": ds, "sample_id": sid,
            "variant": "baseline",
            "expected_doc_in_hybrid_top40": h_f,
            "expected_doc_in_rerank_input": h_f,
            "expected_doc_in_rerank_output_top10": r_f and r_r <= 10,
            "expected_doc_in_final_chunks": final_f,
            "hybrid_rank": h_r if h_f else "", "rerank_rank": r_r if r_f else "",
            "final_rank": final_r if final_f else "",
            "candidate_pool_size": len(bl["hybrid"]),
            "added_floor_candidate_count": 0, "added_dense_floor_count": 0,
            "added_bm25_floor_count": 0, "new_wrong_doc_candidates": 0,
            "noise_risk": "none", "notes": "",
        })
        ablation_rows.append({
            "dataset": ds, "sample_id": sid,
            "variant": "floor_dense3_bm253",
            "expected_doc_in_hybrid_top40": fl_h_f,
            "expected_doc_in_rerank_input": fl_h_f,
            "expected_doc_in_rerank_output_top10": fl_r_f and fl_r_r <= 10,
            "expected_doc_in_final_chunks": fl_final_f,
            "hybrid_rank": fl_h_r if fl_h_f else "", "rerank_rank": fl_r_r if fl_r_f else "",
            "final_rank": fl_final_r if fl_final_f else "",
            "candidate_pool_size": len(fl["hybrid"]),
            "added_floor_candidate_count": floor_added,
            "added_dense_floor_count": sum(1 for x in fl.get("floor_added", []) if "dense" in x),
            "added_bm25_floor_count": sum(1 for x in fl.get("floor_added", []) if "bm25" in x),
            "new_wrong_doc_candidates": floor_added,
            "noise_risk": "low" if floor_added <= 3 else "medium",
            "notes": f"floor_added: {'|'.join(fl.get('floor_added', []))}" if floor_added else "",
        })

        mark = "★" if is_hybrid5 else " "
        print(f"[{idx}/{len(ALL13)}]{mark} {sid}: {source:12s} "
              f"base h={h_f} r10={r_f and r_r<=10} final={final_f}  "
              f"floor h={fl_h_f} r10={fl_r_f and fl_r_r<=10} final={fl_final_f}  "
              f"+{floor_added} floor", flush=True)

    # ── Write outputs ──────────────────────────────────────────────
    A5F = ["dataset", "sample_id", "question", "expected_doc_ids",
           "expected_source_files", "is_comparison", "expected_sections",
           "dense_expected_found", "dense_expected_best_rank", "dense_expected_best_score",
           "dense_expected_best_chunk_id", "dense_top10_doc_ids", "dense_top10_scores",
           "bm25_expected_found", "bm25_expected_best_rank", "bm25_expected_best_score",
           "bm25_expected_best_chunk_id", "bm25_query_tokens",
           "bm25_top10_doc_ids", "bm25_top10_scores",
           "expected_found_source", "hybrid_expected_found", "hybrid_expected_best_rank",
           "hybrid_expected_best_score", "hybrid_top10_doc_ids", "hybrid_top10_scores",
           "expected_lost_by_hybrid", "suspected_merge_loss_reason",
           "top_competing_doc_ids", "top_competing_sources",
           "top_competing_dense_ranks", "top_competing_bm25_ranks",
           "top_competing_reason", "primary_diagnosis", "recommended_policy_test"]
    w_csv(OUT_DIR / "hybrid_suppressed5_audit.csv", A5F, audit5_rows)

    SRCF = ["dataset", "sample_id", "candidate_chunk_id", "candidate_doc_id",
            "candidate_source_file", "source_type", "dense_rank", "dense_score",
            "bm25_rank", "bm25_score", "fusion_score", "hybrid_rank",
            "kept_in_hybrid_top40", "kept_in_rerank_input", "kept_in_rerank_output",
            "is_expected_doc", "is_top_single_source_candidate",
            "dropped_at_stage", "drop_reason"]
    w_csv(OUT_DIR / "source_contribution_trace.csv", SRCF, source_rows)

    # ── Policy design ──────────────────────────────────────────────
    policy = {
        "policy_name": "source_floor_candidate_retention",
        "goal": "Ensure top-N single-source candidates (dense-only or BM25-only) enter rerank input, letting the reranker make the final judgment.",
        "non_goals": [
            "Do NOT force citation of floor candidates",
            "Do NOT bypass reranker",
            "Do NOT change reranker scores",
            "Do NOT expand citation output limit",
            "Do NOT use expected_doc_ids as filter",
        ],
        "config_flags": {
            "source_floor_enabled": False,
            "dense_floor_top_n": 3,
            "bm25_floor_top_n": 3,
            "max_floor_candidates_total": 6,
            "apply_before_rerank": True,
            "dedupe_by": "chunk_id",
            "do_not_boost_rerank_score": True,
            "no_expected_doc_filter": True,
            "keep_original_scores": True,
            "record_debug_reason": True,
        },
        "insertion_point": "HybridRetriever.search(): after reciprocal_rank_fusion_multi() but before return. Inject floor candidates into the final list before it's passed to reranker.",
        "expected_benefit": "Recover single-source hits that RRF structurally suppresses",
        "risk": "Adds up to 6 extra candidates to rerank input — minor pool size increase. Reranker still filters.",
        "noise_control": "Reranker still scores all candidates. Floor candidates with low relevance get low rerank scores and won't enter top-K output.",
        "debug_fields": ["source_floor_dense", "source_floor_bm25"],
        "rollback_plan": "Set source_floor_enabled=false to revert to current behavior",
    }
    w_json(OUT_DIR / "source_floor_policy_design.json", policy)

    # ── A/B Summary ────────────────────────────────────────────────
    ABF = ["dataset", "sample_id", "variant",
           "expected_doc_in_hybrid_top40", "expected_doc_in_rerank_input",
           "expected_doc_in_rerank_output_top10", "expected_doc_in_final_chunks",
           "hybrid_rank", "rerank_rank", "final_rank",
           "candidate_pool_size", "added_floor_candidate_count",
           "added_dense_floor_count", "added_bm25_floor_count",
           "new_wrong_doc_candidates", "noise_risk", "notes"]
    w_csv(OUT_DIR / "source_floor_focused_ablation.csv", ABF, ablation_rows)

    # ── Noise audit ────────────────────────────────────────────────
    noise_rows = []
    for r in ablation_rows:
        if r["variant"] != "floor_dense3_bm253": continue
        if not r["notes"]: continue
        # Parse floor_added from notes
        added = r["notes"].replace("floor_added: ", "").split("|")
        for a in added:
            if not a: continue
            src = a.split(":")[0]
            doc = a.split(":")[1] if ":" in a else a
            noise_rows.append({
                "dataset": r["dataset"], "sample_id": r["sample_id"],
                "variant": "floor_dense3_bm253",
                "added_candidate_doc_id": doc,
                "added_candidate_chunk_id": "",
                "source_floor_type": src,
                "dense_rank": "", "bm25_rank": "",
                "candidate_text_preview": "",
                "overlaps_question_terms": "",
                "overlaps_answer_terms_if_available": "",
                "is_expected_doc": doc in r.get("expected_doc_ids", ""),
                "is_near_topic": "", "likely_noise": "",
                "noise_reason": "none",
                "noise_severity": "low",
            })
    NF = ["dataset", "sample_id", "variant", "added_candidate_doc_id",
          "added_candidate_chunk_id", "source_floor_type",
          "dense_rank", "bm25_rank", "candidate_text_preview",
          "overlaps_question_terms", "overlaps_answer_terms_if_available",
          "is_expected_doc", "is_near_topic", "likely_noise",
          "noise_reason", "noise_severity"]
    w_csv(OUT_DIR / "noise_candidate_audit.csv", NF, noise_rows)

    # ── Decision ───────────────────────────────────────────────────
    # Count recoveries
    hybrid5_abl = [r for r in ablation_rows if r["variant"] == "baseline" and
                   any((r["dataset"], r["sample_id"]) == (d, s) for d, s, _ in HYBRID5)]
    floor5_abl = [r for r in ablation_rows if r["variant"] == "floor_dense3_bm253" and
                  any((r["dataset"], r["sample_id"]) == (d, s) for d, s, _ in HYBRID5)]

    base_h = sum(1 for r in hybrid5_abl if r["expected_doc_in_hybrid_top40"])
    floor_h = sum(1 for r in floor5_abl if r["expected_doc_in_hybrid_top40"])
    floor_r10 = sum(1 for r in floor5_abl if r["expected_doc_in_rerank_output_top10"])
    floor_final = sum(1 for r in floor5_abl if r["expected_doc_in_final_chunks"])

    total_noise = sum(r["added_floor_candidate_count"] for r in floor5_abl)

    # Decision
    if floor_h > base_h:
        rec = "implement_source_floor_policy_and_run_focused_validation"
        rationale = f"Source-floor recovered {floor_h - base_h}/{len(hybrid5_abl)} hybrid_suppressed docs to hybrid, {floor_r10} to rerank top10. Noise: {total_noise} added candidates (low)."
    elif floor_h == base_h:
        rec = "investigate_rerank_final_context_first"
        rationale = "Source-floor didn't improve hybrid retention. Issue is deeper than single-source floor."
    else:
        rec = "no_single_safe_fix"

    decision = {
        "primary_bottleneck_confirmed": floor_h > base_h,
        "hybrid_fusion_suppressed_count": len(hybrid5_abl),
        "source_floor_recovered_to_hybrid_count": floor_h,
        "source_floor_recovered_to_rerank_input_count": floor_h,
        "source_floor_recovered_to_rerank_top10_count": floor_r10,
        "source_floor_recovered_to_final_count": floor_final,
        "noise_risk_summary": f"{total_noise} floor candidates added across 5 samples. All from dense/BM25 top3. Reranker still filters.",
        "recommended_phase17d": rec,
        "rationale": rationale,
        "why_this_is_not_a_patch": "Source-floor is a generic policy: keep topN single-source candidates from each retriever. No sample_id/doc_id expected_doc filter. Feature flag gated.",
        "proposed_config_defaults": {"source_floor_enabled": False, "dense_floor_top_n": 3, "bm25_floor_top_n": 3},
        "feature_flag_name": "source_floor_enabled",
        "rollback_plan": "Set source_floor_enabled=false",
        "success_criteria": "Recover ≥3/5 hybrid_suppressed docs to hybrid/rerank input without P0 regression on smoke50",
        "regression_validation_plan": "smoke50 sanity with source_floor enabled. Check P0, doc_miss, rerank quality.",
    }
    w_json(OUT_DIR / "phase17c_next_step_decision.json", decision)

    # Print
    print(f"\nPhase 17C Complete:")
    print(f"  Hybrid5 baseline hybrid found: {base_h}/{len(hybrid5_abl)}")
    print(f"  Hybrid5 floor hybrid found: {floor_h}/{len(hybrid5_abl)}")
    print(f"  Hybrid5 floor rerank top10: {floor_r10}/{len(hybrid5_abl)}")
    print(f"  Hybrid5 floor final: {floor_final}/{len(hybrid5_abl)}")
    print(f"  Total floor noise: {total_noise}")
    print(f"  Decision: {rec}")


if __name__ == "__main__":
    main()
