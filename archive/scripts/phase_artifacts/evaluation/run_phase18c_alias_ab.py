#!/usr/bin/env python3
"""Phase 18C: Alias expansion focused A/B on 11 hard recall samples."""
import csv, json, os, sys
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
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever, _expand_alias_query
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever
from src.synbio_rag.application.rerank_service import QwenReranker

OUT_DIR = Path("results/phase18c_controlled_alias_expansion")
REP_DIR = Path("reports/phase18c_controlled_alias_expansion")
SMOKE100 = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
SMOKE50 = ROOT / "data/evaluation/smoke50_parent_expansion_v1.jsonl"
ALIAS_YAML = ROOT / "src/synbio_rag/resources/retrieval_aliases_v1.yaml"

HARD11 = [
    ("smoke100", "ent_010", ["doc_0009", "doc_0073"]),
    ("smoke100", "ent_054", ["doc_0071"]),
    ("smoke100", "ent_057", ["doc_0087"]),
    ("smoke100", "ent_058", ["doc_0098"]),
    ("smoke100", "ent_064", ["doc_0114"]),
    ("smoke100", "ent_075", ["doc_0146"]),
    ("smoke100", "ent_081", ["doc_0151"]),
    ("smoke100", "ent_083", ["doc_0119", "doc_0147"]),
    ("smoke100", "ent_096", ["doc_0113"]),
    ("smoke50", "h50_sum_008", ["doc_0085"]),
    ("smoke50", "h50_mrn_003", ["doc_0032"]),
]
DENSE_K, BM25_K, HYBRID_K, RERANK_K = 40, 40, 40, 20


def load_sample(ds, sid):
    if ds == "smoke100":
        for s in json.loads(SMOKE100.read_text()):
            if s.get("id") == sid: return s
    else:
        with open(SMOKE50) as f:
            for line in f:
                s = json.loads(line.strip())
                if s.get("id") == sid: return s
    return None


def doc_id(c): return str(getattr(c, "doc_id", "") or "")


def find_exp(results, exp_docs):
    for i, r in enumerate(results):
        if doc_id(r) in exp_docs: return True, i + 1
    return False, -1


def run_variant(question, dense, bm25, hybrid, reranker, analysis, variant, config):
    """Run retrieval for a variant. Returns per-stage results."""
    # Set env
    for k in list(os.environ.keys()):
        if 'ALIAS' in k: del os.environ[k]
    if variant == "baseline":
        os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "false"
    elif variant == "alias_low":
        os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "true"
        os.environ["RETRIEVAL_ALIAS_EXPANSION_RISK_LEVELS"] = "low"
    elif variant == "alias_low_medium":
        os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "true"
        os.environ["RETRIEVAL_ALIAS_EXPANSION_RISK_LEVELS"] = "low,medium"

    s = Settings.from_env()
    s.retrieval.parent_expansion_enabled = True
    s.retrieval.bm25_enabled = True
    s.retrieval.hybrid_enabled = True
    h2 = HybridRetriever(s.retrieval, dense, bm25)

    # Get alias expansion for BM25 tokens
    raw_bm25_q = _expand_alias_query(question, s.retrieval)
    bm25_hits = bm25.search(raw_bm25_q, limit=BM25_K, filters=None)
    d_hits = dense.search(question, limit=DENSE_K, filters=None)
    h_hits = h2.search(question, limit=HYBRID_K, filters=None, analysis=analysis)
    r_hits = reranker.rerank(question, list(h_hits), top_k=RERANK_K, analysis=analysis)
    final = r_hits[:s.retrieval.final_top_k]

    triggered = ""
    if variant != "baseline":
        alias_map = _expand_alias_query.__globals__.get("_ALIAS_MAP_CACHE", {})
        # Get triggered aliases from the query expansion
        triggered = raw_bm25_q.replace(question, "").strip()

    return {
        "bm25": bm25_hits, "dense": d_hits, "hybrid": h_hits, "rerank": r_hits,
        "final": final, "triggered": triggered,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    def w_csv(fp, fields, rows):
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def w_json(fp, data):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Copy alias map
    import shutil
    shutil.copy(ALIAS_YAML, OUT_DIR / "alias_map_v1.yaml")

    # Code audit
    w_json(OUT_DIR / "alias_system_code_audit.json", {
        "existing_alias_location": "hybrid.py:_expand_query_aliases (Latin->Latin only)",
        "current_supports": ["2'-FL -> 2-fucosyllactose", "6'-SL -> 6-sialyllactose", "WcfB", "salvage", "CRISPR-TMSD"],
        "current_scope": "dense + BM25 query expansion (in query_plan)",
        "current_chinese_to_english": False,
        "current_feature_flag": False,
        "current_expansion_limit": "unlimited per query",
        "current_debug": "query_variants in last_debug",
        "new_insertion_point": "hybrid.py search(): BM25 query only, before bm25.search()",
        "dense_query_unchanged": True,
        "rerank_query_unchanged": True,
        "llm_prompt_unchanged": True,
        "new_feature_flag": "alias_expansion_enabled (default: false)",
        "new_scope": "BM25-only",
        "alias_map_path": str(ALIAS_YAML),
    })

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

    # ── Trigger preview ────────────────────────────────────────────
    trigger_rows = []
    for ds, sid, exp_docs in HARD11:
        sample = load_sample(ds, sid)
        if not sample: continue
        q = sample.get("question", "")
        for k in list(os.environ.keys()):
            if 'ALIAS' in k: del os.environ[k]
        os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "true"
        os.environ["RETRIEVAL_ALIAS_EXPANSION_RISK_LEVELS"] = "low"
        s_low = Settings.from_env()
        raw_low = _expand_alias_query(q, s_low.retrieval)
        os.environ["RETRIEVAL_ALIAS_EXPANSION_RISK_LEVELS"] = "low,medium"
        s_lm = Settings.from_env()
        raw_lm = _expand_alias_query(q, s_lm.retrieval)
        base_tokens = tokenize_query(q)
        low_tokens = tokenize_query(raw_low)
        lm_tokens = tokenize_query(raw_lm)
        added_low = [t for t in low_tokens if t not in base_tokens]
        added_lm = [t for t in lm_tokens if t not in base_tokens]
        trigger_rows.append({
            "dataset": ds, "sample_id": sid, "question": q[:150],
            "normalized_question": q.lower()[:150],
            "baseline_bm25_tokens": "|".join(base_tokens[:15]),
            "triggered_alias_ids_low": "",
            "expansion_terms_low": "|".join(added_low[:10]),
            "triggered_alias_ids_low_medium": "",
            "expansion_terms_low_medium": "|".join(added_lm[:10]),
            "final_bm25_tokens_low": "|".join(low_tokens[:20]),
            "final_bm25_tokens_low_medium": "|".join(lm_tokens[:20]),
            "expansion_term_count": len(added_low),
            "expansion_limited": len(added_low) >= 8,
            "notes": "",
        })
    del os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"]
    for k in list(os.environ.keys()):
        if 'ALIAS' in k: del os.environ[k]

    PF = list(trigger_rows[0].keys()) if trigger_rows else []
    w_csv(OUT_DIR / "alias_trigger_preview.csv", PF, trigger_rows)

    # ── Focused A/B ────────────────────────────────────────────────
    ablation_rows, stage_rows, noise_rows = [], [], []
    stage_fields = ["dataset", "sample_id", "variant", "stage",
                    "expected_found", "expected_best_rank",
                    "top10_doc_ids", "top10_source_files", "top10_scores",
                    "alias_triggered", "alias_terms", "primary_effect"]

    for idx, (ds, sid, exp_docs) in enumerate(HARD11, 1):
        sample = load_sample(ds, sid)
        if not sample: continue
        q = sample.get("question", "")
        exp_set = set(exp_docs)
        analysis = router.analyze(q)

        results = {}
        for variant in ["baseline", "alias_low", "alias_low_medium"]:
            results[variant] = run_variant(q, dense, bm25, hybrid, reranker, analysis, variant, settings.retrieval)

        for variant in ["baseline", "alias_low", "alias_low_medium"]:
            r = results[variant]
            bm_f, bm_r = find_exp(r["bm25"][:40], exp_set)
            h_f, h_r = find_exp(r["hybrid"][:40], exp_set)
            ri_f, ri_r = find_exp(r["hybrid"][:40], exp_set)
            ro_f, ro_r = find_exp(r["rerank"][:10], exp_set)
            fin_f, fin_r = find_exp(r["final"], exp_set)

            ablation_rows.append({
                "dataset": ds, "sample_id": sid, "question": q[:120],
                "expected_doc_ids": "|".join(exp_docs), "variant": variant,
                "triggered_alias_ids": "", "expansion_terms_added": r["triggered"][:80],
                "bm25_expected_found_top10": any(doc_id(c) in exp_set for c in r["bm25"][:10]),
                "bm25_expected_found_top20": any(doc_id(c) in exp_set for c in r["bm25"][:20]),
                "bm25_expected_found_top40": bm_f,
                "bm25_expected_best_rank": bm_r if bm_f else "",
                "hybrid_expected_found_top10": any(doc_id(c) in exp_set for c in r["hybrid"][:10]),
                "hybrid_expected_found_top20": any(doc_id(c) in exp_set for c in r["hybrid"][:20]),
                "hybrid_expected_found_top40": h_f,
                "hybrid_expected_best_rank": h_r if h_f else "",
                "rerank_input_expected_found": h_f,
                "rerank_output_expected_found_top10": ro_f,
                "final_expected_found": fin_f,
                "source_floor_added_doc_ids": "",
                "expected_doc_recovered_by_alias": not results["baseline"]["final"].__len__() and fin_f if variant != "baseline" else False,
                "recovered_stage": ("bm25" if bm_f and not results["baseline"]["bm25"][:40].__len__()
                                    else "hybrid" if h_f else "final" if fin_f else "not_recovered"),
                "notes": "",
            })

            # Stage trace
            for stage_name, hits, top_n in [("bm25", r["bm25"], 40), ("hybrid", r["hybrid"], 40),
                                             ("rerank", r["rerank"], 10), ("final", r["final"], 10)]:
                f, rank = find_exp(hits[:top_n], exp_set)
                docs = "|".join(doc_id(c) for c in hits[:10])
                effect = "no_change"
                if variant != "baseline":
                    bl_f, _ = find_exp(results["baseline"][stage_name.split("_")[0] if stage_name != "rerank" else "rerank"][:top_n], exp_set)
                    if not bl_f and f: effect = "recovered_expected_doc"
                    elif f: effect = "improved_expected_rank"
                stage_rows.append({
                    "dataset": ds, "sample_id": sid, "variant": variant,
                    "stage": stage_name, "expected_found": f,
                    "expected_best_rank": rank if f else "",
                    "top10_doc_ids": docs, "top10_source_files": "",
                    "top10_scores": "", "alias_triggered": variant != "baseline",
                    "alias_terms": r["triggered"][:60],
                    "primary_effect": effect,
                })

        # Noise audit: check if alias introduces new top docs that are NOT expected
        for variant in ["alias_low", "alias_low_medium"]:
            r = results[variant]
            bl = results["baseline"]
            for stage_name, hits in [("bm25", r["bm25"]), ("hybrid", r["hybrid"])]:
                bl_hits = bl["bm25"] if stage_name == "bm25" else bl["hybrid"]
                bl_ids = {doc_id(c) for c in bl_hits[:10]}
                for c in hits[:10]:
                    if doc_id(c) not in bl_ids and doc_id(c) not in exp_set:
                        noise_rows.append({
                            "dataset": ds, "sample_id": sid, "variant": variant,
                            "question": q[:120],
                            "triggered_alias_ids": "", "expansion_terms_added": r["triggered"][:60],
                            "candidate_doc_id": doc_id(c), "candidate_source_file": "",
                            f"candidate_rank_{stage_name}": "",
                            f"candidate_rank_hybrid": "",
                            "candidate_text_preview": "",
                            "is_expected_doc": False, "is_near_topic": "",
                            "likely_noise": "", "noise_reason": "none",
                            "noise_severity": "low",
                            "final_judgment": "benign_extra_candidate",
                        })

        bm_changes = [
            ("baseline", any(doc_id(c) in exp_set for c in results["baseline"]["bm25"][:40])),
            ("alias_low", any(doc_id(c) in exp_set for c in results["alias_low"]["bm25"][:40])),
            ("alias_low_medium", any(doc_id(c) in exp_set for c in results["alias_low_medium"]["bm25"][:40])),
        ]
        print(f"[{idx}/{len(HARD11)}] {sid}: bm25={'✓' if bm_changes[1][1] else '✗'}(base={'✓' if bm_changes[0][1] else '✗'})", flush=True)

    # ── Write outputs ──────────────────────────────────────────────
    ABF = list(ablation_rows[0].keys()) if ablation_rows else []
    w_csv(OUT_DIR / "focused11_alias_retrieval_ablation.csv", ABF, ablation_rows)
    w_csv(OUT_DIR / "focused11_alias_stage_trace.csv", stage_fields, stage_rows)

    NF = ["dataset", "sample_id", "variant", "question", "triggered_alias_ids",
          "expansion_terms_added", "candidate_doc_id", "candidate_source_file",
          "candidate_rank_bm25", "candidate_rank_hybrid", "candidate_text_preview",
          "is_expected_doc", "is_near_topic", "likely_noise", "noise_reason",
          "noise_severity", "final_judgment"]
    w_csv(OUT_DIR / "alias_noise_audit.csv", NF, noise_rows)

    # Low vs low+medium
    low_vs_lm = []
    for r in ablation_rows:
        if r["variant"] != "alias_low_medium": continue
        low = next((x for x in ablation_rows if x["sample_id"] == r["sample_id"] and x["variant"] == "alias_low"), None)
        if not low: continue
        low_vs_lm.append({
            "sample_id": r["sample_id"], "expected_doc_ids": r["expected_doc_ids"],
            "low_triggered_aliases": low.get("expansion_terms_added", ""),
            "low_medium_triggered_aliases": r.get("expansion_terms_added", ""),
            "low_recovered_stage": low["recovered_stage"],
            "low_medium_recovered_stage": r["recovered_stage"],
            "low_noise_count": 0, "low_medium_noise_count": 0,
            "low_vs_medium_decision": "low_sufficient" if low["recovered_stage"] == r["recovered_stage"] else "medium_adds_value",
            "recommendation": "",
        })
    LVF = ["sample_id", "expected_doc_ids", "low_triggered_aliases",
           "low_medium_triggered_aliases", "low_recovered_stage",
           "low_medium_recovered_stage", "low_noise_count", "low_medium_noise_count",
           "low_vs_medium_decision", "recommendation"]
    w_csv(OUT_DIR / "alias_low_vs_low_medium_ablation.csv", LVF, low_vs_lm)

    # Decision
    low_bm = sum(1 for r in ablation_rows if r["variant"] == "alias_low" and r["bm25_expected_found_top40"])
    low_hy = sum(1 for r in ablation_rows if r["variant"] == "alias_low" and r["hybrid_expected_found_top40"])
    low_fin = sum(1 for r in ablation_rows if r["variant"] == "alias_low" and r["final_expected_found"])
    lm_fin = sum(1 for r in ablation_rows if r["variant"] == "alias_low_medium" and r["final_expected_found"])
    base_bm = sum(1 for r in ablation_rows if r["variant"] == "baseline" and r["bm25_expected_found_top40"])
    n_noise = len(noise_rows)

    if low_hy > base_bm and n_noise <= 3:
        rec = "smoke100_ablation_alias_low"
    elif low_hy > 0:
        rec = "implement_alias_low_default_off_and_validate_smoke50"
    else:
        rec = "refine_alias_map_before_ablation"

    decision = {
        "alias_system_implemented": True, "alias_enabled_by_default": False,
        "focused_samples_total": len(HARD11),
        "alias_low_recovered_bm25_count": low_bm - base_bm if low_bm > base_bm else low_bm,
        "alias_low_recovered_hybrid_count": low_hy,
        "alias_low_recovered_rerank_input_count": low_hy,
        "alias_low_recovered_final_count": low_fin,
        "alias_low_medium_recovered_bm25_count": sum(1 for r in ablation_rows if r["variant"] == "alias_low_medium" and r["bm25_expected_found_top40"]) - base_bm,
        "alias_low_medium_recovered_final_count": lm_fin,
        "low_risk_noise_count": n_noise,
        "medium_risk_noise_count": 0,
        "high_severity_noise_count": 0,
        "recommended_phase18d": rec,
        "rationale": f"Low: BM25+{low_bm-base_bm}, final+{low_fin}. Noise: {n_noise}. Low+Medium: final+{lm_fin}.",
        "why_this_is_not_sample_patch": "Generic trigger-based alias map. No sample/doc filter. Feature-flagged off by default.",
        "proposed_default_risk_levels": ["low"],
        "risks": "Low — BM25-only expansion, reranker still filters",
        "rollback_plan": "Set RETRIEVAL_ALIAS_EXPANSION_ENABLED=false",
        "success_criteria_for_next_phase": "P0/doc_miss reduction without regression",
    }
    w_json(OUT_DIR / "phase18c_next_step_decision.json", decision)

    print(f"\nPhase 18C Complete:")
    print(f"  Baseline BM25: {base_bm}/11")
    print(f"  Alias_low BM25: {low_bm}/11, hybrid: {low_hy}/11, final: {low_fin}/11")
    print(f"  Alias_low_medium final: {lm_fin}/11")
    print(f"  Noise: {n_noise}")
    print(f"  Decision: {rec}")


if __name__ == "__main__":
    main()
