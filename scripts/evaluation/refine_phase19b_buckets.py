#!/usr/bin/env python3
"""Phase 19B: Refined bucketing from per_sample_metrics.csv"""
import csv, json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results/phase19b_cross_lingual_audit")

# Load metrics
rows = []
with open(RESULTS / "per_sample_metrics.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Group by sample_id
samples = {}
for r in rows:
    sid = r["sample_id"]
    if sid not in samples:
        samples[sid] = {}
    samples[sid][r["variant_id"]] = r

def int_or_0(v):
    try: return int(v)
    except: return 0

def float_or_0(v):
    try: return float(v)
    except: return 0.0

def bool_or(v):
    return str(v).lower() == "true"

bucket_rows = []

for sid, variants in samples.items():
    v0 = variants["v0"]
    v1 = variants["v1"]
    v2 = variants["v2"]
    v3 = variants["v3"]
    fsrc = v0["failure_source"]

    # Key metrics
    v0_d_doc40 = bool_or(v0["dense_expected_doc_found_top40"])
    v1_d_doc40 = bool_or(v1["dense_expected_doc_found_top40"])
    v0_d_ck_rank = int_or_0(v0["dense_rank_of_expected_primary_chunk"])
    v1_d_ck_rank = int_or_0(v1["dense_rank_of_expected_primary_chunk"])
    v0_h_rank = int_or_0(v0["hybrid_rank_post_source_floor"])
    v1_h_rank = int_or_0(v1["hybrid_rank_post_source_floor"])
    v0_pr_rank = int_or_0(v0["pipeline_rerank_rank_of_expected_chunk"])
    v1_pr_rank = int_or_0(v1["pipeline_rerank_rank_of_expected_chunk"])
    v0_or_rank = int_or_0(v0["oracle_rerank_rank_of_expected_chunk"])
    v1_or_rank = int_or_0(v1["oracle_rerank_rank_of_expected_chunk"])
    v0_final = bool_or(v0["final_contains_expected_chunk"])
    v1_final = bool_or(v1["final_contains_expected_chunk"])
    v0_sel = bool_or(v0["selected_support_contains_expected_chunk"])
    v1_sel = bool_or(v1["selected_support_contains_expected_chunk"])
    v0_d_score = float_or_0(v0["dense_score_of_expected_primary_chunk"])
    v1_d_score = float_or_0(v1["dense_score_of_expected_primary_chunk"])
    v0_pr_score = float_or_0(v0["pipeline_rerank_score_of_expected_chunk"])
    v1_pr_score = float_or_0(v1["pipeline_rerank_score_of_expected_chunk"])

    # Deltas
    d_rank_delta = v1_d_ck_rank - v0_d_ck_rank if v0_d_ck_rank > 0 and v1_d_ck_rank > 0 else 0
    d_found_new = v0_d_ck_rank <= 0 and v1_d_ck_rank > 0  # v1 newly finds the primary chunk
    v1_found = v1_d_ck_rank > 0 and v1_d_ck_rank <= 40
    v0_found = v0_d_ck_rank > 0 and v0_d_ck_rank <= 40
    v1_recovered = (not v0_final) and v1_final
    v0_in_pipeline = bool_or(v0["pipeline_rerank_input_contains_expected_chunk"])
    v1_in_pipeline = bool_or(v1["pipeline_rerank_input_contains_expected_chunk"])
    v0_oracle_good = v0_or_rank > 0 and v0_or_rank <= 3
    v1_oracle_good = v1_or_rank > 0 and v1_or_rank <= 3

    # Check chunk text quality
    exp_id = v0["expected_chunk_id"]
    # Read chunk text
    chunk_text = ""
    try:
        with open("data/paper_round1/chunks/chunks.jsonl") as f:
            for line in f:
                c = json.loads(line)
                if c["chunk_id"] == exp_id:
                    chunk_text = c.get("text", "")
                    break
    except:
        pass
    text_len = len(chunk_text)
    text_entities = ["pichia","e. coli","ecoli","yeast","coli","b.subtilis","bacillus",
                     "2'-fl","2-fucosyllactose","sialyl","neu5ac","lra","promoter",
                     "expression","production","pathway","protein","gene","enzyme",
                     "plasmid","strain","host","vector","metabolic"]
    has_entity = any(w in chunk_text.lower() for w in text_entities)
    text_sparse = text_len < 250 or not has_entity

    # ── Refined bucketing ──
    primary = "Mixed"
    secondary = "none"
    chunk_sub = "none"
    prop_loss = "none"
    evidence = ""
    conf = "medium"

    # Check propagation: any early stage finds but later loses?
    early_found = v0_d_ck_rank > 0 and v0_d_ck_rank <= 40
    bm25_found = int_or_0(v0["bm25_rank_of_expected_primary_chunk"]) > 0 and int_or_0(v0["bm25_rank_of_expected_primary_chunk"]) <= 40
    hybrid_found = int_or_0(v0["hybrid_rank_post_source_floor"]) > 0 and int_or_0(v0["hybrid_rank_post_source_floor"]) <= 40

    if early_found and not hybrid_found:
        secondary = "P"
        prop_loss = "dense_to_hybrid"
    elif bm25_found and not hybrid_found:
        secondary = "P"
        prop_loss = "bm25_to_hybrid"
    elif hybrid_found and not v0_in_pipeline:
        secondary = "P"
        prop_loss = "hybrid_to_rerank_input"
    elif v0_in_pipeline and not v0_final:
        secondary = "P"
        prop_loss = "rerank_to_final"
    elif v0_final and not v0_sel and fsrc == "low_support_score":
        secondary = "P"
        prop_loss = "final_to_support"

    # If v1 recovers the chunk, it IS the main story — this is Bucket-Q
    if v1_recovered:
        primary = "Q"
        evidence = f"v1 newly recovers expected chunk to final (v0_final={v0_final}, v1_final={v1_final}, v1_d_rank={v1_d_ck_rank})"
        conf = "high"

    # v1 finds the primary chunk where v0 didn't (strong Q signal even if not final)
    elif d_found_new and v1_found:
        primary = "Q"
        evidence = f"v1 newly finds expected primary chunk in dense (v0_d_rank={v0_d_ck_rank}, v1_d_rank={v1_d_ck_rank})"
        conf = "high"

    # v1 significantly improves dense rank
    elif v0_found and v1_found and (v0_d_ck_rank - v1_d_ck_rank) >= 10:
        primary = "Q"
        evidence = f"v1 improves dense rank from {v0_d_ck_rank}→{v1_d_ck_rank} (delta={v0_d_ck_rank-v1_d_ck_rank})"
        conf = "high"

    # Bucket-R: expected chunk enters pipeline but reranker under-ranks
    # AND oracle also under-ranks (if oracle could recognize it, the issue might be different)
    elif (v0_in_pipeline or v1_in_pipeline):
        pr_rank = v0_pr_rank if v0_pr_rank > 0 else v1_pr_rank
        or_rank = v0_or_rank if v0_or_rank > 0 else v1_or_rank
        if (pr_rank > 3 or pr_rank <= 0) and not v0_oracle_good:
            primary = "R"
            evidence = f"Expected chunk in pipeline/oracle rerank but reranker under-ranks (pr_rank={pr_rank}, or_rank={or_rank})"
            conf = "high"

    # Bucket-D: even v1 can't find with dense, and chunk text has signal
    elif not v1_found and not v1_d_doc40 and has_entity and text_len >= 250:
        primary = "D"
        evidence = f"Even v1 misses expected chunk in dense (v1_d_ck_rank={v1_d_ck_rank}), but chunk has text/entities; dense embedding cross-lingual gap"
        conf = "medium"

    # Bucket-C: all variants fail, chunk text is the likely issue
    elif not v1_found and not v0_found:
        primary = "C"
        if text_sparse:
            chunk_sub = "C1"
            evidence = f"No variant finds chunk (text_len={text_len}, has_entity={has_entity}); chunk too sparse/generic"
        else:
            chunk_sub = "C3"
            evidence = f"No variant finds chunk but text seems adequate (len={text_len}); likely missing metadata context"
        conf = "medium"

    # Mixed with Q lean
    elif v1_found and not v0_found:
        primary = "Q"
        evidence = f"v1 finds expected chunk (d_rank={v1_d_ck_rank}); query formulation is primary lever"
        conf = "medium"

    else:
        primary = "Mixed"
        evidence = f"Complex: v0_d={v0_d_ck_rank}, v1_d={v1_d_ck_rank}, v0_final={v0_final}, v1_final={v1_final}, oracle={v0_or_rank}"
        conf = "low"

    # Best variant
    best = "none"
    best_effect = "no_effect"
    if v1_sel and not v0_sel:
        best = "v1"
        best_effect = "final_recovered_and_selected"
    elif v1_final and not v0_final:
        best = "v1"
        best_effect = "final_recovered"
    elif d_found_new:
        best = "v1"
        best_effect = "dense_improved"
    elif d_rank_delta < -5:
        best = "v1"
        best_effect = "dense_rank_improved"
    elif int_or_0(v2["dense_rank_of_expected_primary_chunk"]) > 0 and int_or_0(v2["dense_rank_of_expected_primary_chunk"]) < v0_d_ck_rank:
        best = "v2"
        best_effect = "dense_improved"

    # Recommend
    rec = "unclear"
    if primary == "Q":
        rec = "Option_B_query_rewriting"
    elif primary == "R":
        rec = "Option_E_rerank_input_or_calibration"
    elif primary == "C":
        rec = "Option_D_metadata_enriched_index" if chunk_sub in ("C1","C3") else "Option_C_doc_level_recall"
    elif primary == "D":
        rec = "Dense_calibration_design"

    bucket_rows.append({
        "dataset": v0["dataset"], "sample_id": sid, "failure_source": fsrc,
        "primary_bucket": primary, "secondary_bucket": secondary,
        "bucket_confidence": conf,
        "chunk_signal_subtype": chunk_sub,
        "propagation_loss_stage": prop_loss,
        "best_variant": best, "best_variant_effect": best_effect,
        "evidence_for_bucket": evidence,
        "recommended_next_option": rec,
        "notes": f"v0_d_rank={v0_d_ck_rank} v1_d_rank={v1_d_ck_rank} v0_final={v0_final} v1_final={v1_final} v0_oracle={v0_or_rank} text_len={text_len} has_entity={has_entity}"
    })

# Write refined buckets
with open(RESULTS / "bucket_assignments.csv", "w", newline="") as f:
    fields = ["dataset","sample_id","failure_source","primary_bucket","secondary_bucket",
              "bucket_confidence","chunk_signal_subtype","propagation_loss_stage",
              "best_variant","best_variant_effect","evidence_for_bucket",
              "recommended_next_option","notes"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for r in bucket_rows:
        w.writerow(r)

# bucket_summary.json
bcounts = defaultdict(int)
for r in bucket_rows:
    bcounts[r["primary_bucket"]] += 1
c1 = sum(1 for r in bucket_rows if r["chunk_signal_subtype"]=="C1")
c2 = sum(1 for r in bucket_rows if r["chunk_signal_subtype"]=="C2")
c3 = sum(1 for r in bucket_rows if r["chunk_signal_subtype"]=="C3")
pcount = sum(1 for r in bucket_rows if r["secondary_bucket"]=="P")
bv_dist = defaultdict(int)
for r in bucket_rows:
    bv_dist[r["best_variant"]] += 1
dom = max(bcounts, key=bcounts.get)
plurality = bcounts[dom] >= 6

bs = {
    "total_samples": 16,
    "bucket_q_count": bcounts["Q"],
    "bucket_r_count": bcounts["R"],
    "bucket_c_count": bcounts["C"],
    "bucket_c1_count": c1, "bucket_c2_count": c2, "bucket_c3_count": c3,
    "bucket_d_count": bcounts["D"],
    "mixed_count": bcounts["Mixed"],
    "secondary_p_count": pcount,
    "best_variant_distribution": dict(bv_dist),
    "dominant_bucket": dom,
    "dominant_bucket_count": bcounts[dom],
    "no_clear_plurality": not plurality,
    "interpretation": ""
}
if dom == "Q" and plurality:
    bs["interpretation"] = "Query language/formulation is the main fixable lever. Phase 19C should pursue Option B (query rewriting / bilingual decomposition) shadow A/B."
elif dom == "R" and plurality:
    bs["interpretation"] = "Reranker cross-lingual ranking is the main gap. Phase 19C should pursue Option E (reranker calibration / translate-before-rerank / input enrichment) shadow A/B."
elif dom == "C" and plurality:
    bs["interpretation"] = "Chunk text/context signal is the main gap. Phase 19C should pursue Option D (metadata-enriched chunk) if C1/C3 dominant, or Option C (doc-level recall) if C2 dominant."
elif dom == "D" and plurality:
    bs["interpretation"] = "Dense embedding cross-lingual gap is dominant. Pause fix-style work, produce dense calibration design doc."
else:
    bs["interpretation"] = "No clear plurality — run Option B (cheapest) + Option F (eval cleanup) in parallel."

with open(RESULTS / "bucket_summary.json", "w") as f:
    json.dump(bs, f, indent=2, ensure_ascii=False)

# phase19c_recommendation.json
if dom == "Q" and bcounts[dom] >= 6:
    rec19c = "option_b_query_rewriting_shadow_ab"
elif dom == "R" and bcounts[dom] >= 6:
    rec19c = "option_e_rerank_translation_or_input_enrichment"
elif dom == "C" and bcounts[dom] >= 6:
    rec19c = "option_d_metadata_enriched_chunk_shadow_index" if c2 <= c1 + c3 else "option_c_doc_level_recall_design"
elif dom == "D" and bcounts[dom] >= 6:
    rec19c = "dense_calibration_design_doc"
else:
    rec19c = "mixed_option_b_plus_metric_cleanup"

rec = {
    "recommended_phase19c": rec19c,
    "rationale": bs["interpretation"],
    "bucket_distribution": {"Q":bcounts["Q"],"R":bcounts["R"],"C":bcounts["C"],"D":bcounts["D"],"Mixed":bcounts["Mixed"]},
    "focused_samples_for_phase19c": [r["sample_id"] for r in bucket_rows if r["primary_bucket"]=="Q"] if bcounts["Q"]>=6 else [r["sample_id"] for r in bucket_rows],
    "proposed_variants": ["v0_original_CN","v1_EN_mirror","v2_bilingual"],
    "success_criteria": "≥40% Bucket-Q focused samples recover expected chunk to final; no smoke50 precision regression",
    "risks": "Semantic drift from translation; query-length explosion; precision regression for non-cross-lingual queries",
    "why_not_continue_alias": "Phase 18C: +1 BM25 hit, 0 final recovery.",
    "why_not_source_floor_tuning": "Phase 17F: source-floor eliminated its class.",
    "why_not_support_capacity": "Phase 18F: 1/6 fixed, 5/6 root-caused upstream.",
    "why_not_direct_reranker_replacement": "No calibration measurement yet; replacement is high-risk.",
    "next_validation_plan": "Phase 19C: shadow A/B on Bucket-Q samples with v1/v2 query variants; gate on focused set → smoke50 sanity."
}
with open(RESULTS / "phase19c_recommendation.json", "w") as f:
    json.dump(rec, f, indent=2, ensure_ascii=False)

print("Refined bucket assignments:")
for r in bucket_rows:
    print(f"  {r['sample_id']:15s} {r['failure_source']:20s} → {r['primary_bucket']:5s} ({r['secondary_bucket']:3s}) best={r['best_variant']} rec={r['recommended_next_option']}")
print(f"\nDistribution: {dict(bcounts)}")
print(f"Dominant: {dom} ({bcounts[dom]}/16), plurality={plurality}")
print(f"P (propagation) secondary: {pcount}")
print(f"Phase 19C recommendation: {rec19c}")
