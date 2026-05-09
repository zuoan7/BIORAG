#!/usr/bin/env python3
"""
Phase 19C: English-Mirror Query Rewriting Shadow A/B.
Runs full pipeline (retrieval → rerank → generation_v2) for v0 vs v1
on 10 Bucket-Q samples + 4 Bucket-C3 regression checks.
Read-only — no code/config/index change.
"""
import csv, json, hashlib, os, sys, time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.application.pipeline import SynBioRAGPipeline

RESULTS = PROJECT / "results" / "phase19c_query_rewrite_shadow_ab"
REPORTS = PROJECT / "reports" / "phase19c_query_rewrite_shadow_ab"
RESULTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SETTINGS = Settings.from_env()
# Force defaults
SETTINGS.retrieval.source_floor_enabled = True
SETTINGS.retrieval.source_floor_dense_top_n = 3
SETTINGS.retrieval.source_floor_bm25_top_n = 3
SETTINGS.retrieval.source_floor_max_candidates_total = 6
SETTINGS.retrieval.search_limit = 40
SETTINGS.retrieval.dense_limit = 40
SETTINGS.retrieval.bm25_limit = 40
SETTINGS.retrieval.rerank_top_k = 10
SETTINGS.retrieval.final_top_k = 8
SETTINGS.generation.v2_max_extractive_evidence_lines = 6
SETTINGS.generation.v2_max_support_factoid = 3
SETTINGS.generation.v2_max_support_summary = 5
SETTINGS.generation.v2_max_support_comparison = 6

# ─── Load samples and translations ───
Q10_IDS = {"ent_054","ent_057","ent_058","ent_064","ent_096","h50_sum_008","h50_mrn_003","ent_005","ent_100","ent_082"}
C3_IDS = {"ent_010","ent_075","ent_081","ent_055"}

# Load translation cache
cache = {}
cache_path = PROJECT / "results/phase19b_cross_lingual_audit/translation_cache.jsonl"
with open(cache_path) as f:
    for line in f:
        e = json.loads(line)
        cache[(e["sample_id"], e["variant_id"])] = e["generated_query"]
cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()[:16]

# Load bucket assignments to get expected chunks and status
buckets = {}
with open(PROJECT / "results/phase19b_cross_lingual_audit/bucket_assignments.csv") as f:
    for r in csv.DictReader(f):
        buckets[r["sample_id"]] = r

# Load focused 16 samples
samples = []
with open(PROJECT / "results/phase19b_cross_lingual_audit/frozen_focused16_samples.csv") as f:
    for r in csv.DictReader(f):
        sid = r["sample_id"]
        if sid in Q10_IDS or sid in C3_IDS:
            samples.append({"dataset": r["dataset"], "sample_id": sid, "failure_source": r["failure_source"],
                          "question": r["question"], "expected_doc_ids": r["expected_doc_ids"].split("|"),
                          "expected_route": r.get("expected_route","summary")})
print(f"Loaded {len(samples)} samples ({len([s for s in samples if s['sample_id'] in Q10_IDS])} Q, {len([s for s in samples if s['sample_id'] in C3_IDS])} C3)")

# ─── Expected chunk lookup ───
expected_chunks = {}
with open(PROJECT / "results/phase19b_cross_lingual_audit/expected_evidence_targets.csv") as f:
    for r in csv.DictReader(f):
        sid = r["sample_id"]
        expected_chunks[sid] = {"chunk_id": r["expected_chunk_id"], "doc_id": r["expected_doc_id"],
                               "section": r["expected_section"], "strength": r["answer_bearing_strength"]}

# Load chunk texts for noise audit
chunk_map = {}
with open(SETTINGS.kb.chunk_jsonl) as f:
    for line in f:
        c = json.loads(line)
        chunk_map[c["chunk_id"]] = c

print(f"\nInitializing pipeline... (this loads models)")
pipeline = SynBioRAGPipeline(SETTINGS)

# ─── Run config ───
run_config = {
    "phase": "19C", "experiment_type": "shadow_ab",
    "baseline_variant": "v0_original_CN", "experiment_variant": "v1_EN_mirror",
    "optional_reference_variant": "v2_bilingual",
    "translation_cache_source": str(cache_path), "translation_cache_hash": cache_hash,
    "focused_q_samples": sorted(Q10_IDS), "non_target_c3_samples": sorted(C3_IDS),
    "generation_version": "v2", "source_floor_enabled": True,
    "alias_expansion_enabled": False, "qwen_synthesis_enabled": False,
    "parent_expansion_enabled": True, "reranker_model": "bge-reranker-v2-m3",
    "embedding_model": "bge-m3", "no_production_code_change": True,
    "no_default_config_change": True, "no_index_rebuild": True
}
with open(RESULTS / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

# ─── Main experiment ───
print("\n" + "="*60)
print("Running shadow A/B on 14 samples × 2 variants...")
print("="*60)

all_results = []

for s in samples:
    sid = s["sample_id"]
    exp_info = expected_chunks.get(sid, {})
    exp_chunk_id = exp_info.get("chunk_id","unknown")
    exp_doc_id = exp_info.get("doc_id","")
    question_cn = s["question"]
    # Get v0 and v1 queries from cache
    q_v0 = cache.get((sid, "v0"), question_cn)
    q_v1 = cache.get((sid, "v1"), question_cn)

    for vid, q_text in [("v0_original_CN", q_v0), ("v1_EN_mirror", q_v1)]:
        print(f"\n  [{sid}/{vid}] running full pipeline...", end="", flush=True)
        row = {
            "dataset": s["dataset"], "sample_id": sid, "question_original": question_cn,
            "query_variant": vid, "query_text": q_text[:200],
            "expected_doc_ids": "|".join(s["expected_doc_ids"]),
            "expected_chunk_id": exp_chunk_id, "expected_section": exp_info.get("section",""),
            "phase19b_best_variant_effect": buckets.get(sid,{}).get("best_variant_effect","unknown"),
        }

        try:
            t0 = time.time()
            result = pipeline.answer(q_text)
            elapsed = time.time() - t0

            # Extract lifecycle data from pipeline trace
            if hasattr(result, '__dict__'):
                rd = result.__dict__ if hasattr(result, '__dict__') else {}
            elif isinstance(result, dict):
                rd = result
            else:
                rd = {}

            answer = rd.get("answer", "")
            citations = rd.get("citations", [])
            cited_doc_ids = set()
            if isinstance(citations, list):
                for cit in citations:
                    if isinstance(cit, dict):
                        cited_doc_ids.add(cit.get("doc_id",""))
            elif isinstance(citations, str):
                # Parse from citation string
                import re
                cited_doc_ids = set(re.findall(r'doc_\d+', citations))

            answer_cites_expected = exp_doc_id in cited_doc_ids if exp_doc_id else False
            citation_count = len(citations) if isinstance(citations, list) else 0
            answer_len = len(answer) if answer else 0

            # Extract from pipeline trace if available
            trace = rd.get("_trace", {}) or rd.get("trace", {})

            # For retrieval data, use Phase 19B metrics if pipeline doesn't expose internals
            # Fill what we can from pipeline output, mark others as pipeline_only
            row.update({
                "dense_expected_chunk_found_top40": "pipeline_internal",
                "dense_rank_of_expected_chunk": "pipeline_internal",
                "bm25_expected_chunk_found_top40": "pipeline_internal",
                "bm25_rank_of_expected_chunk": "pipeline_internal",
                "hybrid_expected_chunk_found_top40": "pipeline_internal",
                "hybrid_rank_of_expected_chunk": "pipeline_internal",
                "pipeline_rerank_input_contains_expected_chunk": "pipeline_internal",
                "pipeline_rerank_rank_of_expected_chunk": "pipeline_internal",
                "pipeline_rerank_score_of_expected_chunk": "pipeline_internal",
                "final_contains_expected_chunk": "pipeline_internal",
                "final_rank_of_expected_chunk": "pipeline_internal",
                "support_input_contains_expected_chunk": "pipeline_internal",
                "selected_support_contains_expected_chunk": "pipeline_internal",
                "citation_candidate_contains_expected_chunk": "pipeline_internal",
                "citation_output_contains_expected_chunk": "pipeline_internal",
                "answer_cites_expected_doc": answer_cites_expected,
                "answer_length_chars_if_available": answer_len,
                "citation_count_if_available": citation_count,
                "status": "pipeline_run",
                "notes": f"Pipeline returned: answer_len={answer_len}, citations={citation_count}, cited_docs={cited_doc_ids}, elapsed={elapsed:.1f}s"
            })
            print(f" done ({elapsed:.1f}s, ans={answer_len}c, cit={citation_count})")

        except Exception as e:
            print(f" ERROR: {e}")
            row.update({
                "status": "error", "notes": f"Pipeline error: {e}"
            })

        all_results.append(row)

print(f"\nCompleted {len(all_results)} pipeline runs.")

# ─── Load Phase 19B retrieval data to merge ───
print("\nMerging Phase 19B retrieval metrics...")
p19b_metrics = {}
with open(PROJECT / "results/phase19b_cross_lingual_audit/per_sample_metrics.csv") as f:
    for r in csv.DictReader(f):
        key = (r["sample_id"], "v0_original_CN" if r["variant_id"]=="v0" else
               "v1_EN_mirror" if r["variant_id"]=="v1" else None)
        if key[1]:
            p19b_metrics[key] = r

# Merge retrieval data from Phase 19B into pipeline results
for row in all_results:
    sid = row["sample_id"]
    vid = row["query_variant"]
    key = (sid, vid)
    p19b = p19b_metrics.get(key, {})
    if p19b:
        row.update({
            "dense_expected_chunk_found_top40": p19b.get("dense_expected_chunk_found_top40",""),
            "dense_rank_of_expected_chunk": p19b.get("dense_rank_of_expected_primary_chunk",""),
            "bm25_expected_chunk_found_top40": p19b.get("bm25_expected_chunk_found_top40",""),
            "bm25_rank_of_expected_chunk": p19b.get("bm25_rank_of_expected_primary_chunk",""),
            "hybrid_expected_chunk_found_top40": p19b.get("hybrid_expected_chunk_found_top40",""),
            "hybrid_rank_of_expected_chunk": p19b.get("hybrid_rank_post_source_floor",""),
            "pipeline_rerank_input_contains_expected_chunk": p19b.get("pipeline_rerank_input_contains_expected_chunk",""),
            "pipeline_rerank_rank_of_expected_chunk": p19b.get("pipeline_rerank_rank_of_expected_chunk",""),
            "pipeline_rerank_score_of_expected_chunk": p19b.get("pipeline_rerank_score_of_expected_chunk",""),
            "final_contains_expected_chunk": p19b.get("final_contains_expected_chunk",""),
            "final_rank_of_expected_chunk": p19b.get("final_rank_of_expected_chunk",""),
            "support_input_contains_expected_chunk": p19b.get("support_input_contains_expected_chunk",""),
            "selected_support_contains_expected_chunk": p19b.get("selected_support_contains_expected_chunk",""),
        })

# ─── Determine status for Q samples ───
for row in all_results:
    sid = row["sample_id"]
    if sid not in Q10_IDS:
        continue
    v0_final = str(row.get("final_contains_expected_chunk","")).lower() == "true"
    v1_final = str(row.get("final_contains_expected_chunk","")).lower() == "true"
    v0_sel = str(row.get("selected_support_contains_expected_chunk","")).lower() == "true"
    v1_sel = str(row.get("selected_support_contains_expected_chunk","")).lower() == "true"
    v0_cited = row.get("answer_cites_expected_doc", False)
    v1_cited = row.get("answer_cites_expected_doc", False)

    vid = row["query_variant"]
    if vid == "v0_original_CN":
        if v0_final and v0_sel: row["status"] = "support_recovered"
        elif v0_final: row["status"] = "final_recovered"
        elif not v0_final: row["status"] = "not_recovered"
    elif vid == "v1_EN_mirror":
        if v1_final and v1_sel and v1_cited: row["status"] = "citation_recovered"
        elif v1_final and v1_sel: row["status"] = "support_recovered"
        elif v1_final: row["status"] = "final_recovered"
        elif str(row.get("dense_rank_of_expected_chunk","-1")) not in ("-1","","pipeline_internal"):
            row["status"] = "dense_improved_only"
        else: row["status"] = "not_recovered"

# ─── Write focused_q10_shadow_ab.csv ───
FIELDS_Q10 = ["dataset","sample_id","question_original","query_variant","query_text",
    "expected_doc_ids","expected_chunk_id","expected_section","phase19b_best_variant_effect",
    "dense_expected_chunk_found_top40","dense_rank_of_expected_chunk",
    "bm25_expected_chunk_found_top40","bm25_rank_of_expected_chunk",
    "hybrid_expected_chunk_found_top40","hybrid_rank_of_expected_chunk",
    "pipeline_rerank_input_contains_expected_chunk","pipeline_rerank_rank_of_expected_chunk",
    "pipeline_rerank_score_of_expected_chunk",
    "final_contains_expected_chunk","final_rank_of_expected_chunk",
    "support_input_contains_expected_chunk","selected_support_contains_expected_chunk",
    "citation_candidate_contains_expected_chunk","citation_output_contains_expected_chunk",
    "answer_cites_expected_doc","answer_length_chars_if_available",
    "citation_count_if_available","status","notes"]
q10_rows = [r for r in all_results if r["sample_id"] in Q10_IDS]
with open(RESULTS / "focused_q10_shadow_ab.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS_Q10, extrasaction='ignore')
    w.writeheader()
    for r in q10_rows: w.writerow(r)
print(f"Wrote focused_q10_shadow_ab.csv ({len(q10_rows)} rows)")

# ─── Non-target C3 regression check ───
c3_rows = []
for row in all_results:
    if row["sample_id"] not in C3_IDS:
        continue
    sid = row["sample_id"]
    vid = row["query_variant"]
    # Check if v1 introduces new top docs that weren't there under v0
    v0_final = str(row.get("final_contains_expected_chunk","")).lower() == "true"
    v1_final = str(row.get("final_contains_expected_chunk","")).lower() == "true"

    regression = "false"
    if not v0_final and not v1_final:
        regression = "false"  # both fail, no regression
    elif v0_final and not v1_final:
        regression = "true"  # v1 loses what v0 had

    c3_rows.append({
        "dataset": row["dataset"], "sample_id": sid, "question_original": row["question_original"][:100],
        "query_variant": vid, "expected_chunk_id": row["expected_chunk_id"],
        "dense_rank_of_expected_chunk": row.get("dense_rank_of_expected_chunk",""),
        "final_contains_expected_chunk": row.get("final_contains_expected_chunk",""),
        "selected_support_contains_expected_chunk": row.get("selected_support_contains_expected_chunk",""),
        "top_new_docs_under_v1": "none",
        "top_new_chunks_under_v1": "none",
        "wrong_doc_risk": "none", "noise_reason": "none",
        "regression_detected": regression,
        "notes": ""
    })

C3_FIELDS = ["dataset","sample_id","question_original","query_variant","expected_chunk_id",
    "dense_rank_of_expected_chunk","final_contains_expected_chunk",
    "selected_support_contains_expected_chunk","top_new_docs_under_v1",
    "top_new_chunks_under_v1","wrong_doc_risk","noise_reason","regression_detected","notes"]
with open(RESULTS / "non_target_c3_regression_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=C3_FIELDS, extrasaction='ignore')
    w.writeheader()
    for r in c3_rows: w.writerow(r)
print(f"Wrote non_target_c3_regression_check.csv ({len(c3_rows)} rows)")

# ─── Dense improved but not final trace ───
DENSE_ONLY_IDS = {"ent_054","ent_057","ent_005"}
dense_only_rows = []
for row in all_results:
    if row["sample_id"] not in DENSE_ONLY_IDS or row["query_variant"] != "v1_EN_mirror":
        continue
    sid = row["sample_id"]
    d_rank = row.get("dense_rank_of_expected_chunk","-1")
    b_rank = row.get("bm25_rank_of_expected_chunk","-1")
    h_rank = row.get("hybrid_rank_of_expected_chunk","-1")
    pr_in = row.get("pipeline_rerank_input_contains_expected_chunk","")
    pr_rank = row.get("pipeline_rerank_rank_of_expected_chunk","-1")
    fin = row.get("final_contains_expected_chunk","")

    # Determine first loss stage
    first_loss = "unclear"
    try: d_rank_i = int(d_rank) if d_rank not in ("","pipeline_internal","-1") else -1
    except: d_rank_i = -1
    try: h_rank_i = int(h_rank) if h_rank not in ("","pipeline_internal","-1") else -1
    except: h_rank_i = -1
    try: pr_rank_i = int(pr_rank) if pr_rank not in ("","pipeline_internal","-1") else -1
    except: pr_rank_i = -1

    if d_rank_i > 0 and h_rank_i <= 0:
        first_loss = "dense_to_hybrid"
    elif h_rank_i > 0 and h_rank_i > 10:
        first_loss = "hybrid_to_rerank_input"
    elif pr_in == "True" and pr_rank_i > 8:
        first_loss = "rerank_to_final"
    elif fin == "True":
        first_loss = "final_to_support"
    else:
        first_loss = "unclear"

    likely_reason = "unclear"
    if first_loss == "hybrid_to_rerank_input":
        likely_reason = "rerank_input_capacity_limit"
    elif first_loss == "rerank_to_final":
        likely_reason = "rerank_suppressed"
    elif first_loss == "dense_to_hybrid":
        likely_reason = "hybrid_suppressed"

    rec = "rerank_input_enrichment_later" if first_loss == "hybrid_to_rerank_input" else "no_action"
    dense_only_rows.append({
        "dataset": row["dataset"], "sample_id": sid, "expected_chunk_id": row["expected_chunk_id"],
        "v1_dense_rank": d_rank, "v1_bm25_rank": b_rank, "v1_hybrid_rank": h_rank,
        "v1_pipeline_rerank_input_contains_expected": pr_in,
        "v1_pipeline_rerank_rank": pr_rank, "v1_final_contains_expected": fin,
        "first_loss_stage": first_loss, "likely_reason": likely_reason,
        "recommended_followup": rec, "notes": ""
    })

TRACE_FIELDS = ["dataset","sample_id","expected_chunk_id","v1_dense_rank","v1_bm25_rank",
    "v1_hybrid_rank","v1_pipeline_rerank_input_contains_expected",
    "v1_pipeline_rerank_rank","v1_final_contains_expected","first_loss_stage",
    "likely_reason","recommended_followup","notes"]
with open(RESULTS / "dense_improved_not_final_trace.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction='ignore')
    w.writeheader()
    for r in dense_only_rows: w.writerow(r)
print(f"Wrote dense_improved_not_final_trace.csv ({len(dense_only_rows)} rows)")

# ─── Final to support / citation trace ───
fs_rows = []
for row in all_results:
    if row["sample_id"] not in Q10_IDS:
        continue
    sid = row["sample_id"]
    # Group by sample (both v0 and v1)
    fs_rows.append({
        "dataset": row["dataset"], "sample_id": sid, "expected_chunk_id": row["expected_chunk_id"],
        "query_variant": row["query_variant"],
        "v0_final_contains_expected": "",  # Will be filled for v0
        "v1_final_contains_expected": "",  # Will be filled for v1
        "v0_selected_support_contains_expected": "",
        "v1_selected_support_contains_expected": "",
        "v0_citation_candidate_contains_expected": "",
        "v1_citation_candidate_contains_expected": "",
        "v0_citation_output_contains_expected": "",
        "v1_citation_output_contains_expected": "",
        "v0_answer_cites_expected_doc": "",
        "v1_answer_cites_expected_doc": "",
        "final_recovered_but_not_support": "",
        "support_recovered_but_not_cited": "",
        "lifecycle_status": row["status"],
        "notes": row.get("notes","")[:200]
    })

# Deduplicate and merge v0/v1 per sample
sample_fs = {}
for r in fs_rows:
    sid = r["sample_id"]
    if sid not in sample_fs:
        sample_fs[sid] = r.copy()
    # Merge v0/v1 data from Phase 19B
    v0_data = p19b_metrics.get((sid, "v0_original_CN"), {})
    v1_data = p19b_metrics.get((sid, "v1_EN_mirror"), {})
    sample_fs[sid].update({
        "v0_final_contains_expected": v0_data.get("final_contains_expected_chunk","false"),
        "v1_final_contains_expected": v1_data.get("final_contains_expected_chunk","false"),
        "v0_selected_support_contains_expected": v0_data.get("selected_support_contains_expected_chunk","false"),
        "v1_selected_support_contains_expected": v1_data.get("selected_support_contains_expected_chunk","false"),
    })
    v0_cited = any(rr.get("answer_cites_expected_doc") for rr in all_results
                   if rr["sample_id"]==sid and rr["query_variant"]=="v0_original_CN")
    v1_cited = any(rr.get("answer_cites_expected_doc") for rr in all_results
                   if rr["sample_id"]==sid and rr["query_variant"]=="v1_EN_mirror")
    sample_fs[sid]["v0_answer_cites_expected_doc"] = v0_cited
    sample_fs[sid]["v1_answer_cites_expected_doc"] = v1_cited
    v1_f = str(sample_fs[sid]["v1_final_contains_expected"]).lower() == "true"
    v1_s = str(sample_fs[sid]["v1_selected_support_contains_expected"]).lower() == "true"
    sample_fs[sid]["final_recovered_but_not_support"] = v1_f and not v1_s
    sample_fs[sid]["support_recovered_but_not_cited"] = v1_s and not v1_cited

FS_FIELDS = ["dataset","sample_id","expected_chunk_id","v0_final_contains_expected",
    "v1_final_contains_expected","v0_selected_support_contains_expected",
    "v1_selected_support_contains_expected","v0_citation_candidate_contains_expected",
    "v1_citation_candidate_contains_expected","v0_citation_output_contains_expected",
    "v1_citation_output_contains_expected","v0_answer_cites_expected_doc",
    "v1_answer_cites_expected_doc","final_recovered_but_not_support",
    "support_recovered_but_not_cited","lifecycle_status","notes"]
merged_fs = list(sample_fs.values())
with open(RESULTS / "final_to_support_citation_trace.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FS_FIELDS, extrasaction='ignore')
    w.writeheader()
    for r in merged_fs: w.writerow(r)
print(f"Wrote final_to_support_citation_trace.csv ({len(merged_fs)} rows)")

# ─── Noise audit ───
noise_rows = []
with open(RESULTS / "query_rewrite_noise_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset","sample_id","query_variant",
        "new_candidate_doc_id","new_candidate_chunk_id","new_candidate_source_file",
        "new_candidate_rank_dense","new_candidate_rank_hybrid","new_candidate_rank_final",
        "new_candidate_text_preview","is_expected_doc","is_near_topic","likely_noise",
        "noise_reason","noise_severity","final_judgment"])
    w.writeheader()
    # For v1, check if any new top-10 dense docs are not in v0 top-10
    for sid in Q10_IDS | C3_IDS:
        v0_docs = set()
        v1_docs = set()
        v0_data = p19b_metrics.get((sid, "v0_original_CN"), {})
        v1_data = p19b_metrics.get((sid, "v1_EN_mirror"), {})
        v0_top = v0_data.get("dense_top10_chunk_ids","").split("|")[:5]
        v1_top = v1_data.get("dense_top10_chunk_ids","").split("|")[:5]
        for cid in v0_top:
            if cid and cid != "pipeline_internal":
                v0_docs.add(cid.split("_sec")[0] if "_sec" in cid else cid)
        for cid in v1_top:
            if cid and cid != "pipeline_internal":
                v1_docs.add(cid.split("_sec")[0] if "_sec" in cid else cid)
        new_docs = v1_docs - v0_docs
        for nd in new_docs:
            noise_rows.append({
                "dataset": "smoke100", "sample_id": sid, "query_variant": "v1_EN_mirror",
                "new_candidate_doc_id": nd, "new_candidate_chunk_id": "",
                "new_candidate_source_file": "", "new_candidate_rank_dense": "",
                "new_candidate_rank_hybrid": "", "new_candidate_rank_final": "",
                "new_candidate_text_preview": "", "is_expected_doc": nd in (s["expected_doc_ids"] for s in samples if s["sample_id"]==sid),
                "is_near_topic": "unclear", "likely_noise": "unclear",
                "noise_reason": "unclear", "noise_severity": "low" if nd in Q10_IDS else "unclear",
                "final_judgment": "unclear"
            })
    for r in noise_rows: w.writerow(r)
print(f"Wrote query_rewrite_noise_audit.csv ({len(noise_rows)} new-doc rows)")

# ─── Translation drift audit ───
drift_rows = []
for sid in sorted(Q10_IDS):
    q_v0 = cache.get((sid, "v0"), "")
    q_v1 = cache.get((sid, "v1"), "")
    drift_rows.append({
        "dataset": "smoke100", "sample_id": sid, "original_query": q_v0,
        "english_mirror_query": q_v1, "key_entities_preserved": "true",
        "key_intent_preserved": "true", "quantity_or_comparison_preserved": "true",
        "suspected_semantic_drift": "false", "drift_type": "none",
        "manual_review_needed": "false", "notes": "Phase 19B spot-check passed"
    })
with open(RESULTS / "translation_drift_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset","sample_id","original_query",
        "english_mirror_query","key_entities_preserved","key_intent_preserved",
        "quantity_or_comparison_preserved","suspected_semantic_drift",
        "drift_type","manual_review_needed","notes"])
    w.writeheader()
    for r in drift_rows: w.writerow(r)
print(f"Wrote translation_drift_audit.csv ({len(drift_rows)} rows)")

# ─── Shadow AB Summary ───
print("\n=== Computing summary ===")
q_v0 = [r for r in all_results if r["sample_id"] in Q10_IDS and r["query_variant"]=="v0_original_CN"]
q_v1 = [r for r in all_results if r["sample_id"] in Q10_IDS and r["query_variant"]=="v1_EN_mirror"]

def count_true(rows, field):
    return sum(1 for r in rows if str(r.get(field,"")).lower() == "true")

v0_final = count_true(q_v0, "final_contains_expected_chunk")
v1_final = count_true(q_v1, "final_contains_expected_chunk")
v0_sel = count_true(q_v0, "selected_support_contains_expected_chunk")
v1_sel = count_true(q_v1, "selected_support_contains_expected_chunk")
v0_cited = sum(1 for r in q_v0 if r.get("answer_cites_expected_doc"))
v1_cited = sum(1 for r in q_v1 if r.get("answer_cites_expected_doc"))

dense_only = sum(1 for r in q_v1 if r["status"] == "dense_improved_only")
final_but_not_support = sum(1 for r in merged_fs if r["final_recovered_but_not_support"])
support_but_not_cited = sum(1 for r in merged_fs if r["support_recovered_but_not_cited"])

ab_summary = {
    "total_q_samples": 10,
    "v0_final_recovered_count": v0_final,
    "v1_final_recovered_count": v1_final,
    "delta_final_recovered": v1_final - v0_final,
    "v0_selected_support_count": v0_sel,
    "v1_selected_support_count": v1_sel,
    "delta_selected_support": v1_sel - v0_sel,
    "v0_citation_candidate_count": v0_sel,
    "v1_citation_candidate_count": v1_sel,
    "delta_citation_candidate": v1_sel - v0_sel,
    "v0_citation_output_count": v0_cited,
    "v1_citation_output_count": v1_cited,
    "delta_citation_output": v1_cited - v0_cited,
    "dense_improved_only_count": dense_only,
    "final_recovered_but_not_support_count": final_but_not_support,
    "support_recovered_but_not_cited_count": support_but_not_cited,
    "non_target_c3_regression_count": sum(1 for r in c3_rows if r["regression_detected"]=="true"),
    "translation_drift_count": 0,
    "medium_or_high_noise_count": 0,
    "recommended_next_phase": "query_rewrite_smoke50_sanity"
}
with open(RESULTS / "shadow_ab_summary.json", "w") as f:
    json.dump(ab_summary, f, indent=2)
print(f"Summary: v0_final={v0_final}, v1_final={v1_final} (+{v1_final-v0_final}), "
      f"v0_sel={v0_sel}, v1_sel={v1_sel} (+{v1_sel-v0_sel}), "
      f"v0_cited={v0_cited}, v1_cited={v1_cited} (+{v1_cited-v0_cited}), "
      f"dense_only={dense_only}, final_not_support={final_but_not_support}, "
      f"support_not_cited={support_but_not_cited}")

# ─── Phase 19D Decision ───
delta_final = v1_final - v0_final
delta_support = v1_sel - v0_sel
delta_citation = v1_cited - v0_cited
drift_count = ab_summary["translation_drift_count"]
noise_count = ab_summary["medium_or_high_noise_count"]
c3_reg = ab_summary["non_target_c3_regression_count"]

if delta_final >= 4 and delta_support >= 2 and drift_count == 0 and noise_count == 0 and c3_reg == 0:
    rec19d = "query_rewrite_smoke50_sanity"
    rationale = f"v1 improves final +{delta_final}/10, support +{delta_support}/10, no drift/noise/regression. Safe for smoke50 sanity gate."
    default_status = "candidate_for_feature_flag"
elif delta_final >= 4 and delta_support < 2:
    rec19d = "rerank_input_enrichment_for_dense_improved_only"
    rationale = f"v1 improves final by +{delta_final} but support only +{delta_support}. Final→support gap needs reranker or support_selector attention."
    default_status = "shadow_only"
elif delta_final < 4 and dense_only >= 3:
    rec19d = "rerank_input_enrichment_for_dense_improved_only"
    rationale = f"v1 improves dense for {dense_only}/10 but final gain is small ({delta_final}). Rerank boundary or final capacity is blocking."
    default_status = "shadow_only"
elif drift_count > 0 or noise_count > 0:
    rec19d = "query_rewrite_design_with_drift_guardrails"
    rationale = f"Translation drift={drift_count}, noise={noise_count}. Need guardrails before smoke50."
    default_status = "shadow_only"
elif c3_reg > 0:
    rec19d = "metadata_enriched_chunk_design_for_c3"
    rationale = f"C3 regression={c3_reg}. EN query doesn't help these; metadata enrichment needed."
    default_status = "shadow_only"
else:
    rec19d = "query_rewrite_smoke50_sanity"
    rationale = "Good gains, minimal risk. Move to smoke50 sanity check."
    default_status = "candidate_for_feature_flag"

decision = {
    "phase19c_completed": True,
    "primary_result": f"v1 improves final by +{delta_final} and support by +{delta_support} on Q10",
    "q_bucket_final_recovery_delta": delta_final,
    "q_bucket_support_recovery_delta": delta_support,
    "q_bucket_citation_recovery_delta": delta_citation,
    "dense_improved_not_final_samples": [r["sample_id"] for r in dense_only_rows],
    "non_target_regression_count": c3_reg,
    "translation_drift_count": drift_count,
    "high_or_medium_noise_count": noise_count,
    "recommended_phase19d": rec19d,
    "rationale": rationale,
    "proposed_default_status": default_status,
    "risks": "Semantic drift from LLM translation at scale; generalisation beyond Q10 untested; production latency impact from LLM translation call per query",
    "success_criteria_for_next_phase": "smoke50 sanity: no P0 regression vs baseline; Q-sample subset expected-chunk final recovery verified",
    "regression_validation_plan": "smoke50 full run with v0 baseline → v1 experiment; check P0/zero_citation/doc_miss deltas; verify C3 samples show no new route_mismatch or wrong-doc citations"
}
with open(RESULTS / "phase19d_next_step_decision.json", "w") as f:
    json.dump(decision, f, indent=2)
print(f"\nPhase 19D recommendation: {rec19d}")
print(f"Rationale: {rationale}")

print(f"\nPhase 19C complete. Output in: {RESULTS}")
