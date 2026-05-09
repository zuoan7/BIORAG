#!/usr/bin/env python3
"""
Phase 19B: Cross-Lingual Query/Rerank Focused Audit
Read-only — no pipeline changes, no config changes, no index rebuild, no model swap.
"""
import csv, json, hashlib, itertools, os, sys, time, math
from collections import defaultdict
from pathlib import Path

# Add project root
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from openai import OpenAI
from src.synbio_rag.domain.config import Settings, RetrievalConfig, KnowledgeBaseConfig
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever, reciprocal_rank_fusion_multi
from src.synbio_rag.infrastructure.reranker.local_bge import LocalBGEReranker
from src.synbio_rag.domain.schemas import RetrievedChunk

RESULTS_DIR = PROJECT / "results" / "phase19b_cross_lingual_audit"
REPORTS_DIR = PROJECT / "reports" / "phase19b_cross_lingual_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ───
SETTINGS = Settings.from_env()
KC = SETTINGS.kb
RC = SETTINGS.retrieval

# Force defaults per Phase 19B constraints
RC.source_floor_enabled = True
RC.source_floor_dense_top_n = 3
RC.source_floor_bm25_top_n = 3
RC.source_floor_max_candidates_total = 6
RC.search_limit = 40
RC.dense_limit = 40
RC.bm25_limit = 40
RC.rerank_top_k = 10
RC.final_top_k = 8
RC.rrf_k = 60
RC.dense_rrf_weight = 1.0
RC.bm25_rrf_weight = 1.0

DENSE_TOP = 40
BM25_TOP = 40
HYBRID_TOP = 40
RERANK_TOP_K = 10
FINAL_TOP_K = 8

# ─── Samples ───
FOCUSED_SAMPLES = [
    # hard_recall_miss (11)
    {"dataset":"smoke100","sample_id":"ent_010","failure_source":"hard_recall_miss","question":"比较 6′-SL 与 2′-FL 两类 HMO 在当前文库里的工程化合成路径思路。请分别说明它们依赖的关键前体和末端催化步骤。","expected_doc_ids":["doc_0009","doc_0073"],"expected_route":"comparison"},
    {"dataset":"smoke100","sample_id":"ent_054","failure_source":"hard_recall_miss","question":"文库中酿酒酵母超甘露糖基化破坏工作里，蛋白分泌增强的主要机制是什么？","expected_doc_ids":["doc_0071"],"expected_route":"factoid"},
    {"dataset":"smoke100","sample_id":"ent_057","failure_source":"hard_recall_miss","question":"总结文库中降低鼠李糖代谢通量提高毕赤酵母目标蛋白产量的策略。","expected_doc_ids":["doc_0087"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_058","failure_source":"hard_recall_miss","question":"总结文库中利用两种酵母工程化生产 2′-FL 的关键突破。","expected_doc_ids":["doc_0098"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_064","failure_source":"hard_recall_miss","question":"总结文库中毕赤酵母从甲醇生产苹果酸的代谢通量重分配策略。","expected_doc_ids":["doc_0114"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_075","failure_source":"hard_recall_miss","question":"总结文库中多酶级联反应工程化合成 3′-唾液酸乳糖的策略。","expected_doc_ids":["doc_0146"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_081","failure_source":"hard_recall_miss","question":"总结文库中 E. coli BL21(DE3) 中 2′-FL 生产的途径优化策略。","expected_doc_ids":["doc_0151"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_083","failure_source":"hard_recall_miss","question":"比较文库中 E. coli 和 B. subtilis 作为 NeuAc 生产宿主的策略差异和产量表现。","expected_doc_ids":["doc_0119","doc_0147"],"expected_route":"comparison"},
    {"dataset":"smoke100","sample_id":"ent_096","failure_source":"hard_recall_miss","question":"文库中毕赤酵母 LRA4 低表达导致重组蛋白产量提高的机制是什么？","expected_doc_ids":["doc_0113"],"expected_route":"factoid"},
    {"dataset":"smoke50","sample_id":"h50_sum_008","failure_source":"hard_recall_miss","question":"概括一项毕赤酵母双向启动子工程研究：作者如何构建和改造双向启动子库来平衡多基因共表达，这些设计在表达强度、表达比和通路调优上带来了哪些核心发现？","expected_doc_ids":["doc_0085"],"expected_route":"summary"},
    {"dataset":"smoke50","sample_id":"h50_mrn_003","failure_source":"hard_recall_miss","question":"用于构建 gmd-wcaG 与 zwf 过表达系统的宿主、载体和质粒大小分别是什么？","expected_doc_ids":["doc_0032"],"expected_route":"factoid"},
    # low_support_score (5)
    {"dataset":"smoke100","sample_id":"ent_005","failure_source":"low_support_score","question":"总结这篇 6′-sialyllactose 论文中为了提升产量，对宿主和代谢通路做了哪些关键工程化改造。","expected_doc_ids":["doc_0009"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_055","failure_source":"low_support_score","question":"总结文库中 E. coli 模块化增强生产 2′-FL 和 3-FL 的策略和最终产量。","expected_doc_ids":["doc_0081"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_060","failure_source":"low_support_score","question":"总结文库中利用多形拟杆菌 AGE 和 E. coli 醛缩酶全细胞生产 Neu5Ac 的策略。","expected_doc_ids":["doc_0105"],"expected_route":"summary"},
    {"dataset":"smoke100","sample_id":"ent_100","failure_source":"low_support_score","question":"总结文库中环状排列荧光蛋白指示器的设计原理是什么？它如何实现配体检测？","expected_doc_ids":["doc_0090"],"expected_route":"factoid"},
    {"dataset":"smoke100","sample_id":"ent_082","failure_source":"low_support_score","question":"文库中 2′-FL 途径优化工作里，从头途径和补救途径分别包含哪些关键基因？","expected_doc_ids":["doc_0151"],"expected_route":"factoid"},
]

# Phase 18D answer-bearing chunk IDs (primary evidence target per expected doc)
# Selected from answer_bearing_chunk_audit.csv — best evidence-bearing chunk for each expected doc
EXPECTED_CHUNK_MAP = {
    ("ent_010","doc_0009"): {"chunk_id":"doc_0009_sec07_chunk08","section":"Results and Discussion","strength":"partial","source":"phase18d"},
    ("ent_010","doc_0073"): {"chunk_id":"doc_0073_sec05_chunk06","section":"Results","strength":"partial","source":"phase18d"},
    ("ent_054","doc_0071"): {"chunk_id":"doc_0071_sec03_chunk04","section":"Results","strength":"partial","source":"phase18d"},
    ("ent_057","doc_0087"): {"chunk_id":"doc_0087_sec05_chunk06","section":"Results","strength":"partial","source":"phase18d"},
    ("ent_058","doc_0098"): {"chunk_id":"doc_0098_sec11_chunk12","section":"Discussion","strength":"partial","source":"phase18d"},
    ("ent_064","doc_0114"): {"chunk_id":"doc_0114_sec14_chunk15","section":"Discussion","strength":"partial","source":"phase18d"},
    ("ent_075","doc_0146"): {"chunk_id":"doc_0146_sec12_chunk13","section":"Results and Discussion","strength":"strong","source":"phase18d"},
    ("ent_081","doc_0151"): {"chunk_id":"doc_0151_sec01_chunk02","section":"Full Text","strength":"partial","source":"phase18d"},
    ("ent_083","doc_0119"): {"chunk_id":"doc_0119_sec06_chunk07","section":"Full Text","strength":"partial","source":"phase18d"},
    ("ent_083","doc_0147"): {"chunk_id":"doc_0147_sec06_chunk07","section":"Full Text","strength":"partial","source":"phase18d"},
    ("ent_096","doc_0113"): {"chunk_id":"doc_0113_sec02_chunk03","section":"Introduction","strength":"partial","source":"phase18d"},
    ("h50_sum_008","doc_0085"): {"chunk_id":"doc_0085_sec00_chunk01","section":"Title","strength":"partial","source":"phase18d"},
    ("h50_mrn_003","doc_0032"): {"chunk_id":"doc_0032_sec03_chunk04","section":"Materials and Methods","strength":"partial","source":"phase18d"},
}

# For low_support_score samples, expected doc IS in final_chunks.
# We derive the expected chunk as the best answer-bearing chunk in the expected doc.
# From Phase 18D data (answer_bearing_chunk_audit.csv) and chunk text inspection:
EXPECTED_CHUNK_MAP_LOW_SCORE = {
    ("ent_005","doc_0009"): {"chunk_id":"doc_0009_sec08_chunk09","section":"Results and Discussion","strength":"partial","source":"fallback_expected_doc_best_chunk"},
    ("ent_055","doc_0081"): {"chunk_id":"doc_0081_sec04_chunk05","section":"Full Text","strength":"partial","source":"fallback_expected_doc_best_chunk"},
    ("ent_060","doc_0105"): {"chunk_id":"doc_0105_sec06_chunk07","section":"Full Text","strength":"partial","source":"fallback_expected_doc_best_chunk"},
    ("ent_100","doc_0090"): {"chunk_id":"doc_0090_sec02_chunk03","section":"Full Text","strength":"partial","source":"fallback_expected_doc_best_chunk"},
    ("ent_082","doc_0151"): {"chunk_id":"doc_0151_sec01_chunk02","section":"Full Text","strength":"partial","source":"fallback_expected_doc_best_chunk"},
}

# ─── Initialize Components ───
print("Loading embedder...")
embedder = BGEM3Embedder(model_path=KC.embedding_model_path, dim=KC.embedding_dim)

print("Loading dense retriever...")
dense = MilvusRetriever(RC, embedder)

print("Loading BM25 retriever...")
bm25 = BM25Retriever(RC, KC)

print("Loading hybrid retriever...")
hybrid = HybridRetriever(RC, dense, bm25)

print("Loading reranker...")
reranker = LocalBGEReranker(model_path=str(PROJECT / "models" / "BAAI" / "bge-reranker-v2-m3"))

# ─── Translation Cache ───
CACHE_PATH = RESULTS_DIR / "translation_cache.jsonl"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load existing cache
cache = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        for line in f:
            e = json.loads(line)
            cache[(e["sample_id"], e["variant_id"])] = e

from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")
QWEN_KEY = os.environ.get("QWEN_CHAT_API_KEY","")
QWEN_URL = os.environ.get("QWEN_CHAT_API_BASE","")
LLM_CLIENT = OpenAI(api_key=QWEN_KEY, base_url=QWEN_URL) if QWEN_KEY and QWEN_URL else None
if not LLM_CLIENT:
    print("WARNING: No Qwen API configured. Please set QWEN_CHAT_API_KEY and QWEN_CHAT_API_BASE in .env")
    sys.exit(1)

def translate_cn_to_en(query):
    prompt = f'''Translate this Chinese biology research query into a precise English retrieval query. Preserve all scientific terms (organism names, gene names, compound names like 2-FL, pathway names). Output only the English translation, no explanation.

Chinese query: {query}
English query:'''
    resp = LLM_CLIENT.chat.completions.create(model="qwen-plus", messages=[{"role":"user","content":prompt}], temperature=0, max_tokens=200)
    return resp.choices[0].message.content.strip()

def decompose_entities(query_en):
    prompt = f'''Extract the key scientific entities from this biology query as a space-separated English term list. Include: organism names, gene/protein names, compound/product names, pathway names, method/technique names. Output only the term list, no explanation.

Query: {query_en}
Term list:'''
    resp = LLM_CLIENT.chat.completions.create(model="qwen-plus", messages=[{"role":"user","content":prompt}], temperature=0, max_tokens=200)
    return resp.choices[0].message.content.strip()

def get_translated_variants(sample_id, query):
    variants = {}
    for vid, vname in [("v0","original CN"), ("v1","EN-mirror"), ("v2","bilingual"), ("v3","entity list")]:
        cache_key = (sample_id, vid)
        if cache_key in cache:
            variants[vid] = cache[cache_key]["generated_query"]
        else:
            variants[vid] = None
    return variants

def generate_and_cache_variants(sample_id, query):
    """Generate v1, v2, v3 from v0 and cache them."""
    results = {}
    for vid, vname in [("v0","original CN"), ("v1","EN-mirror"), ("v2","bilingual"), ("v3","entity list")]:
        cache_key = (sample_id, vid)
        if cache_key in cache:
            results[vid] = cache[cache_key]["generated_query"]
            continue
        prompt = ""
        temp = 0.0
        if vid == "v0":
            gen = query
            prompt = "original_query"
        elif vid == "v1":
            gen = translate_cn_to_en(query)
            prompt = f"Translate Chinese→English retrieval query.\nQuery: {query}"
        elif vid == "v2":
            v0 = query
            v1 = results.get("v1") or translate_cn_to_en(query)
            gen = f"{v0} | {v1}"
            prompt = "bilingual concatenation of v0 + v1"
        elif vid == "v3":
            v1_en = results.get("v1") or translate_cn_to_en(query)
            gen = decompose_entities(v1_en)
            prompt = f"Extract entity/term list from query.\nEN query: {v1_en}"
        results[vid] = gen
        row = {
            "sample_id": sample_id, "variant_id": vid, "variant_name": vname,
            "original_query": query, "generated_query": gen,
            "translation_model": "qwen-plus", "translation_prompt": prompt[:500],
            "generation_temperature": temp, "generation_method": "llm_translation",
            "prompt_hash": hashlib.sha256(prompt[:500].encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(gen.encode()).hexdigest()[:16],
            "manual_spot_check_status": "pending", "suspected_semantic_drift": False, "notes": ""
        }
        cache[cache_key] = row
    return results

print("\nGenerating translations...")
for s in FOCUSED_SAMPLES:
    sid = s["sample_id"]
    q = s["question"]
    vars_ = generate_and_cache_variants(sid, q)
    s["variants"] = vars_
    print(f"  {sid}: v1={vars_['v1'][:80]}...")

# Save cache
with open(CACHE_PATH, "w") as f:
    for (_, _), row in cache.items():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Translation cache saved: {CACHE_PATH}")

# ─── Load chunk text lookup ───
print("\nLoading chunk text lookup...")
chunk_text_map = {}
chunk_doc_map = {}
chunk_section_map = {}
chunk_title_map = {}
chunk_source_map = {}
with open(KC.chunk_jsonl) as f:
    for line in f:
        c = json.loads(line)
        cid = c["chunk_id"]
        chunk_text_map[cid] = c.get("text","")
        chunk_doc_map[cid] = c.get("doc_id","")
        chunk_section_map[cid] = c.get("section","")
        chunk_title_map[cid] = c.get("title","")
        chunk_source_map[cid] = c.get("source_file","")
print(f"Loaded {len(chunk_text_map)} chunks.")

# ─── Audit Functions ───

def results_to_list(results):
    """Convert RetrievedChunk results to list of dicts."""
    out = []
    for r in results:
        cid = getattr(r, 'chunk_id', '')
        did = getattr(r, 'doc_id', '')
        score = getattr(r, 'score', 0) or getattr(r, 'fusion_score', 0)
        out.append({"chunk_id": cid, "doc_id": did, "score": score, "obj": r})
    return out

def rank_of_doc(doc_id, results_list):
    """Find best rank of any chunk from given doc_id."""
    for i, r in enumerate(results_list):
        if r["doc_id"] == doc_id:
            return i + 1, r["chunk_id"], r["score"]
    return None, None, None

def rank_of_chunk(chunk_id, results_list):
    for i, r in enumerate(results_list):
        if r["chunk_id"] == chunk_id:
            return i + 1, r["score"]
    return None, None

def found_in_top(doc_id, results_list, top_n):
    for r in results_list[:top_n]:
        if r["doc_id"] == doc_id:
            return True
    return False

def found_chunk_in_top(chunk_id, results_list, top_n):
    for r in results_list[:top_n]:
        if r["chunk_id"] == chunk_id:
            return True
    return False

def get_rerank_input(hybrid_results, top_k):
    """Extract candidates that would enter reranker."""
    return hybrid_results[:top_k]

def score_rerank_pairs(query, chunks, batch_size=16):
    """Score (query, chunk_text) pairs with local BGE reranker."""
    scores = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        pairs = [[query, c.get("text","")] for c in batch]
        batch_scores = reranker.score_pairs(pairs)
        scores.extend(batch_scores)
    return scores

def classify_top_wrong(pattern_text):
    if not pattern_text:
        return "unclear"
    t = pattern_text.lower()
    if "same_organism" in t or "pichia" in t and "wrong_product" in t:
        return "same_organism_wrong_product"
    return "unclear"

# ─── Main Audit Loop ───

print("\n" + "="*60)
print("Running per-sample audit...")
print("="*60)

all_metrics = []
oracle_diag_rows = []
bucket_rows = []

for s in FOCUSED_SAMPLES:
    sid = s["sample_id"]
    dset = s["dataset"]
    fsrc = s["failure_source"]
    expected_docs = s["expected_doc_ids"]
    variants = s["variants"]

    # Determine primary expected chunk
    primary_chunk_id = None
    primary_section = None
    primary_strength = None
    primary_source = None
    for doc_id in expected_docs:
        key = (sid, doc_id)
        if fsrc == "hard_recall_miss" and key in EXPECTED_CHUNK_MAP:
            ec = EXPECTED_CHUNK_MAP[key]
            primary_chunk_id = ec["chunk_id"]
            primary_section = ec["section"]
            primary_strength = ec["strength"]
            primary_source = ec["source"]
            break
        elif fsrc == "low_support_score" and key in EXPECTED_CHUNK_MAP_LOW_SCORE:
            ec = EXPECTED_CHUNK_MAP_LOW_SCORE[key]
            primary_chunk_id = ec["chunk_id"]
            primary_section = ec["section"]
            primary_strength = ec["strength"]
            primary_source = ec["source"]
            break
    if not primary_chunk_id:
        # Fallback: use first expected doc's chunk from chunk map
        for doc_id in expected_docs:
            key = (sid, doc_id)
            if key in EXPECTED_CHUNK_MAP:
                ec = EXPECTED_CHUNK_MAP[key]
                primary_chunk_id = ec["chunk_id"]
                primary_section = ec["section"]
                primary_strength = ec.get("strength","unclear")
                primary_source = ec.get("source","fallback")
                break
    if not primary_chunk_id:
        primary_chunk_id = "unknown"
        primary_section = "unknown"
        primary_strength = "unclear"
        primary_source = "fallback_unknown"

    expected_chunk_text = chunk_text_map.get(primary_chunk_id, "")[:200] if primary_chunk_id != "unknown" else ""
    expected_source_file = chunk_source_map.get(primary_chunk_id, "")

    variants_data = {}

    for vid in ["v0","v1","v2","v3"]:
        qtext = variants[vid]
        print(f"  [{sid}/{vid}] dense...", end="", flush=True)

        # ── Dense ──
        try:
            dense_results = dense.search(qtext, limit=DENSE_TOP)
        except Exception as e:
            print(f"dense error: {e}")
            dense_results = []
        dr_list = results_to_list(dense_results)
        d_doc_top10 = found_in_top(expected_docs[0], dr_list, 10) if expected_docs else False
        d_doc_top20 = found_in_top(expected_docs[0], dr_list, 20) if expected_docs else False
        d_doc_top40 = found_in_top(expected_docs[0], dr_list, 40) if expected_docs else False
        d_ck_top10 = found_chunk_in_top(primary_chunk_id, dr_list, 10)
        d_ck_top20 = found_chunk_in_top(primary_chunk_id, dr_list, 20)
        d_ck_top40 = found_chunk_in_top(primary_chunk_id, dr_list, 40)
        d_doc_rank, d_doc_best_chunk, _ = rank_of_doc(expected_docs[0], dr_list) if expected_docs else (None,None,None)
        d_ck_rank, d_ck_score = rank_of_chunk(primary_chunk_id, dr_list)
        d_top10_ids = [r["chunk_id"] for r in dr_list[:10]]
        d_top10_scores = [round(r["score"],6) for r in dr_list[:10]]

        print(" bm25...", end="", flush=True)
        # ── BM25 ──
        try:
            bm25_results = bm25.search(qtext, limit=BM25_TOP)
        except Exception as e:
            print(f"bm25 error: {e}")
            bm25_results = []
        br_list = results_to_list(bm25_results)
        b_doc_top10 = found_in_top(expected_docs[0], br_list, 10) if expected_docs else False
        b_doc_top20 = found_in_top(expected_docs[0], br_list, 20) if expected_docs else False
        b_doc_top40 = found_in_top(expected_docs[0], br_list, 40) if expected_docs else False
        b_ck_top10 = found_chunk_in_top(primary_chunk_id, br_list, 10)
        b_ck_top20 = found_chunk_in_top(primary_chunk_id, br_list, 20)
        b_ck_top40 = found_chunk_in_top(primary_chunk_id, br_list, 40)
        b_doc_rank, b_doc_best_chunk, _ = rank_of_doc(expected_docs[0], br_list) if expected_docs else (None,None,None)
        b_ck_rank, b_ck_score = rank_of_chunk(primary_chunk_id, br_list)
        b_top10_ids = [r["chunk_id"] for r in br_list[:10]]
        b_top10_scores = [round(r["score"],6) for r in br_list[:10]]
        b_query_tokens = len(tokenize_query(qtext))

        print(" hybrid...", end="", flush=True)
        # ── Hybrid (RRF + source-floor) ──
        # Reconstruct hybrid without alias expansion, CJK awareness etc
        # Simple RRF: dense + BM25 only
        dense_rrf = [(dense_results, RC.dense_rrf_weight)]
        sparse_rrf = [(bm25_results, RC.bm25_rrf_weight)]
        try:
            fused = reciprocal_rank_fusion_multi(dense_rrf, sparse_rrf, limit=HYBRID_TOP, rrf_k=RC.rrf_k)
        except Exception as e:
            print(f"RRF error: {e}", end="")
            fused = []
        hr_list = results_to_list(fused)
        h_doc_top10 = found_in_top(expected_docs[0], hr_list, 10) if expected_docs else False
        h_doc_top20 = found_in_top(expected_docs[0], hr_list, 20) if expected_docs else False
        h_doc_top40 = found_in_top(expected_docs[0], hr_list, 40) if expected_docs else False
        h_ck_top10 = found_chunk_in_top(primary_chunk_id, hr_list, 10)
        h_ck_top20 = found_chunk_in_top(primary_chunk_id, hr_list, 20)
        h_ck_top40 = found_chunk_in_top(primary_chunk_id, hr_list, 40)
        h_rank_pre_floor, _ = rank_of_chunk(primary_chunk_id, hr_list)

        # Source-floor: inject expected-doc's dense/BM25 top-3 if not in fused
        sf_added_doc = False
        sf_added_chunk = False
        added_ids = set()
        if RC.source_floor_enabled:
            floor_candidates = []
            for r in dense_results[:RC.source_floor_dense_top_n]:
                cid = getattr(r, 'chunk_id', '')
                if cid not in {x["chunk_id"] for x in hr_list} and cid not in added_ids:
                    floor_candidates.append(("dense_floor", r))
                    added_ids.add(cid)
            for r in bm25_results[:RC.source_floor_bm25_top_n]:
                cid = getattr(r, 'chunk_id', '')
                if cid not in {x["chunk_id"] for x in hr_list} and cid not in added_ids:
                    if len(floor_candidates) < RC.source_floor_max_candidates_total:
                        floor_candidates.append(("bm25_floor", r))
                        added_ids.add(cid)
            for src_type, r in floor_candidates:
                cid = getattr(r, 'chunk_id', '')
                did = getattr(r, 'doc_id', '')
                if did in expected_docs:
                    sf_added_doc = True
                if cid == primary_chunk_id:
                    sf_added_chunk = True
                hr_list.append({"chunk_id":cid, "doc_id":did, "score":0.0, "obj":r, "source_floor":src_type})

        h_rank_post_floor, _ = rank_of_chunk(primary_chunk_id, hr_list)

        # ── Pipeline Rerank ──
        print(" rerank...", end="", flush=True)
        rerank_candidates = hr_list[:RERANK_TOP_K]
        pipeline_expected_in_rerank = primary_chunk_id in {r["chunk_id"] for r in rerank_candidates}
        # Collect chunk texts for reranker
        rerank_chunk_texts = [chunk_text_map.get(r["chunk_id"],"")[:800] for r in rerank_candidates]
        # Score with reranker
        rerank_query = qtext
        try:
            rerank_scores = score_rerank_pairs(rerank_query, [{"text": t} for t in rerank_chunk_texts])
        except Exception as e:
            print(f"rerank error: {e}", end="")
            rerank_scores = [0.0]*len(rerank_chunk_texts)
        # Sort by rerank score
        rank_entries = []
        for i, sc in enumerate(rerank_scores):
            rank_entries.append({**rerank_candidates[i], "rerank_score": round(sc, 6)})
        rank_entries.sort(key=lambda x: x["rerank_score"], reverse=True)

        pr_rank = None
        pr_score = None
        pr_top_competing = None
        pr_top_comp_score = None
        for i, e in enumerate(rank_entries):
            if e["chunk_id"] == primary_chunk_id:
                pr_rank = i + 1
                pr_score = e["rerank_score"]
                break
        # Top competing wrong chunk (different doc)
        for e in rank_entries:
            if e["doc_id"] not in expected_docs:
                pr_top_competing = e["chunk_id"]
                pr_top_comp_score = e["rerank_score"]
                break

        # ── Oracle Rerank ──
        oracle_set = list(rerank_candidates)
        if not pipeline_expected_in_rerank and primary_chunk_id != "unknown":
            oracle_set.append({"chunk_id":primary_chunk_id, "doc_id":primary_chunk_id.split("_sec")[0] if "_sec" in primary_chunk_id else (expected_docs[0] if expected_docs else "unknown"), "score":0.0, "obj":None})
        oracle_texts = [chunk_text_map.get(r["chunk_id"],"")[:800] for r in oracle_set]
        try:
            oracle_scores = score_rerank_pairs(rerank_query, [{"text": t} for t in oracle_texts])
        except Exception as e:
            print(f"oracle rerank error: {e}", end="")
            oracle_scores = [0.0]*len(oracle_texts)
        oracle_entries = []
        for i, sc in enumerate(oracle_scores):
            oracle_entries.append({**oracle_set[i], "rerank_score": round(sc, 6)})
        oracle_entries.sort(key=lambda x: x["rerank_score"], reverse=True)

        oracle_rank = None
        oracle_score = None
        oracle_top_comp = None
        oracle_top_comp_score = None
        for i, e in enumerate(oracle_entries):
            if e["chunk_id"] == primary_chunk_id:
                oracle_rank = i + 1
                oracle_score = e["rerank_score"]
                break
        for e in oracle_entries:
            if e["doc_id"] not in expected_docs and e["chunk_id"] != primary_chunk_id:
                oracle_top_comp = e["chunk_id"]
                oracle_top_comp_score = e["rerank_score"]
                break
        oracle_margin = round(oracle_score - oracle_top_comp_score, 6) if oracle_score is not None and oracle_top_comp_score is not None else None
        oracle_judgment = "unclear"
        if oracle_rank is not None and oracle_rank <= 3:
            oracle_judgment = "reranker_can_recognize"
        elif oracle_rank is not None:
            oracle_judgment = "reranker_under_ranks_expected"

        # ── Final/Support ──
        final_chunks = rank_entries[:FINAL_TOP_K]
        final_has_doc = any(e["doc_id"] in expected_docs for e in final_chunks) if expected_docs else False
        final_has_chunk = primary_chunk_id in {e["chunk_id"] for e in final_chunks}
        final_rank = None
        for i, e in enumerate(final_chunks):
            if e["chunk_id"] == primary_chunk_id:
                final_rank = i + 1
                break
        # For support: we'd need the full support_selector, which requires QueryAnalysis.
        # As a proxy: check if the expected chunk would be in selected_support (top-3 by rerank_score in final).
        support_has_chunk = final_has_chunk and (final_rank is not None and final_rank <= 3)
        support_score_proxy = pr_score
        support_rank_proxy = final_rank

        print(" done")

        # ── Build metrics row ──
        row = {
            "dataset": dset, "sample_id": sid, "failure_source": fsrc,
            "variant_id": vid, "query_text": qtext[:200],
            "expected_doc_ids": "|".join(expected_docs),
            "expected_chunk_id": primary_chunk_id,
            "expected_source_file": expected_source_file,
            "expected_section": primary_section,
            # Dense
            "dense_expected_doc_found_top10": d_doc_top10,
            "dense_expected_doc_found_top20": d_doc_top20,
            "dense_expected_doc_found_top40": d_doc_top40,
            "dense_expected_chunk_found_top10": d_ck_top10,
            "dense_expected_chunk_found_top20": d_ck_top20,
            "dense_expected_chunk_found_top40": d_ck_top40,
            "dense_rank_of_expected_doc_best_chunk": d_doc_rank or -1,
            "dense_rank_of_expected_primary_chunk": d_ck_rank or -1,
            "dense_score_of_expected_primary_chunk": d_ck_score if d_ck_score else 0.0,
            "dense_top10_chunk_ids": "|".join(d_top10_ids[:5]),
            "dense_top10_scores": "|".join(str(x) for x in d_top10_scores[:5]),
            # BM25
            "bm25_expected_doc_found_top10": b_doc_top10,
            "bm25_expected_doc_found_top20": b_doc_top20,
            "bm25_expected_doc_found_top40": b_doc_top40,
            "bm25_expected_chunk_found_top10": b_ck_top10,
            "bm25_expected_chunk_found_top20": b_ck_top20,
            "bm25_expected_chunk_found_top40": b_ck_top40,
            "bm25_rank_of_expected_doc_best_chunk": b_doc_rank or -1,
            "bm25_rank_of_expected_primary_chunk": b_ck_rank or -1,
            "bm25_score_of_expected_primary_chunk": b_ck_score if b_ck_score else 0.0,
            "bm25_query_tokens": b_query_tokens,
            "bm25_top10_chunk_ids": "|".join(b_top10_ids[:5]),
            "bm25_top10_scores": "|".join(str(x) for x in b_top10_scores[:5]),
            # Hybrid
            "hybrid_expected_doc_found_top10": h_doc_top10,
            "hybrid_expected_doc_found_top20": h_doc_top20,
            "hybrid_expected_doc_found_top40": h_doc_top40,
            "hybrid_expected_chunk_found_top10": h_ck_top10,
            "hybrid_expected_chunk_found_top20": h_ck_top20,
            "hybrid_expected_chunk_found_top40": h_ck_top40,
            "hybrid_rank_pre_source_floor": h_rank_pre_floor or -1,
            "hybrid_rank_post_source_floor": h_rank_post_floor or -1,
            "source_floor_added_expected_doc": sf_added_doc,
            "source_floor_added_expected_chunk": sf_added_chunk,
            # Pipeline rerank
            "pipeline_rerank_input_contains_expected_doc": any(r["doc_id"] in expected_docs for r in rerank_candidates) if expected_docs else False,
            "pipeline_rerank_input_contains_expected_chunk": pipeline_expected_in_rerank,
            "pipeline_rerank_rank_of_expected_chunk": pr_rank or -1,
            "pipeline_rerank_score_of_expected_chunk": pr_score if pr_score else 0.0,
            "pipeline_rerank_top_competing_wrong_chunk_id": pr_top_competing or "",
            "pipeline_rerank_top_competing_wrong_score": pr_top_comp_score if pr_top_comp_score else 0.0,
            # Oracle rerank
            "oracle_rerank_rank_of_expected_chunk": oracle_rank or -1,
            "oracle_rerank_score_of_expected_chunk": oracle_score if oracle_score else 0.0,
            "oracle_top_competitor_chunk_id": oracle_top_comp or "",
            "oracle_top_competitor_score": oracle_top_comp_score if oracle_top_comp_score else 0.0,
            "oracle_expected_vs_competitor_margin": oracle_margin if oracle_margin else 0.0,
            "oracle_rerank_judgment": oracle_judgment,
            # Final/Support
            "final_contains_expected_doc": final_has_doc,
            "final_contains_expected_chunk": final_has_chunk,
            "final_rank_of_expected_chunk": final_rank or -1,
            "support_input_contains_expected_chunk": final_has_chunk,
            "support_score_of_expected_chunk": support_score_proxy if support_score_proxy else 0.0,
            "support_rank_of_expected_chunk": support_rank_proxy or -1,
            "selected_support_contains_expected_chunk": support_has_chunk,
            # Noise/competitors
            "top_wrong_doc_pattern": "unclear",
            "notes": ""
        }
        all_metrics.append(row)
        variants_data[vid] = row

    # ── Oracle Rerank Diagnostics only for v0 ──
    v0_row = variants_data["v0"]
    # Build oracle diag for all variants
    for vid in ["v0","v1","v2","v3"]:
        vr = variants_data[vid]
        oracle_diag_rows.append({
            "dataset": dset, "sample_id": sid, "failure_source": fsrc,
            "variant_id": vid, "primary_expected_chunk_id": primary_chunk_id,
            "pipeline_expected_in_rerank": vr["pipeline_rerank_input_contains_expected_chunk"],
            "pipeline_rerank_rank": vr["pipeline_rerank_rank_of_expected_chunk"],
            "pipeline_rerank_score": vr["pipeline_rerank_score_of_expected_chunk"],
            "oracle_rerank_rank": vr["oracle_rerank_rank_of_expected_chunk"],
            "oracle_rerank_score": vr["oracle_rerank_score_of_expected_chunk"],
            "oracle_expected_vs_competitor_margin": vr["oracle_expected_vs_competitor_margin"],
            "oracle_judgment": vr["oracle_rerank_judgment"],
            "notes": ""
        })

    # ── Bucketing ──
    v0 = variants_data["v0"]
    v1 = variants_data["v1"]

    # Compute deltas
    d_score_delta = v1["dense_score_of_expected_primary_chunk"] - v0["dense_score_of_expected_primary_chunk"]
    d_rank_delta = v1["dense_rank_of_expected_primary_chunk"] - v0["dense_rank_of_expected_primary_chunk"] if v0["dense_rank_of_expected_primary_chunk"] > 0 and v1["dense_rank_of_expected_primary_chunk"] > 0 else 0
    r_score_delta = v1["pipeline_rerank_score_of_expected_chunk"] - v0["pipeline_rerank_score_of_expected_chunk"]
    r_rank_delta = v1["pipeline_rerank_rank_of_expected_chunk"] - v0["pipeline_rerank_rank_of_expected_chunk"] if v0["pipeline_rerank_rank_of_expected_chunk"] > 0 and v1["pipeline_rerank_rank_of_expected_chunk"] > 0 else 0
    final_recovery = (not v0["final_contains_expected_chunk"]) and v1["final_contains_expected_chunk"]

    # Update v1 row with deltas
    for vid, vr in [("v0",v0),("v1",v1),("v2",variants_data["v2"]),("v3",variants_data["v3"])]:
        # Find and update the row in all_metrics
        for row in all_metrics:
            if row["sample_id"]==sid and row["variant_id"]==vid:
                if vid=="v1":
                    row["dense_score_delta_v1_minus_v0"] = d_score_delta
                    row["dense_rank_delta_v1_minus_v0"] = d_rank_delta
                    row["rerank_score_delta_v1_minus_v0"] = r_score_delta
                    row["rerank_rank_delta_v1_minus_v0"] = r_rank_delta
                    row["final_recovery_delta_v1_minus_v0"] = final_recovery
                else:
                    row["dense_score_delta_v1_minus_v0"] = 0.0
                    row["dense_rank_delta_v1_minus_v0"] = 0
                    row["rerank_score_delta_v1_minus_v0"] = 0.0
                    row["rerank_rank_delta_v1_minus_v0"] = 0
                    row["final_recovery_delta_v1_minus_v0"] = False
                break

    # Determine primary bucket
    v1_d_rank_ok = v1["dense_rank_of_expected_primary_chunk"] > 0 and v1["dense_rank_of_expected_primary_chunk"] <= 40
    v0_d_rank_ok = v0["dense_rank_of_expected_primary_chunk"] > 0 and v0["dense_rank_of_expected_primary_chunk"] <= 40
    v1_final_has = v1["final_contains_expected_chunk"]
    v0_final_has = v0["final_contains_expected_chunk"]
    v0_oracle_ok = v0["oracle_rerank_rank_of_expected_chunk"] > 0 and v0["oracle_rerank_rank_of_expected_chunk"] <= 3
    v0_pipeline_seen = v0["pipeline_rerank_input_contains_expected_chunk"]
    v0_pipeline_rank_low = v0["pipeline_rerank_rank_of_expected_chunk"] > 3 or v0["pipeline_rerank_rank_of_expected_chunk"] <= 0

    # Check chunk text signal
    exp_text = chunk_text_map.get(primary_chunk_id, "")
    text_short = len(exp_text) < 200
    text_has_entity = any(w.lower() in exp_text.lower() for w in ["pichia","ecoli","e. coli","coli","yeast","2'-fl","2-fucosyllactose","sialyl","neu5ac","lra","promoter","expression","production","pathway","protein","gene","enzyme"])

    primary_bucket = "Mixed"
    chunk_signal_subtype = "none"
    prop_loss_stage = "none"
    evidence_str = ""

    # Bucket-D: dense still fails with v1
    if not v1_d_rank_ok and not v1_final_has:
        if text_has_entity and len(exp_text) >= 200:
            primary_bucket = "D"
            evidence_str = f"v1 dense rank {v1['dense_rank_of_expected_primary_chunk']}, but chunk has entities and length {len(exp_text)}"
        else:
            primary_bucket = "C"
            chunk_signal_subtype = "C1" if (len(exp_text) < 200 or not text_has_entity) else "C3"
            evidence_str = f"v1 dense rank {v1['dense_rank_of_expected_primary_chunk']}, chunk {'short/generic' if chunk_signal_subtype=='C1' else 'missing metadata context'}"
    # Bucket-Q: v1 closes the gap
    elif v1_final_has and not v0_final_has:
        primary_bucket = "Q"
        evidence_str = f"v1 recovers expected chunk to final (v0 final={v0_final_has}, v1 final={v1_final_has})"
    elif d_score_delta > 0.05:
        primary_bucket = "Q"
        evidence_str = f"v1 significantly improves dense score by {d_score_delta:.4f}"
    # Bucket-R: pipeline sees but reranker under-ranks
    elif v0_pipeline_seen and v0_pipeline_rank_low and not v0_oracle_ok:
        primary_bucket = "R"
        evidence_str = f"Pipeline sees expected chunk, reranker ranks it {v0['pipeline_rerank_rank_of_expected_chunk']}, oracle rank {v0['oracle_rerank_rank_of_expected_chunk']}"
    elif v0_pipeline_seen and v0_pipeline_rank_low and v0_oracle_ok:
        # Pipeline rerank low but oracle recognizes — weird, still R
        primary_bucket = "R"
        evidence_str = f"Pipeline rerank low ({v0['pipeline_rerank_rank_of_expected_chunk']}), oracle ok ({v0['oracle_rerank_rank_of_expected_chunk']}) → competitors dominate in pipeline set"
    # Bucket-C: all variants fail
    elif not v0_d_rank_ok and not v1_d_rank_ok:
        primary_bucket = "C"
        if text_short or not text_has_entity:
            chunk_signal_subtype = "C1"
        else:
            chunk_signal_subtype = "C3"
        evidence_str = f"No variant recovers; chunk signal {'weak' if chunk_signal_subtype=='C1' else 'missing metadata'}"
    # Default
    else:
        primary_bucket = "Mixed"
        evidence_str = f"Complex: d_rank={v0['dense_rank_of_expected_primary_chunk']}/{v1['dense_rank_of_expected_primary_chunk']}, final={v0_final_has}/{v1_final_has}, oracle={v0['oracle_rerank_judgment']}"

    # Secondary bucket: propagation failure
    secondary_bucket = "none"
    # Did an early stage find it but later stage lose it?
    found_early = (v0["dense_rank_of_expected_primary_chunk"] > 0 and v0["dense_rank_of_expected_primary_chunk"] <= 40) or (v0["bm25_rank_of_expected_primary_chunk"] > 0 and v0["bm25_rank_of_expected_primary_chunk"] <= 40)
    if found_early and not v0["hybrid_expected_chunk_found_top40"]:
        secondary_bucket = "P"
        prop_loss_stage = "dense_to_hybrid" if v0["dense_rank_of_expected_primary_chunk"] <= 40 else "bm25_to_hybrid"
    elif v0["hybrid_expected_chunk_found_top40"] and not v0["pipeline_rerank_input_contains_expected_chunk"]:
        secondary_bucket = "P"
        prop_loss_stage = "hybrid_to_rerank_input"
    elif v0["pipeline_rerank_input_contains_expected_chunk"] and not v0["final_contains_expected_chunk"]:
        secondary_bucket = "P"
        prop_loss_stage = "rerank_to_final"

    # Best variant
    best_variant = "none"
    best_variant_effect = "no_effect"
    if final_recovery:
        best_variant = "v1"
        best_variant_effect = "final_recovered"
    elif v1["dense_rank_of_expected_primary_chunk"] > 0 and v1["dense_rank_of_expected_primary_chunk"] < v0["dense_rank_of_expected_primary_chunk"]:
        best_variant = "v1"
        best_variant_effect = "dense_improved"
    elif v1["pipeline_rerank_rank_of_expected_chunk"] > 0 and (v0["pipeline_rerank_rank_of_expected_chunk"] <= 0 or v1["pipeline_rerank_rank_of_expected_chunk"] < v0["pipeline_rerank_rank_of_expected_chunk"]):
        best_variant = "v1"
        best_variant_effect = "rerank_improved"

    # Recommended next option
    recommended_next = "unclear"
    if primary_bucket == "Q":
        recommended_next = "Option_B_query_rewriting"
    elif primary_bucket == "R":
        recommended_next = "Option_E_rerank_input_or_calibration"
    elif primary_bucket == "C":
        if chunk_signal_subtype == "C2":
            recommended_next = "Option_C_doc_level_recall"
        else:
            recommended_next = "Option_D_metadata_enriched_index"
    elif primary_bucket == "D":
        recommended_next = "Dense_calibration_design"
    else:
        recommended_next = "unclear"

    bucket_rows.append({
        "dataset": dset, "sample_id": sid, "failure_source": fsrc,
        "primary_bucket": primary_bucket, "secondary_bucket": secondary_bucket,
        "bucket_confidence": "medium" if primary_bucket != "Mixed" else "low",
        "chunk_signal_subtype": chunk_signal_subtype,
        "propagation_loss_stage": prop_loss_stage,
        "best_variant": best_variant, "best_variant_effect": best_variant_effect,
        "evidence_for_bucket": evidence_str,
        "recommended_next_option": recommended_next,
        "notes": f"chunk_text_len={len(exp_text)}, has_entity={text_has_entity}"
    })

# ─── Write Outputs ───

# per_sample_metrics.csv
FIELD_ORDER = [
    "dataset","sample_id","failure_source","variant_id","query_text",
    "expected_doc_ids","expected_chunk_id","expected_source_file","expected_section",
    "dense_expected_doc_found_top10","dense_expected_doc_found_top20","dense_expected_doc_found_top40",
    "dense_expected_chunk_found_top10","dense_expected_chunk_found_top20","dense_expected_chunk_found_top40",
    "dense_rank_of_expected_doc_best_chunk","dense_rank_of_expected_primary_chunk",
    "dense_score_of_expected_primary_chunk","dense_top10_chunk_ids","dense_top10_scores",
    "bm25_expected_doc_found_top10","bm25_expected_doc_found_top20","bm25_expected_doc_found_top40",
    "bm25_expected_chunk_found_top10","bm25_expected_chunk_found_top20","bm25_expected_chunk_found_top40",
    "bm25_rank_of_expected_doc_best_chunk","bm25_rank_of_expected_primary_chunk",
    "bm25_score_of_expected_primary_chunk","bm25_query_tokens","bm25_top10_chunk_ids","bm25_top10_scores",
    "hybrid_expected_doc_found_top10","hybrid_expected_doc_found_top20","hybrid_expected_doc_found_top40",
    "hybrid_expected_chunk_found_top10","hybrid_expected_chunk_found_top20","hybrid_expected_chunk_found_top40",
    "hybrid_rank_pre_source_floor","hybrid_rank_post_source_floor",
    "source_floor_added_expected_doc","source_floor_added_expected_chunk",
    "pipeline_rerank_input_contains_expected_doc","pipeline_rerank_input_contains_expected_chunk",
    "pipeline_rerank_rank_of_expected_chunk","pipeline_rerank_score_of_expected_chunk",
    "pipeline_rerank_top_competing_wrong_chunk_id","pipeline_rerank_top_competing_wrong_score",
    "oracle_rerank_rank_of_expected_chunk","oracle_rerank_score_of_expected_chunk",
    "oracle_top_competitor_chunk_id","oracle_top_competitor_score",
    "oracle_expected_vs_competitor_margin","oracle_rerank_judgment",
    "final_contains_expected_doc","final_contains_expected_chunk","final_rank_of_expected_chunk",
    "support_input_contains_expected_chunk","support_score_of_expected_chunk",
    "support_rank_of_expected_chunk","selected_support_contains_expected_chunk",
    "dense_score_delta_v1_minus_v0","dense_rank_delta_v1_minus_v0",
    "rerank_score_delta_v1_minus_v0","rerank_rank_delta_v1_minus_v0",
    "final_recovery_delta_v1_minus_v0",
    "top_wrong_doc_pattern","notes"
]

metrics_path = RESULTS_DIR / "per_sample_metrics.csv"
with open(metrics_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELD_ORDER, extrasaction='ignore')
    w.writeheader()
    for row in all_metrics:
        w.writerow(row)
print(f"\nWrote {len(all_metrics)} rows to {metrics_path}")

# oracle_rerank_diagnostics.csv
oracle_path = RESULTS_DIR / "oracle_rerank_diagnostics.csv"
ORACLE_FIELDS = ["dataset","sample_id","failure_source","variant_id","primary_expected_chunk_id",
    "pipeline_expected_in_rerank","pipeline_rerank_rank","pipeline_rerank_score",
    "oracle_rerank_rank","oracle_rerank_score","oracle_expected_vs_competitor_margin",
    "oracle_judgment","notes"]
with open(oracle_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ORACLE_FIELDS, extrasaction='ignore')
    w.writeheader()
    for row in oracle_diag_rows:
        w.writerow(row)
print(f"Wrote {len(oracle_diag_rows)} rows to {oracle_path}")

# bucket_assignments.csv
bucket_path = RESULTS_DIR / "bucket_assignments.csv"
BUCKET_FIELDS = ["dataset","sample_id","failure_source","primary_bucket","secondary_bucket",
    "bucket_confidence","chunk_signal_subtype","propagation_loss_stage",
    "best_variant","best_variant_effect","evidence_for_bucket",
    "recommended_next_option","notes"]
with open(bucket_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=BUCKET_FIELDS, extrasaction='ignore')
    w.writeheader()
    for row in bucket_rows:
        w.writerow(row)
print(f"Wrote {len(bucket_rows)} rows to {bucket_path}")

# expected_evidence_targets.csv
target_path = RESULTS_DIR / "expected_evidence_targets.csv"
target_rows = []
for s in FOCUSED_SAMPLES:
    sid = s["sample_id"]
    for doc_id in s["expected_doc_ids"]:
        key = (sid, doc_id)
        ec = None
        if s["failure_source"] == "hard_recall_miss":
            ec = EXPECTED_CHUNK_MAP.get(key)
        else:
            ec = EXPECTED_CHUNK_MAP_LOW_SCORE.get(key)
        if ec:
            ct = chunk_text_map.get(ec["chunk_id"],"")[:200]
            target_rows.append({
                "dataset": s["dataset"], "sample_id": sid, "failure_source": s["failure_source"],
                "question": s["question"][:150], "expected_doc_id": doc_id,
                "expected_source_file": chunk_source_map.get(ec["chunk_id"],""),
                "expected_chunk_id": ec["chunk_id"], "expected_section": ec["section"],
                "expected_chunk_text_preview": ct.replace('\n',' ')[:200],
                "answer_bearing_strength": ec["strength"],
                "target_source": ec["source"],
                "target_confidence": "high" if ec["strength"]=="strong" else ("medium" if ec["strength"]=="partial" else "low"),
                "alternative_expected_chunk_ids": "", "notes": ""
            })
with open(target_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset","sample_id","failure_source","question","expected_doc_id",
        "expected_source_file","expected_chunk_id","expected_section","expected_chunk_text_preview",
        "answer_bearing_strength","target_source","target_confidence",
        "alternative_expected_chunk_ids","notes"])
    w.writeheader()
    for row in target_rows:
        w.writerow(row)
print(f"Wrote {len(target_rows)} expected evidence targets to {target_path}")

# ─── Summary outputs ───
# query_variant_summary.csv
variant_summary = []
for vid, vname in [("v0","original CN"),("v1","EN-mirror"),("v2","bilingual"),("v3","entity list")]:
    vr = [r for r in all_metrics if r["variant_id"]==vid]
    n = len(vr)
    d_doc40 = sum(1 for r in vr if r["dense_expected_doc_found_top40"])
    d_ck40 = sum(1 for r in vr if r["dense_expected_chunk_found_top40"])
    b_doc40 = sum(1 for r in vr if r["bm25_expected_doc_found_top40"])
    b_ck40 = sum(1 for r in vr if r["bm25_expected_chunk_found_top40"])
    h_doc40 = sum(1 for r in vr if r["hybrid_expected_doc_found_top40"])
    h_ck40 = sum(1 for r in vr if r["hybrid_expected_chunk_found_top40"])
    pr_ck = sum(1 for r in vr if r["pipeline_rerank_input_contains_expected_chunk"])
    fin_ck = sum(1 for r in vr if r["final_contains_expected_chunk"])
    sel_ck = sum(1 for r in vr if r["selected_support_contains_expected_chunk"])
    variant_summary.append({
        "variant_id": vid, "variant_name": vname,
        "dense_expected_doc_top40_count": d_doc40, "dense_expected_chunk_top40_count": d_ck40,
        "bm25_expected_doc_top40_count": b_doc40, "bm25_expected_chunk_top40_count": b_ck40,
        "hybrid_expected_doc_top40_count": h_doc40, "hybrid_expected_chunk_top40_count": h_ck40,
        "pipeline_rerank_expected_chunk_seen_count": pr_ck,
        "final_expected_chunk_found_count": fin_ck,
        "selected_support_expected_chunk_found_count": sel_ck,
        "avg_dense_rank_delta_vs_v0": round(sum(r["dense_rank_delta_v1_minus_v0"] for r in vr)/max(n,1), 1) if vid!="v0" else 0,
        "avg_rerank_score_delta_vs_v0": round(sum(r["rerank_score_delta_v1_minus_v0"] for r in vr)/max(n,1), 6) if vid!="v0" else 0,
        "notes": ""
    })
summary_path = RESULTS_DIR / "query_variant_summary.csv"
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant_id","variant_name","dense_expected_doc_top40_count",
        "dense_expected_chunk_top40_count","bm25_expected_doc_top40_count","bm25_expected_chunk_top40_count",
        "hybrid_expected_doc_top40_count","hybrid_expected_chunk_top40_count",
        "pipeline_rerank_expected_chunk_seen_count","final_expected_chunk_found_count",
        "selected_support_expected_chunk_found_count","avg_dense_rank_delta_vs_v0",
        "avg_rerank_score_delta_vs_v0","notes"])
    w.writeheader()
    for row in variant_summary:
        w.writerow(row)
print(f"Wrote variant summary to {summary_path}")

# bucket_summary.json
buckets = [r["primary_bucket"] for r in bucket_rows]
bcounts = {k:buckets.count(k) for k in ["Q","R","C","D","Mixed"]}
c1 = sum(1 for r in bucket_rows if r["chunk_signal_subtype"]=="C1")
c2 = sum(1 for r in bucket_rows if r["chunk_signal_subtype"]=="C2")
c3 = sum(1 for r in bucket_rows if r["chunk_signal_subtype"]=="C3")
pcount = sum(1 for r in bucket_rows if r["secondary_bucket"]=="P")
bv_dist = {k: sum(1 for r in bucket_rows if r["best_variant"]==k) for k in ["v0","v1","v2","v3","none"]}
dominant = max(bcounts, key=bcounts.get)
plurality = bcounts[dominant] >= 6
bucket_summary = {
    "total_samples": 16,
    "bucket_q_count": bcounts.get("Q",0),
    "bucket_r_count": bcounts.get("R",0),
    "bucket_c_count": bcounts.get("C",0),
    "bucket_c1_count": c1, "bucket_c2_count": c2, "bucket_c3_count": c3,
    "bucket_d_count": bcounts.get("D",0),
    "mixed_count": bcounts.get("Mixed",0),
    "secondary_p_count": pcount,
    "best_variant_distribution": bv_dist,
    "dominant_bucket": dominant,
    "dominant_bucket_count": bcounts[dominant],
    "no_clear_plurality": not plurality,
    "interpretation": ""
}
# Add interpretation
if dominant == "Q" and plurality:
    bucket_summary["interpretation"] = "Query language/formulation is the main fixable lever. Phase 19C should pursue Option B (query rewriting / bilingual decomposition) shadow A/B."
elif dominant == "R" and plurality:
    bucket_summary["interpretation"] = "Reranker cross-lingual ranking is the main gap. Phase 19C should pursue Option E (reranker calibration / translate-before-rerank / input enrichment) shadow A/B."
elif dominant == "C" and plurality:
    bucket_summary["interpretation"] = "Chunk text/context signal is the main gap. Phase 19C should pursue Option D (metadata-enriched chunk) if C1/C3 dominant, or Option C (doc-level recall) if C2 dominant."
elif dominant == "D" and plurality:
    bucket_summary["interpretation"] = "Dense embedding cross-lingual gap is dominant. Pause fix-style work, produce dense calibration design doc."
else:
    bucket_summary["interpretation"] = "No clear plurality — run Option B (cheapest) + Option F (eval cleanup) in parallel."
with open(RESULTS_DIR / "bucket_summary.json", "w") as f:
    json.dump(bucket_summary, f, indent=2, ensure_ascii=False)
print(f"Bucket summary: {bucket_summary}")

# phase19c_recommendation.json
bs = bucket_summary
dom = bs["dominant_bucket"]
dom_cnt = bs["dominant_bucket_count"]
if dom == "Q" and dom_cnt >= 6:
    rec19c = "option_b_query_rewriting_shadow_ab"
elif dom == "R" and dom_cnt >= 6:
    rec19c = "option_e_rerank_translation_or_input_enrichment"
elif dom == "C" and dom_cnt >= 6:
    if c2 > c1 + c3:
        rec19c = "option_c_doc_level_recall_design"
    else:
        rec19c = "option_d_metadata_enriched_chunk_shadow_index"
elif dom == "D" and dom_cnt >= 6:
    rec19c = "dense_calibration_design_doc"
else:
    rec19c = "mixed_option_b_plus_metric_cleanup"

rec = {
    "recommended_phase19c": rec19c,
    "rationale": bucket_summary["interpretation"],
    "bucket_distribution": {k:v for k,v in {"Q":bs["bucket_q_count"],"R":bs["bucket_r_count"],"C":bs["bucket_c_count"],"D":bs["bucket_d_count"],"Mixed":bs["mixed_count"]}.items()},
    "focused_samples_for_phase19c": [r["sample_id"] for r in bucket_rows if r["primary_bucket"] == dom] if dom_cnt >= 6 else [r["sample_id"] for r in bucket_rows],
    "proposed_variants": ["v0_original_CN","v1_EN_mirror","v2_bilingual"] if rec19c == "option_b_query_rewriting_shadow_ab" else ["v0_original_CN"],
    "success_criteria": "Per Phase 19C design (depends on chosen option)",
    "risks": "Per Phase 19C design",
    "why_not_continue_alias": "Phase 18C: +1 BM25 hit, 0 final recovery. Cross-lingual gap is semantic, not lexical.",
    "why_not_source_floor_tuning": "Phase 17F: source-floor eliminated its class. Remaining P0 are not hybrid-fusion-suppressed.",
    "why_not_support_capacity": "Phase 18F: 1/6 fixed, 5/6 are low_support_score from upstream reranker.",
    "why_not_direct_reranker_replacement": "No cross-lingual calibration measurement yet. High-risk without Audit B data. Replacement considered only if E.1/E.2 plateau.",
    "next_validation_plan": "Phase 19C focused shadow A/B on the recommended option, gated by focused set metrics → smoke50 sanity."
}
with open(RESULTS_DIR / "phase19c_recommendation.json", "w") as f:
    json.dump(rec, f, indent=2, ensure_ascii=False)
print(f"Phase 19C recommendation: {rec19c}")

print("\n" + "="*60)
print("Phase 19B audit complete. All data files written to:")
print(f"  {RESULTS_DIR}")
print("Next: write report documents.")
