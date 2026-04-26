#!/usr/bin/env python3
"""
512 vs 8192 检索对比诊断脚本
用法:
    python scripts/diagnostics/compare_512_vs_8192.py
"""
import json
import sys
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
URI = "runtime/vectorstores/milvus/papers.db"
COLL_512 = "synbio_papers_exp"
COLL_8192 = "synbio_papers_exp_8192"
MODEL_PATH = "./models/BAAI/bge-m3"
TOP_K = 5

# ── 查询集 ────────────────────────────────────────────
QUERIES = [
    # 上次已用的问题
    {"id": "Q01", "query": "gut microbiota and immune system interaction",
     "expected_doc": "doc_0001", "expected_section": "Results", "category": "Results"},
    {"id": "Q02", "query": "probiotic bacteria fermentation methods",
     "expected_doc": "doc_0001", "expected_section": "Methods", "category": "Methods"},
    {"id": "Q03", "query": "statistical analysis of variance ANOVA",
     "expected_doc": "doc_0001", "expected_section": "Methods", "category": "Methods"},
    {"id": "Q04", "query": "16S rRNA sequencing analysis",
     "expected_doc": "doc_0001", "expected_section": "Methods", "category": "Methods"},
    # Methods 覆盖
    {"id": "Q05", "query": "What bacterial strains were used in the experiments?",
     "expected_doc": "any", "expected_section": "Methods", "category": "Methods"},
    {"id": "Q06", "query": "culture medium and fermentation conditions for Bifidobacterium",
     "expected_doc": "doc_0001", "expected_section": "Methods", "category": "Methods"},
    {"id": "Q07", "query": "RNA isolation and quantitative PCR protocol",
     "expected_doc": "doc_0037", "expected_section": "Methods", "category": "Methods"},
    # Results 覆盖
    {"id": "Q08", "query": "2'-FL utilization by Bifidobacterium infantis",
     "expected_doc": "doc_0001", "expected_section": "Results", "category": "Results"},
    {"id": "Q09", "query": "gene expression changes in yeast during fermentation",
     "expected_doc": "doc_0060", "expected_section": "Results", "category": "Results"},
    # 跨页 section 覆盖
    {"id": "Q10", "query": "selection of Bifidobacterium longum dominant infant fecal microbiota",
     "expected_doc": "doc_0001", "expected_section": "Results", "category": "cross_page"},
    # Figure caption 覆盖
    {"id": "Q11", "query": "NMDS plot Bray-Curtis distance metrics figure",
     "expected_doc": "doc_0001", "expected_section": "Results", "category": "figure_caption"},
    {"id": "Q12", "query": "phylogenetic tree figure showing bacterial relationships",
     "expected_doc": "any", "expected_section": "Results", "category": "figure_caption"},
    # Table caption 覆盖
    {"id": "Q13", "query": "baseline characteristics of selected infants table",
     "expected_doc": "doc_0001", "expected_section": "Methods", "category": "table_caption"},
    {"id": "Q14", "query": "oligonucleotide primers used for PCR amplification table",
     "expected_doc": "any", "expected_section": "Methods", "category": "table_caption"},
    # 表格正文具体信息
    {"id": "Q15", "query": "What is the specific age and delivery mode of infant M1?",
     "expected_doc": "doc_0001", "expected_section": "Methods", "category": "table_content"},
    {"id": "Q16", "query": "concentration of lactate and acetate produced in fermentation",
     "expected_doc": "doc_0001", "expected_section": "Results", "category": "table_content"},
    # References 误召回
    {"id": "Q17", "query": "Akkerman 2019 non-digestible carbohydrates immune system",
     "expected_doc": "NONE", "expected_section": "NOT_References", "category": "references_leak"},
    {"id": "Q18", "query": "bacterial colonization mouse intestine Journal of Science",
     "expected_doc": "NONE", "expected_section": "NOT_References", "category": "references_leak"},
    # 综合
    {"id": "Q19", "query": "adenosine A2A receptor expression in yeast",
     "expected_doc": "doc_0037", "expected_section": "Methods", "category": "Methods"},
    {"id": "Q20", "query": "microarray analysis of gene expression codon usage",
     "expected_doc": "doc_0060", "expected_section": "Results", "category": "Results"},
]

def main():
    from pymilvus import MilvusClient
    from FlagEmbedding import BGEM3FlagModel

    print("Loading BGE-M3 model...")
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)

    client = MilvusClient(uri=URI)

    # Load chunks metadata for token_count lookup
    chunks_meta = {}
    with open("data/small_exp/chunks/chunks.jsonl") as f:
        for line in f:
            c = json.loads(line)
            chunks_meta[c["chunk_id"]] = c

    results = []

    for q_info in QUERIES:
        qid = q_info["id"]
        query = q_info["query"]
        category = q_info["category"]
        expected_doc = q_info["expected_doc"]
        expected_section = q_info["expected_section"]

        print(f"\n{'='*70}")
        print(f"[{qid}] {query}  (category={category})")

        q_vec = model.encode([query], batch_size=1, max_length=512)["dense_vecs"][0]

        row = {"qid": qid, "query": query, "category": category,
               "expected_doc": expected_doc, "expected_section": expected_section}

        for coll_name, label in [(COLL_512, "512"), (COLL_8192, "8192")]:
            try:
                hits = client.search(
                    collection_name=coll_name,
                    data=[q_vec.tolist()],
                    anns_field="embedding",
                    limit=TOP_K,
                    output_fields=["chunk_id", "doc_id", "title", "section", "text"],
                    search_params={"metric_type": "COSINE"},
                )
                ranked = []
                for rank, hit in enumerate(hits[0], 1):
                    cid = hit["entity"]["chunk_id"]
                    meta = chunks_meta.get(cid, {})
                    entry = {
                        "rank": rank,
                        "score": round(hit["distance"], 4),
                        "chunk_id": cid,
                        "doc_id": hit["entity"]["doc_id"],
                        "section": hit["entity"]["section"],
                        "section_path": meta.get("section_path", []),
                        "page_start": meta.get("page_start"),
                        "page_end": meta.get("page_end"),
                        "block_types": meta.get("block_types", []),
                        "token_count": meta.get("token_count", 0),
                        "text_preview": hit["entity"]["text"][:300].replace("\n", " "),
                    }
                    ranked.append(entry)
                row[f"top_{label}"] = ranked
            except Exception as e:
                print(f"  ERROR on {coll_name}: {e}")
                row[f"top_{label}"] = []

        # Quick judgment
        def judge(top_list, exp_doc, exp_sec):
            if not top_list:
                return "fail"
            hit1 = top_list[0]
            doc_ok = (exp_doc == "any" or hit1["doc_id"] == exp_doc)
            sec_ok = (exp_sec in hit1.get("section", "") or 
                      any(exp_sec.lower() in str(s).lower() for s in hit1.get("section_path", [])))
            if exp_doc == "NONE":
                # Should NOT hit references
                has_ref = any("references" in str(s).lower() for s in hit1.get("block_types", []))
                if has_ref:
                    return "fail"
                return "pass"
            if doc_ok and sec_ok:
                return "pass"
            elif doc_ok or sec_ok:
                return "partial"
            return "fail"

        row["judge_512"] = judge(row.get("top_512", []), expected_doc, expected_section)
        row["judge_8192"] = judge(row.get("top_8192", []), expected_doc, expected_section)

        # Compare
        s512 = row["top_512"][0]["score"] if row["top_512"] else 0
        s8192 = row["top_8192"][0]["score"] if row["top_8192"] else 0
        if row["judge_8192"] == "pass" and row["judge_512"] != "pass":
            row["comparison"] = "8192_better"
        elif row["judge_512"] == "pass" and row["judge_8192"] != "pass":
            row["comparison"] = "512_better"
        elif row["judge_512"] == row["judge_8192"]:
            if s8192 > s512 + 0.01:
                row["comparison"] = "8192_slightly_better"
            elif s512 > s8192 + 0.01:
                row["comparison"] = "512_slightly_better"
            else:
                row["comparison"] = "similar"
        else:
            row["comparison"] = "mixed"

        print(f"  512 top1: [{s512:.4f}] {row['top_512'][0]['chunk_id'] if row['top_512'] else 'N/A'} "
              f"judge={row['judge_512']}")
        print(f"  8192 top1: [{s8192:.4f}] {row['top_8192'][0]['chunk_id'] if row['top_8192'] else 'N/A'} "
              f"judge={row['judge_8192']}")
        print(f"  comparison: {row['comparison']}")

        results.append(row)

    client.close()

    # Save full results
    out_path = "results/compare_512_vs_8192_detail.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    # Print summary
    from collections import Counter
    comp_counts = Counter(r["comparison"] for r in results)
    j512 = Counter(r["judge_512"] for r in results)
    j8192 = Counter(r["judge_8192"] for r in results)
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Comparison: {dict(comp_counts)}")
    print(f"512 judgments: {dict(j512)}")
    print(f"8192 judgments: {dict(j8192)}")

    # Table-related queries
    table_cats = {"table_caption", "table_content"}
    table_results = [r for r in results if r["category"] in table_cats]
    print(f"\nTable-related queries ({len(table_results)}):")
    for r in table_results:
        print(f"  {r['qid']}: 512={r['judge_512']}, 8192={r['judge_8192']}, comp={r['comparison']}")

    # References leak check
    ref_results = [r for r in results if r["category"] == "references_leak"]
    print(f"\nReferences leak queries ({len(ref_results)}):")
    for r in ref_results:
        print(f"  {r['qid']}: 512={r['judge_512']}, 8192={r['judge_8192']}")

if __name__ == "__main__":
    main()
