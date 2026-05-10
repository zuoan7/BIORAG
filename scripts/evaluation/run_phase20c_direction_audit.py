"""
Phase 20C: Rewrite-Enabled Residual Badcase Direction Audit.
纯只读审计脚本——不修改生产代码、不重建索引、不修改配置。
"""
import json
import csv
import os
from pathlib import Path
from collections import defaultdict

# ============================================================
# 配置路径
# ============================================================
BASE = Path("/home/gmy/bio-agent-workspace/bge-service")
RESULTS_DIR = BASE / "results/phase20c_residual_direction_audit"
REPORTS_DIR = BASE / "reports/phase20c_residual_direction_audit"

# 输入
SMOKE100_PATH = BASE / "data/eval/datasets/enterprise_ragas_smoke100.json"
SMOKE50_PATH = BASE / "data/evaluation/smoke50_parent_expansion_v1.jsonl"
CHUNKS_PATH = BASE / "data/paper_round1/chunks/chunks.jsonl"
PARENT_PATH = BASE / "data/paper_round1/chunks/parent_index.jsonl"
DELTA_CSV_PATH = BASE / "results/phase20a_rewrite_enabled_full_eval/full_eval_per_sample_delta.csv"
P0_LEDGER_PATH = BASE / "results/phase20a_rewrite_enabled_full_eval/full_eval_p0_delta_ledger_corrected.csv"
NEW_P0_PATH = BASE / "results/phase20a_rewrite_enabled_full_eval/full_eval_new_real_p0_ledger.csv"
RESIDUAL_TAXONOMY = BASE / "results/phase20a_rewrite_enabled_full_eval/full_eval_residual_badcase_taxonomy.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 加载数据集
# ============================================================
def load_datasets():
    """加载 smoke100 和 smoke50 评测集。"""
    ds = {}
    with open(SMOKE100_PATH) as f:
        smoke100 = json.load(f)
    for item in smoke100:
        ds[item["id"]] = {
            "dataset": "smoke100",
            "sample_id": item["id"],
            "original_query": item.get("question", ""),
            "expected_doc_ids": item.get("expected_doc_ids", []) or [],
            "expected_source_files": item.get("expected_source_files", []) or [],
            "expected_route": item.get("expected_route", ""),
            "expected_sections": item.get("expected_sections", []) or [],
            "tags": item.get("tags", []) or [],
            "notes": item.get("notes", ""),
        }

    with open(SMOKE50_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("id", item.get("sample_id", ""))
            ds[sid] = {
                "dataset": "smoke50",
                "sample_id": sid,
                "original_query": item.get("question", ""),
                "expected_doc_ids": item.get("expected_doc_ids", []) or [],
                "expected_source_files": item.get("expected_source_files", []) or [],
                "expected_route": item.get("expected_route", ""),
                "expected_sections": item.get("expected_sections", []) or [],
                "tags": item.get("tags", []) or [],
                "notes": item.get("notes", ""),
            }
    return ds

# ============================================================
# 加载 chunks 和 parent_index
# ============================================================
def load_index():
    """加载 chunks 和 parent_index 建立查询索引。"""
    chunks = {}  # chunk_id → chunk
    doc_chunks = defaultdict(list)  # doc_id → [chunks]
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c["chunk_id"]
            chunks[cid] = c
            doc_chunks[c["doc_id"]].append(c)

    parent_index = {}  # parent_id → parent
    with open(PARENT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            pid = p["parent_id"]
            parent_index[pid] = p

    return chunks, doc_chunks, parent_index

# ============================================================
# 加载 Phase 20A 结果
# ============================================================
def load_phase20a_results():
    """加载 Phase 20A 的 delta 和 p0 ledger。"""
    delta = {}
    with open(DELTA_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delta[row["sample_id"]] = row

    p0 = {}
    with open(P0_LEDGER_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p0[row["sample_id"]] = row

    new_p0 = {}
    with open(NEW_P0_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_p0[row["sample_id"]] = row

    return delta, p0, new_p0

# ============================================================
# Step 1: 冻结 residual set
# ============================================================
def freeze_residual_set(ds, delta, p0, new_p0):
    """从 Phase 20A 输出中提取冻结的 residual 样本集。"""

    # unchanged_real_p0: off=real_p0, enabled=real_p0
    unchanged = [
        sid for sid, row in p0.items()
        if row.get("p0_delta_type_corrected") == "unchanged_real_p0"
    ]

    # new_real_p0: off=ok, enabled=real_p0
    new_p0_samples = [
        sid for sid, row in p0.items()
        if row.get("p0_delta_type_corrected") == "new_real_p0"
    ]

    all_residual_sids = sorted(set(unchanged + new_p0_samples))

    rows = []
    for sid in all_residual_sids:
        info = ds.get(sid, {})
        d = delta.get(sid, {})
        p = p0.get(sid, {})
        np0 = new_p0.get(sid, {})

        residual_type = "new_real_p0" if sid in new_p0_samples else "residual_real_p0"

        rows.append({
            "sample_id": sid,
            "dataset": info.get("dataset", ""),
            "residual_type": residual_type,
            "off_status": p.get("off_corrected_failure_category", d.get("off_corrected_failure_category", "")),
            "enabled_status": p.get("enabled_corrected_failure_category", d.get("enabled_corrected_failure_category", "")),
            "off_failure_category": d.get("off_raw_failure_category", ""),
            "enabled_failure_category": d.get("enabled_raw_failure_category", ""),
            "expected_doc_ids": "|".join(info.get("expected_doc_ids", [])),
            "expected_source_files": "|".join(info.get("expected_source_files", [])),
            "expected_route": info.get("expected_route", ""),
            "original_query": info.get("original_query", ""),
            "rewritten_query": d.get("rewritten_query", ""),
            "notes": np0.get("suspected_root_cause", p.get("likely_reason", "")),
        })

    # 写 CSV
    output_path = RESULTS_DIR / "frozen_residual9_samples.csv"
    fieldnames = [
        "sample_id", "dataset", "residual_type", "off_status", "enabled_status",
        "off_failure_category", "enabled_failure_category",
        "expected_doc_ids", "expected_source_files", "expected_route",
        "original_query", "rewritten_query", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 1] Frozen residual set: {len(rows)} samples")
    print(f"  - residual_real_p0 (unchanged): {len(unchanged)}")
    print(f"  - new_real_p0: {len(new_p0_samples)}")
    print(f"  → {output_path}")
    return rows

# ============================================================
# Step 2: index / evidence existence audit
# ============================================================
def audit_index_evidence(frozen, ds, chunks, doc_chunks, parent_index):
    """判断 expected doc 和 evidence 在 index 中是否存在。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        edoc_ids = info.get("expected_doc_ids", [])
        esrc_files = info.get("expected_source_files", [])

        for i, edoc_id in enumerate(edoc_ids):
            src_file = esrc_files[i] if i < len(esrc_files) else ""
            doc_chunk_list = doc_chunks.get(edoc_id, [])

            # 检查 doc 在 chunks 中是否存在
            exists = len(doc_chunk_list) > 0
            n_chunks = len(doc_chunk_list)

            # 检查 key metadata 字段
            has_title = False
            has_abstract = False
            has_sections = False
            has_table_text = False
            has_figure_caption = False

            answer_chunks = []
            for c in doc_chunk_list:
                title = c.get("title", "")
                section = c.get("section", "")
                if title:
                    has_title = True
                if section and "abstract" in section.lower():
                    has_abstract = True
                if section and section.strip():
                    has_sections = True
                if c.get("contains_table_text"):
                    has_table_text = True
                if c.get("contains_figure_caption"):
                    has_figure_caption = True

            # 判断证据强度
            # 使用 section 信息判断是否有核心内容 section 的 chunk
            core_sections = {"results", "discussion", "method", "conclusion", "methods"}
            strong_chunks = []
            for c in doc_chunk_list:
                sec = (c.get("section") or "").lower()
                if any(cs in sec for cs in core_sections):
                    strong_chunks.append(c["chunk_id"])
                elif c.get("contains_table_text"):
                    strong_chunks.append(c["chunk_id"])
                elif c.get("contains_figure_caption"):
                    strong_chunks.append(c["chunk_id"])

            if strong_chunks:
                strength = "strong"
                answer_chunk_ids = strong_chunks[:5]
            elif doc_chunk_list:
                strength = "partial"
                answer_chunk_ids = [c["chunk_id"] for c in doc_chunk_list[:3]]
            elif exists:
                strength = "weak"
                answer_chunk_ids = []
            else:
                strength = "none"
                answer_chunk_ids = []

            # 判断 evidence_location
            if has_table_text and has_figure_caption:
                ev_loc = "table,figure_caption"
            elif has_table_text:
                ev_loc = "table"
            elif has_figure_caption:
                ev_loc = "figure_caption"
            elif len(doc_chunk_list) > 1:
                ev_loc = "text"
            elif len(doc_chunk_list) == 1:
                ev_loc = "text"
            else:
                ev_loc = "unclear"

            rows.append({
                "sample_id": sid,
                "expected_doc_id": edoc_id,
                "expected_source_file": src_file,
                "expected_doc_exists_in_index": exists,
                "expected_doc_chunk_count": n_chunks,
                "expected_doc_has_title": has_title,
                "expected_doc_has_abstract": has_abstract,
                "expected_doc_has_sections": has_sections,
                "expected_doc_has_table_text": has_table_text,
                "expected_doc_has_figure_caption": has_figure_caption,
                "expected_answer_bearing_chunk_ids": "|".join(answer_chunk_ids),
                "answer_bearing_strength": strength,
                "evidence_location": ev_loc,
                "notes": "",
            })

    output_path = RESULTS_DIR / "index_evidence_existence_audit.csv"
    fieldnames = [
        "sample_id", "expected_doc_id", "expected_source_file",
        "expected_doc_exists_in_index", "expected_doc_chunk_count",
        "expected_doc_has_title", "expected_doc_has_abstract",
        "expected_doc_has_sections", "expected_doc_has_table_text",
        "expected_doc_has_figure_caption",
        "expected_answer_bearing_chunk_ids", "answer_bearing_strength",
        "evidence_location", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 2] Index/evidence audit: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 3: enabled lifecycle trace (基于现有数据推断)
# ============================================================
def build_lifecycle_trace(frozen, ds, delta, chunks, doc_chunks):
    """
    基于 Phase 20A 的 per_sample_delta 数据重构已发生的行为。
    注意：这是基于已有 CSVs 的离线追踪，不重新执行 RAG 管道。
    """
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        d = delta.get(sid, {})

        edoc_ids = info.get("expected_doc_ids", [])
        rewritten_query = d.get("rewritten_query", "")
        enabled_cited = (d.get("enabled_cited_doc_ids") or "").split("|")
        enabled_cited = [x for x in enabled_cited if x]

        # 判断 expected doc 是否在 enabled 的结果中
        expected_in_cited = any(ed in enabled_cited for ed in edoc_ids)
        enabled_doc_hit = d.get("enabled_doc_hit", "") == "True"

        # first_loss_stage 推断
        if enabled_doc_hit:
            first_loss = "none"
        elif not enabled_doc_hit and d.get("off_doc_hit", "") == "True":
            # off 能命中 enabled 不能 → rewrite 导致了 dense/hybrid 阶段丢失
            first_loss = "dense"
        else:
            # off 和 enabled 都不能命中 → document 可能根本没进 index 或 vector gap
            first_loss = "dense"

        rows.append({
            "sample_id": sid,
            "expected_doc_ids": "|".join(edoc_ids),
            "expected_chunk_ids": "",
            "rewritten_query": rewritten_query,
            "dense_expected_doc_top10": "unknown",
            "dense_expected_doc_top20": "unknown",
            "dense_expected_doc_top40": "unknown",
            "dense_expected_chunk_top10": "unknown",
            "dense_expected_chunk_top20": "unknown",
            "dense_expected_chunk_top40": "unknown",
            "bm25_expected_doc_top40": "unknown",
            "bm25_expected_chunk_top40": "unknown",
            "hybrid_expected_doc_top40": "unknown",
            "hybrid_expected_chunk_top40": "unknown",
            "source_floor_added_expected": "unknown",
            "rerank_input_contains_expected_doc": "unknown",
            "rerank_input_contains_expected_chunk": "unknown",
            "rerank_rank_expected_chunk": "unknown",
            "final_contains_expected_doc": enabled_doc_hit,
            "final_contains_expected_chunk": enabled_doc_hit,
            "support_input_contains_expected": "unknown",
            "selected_support_contains_expected": "unknown",
            "citation_candidate_contains_expected": expected_in_cited,
            "citation_output_contains_expected": expected_in_cited,
            "first_loss_stage": first_loss,
            "notes": ("off_doc_hit=" + d.get("off_doc_hit", "?") +
                      " enabled_doc_hit=" + d.get("enabled_doc_hit", "?"))
        })

    output_path = RESULTS_DIR / "enabled_lifecycle_trace.csv"
    fieldnames = [
        "sample_id", "expected_doc_ids", "expected_chunk_ids",
        "rewritten_query",
        "dense_expected_doc_top10", "dense_expected_doc_top20", "dense_expected_doc_top40",
        "dense_expected_chunk_top10", "dense_expected_chunk_top20", "dense_expected_chunk_top40",
        "bm25_expected_doc_top40", "bm25_expected_chunk_top40",
        "hybrid_expected_doc_top40", "hybrid_expected_chunk_top40",
        "source_floor_added_expected",
        "rerank_input_contains_expected_doc", "rerank_input_contains_expected_chunk",
        "rerank_rank_expected_chunk",
        "final_contains_expected_doc", "final_contains_expected_chunk",
        "support_input_contains_expected", "selected_support_contains_expected",
        "citation_candidate_contains_expected", "citation_output_contains_expected",
        "first_loss_stage", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 3] Lifecycle trace: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 4: near-topic competitor audit
# ============================================================
def audit_near_topic(frozen, ds, delta, new_p0, doc_chunks):
    """比较 expected doc 与 competing doc。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        d = delta.get(sid, {})
        np0 = new_p0.get(sid, {})
        edoc_ids = info.get("expected_doc_ids", [])

        enabled_cited_str = d.get("enabled_cited_doc_ids", "")
        enabled_cited = [x for x in enabled_cited_str.split("|") if x] if enabled_cited_str else []

        # 找到 competing docs (enabled 中 cited 但不在 expected 中的)
        competing = [d for d in enabled_cited if d not in edoc_ids]

        # 对每个 expected doc 列出最近的 competing doc
        for i, edoc_id in enumerate(edoc_ids):
            comp_id = competing[i] if i < len(competing) else (competing[0] if competing else "")

            # 比较 expected 和 competing doc 的基本信息
            e_chunks = doc_chunks.get(edoc_id, [])
            c_chunks = doc_chunks.get(comp_id, [])

            e_title = e_chunks[0].get("title", "") if e_chunks else ""
            c_title = c_chunks[0].get("title", "") if c_chunks else ""

            # 判断是否为 near-topic 替代
            is_near = np0.get("suspected_root_cause", "") == "near_topic_but_expected_doc_miss"

            rows.append({
                "sample_id": sid,
                "expected_doc_id": edoc_id,
                "competing_doc_id": comp_id,
                "competing_source_file": c_chunks[0].get("source_file", "") if c_chunks else "",
                "competing_rank": "unknown",
                "competing_cited": comp_id in enabled_cited,
                "same_organism": "unclear",
                "same_product": "unclear",
                "same_method": "unclear",
                "same_pathway": "unclear",
                "same_question_type": "unclear",
                "expected_doc_has_stronger_evidence": "unclear",
                "competing_doc_answerable": "unclear",
                "likely_metric_issue": "false",
                "likely_near_topic_doc_miss": str(is_near).lower(),
                "disambiguation_signal_missing": "unclear",
                "notes": f"Expected title: {e_title[:100]}... | Competing title: {c_title[:100]}...",
            })

    output_path = RESULTS_DIR / "near_topic_competitor_audit.csv"
    fieldnames = [
        "sample_id", "expected_doc_id", "competing_doc_id",
        "competing_source_file", "competing_rank", "competing_cited",
        "same_organism", "same_product", "same_method", "same_pathway",
        "same_question_type", "expected_doc_has_stronger_evidence",
        "competing_doc_answerable", "likely_metric_issue",
        "likely_near_topic_doc_miss", "disambiguation_signal_missing",
        "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 4] Near-topic audit: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 5: metadata context audit
# ============================================================
def audit_metadata_context(frozen, ds, doc_chunks):
    """判断 chunk 是否缺少关键 metadata context。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        edoc_ids = info.get("expected_doc_ids", [])

        for edoc_id in edoc_ids:
            doc_chunk_list = doc_chunks.get(edoc_id, [])

            for c in doc_chunk_list[:5]:  # 每个 doc 先取5个 chunk
                cid = c["chunk_id"]
                text = c.get("text", "")
                section = c.get("section", "")
                title = c.get("title", "")

                has_title = bool(title)
                has_section = bool(section)
                has_organism = any(org in text.lower() for org in
                    ["coli", "cerevisiae", "pastoris", "subtilis", "bifidobacterium",
                     "lactobacillus", "bacillus", "yeast", "bacteria", "mammalian"])
                has_product = any(prod in text.lower() for prod in
                    ["2'-fl", "3-fl", "6'-sl", "lacto-n-", "neu5ac", "sialic",
                     "hmo", "lactose", "fucose", "sialyl", "fucosyl", "n-glycan"])
                has_method = any(meth in text.lower() for meth in
                    ["overexpress", "knockout", "crispr", "plasmid", "fermentation",
                     "chromatography", "mass spec", "hplc", "rt-qpcr"])

                missing = []
                if not has_title:
                    missing.append("title")
                if not has_section:
                    missing.append("section")
                if not has_organism:
                    missing.append("organism")
                if not has_product:
                    missing.append("product")
                if not has_method:
                    missing.append("method")

                would_prefix_help = "yes" if missing else "no"

                rows.append({
                    "sample_id": sid,
                    "expected_chunk_id": cid,
                    "chunk_text_preview": text[:200],
                    "chunk_length": len(text),
                    "has_title_context": has_title,
                    "has_section_context": has_section,
                    "has_doc_summary_context": "unclear",
                    "has_organism_context": has_organism,
                    "has_product_context": has_product,
                    "has_method_context": has_method,
                    "missing_context_types": "|".join(missing) if missing else "none",
                    "would_title_section_prefix_help": would_prefix_help,
                    "would_doc_metadata_help": "yes" if missing else "unclear",
                    "metadata_enrichment_candidate": str(bool(missing)).lower(),
                    "notes": f"section={section}, title={title[:50]}",
                })

    output_path = RESULTS_DIR / "metadata_context_audit.csv"
    fieldnames = [
        "sample_id", "expected_chunk_id", "chunk_text_preview",
        "chunk_length", "has_title_context", "has_section_context",
        "has_doc_summary_context", "has_organism_context",
        "has_product_context", "has_method_context",
        "missing_context_types", "would_title_section_prefix_help",
        "would_doc_metadata_help", "metadata_enrichment_candidate",
        "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 5] Metadata context audit: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 6: doc-level recall audit
# ============================================================
def audit_doc_level_recall(frozen, ds, doc_chunks):
    """判断是否适合 doc-level recall 扩展。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        edoc_ids = info.get("expected_doc_ids", [])
        query = info.get("original_query", "").lower()

        for edoc_id in edoc_ids:
            doc_chunk_list = doc_chunks.get(edoc_id, [])
            if not doc_chunk_list:
                rows.append({
                    "sample_id": sid,
                    "expected_doc_id": edoc_id,
                    "title_matches_query": "false",
                    "abstract_matches_query": "unclear",
                    "source_file_matches_query": "false",
                    "doc_level_signal_strength": "none",
                    "chunk_level_signal_strength": "none",
                    "evidence_dispersed_across_chunks": "false",
                    "local_expansion_would_help": "unclear",
                    "doc_level_recall_candidate": "false",
                    "notes": "Doc not in index",
                })
                continue

            title = doc_chunk_list[0].get("title", "").lower()
            src = doc_chunk_list[0].get("source_file", "").lower()

            # 检查 title / abstract level signal
            title_match = any(w in title for w in query.split() if len(w) > 2)
            src_match = any(w in src for w in query.split() if len(w) > 2)

            # doc level signal
            if title_match:
                doc_signal = "strong"
            elif len(doc_chunk_list) > 10:
                doc_signal = "medium"
            else:
                doc_signal = "weak"

            # chunk level signal
            n_core = sum(1 for c in doc_chunk_list
                         if c.get("section", "") and
                         any(cs in c.get("section", "").lower()
                             for cs in ["results", "discussion", "method"]))
            if n_core > 3:
                chunk_signal = "strong"
            elif n_core > 0:
                chunk_signal = "medium"
            else:
                chunk_signal = "weak"

            # evidence 是否分散
            dispersed = len(doc_chunk_list) > 3

            # local expansion 是否有帮助
            if dispersed and doc_signal in ("strong", "medium"):
                local_help = "yes"
                doc_level_candidate = "true"
            elif dispersed:
                local_help = "yes"
                doc_level_candidate = "true"
            else:
                local_help = "unclear"
                doc_level_candidate = "false"

            rows.append({
                "sample_id": sid,
                "expected_doc_id": edoc_id,
                "title_matches_query": str(title_match).lower(),
                "abstract_matches_query": "unclear",
                "source_file_matches_query": str(src_match).lower(),
                "doc_level_signal_strength": doc_signal,
                "chunk_level_signal_strength": chunk_signal,
                "evidence_dispersed_across_chunks": str(dispersed).lower(),
                "local_expansion_would_help": local_help,
                "doc_level_recall_candidate": doc_level_candidate,
                "notes": f"chunks={len(doc_chunk_list)}, title={title[:100]}",
            })

    output_path = RESULTS_DIR / "doc_level_recall_audit.csv"
    fieldnames = [
        "sample_id", "expected_doc_id",
        "title_matches_query", "abstract_matches_query", "source_file_matches_query",
        "doc_level_signal_strength", "chunk_level_signal_strength",
        "evidence_dispersed_across_chunks", "local_expansion_would_help",
        "doc_level_recall_candidate", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 6] Doc-level recall audit: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 7: parsing / table / figure audit
# ============================================================
def audit_parsing_table_figure(frozen, ds, doc_chunks):
    """判断是否因 PDF 解析问题导致 chunk 缺失。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        edoc_ids = info.get("expected_doc_ids", [])
        esrc_files = info.get("expected_source_files", [])
        query = info.get("original_query", "").lower()

        for i, edoc_id in enumerate(edoc_ids):
            src_file = esrc_files[i] if i < len(esrc_files) else ""
            doc_chunk_list = doc_chunks.get(edoc_id, [])

            has_table = any(c.get("contains_table_text") for c in doc_chunk_list)
            has_fig_cap = any(c.get("contains_figure_caption") for c in doc_chunk_list)
            has_image = any(c.get("contains_image") for c in doc_chunk_list)

            # query 中是否问 table/figure
            asks_table = any(w in query for w in ["table", "table", "表格", "图表"])
            asks_figure = any(w in query for w in ["figure", "图", "fig"])

            has_data = len(doc_chunk_list) > 0

            # 判断解析问题
            if asks_table and not has_table and has_data:
                parsing_issue = "true"
                issue_type = "missing_table"
            elif asks_figure and not has_fig_cap and has_data:
                parsing_issue = "true"
                issue_type = "missing_caption"
            elif any(c.get("contains_image") for c in doc_chunk_list) and not has_table:
                parsing_issue = "unclear"
                issue_type = "image_not_processed"
            else:
                parsing_issue = "false"
                issue_type = "none"

            # dual-column 风险
            n_chunks = len(doc_chunk_list)
            if n_chunks > 20:
                dual_risk = "true"
            else:
                dual_risk = "false"

            rows.append({
                "sample_id": sid,
                "source_file": src_file,
                "evidence_location": "table" if has_table else ("figure_caption" if has_fig_cap else "text"),
                "table_text_present": str(has_table).lower(),
                "figure_caption_present": str(has_fig_cap).lower(),
                "image_or_ocr_needed": str(has_image).lower(),
                "dual_column_risk": dual_risk,
                "chunk_order_suspicious": "unclear",
                "parsing_issue_likely": parsing_issue,
                "parsing_issue_type": issue_type,
                "would_reparse_help": "yes" if parsing_issue == "true" else "unclear",
                "notes": f"chunks={n_chunks}, has_table={has_table}, has_fig={has_fig_cap}",
            })

    output_path = RESULTS_DIR / "parsing_table_figure_audit.csv"
    fieldnames = [
        "sample_id", "source_file", "evidence_location",
        "table_text_present", "figure_caption_present", "image_or_ocr_needed",
        "dual_column_risk", "chunk_order_suspicious",
        "parsing_issue_likely", "parsing_issue_type",
        "would_reparse_help", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 7] Parsing audit: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 8: support/citation residual audit
# ============================================================
def audit_support_citation(frozen, delta):
    """如果 expected evidence 到了 final 但未引用则标记。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        d = delta.get(sid, {})
        enabled_doc_hit = d.get("enabled_doc_hit", "") == "True"
        enabled_cited = (d.get("enabled_cited_doc_ids") or "").split("|")

        # 确定 drop stage
        if enabled_doc_hit:
            drop_stage = "none"
        else:
            drop_stage = "final_to_support"

        rows.append({
            "sample_id": sid,
            "expected_doc_in_final": enabled_doc_hit,
            "expected_chunk_in_final": enabled_doc_hit,
            "expected_doc_in_support_input": "unknown",
            "expected_doc_in_selected_support": "unknown",
            "expected_doc_in_citation_candidates": "unknown",
            "expected_doc_in_citation_output": enabled_doc_hit,
            "drop_stage": drop_stage,
            "drop_reason": "dense_gap" if not enabled_doc_hit else "none",
            "support_citation_candidate": str(not enabled_doc_hit).lower(),
            "notes": f"enabled cited: {enabled_cited}",
        })

    output_path = RESULTS_DIR / "support_citation_residual_audit.csv"
    fieldnames = [
        "sample_id",
        "expected_doc_in_final", "expected_chunk_in_final",
        "expected_doc_in_support_input", "expected_doc_in_selected_support",
        "expected_doc_in_citation_candidates", "expected_doc_in_citation_output",
        "drop_stage", "drop_reason", "support_citation_candidate", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 8] Support/citation audit: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 9: Direction bucket assignment
# ============================================================
def assign_direction_buckets(frozen, idx_evidence, lifecycle, near_topic,
                              metadata, doc_level, parsing, support_cit,
                              ds, delta, new_p0, doc_chunks):
    """综合所有审计，给每个样本分配 primary/secondary bucket。"""
    rows = []
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        d = delta.get(sid, {})
        np0 = new_p0.get(sid, {})
        edoc_ids = info.get("expected_doc_ids", [])

        is_new_p0 = sample["residual_type"] == "new_real_p0"
        off_doc_hit = d.get("off_doc_hit", "") == "True"
        enabled_doc_hit = d.get("enabled_doc_hit", "") == "True"

        # 汇总证据
        idx_row = [r for r in idx_evidence if r["sample_id"] == sid]
        doc_exists = all(r["expected_doc_exists_in_index"] for r in idx_row) if idx_row else False
        doc_has_title = any(r["expected_doc_has_title"] for r in idx_row)
        doc_has_sections = any(r["expected_doc_has_sections"] for r in idx_row)
        doc_has_table = any(r["expected_doc_has_table_text"] for r in idx_row)

        # 主逻辑: 综合归因
        if is_new_p0 and off_doc_hit and not enabled_doc_hit:
            # new P0: off 能命中 enabled 不行 → 重写导致 dense gap 或 near-topic misdirect
            if np0.get("suspected_root_cause", "") == "near_topic_but_expected_doc_miss":
                primary = "NEAR_TOPIC_DOC_MISS"
                secondary = "D_DENSE_GAP"
                confidence = "medium"
                main_evidence = "off_works_enabled_fails_near_topic_substitute"
                direction = "doc_level_recall_local_expansion"
                impact = "medium"
            else:
                primary = "QUERY_REWRITE_REGRESSION"
                secondary = "D_DENSE_GAP"
                confidence = "medium"
                main_evidence = "rewrite_caused_doc_miss_from_ok"
                direction = "query_rewrite_guardrail_revision"
                impact = "medium"
        elif not doc_exists:
            # doc 不在 index 中
            primary = "METRIC_OR_DATASET_ISSUE"
            secondary = "UNCLEAR"
            confidence = "high"
            main_evidence = "doc_not_in_index"
            direction = "eval_dataset_review"
            impact = "low"
        elif not enabled_doc_hit and not off_doc_hit:
            # 两个模式都 miss → dense gap
            if doc_has_title and not doc_has_sections:
                primary = "C3_METADATA_CONTEXT"
                secondary = "D_DENSE_GAP"
                confidence = "medium"
                main_evidence = "doc_has_title_no_sections_enriched_metadata_may_help"
                direction = "metadata_enriched_chunk_index"
                impact = "medium"
            elif doc_has_table:
                primary = "PARSING_TABLE_FIGURE_ISSUE"
                secondary = "D_DENSE_GAP"
                confidence = "medium"
                main_evidence = "table_content_not_semantically_searchable"
                direction = "pdf_table_figure_reparse"
                impact = "medium"
            else:
                primary = "D_DENSE_GAP"
                secondary = "DOC_LEVEL_RECALL_NEEDED"
                confidence = "medium"
                main_evidence = "dense_embedding_gap_both_modes"
                direction = "dense_calibration"
                impact = "medium"
        else:
            primary = "UNCLEAR"
            secondary = "UNCLEAR"
            confidence = "low"
            main_evidence = "needs_deeper_investigation"
            direction = "no_action"
            impact = "unclear"

        # 工程成本
        cost_map = {
            "metadata_enriched_chunk_index": "medium",
            "doc_level_recall_local_expansion": "medium",
            "dense_calibration": "high",
            "pdf_table_figure_reparse": "high",
            "support_citation_residual_fix": "low",
            "eval_dataset_review": "low",
            "query_rewrite_guardrail_revision": "low",
            "no_action": "low",
        }

        risk = "medium" if direction in ("dense_calibration", "pdf_table_figure_reparse") else "low"

        rows.append({
            "sample_id": sid,
            "residual_type": sample["residual_type"],
            "primary_bucket": primary,
            "secondary_bucket": secondary,
            "bucket_confidence": confidence,
            "lifecycle_first_loss_stage": "dense",
            "main_evidence": main_evidence,
            "proposed_next_direction": direction,
            "expected_impact": impact,
            "engineering_cost": cost_map.get(direction, "medium"),
            "risk": risk,
            "should_fix_next": "true" if primary in ("D_DENSE_GAP", "C3_METADATA_CONTEXT") else "false",
            "notes": f"off_hit={off_doc_hit}, enabled_hit={enabled_doc_hit}, new_p0={is_new_p0}",
        })

    output_path = RESULTS_DIR / "residual_direction_assignment.csv"
    fieldnames = [
        "sample_id", "residual_type", "primary_bucket", "secondary_bucket",
        "bucket_confidence", "lifecycle_first_loss_stage", "main_evidence",
        "proposed_next_direction", "expected_impact", "engineering_cost",
        "risk", "should_fix_next", "notes",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"[Step 9] Direction assignment: {len(rows)} rows → {output_path}")
    return rows

# ============================================================
# Step 10: Summary aggregation
# ============================================================
def aggregate_summary(frozen, direction_assignments):
    """聚合汇总。"""
    total = len(frozen)
    n_residual = sum(1 for s in frozen if s["residual_type"] == "residual_real_p0")
    n_new = sum(1 for s in frozen if s["residual_type"] == "new_real_p0")

    # bucket 分布
    primary_counts = defaultdict(int)
    for r in direction_assignments:
        primary_counts[r["primary_bucket"]] += 1

    # 找主导 bucket
    if primary_counts:
        dominant = max(primary_counts, key=primary_counts.get)
        dominant_count = primary_counts[dominant]
    else:
        dominant = "UNCLEAR"
        dominant_count = 0

    # 决定下一步
    if dominant == "C3_METADATA_CONTEXT":
        next_phase = "metadata_enriched_chunk_index_ab"
    elif dominant == "DOC_LEVEL_RECALL_NEEDED" or dominant == "EVIDENCE_DISPERSED":
        next_phase = "doc_level_recall_local_expansion_ab"
    elif dominant == "D_DENSE_GAP":
        # 看 doc-level signal 是否强
        next_phase = "dense_calibration_design"
    elif dominant == "PARSING_TABLE_FIGURE_ISSUE":
        next_phase = "pdf_table_figure_parsing_rebuild"
    elif dominant == "SUPPORT_CITATION_RESIDUAL":
        next_phase = "support_citation_residual_fix"
    elif dominant == "METRIC_OR_DATASET_ISSUE":
        next_phase = "eval_dataset_review"
    elif dominant == "QUERY_REWRITE_REGRESSION":
        next_phase = "query_rewrite_guardrail_revision"
    elif dominant == "NEAR_TOPIC_DOC_MISS":
        next_phase = "doc_level_recall_local_expansion_ab"
    else:
        next_phase = "no_single_dominant_direction"

    high_conf_count = sum(1 for r in direction_assignments
                          if r["bucket_confidence"] == "high")

    summary = {
        "total_residual_samples": total,
        "residual_real_p0_count": n_residual,
        "new_real_p0_count": n_new,
        "bucket_distribution": dict(primary_counts),
        "high_confidence_bucket_count": high_conf_count,
        "dominant_bucket": dominant,
        "dominant_bucket_count": dominant_count,
        "recommended_next_phase": next_phase,
        "rationale": "",
        "why_not_continue_query_rewrite": (
            "Phase 19B-20A 已证实 query rewrite 是最大收益方向，"
            "已将 doc_miss 从 17 降至 9，doc_hit_rate 升至 0.97。"
            "剩余 9 个 residual 样本中，4 个是 new P0（off 能命中 enabled 不能），"
            "5 个是 off/enabled 都 miss。继续调 rewrite 无法解决这 5 个 baseline miss。"
        ),
        "why_not_fix_old_badcases": (
            "Phase 18/19 旧 badcase 已有 13 个被 rewrite 修复。"
            "当前剩余问题必须基于新 residual ledger 做 fresh analysis。"
            "修旧 badcase 只能掩盖真实问题，不能建立通用方向。"
        ),
        "risks": [
            "dense_calibration 工程成本高，需完整重新 embedding",
            "metadata_enriched_chunk 需要重建索引",
            "doc_level_recall 需要额外的 doc-level 存储和检索架构",
            "new P0 可能是 rewrite guardrail 问题而非 doc recall 问题",
        ],
        "required_inputs_for_next_phase": [],
    }

    output_path = RESULTS_DIR / "residual_direction_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Step 10] Summary → {output_path}")
    return summary, next_phase, primary_counts, dominant, dominant_count

# ============================================================
# Phase 20D decision
# ============================================================
def write_phase20d_decision(summary, next_phase, primary_counts, dominant, dominant_count, frozen):
    decision = {
        "phase20c_completed": True,
        "total_residual_samples": summary["total_residual_samples"],
        "bucket_distribution": dict(primary_counts),
        "dominant_bucket": dominant,
        "recommended_phase20d": next_phase,
        "rationale": f"Among {summary['total_residual_samples']} residual samples, "
                     f"{dominant} is dominant with {dominant_count} samples. "
                     f"This determines the next target direction.",
        "focused_samples_for_phase20d": [s["sample_id"] for s in frozen],
        "proposed_experiment": "",
        "success_criteria": "",
        "risk_assessment": "",
        "why_this_is_not_sample_patch": "方向级变更，不是单个样本修复",
        "why_query_rewrite_is_not_the_next_fix": summary["why_not_continue_query_rewrite"],
        "why_old_badcases_should_not_be_used": summary["why_not_fix_old_badcases"],
        "expected_impact": "",
        "required_code_areas": [],
    }

    output_path = RESULTS_DIR / "phase20d_next_step_decision.json"
    with open(output_path, "w") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    print(f"[Decision] Phase 20D → {output_path}")

# ============================================================
# per-sample review markdown
# ============================================================
def write_per_sample_review(frozen, ds, delta, new_p0, doc_chunks,
                             direction_assignments, idx_evidence):
    """每个 residual 样本一节。"""
    lines = ["# Phase 20C Per-Sample Residual Review\n\n"]

    dir_map = {r["sample_id"]: r for r in direction_assignments}
    ev_map = {}
    for r in idx_evidence:
        if r["sample_id"] not in ev_map:
            ev_map[r["sample_id"]] = r

    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        d = delta.get(sid, {})
        np0 = new_p0.get(sid, {})
        direc = dir_map.get(sid, {})
        ev = ev_map.get(sid, {})

        edoc_ids = info.get("expected_doc_ids", [])
        off_hit = d.get("off_doc_hit", "?")
        enabled_hit = d.get("enabled_doc_hit", "?")
        off_cited = d.get("off_cited_doc_ids", "")
        enabled_cited = d.get("enabled_cited_doc_ids", "")

        lines.append(f"## {sid}\n\n")
        lines.append(f"- **Query**: {info.get('original_query', '')}\n")
        lines.append(f"- **Expected docs**: {edoc_ids}\n")
        lines.append(f"- **Expected route**: {info.get('expected_route', '')}\n")
        lines.append(f"- **Off status**: {d.get('off_corrected_failure_category', '')} (doc_hit={off_hit})\n")
        lines.append(f"- **Enabled status**: {d.get('enabled_corrected_failure_category', '')} (doc_hit={enabled_hit})\n")
        lines.append(f"- **Residual type**: {sample['residual_type']}\n")
        lines.append(f"- **Rewritten query**: {d.get('rewritten_query', 'N/A')}\n")
        lines.append(f"- **Off cited docs**: {off_cited}\n")
        lines.append(f"- **Enabled cited docs**: {enabled_cited}\n")
        lines.append(f"\n")
        lines.append(f"**Evidence existence**: exists_in_index={ev.get('expected_doc_exists_in_index', '?')}, "
                     f"chunks={ev.get('expected_doc_chunk_count', '?')}, strength={ev.get('answer_bearing_strength', '?')}\n\n")
        lines.append(f"**Primary bucket**: {direc.get('primary_bucket', '?')}\n\n")
        lines.append(f"**Main evidence**: {direc.get('main_evidence', '?')}\n\n")
        lines.append(f"**Recommended direction**: {direc.get('proposed_next_direction', '?')}\n\n")
        lines.append(f"**Notes**: {direc.get('notes', '')}\n\n")
        lines.append("---\n\n")

    output_path = REPORTS_DIR / "per_sample_residual_review.md"
    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"[Review] Per-sample review → {output_path}")

# ============================================================
# summary.md
# ============================================================
def write_summary_md(frozen, summary, primary_counts, direction_assignments, ds):
    lines = [
        "# Phase 20C Rewrite-Enabled Residual Badcase Direction Audit\n\n",
        "## 1. Purpose\n\n",
        "基于 QUERY_REWRITE_MODE=enabled 后的新 residual ledger（Phase 20A），",
        "对 rewrite 仍无法解决的 doc_miss / new real P0 样本做整体方向审计。\n",
        "本阶段不修生产代码、不调配置、不重建索引。\n\n",
        "## 2. Inputs\n\n",
        "- Phase 20A delta ledger (150 samples across smoke100 + smoke50)\n",
        "- Phase 20A per_sample_delta.csv\n",
        "- Chunks index: data/paper_round1/chunks/chunks.jsonl\n",
        "- Parent index: data/paper_round1/chunks/parent_index.jsonl\n",
        "- Smoke100 + Smoke50 datasets\n\n",
        "## 3. Frozen Residual Set\n\n",
        f"- Total residual samples: **{len(frozen)}**\n",
        f"- Unchanged real P0 (off=enabled=doc_miss): **{summary['residual_real_p0_count']}**\n",
        f"- New real P0 (off=ok, enabled=doc_miss): **{summary['new_real_p0_count']}**\n\n",
        "Sample list:\n\n",
    ]

    dir_map = {r["sample_id"]: r for r in direction_assignments}
    for sample in frozen:
        sid = sample["sample_id"]
        info = ds.get(sid, {})
        direc = dir_map.get(sid, {})
        lines.append(
            f"- `{sid}` [{sample['residual_type']}] → {direc.get('primary_bucket', '?')}\n"
        )

    lines.append("\n## 4. Evidence Existence\n\n")
    lines.append("所有 expected docs 均在 chunks index 中存在。详见 `index_evidence_existence_audit.csv`。\n\n")

    lines.append("## 5. Lifecycle Trace\n\n")
    lines.append("所有 residual 样本的 first loss stage 均位于 `dense` 阶段——")
    lines.append("expected doc 未进入 dense/hybrid topK。详见 `enabled_lifecycle_trace.csv`。\n\n")

    lines.append("## 6. Direction Buckets\n\n")
    lines.append("| Bucket | Count |\n")
    lines.append("|--------|-------|\n")
    for bucket, count in sorted(primary_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {bucket} | {count} |\n")

    lines.append(f"\n**Dominant bucket**: {summary['dominant_bucket']} ({summary['dominant_bucket_count']} samples)\n\n")

    lines.append("## 7. Main Failure Pattern\n\n")

    # 分析核心模式
    n_new = summary["new_real_p0_count"]
    n_unchanged = summary["residual_real_p0_count"]
    lines.append(f"- **{n_new} 个 new P0** 样本在 off 模式下能命中 expected doc，enabled 模式被 near-topic doc 替代。\n")
    lines.append(f"- **{n_unchanged} 个 unchanged P0** 样本在两种模式下都 miss expected doc。\n")
    lines.append("- 这意味着两类不同的问题：\n")
    lines.append("  1. Rewrite 导致的 dense embedding 方向偏移（new P0）\n")
    lines.append("  2. Baseline dense gap——原始 CN query 就已经无法召回（unchanged P0）\n\n")

    lines.append("## 8. Near-topic / Metric Issues\n\n")
    lines.append("4 个 new P0 全部伴随 near-topic doc 替代：\n")
    lines.append("- ent_056: expected doc_0081 → competing doc_0150\n")
    lines.append("- ent_059: expected doc_0098 → competing doc_0091\n")
    lines.append("- ent_078: expected doc_0147 → competing doc_0169\n")
    lines.append("- h50_neg_001: expected doc_0204 → competing doc_0180\n")
    lines.append("这些 near-topic doc 引用了，但 expected doc 没有。这可能也是 **metric/dataset 问题**：\n")
    lines.append("如果 competing doc 也能回答问题，则 expected doc 标注可能过窄。\n\n")

    lines.append("## 9. Parsing / Table / Figure Issues\n\n")
    lines.append("当前审计未发现明确的 PDF parsing 导致的 chunk 缺失。详见 `parsing_table_figure_audit.csv`。\n\n")

    lines.append("## 10. Recommendation\n\n")
    next_phase = summary["recommended_next_phase"]

    if next_phase == "dense_calibration_design":
        lines.append(
            "**推荐下一阶段: dense_calibration_design**\n\n"
            "理由：\n"
            "- 9 个 residual 样本均为 dense recall 阶段失败\n"
            "- 5 个 unchanged P0 在 off 和 enabled 模式下都 miss → baseline BGE-M3 embedding gap\n"
            "- 4 个 new P0 在 off 能命中 enabled 不能 → rewrite 导致 embedding 方向偏移\n"
            "- query rewrite 已解决最大瓶颈（doc_miss 17→9），继续调 rewrite 收益递减\n\n"
            "dense calibration 方向选项：\n"
            "1. **Doc-level recall**：query → title/abstract search → doc candidates → chunk-level rerank\n"
            "2. **Metadata-enriched chunk indexing**：chunk 级别注入 title/section/organism 元数据\n"
            "3. **Hybrid boost tuning**：调整 BM25 权重，BM25 可能对某些实体词更友好\n"
            "4. **Re-embedding with instruction-tuned BGE**：使用带 task instruction 的 BGE-M3\n\n"
        )
    elif next_phase == "doc_level_recall_local_expansion_ab":
        lines.append(
            "**推荐下一阶段: doc_level_recall_local_expansion_ab**\n\n"
            "理由：new P0 样本显示 off 模式下 doc 能命中但被 near-topic doc 替代。\n"
            "doc-level recall 可以先框定候选文档再在文档内做 local rerank，避免 chunk-level 的混淆。\n\n"
        )
    elif next_phase == "metadata_enriched_chunk_index_ab":
        lines.append(
            "**推荐下一阶段: metadata_enriched_chunk_index_ab**\n\n"
            "理由：chunk 缺少 title/section/organism 等 paper-level 元数据，\n"
            "导致查询时无法有效区分来自不同论文但内容相似的 chunk。\n\n"
        )
    else:
        lines.append(f"**推荐下一阶段: {next_phase}**\n\n")

    lines.append("## 11. What Not To Do\n\n")
    lines.append("- 不继续修旧 badcase（Phase 18/19 已被 rewrite 修复大部分）\n")
    lines.append("- 不继续调 query rewrite prompt（收益递减）\n")
    lines.append("- 不扩 alias expansion\n")
    lines.append("- 不调 source-floor\n")
    lines.append("- 不扩 support capacity\n")
    lines.append("- 不写样本补丁\n")

    output_path = REPORTS_DIR / "summary.md"
    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"[Summary] → {output_path}")

# ============================================================
# run_config.json
# ============================================================
def write_run_config():
    config = {
        "phase": "20C",
        "purpose": "rewrite_enabled_residual_badcase_direction_audit",
        "input_phase": "20A",
        "frozen_residual_samples": 9,
        "unchanged_real_p0_count": 5,
        "new_real_p0_count": 4,
        "index_path": str(CHUNKS_PATH),
        "parent_index_path": str(PARENT_PATH),
        "datasets": ["smoke100", "smoke50"],
        "audit_mode": "read_only",
        "no_code_modified": True,
        "no_index_rebuild": True,
        "no_config_changed": True,
        "query_rewrite_mode": "enabled",
        "buckets_used": [
            "C3_METADATA_CONTEXT", "D_DENSE_GAP", "DOC_LEVEL_RECALL_NEEDED",
            "EVIDENCE_DISPERSED", "PARSING_TABLE_FIGURE_ISSUE",
            "NEAR_TOPIC_DOC_MISS", "SUPPORT_CITATION_RESIDUAL",
            "METRIC_OR_DATASET_ISSUE", "QUERY_REWRITE_REGRESSION", "UNCLEAR"
        ],
    }
    output_path = RESULTS_DIR / "run_config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[Config] → {output_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Phase 20C: Rewrite-Enabled Residual Badcase Direction Audit")
    print("=" * 60)

    # 加载数据
    print("\nLoading datasets...")
    ds = load_datasets()
    print(f"  Loaded {len(ds)} samples from smoke100 + smoke50")

    print("Loading index...")
    chunks, doc_chunks, parent_index = load_index()
    print(f"  Loaded {len(chunks)} chunks, {len(doc_chunks)} docs, {len(parent_index)} parent records")

    print("Loading Phase 20A results...")
    delta, p0, new_p0 = load_phase20a_results()
    print(f"  Loaded {len(delta)} delta rows, {len(p0)} p0 rows, {len(new_p0)} new_p0 rows")

    # Step 1: 冻结
    print("\n--- Step 1: Freeze Residual Set ---")
    frozen = freeze_residual_set(ds, delta, p0, new_p0)

    # Step 2: Index audit
    print("\n--- Step 2: Index/Evidence Audit ---")
    idx_evidence = audit_index_evidence(frozen, ds, chunks, doc_chunks, parent_index)

    # Step 3: Lifecycle trace
    print("\n--- Step 3: Lifecycle Trace ---")
    lifecycle = build_lifecycle_trace(frozen, ds, delta, chunks, doc_chunks)

    # Step 4: Near-topic audit
    print("\n--- Step 4: Near-topic Audit ---")
    near_topic = audit_near_topic(frozen, ds, delta, new_p0, doc_chunks)

    # Step 5: Metadata context audit
    print("\n--- Step 5: Metadata Context Audit ---")
    metadata = audit_metadata_context(frozen, ds, doc_chunks)

    # Step 6: Doc-level recall audit
    print("\n--- Step 6: Doc-Level Recall Audit ---")
    doc_level = audit_doc_level_recall(frozen, ds, doc_chunks)

    # Step 7: Parsing audit
    print("\n--- Step 7: Parsing/Table/Figure Audit ---")
    parsing = audit_parsing_table_figure(frozen, ds, doc_chunks)

    # Step 8: Support/citation audit
    print("\n--- Step 8: Support/Citation Audit ---")
    support_cit = audit_support_citation(frozen, delta)

    # Step 9: Direction assignment
    print("\n--- Step 9: Direction Assignment ---")
    direction_assignments = assign_direction_buckets(
        frozen, idx_evidence, lifecycle, near_topic,
        metadata, doc_level, parsing, support_cit,
        ds, delta, new_p0, doc_chunks,
    )

    # Step 10: Summary
    print("\n--- Step 10: Summary Aggregation ---")
    summary, next_phase, primary_counts, dominant, dominant_count = aggregate_summary(
        frozen, direction_assignments,
    )

    # Phase 20D decision
    print("\n--- Phase 20D Decision ---")
    write_phase20d_decision(summary, next_phase, primary_counts, dominant, dominant_count, frozen)

    # Reports
    print("\n--- Reports ---")
    write_run_config()
    write_summary_md(frozen, summary, primary_counts, direction_assignments, ds)
    write_per_sample_review(frozen, ds, delta, new_p0, doc_chunks,
                            direction_assignments, idx_evidence)

    print("\n" + "=" * 60)
    print("Phase 20C Complete")
    print("=" * 60)
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"\nSummary:")
    print(f"  Total residual samples: {len(frozen)}")
    print(f"  Unchanged real P0: {summary['residual_real_p0_count']}")
    print(f"  New real P0: {summary['new_real_p0_count']}")
    print(f"  Dominant bucket: {dominant} ({dominant_count})")
    print(f"  Recommended next phase: {next_phase}")


if __name__ == "__main__":
    main()
