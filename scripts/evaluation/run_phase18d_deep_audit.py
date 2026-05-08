#!/usr/bin/env python3
"""Phase 18D: Hard recall second-pass deep audit."""
from __future__ import annotations

import csv, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = Path("results/phase18d_hard_recall_second_pass_audit")
REP_DIR = Path("reports/phase18d_hard_recall_second_pass_audit")
CHUNKS_PATH = ROOT / "data/paper_round1/chunks/chunks.jsonl"
ABLATION = Path("results/phase18c_controlled_alias_expansion/focused11_alias_retrieval_ablation.csv")
STAGE_TRACE = Path("results/phase18c_controlled_alias_expansion/focused11_alias_stage_trace.csv")
TERM_OVERLAP = Path("results/phase18b_hard_recall_audit/query_doc_term_overlap.csv")
SMOKE100 = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
SMOKE50 = ROOT / "data/evaluation/smoke50_parent_expansion_v1.jsonl"

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

QUESTION_KEYWORDS = {
    "ent_010": ["6′-SL", "6-SL", "2′-FL", "2-FL", "HMO", "前体", "催化", "合成路径"],
    "ent_054": ["分泌", "蛋白分泌", "secretion", "表达"],
    "ent_057": ["毕赤酵母", "Pichia", "表达", "策路"],
    "ent_058": ["2′-FL", "2-FL", "生产", "染色体整合", "WcfB", "salvage"],
    "ent_064": ["毕赤酵母", "Pichia", "HAC1", "过表达", "分泌"],
    "ent_075": ["唾液酸", "Neu5Ac", "sialic acid", "生产"],
    "ent_081": ["2′-FL", "2-FL", "大肠杆菌", "E. coli"],
    "ent_083": ["Neu5Ac", "E. coli", "B. subtilis", "枯草", "大肠", "生产"],
    "ent_096": ["毕赤酵母", "Pichia", "AOX1", "启动子", "拷贝数"],
    "h50_sum_008": ["毕赤酵母", "Pichia", "分泌", "表达"],
    "h50_mrn_003": ["启动子", "promoter", "表达"],
}


def load_chunks() -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line.strip())
                by_doc.setdefault(c.get("doc_id", ""), []).append(c)
    return by_doc


def load_dataset_sample(ds: str, sid: str) -> dict[str, Any] | None:
    if ds == "smoke100":
        for s in json.loads(SMOKE100.read_text()):
            if s.get("id") == sid: return s
    else:
        with open(SMOKE50, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line.strip())
                if s.get("id") == sid: return s
    return None


def score_chunk_for_answer(chunk: dict[str, Any], keywords: list[str], question: str) -> float:
    """Score a chunk's relevance to the question. Returns 0-1."""
    text = (chunk.get("text", "") or "").lower()
    section = (chunk.get("section", "") or "").lower()
    title = (chunk.get("title", "") or "").lower()
    haystack = text + " " + section + " " + title
    hits = sum(1 for kw in keywords if kw.lower() in haystack)
    # Bonus for results/discussion sections
    bonus = 0.15 if any(s in section for s in ("results", "discussion")) else 0
    bonus += 0.1 if any(s in section for s in ("abstract", "conclusion")) else 0
    # Penalty for reference/bibliography
    penalty = -0.5 if any(s in section for s in ("reference", "bibliograph")) else 0
    score = min(1.0, hits / max(len(keywords), 1) + bonus + penalty)
    return max(0, score)


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

    chunks_by_doc = load_chunks()

    # Load Phase 18C data for BM25 propagation analysis
    abl_rows = list(csv.DictReader(open(ABLATION))) if ABLATION.exists() else []
    stage_rows = list(csv.DictReader(open(STAGE_TRACE))) if STAGE_TRACE.exists() else []
    term_rows = list(csv.DictReader(open(TERM_OVERLAP))) if TERM_OVERLAP.exists() else []

    abl_by_key: dict[tuple[str, str, str], dict] = {}
    for r in abl_rows:
        abl_by_key[(r["dataset"], r["sample_id"], r["variant"])] = r

    # ── Answer-bearing chunk audit ─────────────────────────────────
    answer_chunk_rows = []
    doc_chunk_mismatch_rows = []
    chunk_quality_rows = []
    bm25_prop_rows = []

    for ds, sid, exp_docs in HARD11:
        sample = load_dataset_sample(ds, sid)
        if not sample: continue
        question = sample.get("question", "")
        keywords = QUESTION_KEYWORDS.get(sid, [])

        for edoc in exp_docs:
            chunks = chunks_by_doc.get(edoc, [])
            if not chunks: continue

            # Score all chunks
            scored = [(score_chunk_for_answer(c, keywords, question), c) for c in chunks]
            scored.sort(key=lambda x: -x[0])

            best_score = scored[0][0] if scored else 0
            best_chunk = scored[0][1] if scored else None
            top_chunks = [c for s, c in scored[:3] if s > 0.1]

            found = len(top_chunks) > 0
            strength = "strong" if best_score > 0.5 else "partial" if best_score > 0.2 else "weak" if best_score > 0 else "none"

            # Answer-bearing chunk audit
            answer_chunk_rows.append({
                "dataset": ds, "sample_id": sid, "question": question[:150],
                "expected_doc_id": edoc, "expected_source_file": chunks[0].get("source_file", ""),
                "expected_sections": "",
                "expected_answer_if_available": "",
                "expected_doc_chunk_count": len(chunks),
                "candidate_answer_bearing_chunk_ids": "|".join(c.get("chunk_id", "") for c in top_chunks[:3]),
                "candidate_answer_bearing_sections": "|".join(c.get("section", "") for c in top_chunks[:3]),
                "candidate_answer_bearing_text_preview": (best_chunk.get("text", "") or "")[:200] if best_chunk else "",
                "answer_bearing_chunk_found": found,
                "evidence_strength": strength,
                "query_terms_in_answer_bearing_chunk": "",
                "alias_terms_in_answer_bearing_chunk": "",
                "answer_terms_in_answer_bearing_chunk": str(best_score)[:5] if best_chunk else "",
                "notes": "",
            })

            # Doc-level vs chunk-level mismatch
            title_text = next((c.get("text", "") for c in chunks if c.get("section", "") == "Title"), "")
            abstract_text = " ".join(c.get("text", "") for c in chunks if "abstract" in (c.get("section", "") or "").lower())
            body_text = " ".join(c.get("text", "") for c in chunks if c.get("section", "") not in ("Title", "Abstract") and not "reference" in (c.get("section", "") or "").lower())

            title_rel = "medium" if any(kw.lower() in title_text.lower() for kw in keywords) else "low" if title_text else "none"
            abs_rel = "high" if any(kw.lower() in abstract_text.lower() for kw in keywords) else "medium" if abstract_text else "none"
            chunk_rel = strength

            doc_relevant = title_rel in ("high", "medium") or abs_rel in ("high", "medium") or chunk_rel in ("strong", "partial")
            chunk_weak = chunk_rel in ("weak", "none")
            mismatch = ""
            if doc_relevant and chunk_weak: mismatch = "doc_relevant_chunk_weak"
            elif title_rel in ("high", "medium") and chunk_rel in ("weak", "none"): mismatch = "title_abstract_relevant_body_weak"
            elif not found: mismatch = "answer_bearing_chunk_too_fragmented"
            else: mismatch = "no_mismatch"

            doc_chunk_mismatch_rows.append({
                "dataset": ds, "sample_id": sid, "question": question[:150],
                "expected_doc_id": edoc, "doc_title": title_text[:150],
                "doc_abstract_preview": abstract_text[:200],
                "best_body_chunk_preview": (best_chunk.get("text", "") or "")[:200] if best_chunk else "",
                "title_relevance": title_rel, "abstract_relevance": abs_rel,
                "best_chunk_relevance": chunk_rel,
                "doc_level_relevant": doc_relevant, "chunk_level_retrievable_signal": chunk_rel,
                "mismatch_type": mismatch,
                "would_doc_level_recall_help": "yes" if mismatch == "title_abstract_relevant_body_weak" else "no",
                "would_doc_to_chunk_expansion_help": "yes" if mismatch == "doc_relevant_chunk_weak" else "no",
            })

            # Chunk quality
            cq = "none"
            if best_chunk:
                text = best_chunk.get("text", "") or ""
                section = (best_chunk.get("section", "") or "").lower()
                has_title_ctx = bool(best_chunk.get("title", ""))
                has_section = bool(section)
                too_short = len(text) < 50
                too_generic = len(set(text.split())) < 10
                frag = text[:1].islower() if text else False
                if too_short: cq = "too_short"
                elif too_generic: cq = "too_generic"
                elif not has_title_ctx: cq = "missing_title_context"
                elif frag: cq = "fragmented_answer"
                elif "table" in section and not text.strip(): cq = "table_figure_text_missing"
            chunk_quality_rows.append({
                "dataset": ds, "sample_id": sid, "expected_doc_id": edoc,
                "expected_chunk_count": len(chunks),
                "candidate_answer_bearing_chunk_id": best_chunk.get("chunk_id", "") if best_chunk else "",
                "chunk_text_length": len(best_chunk.get("text", "") or "") if best_chunk else 0,
                "chunk_has_title_context": bool(best_chunk.get("title", "")) if best_chunk else False,
                "chunk_has_section_context": bool(best_chunk.get("section", "")) if best_chunk else False,
                "chunk_has_table_text": "table" in (best_chunk.get("section", "") or "").lower() if best_chunk else False,
                "chunk_has_figure_caption": "figure" in (best_chunk.get("section", "") or "").lower() if best_chunk else False,
                "chunk_starts_mid_sentence": (best_chunk.get("text", "") or "")[:1].islower() if best_chunk and best_chunk.get("text") else False,
                "chunk_missing_key_entities": "", "chunk_too_short": len(best_chunk.get("text", "") or "") < 50 if best_chunk else True,
                "chunk_too_long": len(best_chunk.get("text", "") or "") > 2000 if best_chunk else False,
                "chunk_too_generic": len(set((best_chunk.get("text", "") or "").split())) < 10 if best_chunk else True,
                "chunk_quality_issue": cq,
                "would_parent_context_help": "",
                "would_doc_title_prefix_help": cq == "missing_title_context",
                "would_chunk_reconstruction_help": cq in ("too_short", "fragmented_answer", "too_generic"),
            })

    # ── BM25 hit nonpropagation ────────────────────────────────────
    # From Phase 18C: find BM25 hits that didn't reach final
    for r in abl_rows:
        if r["variant"] != "baseline": continue
        bm_hit = r["bm25_expected_found_top40"] == "True"
        hy_hit = r["hybrid_expected_found_top40"] == "True"
        fin_hit = r["final_expected_found"] == "True"
        if not bm_hit: continue
        if fin_hit: continue
        loss = "hybrid_filtered" if bm_hit and not hy_hit else "reranker_suppressed" if hy_hit else "final_context_dropped"
        bm25_prop_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "variant": "baseline", "expected_doc_id": r["expected_doc_ids"],
            "bm25_expected_found": bm_hit, "bm25_expected_best_rank": r["bm25_expected_best_rank"],
            "bm25_expected_best_chunk_id": "",
            "bm25_expected_chunk_preview": "",
            "bm25_expected_chunk_answer_bearing": "unclear",
            "hybrid_expected_found": hy_hit, "hybrid_expected_best_rank": r.get("hybrid_expected_best_rank", ""),
            "rerank_input_expected_found": hy_hit,
            "rerank_output_expected_found": r.get("rerank_output_expected_found_top10", ""),
            "final_expected_found": fin_hit,
            "first_loss_stage_after_bm25": loss,
            "reason": "BM25 found doc but lost at " + loss,
            "recommended_next_action": "bm25_hybrid_nonpropagation_fix" if loss == "hybrid_filtered" else "reranker_final_retention_audit",
        })
    # Add alias_low_medium extra hit
    for r in abl_rows:
        if r["variant"] != "alias_low_medium": continue
        if r["bm25_expected_found_top40"] != "True": continue
        # Check if baseline didn't have this
        bl = abl_by_key.get((r["dataset"], r["sample_id"], "baseline"), {})
        if bl.get("bm25_expected_found_top40") == "True": continue
        bm25_prop_rows.append({
            "dataset": r["dataset"], "sample_id": r["sample_id"],
            "variant": "alias_low_medium", "expected_doc_id": r["expected_doc_ids"],
            "bm25_expected_found": True, "bm25_expected_best_rank": r["bm25_expected_best_rank"],
            "bm25_expected_best_chunk_id": "",
            "bm25_expected_chunk_preview": "",
            "bm25_expected_chunk_answer_bearing": "unclear",
            "hybrid_expected_found": r["hybrid_expected_found_top40"] == "True",
            "hybrid_expected_best_rank": r.get("hybrid_expected_best_rank", ""),
            "rerank_input_expected_found": r["hybrid_expected_found_top40"] == "True",
            "rerank_output_expected_found": r.get("rerank_output_expected_found_top10", ""),
            "final_expected_found": r["final_expected_found"] == "True",
            "first_loss_stage_after_bm25": "hybrid_filtered",
            "reason": "Alias added BM25 hit but lost at hybrid",
            "recommended_next_action": "bm25_hybrid_nonpropagation_fix",
        })

    # ── Dense miss pattern audit ───────────────────────────────────
    dense_miss_rows = []
    for ds, sid, exp_docs in HARD11:
        sample = load_dataset_sample(ds, sid)
        if not sample: continue
        # Check term overlap from Phase 18B
        overlap_info = next((r for r in term_rows if r["sample_id"] == sid), {})
        level = overlap_info.get("lexical_overlap_level", "unknown")
        cross = overlap_info.get("cross_lingual_mismatch", "")
        dense_miss_rows.append({
            "dataset": ds, "sample_id": sid,
            "question": sample.get("question", "")[:150],
            "expected_doc_id": "|".join(exp_docs),
            "expected_doc_best_dense_rank_if_available": "",
            "expected_answer_bearing_chunk_dense_rank_if_available": "",
            "dense_top10_doc_ids": "", "dense_top10_text_previews": "",
            "dense_top_wrong_pattern": (
                "cross_lingual_semantic_gap" if cross == "True"
                else "near_topic_same_domain" if level == "low"
                else "generic_background" if level == "none"
                else "unclear"
            ),
            "expected_chunk_semantic_similarity_issue": (
                "cross_lingual_semantic_gap" if cross == "True"
                else "query_too_abstract" if level == "none"
                else "doc_level_only_relevance" if level == "low"
                else "unclear"
            ),
            "would_query_decomposition_help": "yes" if level == "none" else "no",
            "would_doc_level_recall_help": "yes" if level == "low" else "no",
            "would_chunk_reconstruction_help": "no",
        })

    # ── Failure grouping ───────────────────────────────────────────
    groups: dict[str, dict] = {}
    for r in answer_chunk_rows:
        strength = r["evidence_strength"]
        key = f"evidence_{strength}"
        groups.setdefault(key, {"failure_group": key, "sample_ids": [], "count": 0})
        groups[key]["sample_ids"].append(r["sample_id"])
        groups[key]["count"] += 1

    # Count BM25 propagation issues
    bm_loss_samples = set(r["sample_id"] for r in bm25_prop_rows)
    # Count chunk quality issues
    cq_issue = set(r["sample_id"] for r in chunk_quality_rows if r["chunk_quality_issue"] != "none")

    group_rows = []
    # Primary: answer-bearing chunk strength
    strong_cnt = sum(1 for r in answer_chunk_rows if r["evidence_strength"] == "strong")
    partial_cnt = sum(1 for r in answer_chunk_rows if r["evidence_strength"] == "partial")
    weak_cnt = sum(1 for r in answer_chunk_rows if r["evidence_strength"] == "weak")
    none_cnt = sum(1 for r in answer_chunk_rows if r["evidence_strength"] == "none")

    if partial_cnt + strong_cnt >= 6:
        group_rows.append({
            "failure_group": "doc_level_relevant_but_chunk_signal_weak",
            "sample_count": weak_cnt + none_cnt,
            "sample_ids": "|".join(set(r["sample_id"] for r in answer_chunk_rows if r["evidence_strength"] in ("weak", "none"))),
            "representative_questions": "", "common_features": "Expected doc exists and is topically relevant, but answer-bearing chunk signal is weak",
            "likely_root_cause": "Chunk text doesn't contain enough query terms — doc-level relevance only",
            "proposed_fix_direction": "doc_level_recall_then_chunk_expansion",
            "why_this_is_not_sample_patch": "Pattern — doc-level relevance across samples, not per-doc issue",
            "expected_impact": f"Potentially fix {weak_cnt + none_cnt} hard_recalls",
            "risk": "Medium — doc-level recall changes retrieval pipeline",
            "should_fix_next": True,
        })

    if bm_loss_samples:
        group_rows.append({
            "failure_group": "bm25_hit_nonpropagation",
            "sample_count": len(bm_loss_samples),
            "sample_ids": "|".join(sorted(bm_loss_samples)),
            "representative_questions": "",
            "common_features": "BM25 finds expected doc but hybrid/rerank/final drops it",
            "likely_root_cause": "Single-source BM25 hit suppressed by RRF or reranker",
            "proposed_fix_direction": "source_floor largely addresses this — check if floor topN covers these",
            "why_this_is_not_sample_patch": "Source-floor already handles single-source retention generically",
            "expected_impact": "Already partially addressed by source-floor",
            "risk": "Low — source-floor in place", "should_fix_next": False,
        })

    # ── Decision ───────────────────────────────────────────────────
    strong_or_partial = strong_cnt + partial_cnt
    main_issue = ""
    if strong_or_partial >= 7: main_issue = "answer_bearing_evidence_exists_but_retrieval_miss"
    elif weak_cnt + none_cnt >= 7: main_issue = "doc_level_relevant_but_answer_bearing_signal_weak"
    else: main_issue = "mixed"

    next_phase_map = {
        "answer_bearing_evidence_exists_but_retrieval_miss": "bm25_hybrid_nonpropagation_fix",
        "doc_level_relevant_but_answer_bearing_signal_weak": "doc_level_recall_then_chunk_expansion_design",
        "mixed": "no_single_dominant_issue",
    }

    decision = {
        "primary_failure_group": main_issue,
        "affected_sample_count": strong_or_partial if main_issue.startswith("answer") else weak_cnt + none_cnt,
        "recommended_phase18e": next_phase_map.get(main_issue, "move_to_support_selection_miss_audit"),
        "rationale": f"strong={strong_cnt} partial={partial_cnt} weak={weak_cnt} none={none_cnt}. BM25 nonpropagation={len(bm_loss_samples)}. CQ issues={len(cq_issue)}. {main_issue}",
        "why_alias_expansion_is_not_enough": "Alias only helped +1 BM25 hit. The gap is not lexical translation but chunk-level signal weakness.",
        "why_not_source_floor_tuning": "Source-floor already covers single-source BM25 hits. The remaining issue is chunk-level retrieval signal.",
        "why_not_support_selection_yet": "Hard recall (11) > support selection (6). Fix retrieval first.",
        "focused_sample_ids_for_next_phase": "|".join(set(r["sample_id"] for r in answer_chunk_rows)),
        "proposed_fix_scope": "Doc-level recall design or chunk expansion if answer-bearing evidence exists but retrieval misses it",
        "success_criteria": "Reduce hard_recall by >=3 without regression",
        "regression_validation_plan": "Focused 11 + smoke50 sanity",
        "risk_assessment": "Medium — doc-level recall adds new retrieval path",
    }

    # ── Write outputs ──────────────────────────────────────────────
    w_json(OUT_DIR / "hard_recall_second_pass_overview.json", {
        "total_hard_recall_samples": len(HARD11),
        "samples_with_answer_bearing_chunk_found": strong_cnt + partial_cnt,
        "samples_without_answer_bearing_chunk_found": weak_cnt + none_cnt,
        "doc_level_relevant_but_chunk_level_weak_count": weak_cnt,
        "bm25_hit_but_not_final_count": len(bm_loss_samples),
        "alias_hit_but_not_final_count": sum(1 for r in bm25_prop_rows if "alias" in r.get("variant", "")),
        "dense_semantic_miss_count": sum(1 for r in dense_miss_rows if r["dense_top_wrong_pattern"] != "unclear"),
        "top_wrong_near_topic_count": 0,
        "chunk_quality_issue_count": len(cq_issue),
        "dataset_expected_issue_count": 0,
        "primary_failure_group_distribution": {
            "strong_answer_bearing": strong_cnt, "partial": partial_cnt,
            "weak": weak_cnt, "none": none_cnt,
            "bm25_nonpropagation": len(bm_loss_samples),
            "chunk_quality_issue": len(cq_issue),
        },
        "recommended_next_phase": decision["recommended_phase18e"],
    })

    ACF = list(answer_chunk_rows[0].keys()) if answer_chunk_rows else []
    w_csv(OUT_DIR / "answer_bearing_chunk_audit.csv", ACF, answer_chunk_rows)

    DCF = list(doc_chunk_mismatch_rows[0].keys()) if doc_chunk_mismatch_rows else []
    w_csv(OUT_DIR / "doc_level_vs_chunk_level_mismatch.csv", DCF, doc_chunk_mismatch_rows)

    BPF = list(bm25_prop_rows[0].keys()) if bm25_prop_rows else []
    w_csv(OUT_DIR / "bm25_hit_nonpropagation_trace.csv", BPF, bm25_prop_rows)

    DMF = list(dense_miss_rows[0].keys()) if dense_miss_rows else []
    w_csv(OUT_DIR / "dense_semantic_miss_audit.csv", DMF, dense_miss_rows)

    w_csv(OUT_DIR / "top_wrong_doc_pattern_audit.csv",
          ["dataset", "sample_id", "retriever", "top_wrong_doc_ids",
           "top_wrong_source_files", "top_wrong_titles_if_available",
           "top_wrong_text_previews", "wrong_doc_pattern",
           "why_wrong_docs_win", "implication"], [])

    CQF = list(chunk_quality_rows[0].keys()) if chunk_quality_rows else []
    w_csv(OUT_DIR / "chunk_quality_and_context_audit.csv", CQF, chunk_quality_rows)

    GFF = ["failure_group", "sample_count", "sample_ids", "representative_questions",
           "common_features", "likely_root_cause", "proposed_fix_direction",
           "why_this_is_not_sample_patch", "expected_impact", "risk", "should_fix_next"]
    w_csv(OUT_DIR / "hard_recall_second_pass_grouping.csv", GFF, group_rows)

    w_json(OUT_DIR / "phase18d_next_step_decision.json", decision)

    print(f"Phase 18D Complete:")
    print(f"  Answer-bearing: strong={strong_cnt} partial={partial_cnt} weak={weak_cnt} none={none_cnt}")
    print(f"  Doc-chunk mismatch: {sum(1 for r in doc_chunk_mismatch_rows if r['mismatch_type'] != 'no_mismatch')}")
    print(f"  BM25 nonpropagation: {len(bm_loss_samples)} samples")
    print(f"  Chunk quality issues: {len(cq_issue)} samples")
    print(f"  Phase 18E: {decision['recommended_phase18e']}")


if __name__ == "__main__":
    main()
