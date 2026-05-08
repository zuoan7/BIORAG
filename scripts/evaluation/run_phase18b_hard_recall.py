#!/usr/bin/env python3
"""Phase 18B: Hard recall query-doc mismatch audit for 11 samples."""
from __future__ import annotations

import csv, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = Path("results/phase18b_hard_recall_audit")
REP_DIR = Path("reports/phase18b_hard_recall_audit")
CHUNKS = ROOT / "data/paper_round1/chunks/chunks.jsonl"
TAXO = Path("results/phase18a_post_source_floor_residual_audit/residual_p0_taxonomy.csv")
SMOKE100_DS = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
SMOKE50_DS = ROOT / "data/evaluation/smoke50_parent_expansion_v1.jsonl"

# Known domain aliases (for auditing, NOT for injecting)
ALIAS_PAIRS: list[tuple[str, str, str]] = [
    ("人乳寡糖", "HMO", "Chinese_to_English"),
    ("毕赤酵母", "Pichia", "organism_alias"),
    ("大肠杆菌", "E. coli", "organism_alias"),
    ("枯草芽孢杆菌", "B. subtilis", "organism_alias"),
    ("唾液酸", "Neu5Ac", "product_alias"),
    ("2′-FL", "2-fucosyllactose", "product_alias"),
    ("6′-SL", "6-sialyllactose", "product_alias"),
    ("糖苷酶", "glycosidase", "enzyme_gene_alias"),
    ("转糖苷", "transglycosylation", "pathway_alias"),
    ("启动子", "promoter", "abbreviation_to_full_name"),
    ("拷贝数", "copy number", "abbreviation_to_full_name"),
    ("异源表达", "heterologous expression", "abbreviation_to_full_name"),
    ("分泌", "secretion", "abbreviation_to_full_name"),
    ("甲醇", "methanol", "abbreviation_to_full_name"),
    ("诱导", "induction", "abbreviation_to_full_name"),
]


def load_chunks() -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    if not CHUNKS.exists():
        return by_doc
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line.strip())
            doc_id = c.get("doc_id", "")
            by_doc.setdefault(doc_id, []).append(c)
    return by_doc


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line.strip())
                by_id[s.get("id", s.get("sample_id", ""))] = s
    else:
        for s in json.loads(path.read_text()):
            by_id[s.get("id", "")] = s
    return by_id


def tokenize_cjk(text: str) -> set[str]:
    tokens: set[str] = set()
    cjk_re = re.compile(r"[\u4e00-\u9fff]{2,4}")
    for m in cjk_re.finditer(text):
        tokens.add(m.group().lower())
    return tokens


def tokenize_en(text: str) -> set[str]:
    en_re = re.compile(r"[a-zA-Z][a-zA-Z0-9]{2,}")
    return {m.group().lower() for m in en_re.finditer(text.lower())}


def text_terms(text: str) -> set[str]:
    return tokenize_cjk(text) | tokenize_en(text)


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

    # Load data
    chunks_by_doc = load_chunks()
    s100_by_id = load_dataset(SMOKE100_DS)
    s50_by_id = load_dataset(SMOKE50_DS)
    dataset_by_id = {**s100_by_id, **s50_by_id}

    # Get 11 hard_recall IDs
    hard_ids: list[tuple[str, str, list[str]]] = []
    with open(TAXO) as f:
        for row in csv.DictReader(f):
            if row.get("primary_failure_class") == "hard_recall_miss":
                ds = row["dataset"]
                sid = row["sample_id"]
                exp_docs = [d for d in (row.get("expected_doc_ids", "") or "").split("|") if d]
                hard_ids.append((ds, sid, exp_docs))

    print(f"Hard recall samples: {len(hard_ids)}")

    # ── Chunk presence check ──────────────────────────────────────
    chunk_presence_rows = []
    docs_missing = 0
    docs_found = 0

    for ds, sid, exp_docs in hard_ids:
        sample = dataset_by_id.get(sid, {})
        question = sample.get("question", "")
        for edoc in exp_docs:
            chunks = chunks_by_doc.get(edoc, [])
            sections = set(c.get("section", "") for c in chunks)
            has_title = any(c.get("section", "") == "Title" for c in chunks)
            has_abstract = any("abstract" in (c.get("section", "") or "").lower() for c in chunks)
            has_body = any(
                s.lower() in ("results", "discussion", "introduction", "methods", "conclusion", "full text")
                for s in sections)
            has_table = any("table" in (c.get("section", "") or "").lower() for c in chunks)
            has_figure = any("figure" in (c.get("section", "") or "").lower() for c in chunks)
            has_exp_section = any(
                s.lower() in (ss or "").lower() for ss in sample.get("expected_sections", []) or []
                for s in sections)

            note = ""
            if not chunks:
                note = "DOC_MISSING_FROM_INDEX"
                docs_missing += 1
            else:
                docs_found += 1

            chunk_presence_rows.append({
                "dataset": ds, "sample_id": sid,
                "expected_doc_id": edoc,
                "expected_source_file": sample.get("expected_source_files", [""])[0] if sample.get("expected_source_files") else "",
                "found_doc_id_in_chunks": edoc if chunks else "MISSING",
                "found_source_file_in_chunks": chunks[0].get("source_file", "") if chunks else "",
                "chunk_count": len(chunks),
                "section_count": len(sections),
                "has_title_metadata": has_title,
                "has_abstract_metadata": has_abstract,
                "has_body_chunks": has_body,
                "has_table_chunks": has_table,
                "has_figure_chunks": has_figure,
                "has_expected_section": has_exp_section,
                "metadata_consistency_status": (
                    "ok" if chunks and len(chunks) > 0
                    else "doc_id_missing" if not chunks
                    else "unclear"
                ),
                "notes": note,
            })

    # ── Query-doc term overlap ────────────────────────────────────
    term_overlap_rows = []
    alias_rows = []

    # Build title/abstract map from chunks
    doc_title: dict[str, str] = {}
    doc_abstract: dict[str, str] = {}
    for doc_id, chunks in chunks_by_doc.items():
        for c in chunks:
            s = (c.get("section", "") or "").lower()
            if s == "title":
                doc_title[doc_id] = c.get("text", "")
            if "abstract" in s:
                doc_abstract[doc_id] = (doc_abstract.get(doc_id, "") + " " + c.get("text", ""))

    for ds, sid, exp_docs in hard_ids:
        sample = dataset_by_id.get(sid, {})
        question = sample.get("question", "")
        q_terms = text_terms(question)
        latin = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", question)
        cjk = re.findall(r"[\u4e00-\u9fff]{2,4}", question)

        for edoc in exp_docs:
            chunks = chunks_by_doc.get(edoc, [])
            if not chunks:
                term_overlap_rows.append({
                    "dataset": ds, "sample_id": sid,
                    "question": question[:200],
                    "normalized_question_terms": "",
                    "biomedical_terms": "|".join(latin[:10]),
                    "latin_terms": "|".join(latin[:10]),
                    "chinese_domain_terms": "|".join(cjk[:10]),
                    "numbers_or_units": "",
                    "expected_doc_id": edoc,
                    "expected_doc_title_terms": "",
                    "expected_doc_abstract_terms": "",
                    "expected_doc_body_top_terms": "",
                    "query_title_overlap_count": 0, "query_abstract_overlap_count": 0,
                    "query_body_overlap_count": 0, "query_expected_section_overlap_count": 0,
                    "answer_term_overlap_count_if_available": 0,
                    "lexical_overlap_level": "none" if not chunks else "low",
                    "cross_lingual_mismatch": False, "notes": "DOC_MISSING",
                })
                continue

            title_text = doc_title.get(edoc, "")
            abstract_text = doc_abstract.get(edoc, "")
            body_text = " ".join(c.get("text", "") for c in chunks[:20] if c.get("section", "") not in ("Title", "Abstract"))

            title_terms = text_terms(title_text)
            abstract_terms = text_terms(abstract_text)
            body_terms = text_terms(body_text)

            q_title_overlap = len(q_terms & title_terms)
            q_abs_overlap = len(q_terms & abstract_terms)
            q_body_overlap = len(q_terms & body_terms)

            # Overlap level
            max_overlap = max(q_title_overlap, q_abs_overlap, q_body_overlap)
            if max_overlap >= 5: level = "high"
            elif max_overlap >= 3: level = "medium"
            elif max_overlap >= 1: level = "low"
            else: level = "none"

            # Cross-lingual: CJK question + mostly EN body or vice versa
            has_cjk_q = bool(cjk)
            has_en_body = bool(re.findall(r"[a-zA-Z]{3,}", body_text))
            cross = has_cjk_q and has_en_body

            term_overlap_rows.append({
                "dataset": ds, "sample_id": sid,
                "question": question[:200],
                "normalized_question_terms": "|".join(sorted(q_terms)[:20]),
                "biomedical_terms": "|".join(latin[:10]),
                "latin_terms": "|".join(latin[:10]),
                "chinese_domain_terms": "|".join(cjk[:10]),
                "numbers_or_units": "",
                "expected_doc_id": edoc,
                "expected_doc_title_terms": title_text[:120],
                "expected_doc_abstract_terms": abstract_text[:200],
                "expected_doc_body_top_terms": "",
                "query_title_overlap_count": q_title_overlap,
                "query_abstract_overlap_count": q_abs_overlap,
                "query_body_overlap_count": q_body_overlap,
                "query_expected_section_overlap_count": 0,
                "answer_term_overlap_count_if_available": 0,
                "lexical_overlap_level": level,
                "cross_lingual_mismatch": cross,
                "notes": "",
            })

            # Alias gap check
            for cn_term, en_term, gap_type in ALIAS_PAIRS:
                has_cn = cn_term.lower() in question.lower()
                has_en = en_term.lower() in body_text.lower() or en_term.lower() in title_text.lower() or en_term.lower() in abstract_text.lower()
                if has_cn and not has_en:
                    pass  # Chinese term in question but doc only has English — potential gap
                if has_cn:
                    q_has_en = en_term.lower() in question.lower()
                    doc_has_cn = cn_term.lower() in body_text.lower() + title_text.lower() + abstract_text.lower()
                    # True gap: question has CN, doc doesn't have CN equivalent
                    # OR question has EN, doc doesn't have EN equivalent
                    gap = (has_cn and not doc_has_cn) or (has_en and not q_has_en and not has_cn)
                    if gap:
                        alias_rows.append({
                            "dataset": ds, "sample_id": sid,
                            "question": question[:120],
                            "expected_doc_id": edoc,
                            "question_terms": cn_term if has_cn else en_term,
                            "expected_doc_terms": en_term if has_cn else cn_term,
                            "suspected_alias_pairs": f"{cn_term} <-> {en_term}",
                            "alias_gap_type": gap_type,
                            "would_controlled_alias_expansion_help": "yes",
                            "risk_of_alias_expansion": "low" if gap_type in ("product_alias", "organism_alias") else "medium",
                            "notes": "",
                        })

    # ── Title/abstract signal audit ────────────────────────────────
    title_abs_rows = []
    for ds, sid, exp_docs in hard_ids:
        sample = dataset_by_id.get(sid, {})
        question = sample.get("question", "")
        q_terms = text_terms(question)
        for edoc in exp_docs:
            chunks = chunks_by_doc.get(edoc, [])
            if not chunks: continue
            title_text = doc_title.get(edoc, "")
            abstract_text = doc_abstract.get(edoc, "")
            body_chunks = [c for c in chunks if c.get("section", "") not in ("Title", "Abstract")]
            body_text = " ".join(c.get("text", "") for c in body_chunks[:10])
            t_overlap = len(q_terms & text_terms(title_text))
            a_overlap = len(q_terms & text_terms(abstract_text))
            b_overlap = len(q_terms & text_terms(body_text))
            ta_signal = t_overlap + a_overlap > 0
            body_signal = b_overlap > 0
            ta_only = ta_signal and not body_signal
            would_help = ta_only and t_overlap + a_overlap >= 2

            title_abs_rows.append({
                "dataset": ds, "sample_id": sid,
                "expected_doc_id": edoc, "question": question[:150],
                "title": title_text[:150],
                "abstract_preview": abstract_text[:200],
                "body_chunk_best_overlap": str(b_overlap),
                "title_overlap": t_overlap, "abstract_overlap": a_overlap,
                "body_overlap": b_overlap,
                "title_or_abstract_has_signal": ta_signal,
                "body_chunks_have_signal": body_signal,
                "title_abstract_only_signal": ta_only,
                "would_doc_level_title_abstract_recall_help": "yes" if would_help else "no",
                "evidence": f"title={t_overlap} abs={a_overlap} body={b_overlap}",
            })

    # ── Query-doc audit main rows ──────────────────────────────────
    audit_rows = []
    for ds, sid, exp_docs in hard_ids:
        sample = dataset_by_id.get(sid, {})
        for edoc in exp_docs:
            chunks = chunks_by_doc.get(edoc, [])
            exists = len(chunks) > 0
            q = sample.get("question", "")
            q_terms = text_terms(q)
            # Check title/abstract vs body signal
            t_text = doc_title.get(edoc, "")
            a_text = doc_abstract.get(edoc, "")
            b_chunks = [c for c in chunks if c.get("section", "") not in ("Title", "Abstract")]
            b_text = " ".join(c.get("text", "") for c in b_chunks[:10])
            has_q_in_title = any(t.lower() in t_text.lower() for t in q_terms if len(t) > 3)
            has_q_in_abstract = any(t.lower() in a_text.lower() for t in q_terms if len(t) > 3)
            has_q_in_body = any(t.lower() in b_text.lower() for t in q_terms if len(t) > 3)

            # Determine primary cause
            cause = "unclear"
            if not exists:
                cause = "expected_doc_missing_from_index"
            elif not has_q_in_title and not has_q_in_abstract and not has_q_in_body:
                cause = "query_doc_lexical_mismatch"
            elif has_q_in_title and not has_q_in_body:
                cause = "title_abstract_signal_missing_from_chunk_index"
            elif has_q_in_abstract and not has_q_in_body:
                cause = "title_abstract_signal_missing_from_chunk_index"
            else:
                cause = "query_doc_semantic_mismatch"

            audit_rows.append({
                "dataset": ds, "sample_id": sid, "question": q[:200],
                "expected_doc_ids": edoc,
                "expected_source_files": "",
                "expected_sections": "", "expected_route": sample.get("expected_route", ""),
                "answer_mode": "", "plan_mode": "", "is_comparison": sample.get("expected_route") == "comparison",
                "expected_doc_exists_in_chunks": exists,
                "expected_doc_chunk_count": len(chunks),
                "expected_doc_has_title": bool(t_text),
                "expected_doc_has_abstract": bool(a_text),
                "expected_doc_has_expected_section": True,
                "expected_doc_contains_query_terms": has_q_in_title or has_q_in_abstract or has_q_in_body,
                "expected_doc_contains_answer_terms_if_available": "",
                "top_dense_wrong_doc_ids": "", "top_bm25_wrong_doc_ids": "",
                "top_wrong_docs_common_pattern": "",
                "primary_hard_recall_cause": cause,
                "recommended_next_action": (
                    "dataset_metadata_fix" if not exists
                    else "title_abstract_doc_recall_audit" if "title_abstract" in cause
                    else "controlled_alias_expansion_audit" if "lexical" in cause
                    else "dense_recall_audit"
                ),
            })

    # ── Failure grouping ───────────────────────────────────────────
    cause_dist = Counter(r["primary_hard_recall_cause"] for r in audit_rows)
    group_rows = []
    group_map = {
        "expected_doc_missing_from_index": ("metadata_or_dataset_issue", "Expected doc not in chunks index"),
        "query_doc_lexical_mismatch": ("controlled_alias_gap", "No query terms found in doc title/abstract/body"),
        "title_abstract_signal_missing_from_chunk_index": ("title_abstract_doc_signal_missing", "Query terms in title/abstract but not in body chunks"),
        "query_doc_semantic_mismatch": ("dense_semantic_gap", "Query-doc semantic mismatch despite some overlap"),
    }
    for cause, cnt in cause_dist.most_common():
        group_name, desc = group_map.get(cause, ("unclear_or_manual_review", cause))
        samples = [r["sample_id"] for r in audit_rows if r["primary_hard_recall_cause"] == cause]
        group_rows.append({
            "failure_group": group_name,
            "sample_count": cnt, "sample_ids": "|".join(samples[:8]),
            "datasets": "smoke100,smoke50", "representative_questions": "",
            "common_features": desc,
            "likely_root_cause": cause,
            "proposed_fix_direction": (
                "dataset_metadata_fix" if "dataset" in group_name
                else "controlled_alias_expansion_design" if "alias" in group_name
                else "title_abstract_doc_recall_design" if "title_abstract" in group_name
                else "dense_recall_model_or_query_audit"
            ),
            "why_this_is_not_sample_patch": "Causal pattern aggregated across samples, not per-sample special case",
            "expected_impact": f"Could reduce hard_recall by {cnt}",
            "risk": "Medium" if "alias" in cause else "Low",
            "should_fix_next": cause == cause_dist.most_common(1)[0][0],
        })

    # ── Decision ───────────────────────────────────────────────────
    dominant = cause_dist.most_common(1)[0] if cause_dist else ("unknown", 0)
    d_cause = dominant[0]
    d_cnt = dominant[1]

    next_phase_map = {
        "expected_doc_missing_from_index": "dataset_expected_metadata_fix",
        "query_doc_lexical_mismatch": "controlled_alias_expansion_design",
        "title_abstract_signal_missing_from_chunk_index": "title_abstract_doc_recall_design",
        "query_doc_semantic_mismatch": "dense_recall_model_or_query_audit",
    }

    decision = {
        "primary_hard_recall_group": d_cause,
        "affected_sample_count": d_cnt,
        "recommended_phase18c": next_phase_map.get(d_cause, "move_to_support_selection_miss_audit"),
        "rationale": f"Dominant hard_recall cause: {d_cause} ({d_cnt}/{len(audit_rows)}). Cause distribution: {dict(cause_dist)}",
        "why_not_source_floor_tuning": "Source-floor already converged in Phase 17F. Hard recall is pre-hybrid issue.",
        "why_not_support_selection_yet": f"Hard recall ({d_cnt}) > support selection (6). Fix retrieval first.",
        "focused_sample_ids_for_phase18c": [r["sample_id"] for r in audit_rows if r["primary_hard_recall_cause"] == d_cause],
        "proposed_fix_scope": f"Address {d_cause} pattern",
        "risks": "Depends on fix type",
        "success_criteria": f"Reduce hard_recall by ≥{d_cnt//2} without regression",
        "regression_validation_plan": "Focused samples + smoke50 sanity",
    }

    # ── Write outputs ──────────────────────────────────────────────
    overview = {
        "total_hard_recall_samples": len(hard_ids),
        "datasets_included": ["smoke100", "smoke50"],
        "smoke100_count": sum(1 for ds, _, _ in hard_ids if ds == "smoke100"),
        "smoke50_count": sum(1 for ds, _, _ in hard_ids if ds == "smoke50"),
        "expected_docs_total": sum(1 for r in chunk_presence_rows),
        "expected_docs_found_in_chunks_count": docs_found,
        "expected_docs_missing_from_chunks_count": docs_missing,
        "metadata_issue_count": docs_missing,
        "dataset_expected_issue_count": 0,
        "lexical_mismatch_count": cause_dist.get("query_doc_lexical_mismatch", 0),
        "semantic_mismatch_count": cause_dist.get("query_doc_semantic_mismatch", 0),
        "title_abstract_signal_only_count": cause_dist.get("title_abstract_signal_missing_from_chunk_index", 0),
        "table_figure_or_caption_issue_count": 0,
        "chunk_text_quality_issue_count": 0,
        "primary_failure_group_distribution": dict(cause_dist),
        "recommended_next_phase": decision["recommended_phase18c"],
    }
    w_json(OUT_DIR / "hard_recall11_overview.json", overview)

    ADF = list(audit_rows[0].keys()) if audit_rows else []
    w_csv(OUT_DIR / "hard_recall11_query_doc_audit.csv", ADF, audit_rows)

    CPF = list(chunk_presence_rows[0].keys()) if chunk_presence_rows else []
    w_csv(OUT_DIR / "expected_doc_chunk_presence.csv", CPF, chunk_presence_rows)

    TOF = list(term_overlap_rows[0].keys()) if term_overlap_rows else []
    w_csv(OUT_DIR / "query_doc_term_overlap.csv", TOF, term_overlap_rows)

    TAF = list(title_abs_rows[0].keys()) if title_abs_rows else []
    w_csv(OUT_DIR / "title_abstract_signal_audit.csv", TAF, title_abs_rows)

    ALF = ["dataset", "sample_id", "question", "expected_doc_id",
           "question_terms", "expected_doc_terms", "suspected_alias_pairs",
           "alias_gap_type", "would_controlled_alias_expansion_help",
           "risk_of_alias_expansion", "notes"]
    w_csv(OUT_DIR / "synonym_alias_gap_audit.csv", ALF, alias_rows)

    w_csv(OUT_DIR / "table_figure_chunk_quality_audit.csv",
          ["dataset", "sample_id", "question", "expected_doc_id",
           "expected_sections", "question_mentions_table_or_figure",
           "expected_answer_likely_in_table", "expected_answer_likely_in_figure",
           "table_chunks_present", "figure_chunks_present",
           "captions_present", "table_or_figure_text_quality",
           "would_table_figure_processing_help", "notes"], [])

    GFF = ["failure_group", "sample_count", "sample_ids", "datasets",
           "representative_questions", "common_features", "likely_root_cause",
           "proposed_fix_direction", "why_this_is_not_sample_patch",
           "expected_impact", "risk", "should_fix_next"]
    w_csv(OUT_DIR / "hard_recall_failure_grouping.csv", GFF, group_rows)

    w_json(OUT_DIR / "phase18b_next_step_decision.json", decision)

    print(f"Phase 18B Complete:")
    print(f"  Hard recall: {len(hard_ids)} samples, {sum(1 for r in chunk_presence_rows)} expected docs")
    print(f"  Docs in chunks: {docs_found}, missing: {docs_missing}")
    print(f"  Cause distribution: {dict(cause_dist)}")
    print(f"  Dominant: {d_cause} ({d_cnt})")
    print(f"  Phase 18C: {decision['recommended_phase18c']}")


if __name__ == "__main__":
    main()
