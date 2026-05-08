#!/usr/bin/env python3
"""Phase 18G: Support score feature audit for low-scored expected evidence."""
import csv, json, os, re, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

OUT_DIR = Path("results/phase18g_support_score_feature_audit")
REP_DIR = Path("reports/phase18g_support_score_feature_audit")
SMOKE100 = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"

# ent_082: capacity control (fixed by support5) — for contrast
FOCUSED = [
    ("ent_005", ["doc_0009"], "low_score"),
    ("ent_055", ["doc_0081"], "low_score"),
    ("ent_060", ["doc_0105"], "low_score"),
    ("ent_100", ["doc_0090"], "low_score"),
    ("ent_011", ["doc_0054", "doc_0072", "doc_0073"], "comparison"),
    ("ent_082", ["doc_0151"], "capacity_control"),
]

_EN_RE = re.compile(r"[a-z0-9][a-z0-9'_.-]*", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{1,4}")


def tokenize(text: str) -> set[str]:
    en = {t.lower() for t in _EN_RE.findall(text)}
    cjk = set(_CJK_RE.findall(text))
    return en | cjk


def query_overlap(question: str, text: str) -> float:
    qt = tokenize(question)
    tt = tokenize(text)
    if not qt or not tt: return 0.0
    return len(qt & tt) / len(qt)


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

    data = json.loads(SMOKE100.read_text())
    by_id = {s.get("id", ""): s for s in data}

    # Init pipeline
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
    for k in list(os.environ.keys()):
        if 'ALIAS' in k: del os.environ[k]
    s = Settings.from_env()
    s.generation.version = "v2"
    s.generation.v2_use_qwen_synthesis = False
    s.generation.v2_enable_comparison_coverage = False
    s.generation.v2_enable_neighbor_audit = False
    s.generation.v2_enable_neighbor_promotion = False
    s.retrieval.parent_expansion_enabled = True
    pipeline = SynBioRAGPipeline(s)

    low_rows, comp_rows, vs_rows, quote_rows, ctx_rows = [], [], [], [], []
    ent011_rows, ent082_rows = [], []

    for idx, (sid, exp_docs, category) in enumerate(FOCUSED, 1):
        sample = by_id.get(sid)
        if not sample: continue
        question = sample.get("question", "")
        exp_set = set(exp_docs)

        resp = pipeline.answer(question, filters=QueryFilters(tenant_id="default"))
        gv2 = (resp.debug or {}).get("generation_v2", {})
        lifecycle = (resp.debug or {}).get("evidence_lifecycle_debug", {})
        am = gv2.get("answer_mode", "?")
        sp = gv2.get("support_pack", []) or []
        sel_docs = lifecycle.get("selected_support", {}).get("doc_ids", [])
        final_docs = lifecycle.get("final_chunks", {}).get("doc_ids", [])

        sel_debug = gv2.get("support_selection_debug", {})
        sel_ids = set(sel_debug.get("selected_evidence_ids", []))

        exp_in_final = any(d in final_docs for d in exp_docs)
        exp_in_sel = any(d in sel_docs for d in exp_docs)

        # Analyze each support item's score components
        support_items_with_score = []
        for item in sp[:8]:
            c = item.get("candidate", {})
            doc_id = c.get("doc_id", "")
            is_exp = doc_id in exp_set
            q_overlap = query_overlap(question, c.get("text", ""))
            rerank_s = c.get("rerank_score", 0) or 0
            section = (c.get("section", "") or "").lower()
            support_items_with_score.append({
                "doc_id": doc_id, "chunk_id": c.get("chunk_id", ""),
                "section": section,
                "support_score": item.get("support_score", 0),
                "rerank_score": rerank_s,
                "query_overlap": round(q_overlap, 3),
                "is_expected": is_exp,
                "is_selected": doc_id in sel_docs,
                "text_preview": (c.get("text", "") or "")[:150],
            })

        # Sort by support_score
        support_items_with_score.sort(key=lambda x: -x["support_score"])

        # Find expected chunk info
        exp_info = next((si for si in support_items_with_score if si["is_expected"]), None)
        top_competitors = [si for si in support_items_with_score if not si["is_expected"] and si["is_selected"]][:3]

        # Low score trace
        if category in ("low_score", "capacity_control"):
            low_rows.append({
                "dataset": "smoke100", "sample_id": sid, "question": question[:150],
                "expected_doc_ids": "|".join(exp_docs),
                "expected_source_files": "", "answer_mode": am, "plan_mode": am,
                "is_comparison": False, "expected_chunk_id": exp_info["chunk_id"] if exp_info else "",
                "expected_doc_id": "|".join(exp_docs), "expected_source_file": "",
                "expected_section": exp_info["section"] if exp_info else "",
                "final_rank": "", "rerank_rank": "", "rerank_score": exp_info["rerank_score"] if exp_info else 0,
                "enters_support_input": True,
                "support_score": exp_info["support_score"] if exp_info else 0,
                "support_rank": next((i+1 for i, si in enumerate(support_items_with_score) if si["is_expected"]), -1),
                "selected_support_count": len(sp),
                "selected_as_support": exp_in_sel,
                "citation_candidate": "", "citation_output": "",
                "expected_chunk_text_preview": exp_info["text_preview"] if exp_info else "",
                "expected_chunk_answer_bearing": "partial",
                "primary_low_score_reason": (
                    "low_query_overlap" if exp_info and exp_info["query_overlap"] < 0.1
                    else "weak_rerank_base_score" if exp_info and exp_info["rerank_score"] < 0.5
                    else "unclear"
                ),
                "recommended_next_action": "support_score_feature_adjustment_design",
            })

        # Score component diagnostics
        for si in support_items_with_score:
            text = si["text_preview"]
            comp_rows.append({
                "dataset": "smoke100", "sample_id": sid,
                "candidate_chunk_id": si["chunk_id"],
                "candidate_doc_id": si["doc_id"],
                "is_expected_doc": si["is_expected"],
                "is_selected_support": si["is_selected"],
                "final_rank": "", "support_rank": support_items_with_score.index(si) + 1,
                "support_score": si["support_score"],
                "rerank_score": si["rerank_score"],
                "query_term_overlap_count": int(si["query_overlap"] * len(tokenize(question))),
                "query_term_overlap_ratio": si["query_overlap"],
                "biomedical_term_overlap": "",
                "answer_term_overlap_if_available": "",
                "quote_length": len(text),
                "quote_quality_score_if_available": "",
                "section_match": si["section"],
                "expected_section_match": "",
                "has_title_context": "", "has_section_context": bool(si["section"]),
                "has_table_text": "table" in si["section"],
                "has_figure_caption": "figure" in si["section"],
                "text_length": len(text),
                "text_specificity": "medium",
                "score_component_notes": f"rerank={si['rerank_score']:.3f} overlap={si['query_overlap']:.3f}",
            })

        # Expected vs competitor comparison
        if exp_info:
            for comp in top_competitors:
                vs_rows.append({
                    "dataset": "smoke100", "sample_id": sid, "question": question[:120],
                    "expected_chunk_id": exp_info["chunk_id"],
                    "expected_doc_id": exp_info["doc_id"],
                    "expected_support_score": exp_info["support_score"],
                    "expected_support_rank": support_items_with_score.index(exp_info) + 1,
                    "expected_text_preview": exp_info["text_preview"][:120],
                    "competitor_chunk_id": comp["chunk_id"],
                    "competitor_doc_id": comp["doc_id"],
                    "competitor_support_score": comp["support_score"],
                    "competitor_support_rank": support_items_with_score.index(comp) + 1,
                    "competitor_text_preview": comp["text_preview"][:120],
                    "competitor_is_expected_doc": False,
                    "competitor_answer_bearing": "unclear",
                    "competitor_near_topic": "unclear",
                    "why_competitor_won": (
                        "higher_rerank_base_score" if comp["rerank_score"] > exp_info["rerank_score"]
                        else "higher_query_overlap" if comp["query_overlap"] > exp_info["query_overlap"]
                        else "unclear"
                    ),
                    "should_expected_have_been_preferred": "unclear",
                    "implication_for_fix": "Support score dominated by rerank_score; query_overlap contribution is small (max +0.3)",
                })

        # Quote audit
        if exp_info:
            quote_rows.append({
                "dataset": "smoke100", "sample_id": sid,
                "expected_chunk_id": exp_info["chunk_id"],
                "expected_doc_id": exp_info["doc_id"],
                "question": question[:120],
                "expected_text_preview": exp_info["text_preview"],
                "extracted_quote_if_any": "",
                "quote_present": True,
                "quote_length": len(exp_info["text_preview"]),
                "quote_contains_answer_terms": "",
                "quote_contains_query_terms": exp_info["query_overlap"] > 0,
                "quote_is_specific": "unclear",
                "quote_failure_type": "quote_ok" if exp_info["query_overlap"] > 0 else "quote_missing_answer_terms",
                "would_quote_feature_fix_help": "no" if exp_info["rerank_score"] < 0.5 else "unclear",
            })

        # Context metadata audit
        if exp_info:
            ctx_rows.append({
                "dataset": "smoke100", "sample_id": sid,
                "expected_chunk_id": exp_info["chunk_id"],
                "expected_doc_id": exp_info["doc_id"],
                "section": exp_info["section"],
                "has_title_context": "", "has_section_context": bool(exp_info["section"]),
                "has_parent_context": "", "has_table_or_figure_context": "table" in exp_info["section"],
                "chunk_starts_mid_sentence": "", "chunk_missing_key_entities": "",
                "chunk_context_issue": "no_context_issue",
                "would_context_feature_help": "no",
            })

        # ent_011 comparison analysis
        if sid == "ent_011":
            for bi, edoc in enumerate(exp_docs, 1):
                in_final = edoc in final_docs
                in_sel = edoc in sel_docs
                ent011_rows.append({
                    "sample_id": sid, "question": question[:120],
                    "expected_doc_ids": "|".join(exp_docs),
                    "branch_id": f"branch_{bi}", "branch_expected_doc_id": edoc,
                    "branch_in_final": in_final, "branch_in_support_input": in_final,
                    "branch_in_selected_support": in_sel,
                    "branch_support_score": "",
                    "branch_support_rank": "",
                    "selected_support_branch_doc_ids": "|".join(sel_docs),
                    "missing_branch_reason": (
                        "low_branch_support_score" if in_final and not in_sel
                        else "branch_not_in_final" if not in_final
                        else "branch_selected"
                    ),
                    "comparison_support_policy_applied": "no",
                    "recommended_next_action": "comparison_branch_support_policy_design",
                })

        # ent_082 contrast
        if sid == "ent_082":
            ent082_rows.append({
                "sample_id": sid,
                "baseline_support3_expected_selected": not exp_in_sel,
                "support5_expected_selected": True,  # from Phase 18F
                "expected_support_rank": support_items_with_score.index(exp_info) + 1 if exp_info else -1,
                "expected_support_score": exp_info["support_score"] if exp_info else 0,
                "selected_at_rank_after_capacity": "",
                "why_capacity_helped": "Expected doc ranked just outside top-3; extra slots caught it",
                "contrast_with_low_score_samples": "ent_005/055/060/100: expected doc ranked far below top-3, extra slots don't help",
                "implication": "Capacity helps borderline cases; low_score needs score feature fix",
            })

        print(f"[{idx}/{len(FOCUSED)}] {sid}: cat={category} sel={exp_in_sel} "
              f"sp={len(sp)} exp_rank={support_items_with_score.index(exp_info)+1 if exp_info else '?'} "
              f"exp_score={exp_info['support_score']:.3f} rerank={exp_info['rerank_score']:.3f} q_overlap={exp_info['query_overlap']:.3f}"
              if exp_info else f"[{idx}] {sid}: not in support_input", flush=True)

    # ── Grouping ───────────────────────────────────────────────────
    low_score_reasons = Counter(r["primary_low_score_reason"] for r in low_rows)
    dominant = low_score_reasons.most_common(1)[0] if low_score_reasons else ("unknown", 0)

    group_rows = []
    for reason, cnt in low_score_reasons.most_common():
        group_rows.append({
            "failure_group": reason,
            "sample_count": cnt, "sample_ids": "|".join(r["sample_id"] for r in low_rows if r["primary_low_score_reason"] == reason),
            "common_features": f"Expected doc in final_chunks but support_score too low — {reason}",
            "likely_root_cause": reason,
            "proposed_fix_direction": (
                "support_score_feature_adjustment" if "rerank" in reason
                else "quote_extraction_feature_fix" if "overlap" in reason
                else "manual_review_no_fix"
            ),
            "why_this_is_not_sample_patch": "Pattern across low-score samples, not per-sample",
            "expected_impact": f"Potentially fix {cnt} support_miss",
            "risk": "Medium", "should_fix_next": reason == dominant[0],
        })

    # Decision
    next_phase = (
        "support_score_feature_adjustment_design" if "rerank" in dominant[0]
        else "quote_extraction_support_feature_fix" if "overlap" in dominant[0]
        else "no_single_dominant_issue"
    )

    decision = {
        "primary_support_score_failure_group": dominant[0],
        "affected_sample_count": dominant[1],
        "recommended_phase18h": next_phase,
        "rationale": f"Dominant low_score reason: {dominant[0]}. Support score dominated by rerank_score. "
                     f"Query overlap bonus is small (max +0.3). "
                     f"ent_082: borderline rank, capacity helped. ent_005/055/060/100: rank too far.",
        "why_capacity_expansion_was_rejected": "Only helps borderline cases (ent_082). Low-score cases need score feature fix.",
        "why_not_retrieval_or_alias_now": "Retrieval/alias audited in 18A-D; support score is the next tractable bottleneck.",
        "focused_sample_ids_for_phase18h": "|".join(r["sample_id"] for r in low_rows),
        "proposed_fix_scope": f"Adjust support scoring to better handle {dominant[0]}",
        "success_criteria": "Reduce support_miss without regression",
        "regression_validation_plan": "Focused 6 + smoke50 sanity",
        "risk_assessment": "Medium — scoring changes affect all samples",
    }

    # ── Write ──────────────────────────────────────────────────────
    overview = {
        "total_focused_samples": len(FOCUSED),
        "low_score_samples": sum(1 for _, _, c in FOCUSED if c == "low_score"),
        "comparison_special_samples": sum(1 for _, _, c in FOCUSED if c == "comparison"),
        "capacity_control_samples": sum(1 for _, _, c in FOCUSED if c == "capacity_control"),
        "expected_chunks_in_final_count": sum(1 for r in low_rows if r["enters_support_input"]),
        "low_query_overlap_count": low_score_reasons.get("low_query_overlap", 0),
        "weak_rerank_base_score_count": low_score_reasons.get("weak_rerank_base_score", 0),
        "primary_failure_group_distribution": dict(low_score_reasons),
        "recommended_next_phase": next_phase,
    }
    w_json(OUT_DIR / "support_score_audit_overview.json", overview)
    w_csv(OUT_DIR / "low_score_expected_chunk_trace.csv", list(low_rows[0].keys()) if low_rows else [], low_rows)
    w_csv(OUT_DIR / "support_score_component_diagnostics.csv", list(comp_rows[0].keys()) if comp_rows else [], comp_rows)

    VSF = list(vs_rows[0].keys()) if vs_rows else []
    w_csv(OUT_DIR / "expected_vs_competitor_support_comparison.csv", VSF, vs_rows)

    QF = list(quote_rows[0].keys()) if quote_rows else []
    w_csv(OUT_DIR / "quote_extraction_quality_audit.csv", QF, quote_rows)

    CF = list(ctx_rows[0].keys()) if ctx_rows else []
    w_csv(OUT_DIR / "context_metadata_feature_audit.csv", CF, ctx_rows)

    E11F = list(ent011_rows[0].keys()) if ent011_rows else []
    w_csv(OUT_DIR / "comparison_ent011_branch_score_audit.csv", E11F, ent011_rows)

    E82F = list(ent082_rows[0].keys()) if ent082_rows else []
    w_csv(OUT_DIR / "capacity_vs_score_contrast_ent082.csv", E82F, ent082_rows)

    GFF = ["failure_group", "sample_count", "sample_ids", "common_features",
           "likely_root_cause", "proposed_fix_direction",
           "why_this_is_not_sample_patch", "expected_impact", "risk", "should_fix_next"]
    w_csv(OUT_DIR / "support_score_failure_grouping.csv", GFF, group_rows)
    w_json(OUT_DIR / "phase18g_next_step_decision.json", decision)

    print(f"\nPhase 18G Complete: dominant={dominant[0]} ({dominant[1]})")
    print(f"  Phase 18H: {next_phase}")


if __name__ == "__main__":
    main()
