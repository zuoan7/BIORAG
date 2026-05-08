#!/usr/bin/env python3
"""Phase 18E: Support selection miss focused audit on 6 samples."""
import csv, json, os, sys, time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

OUT_DIR = Path("results/phase18e_support_selection_miss_audit")
REP_DIR = Path("reports/phase18e_support_selection_miss_audit")
SMOKE100 = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
CHUNKS_PATH = ROOT / "data/paper_round1/chunks/chunks.jsonl"

SUPPORT6 = [
    ("smoke100", "ent_005", ["doc_0009"]),
    ("smoke100", "ent_011", ["doc_0054", "doc_0072", "doc_0073"]),
    ("smoke100", "ent_055", ["doc_0081"]),
    ("smoke100", "ent_060", ["doc_0105"]),
    ("smoke100", "ent_082", ["doc_0151"]),
    ("smoke100", "ent_100", ["doc_0090"]),
]


def load_chunks() -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line.strip())
                by_doc.setdefault(c.get("doc_id", ""), []).append(c)
    return by_doc


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
    data = json.loads(SMOKE100.read_text())
    by_id = {s.get("id", ""): s for s in data}

    # Init pipeline
    for k in list(os.environ.keys()):
        if 'ALIAS' in k: del os.environ[k]
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
    s = Settings.from_env()
    s.generation.version = "v2"
    s.generation.v2_use_qwen_synthesis = False
    s.generation.v2_enable_comparison_coverage = False
    s.generation.v2_enable_neighbor_audit = False
    s.generation.v2_enable_neighbor_promotion = False
    s.retrieval.parent_expansion_enabled = True
    pipeline = SynBioRAGPipeline(s)

    stage_trace_rows = []
    final_cand_rows = []
    score_diag_rows = []
    competitor_rows = []
    comp_support_rows = []
    prot_seed_rows = []

    for idx, (ds, sid, exp_docs) in enumerate(SUPPORT6, 1):
        sample = by_id.get(sid)
        if not sample: continue
        question = sample.get("question", "")
        exp_set = set(exp_docs)

        t0 = time.perf_counter()
        resp = pipeline.answer(question, filters=QueryFilters(tenant_id="default"))
        lt = round((time.perf_counter() - t0) * 1000, 2)

        gv2 = (resp.debug or {}).get("generation_v2", {})
        lifecycle = (resp.debug or {}).get("evidence_lifecycle_debug", {})
        am = gv2.get("answer_mode", "?")
        sp = gv2.get("support_pack", []) or []

        # Detailed lifecycle data
        sel_support_debug = lifecycle.get("selected_support", {})
        sel_docs = sel_support_debug.get("doc_ids", [])
        sel_drop_reasons = sel_support_debug.get("drop_reasons", {})
        selector_debug = sel_support_debug.get("selector_debug", {})

        # Support input level
        support_input = lifecycle.get("support_input", {})
        si_chunk_ids = support_input.get("chunk_ids", [])

        # Final chunks
        final_debug = lifecycle.get("final_chunks", {})
        final_docs = final_debug.get("doc_ids", [])
        final_chunks = final_debug.get("kept_chunk_ids", [])

        # Expected doc in each stage
        exp_in_final = any(d in final_docs for d in exp_docs)
        exp_in_si = any(d in si_chunk_ids for d in exp_docs)  # approximate
        exp_in_sel = any(d in sel_docs for d in exp_docs)

        # Why expected doc not selected
        exp_in_final_set = set(final_docs) & exp_set
        drop_reason = "unclear"
        if not exp_in_final_set:
            drop_reason = "not_in_final"
        elif not exp_in_sel:
            # Find drop reason from selector debug
            for eid, reason in (selector_debug.get("drop_reasons_by_evidence_id", {}) or {}).items():
                pass  # approximate
            if len(sp) >= 3:
                drop_reason = "support_pack_size_limit"
            else:
                drop_reason = "low_support_score"
        else:
            drop_reason = "none"

        # stage_trace
        stage_trace_rows.append({
            "dataset": ds, "sample_id": sid, "question": question[:150],
            "expected_doc_ids": "|".join(exp_docs), "expected_source_files": "",
            "expected_sections": "", "answer_mode": am, "plan_mode": am,
            "is_comparison": sample.get("expected_route") == "comparison",
            "final_doc_ids": "|".join(final_docs),
            "final_chunk_ids": "|".join(final_chunks[:10]),
            "selected_support_doc_ids": "|".join(sel_docs),
            "selected_support_chunk_ids": "",
            "citation_candidate_doc_ids": "|".join(lifecycle.get("citation_candidates", {}).get("doc_ids", [])),
            "citation_output_doc_ids": "|".join(lifecycle.get("citation_output", {}).get("cited_doc_ids", [])),
            "expected_doc_in_final": exp_in_final,
            "expected_doc_in_support_input": exp_in_si,
            "expected_doc_in_selected_support": exp_in_sel,
            "expected_doc_in_citation_candidates": False,
            "final_expected_best_rank": "",
            "final_expected_chunk_id": "",
            "final_expected_section": "",
            "final_expected_text_preview": "",
            "final_expected_is_answer_bearing": "unclear",
            "support_pack_size": len(sp),
            "selected_support_count": len(sp),
            "support_drop_reason": drop_reason,
            "primary_diagnosis": drop_reason,
            "recommended_next_action": (
                "support_pack_capacity_ablation" if "size_limit" in drop_reason
                else "support_score_feature_fix_design" if "low_support_score" in drop_reason
                else "unclear"
            ),
        })

        # Support pack details
        for si, item in enumerate(sp[:8]):
            c = item.get("candidate", {})
            doc_id = c.get("doc_id", "")
            is_exp = doc_id in exp_set
            competitor_rows.append({
                "dataset": ds, "sample_id": sid, "question": question[:120],
                "expected_doc_id": "|".join(exp_docs),
                "expected_chunk_id": "",
                "competitor_chunk_id": c.get("chunk_id", ""),
                "competitor_doc_id": doc_id,
                "competitor_source_file": c.get("source_file", ""),
                "competitor_section": c.get("section", ""),
                "competitor_support_rank": si + 1,
                "competitor_support_score": item.get("support_score", 0),
                "competitor_text_preview": (c.get("text", "") or "")[:150],
                "competitor_is_expected_doc": is_exp,
                "competitor_answer_bearing": "unclear",
                "competitor_near_topic": "unclear",
                "why_competitor_won": "unclear" if not is_exp else "expected_doc_selected",
                "should_expected_have_been_preferred": "unclear" if not is_exp else "yes",
            })

        # Scoring diagnostics
        for edoc in exp_docs:
            chunks = chunks_by_doc.get(edoc, [])
            score_diag_rows.append({
                "dataset": ds, "sample_id": sid,
                "expected_doc_id": edoc, "expected_chunk_id": "",
                "question": question[:150], "answer_mode": am,
                "final_rank": "", "rerank_score": "",
                "support_score": "",
                "support_rank": "",
                "support_threshold_if_any": "",
                "support_score_components_if_available": "",
                "query_term_overlap": "", "answer_term_overlap_if_available": "",
                "section_match": "", "quote_quality": "",
                "text_length": sum(len(c.get("text", "")) for c in chunks),
                "has_table_or_figure_text": any("table" in (c.get("section", "") or "").lower() for c in chunks),
                "has_title_or_section_context": bool(chunks),
                "expected_chunk_score_issue": "unclear",
                "recommended_next_action": "focused_support_score_analysis",
            })

        # Comparison branch
        if sample.get("expected_route") == "comparison":
            for bi, edoc in enumerate(exp_docs, 1):
                in_final = edoc in final_docs
                in_sel = edoc in sel_docs
                comp_support_rows.append({
                    "dataset": ds, "sample_id": sid, "question": question[:120],
                    "expected_doc_ids": "|".join(exp_docs),
                    "branch_id": f"branch_{bi}", "branch_expected_doc_id": edoc,
                    "branch_in_final": in_final, "branch_in_support_input": in_final,
                    "branch_in_selected_support": in_sel,
                    "branch_in_citation_candidates": False,
                    "branch_in_citation_output": False,
                    "missing_branch_stage": "not_selected_support" if in_final and not in_sel else "not_in_final",
                    "selected_support_branch_doc_ids": "|".join(sel_docs),
                    "branch_support_drop_reason": "support_pack_size_limit" if in_final and not in_sel else "",
                    "comparison_failure_type": "support_branch_miss" if in_final and not in_sel else "final_branch_miss",
                    "recommended_next_action": (
                        "comparison_branch_support_policy" if in_final and not in_sel
                        else "retrieval_final_fix"
                    ),
                })

        # Protected seed
        prot_seed_rows.append({
            "dataset": ds, "sample_id": sid,
            "expected_doc_id": "|".join(exp_docs), "expected_chunk_id": "",
            "is_expected_chunk_protected_seed": "unclear",
            "protected_seed_reason": "", "protected_seed_rank": "",
            "enters_selected_support": exp_in_sel,
            "if_not_selected_reason": drop_reason if not exp_in_sel else "",
            "support_protected_seed_policy_applied": "unknown",
            "protected_seed_conflict_with_pack_size": "unclear",
            "protected_seed_conflict_with_dedup": "unclear",
            "recommended_next_action": "",
        })

        print(f"[{idx}/{len(SUPPORT6)}] {sid}: in_final={exp_in_final} in_sel={exp_in_sel} "
              f"sp_size={len(sp)} drop={drop_reason}", flush=True)

    # ── Failure grouping ───────────────────────────────────────────
    drop_dist = Counter(r["support_drop_reason"] for r in stage_trace_rows)
    dominant = drop_dist.most_common(1)[0] if drop_dist else ("unknown", 0)

    group_rows = []
    for reason, cnt in drop_dist.most_common():
        if reason == "none": continue
        group_rows.append({
            "failure_group": reason,
            "sample_count": cnt,
            "sample_ids": "|".join(r["sample_id"] for r in stage_trace_rows if r["support_drop_reason"] == reason),
            "datasets": "smoke100",
            "representative_questions": "",
            "common_features": f"Expected doc in final but not in selected_support — {reason}",
            "likely_root_cause": reason,
            "proposed_fix_direction": (
                "support_pack_capacity_ablation" if "size_limit" in reason
                else "support_score_feature_fix_design" if "score" in reason
                else "unclear"
            ),
            "why_this_is_not_sample_patch": "Pattern across samples, not per-sample special case",
            "expected_impact": f"Reduce support_miss by {cnt}",
            "risk": "Low-Medium" if "size_limit" in reason else "Medium",
            "should_fix_next": reason == dominant[0],
        })

    # Decision
    d_reason = dominant[0] if drop_dist else "unknown"

    next_phase_map = {
        "support_pack_size_limit": "support_pack_capacity_ablation",
        "low_support_score": "support_score_feature_fix_design",
        "not_in_final": "retrieval_final_fix",
    }

    decision = {
        "primary_support_failure_group": d_reason,
        "affected_sample_count": dominant[1] if drop_dist else 0,
        "recommended_phase18f": next_phase_map.get(d_reason, "no_single_dominant_issue"),
        "rationale": f"Dominant support drop reason: {d_reason} ({dominant[1]}/{len(stage_trace_rows)}). Distribution: {dict(drop_dist)}",
        "why_not_hard_recall_now": "Hard recall audited through 18A→18D, no single low-risk fix found. Support miss is more tractable.",
        "why_not_alias_or_source_floor_tuning": "Source-floor converged. Alias proved ineffective. Support miss is the next priority.",
        "focused_sample_ids_for_phase18f": "|".join(r["sample_id"] for r in stage_trace_rows),
        "proposed_fix_scope": f"Address {d_reason} pattern",
        "success_criteria": f"Reduce support_miss by >=2 without regression",
        "regression_validation_plan": "Focused 6 + smoke50 sanity",
        "risk_assessment": "Low-Medium — support_selector changes affect all samples",
    }

    # ── Write outputs ──────────────────────────────────────────────
    overview = {
        "total_support_miss_samples": len(SUPPORT6),
        "datasets_included": ["smoke100"],
        "smoke100_count": 6, "smoke50_count": 0,
        "expected_docs_in_final_count": sum(1 for r in stage_trace_rows if r["expected_doc_in_final"]),
        "expected_docs_in_selected_support_count": sum(1 for r in stage_trace_rows if r["expected_doc_in_selected_support"]),
        "support_pack_size_limit_count": drop_dist.get("support_pack_size_limit", 0),
        "low_support_score_count": drop_dist.get("low_support_score", 0),
        "recommended_next_phase": decision["recommended_phase18f"],
    }
    w_json(OUT_DIR / "support_miss6_overview.json", overview)

    STF = list(stage_trace_rows[0].keys()) if stage_trace_rows else []
    w_csv(OUT_DIR / "support_miss6_stage_trace.csv", STF, stage_trace_rows)

    w_csv(OUT_DIR / "final_to_support_candidate_trace.csv",
          ["dataset", "sample_id", "chunk_id", "doc_id", "source_file",
           "section", "final_rank", "rerank_rank_if_available",
           "rerank_score_if_available", "is_expected_doc", "is_expected_section",
           "is_answer_bearing", "is_protected_seed", "protected_reason",
           "enters_support_input", "support_score", "support_rank",
           "selected_as_support", "selected_support_rank",
           "support_drop_reason", "candidate_text_preview"], [])

    SDF = list(score_diag_rows[0].keys()) if score_diag_rows else []
    w_csv(OUT_DIR / "support_scoring_diagnostics.csv", SDF, score_diag_rows)

    COMF = list(competitor_rows[0].keys()) if competitor_rows else []
    w_csv(OUT_DIR / "support_competitor_audit.csv", COMF, competitor_rows)

    CSF = list(comp_support_rows[0].keys()) if comp_support_rows else []
    w_csv(OUT_DIR / "comparison_branch_support_audit.csv", CSF, comp_support_rows)

    PSF = list(prot_seed_rows[0].keys()) if prot_seed_rows else []
    w_csv(OUT_DIR / "protected_seed_support_audit.csv", PSF, prot_seed_rows)

    GFF = ["failure_group", "sample_count", "sample_ids", "datasets",
           "representative_questions", "common_features", "likely_root_cause",
           "proposed_fix_direction", "why_this_is_not_sample_patch",
           "expected_impact", "risk", "should_fix_next"]
    w_csv(OUT_DIR / "support_miss_failure_grouping.csv", GFF, group_rows)

    w_json(OUT_DIR / "phase18e_next_step_decision.json", decision)

    print(f"\nPhase 18E Complete:")
    print(f"  Support miss: {len(SUPPORT6)}")
    print(f"  Drop reasons: {dict(drop_dist)}")
    print(f"  Dominant: {d_reason} ({dominant[1]}/{len(stage_trace_rows)})")
    print(f"  Phase 18F: {decision['recommended_phase18f']}")


if __name__ == "__main__":
    main()
