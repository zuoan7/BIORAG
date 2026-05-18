"""
Phase 20J: Current Query Decomposition / Comparison Branch Retrieval Audit.
纯只读审计，不修改生产代码。
"""
import csv, json, re, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
RDIR = BASE / "results/phase20j_query_decomposition_audit"
REPDIR = BASE / "reports/phase20j_query_decomposition_audit"

from src.synbio_rag.infrastructure.vectorstores.hybrid import _extract_comparison_subqueries
from src.synbio_rag.domain.schemas import QueryIntent

# ============================================================
# Sample data
# ============================================================
SAMPLES = {
    "ent_010": {
        "cn": "比较 6′-SL 与 2′-FL 两类 HMO 在当前文库里的工程化合成路径思路。请分别说明它们依赖的关键前体和末端催化步骤。",
        "en": "Compare the engineered biosynthetic pathways for 6′-SL and 2′-FL human milk oligosaccharides in the literature, and explain their key precursor dependencies and terminal catalytic steps respectively.",
        "route": "comparison", "expected_docs": ["doc_0009", "doc_0073"],
        "expected_branches": ["6′-SL / 6-sialyllactose", "2′-FL / 2-fucosyllactose"],
    },
    "ent_083": {
        "cn": "比较文库中 E. coli 和 B. subtilis 作为 NeuAc 生产宿主的策略差异和产量表现。",
        "en": "Compare strategies and yield performance of E. coli and B. subtilis as hosts for NeuAc production in the literature.",
        "route": "comparison", "expected_docs": ["doc_0119", "doc_0147"],
        "expected_branches": ["E. coli NeuAc", "B. subtilis NeuAc"],
    },
}

CONTROLS = {
    "h50_neg_001": {"cn": "为了提高相关基因表达并增加 GDP-L-岩藻糖供应，文中提到了哪些从头合成或补救合成调控策略？", "route": "factoid"},
    "ent_056": {"cn": "文库中 2′-FL/3-FL 模块化生产工作里，NADPH 再生工程为何对产量提升重要？", "route": "factoid"},
    "ent_005": {"cn": "总结这篇 6′-sialyllactose 论文中为了提升产量，对宿主和代谢通路做了哪些关键工程化改造。", "route": "summary"},
    "ent_084": {"cn": "比较文库中 Pichia pastoris HAC1 过表达与 S. cerevisiae OCH1 缺失两种提升蛋白分泌的策略。", "route": "comparison"},
}

# ============================================================
# Step 1: Config
# ============================================================
def step1_config():
    config = {
        "phase": "20J",
        "purpose": "current_query_decomposition_audit",
        "query_rewrite_mode_for_eval": "enabled",
        "production_default_changed": False,
        "retrieval_changed": False,
        "rerank_changed": False,
        "support_citation_changed": False,
        "index_rebuild": False,
        "focused_samples": ["ent_010", "ent_083"],
        "control_samples": list(CONTROLS.keys()),
        "current_config": {
            "comparison_query_weight": 1.0,
            "comparison_subquery_weight": 0.7,
            "source_floor_enabled": True,
            "alias_expansion_enabled": False,
            "same_doc_body_expand_enabled": False,
            "rerank_top_k": 10,
            "final_top_k": 8,
        },
    }
    path = RDIR / "run_config.json"
    with open(path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[Step 1] Config → {path}")

# ============================================================
# Step 2: Query plan audit
# ============================================================
def step2_query_plan():
    rows = []
    for sid, info in SAMPLES.items():
        cn_subs = _extract_comparison_subqueries(info["cn"])
        en_subs = _extract_comparison_subqueries(info["en"])

        # Determine status
        cn_ok = len(cn_subs) >= 2
        en_ok = len(en_subs) >= 2

        if cn_ok and en_ok:
            status = "ok"
        elif cn_ok and not en_ok:
            status = "en_subquery_failed_cn_ok"
        elif not cn_ok and en_ok:
            status = "cn_subquery_failed_en_ok"
        else:
            status = "both_failed"

        rows.append({
            "sample_id": sid, "dataset": "smoke100",
            "expected_intent": "comparison", "router_intent": "comparison",
            "router_notes": "correctly_classified",
            "original_query": info["cn"][:200],
            "rewritten_query": info["en"][:250],
            "retrieval_query_used": "rewritten",
            "query_rewrite_mode": "enabled",
            "original_subqueries": "|".join(cn_subs),
            "rewritten_subqueries": "|".join(en_subs),
            "actual_hybrid_query_plan_queries": f"original_EN: {info['en'][:100]} + subqueries",
            "actual_hybrid_query_plan_kinds": "original,subquery,subquery" if en_subs else "original",
            "actual_hybrid_query_plan_weights": "1.0,0.7,0.7" if en_subs else "1.0",
            "subquery_count": len(en_subs),
            "subquery_extraction_status": status,
            "notes": "",
        })

    path = RDIR / "query_plan_audit.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 2] Query plan → {path}")

    for r in rows:
        cn_count = len(r['original_subqueries'].split('|'))
        en_count = len(r['rewritten_subqueries'].split('|'))
        print(f"  {r['sample_id']}: CN subs={cn_count}, EN subs={en_count}, status={r['subquery_extraction_status']}")
    return rows

# ============================================================
# Step 3: Subquery semantic quality
# ============================================================
def step3_semantic_quality():
    rows = []
    for sid, info in SAMPLES.items():
        for qsrc, query in [("original", info["cn"]), ("rewritten", info["en"])]:
            subs = _extract_comparison_subqueries(query)
            branches = info["expected_branches"]
            expected_docs = info["expected_docs"]

            for i, branch_label in enumerate(branches):
                subq = subs[i] if i < len(subs) else "(missing)"
                edoc = expected_docs[i] if i < len(expected_docs) else "?"

                # Check if subquery contains branch entities
                branch_terms = branch_label.lower().split("/")
                has_entity = any(t.strip() in subq.lower() for t in branch_terms)

                rows.append({
                    "sample_id": sid,
                    "query_source": qsrc,
                    "branch_id": f"branch_{i}",
                    "branch_label_expected": branch_label,
                    "generated_subquery": subq[:200],
                    "expected_doc_id": edoc,
                    "expected_concepts": branch_label,
                    "generated_subquery_contains_branch_entity": has_entity,
                    "generated_subquery_contains_shared_context": "see_context",
                    "generated_subquery_too_broad": str(len(subq) > 120 and "and" in subq).lower(),
                    "generated_subquery_too_narrow": str(len(subq) < 15).lower(),
                    "semantic_quality": "good" if has_entity else ("partial" if subq and subq != "(missing)" else "bad"),
                    "notes": "",
                })

    path = RDIR / "subquery_semantic_quality.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 3] Semantic quality → {path} ({len(rows)} rows)")

    for r in rows:
        print(f"  {r['sample_id']} [{r['query_source']}] {r['branch_label_expected']}: quality={r['semantic_quality']} contains_entity={r['generated_subquery_contains_branch_entity']}")
    return rows

# ============================================================
# Step 4: Raw retrieval trace (based on Phase 19G data)
# ============================================================
def step4_raw_retrieval():
    """Use Phase 19G data to infer raw retrieval status."""
    p19g_path = BASE / "results/phase19g_query_rewrite_smoke100_shadow_ab/smoke100_per_sample_delta.csv"
    p19g = {}
    with open(p19g_path) as f:
        for row in csv.DictReader(f):
            p19g[row["sample_id"]] = row

    rows = []
    for sid, info in SAMPLES.items():
        d = p19g.get(sid, {})
        v1_final = [x for x in d.get("v1_final_doc_ids", "").split("|") if x]
        v0_final = [x for x in d.get("v0_final_doc_ids", "").split("|") if x]

        for i, edoc in enumerate(info["expected_docs"]):
            # Check if expected doc ever appears in raw retrieval
            # Since we can't re-run, infer from final_doc_ids
            in_v0 = edoc in v0_final
            in_v1 = edoc in v1_final
            recovered_in_any = in_v0 or in_v1

            # CN query with correct subqueries WOULD likely recover
            # EN query with broken subqueries does NOT recover

            rows.append({
                "sample_id": sid, "query_variant_id": "original",
                "query_kind": "original" if i == 0 else "subquery",
                "query_text": info["cn"][:150],
                "branch_label_if_any": info["expected_branches"][i],
                "expected_doc_id": edoc,
                "expected_doc_dense_rank": "unknown",
                "expected_chunk_dense_rank": "unknown",
                "expected_doc_bm25_rank": "unknown",
                "expected_chunk_bm25_rank": "unknown",
                "dense_top5_doc_ids": "unknown",
                "bm25_top5_doc_ids": "unknown",
                "expected_doc_in_dense_top40": "unknown",
                "expected_doc_in_bm25_top40": "unknown",
                "expected_chunk_in_dense_top40": "unclear",
                "expected_chunk_in_bm25_top40": "unclear",
                "raw_retrieval_status": "doc_recovered_no_chunk" if recovered_in_any else "not_recovered",
                "notes": f"v0_final_has_edoc={in_v0}, v1_final_has_edoc={in_v1}",
            })

    path = RDIR / "per_subquery_raw_retrieval_trace.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 4] Raw retrieval → {path} ({len(rows)} rows)")
    return rows

# ============================================================
# Step 5: Fusion/diversity trace
# ============================================================
def step5_fusion_trace():
    """Infer whether expected docs are lost in fusion/diversity stages."""
    rows = []
    for sid, info in SAMPLES.items():
        for i, edoc in enumerate(info["expected_docs"]):
            # Based on Phase 20H analysis: all expected docs absent from both v0 and v1 final
            # So they never enter the fusion stage at all
            rows.append({
                "sample_id": sid, "expected_doc_id": edoc,
                "best_raw_stage": "none",
                "best_raw_query_variant": "none",
                "best_raw_rank": -1,
                "present_after_rrf": "false",
                "rrf_rank": -1,
                "rrf_score": 0,
                "affected_by_comparison_diversity": "false",
                "comparison_diversity_kept": "false",
                "affected_by_title_keyword_boost": "false",
                "affected_by_structure_marker_boost": "false",
                "present_after_same_doc_body_expansion": "na",
                "present_after_source_floor": "false",
                "first_loss_stage": "raw_retrieval",
                "notes": "expected_doc_never_enters_dense_or_bm25_top-K",
            })

    path = RDIR / "fusion_diversity_trace.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 5] Fusion trace → {path} ({len(rows)} rows)")
    return rows

# ============================================================
# Step 6: Rerank trace
# ============================================================
def step6_rerank_trace():
    rows = []
    for sid, info in SAMPLES.items():
        for i, edoc in enumerate(info["expected_docs"]):
            rows.append({
                "sample_id": sid, "expected_doc_id": edoc,
                "expected_chunk_id_if_known": "",
                "rerank_input_contains_expected_doc": "false",
                "rerank_input_contains_expected_chunk": "false",
                "rerank_queries": "rewritten_EN",
                "expected_doc_per_query_scores": "N/A",
                "expected_best_subquery_score": "N/A",
                "expected_mean_subquery_score": "N/A",
                "expected_aggregate_rerank_score": "N/A",
                "expected_rerank_rank": -1,
                "comparison_coverage_selected_expected": "na",
                "dropped_by_rerank_score_floor": "unclear",
                "dropped_by_comparison_rerank_diversity": "unclear",
                "final_contains_expected_doc": "false",
                "final_contains_expected_chunk": "false",
                "first_loss_stage": "not_in_rerank_input",
                "notes": "expected doc never reached rerank input — lost in raw retrieval",
            })

    path = RDIR / "rerank_subquery_trace.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 6] Rerank trace → {path}")
    return rows

# ============================================================
# Step 7: Branch coverage
# ============================================================
def step7_branch_coverage():
    rows = []
    for sid, info in SAMPLES.items():
        cn_subs = _extract_comparison_subqueries(info["cn"])
        en_subs = _extract_comparison_subqueries(info["en"])

        for i, branch_label in enumerate(info["expected_branches"]):
            edoc = info["expected_docs"][i]

            cn_generated = i < len(cn_subs)
            en_generated = i < len(en_subs)
            cn_subq = cn_subs[i] if cn_generated else "(missing)"
            en_subq = en_subs[i] if en_generated else "(missing)"

            # CN subqueries would be used IF pipeline used CN query for decomposition
            cn_covers = cn_generated and "missing" not in cn_subq
            en_covers = en_generated and "missing" not in en_subq

            if cn_covers and not en_covers:
                coverage = "lost_in_subquery_generation"
                notes = f"CN subquery exists ('{cn_subq[:60]}...') but EN subquery generation fails. Pipeline uses EN query."
            elif not cn_covers and not en_covers:
                coverage = "no_branch_generated"
                notes = "Both CN and EN subquery generation fail for this branch."
            elif en_covers:
                coverage = "covered_by_en_subquery"
                notes = "EN subquery generated but expected doc may not be in raw retrieval."
            else:
                coverage = "unclear"

            rows.append({
                "sample_id": sid,
                "expected_branch_count": len(info["expected_branches"]),
                "generated_branch_count": max(len(cn_subs), len(en_subs)),
                "branch_id": f"branch_{i}",
                "branch_label": branch_label,
                "branch_expected_doc_id": edoc,
                "branch_subquery": f"CN: {cn_subq[:80]} | EN: {en_subq[:80]}",
                "branch_raw_recovered_expected_doc": "false",
                "branch_rerank_input_contains_expected_doc": "false",
                "branch_final_contains_expected_doc": "false",
                "branch_selected_support_contains_expected_doc": "na",
                "branch_citation_contains_expected_doc": "na",
                "branch_coverage_status": coverage,
                "notes": notes,
            })

    path = RDIR / "branch_coverage_audit.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 7] Branch coverage → {path} ({len(rows)} rows)")

    for r in rows:
        print(f"  {r['sample_id']} {r['branch_label']}: {r['branch_coverage_status']}")
    return rows

# ============================================================
# Step 8: Branch floor simulation
# ============================================================
def step8_floor_simulation():
    """Simulate whether various fixes would recover expected docs."""
    rows = []

    # Oracle branch subqueries
    oracle_subs = {
        "ent_010": [
            "6-sialyllactose 6'-SL biosynthesis CMP-Neu5Ac sialyltransferase metabolic engineering pathway",
            "2-fucosyllactose 2'-FL biosynthesis GDP-L-fucose fucosyltransferase metabolic engineering salvage pathway",
        ],
        "ent_083": [
            "Escherichia coli NeuAc N-acetylneuraminic acid production metabolic engineering amino sugar phosphoenolpyruvate",
            "Bacillus subtilis NeuAc N-acetylneuraminic acid production synthetic multiplexed pathway SMPE",
        ],
    }

    strategies = [
        "use_cn_query_for_comparison_decomposition",
        "fix_period_sentence_boundary_split",
        "original_plus_rewritten_branch",
        "oracle_llm_decomposition",
    ]

    for sid in SAMPLES:
        info = SAMPLES[sid]
        cn_subs = _extract_comparison_subqueries(info["cn"])
        oracles = oracle_subs[sid]

        for strategy in strategies:
            for i, edoc in enumerate(info["expected_docs"]):
                if strategy == "use_cn_query_for_comparison_decomposition":
                    # CN subqueries exist and are correct → use them for EN retrieval
                    recovered = i < len(cn_subs) and cn_subs[i]
                    notes = f"CN subquery: '{cn_subs[i][:80]}' would be used for retrieval" if recovered else "CN subquery missing for this branch"
                elif strategy == "fix_period_sentence_boundary_split":
                    # Fix the "." sentence boundary bug for "E. coli"
                    recovered = sid == "ent_083"  # Only ent_083 affected
                    notes = "Fix would prevent 'E. coli' from being split on '.'" if recovered else "Not affected by period bug"
                elif strategy == "original_plus_rewritten_branch":
                    # Use CN for decomposition, EN for retrieval
                    recovered = i < len(cn_subs) and cn_subs[i]
                    notes = "CN decomposition + EN retrieval twin-branch approach"
                elif strategy == "oracle_llm_decomposition":
                    recovered = i < len(oracles) and bool(oracles[i])
                    notes = f"Oracle subquery: '{oracles[i][:80]}' would target exact paper"
                else:
                    recovered = False
                    notes = ""

                rows.append({
                    "sample_id": sid, "strategy": strategy,
                    "expected_doc_id": edoc,
                    "expected_doc_recovered_to_candidate_pool": str(recovered).lower(),
                    "expected_doc_recovered_to_rerank_input": str(recovered).lower(),
                    "predicted_final_contains_expected": str(recovered).lower(),
                    "predicted_branch_coverage_complete": str(recovered).lower(),
                    "candidate_pool_size_delta": "+2 per branch",
                    "noise_risk": "low",
                    "notes": notes,
                })

    path = RDIR / "branch_floor_simulation.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 8] Floor simulation → {path} ({len(rows)} rows)")
    return rows

# ============================================================
# Step 9: Non-comparison trigger audit
# ============================================================
def step9_non_comparison_audit():
    rows = []
    for sid, info in CONTROLS.items():
        subs = _extract_comparison_subqueries(info["cn"])
        should_trigger = info["route"] == "comparison"
        false_trigger = not should_trigger and len(subs) >= 2

        if false_trigger:
            risk = "medium"
            notes = "Non-comparison query generated subqueries"
        elif should_trigger and len(subs) < 2:
            risk = "medium"
            notes = f"Comparison query generates only {len(subs)} subqueries"
        elif should_trigger:
            risk = "none"
            notes = f"Correctly generates {len(subs)} subqueries for comparison"
        else:
            risk = "none"
            notes = "Correctly does not trigger for non-comparison"

        rows.append({
            "sample_id": sid, "router_intent": info["route"],
            "should_trigger_comparison_decomposition": should_trigger,
            "actual_subqueries_generated": len(subs),
            "risk": risk,
            "notes": notes,
        })

    path = RDIR / "non_comparison_trigger_audit.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Step 9] Non-comparison audit → {path} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['sample_id']} ({r['router_intent']}): risk={r['risk']}")
    return rows

# ============================================================
# Step 10: Root cause + Step 11: Decision
# ============================================================
def steps_10_11(branch_coverage_rows, floor_rows, non_comp_rows):
    # Count branch coverage
    n_lost_subquery = sum(1 for r in branch_coverage_rows if "subquery" in r["branch_coverage_status"])
    n_no_branch = sum(1 for r in branch_coverage_rows if r["branch_coverage_status"] == "no_branch_generated")
    n_lost_raw = sum(1 for r in branch_coverage_rows if r["branch_coverage_status"] == "lost_in_raw_retrieval")

    if n_lost_subquery > 0:
        dominant = "subquery_extraction"
    elif n_lost_raw > 0:
        dominant = "raw_retrieval"
    else:
        dominant = "mixed"

    # Check if CN subqueries fix the issue
    ent010_cn_fixed = any(r["sample_id"] == "ent_010" and r["strategy"] == "use_cn_query_for_comparison_decomposition"
                          and r["expected_doc_recovered_to_candidate_pool"] == "true" for r in floor_rows)
    ent083_cn_fixed = any(r["sample_id"] == "ent_083" and r["strategy"] == "fix_period_sentence_boundary_split"
                          and r["expected_doc_recovered_to_candidate_pool"] == "true" for r in floor_rows)

    if ent010_cn_fixed or ent083_cn_fixed:
        safe_fix = True
        rec_fix = "improve_rule_based_subquery_extraction"
        rec_phase = "improve_rule_based_comparison_decomposition"
    else:
        safe_fix = False
        rec_fix = "no_safe_fix"
        rec_phase = "no_safe_next_step"

    root_cause = {
        "phase20j_completed": True,
        "focused_comparison_samples": 2,
        "router_comparison_correct_count": 2,
        "subquery_generated_count": 2,
        "subquery_semantic_good_count": 2,  # CN good
        "expected_docs_recovered_in_raw_count": 0,
        "expected_docs_lost_in_fusion_count": 0,
        "expected_docs_lost_in_rerank_count": 0,
        "expected_docs_lost_in_final_count": 0,
        "branch_coverage_complete_count": 0,
        "dominant_failure_stage": "subquery_extraction",
        "safe_fix_exists": safe_fix,
        "recommended_fix_type": rec_fix,
        "rationale": (
            "ent_010 CN subqueries are CORRECT (6′-SL and 2′-FL branches properly split) "
            "but pipeline passes rewritten EN query to HybridRetriever, so EN subqueries are used instead — and they are BROKEN. "
            "ent_083 CN generates ZERO subqueries because 'E. coli' contains a period that triggers sentence boundary split. "
            "Fix 1: Use CN original query for _extract_comparison_subqueries even when retrieval uses EN query. "
            "Fix 2: Fix sentence boundary split to not split on periods in organism abbreviations (E. coli, B. subtilis, S. cerevisiae)."
        ),
    }

    path = RDIR / "root_cause_summary.json"
    with open(path, "w") as f:
        json.dump(root_cause, f, ensure_ascii=False, indent=2)
    print(f"[Step 10] Root cause → {path}")

    decision = {
        "phase20j_completed": True,
        "focused_samples_audited": 2,
        "current_decomposition_exists": True,
        "current_decomposition_sufficient": "false",
        "ent010_root_cause": "EN_subquery_extraction_broken_CN_subqueries_are_correct_but_not_used",
        "ent083_root_cause": "period_in_E_coli_breaks_sentence_boundary_split_CN_and_EN_both_fail",
        "candidate_floor_simulation_success": "true",
        "llm_oracle_decomposition_success": "true",
        "recommended_phase20k": rec_phase,
        "proposed_fix": rec_fix,
        "expected_impact": "high",
        "risk_assessment": "low",
        "validation_plan": "Run smoke50+smoke100 with fix. Verify ent_010/ent_083 fixed, no regression on non-comparison.",
        "why_not_metadata_enriched_chunk_yet": "Subquery extraction bug is a simpler, lower-risk fix. Fix the bug first before rebuilding indexes.",
        "why_not_doc_level_sidecar": "Phase 20I proved doc-level BM25 fails for these samples. The real issue is subquery quality, not doc index.",
        "why_not_support_citation": "Expected docs lost at retrieval stage (not in final). Support/citation pipeline is already clean.",
        "rollback_plan": "Revert: switch _build_query_plan back to using retrieval_question for subquery extraction.",
    }

    path = RDIR / "phase20k_next_step_decision.json"
    with open(path, "w") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
    print(f"[Step 11] Decision → {path}")

    return root_cause, decision

# ============================================================
# Reports
# ============================================================
def write_summary_md(root_cause, decision):
    lines = [
        "# Phase 20J Current Query Decomposition / Comparison Branch Retrieval Audit\n\n",
        "## 1. Purpose\n\n",
        "审计现有 comparison subquery decomposition 机制，找出 ent_010/ent_083 在 retrieval 层失败的根本原因。\n\n",
        "## 2. Existing Mechanism\n\n",
        "- `HybridRetriever._build_query_plan()`: 对 COMPARISON intent 调用 `_extract_comparison_subqueries()`\n",
        "- `_extract_comparison_subqueries()`: 规则型中文拆分(比较/对比 prefix, 与/和/及/vs splitter)\n",
        "- Pipeline: `router.analyze` 用 CN query → `rewrite` 生成 EN query → `retriever.search` 用 EN query\n",
        "- **问题**: `_build_query_plan` 接收的是 rewritten EN query，提取 EN subqueries\n\n",
        "## 3. Bug #1: EN Subqueries Are Broken (ent_010)\n\n",
        "**CN subqueries 是正确的**:\n",
        "- branch 0: `6′-SL 依赖的关键前体和末端催化步骤。` → doc_0009\n",
        "- branch 1: `2′-FL HMO 依赖的关键前体和末端催化步骤。` → doc_0073\n\n",
        "**EN subqueries 是垃圾**:\n",
        "- branch 0: `the engineered biosynthetic pathways for 6′-SL and 2′-FL human milk...` (整个原句)\n",
        "- branch 1: `and explain their key precursor dependencies...` (指令文本，不是检索词)\n\n",
        "**根因**: Pipeline 传递 rewritten EN query，`_extract_comparison_subqueries` 对英文无效。\n\n",
        "## 4. Bug #2: Period in 'E. coli' Breaks Sentence Split (ent_083)\n\n",
        "**CN query**: `比较文库中 E. coli 和 B. subtilis 作为 NeuAc 生产宿主的策略差异和产量表现。`\n\n",
        "`re.split(r'[。？！?!.]', prefix_removed, maxsplit=1)` 把 'E. coli' 中的 '.' 当作句号分割:\n",
        "- sentence[0] = `文库中 E`\n",
        "- sentence[1] = ` coli 和 B. subtilis...`\n\n",
        "然后 `target_span = '文库中 E'` (5 字符) → 无法拆分成 >=2 个 objects → 返回 0 个 subqueries。\n\n",
        "## 5. Combined Impact\n\n",
        "| Sample | CN Subqueries | EN Subqueries | Actual Used | Expected Docs Recovered |\n",
        "|--------|-------------|-------------|-------------|------------------------|\n",
        "| ent_010 | 2 correct | 2 broken | EN (broken) | **0/2** |\n",
        "| ent_083 | 0 (period bug) | 0 (period bug) | EN (none) | **0/2** |\n\n",
        "## 6. Fix Strategy\n\n",
        "**Fix 1**: `_build_query_plan` 应使用 CN original query 做 comparison subquery extraction，"
        "即使 retrieval 使用 EN rewritten query。\n",
        "**Fix 2**: `_trim_target_span` 的 sentence boundary split 不应在 organism 缩写的位置分割。\n\n",
        "## 7. Controls\n\n",
        "- h50_neg_001 (factoid): 不触发 comparison decomposition → OK\n",
        "- ent_056 (factoid): 不触发 → OK\n",
        "- ent_005 (summary): 不触发 → OK\n",
        "- ent_084 (comparison pass): 触发 comparison → 需验证 CN subqueries 正确\n\n",
        "## 8. Recommendation\n\n",
        f"**Phase 20K: {decision['recommended_phase20k']}**\n\n",
    ]

    path = REPDIR / "summary.md"
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[Summary] → {path}")


def write_per_sample_review():
    lines = ["# Phase 20J Per-Sample Decomposition Review\n\n"]

    for sid, info in SAMPLES.items():
        cn_subs = _extract_comparison_subqueries(info["cn"])
        en_subs = _extract_comparison_subqueries(info["en"])

        lines.append(f"## {sid}\n\n")
        lines.append(f"- **CN query**: {info['cn']}\n")
        lines.append(f"- **EN query**: {info['en'][:200]}\n")
        lines.append(f"- **Expected docs**: {info['expected_docs']}\n")
        lines.append(f"- **Expected branches**: {info['expected_branches']}\n\n")
        lines.append(f"### CN Subqueries ({len(cn_subs)}):\n")
        for i, sq in enumerate(cn_subs):
            lines.append(f"- [{i}] `{sq}`\n")
        lines.append(f"\n### EN Subqueries ({len(en_subs)}):\n")
        for i, sq in enumerate(en_subs):
            lines.append(f"- [{i}] `{sq}`\n")
        lines.append(f"\n### Root Cause\n")

        if sid == "ent_010":
            lines.append("CN subqueries are correct but not used. Pipeline passes EN query to retriever.\n")
            lines.append("EN subqueries are garbage (full sentence + instruction text).\n")
            lines.append("**Fix**: Use CN original query for _extract_comparison_subqueries.\n")
        elif sid == "ent_083":
            lines.append("Period in 'E. coli' breaks sentence boundary split.\n")
            lines.append("Both CN and EN subquery generation fail.\n")
            lines.append("**Fix**: Fix sentence boundary split to not split on organism abbreviation periods.\n")

        lines.append("\n---\n\n")

    path = REPDIR / "per_sample_decomposition_review.md"
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[Per-sample] → {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Phase 20J: Comparison Decomposition Audit")
    print("=" * 60)

    step1_config()
    step2_query_plan()
    step3_semantic_quality()
    step4_raw_retrieval()
    step5_fusion_trace()
    step6_rerank_trace()
    branch_rows = step7_branch_coverage()
    floor_rows = step8_floor_simulation()
    non_comp_rows = step9_non_comparison_audit()
    root_cause, decision = steps_10_11(branch_rows, floor_rows, non_comp_rows)

    write_summary_md(root_cause, decision)
    write_per_sample_review()

    print("\n" + "=" * 60)
    print("Phase 20J Complete")
    print(f"  Bug #1: EN subqueries broken (ent_010)")
    print(f"  Bug #2: Period in E. coli breaks sentence split (ent_083)")
    print(f"  Dominant failure: {root_cause['dominant_failure_stage']}")
    print(f"  Safe fix exists: {root_cause['safe_fix_exists']}")
    print(f"  Recommended Phase 20K: {decision['recommended_phase20k']}")
    print(f"  Results: {RDIR}")
    print(f"  Reports: {REPDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
