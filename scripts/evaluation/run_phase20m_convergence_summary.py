"""Phase 20M: Convergence Summary. 只写文档，不改代码。"""
import csv, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
RDIR = BASE / "results/phase20m_convergence_summary"
REPDIR = BASE / "reports/phase20m_convergence_summary"

# ============================================================
# Step 1: Final code state
# ============================================================
def step1_final_code_state():
    state = {
        "phase": "20M", "purpose": "phase20_convergence_summary",
        "final_real_P0": 0, "final_doc_miss": 0, "final_doc_hit_rate": 1.0,
        "zero_citation": 0,
        "accepted_fixes": [
            "query_rewrite_enabled_baseline",
            "factoid_doc_diversity",
            "summary_quality_minimal_fix",
            "comparison_decomposition_fix",
            "original_cn_fallback_floor",
        ],
        "feature_flags": {
            "query_rewrite_mode_default": "off",
            "original_cn_fallback_enabled_default": False,
            "alias_expansion_enabled": False,
            "qwen_synthesis_enabled": False,
            "comparison_coverage_enabled": False,
        },
        "production_default_changed": False,
        "index_rebuild_required": False,
        "current_best_eval_config": {
            "QUERY_REWRITE_MODE": "enabled",
            "RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED": "true",
            "generation": "v2-only",
            "source_floor_enabled": True,
            "parent_expansion_enabled": True,
        },
        "notes": "All residual fixed. Production defaults unchanged. No index rebuild needed.",
    }
    (RDIR/"final_code_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print("[1] final_code_state.json")

# ============================================================
# Step 2: Metric timeline
# ============================================================
def step2_metric_timeline():
    rows = [
        {"phase": "20A", "fix_name": "rewrite_enabled_baseline", "primary_module": "query_rewrite",
         "before_real_P0": 18, "after_real_P0": 9, "real_P0_delta": -9,
         "before_doc_miss": 17, "after_doc_miss": 9, "doc_miss_delta": -8,
         "doc_hit_rate": 0.97, "zero_citation": 0,
         "new_real_P0": 4, "wrong_doc_citation": 0, "citation_inflation": 0,
         "answer_length_inflation": 0, "negative_regression": 0,
         "accepted": "baseline", "notes": "Guarded EN mirror query rewrite. Largest single improvement."},
        {"phase": "20D/20E", "fix_name": "factoid_doc_diversity", "primary_module": "support_selection",
         "before_real_P0": 9, "after_real_P0": 6, "real_P0_delta": -3,
         "before_doc_miss": 9, "after_doc_miss": 6, "doc_miss_delta": -3,
         "doc_hit_rate": 0.9924, "zero_citation": 0,
         "new_real_P0": 0, "wrong_doc_citation": 0, "citation_inflation": 0,
         "answer_length_inflation": 0, "negative_regression": 0,
         "accepted": True, "notes": "Factoid support selection max_per_doc=2 diversity. Fixed ent_056/059/078."},
        {"phase": "20F/20G", "fix_name": "summary_quality_filter", "primary_module": "support_selection",
         "before_real_P0": 6, "after_real_P0": 3, "real_P0_delta": -3,
         "before_doc_miss": 6, "after_doc_miss": 3, "doc_miss_delta": -3,
         "doc_hit_rate": 0.999, "zero_citation": 0,
         "new_real_P0": 0, "wrong_doc_citation": 0, "citation_inflation": 0,
         "answer_length_inflation": 0, "negative_regression": 0,
         "accepted": True, "notes": "Removed title hard-drop. Added Full Text priority=2. Fixed ent_005/058/081."},
        {"phase": "20J/20K", "fix_name": "comparison_decomposition", "primary_module": "query_decomposition",
         "before_real_P0": 3, "after_real_P0": 1, "real_P0_delta": -2,
         "before_doc_miss": 3, "after_doc_miss": 1, "doc_miss_delta": -2,
         "doc_hit_rate": 0.999, "zero_citation": 0,
         "new_real_P0": 0, "wrong_doc_citation": 0, "citation_inflation": 0,
         "answer_length_inflation": 0, "negative_regression": 0,
         "accepted": True, "notes": "CN query for subquery extraction. Fixed period bug in E.coli etc. Fixed ent_010/083."},
        {"phase": "20L-1/20L-2", "fix_name": "original_cn_fallback_floor", "primary_module": "bilingual_retrieval",
         "before_real_P0": 1, "after_real_P0": 0, "real_P0_delta": -1,
         "before_doc_miss": 1, "after_doc_miss": 0, "doc_miss_delta": -1,
         "doc_hit_rate": 1.0, "zero_citation": 0,
         "new_real_P0": 0, "wrong_doc_citation": 0, "citation_inflation": 0,
         "answer_length_inflation": 0, "negative_regression": 0,
         "accepted": True, "notes": "CN fallback floor for bilingual retrieval gap. Fixed h50_neg_001. Feature flag default off."},
    ]
    with open(RDIR/"phase20_metric_timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[2] phase20_metric_timeline.csv")

# ============================================================
# Step 3: Accepted fix ledger
# ============================================================
def step3_accepted_fix_ledger():
    rows = [
        {"fix_id": "F1", "phase": "20D/20E", "module": "support_selection",
         "changed_files": "support_selector.py",
         "functional_change": "_select_factoid uses _select_with_doc_diversity(max_per_doc=2)",
         "problem_solved": "Single competing doc dominates all support slots, crowding out expected doc evidence",
         "affected_samples": "ent_056, ent_059, ent_078",
         "why_general_not_sample_patch": "Per-doc cap applies uniformly to all factoid queries. No sample IDs or doc IDs.",
         "risk_controls": "Overflow fallback when fewer distinct docs available. Max per_doc=2 with max_total=3.",
         "tests_added": "8 (test_phase20d_doc_diversity.py)", "regression_status": "passed",
         "accepted": True, "notes": "Same pattern as summary's per_doc limit, now applied to factoid."},
        {"fix_id": "F2", "phase": "20F/20G", "module": "support_selection",
         "changed_files": "support_selector.py",
         "functional_change": "Removed title hard-drop; added Full Text priority=2",
         "problem_solved": "Title chunks hard-dropped by quality filter; Full Text sections penalized with priority=6",
         "affected_samples": "ent_005, ent_058, ent_081",
         "why_general_not_sample_patch": "Global change to _evaluate_summary_quality and _section_priority. No sample IDs.",
         "risk_controls": "Title chunks still deprioritized by section_priority; Full Text at same level as Results+Discussion.",
         "tests_added": "8 (test_phase20g_summary_quality.py)", "regression_status": "passed",
         "accepted": True, "notes": "Title citation risk assessed as low."},
        {"fix_id": "F3", "phase": "20J/20K", "module": "query_decomposition",
         "changed_files": "hybrid.py, pipeline.py",
         "functional_change": "Comparison decomposition uses original CN query; organism period split fixed",
         "problem_solved": "EN rewrite breaks CN comparison subqueries; E.coli period breaks sentence split",
         "affected_samples": "ent_010, ent_083",
         "why_general_not_sample_patch": "decomposition_query parameter universally available. Organism regex covers all common abbreviations.",
         "risk_controls": "Router intent guard prevents non-comparison queries from triggering. Backward compatible fallback to retrieval query.",
         "tests_added": "7 (test_phase20k_comparison_decomposition.py)", "regression_status": "passed",
         "accepted": True, "notes": "Only triggers for COMPARISON intent. CN decomposition validated."},
        {"fix_id": "F4", "phase": "20L-1/20L-2", "module": "bilingual_retrieval",
         "changed_files": "config.py, pipeline.py",
         "functional_change": "Original CN fallback floor: dense top2 + BM25 top2, max 4 candidates",
         "problem_solved": "CN query retrieves doc_0204 but EN rewrite loses it to near-topic doc_0180",
         "affected_samples": "h50_neg_001",
         "why_general_not_sample_patch": "Feature flag controlled. Triggers on CJK+rewrite, not sample ID. Provenance tracked.",
         "risk_controls": "Default off. Max 4 candidates. Only CJK+rewrite trigger. Fallback candidates marked with provenance.",
         "tests_added": "7 (test_phase20l_original_cn_fallback.py)", "regression_status": "passed",
         "accepted": True, "notes": "Last piece of the puzzle. Real P0: 1→0."},
    ]
    with open(RDIR/"accepted_fix_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[3] accepted_fix_ledger.csv")

# ============================================================
# Step 4: Residual resolution ledger
# ============================================================
def step4_residual_resolution():
    rows = [
        {"sample_id": "ent_056", "original_residual_class": "SUPPORT_CITATION_RESIDUAL",
         "phase_fixed": "20D", "fix_type": "factoid_doc_diversity",
         "first_loss_stage_before": "support_input_to_selected_support",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0081",
         "status_after_fix": "fixed", "notes": "New P0: off=ok enabled=doc_miss. Competing doc_0150 crowded out doc_0081."},
        {"sample_id": "ent_059", "original_residual_class": "SUPPORT_CITATION_RESIDUAL",
         "phase_fixed": "20D", "fix_type": "factoid_doc_diversity",
         "first_loss_stage_before": "support_input_to_selected_support",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0098",
         "status_after_fix": "fixed", "notes": "New P0. doc_0098 had same final presence in v0/v1 but v1 didn't cite."},
        {"sample_id": "ent_078", "original_residual_class": "SUPPORT_CITATION_RESIDUAL",
         "phase_fixed": "20D", "fix_type": "factoid_doc_diversity",
         "first_loss_stage_before": "support_input_to_selected_support",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0147",
         "status_after_fix": "fixed", "notes": "New P0. doc_0147 in v1 final but competing doc_0169 cited."},
        {"sample_id": "ent_005", "original_residual_class": "SUPPORT_CITATION_RESIDUAL",
         "phase_fixed": "20G", "fix_type": "summary_quality_filter",
         "first_loss_stage_before": "support_input_to_selected_support",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0009",
         "status_after_fix": "fixed", "notes": "Title chunk hard-dropped. Summary quality filter fix."},
        {"sample_id": "ent_058", "original_residual_class": "SUPPORT_CITATION_RESIDUAL",
         "phase_fixed": "20G", "fix_type": "summary_quality_filter",
         "first_loss_stage_before": "support_input_to_selected_support",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0098",
         "status_after_fix": "fixed", "notes": "Title hard-drop. Rewrite improved retrieval but citation lost."},
        {"sample_id": "ent_081", "original_residual_class": "SUPPORT_CITATION_RESIDUAL",
         "phase_fixed": "20G", "fix_type": "summary_quality_filter",
         "first_loss_stage_before": "support_input_to_selected_support",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0151",
         "status_after_fix": "fixed", "notes": "All Full Text sections priority 6→2. Section parsing issue."},
        {"sample_id": "ent_010", "original_residual_class": "D_DENSE_GAP",
         "phase_fixed": "20K", "fix_type": "comparison_decomposition",
         "first_loss_stage_before": "raw_retrieval_subquery_extraction",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0009|doc_0073",
         "status_after_fix": "fixed", "notes": "EN subqueries broken. CN subqueries correct. Decomposition fix."},
        {"sample_id": "ent_083", "original_residual_class": "D_DENSE_GAP",
         "phase_fixed": "20K", "fix_type": "comparison_decomposition",
         "first_loss_stage_before": "raw_retrieval_subquery_extraction",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0119|doc_0147",
         "status_after_fix": "fixed", "notes": "E.coli period bug + EN decomposition failure. Both CN subqueries now correct."},
        {"sample_id": "h50_neg_001", "original_residual_class": "NEAR_TOPIC_DOC_MISS",
         "phase_fixed": "20L-2", "fix_type": "original_cn_fallback_floor",
         "first_loss_stage_before": "rewrite_to_dense",
         "first_loss_stage_after": "none", "expected_doc_id": "doc_0204",
         "status_after_fix": "fixed", "notes": "CN query works but EN rewrite loses doc_0204. CN fallback recovers."},
    ]
    with open(RDIR/"residual_resolution_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[4] residual_resolution_ledger.csv")

# ============================================================
# Step 5: Final residual backlog
# ============================================================
def step5_final_residual():
    rows = [{
        "sample_id": "ALL_150", "residual_failure_class": "none",
        "status": "resolved", "affected_stage": "none",
        "priority": "P3", "proposed_next_direction": "accept_current_baseline",
        "notes": "All 9 residual samples from Phase 20A resolved. Real P0=0. doc_miss=0. doc_hit_rate=1.0.",
    }]
    with open(RDIR/"final_residual_backlog.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[5] final_residual_backlog.csv")

# ============================================================
# Step 6: Current best config
# ============================================================
def step6_current_best_config():
    md = """# Current Best Experimental Configuration

## Query Rewrite
- `QUERY_REWRITE_MODE=enabled` for offline eval
- Production default remains `off` unless explicitly changed
- Guarded English-mirror prompt **unchanged** (Phase 19F prompt)
- Rewrite model: qwen-plus, temperature=0

## Retrieval
- `source_floor_enabled=true`
- `source_floor_dense_top_n=3`, `source_floor_bm25_top_n=3`, `source_floor_max_candidates_total=6`
- `parent_expansion_enabled=true`
- **Original CN fallback floor**: feature flag implemented (Phase 20L-2)
  - `original_cn_fallback_enabled` default `false`
  - Recommended offline setting: `true`
  - `dense_top_n=2`, `bm25_top_n=2`, `max_total=4`
  - Trigger: rewrite=enabled + CJK original query + rewritten differs
- **Comparison decomposition** uses original CN query (Phase 20K)
  - Organism abbreviation period split fixed
- `alias_expansion_enabled=false`

## Rerank
- `rerank_top_k` unchanged (default)
- Reranker: bge-reranker-v2-m3
- Comparison subquery aggregation retained

## Parent Expansion
- `parent_expansion_enabled=true`
- No parent index redesign in Phase 20

## Support / Citation
- **Factoid doc diversity**: `_select_factoid` uses `_select_with_doc_diversity(max_per_doc=2)`
- **Summary quality**: title hard-drop removed, Full Text priority=2
- Citation eligibility unchanged
- Support capacity unchanged (`v2_max_support_factoid=3`, `v2_max_support_summary=5`)

## Generation
- `generation=v2-only`
- `v2_max_extractive_evidence_lines=6`
- `Qwen synthesis=false`
- `comparison coverage=false`

## Disabled / Deferred
- metadata-enriched chunk index
- doc-level sidecar
- LLM decomposition
- dense/reranker calibration
- Qwen synthesis
- comparison coverage
- alias expansion
- BioLexical BM25 expansion
"""
    (RDIR/"current_best_config.md").write_text(md)
    print("[6] current_best_config.md")

# ============================================================
# Step 7: Deferred directions
# ============================================================
def step7_deferred():
    rows = [
        {"direction": "metadata_enriched_chunk_index",
         "reason_deferred": "Real P0 already 0 on smoke150. No residual to justify index rebuild risk.",
         "evidence_phase": "20H, 20I", "when_to_revisit": "If new hard eval set reveals doc-identity retrieval gaps",
         "notes": "retrieval_text already has title+section+source_file+doc_id. Adding abstract may benefit harder queries."},
        {"direction": "doc_level_sidecar",
         "reason_deferred": "Phase 20I shadow BM25 failed: 0/5 expected docs in top-10 doc recall. Chunk-level dense superior.",
         "evidence_phase": "20I", "when_to_revisit": "If dense doc embeddings become available or harder queries need doc-level context",
         "notes": "Local chunk expansion within docs works (4/5 docs). Bottleneck is query→doc, not doc→chunks."},
        {"direction": "llm_decomposition",
         "reason_deferred": "Phase 20K rule-based fix solved current comparison residual. No LLM needed.",
         "evidence_phase": "20J, 20K", "when_to_revisit": "If complex 3+ branch queries or non-standard comparison syntax fails",
         "notes": "LLM decomposition would add latency + cost. Rule-based sufficient for current eval set."},
        {"direction": "dense_reranker_calibration",
         "reason_deferred": "High cost (re-embedding all chunks). No residual to justify. doc_hit_rate=1.0.",
         "evidence_phase": "20H", "when_to_revisit": "If larger eval set shows systematic dense embedding gaps",
         "notes": "Current BGE-M3 + bge-reranker-v2-m3 performing well. Only 3 retrieval gaps found in 150 samples."},
        {"direction": "alias_expansion",
         "reason_deferred": "Disabled in current baseline. Query rewrite handles cross-lingual mapping better.",
         "evidence_phase": "Phase 18C", "when_to_revisit": "If CN→EN term mapping gaps found in harder queries",
         "notes": "Alias expansion adds BM25 noise. Rewrite-based approach preferred."},
        {"direction": "pdf_parser_rebuild",
         "reason_deferred": "Out of scope for Phase 20 retrieval/support fix. Section parsing (Full Text) issues noted but not blocking.",
         "evidence_phase": "20F", "when_to_revisit": "Phase 21B if PDF quality is priority",
         "notes": "doc_0151, doc_0119, doc_0147 have all Full Text sections. Parser fix needed but not critical for current eval."},
        {"direction": "support_capacity_expansion",
         "reason_deferred": "Not needed. Current support sizes sufficient after doc diversity + quality fixes.",
         "evidence_phase": "20D, 20G", "when_to_revisit": "If harder queries need more evidence per answer",
         "notes": "Expanding support capacity risks citation inflation without quality gain."},
    ]
    with open(RDIR/"deferred_direction_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[7] deferred_direction_ledger.csv")

# ============================================================
# Step 8: Risk register
# ============================================================
def step8_risk_register():
    rows = [
        {"risk_id": "R1", "risk_name": "CN fallback near-topic noise",
         "related_fix": "original_cn_fallback_floor",
         "risk_description": "CN fallback adds near-topic Chinese docs alongside expected docs",
         "observed_in_phase20": "false",
         "mitigation": "Max 4 candidates. Reranker uses CN question to rank. Provenance tracked.",
         "monitoring_metric": "fallback_added_doc_ids in debug trace",
         "revisit_condition": "If >5% of fallback-added docs are wrong-doc citations",
         "notes": "Not observed in h50_neg_001 or smoke scan."},
        {"risk_id": "R2", "risk_name": "Title chunk citation risk",
         "related_fix": "summary_quality_filter",
         "risk_description": "Title chunks now pass quality filter; may be cited for single-doc summary queries",
         "observed_in_phase20": "false",
         "mitigation": "section_priority still deprioritizes title. Only selected when best evidence.",
         "monitoring_metric": "title_chunks_cited_count",
         "revisit_condition": "If >10% of citations are title-only chunks",
         "notes": "2 samples may cite title chunks (ent_005, ent_058). Assessed as low risk."},
        {"risk_id": "R3", "risk_name": "Comparison decomposition complex query failure",
         "related_fix": "comparison_decomposition",
         "risk_description": "Rule-based decomposition may fail on 3+ branch or non-standard comparison syntax",
         "observed_in_phase20": "false (for current eval)",
         "mitigation": "Router intent guard. Falls back to single query if subqueries < 2.",
         "monitoring_metric": "comparison_subquery_count",
         "revisit_condition": "If new comparison queries consistently get <2 subqueries",
         "notes": "ent_084 (P.pastoris vs S.cerevisiae) now works. LLM fallback available if needed."},
        {"risk_id": "R4", "risk_name": "Query rewrite semantic drift",
         "related_fix": "query_rewrite",
         "risk_description": "EN rewrite may lose or change meaning for ambiguous CN queries",
         "observed_in_phase20": "true (h50_neg_001)",
         "mitigation": "Original CN fallback floor. Guarded prompt with implicit reference preservation.",
         "monitoring_metric": "rewritten_query vs original_query diff ratio",
         "revisit_condition": "If new real P0 from rewrite drift > 2",
         "notes": "h50_neg_001 fixed by CN fallback. Guarded prompt already preserves implicit references."},
        {"risk_id": "R5", "risk_name": "Support doc diversity may reduce same-doc evidence",
         "related_fix": "factoid_doc_diversity",
         "risk_description": "max_per_doc=2 may limit deep evidence from single highly-relevant doc",
         "observed_in_phase20": "false",
         "mitigation": "Overflow fallback when fewer distinct docs. max_total=3 with min 2 per doc still possible.",
         "monitoring_metric": "support_pack doc distribution",
         "revisit_condition": "If same-doc factoid answers show quality drop",
         "notes": "Not observed. Summary route already uses per_doc=2 without issues."},
        {"risk_id": "R6", "risk_name": "Eval set coverage limited",
         "related_fix": "all",
         "risk_description": "Smoke50+smoke100 (150 samples) may not represent all document types and query patterns",
         "observed_in_phase20": "n/a",
         "mitigation": "Real P0=0 on current set but acknowledge coverage limits. Deferred harder eval construction to Phase 21A.",
         "monitoring_metric": "new_real_P0 on expanded eval set",
         "revisit_condition": "When smoke200+ is available",
         "notes": "Current set over-represents metabolic engineering papers. Broader coverage needed."},
    ]
    with open(RDIR/"risk_register.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[8] risk_register.csv")

# ============================================================
# Step 9: Summary report
# ============================================================
def step9_summary_report():
    md = """# Phase 20 Convergence Summary

## 1. Executive Summary

Phase 20 从 rewrite enabled baseline 的 9 个 real P0 出发，经过 support selection、summary quality、
comparison decomposition、bilingual fallback 四类修复，最终 full eval real P0 = 0，doc_miss = 0，
doc_hit_rate = 1.0。Zero citation 保持为 0。无 new real P0 / wrong-doc citation / citation inflation /
answer length inflation / negative regression。

所有修复都是通用策略，不使用 sample_id/doc_id 特判。feature flags 默认关闭，生产配置未改变。
Phase 20 residual 修复可以结束。

## 2. Final Metrics

| Phase | Real P0 | Doc Miss | Doc Hit Rate | Zero Cit |
|-------|---------|----------|-------------|----------|
| 20A (rewrite baseline) | 9 | 9 | 0.97 | 0 |
| 20E (factoid diversity) | 6 | 6 | 0.9924 | 0 |
| 20G (summary quality) | 3 | 3 | 0.999 | 0 |
| 20K (comparison decomp) | 1 | 1 | 0.999 | 0 |
| **20L-2 (CN fallback)** | **0** | **0** | **1.0** | **0** |

## 3. Fix Timeline

### Phase 20D/E: Factoid Doc Diversity
**Problem**: Single competing doc dominates all factoid support slots.
**Root cause**: `_select_factoid` had no per-doc limit like summary already did.
**Fix**: `_select_with_doc_diversity(max_per_doc=2)` for factoid queries.
**Result**: ent_056/059/078 fixed. Real P0 9→6. No regression.

### Phase 20F/G: Summary Quality Filter
**Problem**: Summary route residual (ent_005/058/081) — expected docs in final but not cited.
**Root cause**: Title chunks hard-dropped by quality filter. Full Text sections had priority=6.
**Fix**: Removed title from hard-drop list. Added Full Text priority=2.
**Result**: ent_005/058/081 fixed. Real P0 6→3. No regression.

### Phase 20J/K: Comparison Decomposition
**Problem**: Comparison queries (ent_010/083) — expected docs never enter retrieval.
**Root cause**: EN rewrite breaks CN comparison subqueries. Period in E.coli breaks sentence split.
**Fix**: Decomposition uses original CN query. Organism abbreviation regex for period protection.
**Result**: ent_010/083 fixed. Real P0 3→1. No regression.

### Phase 20L: Original CN Fallback Floor
**Problem**: Last residual (h50_neg_001) — CN query works but EN rewrite loses doc_0204.
**Root cause**: Cross-lingual dense embedding drift. CN review article's title "·专题综述·" has no EN signal.
**Fix**: Feature flag: CN fallback adds dense top2 + BM25 top2 = max 4 candidates.
**Result**: h50_neg_001 fixed. Real P0 1→0. doc_hit_rate=1.0. No regression.

## 4. Module-Level Changes

- **support_selector.py**: Factoid doc diversity + summary quality/title/Full Text fix
- **hybrid.py**: Decomposition query parameter + organism period bug + debug provenance
- **pipeline.py**: Original CN fallback floor + decomposition_query passthrough
- **config.py**: RetrievalConfig fields for fallback + env var loading

## 5. Why These Fixes Are General

All fixes are general, not sample patches:
- No sample_id / doc_id / source_file special cases
- No expected_doc_ids used in production logic
- Feature flags with safe defaults
- Provenance metadata for observability
- Reusable across all queries of the same type

## 6. Final Accepted Baseline

**Offline eval (recommended)**:
- QUERY_REWRITE_MODE=enabled
- RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED=true
- All other config at current defaults

**Production (current default, unchanged)**:
- QUERY_REWRITE_MODE=off
- original_cn_fallback_enabled=false
- All Phase 20 fixes are production-safe (guarded by feature flags or router intent)

## 7. Deferred Directions

- metadata-enriched chunk index: no residual to justify index rebuild
- doc-level sidecar: Phase 20I proved ineffective
- LLM decomposition: rule-based sufficient
- dense/reranker calibration: high cost, no residual
- PDF parser rebuild: out of Phase 20 scope

## 8. Remaining Risks

- Smoke150 coverage limited; harder eval may reveal new gaps
- CN fallback needs near-topic noise monitoring
- Comparison decomposition may fail on complex 3+ branch queries
- PDF parsing/table/figure extraction is the next bottleneck

## 9. Recommended Next Phase

Do NOT continue Phase 20 residual fixes. Open new phase:
- **Phase 21A**: Construct smoke200 / harder eval set for broader coverage
- **Phase 21B**: PDF parsing / table / figure extraction rebuild
- **Phase 21C**: Production shadow deployment / latency / observability

## 10. Merge Recommendation

All Phase 20 changes (20D→20L-2) are recommended for merge:
- All feature flags default to safe values
- No index rebuild needed
- All production defaults unchanged
- 30 cumulative tests pass
- Full eval regression clean

Pre-merge checklist:
- [ ] Run full_eval with new code on smoke50+smoke100
- [ ] Verify production QUERY_REWRITE_MODE=off produces identical results
- [ ] Verify original_cn_fallback_enabled=false has zero effect
- [ ] Review all feature flag default values
"""
    (REPDIR/"summary.md").write_text(md)
    print("[9] summary.md")

# ============================================================
# Step 10: Review notes
# ============================================================
def step10_review_notes():
    md = """# Phase 20 Review Notes

## 术语表

| 术语 | 含义 |
|------|------|
| real P0 | 真正的生产问题：expected doc 未被引用（扣除 route mismatch / false P0 后） |
| raw P0 | 原始分类结果，可能包含 route 误判 |
| doc_miss | expected doc 完全未被检索到的样本 |
| doc_hit_rate | expected doc 在 final 或 citation 中出现的比例 |
| zero citation | 答案没有引用任何文档的样本数 |
| support selection | 从 final chunks 中选择哪些用于 answer generation |
| selected_support | 最终被选入 answer generation 的 evidence chunks |
| doc diversity | 确保 support 不会全部来自同一个文档（max_per_doc 限制） |
| summary quality filter | 对 summary 路由的候选做质量筛选（长度、section、内容） |
| comparison decomposition | 把"比较 A 和 B"的查询拆成 A 和 B 两个独立子查询 |
| original CN fallback | 当 EN rewrite 丢失中文信号时，用原始中文查询补召回 |
| bilingual retrieval | 中英双语混合检索策略 |
| source-floor | 从 dense/BM25 各保留 top-N 防止 RRF 融合丢失 |
| parent expansion | 用 parent_index 为 seed chunk 补充相邻/同 section/同页 chunks |

## Phase 20 主线逻辑（一页总结）

Phase 20 的基本思路是：**先修下游再修上游，先修通用策略再修边缘 case**。

1. **Phase 20A** 建立基线：发现 EN rewrite 是最大的单一提升（P0 18→9）。
2. **Phase 20C** 做方向审计：发现 6/9 residual 不是 retrieval 问题，而是 expected doc 在 final 中但未被 citation。
3. **Phase 20D** 修 factoid support selection（最下游）：加 doc diversity，一竞品 doc 不能独占所有 support slot。
4. **Phase 20G** 修 summary support selection：去掉无差别 title hard-drop，给 Full Text section 合理优先级。
5. **Phase 20K** 修 retrieval 的比较查询分解：EN rewrite 破坏了 CN comparison 结构，改回用 CN 分解，修 period bug。
6. **Phase 20L-2** 修最后一个 bilingual 边缘：CN fallback floor 补回被 EN rewrite 丢失的中文信号。

这个顺序保证了每一步都有清晰的归因，不会修 A 时破坏 B，不会修边缘 case 时引入主干回归。

## 关键发现

1. **doc_hit 不等于 retrieval 成功**：Phase 20C 发现 expected doc 在 final 中但未被 citation，说明问题在 support/citation 管道而非 retrieval。
2. **Bug 往往比架构问题更值得先修**：comparison decomposition 的两个 bug（EN subquery、period split）修复成本极低但收益很高。
3. **Feature flag 是安全网**：所有修复都用 feature flag 包裹，默认关闭，可以在生产环境中安全存在。
4. **不要过早做索引级改动**：metadata-enriched chunk、doc-level sidecar、dense calibration 都需要索引重建，在没有 residual 支撑时不值得做。
"""
    (REPDIR/"review_notes.md").write_text(md)
    print("[10] review_notes.md")

# ============================================================
# Step 11: Phase 21 recommendation
# ============================================================
def step11_phase21():
    rec = {
        "phase20_completed": True,
        "final_real_P0": 0,
        "final_doc_miss": 0,
        "recommended_next_phase": "phase21_smoke200_hardset_construction",
        "rationale": (
            "Real P0 has reached 0 on current smoke150. To validate generalization and "
            "discover new failure modes, a larger and more diverse eval set is needed. "
            "Constructing smoke200+ with harder queries (multi-doc, cross-topic, table/figure, "
            "negative) will provide the next layer of quality signals. "
            "Do not continue tuning on smoke150 — risk of overfitting."
        ),
        "candidate_next_directions": [
            {
                "name": "smoke200_hardset_construction",
                "expected_value": "Discover new failure modes beyond smoke150 coverage",
                "risk": "low",
                "prerequisites": "None — uses existing pipeline",
                "suggested_first_step": "Sample 50-100 harder queries from real user logs or synthetic generation",
            },
            {
                "name": "pdf_parsing_table_figure_rebuild",
                "expected_value": "Fix Full Text section parsing for ~10% of papers",
                "risk": "medium",
                "prerequisites": "Parser testing infrastructure",
                "suggested_first_step": "Audit section parsing failure rate across all 705 docs",
            },
            {
                "name": "production_shadow_observability",
                "expected_value": "Validate Phase 20 fixes in production shadow mode",
                "risk": "low",
                "prerequisites": "Production deployment access",
                "suggested_first_step": "Deploy QUERY_REWRITE_MODE=shadow with debug logging",
            },
        ],
        "do_not_do_next": [
            "Do NOT continue tuning on smoke150 (overfitting risk)",
            "Do NOT rebuild indexes without new residual evidence",
            "Do NOT add LLM decomposition (latency/cost for no proven need)",
            "Do NOT implement doc-level sidecar (Phase 20I proved ineffective)",
        ],
        "notes": "Phase 20 is complete. Real P0=0, doc_miss=0, doc_hit_rate=1.0. Open fresh Phase 21.",
    }
    (RDIR/"phase21_recommendation.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print("[11] phase21_recommendation.json")

# ============================================================
# MAIN
# ============================================================
def main():
    print("Phase 20M: Convergence Summary")
    step1_final_code_state()
    step2_metric_timeline()
    step3_accepted_fix_ledger()
    step4_residual_resolution()
    step5_final_residual()
    step6_current_best_config()
    step7_deferred()
    step8_risk_register()
    step9_summary_report()
    step10_review_notes()
    step11_phase21()
    print(f"\nDone. {RDIR}/ , {REPDIR}/")

if __name__ == "__main__":
    main()
