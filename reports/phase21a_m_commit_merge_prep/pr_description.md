# Phase 21A: Smoke200 Frozen Baseline Convergence

## Summary

本 PR 收敛 smoke150/smoke200 frozen baseline，稳定以下 RAG pipeline 行为：
- Query rewrite wiring（修复 100% fallback）
- Support diversity retention（消除 same-doc overcrowding）
- Negative route classification（explicit abstain → 0 citations）
- Frozen eval rewrite cache（确保评测可复现）
- Citation completion for used-support items
- CN comparison parser pattern

同时修正 eval dataset bad5 → replacement5，更新 rewrite caches / manifest / registry。

## What Changed

### Code fixes (7 fixes, 9 src files, +401/-21 lines)

| Phase | Fix | Files | Effect |
|-------|-----|-------|--------|
| 9C | Rewrite wiring | `pipeline.py`, `config.py`, `query_rewrite_service.py`, `openai_compatible.py` | llm_client=None → 100% fallback fixed |
| 9G | Support diversity | `support_selector.py` | Same-doc duplicate swap prevents overcrowding |
| 9I | Negative route | `schemas.py`, `router.py`, `pipeline.py` | Explicit abstain → negative route → 0 citations |
| 9K | Frozen rewrite cache | `query_rewrite_service.py`, `config.py`, `pipeline.py` | Pre-computed cache ensures reproducibility |
| 9M | Second support fix | `support_selector.py` | Scored_median bypass + close-margin capacity |
| 9O | Citation completion | `citation_binder.py` | Used-support items get cited even without markers |
| 9Q | CN parser pattern | `branch_parser.py` | A和B作为X comparison parsing |

### Dataset / Cache (7 files, git add -f)

- `data/eval/datasets/smoke150.jsonl` — canonicalized (smoke50 + smoke100 → 150 IDs)
- `data/eval/datasets/smoke200.jsonl` — bad5 replaced (p21a_added50_014-018 → p21a9v2_fact_001-005)
- `data/eval/rewrite_cache/smoke150_rewrites.jsonl` — 150 frozen rewrites
- `data/eval/rewrite_cache/smoke200_rewrites.jsonl` — 200 frozen rewrites
- `data/eval/manifests/smoke200_manifest.json` — distribution, routes, categories
- `data/eval/manifests/smoke150_manifest.json` — canonicalization record
- `data/eval/registry.json` — canonical paths for all datasets

### Evaluation Diagnostics (16 scripts + 3 tests)

- Phase 21A diagnostic/validation scripts (`scripts/evaluation/run_phase21a*`)
- Unit tests (`tests/test_phase21a9c/g/i_*.py`)
- Convergence reports (`results/phase21a9y_*/`, `reports/phase21a9y_*/`)
- Final closeout reports (`results/phase21a_z_*/`, `reports/phase21a_z_*/`)

## Validation

### Final Smoke150 Baseline
- real_P0 = 3 (ent_058, ent_083, ent_094)
- frozen_rewrite = 150/150, live = 0, fallback = 0
- 3 runs identical

### Final Smoke200 Baseline
- real_P0 = 6/200 (3 smoke150 + 3 replacement5)
- doc_hit_rate = 1.000, doc_miss = 0
- frozen_rewrite = 200/200, live = 0, fallback = 0
- negative = 0, citation_binding = 0

### Safety
- negative_safety: **PASS** (6/6 negative, 0 citation)
- citation_safety: **PASS** (0 wrong-doc, 0 citation_inflation)
- rewrite_safety: **PASS** (100% frozen, 0 live)
- dataset_safety: **PASS** (200 unique IDs, no conflicts)

## Known Residuals (6 total, not blocking merge)

| ID | Sample | Class | Deferred Reason |
|----|--------|-------|-----------------|
| ent_058 | smoke150 | support_selection | Close-margin capacity cutoff |
| ent_083 | smoke150 | branch_query_context | Query enrichment needed |
| ent_094 | smoke150 | support_selection | Expected doc score negative |
| p21a9v2_fact_001 | replacement | support_selection | Same pattern as ent_058 |
| p21a9v2_fact_002 | replacement | retrieval_doc_miss | Query-doc matching (P2) |
| p21a9v2_fact_004 | replacement | retrieval_doc_miss | Query-doc matching (P2) |

## RAGAS Status

**RAGAS excluded from this PR's merge gate.** RAGAS evaluation exposed evaluator issues (BGE-M3 score scale, judge strictness, reference format incompatibility) rather than pipeline quality failures. RAGAS scripts, results, and reports are not included in this PR. RAGAS metric adaptation will be handled in a separate Phase 21B-Eval.

## Risk Assessment

- **No sample/doc ID hardcoding** — all fixes are generic rules/policies
- **No experimental flags** — qwen_synthesis=false, alias_expansion=false
- **Backward compatible** — v1 pipeline path unchanged
- **Residuals documented** — 6 explained, 10 backlog items for Phase 21B
- **3-commit split** — code, data, diagnostics separated for review clarity

## Backlog / Next Phase

Suggested Phase 21B scope:
1. PDF parser / table / figure / double-column cleanup (high priority)
2. Branch query context enhancement (ent_083 pattern)
3. Support selection residual (BL-001~004)
4. RAGAS metric adaptation / reference answer construction
5. Negative abstention detector improvement
6. Future larger-scale evaluation

## Review Notes

- **Commit 1** (code): Review src changes for correctness and no regression. Key files: `support_selector.py` (+158 lines) and `query_rewrite_service.py` (+107 lines).
- **Commit 2** (data): Verify smoke200.jsonl has 200 unique IDs, 150 smoke150 + 50 added50. Confirm bad5 removed, replacement5 added.
- **Commit 3** (diag): Verify scripts run with frozen cache. Reports are documentation artifacts.
