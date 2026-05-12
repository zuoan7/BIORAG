# Phase 21A-Z Final Closeout Without RAGAS

## 1. Purpose
基于 Phase 21A smoke150 / smoke200 frozen baseline 做最终收口、归档、提交准备和合并建议。**RAGAS 明确排除在 Phase 21A 合并门控之外。**

## 2. Final Smoke Baselines

### Smoke150
- real_P0 = 3 (ent_058, ent_083, ent_094)
- frozen rewrite = 150/150, live = 0, fallback = 0
- 3 runs identical
- negative = 0, citation_binding = 0

### Smoke200
- real_P0 = 6/200
- smoke150 P0 = 3, added50 P0 = 3 (all replacement5)
- unchanged added50 real_P0 = 0
- doc_miss = 0, doc_hit_rate = 1.000
- frozen rewrite = 200/200, live = 0, fallback = 0

## 3. Code Fixes Kept (7 fixes, 9 src files)

| Phase | Fix | Files | Validation |
|-------|-----|-------|-----------|
| 9C | Rewrite wiring | pipeline.py, config.py, query_rewrite_service.py, openai_compatible.py | 12/15 probe success |
| 9G | Support diversity retention | support_selector.py | 3/7 fixed, 0 citation inflation |
| 9I | Negative route | schemas.py, router.py, support_selector.py, pipeline.py | 5/5 smoke150 negative fixed |
| 9K | Frozen eval rewrite cache | query_rewrite_service.py, config.py, pipeline.py | 150/150 frozen, 3 runs identical |
| 9M | Second support fix | support_selector.py | real_P0 9→6 |
| 9O | Used-support citation completion | citation_binder.py | citation_binding 3→0 |
| 9Q | CN comparison parser pattern | branch_parser.py | Parser correct; retrieval gap separate |

Total: +401 lines, -21 lines across 9 src files.

## 4. Dataset / Cache / Manifest Fixes

- **smoke150 canonicalization**: smoke50 + smoke100 → smoke150 (150 unique IDs)
- **smoke200 bad5 replacement**: p21a_added50_014-018 → p21a9v2_fact_001-005
- **rewrite cache**: smoke150 (150) + smoke200 (200), all entries frozen
- **manifest**: smoke200_manifest.json updated with full distribution
- **registry**: registry.json updated with canonical paths for all datasets

## 5. Safety

| Check | Status |
|-------|--------|
| negative_safety | PASS (6/6 smoke200, citation=0) |
| citation_safety | PASS (0 wrong-doc, 0 citation_inflation) |
| rewrite_safety | PASS (200/200 frozen, 0 live) |
| dataset_safety | PASS (200 unique IDs, no conflicts) |

## 6. Final Residuals (6 total, all documented)

| ID | Sample | Class | Deferred |
|----|--------|-------|----------|
| ent_058 | smoke150 | support_selection | Close-margin, generic fix needed |
| ent_083 | smoke150 | branch_query_context | Query enrichment, not parser |
| ent_094 | smoke150 | support_selection | Expected doc score negative |
| p21a9v2_fact_001 | added50_replacement | support_selection | Same as ent_058 pattern |
| p21a9v2_fact_002 | added50_replacement | retrieval_doc_miss | Query-doc matching (P2) |
| p21a9v2_fact_004 | added50_replacement | retrieval_doc_miss | Query-doc matching (P2) |

## 7. RAGAS Exclusion

**RAGAS 结果不纳入 Phase 21A 是否合并的门控。** 当前 RAGAS 暴露的是评估器/指标适配问题（BGE-M3 标度、judge 过严、reference 格式），而非 pipeline 质量问题。RAGAS 相关脚本和报告已归档但标记为 excluded_from_closeout。后续单独 Phase 21B-Eval 调整。

## 8. Backlog Handoff

10 项 backlog 移交：
- BL-001~004: support_selection improvements
- BL-005~006: retrieval query-doc matching
- BL-007: RAGAS metric adaptation
- BL-008: reference answer construction
- BL-009: PDF parser/table/figure cleanup (high priority)
- BL-010: negative abstention detector improvement

## 9. Commit / Merge Recommendation

**recommended_next_step = commit_phase21a**

建议 3 步提交：
1. **Commit 1 (code)**: 9 src files — 7 fixes
2. **Commit 2 (data)**: datasets, rewrite caches, manifest, registry
3. **Commit 3 (tools)**: diagnostic scripts, tests (excluding RAGAS)

所有改动已审查，无 blocking issues。smoke baseline 稳定。RAGAS 排除。ready to merge after PR review。

## 10. Next Step

1. commit Phase 21A (3-commit split)
2. PR + code review
3. merge to main
4. start Phase 21B: parser/table/figure cleanup (BL-009)
5. start Phase 21B-Eval: RAGAS metric adaptation (BL-007, BL-008)
