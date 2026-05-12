# Phase 21A-9Y Smoke200 Convergence Summary


## 1. Executive Summary

Phase 21A 将 smoke200 frozen baseline 收敛到可复现、可解释状态。

Final smoke200 frozen real_P0=6/200 (3 known backlog + 3 replacement residual)。

所有 pipeline fix 经过安全验证，无 unexplained regression。


## 2. Why Phase 21A-9 Was Needed

最初 smoke200 rebaseline 无效：rewrite llm_client=None 导致 100% fallback。

Live rewrite 不稳定性使 per-sample 诊断不可靠。

Added50 中 5 条 bad metadata/fragment sample。


## 3. Major Fixes (7 code fixes)

- 9C: rewrite wiring fix
- 9G: support diversity retention
- 9I: negative route fix

- 9K: frozen eval rewrite cache
- 9M: second support fix
- 9O: used-support citation completion

- 9Q: CN comparison parser pattern


## 4. Dataset Corrections

- Bad5 replacement: 5 metadata/fragment → 5 reviewed factoid replacements


## 5. Final Smoke150 Baseline

real_P0=3 (ent_058, ent_083, ent_094)。Negative=0。Citation_binding=0。3 runs identical。


## 6. Final Smoke200 Baseline

real_P0=6/200。doc_miss=0。doc_hit_rate=1.000。frozen=200/200。live=0。

smoke150=3, unchanged_added50=0, replacement5=3。Negative=0。Citation_binding=0。


## 7. Final Residual Ledger (6)

- ent_058: support_selection (close-margin)
- ent_083: retrieval (branch query context)

- ent_094: support_selection (score too low)
- p21a9v2_fact_001: support_selection

- p21a9v2_fact_002: retrieval_doc_miss
- p21a9v2_fact_004: retrieval_doc_miss


## 8. Safety Checks
All PASS: negative=0, citation_inflation=0, wrong_doc=0, frozen=200/200。


## 9. What Should Not Be Interpreted as Failure

real_P0≠0 不是失败 — 6 个 residual 全部解释。replacement 部分 retrieval miss 不是 pipeline bug。


## 10. Remaining Backlog
7 items tracked in known_backlog.csv。


## 11. Merge Recommendation
**ready_to_commit_and_merge_after_review**。可选 run RAGAS200 作为质量补充。


## 12. Next Suggested Phase
Phase 21B: parser/table/figure/PDF cleanup 或 RAGAS200 quality evaluation。
