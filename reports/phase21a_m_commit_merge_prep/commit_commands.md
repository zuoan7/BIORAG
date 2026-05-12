# Phase 21A-M Commit Commands

以下命令仅供建议，不自动执行。请在确认后手动运行。

## Pre-commit checks

```bash
# 确认当前分支和状态
git status
git branch --show-current

# 确认没有意外 staged 文件
git diff --cached --stat

# 确认 diff 只包含预期的 9 个 src 文件
git diff --name-only
```

## Commit 1: Code fixes

```bash
# 添加 9 个 src 文件
git add \
  src/synbio_rag/application/generation_v2/branch_parser.py \
  src/synbio_rag/application/generation_v2/citation_binder.py \
  src/synbio_rag/application/generation_v2/support_selector.py \
  src/synbio_rag/application/pipeline.py \
  src/synbio_rag/domain/config.py \
  src/synbio_rag/domain/router.py \
  src/synbio_rag/domain/schemas.py \
  src/synbio_rag/infrastructure/clients/openai_compatible.py \
  src/synbio_rag/rewrite/query_rewrite_service.py

# 添加 3 个支持测试
git add \
  tests/test_phase21a9c_query_rewrite_wiring.py \
  tests/test_phase21a9g_support_retention.py \
  tests/test_phase21a9i_negative_abstention.py

# 提交
git commit -m "$(cat <<'EOF'
Phase 21A: stabilize RAG pipeline and smoke support/citation behavior

7 fixes across 9 src files (+401/-21 lines):

- 9C: Fix rewrite wiring — llm_client=None caused 100% fallback
  (pipeline.py, config.py, query_rewrite_service.py, openai_compatible.py)

- 9G: Support diversity retention — same-doc duplicate swap
  prevents overcrowding; no sample/doc IDs (support_selector.py)

- 9I: Negative route classification — explicit abstain clause
  routes to negative intent → zero citations emitted
  (schemas.py, router.py, pipeline.py, support_selector.py)

- 9K: Frozen eval rewrite cache — pre-computed JSONL cache
  with fail-fast guards ensures 100% reproducible eval
  (query_rewrite_service.py, config.py, pipeline.py)

- 9M: Second support fix — scored_median bypass +
  close-margin distinct-doc capacity+1 (support_selector.py)

- 9O: Used-support citation completion — citation-eligible
  selected_support items get cited even without markers
  (citation_binder.py)

- 9Q: CN comparison parser pattern — zh_compare_a_and_b_as_x
  for A和B作为X syntax (branch_parser.py)

Validation:
- smoke150 real_P0=3, smoke200 real_P0=6/200
- doc_hit_rate=1.000, doc_miss=0
- frozen_rewrite=200/200, live=0, fallback=0
- 3 runs identical, 0 regression
EOF
)"
```

## Commit 2: Eval datasets / caches / manifest

```bash
# 强制添加 gitignored data/eval/ 文件
git add -f \
  data/eval/datasets/smoke150.jsonl \
  data/eval/datasets/smoke200.jsonl \
  data/eval/rewrite_cache/smoke150_rewrites.jsonl \
  data/eval/rewrite_cache/smoke200_rewrites.jsonl \
  data/eval/manifests/smoke150_manifest.json \
  data/eval/manifests/smoke200_manifest.json \
  data/eval/registry.json

# 提交
git commit -m "$(cat <<'EOF'
Phase 21A: canonicalize smoke datasets and freeze rewrite caches

- smoke150.jsonl: smoke50 + smoke100 canonicalized (150 unique IDs)
- smoke200.jsonl: bad5 (p21a_added50_014-018) replaced with reviewed
  factoid replacements (p21a9v2_fact_001-005)
- smoke150_rewrites.jsonl: 150 frozen rewrite entries
- smoke200_rewrites.jsonl: 200 frozen rewrite entries
- smoke150_manifest.json, smoke200_manifest.json: distribution,
  route/category counts, component sources
- registry.json: canonical paths for all eval datasets

Note: git add -f required (data/ in .gitignore).
EOF
)"
```

## Commit 3: Diagnostics / reports

```bash
# 添加 Phase 21A 诊断脚本
git add \
  scripts/evaluation/run_phase21a0_layout_audit.py \
  scripts/evaluation/run_phase21a1_canonicalize_smoke150.py \
  scripts/evaluation/run_phase21a2_equivalence_validation.py \
  scripts/evaluation/run_phase21a9_smoke200_rebaseline.py \
  scripts/evaluation/run_phase21a9b_rewrite_fallback_audit.py \
  scripts/evaluation/run_phase21a9c_rewrite_wiring_fix.py \
  scripts/evaluation/run_phase21a9d_remaining_smoke150_regression.py \
  scripts/evaluation/run_phase21a9e_real_remaining_regression_audit.py \
  scripts/evaluation/run_phase21a9f_support_citation_targeted_audit.py \
  scripts/evaluation/run_phase21a9g_support_retention_fix.py \
  scripts/evaluation/run_phase21a9g_v_validation.py \
  scripts/evaluation/run_phase21a9h_negative_audit.py \
  scripts/evaluation/run_phase21a9i_negative_fix.py \
  scripts/evaluation/run_phase21a9j_remaining_support_audit.py \
  scripts/evaluation/run_phase21a9k_frozen_rewrite_cache.py \
  scripts/evaluation/run_phase21a9l_frozen_support_audit.py

# 添加强制 gitignored results/reports 中的关键收敛文件
git add -f \
  results/phase21a9y_smoke200_convergence_summary/run_config.json \
  results/phase21a9y_smoke200_convergence_summary/final_smoke150_baseline.json \
  results/phase21a9y_smoke200_convergence_summary/final_smoke200_baseline.json \
  results/phase21a9y_smoke200_convergence_summary/final_residual_ledger.csv \
  results/phase21a9y_smoke200_convergence_summary/code_fix_ledger.csv \
  results/phase21a9y_smoke200_convergence_summary/dataset_fix_ledger.csv \
  results/phase21a9y_smoke200_convergence_summary/safety_summary.json \
  results/phase21a9y_smoke200_convergence_summary/known_backlog.csv \
  results/phase21a9y_smoke200_convergence_summary/merge_readiness_assessment.json \
  results/phase21a_z_final_closeout_without_ragas/final_smoke_baseline_acceptance.json \
  results/phase21a_z_final_closeout_without_ragas/final_backlog_handoff.csv \
  results/phase21a_z_final_closeout_without_ragas/ragas_exclusion_note.json \
  results/phase21a_z_final_closeout_without_ragas/final_code_change_summary.csv \
  reports/phase21a9y_smoke200_convergence_summary/summary.md \
  reports/phase21a_z_final_closeout_without_ragas/summary.md

# 提交
git commit -m "$(cat <<'EOF'
Phase 21A: add smoke diagnostics and convergence reports

- 16 Phase 21A diagnostic/validation scripts
  (rewrite audit, support audit, negative audit, smoke200 rebaseline)
- smoke200 convergence summary: run_config, baselines, residual ledger,
  code/dataset fix ledgers, safety summary, known backlog
- Phase 21A-Z final closeout: baseline acceptance, backlog handoff,
  RAGAS exclusion note, code change summary
- Convergence reports (phase21a9y, phase21a_z)

Note: git add -f required for results/ and reports/ (in .gitignore).
EOF
)"
```

## Post-commit checks

```bash
# 验证 3 个 commit
git log --oneline -3

# 确认 RAGAS 文件未被提交
git status

# RAGAS 文件应仍显示为 untracked
```

## RAGAS 文件处理

以下 RAGAS 文件保持 untracked，不提交：

- `scripts/evaluation/run_ragas200_fixed.py`
- `scripts/evaluation/run_phase21a_r1b_ragas_slimmed.py`
- `scripts/evaluation/run_phase21a_r2_ragas200_v1.py`
- `tests/test_phase21a_r1b_ragas_input_slimming.py`
- `results/phase21a_r1b_ragas_input_slimming/`
- `results/phase21a_r2_ragas200_v1_baseline/`
- `results/phase21a_r2a_ragas_manual_calibration/`
- `reports/phase21a_r1b_ragas_input_slimming/`
- `reports/phase21a_r2_ragas200_v1_baseline/`
- `reports/phase21a_r2a_ragas_manual_calibration/`

如需保留 RAGAS 审计结果，可在 Phase 21B-Eval 时单独处理。

## PR 创建后

```bash
# 如果需要创建 PR（假设使用 GitHub）
gh pr create \
  --title "Phase 21A: Smoke200 Frozen Baseline Convergence" \
  --body "$(cat reports/phase21a_m_commit_merge_prep/pr_description.md)"
```
