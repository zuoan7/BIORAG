# Results And Reports Artifact Inventory

This inventory records the PR6 tracked-artifact audit for results, reports, and
constructed datasets. It is descriptive only: PR6 does not delete, move, or
promote any artifact.

## Retention Policy

PR6 uses this retention rule:

- Keep staged baselines and baseline acceptance records.
- Keep major phase closeout, guardrail, and summary reports.
- Keep constructed datasets, manifests, registries, and preview fixtures.
- Keep Phase7 table preview/prototype artifacts that tests, README, or retained
  scripts still use.

Files without an exact current path reference may still be protected if they are
part of one of these retained artifact packages.

## Audit Scope

Tracked files audited:

```bash
git ls-files reports results data/eval data/evaluation data/experiments | sort
```

Summary:

| Scope | Tracked files | Bytes | Disposition | Reason |
| --- | ---: | ---: | --- | --- |
| `data/eval` | 5 | 596545 | `protected_constructed_dataset` | Current smoke150/smoke200 datasets, manifests, and registry. |
| `data/evaluation` | 2 | 49684 | `protected_constructed_dataset` | Legacy constructed evaluation fixtures retained as datasets. |
| `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run` | 3 | 26510 | `protected_phase7_preview` | Phase7Q-1 mapper input fixture and manifest used by tests/scripts. |
| `data/experiments/v7_phase7_table_citation_schema_prototype` | 4 | 22089 | `protected_phase7_preview` | Phase7Q citation schema examples and matrices used by tests/scripts. |
| `reports/phase5d_closeout` | 6 | 19469 | `protected_major_summary_report` | Major phase closeout report package. |
| `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit` | 4 | 18548 | `protected_major_summary_report` | Major Phase7 table/figure audit and guardrail package. |
| `reports/v7_phase7_table_citation_binder_prototype_dry_run` | 8 | 13314 | `protected_phase7_preview` | Phase7Q-1 mapper report package, linked from README and used by tests/scripts. |
| `reports/v7_phase7_table_citation_schema_prototype` | 8 | 18341 | `protected_phase7_preview` | Phase7Q schema report package, linked from README and used by tests/scripts. |
| `results/phase21a9y_smoke200_convergence_summary` | 12 | 16567 | `protected_stage_baseline` | Phase21A smoke200 convergence baseline and ledger package. |
| `results/phase21a_m_commit_merge_prep` | 5 | 19881 | `protected_major_summary_report` | Phase21A merge-prep closeout and RAGAS file-handling decision package. |
| `results/phase21a_z_final_closeout_without_ragas` | 8 | 15680 | `protected_stage_baseline` | Final closeout without RAGAS, including baseline acceptance and next-step decision. |
| `results/v7_phase7_table_citation_binder_prototype_dry_run` | 4 | 15797 | `protected_phase7_preview` | Phase7Q-1 mapper dry-run outputs used by focused tests/scripts. |
| `results/v7_phase7_table_citation_schema_prototype` | 1 | 1579 | `protected_phase7_preview` | Phase7Q schema validation output used by focused tests/scripts. |

Total tracked files in scope: 70.
Total tracked bytes in scope: 834004.
`du -ch` reported 980K on disk.

## Ignored But Protected Official Baseline

`reports/phase5f_eval_semantic_enhancement_v2/` is not part of the tracked
artifact matrix because `reports/` is ignored by `.gitignore` and the files in
this directory are not tracked by git in the current checkout.

It is still explicitly protected by the PR6 retention policy:

- `reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl`
  is the official clean baseline dataset path in
  `configs/baseline_registry.yaml`.
- `configs/baseline_registry.yaml` pins the dataset SHA256:
  `39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3`.
- `reports/phase5f_eval_semantic_enhancement_v2/summary.md` and
  `reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2_summary.md`
  are the summary reports for that official denominator.
- Retained Phase6/Phase7 reports and scripts repeatedly treat
  `strict_main_eval_set_v2.jsonl` as official dataset input and guard against
  modifying it.

Disposition: keep in place if present locally; do not delete, archive, rewrite,
or add a replacement in cleanup. If this official dataset must become tracked in
a future PR, do that as an explicit baseline-data PR, not as cleanup.

## Reference Proof

Reference checks searched exact paths in:

```bash
README.md docs tests configs scripts app src
```

Generated cleanup inventory docs were excluded as blockers.

Direct current references found:

- `README.md` links to:
  - `reports/v7_phase7_table_citation_schema_prototype/phase7q_summary.md`
  - `reports/v7_phase7_table_citation_binder_prototype_dry_run/phase7q1_summary.md`
- `scripts/evaluation/run_phase21a9_smoke200_rebaseline.py` reads:
  - `data/eval/datasets/smoke200.jsonl`
  - `data/eval/datasets/smoke150.jsonl`
  - `data/eval/manifests/smoke200_manifest.json`
  - `data/eval/registry.json`
- `scripts/evaluation/run_phase21a_r1b_ragas_slimmed.py`,
  `scripts/evaluation/run_biorag_eval_full200.py`, and
  `scripts/evaluation/biorag_eval/collect_records.py` read
  `data/eval/datasets/smoke150.jsonl` or `data/eval/datasets/smoke200.jsonl`.
- `tests/test_phase7q_table_citation_schema_prototype.py` reads
  `data/experiments/v7_phase7_table_citation_schema_prototype` and
  `results/v7_phase7_table_citation_schema_prototype`.
- `tests/test_phase7q1_table_citation_mapper_dry_run.py` reads
  `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run`,
  `results/v7_phase7_table_citation_binder_prototype_dry_run`, and
  `reports/v7_phase7_table_citation_binder_prototype_dry_run`.
- `scripts/evaluation/phase7q_build_table_citation_schema.py`,
  `scripts/evaluation/phase7q_validate_citation_schema_examples.py`, and
  `scripts/evaluation/phase7s_canonical_source_dry_run.py` reference Phase7Q
  schema data, result, and report paths.
- `scripts/evaluation/phase7q1_build_mapper_fixture.py`,
  `scripts/evaluation/phase7q1_table_citation_mapper_dry_run.py`, and
  `scripts/evaluation/phase7q1_validate_mapper_outputs.py` reference Phase7Q-1
  mapper data, result, and report paths.

No exact path references were found for these retained packages:

- `reports/phase5d_closeout`
- `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit`
- `results/phase21a9y_smoke200_convergence_summary`
- `results/phase21a_m_commit_merge_prep`
- `results/phase21a_z_final_closeout_without_ragas`

They remain protected by the PR6 retention policy because they are major phase
closeout or staged baseline packages, not disposable generated scratch output.

## Candidate Matrix

| Artifact package | Files | Status | Archive candidate? | Notes |
| --- | ---: | --- | --- | --- |
| `data/eval` | 5 | `protected_constructed_dataset` | no | Current smoke150/smoke200 datasets and registry. |
| `data/evaluation` | 2 | `protected_constructed_dataset` | no | Historical constructed evaluation fixtures retained as datasets. |
| `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run` | 3 | `protected_phase7_preview` | no | Phase7Q-1 mapper fixture package. |
| `data/experiments/v7_phase7_table_citation_schema_prototype` | 4 | `protected_phase7_preview` | no | Phase7Q schema fixture package. |
| `reports/phase5d_closeout` | 6 | `protected_major_summary_report` | no | Major phase summary and usage package. |
| `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit` | 4 | `protected_major_summary_report` | no | Phase7 table/figure audit guardrail package. |
| `reports/v7_phase7_table_citation_binder_prototype_dry_run` | 8 | `protected_phase7_preview` | no | README-linked and test/script-backed preview report package. |
| `reports/v7_phase7_table_citation_schema_prototype` | 8 | `protected_phase7_preview` | no | README-linked and test/script-backed preview report package. |
| `results/phase21a9y_smoke200_convergence_summary` | 12 | `protected_stage_baseline` | no | Baseline, ledger, safety, and next-step package. |
| `results/phase21a_m_commit_merge_prep` | 5 | `protected_major_summary_report` | no | Merge-prep and RAGAS file-handling decision package. |
| `results/phase21a_z_final_closeout_without_ragas` | 8 | `protected_stage_baseline` | no | Final closeout and baseline acceptance package. |
| `results/v7_phase7_table_citation_binder_prototype_dry_run` | 4 | `protected_phase7_preview` | no | Mapper dry-run output package. |
| `results/v7_phase7_table_citation_schema_prototype` | 1 | `protected_phase7_preview` | no | Schema validation output. |

PR6 archive candidates: 0.
PR6 unknown tracked artifact packages: 0.

## File Listing

### `data/eval`

- `data/eval/datasets/smoke150.jsonl`
- `data/eval/datasets/smoke200.jsonl`
- `data/eval/manifests/smoke150_manifest.json`
- `data/eval/manifests/smoke200_manifest.json`
- `data/eval/registry.json`

### `data/evaluation`

- `data/evaluation/smoke150_manifest.json`
- `data/evaluation/smoke50_parent_expansion_v1.jsonl`

### `data/experiments`

- `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run/input_artifact_manifest.csv`
- `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run/mapper_input_fixture.jsonl`
- `data/experiments/v7_phase7_table_citation_binder_prototype_dry_run/mapper_input_fixture_summary.csv`
- `data/experiments/v7_phase7_table_citation_schema_prototype/citation_guard_delta_matrix.csv`
- `data/experiments/v7_phase7_table_citation_schema_prototype/citation_mapping_matrix.csv`
- `data/experiments/v7_phase7_table_citation_schema_prototype/citation_prototype_examples.jsonl`
- `data/experiments/v7_phase7_table_citation_schema_prototype/table_evidence_citation_schema.json`

### `reports`

- `reports/phase5d_closeout/backlog.md`
- `reports/phase5d_closeout/command_cheatsheet.md`
- `reports/phase5d_closeout/merge_decision.md`
- `reports/phase5d_closeout/metrics_table.md`
- `reports/phase5d_closeout/summary.md`
- `reports/phase5d_closeout/usage.md`
- `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit/audit_plan.md`
- `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit/baseline_guardrail_audit.md`
- `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit/classification_rubric.md`
- `reports/v7_phase6a_structured_table_figure_ocr_vlm_audit/source_inventory.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/input_artifact_manifest.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/mapper_contract.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/mapper_dry_run_report.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/mapper_input_fixture_report.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/mapper_validation_report.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/phase7q1_guardrail.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/phase7q1_summary.md`
- `reports/v7_phase7_table_citation_binder_prototype_dry_run/phase7q_to_q1_delta.md`
- `reports/v7_phase7_table_citation_schema_prototype/citation_guard_delta.md`
- `reports/v7_phase7_table_citation_schema_prototype/citation_mapping_from_retrieved_chunk.md`
- `reports/v7_phase7_table_citation_schema_prototype/citation_prototype_examples.md`
- `reports/v7_phase7_table_citation_schema_prototype/current_citation_gap_analysis.md`
- `reports/v7_phase7_table_citation_schema_prototype/phase7q_guardrail.md`
- `reports/v7_phase7_table_citation_schema_prototype/phase7q_summary.md`
- `reports/v7_phase7_table_citation_schema_prototype/schema_validation_report.md`
- `reports/v7_phase7_table_citation_schema_prototype/table_evidence_citation_schema.md`

### `results`

- `results/phase21a9y_smoke200_convergence_summary/artifact_index.json`
- `results/phase21a9y_smoke200_convergence_summary/change_inventory.json`
- `results/phase21a9y_smoke200_convergence_summary/code_fix_ledger.csv`
- `results/phase21a9y_smoke200_convergence_summary/dataset_fix_ledger.csv`
- `results/phase21a9y_smoke200_convergence_summary/final_residual_ledger.csv`
- `results/phase21a9y_smoke200_convergence_summary/final_smoke150_baseline.json`
- `results/phase21a9y_smoke200_convergence_summary/final_smoke200_baseline.json`
- `results/phase21a9y_smoke200_convergence_summary/known_backlog.csv`
- `results/phase21a9y_smoke200_convergence_summary/merge_readiness_assessment.json`
- `results/phase21a9y_smoke200_convergence_summary/phase21a_final_next_step_decision.json`
- `results/phase21a9y_smoke200_convergence_summary/run_config.json`
- `results/phase21a9y_smoke200_convergence_summary/safety_summary.json`
- `results/phase21a_m_commit_merge_prep/commit_split_plan.json`
- `results/phase21a_m_commit_merge_prep/final_git_diff_review.json`
- `results/phase21a_m_commit_merge_prep/merge_checklist.json`
- `results/phase21a_m_commit_merge_prep/ragas_file_handling_decision.json`
- `results/phase21a_m_commit_merge_prep/run_config.json`
- `results/phase21a_z_final_closeout_without_ragas/commit_readiness_assessment.json`
- `results/phase21a_z_final_closeout_without_ragas/final_backlog_handoff.csv`
- `results/phase21a_z_final_closeout_without_ragas/final_code_change_summary.csv`
- `results/phase21a_z_final_closeout_without_ragas/final_next_step_decision.json`
- `results/phase21a_z_final_closeout_without_ragas/final_smoke_baseline_acceptance.json`
- `results/phase21a_z_final_closeout_without_ragas/git_status_diff_audit.json`
- `results/phase21a_z_final_closeout_without_ragas/ragas_exclusion_note.json`
- `results/phase21a_z_final_closeout_without_ragas/run_config.json`
- `results/v7_phase7_table_citation_binder_prototype_dry_run/mapped_table_evidence_citations.jsonl`
- `results/v7_phase7_table_citation_binder_prototype_dry_run/mapper_blocked_records.jsonl`
- `results/v7_phase7_table_citation_binder_prototype_dry_run/mapper_dry_run_results.csv`
- `results/v7_phase7_table_citation_binder_prototype_dry_run/mapper_validation_results.csv`
- `results/v7_phase7_table_citation_schema_prototype/schema_validation_results.csv`

## PR7 Recommendation

Do not quarantine tracked results/reports artifacts in PR7 from this audited
set. Under the PR6 retention policy, the tracked set is small and consists of
baselines, closeout reports, constructed datasets, or Phase7 preview fixtures.

A future cleanup can separately inspect untracked or ignored local outputs, but
that should be local workspace hygiene unless those files are intentionally
tracked by git.

## Phase7 Recheck After Runtime Cleanup

After the Phase0-6 runtime/config cleanup checkpoint
`7d751a5 refactor: stabilize rag runtime cleanup through phase6`, the
artifact inventory was rechecked without moving, deleting, promoting, or
tracking any generated artifact.

Tracked artifact scope remains unchanged:

- `git ls-files reports results data/eval data/evaluation data/experiments`
  reports 70 tracked files.
- The tracked package distribution still matches the PR6 matrix above.
- `git status --short -- scripts reports results data/eval data/evaluation
  data/experiments archive` was clean before this documentation update.

Local generated-output scope remains intentionally broader than the tracked
audit:

- `find results reports artifacts -maxdepth 3 -type f` reports 1600 local files
  in this checkout.
- Those local files include ignored generated phase packages, official baseline
  material, preview fixtures, and local reports already classified by
  `docs/architecture/pr7_local_artifact_inventory.md`.

Disposition for this Phase7 pass: keep tracked and ignored artifacts in place.
No artifact package has enough proof for deletion or archive movement under
`docs/architecture/cleanup_policy.md`.

Verification for this recheck:

- `pytest tests/test_phase7*.py -q`: passed, 278 tests.
- `pytest --collect-only -q`: passed, 1047 tests collected.
- `git diff --check`: passed.
- `git status --short -- src config scripts reports results data/eval
  data/evaluation data/experiments archive`: clean.

## Verification

Commands run for PR6 audit:

```bash
git diff --check
pytest --collect-only -q
```

Results:

- `git diff --check`: passed.
- `pytest --collect-only -q`: passed, 1042 tests collected.
- No RAGAS, Qwen, embedding, rerank, retrieval evaluation, or index build was
  run.
