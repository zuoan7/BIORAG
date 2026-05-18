# Scripts Archive Candidates

This matrix records the PR2 scan of Python scripts that originally lived under `scripts/`.

## Scan Method

- Enumerated scripts with `find scripts -type f -name '*.py' | sort` before quarantine.
- Parsed Python imports with `ast` for `tests/`, `app/`, `src/`, and `scripts/` internal dependencies.
- Searched path/module references in `tests`, `README.md`, `docs`, `app`, `src`, and `configs`.
- User-entry docs were limited to `README.md`, `docs/startup.md`, `docs/README.md`, and `docs/architecture/cleanup_policy.md`; generated inventory docs were excluded from archive blocking.
- After the first move, scanned remaining `scripts/` imports again and archived two additional historical generation eval scripts that depended on archived generation-stage code.

Bulk rg proof command used for archive candidates:

```bash
rg -n '<path>|<module>' tests README.md docs app src configs -g '!docs/architecture/script_inventory.md' -g '!docs/architecture/scripts_*.md'
```

For rows marked `archive_candidate`, the AST/reference scan found no test import/path ref, no current README/docs user-entry ref, no app/src ref, no config ref, and no current accepted-baseline classification.

## Status Counts

| Status | Count |
| --- | ---: |
| `keep` | 54 |
| `protected_by_tests` | 60 |
| `archive_candidate` | 114 |
| `delete_candidate` | 0 |
| `unknown` | 53 |

## Candidate Matrix

| Original script | Status | Tests import/path ref | README/docs user ref | app/src ref | Current accepted baseline | Archive path | Check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `scripts/audit/audit_parsed_clean_quality.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/audit/audit_pdf_extraction_safe.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/data_prep/lookup_pdf_map.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/data_prep/rebuild_pdf_map_clean.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/data_prep/rename_pdfs_and_build_map.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/data_prep/restore_pdf_names_from_map.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/audit_parsed_raw_v4.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/audit_two_column_order_v5.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/audit_two_column_recall_v4.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/check_chunk_structure_contract.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/check_holdout50_quality.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/check_parent_index_contract.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/chunk_retrieval_smoke_v5.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/compare_512_vs_8192.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/debug_parent_expansion.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/diagnose_pdf_reading_order.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/generate_full_ingestion_spotcheck_v5.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/run_parent_expansion_small_smoke.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/run_phase12e_diagnostic_smoke.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/diagnostics/run_phase12e_diagnostic_smoke.py` | bulk-rg + AST scan |
| `scripts/diagnostics/search_small_milvus.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/summarize_full_ingestion_regression_v5.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/test_references_final_cleanup.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/test_references_state_fix.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/validate_chunks_v4.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/validate_evidence_pack_v5.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/validate_parsed_clean_v4.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/diagnostics/validate_parsed_raw_v4.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/audit_phase5_table_content_loss.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5_table_content_loss.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5c1_table_preservation.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5c1_table_preservation.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5c4_full_preflight.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5c4_full_preflight.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5d_false_fragment_captions.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5d_false_fragment_captions.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5e_section_metadata.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5e_section_metadata.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5f4_asset_promotion.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5f4_asset_promotion.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5f4_index_assets.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5f4_index_assets.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_phase5f_eval_quality.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/audit_phase5f_eval_quality.py` | bulk-rg + AST scan |
| `scripts/evaluation/audit_table_figure_retrieval_text.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/__init__.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/aggregate_scores.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/collect_records.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/judge_prompts.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/qwen_judge.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/rule_metrics.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/biorag_eval/schemas.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/build_diagnostics_ledger.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/build_phase4e3_approved_eval_set.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/build_phase4e3_approved_eval_set.py` | bulk-rg + AST scan |
| `scripts/evaluation/build_phase5f_clean_eval_sets.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/build_phase5f_clean_eval_sets.py` | bulk-rg + AST scan |
| `scripts/evaluation/build_ragas_dataset.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/build_round8_diagnostic_sets.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/build_table_retrieval_preview_queries.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/cleanup_phase5f_eval_sets_semantic.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/cleanup_phase5f_eval_sets_semantic.py` | bulk-rg + AST scan |
| `scripts/evaluation/diagnose_p0_failure_layers.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/dryrun_phase5c4_full_table_enhancement.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/dryrun_phase5c4_full_table_enhancement.py` | bulk-rg + AST scan |
| `scripts/evaluation/enhance_p0_diagnostics.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/enhance_phase5f_eval_semantic_quality_v2.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/enhance_phase5f_eval_semantic_quality_v2.py` | bulk-rg + AST scan |
| `scripts/evaluation/evaluate_chunk_evidence_audit.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/evaluate_e2e_small.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/evaluate_existing_hybrid_retrieval.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/evaluate_guarded_reranker.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/evaluate_phase4e0_retrieval_sanity.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/evaluate_phase4e0_retrieval_sanity.py` | bulk-rg + AST scan |
| `scripts/evaluation/evaluate_phase4e3_manual_eval_set.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/evaluate_phase4e3_manual_eval_set.py` | bulk-rg + AST scan |
| `scripts/evaluation/evaluate_phase5c2_table_retrieval_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/evaluate_phase5c2_table_retrieval_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/evaluate_phase5c3_table_retrieval_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/evaluate_phase5c3_table_retrieval_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/evaluate_phase5c5_full_retrieval_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/evaluate_phase5c5_full_retrieval_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/evaluate_ragas.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/evaluate_retrieval.py` | `keep` | no | yes | no | no | `` |  |
| `scripts/evaluation/evaluate_retrieval_only.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/generate_baseline_regression_report.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/generate_p0_reconciliation_diff.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/generate_review_candidates.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/merge_ragas_with_eval_metrics.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/phase4_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/phase4_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/phase5_analyze.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/phase5_analyze.py` | bulk-rg + AST scan |
| `scripts/evaluation/phase6_analyze.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/phase6_analyze.py` | bulk-rg + AST scan |
| `scripts/evaluation/phase7_diagnose_summary_gaps.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7l_evidence_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7l_rollback_check.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7l_sandbox_merge_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7l_sidecar_retriever_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7l_table_rag_smoke_common.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7l_table_unit_adapter_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7m_contract_regression_check.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7m_failure_path_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7m_generation_v2_contract_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7m_policy_matrix_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7q1_build_mapper_fixture.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7q1_table_citation_mapper_dry_run.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7q1_validate_mapper_outputs.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7q_build_table_citation_schema.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7q_validate_citation_schema_examples.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7r_build_table_index_production_proposal.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7r_validate_table_index_production_proposal.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7s_canonical_source_dry_run.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7s_production_readiness_gate_dry_run.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7s_validate_readiness_dry_run.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_build_query_fixture.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_citation_guard_smoke.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_merge_smoke.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_rerank_smoke.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_rollback_smoke.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_shadow_smoke.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7u_validate_preview_eval_smoke.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7v_fast_ab_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7v_fast_build_ab_fixture.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7v_fast_citation_guard_regression.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7v_fast_replay_phase7u_misses.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7v_fast_rollback_regression.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7v_fast_validate.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/phase7w_slim_answer_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7w_slim_build_fixture.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7w_slim_mainchain_evidence_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7w_slim_pipeline_seam_smoke.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7w_slim_validate.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7x_final_answer_acceptance.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7x_final_build_query_set.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7x_final_guardrail_check.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7x_final_mainchain_ab_acceptance.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase7x_final_validate.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/phase8_analyze.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/phase8_analyze.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase4e0_subset.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase4e0_subset.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase4e3_manual_eval_set.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase4e3_manual_eval_set.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase5c2_baseline_chunks.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase5c2_baseline_chunks.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase5c2_table_retrieval_eval.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase5c2_table_retrieval_eval.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase5c3_baseline_chunks.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase5c3_baseline_chunks.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase5c3_table_retrieval_eval.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase5c3_table_retrieval_eval.py` | bulk-rg + AST scan |
| `scripts/evaluation/prepare_phase5c5_full_retrieval_eval.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/prepare_phase5c5_full_retrieval_eval.py` | bulk-rg + AST scan |
| `scripts/evaluation/promote_phase5f_official_baseline.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/promote_phase5f_official_baseline.py` | bulk-rg + AST scan |
| `scripts/evaluation/recompute_phase5c2_stable_target_metrics.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/recompute_phase5c2_stable_target_metrics.py` | bulk-rg + AST scan |
| `scripts/evaluation/refine_phase19b_buckets.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/refine_phase19b_buckets.py` | bulk-rg + AST scan |
| `scripts/evaluation/render_table_retrieval_evidence_cards.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/review_phase4e3_normal_misses.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/review_phase4e3_normal_misses.py` | bulk-rg + AST scan |
| `scripts/evaluation/review_phase5f4_failures.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/review_phase5f4_failures.py` | bulk-rg + AST scan |
| `scripts/evaluation/review_phase5f_normal_eval_quality.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/review_phase5f_normal_eval_quality.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_biorag_eval_calibration_v2.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_biorag_eval_full200.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/run_biorag_eval_pilot.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_biorag_eval_v31_dedup.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_biorag_eval_v3_fix.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_generation_smoke100.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_smoke100.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage1_5_compare.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage1_5_compare.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2b_qwen_synthesis.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2b_qwen_synthesis.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2c1_comparison_hotfix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2c1_comparison_hotfix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2c2_branch_parser.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2c2_branch_parser.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2c3_validator_polish.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2c3_validator_polish.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2c_comparison_coverage.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2c_comparison_coverage.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2d_summary_support.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2d_summary_support.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2e01_neighbor_gate_calibration.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2e01_neighbor_gate_calibration.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_stage2e_neighbor_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_stage2e_neighbor_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_generation_v2_baseline_matrix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_generation_v2_baseline_matrix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase12f_smoke100.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase12f_smoke100.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase12g_holdout50_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase12g_holdout50_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16b_evidence_lifecycle_debug.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16b_evidence_lifecycle_debug.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16c_citation_contract.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16c_citation_contract.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16d_smoke100.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16d_smoke100.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16e_focused_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16e_focused_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16e_marker_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16e_marker_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16f_smoke100_lines6.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16f_smoke100_lines6.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16g_smoke50.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16g_smoke50.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase16h_regression.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase16h_regression.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase17a_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase17a_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase17b_retrieval_trace.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase17b_retrieval_trace.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase17c_hybrid_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase17c_hybrid_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase17d_source_floor.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase17d_source_floor.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase17e_smoke100_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase17e_smoke100_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase17f_regression.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase17f_regression.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18a_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18a_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18b_hard_recall.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18b_hard_recall.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18c_alias_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18c_alias_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18d_deep_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18d_deep_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18e_support_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18e_support_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18f_support_capacity_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18f_support_capacity_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase18g_score_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase18g_score_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19b_cross_lingual_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19b_cross_lingual_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19c_query_rewrite_shadow_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19c_query_rewrite_shadow_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19d_smoke50_sanity.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19d_smoke50_sanity.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19e_metric_cleanup.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19e_metric_cleanup.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19f_metric_cleanup_guardrail.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19f_metric_cleanup_guardrail.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19g_smoke100_shadow_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19g_smoke100_shadow_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19h_eval_taxonomy_flag.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19h_eval_taxonomy_flag.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19i_feature_flag_regression.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19i_feature_flag_regression.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19j_e2e_regression.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19j_e2e_regression.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase19k_shadow_rollout_plan.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase19k_shadow_rollout_plan.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20a_full_eval_reclassify.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20a_full_eval_reclassify.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20c_direction_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20c_direction_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20d_support_citation_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20d_support_citation_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20e_rebaseline.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20e_rebaseline.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20f_summary_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20f_summary_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20g_summary_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20g_summary_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20h_retrieval_strategy_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20h_retrieval_strategy_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20i_doc_sidecar_ab.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20i_doc_sidecar_ab.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20j_decomposition_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20j_decomposition_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20k_comparison_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20k_comparison_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20l1_cn_fallback_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase20l1_cn_fallback_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase20l2_cn_fallback_feature.py` | `keep` | no | yes | no | yes | `` |  |
| `scripts/evaluation/run_phase20m_convergence_summary.py` | `keep` | no | no | no | yes | `` |  |
| `scripts/evaluation/run_phase21a0_layout_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a0_layout_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a1_canonicalize_smoke150.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a1_canonicalize_smoke150.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a2_equivalence_validation.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a2_equivalence_validation.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9_smoke200_rebaseline.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/run_phase21a9b_rewrite_fallback_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9b_rewrite_fallback_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9c_rewrite_wiring_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9c_rewrite_wiring_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9d_remaining_smoke150_regression.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9d_remaining_smoke150_regression.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9e_real_remaining_regression_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9e_real_remaining_regression_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9f_support_citation_targeted_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9f_support_citation_targeted_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9g_support_retention_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9g_support_retention_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9g_v_validation.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9g_v_validation.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9h_negative_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9h_negative_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9i_negative_fix.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9i_negative_fix.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9j_remaining_support_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9j_remaining_support_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9k_frozen_rewrite_cache.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9k_frozen_rewrite_cache.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a9l_frozen_support_audit.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a9l_frozen_support_audit.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase21a_r1b_ragas_slimmed.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/run_phase21a_r2_ragas200_v1.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase21a_r2_ragas200_v1.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_phase5f4_clean_main_baseline.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/run_phase5f4_clean_main_baseline.py` | bulk-rg + AST scan |
| `scripts/evaluation/run_ragas200_fixed.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_ragas_regression.py` | `keep` | no | yes | no | no | `` |  |
| `scripts/evaluation/run_ragas_smoke100.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_smoke100_regression.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/run_table_retrieval_wiring_preview.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/run_validation_suite.py` | `keep` | no | yes | no | no | `` |  |
| `scripts/evaluation/select_phase5c3_representative_docs.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/select_phase5c3_representative_docs.py` | bulk-rg + AST scan |
| `scripts/evaluation/signoff_phase5d_false_fragment_captions.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/signoff_phase5d_false_fragment_captions.py` | bulk-rg + AST scan |
| `scripts/evaluation/signoff_phase5e_section_repair_candidates.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/signoff_phase5e_section_repair_candidates.py` | bulk-rg + AST scan |
| `scripts/evaluation/signoff_table_figure_remaining_risks.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/summarize_phase5c3_index_build.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/summarize_phase5c3_index_build.py` | bulk-rg + AST scan |
| `scripts/evaluation/supplement_phase5f_normal_controls.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/supplement_phase5f_normal_controls.py` | bulk-rg + AST scan |
| `scripts/evaluation/v7_phase6b_flat_vs_table_object_representation.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/v7_phase6b_offline_table_object_coverage.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/v7_phase6c_flat_vs_table_object_representation.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/v7_phase6c_offline_table_object_coverage_check.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/v7_phase6c_offline_table_object_coverage_check_rerun.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/v7_phase6f_flat_vs_table_object_representation.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/v7_phase6f_offline_table_object_coverage_check.py` | `keep` | no | no | no | no | `` |  |
| `scripts/evaluation/validate_enterprise_dataset.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/evaluation/validate_expanded_table_seed_consistency.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/validate_hybrid_extractor_against_gold_seed.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/evaluation/validate_phase5d_caption_cleanup.py` | `archive_candidate` | no | no | no | no | `archive/scripts/phase_artifacts/evaluation/validate_phase5d_caption_cleanup.py` | bulk-rg + AST scan |
| `scripts/evaluation/validate_table_retrieval_wiring_preview.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/extraction/align_chunk_pdfplumber_tables.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/analyze_human_review_label_errors.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/apply_bulk_binding_review_and_build_expanded_seed.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/apply_hybrid_rule_fixes_v2.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/apply_hybrid_source_review_decisions.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/build_expanded_table_review_candidates.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/build_hybrid_table_objects_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/build_phase7j_preview_subset.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/build_seed_drafts_from_review_labels.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/build_table_index_units_preview.py` | `keep` | no | no | no | no | `` |  |
| `scripts/extraction/build_table_index_units_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/construct_hybrid_table_gold_seed.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/export_table_review_crops.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/extract_table_objects_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/extract_tables_pdfplumber_v1.py` | `keep` | no | no | no | no | `` |  |
| `scripts/extraction/freeze_human_table_review_labels.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/qa_table_index_units_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/reconstruct_logical_cells_v2.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/render_expanded_table_review_pack.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/render_hybrid_table_objects_markdown.py` | `keep` | no | no | no | no | `` |  |
| `scripts/extraction/render_table_index_units_review.py` | `keep` | no | no | no | no | `` |  |
| `scripts/extraction/render_table_objects_markdown.py` | `keep` | no | no | no | no | `` |  |
| `scripts/extraction/review_hybrid_binding_candidates.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/run_hybrid_table_extractor_v2.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/validate_hybrid_table_objects_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/validate_table_index_unit_qa_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/validate_table_index_units_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/extraction/validate_table_objects_v1.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/ingestion/audit_cleaning_rules.py` | `keep` | no | no | no | no | `` |  |
| `scripts/ingestion/audit_context_rule_samples.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/ingestion/audit_table_figure_evidence.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/ingestion/build_evidence_pack_v5.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/ingestion/build_parent_index.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/ingestion/build_round1_kb.py` | `keep` | no | yes | no | no | `` |  |
| `scripts/ingestion/clean_parsed_structure.py` | `protected_by_tests` | yes | no | yes | no | `` |  |
| `scripts/ingestion/cleanup_false_fragment_captions.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/ingestion/document_cleaning_v5.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/ingestion/enhance_table_like_paragraphs_pilot.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/ingestion/import_to_milvus.py` | `protected_by_tests` | yes | yes | yes | no | `` |  |
| `scripts/ingestion/pdf_to_structured.py` | `protected_by_tests` | yes | yes | yes | no | `` |  |
| `scripts/ingestion/phase4_shadow_table_figure_parse.py` | `protected_by_tests` | yes | no | no | no | `` |  |
| `scripts/ingestion/preprocess_and_chunk.py` | `protected_by_tests` | yes | yes | yes | no | `` |  |
| `scripts/ingestion/rebuild_docs_by_id.py` | `unknown` | no | no | no | no | `` |  |
| `scripts/ops/interactive_rag_cli.py` | `keep` | no | yes | no | no | `` |  |
