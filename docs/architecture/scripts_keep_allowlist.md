# Scripts Keep Allowlist

Cleanup PR2 uses this allowlist before moving phase artifacts out of `scripts/`.

## Assumptions

- `README.md`, `docs/startup.md`, and `docs/architecture/cleanup_policy.md` are treated as current user-entry documentation.
- Existing architecture inventory references are not treated as user-entry references for archive blocking.
- Phase7 table evidence/citation preview and final-chain scripts stay in `scripts/` for this PR; cleanup must not promote them to production or weaken preview guards.
- `configs/baseline_registry.yaml` and accepted baseline data are not edited.

## Category Counts

| Category | Count | Reason |
| --- | ---: | --- |
| `production_or_ops_entry` | 1 | Current operator-facing entry point; moving it would break documented local API debugging. |
| `current_ingestion_entry` | 8 | Current ingestion/build entry or direct ingestion-chain dependency. |
| `current_eval_entry` | 17 | Current documented or retained evaluation entry; verification does not execute RAGAS/Qwen/retrieval eval. |
| `current_phase7_final_pipeline_entry` | 79 | Phase7 table preview/final-chain or extraction support; retained to keep preview boundary intact. |
| `protected_by_tests` | 60 | Directly imported by tests or referenced by test source paths. |
| `referenced_by_docs` | 9 | Referenced by current README/startup/cleanup docs as a user or policy entry. |
| `referenced_by_app_src` | 4 | Referenced by app/src runtime or current builder code. |
| `current_accepted_baseline_entry` | 2 | Part of the current accepted Phase20 baseline documentation path. |
| `dependency_of_keep` | 2 | Imported by another allowlisted script; retained to avoid breaking current entries. |

## Allowlisted Files

| Script | Allowlist reason |
| --- | --- |
| `scripts/diagnostics/validate_parsed_raw_v4.py` | protected_by_tests |
| `scripts/evaluation/biorag_eval/__init__.py` | current_eval_entry |
| `scripts/evaluation/biorag_eval/aggregate_scores.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/biorag_eval/collect_records.py` | current_eval_entry |
| `scripts/evaluation/biorag_eval/judge_prompts.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/biorag_eval/qwen_judge.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/biorag_eval/rule_metrics.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/biorag_eval/schemas.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/evaluate_e2e_small.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/evaluate_existing_hybrid_retrieval.py` | current_eval_entry |
| `scripts/evaluation/evaluate_guarded_reranker.py` | dependency_of_keep |
| `scripts/evaluation/evaluate_ragas.py` | current_eval_entry, protected_by_tests |
| `scripts/evaluation/evaluate_retrieval.py` | current_eval_entry, referenced_by_docs |
| `scripts/evaluation/phase7_diagnose_summary_gaps.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7l_evidence_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7l_rollback_check.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7l_sandbox_merge_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7l_sidecar_retriever_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7l_table_rag_smoke_common.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7l_table_unit_adapter_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7m_contract_regression_check.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7m_failure_path_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7m_generation_v2_contract_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7m_policy_matrix_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7q1_build_mapper_fixture.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7q1_table_citation_mapper_dry_run.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7q1_validate_mapper_outputs.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7q_build_table_citation_schema.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7q_validate_citation_schema_examples.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7r_build_table_index_production_proposal.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7r_validate_table_index_production_proposal.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7s_canonical_source_dry_run.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7s_production_readiness_gate_dry_run.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7s_validate_readiness_dry_run.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_build_query_fixture.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_citation_guard_smoke.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_merge_smoke.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_rerank_smoke.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_rollback_smoke.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_shadow_smoke.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7u_validate_preview_eval_smoke.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7v_fast_ab_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7v_fast_build_ab_fixture.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7v_fast_citation_guard_regression.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7v_fast_replay_phase7u_misses.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7v_fast_rollback_regression.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7v_fast_validate.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/evaluation/phase7w_slim_answer_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7w_slim_build_fixture.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7w_slim_mainchain_evidence_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7w_slim_pipeline_seam_smoke.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7w_slim_validate.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7x_final_answer_acceptance.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7x_final_build_query_set.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7x_final_guardrail_check.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7x_final_mainchain_ab_acceptance.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/phase7x_final_validate.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/render_table_retrieval_evidence_cards.py` | protected_by_tests |
| `scripts/evaluation/run_biorag_eval_full200.py` | current_eval_entry |
| `scripts/evaluation/run_phase20l2_cn_fallback_feature.py` | current_accepted_baseline_entry, current_eval_entry, referenced_by_docs |
| `scripts/evaluation/run_phase20m_convergence_summary.py` | current_accepted_baseline_entry, current_eval_entry |
| `scripts/evaluation/run_phase21a9_smoke200_rebaseline.py` | current_eval_entry |
| `scripts/evaluation/run_phase21a_r1b_ragas_slimmed.py` | protected_by_tests |
| `scripts/evaluation/run_ragas_regression.py` | current_eval_entry, referenced_by_docs |
| `scripts/evaluation/run_table_retrieval_wiring_preview.py` | protected_by_tests |
| `scripts/evaluation/run_validation_suite.py` | current_eval_entry, referenced_by_docs |
| `scripts/evaluation/v7_phase6b_flat_vs_table_object_representation.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/v7_phase6b_offline_table_object_coverage.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/v7_phase6c_flat_vs_table_object_representation.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/v7_phase6c_offline_table_object_coverage_check.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/v7_phase6c_offline_table_object_coverage_check_rerun.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/v7_phase6f_flat_vs_table_object_representation.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/v7_phase6f_offline_table_object_coverage_check.py` | current_phase7_final_pipeline_entry |
| `scripts/evaluation/validate_expanded_table_seed_consistency.py` | protected_by_tests |
| `scripts/evaluation/validate_hybrid_extractor_against_gold_seed.py` | protected_by_tests |
| `scripts/extraction/align_chunk_pdfplumber_tables.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/analyze_human_review_label_errors.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/apply_bulk_binding_review_and_build_expanded_seed.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/apply_hybrid_rule_fixes_v2.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/apply_hybrid_source_review_decisions.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/build_expanded_table_review_candidates.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/build_hybrid_table_objects_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/build_phase7j_preview_subset.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/build_seed_drafts_from_review_labels.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/build_table_index_units_preview.py` | current_phase7_final_pipeline_entry |
| `scripts/extraction/build_table_index_units_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/construct_hybrid_table_gold_seed.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/export_table_review_crops.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/extract_table_objects_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/extract_tables_pdfplumber_v1.py` | current_phase7_final_pipeline_entry |
| `scripts/extraction/freeze_human_table_review_labels.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/qa_table_index_units_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/reconstruct_logical_cells_v2.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/render_expanded_table_review_pack.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/render_hybrid_table_objects_markdown.py` | current_phase7_final_pipeline_entry |
| `scripts/extraction/render_table_index_units_review.py` | current_phase7_final_pipeline_entry |
| `scripts/extraction/render_table_objects_markdown.py` | current_phase7_final_pipeline_entry |
| `scripts/extraction/review_hybrid_binding_candidates.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/run_hybrid_table_extractor_v2.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/validate_hybrid_table_objects_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/validate_table_index_unit_qa_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/validate_table_index_units_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/extraction/validate_table_objects_v1.py` | current_phase7_final_pipeline_entry, protected_by_tests |
| `scripts/ingestion/audit_cleaning_rules.py` | dependency_of_keep |
| `scripts/ingestion/build_parent_index.py` | current_ingestion_entry, protected_by_tests |
| `scripts/ingestion/build_round1_kb.py` | current_ingestion_entry, referenced_by_docs |
| `scripts/ingestion/clean_parsed_structure.py` | current_ingestion_entry, protected_by_tests, referenced_by_app_src |
| `scripts/ingestion/cleanup_false_fragment_captions.py` | current_ingestion_entry, protected_by_tests |
| `scripts/ingestion/document_cleaning_v5.py` | current_ingestion_entry, protected_by_tests |
| `scripts/ingestion/import_to_milvus.py` | current_ingestion_entry, protected_by_tests, referenced_by_app_src, referenced_by_docs |
| `scripts/ingestion/pdf_to_structured.py` | current_ingestion_entry, protected_by_tests, referenced_by_app_src, referenced_by_docs |
| `scripts/ingestion/phase4_shadow_table_figure_parse.py` | protected_by_tests |
| `scripts/ingestion/preprocess_and_chunk.py` | current_ingestion_entry, protected_by_tests, referenced_by_app_src, referenced_by_docs |
| `scripts/ops/interactive_rag_cli.py` | production_or_ops_entry, referenced_by_docs |

## Retained But Not Allowlisted

Files marked `unknown` in `scripts_archive_candidates.md` remain in `scripts/` because this PR did not have enough proof to archive them. They are not current-entry allowlist items and should be rechecked in a later cleanup round.
