# v7-phase6A Audit Plan

Generated at: 2026-05-15

## Scope

v7-phase6A is the first audit stage after v7-phase5 closeout. The purpose is to
decide whether structured table extraction, figure-image handling, OCR, VLM,
object schemas, object-relation indexing, and finer-grained eval are justified
for later phases.

This plan uses `v7` as the documentation-governance prefix. It is not the same
as a future project "Phase 7".

## Stage Position

Stage name:

`v7-phase6A Structured Table / Figure-Image / OCR-VLM Feasibility Audit`

Stage type:

read-only audit and feasibility planning.

Primary question:

Which table and figure failures are already solvable by existing text chunks,
which are retrieval gaps, which indicate parser loss, and which require
structured table extraction, OCR, VLM, or human PDF review?

## Hard Guardrails

v7-phase6A must not:

- modify code;
- run tests;
- rebuild chunks, BM25, Milvus, or any index;
- call Qwen, RAGAS, OCR, VLM, or embedding/rerank services;
- write back to production `parsed_clean`;
- modify `strict_main_eval_set_v2`;
- modify `phase5f_official_clean_baseline`;
- treat `current_default` / `synbio_papers` as the official clean baseline;
- enable Phase 5C table enhancement by default;
- enable Phase 5D caption cleanup by default;
- claim structured table extraction, OCR, VLM, `table_object`, or
  `figure_object` are implemented.

## Official Baseline Pins

The audit must preserve the current official clean baseline:

| Item | Value |
|---|---|
| official dataset | `reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl` |
| dataset SHA256 | `39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3` |
| official baseline | `phase5f_official_clean_baseline` |
| chunks | `15,802` |
| BM25 records | `15,802` |
| Milvus collection | `synbio_phase5f_official_clean_baseline` |
| vector dimension | `1024` |
| doc_hit@10 | `95.6%` |
| stable_block_hit@10 | `95.6%` |
| stable_block_hit@20 | `95.6%` |

## Phase6A Substage Plan

| Substage | Purpose | Outputs |
|---|---|---|
| v7-phase6A-0 | Preflight: freeze audit scope, baseline pins, and classification rubric | `audit_plan.md`, `source_inventory.md`, `classification_rubric.md` |
| v7-phase6A-1 | Baseline Guardrail Audit | `baseline_guardrail_audit.md` |
| v7-phase6A-2 | Table Coverage Audit | `table_coverage_audit.md`, `table_sample_classification.csv` |
| v7-phase6A-3 | Figure Coverage Audit | `figure_coverage_audit.md`, `figure_sample_classification.csv` |
| v7-phase6A-4 | OCR/VLM Feasibility | `ocr_vlm_need_matrix.md` |
| v7-phase6A-5 | Schema / Eval / Object-Relation Candidate Design | `schema_eval_object_relation_candidate_design.md` |
| v7-phase6A-6 | Closeout Gate | `phase6a_closeout.md`, `next_phase_decision.md` |

## Audit Inputs

Required inputs for Phase6A-0/1:

- `configs/baseline_registry.yaml`
- `data/baselines/phase5f_official_clean_baseline/manifest.json`
- `data/baselines/phase5f_official_clean_baseline/README.md`
- `reports/phase5_overall_closeout/final_summary.md`
- `reports/phase5_overall_closeout/official_assets.md`
- `reports/phase5_overall_closeout/phase6_planning_note.md`
- `reports/phase5_overall_closeout/merge_decision.md`
- `reports/phase5f5_closeout/summary.md`
- `reports/phase5f5_closeout/baseline_protocol.md`
- `reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl`
- `docs/BIORAG_NEXT_STAGE.md`

Optional read-only supporting inputs for later substages:

- existing Phase 5 table/figure audit scripts;
- existing parsed_clean/chunk inspection scripts;
- existing table/figure diagnostic reports;
- existing Phase 21B parser/table/figure/numeric audit artifacts, if treated
  explicitly as supporting context rather than the Phase 5 official baseline.

## Classification Outputs

Phase6A-2 and Phase6A-3 should classify samples without running new retrieval or
model calls. Each classification row should include:

- `sample_id`
- `query_type`
- `classification`
- `confidence`
- `evidence_basis`
- `requires_human_review`
- `notes`

The classifications are audit labels, not production behavior.

## Success Criteria For Phase6A-0/1

Phase6A-0 is complete when:

- audit boundaries are documented;
- baseline pins are listed;
- classification terms are defined;
- source inventory exists;
- no code, test, model, or index action has been performed.

Phase6A-1 is complete when:

- official baseline and legacy reference boundaries are documented;
- Phase 5 default-off capabilities are not promoted to defaults;
- known stale or conflicting documentation entry points are identified;
- the next audit stages have a stable guardrail to work against.

## Non-Goals

v7-phase6A does not decide production migration. It only decides whether later
candidate tracks should be opened. Any future implementation track must use
separate isolated assets and a new decision gate.
