# v7-phase6A Source Inventory

Generated at: 2026-05-15

## Purpose

This inventory records the sources used to prepare v7-phase6A-0/1. It is a
read-only inventory. It does not mark any source as modified.

## Primary Sources

| Source | Role In Phase6A | Notes |
|---|---|---|
| `configs/baseline_registry.yaml` | Current registry for official clean baseline and legacy production reference | Confirms `phase5f_official_clean_baseline` and `current_default` separation |
| `data/baselines/phase5f_official_clean_baseline/manifest.json` | Durable baseline asset manifest | Confirms dataset, chunks, BM25, Milvus, model, vector dimension, and feature flags |
| `data/baselines/phase5f_official_clean_baseline/README.md` | Baseline asset README | States this is not table-enhancement-ON and not production current default |
| `reports/phase5f5_closeout/summary.md` | Phase 5F-5 closeout summary | Freezes official dataset and retrieval-only metrics |
| `reports/phase5f5_closeout/baseline_protocol.md` | Future comparison protocol | Defines required pins and forbidden baseline practices |
| `reports/phase5f5_closeout/official_baseline_asset_closeout.md` | Asset closeout | Confirms asset paths and legacy production reference boundary |
| `reports/phase5_overall_closeout/final_summary.md` | Phase 5 final summary | States Phase 5 did not implement structured tables, OCR, VLM, schemas, or generation/RAGAS |
| `reports/phase5_overall_closeout/official_assets.md` | Official asset registry | Human-readable baseline and experimental variant registry |
| `reports/phase5_overall_closeout/merge_decision.md` | Phase 5 merge decision | Confirms Phase 5C/5D are default-off capabilities |
| `reports/phase5_overall_closeout/phase6_planning_note.md` | Phase 6 planning seed | Recommends starting with structured table / figure-image / OCR-VLM feasibility audit |
| `reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl` | Official clean-main denominator | Contains the table/figure/control sample set for later classification |
| `docs/BIORAG_NEXT_STAGE.md` | Long-range design context | Useful for candidate schemas, but it is a design plan, not implemented capability |

## Supporting Context Sources

| Source | Role In Phase6A | Boundary |
|---|---|---|
| `scripts/evaluation/audit_table_figure_retrieval_text.py` | Existing read-only table/figure retrieval-text audit capability | Can inform later audit design; not executed in Phase6A-0/1 |
| `scripts/evaluation/audit_phase5_table_content_loss.py` | Existing read-only table content loss audit capability | Can inform later table audit; not executed in Phase6A-0/1 |
| `scripts/ingestion/audit_table_figure_evidence.py` | Existing table/figure evidence audit utility | Candidate support for later read-only inspection |
| `scripts/audit/audit_parsed_clean_quality.py` | Existing parsed_clean quality audit script | Candidate support for parser-loss inspection |
| `results/phase21b_fix1a_numeric_evidence_chain_audit/` | Later parser/table/figure numeric audit context | Supporting evidence only; not the Phase 5 official clean baseline |
| `reports/phase21b_plan0_major_failure_roadmap/` | Later repair-roadmap context | Useful warning about parser/table/figure issues; not a Phase6A decision source by itself |

## Sources With Known Baseline Drift

| Source | Drift |
|---|---|
| `README.md` | Mentions Phase 20 convergence baseline; does not supersede Phase 5F official clean baseline for v7-phase6A |
| `docs/README.md` | Mentions older Phase 9 / Phase 11 baseline state; does not supersede Phase 5F official clean baseline |
| `results/phase-reports/phase7_*.md` | Historical Phase 7 smoke reports; must not be confused with future Phase 7 or the v7 prefix |

## Source Priority Rule

For v7-phase6A, source priority is:

1. `configs/baseline_registry.yaml` and baseline manifest.
2. Phase 5F / Phase 5 overall closeout reports.
3. Phase 6 planning note.
4. Design docs and later-phase repair artifacts as supporting context only.

Older README entries and historical Phase 7 smoke reports must not override the
Phase 5F official clean baseline.

## Inventory Conclusion

The repository contains enough baseline and planning material to start
v7-phase6A-0/1. Phase6A-2 and later should use this inventory as the boundary
for read-only sample classification.
