# Evaluation Sets

## smoke100

- **Path**: `data/eval/datasets/enterprise_ragas_smoke100.json`
- **Samples**: 100
- **Role**: Historical main regression set
- **Usage**: Baseline comparison for retrieval/generation pipeline changes
- **Status**: Stable. Do not modify without explicit re-baselining.
- **Categories**: enterprise/summary/comparison/factoid

## smoke50_parent_expansion_v1

- **Path**: `data/evaluation/smoke50_parent_expansion_v1.jsonl`
- **Samples**: 50
- **Origin**: Built as Phase 12G holdout50; promoted to formal extension after Phase 12G/12H validation
- **Role**: Parent expansion regression extension
- **Status**: Merged into main as formal regression asset. No longer an unseen holdout.
- **Categories**:
  - summary: 10
  - comparison: 10
  - table_figure_caption: 10
  - method_result_numeric: 5
  - negative_trigger_robustness: 5
  - factoid: 10
- **Retrieval modes**: open_retrieval (41), targeted_retrieval (9)

## smoke150

- **Manifest**: `data/evaluation/smoke150_manifest.json`
- **Composition**: smoke100 + smoke50_parent_expansion_v1
- **Total**: 150 samples
- **Role**: Primary regression suite for parent_expansion default-on pipeline
- **Extension path**: smoke200 can be built by adding another smoke50 extension

## Holdout Concept

- smoke50_parent_expansion_v1 was built as an unseen holdout (Phase 12G) but is now a fixed regression extension after merging.
- Future parameter tuning against smoke150 requires building a NEW holdout set.
- Do not tune parent_expansion parameters against smoke50 or smoke150 and claim holdout status.

## Reconstruction

smoke100 is versioned in git. smoke50_parent_expansion_v1 is versioned in git. Both are check-in artifacts. No regeneration needed.

For smoke150 scoring, use the manifest to compose both sets into a single eval run.
