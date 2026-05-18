# BIORAG Script Inventory

This inventory reflects Cleanup PR3 after the delete-after-quarantine pass. It is descriptive; production code, retrieval, generation, accepted data, and configs were not changed by this document.

## Counts

Original Python scripts under `scripts/`: 281.
Python scripts remaining under `scripts/`: 167.
Python scripts quarantined in PR2 under `archive/scripts/phase_artifacts/`: 114.
Python scripts currently archived under `archive/scripts/phase_artifacts/`: 0.
Python scripts deleted after quarantine in PR3: 114.
Retained `unknown` scripts still under `scripts/`: 53.

| Directory | Original | PR2 quarantined | PR3 deleted_after_quarantine | Remaining |
| --- | ---: | ---: | ---: | ---: |
| `scripts/audit` | 2 | 0 | 0 | 2 |
| `scripts/data_prep` | 4 | 0 | 0 | 4 |
| `scripts/diagnostics` | 21 | 1 | 1 | 20 |
| `scripts/evaluation` | 210 | 113 | 113 | 97 |
| `scripts/extraction` | 28 | 0 | 0 | 28 |
| `scripts/ingestion` | 15 | 0 | 0 | 15 |
| `scripts/ops` | 1 | 0 | 0 | 1 |

## Current Keep Groups

- `production_or_ops_entry`: `scripts/ops/interactive_rag_cli.py`.
- `current_ingestion_entry`: build-round, PDF parse, clean, chunk, import, parent-index, document-cleaning, and caption-cleanup entries used by current ingestion tests or docs.
- `current_eval_entry`: documented validation/RAGAS/retrieval entries, `evaluate_ragas.py`, `evaluate_e2e_small.py`, the biorag eval package, and retained Phase20 accepted-baseline scripts.
- `current_phase7_final_pipeline_entry`: all Phase7/v7 table preview/final-chain evaluation scripts and `scripts/extraction/` remain in place for this PR.
- `protected_by_tests`: 60 scripts have direct test import/path protection.
- `referenced_by_docs`: current README/startup/cleanup-policy entries remain in place.

The full allowlist is recorded in `docs/architecture/scripts_keep_allowlist.md`.

## PR2 Quarantined And PR3 Deleted Groups

- Historical Phase4/5/6/8 audit, prepare, review, signoff, and table-retrieval eval scripts.
- Historical generation stage and generation eval scripts (`run_generation_stage*`, `run_generation_smoke100.py`, `run_generation_v2_baseline_matrix.py`).
- Historical Phase12 through Phase21 evaluation scripts that were not current entries, not test-protected, not docs/user-entry referenced, and not Phase7 preview/final-chain scripts.
- One diagnostics phase smoke: `scripts/diagnostics/run_phase12e_diagnostic_smoke.py`.

Archived files were moved to `archive/scripts/phase_artifacts/` with their
original subdirectory under that archive root in PR2. PR3 deleted those 114
quarantined Python scripts from the archive path. The full candidate matrix and
proof method are recorded in `docs/architecture/scripts_archive_candidates.md`.

## Not Archived In PR2

- Current ops, ingestion, evaluation, Phase7 preview/final, and test-protected scripts.
- Non-phase audit/data-prep/diagnostics/evaluation scripts marked `unknown`; these require a separate proof pass.
- Official datasets, baseline data, accepted baseline artifacts, production configs, and source code under `app/` or `src/`.

## Cleanup Notes

- No tests were moved to `tests/legacy/`; archive candidates had no direct test import/path refs.
- PR3 deleted only the 114 PR2-quarantined Python scripts under `archive/scripts/phase_artifacts/`.
- No current Python files under `scripts/` were deleted or moved in PR3.
- The 53 `unknown` scripts remain unprocessed in `scripts/`.
- Phase7 table evidence/citation remains preview/offline; no production wiring was changed.
- Generation v2, the old generation path, retrieval, BM25, dense, hybrid, rerank, and `/v1/ask` production behavior were not changed.
