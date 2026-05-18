# Scripts Cleanup Report

Cleanup PR2 quarantined historical phase artifacts from `scripts/` into
`archive/scripts/phase_artifacts/`. Cleanup PR3 deleted that quarantined set
after the PR2 verification showed no collect decrease and focused checks passed.
Cleanup PR4 proof-checked the retained `unknown` scripts and quarantined the
subset that stayed inside the cleanup guardrails. Cleanup PR5 deleted the
PR4-quarantined unknown candidates after the same focused checks had passed.

## Summary

| Item | Count | Notes |
| --- | ---: | --- |
| Original Python scripts under `scripts/` | 281 | Counted before quarantine. |
| Scripts remaining under `scripts/` | 120 | 54 keep, 61 protected-by-tests, and 5 retained `unknown` scripts. |
| PR2 quarantined scripts | 114 | 113 from `scripts/evaluation`, 1 from `scripts/diagnostics`. |
| PR3 `deleted_after_quarantine` scripts | 114 | Deleted only from `archive/scripts/phase_artifacts/`. |
| Archived Python scripts currently under `archive/scripts/phase_artifacts/` | 0 | No `.py` files remain in the working tree under the archive path. |
| PR4 quarantined unknown scripts | 47 | Moved to `archive/scripts/unknown_candidates/`. |
| PR5 `deleted_after_quarantine` unknown scripts | 47 | Deleted only from `archive/scripts/unknown_candidates/`. |
| Archived Python scripts currently under `archive/scripts/unknown_candidates/` | 0 | The archive root was removed after it emptied. |
| PR4 reclassified test-protected scripts | 1 | `scripts/diagnostics/chunk_retrieval_smoke_v5.py` is dynamically loaded by tests. |
| Retained `unknown` scripts | 5 | Still in `scripts/ingestion`; Phase7 guardrails treat ingestion git drift as pipeline drift. |
| Legacy tests moved | 0 | Archive candidates had no direct test import/path references. |

## What Moved In PR2 And Was Deleted In PR3

Archived files were moved under `archive/scripts/phase_artifacts/`, preserving
their original subdirectory below that archive root. PR3 deleted those archived
Python scripts from the archive path without moving or deleting any current
`scripts/` file.

The moved set consists of:

- Historical Phase4/5/6/8 audit, prepare, review, signoff, and table-retrieval
  evaluation scripts.
- Historical generation stage/eval scripts, including `run_generation_stage*`,
  `run_generation_smoke100.py`, and `run_generation_v2_baseline_matrix.py`.
- Historical Phase12 through Phase21 evaluation scripts that were not current
  entries, not test-protected, not user-doc referenced, and not Phase7
  preview/final-chain scripts.
- `scripts/diagnostics/run_phase12e_diagnostic_smoke.py`.

The full proof matrix is in `docs/architecture/scripts_archive_candidates.md`.
The keep allowlist is in `docs/architecture/scripts_keep_allowlist.md`.

After PR3, `archive/scripts/phase_artifacts/` has zero working-tree Python scripts.
No non-PR2-quarantine Python script was found under that archive path during the
pre-delete cross-check.

## What Moved In PR4 And Was Deleted In PR5

PR4 moved 47 previously unknown scripts to `archive/scripts/unknown_candidates/`,
preserving their original subdirectory below that archive root:

- 2 from `scripts/audit`;
- 4 from `scripts/data_prep`;
- 18 from `scripts/diagnostics`;
- 23 from `scripts/evaluation`.

PR5 deleted those 47 archived Python scripts and removed the
`archive/scripts/unknown_candidates/` directory after it emptied. The pre-delete
cross-check matched all 47 documented PR4 quarantine paths to the 47 working-tree
Python files under that archive root, with zero extra archive `.py` files and
zero missing documented paths.

PR4 and PR5 did not move or delete the 5 `scripts/ingestion` unknown scripts
because the Phase7 baseline and rollback guardrails check
`git status --short -- scripts/ingestion` and fail on drift.

One file was restored during validation and reclassified as test-protected:
`scripts/diagnostics/chunk_retrieval_smoke_v5.py`. `pytest --collect-only -q`
exposed that `tests/test_chunk_retrieval_smoke_v5.py` dynamically loads it by
file path.

## Tests Legacy Handling

No tests were moved to `tests/legacy/`.

Reason: every archived script had `tests import/path ref = no` in the scan
matrix. Since no current test directly imported or path-read the archived
scripts, moving tests would have reduced the active suite without a matching
test ownership reason.

Manual-run note: not applicable for tests; no legacy tests were moved.

## Collect Count

| Checkpoint | Collected tests |
| --- | ---: |
| Before quarantine | 1042 |
| After quarantine | 1042 |
| After delete-after-quarantine | 1042 |
| After PR4 unknown quarantine | 1042 |
| After PR5 unknown delete-after-quarantine | 1042 |
| Delta | 0 |

There was no collect decrease. No tests were moved out of the active suite, and
no import failure was introduced by the final PR4 quarantine set or the PR5
delete-after-quarantine pass. An intermediate PR4 attempt to move
`scripts/diagnostics/chunk_retrieval_smoke_v5.py` caused a collect error; it was
restored and marked test-protected before final verification.

## Verification

Commands run:

```bash
python -m compileall app src scripts tests
pytest --collect-only -q
pytest tests/test_generation_v2.py -q
pytest tests/test_phase7_table_retrieval_wiring_preview.py tests/test_phase7l_table_rag_smoke.py tests/test_phase7m_sandbox_contract_hardening.py tests/test_phase7q_table_citation_schema_prototype.py tests/test_phase7q1_table_citation_mapper_dry_run.py tests/test_phase7r_table_index_production_proposal.py tests/test_phase7s_production_readiness_dry_run.py tests/test_phase7t_table_preview_scaffold.py tests/test_phase7u_table_preview_eval_smoke.py tests/test_phase7v_fast_type_aware_merge.py tests/test_phase7w_slim_mainchain_preview.py tests/test_phase7x_final_default_on_table_preview.py -q
```

Results:

- PR2 `compileall` including `archive`: passed.
- PR3 `python -m compileall app src scripts tests`: passed.
- PR4 `python -m compileall app src scripts tests`: passed.
- PR5 `python -m compileall app src scripts tests`: passed.
- `pytest --collect-only -q`: passed, 1042 tests collected.
- `pytest tests/test_generation_v2.py -q`: passed, 16 tests.
- Phase7/table preview focused set: passed, 96 tests.

No RAGAS, Qwen, embedding, rerank, or retrieval evaluation was run.

## Production Impact

No production code under `app/` or `src/` was changed by this cleanup pass.
No official dataset, baseline data, accepted baseline artifact, or baseline
registry was changed.

No config file was edited in PR5.

The current ops, ingestion, and evaluation entries remain in `scripts/`.

Generation v2, the old generation path, retrieval, BM25, dense, hybrid, rerank,
and `/v1/ask` production behavior were not changed.

## Phase7 Boundary

Phase7 table evidence/citation remains preview/offline. This cleanup did not:

- wire Phase7 table evidence into production generation;
- relax production citation guards;
- move Phase7/v7 preview/final-chain scripts out of `scripts/`;
- change `src/synbio_rag/application/table_preview.py`;
- change production `CitationBinder` behavior.

PR4 and PR5 left `scripts/ingestion` unchanged in the final diff so the Phase7
baseline and rollback guardrails continue to report no ingestion pipeline drift.

## Failed Or Restored Files

No PR2/PR3 archived file had to be restored.

During PR4 validation, one intermediate unknown quarantine was restored before
finalizing the move set:

- `scripts/diagnostics/chunk_retrieval_smoke_v5.py`

Reason: `pytest --collect-only -q` failed because
`tests/test_chunk_retrieval_smoke_v5.py` dynamically loads that file by path.
It is now classified as `protected_by_tests` and remains under `scripts/`.

One post-move closure check found that two remaining historical generation eval
scripts still imported an archived generation-stage helper:

- `scripts/evaluation/run_generation_smoke100.py`
- `scripts/evaluation/run_generation_v2_baseline_matrix.py`

Both had no test, user-doc, app/src, config, or current accepted-baseline
reference, so they were archived with the rest of the generation-stage artifact
set. A second closure check confirmed that remaining `scripts/` files no longer
import or path-reference archived original scripts.

## Next Cleanup Round

The remaining cleanup decision is the 5 retained `scripts/ingestion` unknown
scripts. They need a separate plan because moving them trips Phase7 ingestion
drift guardrails even though the reference scan found no direct tests, app/src
refs, or user-doc refs.
