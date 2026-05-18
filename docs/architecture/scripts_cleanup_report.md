# Scripts Cleanup Report

Cleanup PR2 quarantined historical phase artifacts from `scripts/` into
`archive/scripts/phase_artifacts/`. Cleanup PR3 deleted that quarantined set
after the PR2 verification showed no collect decrease and focused checks passed.

## Summary

| Item | Count | Notes |
| --- | ---: | --- |
| Original Python scripts under `scripts/` | 281 | Counted before quarantine. |
| Scripts remaining under `scripts/` | 167 | 114 explicit keep/protected scripts plus 53 retained `unknown` scripts. |
| PR2 quarantined scripts | 114 | 113 from `scripts/evaluation`, 1 from `scripts/diagnostics`. |
| PR3 `deleted_after_quarantine` scripts | 114 | Deleted only from `archive/scripts/phase_artifacts/`. |
| Archived Python scripts currently under `archive/scripts/phase_artifacts/` | 0 | No `.py` files remain in the working tree under the archive path. |
| Retained `unknown` scripts | 53 | Still in `scripts/`; not processed in PR3. |
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
| Delta | 0 |

There was no collect decrease. No tests were moved out of the active suite, and
no import failure was introduced by the quarantine or delete-after-quarantine
step.

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
- `pytest --collect-only -q`: passed, 1042 tests collected.
- `pytest tests/test_generation_v2.py -q`: passed, 16 tests.
- Phase7/table preview focused set: passed, 96 tests.

No RAGAS, Qwen, embedding, rerank, or retrieval evaluation was run.

## Production Impact

No production code under `app/` or `src/` was changed by this cleanup pass.
No official dataset, baseline data, accepted baseline artifact, or baseline
registry was changed.

`config/settings.example.env` was already modified before this cleanup pass and
was not edited as part of the cleanup work.

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

## Failed Or Restored Files

No archived file had to be restored.

One post-move closure check found that two remaining historical generation eval
scripts still imported an archived generation-stage helper:

- `scripts/evaluation/run_generation_smoke100.py`
- `scripts/evaluation/run_generation_v2_baseline_matrix.py`

Both had no test, user-doc, app/src, config, or current accepted-baseline
reference, so they were archived with the rest of the generation-stage artifact
set. A second closure check confirmed that remaining `scripts/` files no longer
import or path-reference archived original scripts.

## Next Cleanup Round

The 53 retained `unknown` scripts should get their own proof pass before any
archive or deletion decision. PR3 deliberately did not expand the cleanup scope
or process unknown scripts.
