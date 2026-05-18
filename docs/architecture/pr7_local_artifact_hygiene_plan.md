# PR7 Local Artifact Hygiene Plan

PR7 should start after PR6 is reviewed or merged. PR6 found no tracked
results/reports archive candidates: the tracked artifact set is small and
protected as staged baselines, major phase summaries, constructed datasets, or
Phase7 preview fixtures.

## Goal

Audit ignored and untracked local artifacts without changing production behavior
or deleting protected baseline evidence.

PR7 should answer:

- Which ignored/untracked artifact directories are local runtime/cache output?
- Which ignored/untracked artifacts are protected baseline or phase evidence
  that must stay in place locally?
- Is there any safe local-only cleanup action, or should the cleanup thread close
  with no repository deletion?

## Protected By Default

Do not delete, move, rewrite, or replace these in PR7:

- `reports/phase5f_eval_semantic_enhancement_v2/`
  - contains `strict_main_eval_set_v2.jsonl`, the official clean baseline
    dataset path in `configs/baseline_registry.yaml`;
  - contains `summary.md` and `strict_main_eval_set_v2_summary.md`, the summary
    reports for that official denominator.
- `data/baselines/phase5f_official_clean_baseline/`
- `configs/baseline_registry.yaml`
- `data/eval/`, `data/evaluation/`, and tracked `data/experiments/v7_phase*`
  files audited in PR6.
- Tracked `reports/phase*`, `reports/v7_phase*`, `results/phase*`, and
  `results/v7_phase*` files audited in PR6.
- `models/`, `runtime/`, `vector_db/`, and `logs/` unless the user explicitly
  asks for local workspace cleanup and accepts that these are local machine
  state, not repository cleanup.

## Strict Boundaries

Do not:

- edit `app/`, `src/`, scripts, configs, accepted baselines, or tracked
  artifacts;
- change `/v1/ask`, retrieval, BM25, dense, hybrid, rerank, generation v2, old
  generation, or Phase7 preview behavior;
- run RAGAS, Qwen, embedding, rerank, retrieval evaluation, index builds, or
  model downloads;
- turn ignored local artifacts into tracked files as part of cleanup;
- use directory name alone as proof that something can be deleted.

## Audit Commands

Use read-only commands first:

```bash
git status --short --ignored
git ls-files --others --exclude-standard
git ls-files --ignored --others --exclude-standard
du -sh reports results data/eval data/evaluation data/experiments data/baselines logs runtime vector_db models 2>/dev/null
find reports results data/experiments logs runtime vector_db -maxdepth 2 -type f | sort
```

Then classify ignored/untracked paths into:

- `protected_official_baseline`
- `protected_phase_evidence`
- `local_runtime_state`
- `local_cache`
- `local_generated_output`
- `unknown_keep`

## Candidate Rules

PR7 should not delete anything by default. It may recommend deletion only for
local-only, ignored artifacts that meet all conditions:

- not referenced by `README.md`, `docs`, `tests`, `configs`, `scripts`, `app`,
  or `src`;
- not under `reports/phase5f_eval_semantic_enhancement_v2/`;
- not under `data/baselines/`;
- not needed by Phase7 preview focused tests;
- not model, vector DB, or runtime state unless the user explicitly asks for
  local workspace cleanup.

If any condition is uncertain, keep and document it.

## Verification

If PR7 is documentation-only:

```bash
git diff --check
pytest --collect-only -q
```

If PR7 deletes local ignored artifacts, verify the expected local-only state
with `git status --short --ignored` and do not commit deletion of ignored files.

If PR7 changes tracked docs only, no full compile/test run is required beyond
collect unless the diff touches code or tracked artifact paths.

## Deliverables

- An ignored/untracked artifact inventory document.
- A protected local baseline note covering
  `reports/phase5f_eval_semantic_enhancement_v2/`.
- A recommendation:
  - close cleanup with no tracked deletions; or
  - perform user-approved local-only cleanup; or
  - open a separate baseline-data tracking PR if official baseline files must be
    tracked intentionally.

