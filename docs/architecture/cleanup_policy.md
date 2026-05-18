# BIORAG Cleanup Policy

This policy scopes repository cleanup work after the current inventory pass.

## Round 1 Scope

The first cleanup round is limited to boundary documentation plus
configuration, documentation, and test alignment. It must not become an
algorithm rewrite, API migration, or script purge.

Round 1 allowed work:

- Add current inventory and script inventory documents.
- Add cleanup policy.
- Align `README.md`, `docs/README.md`, and obvious entry documentation.
- Update `config/settings.example.env` to reflect current configurable flags.
- Fix tests that contradict current code defaults.
- Remove or simplify duplicate pytest configuration only when test discovery is
  unchanged.

Round 1 forbidden work:

- Do not change `/v1/ask` behavior.
- Do not change retrieval, BM25, dense, hybrid, rerank, or generation_v2
  algorithms.
- Do not move `app/main.py`.
- Do not delete source behavior without an explicit cleanup decision.
- Do not delete Phase7 experimental scripts or artifacts.
- Do not promote Phase7 table evidence/citation to production.
- Do not run RAGAS, Qwen, embedding, rerank, or retrieval evaluation as part of
  cleanup verification.

## Deletion Rules

Any deletion or permanent archive move must be proven before the change. The
minimum proof is:

- No production import.
- No tests import.
- No `README.md` or `docs/` link.
- No dependency from current baseline or accepted report artifacts.
- A grep/ripgrep check is recorded in the PR or cleanup note.
- The relevant test collection and focused tests still pass.

If any condition is uncertain, keep the file and classify it instead.

## Scripts Policy

Phase scripts under `scripts/` default to `legacy_candidate` in the first round.
They are not deleted simply because they are historical or phase-labeled.

Protected script groups:

- Current ops entries, such as `scripts/ops/interactive_rag_cli.py`.
- Current ingestion entries, such as `scripts/ingestion/build_round1_kb.py`.
- Current evaluation entries, such as `scripts/evaluation/run_validation_suite.py`
  and `scripts/evaluation/run_ragas_regression.py`.
- Any script directly imported by tests.

## Phase7 Policy

Phase7 table evidence/citation work stays preview/offline/experimental unless a
separate production promotion PR changes that boundary.

Cleanup must preserve these constraints:

- `preview_only` means not production-ready.
- `production_ready=false` blocks production promotion.
- Missing value-level bbox data must stay missing; do not synthesize or imply it.
- CSV, crop, and markdown paths remain debug provenance only.
- Formal production citation behavior must not be relaxed during cleanup.

## Old Generation Policy

Old generation support has an explicit cleanup decision: the runtime is v2-only.
Do not reintroduce `GENERATION_VERSION=old`, `GenerationConfig.version`, or the
deleted old generation modules without a new decision record.

## App Entry Policy

`app/main.py` stays in place in the first cleanup round. A future wrapper
direction can be documented, but the service entry should not be moved until a
dedicated import/API normalization PR.

## Future PR Sequence

- PR2: legacy classification and very small archival moves for files proven
  unreferenced.
- PR3: API wrapper and import normalization.
- PR8: old generation removal completed; runtime remains v2-only.
- PR5: large `scripts/evaluation` archival plan.
