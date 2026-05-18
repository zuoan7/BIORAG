# PR8 Source Architecture Consolidation Plan

PR8 should start after PR7 local artifact hygiene is reviewed or merged. PR7 is
artifact inventory only. PR8 is the first source-code cleanup round and should
not delete source behavior by assumption.

## Assumptions

- The production API remains `/v1/ask`.
- `GenerationConfig.version` defaults to `v2`, but `version="old"` is still an
  explicit supported path until the project decides otherwise.
- Phase7 table preview remains preview/debug-only and must not become formal
  table citation support as part of cleanup.
- Existing environment variable names remain stable in PR8.
- PR8 may add documentation and source inventory first. Code changes, if any,
  should be behavior-preserving and split into narrow follow-up PRs.

If any assumption is false, stop and update this plan before editing source.

## Goal

Reduce source-code confusion by mapping the current RAG source architecture,
separating current production paths from legacy and preview paths, and defining
small behavior-preserving consolidation steps.

PR8 should answer:

- What is the exact `/v1/ask` runtime chain today?
- Which modules are production-current, legacy-supported, preview-only, or
  cleanup candidates?
- Which patches are policy/configuration concerns versus orchestration logic?
- Which refactors can be done without changing retrieval, rerank, generation,
  or citation behavior?
- What decision is needed before old generation can be removed or isolated?

## Non-Goals

Do not:

- delete `src/` or `app/` modules in PR8 without a specific decision record;
- remove `GenerationConfig.version="old"` support silently;
- change `/v1/ask` response shape, citations, confidence, debug payload, or
  session/audit behavior;
- change dense, BM25, hybrid retrieval, rerank ordering, parent expansion,
  same-doc expansion, original-CN fallback, query rewrite, generation v2,
  old generation, or Phase7 table preview behavior;
- change default environment variable values;
- edit `reports/`, `results/`, `data/baselines/`, accepted baselines, or
  artifact cleanup docs except to reference source decisions;
- run RAGAS, Qwen, embedding, rerank evaluation, index builds, model downloads,
  or retrieval benchmark runs.

## Current Source Map

Production entry:

```text
app/main.py
  -> RAGApplicationService.ask()
  -> SynBioRAGPipeline.answer()
  -> retrieval + optional table preview + rerank + expansion
  -> generation_v2 by default, old generation when configured
```

Current source roles:

| Area | Path | Current role | PR8 disposition |
| --- | --- | --- | --- |
| API entry | `app/main.py` | FastAPI `/v1/ask`, session lookup, health | `keep_current` |
| Reranker entry | `app/reranker_main.py` | Optional standalone `/v1/rerank` service | `keep_current` |
| App service | `src/synbio_rag/application/rag_service.py` | Session/audit wrapper around pipeline | `keep_current` |
| Main orchestration | `src/synbio_rag/application/pipeline.py` | Retrieval, rewrite, preview, rerank, expansion, generation dispatch | `refactor_candidate` |
| Generation v2 | `src/synbio_rag/application/generation_v2/` | Default answer chain | `keep_current` |
| Old generation | `src/synbio_rag/application/generation_service.py` | Explicit old generation path | `keep_legacy_supported` |
| Old context | `src/synbio_rag/application/context_builder.py` | Old generation context builder | `keep_legacy_supported` |
| Old neighbor path | `src/synbio_rag/application/neighbor_expansion.py` | Old/diagnostic neighbor expansion | `keep_legacy_supported` |
| Parent expansion | `src/synbio_rag/application/parent_expansion.py` | Current v2 context expansion after rerank seeds | `keep_current` |
| Rerank adapter | `src/synbio_rag/application/rerank_service.py` | Reranker service client and fallback | `keep_current` |
| Table preview | `src/synbio_rag/application/table_preview.py` | Phase7 preview candidate sidecar | `isolate_preview` |
| Query rewrite | `src/synbio_rag/rewrite/query_rewrite_service.py` | Feature-flagged rewrite service | `keep_current_flagged` |
| Config | `src/synbio_rag/domain/config.py` | Dataclasses, env parsing, path resolution, profile enforcement | `refactor_candidate` |
| Schemas | `src/synbio_rag/domain/schemas.py` | Domain request/response dataclasses | `keep_current` |
| Router | `src/synbio_rag/domain/router.py` | Query intent analysis | `keep_current` |
| Confidence | `src/synbio_rag/domain/confidence.py` | Confidence policy | `keep_current` |
| Infrastructure | `src/synbio_rag/infrastructure/*` | Embedding, Milvus, BM25, hybrid, reranker, persistence | `keep_current` |
| Ingestion helpers | `src/synbio_rag/ingestion/*` | Reusable ingestion libraries used by scripts/tests | `keep_current` |

## Observed Architecture Pressure

The current source tree has three cleanup pressures:

1. `pipeline.py` is doing too much.
   - It owns runtime orchestration.
   - It builds query rewrite clients.
   - It runs table preview.
   - It supplements summary sections.
   - It runs original-CN fallback.
   - It dispatches generation v2 versus old generation.

2. `domain/config.py` is mixing concerns.
   - Config dataclasses live beside all env parsing.
   - Path resolution and directory creation live beside profile policy.
   - Phase flags and legacy flags are interleaved with current defaults.

3. Generation boundaries are unclear.
   - `generation_v2/` is current default.
   - `generation_service.py` still contains a large old generation path.
   - Tests prove `GENERATION_VERSION=old` can still be selected.
   - Removing old generation needs an explicit product/compatibility decision.

## PR8 Deliverables

PR8 should be documentation and source inventory first:

- `docs/architecture/pr8_source_architecture_consolidation_plan.md`
- optional `docs/architecture/source_runtime_chain_inventory.md`
- optional `docs/architecture/source_module_disposition_matrix.md`

The inventory should classify each source module as:

- `keep_current`
- `keep_current_flagged`
- `keep_legacy_supported`
- `isolate_preview`
- `refactor_candidate`
- `delete_candidate`
- `unknown_keep`

PR8 should not contain large behavior-changing source edits. If a tiny
mechanical source cleanup is included, it must be directly justified by the
inventory and verified with focused tests.

## Proposed PR Sequence

### PR8A: Runtime Chain Inventory

Scope:

- map `/v1/ask` from FastAPI request through response;
- list direct collaborators of `SynBioRAGPipeline.answer()`;
- identify which debug keys and response fields are externally visible;
- identify old generation branch conditions;
- identify Phase7 preview branch conditions.

Verification:

```bash
git diff --check
pytest --collect-only -q
pytest tests/test_generation_v2.py -q
pytest tests/test_phase7t_table_preview_scaffold.py -q
pytest tests/test_phase7w_slim_mainchain_preview.py -q
```

Expected output:

- no source behavior change;
- clear list of current runtime seams;
- no deletion candidates accepted yet.

### PR8B: Config Boundary Plan

Scope:

- inventory `Settings.from_env()` by config group;
- separate future extraction targets:
  - retrieval env parsing;
  - generation env parsing;
  - table preview env parsing;
  - query rewrite env parsing;
  - path resolution;
  - profile/forbidden flag enforcement;
- keep all env names and defaults unchanged.

Candidate future files:

```text
src/synbio_rag/domain/config.py
src/synbio_rag/domain/config_env.py
src/synbio_rag/domain/config_paths.py
src/synbio_rag/domain/config_profiles.py
```

Verification:

```bash
pytest tests/test_e2e_eval_config.py -q
pytest tests/test_generation_v2_config_profiles.py -q
pytest tests/test_parent_expansion_default_config.py -q
pytest tests/test_phase7t_table_preview_scaffold.py -q
```

Expected output:

- no default changes;
- no env variable rename;
- no changed resolved local paths.

### PR8C: Pipeline Orchestration Boundary Plan

Scope:

- identify behavior-preserving extraction points from `pipeline.py`;
- define names before moving code;
- keep `SynBioRAGPipeline.answer()` as the external application method.

Candidate extraction seams:

| Current concern | Candidate module | Rule |
| --- | --- | --- |
| query rewrite client setup | `application/query_rewrite_adapter.py` | no prompt or cache behavior change |
| table preview orchestration | `application/table_preview_pipeline.py` | no formal citation behavior change |
| original-CN fallback | `application/original_cn_fallback.py` | no retrieval scoring change |
| summary section supplement | `application/summary_supplement.py` | no chunk ordering change |
| v2 evidence/response assembly | `application/evidence_selection_stage.py`, `application/generation_v2_response.py` | preserve v2 debug and citation semantics |
| legacy generation branch | `application/legacy_generation_flow.py` | preserve `GENERATION_VERSION=old` behavior |

Verification:

```bash
pytest tests/test_phase21a9c_query_rewrite_wiring.py -q
pytest tests/test_generation_v2.py -q
pytest tests/test_parent_expansion.py -q
pytest tests/test_pipeline_summary_supplement.py -q
pytest tests/test_phase7t_table_preview_scaffold.py -q
pytest tests/test_phase7x_final_default_on_table_preview.py -q
```

Expected output:

- call graph is easier to read;
- no result ordering or debug payload changes unless a focused test is updated
  with an explicit reason.

### PR8D: Legacy Generation Decision

Scope:

- decide whether `GenerationConfig.version="old"` remains supported;
- document the decision before deleting or moving old-generation code.

Options:

| Option | Action | Tradeoff |
| --- | --- | --- |
| Keep old support | Move old generation into an explicit legacy package | Lowest behavior risk, keeps maintenance cost. |
| Deprecate old support | Add decision doc and warning path first | Medium risk, gives migration window. |
| Remove old support | Delete old generation after tests and docs prove unused | Highest cleanup value, highest compatibility risk. |

No deletion should happen until one option is accepted.

Verification if kept:

```bash
pytest tests/test_generation_v2.py::test_generation_version_env_can_select_old -q
```

Verification if deprecated or removed:

```bash
rg -n "version=\"old\"|GENERATION_VERSION.*old|generation.version == \"old\"" README.md docs tests app src
```

Expected output:

- old generation has one explicit owner and status;
- no silent compatibility break.

## Source Cleanup Rules

Use these rules for any source edit after PR8 inventory:

- Every edit must preserve `/v1/ask` behavior unless the PR title says it is a
  behavior change.
- Prefer moving cohesive helpers over rewriting logic.
- Keep public dataclass fields, config names, env vars, and response keys stable.
- Do not merge Phase7 preview citation with formal citation code.
- Do not make old generation look current; either isolate it or deprecate it.
- Do not delete code only because it looks old. First prove its status with
  references, tests, and runtime branch conditions.
- Keep refactors smaller than the test surface they require.

## Candidate Deletion Rules

A source file or function can become a deletion candidate only if all are true:

- no production import path reaches it;
- no tests import or assert it;
- no docs or config describe it as supported;
- no script uses it as a library;
- no baseline, Phase7 preview, or old-generation compatibility decision depends
  on it;
- a focused removal test or collection run confirms the deletion.

If any condition is uncertain, classify as `unknown_keep` or
`keep_legacy_supported`.

## Risk Matrix

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Old generation removal breaks configured users | `GENERATION_VERSION=old` is tested | Require explicit decision before deletion. |
| Config refactor changes defaults | Env parsing is long and interleaved | Golden tests for config defaults and env overrides. |
| Pipeline extraction changes ordering | Retrieval/rerank/generation order is behavior | Move code first, then compare focused tests. |
| Phase7 preview leaks into formal citation | Preview data is debug-only | Keep citation guard tests mandatory. |
| Debug payload changes break evaluation tooling | Tests and scripts inspect debug fields | Inventory debug keys before edits. |
| Query rewrite changes cached behavior | Rewrite mode is feature-flagged | Preserve cache version and prompt wiring. |

## Verification Baseline

Minimum verification for documentation-only PR8:

```bash
git diff --check
pytest --collect-only -q
```

Minimum verification for behavior-preserving source refactors:

```bash
git diff --check
pytest tests/test_generation_v2.py -q
pytest tests/test_e2e_eval_config.py -q
pytest tests/test_parent_expansion.py -q
pytest tests/test_phase21a9c_query_rewrite_wiring.py -q
pytest tests/test_pipeline_summary_supplement.py -q
pytest tests/test_phase7t_table_preview_scaffold.py -q
pytest tests/test_phase7w_slim_mainchain_preview.py -q
pytest tests/test_phase7x_final_default_on_table_preview.py -q
```

Run broader tests only after source movement is complete or if focused tests
expose uncertainty.

## Recommended First Step

Start PR8 with source inventory only:

1. Create `cleanup/pr8-source-architecture-consolidation` after PR7 is committed
   or merged.
2. Add a runtime chain inventory for `/v1/ask`.
3. Add a module disposition matrix for `app/` and `src/synbio_rag/`.
4. Decide whether PR8 remains documentation-only or splits code movement into
   PR9+.

Do not start by deleting source files. The first useful cleanup is to make the
current seams visible, then move one seam at a time with focused tests.
