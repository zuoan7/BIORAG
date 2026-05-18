# BIORAG Current Inventory

This inventory records the current repository boundary before any larger cleanup.
It is descriptive only: files listed here are not moved, deleted, or promoted by
this document.

## production_entry

- `app/main.py`: current FastAPI service entry. It exposes `/healthz`,
  `/v1/sessions/{session_id}`, and `/v1/ask`.
- `/v1/ask`: request model is in `app/main.py`; execution flows through
  `RAGApplicationService.ask()` and `SynBioRAGPipeline.answer()`.
- `src/synbio_rag/application/rag_service.py`: application service wrapper for
  sessions, audit logging, and pipeline invocation.
- `src/synbio_rag/application/pipeline.py`: current main RAG orchestration path.
- `src/synbio_rag/application/generation_v2/`: current generation chain. The
  runtime is v2-only; `GenerationConfig.version` and `GENERATION_VERSION` are no
  longer supported.
- `src/synbio_rag/application/rerank_service.py`: main-process local BGE
  reranker orchestration and guarded rerank logic.
- `src/synbio_rag/application/parent_expansion.py`: current v2 context expansion
  path after rerank seeds.
- `src/synbio_rag/rewrite/query_rewrite_service.py`: feature-flagged query
  rewrite service. Production default remains `QUERY_REWRITE_MODE=off`.
- `src/synbio_rag/application/table_preview.py`: preview table sidecar candidate
  provider. It can add preview candidates before rerank when enabled, but formal
  table citation remains blocked by default.

## production_core

- `src/synbio_rag/domain/`: configuration, request/response dataclasses, router,
  and confidence policy.
- `src/synbio_rag/application/`: main application services, pipeline, rerank,
  parent expansion, table preview, and generation modules.
- `src/synbio_rag/application/generation_v2/`: evidence ledger, support
  selection, answer planning/building, citation binding, validation, and optional
  Qwen synthesis.
- `src/synbio_rag/infrastructure/`: embedding, reranker, vectorstores, persistence,
  external clients, and parent index store.
- `src/synbio_rag/rewrite/`: query rewrite service and prompt-backed rewrite
  wiring.
- `src/synbio_rag/ingestion/`: reusable ingestion helpers used by current
  ingestion scripts.
- `resources/prompts/query_rewrite_en_mirror.txt`: query rewrite prompt resource.
- `config/settings.example.env`: documented environment variable example.

## experimental_preview

Phase7 table evidence and citation work remains preview/prototype/offline unless
explicitly promoted in a later cleanup or feature PR.

- `src/synbio_rag/application/table_preview.py` adapts offline table units into
  preview `RetrievedChunk` objects. These chunks use debug source metadata and
  must not be treated as formal table citations.
- `schemas/table_object_v1.yaml` and `schemas/table_index_unit_v1.yaml` describe
  offline table object/index unit formats.
- `docs/table_object_schema_v1.md` and `docs/table_index_unit_v1.md` document
  Phase7 preview schemas.
- `scripts/extraction/*` contains Phase7 table extraction, QA, review, and
  rendering scripts.
- `scripts/evaluation/phase7*` and `scripts/evaluation/v7_phase6*` contain
  preview smoke tests, readiness dry runs, and proposal/report generators.
- `data/experiments/v7_phase7_*`, `reports/v7_phase7_*`, and
  `results/v7_phase7_*` are experiment artifacts or fixtures, not production
  baseline assets by default.

Phase7 constraints:

- Table evidence / citation is still preview/prototype/offline artifact work.
- Table units with `preview_only` or `production_ready=false` are not
  production-ready.
- `value_bboxes_available=false` must not be converted into value-level bbox
  claims.
- CSV, PDF crop, and markdown card paths are debug provenance only and must not
  enter formal citation source fields.
- Production `CitationBinder` must not be changed to allow formal Phase7 table
  citation as part of cleanup.
- Cleanup must not alter `/v1/ask` behavior when Phase7 preview data is present.

## legacy_candidate

These items may be considered for later archival or explicit support decisions.
They are not deleted in this round.

- Old generation support has been removed. `generation_service.py`,
  `context_builder.py`, and `legacy_generation_flow.py` are no longer part of
  the source tree.
- `src/synbio_rag/application/neighbor_expansion.py`: retained as the generation
  v2 neighbor-audit index source; it is no longer a generation branch.
- `src/synbio_rag/application/query_expansion.py`: prototype query expansion; no
  current main pipeline import was found in this inventory pass.
- `scripts/evaluation/run_phase*`, `scripts/evaluation/phase*`,
  `scripts/evaluation/audit_phase*`, and many `scripts/extraction/phase*`
  workflows: one-off phase scripts and report builders.
- Historical reports under `reports/phase*`, `results/phase*`, and
  `docs/BIORAG_NEXT_STAGE.md`: useful audit trail, not current API logic.

## artifact_or_fixture

Do not delete these by directory name alone. Some ignored artifact directories
also contain tracked test fixtures or current baseline references.

- Runtime/local assets:
  - `models/`: local BGE model assets, ignored by git.
  - `runtime/`: local Milvus and runtime state, ignored by git.
  - `logs/` and `vector_db/`: local runtime leftovers, ignored by git.
- Knowledge base and datasets:
  - `data/paper_round1/`: local paper corpus and generated KB assets.
  - `data/eval/datasets/` and `data/eval/manifests/`: tracked evaluation inputs.
  - `data/evaluation/`: mixed historical evaluation fixtures.
  - `data/experiments/`: mostly Phase experiment artifacts; some files are
    tracked fixtures for tests.
- Reports and results:
  - `reports/phase*`, `reports/v7_phase*`: historical reports and accepted
    report references.
  - `results/phase*`, `results/v7_phase*`: historical outputs; a small tracked
    subset is used by tests or documentation.
  - `configs/baseline_registry.yaml`: tracked baseline registry; do not edit
    unless the baseline contract changes.
- Test fixtures:
  - `tests/fixtures/`: explicit test fixtures.
  - Any file imported or read by tests should be treated as protected until a
    targeted grep/ripgrep check proves otherwise.
