# Runtime and Configuration Contract

This document freezes the current runtime/configuration contract before larger
cleanup work. It is descriptive only: it does not promote Phase7 table evidence,
change `/v1/ask`, or redefine retrieval/generation behavior.

## Runtime Entry

`/v1/ask` remains the production answer endpoint.

```text
app/main.py::ask()
  -> RAGApplicationService.ask()
      -> load session history
      -> SynBioRAGPipeline.answer()
      -> append user/assistant turns
      -> adapt RAGPipelineResponse to legacy RAGResponse
      -> hide debug unless include_debug=true
      -> write audit log
  -> serialize RAGResponse
```

The runtime answer chain is generation v2-only. `GenerationConfig.version` and
`GENERATION_VERSION` are not supported configuration surfaces.

## Main Pipeline

`SynBioRAGPipeline.answer()` currently coordinates these steps:

1. Query intent analysis.
2. Query rewrite when `QUERY_REWRITE_MODE` enables it.
3. Hybrid retrieval with filter fallback.
4. Optional original Chinese fallback floor.
5. Optional Phase7 table preview candidate pass.
6. Local BGE rerank.
7. Final seed selection and `rerank_rank` metadata annotation.
8. Summary section supplement for summary queries.
9. Generation v2 evidence selection with parent expansion.
10. Seed/final confidence scoring.
11. Generation v2 answer construction.
12. `RAGPipelineResponse` assembly.

The current pipeline is still a patch orchestrator. Future stage extraction
should preserve this behavior first, then move responsibilities in small steps.

Phase 3 keeps `SynBioRAGPipeline.answer()` as the public application facade and
moves the orchestration blocks into internal stage classes:

- `RetrievalStage`: query analysis, query rewrite, filter fallback retrieval,
  original-CN fallback, and Phase7 table preview.
- `RerankStage`: local rerank, final seed slicing, and `rerank_rank` annotation.
- `ContextStage`: summary supplement plus generation v2 evidence/parent
  expansion.
- `GenerationStage`: confidence scoring, Phase7 merged-preview citation
  requirement relaxation, and generation v2 invocation.
- `ResponseStage`: v2 internal `RAGPipelineResponse` assembly.

The stage split is behavior-preserving. It does not introduce new configuration,
does not change debug keys, and does not change Phase7 preview/citation policy.

Phase 4 keeps `/v1/ask` behavior unchanged while moving retrieval policy out of
`infrastructure/vectorstores/hybrid.py`. `SynBioRAGPipeline.__init__()` now
constructs and injects `RetrievalQueryPlanner`, `AliasExpansionPolicy`, and
`RetrievalPostProcessor` into `HybridRetriever`. The infrastructure retriever
keeps dense/BM25 calls, RRF fusion, and existing debug assembly, but no longer
imports application-layer private helper functions directly.

Phase 5 isolates Phase7 table preview by default. Preview candidate loading and
debug remain available, but table preview chunks do not enter rerank input unless
`TABLE_PREVIEW_MERGE_ENABLED=true` is set explicitly for eval or experiment
runs. This does not promote preview table evidence to formal citation support.

Phase 6 separates the internal response model from the legacy `/v1/ask` DTO.
Generation v2 and pipeline stages now return `RAGPipelineResponse`, which does
not carry external-tool compatibility fields. `RAGApplicationService.ask()`
adapts that internal response to `RAGResponse` immediately before debug hiding,
audit logging, and API serialization.

## Response and Debug Surface

The external `/v1/ask` response keeps these fields:

- `session_id`
- `answer`
- `confidence`
- `route`
- `citations`
- `used_external_tool`
- `tool_name`
- `tool_result`
- `external_references`
- `debug`

`used_external_tool`, `tool_name`, `tool_result`, and `external_references` are
kept only for `/v1/ask` compatibility. The legacy response adapter sets them to
`False`, `None`, `None`, and `[]`.

When `include_debug=true`, the top-level debug keys are:

- `analysis_notes`
- `retrieved_count`
- `reranked_count`
- `seed_context_count`
- `final_context_count`
- `context_chars`
- `latency_ms`
- `seed_confidence`
- `final_confidence`
- `tenant_id`
- `hybrid_enabled`
- `bm25_enabled`
- `retrieval_hits`
- `rerank_hits`
- `neighbor_expansion`
- `original_cn_fallback`
- `table_preview`
- `parent_expansion`
- `filter_strategy`
- `generation_v2`
- `evidence_lifecycle_debug`
- `query_rewrite`

`generation_v2` remains the nested compatibility surface for answer planning,
support selection, citation binding, Qwen synthesis, summary selection,
comparison coverage, neighbor audit, candidate/support-pack diagnostics, and
evidence lifecycle data.

## Effective Env Contract

`Settings.from_env()` reads `.env` plus process environment variables. Process
environment values take precedence over `.env`; default dataclass values are used
when a supported key is absent.

The safe copy/edit contract is `config/settings.example.env`. It lists the
current supported runtime keys with defaults that should not change behavior when
copied as-is. Source-supported keys include:

- `APP_ENV`
- `SYNBIO_MILVUS_URI`, `MILVUS_URI`, `MILVUS_COLLECTION`
- `BGE_M3_MODEL_PATH`, `BGE_EMBED_MAX_LENGTH`, `BGE_RERANKER_MODEL_PATH`
- `QWEN_CHAT_API_BASE`, `QWEN_CHAT_API_KEY`
- `AUDIT_LOG_PATH`, `SESSION_STORE_PATH`
- `QUERY_REWRITE_*` and `EVAL_REWRITE_*`
- `GENERATION_V2_*`
- `RETRIEVAL_*` and `BIORAG_RERANK_MODE`
- `TABLE_PREVIEW_*`
- `TABLE_ENHANCEMENT_*`

`GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION` and
`GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN` are parsed only so the hard
guard can force them back to `false` with a warning.

## Retrieval Config Split

`RetrievalConfig` is now a migration compatibility shell around smaller config
objects:

- `core`: dense/BM25/vector-store basics, retrieval limits, fusion weights, and
  BM25 parameters.
- `comparison`: comparison query/subquery retrieval knobs.
- `rerank`: rerank sizing, mode, guarded rerank weights, and rerank aggregation.
- `evidence_boost`: title/table/figure/evidence/section boost weights.
- `same_doc`: same-doc body coverage, same-doc expansion, and source-floor
  policy.
- `alias_expansion`: query-time domain alias expansion settings.
- `context_expansion`: neighbor, parent, and protected-seed context expansion.
- `original_cn_fallback`: original Chinese query fallback settings.
- `table_preview`: Phase7 table preview settings.
- `index_contract`: structured index schema/type/search-parameter contract.

During the migration, existing callers may continue using
`settings.retrieval.<legacy_field>`. Reads and writes are delegated to the
appropriate sub-config. New code should prefer the grouped config when it is
already touching a specific policy boundary, but broad call-site migration is not
part of Phase 2.

## Deprecated or Ignored Env Keys

These keys are not read as active configuration and must not appear as active
assignments in `config/settings.example.env`:

- `GENERATION_VERSION`
- `GENERATION_V2_USE_EXTERNAL_TOOLS`
- `ROUND8_*`
- `QWEN_RERANK_API_BASE`
- `QWEN_RERANK_API_KEY`
- `RERANKER_SERVICE_URL`

`Settings.from_env()` now warns, but does not fail, when it sees deprecated keys
or unknown project-prefixed keys. This is a migration diagnostic only.

## Phase7 Table Preview

Phase7 table evidence remains preview/offline. Current defaults are:

- `TABLE_PREVIEW_ENABLED=true`
- `TABLE_PREVIEW_MERGE_ENABLED=false`
- `TABLE_PREVIEW_ALLOW_FORMAL_CITATION=false`

By default, preview candidates are shadow-only debug data and do not enter rerank
input. When `TABLE_PREVIEW_MERGE_ENABLED=true` explicitly adds preview chunks,
the pipeline can pass those chunks into rerank and run generation with
`v2_require_citation=false` for that answer. Formal table citation is still
blocked by `CitationBinder`; debug provenance paths, CSV paths, PDF crops, and
markdown cards are not formal citation sources.

Future work must decide separately whether preview should become eval-only merge
behind a stronger environment gate or a formal `TableEvidenceIndex`. Cleanup PRs
must not promote Phase7 preview into production citation support.
