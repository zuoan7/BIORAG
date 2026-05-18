# Source Runtime Chain Inventory

This inventory records the PR8A source architecture review for the current
`/v1/ask` runtime chain. It is descriptive only: no source behavior is changed
by this review.

## Scope

Reviewed paths:

- `app/main.py`
- `src/synbio_rag/application/rag_service.py`
- `src/synbio_rag/application/pipeline.py`
- `src/synbio_rag/application/generation_v2/`
- `src/synbio_rag/application/generation_service.py`
- `src/synbio_rag/application/context_builder.py`
- `src/synbio_rag/application/neighbor_expansion.py`
- `src/synbio_rag/application/parent_expansion.py`
- `src/synbio_rag/application/rerank_service.py`
- `src/synbio_rag/application/table_preview.py`
- `src/synbio_rag/application/table_preview_pipeline.py`
- `src/synbio_rag/application/summary_supplement.py`
- `src/synbio_rag/application/original_cn_fallback.py`
- `src/synbio_rag/application/query_rewrite_adapter.py`
- `src/synbio_rag/application/evidence_selection_stage.py`
- `src/synbio_rag/application/generation_v2_response.py`
- `src/synbio_rag/application/legacy_generation_flow.py`
- `src/synbio_rag/domain/config.py`
- `src/synbio_rag/domain/schemas.py`
- `src/synbio_rag/rewrite/query_rewrite_service.py`

Out of scope:

- source deletion;
- behavior changes;
- script cleanup;
- artifact cleanup;
- RAGAS, Qwen, embedding, rerank evaluation, retrieval evaluation, index builds,
  or model downloads.

## Entry Chain

Current `/v1/ask` runtime chain:

```text
app/main.py::ask()
  -> build QueryFilters from request fields
  -> RAGApplicationService.ask()
      -> choose/create session_id
      -> load session history
      -> SynBioRAGPipeline.answer()
      -> append user/assistant turns
      -> hide debug unless include_debug=true
      -> write audit log
  -> serialize RAGResponse to API dict
```

Externally visible response fields from `app/main.py`:

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

`include_debug=false` clears `response.debug` in
`RAGApplicationService.ask()` before audit serialization and API return.

## Pipeline Construction

`SynBioRAGPipeline.__init__()` eagerly constructs most collaborators:

| Collaborator | Source | Purpose |
| --- | --- | --- |
| `BGEM3Embedder` | `infrastructure/embedding/bge.py` | Dense embedding adapter. |
| `QueryRouter` | `domain/router.py` | Query intent, search limit, rerank top-k. |
| `MilvusRetriever` | `infrastructure/vectorstores/milvus.py` | Dense retrieval. |
| `BM25Retriever` | `infrastructure/vectorstores/bm25.py` | Lexical retrieval. |
| `HybridRetriever` | `infrastructure/vectorstores/hybrid.py` | Dense/BM25 fusion and retrieval patches. |
| `QwenReranker` | `application/rerank_service.py` | Rerank service/local fallback and guarded rerank modes. |
| `ContextBuilder` | `application/context_builder.py` | Old generation context builder. |
| `ChunkNeighborExpander` | `application/neighbor_expansion.py` | Old generation neighbor expansion and v2 neighbor audit index source. |
| `QwenChatGenerator` | `application/generation_service.py` | Old generation answer path. |
| `GenerationV2Service` | `application/generation_v2/service.py` | Current default generation path. |
| `ParentContextExpander` | `application/parent_expansion.py` | Current v2 parent/window/caption/page expansion. |
| `TablePreviewCandidateProvider` | `application/table_preview.py` | Phase7 preview-only table candidate provider. |
| `ConfidenceScorer` | `domain/confidence.py` | Confidence score from selected chunks. |
| `ExternalToolManager` | `infrastructure/external_tools/literature_search.py` | Old generation external tool fallback. |
| `QueryRewriteService` | `rewrite/query_rewrite_service.py` | Feature-flagged query rewrite. |

Review note: construction mixes current default behavior, old generation
support, preview-only behavior, and optional feature patches in one class.
That is the main cleanup pressure.

## Answer Flow

`SynBioRAGPipeline.answer()` currently executes these stages:

1. Analyze query intent with `QueryRouter.analyze(question)`.
2. Rewrite the retrieval query with `QueryRewriteService.rewrite(...)`.
3. Retrieve chunks with `_search_with_filter_fallback(...)`.
4. Optionally merge original Chinese query fallback candidates via
   `_run_original_cn_fallback(...)`.
5. Optionally run Phase7 table preview through `_run_table_preview(...)`.
6. Rerank retrieved candidates with `QwenReranker.rerank(...)`.
7. Select `seed_chunks` from top reranked candidates.
8. Annotate `rerank_rank` on seed chunk metadata.
9. For v2 summary queries, supplement top documents with abstract/conclusion
   chunks using `_supplement_summary_sections(...)`.
10. Branch on `settings.generation.version`.

Generation v2 branch:

```text
if settings.generation.version == "v2":
  -> ParentContextExpander.expand()
  -> protect top rerank seeds
  -> build evidence lifecycle debug
  -> ConfidenceScorer.score(seed_chunks/final_chunks)
  -> possibly relax v2_require_citation for table preview answers
  -> GenerationV2Service.run()
  -> suppress citations for negative intent
  -> return RAGResponse with v2 debug payload
```

Old generation branch:

```text
else:
  -> ChunkNeighborExpander.expand(seed_chunks)
  -> ContextBuilder.build()
  -> QwenChatGenerator.assess_evidence()
  -> QwenChatGenerator.generate()
  -> ConfidenceScorer.score()
  -> ExternalToolManager.run_if_needed()
  -> QwenChatGenerator.build_citations()
  -> suppress citations for negative intent
  -> QwenChatGenerator.validate_generated_answer()
  -> return RAGResponse with old debug payload
```

## Branch Conditions

| Branch | Condition | Default | Cleanup meaning |
| --- | --- | --- | --- |
| Generation v2 | `settings.generation.version == "v2"` | yes | Production-current branch. |
| Old generation | any generation version other than `v2`; tested with `old` | no | Legacy-supported until explicit decision. |
| Query rewrite | `settings.query_rewrite.mode` via `QueryRewriteMode` | `off` | Feature-flagged. |
| Original CN fallback | `retrieval.original_cn_fallback_enabled` and rewrite/CJK gates | off | Retrieval patch, not independent product flow. |
| Table preview | generation v2 plus retrieval table preview config | on by default | Preview-only; formal citation remains blocked. |
| Summary supplement | generation v2, summary intent, nonempty seeds | route-dependent | Extracted helper; preserve seed order/debug shape. |
| Parent expansion | generation v2 branch, expander config gates | on by default | Current v2 context expansion. |
| Neighbor expansion | old branch only for answer context | on by default | Legacy branch behavior. |
| Neighbor audit | v2 service with audit engine and v2 flag | off | Dry-run diagnostic only. |
| External tools | old generation branch only in pipeline | off by confidence/analysis | Not used by v2 pipeline path. |

## Debug Payload Surface

Both branches expose these top-level debug keys when `include_debug=true`:

- `query_rewrite`
- `analysis_notes`
- `retrieved_count`
- `reranked_count`
- `seed_context_count`
- `final_context_count`
- `context_chars`
- `latency_ms`
- `tenant_id`
- `hybrid_enabled`
- `bm25_enabled`
- `retrieval_hits`
- `rerank_hits`
- `original_cn_fallback`
- `table_preview`
- `filter_strategy`

Generation v2 additionally exposes:

- `seed_confidence`
- `final_confidence`
- `neighbor_expansion`
- `parent_expansion`
- `generation_v2`
- `evidence_lifecycle_debug`

Old generation additionally exposes:

- `neighbor_expansion`
- `evidence_quality`

`generation_v2` nested debug includes answer planning, support selection,
citation binding, qwen synthesis, summary selection, comparison coverage,
neighbor audit, candidates, support pack, and evidence lifecycle data. Tests
assert many of these nested keys, so debug payload shape should be treated as
part of the compatibility surface during refactors.

## Current Seams

Behavior-preserving extraction seams found in `pipeline.py`:

| Current function/concern | Current owner | Target owner | Status | Risk |
| --- | --- | --- | --- | --- |
| Query rewrite client setup | imported `_build_query_rewrite_llm_client` alias | `application/query_rewrite_adapter.py` | extracted | medium: eval cache/fail-fast behavior |
| Filter fallback retrieval | `_search_with_filter_fallback` | keep in pipeline or move to `retrieval_flow.py` | pending | medium: filter debug keys |
| Original CN fallback | imported `_run_original_cn_fallback` alias | `application/original_cn_fallback.py` | extracted | medium: candidate ordering |
| Table preview gate | imported `_run_table_preview` alias | `application/table_preview_pipeline.py` | extracted | high: preview must not become formal citation |
| Summary section supplement | imported `_supplement_summary_sections` alias | `application/summary_supplement.py` | extracted | medium: seed order/context changes |
| Generation v2 response assembly | v2 branch response block | `application/generation_v2_response.py` | extracted | high: v2 debug shape |
| Generation v2 evidence selection | v2 branch parent/protection block | `application/evidence_selection_stage.py` | extracted | medium: evidence lifecycle debug shape |
| Legacy generation flow | old branch body | `application/legacy_generation_flow.py` | extracted | high: old branch compatibility |

First behavior-preserving split completed:

- `src/synbio_rag/application/summary_supplement.py`
- `src/synbio_rag/application/original_cn_fallback.py`
- `src/synbio_rag/application/table_preview_pipeline.py`
- `src/synbio_rag/application/query_rewrite_adapter.py`
- `src/synbio_rag/application/evidence_selection_stage.py`
- `src/synbio_rag/application/generation_v2_response.py`
- `src/synbio_rag/application/legacy_generation_flow.py`

`pipeline.py` keeps the previous private helper names as imported aliases, so
existing focused tests and private imports continue to work.

## Review Findings

1. No source module is currently a safe deletion candidate.

   `generation_service.py` is large and legacy-looking, but tests and policy
   still treat `GENERATION_VERSION=old` as selectable. Delete only after an
   explicit old-generation decision.

2. `pipeline.py` remains the highest-value refactor target.

   It still combines pipeline construction, retrieval fallback, and generation
   version dispatch. The split moved query rewrite setup, summary supplement,
   original-CN fallback, table preview gate, v2 evidence/response assembly, and
   legacy generation flow into dedicated modules without changing the public
   runtime chain.

3. `domain/config.py` is the second highest-value refactor target.

   It contains dataclasses, 100 distinct env keys, profile handling, forbidden
   flag enforcement, path resolution, and directory creation. Future extraction
   should preserve every env name and default.

4. Phase7 table preview is current but preview-only.

   It is enabled by default and can enter rerank input for table-like queries,
   but citations must remain debug/provenance-only. Refactors must keep
   `table_preview_allow_formal_citation` default false and preserve citation
   guard tests.

5. Generation v2 is modular internally, but service-level helpers still contain
   patch history.

   `GenerationV2Service.run()` delegates to ledger, selector, planner, builder,
   synthesizer, binder, validator, and neighbor audit. The remaining local
   helpers are limited support pack/entity fallback logic.

## Recommended Next PRs

1. Config boundary extraction plan or tests first.
   - Add golden tests around env defaults and selected env overrides before
     moving parsing code.

2. Continue extracting one pipeline seam at a time.
   - Remaining candidates: startup construction helpers or filter fallback.
   - Preserve function signatures and debug keys first.

3. Decide old generation status.
   - Keep, deprecate, or remove.
   - Until that decision, classify old generation as `keep_legacy_supported`.

4. Only after those steps consider deletion.
   - Current review found no deletion-safe source file.

## Verification Used For This Review

Review and verification commands:

```bash
git status --short --branch
find src/synbio_rag app -maxdepth 4 -type f | sort
wc -l app/main.py src/synbio_rag/application/*.py src/synbio_rag/domain/config.py
rg -n "GenerationConfig|generation.version|table_preview|query_rewrite" src app tests
rg -n "version=\"old\"|GENERATION_VERSION.*old|generation.version == \"old\"" README.md docs tests app src
pytest tests/test_pipeline_summary_supplement.py tests/test_phase20l_original_cn_fallback.py tests/test_phase21a9c_query_rewrite_wiring.py tests/test_phase7t_table_preview_scaffold.py tests/test_phase7w_slim_mainchain_preview.py tests/test_phase7x_final_default_on_table_preview.py -q
pytest tests/test_generation_v2.py tests/test_parent_expansion.py tests/test_phase7t_table_preview_scaffold.py tests/test_phase7w_slim_mainchain_preview.py tests/test_phase7x_final_default_on_table_preview.py tests/test_phase21a9c_query_rewrite_wiring.py tests/test_phase20l_original_cn_fallback.py tests/test_pipeline_summary_supplement.py -q
pytest --collect-only -q
```

Source code was edited only to move isolated helper implementations out of
`pipeline.py`; behavior and compatibility wrappers were preserved.
