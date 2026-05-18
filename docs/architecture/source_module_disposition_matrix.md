# Source Module Disposition Matrix

This matrix records source module disposition after old generation removal.

## Summary

The runtime is v2-only. No source file may reintroduce
`GENERATION_VERSION=old`, `GenerationConfig.version`, or the deleted old
generation modules without a new explicit decision.

Highest-value remaining refactor candidates:

- `src/synbio_rag/domain/config.py`
- `src/synbio_rag/application/rerank_service.py`
- `src/synbio_rag/infrastructure/vectorstores/hybrid.py`
- `src/synbio_rag/application/parent_expansion.py`

## Application Layer

| Path | Disposition | Reason |
| --- | --- | --- |
| `src/synbio_rag/application/rag_service.py` | `keep_current` | Session/audit wrapper around pipeline. |
| `src/synbio_rag/application/pipeline.py` | `keep_current` | v2-only runtime orchestration. |
| `src/synbio_rag/application/evidence_selection_stage.py` | `keep_current` | v2 parent expansion, protected seeds, lifecycle debug. |
| `src/synbio_rag/application/generation_v2_response.py` | `keep_current` | v2 response/debug assembly. |
| `src/synbio_rag/application/generation_v2/` | `keep_current` | Only generation path. |
| `src/synbio_rag/application/rerank_service.py` | `keep_current` | Main-process local BGE reranker and guarded rerank logic. |
| `src/synbio_rag/application/parent_expansion.py` | `keep_current` | v2 context expansion. |
| `src/synbio_rag/application/summary_supplement.py` | `keep_current` | Summary Abstract/Conclusion supplement. |
| `src/synbio_rag/application/original_cn_fallback.py` | `keep_current_flagged` | Original-CN fallback behind retrieval flags. |
| `src/synbio_rag/application/query_rewrite_adapter.py` | `keep_current_flagged` | Query rewrite service setup. |
| `src/synbio_rag/application/table_preview.py` | `isolate_preview` | Phase7 preview sidecar, not formal citation. |
| `src/synbio_rag/application/table_preview_pipeline.py` | `isolate_preview` | Preview gate into v2 rerank input. |
| `src/synbio_rag/application/neighbor_expansion.py` | `keep_current` | Retained only as v2 neighbor-audit index source. |
| `src/synbio_rag/application/query_expansion.py` | `unknown_keep` | Prototype expansion; not part of current main chain. |

## Removed

| Path | Disposition | Reason |
| --- | --- | --- |
| `src/synbio_rag/application/generation_service.py` | `removed_old_generation` | Old answer generator. |
| `src/synbio_rag/application/context_builder.py` | `removed_old_generation` | Old prompt context builder. |
| `src/synbio_rag/application/legacy_generation_flow.py` | `removed_old_generation` | Old branch response flow. |
| `app/reranker_main.py` | `removed_reranker_http_service` | Standalone HTTP reranker service is no longer a supported call path. |
| `src/synbio_rag/infrastructure/reranker/client.py` | `removed_reranker_http_service` | HTTP reranker client is no longer a supported call path. |
| `src/synbio_rag/infrastructure/external_tools/literature_search.py` | `removed_old_generation` | Used only by old branch fallback. |
| `tests/test_round8_policy.py` | `removed_old_generation_test` | Tested old generator policy. |
| `tests/test_e2e_support_pack.py` | `removed_old_generation_test` | Tested old generator support-pack logic. |

## Remaining Cleanup Queue

1. `domain/config.py`
   - Split env parsing by group.
   - Preserve current v2 env names and defaults.

2. `application/rerank_service.py`
   - If it grows again, separate local model scoring from guarded rerank policy.

3. `infrastructure/vectorstores/hybrid.py`
   - Separate fusion mechanics from route/comparison/alias/source-floor policy.

4. `application/neighbor_expansion.py`
   - If neighbor audit remains, consider renaming/splitting the index-loading
     subset from the old `expand()` API in a later behavior-preserving PR.
