# Source Module Disposition Matrix

This matrix records the PR8 source-code review disposition for tracked modules
under `app/` and `src/synbio_rag/`. It is descriptive only and does not approve
source deletion.

## Disposition Vocabulary

- `keep_current`: current production or library path.
- `keep_current_flagged`: current path behind an explicit feature/config flag.
- `keep_legacy_supported`: old behavior that is still selectable or tested.
- `isolate_preview`: preview/offline path that should remain separated from
  formal production behavior.
- `refactor_candidate`: current behavior to keep, but source boundary is too
  broad or mixed.
- `delete_candidate`: safe deletion candidate after proof. None found in this
  review.
- `unknown_keep`: not enough evidence to delete.

## Summary

App/source files reviewed: 69 Python files plus one YAML resource.

| Disposition | Count | Meaning |
| --- | ---: | --- |
| `keep_current` | 41 | Current API, domain, infrastructure, ingestion helpers, or generation v2 internals. |
| `keep_current_flagged` | 7 | Current behavior behind config or optional service flags. |
| `keep_legacy_supported` | 4 | Old generation path still selectable/test-backed. |
| `isolate_preview` | 2 | Phase7 table preview sidecar and pipeline gate. |
| `refactor_candidate` | 4 | Keep behavior, split boundaries later. |
| `unknown_keep` | 12 | Package markers or low-information modules; no deletion proof. |
| `delete_candidate` | 0 | No safe source deletion candidate found. |

The highest-risk/highest-value refactor candidates are:

- `src/synbio_rag/application/pipeline.py`
- `src/synbio_rag/domain/config.py`
- `src/synbio_rag/application/rerank_service.py`
- `src/synbio_rag/infrastructure/vectorstores/hybrid.py`

## Application Entry

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `app/main.py` | 84 | `keep_current` | FastAPI `/v1/ask`, session lookup, and health entry. |
| `app/reranker_main.py` | 58 | `keep_current_flagged` | Optional standalone reranker service entry. |
| `src/synbio_rag/__init__.py` | 7 | `keep_current` | Public package exports used by `app/main.py`. |
| `src/synbio_rag/api/__init__.py` | 1 | `unknown_keep` | Empty package marker; harmless but not worth deleting alone. |

## Application Layer

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `src/synbio_rag/application/rag_service.py` | 51 | `keep_current` | Session/audit wrapper around pipeline. |
| `src/synbio_rag/application/pipeline.py` | 326 | `refactor_candidate` | Main orchestration still mixes construction, retrieval fallback, and generation dispatch after helper extraction. |
| `src/synbio_rag/application/rerank_service.py` | 1095 | `refactor_candidate` | Current rerank adapter plus fallback/guarded policies; too broad for one module. |
| `src/synbio_rag/application/parent_expansion.py` | 882 | `keep_current` | Current v2 parent/window/caption/page context expansion; heavily tested. |
| `src/synbio_rag/application/evidence_selection_stage.py` | 79 | `keep_current` | Extracted v2 parent expansion, protected seed, and evidence lifecycle stage. |
| `src/synbio_rag/application/generation_v2_response.py` | 89 | `keep_current` | Extracted v2 response/debug assembly. |
| `src/synbio_rag/application/summary_supplement.py` | 155 | `keep_current` | Extracted summary Abstract/Conclusion supplement helper. |
| `src/synbio_rag/application/original_cn_fallback.py` | 97 | `keep_current_flagged` | Extracted original-CN fallback helper behind retrieval config gates. |
| `src/synbio_rag/application/query_rewrite_adapter.py` | 53 | `keep_current_flagged` | Extracted query rewrite service and optional LLM client setup. |
| `src/synbio_rag/application/table_preview.py` | 471 | `isolate_preview` | Phase7 preview-only table sidecar; keep isolated from formal citation. |
| `src/synbio_rag/application/table_preview_pipeline.py` | 40 | `isolate_preview` | Extracted pipeline gate for Phase7 preview; preserves generation-v2-only guard. |
| `src/synbio_rag/application/legacy_generation_flow.py` | 95 | `keep_legacy_supported` | Extracted old generation branch while `GENERATION_VERSION=old` remains selectable. |
| `src/synbio_rag/application/context_builder.py` | 104 | `keep_legacy_supported` | Old generation context builder. |
| `src/synbio_rag/application/generation_service.py` | 1879 | `keep_legacy_supported` | Old generation path still selectable via `GENERATION_VERSION=old` and tested. |
| `src/synbio_rag/application/neighbor_expansion.py` | 171 | `keep_legacy_supported` | Old generation neighbor expansion and v2 neighbor audit index source. |
| `src/synbio_rag/application/query_expansion.py` | 173 | `unknown_keep` | Prototype expansion; no deletion proof in this review. |
| `src/synbio_rag/application/__init__.py` | 1 | `unknown_keep` | Package marker. |

## Generation V2

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `src/synbio_rag/application/generation_v2/service.py` | 340 | `keep_current` | Current default generation coordinator. |
| `src/synbio_rag/application/generation_v2/models.py` | 140 | `keep_current` | Generation v2 dataclasses and result types. |
| `src/synbio_rag/application/generation_v2/evidence_ledger.py` | 63 | `keep_current` | Converts retrieved chunks to evidence candidates. |
| `src/synbio_rag/application/generation_v2/support_selector.py` | 870 | `keep_current` | Support pack selection, summary policy, document diversity. |
| `src/synbio_rag/application/generation_v2/answer_planner.py` | 377 | `keep_current` | Answer mode and branch planning. |
| `src/synbio_rag/application/generation_v2/answer_builder.py` | 259 | `keep_current` | Extractive answer construction. |
| `src/synbio_rag/application/generation_v2/citation_binder.py` | 255 | `keep_current` | Formal citation binding and preview citation blocking. |
| `src/synbio_rag/application/generation_v2/validator.py` | 54 | `keep_current` | Final answer validation. |
| `src/synbio_rag/application/generation_v2/qwen_synthesizer.py` | 480 | `keep_current_flagged` | Optional Qwen synthesis behind v2 config. |
| `src/synbio_rag/application/generation_v2/comparison_coverage.py` | 507 | `keep_current_flagged` | Optional comparison coverage policy. |
| `src/synbio_rag/application/generation_v2/neighbor_audit.py` | 536 | `keep_current_flagged` | Dry-run neighbor audit; no mutation of support/citations. |
| `src/synbio_rag/application/generation_v2/branch_parser.py` | 198 | `keep_current` | Comparison branch parsing helper. |
| `src/synbio_rag/application/generation_v2/guardrails.py` | 335 | `keep_current` | Existence and support guardrails. |
| `src/synbio_rag/application/generation_v2/evidence_lifecycle_debug.py` | 257 | `keep_current` | Debug payload helpers; treated as compatibility surface. |
| `src/synbio_rag/application/generation_v2/__init__.py` | 3 | `keep_current` | Public generation v2 export. |

## Domain Layer

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `src/synbio_rag/domain/config.py` | 1012 | `refactor_candidate` | Dataclasses, env parsing, profiles, path resolution, and directory creation are mixed. |
| `src/synbio_rag/domain/schemas.py` | 93 | `keep_current` | Request/response/chunk/citation dataclasses. |
| `src/synbio_rag/domain/router.py` | 106 | `keep_current` | Query intent and retrieval sizing. |
| `src/synbio_rag/domain/confidence.py` | 28 | `keep_current` | Confidence policy. |
| `src/synbio_rag/domain/__init__.py` | 1 | `unknown_keep` | Package marker. |

## Infrastructure Layer

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `src/synbio_rag/infrastructure/embedding/bge.py` | 23 | `keep_current` | Dense embedding adapter. |
| `src/synbio_rag/infrastructure/vectorstores/milvus.py` | 145 | `keep_current` | Dense Milvus retrieval. |
| `src/synbio_rag/infrastructure/vectorstores/bm25.py` | 433 | `keep_current` | Lexical retrieval. |
| `src/synbio_rag/infrastructure/vectorstores/hybrid.py` | 798 | `refactor_candidate` | Current dense/BM25 fusion plus multiple retrieval policy patches. |
| `src/synbio_rag/infrastructure/reranker/client.py` | 19 | `keep_current` | Reranker service client wrapper. |
| `src/synbio_rag/infrastructure/reranker/local_bge.py` | 29 | `keep_current` | Local BGE reranker adapter. |
| `src/synbio_rag/infrastructure/clients/openai_compatible.py` | 62 | `keep_current` | OpenAI-compatible HTTP client used by LLM/rewrite/rerank. |
| `src/synbio_rag/infrastructure/external_tools/literature_search.py` | 114 | `keep_current` | External literature tool manager used by old branch. |
| `src/synbio_rag/infrastructure/index/parent_store.py` | 378 | `keep_current` | Parent index store for v2 expansion. |
| `src/synbio_rag/infrastructure/persistence/audit.py` | 37 | `keep_current` | Audit serialization/logging. |
| `src/synbio_rag/infrastructure/persistence/session_store.py` | 33 | `keep_current` | Session persistence. |
| `src/synbio_rag/infrastructure/__init__.py` | 1 | `unknown_keep` | Package marker. |
| `src/synbio_rag/infrastructure/clients/__init__.py` | 1 | `unknown_keep` | Package marker. |
| `src/synbio_rag/infrastructure/embedding/__init__.py` | 1 | `unknown_keep` | Package marker. |
| `src/synbio_rag/infrastructure/external_tools/__init__.py` | 1 | `unknown_keep` | Package marker. |
| `src/synbio_rag/infrastructure/index/__init__.py` | 3 | `keep_current` | Parent index public export. |
| `src/synbio_rag/infrastructure/persistence/__init__.py` | 1 | `unknown_keep` | Package marker. |
| `src/synbio_rag/infrastructure/reranker/__init__.py` | 6 | `keep_current` | Reranker public exports. |
| `src/synbio_rag/infrastructure/vectorstores/__init__.py` | 1 | `unknown_keep` | Package marker. |

## Ingestion And Evaluation Libraries

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `src/synbio_rag/ingestion/kb_builder.py` | 214 | `keep_current` | Reusable KB construction helper. |
| `src/synbio_rag/ingestion/cleaning_rules.py` | 481 | `keep_current` | Tested document-cleaning rules. |
| `src/synbio_rag/ingestion/caption_cleanup.py` | 550 | `keep_current` | Caption cleanup library used by CLI/tests. |
| `src/synbio_rag/ingestion/table_enhancement.py` | 817 | `keep_current` | Table enhancement library used by scripts/tests. |
| `src/synbio_rag/ingestion/__init__.py` | 1 | `unknown_keep` | Package marker. |
| `src/synbio_rag/evaluation/failure_taxonomy.py` | 170 | `keep_current` | Evaluation taxonomy library used by tests/scripts. |
| `src/synbio_rag/evaluation/__init__.py` | 1 | `unknown_keep` | Package marker. |

## Rewrite And Resources

| Path | Lines | Disposition | Reason |
| --- | ---: | --- | --- |
| `src/synbio_rag/rewrite/query_rewrite_service.py` | 291 | `keep_current_flagged` | Feature-flagged query rewrite service, default off. |
| `src/synbio_rag/rewrite/__init__.py` | 1 | `keep_current` | Public rewrite exports. |
| `src/synbio_rag/resources/retrieval_aliases_v1.yaml` | 148 | `keep_current` | Alias expansion resource used by retrieval policy. |

## Refactor Queue

Recommended order:

1. `domain/config.py`
   - Add golden tests around defaults and env overrides.
   - Extract env parsing by config group without renaming env vars.

2. `application/pipeline.py`
   - Remaining cleanup should focus on startup construction or filter fallback
     only if it reduces coupling without changing debug keys.

3. `application/rerank_service.py`
   - Separate service client adapter, local fallback, and guarded rerank policy.

4. `infrastructure/vectorstores/hybrid.py`
   - Separate fusion mechanics from route/comparison/alias/source-floor policy.

5. Legacy generation decision.
   - Keep, deprecate, or remove `generation_service.py`,
     `context_builder.py`, and old neighbor expansion as a deliberate decision.

## Deletion Review Result

No `delete_candidate` source files were found.

Reasons:

- old generation is still selectable and tested;
- Phase7 preview is active but preview-only;
- ingestion/evaluation helpers are test-backed or script-backed;
- package marker files are low value to delete and may affect imports;
- infrastructure modules are reached by the current pipeline or optional
  service entry.

Any future deletion should start from a narrower decision, not from this matrix.
