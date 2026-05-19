# Source Runtime Chain Inventory

This inventory records the v2-only `/v1/ask` runtime chain after old generation
removal.

## Entry Chain

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

Externally visible response fields remain:

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

## Pipeline Construction

`SynBioRAGPipeline.__init__()` constructs:

| Collaborator | Source | Purpose |
| --- | --- | --- |
| `BGEM3Embedder` | `infrastructure/embedding/bge.py` | Dense embedding adapter. |
| `QueryRouter` | `domain/router.py` | Query intent and retrieval sizing. |
| `MilvusRetriever` | `infrastructure/vectorstores/milvus.py` | Dense retrieval. |
| `BM25Retriever` | `infrastructure/vectorstores/bm25.py` | Lexical retrieval. |
| `RetrievalQueryPlanner` | `application/retrieval_query_planner.py` | Retrieval query variants, comparison subqueries, and CJK signal. |
| `AliasExpansionPolicy` | `application/alias_expansion_policy.py` | BM25-only controlled alias expansion. |
| `RetrievalPostProcessor` | `application/retrieval_postprocessor.py` | Retrieval boosts, comparison diversity, same-doc expansion, and source floor. |
| `HybridRetriever` | `infrastructure/vectorstores/hybrid.py` | Dense/BM25 retrieval and fusion using injected application policies. |
| `LocalBGERerankerService` | `application/rerank_service.py` | Main-process local BGE reranker and guarded rerank logic. |
| `GenerationV2Service` | `application/generation_v2/service.py` | Only answer generation path. |
| `ParentContextExpander` | `application/parent_expansion.py` | v2 parent/window/caption/page expansion. |
| `TablePreviewCandidateProvider` | `application/table_preview.py` | Phase7 preview-only table candidates; shadow-only by default. |
| `ConfidenceScorer` | `domain/confidence.py` | Confidence score from selected chunks. |
| `QueryRewriteService` | `rewrite/query_rewrite_service.py` | Feature-flagged query rewrite. |

`ChunkNeighborExpander` is instantiated only as a corpus index source for the
optional generation v2 neighbor audit engine. It is not an answer branch.

## Answer Flow

`SynBioRAGPipeline.answer()` executes:

1. Analyze query intent.
2. Rewrite retrieval query when query rewrite is enabled.
3. Retrieve chunks with filter fallback.
4. Optionally merge original Chinese query fallback candidates.
5. Optionally run Phase7 table preview. The default is shadow-only debug;
   explicit preview merge can add candidates before rerank.
6. Rerank candidates.
7. Select final seed chunks and annotate `rerank_rank`.
8. For summary queries, supplement Abstract/Conclusion chunks from top docs.
9. Run v2 parent/evidence selection.
10. Score seed/final confidence.
11. Run `GenerationV2Service`.
12. Build a v2 internal `RAGPipelineResponse`.
13. Adapt to legacy `RAGResponse` in `RAGApplicationService.ask()`.

There is no generation-version branch. `GenerationConfig.version` and
`GENERATION_VERSION` are no longer supported.

## Debug Surface

The v2 response debug payload keeps these top-level keys when debug is included:

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

`generation_v2` nested debug remains the compatibility surface for answer
planning, support selection, citation binding, Qwen synthesis, summary
selection, comparison coverage, neighbor audit, candidates, support pack, and
evidence lifecycle data.

## Removed Old Chain

The following old-only modules were removed:

- `src/synbio_rag/application/generation_service.py`
- `src/synbio_rag/application/context_builder.py`
- `src/synbio_rag/application/legacy_generation_flow.py`
- `src/synbio_rag/infrastructure/external_tools/literature_search.py`

The old Round8 generation policy tests and old support-pack tests were removed
with that source path.
