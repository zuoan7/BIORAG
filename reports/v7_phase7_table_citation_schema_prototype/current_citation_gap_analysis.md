# Current Citation Gap Analysis

The current `Citation` dataclass in `src/synbio_rag/domain/schemas.py` has a normal text-chunk shape: `chunk_id`, `doc_id`, `title`, `source_file`, `section`, page range, score, and quote. `CitationBinder` builds this object from `CitationCandidate.source_file` after checking only basic text and metadata presence.

For table evidence, that shape is unsafe:

- `source_file` can confuse canonical paper source with debug paths such as CSV tables or PDF crops.
- There is no `citation_type`, so downstream code cannot distinguish table evidence from ordinary text evidence.
- There is no `citation_scope`, so table, row, cell-group, and forbidden value-level claims are indistinguishable.
- There is no table-specific scope: no `table_id`, `row_label`, selected `header_path`, or table page scope tied to the cited table.
- There is no `limitations` object to surface `production_ready=false`, `preview_only`, `value_bboxes_available=false`, or warning-level binding.
- There is no debug provenance layer separate from formal citation source.
- There is no way to express `value_bboxes_available=false` while still allowing a row or cell-group summary citation.
- There is no binding warning level in the public citation.
- There is no typed guard that blocks `preview_only` or `production_ready=false` table units from becoming formal citations.

Phase7M showed that debug metadata can flow through ledger/support/citation-candidate construction, but no formal citation was emitted. Phase7N and Phase7O kept CSV/crop paths debug-only and identified typed table citation as a blocker. Phase7P showed reranker score cannot be used as a production safety signal, so citation safety must live in schema and guard logic, not ranking.
