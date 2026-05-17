# Citation Mapping From RetrievedChunk

This matrix defines a prototype mapping from table-adapted `RetrievedChunk.metadata` and `CitationCandidate` fields into `TableEvidenceCitation`.

Formal citation fields identify the paper, table, page, and evidence scope. Debug fields preserve CSV/crop/markdown and table-index traceability only.

| target_field | source_field | citation_layer | mapping_rule |
| --- | --- | --- | --- |
| `doc_id` | `RetrievedChunk.doc_id or metadata.doc_id` | formal | Copy document identity; mismatch between chunk doc_id and metadata doc_id blocks. |
| `canonical_source.paper_title` | `RetrievedChunk.title after table-caption prefix stripping or paper metadata title` | formal | Use canonical paper title when available; table caption alone is not a paper title. |
| `canonical_source.source_file` | `RetrievedChunk.source_file only if it is canonical paper source` | formal | Must not copy source_csv_path or source_pdf_crop_path; debug paths block formal mapping. |
| `table_scope.table_id` | `metadata.table_id` | formal | Copy table id; missing value blocks table citation. |
| `table_scope.table_caption` | `metadata.caption or RetrievedChunk.title` | formal | Use caption for table scope, not paper identity. |
| `evidence_scope.row_label` | `metadata.row_label` | formal | Required for row and cell_group scopes; null allowed for table scope. |
| `evidence_scope.header_path` | `metadata.header_path` | formal | Flatten selected header hierarchy into string array for the cited table/row/cell group. |
| `table_scope.page_start` | `RetrievedChunk.page_start` | formal | Copy page start when available; null otherwise. |
| `table_scope.page_end` | `RetrievedChunk.page_end` | formal | Copy page end when available; null otherwise. |
| `evidence_scope.table_unit_type` | `metadata.table_unit_type` | formal guard | Allowed values are table_unit, row_unit, cell_group_unit. |
| `evidence_scope.citation_scope` | `derived from metadata.table_unit_type and query type` | formal guard | table_unit -> table; row_unit -> row; cell_group_unit -> cell_group; value is forbidden. |
| `quote.text` | `RetrievedChunk.text or metadata.retrieval_text` | formal quote | Use bounded table/row/cell-group summary text; no generated answer text. |
| `provenance_debug.source_csv_path` | `metadata.source_csv_path` | debug | Copy only into provenance_debug; never into canonical_source. |
| `provenance_debug.source_pdf_crop_path` | `metadata.source_pdf_crop_path` | debug | Copy only into provenance_debug; never into canonical_source. |
| `provenance_debug.table_index_unit_id` | `metadata.table_index_unit_id or chunk_id suffix` | debug | Trace prototype table unit identity. |
| `provenance_debug.seed_id` | `metadata.seed_id` | debug | Trace seed grouping only. |
| `provenance_debug.candidate_id` | `metadata.candidate_id` | debug | Trace extraction candidate only. |
| `limitations.production_ready` | `metadata.production_ready` | limitation guard | Phase7Q requires false; false blocks production formal citation. |
| `limitations.index_unit_status` | `metadata.index_unit_status` | limitation guard | Phase7Q requires preview_only; preview_only blocks production formal citation. |
| `limitations.value_bboxes_available` | `metadata.value_bboxes_available` | limitation guard | False forces value_level_citation_claim_allowed=false. |
| `limitations.binding_review_level` | `metadata.binding_review_limitation/reference_ok/unit_or_note_ok` | limitation guard | Current warning-level binding maps to warning and must be surfaced. |

Mapping blocks:

- If `metadata.doc_id` conflicts with `RetrievedChunk.doc_id`, block.
- If `source_csv_path` or `source_pdf_crop_path` would enter `canonical_source.source_file`, block.
- If `citation_scope=value`, block.
- If `production_ready=false` or `index_unit_status=preview_only`, block production formal citation and allow debug provenance only for otherwise valid examples.
- If query context is `non_table_query`, block table citation even if a reranker ranks table evidence highly.
