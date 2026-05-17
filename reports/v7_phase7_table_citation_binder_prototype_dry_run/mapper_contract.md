# Phase7Q-1 Mapper Contract

## Input

The mapper consumes a dry-run fixture row with:

- `query_id` and `query_type`
- expected mapper status
- a table-adapted `retrieved_chunk` object with `chunk_id`, `doc_id`, `source_file`, text/page fields, and `metadata`

The fixture is built from existing Phase7M/7L/7P artifacts. It does not query retrieval stores or rerun ranking.

## Output

Each input becomes either:

- a `TableEvidenceCitation` prototype object with `mapper_status=mapped_with_warnings`; or
- a blocked record with `mapper_status=blocked` and structured `block_reasons`.

No output is a production citation. All mapped records remain formal-citation blocked because Phase7 table units are still `production_ready=false` and `index_unit_status=preview_only`.

## Mapping Rules

- `canonical_source` is formal source only. If `RetrievedChunk.source_file` is a CSV or crop path, the mapper does not copy it into `canonical_source.source_file`.
- `provenance_debug` receives CSV/crop/markdown paths and table-index trace ids.
- `table_unit_type` maps to citation scope: `table_unit -> table`, `row_unit -> row`, `cell_group_unit -> cell_group`.
- `citation_scope=value` is forbidden.
- `query_type=non_table_query` blocks table citation.
- `object_type != table_index_unit` blocks table citation.
- Missing `doc_id`, `table_id`, `table_unit_type`, or text blocks mapping.
- `production_ready=false`, `index_unit_status=preview_only`, `value_bboxes_available=false`, and warning-level binding are surfaced as limitations and warnings.

## Block Modes

- `normal_chunk_not_table_evidence`
- `non_table_query_blocks_table_citation`
- `missing_doc_id`
- `missing_table_id`
- `invalid_table_unit_type`
- `missing_quote_text`
- `citation_scope_value_forbidden`
- `invalid_citation_scope`

## Warning Modes

- `canonical_source_file_unresolved`
- `production_ready_false_blocks_formal_citation`
- `preview_only_blocks_formal_citation`
- `value_bboxes_unavailable`
- `binding_warning_level`
