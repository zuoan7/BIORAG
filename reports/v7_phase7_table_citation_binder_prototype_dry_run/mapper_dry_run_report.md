# Phase7Q-1 Mapper Dry-Run Report

- input_count: 8
- mapped_count: 4
- blocked_count: 4
- expectation_pass_count: 8
- production_citation_count: 0
- answer_generated: false

| fixture_id | mapper_status | formal_citation_allowed | debug_provenance_only | expectation_pass | block_reason | warning_reason |
| --- | --- | --- | --- | --- | --- | --- |
| table_level_from_phase7m | mapped_with_warnings | false | true | true | - | canonical_source_file_unresolved;production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;value_bboxes_unavailable;binding_warning_level |
| row_level_from_phase7m | mapped_with_warnings | false | true | true | - | canonical_source_file_unresolved;production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;value_bboxes_unavailable;binding_warning_level |
| cell_group_from_phase7m | mapped_with_warnings | false | true | true | - | canonical_source_file_unresolved;production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;value_bboxes_unavailable;binding_warning_level |
| csv_source_file_sanitized | mapped_with_warnings | false | true | true | - | canonical_source_file_unresolved;production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;value_bboxes_unavailable;binding_warning_level |
| malformed_missing_table_id | blocked | false | false | true | missing_table_id | - |
| malformed_value_scope | blocked | false | false | true | citation_scope_value_forbidden;invalid_citation_scope | - |
| non_table_query_table_candidate | blocked | false | false | true | non_table_query_blocks_table_citation | - |
| normal_chunk_not_mapped | blocked | false | false | true | normal_chunk_not_table_evidence;non_table_query_blocks_table_citation | - |

The mapper is intentionally standalone and does not import or modify production `CitationBinder`. CSV/crop source paths are retained only under `provenance_debug`; unresolved canonical source files remain `null` rather than being replaced by debug paths.
