# Phase7Q-1 Mapper Validation Report

- validation_status: `pass_with_warnings`
- record_count: 8
- mapped_count: 4
- blocked_count: 4
- pass_count: 8

Validation checks the mapped `TableEvidenceCitation` shape, formal/debug source separation, no value-level claim, `production_ready=false`, `preview_only`, expected blocked records, non-table query blocking, and normal chunk blocking.

| fixture_id | record_kind | validation_status | check_pass | block_reason | warning_reason |
| --- | --- | --- | --- | --- | --- |
| table_level_from_phase7m | mapped | pass_with_warnings | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| row_level_from_phase7m | mapped | pass_with_warnings | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| cell_group_from_phase7m | mapped | pass_with_warnings | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| csv_source_file_sanitized | mapped | pass_with_warnings | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| malformed_missing_table_id | blocked | blocked_expected | true | missing_table_id | - |
| malformed_value_scope | blocked | blocked_expected | true | citation_scope_value_forbidden;invalid_citation_scope | - |
| non_table_query_table_candidate | blocked | blocked_expected | true | non_table_query_blocks_table_citation | - |
| normal_chunk_not_mapped | blocked | blocked_expected | true | normal_chunk_not_table_evidence;non_table_query_blocks_table_citation | - |
