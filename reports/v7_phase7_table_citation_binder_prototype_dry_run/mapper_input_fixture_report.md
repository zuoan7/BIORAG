# Phase7Q-1 Mapper Input Fixture

- fixture_count: 8
- mapped_expected_count: 4
- blocked_expected_count: 4

| fixture_id | fixture_type | query_type | object_type | table_unit_type | expected_mapper_status | expected_block_reason_contains |
| --- | --- | --- | --- | --- | --- | --- |
| `table_level_from_phase7m` | table_level | table_lookup | table_index_unit | table_unit | mapped_with_warnings | - |
| `row_level_from_phase7m` | row_level | row_lookup | table_index_unit | row_unit | mapped_with_warnings | - |
| `cell_group_from_phase7m` | cell_group_level | metric_lookup | table_index_unit | cell_group_unit | mapped_with_warnings | - |
| `csv_source_file_sanitized` | csv_source_file_sanitized | row_lookup | table_index_unit | row_unit | mapped_with_warnings | - |
| `malformed_missing_table_id` | malformed_missing_table_id | row_lookup | table_index_unit | row_unit | blocked | missing_table_id |
| `malformed_value_scope` | malformed_value_scope | row_lookup | table_index_unit | row_unit | blocked | citation_scope_value_forbidden |
| `non_table_query_table_candidate` | non_table_query_blocked | non_table_query | table_index_unit | table_unit | blocked | non_table_query_blocks_table_citation |
| `normal_chunk_not_mapped` | normal_chunk_not_mapped | non_table_query | normal_chunk |  | blocked | normal_chunk_not_table_evidence |
