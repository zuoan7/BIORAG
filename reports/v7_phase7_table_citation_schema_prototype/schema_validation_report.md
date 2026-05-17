# Phase7Q Schema Validation Report

- validation_status: `pass_with_warnings`
- example_count: 5
- pass_count: 5
- blocked_count: 2
- pass_with_warnings_count: 3
- output: `results/v7_phase7_table_citation_schema_prototype/schema_validation_results.csv`

## Checks

The validator checks required fields, `citation_type=table_evidence`, legal citation scopes, no `value` scope, CSV/crop path exclusion from `canonical_source`, value-level claim blocking, `production_ready=false`, `index_unit_status=preview_only`, required limitations, and blocked example labels.

## Results

| example_id | actual_validation_status | formal_citation_allowed | check_pass | block_reason | warning_reason |
| --- | --- | --- | --- | --- | --- |
| phase7q_example_table_level | pass_with_warnings | false | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| phase7q_example_row_level | pass_with_warnings | false | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| phase7q_example_cell_group_level | pass_with_warnings | false | true | - | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| phase7q_example_malformed_blocked | blocked | false | true | invalid_citation_scope;citation_scope_value_forbidden;source_csv_path_in_canonical_source;canonical_source.source_file_looks_like_csv;value_level_citation_claim_allowed_not_false;value_bbox_false_but_value_claim_allowed | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |
| phase7q_example_non_table_query_blocked | blocked | false | true | non_table_query_blocks_table_citation | production_ready_false_blocks_formal_citation;preview_only_blocks_formal_citation;binding_warning_level;value_bboxes_unavailable |

## Decision

The schema prototype passes structural validation with warnings because every valid example remains `production_ready=false`, `index_unit_status=preview_only`, value bboxes are unavailable, and formal production citation remains blocked.
