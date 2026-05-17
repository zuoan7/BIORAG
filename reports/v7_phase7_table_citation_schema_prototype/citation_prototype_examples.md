# Citation Prototype Examples

The JSONL examples are stored at:

- `data/experiments/v7_phase7_table_citation_schema_prototype/citation_prototype_examples.jsonl`

Each example contains a `schema_object`, expected validation status, block or warning reason, formal citation allowance, and debug-provenance-only flag.

| example_id | type | expected_validation_status | formal_citation_allowed | debug_provenance_only | reason |
| --- | --- | --- | --- | --- | --- |
| `phase7q_example_table_level` | table_level | pass_with_warnings | False | True | production_ready=false; index_unit_status=preview_only; value_bboxes_available=false; binding_review_level=warning |
| `phase7q_example_row_level` | row_level | pass_with_warnings | False | True | production_ready=false; index_unit_status=preview_only; value_bboxes_available=false; binding_review_level=warning |
| `phase7q_example_cell_group_level` | cell_group_level | pass_with_warnings | False | True | production_ready=false; index_unit_status=preview_only; value_bboxes_available=false; binding_review_level=warning |
| `phase7q_example_malformed_blocked` | malformed_blocked | blocked | False | False | citation_scope=value; canonical_source.source_file is CSV debug path; value-level claim requested |
| `phase7q_example_non_table_query_blocked` | non_table_query_blocked | blocked | False | False | non_table_query blocks table evidence citation |

The first three examples are structurally valid table, row, and cell-group citations, but they remain `pass_with_warnings` because current preview units are not production-ready formal citations.

The malformed example is blocked for value scope, CSV-as-formal-source, and value-level claim attempt. The non-table-query example is blocked by query context.
