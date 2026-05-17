# Citation Guard Delta

The prototype schema adds guard surface that the current `CitationBinder` public citation cannot express.

| guard | current_binder_gap | schema_delta | phase7q_status |
| --- | --- | --- | --- |
| formal_source_debug_provenance_separation | Citation.source_file can hold a technical path. | canonical_source is formal; provenance_debug holds CSV/crop/markdown paths. | prototype_guard |
| csv_crop_path_formal_block | CSV path may appear in CitationCandidate.source_file debug. | Validator blocks CSV/crop equality or file extension in canonical_source.source_file. | prototype_guard |
| no_value_level_claim | No typed citation_scope exists. | citation_scope enum excludes value; value claim flag must be false. | prototype_guard |
| preview_only_production_ready_surface | Production readiness is only metadata/debug. | limitations explicitly exposes production_ready=false and index_unit_status=preview_only. | prototype_guard |
| binding_warning_surface | Warning-level binding is not present in Citation. | limitations.binding_review_level records warning/reviewed/verified. | prototype_guard |
| citation_scope_restriction | Citation has no table, row, or cell_group scope. | evidence_scope.citation_scope is table/row/cell_group only. | prototype_guard |
| malformed_metadata_block | Binder only checks chunk_id/doc_id/source_file/text presence. | Validator blocks missing fields, invalid scope, debug path in formal source, and value claims. | prototype_guard |
| non_table_query_block | CitationBinder itself has no query-type table evidence block. | Example context blocks table citation under non_table_query. | prototype_guard |

This is still a prototype delta. It does not change production binding behavior and does not enable formal table citation.
