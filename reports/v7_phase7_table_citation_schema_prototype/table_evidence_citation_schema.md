# TableEvidenceCitation Schema Prototype

`TableEvidenceCitation` is a typed prototype for table evidence citation. It is not a production replacement for the current `Citation` dataclass.

## Required Shape

The structured schema is stored at:

- `data/experiments/v7_phase7_table_citation_schema_prototype/table_evidence_citation_schema.json`

Top-level fields:

- `citation_type`: constant `table_evidence`.
- `citation_id`: prototype citation id.
- `doc_id`: canonical document id.
- `canonical_source`: formal citation source with paper title, canonical source file, DOI, and PMID.
- `table_scope`: table id, caption, and page range.
- `evidence_scope`: table unit type, formal citation scope, row label, header path, and source-span granularity.
- `quote`: bounded table/row/cell-group quote text.
- `provenance_debug`: CSV/crop/markdown/unit/seed/candidate traceability only.
- `limitations`: explicit readiness, bbox, binding, and value-claim limits.

## Formal Source Rule

`canonical_source` is the only formal citation source. `source_csv_path`, `source_pdf_crop_path`, and `source_markdown_path` belong only in `provenance_debug`.

`canonical_source.source_file` must not equal a CSV path, a PDF crop path, or any debug artifact path. If only debug paths are available, the prototype citation can be retained for debug but must not become a formal production citation.

## Scope Rule

Allowed `citation_scope` values are:

- `table`
- `row`
- `cell_group`

`value` is intentionally not allowed. While `value_bboxes_available=false`, `value_level_citation_claim_allowed` must remain `false`.

## Limitation Rule

Phase7Q keeps all preview table units non-production:

- `production_ready=false`
- `index_unit_status=preview_only`
- `value_bboxes_available=false`
- `binding_review_level=warning`
- `value_level_citation_claim_allowed=false`

These limitations must be visible to downstream consumers instead of being hidden in debug metadata.
