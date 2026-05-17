# Phase7Q To Phase7Q-1 Delta

Phase7Q produced a typed schema prototype, mapping matrix, examples, and schema validation.

Phase7Q-1 adds a dry-run mapper around that schema:

- Reads existing Phase7M/7L/7P artifacts instead of hand-written examples only.
- Converts table-adapted candidate/debug records into `TableEvidenceCitation` prototype objects.
- Emits blocked records for malformed metadata, non-table query table candidates, and normal chunks.
- Keeps CSV/crop/markdown paths in `provenance_debug`.
- Leaves `canonical_source.source_file` null when only debug paths are available.
- Keeps all mapped objects `debug_provenance_only=true` and formal production citation blocked.

This still does not change production `CitationBinder`, the current `Citation` dataclass, retrieval, reranking, or answer generation.
