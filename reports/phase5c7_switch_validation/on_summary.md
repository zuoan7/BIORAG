# ON Mode Summary

Used `run_table_enhancement` with explicit ON config, then chunked the isolated enhanced parsed_clean output with the same preprocessing module.

- processed_docs: 8
- enhanced_parsed_clean_path: `/tmp/biorag_phase5c7_switch_on/parsed_clean_table_enhanced`
- chunks_path: `/tmp/biorag_phase5c7_switch_on/chunks/chunks.jsonl`
- audit_path: `/tmp/biorag_phase5c7_switch_on/audit`
- association_count: 57
- confidence distribution: {'high': 2, 'medium': 43, 'low': 12}
- accepted_long_prose: 0
- suspicious docs: ['doc_0090']
- table_related_metadata_count: 57
- chunks_schema_field_set_count: 1
- accepted associated block types: {'subsection_heading': 11, 'paragraph': 46}
- accepted_disallowed_block_type_count: 0
- whether_main_paths_touched: false
