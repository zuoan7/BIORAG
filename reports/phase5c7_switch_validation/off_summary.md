# OFF Mode Summary

Used minimal script combination instead of `build_round1_kb.py` because the formal builder always proceeds through PDF conversion, chunking, and Milvus import; this phase explicitly forbids full Milvus/full BM25/full eval. The validation still uses the integrated default-off path by chunking the selected old parsed_clean input without running table enhancement.

- processed_docs: 8
- chunk_count: 204
- table_related_metadata_count: 0 (expected 0)
- table_related_chunk_count: 0 (expected 0)
- chunks_schema_field_set_count: 1
- used_input_dir: `/tmp/biorag_phase5c7_switch_validation/selected_parsed_clean`
- used_output_dir: `/tmp/biorag_phase5c7_switch_off`
- whether_main_paths_touched: false
