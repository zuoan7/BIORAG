# Phase 5D-4 Chunk Static Diff

- baseline_chunk_count: 15802
- cleanup_chunk_count: 15738
- chunk_count_delta: -64
- table_focused_chunk_count before/after: 826 / 799
- figure_focused_chunk_count before/after: 2877 / 2868
- caption_only_false_chunk_count before/after: 36 / 0
- false_caption_focused_chunk_count before/after: 36 / 0
- demoted_retained_as_paragraph_or_noncaption_chunks: 36 / 46
- demoted_text_retained_in_parsed: 46 / 46
- demoted_metadata_ok_in_parsed: 46 / 46
- chunks_schema_same: true
- field_set_count baseline/cleanup: 1 / 1
- top_level_field_diff baseline_only: []
- top_level_field_diff cleanup_only: []
- normal_paragraph_chunk_count before/after: 11282 / 11254
- doc_0367_figure5_chunk_exists: true

## Interpretation
All demoted false/fragment caption blocks remain in parsed_clean with paragraph type and cleanup metadata. The 36 demoted blocks that appeared as caption-focused chunks in the baseline no longer appear as table_caption/figure_caption focused chunks in cleanup; 36 demoted blocks appear in cleanup as non-caption paragraph-style evidence.
