# Phase 5D Closeout Summary

## 1. Stage Goal

Phase 5D addressed conservative cleanup for parser false / fragment captions. The specific problem was that some Table/Figure captions were parser artifacts: short fragments such as split organism prefixes, single-letter continuation fragments, or page-header/footer strings. These false captions polluted table/figure evidence chunks, caption-level eval targets, risk slices, and future object-candidate inventories.

Phase 5D did not attempt OCR, pdfplumber/camelot/tabula parsing, structured table extraction, table_object or figure_object creation, row/cell-level understanding, Qwen generation eval, or final RAGAS evaluation.

## 2. Technical Approach

The cleanup capability is based on Phase 5D-1 audit and Phase 5D-2 sign-off. It only handles high-confidence `safe_to_demote` fragment cases and leaves all other short captions protected or skipped.

The implementation does not delete blocks. It preserves original text, demotes/retypes selected fragment captions to paragraph-style blocks, and records cleanup metadata:

- `original_block_type`
- `caption_cleanup_rule_id`
- `caption_cleanup_reason`

It does not modify chunk main fields, does not modify the `Chunk` dataclass, and does not write back to production `parsed_clean`. The validated path writes an experimental `parsed_clean` output and remains default-off.

## 3. Stage Results

### Phase 5D-1 Audit

- table_caption: 837
- figure_caption: 2898
- total captions: 3735
- false / fragment caption candidates: 178
- high-confidence false candidates: 75
- protected short captions: 1172
- main patterns:
  - very_short_no_semantic_anchor: 90
  - broken_organism_or_abbreviation_prefix: 41
  - number_only_caption: 30
  - supplementary_reference_fragment: 13
  - article_plus_single_letter_fragment: 4

### Phase 5D-2 Sign-Off

- safe_to_demote: 48
- keep_as_caption: 13
- needs_manual_pdf_check: 2
- eval_only_noise: 95
- uncertain: 20
- high-confidence candidates safe_to_demote: 43 / 75 = 57.3%
- protected short caption review:
  - correctly_protected: 59
  - uncertain: 1
  - should_be_candidate: 0

Conclusion: real short captions are common, so length-only cleanup is unsafe.

### Phase 5D-3 Cleanup Experiment

- cleanup experimental path completed: yes
- demoted_count: 46
- skipped protected: 0
- skipped uncertain: 20
- skipped eval_only_noise: 95
- skipped needs_manual_pdf_check: 2
- table_caption before/after: 837 / 800
- figure_caption before/after: 2898 / 2889
- protected violation: 0
- doc_0367 Figure 5 preserved: true
- parsed JSON readable: 708 / 708
- failed docs: 0
- chunk generation passed
- chunks: 15738
- schema field set count: 1
- Chunk dataclass unchanged: true
- chunks JSONL main fields unchanged: true
- table-focused chunks still exist: 799
- figure-focused chunks still exist: 2868
- doc_0367 Figure 5 chunk exists: true
- cleanup unit / CLI tests: 13 passed
- key regression tests: 76 passed

Rule hits:

- `phase5d3_broken_organism_or_abbreviation_prefix`: 39
- `phase5d3_article_plus_single_letter_fragment`: 4
- `phase5d3_page_header_footer_caption_fragment`: 3

### Phase 5D-4 Static Audit

- static audit passed
- acceptable_demote: 46
- questionable_needs_manual_check: 0
- likely_false_positive_demote: 0
- false_positive_demote: 0
- protected violation: 0
- approved eval target caption was not modified
- doc_0367 Figure 5 preserved
- chunks schema unchanged
- field set count: 1 / 1
- top-level fields unchanged
- false/fragment caption-focused chunks: 36 -> 0
- table-focused chunks: 826 -> 799
- figure-focused chunks: 2877 -> 2868
- demoted text was not written back to production `parsed_clean`
- chunk main fields unchanged

### Phase 5D-4S Retrieval Smoke

- method: in-memory BM25-only
- no full BM25 cache
- no Milvus
- no Qwen
- no generation eval / RAGAS / OCR
- top_k: 10
- main denominator: 71 / 71 passed
- table_caption_protection: 25 / 25 cleanup hit
- figure_caption_protection: 26 / 26 cleanup hit
- normal_control: 20 / 20 cleanup hit
- false_caption_noise cleanup caption-focused hit: 0 / 25
- doc_0367 Figure 5 cleanup top10 hit: true
- retrieval_smoke_pass: true

## 4. Current Conclusion

Phase 5D is accepted.

The cleanup can be kept as a default-off optional capability. No full index rebuild is required for this closeout. Do not enable cleanup by default. Do not write the experimental cleanup output back to production `parsed_clean` without an explicit rebuild and evaluation plan.

## 5. Known Limitations

- The cleanup only handles high-confidence fragment cases; it does not clean all short captions.
- Broad `number_only_caption` cleanup was not enabled automatically.
- Broad `very_short_no_semantic_anchor` cleanup was not enabled automatically.
- `eval_only_noise` remains primarily an eval/probe-layer concern.
- `uncertain` and `needs_manual_pdf_check` samples were not changed.
- Real short captions are common and still require protection rules.
- No OCR or structured table/figure parsing was performed.
- No row/cell-level table understanding was added.
- No table_object or figure_object schema was introduced.
