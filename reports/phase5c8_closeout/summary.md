# Phase 5C Closeout Summary

## 1. Phase Goal

Phase 5C addressed the table content loss behind `table_text=0`. The audit showed that most missing table content was already present in `parsed_clean` as paragraphs, subsection-like blocks, or list-like text, but was not represented as `table_text`. Phase 5C therefore used conservative table-like paragraph preservation and caption-nearby association to make that existing text available to table-focused chunks and improve `table_content` retrieval.

Phase 5C did not include OCR, pdfplumber/camelot/tabula extraction, structured table extraction, `table_object`, row/cell-level table understanding, image understanding, Qwen generation evaluation, or RAGAS final evaluation.

## 2. Technical Approach

The integrated implementation identifies table-like paragraph blocks within a conservative window around `table_caption` blocks. Accepted blocks receive metadata such as `table_related`, `table_related_type`, `associated_table_caption_block_id`, `associated_table_caption_text`, `association_confidence`, and rule-hit fields. Table captions also record associated block ids and association counts in caption metadata.

During chunking, table-related text can join table-focused chunks through the existing metadata path. The chunk top-level schema is unchanged, the `Chunk` dataclass is unchanged, and table association metadata is kept inside `source_block_metadata`. The feature is default-off through `TableEnhancementConfig.enabled=False`, supports explicit enablement, and supports dry-run audit mode.

## 3. Validation Results

Phase 5A/B Table Content Loss Audit:
- selected docs: 50
- table-like paragraph detected in 45/50 selected docs
- cleaning deletion/downgrade was 2/50 and not the main cause
- OCR candidate count was 0/50
- recommended conservative table-like paragraph preservation plus caption-nearby association

Phase 5C-1 Pilot:
- enhanced parsed_clean success: 50/50
- enhanced chunks success: 50/50, 1570 chunks
- table_related associations: 161
- table-related text reached 84 table-focused chunks
- chunk schema unchanged
- chunk delta: +12 / +0.77%
- token delta: -715 / -0.14%
- accepted_long_prose: 0

Phase 5C-2 / 5C-2M Retrieval A/B:
- table_content hybrid doc_hit@10: 0.625 -> 0.975
- table_content corrected_chunk_hit@10: 0.450 -> 0.975
- caption-level apparent regressions were target mapping artifacts
- corrected caption_level_table enhanced chunk_hit@10 recovered to 1.000
- normal_control did not regress
- no enhanced table-related takeover

Phase 5C-3 Representative Expansion:
- selected docs: 120
- chunks: 3739 -> 3766, +27 / +0.72%
- token delta: -1185 / -0.08%
- schema drift: false
- table_related associations: 256
- accepted_long_prose: 0
- table_content doc_hit@10: 0.543 -> 0.957
- table_content corrected_chunk_hit@10: 0.343 -> 0.957
- caption_level_table corrected_chunk_hit@10: 1.000 -> 1.000
- normal_control corrected_chunk_hit@10: 0.850 -> 0.850
- no table-related takeover

Phase 5C-4 Full-Corpus Preflight:
- total docs: 708
- dry-run safe
- table_related associations: 652
- confidence: high 107, medium 356, low 189
- accepted_long_prose: 0
- suspicious uncontrolled growth docs: 0
- enhanced parsed_clean/chunks success: 708/708
- chunks: 15802 -> 15902, +100 / +0.63%
- token delta: -3795 / -0.06%
- schema same: true
- BGE tokenizer >4096 chunks: 65 -> 65
- Milvus VARCHAR risk: 0

Phase 5C-5 Full Experimental Retrieval A/B:
- baseline/enhanced: 15802 -> 15902 chunks, Milvus rows, and BM25 records
- no main Milvus/BM25/chunks overwritten
- Milvus VARCHAR risk: 0
- BGE tokenizer truncation risk: 65 -> 65
- table_content doc_hit@10: 0.315 -> 0.908
- table_content corrected_chunk_hit@10: 0.238 -> 0.908
- caption_level_table corrected_chunk_hit@10: 0.914 -> 0.886, drop <0.05; both regressions were @10 near-misses within @20
- figure_caption corrected_chunk_hit@10: 0.860 -> 0.860
- normal_control doc_hit@10: 0.822 -> 0.822
- normal_control corrected_chunk_hit@10: 0.733 -> 0.733
- false_positive_retrieval_count: 1
- enhanced_table_related_takeover_count: 2, with no normal hit decline
- table_related occupancy: 0.010
- gate passed

Phase 5C-6B Default-Off Integration:
- added `src/synbio_rag/ingestion/table_enhancement.py`
- added `TableEnhancementConfig`
- wired `kb_builder` and `build_round1_kb.py`
- pilot script now reuses the formal module
- default off confirmed
- ON mode uses isolated enhanced parsed_clean
- dry-run writes audit and does not write enhanced parsed JSON
- audit outputs: `association_audit.csv`, `doc_level_stats.csv`, `summary.md`, `false_positive_review.md`
- chunks top-level fields unchanged
- tests passed: table enhancement 12, index contract 63, key regression 76
- smoke passed: off no table_related, on isolated output with associations, dry-run audit only

Phase 5C-7 Switch Validation:
- default-off: pass
- OFF / ON / DRY-RUN: pass
- ON association_count: 57
- chunks schema: same, top-level field diff empty
- main paths untouched
- py_compile passed
- table enhancement tests: 12 passed
- key regression tests: 76 passed
- full index/retrieval intentionally skipped because Phase 5C-5 already completed full A/B

Overall, table_content retrieval improved substantially. caption_level_table did not show a material true regression, figure_caption was unchanged, normal_control did not regress, table-related takeover risk was acceptable, schema did not drift, chunk/token growth was controlled, truncation risk did not increase, and default-off switch validation passed.

## 4. Merge Recommendation

Recommendation:
- Code can be merged with table enhancement default-off.
- Do not enable by default.
- Do not overwrite main parsed_clean / chunks / Milvus / BM25.
- Use explicit config or CLI flag to enable.
- Keep audit and safety gates.
- Run formal on/off rebuild before any production-like use.

中文建议：
- 建议合并代码。
- 默认保持关闭。
- 仅在显式配置或 CLI 参数下启用。
- 合并不等于正式启用。
- 后续正式使用前仍需按文档执行 on/off build 和 regression。

## 5. Known Limitations

- This is not structured table extraction.
- There is no row/cell-level table understanding.
- There is no OCR.
- Low-confidence associations still require audit monitoring.
- `false_positive_retrieval_count=1`.
- `enhanced_table_related_takeover_count=2`, but normal hit did not decline.
- caption_level_table had a small @10 near-miss decline below threshold.
- Parser false/fragment captions remain backlog.
- Section metadata cleanup remains backlog.

## 6. Rollback Strategy

Default-off is the primary rollback. Disabling `TableEnhancementConfig.enabled` or using `--no-table-enhancement` restores the old build path. ON outputs are isolated by suffix and should not overwrite main `parsed_clean`, chunks, Milvus, or BM25 paths. If an ON build fails, continue using the OFF baseline. If audit or safety gates fail, do not proceed to index build.

## 7. Next Steps

Phase 5C can close. Do not create a Phase 5C-9 for more table enhancement validation. Prepare merge with default-off. After merge, the next useful cleanup stages are parser false/fragment caption cleanup and section metadata cleanup. Structured table extraction should be opened later as a separate Phase 5D/6 effort, not mixed into Phase 5C.
