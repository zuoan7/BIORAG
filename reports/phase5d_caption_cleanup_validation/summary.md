# Phase 5D-4 Caption Cleanup Validation Summary

- static_audit_pass: true
- acceptable_demote: 46 / 46
- questionable_needs_manual_check: 0
- likely_false_positive_demote: 0
- protected_captions_all_preserved: true
- protected_violation_count: 0
- doc_0367_figure5_preserved: true
- chunks_schema_same: true
- false_fragment_caption_focused_chunks_reduced: true
- false_caption_focused_chunk_count before/after: 36 / 0
- retrieval_smoke_pass: true
- table_caption_protection_no_obvious_regression: true
- figure_caption_protection_no_obvious_regression: true
- normal_control_no_obvious_regression: true
- phase5d4_pass: true
- recommend_enter_phase5d5: true
- phase5d5_index_recommendation: continue experimental path; do not overwrite main Milvus/BM25. A full experimental index is only needed if Phase 5D-5 explicitly broadens beyond this smoke.
- cleanup_default_off_should_remain: yes

## Answered Gates
1. Static audit passed.
2. Demoted 46 rows: 46 acceptable, 0 questionable, 0 likely false positive.
3. Protected captions preserved: true.
4. doc_0367 Figure 5 preserved: true.
5. Chunk schema preserved: true.
6. False/fragment caption focused chunks reduced from 36 to 0.
7. Retrieval smoke passed: true.
8. Table/figure caption protection and normal controls show no obvious regression.
9. Recommendation: proceed to Phase 5D-5 on the experimental path only; keep cleanup default-off.
