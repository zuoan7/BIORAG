# Phase 5D-4 Retrieval Smoke Summary

- retrieval_smoke_pass: true
- retrieval_method: in_memory_bm25
- top_k: 10
- main_denominator passed/total: 71 / 71
- table_caption_protection cleanup_hit/total: 25 / 25
- figure_caption_protection cleanup_hit/total: 26 / 26
- normal_control cleanup_hit/total: 20 / 20
- false_caption_noise cleanup_caption_focused_hit/total: 0 / 25
- doc_0367_figure5_cleanup_hit_top10: true

No full BM25 cache, Milvus index, generation eval, Qwen call, RAGAS, OCR, or parser rebuild was used.
