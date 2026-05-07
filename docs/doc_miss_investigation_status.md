# Doc Miss 问题排查与修复状态

**日期**: 2026-05-07 | **分支**: main | **数据**: smoke100 (100 条)

## 当前 Baseline (Phase 14F)

| 指标 | 值 |
|------|-----|
| total_P0 | 33 |
| doc_miss | 22 |
| route_mismatch | 11 |
| doc_hit_rate | 0.7474 |
| zero_citation | 0 |
| min_cit_pass | 0.97 |
| latency_p95 | ~1.9s |

## 已完成修复及效果

| Phase | 修复内容 | 修改文件 | 效果 |
|-------|---------|---------|------|
| **13E-1** | Negative query metadata fix (6 samples) | `smoke100.json`, `evaluate_ragas.py` | 评测噪声清除: 6 条 negative query 不再误判 doc_miss |
| **14D** | BM25 tokenizer: prime/Greek/Unicode normalization | `bm25.py` | 术语 token 正确: "2′-FL"→["2'-fl"], 但中文噪声仍主导 |
| **14E** | BM25 query-side CJK noise suppression | `bm25.py::tokenize_query` | BM25 top40 命中: 0→13/23 (focused), 但 smoke100 delta 仅 -1 P0 |
| **15A** | Protected rerank seeds in final_chunks | `config.py`, `pipeline.py` | Expected docs in final_chunks ✅, 但 citation 仍 0/11 |
| **15C** | Protected support seed selection | `config.py`, `support_selector.py` | 4/11 focused samples end-to-end fixed |

## 已修改的核心文件

| 文件 | 变更摘要 |
|------|---------|
| `src/synbio_rag/infrastructure/vectorstores/bm25.py` | Tokenizer: prime/Greek/Unicode + `tokenize_query()` CJK filter |
| `src/synbio_rag/domain/config.py` | `protect_rerank_seeds_*` + `v2_protect_support_seeds_*` |
| `src/synbio_rag/application/pipeline.py` | rerank_rank metadata + protected seed retention |
| `src/synbio_rag/application/generation_v2/support_selector.py` | `_ensure_protected_support_seeds` |
| `scripts/evaluation/evaluate_ragas.py` | negative_query skip_doc_hit |
| `data/eval/datasets/enterprise_ragas_smoke100.json` | negative_query metadata |
| `src/synbio_rag/domain/confidence.py` | None score safety (or 0.0) |

## 22 条 Doc Miss 症状分组 (Phase 14G + 15D 更新)

```
已修复 (4): ent_013, ent_040, ent_066, ent_077
  → 根因: support_selector 丢弃 rerank top seed → Phase 15C 修复

Support OK 但 Citation 丢 (2): ent_074, ent_086
  → 根因: plan=partial, expected doc 在 selected_support 但 citation_binder 丢弃
  → 待修: citation_binder

Expected doc 不在 Rerank (5): ent_005, 011, 055, 060, 100
  → 根因: hybrid/rerank 环节丢失 expected doc
  → 待修: reranker near-topic or hybrid fusion

Recall Hard Miss (约 6-8):
  → 根因: dense+BM25 都未找到 expected doc
  → 待修: dense recall improve or query/doc matching

Route Mismatch (3): ent_021, 035, 039
  → 根因: router 分类不匹配 expected_route
  → 待修: router audit
```

## 各检索阶段瓶颈总结

| 阶段 | 找到率 (22 doc_miss) | 瓶颈程度 |
|------|---------------------|---------|
| BM25 (CJK-filtered) | 11/22 | ✅ 已修复 (Phase 14E) |
| Dense | 16/22 | OK |
| Hybrid | 15/22 | 小瓶颈 (2 lost) |
| Rerank | 11/22 | 中瓶颈 (4 lost) |
| Final Context | 11/22 | ✅ 已修复 (Phase 15A) |
| Support Selection | 6/11 | ✅ 已修复 (Phase 15C) |
| Citation Binding | 0/11→4/11 | **剩余瓶颈** |

## 关键发现

1. **BM25 0→13 命中但 smoke100 doc_miss 只 -1**：BM25 候选层改善被后续 fusion/rerank/citation 逐层消耗——这是"候选保留"问题，不是"召回"问题。

2. **中文单字噪声是 BM25 致命缺陷**："的/中/为/是/和"等单字匹配数百文档产生 score 100+，压过英文术语 signal (score ~6-10)。移除后 BM25 恢复 13/23。

3. **Support_selector 的 sp_size=3 是瓶颈**：Phase 15C 在不变大 support_pack 的前提下插入 protected seeds，4/11 修复。

4. **Phase 14G "rerank 11/22" 包含过估计**：Phase 15D 发现 5 条 expected doc 实际不在 rerank output 中（exp_rerank_ranks 为空）。

5. **Qwen synthesis 不影响 P0/citation** (Phase 13B)：只增加 latency ~14s。

## 下一步修复优先级

| 优先级 | 行动 | 影响样本 |
|--------|------|---------|
| **P0** | Citation binder fix for partial-mode | ent_074, 086 |
| P1 | Reranker/hybrid audit for 5 hard losses | ent_005, 011, 055, 060, 100 |
| P2 | Dense recall improve for hard misses | ~6-8 samples |
| P3 | Route mismatch audit | ent_021, 035, 039 |
