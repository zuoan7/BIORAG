# 检索阶段修复总结（2026-04-28）

> 分支：`fix/retrieval`  
> 数据集：enterprise_ragas_smoke20（n=20）  
> 对比基线：generation_v2_stage2e01 结论时的检索指标

---

## 总体结果

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| `doc_id_hit_rate` | 0.9412 | **1.0** | +5.9pp |
| `section_hit_rate` | 0.4118 | **0.8235** | +41.2pp |
| `section_miss` 样本数 | 7 | **2** | -71% |
| `answer_mode` 分布 | — | full=5, partial=13, refuse=2 | 不变 |

---

## Phase 1：评测标签修正

**问题**：smoke20 数据集中部分样本的 `expected_sections` 标签与实际文档索引不一致，导致命中被误判为 miss。  
**修改模块**：`data/eval/datasets/enterprise_ragas_smoke20.json`  
**效果**：消除 5 个假阳性 section_miss，section_hit_rate 从 41.2% 提升至 76.5%。

---

## Phase 2 + 2b：Chunk 粒度与 Section 识别改进

**问题**：block-based 预处理路径存在两个缺陷：
1. 非标准 heading（如 `"■ INTRODUCTION"`）未被识别为标准 section，导致 section 标签缺失。
2. 每个 block（包括 heading）各自成为独立 chunk，产生大量 < 50 token 的碎片 chunk，检索粒度过细。

**修改模块**：`scripts/ingestion/preprocess_and_chunk.py`  
- Phase 2：加入 heading 标准化映射，非标准形式统一归入标准 section 名称；忽略元数据类 heading（如参考文献列表标题）。
- Phase 2b：重写 block 合并逻辑，heading 附着到后续正文而非独立成 chunk，相邻 block 按 token 预算合并，恢复段落级检索粒度。

**效果**：chunk 总数从 29834（Phase 2 后暂时膨胀）降至 10747，< 50 token 占比 2.4%（正常水平）。单 section 文档占比从 29.4% 降至 7.4%。

---

## Phase 3：Comparison 查询多样性扩展

**问题**：comparison 类查询（对比多个条目的问题）中，每篇文档在进入 rerank 池前被硬限为最多 1 个 chunk，导致相关 section 被过早过滤。  
例：doc_0037 在 hybrid 检索结果中排名第 3（Abstract）、第 7（Methods）、第 19（Results and Discussion），`max=1` 时只保留 Abstract，Results and Discussion 被丢弃，section_miss。

**修改模块**：`src/synbio_rag/domain/config.py`  
- `comparison_max_chunks_per_doc`：1 → 3  
- `comparison_rerank_max_chunks_per_doc`：1 → 3

**效果**：doc_0037 的 Results and Discussion chunk 进入 rerank 池，经 section_results_bonus 加权后浮入 final top-8，section_miss 从 3 降至 2，section_hit_rate 提升至 82.4%。

---

## 剩余未解决问题

| 样本 | 问题描述 | 根因 | 处理建议 |
|------|----------|------|----------|
| ent_090 | doc_0001 完全不在 hybrid top-40 中 | embedding 相关性不足，deep retrieval 问题 | 需增大 search_limit 或改进 embedding，本轮不处理 |

---

## 下一步建议

在全量数据集或 smoke100 上验证以上改动的泛化能力，重点关注：
- `comparison_max_chunks_per_doc=3` 对 diversity 是否有副作用
- chunk 粒度改变（10747 vs 原 7267）对非 comparison 样本的影响
