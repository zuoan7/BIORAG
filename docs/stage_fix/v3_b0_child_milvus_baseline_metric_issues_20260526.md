# v3 B0 child Milvus 最新 baseline 指标问题说明

## 基准声明

本说明以 2026-05-26 在新重建 child Milvus 向量库上完成的 B0 全量评测作为最新 baseline。

- run id: `20260526_v3_b0_child_milvus_rewrite_enabled_deepseek_answer`
- result dir: `results/evaluation/v3_baseline_b0_20260526_v3_b0_child_milvus_rewrite_enabled_deepseek_answer`
- report dir: `reports/evaluation/v3_baseline_b0_20260526_v3_b0_child_milvus_rewrite_enabled_deepseek_answer`
- dataset: `data/eval/datasets/v3_baseline_dataset.jsonl`
- 样本数: all200
- 变体: `b0_stable`
- query rewrite: `enabled`
- answer model: `deepseek-v4-flash`
- RAG answer concurrency: `1`
- judge concurrency schedule: `[10, 5, 2]`
- Milvus collection: `synbio_papers`
- Milvus 数据口径: `data/paper_round1/chunks/child_chunks.jsonl`, 57396 rows

对照旧 run:

- old run id: `20260525_v3_b0_rewrite_enabled_deepseek_answer`
- 旧 run 不是 child Milvus 口径，不能继续作为后续修改 baseline。
- 后续修改、验收和回归对比优先使用本次 child Milvus B0 结果。

## 最新 baseline 核心指标

| 指标 | All200 |
|---|---:|
| doc recall@10 | 0.839474 |
| doc hit@10 | 86.3158% |
| strict doc all hit@10 | 81.5789% |
| doc MRR@10 | 0.761065 |
| parent hit@10 | 75.5% |
| parent MRR@10 | 0.563407 |
| exact child hit@10 | 20.0% |
| exact child MRR@10 | 0.141673 |
| matched child at retrieval | 83.5714% |
| matched child at final | 72.8571% |
| matched child at support | 67.8571% |
| matched child at citation | 67.6923% |
| citation parent hit | 66.3158% |
| citation child evidence hit | 67.6923% |
| answer correctness pass | 17.0% |
| faithfulness pass | 92.0% |
| citation pass | 87.0% |
| critical error rate | 14.0% |

Actionable198 口径排除 `v3_pc_018`、`v3_pc_153`，保留 `v3_pc_148`、`v3_pc_149`。核心趋势与 all200 一致:

| 指标 | Actionable198 |
|---|---:|
| doc recall@10 | 0.843085 |
| doc hit@10 | 86.7021% |
| parent hit@10 | 75.7576% |
| parent MRR@10 | 0.564047 |
| matched child at retrieval | 83.4532% |
| matched child at support | 67.6259% |
| citation child evidence hit | 67.4419% |
| answer correctness pass | 17.1717% |
| faithfulness pass | 91.9192% |
| citation pass | 86.8687% |

## 相比旧库的主要变化

| 指标 | 旧库 B0 | 新 child Milvus B0 | 变化 |
|---|---:|---:|---:|
| doc recall@10 | 0.847368 | 0.839474 | -0.007894 |
| doc hit@10 | 87.3684% | 86.3158% | -1.0526pp |
| doc MRR@10 | 0.768106 | 0.761065 | -0.007041 |
| parent hit@10 | 74.0% | 75.5% | +1.5pp |
| parent MRR@10 | 0.548960 | 0.563407 | +0.014447 |
| exact child hit@10 | 22.5% | 20.0% | -2.5pp |
| matched child at retrieval | 66.4286% | 83.5714% | +17.1428pp |
| matched child at support | 57.1429% | 67.8571% | +10.7142pp |
| citation child evidence hit | 59.2308% | 67.6923% | +8.4615pp |
| citation parent hit | 66.3158% | 66.3158% | 0 |
| answer correctness pass | 14.0% | 17.0% | +3.0pp |
| faithfulness pass | 90.5% | 92.0% | +1.5pp |
| citation pass | 89.5% | 87.0% | -2.5pp |

结论: 新 child Milvus 不是整体退化。它牺牲了一点 doc-level 宽召回，但提升了 parent 排名和 child evidence 保留。当前问题不是简单“child index 无效”，而是 child-only 召回不适合作为所有 route 的唯一 dense 入口。

## 暴露出的指标问题

### P0: 模型答案大量回退

虽然本阶段优先分析检索和证据链路，但端到端系统仍暴露出明显生成链路问题。

- enabled rate: 100.0%
- attempted rate: 100.0%
- used model rate: 43.0%
- fallback rate: 57.0%
- answer source: `fallback_extractive: 114`, `model_generated: 86`
- fallback reason: `validation_failed: 113`, `client_error: 1`

问题判断:

- DeepSeek 大部分请求不是没有尝试，而是生成后未通过系统校验。
- `validation_failed` 是后续提升 answer pass 的第一定位入口。
- 在修检索前，不应把 answer correctness 低分直接归因于 child Milvus。

### P1: child-only dense 召回削弱宽语义问题

doc-level 指标相对旧库小幅下降:

- doc recall@10: -0.007894
- doc hit@10: -1.0526pp
- doc MRR@10: -0.007041

下降集中在宽语义 route/category:

| 类别 | doc hit 变化 | parent hit 变化 |
|---|---:|---:|
| comparison | -13.3333pp | -13.3334pp |
| cross_lingual | -10.0pp | -10.0pp |
| summary_review | -6.6667pp | -20.0pp |
| figure_caption | -6.25pp | +3.125pp |

问题判断:

- parent/chunk 向量承载更宽上下文，更适合 comparison、summary、cross-lingual。
- child 向量更局部，适合精确证据定位，但不天然适合文档级 broad recall。
- 当前 topK 预算没有随着 child 粒度变细而调整，可能导致正确 doc 被更多 child 候选竞争掉。

### P2: parent 命中率仍是主要上限

最新 baseline:

- parent hit@10: 75.5%
- parent MRR@10: 0.563407
- doc miss: 26 条
- doc hit 但 parent chunk miss: 21 条

问题判断:

- 约四分之一样本还不能稳定拿到正确父块。
- parent hit 是 support/citation 和最终答案的硬上限。
- 需要优先区分两类问题:
  - 正确 doc 未进入 top10。
  - 正确 doc 已进入，但 gold parent 未进入 top10。

### P3: child evidence 命中后仍在链路中流失

child evidence 链路相对旧库明显提升，但从 retrieval 到 citation 仍有下降:

| 阶段 | 命中率 |
|---|---:|
| matched child at retrieval | 83.5714% |
| matched child at final | 72.8571% |
| matched child at support | 67.8571% |
| matched child at citation | 67.6923% |

诊断桶:

- parent hit 但 support miss: 18 条
- parent hit 但 citation miss: 17 条

问题判断:

- child 检索已经提供了更好的证据定位信号。
- 主要流失发生在 materialize parent、final context、support selector、citation binder 之间。
- 后续优化不能只看 retrieval top10，还要检查 matched child metadata 是否被稳定传递和使用。

### P4: exact child hit 指标会低估真实 child evidence 链路

最新 baseline:

- exact child hit@10: 20.0%
- matched child at retrieval: 83.5714%
- citation child evidence hit: 67.6923%

问题判断:

- 当前链路会把 child hit materialize 回 parent，并通过 `matched_child_chunk_ids` 保存 child 证据信息。
- `exact_gold_chunk_hit@10` 只检查 top10 chunk id 是否精确等于 gold child id，因此会低估 child evidence 是否进入链路。
- 后续判断 child index 效果时，应优先看 `matched_child_*` 和 `citation_child_evidence_hit`，不要单独用 exact child hit 做结论。

### P5: route/category 缺少分流策略

新 child Milvus 对局部事实类更有利:

| 类别 | doc hit 变化 | parent hit 变化 | child support 变化 |
|---|---:|---:|---:|
| caption_level_table | +10.7142pp | +14.2857pp | +20.0pp |
| normal_factoid | +4.0pp | +16.0pp | +20.0pp |
| negative_near_topic | N/A | +20.0pp | +40.0pp |

但对宽语义类不稳定:

- comparison
- summary_review
- cross_lingual

问题判断:

- 单一检索策略同时覆盖所有 route，导致局部事实类和宽语义类互相牵制。
- child index 应作为 evidence localization 层，而不是所有问题的唯一 broad recall 层。

## 后续修改方向

优先级建议:

1. 保留本次 child Milvus B0 作为最新 baseline，不再用旧 parent/chunk run 作为主基准。
2. 先做检索策略分层，不急于删除 child index:
   - factoid/table/caption: child dense 优先。
   - comparison/summary/cross-lingual: parent/doc recall 优先。
3. 引入或验证 parent aggregation:
   - 先扩大 child candidate pool。
   - 再按 parent/doc 聚合分数。
   - 最后 rerank 聚合后的 parent candidates。
4. 单独定位 parent miss:
   - doc miss: 看 query rewrite、dense/BM25/source floor 是否召回正确 doc。
   - doc hit parent miss: 看同文档 parent 排序、section/body expansion、rerank 输入是否足够。
5. 单独定位 child evidence 流失:
   - retrieval matched child 是否进入 materialized parent。
   - final context 是否保留 matched child。
   - support selector 是否选择带 gold child 的 parent。
   - citation binder 是否把 matched child 传到最终 citation。
6. 生成链路单独开线排查:
   - 重点拆 `validation_failed: 113`。
   - 不要把 answer correctness 低分直接归因到 child Milvus。

## 验收基准

后续每次修改至少对比以下指标:

- doc recall@10
- doc hit@10
- parent hit@10
- parent MRR@10
- matched child at retrieval
- matched child at final
- matched child at support
- matched child at citation
- citation parent hit
- citation child evidence hit
- citation pass
- answer model used rate
- fallback rate
- fallback reason counts
- judge error count

建议首轮目标:

- 不牺牲 `matched child at support/citation` 的前提下，恢复或提升 comparison/summary/cross-lingual 的 doc recall。
- parent hit@10 从 75.5% 提升到至少 80%。
- citation child evidence hit 从 67.7% 提升到至少 72%。
- `validation_failed` 单独下降，不与检索优化混在同一个实验里判断。

