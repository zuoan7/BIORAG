# v3 B0+rewrite full no-judge parent miss 审计状态

## 背景

本文件记录 2026-05-24 这一轮对 v3 B0+rewrite full no-judge 后续最大失败桶的审计状态。

当前目标是维护问题归因和下一步修复顺序，不记录已执行的代码改动方案。本轮没有修改 score floor、same-doc expansion、neighbor expansion、support selector、generation/citation binding，也没有清理 dirty worktree。

关键输入：

- `results/evaluation/v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge/b0_rewrite_enabled/summary.json`
- `results/evaluation/v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge/b0_rewrite_enabled/results.jsonl`
- `results/evaluation/v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge/rewrite_cache.jsonl`
- `data/eval/datasets/v3_baseline_dataset.jsonl`
- `data/paper_round1/chunks/parent_chunks.jsonl`
- `data/paper_round1/chunks/child_chunks.jsonl`

## 当前状态

`SynBioRAGPipeline.answer()` 的 rerank query wiring 已修复并验证：

- B0+rewrite enabled 时，retrieval 和 rerank 均使用 rewritten English query。
- 原始中文 query 仍用于 route/analysis、context、generation。
- query wiring 验证结果：
  - `total`: 200
  - `missing_query_bundle`: 0
  - `missing_rerank_hits`: 0
  - `query_bundle.rerank_query != query_bundle.rewritten_query`: 0
  - `rerank_hits.rerank_query != query_bundle.rewritten_query`: 0
  - `retrieval_query != rewritten_query`: 0
  - `original_query != rewritten_query`: 170/200

Full no-judge eval 已完成，结果路径：

- `results/evaluation/v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge/`
- `reports/evaluation/v3_b0_rewrite_enabled_20260524_rerank_query_wiring_full_nojudge/`

注意：该 full eval 当时在 conda `base` 环境运行，reranker 实际退回 CPU。后续如需重新跑 eval，使用：

- `conda run -n bge python ...`
- `/home/gmy/miniconda3/envs/bge/bin/python ...`

## 指标和剩余桶

相对 B0 baseline，本轮 full no-judge 规则指标：

| metric | B0 baseline | B0+rewrite | delta |
|---|---:|---:|---:|
| `doc_hit_at10_rate` | 0.836842 | 0.905263 | +0.068421 |
| `parent_chunk_hit_at10_rate` | 0.560 | 0.765 | +0.205 |
| `exact_gold_chunk_hit_at10_rate` | 0.120 | 0.155 | +0.035 |
| `support_parent_chunk_hit_rate` | 0.435 | 0.700 | +0.265 |
| `support_child_evidence_hit_rate` | 0.265 | 0.555 | +0.290 |
| `citation_parent_chunk_hit_rate` | 0.431579 | 0.705263 | +0.273684 |
| `citation_child_evidence_hit_rate` | 0.263158 | 0.552632 | +0.289474 |
| `route_match_rate` | 0.715 | 0.715 | 0.000 |

计数口径：

- parent hit: 153/200
- support parent hit: 140/200
- citation parent hit: 134/190
- support child hit: 111/200
- citation child hit: 105/190

剩余 diagnostic buckets：

| bucket | count | 状态 |
|---|---:|---|
| `doc_miss` | 18 | 未进入本轮审计 |
| `doc_hit_parent_chunk_miss` | 26 | 本轮已完整审计 |
| `parent_hit_support_miss` | 13 | 待 parent miss 处理后再审 |
| `parent_hit_citation_miss` | 12 | 待 parent miss 处理后再审 |
| `evidence_hit_answer_failed` | 0 | 本轮 skip judge，不能与 baseline judge run 直接比较 |
| `judge_schema_or_call_issue` | 0 | 无 judge 调用 |

## 最大桶归因

本轮审计了全部 26 个 `doc_hit_parent_chunk_miss` 样本。

主因分布：

| first break | count | samples | 后续维护方向 |
|---|---:|---|---|
| raw 命中 gold，但 rerank/score floor 后出局 | 8 | `v3_pc_020`, `v3_pc_039`, `v3_pc_132`, `v3_pc_139`, `v3_ra_002`, `v3_ra_016`, `v3_ra_022`, `v3_ra_029` | 做 score 分布和 rerank trace 审计；先不要改 score floor |
| gold id 与当前 chunk index 漂移 | 7 | `v3_ra_009`, `v3_ra_014`, `v3_ra_018`, `v3_ra_019`, `v3_ra_021`, `v3_ra_026`, `v3_ra_027` | 先审 dataset gold remap 口径，否则会误判 retrieval |
| raw 未召回 gold，只召回同文档邻近/结构块 | 8 | `v3_pc_098`, `v3_ra_005`, `v3_ra_007`, `v3_ra_020`, `v3_ra_023`, `v3_ra_024`, `v3_ra_025`, `v3_ra_028` | 审 raw retrieval、chunk boundary、table/figure/caption 结构召回 |
| comparison route/selection/topk 问题 | 2 | `v3_pc_151`, `v3_pc_154` | 审 route 和 comparison coverage，不转 support/generation |
| comparison 一侧 gold raw 缺失 | 1 | `v3_pc_153` | 审 query decomposition/route；raw 阶段已经断 |

关键判断：

- 这 26 个样本的大部分问题和 English mirror rewrite 无直接关系。
- 7 个 gold mapping drift 样本属于 dataset/gold 与当前 chunk index 不一致。
- 18 个 `v3_ra_*` 样本为 `identity_fallback`，query 本身就是英文，rewrite 没有引入语义变化。
- 8 个 score-floor 样本中，gold parent 已进入 raw retrieval，first break 在 rerank selection/score floor 之后，不是 retrieval query 直接 miss。
- `v3_pc_*` 中仍可能存在 rewrite 与 rerank 分数形态的交互，但当前证据不支持把最大桶主因归咎为 English rewrite。

## 问题维护 ledger

| 优先级 | 问题 | 当前状态 | 下一步 |
|---|---|---|---|
| P0 | `doc_hit_parent_chunk_miss` 中 7 个 gold id 漂移 | 已完成标注 remap：7 个样本的 `target_chunk_id_candidate` 和 `answer_rubric.source_trace.chunk_ids` 已更新到当前 parent id | 离线重算 rule metrics，确认 parent miss bucket 变化 |
| P1 | 8 个 raw 命中后被 score floor/drop | 已定位 target trace，如 `raw#6/pre#2/postNone/finalNone/score_floor` | 审 rerank score 分布和 target/top competitor 差异；暂不改 `RETRIEVAL_RERANK_SCORE_FLOOR_RATIO` |
| P1 | table/figure/caption 结构召回缺口 | 已定位 raw 无 gold 但召回同文档邻近或结构块 | 分 table、figure_caption、caption_level_table 做 raw retrieval 结构审计 |
| P2 | comparison route/selection/topk | `v3_pc_151` route 到 FACTOID，`v3_pc_153` route 到 SUMMARY，`v3_pc_154` comparison selection 选同文档非 gold 块 | 单独审 route/comparison coverage；不要混入 support/generation |
| P2 | support/citation 剩余桶 | 尚未进入本轮审计 | 等 parent chunk miss 主因处理后再转向 selector/generation |

## 明确不做

- 不要直接改 `RETRIEVAL_RERANK_SCORE_FLOOR_RATIO`。
- 不要直接改 same-doc expansion 或 neighbor expansion。
- 不要直接改 support selector。
- 不要直接改 generation/citation binding。
- 不要用 conda `base` 环境跑需要 GPU 的 eval。
- 不要把 skip-judge 的 `evidence_hit_answer_failed = 0` 当成答案质量已解决。
- 不要清理当前 dirty worktree。

## 下一步建议

1. 对 P0 标注更新后的 dataset 做离线 rule metric 重算，确认 parent miss bucket 变化。
2. 再对 8 个 score-floor 样本做 rerank trace 复盘，比较 target 与 top competitor 的 query scores、section、structure flags、same-doc 距离。
3. 之后再审 8 个 raw retrieval 结构缺口，按 table、figure_caption、caption_level_table 分桶。
4. 最后单独处理 comparison route/coverage，再决定是否进入 support/citation bucket。

## 2026-05-24 P0 标注更新记录

已直接更新 `data/eval/datasets/v3_baseline_dataset.jsonl` 中 7 个 P0 样本的证据标注：

| sample_id | old target | new target |
|---|---|---|
| `v3_ra_009` | `doc_0110_sec19_chunk20` | `doc_0110_sec14_chunk15` |
| `v3_ra_014` | `doc_0035_sec12_chunk13` | `doc_0035_sec09_chunk10` |
| `v3_ra_018` | `doc_0087_sec18_chunk19` | `doc_0087_sec08_chunk09` |
| `v3_ra_019` | `doc_0366_sec20_chunk21` | `doc_0366_sec11_chunk12` |
| `v3_ra_021` | `doc_0199_sec25_chunk26` | `doc_0199_sec10_chunk11` |
| `v3_ra_026` | `doc_0001_sec15_chunk16` | `doc_0001_sec08_chunk09` |
| `v3_ra_027` | `doc_0003_sec15_chunk16` | `doc_0003_sec08_chunk09` |

更新范围：

- 已更新 `target_chunk_id_candidate`。
- 已更新 `answer_rubric.source_trace.chunk_ids`。
- 未修改 `expected_answer`、`answer_rubric.must_include`、`answer_rubric.reject_if`、`stable_target_block_ids`。
- 验证通过：7/7 新 parent id 存在于当前 `parent_chunks.jsonl`，并覆盖对应 `stable_target_block_ids`。

注意：`data/` 目录被 `.gitignore` 忽略，普通 `git status` 不会显示这次 dataset 文件改动。
