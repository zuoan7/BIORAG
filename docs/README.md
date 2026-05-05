# BIORAG 文档索引

本文档是当前项目状态的稳定入口。历史 round 文档保留事实记录，但不代表当前推荐方案。

## 当前权威入口

- [当前评测基线](evaluation/current_baseline.md)：Phase 9 accepted baseline、核心指标、scope freeze、quick check / smoke100 命令。
- [Phase 4–10B 复盘报告](reports/phase4_to_10b_review.md)：从 Phase 4 到 Phase 10B 的历史决策和收口原因。
- [Phase 11 下一阶段规划](reports/phase11_next_stage_plan.md)：当前为什么不继续修主链路，以及下一阶段候选方向。
- [Phase 11E runtime-stable candidate](reports/phase11e_runtime_stable_candidate.md)：Hotfix 11D-b 后的 runtime-stable 候选记录，不替代 Phase 9 accepted quality baseline。

## 当前状态摘要

Accepted baseline：**Phase 9 accepted baseline**

smoke100 结果目录：`results/ragas/smoke100_20260504_214135/`

Runtime-stable candidate：**Phase 11E runtime-stable baseline candidate**

candidate 结果目录：`results/ragas/smoke100_20260505_151754/`

Phase 11E 只代表 API collection error 已收敛，不替代 Phase 9 accepted quality baseline。

核心指标：

| 指标 | 当前值 |
|------|--------|
| faithfulness | 0.6886 |
| answer_relevancy | 0.3185 |
| calibrated P0 | 8 |
| false_refusal | 0 |
| 新增 hallucination | 0 |
| Qwen citation loss | 0 |

当前没有达到 backlog trigger condition，**不建议现在修主链路**。

评测报告中的 `rule_review_candidate_count` 不应称为 calibrated P0；它是规则生成的 P0 review candidate 数量，不保证是 `raw_p0_count` 的子集。

## 当前冻结专项

以下专项仍被冻结：

- Fix C
- comparison guardrail
- summary detail evidence fix
- KB section labeling / re-chunking
- secondary judge
- retrieval / rerank 参数调整
- Qwen prompt 调整

## 后续触发条件

| 触发条件 | 后续动作 |
|----------|----------|
| `comparison_branch_miss >= 3` | 开启 comparison branch guardrail |
| `summary_detail_missing >= 3` | 开启 summary detail evidence fix |
| KB section 问题成批出现或启动 KB 更新 | 考虑 KB section labeling / re-chunking |
| `judge_artifact` 多轮持续 >10 且人工确认影响工程决策 | 考虑 secondary judge |
| `factoid_entity_mismatch >= 3` 且术语映射表就绪 | 考虑 Fix C |

## 运行与评测

- [启动说明](startup.md)
- [当前评测基线与命令](evaluation/current_baseline.md)
- [RAGAS 评测说明](ragas_evaluation.md)

## 历史设计与报告

- [Generation v2 设计](generation_v2_design.md)
- [Generation v2 配置](generation_v2_config.md)
- [Generation v2 审计](generation_v2_audit.md)
- [Generation v2 评测总结](generation_v2_eval_summary.md)
- [Generation fix 状态报告](generation_fix_status_report.md)
- [Systemic rerank / generation fix 报告](systemic_rerank_generation_fix_report.md)
- [Retrieval phase fix 总结](retrieval_phase_fix_summary.md)
- [Diagnostics phase 总结](diagnostics_phase_summary.md)
- [Parsed clean pipeline](parsed_clean_pipeline.md)
- [项目结构](project_structure.md)

## Superseded Round 文档

以下文档已在原路径顶部标注 Superseded，保留作历史记录，不作为当前修改依据：

- [Round 2 regression guide](round2_regression_guide.md)
- [Round 2 regression follow-up](round2_regression_followup.md)
- [Round 2 post-evaluation summary and plan](round2_post_evaluation_summary_and_plan.md)
- [Round 3 guide](round3_guide.md)
- [Round 4 guide](round4_guide.md)
- [Round 5 guide](round5_guide.md)
- [Round 7 guide](round7_guide.md)
- [Round 8 guide](round8_guide.md)
