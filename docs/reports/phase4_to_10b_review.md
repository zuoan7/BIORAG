# Phase 4–10B RAG 优化复盘报告

**Date**: 2026-05-04
**Branch**: `fix/generation`
**Scope**: Phase 4 (support selector) → Phase 10B (evaluation noise cleanup)

---

## 1. 背景和起点

本轮优化前，系统已经历 Qwen+coverage 配置下的第二次 smoke100 RAGAS 评测。此时面临的核心问题：

- **calibrated P0 为 15 个**，其中 false_refusal 3 个、summary_fragment_evidence 6 个、citation_not_supporting_claim 4 个、factoid_entity_or_numeric_mismatch 2 个；
- **RAGAS faithfulness 为 0.642**，低于建议阈值 0.85，但不能确定多少是真实幻觉、多少是 judge 误判；
- answer_relevancy 和 factual_correctness 在 qwen-plus + 合成生物学中英混合文本上基本失效；
- 需要从"指标低分"转入"failure layer 归因"，而不是为了抬分数调参。

起点数据集：`enterprise_ragas_eval_v1.json`，100 条样本。评测配置：Qwen+coverage 开启，judge model 为 qwen-plus。

---

## 2. Phase 4：Summary section boost + limited_support_pack fallback

### 2.1 Fix A：Summary section priority 重排

**问题**：summary route 的 support_selector 将 Results/Discussion 正文碎片排在 Abstract/Conclusion 之前，导致 summary 答案引用局部正文而非总结性章节。

**修改**：`support_selector.py` — `_section_priority()` 重排为 Abstract(0) > Conclusion(1) > Results+Discussion(2) > Results(3) > Discussion(4) > Introduction(5)；新增 `_is_bibliography_like()` 检测并降权 bibliography/reference-list 样式 chunk。

### 2.2 Fix B：limited_support_pack fallback

**问题**：3 个样本（ent_065/071/100）的 support_pack=0，但 final_chunks 中实际包含问题核心实体。support_selector 因分数阈值过严拒绝了所有候选，导致 false refusal。

**修改**：`service.py` — support_pack=0 时，用实体匹配（text+title 的 CJK 3-4 gram + 英文 domain terms）或 rerank score fallback 构建 1-3 个 limited support chunks。

### 2.3 初步结果

- Fix B 将 3 条 false_refusal 全部修复（refuse → 有引用的 substantive/limited answer，citation 0→3；后续审计未发现 unsupported full answer）；
- Fix A 对 6 个 summary fragment 样本启用了 section boost；
- Phase 4.5 审计确认 Fix B 零 unsupported full answer，Fix A 部分有效；
- 单元测试 205 passed，quick check 10 route match 10/10。

---

## 3. Phase 4.5：Fix A / Fix B 风险审计

在进入 smoke100 全量回归之前，对 Phase 4 的两个修复做了定向风险审计：

- **Fix B**：逐条检查 ent_065/071/100 的 limited_support_pack 触发后的答案。结论：**未发现 unsupported full answer**——3 条答案均有 citation 支撑核心声明，不需要收紧 answer_mode。
- **Fix A**：检查 6 个 summary_fragment_evidence 样本的 section boost 效果。结论：**部分有效**——当候选中有 Abstract/Conclusion 时生效（3/6 改善），但纯 body-text 文档无法受益。
- 新增 4 个回归测试覆盖 Fix A 的设计变更（Abstract 优先、bibliography 降权、Results 不过滤、非 summary 不受影响），全部通过。
- 单元测试 205 passed，quick check 10 route match 10/10。

**审计结论**：Phase 4 改动安全，允许进入 smoke100 全量回归。

---

## 4. Phase 5：Phase 4 smoke100 回归

以 Phase 4 代码重跑完整 Qwen+coverage smoke100 RAGAS，与 Phase 4 前基线对比：

| 指标 | Phase 4 前基线 | Phase 5（含 Fix A+B） | 变化 |
|------|-------------|---------------------|------|
| faithfulness | 0.642 | **0.691** | **+7.7%** |
| context_recall | 0.768 | 0.770 | +0.001 |
| context_precision | 0.690 | 0.678 | -0.012 |
| refusal_count | 8 | 4 | -4 |
| calibrated P0 | 15 | **12** | -3 |
| false_refusal | 3 | **0** | 消除 |

**结论**：接受 Phase 4 作为新基线。

---

## 4. Phase 6：剩余 P0 诊断与原决策修正

Phase 5 后剩余 12 个 calibrated P0。对齐 Phase 4 前的 15 个原 P0：

- **已解决 10 个**：ent_065/071/100（Fix B） + ent_011/012/017/028/047/074/083（Fix A + 整体 faith 提升）；
- **仍为 P0 5 个**：ent_013/024/040/062/084，全部是 summary_fragment_evidence；
- **新增 P0 7 个**：ent_015/016/032/036/042/051/092，为 pre-existing low faithfulness，非 Phase 4 引入。

当时对剩余 P0 的 failure layer 统计显示 summary_fragment_evidence 占主导（10/12），因此 Phase 6 **初始决策为 `proceed_to_summary_retrieval_fix`**，认为 root cause 是检索未返回 Abstract/Conclusion。

**这个判断后来被 Phase 7 和 Phase 8 修正**——见下文。

---

## 5. Phase 7：Summary retrieval supplement（基础设施就位，但受 KB section 标签限制）

**目标**：在 pipeline.py 中实现 route=summary 时自动从 Milvus 补充 top documents 的 Abstract/Conclusion chunks。

**实现**：pipeline.py 中 `_supplement_summary_sections()` 函数，通过 Milvus client 查询 `doc_id + section == "Abstract"/"Conclusion"`，最多补充 5 个 chunk。

**结果**：supplement 逻辑正确但**未对任何目标样本实际补入 chunk**。根因是 KB 中大量文档的所有 chunks 均标注为 "Full Text"，没有细粒度 section 标签（Abstract/Conclusion/Results/Discussion）。

**关键发现**：summary P0 的问题不全是 retrieval 不返回 Abstract/Conclusion——很多时候是 KB 根本没有这些 section 标签。Phase 6 的 `summary_retrieval_fix` 判断需要修正。

**决策**：`stop_and_manual_review`，等待人工审核数据后再决定方向。

---

## 6. Phase 8：人工审核标签分析

对 43 个候选样本进行了人工审核（人工审核标签文件与合并结果以仓库实际文件名为准，如 `human_review_candidates_reviewed.csv` 和 `phase8_manual_review_merged.csv`），结果：

| 人工指标 | 分布 |
|---------|------|
| answer_correct: yes | 14 |
| answer_correct: partial | 23 |
| answer_correct: no | 6 |
| **hallucination: no** | **41/43 (95%)** |
| hallucination: yes/unsure | 2 |
| citation_support: yes | 15 |
| citation_support: partial | 27 |
| citation_support: no | 1 |

失败层归因（人工标注）：

| 层 | 数量 | 主要标签 |
|----|------|---------|
| generation_answer | 20 | answer_fragmentary, citation_not_supporting_claim |
| evaluation_noise | 15 | judge_artifact (11), correct_refusal |
| retrieval_candidate | 6 | comparison_branch_miss |
| kb_ingestion | 2 | section label missing |

**核心发现**：
- **系统不是幻觉严重**——95% 样本无幻觉；
- 主要问题是**答案碎片化**（answer_fragmentary）和 **partial citation support**；
- judge_artifact 是最大的单一标签（11/43），说明 RAGAS 对当前答案风格偏严；
- 最大可修复类是 generation_answer 层，而非 retrieval 或 KB 层。

**决策**：`proceed_to_summary_answer_builder_fix`——修正了 Phase 6 的 retrieval_fix 方向。

---

## 7. Phase 9：Summary answer builder fix

### 7.1 问题

Summary route 的用户侧答案长期以 evidence snippet 格式输出：

```
Results 证据显示：raw English chunk text... [E1]
Introduction 证据显示：another raw text... [E2]
```

这种格式让 RAGAS judge 难以评估答案质量（faithfulness 被压低），也让用户难以阅读。

### 7.2 修改

`answer_builder.py` — summary route 新增 `_build_summary_claims()` 函数，将 support_pack 转化为结构化 supported claims：

```
- 结论: organized claim text... [E1] [doc_0007]
- 摘要: another claim... [E2] [doc_0198]

证据限制：
- 仅有 2 条合格证据，覆盖面有限。
```

**约束**：每条 claim 必须来自 support_pack，保留 citation；bibliography-like 内容自动过滤；evidence snippet 仍保留在 debug 中供调试。

### 7.3 效果

对 9 个 Phase 8A 识别的 answer_fragmentary 目标样本验证：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| evidence-snippet 格式 | 9/9 | **0/9** |
| structured claims 格式 | 0 | **7/9** |
| Qwen 自然语言重写 | 0 | **2/9** |
| citation 保留 | — | 全部 |
| 新增幻觉 | — | 0 |

**影响范围**：仅限 route=summary，factoid/comparison/experiment 不受影响。单元测试 204 passed，quick check 10/10。

---

## 8. Phase 9 smoke100 / Phase 10A：剩余 P0 风险审计

### 8.1 全局指标

| 指标 | Phase 5 | Phase 9 | 变化 |
|------|---------|---------|------|
| faithfulness | 0.691 | **0.689** | -0.4%（稳定） |
| context_recall | 0.770 | 0.750 | -2.5% |
| answer_relevancy | 0.283 | **0.319** | **+12.7%** |
| calibrated P0 | 12 | **8** | **-33%** |
| false_refusal | 0 | 0 | 稳定 |

### 8.2 剩余 8 个 P0 审计

| Sample | Route | Faith | Real Issue? | Type |
|--------|-------|-------|------------|------|
| ent_021 | (error) | None | ❌ | API 采集错误 |
| ent_010 | comparison | 0.0 | ⚠️ | comparison_branch_miss |
| ent_055 | summary | 0.067 | ❌ | judge_artifact（答案正确但 judge 过严） |
| ent_012 | factoid | 0.2 | ❌ | reference 命名差异（中文名 vs 英文缩写） |
| ent_032 | summary | 0.25 | ⚠️ | summary_detail_missing（缺定量结果） |
| ent_084 | comparison | 0.25 | ❌ | comparison 旧格式（Phase 9 未覆盖） |
| ent_040 | summary | 0.29 | ⚠️ | summary_fragment_evidence（KB 限制） |
| ent_022 | comparison | 0.43 | ❌ | acceptable partial（自认 incomplete） |

**5/8 不是真实质量问题**。3/8 是真实问题但分布在 3 个不同 failure type，无统一修复点。

### 8.3 Phase 9 风险确认

- 无 evidence 被压缩为更强 claim
- 无 partial→full conclusion 误升
- 无 Qwen 丢失 citation
- 无 Qwen 新增 support_pack 外事实
- comparison/factoid 未受影响
- false_refusal 仍为 0

**决策**：`accept_phase9_and_stop`。

---

## 9. Phase 10B：Evaluation noise ledger + real issue backlog

### 9.1 Evaluation Noise Ledger（5 个非真实 P0）

| Sample | Type | Why Not Real |
|--------|------|-------------|
| ent_021 | api_data_collection_error | API call failed，不是答案质量问题 |
| ent_055 | judge_artifact | 答案正确描述策略，judge 期望的定量数据不在 Abstract chunk 中 |
| ent_012 | reference_naming_mismatch | 给出 4 个正确 HMO 例子，但中文全名 vs 英文缩写 mismatch |
| ent_084 | comparison_old_format | 仍用旧 evidence-snippet 格式（Phase 9 只覆盖 summary） |
| ent_022 | acceptable_partial | 自认 comparison_evidence_incomplete，faith=0.43 尚可 |

这些样本 `should_count_as_p0=false`，不应触发主链路修复。

### 9.2 Real Issue Backlog（3 个真实问题）

| Sample | Issue | Trigger to Revisit |
|--------|-------|-------------------|
| ent_010 | comparison_branch_miss | ≥3 comparison P0 时开专项 |
| ent_032 | summary_detail_missing (定量缺失) | ≥3 定量 summary P0 时开专项 |
| ent_040 | summary_fragment_evidence (KB limit) | KB re-chunking 项目启动时处理 |

每类目前只有 1 条样本，无法支撑定向修复。设定 `trigger_condition`，等同类问题成规模后再开专项。

---

## 10. 当前最终基线

| 指标 | 值 | 说明 |
|------|-----|------|
| faithfulness | **0.689** | 稳定（vs Phase 4 前 0.642，+7.3%） |
| answer_relevancy | **0.319** | Phase 9 提升 +12.7% |
| context_recall | 0.750 | 较 Phase 5 轻微下降；由于 Phase 9 未修改 retrieval/rerank/KB，暂无证据表明是 retrieval 退化，后续仅作监控 |
| calibrated P0 | **8** | 从 15 降至 8（-47%） |
| false_refusal | **0** | Phase 4 起持续为零 |
| 新增 hallucination | 0 | Phase 9 不引入新幻觉 |
| Qwen citation loss | 0 | citation 全部保留 |

**输出目录**：`results/ragas/smoke100_20260504_214135/`（Phase 9 smoke100）

**代码基线**：`fix/generation` 分支，领先 main 4 个 commit。

精确指标和可执行命令以 `docs/evaluation/current_baseline.md` 为准。

---

## 11. 当前 Scope Freeze

以下方向**当前不做**，原因如下：

| 方向 | 不做的原因 |
|------|-----------|
| Fix C（factoid entity/numeric validation） | 仅 1 条 factoid P0，术语映射表未就绪 |
| Claim-level citation validation | 无统一 failure pattern；5/8 P0 非真实问题 |
| Comparison branch guardrail | 仅 1 条 comparison P0（ent_010） |
| KB re-chunking / section labeling | 需独立 KB 项目；仅 1 条 P0 受此影响 |
| Judge model 更换 | 当前 judge 噪声已知、可管理；更换引入新变量 |
| Retrieval/rerank 参数调整 | context_recall 较 Phase 5 下降 -2.5%，但 Phase 9 未修改检索链路，暂无证据表明是 retrieval 退化 |
| Qwen prompt 修改 | 当前 answers 无幻觉、有 citation；不需要调 prompt |

---

## 12. 后续触发式计划

| 触发条件 | 行动 |
|---------|------|
| comparison_branch_miss ≥3 | 开启 comparison branch guardrail (per-branch support_pack + citation) |
| summary_detail_missing ≥3（定量题） | 开启 summary detail evidence fix (Results/Discussion body supplements Abstract) |
| KB re-chunking 项目启动 | 重新分块后补上 Abstract/Conclusion section 标签，让 Phase 7 supplement 生效 |
| judge_artifact 在多轮 smoke100 中持续 >10 条，且人工审核确认其影响工程决策 | 考虑 secondary judge（如 gpt-4o）对照实验 |
| factoid_entity_mismatch ≥3 | 开启 Fix C（需要术语映射表就绪） |

---

## 13. 本轮关键经验

1. **不能只看 RAGAS 分数**——Phase 8 人工审核发现 95% 样本无幻觉，而 RAGAS faithfulness 仅 0.689。Judge artifact（11/43）是最大单一人类标签。必须用人工审核校准自动化指标。

2. **false_refusal 和 low_faithfulness 要分开处理**——前者是 support_pack 阈值问题（Fix B），后者更多是答案结构和证据组织问题（Phase 9）。

3. **summary 问题不等于一定缺 Abstract/Conclusion**——Phase 6 初始判断认为需要 retrieval fix，但 Phase 7 发现 KB section 标签限制了 supplement 生效，Phase 8 发现真正的最大类别是 generation_answer 层的 answer_fragmentary。方向从 retrieval → KB → answer_builder 逐步修正。

4. **answer builder 比 prompt 自由生成更可控**——Phase 9 的 supported-claims 结构在不改 Qwen prompt 的前提下，将 evidence snippet 格式转换为结构化输出，citation 保留、幻觉零新增、faithfulness 稳定。

5. **evaluation noise ledger 和 real issue backlog 是防止过拟合的关键工具**——明确区分"评测噪声"和"真实问题"，为每类真实问题设定 trigger_condition（≥3 条），避免为了 1-2 条样本过度修改主链路。
