# Generation V2 设计文档

> 更新日期：2026-04-28  
> 状态：已完成阶段性收口（Stage 1 ~ Stage 2E.0.1）

---

## 1. 为什么要重做 generation

旧 `generation_service` 存在以下问题：

- **逻辑混杂**：neighbor expansion、context building、answer generation、citation、validation 全部耦合在一个大类中。
- **补丁化严重**：每次修复都在原有逻辑上打补丁，难以区分哪条路径真正在跑。
- **可解释性差**：debug 信息不一致，无法追溯某个样本为何得到特定答案。
- **回滚困难**：旧逻辑与新逻辑共存，边界模糊，实验开关随处打补丁。

V2 的目标：**低变量、可解释、可回滚**。

---

## 2. V2 核心设计原则

1. **Seed-only baseline**：只有来自 retrieval/rerank 的 seed_chunks 进入 support_pack，不默认扩展 neighbor。
2. **Support_pack 是唯一可引用证据来源**：citation 只能绑定到 support_pack 中的 chunk，任何不在 support_pack 的内容都不可引用。
3. **Answer_plan 先行**：answer_mode（full/partial/refuse）由 AnswerPlanner 决定，Qwen 不能改变它。
4. **Qwen 只做受控改写**：Synthesizer 接受 support_pack 和 extractive answer，只改写表达，不扩充证据、不改 answer_mode。
5. **Validator 防止 citation 漂移**：FinalValidator 过滤幽灵引用，强制 zero-citation non-refuse 降为 refuse。
6. **Debug 可解释每一步**：每个模块的决策都有 debug 字段，可追溯 mode 的来源、citation 的绑定过程、Qwen 的 fallback 原因等。
7. **Neighbor 不默认参与答案**：NeighborAuditEngine 是 dry-run 诊断工具，不修改任何上游结果。

---

## 3. 当前主链路

```
                          ┌─────────────────┐
seed_chunks (retrieval)   │   GenerationV2   │
     │                    │     Service      │
     │                    └────────┬─────────┘
     ▼                             │
EvidenceLedger.build()             │
     │  EvidenceCandidate[]        │
     ▼                             │
SupportSelector.select()           │
     │  SupportItem[]              │
     ▼                             │
AnswerPlanner.plan()               │
     │  AnswerPlan (mode/reason/   │
     │  covered_branches/missing)  │
     ▼                             │
ExtractiveAnswerBuilder.build()    │
     │  extractive_answer: str     │
     ▼                             │
QwenSynthesizer.synthesize()  ←── v2_use_qwen_synthesis
     │  SynthesisResult            │
     ▼                             │
CitationBinder.bind()             │
     │  (answer, citations)        │
     ▼                             │
FinalValidator.validate()          │
     │  (answer, citations, plan)  │
     ▼                             │
NeighborAuditEngine.run()  ←────── v2_enable_neighbor_audit (dry-run only)
     │  NeighborAuditResult        │
     ▼                             │
GenerationV2Result ────────────────┘
(answer, citations, answer_plan, support_pack, debug)
```

---

## 4. 核心模块说明

| 模块 | 功能 |
|------|------|
| **EvidenceLedger** | 将 RetrievedChunk 转为带 features 的 EvidenceCandidate。features 包含 section_type、score、表格/图片标注等。 |
| **SupportSelector** | 按 intent 三路选 support_pack。summary 路径有 qualified_count 筛选和 selection debug；comparison 路径有 branch-aware 打分。 |
| **AnswerPlanner** | 决定 full/partial/refuse 及原因。集成 existence guardrail（防止文库存在性问题被弱证据误判）和 comparison coverage（可选）。 |
| **ExtractiveAnswerBuilder** | 不调 LLM，直接从 support_pack 拼装初稿。保证即使 Qwen 超时/失败也有可用答案。 |
| **QwenSynthesizer** | 调用 Qwen API 改写表达。有 overclaim 检测、citation 合法性校验、字数限制。不合规则 fallback。 |
| **CitationBinder** | 解析答案中的 [E1]...[En] 标记，绑定到对应 support_pack item。输出有序 citation 列表。 |
| **FinalValidator** | 后置护栏：过滤幽灵引用，full+missing_branches 降为 partial，zero citation non-refuse 降为 refuse。 |
| **NeighborAuditEngine** | 诊断工具：分析 seed 的 ±N 邻居，按 score/semantic/section gate 分类。不写入 support_pack。 |

---

## 5. 已完成阶段总结

### Stage 1
建立 seed-only v2 旁路链路。EvidenceLedger → SupportSelector → AnswerPlanner → ExtractiveAnswerBuilder → CitationBinder → Validator。不走旧链路的 neighbor expansion / ContextBuilder / generator。

### Stage 1.5
修 v2 诊断兼容性。raw_records 输出更完整；区分 refusal_no_citation 和 zero_citation_substantive_answer；修 comparison parser / planner 误拒答。

### Stage 2A
加 existence/absence guardrail。防止"文库中是否有……"类问题被弱相关证据误判为 full。典型样本 ent_094 从 full 降为 partial。

### Stage 2B
加 optional Qwen synthesis。受控改写，不改 answer_mode / support_pack / citation。不合规输出 fallback 到 extractive answer。

### Stage 2C ~ 2C.3
加 comparison branch-aware coverage。支持 branch → evidence → allowed citation set。修中文 comparison parser；修 partial comparison validator 误伤。comparison coverage 作为可配置实验功能。

### Stage 2D / 2D.1
加 summary_selection debug 和 summary support quality gate。结论：ent_015/ent_026/ent_064 support_pack_count=1 且 candidate_count=1，瓶颈在 seed candidates，不在 selector。修 summary partial Qwen validator 误伤。

### Stage 2E / 2E.0.1
加 neighbor dry-run audit。score source 优先 rerank_score；gate 收紧（semantic gate 必须有 query_overlap 或 branch_overlap）；no-support/refuse 样本 blocked；score floor=0.05。**结论：ent_021 的 11 个 false positive promoted 降为 0；ent_015/ent_026/ent_064 的邻居为 context_only（内容不足），当前邻居审计不支持 direct promotion。**

---

## 6. 当前建议默认状态

| 功能 | 建议默认 | 说明 |
|------|---------|------|
| neighbor audit | 关闭 | 保留为诊断工具；production 关闭 |
| **neighbor promotion** | **永久禁用** | hard guard 强制 false；NOT IMPLEMENTED |
| comparison coverage | 关闭（可按需开） | 实验功能；开启前先评测 |
| Qwen synthesis | 关闭（可按需开） | 需 Qwen API 可用；开启前先评测 |
| existence guardrail | 保留，始终启用 | 核心安全护栏 |
| summary selector debug | 保留 | 诊断用，不影响输出 |
| final validator zero citation | 保留，始终启用 | 核心安全护栏 |

---

## 6a. 合并策略

### 推荐方式

合并 generation_v2 核心代码时：

1. **保持 `GENERATION_VERSION=old` 为生产默认**，不擅自切换。
2. 如需启用 v2，设置：
   ```env
   GENERATION_VERSION=v2
   GENERATION_V2_PROFILE=stable
   ```
3. **`stable` profile = v2 最小保守链路**：extractive only，无 Qwen，无 comparison，无 neighbor。
4. 实验性功能通过 profile 逐步解锁，不在 production 直接设置裸开关。
5. `v2_enable_neighbor_promotion` 无论任何情况必须关闭（hard guard 已保证）。

### Profile 层收益

引入 `GENERATION_V2_PROFILE` 的目的是**减少裸开关数量**。9 个 v2 实验开关只需一个变量控制，显式 env 仍可覆盖非危险开关。

### 何时可以切换到 v2 默认

满足以下条件时可考虑将 `GENERATION_VERSION` 默认改为 `v2`：

1. smoke100+ 验证 route/doc/section 指标不低于当前 old baseline；
2. answer_mode 质量（full/partial/refuse 分布）明显优于 old；
3. 所有现有测试（131 tests）持续通过；
4. FinalValidator 零引用护栏在更大集上仍有效。

---

## 7. 已知限制

1. **summary 样本 qualified_count=1**：ent_015/ent_026/ent_064 的 candidate_count=1，说明 seed candidates 在 retrieval/rerank 阶段不足。不是 generation 层能解决的问题。
2. **neighbor audit 未找到可靠 summary promotion 证据**：±1 window 内的邻居 text_preview 主要是缩略词表、方法描述等，不含 2′-FL 机制等有效信息。
3. **section_miss / doc_miss**：section_hit_rate=41%，大部分是 retrieval/rerank/chunking 层问题，generation 层无法弥补。
4. **Qwen 非确定性**：相同输入 citation_count 可能有 ±1 波动（LLM 非确定性），非 neighbor audit 引入。
5. **comparison coverage 复杂度**：branch parse 在中文问题上偶有失败，fallback 到默认逻辑。

---

## 8. 后续建议

1. **先跑更大评测**（如 smoke100 或完整数据集），在更大样本上确认各阶段改进是否稳定。
2. **候选不足问题转向 retrieval/rerank**：summary 样本的根本瓶颈是 seed candidates 不足，应在 retrieval top-k、rerank 策略、chunking 粒度上改进。
3. **不建议继续堆 generation 规则**：当前 generation 层已足够复杂；新增规则的边际收益递减，维护成本递增。
4. **消融实验**：在当前基线上做消融（见 `generation_v2_eval_summary.md`），决定 Qwen synthesis / comparison coverage 是否值得默认开启。
5. **neighbor promotion 未来考虑**：仅在 retrieval 层候选已充分但 generation 层仍丢失时才有价值；需要更长 window 和更好的 semantic gate 再评估。
