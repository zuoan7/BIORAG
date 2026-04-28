# Generation V2 代码审计报告

> 审计日期：2026-04-28  
> 分支：refactor/generation

---

## 1. 当前 generation_v2 模块列表

| 文件 | 职责 |
|------|------|
| `evidence_ledger.py` | 将 seed_chunks 转换为 EvidenceCandidate 列表，附加 features |
| `support_selector.py` | 从候选中选出 support_pack（factoid/summary/comparison 三路策略） |
| `answer_planner.py` | 根据 support_pack + intent 决定 answer_mode（full/partial/refuse）和 allowed_scope |
| `answer_builder.py` | 抽取式构建初稿答案（extractive，不调 LLM） |
| `citation_binder.py` | 将答案中的引用标记 [E1]...[En] 绑定到 support_pack 对应 chunk |
| `validator.py` | 后置校验：过滤幽灵引用、强制 refuse when zero citation |
| `guardrails.py` | existence/absence 问题检测 + support 充分性评估 |
| `qwen_synthesizer.py` | 受控 Qwen 改写（不改 answer_mode/support_pack/citation） |
| `branch_parser.py` | comparison 问题分支解析（中英文） |
| `comparison_coverage.py` | 按 branch 计算 coverage，生成 allowed_citation_evidence_ids |
| `neighbor_audit.py` | dry-run 邻居审计（不进入 support_pack/citations/Qwen） |
| `models.py` | 共享数据类（AnswerPlan, EvidenceCandidate, SupportItem, GenerationV2Result 等） |
| `service.py` | 主入口，串联上述模块，输出 GenerationV2Result + debug |

---

## 2. 当前 v2 主链路

```
seed_chunks (来自 retrieval/rerank)
    │
    ▼
EvidenceLedgerBuilder.build()
    → 生成 EvidenceCandidate 列表（带 features: section_type, rerank_score 等）
    │
    ▼
SupportPackSelector.select()
    → 按 intent 路由（factoid/summary/comparison）
    → 输出 support_pack: list[SupportItem]
    │
    ▼
AnswerPlanner.plan()
    → existence guardrail 检查
    → comparison branch coverage（可选，v2_enable_comparison_coverage）
    → 决定 mode: full | partial | refuse
    → 输出 AnswerPlan
    │
    ▼
ExtractiveAnswerBuilder.build()
    → 从 support_pack 抽取初稿答案文本
    │
    ▼
QwenSynthesizer.synthesize()   ← 可选（v2_use_qwen_synthesis）
    → 受控 LLM 改写
    → 不合规 → fallback 到 extractive answer
    │
    ▼
CitationBinder.bind()
    → 解析 [E1]...[En] 标记，绑定到 support_pack
    │
    ▼
FinalValidator.validate()
    → 过滤幽灵引用
    → full + missing_branches → 降为 partial
    → v2_require_citation=true 且 citations=[] → refuse
    │
    ▼
NeighborAuditEngine.run()      ← 可选（v2_enable_neighbor_audit）
    → dry-run only，不修改任何上游结果
    │
    ▼
GenerationV2Result（answer, citations, answer_plan, support_pack, debug）
```

---

## 3. 核心功能

| 功能 | 状态 | 说明 |
|------|------|------|
| support_pack 构建 | 核心，默认启用 | 唯一可引用证据来源 |
| answer_plan (mode 决策) | 核心，默认启用 | full/partial/refuse 三态 |
| citation binding | 核心，默认启用 | 绑定 [E1]...[En] 标记 |
| final validator | 核心，默认启用 | 过滤幽灵引用，防止 0 citation non-refuse |
| existence guardrail | 核心，默认启用 | 防止文库存在性问题被弱证据误判 |
| Qwen controlled synthesis | 扩展，默认关闭 | 改写表达，不改 answer_mode/citation |

---

## 4. 实验功能 / 开关功能

| 功能 | 默认 | 状态说明 |
|------|------|---------|
| comparison branch coverage | 关闭 | 实验功能，可配置；改善 comparison 覆盖度 |
| summary_selection debug | 已有 | 诊断用，不影响输出 |
| neighbor_audit | 关闭 | dry-run 诊断，不影响 support_pack/citations/answer |
| **neighbor promotion** | **关闭** | **NOT IMPLEMENTED** — 代码中 `v2_enable_neighbor_promotion=False`，service.py 中 neighbor 不进入 support_pack |
| external tools | 关闭 | NOT IMPLEMENTED — `v2_use_external_tools=False`，无调用路径 |
| Phase A / claim extraction | 关闭 | NOT IMPLEMENTED — 无相关代码路径 |
| conversation history | 关闭 | `v2_use_history=False`，history 参数传入但未实际使用 |

---

## 5. 当前默认配置

所有 `GENERATION_V2_*` 环境变量及默认值：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `GENERATION_VERSION` | `old` | 生成版本（old / v2） |
| `GENERATION_V2_USE_QWEN_SYNTHESIS` | `false` | Qwen 受控改写 |
| `GENERATION_V2_ENABLE_COMPARISON_COVERAGE` | `false` | comparison branch coverage |
| `GENERATION_V2_QWEN_SYNTHESIS_TIMEOUT_SECONDS` | `30` | Qwen 超时 |
| `GENERATION_V2_QWEN_SYNTHESIS_MAX_CHARS_PER_EVIDENCE` | `1200` | 每条证据最大字符数 |
| `GENERATION_V2_QWEN_SYNTHESIS_MAX_OUTPUT_CHARS` | `3000` | 输出最大字符数 |
| `GENERATION_V2_USE_EXTERNAL_TOOLS` | `false` | 外部工具（未实现） |
| `GENERATION_V2_USE_HISTORY` | `false` | 历史对话（未实现） |
| `GENERATION_V2_MAX_SUPPORT_FACTOID` | `3` | factoid 最大 support 数 |
| `GENERATION_V2_MAX_SUPPORT_SUMMARY` | `5` | summary 最大 support 数 |
| `GENERATION_V2_MAX_SUPPORT_COMPARISON` | `6` | comparison 最大 support 数 |
| `GENERATION_V2_MIN_SUPPORT_SCORE` | `0.0` | 最低支持分 |
| `GENERATION_V2_REQUIRE_CITATION` | `true` | non-refuse 必须有引用 |
| `GENERATION_V2_ENABLE_NEIGHBOR_AUDIT` | `false` | 邻居审计（dry-run） |
| `GENERATION_V2_NEIGHBOR_WINDOW` | `1` | 邻居窗口大小 |
| `GENERATION_V2_MAX_NEIGHBORS_PER_SEED` | `2` | 每个 seed 最多邻居数 |
| `GENERATION_V2_NEIGHBOR_PROMOTION_DRY_RUN` | `true` | 邻居 dry-run 模式 |
| `GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION` | `false` | 邻居实际 promotion（禁用） |
| `GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN` | `false` | 邻居不进入 Qwen（禁用） |
| `GENERATION_V2_NEIGHBOR_SCORE_DECAY_DISTANCE1` | `0.45` | 距离=1 的分数衰减系数 |
| `GENERATION_V2_NEIGHBOR_SCORE_DECAY_DISTANCE2` | `0.25` | 距离=2 的分数衰减系数 |
| `GENERATION_V2_NEIGHBOR_MIN_PROMOTION_SCORE` | `0.05` | 邻居最低 promotion 分数门槛 |

---

## 6. 风险点检查

| 风险项 | 检查结果 |
|--------|---------|
| 样本 ID 特判（ent_xxx 等） | **无** — 代码中无任何样本 ID 条件分支 |
| expected_doc_ids / expected_sections 进入运行时 | **无** — 仅出现在评测脚本和 debug 输出中 |
| neighbor 进入 support_pack | **无** — service.py:64 注释明确，neighbor_audit_engine.run() 结果仅写入 debug |
| neighbor 进入 Qwen prompt | **无** — synthesizer 只接受 support_pack，`v2_include_neighbor_context_in_qwen=False` |
| citation fallback 到 chunks[:3] | **无** — CitationBinder 严格绑定 support_pack，无 fallback 路径 |
| old path 被意外修改 | **无** — old/v2 路由在 pipeline.py 层分叉，互不干扰 |
| Qwen 改变 answer_mode | **无** — synthesizer 只返回改写后的 answer text，不修改 plan |
| non-refuse answer 无 citation | **有保护** — FinalValidator.validate() 强制触发 zero_citation_guardrail |

**结论：当前代码无明显安全隐患，关键 invariant 均有代码保护。**
