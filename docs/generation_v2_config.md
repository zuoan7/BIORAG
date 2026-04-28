# Generation V2 配置说明

> 更新日期：2026-04-28

---

## 概览

Generation V2 使用环境变量或 `.env` 文件配置。所有字段定义在 `src/synbio_rag/domain/config.py` 的 `GenerationConfig` dataclass 中。

---

## Profile（推荐入口）

### `GENERATION_V2_PROFILE`

- **默认值**：`stable`
- **可选值**：`stable` | `qwen` | `comparison` | `debug`
- **说明**：用单个变量同时配置多个实验开关，取代手动设置裸开关组合。Profile 在读取 env 时先生效，显式 env 变量可在 profile 基础上覆盖单个非危险开关。
- **危险开关**（`v2_enable_neighbor_promotion`、`v2_include_neighbor_context_in_qwen`）无论 profile 还是 env 设置，均强制为 `false`（hard guard）。

#### Profile 语义对照表

| profile | qwen_synthesis | comparison_coverage | neighbor_audit | 适用场景 |
|---------|---------------|---------------------|----------------|---------|
| `stable` | false | false | false | 生产默认 / 稳定基线 |
| `qwen` | true | false | false | 表达质量实验 |
| `comparison` | true | true | false | comparison 类实验 |
| `debug` | true | true | true | 诊断评测（非 production） |

> **不存在** `promotion` profile。neighbor promotion 永久禁用。

#### 推荐使用方式

```env
# 生产稳定基线
GENERATION_VERSION=v2
GENERATION_V2_PROFILE=stable

# Qwen 实验
GENERATION_VERSION=v2
GENERATION_V2_PROFILE=qwen

# comparison 实验
GENERATION_VERSION=v2
GENERATION_V2_PROFILE=comparison

# 诊断评测（含 neighbor audit dry-run）
GENERATION_VERSION=v2
GENERATION_V2_PROFILE=debug
```

---

## 基础开关

### `GENERATION_VERSION`

- **默认值**：`old`
- **可选值**：`old` | `v2`
- **说明**：控制使用旧生成链路（old）还是 v2 新链路。
- **推荐生产**：按项目当前约定，不要擅自改变。
- **风险**：切换为 `v2` 会完全绕过旧链路，需先验证 v2 行为符合预期。

---

## Qwen 合成

### `GENERATION_V2_USE_QWEN_SYNTHESIS`

- **默认值**：`false`
- **说明**：启用 Qwen LLM 对抽取式答案进行受控改写。
- **推荐生产**：评估后酌情开启；需要 Qwen API 可用。
- **实验功能**：是
- **不变量**：Qwen 不修改 answer_mode、support_pack、citation。不合规输出自动 fallback 到 extractive answer。

### `GENERATION_V2_QWEN_SYNTHESIS_TIMEOUT_SECONDS`

- **默认值**：`30`
- **说明**：Qwen API 单次调用超时秒数。

### `GENERATION_V2_QWEN_SYNTHESIS_MAX_CHARS_PER_EVIDENCE`

- **默认值**：`1200`
- **说明**：每条证据输入 Qwen 的最大字符数（防止 prompt 过长）。

### `GENERATION_V2_QWEN_SYNTHESIS_MAX_OUTPUT_CHARS`

- **默认值**：`3000`
- **说明**：Qwen 输出的最大字符数（超出则 fallback）。

---

## Comparison Coverage

### `GENERATION_V2_ENABLE_COMPARISON_COVERAGE`

- **默认值**：`false`
- **说明**：为 comparison 类问题启用 branch-aware coverage 分析。
  - 解析问题中涉及的比较分支（如 A vs B）
  - 生成 `allowed_citation_evidence_ids`（每个 branch 的可引用证据）
  - 影响 answer_mode（partial/full 判断更精确）
- **推荐生产**：实验功能，建议先在评测中验证后开启。
- **风险**：branch parse 失败时会 fallback 到默认 coverage 逻辑。

---

## Support Pack 控制

### `GENERATION_V2_MAX_SUPPORT_FACTOID`

- **默认值**：`3`
- **说明**：factoid 类问题 support_pack 最大条目数。

### `GENERATION_V2_MAX_SUPPORT_SUMMARY`

- **默认值**：`5`
- **说明**：summary 类问题 support_pack 最大条目数。

### `GENERATION_V2_MAX_SUPPORT_COMPARISON`

- **默认值**：`6`
- **说明**：comparison 类问题 support_pack 最大条目数。

### `GENERATION_V2_MIN_SUPPORT_SCORE`

- **默认值**：`0.0`
- **说明**：候选证据最低支持分（低于此分的候选从 support_pack 中排除）。
- **推荐**：保持 0.0；调高可能造成 support_pack 为空。

### `GENERATION_V2_REQUIRE_CITATION`

- **默认值**：`true`
- **说明**：非 refuse 答案必须有至少一条 citation。零引用时触发 FinalValidator 强制 refuse。
- **推荐生产**：保持 `true`，防止幽灵答案。
- **风险**：设为 `false` 会允许无引用的实质性回答，可能产生幻觉。

---

## Neighbor Audit（诊断用）

### `GENERATION_V2_ENABLE_NEIGHBOR_AUDIT`

- **默认值**：`false`
- **说明**：启用邻居 dry-run 审计。分析 seed chunks 的上下游邻居，输出 promotion 潜力诊断。
- **不变量**：审计结果仅写入 debug，不影响 support_pack / citations / answer。
- **推荐生产**：关闭；仅在评测/诊断时开启。

### `GENERATION_V2_NEIGHBOR_WINDOW`

- **默认值**：`1`
- **说明**：邻居窗口大小（向前 ±N 个 chunk）。

### `GENERATION_V2_MAX_NEIGHBORS_PER_SEED`

- **默认值**：`2`
- **说明**：每个 seed chunk 最多保留的邻居候选数（按 neighbor_score 降序保留）。

### `GENERATION_V2_NEIGHBOR_MIN_PROMOTION_SCORE`

- **默认值**：`0.05`
- **说明**：邻居 promotion 最低分门槛。低于此分的邻居标注为 context_only（reason: score_below_floor）。

### `GENERATION_V2_NEIGHBOR_SCORE_DECAY_DISTANCE1`

- **默认值**：`0.45`
- **说明**：距离=1 的 seed score 衰减系数。`neighbor_score = seed_score * decay + bonus`

### `GENERATION_V2_NEIGHBOR_SCORE_DECAY_DISTANCE2`

- **默认值**：`0.25`
- **说明**：距离=2 的衰减系数。

---

## Neighbor Promotion（永久禁用）

以下两个开关受 **hard guard** 保护，即使在 env 中设为 `true` 也会被强制覆盖为 `false` 并发出 warning。**不要在任何环境中设为 true。**

### `GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION`

- **默认值**：`false`
- **Hard Guard**：是（设为 true 会被强制覆盖为 false + warning）
- **说明**：neighbor promotion 当前不支持进入主链路（`service.py` 中 neighbor audit 完全 dry-run，不写入 support_pack / Qwen / CitationBinder）。

### `GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN`

- **默认值**：`false`
- **Hard Guard**：是（设为 true 会被强制覆盖为 false + warning）
- **说明**：邻居不进入 Qwen prompt。保持关闭。

### `GENERATION_V2_NEIGHBOR_PROMOTION_DRY_RUN`

- **默认值**：`true`
- **说明**：dry-run 标志，控制 audit 结果标注 `dry_run_promoted` 还是实际 promotion。neighbor audit 始终 dry-run，此标志保持 `true`。

---

## 未实现功能（配置存在但无代码路径）

| 变量 | 默认 | 说明 |
|------|------|------|
| `GENERATION_V2_USE_EXTERNAL_TOOLS` | `false` | 外部工具调用，NOT IMPLEMENTED |
| `GENERATION_V2_USE_HISTORY` | `false` | 历史对话，NOT IMPLEMENTED |

---

## 功能分级

| 功能 | 分级 | 说明 |
|------|------|------|
| generation_v2 主链路 | **生产可用** | EvidenceLedger + SupportSelector + AnswerPlanner + ExtractiveAnswerBuilder + CitationBinder + FinalValidator |
| existence guardrail | **始终启用** | 防止文库存在性误判，不可关闭 |
| FinalValidator 零引用护栏 | **始终启用** | 防止幽灵答案，不可关闭 |
| summary_selection debug | 生产可用 | 不影响输出，diagnostics only |
| Qwen synthesis | **实验开关** | 表达质量改善，需 Qwen API；smoke20 fallback=55%，未默认开启 |
| comparison coverage | **实验开关** | comparison 样本 citation +2，但 branch parse 中文偶有失败 |
| neighbor audit | **诊断开关** | dry-run only，不影响答案；production 建议关闭 |
| neighbor promotion | **永久禁用** | hard guard 强制 false；NOT IMPLEMENTED |
| neighbor context in Qwen | **永久禁用** | hard guard 强制 false |
| external tools / history | **未实现** | 配置存在但无代码路径 |

---

## 保守默认配置一览（推荐生产基线）

```env
# 推荐：通过 profile 一键配置
GENERATION_VERSION=v2
GENERATION_V2_PROFILE=stable

# 等价的裸开关配置（stable profile 展开值）
GENERATION_V2_USE_QWEN_SYNTHESIS=false
GENERATION_V2_ENABLE_COMPARISON_COVERAGE=false
GENERATION_V2_REQUIRE_CITATION=true
GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false
GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION=false   # hard guard；无需手动设置
GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN=false  # hard guard；无需手动设置
GENERATION_V2_USE_EXTERNAL_TOOLS=false
GENERATION_V2_USE_HISTORY=false
```
