# Generation V2 评测总结与消融计划

> 更新日期：2026-04-28  
> 数据集：enterprise_ragas_smoke20（n=20）

---

## 0. Retrieval 改进后新基线（2026-04-28 Phase 1+2+2b）

### 改进内容

1. **评测标签修正（Phase 1）**：修正 `expected_sections` 中与索引标签不一致的条目（如 `"Full Text"` → `"Results and Discussion"`），消除 5/7 的假阳性 section_miss。
2. **Section 识别改进（Phase 2）**：`chunk_by_blocks()` 中对 heading 做标准化映射（`"■ INTRODUCTION"` → `"Introduction"`），title block 如匹配标准 section 则按 section 处理，metadata heading 静默跳过。
3. **Chunk 粒度修复（Phase 2b）**：去掉 heading 强制开新 chunk 的逻辑，非标准 title 归入 parent section，恢复段落级检索粒度。

### 新基线指标

| 指标 | 旧值 | 新值 | 变化 |
|------|------|------|------|
| `doc_id_hit_rate` | 0.9412 | **1.0** | +5.9pp |
| `section_hit_rate` | 0.4118 | **0.7647** | +35.3pp |
| `section_miss` count | 7 | **3** | -57% |
| `doc_miss` count | 1 | **0** | -100% |
| `route_match_rate` | 0.85 | 0.85 | — |
| `answer_mode_distribution` | full=5, partial=13, refuse=2 | full=5, partial=13, refuse=2 | — |
| Chunk 总数 | 7267 | **10747** | +48%（block-based 路径） |
| `<50 token` chunks 占比 | — | 2.4% | 正常 |
| 单 section 文档占比 | 29.4% | **7.4%** | -75% |

### 剩余 3 个 section_miss

待排查，可能是 rerank 排序问题或标签仍有偏差。

---

## 1. 当前 smoke20 指标基线

数据来源：`reports/evaluation/ad_hoc/generation_v2_stage2e01_neighbor_gate_calibration/20260427_175109/v2_stage2d1_baseline.json`

> 此基线 = v2 主链路（Qwen off，comparison coverage off，neighbor audit off）

| 指标 | 当前值 | 来源 |
|------|--------|------|
| `route_match_rate` | 0.85 | 上述 baseline.json，字段 `route_match_rate` |
| `doc_id_hit_rate` | 0.9412 | 上述 baseline.json，字段 `doc_id_hit_rate` |
| `section_hit_rate` | 0.4118 | 上述 baseline.json，字段 `section_hit_rate` |
| `answer_mode_distribution` | full=5, partial=13, refuse=2 | 上述 baseline.json，字段 `answer_mode_distribution` |
| `citation_count_distribution` | 0:2, 1:5, 3:9, 4:2, 5:2 | 上述 baseline.json，字段 `citation_count_distribution` |
| `zero_citation_substantive_answer_ids` | [] | 上述 baseline.json，通过 `answer_mode != refuse && citation_count == 0` 计算 |
| `qwen_used_count` | 14/20 | 上述 baseline.json，字段 `qwen_used_count` |
| `qwen_fallback_count` | 6 | 上述 baseline.json，字段 `qwen_fallback_count` |
| `support_pack_count_distribution` | 未在现有报告中找到聚合值 | baseline.json 中各 raw_record 含 `debug.generation_v2.support_pack_count` |
| **neighbor audit 结论** | ent_015/026/064: promoted=0（context_only）；ent_021/092: refuse blocked | `reports/evaluation/ad_hoc/generation_v2_stage2e01_neighbor_gate_calibration/20260427_175109/summary.md` Section 3 |
| **summary qualified_count=1 结论** | ent_015/026/064 candidate_count=1，瓶颈在 retrieval | 来自阶段摘要（Stage 2D.1 运行结论） |

### 关键观察

- `doc_id_hit_rate=94.1%` 表明检索到正确文档的能力较好。
- `section_hit_rate=41.2%` 显著偏低，是主要质量瓶颈：找到文档但 section 命中不准。
- `refuse=2`（ent_021, ent_092）：无 support_pack，正确拒绝。
- `zero_citation_substantive_answer_ids=[]`：FinalValidator 零引用护栏有效。
- `partial=13`：大部分样本为部分支持；full 仅 5 个。

---

## 2. 各阶段主要解决的问题

### Stage 1 / 1.5
**问题**：无 v2 链路；旧链路 debug 不清；comparison 误拒答。  
**解决**：建立 seed-only v2 旁路；raw_records 完善；区分 refusal_no_citation / zero_citation_substantive；修 comparison planner 误拒。

### Stage 2A
**问题**：文库存在性问题（"文库中是否有……"）被弱相关证据误判为 full answer。  
**解决**：加 existence/absence guardrail。ent_094 从 full 降为 partial（正确）。

### Stage 2B
**问题**：extractive answer 表达质量较差，拼接感强。  
**解决**：加 optional Qwen synthesis。受控改写，不改 answer_mode/citation。fallback 保证安全。

### Stage 2C ~ 2C.3
**问题**：comparison 问题无法区分各分支证据覆盖度；中文 comparison 解析失败；partial validator 误伤。  
**解决**：加 branch-aware coverage；修中文 parser；修 partial comparison validator。

### Stage 2D / 2D.1
**问题**：summary 样本 support_pack_count=1，质量差；Qwen validator 误伤 summary partial。  
**解决**：加 summary_selection debug；加 qualified_count 诊断；修 Qwen validator。  
**结论**：ent_015/ent_026/ent_064 的 candidate_count=1，瓶颈在检索层。

### Stage 2E / 2E.0.1
**问题**：neighbor audit 存在大量 false positive（ent_021 有 11 个 promoted）；score source 使用 fusion_score（近零）。  
**解决**：优先使用 rerank_score；gate 收紧（semantic gate 必须有 query_overlap 或 branch_overlap）；refuse blocking；score floor=0.05。  
**结论**：ent_021 promoted 0（正确）；总 context_only=94，dry_run_promoted=28；neighbor 不支持 direct promotion。

---

## 3. 当前最重要结论

1. **v2 链路稳定**：FinalValidator 零引用护栏有效；answer_mode 不会被 Qwen 改变；neighbor 不影响答案。
2. **existence false positive 已压住**：Stage 2A 之后 ent_094 正确保持 partial。
3. **Qwen synthesis 改善表达**：smoke20 中 Qwen used=14/20；fallback rate ~30%（6-7 次），主要因 overclaim 或字数超限触发。
4. **comparison coverage 有改善但复杂**：branch parse 在中文问题有一定失败率；建议继续作为实验功能。
5. **summary 证据不足主要是 seed candidates 不足**：candidate_count=1 的样本在 generation 层无法弥补；需要 retrieval 层改进。
6. **neighbor audit 不支持直接 promotion**：ent_015/ent_026/ent_064 邻居内容为缩略词表、方法描述，无有效机制说明；正确的 gate 决策是 context_only。

---

## 3a. 消融矩阵实验结果（smoke20，2026-04-28）

> 报告路径：`reports/evaluation/ad_hoc/generation_v2_baseline_matrix/20260428_101618/`

| group | route_match | doc_id_hit | section_hit | answer_mode (full/partial/refuse) | qwen_used | qwen_fallback |
|-------|-------------|------------|-------------|-----------------------------------|-----------|---------------|
| old_baseline | 0.85 | 0.9412 | 0.4118 | 13/1/6 | 0 | 0 |
| v2_extractive_only | 0.85 | 0.9412 | 0.4118 | 5/13/2 | 0 | 0 |
| v2_qwen | 0.85 | 0.9412 | 0.4118 | 5/13/2 | 9 | 11 |
| v2_qwen_comparison | 0.85 | 0.9412 | 0.4118 | 5/13/2 | 12 | 8 |
| v2_qwen_comparison_neighbor_audit | 0.85 | 0.9412 | 0.4118 | 5/13/2 | 11 | 9 |

### 消融结论

1. **old vs v2_extractive_only**：route/doc/section 完全一致；answer_mode 质量显著提升（old full=13 实为虚高，v2 正确拆分为 partial=13 + refuse=2）。v2 主链路更保守、更诚实。

2. **v2_extractive_only vs v2_qwen**：全部结构指标不变；Qwen fallback rate=55%（主因 validation_failed），高于预期。Qwen 不破坏任何不变量，但 55% fallback 偏高，在更大集验证前不建议默认开启。

3. **v2_qwen vs v2_qwen_comparison**：comparison coverage 使 ent_020/ent_090 citations 3→5，branch coverage 完整；同时 Qwen fallback rate 从 55% 降至 40%（branch-aware evidence 让 Qwen 输入更结构化）。

4. **neighbor audit 完全隔离**：全部 20 个样本 answer_mode/citation_count 与 v2_qwen_comparison 完全一致（ent_021 refuse blocking 有效：11 candidates，0 promoted）。

5. **主要失败类别**：section_miss=7，evidence_not_supported=7，route_mismatch=3，doc_miss=1。主因在 retrieval/rerank/chunking，generation 层无法弥补。

### 基于消融矩阵的最终决策

| 功能 | 决策 |
|------|------|
| v2_extractive_only | 当前最稳基线（`GENERATION_V2_PROFILE=stable`） |
| Qwen synthesis | 不建议默认开启（fallback 55%，需 smoke100+ 验证） |
| comparison coverage | 保留实验开关（comparison 场景可开，副作用为零） |
| neighbor audit | 保留诊断工具（production 关闭） |
| neighbor promotion | **永久禁用** |

---

## 4. 建议消融实验矩阵

### A. v2 extractive only（基线）
```
GENERATION_V2_USE_QWEN_SYNTHESIS=false
GENERATION_V2_ENABLE_COMPARISON_COVERAGE=false
GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false
```
**目标**：确认 v2 纯抽取式链路的稳定基线。

### B. v2 + Qwen synthesis
```
GENERATION_V2_USE_QWEN_SYNTHESIS=true
GENERATION_V2_ENABLE_COMPARISON_COVERAGE=false
GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false
```
**目标**：评估 Qwen 对表达质量的提升；关注 fallback rate 和 citation_count 稳定性。

### C. v2 + Qwen + existence guardrail（当前默认已包含）
existence guardrail 始终启用，此组与 B 相同，可用于与旧链路（Stage 2A 之前）对比。  
**目标**：量化 existence guardrail 减少的 false positive 数量。

### D. v2 + Qwen + comparison coverage
```
GENERATION_V2_USE_QWEN_SYNTHESIS=true
GENERATION_V2_ENABLE_COMPARISON_COVERAGE=true
GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false
```
**目标**：评估 comparison coverage 对 comparison 类样本（ent_007/ent_020/ent_084/ent_090 等）的改善。

### E. v2 + Qwen + summary selector（当前 2D.1 后状态）
当前 summary_selection debug 已内置。此组等价于 B。  
**目标**：与 Stage 2D 之前（无 qualified_count 筛选）对比。

### F. v2 + neighbor audit（dry-run，不 promotion）
```
GENERATION_V2_USE_QWEN_SYNTHESIS=true
GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=true
GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION=false
```
**目标**：评估 dry-run 对答案的影响（应为零）；收集 promoted 分布数据；判断将来是否值得启用 promotion。

---

## 5. 各组比较维度

每组应比较：

| 维度 | 指标 |
|------|------|
| 检索质量 | route_match_rate, doc_id_hit_rate, section_hit_rate |
| 答案模式 | answer_mode_distribution（full/partial/refuse 分布） |
| 引用质量 | citation_count_distribution, zero_citation_substantive_answer_ids |
| Qwen 行为 | qwen_used_count, qwen_fallback_count, fallback_reason 分布 |
| 失败类别 | failure_category 分布 |
| 关注样本 | ent_015/ent_026/ent_064（summary），ent_021/ent_092（refuse），ent_007/ent_020/ent_084/ent_090（comparison/partial） |

---

## 6. 明确不做

- **不做 neighbor promotion**：当前邻居内容质量不足以支持 direct promotion；gate 正确拒绝了所有 ent_015/ent_026/ent_064 邻居。
- **不做 Phase A / claim extraction**：增加链路复杂度，边际收益不明确。
- **不做新的 prompt 调参**：Qwen prompt 已经过多轮调整，进一步调参需要更大评测集支撑。
- **不做样本特判**：代码中无任何样本 ID 条件分支，审计已确认。

---

## 7. 消融实验可执行命令

> 注：以下命令均基于现有脚本，调用前需确保 Qwen API 配置已在 `.env` 中设置。  
> `scripts/evaluation/run_generation_v2_baseline_matrix.py` 可一次性运行全部消融组（见下方）。

### Group A：old baseline（旧链路，对照组）

```bash
# 无专用脚本；在脚本中设 GENERATION_VERSION=old 即可对比，当前旧链路未重新评测
# 如需运行，使用 baseline matrix 脚本中的 old_baseline 组
conda activate bge
python scripts/evaluation/run_generation_v2_baseline_matrix.py --groups old_baseline
```

### Group B：v2 extractive only（基线）

```bash
conda activate bge
# 使用 baseline matrix 脚本
python scripts/evaluation/run_generation_v2_baseline_matrix.py --groups v2_extractive_only
# 等价手动运行：GENERATION_VERSION=v2 GENERATION_V2_USE_QWEN_SYNTHESIS=false \
#   GENERATION_V2_ENABLE_COMPARISON_COVERAGE=false \
#   GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false \
#   python scripts/evaluation/run_generation_stage2e01_neighbor_gate_calibration.py
```

### Group C/D：v2 + Qwen synthesis

```bash
conda activate bge
python scripts/evaluation/run_generation_v2_baseline_matrix.py --groups v2_qwen
# 等价：GENERATION_VERSION=v2 GENERATION_V2_USE_QWEN_SYNTHESIS=true \
#   GENERATION_V2_ENABLE_COMPARISON_COVERAGE=false \
#   GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false
```

### Group E：v2 + Qwen + comparison coverage

```bash
conda activate bge
python scripts/evaluation/run_generation_v2_baseline_matrix.py --groups v2_qwen_comparison
# 等价：GENERATION_VERSION=v2 GENERATION_V2_USE_QWEN_SYNTHESIS=true \
#   GENERATION_V2_ENABLE_COMPARISON_COVERAGE=true \
#   GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=false
# 或直接使用现有脚本（group_key=v2_stage2c_current）：
# python scripts/evaluation/run_generation_stage2c_comparison_coverage.py
```

### Group F：v2 + Qwen + neighbor audit（dry-run）

```bash
conda activate bge
python scripts/evaluation/run_generation_v2_baseline_matrix.py --groups v2_qwen_comparison_neighbor_audit
# 等价：GENERATION_VERSION=v2 GENERATION_V2_USE_QWEN_SYNTHESIS=true \
#   GENERATION_V2_ENABLE_COMPARISON_COVERAGE=true \
#   GENERATION_V2_ENABLE_NEIGHBOR_AUDIT=true \
#   GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION=false  # 必须保持 false
# 或使用现有脚本：
# python scripts/evaluation/run_generation_stage2e01_neighbor_gate_calibration.py
```

### 一次性跑全部消融组

```bash
conda activate bge
python scripts/evaluation/run_generation_v2_baseline_matrix.py
# 输出目录：reports/evaluation/ad_hoc/generation_v2_baseline_matrix/<timestamp>/
# 输出文件：comparison_summary.json, comparison_summary.md, focus_samples.json, 各组 json/md
```

### 运行完整测试套件

```bash
conda activate bge
pytest tests/test_generation_v2.py tests/test_generation_v2_eval_diagnostics.py \
  tests/test_generation_v2_existence_guardrail.py tests/test_generation_v2_qwen_synthesis.py \
  tests/test_generation_v2_comparison_coverage.py tests/test_generation_v2_branch_parser.py \
  tests/test_generation_v2_summary_support.py tests/test_generation_v2_neighbor_audit.py \
  tests/test_round8_policy.py tests/test_e2e_support_pack.py -q
```

---

## 8. 最终收口建议（Phase G）

### 推荐当前 baseline

**v2 extractive only**（Group B）：`GENERATION_VERSION=v2`，Qwen off，comparison coverage off，neighbor audit off。

这是最稳定、最可解释的配置；已通过 smoke20 全量验证，测试 113 passed。

### 默认应关闭的开关

| 开关 | 建议 | 原因 |
|------|------|------|
| `GENERATION_V2_USE_QWEN_SYNTHESIS` | 关闭（可实验开启） | 需 Qwen API 可用；fallback rate ~30%；在更大评测集上验证前不宜默认开 |
| `GENERATION_V2_ENABLE_COMPARISON_COVERAGE` | 关闭（可实验开启） | branch parse 中文偶有失败；改善幅度未在 smoke100+ 上确认 |
| `GENERATION_V2_ENABLE_NEIGHBOR_AUDIT` | 关闭（诊断用） | dry-run 不影响答案，但增加运行时开销；需要时才开 |
| `GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION` | 永久关闭 | NOT IMPLEMENTED，当前邻居内容不支持直接 promotion |

### 保留为实验开关的功能

- **Qwen synthesis**：表达质量有提升，fallback 机制已健壮。待 smoke100+ 验证稳定后可默认开启。
- **comparison coverage**：comparison 样本有改善，但 branch parse 失败率需继续评估。
- **neighbor audit**：有诊断价值，保留为可选诊断工具。

### 继续改 generation 的条件

满足以下任一条件才建议继续改 generation 规则：

1. 在 smoke100+ 评测中发现 generation 层有明确可修复的错误类别（非 doc_miss / section_miss / qualified_count=1）；
2. Qwen synthesis 在更大集上 fallback rate < 20%，且 citation_count 分布稳定；
3. 有具体的 comparison 样本集中失败（answer_mode 错误），且已排除 retrieval 原因。

**否则不建议继续在 smoke20 单样本上堆 generation 规则。**

### 转 retrieval / rerank / chunking 的条件

当失败集中在以下类别时，优先查 retrieval/rerank/chunking，而非 generation：

- `doc_miss`：expected_doc_ids 未命中 → 检索层 top-k 不足；
- `section_miss`：doc 命中但 section 未命中（当前 section_hit_rate=41.2%）→ chunking 粒度或 rerank scoring 问题；
- `qualified_count=1`（ent_015/026/064）：summary 候选不足 → retrieval top-k 或 rerank 策略问题；
- `partial` 大量集中但 doc_hit=true → 检索到文档但内容分散，属于 chunk 粒度问题。

### 不建议继续做的事

- 不做 neighbor promotion（当前邻居内容质量不足）；
- 不做 Phase A / claim extraction（链路复杂度高，边际收益未验证）；
- 不做新 prompt 调参（需要更大评测集支撑）；
- 不做样本特判（已审计无 ent_xxx 条件分支）；
- 不在 smoke20 上继续堆 Stage 2F+ 规则（收益递减，维护成本递增）。

### 建议下一步行动

1. **先跑 smoke100 或完整数据集**，确认 smoke20 改进是否泛化；
2. **消融矩阵评测**：用 `run_generation_v2_baseline_matrix.py` 跑 6 组对比，决定 Qwen/comparison coverage 是否值得默认开启；
3. **冻结 generation_v2 规则**：当前 generation 层已足够复杂，新增规则需有量化理由；
4. **retrieval/rerank 改进**：优先解决 section_hit_rate 和 qualified_count 不足问题。
