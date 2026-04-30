# BIORAG 生成阶段修复 — 状态报告

**Date**: 2026-04-30
**Branch**: `fix/generation`
**Base**: `main`
**Commits ahead of main**: 1 (`b48d440`)

---

## 1. 当前分支总览

### 修改的文件 (6 个主链路文件 + 1 个测试文件)

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/.../support_selector.py` | +67 | Fix A: section priority 重排, bibliography-like 检测 |
| `src/.../service.py` | +130 | Fix B: limited_support_pack fallback |
| `src/.../pipeline.py` | +193 | Phase 7C: summary section supplement |
| `src/.../answer_builder.py` | +158 | Phase 9: summary supported-claims 输出 |
| `scripts/.../generate_review_candidates.py` | +398 | 校准后的 P0 分级规则 |
| `tests/test_generation_v2_summary_support.py` | +98 | Fix A 回归测试 (4 个新测试) |

### 新增的诊断/分析脚本 (7 个)

| 文件 | 用途 |
|------|------|
| `diagnose_p0_failure_layers.py` | Phase 1: 15 P0 failure-layer 诊断 |
| `enhance_p0_diagnostics.py` | Phase 2: 增强诊断 (support/citation/factoid) |
| `phase4_audit.py` | Phase 4.5: Fix A/B 审计 |
| `phase5_analyze.py` | Phase 5: 前后对比分析 |
| `phase6_analyze.py` | Phase 6: 残差 P0 分析与下一轮决策 |
| `phase7_diagnose_summary_gaps.py` | Phase 7A: Summary retrieval gap 诊断 |
| `phase8_analyze.py` | Phase 8A: 人工审核标签合并与统计 |

---

## 2. 核心修复内容

### Fix A: Summary section boost (Phase 4)
- **位置**: `support_selector.py`
- **效果**: Abstract/Conclusion 章节优先级从最低提升到最高；bibliography-like chunk 自动降权
- **覆盖**: summary route 样本的 support selection 质量提升
- **状态**: ✅ 已实现，测试通过

### Fix B: limited_support_pack fallback (Phase 4)
- **位置**: `service.py`
- **效果**: support_pack=0 时，用实体匹配 (text+title) 或 rerank score 构建 1-3 个 limited support chunks
- **覆盖**: 3 条 false_refusal 全部消除 (ent_065/071/100)
- **状态**: ✅ 已实现，零新增风险

### Phase 7C: Summary section supplement (Phase 7)
- **位置**: `pipeline.py`
- **效果**: route=summary 时自动从 Milvus 补充 top docs 的 Abstract/Conclusion chunks
- **限制**: KB 中大量文档只有 "Full Text" 标签，无细粒度 section label，导致 supplement 在当前 KB 上无实际命中
- **状态**: ⚠️ 基础设施已就位，受 KB section labeling 限制未生效

### Phase 9: Summary supported-claims output (Phase 9)
- **位置**: `answer_builder.py`
- **效果**: summary 答案从 "Section 证据显示：raw text [E1]" 变为 "- 章节: claim... [E1]" 结构化格式
- **覆盖**: 9 个 answer_fragmentary 目标样本
- **状态**: ✅ 已实现，7/9 targets 输出 claims 格式

---

## 3. RAGAS 评测历程

| Phase | faithfulness | context_recall | P0 数量 | 关键变化 |
|-------|-------------|----------------|---------|---------|
| Baseline (no Qwen) | 0.597 | 0.750 | — | 模板答案，RAGAS 失效 |
| Qwen+coverage (Phase 4前) | 0.642 | 0.768 | 15 | 自然语言答案，可评测 |
| **Phase 4 (Fix A+B)** | **0.691** | 0.769 | **12** | false_refusal 消除，faith +7.7% |
| Phase 7 (supplement) | — | — | 12 | KB 限制，未实际生效 |
| Phase 9 (claims) | 未重跑 | — | — | 答案格式改善，语义质量待 RAGAS 验证 |

---

## 4. 当前 P0 样本状态

### Phase 5 后剩余 12 个 calibrated P0

**5 条原 P0 仍为 P0** (summary_fragment_evidence):
- ent_013, ent_024, ent_040, ent_062, ent_084
- 根因: 检索未返回 Abstract/Conclusion，KB section 标签缺失

**7 条新增 P0** (pre-existing low faithfulness，非 Phase 4 引入):
- ent_015, ent_016, ent_032, ent_036, ent_042, ent_051, ent_092

### 人工审核 (43 条) 关键发现

| 指标 | 分布 |
|------|------|
| answer_correct: yes | 14 |
| answer_correct: partial | 23 |
| answer_correct: no | 6 |
| **hallucination: no** | **41/43 (95%)** |
| hallucination: yes/unsure | 2 |

**RAGAS faithfulness 严重低估了答案质量。** judge_artifact 是最大的单一人类标签 (11/43)。

### 失败层归因 (Phase 8A)

| 层 | 数量 | 主要问题 |
|----|------|---------|
| generation_answer | 20 | answer_fragmentary, citation_not_supporting_claim |
| evaluation_noise | 15 | judge_artifact, correct_refusal |
| retrieval_candidate | 6 | comparison_branch_miss |
| kb_ingestion | 2 | section label missing |

---

## 5. 待解决问题 (下次继续)

### A. KB section labeling (阻塞 summary supplement)
- 当前 KB 中大量文档的全部 chunks 标注为 "Full Text"
- 缺少 Abstract / Conclusion / Results / Discussion 等细粒度 section 标签
- Phase 7C supplement 基础设施已就位，仅等 KB 重分块后生效
- **不确定**: 是否需要全库 re-chunk，还是可以增量更新 section 标签

### B. RAGAS judge 校准
- qwen-plus 在合成生物学中英混合文本上的 faithfulness 判断偏严
- judge_artifact 占人工审核样本的 25% (11/43)
- **不确定**: 更换 judge model (如 gpt-4o) 是否能改善；需要对比实验

### C. Summary retrieval Abstract/Conclusion 覆盖
- 6/10 summary P0 的 Abstract/Conclusion 根本没进入候选池
- 当前 top_k 和 rerank 参数可能对 summary-section chunks 不够友好
- **不确定**: 是 retrieval query 表达问题还是 semantic mismatch

### D. factoid entity/numeric validation (Fix C)
- 2 条 factoid P0 涉及实体/数字不在 citation 中
- 需要合成生物学中英文术语映射表
- **不确定**: 是否优先修（当前只有 2 条 P0）

### E. claim-to-citation 精确绑定
- Phase 9 后 claims 已结构化但 claim 级别精确 citation 验证未实现
- 人工审核显示 citation_support=partial 占 27/43
- **不确定**: 是否需要 LLM 辅助验证还是纯规则可行

### F. 是否需要重跑 smoke100
- Phase 9 后答案格式变了但语义未变
- RAGAS faithfulness 可能因为结构化 claims 变得更可评测
- **不确定**: 重跑 smoke100 的成本 (~22min + API 调用) vs 收益

---

## 6. 未修改的模块 (保护清单)

以下模块在整个 Phase 4-9 过程中未被修改：
- `generation_v2/qwen_synthesizer.py` — Qwen prompt 未变
- `generation_v2/citation_binder.py` — 引用绑定逻辑未变
- `generation_v2/validator.py` — 最终校验未变
- `generation_v2/answer_planner.py` — 回答计划未变
- `generation_v2/branch_parser.py` — 分支解析未变
- `generation_v2/comparison_coverage.py` — 比较覆盖未变
- `generation_v2/neighbor_audit.py` — 邻居审计未变
- `generation_service.py` — 旧 pipeline 未变
- `infrastructure/vectorstores/` — 检索/rerank 参数未变
- `domain/config.py` — 配置默认值未变
- `data/eval/datasets/enterprise_ragas_eval_v1.json` — 数据集未变

---

## 7. 关键产出文件路径

```
results/ragas/
├── smoke100_20260430_113510/    # Phase 4 基线 (Qwen+coverage)
│   ├── ragas_scores.jsonl
│   ├── p0_failure_layer_diagnosis.json
│   ├── phase4_validation_report.md
│   └── human_review_candidates_calibrated.csv
│
├── smoke100_20260430_153147/    # Phase 5 新基线 (含 Fix A+B)
│   ├── ragas_scores.jsonl
│   ├── phase5_final_report.md
│   ├── phase6_final_report.md        # P0 残差分析 + 下一轮决策
│   ├── phase7_final_report.md        # Summary supplement (KB限制)
│   ├── phase8_manual_review_summary.md  # 人工审核统计
│   ├── phase8_next_phase_decision.md    # → answer_builder_fix
│   ├── phase9_final_report.md        # Claims 输出
│   ├── human_review_candidates_reviewed.csv  # 人工标签 ★
│   └── human_review_candidates_calibrated.csv
│
└── STATUS_REPORT.md              # ← 本文件
```

---

## 8. 下次继续的建议顺序

1. **确认是否需要重跑 smoke100** — 验证 Phase 9 claims 格式是否改善 faithfulness
2. **review ent_043, ent_047** — 仅有的 2 条疑似 hallucination，需人工确认
3. **评估 KB section labeling 方案** — 如果可行，Phase 7C supplement 就能生效
4. **对比 judge model** — 测试 gpt-4o 或其他 judge 的 faithfulness 判断是否更准确
5. **评估 Fix C 优先级** — 取决于 factoid P0 在重跑后的实际数量
