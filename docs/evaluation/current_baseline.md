# BIORAG 当前评测基线文档

**Date**: 2026-05-04
**基线名称**: Phase 9 accepted baseline
**代码分支**: `fix/generation`

---

## 1. 当前 Accepted Baseline

| 指标 | 值 | 说明 |
|------|-----|------|
| faithfulness | **0.6886** | 稳定（vs Phase 4 前 0.642，+7.3%） |
| answer_relevancy | **0.3185** | Phase 9 claims 格式提升 +12.7% |
| context_recall | 0.7502 | 较 Phase 5 轻微下降，暂无证据是 retrieval 退化 |
| context_precision | 0.6748 | 稳定 |
| calibrated P0 | **8** | 从 15 降至 8（-47%） |
| false_refusal | **0** | 持续为零 |
| 新增 hallucination | 0 | Phase 9 不引入新幻觉 |
| Qwen citation loss | 0 | citation 全部保留 |

**smoke100 输出目录**: `results/ragas/smoke100_20260504_214135/`

---

## 1.1 Phase 11E runtime-stable baseline candidate

Phase 11E 是 Hotfix 11D-b 之后的 runtime-stable baseline candidate。它**不替代** Phase 9 accepted baseline；Phase 9 accepted baseline 仍然是当前 accepted quality baseline。

Phase 11E candidate 的作用是记录一个已修复 API collection error 的可运行候选口径：

- 修复对象：`src/synbio_rag/application/pipeline.py::_supplement_summary_sections`
- 问题：`RetrievedChunk` schema 不支持顶层 `chunk_index` 参数
- 修复方式：移除 `chunk_index=...`，并将该值保留到 `metadata["chunk_index"]`
- 结果：`ent_021` 不再触发 API 500，完整 smoke100 API collection `error_count=0`

Phase 11E 完整 smoke100 输出目录：

`results/ragas/smoke100_20260505_151754/`

| 指标 | Phase 11E candidate |
|------|---------------------|
| faithfulness | `0.6742` |
| answer_relevancy | `0.3087` |
| raw_p0_count | `10` |
| rule_review_candidate_count | `11` |
| noise_adjusted_p0_count | `9` |
| qwen_citation_loss_count | `0` |
| false_refusal | `0` |
| new_hallucination | `0` |
| API collection error_count | `0` |
| zero_citation_count | `0` |

不直接替代 Phase 9 的原因：

- `rule_review_candidate_count = 11` vs Phase 9 `8`，仍有 warning
- `noise_adjusted_p0_count = 9` vs Phase 9 `3`，仍有 warning
- 本候选基线只证明 runtime 500 已收敛，不证明质量口径优于或等价于 Phase 9 accepted baseline

术语说明：后续报告中不要把 `rule_review_candidate_count` 简写为 calibrated P0。`rule_review_candidate_count` 是规则生成的 P0 review candidate 数量，不保证是 `raw_p0_count` 的子集。

---

## 2. 本轮主要修复内容

本轮不是单纯修检索，而是从 Phase 4 到 Phase 9 的完整优化过程：

| Phase | 内容 | 效果 |
|-------|------|------|
| Phase 4 Fix A | Summary section priority 重排：Abstract/Conclusion 优先 | summary support selection 质量改善 |
| Phase 4 Fix B | limited_support_pack fallback：support_pack=0 时实体匹配兜底 | false_refusal 3→0 |
| Phase 7 | Summary retrieval supplement（基础设施就位，受 KB section 标签限制） | 补充逻辑正确但 KB 无细粒度 section 标签，未实际生效 |
| Phase 8 | 人工审核 43 条样本 | 确认系统不是幻觉严重（95% 无幻觉），主问题是 answer_fragmentary |
| Phase 9 | Summary answer builder：evidence snippets → structured supported claims | P0 12→8，answer_relevancy +12.7% |
| Phase 10B | Evaluation noise ledger + real issue backlog | 5/8 剩余 P0 标记为非真实问题，3 个真实问题进入 backlog |

---

## 3. 当前推荐运行配置

以下为从 `Settings.from_env()` 读取的实际配置：

| 配置项 | 值 |
|--------|-----|
| `generation.version` | `v2` |
| `GENERATION_V2_PROFILE` | `stable`（显式 env 覆盖下列两个实验开关） |
| `v2_use_qwen_synthesis` | `true` |
| `v2_enable_comparison_coverage` | `true` |
| `v2_enable_neighbor_audit` | `false` |
| `v2_enable_neighbor_promotion` | `false` |
| `retrieval.neighbor_expansion_enabled` | `true` |
| `retrieval.rerank_mode` | `local` |
| `retrieval.hybrid_enabled` | `true` |
| `retrieval.bm25_enabled` | `true` |
| `retrieval.comparison_max_chunks_per_doc` | `3` |
| `round8.enable_round8_policy` | `false` |

**Phase 4 修复相关**（代码内置，不需要配置开关）：
- Fix A：summary section boost（`support_selector.py` 中的 `_section_priority()`）
- Fix B：limited_support_pack fallback（`service.py` 中的 `_build_limited_support_pack()`）
- Phase 7：summary section supplement（`pipeline.py` 中的 `_supplement_summary_sections()`）
- Phase 9：summary supported-claims（`answer_builder.py` 中的 `_build_summary_claims()`）

**环境变量**（在 `.env` 中配置，不提交到仓库）：

| 变量 | 用途 |
|------|------|
| `QWEN_CHAT_API_BASE` | Judge LLM + Qwen synthesis API base URL |
| `QWEN_CHAT_API_KEY` | Judge LLM + Qwen synthesis API key |
| `GENERATION_V2_USE_QWEN_SYNTHESIS` | 开启 Qwen synthesis（当前 `true`） |
| `GENERATION_V2_ENABLE_COMPARISON_COVERAGE` | 开启 comparison coverage（当前 `true`） |
| `BIORAG_RERANK_MODE` | 设为 `local` 使用本地 BGE reranker |

**RAGAS judge / embedding 口径**：

| 配置项 | 值 |
|--------|-----|
| judge model | `qwen-plus`（`RAGAS_JUDGE_MODEL` 未设置时的默认值） |
| judge API | `QWEN_CHAT_API_BASE` / `QWEN_CHAT_API_KEY` |
| embedding provider | `local_bge` |
| embedding model | `models/BAAI/bge-m3` |
| embedding dim | `1024` |

---

## 4. 当前评测命令

### 4.1 启动 RAG API 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

### 4.2 数据集校验

```bash
python scripts/evaluation/validate_enterprise_dataset.py
```

### 4.3 Quick Check 10

构建 10 条样本的 RAGAS dataset 并评测：

```bash
# Step 1: 构建 10 条 dataset
python scripts/evaluation/build_ragas_dataset.py \
  --base-url http://127.0.0.1:9000 \
  --context-source final_chunks \
  --include-debug \
  --max-samples 10

# Step 2: 运行 RAGAS
python scripts/evaluation/run_ragas_smoke100.py \
  --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \
  --metrics context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness \
  --max-samples 10
```

### 4.4 Full Smoke100

```bash
# Step 1: 构建 100 条 dataset（需要 API 运行中）
python scripts/evaluation/build_ragas_dataset.py \
  --base-url http://127.0.0.1:9000 \
  --context-source final_chunks \
  --include-debug \
  --timeout 600

# Step 2: 运行 RAGAS（约 22 分钟，500 次 LLM judge 调用）
python scripts/evaluation/run_ragas_smoke100.py \
  --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \
  --metrics context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness \
  --timeout 600

# Step 3: 合并项目指标
python scripts/evaluation/merge_ragas_with_eval_metrics.py \
  --ragas results/ragas/smoke100_<timestamp>/ragas_scores.jsonl \
  --output-dir results/ragas/smoke100_<timestamp>/

# Step 4: 生成 calibrated review candidates
python scripts/evaluation/generate_review_candidates.py \
  --input results/ragas/smoke100_<timestamp>/ragas_eval_joined.jsonl \
  --output-dir results/ragas/smoke100_<timestamp>/
```

### 4.5 Phase 11B / 11C 一键 smoke100 回归

Phase 11B 新增一键评测工具，封装：

1. build responses / RAGAS dataset
2. run RAGAS
3. merge project metrics
4. generate review candidates
5. generate warning-only baseline regression report

默认 `--preset stable` 回归配置：不开启 comparison profile，不开启 Qwen synthesis，不开启 neighbor audit / promotion。该口径用于稳定评测工具链，不可直接当作 Phase 9 baseline 退化结论。

Phase 9 accepted baseline 复现必须使用 `--preset phase9_accepted`。该 preset 会显式打印并写入 manifest：

- dataset path / sha256 / sample_count / sample IDs 校验
- generation profile
- Qwen synthesis flag
- comparison coverage flag
- neighbor audit / promotion flags
- base-url
- timestamp
- 是否与 Phase 9 baseline 可比

`run_smoke100_regression.py` 只支持一个 `--base-url`。如需使用 `9010` 等临时端口，应同时传入 `--port 9010` 和 `--base-url http://127.0.0.1:9010`；不要在同一命令中传两个 `--base-url`。

Phase 9 可比复现：

```bash
python scripts/evaluation/run_smoke100_regression.py \
  --preset phase9_accepted \
  --base-url http://127.0.0.1:9000
```

Phase 9 可比复现并由脚本临时启动 API 服务：

```bash
python scripts/evaluation/run_smoke100_regression.py \
  --preset phase9_accepted \
  --start-server \
  --base-url http://127.0.0.1:9000
```

如果 API 服务已经用对应配置启动：

```bash
python scripts/evaluation/run_smoke100_regression.py \
  --preset stable \
  --base-url http://127.0.0.1:9000
```

如果需要脚本临时启动并在结束后停止 API 服务：

```bash
python scripts/evaluation/run_smoke100_regression.py \
  --preset stable \
  --start-server \
  --base-url http://127.0.0.1:9000
```

输出目录：`results/ragas/smoke100_<timestamp>/`

关键输出：

| 文件 | 说明 |
|------|------|
| `smoke100_pipeline_manifest.json` | 本次一键流程配置与命令记录 |
| `ragas_scores.jsonl` | 原始 RAGAS per-sample 结果 |
| `ragas_summary.json` / `ragas_summary.md` | RAGAS 汇总 |
| `ragas_eval_joined.jsonl` | RAGAS 与项目指标合并结果 |
| `human_review_candidates_calibrated.csv` | review candidates 与规则生成优先级 |
| `calibration_summary.json` | raw P0 / rule-review candidate 汇总；不要将 `rule_review_candidate_count` 当作人工 calibrated P0 |
| `baseline_regression_report.json` / `.md` | Phase 9 baseline warning-only 回归对比，含 Phase 10B noise ledger 排除明细 |

### 4.6 数据集路径

Phase 9 accepted baseline 的文档历史名为：`data/eval/datasets/enterprise_ragas_eval_v1.json`（100 条）。

当前工作区该文件已不存在；`archive/backups/enterprise_ragas_eval_v1.json.bak_before_final_dataset_cleanup` 是 final dataset cleanup 之前的备份，不等同于 Phase 9 accepted baseline 使用的最终 smoke100 口径。

当前 canonical smoke100 数据集为：

`data/eval/datasets/enterprise_ragas_smoke100.json`

manifest / provenance：

| 字段 | 值 |
|------|-----|
| sample_count | `100` |
| sample IDs | `ent_001` 到 `ent_100` |
| sha256 | `1e413d826dad87ad324dfa6cf9d2a6fe4897d6a5a55cacad54455aeaf1e4230e` |
| Phase 9 output match | `results/ragas/smoke100_20260504_214135/ragas_scores.jsonl` 的 question、expected_doc_ids、expected_section_groups 与该文件一致；`ent_021` 因 API collection error 在输出中保留为空 reference/route/scenario |

`--preset phase9_accepted` 使用上述 canonical 文件；如果该文件缺失，脚本会停止，不会 fallback 后继续声称是 Phase 9 baseline 对比。`--preset stable` 可保留 fallback 行为，但 regression report 会标记为 non-comparable run。

---

## 5. 当前不做的事项（Scope Freeze）

| 方向 | 不做的原因 |
|------|-----------|
| Fix C（factoid entity/numeric validation） | 仅 1 条 factoid P0，术语映射表未就绪 |
| Claim-level citation validation | 无统一 failure pattern；5/8 P0 非真实问题 |
| Comparison branch guardrail | 仅 1 条 comparison P0（ent_010） |
| KB re-chunking / section labeling | 需独立 KB 项目；仅 1 条 P0 受此影响 |
| Judge model 更换 | 当前 judge 噪声已知、可管理；更换引入新变量 |
| Retrieval/rerank 参数调整 | context_recall 下降 -2.5%，暂无证据是 retrieval 退化 |
| Qwen prompt 修改 | 当前 answers 无幻觉、有 citation；不需要调 prompt |

---

## 6. Evaluation Noise Ledger

**文件**: `results/ragas/smoke100_20260504_214135/phase10b_evaluation_noise_ledger.csv`

记录 5 个非真实 P0 样本，标记 `should_count_as_p0=false`：

| Sample | Type | 说明 |
|--------|------|------|
| ent_021 | api_data_collection_error | API call failed |
| ent_055 | judge_artifact | 答案正确但 judge 过严 |
| ent_012 | reference_naming_mismatch | 中文名 vs 英文缩写 mismatch |
| ent_084 | comparison_old_format | 仍用旧 evidence-snippet 格式 |
| ent_022 | acceptable_partial | 自认 comparison_evidence_incomplete |

**用途**：这些样本不作为主链路修复依据；后续 smoke100 重跑时可参考此 ledger 排除已知噪声。

---

## 7. Real Issue Backlog

**文件**: `results/ragas/smoke100_20260504_214135/phase10b_real_issue_backlog.csv`

记录 3 个真实问题，附带触发条件：

| Sample | Issue | Trigger to Revisit |
|--------|-------|-------------------|
| ent_010 | comparison_branch_miss | ≥3 comparison P0 |
| ent_032 | summary_detail_missing（定量缺失） | ≥3 定量 summary P0 |
| ent_040 | summary_fragment_evidence（KB limit） | KB re-chunking 项目启动 |

**用途**：同类问题累计 ≥3 时启动专项修复；当前每类仅 1 条，不做针对性修改。

---

## 8. 后续触发条件

| 触发条件 | 行动 |
|---------|------|
| comparison_branch_miss ≥3 | 开启 comparison branch guardrail (per-branch support_pack + citation) |
| summary_detail_missing ≥3（定量题） | 开启 summary detail evidence fix (Results/Discussion body supplements Abstract) |
| KB section 问题成批出现或启动 KB 更新 | KB section labeling / re-chunking，让 Phase 7 supplement 生效 |
| judge_artifact 多轮持续 >10 条，且人工审核确认影响工程决策 | 考虑 secondary judge（如 gpt-4o）对照实验 |
| factoid_entity_mismatch ≥3 | 开启 Fix C（需要术语映射表就绪） |
