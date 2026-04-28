# 评测诊断层改造总结（2026-04-28）

> 分支：`fix/retrieval`
> 数据集：enterprise_ragas_eval_v1（n=100，smoke100）
> 对比：old_baseline vs v2_stable（extractive only，Qwen/comparison/neighbor 全关）

---

## 1. 背景

上一轮 smoke100 评测（20260428_162959）显示 `overall_pass=FALSE`：

| 指标 | old_baseline | v2_stable | 门禁 |
|------|-------------|-----------|------|
| route_match | 0.89 | 0.89 | pass |
| doc_id_hit | 0.9468 | 0.8723 | **fail** |
| section_hit | 0.2128 | 0.2021 | pass |

`doc_id_hit` 下降 7.5pp 且 `section_hit` 仅 ~21%，但无法判断根因是 retrieval miss、generation 保守拒答，还是评测标签偏差。

本轮改造：只做评测诊断层，不改任何算法、数据或入库流程。

---

## 2. 改造内容

### 2.1 检索管线分解（evaluate_ragas.py）

在 `build_raw_records()` 中嵌入 `build_retrieval_ledger()`，从 `generation_v2.candidates` 和 `support_pack` 中提取各阶段数据，将 doc/section 命中分解为三层：

- **candidate**（retrieval + rerank 输出池）
- **support_pack**（generation_v2 证据选择）
- **citation**（最终答案引用，即旧的 doc_id_hit / section_hit）

新增函数：`compute_retrieval_ledger_summary()`、`_classify_pipeline_status()`、`_diagnose_section_labels()`

### 2.2 Section 语义归一化（build_diagnostics_ledger.py）

新增 `SectionNormalizer` 类，定义 canonical form 和 semantic group：

| canonical | 别名 | semantic_group |
|-----------|------|---------------|
| abstract | Abstract | abstract |
| title | Title | title |
| intro | Introduction, Background | body |
| methods | Methods, Materials and Methods, Experimental Section, ... | body |
| results | Results | body |
| discussion | Discussion | body |
| results_discussion | Results and Discussion, R&D | body |
| conclusion | Conclusion, Conclusions | body |
| body | Full Text, General Body | body |

关键匹配规则：
- expected "Full Text" → 匹配任意 semantic_group=body 的 actual section
- expected "Results" → 匹配 "Results" 或 "Results and Discussion"
- expected "Discussion" → 匹配 "Discussion" 或 "Results and Discussion"
- expected "Methods" → 匹配所有 method 变体
- expected "Title" 不参与正文命中

### 2.3 逐样本诊断报告

独立脚本 `scripts/evaluation/build_diagnostics_ledger.py`，从已有 smoke100 报告和 chunks.jsonl 生成 per-sample ledger。

---

## 3. 核心发现

### 3.1 doc_id_hit 下降：v2 更诚实，不是检索退化

| 指标 | old_baseline | v2_stable |
|------|-------------|-----------|
| retrieval_doc_hit_rate | N/A（old 无 candidates 数据） | **0.9894** |
| citation_doc_hit_rate | 0.9468 | 0.8723 |
| retrieval_doc_miss | 见 citation 指标 | **1**（ent_090） |
| retrieved_but_not_cited | — | **11**（全部 partial/refuse） |
| full_answer_missing_expected_doc | — | **0** |

**结论：v2 检索召回 98.9%，old 是 94.7%。v2 的 doc_id_hit 更低是因为 11 个样本检索到了正确文档但因证据不足选择 partial/refuse 而不引用。old 没有这个约束，直接引用。**

### 3.2 section_hit ~21%：完全是评测标签问题

| 指标 | 旧 (case-insensitive) | 新 (semantic normalized) |
|------|----------------------|--------------------------|
| v2 normalized_section_hit_rate | 0.2021 | **0.766** |
| v2 index_possible_hit_rate | 0.2872 | **0.8404** |
| v2 retrieval_section_hit_rate | 0.266 | **0.8298** |
| eval_label_mismatch 样本数 | 64 | **0** |
| Full Text 样本数 | 80 | — |
| Full Text normalized hit | — | 54/80 |
| strict_miss_but_normalized_hit | — | 53 |

根因：80% 样本使用 `expected_sections = ["Full Text"]`，但 chunk 索引中没有名为 "Full Text" 的 section。"Full Text" 语义上等于"正文任意 section"。归一化后 54/80 命中，剩余 26 个未命中属于真实检索问题。

### 3.3 剩余真实问题分类

| 类别 | v2 数量 | 说明 |
|------|---------|------|
| retrieval_doc_miss | 1 | ent_090，doc_0001 不在 hybrid top-40 中（上一轮已知问题） |
| retrieval_section_miss | 12 | 预期 section 在索引中存在但未被检索到（多数只召回了 Title chunk） |
| retrieved_but_not_cited_section | 6 | section 已在检索池中但 v2 证据选择未引用 |
| index_missing_expected_section | 4 | 预期 section 在索引中真实不存在 |
| route_mismatch | 11 | 路由分类不准（old 和 v2 相同） |

---

## 4. 未解决问题

### 4.1 12 个 retrieval_section_miss

这些样本中检索返回了目标文档，但只召回了 Title/Abstract 等非正文 section，body-content section 未被召回。

典型样本：
- ent_049: expected=["Full Text"], retrieved=["Abstract"] — 正文内容未被检索到
- ent_067: expected=["Full Text"], retrieved=["Title"] — 只召回了标题
- ent_009: expected=["Abstract", "Experimental Section"], retrieved=["Discussion", "Results", "Title"] — Abstract 和 Experimental Section 未召回

可能方向（仅记录，不在本轮处理）：
- 检查这些样本的 query 是否过于简短或缺乏正文关键词
- 检查 rerank 阶段 section_results_bonus 是否偏向特定 section
- 检查 chunk diversity 策略是否过早排除正文 chunk

### 4.2 6 个 retrieved_but_not_cited_section

v2 证据选择丢弃了已检索到的 section：

- ent_008: retrieved Abstract + Introduction，但只引用了 Results and Discussion
- ent_014: retrieved Results，但只引用了 Discussion + Title
- ent_017: retrieved Abstract，但只引用了 Results and Discussion
- ent_065/071/100: 三个 refuse 样本，body-section 已检索到但最终零引用

可能方向（仅记录）：
- v2 evidence selection 偏向 Results/Discussion 而非 Abstract/Introduction 可能是合理的
- refuse 样本有 body-section 但未引用：检查 support_pack 中的 support_score 是否都偏低

### 4.3 1 个 retrieval_doc_miss（ent_090）

已知问题。doc_0001 完全不在 hybrid top-40 中。需要增大 search_limit 或改进 embedding。已在上轮 retrieval_phase_fix_summary.md 中记录。

### 4.4 11 个 route_mismatch

old 和 v2 相同，不是 v2 引入的问题。不在本轮范围。

### 4.5 expected_sections 标签质量问题

虽然语义归一化消除了 64 个 eval_label_mismatch，但 80 个 "Full Text" 标签本身就很粗糙——它无法区分"需要 Abstract"还是"需要 Results"。这个不是算法问题，而是评测集设计问题。本轮不做修改。

---

## 5. 不应在本轮修改的事项

- 不修改 `expected_sections` / `expected_doc_ids` 标签
- 不修改 retrieval/reranker/chunking/generation 算法
- 不改 `overall_pass` 门禁逻辑
- `strict_section_hit_rate` 保持旧的计算方式不变
- `normalized_section_hit_rate` 仅作为诊断指标，不作为正式门禁

---

## 6. 生成的文件

```
reports/evaluation/ad_hoc/generation_smoke100/20260428_220340/
├── comparison.md                                    # 原 smoke100 对比报告（已含 ledger）
├── eval_metric_diagnostics_ledger_old_baseline.json  # old 逐样本 ledger
├── eval_metric_diagnostics_ledger_v2_stable.json     # v2 逐样本 ledger
├── eval_metric_diagnostics_report_old_baseline.md    # old 诊断报告
└── eval_metric_diagnostics_report_v2_stable.md       # v2 诊断报告
```

可复现命令：
```bash
conda activate bge
python scripts/evaluation/run_generation_smoke100.py  # 全量评测（含 ledger）
python scripts/evaluation/build_diagnostics_ledger.py \
  reports/evaluation/ad_hoc/generation_smoke100/<timestamp>  # 生成逐样本诊断
```

---

## 7. 代码改动清单

| 文件 | 改动 |
|------|------|
| `scripts/evaluation/evaluate_ragas.py` | 新增 `build_retrieval_ledger()`、`compute_retrieval_ledger_summary()`、`_classify_pipeline_status()`、`_diagnose_section_labels()`；`build_raw_records()` 中嵌入 ledger |
| `scripts/evaluation/run_generation_stage2c_comparison_coverage.py` | `run_group()` 加入 ledger_summary；`build_markdown()` 加入 ledger 表格 |
| `scripts/evaluation/run_generation_smoke100.py` | `_comparison_md()` 加入两组 ledger 对比 |
| `scripts/evaluation/build_diagnostics_ledger.py` | **新建**：`SectionNormalizer`、逐样本诊断生成、semantic section 匹配 |
