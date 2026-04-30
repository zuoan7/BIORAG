# RAGAS 自动化测评使用说明

## 1. 为什么引入 RAGAS

BIORAG 项目已有项目自定义的结构化指标：

- `doc_id_hit`: 期望文档是否被检索并引用
- `section_norm_hit`: 期望 section（group-level）是否命中
- `citation_count`: 引用数量
- `route_match_rate`: 路由匹配率

这些指标评估的是**检索结构**，但无法回答以下问题：

- 检索到的上下文是否**语义上覆盖**了参考答案？（context_recall）
- 检索到的上下文中，有多少是**真正相关**的？（context_precision）
- 生成的答案是否**忠实于**检索到的证据？（faithfulness）
- 答案与问题的**相关度**如何？（answer_relevancy）
- 答案中的事实是否与参考答案**一致**？（factual_correctness）

**RAGAS 填补了语义质量评估的空白**，与项目结构化指标互补。

## 2. RAGAS 指标与现有指标的关系

```
检索结构指标 (project)         语义检索指标 (RAGAS)        答案质量指标 (RAGAS)
─────────────────────────    ────────────────────────    ───────────────────────────
doc_id_hit                   context_recall              faithfulness
section_norm_hit             context_precision           answer_relevancy
citation_count                                          factual_correctness
route_match_rate
```

**互补关系**，不是替代关系：

- `doc_id_hit=true` + `faithfulness=low`：检索到了正确文档，但答案可能没忠实使用证据
- `doc_id_hit=false` + `factual_correctness=high`：可能 reference 标注需要复核
- `section_norm_hit=true` + `context_recall=low`：section 标签命中了但上下文语义覆盖不足

## 3. 如何构建 RAGAS Dataset

### 3.1 调用 RAG API 获取答案

```bash
# 使用 final_chunks（全部候选 chunks）作为 contexts
python scripts/evaluation/build_ragas_dataset.py \
  --base-url http://127.0.0.1:9000 \
  --context-source final_chunks \
  --include-debug

# 使用 cited_chunks（仅引用 chunks）作为 contexts
python scripts/evaluation/build_ragas_dataset.py \
  --base-url http://127.0.0.1:9000 \
  --context-source cited_chunks
```

**输出文件**:

- `results/ragas/smoke100_ragas_dataset_final_chunks.jsonl`
- `results/ragas/smoke100_ragas_dataset_cited_chunks.jsonl`

### 3.2 Context 模式说明

| 模式 | 来源 | 适用场景 |
|------|------|---------|
| `final_chunks` | `api_response.debug.generation_v2.candidates` | 评估检索+生成整体链路 |
| `cited_chunks` | `api_response.citations[].quote` | 严格评估引用是否足以支撑答案 |

**建议**: 默认使用 `final_chunks`，在需要严格评估 citation 质量时使用 `cited_chunks`。

## 4. 如何运行 RAGAS Smoke100

### 4.1 环境变量

```bash
export QWEN_CHAT_API_BASE=https://your-api-endpoint/v1
export QWEN_CHAT_API_KEY=your-api-key
export RAGAS_JUDGE_MODEL=qwen-plus        # 可选，默认 qwen-plus
```

### 4.2 运行完整评测

```bash
python scripts/evaluation/run_ragas_smoke100.py \
  --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \
  --output-dir results/ragas/smoke100_$(date +%Y%m%d_%H%M%S) \
  --metrics context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness
```

### 4.3 跳过某些指标

```bash
python scripts/evaluation/run_ragas_smoke100.py \
  --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \
  --skip-metrics factual_correctness \
  --metrics context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness
```

### 4.4 限制样本数（快速测试）

```bash
python scripts/evaluation/run_ragas_smoke100.py \
  --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \
  --max-samples 10
```

## 5. 输出文件说明

### 构建阶段 (`build_ragas_dataset.py`)

| 文件 | 内容 |
|------|------|
| `smoke100_ragas_dataset_<mode>.jsonl` | RAGAS 格式数据集，每行一个样本 |
| `smoke100_ragas_dataset_<mode>_summary.json` | 数据集统计摘要 |

### 测评阶段 (`run_ragas_smoke100.py`)

| 文件 | 内容 |
|------|------|
| `ragas_scores.jsonl` | 每行包含元数据 + ragas_scores 的 per-sample 文件 |
| `ragas_summary.json` | JSON 格式汇总（全局 + 分组指标） |
| `ragas_summary.md` | Markdown 格式汇总报告 |
| `ragas_low_score_cases.md` | 每个指标 Bottom 10 样本列表 |

### 合并阶段 (`merge_ragas_with_eval_metrics.py`)

| 文件 | 内容 |
|------|------|
| `ragas_eval_joined.jsonl` | RAGAS 分数 + 项目指标合并 |
| `ragas_eval_joined_summary.md` | 交叉分析报告 |

### 审核候选 (`generate_review_candidates.py`)

| 文件 | 内容 |
|------|------|
| `human_review_candidates.csv` | 人工审核候选集，包含优先级和推测问题类型 |

## 6. 如何查看低分样本

### 6.1 直接查看 Markdown 报告

```bash
cat results/ragas/<timestamp>/ragas_low_score_cases.md
```

### 6.2 用 jq 过滤 JSONL

```bash
# 找 faithfulness < 0.5 的样本
cat results/ragas/<timestamp>/ragas_scores.jsonl | \
  python3 -c "
import sys, json
for line in sys.stdin:
    item = json.loads(line)
    scores = item.get('ragas_scores', {})
    faith = scores.get('faithfulness')
    if isinstance(faith, (int, float)) and faith < 0.5:
        print(json.dumps({
            'sample_id': item['sample_id'],
            'faithfulness': faith,
            'answer': item['answer'][:120],
        }, ensure_ascii=False))
"
```

## 7. 如何生成人工审核候选集

```bash
python scripts/evaluation/generate_review_candidates.py \
  --input results/ragas/<timestamp>/ragas_scores.jsonl \
  --output-dir results/ragas/<timestamp>/
```

CSV 中的 `human_*` 字段留空，由人工审核时填写。

## 8. 注意事项

### RAGAS 不能完全替代人工审核

- RAGAS 是自动化回归工具，用于快速发现**候选问题**样本
- 最终的答案质量判定仍需人工确认
- 建议将 RAGAS 作为 CI 中的**预警信号**而非 gate

### factual_correctness 依赖 reference answer 质量

- 如果 reference 写得过于简略，正常答案可能被判为"不完整"
- 如果 reference 包含数据集中未提及的信息，会导致全部样本分数偏低
- **禁止为了提高 RAGAS 分数而修改 reference**

### faithfulness != correctness

- faithfulness 衡量答案是否被上下文**支持**
- 上下文本身可能不完整或不正确
- 需要结合 factual_correctness 和人工判断

### judge LLM 可能误判专业术语

- 合成生物学领域的中英文术语、缩写、同义表达可能被 judge 误判
- 建议人工审核时关注术语层面的 false positive/negative

### negative / abstain 样本需要单独解释

- 拒答样本的 faithfulness 可能异常偏高（因为没有做出任何声称）
- 在计算全局平均值时，应按 `answer_mode` 分组展示
- 不要将拒答样本的 RAGAS 分数与正常样本直接比较

### 不要过度依赖分数

- RAGAS 分数是趋势指标，不是绝对质量标准
- 一个分数为 0.75 的答案不一定比 0.80 的差很多
- 关注分数**分布**和**极端值**，而非小数点后的差异
