# BIORAG — 合成生物学领域 RAG 服务

面向合成生物学领域的知识问答 RAG 服务。支持论文知识库的语义检索、混合检索（Dense + BM25）、重排序与证据约束生成。

## 当前项目状态

当前 accepted baseline 是 **Phase 20 convergence baseline**（Phase 20M）。

权威实验基线文档：[reports/phase20m_convergence_summary/summary.md](reports/phase20m_convergence_summary/summary.md)

### Final Phase 20 核心指标 (smoke50 + smoke100 = 150 samples)

| 指标 | Phase 20A (rewrite baseline) | Phase 20L-2 (final) |
|------|---------------------------|---------------------|
| corrected real P0 | 9 | **0** |
| doc_miss | 9 | **0** |
| doc_hit_rate | 0.97 | **1.0** |
| zero_citation | 0 | 0 |
| new real P0 | - | 0 |
| wrong-doc citation | - | 0 |
| citation inflation | - | 0 |

### Phase 20 已接受的修复

| Fix | Phase | Module | 说明 |
|-----|-------|--------|------|
| Factoid doc diversity | 20D | support_selector | `_select_factoid` 新增 max_per_doc=2，防止单文档独占 support |
| Summary quality filter | 20G | support_selector | 移除 title hard-drop，Full Text section priority=2 |
| Comparison decomposition | 20K | hybrid/pipeline | CN query 用于 comparison subquery，修复 period bug |
| CN fallback floor | 20L-2 | pipeline/config | EN rewrite 后 CN fallback，dense 2 + BM25 2，max 4 candidates |

### 生产默认 vs 实验最佳配置

| Feature Flag | 生产默认 | 离线实验推荐 |
|-------------|---------|------------|
| QUERY_REWRITE_MODE | `off` | `enabled` |
| original_cn_fallback_enabled | `false` | `true` |
| alias_expansion_enabled | `false` | `false` |
| qwen_synthesis | `false` | `false` |

生产环境默认不改动，所有实验功能通过 feature flag 或 env var 开启。

## 核心特性

- **混合检索**: Dense（BGE-M3）+ BM25 + RRF 融合
- **智能路由**: QueryRouter 自动分析问题意图（factoid / summary / comparison）
- **英文镜像查询改写**: 中文查询 → 英文检索查询，显著改善跨语言召回
- **多级重排**: BGE-Reranker-v2-m3 / subquery aggregation / 启发式兜底
- **Parent-child 索引**: chunk_window / section_path / caption_context / page 7 种 parent 类型
- **Source-floor**: 保留 dense/BM25 各 top-N 候选，防止 RRF 融合丢失
- **证据约束生成**: v2 generation pipeline，支持 support selection / citation binding / answer validation

## 技术栈

- Python 3.10+
- FastAPI + Uvicorn
- Milvus Lite（本地向量存储）
- BGE-M3（Embedding）/ BGE-Reranker-v2-m3（重排）
- Qwen-plus / OpenAI-compatible LLM（生成）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp config/settings.example.env .env
# 按需编辑 .env，配置模型路径、API 密钥等
```

### 3. 构建知识库

```bash
python scripts/ingestion/build_round1_kb.py
```

### 4. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

如需独立 reranker 服务：

```bash
uvicorn app.reranker_main:app --host 0.0.0.0 --port 9001
```

### 5. 请求示例

```bash
curl -X POST http://127.0.0.1:9000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "比较 2'\''-FL 和 6'\''-SL 的工程化合成路径差异",
    "session_id": "demo-session-1",
    "tenant_id": "default"
  }'
```

## 关键配置

核心参数在 `src/synbio_rag/domain/config.py`，可通过 `.env` 覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QUERY_REWRITE_MODE` | `off` | 英文镜像查询改写（`off`/`shadow`/`enabled`） |
| `RETRIEVAL_SOURCE_FLOOR_ENABLED` | `true` | 保留 dense/BM25 各 top-N 候选 |
| `RETRIEVAL_SOURCE_FLOOR_DENSE_TOP_N` | `3` | Dense source-floor 保留数 |
| `RETRIEVAL_SOURCE_FLOOR_BM25_TOP_N` | `3` | BM25 source-floor 保留数 |
| `RETRIEVAL_PARENT_EXPANSION_ENABLED` | `true` | Parent-child index 扩展 |
| `RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED` | `false` | EN rewrite 后 CN fallback floor |
| `RETRIEVAL_ORIGINAL_CN_FALLBACK_MAX_TOTAL` | `4` | CN fallback 最大候选数 |
| `RETRIEVAL_SEARCH_LIMIT` | `40` | 混合检索召回数 |
| `RETRIEVAL_RERANK_TOP_K` | `10` | Rerank 后保留数 |
| `RETRIEVAL_FINAL_TOP_K` | `8` | 最终 seed chunk 数 |
| `GENERATION_VERSION` | `v2` | 生成 pipeline 版本 |

完整参数列表见 `src/synbio_rag/domain/config.py`。

## 评测

当前 smoke100 / smoke50 评测以 Phase 20 脚本为准。

```bash
# 全量离线评测 (smoke50 + smoke100)
python scripts/evaluation/run_phase20l2_cn_fallback_feature.py

# Phase 20 相关测试
pytest tests/test_phase20*.py -v
```

## 项目结构

```
biorag/
├── app/                    # FastAPI 入口
├── config/                 # 环境变量模板
├── data/                   # 论文、评测集
│   ├── eval/               # 评测数据集 (smoke50, smoke100)
│   └── paper_round1/       # 生产知识库 (chunks, parent_index)
├── models/                 # 本地模型（BGE-M3、BGE-Reranker）
├── reports/                # Phase 评测报告
├── runtime/                # 运行时产物（Milvus、日志）
├── scripts/
│   ├── evaluation/         # 离线评测脚本 (Phase 20A-20M)
│   ├── ingestion/          # 建库与预处理
│   └── ops/                # 调试工具
├── src/synbio_rag/         # 业务源码
│   ├── application/        # pipeline, generation_v2, rerank
│   ├── domain/             # config, schemas, router
│   ├── infrastructure/     # vectorstores (hybrid, BM25, Milvus)
│   ├── rewrite/            # query rewrite service
│   └── evaluation/         # failure taxonomy
├── tests/                  # 测试 (Phase 20: 30 tests)
├── results/                # Phase 实验结果 (20A-20M)
└── resources/prompts/      # LLM prompt 模板
```

## Phase 20 文档索引

- [Phase 20M 收敛总结](reports/phase20m_convergence_summary/summary.md)
- [Phase 20M 复习笔记](reports/phase20m_convergence_summary/review_notes.md)
- [Phase 20L-2 CN Fallback 报告](reports/phase20l2_original_cn_fallback_floor_feature_ab/summary.md)
- [Phase 20K Comparison Decomposition 报告](reports/phase20k_comparison_decomposition_fix/summary.md)
- [Phase 20G Summary Quality 报告](reports/phase20g_summary_quality_minimal_fix/summary.md)
- [Phase 20E Factoid Diversity 报告](reports/phase20e_support_diversity_rebaseline/summary.md)
- [当前实验最佳配置](results/phase20m_convergence_summary/current_best_config.md)

## License

MIT
