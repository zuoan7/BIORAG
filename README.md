# SynBio RAG Service (prototype-v1)

面向合成生物学领域的知识问答 RAG 服务。支持论文知识库的语义检索、混合检索（Dense + BM25）、重排序与证据约束生成。

## 核心特性

- **混合检索**: Dense（BGE-M3）+ BM25 + RRF 融合
- **智能路由**: QueryRouter 自动分析问题意图（factoid / summary / comparison）
- **多级重排**: 本地 BGE-Reranker / LLM rerank / 启发式兜底
- **Section fallback**: 自动修复 PDF parser 产生的 section metadata 错误
- **Body coverage** (可选): post-rerank same-doc body section 覆盖，预算内替换
- **证据约束生成**: 两阶段生成，支持会话与审计日志

## 技术栈

- Python 3.10+
- FastAPI + Uvicorn
- Milvus Lite（本地向量存储）
- BGE-M3（Embedding）/ BGE-Reranker-v2-m3（重排）
- Qwen / OpenAI-compatible LLM（生成）

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

> 详细步骤见 [docs/startup.md](docs/startup.md)。

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

## 项目结构

```
bge-service/
├── app/                    # FastAPI 入口
├── config/                 # 环境变量模板
├── data/                   # 论文、评测集
├── docs/                   # 文档
├── models/                 # 本地模型（BGE-M3、BGE-Reranker）
├── reports/                # 评测报告
├── runtime/                # 运行时产物（日志、向量库）
├── scripts/
│   ├── evaluation/         # 离线评测
│   ├── ingestion/          # 建库、局部重建
│   └── ops/                # 调试工具
├── src/synbio_rag/         # 业务源码
├── tests/                  # 测试
├── results/                # 诊断与修复报告
│   ├── early-diagnosis/    # 早期诊断
│   ├── phase-reports/      # Phase 1-7 阶段报告
│   └── final/              # 最终收口报告
└── archive/                # 历史备份与旧评测数据
```

## 评测

### smoke100 门禁

```bash
# baseline（默认配置）
python scripts/evaluation/run_generation_smoke100.py

# 开启 body coverage
RETRIEVAL_SAME_DOC_BODY_COVERAGE_ENABLED=true \
python scripts/evaluation/run_generation_smoke100.py
```

### prototype-v1 结果

| 指标 | baseline | coverage-enabled |
|------|----------|-----------------|
| retrieval_section_miss | 1 | **0** |
| section_norm_hit_rate | 0.883 | **0.894** |
| doc_id_hit_rate | 0.9255 | 0.9255 |
| route_match_rate | 0.89 | 0.89 |
| regression | — | **0** |

### 其他评测命令

| 命令 | 说明 |
|------|------|
| `python scripts/evaluation/validate_enterprise_dataset.py` | 数据集校验 |
| `python scripts/evaluation/evaluate_ragas.py --base-url http://127.0.0.1:9000` | RAGAS 评测 |
| `python scripts/evaluation/run_ragas_regression.py --base-url http://127.0.0.1:9000` | RAGAS 回归 |
| `python scripts/evaluation/build_diagnostics_ledger.py <report_dir>` | 生成诊断账本 |
| `python scripts/ops/interactive_rag_cli.py` | 交互式调试 |

## 关键配置

核心参数在 `src/synbio_rag/domain/config.py`，可通过 `.env` 覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RETRIEVAL_SEARCH_LIMIT` | `40` | 混合检索召回数 |
| `RETRIEVAL_RERANK_TOP_K` | `10` | Rerank 后保留数 |
| `RETRIEVAL_FINAL_TOP_K` | `8` | 最终 seed chunk 数 |
| `RETRIEVAL_SAME_DOC_BODY_COVERAGE_ENABLED` | `false` | 同文档 body coverage（可选） |
| `RETRIEVAL_SECTION_GROUP_COVERAGE_LEVEL2_ENABLED` | `false` | Level 2 group coverage（实验性） |

完整参数列表见 `src/synbio_rag/domain/config.py`。

## 检索修复记录 (prototype-v1)

原始 12 个 retrieval_section_miss 已全部修复。详见 [results/final/BIORAG_Prototype_v1_Retrieval_Fix_Summary.md](results/final/BIORAG_Prototype_v1_Retrieval_Fix_Summary.md)。

## 文档

- [启动说明](docs/startup.md)
- [最终修复总结](results/final/BIORAG_Prototype_v1_Retrieval_Fix_Summary.md)

## License

MIT
