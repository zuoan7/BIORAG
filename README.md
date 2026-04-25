# SynBio RAG Service

面向合成生物学领域的知识问答 RAG（Retrieval-Augmented Generation）服务。支持基于论文知识库的语义检索、混合检索（Dense + BM25）、重排序、邻居扩展与证据约束生成。

## 核心特性

- **混合检索**：Dense（BGE-M3）+ BM25 + RRF 融合
- **智能路由**：QueryRouter 自动分析问题意图（factoid / summary / comparison / experiment）
- **多级重排**：独立 reranker 服务 / 本地 BGE-Reranker / LLM rerank / 启发式兜底
- **邻居扩展**：ChunkNeighborExpander 补齐上下文，减少断句
- **证据约束生成**：两阶段生成（Claim 抽取 → 答案合成），低置信度自动触发外部文献检索
- **会话与审计**：多轮会话支持，完整审计日志

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

> 首次使用需要准备知识库。详细步骤见 [docs/startup.md](docs/startup.md)。

### 4. 启动服务

启动主 RAG 服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

如需独立 reranker 服务：

```bash
uvicorn app.reranker_main:app --host 0.0.0.0 --port 8001
```

并在 `.env` 中配置：

```bash
RERANKER_SERVICE_URL=http://127.0.0.1:8001
```

### 5. 请求示例

```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "比较 2'-FL 和 6'-SL 的工程化合成路径差异",
    "session_id": "demo-session-1",
    "tenant_id": "acme-bio",
    "sections": ["Results", "Discussion"],
    "include_debug": true
  }'
```

返回字段：

- `answer`：生成答案
- `confidence`：置信度评分
- `route`：问题路由类型
- `citations`：引用来源
- `used_external_tool`：是否使用外部文献工具
- `external_references`：外部文献引用
- `debug.*`：调试信息（检索命中、重排结果、邻居扩展等）

## 项目结构

```
bge-service/
├── app/                    # FastAPI 入口（主服务 / Reranker 服务）
├── config/                 # 环境变量模板
├── data/                   # 数据目录（论文、评测集）
│   ├── eval/
│   ├── paper_round1/
│   └── patent_round1/
├── docs/                   # 文档（启动说明、项目规范等）
├── models/                 # 本地模型资产（BGE-M3、BGE-Reranker）
├── reports/                # 评测报告与回归结果
├── runtime/                # 运行时产物（日志、向量库）
├── scripts/                # 脚本
│   ├── data_prep/          # 数据准备
│   ├── evaluation/         # 离线评测与回归
│   ├── ingestion/          # 建库与入库
│   └── ops/                # 运维调试工具
├── src/synbio_rag/         # 业务源码
│   ├── application/        # 应用层（服务、Pipeline）
│   ├── domain/             # 领域层（配置、路由、检索器）
│   └── infrastructure/     # 基础设施层
├── tests/                  # 测试（smoke / integration / evaluation）
├── pyproject.toml          # 项目配置
├── pytest.ini              # 测试配置
└── requirements.txt        # 依赖
```

## RAG 主流程

```text
用户请求 /v1/ask
  -> 读取 session history
  -> QueryRouter 分析意图
  -> HybridRetriever 检索
     -> Dense（Milvus + BGE-M3）
     -> BM25
     -> RRF 融合
  -> Reranker 重排
  -> 取前 final_top_k 个 seed chunks
  -> 邻居扩展
  -> ContextBuilder 组装证据上下文
  -> QwenChatGenerator 评估证据并生成答案
  -> ConfidenceScorer 打分
  -> 低置信度或实验类问题时触发外部文献工具
  -> 返回 answer / citations / confidence / debug
```

## 离线建库流程

```bash
python scripts/ingestion/build_round1_kb.py
```

执行链路：

```text
pdf_to_structured.py
  -> preprocess_and_chunk.py
  -> import_to_milvus.py
```

## 常用开发命令

| 命令 | 说明 |
|------|------|
| `python scripts/evaluation/evaluate_retrieval.py` | 检索评测 |
| `python scripts/evaluation/run_ragas_regression.py` | RAGAS 回归 |
| `python scripts/evaluation/run_validation_suite.py --label round5` | 验证套件（smoke → full） |
| `python scripts/ops/interactive_rag_cli.py` | 交互式调试 CLI |

更多命令见 [docs/startup.md](docs/startup.md)。

## 关键配置参数

核心参数在 `src/synbio_rag/domain/config.py` 中定义，可通过 `.env` 覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MILVUS_URI` | `./runtime/vectorstores/milvus/papers.db` | Milvus 连接地址 |
| `BGE_M3_MODEL_PATH` | `./models/BAAI/bge-m3` | Embedding 模型路径 |
| `BGE_RERANKER_MODEL_PATH` | `./models/BAAI/bge-reranker-v2-m3` | Reranker 模型路径 |
| `RETRIEVAL_SEARCH_LIMIT` | `30` | 检索召回规模 |
| `RETRIEVAL_RERANK_TOP_K` | `12` | Rerank 后保留数 |
| `RETRIEVAL_FINAL_TOP_K` | `10` | 邻居扩展前 seed chunk 数 |
| `RETRIEVAL_NEIGHBOR_EXPANSION_ENABLED` | `true` | 是否启用邻居扩展 |

完整参数列表见源码 `src/synbio_rag/domain/config.py`。

## 文档索引

- [启动说明](docs/startup.md)
- [项目目录规范](docs/project_structure.md)
- [RAG 主流程详解](docs/README.md)（如有）

## License

MIT
