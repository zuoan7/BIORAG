# SynBio RAG 项目启动命令说明

本文档对应当前仓库 `/home/gmy/bio-agent-workspace/bge-service`。

## 1. 环境准备

推荐先进入项目根目录再执行命令：

```bash
cd /home/gmy/bio-agent-workspace/bge-service
```

安装依赖：

```bash
pip install -r requirements.txt
```

初始化环境变量文件：

```bash
cp config/settings.example.env .env
```

## 2. 主 RAG 服务启动

启动命令：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

健康检查：

```bash
curl http://127.0.0.1:9000/healthz
```

问答接口：

```bash
curl -X POST http://127.0.0.1:9000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "总结 CRISPR 在细菌基因调控中的应用",
    "session_id": "demo-session-1",
    "tenant_id": "acme-bio",
    "include_debug": true
  }'
```

## 3. 独立 reranker 服务启动

如果希望把 rerank 从主服务中拆出去，先启动 reranker：

```bash
uvicorn app.reranker_main:app --host 0.0.0.0 --port 9001
```

然后在 `.env` 中配置：

```bash
RERANKER_SERVICE_URL=http://127.0.0.1:9001
```

健康检查：

```bash
curl http://127.0.0.1:9001/healthz
```

## 4. 知识库构建命令

首次使用前需要准备知识库：

```bash
python scripts/ingestion/build_round1_kb.py
```

如果你只想做单项处理，也可以使用这些脚本：

```bash
python scripts/ingestion/pdf_to_structured.py
python scripts/ingestion/preprocess_and_chunk.py
python scripts/ingestion/import_to_milvus.py
```

## 5. 常用开发命令

检索评测：

```bash
python scripts/evaluation/evaluate_retrieval.py
```

RAGAS 回归：

```bash
python scripts/evaluation/run_ragas_regression.py
```

日常代码改动先跑最小验证集，通过后自动升到全量：

```bash
python scripts/evaluation/run_validation_suite.py --label round5
```

只跑最小验证集：

```bash
python scripts/evaluation/run_validation_suite.py --stage smoke --label round5
```

只跑全量验证：

```bash
python scripts/evaluation/run_validation_suite.py --stage full --label round5
```

按标签归档本轮结果：

```bash
python scripts/evaluation/run_ragas_regression.py --label round4_retrieval
```

交互式命令行调试：

```bash
python scripts/ops/interactive_rag_cli.py
```

如果主服务不在 `9000`，可以显式指定地址：

```bash
python scripts/ops/interactive_rag_cli.py --base-url http://127.0.0.1:8002
```

如果希望默认打开调试信息：

```bash
python scripts/ops/interactive_rag_cli.py --debug
```

进入 CLI 后，直接输入问题就会调用 `/v1/ask`。常用命令：

```text
/health
/show
/history
/debug on
/reset
/quit
```

## 6. 当前启动注意事项

- 现在代码会按项目根目录解析本地路径，避免因为启动目录不同导致 `.env`、`runtime/logs/`、`data/`、`runtime/vectorstores/milvus/`、`models/` 路径错误。
- 离线评测数据默认放在 `data/eval/datasets/`，报告默认输出到 `reports/evaluation/`，并以时间戳新文件归档，不直接覆盖历史结果。
- `run_validation_suite.py` 默认使用 `enterprise_ragas_smoke20.json` 作为最小验证集，`enterprise_ragas_eval_v1.json` 作为全量验证集。
- 如果 `9000` 或 `9001` 端口已被占用，`uvicorn` 会直接报端口冲突，需要换端口或先停掉旧进程。
- 如果知识库未构建完成，服务可以启动，但 `/v1/ask` 可能在检索阶段报错或返回空结果。
- 如果使用 QWEN/OpenAI-compatible 接口，需要在 `.env` 里配置对应的 `API_BASE` 和 `API_KEY`。
