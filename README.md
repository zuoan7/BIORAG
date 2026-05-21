# BIORAG — 合成生物学领域 RAG 服务

BIORAG 是面向合成生物学论文知识库的企业级 RAG 服务。当前主链路围绕
FastAPI 问答接口、child chunk 级 Dense + BM25 混合检索、parent chunk
materialization、本地 BGE rerank、证据选择和 generation v2 证据约束生成构建，
目标是让回答可追溯到知识库 chunk、doc、section 和 citation。

## 当前能力

- **问答服务**：`app/main.py` 暴露 `/v1/ask`、`/v1/sessions/{session_id}` 和
  `/healthz`，应用层入口为 `RAGApplicationService`。
- **意图路由**：`QueryRouter` 识别 `factoid`、`summary`、`comparison`、
  `negative` 等查询意图，并决定检索与 rerank 规模。
- **混合检索**：`HybridRetriever` 组合 Milvus dense 检索与 BM25 检索；当
  `child_chunks.jsonl` 存在时，Milvus/BM25 使用 child chunk 做召回，再通过
  `ParentStore` materialize 回 parent chunk，citation 保持 parent chunk 级别。
- **parent-child 证据视图**：child 命中的 `matched_child_snippets` 会进入 rerank
  和 generation v2；rerank/generation 优先使用 `matched_child_evidence`，parent
  正文主要作为 citation 和局部上下文。
- **查询改写**：`QUERY_REWRITE_MODE` 默认关闭；可按需启用中文问题的英文镜像
  改写，并保留 original-CN fallback 作为跨语言召回补偿。
- **本地重排**：`LocalBGERerankerService` 负责 BGE rerank、comparison 子查询
  聚合、route/evidence/structure bonus、guarded rerank 和同文档正文覆盖策略。
- **上下文扩展**：`ParentContextExpander` 基于 `parent_index.jsonl` 扩展
  `chunk_window`、`caption_context`、`page`、`section_path`、`evidence_type_context`
  等 parent context；当前默认关闭，离线 sweep 可显式打开。
- **表格候选预览**：`table_preview` 可把离线 table index unit 作为候选合入检索
  池，但默认仍是 preview-only；`TABLE_PREVIEW_ALLOW_FORMAL_CITATION=false` 时不
  生成正式表格 citation。
- **generation v2**：`GenerationV2Service` 将 evidence ledger、support
  selection、answer planning、extractive/Qwen synthesis、citation binding 和
  validation 拆成独立阶段。
- **会话与审计**：`SessionStore` 保存多轮会话，`AuditLogger` 写入请求、过滤器
  和响应审计记录。
- **离线评测**：支持 retrieval/RAGAS 回归、enterprise gates、biorag rule metrics
  和多阶段 smoke 验证。

## 主链路

一次 `/v1/ask` 请求的核心执行顺序：

```text
FastAPI request
  -> RAGApplicationService
  -> SynBioRAGPipeline
  -> QueryRouter
  -> optional query rewrite
  -> HybridRetriever (child chunk Milvus dense + BM25 + RRF/post-process)
  -> ParentStore materialize child hits to parent chunks
  -> original-CN fallback
  -> table preview merge
  -> LocalBGERerankerService (matched_child_evidence focused rerank text)
  -> summary supplement
  -> optional ParentContextExpander + evidence selection
  -> GenerationV2Service (child-focused evidence, parent-level citations)
  -> response/citation/debug assembly
  -> session + audit persistence
```

## 模块划分

```text
app/
  main.py                         # FastAPI HTTP 入口

src/synbio_rag/
  application/
    rag_service.py                # 应用服务、session、audit 编排
    pipeline.py                   # RAG 主链路编排
    query_rewrite_adapter.py      # query rewrite 接入
    original_cn_fallback.py       # 中文原问题 fallback 检索补偿
    query_semantics.py            # query 语义拆分、alias、comparison 子查询
    retrieval_postprocess.py      # 检索后处理：source-floor、boost、diversity
    rerank_*.py                   # rerank query/features/selection/guarded policy
    parent_expansion*.py          # parent context 扩展、候选、计划和 debug
    neighbor_*.py                 # chunk 邻居索引与 generation v2 audit
    table_preview*.py             # preview-only table candidate sidecar
    evidence_selection_stage.py   # generation 前 evidence selection
    generation_v2/                # 证据约束生成链路
      evidence_ledger.py
      support_selector.py
      summary_support_selector.py
      support_retention.py
      limited_support.py
      answer_planner.py
      answer_builder.py
      qwen_synthesizer.py
      citation_binder.py
      validator.py

  domain/
    config.py                     # Settings 和 feature flags
    router.py                     # QueryRouter
    schemas.py                    # QueryAnalysis / RetrievedChunk / RAGResponse
    confidence.py                 # confidence scoring

  infrastructure/
    embedding/bge.py              # BGE-M3 embedding adapter
    reranker/local_bge.py         # BGE reranker adapter
    vectorstores/
      milvus.py                   # dense retrieval
      bm25.py                     # sparse retrieval
      fusion.py                   # RRF helpers
      hybrid.py                   # hybrid retrieval orchestration
    index/parent_store.py         # parent index loader 与 child->parent materialization
    clients/openai_compatible.py  # Qwen/OpenAI-compatible client
    persistence/                  # audit/session persistence

  ingestion/                      # 清洗、chunk、table enhancement 库代码
  rewrite/                        # query rewrite service
  evaluation/                     # failure taxonomy

scripts/
  ingestion/                      # 建库、清洗、parent-child index、Milvus 导入
  extraction/                     # 表格抽取和 table index unit 离线流程
  evaluation/                     # RAGAS、enterprise gate、phase smoke
  diagnostics/                    # 诊断脚本
  ops/                            # 运维/交互式工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp config/settings.example.env .env
# 按需编辑 .env，配置模型路径、Milvus 路径和 Qwen/OpenAI-compatible API。
```

常用本地资源默认路径：

- `MILVUS_URI=./runtime/vectorstores/milvus/papers.db`
- `BGE_M3_MODEL_PATH=./models/BAAI/bge-m3`
- `BGE_RERANKER_MODEL_PATH=./models/BAAI/bge-reranker-v2-m3`
- `parent_chunks.jsonl: ./data/paper_round1/chunks/parent_chunks.jsonl`
- `child_chunks.jsonl: ./data/paper_round1/chunks/child_chunks.jsonl`
- `RETRIEVAL_PARENT_INDEX_PATH=./data/paper_round1/chunks/parent_index.jsonl`

### 3. 构建知识库

```bash
python scripts/ingestion/build_round1_kb.py
```

该命令当前会生成 `chunks.jsonl`、`parent_chunks.jsonl`、`child_chunks.jsonl` 和
`parent_index.jsonl`，并把 `child_chunks.jsonl` 导入 Milvus。BM25 也会优先读取
`child_chunks.jsonl`，使 dense/BM25 召回保持同一 child chunk 粒度。

必要时可单独运行：

```bash
python scripts/ingestion/build_parent_child_index.py \
  --chunks data/paper_round1/chunks/chunks.jsonl \
  --parent-output data/paper_round1/chunks/parent_chunks.jsonl \
  --child-output data/paper_round1/chunks/child_chunks.jsonl \
  --parent-index-output data/paper_round1/chunks/parent_index.jsonl

python scripts/ingestion/import_to_milvus.py \
  --jsonl data/paper_round1/chunks/child_chunks.jsonl \
  --collection_name synbio_papers \
  --milvus_uri ./runtime/vectorstores/milvus/papers.db
```

### 4. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

Reranker 在主服务进程内加载本地 BGE-Reranker 模型，不需要单独启动 HTTP
reranker 服务。

### 5. 请求示例

```bash
curl -X POST http://127.0.0.1:9000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "比较 2'\''-FL 和 6'\''-SL 的工程化合成路径差异",
    "session_id": "demo-session-1",
    "tenant_id": "default",
    "include_debug": true
  }'
```

## 关键配置

核心配置在 `src/synbio_rag/domain/config.py`，可通过 `.env` 覆盖。

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `RETRIEVAL_HYBRID_ENABLED` | `true` | 启用 Dense + BM25 混合检索 |
| `RETRIEVAL_BM25_ENABLED` | `true` | 启用 BM25 sparse 检索 |
| `RETRIEVAL_SEARCH_LIMIT` | `40` | 每轮检索候选规模 |
| `RETRIEVAL_RERANK_TOP_K` | `10` | rerank 后保留候选数 |
| `RETRIEVAL_FINAL_TOP_K` | `8` | generation 前 seed chunk 数 |
| `RETRIEVAL_RERANK_MODE` | `plain` | rerank 模式，支持关闭或 guarded 策略 |
| `RETRIEVAL_SOURCE_FLOOR_ENABLED` | `true` | 保留 dense/BM25 单源 top-N 候选 |
| `RETRIEVAL_PARENT_EXPANSION_ENABLED` | `false` | 启用 parent context 扩展 |
| `RETRIEVAL_PARENT_EXPANSION_MAX_TOTAL` | `12` | parent expansion 打开后的 final context 上限 |
| `RETRIEVAL_PARENT_EXPANSION_PER_SEED_LIMIT` | `2` | 每个 seed 最多补充的 parent context 数 |
| `QUERY_REWRITE_MODE` | `off` | query rewrite 模式：`off`/`shadow`/`enabled` |
| `RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED` | `false` | 英文改写后追加中文原问题 fallback |
| `TABLE_PREVIEW_ENABLED` | `true` | 合入 preview-only table candidate |
| `TABLE_PREVIEW_ALLOW_FORMAL_CITATION` | `false` | 是否允许 table preview 进入正式 citation |
| `GENERATION_V2_PROFILE` | `stable` | generation v2 配置档位 |
| `GENERATION_V2_USE_QWEN_SYNTHESIS` | `false` | 候选功能：使用 Qwen 生成综合答案，默认关闭 |
| `GENERATION_V2_ENABLE_COMPARISON_COVERAGE` | `false` | 候选功能：comparison 覆盖分析增强，默认关闭 |
| `GENERATION_V2_ENABLE_NEIGHBOR_AUDIT` | `false` | 邻居证据 dry-run audit |

`GENERATION_V2_PROFILE` 支持：

| Profile | 行为 |
|---------|------|
| `stable` | 默认稳定档，extractive generation，关闭 comparison/neighbor 增强 |
| `qwen` | 开启 Qwen synthesis |
| `comparison` | 开启 Qwen synthesis 和 comparison coverage |
| `debug` | 开启 Qwen synthesis、comparison coverage 和 neighbor audit |

检查当前运行环境中两个候选开关的实际值：

```bash
python -W default -c "from src.synbio_rag.domain.config import Settings; s=Settings.from_env(); print(s.generation.v2_use_qwen_synthesis, s.generation.v2_enable_comparison_coverage)"
```

## 评测与验证

最小链路验证：

```bash
python scripts/evaluation/evaluate_ragas.py \
  --dataset data/eval/datasets/enterprise_ragas_smoke20.json \
  --base-url http://127.0.0.1:9000 \
  --include-debug \
  --skip-generation-metrics
```

日常 gate：

```bash
python scripts/evaluation/run_validation_suite.py --stage smoke --label local
```

完整 RAGAS 回归：

```bash
python scripts/evaluation/run_ragas_regression.py \
  --dataset data/eval/datasets/enterprise_ragas_smoke20.json \
  --label local
```

单元测试：

```bash
pytest -q
```

parent-child 参数 sweep：

```bash
python scripts/evaluation/run_parent_child_param_sweep.py --round recall
python scripts/evaluation/run_parent_child_param_sweep.py --round parent --base-profile <recall_winner>
python scripts/evaluation/run_parent_child_param_sweep.py --round generation --base-profile <recall_winner> --parent-profile <parent_winner>
```

最近一次 smoke20 链路验证结果：

- 数据集：`data/eval/datasets/enterprise_ragas_smoke20.json`
- 样本数：20
- `enterprise_gates.overall_status`: `pass`
- `doc_id_hit_rate`: `1.0`
- `source_file_hit_rate`: `1.0`
- `route_match_rate`: `0.8`
- `min_citation_pass_rate`: `1.0`
- 报告：`results/ad_hoc/enterprise_ragas_smoke20/20260518_231308/report.json`

该验证使用 `--skip-generation-metrics`，因此 faithfulness 和 answer relevancy 的
RAGAS 裁判指标为 `not_applicable`。需要质量回归时应运行不跳过 generation
metrics 的 RAGAS 流程。

## 当前状态说明

- accepted baseline 仍以 Phase 20 convergence baseline（Phase 20M）为当前记录。
- production 默认保持 query rewrite、original-CN fallback、Qwen synthesis 等实验能力
  关闭；离线实验通过 feature flag 或 profile 显式启用。
- 当前代码已支持 parent-child 检索：child chunk 用于召回，命中的 child 片段进入
  rerank 和 generation，最终 citation 仍绑定 parent chunk。
- `ParentContextExpander` 是 child->parent materialization 之后的额外上下文扩展；
  当前默认关闭，parent-child sweep 用它评估不同 context budget。
- table preview 是候选召回/诊断 sidecar，不等同于 production formal table citation。
- 历史 Phase7/Phase20 脚本和报告仍保留为回归与审计材料，当前 README 以正在运行的
  模块边界和主链路为准。

## License

MIT
