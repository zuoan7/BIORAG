# SynBio RAG 项目目录规范

面向企业级 RAG 项目，当前仓库建议按“源码、数据、测试、评测结果、运行时产物”分层管理。

## 当前目录职责

```text
bge-service/
├── app/                         # FastAPI 入口
├── config/                      # 环境变量模板和配置示例
├── data/
│   ├── eval/
│   │   ├── datasets/            # 评测集、样本、模板（含 smoke20 / full100）
│   │   └── *.md                 # 评测集使用说明
│   ├── paper_round1/            # 论文原始数据、解析结果、chunk
│   └── patent_round1/           # 专利原始数据
├── docs/                        # 启动、回归、结构说明
├── models/                      # 本地模型资产
├── reports/
│   └── evaluation/
│       ├── ragas/               # 单次 RAGAS 评测记录
│       └── regression/          # baseline / candidate / compare / diagnostic 归档
├── runtime/
│   ├── logs/                    # 审计日志、会话存储
│   └── vectorstores/            # 本地向量库运行产物
├── scripts/
│   ├── data_prep/               # 数据准备与文件整理
│   ├── evaluation/              # 离线评测、回归脚本
│   ├── ingestion/               # 建库与入库脚本
│   └── ops/                     # 运维和调试工具
├── src/synbio_rag/              # 业务源码
└── tests/
    ├── fixtures/                # 测试样本与夹具
    ├── integration/             # Milvus / 检索链路集成测试
    └── smoke/                   # 模型与依赖烟雾测试
```

## 归档原则

- 测试代码统一放在 `tests/`，不要再散落到 `scripts/`。
- 评测输入统一放在 `data/eval/datasets/`。
- 日常验证建议固定保留两套集: `enterprise_ragas_smoke20.json` 用于快速回归，`enterprise_ragas_eval_v1.json` 用于全量放行。
- 评测输出统一放在 `reports/evaluation/`，便于回归比较和审计留存。
- `reports/evaluation/ragas/runs/` 用于保存单次评测结果，默认按时间戳生成新文件。
- `reports/evaluation/regression/` 按 `baselines/`、`candidates/`、`compares/`、`diagnostics/`、`targeted/`、`sweeps/` 分层归档；同类结果下可继续按 `round3`、`systemic_fix`、`default` 等标签分目录。
- 运行时日志、会话状态、Milvus Lite 文件统一放在 `runtime/`，并默认不纳入版本管理。
- `scripts/` 只保留可执行工作流，不放测试断言代码。

## 后续建议

- 若继续扩展，建议补 `Makefile` 或 `justfile` 统一常用命令。
- 若进入多人协作，建议新增 `docs/adr/` 存放架构决策记录。
- 若需要更完整交付，建议追加 `deploy/` 或 `infra/` 保存 Docker / Compose / CI 配置。
