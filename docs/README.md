# BIORAG 文档索引

本文档只是文档索引。当前项目状态、accepted baseline、生产默认配置和
Phase7 preview 边界以仓库根目录的 [README.md](../README.md) 为准。

## 当前入口

- [项目 README](../README.md)：当前权威项目状态入口。
- [启动说明](startup.md)：本地服务、reranker 服务、知识库构建和 CLI。
- [当前仓库 inventory](architecture/current_inventory.md)：production、
  preview、legacy candidate 和 artifact/fixture 边界。
- [cleanup policy](architecture/cleanup_policy.md)：后续清理删除和归档规则。
- [scripts inventory](architecture/script_inventory.md)：`scripts/` 当前分类清单。
- [scripts keep allowlist](architecture/scripts_keep_allowlist.md)：Cleanup PR2
  脚本保留白名单。
- [scripts archive candidates](architecture/scripts_archive_candidates.md)：
  Cleanup PR2 脚本归档候选矩阵。
- [scripts cleanup report](architecture/scripts_cleanup_report.md)：Cleanup
  PR2-PR5 quarantine/delete-after-quarantine 结果和验证记录。
- [tracked artifact inventory](architecture/results_reports_artifact_inventory.md)：
  tracked reports/results/data artifact 保留矩阵。
- [local artifact inventory](architecture/pr7_local_artifact_inventory.md)：
  ignored/untracked local artifact 分类和保留建议。

## Phase7 表格证据边界

Phase7 table evidence / citation 仍是 preview / prototype / offline artifact。
Table units 不能视为 production-ready；`value_bboxes_available=false` 不能被
解释成 value-level bbox 已存在；CSV/PDF crop/markdown card 路径只能作为 debug
provenance。清理工作不得把 Phase7 table citation 接入 production `/v1/ask`。

## 当前保留文档

- [BIORAG next stage notes](BIORAG_NEXT_STAGE.md)：历史规划和后续讨论材料。
- [table_object_v1 schema](table_object_schema_v1.md)：Phase7 table object
  preview schema。
- [table_index_unit_v1](table_index_unit_v1.md)：Phase7 table index unit preview
  schema。

## Archived / Not Yet Restored

旧版 `docs/README.md` 曾引用 `docs/evaluation/current_baseline.md`、
`docs/generation_v2_design.md`、`docs/project_structure.md`、`docs/ragas_evaluation.md`
等路径。当前仓库没有这些文件，本轮不重建内容，也不把它们作为当前状态依据。
