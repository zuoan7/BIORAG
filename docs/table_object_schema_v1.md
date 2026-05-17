# table_object_v1 结构说明

## 1. 主格式

`table_object_v1` 的主格式是 JSONL：

`data/experiments/v7_phase7_table_extraction_mvp/table_objects.jsonl`

每一行保存一个离线抽取的 `table_object`。它是本轮 Phase7A 的结构化 source of truth，包含 table、caption、rows、columns、cells、footnotes、references、source_spans、warnings 和 validation_status。

## 2. Markdown 只是审阅视图

`table_objects_review.md` 只是从 JSONL 派生出来的人工审阅视图，用于快速查看 caption、block id、warnings、source_span 限制和 Markdown table preview。

Markdown 不是中间主数据结构，也不应作为后续结构化处理的权威输入。

## 3. JSON 不直接用于 embedding

完整 `table_object` JSON 不应被直接拼接后 embedding。原因是 JSON 字段、id、warnings 和内部结构会把机器格式噪声混入向量文本，且不一定对应用户检索时需要的事实粒度。

后续如果进入索引设计，应从 JSON 派生自然语言化的 indexable units，而不是直接 embedding JSON。

## 4. 后续应派生 index units

推荐从 `table_object` 派生以下视图：

- `table_summary`：表级摘要，包含 doc_id、table_id、caption、主要列和整体限制。
- `row_fact`：行级事实，表达某一行的 row label、列值和 source_span。
- `cell_fact`：单元格事实，表达 row、column、value_raw、unit、literal、footnote/reference 限制。
- `caption_context`：caption 和邻近上下文，用于支持 caption/context 级问题。

Phase7A 可生成 `table_index_units.preview.jsonl` 作为设计预览，但不写入 Milvus，不建 BM25，不跑 retrieval。

## 5. source_span 粒度限制

本轮输入是 official chunks。当前 chunks 可提供 chunk、block、page、block preview 等信息，但不能提供 value-level bbox。

`source_span_granularity` 可取：

- `table_level`
- `table_row_level`
- `row_level`
- `cell_level`
- `value_level`
- `mixed_or_unclear`

Phase7A 不伪造 value-level bbox。若只能定位到 table row、row 或 block，应如实记录 `source_span_table_row_level_only`、`source_span_not_value_level`、`no_value_level_bbox` 等 warning。

## 6. warnings 与 validation_status

`warnings` 用于保留抽取和绑定风险，不能吞掉。典型 warning 包括：

- parser/boundary 类：`no_table_text_flag`、`caption_body_split`、`body_as_paragraph`、`body_as_table_caption`、`parser_boundary_warning`。
- binding 类：`numeric_column_order_uncertain`、`unit_scope_uncertain`、`footnote_binding_uncertain`、`reference_binding_uncertain`。
- provenance 类：`source_span_table_row_level_only`、`source_span_not_value_level`、`no_value_level_bbox`。
- complex table 类：`matrix_binding_uncertain`、`metric_level_cell_gap_risk`、`mixed_table_block_risk`、`target_mapping_risk`。

`validation_status` 只表示本轮离线结构化对象是否足够进入人工审阅：

- `pass`：核心字段完整，且没有 warning。
- `pass_with_warnings`：核心字段完整，但存在非阻断 warning。
- `partial`：表体、行列单元格、caption 或 boundary/binding 存在明显缺口，需要人工复核。
- `fail`：缺少 source_span、chunk/source block 或最基本的对象字段，无法审阅。

本轮的 `pass_with_warnings` 不等于 production confirmed，也不等于 gold confirmed。

## 7. 非 production 声明

`table_object_v1` 是 Phase7A 的离线 MVP schema。它不修改 ingestion 主链路，不接 RAG，不写 Milvus，不重建 BM25，不运行 retrieval，不调用 Qwen/RAGAS/OCR/VLM，不修改 official baseline，也不进入 Route C implementation。

Route C 仍只是 backlog。若未来要进入 production structured table extraction，需要单独设计 value-level provenance、bbox 或等价策略、extraction quality gate、index/retrieval integration、rollback 与监控。
