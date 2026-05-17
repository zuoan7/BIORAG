# table_index_unit_v1 设计说明

## 1. 设计定位

`table_index_unit_v1` 是 Phase7I 的离线 preview schema，用于把 Phase7H 已通过一致性验证的 15 条 formal seed 转成未来 RAG 可使用的最小表格索引单元。

本 schema 不代表 production index，不写入 Milvus，不构建 BM25，不运行 embedding、rerank 或 retrieval，也不修改 official baseline。

## 2. 为什么不直接 embedding JSON

完整 JSON 适合作为结构化交换格式，但不适合作为 embedding 文本。原因有三点：

- JSON key、路径、内部 id、布尔状态和 guardrail 字段会引入大量机器格式噪声。
- 用户检索时通常问的是某张表、某一行或某组指标，而不是完整 JSON 对象。
- provenance、guardrail 与语义内容应分层保存；直接 embedding JSON 容易让限制说明、路径和状态字段压过真正的表格事实。

因此 Phase7I 只从结构化输入派生自然语言或紧凑半结构化的 `content_text_for_embedding`，并把来源和限制放在 `provenance` / `guardrail` 中。

## 3. 三类 unit

Phase7I 只采用三类最小 unit：

- `table_unit`：一张表一个 unit，用于表级召回、caption/table_id/doc_id/page 查询和表格主题理解。
- `row_unit`：一行一个 unit，用于按 strain、condition、construct、taxon、treatment 等行级对象召回指标，是后续 table-aware RAG 最常用的证据粒度。
- `cell_group_unit`：一行中的若干关键值聚合成一个 unit，用于让未来检索能命中同一行内的多个指标组合。

## 4. 为什么不做 value-level cell unit

Phase7H 明确记录 `value_bboxes_available=false`，且 source span 是 cell-level，不是 value-level。Phase7I 不能把 cell bbox 写成 value bbox，也不能声称逐值绑定已完全确认。

独立 value-level cell unit 会给人一种“每个值已有稳定 provenance 和 binding”的错觉。本轮只生成 row 级和 cell group 级 preview，并在每条 unit 中继承 warning-level limitation。

## 5. 为什么不单独拆 table_context_unit

caption、note、abbreviation、footnote 目前先保留在 `table_unit` 或 metadata/guardrail/provenance 中。单独拆 `table_context_unit` 会扩大 unit 类型和验证面，且 Phase7I 的目标是先验证最小可索引单元，不是构造完整 production table context graph。

后续如果 Phase7J 发现 caption/context 召回确有独立价值，可以在新阶段单独设计 context unit。

## 6. content_text_for_embedding 写法

`content_text_for_embedding` 应当面向未来检索文本，而不是面向程序解析。推荐写法：

- 包含 `doc_id`、`table_id`、caption 或行级上下文。
- 对 row/cell group 使用自然语言或半结构化片段表达：`column=value`。
- 简要保留 warning limitation，例如 binding 仍是 warning-level，value-level coordinates 不可用。
- 避免堆入文件路径、JSON key、内部 bbox 细节或过长原始表格全文。

## 7. content_markdown 用途

`content_markdown` 是人工审阅视图，用于快速检查某条 unit 的内容是否可理解、是否保留了 row label、column header、key values 和 limitation。

它不是权威结构化数据，也不是 production prompt 模板。权威结构仍是 JSONL 中的字段和子对象。

## 8. metadata / provenance / guardrail 的边界

`metadata` 保存表格语义信息，例如 page、row_index、row_label、column_headers、header_path、row_values、cell_group_values、table_shape 和 header_structure_type。

`provenance` 保存来源和可追溯限制，例如 source CSV、Markdown card、PDF crop、source_span_granularity、value_bboxes_available 和 cell_bboxes_available。`value_bboxes_available` 必须继承为 `false`。

`guardrail` 保存状态与阶段限制，例如 seed_status、binding_review_mode、binding_review_limitation、unit_or_note_ok、reference_ok、seed_warnings、index_unit_status、is_official_benchmark_seed 和 production_ready。

## 9. 多层表头处理

多行或 spanning header 使用 `header_path` 表达。例如：

```json
["Abundance, % (mean ± SD)", "Control"]
```

如果 header 结构无法稳定判断，则使用 `header_structure_type=uncertain_header`，并保留可恢复的 column header 文本。Phase7I 不回修 extractor，也不重建 CSV。

## 10. 为什么 Phase7I 不是 production index

Phase7I 只是 table index unit design and preview。它不读取或查询 BM25 index，不访问或写入 Milvus，不运行 embedding/rerank/retrieval，不调用 Qwen/RAGAS/OCR/VLM，不修改 ingestion pipeline，不修改 official dataset、official baseline、chunks、configs 或 baseline registry。

所有 unit 的 `guardrail.index_unit_status` 固定为 `preview_only`，`guardrail.production_ready=false`，`guardrail.is_official_benchmark_seed=false`。
