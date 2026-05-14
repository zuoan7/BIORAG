---
title: "BIORAG 改进方案"
project: "zuoan7/BIORAG"
domain: "Synthetic Biology RAG"
generated_at: "2026-05-12"
purpose: "供 Claude Code / Codex / ChatGPT 等 AI 编程助手读取，用于理解当前系统问题、改进方向和后续实施优先级。"
status: "design_plan"
---

# BIORAG 改进方案

## 0. 背景与目标

BIORAG 是一个面向合成生物学论文的 RAG 系统。当前系统已经具备较完整的工程链路，包括 PDF 预处理、chunk 构建、Milvus/BM25 混合检索、query rewrite、RRF 融合、rerank、parent expansion、evidence ledger、support pack、answer planning、extractive answer、可选 Qwen synthesis、citation binding 和 final validation。

本方案的目标不是推倒重写系统，而是在现有架构上逐步增强以下能力：

1. 提高论文 PDF 解析质量，尤其是双栏、表格和图片。
2. 优化索引粒度，避免 chunk 过大或过碎导致检索和生成效果不稳定。
3. 增强中文问题到英文论文库的跨语言检索能力。
4. 提高生成阶段的证据粒度和 citation 忠实度。
5. 建立系统化诊断与 A/B 测试机制，避免凭感觉改动。

核心判断：

```text
当前系统方向正确，但仍偏向“粗粒度文本 RAG”。
下一步应逐步升级为“结构化论文证据 RAG”。
```

---

## 1. 当前系统总览

### 1.1 当前主链路

```text
PDF 论文
  -> PDF 预处理与结构化解析
  -> 清洗噪声、恢复双栏阅读顺序
  -> chunk 构建
  -> Milvus dense index + BM25 sparse index
  -> parent_index sidecar
  -> 用户问题分类 QueryRouter
  -> 可选 query rewrite
  -> Dense + BM25 hybrid retrieval
  -> RRF 融合
  -> source-floor 补回单路 top-N chunk
  -> rerank
  -> seed_chunks
  -> parent expansion 得到 final_chunks
  -> EvidenceLedgerBuilder
  -> SupportPackSelector
  -> AnswerPlanner
  -> ExtractiveAnswerBuilder
  -> 可选 QwenSynthesizer
  -> CitationBinder
  -> FinalValidator
  -> 最终答案与引用
```

### 1.2 当前系统的总体评价

当前系统的优点：

```text
1. 不是裸向量检索 + 裸 LLM 生成。
2. 检索阶段有 hybrid、RRF、source-floor、rerank、parent expansion。
3. 生成阶段有 evidence ledger、support pack、answer plan、citation binding 和 final validation。
4. 对 summary、comparison、factoid 等不同问题类型已有差异化处理。
5. 整体设计偏保守，能降低幻觉风险。
```

当前系统的主要问题：

```text
1. PDF 解析仍然是启发式，复杂图表混排、表格行列结构、图片内容理解仍不充分。
2. chunk 粒度和 parent expansion 的最佳组合尚未通过 A/B 实验证明。
3. query rewrite、original CN fallback、alias expansion、source-floor 等机制的真实贡献需要统计。
4. rerank 阶段如果只用中文原问题对英文 chunk 打分，可能存在跨语言打分不足。
5. 生成阶段 evidence 粒度仍是 chunk 级，不是 claim/句子级。
6. citation binding 能保证引用合法，但不能完全证明每个事实都被引用语义支持。
```

---

# 2. 阶段一：PDF 预处理改进

## 2.1 当前状态

当前 PDF 预处理已经做了 layout-aware 解析，不是简单 `page.get_text()`。系统会保留文本 bbox、字体、页码、block、line、column 等信息，并尝试根据 bbox 判断左栏、右栏和跨栏区域，恢复双栏论文阅读顺序。

当前已经具备：

```text
1. 双栏页面判断。
2. 左右栏阅读顺序恢复。
3. 页眉、页脚、DOI、Journal Pre-proof、版权、参考文献等噪声清理。
4. figure caption / table caption / table text 标记。
5. image block bbox 和 metadata 保留。
```

但当前还不是完整的视觉文档理解系统。

## 2.2 主要问题

```text
1. 双栏恢复仍然依赖启发式规则。
2. 复杂图表混排页仍可能乱序。
3. 表格没有稳定恢复行列结构。
4. 图片没有 OCR，也没有视觉模型摘要。
5. figure/table caption 保留了，但图表本体内容没有真正结构化。
6. section 识别、页眉页脚清理错误会向后传播到索引、检索和生成阶段。
```

## 2.3 改进方向

### 2.3.1 建立 parsed 输出抽样审计机制

先不要重写 PDF parser，应先建立稳定的质量审计流程。

建议每轮解析后抽样 20 篇论文，检查：

```text
1. 双栏阅读顺序是否错乱。
2. Abstract / Introduction / Methods / Results / Discussion / Conclusion 是否识别正确。
3. figure caption 是否完整保留。
4. table caption 是否完整保留。
5. table text 是否乱序到无法理解。
6. header/footer/reference 噪声是否残留。
7. 是否误删正文。
```

建议输出：

```text
reports/parsed_quality/parsed_quality_report.json
reports/parsed_quality/parsed_quality_report.md
```

建议记录字段：

```json
{
  "doc_id": "doc_0001",
  "source_file": "paper.pdf",
  "two_column_order_ok": true,
  "section_detection_ok": true,
  "figure_caption_ok": true,
  "table_caption_ok": true,
  "table_text_readable": false,
  "noise_residual_level": "low",
  "major_issues": ["table_text_disordered"],
  "notes": "Table 2 content flattened and row-column relation lost."
}
```

### 2.3.2 表格对象化

当前表格主要以 `[TABLE CAPTION]` 和 `[TABLE TEXT]` 形式进入 chunk。后续应将表格单独建模为 table object。

目标结构：

```json
{
  "object_id": "doc001_table_2",
  "type": "table",
  "doc_id": "doc001",
  "page": 7,
  "bbox": [60, 200, 540, 420],
  "caption": "Table 2. Engineered strains and product titers",
  "markdown": "| Strain | Product | Titer |\n|---|---|---|\n| E. coli A | 2'-FL | 3.2 g/L |",
  "csv_path": "assets/doc001/table_2.csv",
  "json_path": "assets/doc001/table_2.json",
  "source_block_ids": ["b123", "b124"],
  "linked_chunk_ids": ["chunk_018", "chunk_019"]
}
```

优先级：

```text
P1：先保留 markdown/csv/json 表格结构。
P2：再做表格摘要 table_summary。
P3：最后做复杂跨页表格、多级表头等增强。
```

### 2.3.3 图片对象化

当前 image block 主要保留 bbox/metadata，不做 OCR 和视觉理解。后续应把图片变成 figure object。

目标结构：

```json
{
  "object_id": "doc001_fig_3",
  "type": "figure",
  "doc_id": "doc001",
  "page": 5,
  "bbox": [80, 120, 500, 360],
  "caption": "Fig. 3. Biosynthetic pathway of 2'-FL production",
  "image_path": "assets/doc001/fig_3.png",
  "ocr_text": "GDP-L-fucose, lactose, fucosyltransferase...",
  "visual_summary": "This figure shows the biosynthetic pathway of 2'-FL, including GDP-L-fucose precursor and fucosyltransferase.",
  "linked_chunk_ids": ["chunk_011", "chunk_012"]
}
```

建议分步实现：

```text
Step 1：按 bbox 裁剪 figure image，保存 image_path。
Step 2：将 figure caption 和 image_path 绑定。
Step 3：接 OCR，提取图中文字。
Step 4：接 vision model，生成 visual_summary。
```

## 2.4 阶段一优先级

```text
P0：建立 parsed 抽样审计。
P1：表格结构化为 markdown/csv/json。
P2：图片裁剪 + caption 绑定。
P3：图片 OCR / visual_summary。
```

---

# 3. 阶段二：索引构建改进

## 3.1 当前状态

当前系统采用 chunk-level 检索 + parent index 上下文扩展。

子块写入 Milvus，包含：

```text
chunk_id
text
retrieval_text
embedding
doc_id
source_file
title
section
page_start
page_end
chunk_index
content_kind
quality_score
contains_table_text
contains_table_caption
contains_figure_caption
contains_image
object_type
object_id
metadata_json
```

同时系统构建 `parent_index.jsonl`，包含多类 parent：

```text
doc
section
section_path
page
chunk_window
caption_context
evidence_type_context
```

注意：parent index 不是主要用于向量召回，而是在命中 child chunk 后补上下文。

## 3.2 当前问题

```text
1. 默认 chunk_size=800、chunk_overlap=120，可能偏大。
2. 如果改小 chunk，但 parent expansion 不增强，会导致上下文断裂。
3. parent expansion 当前偏保守，某些场景每个 seed 只补少量 chunk。
4. parent index 目前主要是 chunk-to-chunk/context relation，还没有充分纳入 table/figure object。
5. section/path/page 等 metadata 依赖预处理质量。
```

## 3.3 改进方向

### 3.3.1 做 chunk size A/B 测试

不要凭感觉修改 chunk 参数。建议做三组实验：

```text
A. 当前配置
chunk_size=800
chunk_overlap=120
parent_expansion_per_seed_limit=1~2

B. 中等配置
chunk_size=600
chunk_overlap=100
parent_expansion_per_seed_limit=2

C. 小块配置
chunk_size=450
chunk_overlap=80
parent_expansion_per_seed_limit=2~3
```

观察指标：

```text
doc_id_hit_rate
chunk_hit_rate
rerank_top3_hit_rate
citation_correctness
answer_correctness
support_pack_noise_rate
zero_citation_rate
```

推荐判断标准：

```text
如果 B 优于 A：说明 800 chunk 偏大。
如果 C 召回准但生成差：说明 chunk 过碎或 parent expansion 噪声过大。
如果 A 仍最稳：说明当前大 chunk 策略适合现有数据集。
```

### 3.3.2 建立 parent expansion 贡献统计

当前 parent index 存在，但需要证明它真的贡献了最终答案。

建议在 debug 中记录：

```json
{
  "parent_expansion": {
    "input_seed_chunk_ids": [],
    "added_chunk_ids": [],
    "added_parent_types": ["chunk_window", "caption_context", "section_path"],
    "added_count": 0,
    "final_chunk_ids": [],
    "used_in_support_pack_chunk_ids": [],
    "used_in_citation_chunk_ids": []
  }
}
```

重点关注：

```text
1. parent expansion 补回的 chunk 是否进入 support_pack。
2. parent expansion 补回的 chunk 是否最终被引用。
3. 哪类 parent_type 最有贡献。
4. 哪类 parent_type 只是增加噪声。
```

### 3.3.3 建立 table / figure object index

后续索引对象应分为：

```text
text_chunk
table_object
figure_object
```

每类对象分别构建 retrieval_text：

```text
text_chunk.retrieval_text = title + section + text

table_object.retrieval_text = caption + markdown + table_summary

figure_object.retrieval_text = caption + ocr_text + visual_summary
```

这样表格和图片也能被检索，但向量化的是它们的文字表达，而不是原始图片像素。

### 3.3.4 parent index 加入 object relation

目标关系：

```text
caption_chunk -> table_object
caption_chunk -> figure_object
table_object -> surrounding_text_chunks
figure_object -> surrounding_text_chunks
object -> page_parent
object -> section_parent
```

这样用户问图表时，系统能召回：

```text
图表对象 + caption + 相关正文 + 所在 section/page 上下文
```

## 3.4 阶段二优先级

```text
P0：chunk_size A/B 测试。
P1：parent expansion 贡献统计。
P2：table/figure object index。
P3：object relation parent index。
```

---

# 4. 阶段三：检索阶段改进

## 4.1 当前状态

当前检索链路：

```text
QueryRouter 问题分类
  -> query rewrite 可选 off / shadow / enabled
  -> Dense retrieval
  -> BM25 retrieval
  -> RRF 融合
  -> structure marker boost / title boost / alias expansion
  -> source-floor 补回 dense/BM25 单路 top-N chunk
  -> rerank
  -> seed_chunks
```

当前问题类型包括：

```text
factoid
summary
comparison
experiment
unknown
```

query rewrite 默认不是一定开启，需要看配置。开启后，中文问题可改写为英文 query 用于检索。

CJK 机制不是翻译，也不是过滤中文，而是：如果实际检索 query 中包含中文字符，则降低 BM25 权重，避免中文问题对英文论文 BM25 造成噪声。

## 4.2 当前问题

```text
1. 如果 retrieval 使用英文 rewrite，但 rerank 仍使用中文原问题，可能出现中文 query + 英文 chunk 的跨语言 rerank 风险。
2. CJK 降权只是防噪声机制，不是召回增强机制。
3. original CN fallback 的真实贡献需要统计。
4. source-floor 补回单路 top-N chunk，但补回后是否进入 rerank/support/citation 需要验证。
5. 领域 alias 扩展分散在硬编码和 YAML 词表中，需要统一管理。
6. comparison 问题依赖分支拆解，若某一分支召回不足，生成阶段容易 partial。
```

## 4.3 改进方向

### 4.3.1 双 query rerank

当前可能是：

```text
retrieval_query = rewritten English query
rerank_query = original Chinese question
```

建议新增实验模式：

```text
rerank_query = original Chinese question + rewritten English query
```

分数聚合方式：

```text
方案 1：score = max(score_cn, score_en)
方案 2：score = 0.4 * score_cn + 0.6 * score_en
方案 3：score = max(score_cn, score_en) + alpha * mean(score_cn, score_en)
```

预期效果：

```text
中文问题：保留用户原始意图。
英文 rewrite：更好对齐英文论文 chunk。
双 query 聚合：降低 rewrite 丢意图和跨语言 rerank 弱的问题。
```

### 4.3.2 original CN fallback 贡献统计

既然 original CN fallback 已修复，应统计它是否真的有用。

建议记录：

```text
original_cn_fallback.triggered
original_cn_fallback.reason
original_cn_fallback.fallback_added_count
original_cn_fallback.fallback_added_chunk_ids
original_cn_fallback.fallback_added_doc_ids
fallback chunks 是否进入 rerank top_k
fallback chunks 是否进入 support_pack
fallback chunks 是否最终被引用
```

判断：

```text
如果 fallback_added_count 经常为 0：它只是保险丝。
如果 fallback chunk 经常进入 citation：它很重要。
如果 fallback chunk 经常引入错误：需要收紧。
```

### 4.3.3 source-floor 贡献统计

source-floor 的设计是补回 dense 和 BM25 单路 top-N chunk，避免某一路强相关候选被 RRF 压掉。

建议统计：

```text
source_floor_added_count
source_floor_added_chunk_ids
source_floor_added_source: dense_floor / bm25_floor
是否进入 rerank top_k
是否进入 support_pack
是否被最终引用
```

### 4.3.4 统一领域 alias 管理

当前存在两类 alias：

```text
1. hybrid.py 中的硬编码 alias。
2. retrieval_aliases_v1.yaml 中的受控 alias。
```

建议：

```text
短期：打开 low-risk YAML alias 做实验。
中期：将硬编码 alias 逐步迁移到 YAML。
长期：建立 alias 命中 debug。
```

alias debug 示例：

```json
{
  "alias_expansion": {
    "enabled": true,
    "triggered_aliases": ["organism.ecoli", "molecular.heterologous_expression"],
    "added_terms": ["E. coli", "Escherichia coli", "heterologous expression"],
    "scope": "bm25_only"
  }
}
```

### 4.3.5 按 intent 定制检索 profile

建议不同问题使用不同策略：

```text
factoid:
  目标：精确命中关键事实。
  策略：较强 rerank，适度 parent expansion。

summary:
  目标：覆盖多文档、多 section。
  策略：扩大召回，偏好 Abstract/Conclusion/Results。

comparison:
  目标：每个比较分支都至少召回候选。
  策略：分支子查询、doc diversity、branch coverage。

table/figure:
  目标：召回结构化对象。
  策略：table/figure marker boost + object index。
```

## 4.4 阶段三优先级

```text
P0：双 query rerank A/B。
P1：original CN fallback 贡献统计。
P2：source-floor 贡献统计。
P3：low-risk alias expansion 实验。
P4：按 intent 定制 retrieval profile。
```

---

# 5. 阶段四：生成阶段改进

## 5.1 当前状态

当前 generation_v2 不是裸 LLM 生成，而是：

```text
EvidenceLedgerBuilder
  -> SupportPackSelector
  -> AnswerPlanner
  -> ExtractiveAnswerBuilder
  -> QwenSynthesizer optional
  -> CitationBinder
  -> FinalValidator
```

当前生成阶段的优点：

```text
1. 先生成抽取式草稿，即使 LLM 不可用也能回答。
2. Qwen synthesis 是可选的。
3. Qwen 输出必须使用给定 [E#]，不能创造引用。
4. 如果 Qwen 输出校验失败，会回退到抽取式答案。
5. CitationBinder 会将 [E#] 绑定到真实 chunk citation。
6. FinalValidator 会检查 citation 是否为空或非法。
```

## 5.2 当前问题

```text
1. evidence 粒度是 chunk 级，不是 claim/句子级。
2. chunk 引用合法不代表每个事实都被精确支持。
3. SupportPackSelector 依赖较多启发式规则，可能过度偏好 Results、数字、表格/图注标记。
4. limited_support_pack fallback 可能救回答案，也可能引入弱证据。
5. Qwen validation 严格，能防幻觉，但可能导致频繁 fallback，答案变生硬。
6. 表格和图片问题受上游结构化程度限制，生成阶段难以补救。
7. comparison 问题容易 partial，可能用户体验偏保守。
```

## 5.3 改进方向

### 5.3.1 从 chunk-level evidence 升级为 claim-level evidence

当前：

```text
一个 chunk = 一个 evidence candidate
```

建议：

```text
一个 chunk -> 多个 evidence claim
```

claim 示例：

```json
{
  "claim_id": "E1_C2",
  "chunk_id": "chunk_012",
  "doc_id": "doc001",
  "claim_text": "The engineered E. coli strain produced 3.2 g/L 2'-FL.",
  "claim_type": "result",
  "entities": ["E. coli", "2'-FL"],
  "numbers": ["3.2 g/L"],
  "section": "Results",
  "page_start": 6,
  "page_end": 6
}
```

claim_type 可以包括：

```text
definition
method
result
comparison
limitation
table_row
figure_caption
background
```

### 5.3.2 claim-citation 对齐检查

当前 CitationBinder 只能保证 citation 来自 support_pack。后续应检查答案中的事实是否被引用真正支持。

短期规则检查：

```text
1. 数字是否一致。
2. 单位是否一致。
3. 实体是否一致。
4. 表格行列对象是否一致。
```

中期模型检查：

```text
answer_claim + cited_evidence -> support / contradict / not enough info
```

### 5.3.3 limited_support_pack_used 专项评测

建议统计：

```text
limited_support_pack_used=true 的样本数量
这些样本是否最终正确
这些样本是否更容易 partial/refuse
这些样本 citation 是否可靠
这些样本是否来自弱相关候选
```

判断策略：

```text
如果救回率高，保留并优化。
如果错误率高，收紧 fallback。
```

### 5.3.4 table / figure 专用生成路径

当 support_pack 包含 table_object：

```text
使用 table caption + markdown/csv/json + surrounding chunks。
不要只基于 table text flat string。
```

当 support_pack 包含 figure_object：

```text
使用 figure caption + OCR + visual_summary + surrounding chunks。
不要只基于 figure caption。
```

### 5.3.5 优化 partial/refuse 表达

当前 partial/refuse 偏保守。建议细分 reason：

```text
partial_because_missing_branch
partial_because_single_doc
partial_because_indirect_evidence
partial_because_structured_data_missing
partial_because_low_support_count
partial_because_abstract_only
```

然后生成更自然的限制说明。

例如：

```text
当前证据可以回答 A 部分，但 B 分支没有直接证据，因此下面只给出基于已检索证据的有限比较。
```

而不是笼统地说：

```text
当前知识库证据不足。
```

## 5.4 阶段四优先级

```text
P0：统计 Qwen fallback / partial / refuse / limited_support_pack 的触发原因。
P1：claim-level evidence prototype。
P2：claim-citation 对齐检查。
P3：table/figure 专用生成路径。
P4：优化 partial/refuse 表达。
```

---

# 6. 总体路线图

## 6.1 第一阶段：诊断优先，不改大架构

目标：确认真实瓶颈在哪里。

任务：

```text
1. parsed 抽样审计。
2. parent expansion 贡献统计。
3. original CN fallback 贡献统计。
4. source-floor 贡献统计。
5. Qwen fallback / partial / refuse 触发统计。
6. limited_support_pack_used 样本专项检查。
```

输出：

```text
reports/diagnostics/parsed_quality_report.md
reports/diagnostics/retrieval_contribution_report.md
reports/diagnostics/generation_guardrail_report.md
```

## 6.2 第二阶段：检索 A/B 测试

目标：验证 chunk size、rewrite、rerank、alias 的影响。

实验：

```text
A. 当前配置 baseline。
B. chunk_size=600, overlap=100。
C. chunk_size=450, overlap=80。
D. rerank 使用英文 rewrite。
E. rerank 使用中文原问题 + 英文 rewrite 双 query。
F. low-risk alias expansion 开启。
```

核心指标：

```text
doc_id_hit_rate
chunk_hit_rate
rerank_top3_hit_rate
citation_correctness
final_answer_correctness
comparison_branch_coverage
support_pack_noise_rate
zero_citation_rate
```

## 6.3 第三阶段：结构化证据增强

目标：解决表格和图片问题。

任务：

```text
1. table object：caption + markdown + csv/json + bbox。
2. figure object：caption + image crop + OCR + visual_summary。
3. object index：table/figure 单独进入检索。
4. object relation：table/figure 与周围正文 chunk 关联。
```

## 6.4 第四阶段：生成可信度增强

目标：从“引用合法”升级为“引用真正支持结论”。

任务：

```text
1. chunk-level evidence -> claim-level evidence。
2. answer claim -> citation support 检查。
3. 数字/单位/实体一致性检查。
4. table evidence 专用回答。
5. comparison 分支覆盖更细化。
```

---

# 7. 推荐优先级清单

## 7.1 P0：必须先做

```text
1. parsed 抽样审计。
2. parent expansion 贡献统计。
3. source-floor 贡献统计。
4. original CN fallback 贡献统计。
5. Qwen fallback / partial / refuse / limited_support_pack 触发统计。
```

原因：

```text
当前系统已经有很多机制，但不清楚哪些机制真的产生贡献。
如果不先诊断，后续容易凭感觉改错方向。
```

## 7.2 P1：高价值实验

```text
1. chunk_size A/B：800 vs 600 vs 450。
2. dual-query rerank：中文原问题 + 英文 rewrite。
3. low-risk alias expansion 实验。
```

原因：

```text
这些改动成本低，但可能明显影响检索质量。
```

## 7.3 P2：结构化增强

```text
1. table object。
2. figure crop + caption binding。
3. object index。
4. object relation parent index。
```

原因：

```text
合成生物学论文中大量关键结果在表格和图片里。
当前系统对图表的处理仍然偏弱。
```

## 7.4 P3：生成可信度增强

```text
1. claim-level evidence。
2. claim-citation 对齐校验。
3. table/figure 专用生成路径。
4. partial/refuse 表达优化。
```

原因：

```text
当前 citation 是 chunk 级合法性验证，不是 claim 级忠实性验证。
要进一步提升科研问答可信度，需要更细的 evidence 单元。
```

---

# 8. 不建议立即做的事情

短期不建议：

```text
1. 不要一上来重写整个 PDF parser。
2. 不要一上来把 chunk_size 改得很小。
3. 不要一上来打开所有 alias expansion。
4. 不要放宽 generation validation 来追求更流畅答案。
5. 不要跳过诊断直接做大规模重构。
```

原因：

```text
当前系统问题未必来自单一模块。
如果没有失败样本和指标支撑，很容易引入新的噪声或回归。
```

---

# 9. 给 AI 编程助手的执行原则

如果让 Claude Code / Codex 修改本项目，应遵守：

```text
1. 优先加诊断和报告，不优先大改架构。
2. 每个功能必须 feature flag 控制。
3. 每个改动必须能通过小规模 eval 验证。
4. 不要破坏当前 baseline 行为。
5. 不要默认开启实验功能。
6. 改动前先读相关 debug 和测试。
7. 尽量最小实现，避免一次性重构多个阶段。
```

建议任务拆分：

```text
Task 1：增加 parent/source-floor/original-cn-fallback 贡献统计。
Task 2：增加 generation guardrail 触发统计报告。
Task 3：实现 dual-query rerank 实验开关。
Task 4：实现 chunk_size A/B 批处理脚本。
Task 5：实现 table object prototype。
Task 6：实现 claim-level evidence prototype。
```

---

# 10. 一句话总结

当前 BIORAG 已经有完整的 RAG 工程骨架，下一步不是推倒重来，而是沿着以下方向升级：

```text
预处理阶段：从 layout-aware 文本解析升级到 table/figure object 解析。
索引阶段：从 chunk index 升级到 text/table/figure object index + relation index。
检索阶段：从单 query hybrid 升级到跨语言 dual-query、alias-aware、intent-aware retrieval。
生成阶段：从 chunk-level citation 升级到 claim-level evidence 和 claim-citation 对齐校验。
```

最终目标：

```text
把当前“粗粒度文本 RAG”升级为“结构化论文证据 RAG”。
```
