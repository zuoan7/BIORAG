# parsed_clean Pipeline 文档

## 概述

本文档描述了改进后的文档结构化处理链路，从 PDF 到可检索的 chunks。

## 新链路

```
PDF
  → parsed_raw/        （原始抽取，不覆盖）
  → parsed_clean/      （清洗后的结构化 JSON，chunking 唯一输入）
  → parsed_preview/    （Markdown 预览，仅用于人工审计）
  → chunks/            （分块后的 JSONL）
  → Milvus / BM25
```

### 目录结构

```
data/paper_round1/
  paper/               # 原始 PDF
  parsed_raw/          # 原始抽取 JSON（pdf_to_structured.py 输出）
  parsed_clean/        # 清洗后 JSON（clean_parsed_structure.py 输出）
  parsed_preview/      # Markdown 预览（人工审计用）
  chunks/              # 分块 JSONL（preprocess_and_chunk.py 输出）
```

## 核心原则

1. **parsed_raw 是原始抽取结果，尽量保留，不覆盖。**
2. **parsed_clean 是后续 chunking 的唯一输入。**
3. **parsed_preview 是从 parsed_clean 自动导出的 Markdown，只用于人工审计，不参与 pipeline。**
4. **preprocess_and_chunk.py 只能读取 parsed_clean JSON，不要同时读取 TXT。**
5. 第一阶段不做图片 OCR。
6. 第一阶段不强行做复杂表格抽取，先标记 `table_caption` / `table_text`。
7. 所有修改兼容旧版 parsed JSON：如果 JSON 没有 `pages[].blocks`，就回退使用 `pages[].text`。

## 清洗脚本

### `scripts/ingestion/clean_parsed_structure.py`

```bash
python scripts/ingestion/clean_parsed_structure.py \
  --input_dir data/paper_round1/parsed_raw \
  --output_dir data/paper_round1/parsed_clean \
  --preview_dir data/paper_round1/parsed_preview
```

#### 清洗能力

1. **修复断词**：保守词典规则，覆盖 Bifidobacterium、Klebsiella、fermentation、primary 等常见断词
2. **降级误判标题**：16S rRNA、27F/1492R 引物、13CH 化合物等被还原为正文
3. **拆分 inline subsection**：将 "2.1. Materials Six commercially available..." 拆成 subsection_heading + paragraph
4. **分离 Figure/Table caption**：将 Fig. / Table 标题切为独立 block
5. **标记 References**：References 之后的 block type 标记为 references
6. **输出结构化 JSON + Markdown 预览**

#### parsed_clean JSON 格式

```json
{
  "doc_id": "doc_0001",
  "source_file": "doc_0001.pdf",
  "total_pages": 9,
  "parser_stage": "parsed_clean_v1",
  "pages": [
    {
      "page": 1,
      "text": "清洗后的页面级文本",
      "blocks": [
        {
          "block_id": "p1_b0001",
          "type": "title | abstract | section_heading | subsection_heading | paragraph | figure_caption | table_caption | table_text | references | noise",
          "text": "...",
          "section_path": ["3. Results and discussion", "3.2. Gas production"],
          "page": 1
        }
      ]
    }
  ]
}
```

#### Block 类型说明

| 类型 | 说明 |
|------|------|
| `title` | 论文标题 |
| `abstract` | 摘要 |
| `section_heading` | 大节标题（Introduction, Methods 等） |
| `subsection_heading` | 子节标题（2.1. Materials 等） |
| `paragraph` | 正文段落 |
| `figure_caption` | 图片标题 |
| `table_caption` | 表格标题 |
| `table_text` | 表格内容 |
| `references` | 参考文献 |
| `noise` | 噪声文本 |

## 审计脚本

### `scripts/audit/audit_parsed_clean_quality.py`

```bash
python scripts/audit/audit_parsed_clean_quality.py data/paper_round1/parsed_clean
```

输出每个文档的结构质量指标：
- `false_heading_count` — 误判标题数
- `subsection_heading_count` — 子节标题数
- `figure_caption_count` — 图片标题数
- `table_caption_count` — 表格标题数
- `references_block_count` — 参考文献块数
- `broken_word_count` — 残存断词数
- `empty_block_count` — 空 block 数
- `pages_count` / `block_count`

报告输出到 `reports/parsed_clean_quality.json` 和 `reports/parsed_clean_quality.txt`。

## 测试命令

### 测试 1：单文档测试

```bash
# 准备
mkdir -p tmp/parsed_raw_one tmp/parsed_clean_one tmp/parsed_preview_one tmp/chunks_raw_one tmp/chunks_clean_one
cp data/paper_round1/parsed/doc_0001.json tmp/parsed_raw_one/

# 清洗
python scripts/ingestion/clean_parsed_structure.py \
  --input_dir tmp/parsed_raw_one \
  --output_dir tmp/parsed_clean_one \
  --preview_dir tmp/parsed_preview_one

# 审计
python scripts/audit/audit_parsed_clean_quality.py tmp/parsed_clean_one

# 对比 raw 与 clean 的 chunk 结果
python scripts/ingestion/preprocess_and_chunk.py \
  --input_dir tmp/parsed_raw_one \
  --output_dir tmp/chunks_raw_one \
  --chunk_size 800 --chunk_overlap 120

python scripts/ingestion/preprocess_and_chunk.py \
  --input_dir tmp/parsed_clean_one \
  --output_dir tmp/chunks_clean_one \
  --chunk_size 800 --chunk_overlap 120

# 检查关键证据
grep -n "98–104 mM" tmp/chunks_clean_one/chunks.jsonl
grep -n "Klebsiella pneumoniae" tmp/chunks_clean_one/chunks.jsonl
grep -n "Fig. 3" tmp/chunks_clean_one/chunks.jsonl
grep -n "Table 1" tmp/chunks_clean_one/chunks.jsonl
```

**验收标准**：
- `false_heading_count` 应为 0 或显著下降
- `broken_word_count` 应显著下降
- `subsection_heading_count` > 0
- `figure_caption_count` > 0
- `table_caption_count` > 0
- 不应丢失关键证据（98–104 mM、Klebsiella pneumoniae、Fig. 3、Table 1）

### 测试 2：20 篇小批量测试

```bash
mkdir -p tmp/parsed_raw_20 tmp/parsed_clean_20 tmp/parsed_preview_20 tmp/chunks_clean_20

find data/paper_round1/parsed -maxdepth 1 -name "*.json" | head -n 20 | xargs -I{} cp {} tmp/parsed_raw_20/

python scripts/ingestion/clean_parsed_structure.py \
  --input_dir tmp/parsed_raw_20 \
  --output_dir tmp/parsed_clean_20 \
  --preview_dir tmp/parsed_preview_20

python scripts/audit/audit_parsed_clean_quality.py tmp/parsed_clean_20

python scripts/ingestion/preprocess_and_chunk.py \
  --input_dir tmp/parsed_clean_20 \
  --output_dir tmp/chunks_clean_20 \
  --chunk_size 800 --chunk_overlap 120
```

**验收标准**：
- 无 Python 异常
- `failed_docs.log` 为空或无新增
- `chunks.jsonl` 正常生成
- 随机抽 3 篇 `parsed_preview`，人工可读性明显好于 raw txt

### 测试 3：全量测试（先不入 Milvus）

```bash
# 如果 parsed_raw 不存在，先用 parsed 目录作为 parsed_raw
mkdir -p data/paper_round1/parsed_raw
cp data/paper_round1/parsed/*.json data/paper_round1/parsed_raw/

python scripts/ingestion/clean_parsed_structure.py \
  --input_dir data/paper_round1/parsed_raw \
  --output_dir data/paper_round1/parsed_clean \
  --preview_dir data/paper_round1/parsed_preview

python scripts/audit/audit_parsed_clean_quality.py data/paper_round1/parsed_clean \
  > reports/parsed_clean_quality.txt

python scripts/ingestion/preprocess_and_chunk.py \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir data/paper_round1/chunks \
  --chunk_size 800 --chunk_overlap 120
```

**验收标准**：
- `parsed_clean` 文件数 = `parsed_raw` 文件数
- `parsed_preview` 文件数 = `parsed_raw` 文件数
- `chunks.jsonl` 正常生成
- 不丢失关键证据

### 测试 4：关键证据 grep 检查

```bash
grep -n "Klebsiella pneumoniae" data/paper_round1/chunks/chunks.jsonl | head
grep -n "sialylated HMOs" data/paper_round1/chunks/chunks.jsonl | head
grep -n "lactate" data/paper_round1/chunks/chunks.jsonl | head
grep -n "acetate" data/paper_round1/chunks/chunks.jsonl | head
grep -n "Fig. 3" data/paper_round1/chunks/chunks.jsonl | head
grep -n "Table 1" data/paper_round1/chunks/chunks.jsonl | head
```

### 测试 5：Milvus rebuild

```bash
python scripts/ingestion/import_to_milvus.py \
  --jsonl ./data/paper_round1/chunks/chunks.jsonl \
  --collection_name synbio_papers \
  --milvus_uri ./runtime/vectorstores/milvus/papers.db \
  --embedding bge-m3 \
  --model_path ./models/BAAI/bge-m3 \
  --dim 1024 \
  --rebuild

# 验证行数匹配
python - <<'PY'
from pathlib import Path
from pymilvus import MilvusClient

jsonl = Path("./data/paper_round1/chunks/chunks.jsonl")
chunk_count = sum(1 for _ in jsonl.open("r", encoding="utf-8"))

client = MilvusClient("./runtime/vectorstores/milvus/papers.db")
stats = client.get_collection_stats("synbio_papers")
row_count = int(stats.get("row_count", 0))

print("chunks:", chunk_count)
print("milvus:", row_count)
print("match:", chunk_count == row_count)
PY
```

## 修改清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/ingestion/clean_parsed_structure.py` | 新增 | parsed_raw → parsed_clean + preview 清洗 |
| `scripts/ingestion/preprocess_and_chunk.py` | 修改 | 支持 blocks、保留 subsection 编号 |
| `src/synbio_rag/ingestion/kb_builder.py` | 修改 | 增加清洗步骤 |
| `src/synbio_rag/domain/config.py` | 修改 | 增加 parsed_raw_dir、parsed_preview_dir |
| `scripts/audit/audit_parsed_clean_quality.py` | 新增 | 质量审计脚本 |
| `docs/parsed_clean_pipeline.md` | 新增 | 本文档 |
