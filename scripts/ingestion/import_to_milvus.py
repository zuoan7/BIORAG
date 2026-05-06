#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus 导入脚本 —— 将 preprocess_and_chunk.py 输出的 JSONL 导入 Milvus。

用法:
    # 使用 mock embedding（默认，用于测试流程）
    python import_to_milvus.py --jsonl ./chunks/chunks.jsonl

    # 使用 BGE-M3 真实 embedding
    python import_to_milvus.py --jsonl ./chunks/chunks.jsonl --embedding bge-m3 --model_path /path/to/bge-m3

    # 指定 Milvus 服务地址
    python import_to_milvus.py --jsonl ./chunks/chunks.jsonl --milvus_uri http://localhost:19530

    # 使用本地 Milvus Lite（默认）
    python import_to_milvus.py --jsonl ./chunks/chunks.jsonl --milvus_uri ./runtime/vectorstores/milvus/papers.db

关于 embedding 维度与 metric_type 的说明:
    - BGE-M3 的 dense_vecs 维度为 1024
    - metric_type 统一使用 COSINE，因为:
      1. 文本语义检索场景下，余弦相似度比 L2 更稳定
      2. 统一 metric_type 可避免后续混合检索时的兼容问题
    - 首版建议使用稳定 chunk_id 作为主键，原因:
      1. 重复导入时自动去重，不会产生重复数据
      2. 支持按 doc_id 删除后重插，chunk_id 可复现
      3. 方便调试和溯源
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from pymilvus import (
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema,
)

DEFAULT_BGE_EMBED_MAX_LENGTH = 4096

# Phase 12A: Milvus VARCHAR 字段存储上限（字符数，非 token 数）
# 注意：这些值必须与 build_collection_schema 中的 FieldSchema max_length 保持一致
TEXT_VARCHAR_MAX_LENGTH = 16384
RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH = 16384
METADATA_JSON_VARCHAR_MAX_LENGTH = 8192

# ============================================================
# Embedding 接口
# ============================================================

class BaseEmbedder:
    """Embedding 基类，所有实现继承此类。"""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class MockEmbedder(BaseEmbedder):
    """
    Mock embedding，返回随机向量，仅用于测试流程。
    不保证语义质量，生产环境请替换为真实模型。
    """

    def __init__(self, dim: int = 1024):
        super().__init__(dim)

    def encode(self, texts: list[str]) -> list[list[float]]:
        import random
        return [
            self._normalize([random.gauss(0, 1) for _ in range(self.dim)])
            for _ in texts
        ]

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0:
            return vec
        return [x / norm for x in vec]


class BGEEmbedder(BaseEmbedder):
    """
    BGE-M3 真实 embedding。
    依赖: pip install FlagEmbedding
    """

    def __init__(
        self,
        model_path: str,
        dim: int = 1024,
        use_fp16: bool = True,
        max_length: int = DEFAULT_BGE_EMBED_MAX_LENGTH,
    ):
        super().__init__(dim)
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        self.max_length = max_length
        print(f"  BGE-M3 embed max_length={self.max_length}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        # 使用真实 BGE tokenizer 检查是否会超过 max_length。
        # 注意：这里的 i 是当前 encode batch 内的 index，不是全局 chunk index。
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            token_lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
            over_limit = [
                (i, n) for i, n in enumerate(token_lengths)
                if n > self.max_length
            ]

            if over_limit:
                examples = ", ".join(
                    f"#{i}:{n}" for i, n in over_limit[:10]
                )
                logger.warning(
                    "本批次 %d/%d 条文本实际 tokenizer tokens > max_length=%d，"
                    "超出部分将被模型截断。示例: %s",
                    len(over_limit),
                    len(texts),
                    self.max_length,
                    examples,
                )

        output = self.model.encode(texts, batch_size=32, max_length=self.max_length)
        dense = output["dense_vecs"]
        return [vec.tolist() for vec in dense]


def create_embedder(
    embedding_type: str,
    model_path: Optional[str] = None,
    dim: int = 1024,
    embed_max_length: int = DEFAULT_BGE_EMBED_MAX_LENGTH,
) -> BaseEmbedder:
    if embedding_type == "mock":
        return MockEmbedder(dim=dim)
    elif embedding_type == "bge-m3":
        if not model_path:
            raise ValueError("使用 bge-m3 embedding 必须指定 --model_path")
        return BGEEmbedder(model_path=model_path, dim=dim, max_length=embed_max_length)
    else:
        raise ValueError(f"不支持的 embedding 类型: {embedding_type}")


# ============================================================
# Collection Schema
# ============================================================

def build_collection_schema(dim: int) -> CollectionSchema:
    """
    构建 Milvus collection schema (Phase 12A v2)。

    原有字段:
    - chunk_id (VARCHAR, PK): 稳定可复现的主键
    - doc_id (VARCHAR): 文档 ID
    - source_file (VARCHAR): 源文件名
    - title (VARCHAR): 论文标题
    - section (VARCHAR): 所属 section
    - page_start (INT64): 起始页码
    - page_end (INT64): 结束页码
    - chunk_index (INT64): chunk 序号
    - text (VARCHAR): 清洗后的 chunk 文本（给生成/引用/citation 使用）
    - embedding (FLOAT_VECTOR): 向量

    Phase 12A 新增结构化索引字段:
    - retrieval_text (VARCHAR): embedding/BM25/rerank 使用的文本
    - content_kind (VARCHAR): body/table_text/table_caption/figure_caption/image_related/references/metadata
    - quality_score (FLOAT): chunk 质量分
    - contains_table_text (BOOL)
    - contains_table_caption (BOOL)
    - contains_figure_caption (BOOL)
    - contains_image (BOOL)
    - object_type (VARCHAR): body/table/figure/image/references/metadata
    - object_id (VARCHAR): 稳定 object 标识符
    - metadata_json (VARCHAR): 复杂结构字段的 JSON 序列化
    """
    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
            description="稳定可复现的 chunk 主键",
        ),
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="文档 ID",
        ),
        FieldSchema(
            name="source_file",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="源 PDF 文件名",
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="论文标题",
        ),
        FieldSchema(
            name="section",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="所属 section",
        ),
        FieldSchema(
            name="page_start",
            dtype=DataType.INT64,
            description="起始页码，-1 表示未知",
        ),
        FieldSchema(
            name="page_end",
            dtype=DataType.INT64,
            description="结束页码，-1 表示未知",
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
            description="chunk 序号",
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=TEXT_VARCHAR_MAX_LENGTH,
            description="清洗后的 chunk 文本（生成/引用/citation 使用）",
        ),
        FieldSchema(
            name="retrieval_text",
            dtype=DataType.VARCHAR,
            max_length=RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH,
            description="embedding/BM25/rerank 使用的文本",
        ),
        FieldSchema(
            name="content_kind",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="body/table_text/table_caption/figure_caption/image_related/references/metadata",
        ),
        FieldSchema(
            name="quality_score",
            dtype=DataType.FLOAT,
            description="chunk 质量分",
        ),
        FieldSchema(
            name="contains_table_text",
            dtype=DataType.BOOL,
            description="是否包含表格正文",
        ),
        FieldSchema(
            name="contains_table_caption",
            dtype=DataType.BOOL,
            description="是否包含表格标题",
        ),
        FieldSchema(
            name="contains_figure_caption",
            dtype=DataType.BOOL,
            description="是否包含图片标题",
        ),
        FieldSchema(
            name="contains_image",
            dtype=DataType.BOOL,
            description="是否包含图片",
        ),
        FieldSchema(
            name="object_type",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="body/table/figure/image/references/metadata",
        ),
        FieldSchema(
            name="object_id",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="稳定 object 标识符",
        ),
        FieldSchema(
            name="metadata_json",
            dtype=DataType.VARCHAR,
            max_length=METADATA_JSON_VARCHAR_MAX_LENGTH,
            description="复杂结构字段的 JSON 序列化",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="文本向量",
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description="合成生物学论文 RAG chunk 索引 (Phase 12A v2)",
    )


# ============================================================
# Phase 12A: 结构化字段推导
# ============================================================

def derive_content_kind(chunk: dict) -> str:
    """根据 chunk 的 contains_* 布尔字段推导 content_kind。"""
    if chunk.get("contains_table_text"):
        return "table_text"
    if chunk.get("contains_table_caption"):
        return "table_caption"
    if chunk.get("contains_figure_caption"):
        return "figure_caption"
    if chunk.get("contains_image"):
        return "image_related"
    if chunk.get("contains_references"):
        return "references"
    if chunk.get("contains_metadata"):
        return "metadata"
    return "body"


def derive_object_type(content_kind: str) -> str:
    """根据 content_kind 推导 object_type。"""
    if content_kind in ("table_text", "table_caption"):
        return "table"
    if content_kind == "figure_caption":
        return "figure"
    if content_kind == "image_related":
        return "image"
    if content_kind == "references":
        return "references"
    if content_kind == "metadata":
        return "metadata"
    return "body"


def infer_object_id(chunk: dict, object_type: str) -> str:
    """保守推导 object_id，不做复杂解析。"""
    doc_id = chunk.get("doc_id", "")
    # 已有显式 ID 则直接使用
    for key in ("object_id", "table_id", "figure_id", "image_id"):
        val = chunk.get(key)
        if val:
            return str(val)
    # 使用 source_block_ids 或 block_ids 的第一个元素
    block_ids = chunk.get("source_block_ids") or chunk.get("block_ids") or []
    if block_ids and len(block_ids) > 0:
        return f"{doc_id}::{object_type}::{block_ids[0]}"
    return ""


def build_metadata_json(chunk: dict) -> str:
    """将复杂结构字段序列化为 metadata_json，缺失时使用默认值。"""
    import json as _json

    # 摘要 source_block_metadata，避免过大
    sbm = chunk.get("source_block_metadata") or []
    if isinstance(sbm, list):
        sbm_summary = [
            {k: v for k, v in (item if isinstance(item, dict) else {}).items()
             if k in ("block_id", "type", "page", "bbox", "text_length")}
            for item in sbm[:20]
        ]
    else:
        sbm_summary = []

    payload = {
        # 当前已有字段
        "section_path": chunk.get("section_path") or [],
        "block_types": chunk.get("block_types") or [],
        "source_block_ids": chunk.get("source_block_ids") or [],
        "block_ids": chunk.get("block_ids") or [],
        "evidence_types": chunk.get("evidence_types") or [],
        "page_numbers": chunk.get("page_numbers") or [],
        "layout_columns": chunk.get("layout_columns") or 1,
        "reading_order_span": chunk.get("reading_order_span") or [],
        "bbox_span": chunk.get("bbox_span") or [],
        "excluded_block_counts": chunk.get("excluded_block_counts") or {},
        "contains_references": chunk.get("contains_references") or False,
        "contains_metadata": chunk.get("contains_metadata") or False,
        "contains_noise": chunk.get("contains_noise") or False,
        "parser_stage": chunk.get("parser_stage") or "",
        "source_block_metadata": sbm_summary,
        # 图表增强预留字段（当前填默认空值）
        "table_id": chunk.get("table_id") or "",
        "figure_id": chunk.get("figure_id") or "",
        "image_id": chunk.get("image_id") or "",
        "object_id": chunk.get("object_id") or "",
        "object_type": chunk.get("object_type") or "",
        "table_caption": chunk.get("table_caption") or "",
        "table_markdown": chunk.get("table_markdown") or "",
        "table_csv_path": chunk.get("table_csv_path") or "",
        "table_json_path": chunk.get("table_json_path") or "",
        "figure_caption": chunk.get("figure_caption") or "",
        "image_path": chunk.get("image_path") or "",
        "image_hash": chunk.get("image_hash") or "",
        "ocr_text": chunk.get("ocr_text") or "",
        "visual_summary": chunk.get("visual_summary") or "",
        "asset_uri": chunk.get("asset_uri") or "",
    }
    return _json.dumps(payload, ensure_ascii=False)


# ============================================================
# 读取 JSONL
# ============================================================

def read_jsonl(filepath: Path) -> list[dict]:
    """读取 JSONL 文件，返回 dict 列表。"""
    records = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败: {e}")
    return records


# ============================================================
# 删除旧数据
# ============================================================

def delete_by_doc_ids(
    client: MilvusClient,
    collection_name: str,
    doc_ids: list[str],
) -> int:
    """
    按 doc_id 删除旧数据，返回删除的条数。
    如果 collection 不存在则跳过。
    """
    if collection_name not in client.list_collections():
        return 0

    total_deleted = 0
    for doc_id in doc_ids:
        client.delete(
            collection_name=collection_name,
            filter=f'doc_id == "{doc_id}"',
        )
        total_deleted += 1

    return total_deleted


# ============================================================
# 批量插入
# ============================================================

# 全局截断统计（在 upsert_chunks 中累积，main 结束时汇总输出）
_truncation_stats: dict[str, list[dict]] = {
    "text": [],
    "retrieval_text": [],
    "metadata_json": [],
}


def _safe_truncate(value: str, max_len: int, field_name: str, chunk_id: str) -> str:
    """安全截断字符串到指定长度，超长时记录统计并发出警告。"""
    if len(value) <= max_len:
        return value
    logger.warning(
        "%s 字段超过 VARCHAR(%d) 限制 (长度=%d)，将被截断: chunk_id=%s",
        field_name, max_len, len(value), chunk_id,
    )
    if field_name in _truncation_stats:
        _truncation_stats[field_name].append({
            "chunk_id": chunk_id,
            "original_chars": len(value),
            "max_allowed": max_len,
        })
    return value[:max_len]


def log_truncation_summary() -> None:
    """汇总输出截断统计。"""
    total = sum(len(v) for v in _truncation_stats.values())
    if total == 0:
        return
    print()
    print("=" * 60)
    print("VARCHAR 截断统计")
    print("=" * 60)
    for field_name, items in _truncation_stats.items():
        if not items:
            continue
        orig_lens = [item["original_chars"] for item in items]
        print(f"  {field_name}:")
        print(f"    截断数量:     {len(items)}")
        print(f"    原始长度范围: {min(orig_lens)} – {max(orig_lens)}")
        print(f"    原始长度平均: {sum(orig_lens) / len(orig_lens):.0f}")
    print()


def upsert_chunks(
    client: MilvusClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    batch_size: int = 500,
) -> int:
    """
    批量 upsert chunk + embedding 到 Milvus。
    使用 upsert 而非 insert，确保 chunk_id 主键重复时更新而非报错。
    返回成功 upsert 的条数。
    """
    inserted = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        data = []
        for chunk, emb in zip(batch_chunks, batch_embs):
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            retrieval_text_raw = chunk.get("retrieval_text") or chunk.get("text", "")
            text_raw = chunk.get("text", "")
            content_kind = derive_content_kind(chunk)
            object_type = derive_object_type(content_kind)
            object_id = infer_object_id(chunk, object_type)
            cid = chunk["chunk_id"]

            # 记录原始长度用于截断统计
            text_original_chars = len(text_raw)
            retrieval_text_original_chars = len(retrieval_text_raw)

            # 安全截断 VARCHAR 字段，防止超长导致 Milvus 插入失败
            retrieval_text = _safe_truncate(
                retrieval_text_raw, RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH, "retrieval_text", cid
            )
            text = _safe_truncate(text_raw, TEXT_VARCHAR_MAX_LENGTH, "text", cid)
            metadata_json = _safe_truncate(
                build_metadata_json(chunk), METADATA_JSON_VARCHAR_MAX_LENGTH, "metadata_json", cid
            )

            row = {
                "chunk_id": cid,
                "doc_id": chunk["doc_id"],
                "source_file": chunk.get("source_file", ""),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "page_start": page_start if page_start is not None else -1,
                "page_end": page_end if page_end is not None else -1,
                "chunk_index": chunk.get("chunk_index", 0),
                "text": text,
                "retrieval_text": retrieval_text,
                "content_kind": content_kind,
                "quality_score": chunk.get("quality_score") or 0.0,
                "contains_table_text": bool(chunk.get("contains_table_text")),
                "contains_table_caption": bool(chunk.get("contains_table_caption")),
                "contains_figure_caption": bool(chunk.get("contains_figure_caption")),
                "contains_image": bool(chunk.get("contains_image")),
                "object_type": object_type,
                "object_id": object_id,
                "metadata_json": metadata_json,
                "embedding": emb,
            }
            # 向 metadata_json 注入截断标记
            if (text_original_chars > TEXT_VARCHAR_MAX_LENGTH
                    or retrieval_text_original_chars > RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH):
                import json as _json2
                try:
                    mj = _json2.loads(metadata_json) if isinstance(metadata_json, str) else {}
                except Exception:
                    mj = {}
                mj["text_truncated"] = text_original_chars > TEXT_VARCHAR_MAX_LENGTH
                mj["retrieval_text_truncated"] = (
                    retrieval_text_original_chars > RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH
                )
                mj["text_original_chars"] = text_original_chars
                mj["retrieval_text_original_chars"] = retrieval_text_original_chars
                row["metadata_json"] = _safe_truncate(
                    _json2.dumps(mj, ensure_ascii=False),
                    METADATA_JSON_VARCHAR_MAX_LENGTH,
                    "metadata_json",
                    cid,
                )

            data.append(row)

        client.upsert(collection_name=collection_name, data=data)
        inserted += len(data)
        print(f"    已 upsert {inserted}/{len(chunks)}")

    return inserted


# ============================================================
# 创建索引
# ============================================================

def create_index(client: MilvusClient, collection_name: str) -> None:
    """
    为 collection 创建索引。
    使用 IVF_FLAT 索引 + COSINE metric，兼顾速度和精度。
    """
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )

    index_params.add_index(
        field_name="doc_id",
        index_type="",  # 标量字段默认索引
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )

    print(f"  索引创建完成")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="将 JSONL chunk 导入 Milvus"
    )
    parser.add_argument(
        "--jsonl", required=True,
        help="JSONL 文件路径（preprocess_and_chunk.py 的输出）",
    )
    parser.add_argument(
        "--collection_name", default="synbio_papers",
        help="Milvus collection 名称（默认: synbio_papers）",
    )
    parser.add_argument(
        "--milvus_uri", default="./runtime/vectorstores/milvus/papers.db",
        help="Milvus 连接地址（默认: 本地 Milvus Lite）",
    )
    parser.add_argument(
        "--embedding", choices=["mock", "bge-m3"], default="mock",
        help="embedding 方式（默认: mock，用于测试）",
    )
    parser.add_argument(
        "--model_path", default=None,
        help="BGE-M3 模型路径（--embedding bge-m3 时必填）",
    )
    parser.add_argument(
        "--dim", type=int, default=1024,
        help="embedding 维度（默认: 1024，BGE-M3 dense_vecs 维度）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=500,
        help="批量插入大小（默认: 500）",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="是否重建 collection（删除已有 collection 后重建）",
    )
    parser.add_argument(
        "--delete_doc_ids", nargs="*", default=None,
        help="导入前按 doc_id 删除旧数据（支持多个 doc_id）",
    )
    parser.add_argument(
        "--embed-max-length", type=int, default=None,
        help=f"BGE embedding 最大 token 长度（默认: {DEFAULT_BGE_EMBED_MAX_LENGTH}，可通过 BGE_EMBED_MAX_LENGTH 环境变量配置）",
    )

    args = parser.parse_args()

    # 解析 embed_max_length: CLI 参数 > 环境变量 > 默认值 DEFAULT_BGE_EMBED_MAX_LENGTH
    if args.embed_max_length is not None:
        embed_max_length = args.embed_max_length
    elif os.environ.get("BGE_EMBED_MAX_LENGTH"):
        embed_max_length = int(os.environ["BGE_EMBED_MAX_LENGTH"])
    else:
        embed_max_length = DEFAULT_BGE_EMBED_MAX_LENGTH

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.exists():
        print(f"[ERROR] JSONL 文件不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Milvus 导入")
    print("=" * 60)
    print(f"JSONL:            {jsonl_path}")
    print(f"Collection:       {args.collection_name}")
    print(f"Milvus URI:       {args.milvus_uri}")
    print(f"Embedding:        {args.embedding}")
    print(f"维度:             {args.dim}")
    print(f"Embed max_length: {embed_max_length}")
    print()

    # 1. 读取 JSONL
    print("[1/6] 读取 JSONL ...")
    chunks = read_jsonl(jsonl_path)
    print(f"  读取到 {len(chunks)} 个 chunk")

    if not chunks:
        print("[WARN] 没有数据可导入，退出")
        return

    # 2. 创建 embedder
    print("[2/6] 初始化 Embedder ...")
    embedder = create_embedder(
        embedding_type=args.embedding,
        model_path=args.model_path,
        dim=args.dim,
        embed_max_length=embed_max_length,
    )
    print(f"  Embedder: {type(embedder).__name__}, dim={embedder.dim}")

    # 3. 生成 embedding
    print("[3/6] 生成 Embedding ...")
    texts = [c.get("retrieval_text") or c.get("text", "") for c in chunks]
    start_time = time.time()

    embeddings = []
    encode_batch = 64
    for i in range(0, len(texts), encode_batch):
        batch = texts[i : i + encode_batch]
        batch_embs = embedder.encode(batch)
        embeddings.extend(batch_embs)
        done = min(i + encode_batch, len(texts))
        print(f"  编码进度: {done}/{len(texts)}")

    elapsed = time.time() - start_time
    print(f"  编码完成，耗时 {elapsed:.2f}s")

    # 4. 连接 Milvus & 创建 collection
    print("[4/6] 连接 Milvus & 创建 Collection ...")
    client = MilvusClient(args.milvus_uri)

    if args.rebuild and args.collection_name in client.list_collections():
        client.drop_collection(args.collection_name)
        print(f"  已删除旧 collection: {args.collection_name}")

    if args.collection_name not in client.list_collections():
        schema = build_collection_schema(dim=args.dim)
        client.create_collection(
            collection_name=args.collection_name,
            schema=schema,
        )
        print(f"  已创建 collection: {args.collection_name}")
    else:
        print(f"  Collection 已存在: {args.collection_name}")

    # 5. 删除旧数据（按 doc_id）
    if args.delete_doc_ids:
        print("[5/6] 删除旧数据 ...")
        doc_ids_to_delete = args.delete_doc_ids
        deleted = delete_by_doc_ids(client, args.collection_name, doc_ids_to_delete)
        print(f"  已删除 {deleted} 个 doc_id 的旧数据")
    else:
        print("[5/6] 跳过旧数据删除")

    # 6. 批量 upsert
    print("[6/6] 批量 upsert ...")
    inserted = upsert_chunks(
        client=client,
        collection_name=args.collection_name,
        chunks=chunks,
        embeddings=embeddings,
        batch_size=args.batch_size,
    )
    print(f"  upsert 完成: {inserted} 条")
    log_truncation_summary()

    # 创建索引
    print()
    print("创建索引 ...")
    create_index(client, args.collection_name)

    # 刷新
    client.flush(args.collection_name)

    # 统计
    stats = client.get_collection_stats(args.collection_name)
    row_count = stats.get("row_count", "N/A")

    print()
    print("=" * 60)
    print("导入完成")
    print("=" * 60)
    print(f"Collection:  {args.collection_name}")
    print(f"总行数:      {row_count}")
    print(f"Embedding:   {args.embedding} (dim={args.dim})")


if __name__ == "__main__":
    main()
