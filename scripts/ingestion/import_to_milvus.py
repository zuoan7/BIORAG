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
import sys
import time
from pathlib import Path
from typing import Optional

from pymilvus import (
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema,
)


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

    def __init__(self, model_path: str, dim: int = 1024, use_fp16: bool = True):
        super().__init__(dim)
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def encode(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(texts, batch_size=32, max_length=512)
        dense = output["dense_vecs"]
        return [vec.tolist() for vec in dense]


def create_embedder(
    embedding_type: str,
    model_path: Optional[str] = None,
    dim: int = 1024,
) -> BaseEmbedder:
    if embedding_type == "mock":
        return MockEmbedder(dim=dim)
    elif embedding_type == "bge-m3":
        if not model_path:
            raise ValueError("使用 bge-m3 embedding 必须指定 --model_path")
        return BGEEmbedder(model_path=model_path, dim=dim)
    else:
        raise ValueError(f"不支持的 embedding 类型: {embedding_type}")


# ============================================================
# Collection Schema
# ============================================================

def build_collection_schema(dim: int) -> CollectionSchema:
    """
    构建 Milvus collection schema。

    字段说明:
    - chunk_id (VARCHAR, PK): 稳定可复现的主键，格式 doc_0001_sec03_chunk02
    - doc_id (VARCHAR): 文档 ID，用于按文档删除/查询
    - source_file (VARCHAR): 源文件名
    - title (VARCHAR): 论文标题
    - section (VARCHAR): 所属 section
    - page_start (INT64): 起始页码（可为 null，用 -1 占位）
    - page_end (INT64): 结束页码（可为 null，用 -1 占位）
    - chunk_index (INT64): chunk 序号
    - text (VARCHAR): 清洗后的 chunk 文本
    - embedding (FLOAT_VECTOR): 向量，维度由 dim 参数决定
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
            max_length=8192,
            description="清洗后的 chunk 文本",
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
        description="合成生物学论文 RAG chunk 索引",
    )


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

            row = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_file": chunk.get("source_file", ""),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "page_start": page_start if page_start is not None else -1,
                "page_end": page_end if page_end is not None else -1,
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk.get("text", ""),
                "embedding": emb,
            }
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

    args = parser.parse_args()

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
