#!/usr/bin/env python3
"""
局部重建指定文档的 chunks + Milvus vectors。

只删除/更新指定的 doc_ids，不影响其他文档。

用法:
    # dry-run 预览变更
    python scripts/ingestion/rebuild_docs_by_id.py --doc_ids doc_0008,doc_0078 --dry-run

    # 实际执行（仅重建 chunks.jsonl，不更新 Milvus）
    python scripts/ingestion/rebuild_docs_by_id.py --doc_ids doc_0008,doc_0078 --no-milvus

    # 完整重建（chunks + Milvus）
    python scripts/ingestion/rebuild_docs_by_id.py --doc_ids doc_0008,doc_0078
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings


def load_parsed_doc(doc_id: str) -> Optional[dict]:
    """从 parsed_clean 加载文档结构化数据。"""
    parsed_dir = ROOT / "data" / "paper_round1" / "parsed_clean"
    path = parsed_dir / f"{doc_id}.json"
    if not path.exists():
        print(f"  [WARN] parsed_clean not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["doc_id"] = doc_id
    data["source_file"] = data.get("source_file", f"{doc_id}.pdf")
    data["has_blocks"] = bool(data.get("pages"))
    data["raw_text"] = data.get("raw_text", "")
    return data


def chunk_doc(doc_data: dict) -> list[dict]:
    """对单个文档运行 chunk_by_blocks（包含 section fallback）。"""
    from scripts.ingestion.preprocess_and_chunk import process_document

    chunks, _ = process_document(doc_data)
    return [c.__dict__ if hasattr(c, '__dict__') else c for c in chunks]


def compute_section_summary(chunks: list[dict]) -> dict:
    """计算 section 分布摘要。"""
    dist = Counter(c.get("section", "?") for c in chunks)
    body_sections = {
        "Introduction", "Background", "Methods", "Materials and Methods",
        "Experimental Section", "Experimental Procedures", "Results",
        "Results and Discussion", "Discussion", "Conclusion", "Conclusions",
        "Full Text",
    }
    body_count = sum(n for s, n in dist.items() if s in body_sections)
    title_au_unknown = sum(
        n for s, n in dist.items() if s in ("Title", "Abstract", "Unknown", "")
    )
    return {
        "total": len(chunks),
        "section_dist": dict(dist),
        "body_count": body_count,
        "title_au_unknown": title_au_unknown,
        "has_body": body_count > 0,
    }


def rebuild_chunks_jsonl(
    doc_ids: list[str],
    chunks_dir: Path,
    dry_run: bool = False,
) -> dict:
    """更新 chunks.jsonl：删除旧 chunk，插入新 chunk。"""
    jsonl_path = chunks_dir / "chunks.jsonl"
    if not jsonl_path.exists():
        print(f"  [ERROR] chunks.jsonl not found: {jsonl_path}")
        return {}

    # 备份
    backup_path = chunks_dir / f"chunks.jsonl.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not dry_run:
        shutil.copy2(jsonl_path, backup_path)
        print(f"  Backup: {backup_path}")

    # 读取旧 chunks
    old_entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                old_entries.append(json.loads(line))

    target_set = set(doc_ids)
    kept = [e for e in old_entries if e.get("doc_id") not in target_set]
    removed = [e for e in old_entries if e.get("doc_id") in target_set]
    print(f"  Removed {len(removed)} old chunks (out of {len(old_entries)} total)")

    # 生成新 chunks
    new_entries = []
    for doc_id in doc_ids:
        doc_data = load_parsed_doc(doc_id)
        if doc_data is None:
            print(f"  [SKIP] {doc_id}: no parsed data")
            continue
        chunks = chunk_doc(doc_data)
        for c in chunks:
            c["doc_id"] = doc_id
            c["source_file"] = doc_data.get("source_file", f"{doc_id}.pdf")
        new_entries.extend(chunks)
        summary = compute_section_summary(chunks)
        print(f"  {doc_id}: {summary['total']} chunks, body={summary['body_count']}, "
              f"sections={summary['section_dist']}")
    print(f"  Generated {len(new_entries)} new chunks")

    if not dry_run:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in kept + new_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  Written: {len(kept) + len(new_entries)} total chunks")

    # 返回变更摘要
    old_summaries = {}
    for doc_id in doc_ids:
        old_doc_chunks = [e for e in removed if e.get("doc_id") == doc_id]
        old_summaries[doc_id] = compute_section_summary(old_doc_chunks)
    new_summaries = {}
    for doc_id in doc_ids:
        new_doc_chunks = [e for e in new_entries if e.get("doc_id") == doc_id]
        new_summaries[doc_id] = compute_section_summary(new_doc_chunks)

    return {
        "removed_count": len(removed),
        "new_count": len(new_entries),
        "kept_count": len(kept),
        "old_summaries": old_summaries,
        "new_summaries": new_summaries,
    }


def rebuild_milvus(
    doc_ids: list[str],
    settings: Settings,
    dry_run: bool = False,
) -> None:
    """
    从 chunks.jsonl 中读取指定 doc_ids 的 chunks，更新 Milvus。
    先删除旧 vectors，再插入新 vectors。
    """
    try:
        from pymilvus import Collection, connections
    except ImportError:
        print("  [WARN] pymilvus not installed, skipping Milvus update")
        return

    uri = settings.retrieval.milvus_uri
    collection_name = settings.retrieval.collection_name

    print(f"  Connecting to Milvus: {uri}")
    connections.connect(uri=uri)
    col = Collection(collection_name)

    # 删除旧 vectors
    delete_expr = f"doc_id in {json.dumps(doc_ids)}"
    print(f"  Deleting old vectors: {delete_expr}")
    if not dry_run:
        col.delete(delete_expr)
        col.flush()
        print(f"  Deleted. Collection count: {col.num_entities}")

    # 读取新 chunks
    jsonl_path = ROOT / "data" / "paper_round1" / "chunks" / "chunks.jsonl"
    new_chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("doc_id") in doc_ids:
                new_chunks.append(rec)

    if not new_chunks:
        print("  [WARN] No new chunks found")
        return

    # 生成 embeddings 并插入
    from src.synbio_rag.infrastructure.embedding import EmbeddingService
    embedder = EmbeddingService(
        model_name="bge-m3",
        model_path=settings.kb.embedding_model_path,
        dim=settings.kb.embedding_dim,
    )

    print(f"  Embedding {len(new_chunks)} chunks...")
    texts = [c.get("retrieval_text", c.get("text", "")) for c in new_chunks]
    vectors = embedder.encode(texts, show_progress_bar=True)

    # 构造 Milvus insert 数据
    insert_data = []
    for i, chunk in enumerate(new_chunks):
        insert_data.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source_file": chunk.get("source_file", ""),
            "title": chunk.get("title", ""),
            "section": chunk.get("section", ""),
            "text": chunk.get("text", ""),
            "retrieval_text": chunk.get("retrieval_text", ""),
            "vector": vectors[i].tolist(),
        })

    if not dry_run:
        # Batch insert
        batch_size = 100
        for i in range(0, len(insert_data), batch_size):
            batch = insert_data[i:i + batch_size]
            col.insert(batch)
        col.flush()
        print(f"  Inserted {len(insert_data)} vectors. Collection count: {col.num_entities}")

    connections.disconnect("default")


def main():
    parser = argparse.ArgumentParser(description="局部重建指定文档的索引")
    parser.add_argument(
        "--doc_ids", required=True,
        help="逗号分隔的 doc_ids，例如 doc_0008,doc_0078",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览变更但不实际写入",
    )
    parser.add_argument(
        "--no-milvus", action="store_true",
        help="只重建 chunks.jsonl，不更新 Milvus",
    )
    args = parser.parse_args()

    doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()]
    if not doc_ids:
        print("[ERROR] No doc_ids provided")
        sys.exit(1)

    print("=" * 60)
    print(f"局部重建: {len(doc_ids)} 个文档")
    print(f"doc_ids: {doc_ids}")
    print(f"mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"milvus: {'SKIP' if args.no_milvus else 'UPDATE'}")
    print("=" * 60)
    print()

    t0 = time.time()

    chunks_dir = ROOT / "data" / "paper_round1" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 重建 chunks.jsonl
    print("[1/2] Rebuilding chunks.jsonl...")
    change_summary = rebuild_chunks_jsonl(doc_ids, chunks_dir, dry_run=args.dry_run)

    # Step 2: 更新 Milvus
    if not args.no_milvus:
        print("\n[2/2] Updating Milvus...")
        settings = Settings.from_env()
        rebuild_milvus(doc_ids, settings, dry_run=args.dry_run)
    else:
        print("\n[2/2] Milvus update SKIPPED (--no-milvus)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # 输出变更摘要
    print("\n" + "=" * 60)
    print("变更摘要")
    print("=" * 60)
    for doc_id in doc_ids:
        old_s = change_summary.get("old_summaries", {}).get(doc_id, {})
        new_s = change_summary.get("new_summaries", {}).get(doc_id, {})
        print(f"\n  {doc_id}:")
        print(f"    chunks: {old_s.get('total', '?')} → {new_s.get('total', '?')}")
        print(f"    body:   {old_s.get('body_count', '?')} → {new_s.get('body_count', '?')}")
        print(f"    old sections: {old_s.get('section_dist', {})}")
        print(f"    new sections: {new_s.get('section_dist', {})}")

    if args.dry_run:
        print("\n[DRY-RUN] No changes were written.")


if __name__ == "__main__":
    main()
