from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ...domain.config import KnowledgeBaseConfig, RetrievalConfig
from ...domain.schemas import QueryFilters, RetrievedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+|[\u4e00-\u9fff]")


class BM25Retriever:
    def __init__(
        self,
        retrieval_config: RetrievalConfig,
        kb_config: KnowledgeBaseConfig,
        milvus_client: Any | None = None,
    ):
        self.retrieval_config = retrieval_config
        self.kb_config = kb_config
        self.milvus_client = milvus_client
        self._records: list[RetrievedChunk] = []
        self._doc_len: list[int] = []
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._term_freqs: list[Counter[str]] = []
        self._avgdl = 0.0
        self._loaded = False

    def search(
        self,
        question: str,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        if not self.retrieval_config.bm25_enabled:
            return []
        self._ensure_index()
        if not self._records:
            return []

        query_terms = _tokenize(question)
        if not query_terms:
            return []

        scored: list[RetrievedChunk] = []
        for idx, chunk in self._filter_records(filters):
            score = self._score(query_terms, idx)
            if score <= 0:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    title=chunk.title,
                    section=chunk.section,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    bm25_score=score,
                    metadata=dict(chunk.metadata),
                )
            )

        scored.sort(key=lambda item: item.bm25_score, reverse=True)
        return scored[:limit]

    def _ensure_index(self) -> None:
        if self._loaded:
            return
        if self._load_cache():
            self._loaded = True
            return
        self._records = self._load_records()
        self._build_index()
        self._save_cache()
        self._loaded = True

    def _build_index(self) -> None:
        self._doc_len = []
        self._doc_freq = defaultdict(int)
        self._term_freqs = []
        for chunk in self._records:
            terms = _tokenize(_retrieval_text(chunk))
            tf = Counter(terms)
            self._term_freqs.append(tf)
            self._doc_len.append(len(terms))
            for term in tf:
                self._doc_freq[term] += 1
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0

    def _load_records(self) -> list[RetrievedChunk]:
        jsonl_path = Path(self.kb_config.chunk_jsonl)
        if jsonl_path.exists():
            return self._load_from_jsonl(jsonl_path)
        if self.milvus_client is not None:
            return self._load_from_milvus()
        return []

    def _load_from_jsonl(self, path: Path) -> list[RetrievedChunk]:
        records: list[RetrievedChunk] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata: dict[str, Any] = {
                    "chunk_index": item.get("chunk_index"),
                    "retrieval_text": item.get("retrieval_text", ""),
                    "content_kind": item.get("content_kind", "body"),
                    "quality_score": item.get("quality_score", 0.0),
                    "contains_table_text": item.get("contains_table_text", False),
                    "contains_table_caption": item.get("contains_table_caption", False),
                    "contains_figure_caption": item.get("contains_figure_caption", False),
                    "contains_image": item.get("contains_image", False),
                    "object_type": item.get("object_type", "body"),
                    "object_id": item.get("object_id", ""),
                    "block_types": item.get("block_types", []),
                    "evidence_types": item.get("evidence_types", []),
                }
                records.append(
                    RetrievedChunk(
                        chunk_id=item.get("chunk_id", ""),
                        doc_id=item.get("doc_id", ""),
                        source_file=item.get("source_file", ""),
                        title=item.get("title", ""),
                        section=item.get("section", ""),
                        text=item.get("text", ""),
                        page_start=item.get("page_start"),
                        page_end=item.get("page_end"),
                        metadata=metadata,
                    )
                )
        return records

    def _load_from_milvus(self) -> list[RetrievedChunk]:
        iterator = self.milvus_client.query_iterator(
            collection_name=self.retrieval_config.collection_name,
            batch_size=self.retrieval_config.bm25_batch_size,
            output_fields=[
                "chunk_id",
                "doc_id",
                "source_file",
                "title",
                "section",
                "page_start",
                "page_end",
                "chunk_index",
                "text",
                "retrieval_text",
                "content_kind",
                "quality_score",
                "contains_table_text",
                "contains_table_caption",
                "contains_figure_caption",
                "contains_image",
                "object_type",
                "object_id",
                "metadata_json",
            ],
        )
        records: list[RetrievedChunk] = []
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                for row in rows:
                    parsed = _safe_parse_metadata_json(row.get("metadata_json", ""))
                    metadata: dict[str, Any] = {
                        "chunk_index": row.get("chunk_index"),
                        "retrieval_text": row.get("retrieval_text", ""),
                        "content_kind": row.get("content_kind", "body"),
                        "quality_score": row.get("quality_score", 0.0),
                        "contains_table_text": row.get("contains_table_text", False),
                        "contains_table_caption": row.get("contains_table_caption", False),
                        "contains_figure_caption": row.get("contains_figure_caption", False),
                        "contains_image": row.get("contains_image", False),
                        "object_type": row.get("object_type", "body"),
                        "object_id": row.get("object_id", ""),
                    }
                    metadata.update(parsed)
                    records.append(
                        RetrievedChunk(
                            chunk_id=row.get("chunk_id", ""),
                            doc_id=row.get("doc_id", ""),
                            source_file=row.get("source_file", ""),
                            title=row.get("title", ""),
                            section=row.get("section", ""),
                            text=row.get("text", ""),
                            page_start=row.get("page_start"),
                            page_end=row.get("page_end"),
                            metadata=metadata,
                        )
                    )
        finally:
            iterator.close()
        return records

    def _filter_records(self, filters: QueryFilters | None) -> list[tuple[int, RetrievedChunk]]:
        if not filters:
            return list(enumerate(self._records))
        doc_ids = set(filters.doc_ids)
        sections = set(filters.sections)
        source_files = set(filters.source_files)
        items: list[tuple[int, RetrievedChunk]] = []
        for idx, chunk in enumerate(self._records):
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            if sections and chunk.section not in sections:
                continue
            if source_files and chunk.source_file not in source_files:
                continue
            items.append((idx, chunk))
        return items

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        score = 0.0
        tf = self._term_freqs[doc_idx]
        dl = self._doc_len[doc_idx]
        n_docs = len(self._records)
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.retrieval_config.bm25_k1 * (
                1
                - self.retrieval_config.bm25_b
                + self.retrieval_config.bm25_b * dl / max(self._avgdl, 1.0)
            )
            score += idf * (freq * (self.retrieval_config.bm25_k1 + 1)) / max(denom, 1e-9)
        return score

    def _load_cache(self) -> bool:
        cache_path = Path(self.retrieval_config.bm25_cache_path)
        jsonl_path = Path(self.kb_config.chunk_jsonl)
        if not cache_path.exists():
            return False
        if jsonl_path.exists() and cache_path.stat().st_mtime < jsonl_path.stat().st_mtime:
            return False
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        self._records = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                source_file=item["source_file"],
                title=item["title"],
                section=item["section"],
                text=item["text"],
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                metadata=item.get("metadata", {"chunk_index": item.get("chunk_index")}),
            )
            for item in payload.get("records", [])
        ]
        self._doc_len = [int(value) for value in payload.get("doc_len", [])]
        self._doc_freq = defaultdict(int, {k: int(v) for k, v in payload.get("doc_freq", {}).items()})
        self._term_freqs = [Counter({k: int(v) for k, v in item.items()}) for item in payload.get("term_freqs", [])]
        self._avgdl = float(payload.get("avgdl", 0.0))
        return True

    def _save_cache(self) -> None:
        cache_path = Path(self.retrieval_config.bm25_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [
                {
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "source_file": item.source_file,
                    "title": item.title,
                    "section": item.section,
                    "text": item.text,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "metadata": dict(item.metadata),
                }
                for item in self._records
            ],
            "doc_len": self._doc_len,
            "doc_freq": dict(self._doc_freq),
            "term_freqs": [dict(counter) for counter in self._term_freqs],
            "avgdl": self._avgdl,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _retrieval_text(chunk: RetrievedChunk) -> str:
    # Phase 12A: 优先使用 metadata 中的 retrieval_text
    rt = chunk.metadata.get("retrieval_text", "")
    if rt:
        return rt
    # fallback: 用 title/section/source_file + text 构造
    metadata_lines = []
    if chunk.title:
        metadata_lines.append(f"title {chunk.title}")
    if chunk.section:
        metadata_lines.append(f"section {chunk.section}")
    if chunk.source_file:
        metadata_lines.append(f"source_file {chunk.source_file}")
    if chunk.doc_id:
        metadata_lines.append(f"doc_id {chunk.doc_id}")
    metadata_lines.append(chunk.text or "")
    return "\n".join(metadata_lines)


def _safe_parse_metadata_json(raw: str) -> dict[str, Any]:
    """安全解析 metadata_json，失败时返回标记。"""
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {"metadata_json_parse_error": True}
