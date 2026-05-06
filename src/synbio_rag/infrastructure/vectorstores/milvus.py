from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from ...domain.config import RetrievalConfig
from ...domain.schemas import QueryFilters, RetrievedChunk


class MilvusRetriever:
    def __init__(self, config: RetrievalConfig, embedder: Any):
        self.config = config
        self.embedder = embedder
        uri = config.milvus_uri
        if not uri.startswith(("http://", "https://", "unix://", "tcp://")):
            uri = str(Path(config.milvus_uri).resolve())
        self.client = MilvusClient(uri)

    def search(
        self,
        question: str,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        query_vec = self.embedder.encode([question])[0]
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=[query_vec],
            anns_field=self.config.vector_field,
            limit=limit,
            search_params=self._build_search_params(),
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
            filter=build_scalar_filter(filters, self.config.max_filter_items),
        )
        if not results:
            return []

        chunks: list[RetrievedChunk] = []
        score_floor = filters.min_score if filters and filters.min_score is not None else self.config.score_floor
        for hit in results[0]:
            entity = hit.get("entity", {})
            score = float(hit.get("distance", 0.0))
            if score < score_floor:
                continue

            # 安全解析 metadata_json
            parsed_meta = _safe_parse_metadata_json(entity.get("metadata_json", ""))

            metadata: dict[str, Any] = {
                "tenant_id": filters.tenant_id if filters else "default",
                "chunk_index": entity.get("chunk_index"),
                "retrieval_text": entity.get("retrieval_text", ""),
                "content_kind": entity.get("content_kind", "body"),
                "quality_score": entity.get("quality_score", 0.0),
                "contains_table_text": entity.get("contains_table_text", False),
                "contains_table_caption": entity.get("contains_table_caption", False),
                "contains_figure_caption": entity.get("contains_figure_caption", False),
                "contains_image": entity.get("contains_image", False),
                "object_type": entity.get("object_type", "body"),
                "object_id": entity.get("object_id", ""),
            }
            # merge metadata_json 内容
            metadata.update(parsed_meta)

            chunks.append(
                RetrievedChunk(
                    chunk_id=entity.get("chunk_id", ""),
                    doc_id=entity.get("doc_id", ""),
                    source_file=entity.get("source_file", ""),
                    title=entity.get("title", ""),
                    section=entity.get("section", ""),
                    text=entity.get("text", ""),
                    page_start=_normalize_page(entity.get("page_start")),
                    page_end=_normalize_page(entity.get("page_end")),
                    vector_score=score,
                    metadata=metadata,
                )
            )
        return chunks

    def _build_search_params(self) -> dict[str, Any]:
        """根据 index_type 构造 search_params。"""
        params: dict[str, Any] = {
            "metric_type": self.config.metric_type,
        }
        if self.config.index_type == "HNSW":
            params["params"] = {"ef": self.config.hnsw_ef}
        else:
            params["params"] = {"nprobe": self.config.nprobe}
        return params


def build_scalar_filter(filters: QueryFilters | None, max_filter_items: int) -> str:
    if not filters:
        return ""
    clauses: list[str] = []
    if filters.doc_ids:
        values = ",".join(f'"{item}"' for item in filters.doc_ids[:max_filter_items])
        clauses.append(f"doc_id in [{values}]")
    if filters.sections:
        values = ",".join(f'"{item}"' for item in filters.sections[:max_filter_items])
        clauses.append(f"section in [{values}]")
    if filters.source_files:
        values = ",".join(f'"{item}"' for item in filters.source_files[:max_filter_items])
        clauses.append(f"source_file in [{values}]")
    return " and ".join(clauses)


def _normalize_page(value: Any) -> int | None:
    if value in (-1, None):
        return None
    return int(value)


def _safe_parse_metadata_json(raw: str) -> dict[str, Any]:
    """安全解析 metadata_json，失败时返回标记。"""
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {"metadata_json_parse_error": True}
