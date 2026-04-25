from __future__ import annotations

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
            search_params={
                "metric_type": self.config.metric_type,
                "params": {"ef": self.config.ef},
            },
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
                    metadata={
                        "tenant_id": filters.tenant_id if filters else "default",
                        "chunk_index": entity.get("chunk_index"),
                    },
                )
            )
        return chunks


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
