from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..domain.config import KnowledgeBaseConfig, RetrievalConfig
from ..domain.schemas import RetrievedChunk


class ChunkNeighborExpander:
    def __init__(self, kb_config: KnowledgeBaseConfig, retrieval_config: RetrievalConfig):
        self.kb_config = kb_config
        self.retrieval_config = retrieval_config
        self._loaded = False
        self._by_id: dict[str, RetrievedChunk] = {}
        self._positions: dict[str, tuple[str, int]] = {}
        self._doc_chunks: dict[str, list[RetrievedChunk]] = {}
        self.last_debug: dict[str, object] = {}

    def expand(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.last_debug = {
            "enabled": self.retrieval_config.neighbor_expansion_enabled,
            "input_count": len(chunks),
            "window_size": self.retrieval_config.neighbor_window_size,
        }
        if not self.retrieval_config.neighbor_expansion_enabled or not chunks:
            self.last_debug["output_count"] = len(chunks)
            return chunks

        self._ensure_loaded()
        if not self._by_id:
            self.last_debug["output_count"] = len(chunks)
            self.last_debug["reason"] = "neighbor_index_empty"
            return chunks

        window_size = max(0, int(self.retrieval_config.neighbor_window_size))
        max_chunks = max(len(chunks), int(self.retrieval_config.neighbor_expansion_max_chunks))
        expanded: list[RetrievedChunk] = []
        seen: set[str] = set()
        added_neighbors = 0

        for chunk in chunks:
            for candidate in self._window_for(chunk, window_size):
                if candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                if candidate.chunk_id == chunk.chunk_id:
                    expanded.append(chunk)
                else:
                    expanded.append(self._clone_neighbor(candidate, anchor=chunk))
                    added_neighbors += 1
                if len(expanded) >= max_chunks:
                    expanded = self._sort_by_document_order(expanded)
                    self.last_debug.update(
                        {
                            "output_count": len(expanded),
                            "added_neighbors": added_neighbors,
                            "truncated": True,
                        }
                    )
                    return expanded

        expanded = self._sort_by_document_order(expanded)
        self.last_debug.update(
            {
                "output_count": len(expanded),
                "added_neighbors": added_neighbors,
                "truncated": False,
            }
        )
        return expanded

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        path = Path(self.kb_config.chunk_jsonl)
        if not path.exists():
            self._loaded = True
            return

        grouped: dict[str, list[tuple[int, RetrievedChunk]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as handle:
            for ordinal, raw in enumerate(handle):
                raw = raw.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                chunk = RetrievedChunk(
                    chunk_id=item.get("chunk_id", ""),
                    doc_id=item.get("doc_id", ""),
                    source_file=item.get("source_file", ""),
                    title=item.get("title", ""),
                    section=item.get("section", ""),
                    text=item.get("text", ""),
                    page_start=item.get("page_start"),
                    page_end=item.get("page_end"),
                    metadata={
                        "chunk_index": item.get("chunk_index", ordinal),
                        "quality_score": item.get("quality_score"),
                    },
                )
                if not chunk.chunk_id or not chunk.doc_id:
                    continue
                chunk_index = _safe_int(item.get("chunk_index"), ordinal)
                grouped[chunk.doc_id].append((chunk_index, chunk))
                self._by_id[chunk.chunk_id] = chunk

        for doc_id, pairs in grouped.items():
            ordered = [chunk for _idx, chunk in sorted(pairs, key=lambda pair: pair[0])]
            self._doc_chunks[doc_id] = ordered
            for position, chunk in enumerate(ordered):
                self._positions[chunk.chunk_id] = (doc_id, position)
        self._loaded = True

    def _sort_by_document_order(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        def sort_key(chunk: RetrievedChunk) -> tuple[str, int]:
            position = self._positions.get(chunk.chunk_id)
            if position:
                return (position[0], position[1])
            idx = chunk.metadata.get("chunk_index", 0)
            return (chunk.doc_id, idx)

        return sorted(chunks, key=sort_key)

    def _window_for(self, chunk: RetrievedChunk, window_size: int) -> list[RetrievedChunk]:
        position = self._positions.get(chunk.chunk_id)
        if not position:
            return [chunk]
        doc_id, idx = position
        doc_chunks = self._doc_chunks.get(doc_id) or []
        ordered = [doc_chunks[idx]]
        for distance in range(1, window_size + 1):
            prev_idx = idx - distance
            next_idx = idx + distance
            if prev_idx >= 0:
                ordered.append(doc_chunks[prev_idx])
            if next_idx < len(doc_chunks):
                ordered.append(doc_chunks[next_idx])
        return ordered

    def _clone_neighbor(self, neighbor: RetrievedChunk, anchor: RetrievedChunk) -> RetrievedChunk:
        cloned = RetrievedChunk(
            chunk_id=neighbor.chunk_id,
            doc_id=neighbor.doc_id,
            source_file=neighbor.source_file,
            title=neighbor.title,
            section=neighbor.section,
            text=neighbor.text,
            page_start=neighbor.page_start,
            page_end=neighbor.page_end,
            vector_score=anchor.vector_score,
            bm25_score=anchor.bm25_score,
            rerank_score=max(anchor.rerank_score - 0.01, 0.0),
            fusion_score=anchor.fusion_score,
            metadata=dict(neighbor.metadata),
        )
        cloned.metadata.update(
            {
                "expanded_from_chunk_id": anchor.chunk_id,
                "neighbor_expansion": True,
            }
        )
        return cloned


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
