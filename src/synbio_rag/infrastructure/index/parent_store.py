from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.schemas import RetrievedChunk


@dataclass
class ParentRecord:
    parent_id: str
    parent_type: str
    doc_id: str
    source_file: str = ""
    title: str = ""
    section: str = ""
    section_path: list[str] = field(default_factory=list)
    section_path_key: str = ""
    anchor_chunk_id: str = ""
    child_chunk_ids: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    page_numbers: list[int] = field(default_factory=list)
    page_number: int | None = None
    block_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    content_kinds: list[str] = field(default_factory=list)
    contains_table_caption: bool = False
    contains_figure_caption: bool = False
    contains_table_text: bool = False
    contains_image: bool = False
    text_preview: str = ""
    caption_kind: str = ""
    evidence_type: str = ""


class ParentStore:
    def __init__(
        self,
        parents: dict[str, ParentRecord],
        parents_by_chunk: dict[str, list[str]],
        chunk_by_id: dict[str, RetrievedChunk] | None = None,
    ) -> None:
        self._parents = parents
        self._parents_by_chunk = parents_by_chunk
        self._chunk_by_id = chunk_by_id or {}
        self._parents_by_type: dict[str, list[ParentRecord]] = {}
        self._parents_by_doc: dict[str, list[ParentRecord]] = {}
        self._page_parents: dict[tuple[str, int], ParentRecord] = {}
        self._section_path_parents: dict[str, list[ParentRecord]] = {}
        self._evidence_parents: dict[tuple[str, str], list[ParentRecord]] = {}
        self._build_indexes()

    @classmethod
    def from_jsonl(cls, path: str | Path, chunk_jsonl_path: str | Path | None = None) -> "ParentStore":
        parents: dict[str, ParentRecord] = {}
        parents_by_chunk: dict[str, list[str]] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                record = ParentRecord(
                    parent_id=str(item.get("parent_id") or ""),
                    parent_type=str(item.get("parent_type") or ""),
                    doc_id=str(item.get("doc_id") or ""),
                    source_file=str(item.get("source_file") or ""),
                    title=str(item.get("title") or ""),
                    section=str(item.get("section") or ""),
                    section_path=[str(v) for v in item.get("section_path") or []],
                    section_path_key=str(item.get("section_path_key") or ""),
                    anchor_chunk_id=str(item.get("anchor_chunk_id") or ""),
                    child_chunk_ids=[str(v) for v in item.get("child_chunk_ids") or [] if str(v or "").strip()],
                    page_start=_safe_int(item.get("page_start")),
                    page_end=_safe_int(item.get("page_end")),
                    page_numbers=_coerce_int_list(item.get("page_numbers")),
                    page_number=_safe_int(item.get("page_number")),
                    block_ids=_coerce_str_list(item.get("block_ids")),
                    source_block_ids=_coerce_str_list(item.get("source_block_ids")),
                    content_kinds=_coerce_str_list(item.get("content_kinds")),
                    contains_table_caption=bool(item.get("contains_table_caption")),
                    contains_figure_caption=bool(item.get("contains_figure_caption")),
                    contains_table_text=bool(item.get("contains_table_text")),
                    contains_image=bool(item.get("contains_image")),
                    text_preview=str(item.get("text_preview") or ""),
                    caption_kind=str(item.get("caption_kind") or ""),
                    evidence_type=str(item.get("evidence_type") or ""),
                )
                if not record.parent_id:
                    continue
                parents[record.parent_id] = record
                for chunk_id in record.child_chunk_ids:
                    parents_by_chunk.setdefault(chunk_id, []).append(record.parent_id)

        chunk_by_id = cls._load_chunks(chunk_jsonl_path) if chunk_jsonl_path else {}
        return cls(parents=parents, parents_by_chunk=parents_by_chunk, chunk_by_id=chunk_by_id)

    def get_parent(self, parent_id: str) -> ParentRecord | None:
        return self._parents.get(parent_id)

    def get_parents_for_chunk(self, chunk_id: str) -> list[ParentRecord]:
        return [
            self._parents[parent_id]
            for parent_id in self._parents_by_chunk.get(chunk_id, [])
            if parent_id in self._parents
        ]

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None:
        return self._chunk_by_id.get(chunk_id)

    def get_parents_by_type(self, parent_type: str) -> list[ParentRecord]:
        return list(self._parents_by_type.get(parent_type, []))

    def get_parents_for_doc(self, doc_id: str, parent_type: str | None = None) -> list[ParentRecord]:
        parents = self._parents_by_doc.get(doc_id, [])
        if parent_type is None:
            return list(parents)
        return [parent for parent in parents if parent.parent_type == parent_type]

    def get_page_parent(self, doc_id: str, page_number: int) -> ParentRecord | None:
        return self._page_parents.get((doc_id, int(page_number)))

    def get_section_path_parents(self, doc_id: str) -> list[ParentRecord]:
        return [
            parent
            for parent in self._section_path_parents.get(doc_id, [])
            if parent.parent_type == "section_path"
        ]

    def get_evidence_parents(self, doc_id: str, evidence_type: str) -> list[ParentRecord]:
        return list(self._evidence_parents.get((doc_id, str(evidence_type)), []))

    def get_children(self, parent_id: str) -> list[str] | list[RetrievedChunk]:
        parent = self._parents.get(parent_id)
        if not parent:
            return []
        if not self._chunk_by_id:
            return list(parent.child_chunk_ids)
        return [
            self._chunk_by_id[chunk_id]
            for chunk_id in parent.child_chunk_ids
            if chunk_id in self._chunk_by_id
        ]

    def get_parent_types_for_chunk(self, chunk_id: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for parent in self.get_parents_for_chunk(chunk_id):
            if parent.parent_type in seen:
                continue
            seen.add(parent.parent_type)
            ordered.append(parent.parent_type)
        return ordered

    def expand_by_parent(
        self,
        seed_chunks: list[RetrievedChunk],
        parent_types: list[str],
        max_total: int,
        per_seed_limit: int,
    ) -> list[RetrievedChunk]:
        if not seed_chunks or max_total <= 0:
            return []

        selected_parent_types = {str(v) for v in parent_types if str(v).strip()}
        expanded: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for seed in seed_chunks:
            if seed.chunk_id not in seen_chunk_ids and len(expanded) < max_total:
                expanded.append(seed)
                seen_chunk_ids.add(seed.chunk_id)

        if len(expanded) >= max_total or not self._chunk_by_id or not selected_parent_types:
            return expanded[:max_total]

        effective_limit = max(0, int(per_seed_limit))
        for seed in seed_chunks:
            chosen = 0
            for parent in self.get_parents_for_chunk(seed.chunk_id):
                if parent.parent_type not in selected_parent_types:
                    continue
                if effective_limit and chosen >= effective_limit:
                    break
                chosen += 1
                for child in self.get_children(parent.parent_id):
                    if not isinstance(child, RetrievedChunk):
                        continue
                    if child.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(child.chunk_id)
                    expanded.append(_clone_chunk(child, parent))
                    if len(expanded) >= max_total:
                        return expanded

        return expanded

    def expand_caption_context(self, seed_chunk_id: str) -> list[RetrievedChunk]:
        return self._expand_for_chunk(seed_chunk_id, preferred_type="caption_context", anchor_only=True)

    def expand_page_context(self, seed_chunk_id: str) -> list[RetrievedChunk]:
        preferred_page = None
        seed = self._chunk_by_id.get(seed_chunk_id)
        if seed:
            page_numbers = seed.metadata.get("page_numbers") or []
            if isinstance(page_numbers, list) and page_numbers:
                try:
                    preferred_page = int(page_numbers[0])
                except (TypeError, ValueError):
                    preferred_page = None
        return self._expand_for_chunk(
            seed_chunk_id,
            preferred_type="page",
            anchor_only=False,
            predicate=lambda parent: preferred_page is None or parent.page_number == preferred_page,
        )

    def expand_section_path_context(self, seed_chunk_id: str) -> list[RetrievedChunk]:
        preferred_key = ""
        seed = self._chunk_by_id.get(seed_chunk_id)
        if seed:
            section_path = seed.metadata.get("section_path") or []
            if isinstance(section_path, list):
                preferred_key = " > ".join(str(part).strip() for part in section_path if str(part).strip())
        return self._expand_for_chunk(
            seed_chunk_id,
            preferred_type="section_path",
            anchor_only=False,
            predicate=lambda parent: not preferred_key or parent.section_path_key == preferred_key,
        )

    def _expand_for_chunk(
        self,
        chunk_id: str,
        preferred_type: str,
        anchor_only: bool,
        predicate: Any | None = None,
    ) -> list[RetrievedChunk]:
        if not self._chunk_by_id:
            return []
        matches: list[ParentRecord] = []
        for parent in self.get_parents_for_chunk(chunk_id):
            if parent.parent_type != preferred_type:
                continue
            if anchor_only and parent.anchor_chunk_id != chunk_id:
                continue
            if predicate is not None and not predicate(parent):
                continue
            matches.append(parent)
        if not matches and anchor_only:
            matches = [parent for parent in self.get_parents_for_chunk(chunk_id) if parent.parent_type == preferred_type]
        if not matches and predicate is not None:
            matches = [
                parent
                for parent in self.get_parents_for_chunk(chunk_id)
                if parent.parent_type == preferred_type
            ]
        if not matches:
            return []
        anchor_parent = matches[0]
        children = self.get_children(anchor_parent.parent_id)
        if not children or not isinstance(children[0], RetrievedChunk):
            return []
        return [_clone_chunk(child, anchor_parent) for child in children]  # type: ignore[arg-type]

    def _build_indexes(self) -> None:
        parents_by_type: dict[str, list[ParentRecord]] = {}
        parents_by_doc: dict[str, list[ParentRecord]] = {}
        page_parents: dict[tuple[str, int], ParentRecord] = {}
        section_path_parents: dict[str, list[ParentRecord]] = {}
        evidence_parents: dict[tuple[str, str], list[ParentRecord]] = {}

        for parent in self._parents.values():
            parents_by_type.setdefault(parent.parent_type, []).append(parent)
            parents_by_doc.setdefault(parent.doc_id, []).append(parent)
            if parent.parent_type == "page" and parent.page_number is not None:
                page_parents[(parent.doc_id, parent.page_number)] = parent
            if parent.parent_type == "section_path":
                section_path_parents.setdefault(parent.doc_id, []).append(parent)
            if parent.parent_type == "evidence_type_context" and parent.evidence_type:
                evidence_parents.setdefault((parent.doc_id, parent.evidence_type), []).append(parent)

        self._parents_by_type = parents_by_type
        self._parents_by_doc = parents_by_doc
        self._page_parents = page_parents
        self._section_path_parents = section_path_parents
        self._evidence_parents = evidence_parents

    @staticmethod
    def _load_chunks(path: str | Path | None) -> dict[str, RetrievedChunk]:
        if path is None:
            return {}
        chunk_by_id: dict[str, RetrievedChunk] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for ordinal, raw in enumerate(handle):
                raw = raw.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                chunk_id = str(item.get("chunk_id") or "")
                if not chunk_id:
                    continue
                chunk_by_id[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=str(item.get("doc_id") or ""),
                    source_file=str(item.get("source_file") or ""),
                    title=str(item.get("title") or ""),
                    section=str(item.get("section") or ""),
                    text=str(item.get("text") or ""),
                    page_start=_safe_int(item.get("page_start")),
                    page_end=_safe_int(item.get("page_end")),
                    metadata={
                        "chunk_index": _safe_int(item.get("chunk_index"), ordinal) or ordinal,
                        "page_numbers": _coerce_int_list(item.get("page_numbers")),
                        "section_path": _coerce_str_list(item.get("section_path")),
                        "evidence_types": _coerce_str_list(item.get("evidence_types")),
                        "contains_table_caption": bool(item.get("contains_table_caption")),
                        "contains_figure_caption": bool(item.get("contains_figure_caption")),
                        "contains_table_text": bool(item.get("contains_table_text")),
                        "contains_image": bool(item.get("contains_image")),
                        "parent_store_loaded": True,
                    },
                )
        return chunk_by_id


def _clone_chunk(chunk: RetrievedChunk, parent: ParentRecord) -> RetrievedChunk:
    cloned = RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        title=chunk.title,
        section=chunk.section,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score,
        rerank_score=chunk.rerank_score,
        fusion_score=chunk.fusion_score,
        metadata=dict(chunk.metadata),
    )
    cloned.metadata.update(
        {
            "expanded_by_parent": True,
            "parent_id": parent.parent_id,
            "parent_type": parent.parent_type,
            "anchor_chunk_id": parent.anchor_chunk_id,
        }
    )
    return cloned


def _safe_int(value: object, default: int | None = None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v or "").strip()]


def _coerce_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    output: list[int] = []
    for item in value:
        coerced = _safe_int(item)
        if coerced is not None:
            output.append(coerced)
    return output
