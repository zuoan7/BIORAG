from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path for scripts imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingestion.build_parent_index import build_parent_records, make_parent_id
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.index.parent_store import ParentStore


def _chunk(
    chunk_id: str,
    doc_id: str,
    chunk_index: int,
    *,
    section: str = "Body",
    section_path: list[str] | None = None,
    table_caption: bool = False,
    figure_caption: bool = False,
    text: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "title": f"title-{doc_id}",
        "section": section,
        "section_path": section_path or [section],
        "chunk_index": chunk_index,
        "text": text or f"text for {chunk_id}",
        "page_start": chunk_index,
        "page_end": chunk_index,
        "page_numbers": [chunk_index],
        "block_ids": [f"b{chunk_index}"],
        "source_block_ids": [f"sb{chunk_index}"],
        "contains_table_caption": table_caption,
        "contains_figure_caption": figure_caption,
        "contains_table_text": False,
        "contains_image": False,
        "contains_references": False,
        "contains_metadata": False,
        "contains_noise": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_parent_id_stability():
    assert make_parent_id("doc", "doc_1", "doc_1::doc") == make_parent_id("doc", "doc_1", "doc_1::doc")


def test_doc_parent_aggregation():
    parents = build_parent_records(
        [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2), _chunk("c3", "d2", 1)],
        window_size=1,
    )
    doc_parent = next(parent for parent in parents if parent["parent_type"] == "doc" and parent["doc_id"] == "d1")
    assert doc_parent["child_chunk_ids"] == ["c1", "c2"]
    assert doc_parent["page_numbers"] == [1, 2]


def test_section_parent_aggregation():
    parents = build_parent_records(
        [
            _chunk("c1", "d1", 1, section="Intro"),
            _chunk("c2", "d1", 2, section="Intro"),
            _chunk("c3", "d1", 3, section="Methods"),
        ],
        window_size=1,
    )
    intro_parent = next(
        parent
        for parent in parents
        if parent["parent_type"] == "section" and parent["doc_id"] == "d1" and parent["section"] == "Intro"
    )
    assert intro_parent["child_chunk_ids"] == ["c1", "c2"]


def test_chunk_window_parent_neighbors():
    parents = build_parent_records(
        [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2), _chunk("c3", "d1", 3)],
        window_size=1,
    )
    window_parent = next(
        parent
        for parent in parents
        if parent["parent_type"] == "chunk_window" and parent["anchor_chunk_id"] == "c2"
    )
    assert window_parent["child_chunk_ids"] == ["c1", "c2", "c3"]


def test_caption_context_only_for_caption_chunks():
    parents = build_parent_records(
        [
            _chunk("c1", "d1", 1),
            _chunk("c2", "d1", 2, table_caption=True),
            _chunk("c3", "d1", 3, figure_caption=True),
        ],
        window_size=1,
    )
    caption_parents = [parent for parent in parents if parent["parent_type"] == "caption_context"]
    assert [parent["anchor_chunk_id"] for parent in caption_parents] == ["c2", "c3"]
    assert {parent["caption_kind"] for parent in caption_parents} == {"table_caption", "figure_caption"}


def test_parent_store_lookup_and_children(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2, table_caption=True), _chunk("c3", "d1", 3)]
    parents = build_parent_records(chunks, window_size=1)
    chunk_path = tmp_path / "chunks.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    _write_jsonl(chunk_path, chunks)
    _write_jsonl(parent_path, parents)

    store = ParentStore.from_jsonl(parent_path, chunk_jsonl_path=chunk_path)
    window_parent = next(parent for parent in parents if parent["parent_type"] == "chunk_window" and parent["anchor_chunk_id"] == "c2")
    parent = store.get_parent(window_parent["parent_id"])
    assert parent is not None
    assert parent.parent_type == "chunk_window"

    related = store.get_parents_for_chunk("c2")
    assert {parent.parent_type for parent in related} >= {
        "doc", "section", "section_path", "page", "chunk_window", "caption_context"
    }

    children = store.get_children(window_parent["parent_id"])
    assert all(isinstance(child, RetrievedChunk) for child in children)
    assert [child.chunk_id for child in children] == ["c1", "c2", "c3"]


def test_orphan_child_detection_shape():
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2)]
    parents = build_parent_records(chunks, window_size=1)
    child_ids = {chunk_id for parent in parents for chunk_id in parent["child_chunk_ids"]}
    assert child_ids == {"c1", "c2"}


def test_expand_by_parent_returns_related_chunks(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2, figure_caption=True), _chunk("c3", "d1", 3)]
    parents = build_parent_records(chunks, window_size=1)
    chunk_path = tmp_path / "chunks.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    _write_jsonl(chunk_path, chunks)
    _write_jsonl(parent_path, parents)
    store = ParentStore.from_jsonl(parent_path, chunk_jsonl_path=chunk_path)

    seed = RetrievedChunk(
        chunk_id="c2",
        doc_id="d1",
        source_file="d1.pdf",
        title="title-d1",
        section="Body",
        text="seed text",
        metadata={"chunk_index": 2},
    )
    expanded = store.expand_by_parent([seed], parent_types=["caption_context"], max_total=5, per_seed_limit=1)
    assert [chunk.chunk_id for chunk in expanded] == ["c2", "c1", "c3"]
    assert expanded[1].metadata["expanded_by_parent"] is True
