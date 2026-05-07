from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingestion.build_parent_index import build_parent_records, make_parent_id
from src.synbio_rag.infrastructure.index.parent_store import ParentStore


def _chunk(
    chunk_id: str,
    doc_id: str,
    chunk_index: int,
    *,
    section: str = "Body",
    section_path: list[str] | None = None,
    page_numbers: list[int] | None = None,
    evidence_types: list[str] | None = None,
    table_caption: bool = False,
    figure_caption: bool = False,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "title": f"title-{doc_id}",
        "section": section,
        "section_path": section_path if section_path is not None else [section],
        "chunk_index": chunk_index,
        "text": f"text for {chunk_id}",
        "page_start": (page_numbers or [chunk_index])[0],
        "page_end": (page_numbers or [chunk_index])[-1],
        "page_numbers": page_numbers or [chunk_index],
        "block_ids": [f"b{chunk_index}"],
        "source_block_ids": [f"sb{chunk_index}"],
        "evidence_types": evidence_types or [],
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


def test_section_path_parent_builds_from_section_path():
    chunks = [
        _chunk("c1", "d1", 1, section="Intro", section_path=["Intro", "Background"]),
        _chunk("c2", "d1", 2, section="Intro", section_path=["Intro", "Background"]),
        _chunk("c3", "d1", 3, section="Intro", section_path=["Intro", "Aim"]),
    ]
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=5)
    sp_parent = next(
        parent
        for parent in parents
        if parent["parent_type"] == "section_path" and parent["section_path_key"] == "Intro > Background"
    )
    assert sp_parent["child_chunk_ids"] == ["c1", "c2"]


def test_page_parent_builds_for_multi_page_chunks():
    chunks = [
        _chunk("c1", "d1", 1, page_numbers=[1, 2]),
        _chunk("c2", "d1", 2, page_numbers=[2]),
    ]
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=5)
    page1 = next(parent for parent in parents if parent["parent_type"] == "page" and parent["page_number"] == 1)
    page2 = next(parent for parent in parents if parent["parent_type"] == "page" and parent["page_number"] == 2)
    assert page1["child_chunk_ids"] == ["c1"]
    assert page2["child_chunk_ids"] == ["c1", "c2"]


def test_evidence_type_context_uses_explicit_types_and_fallback():
    chunks = [
        _chunk("c1", "d1", 1, evidence_types=["figure_caption", "paragraph"], figure_caption=True),
        _chunk("c2", "d1", 2, section="Materials and Methods"),
        _chunk("c3", "d1", 3, section="Results"),
    ]
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=5)
    explicit = next(
        parent
        for parent in parents
        if parent["parent_type"] == "evidence_type_context" and parent["evidence_type"] == "figure_caption"
    )
    method = next(
        parent
        for parent in parents
        if parent["parent_type"] == "evidence_type_context" and parent["evidence_type"] == "method"
    )
    result = next(
        parent
        for parent in parents
        if parent["parent_type"] == "evidence_type_context" and parent["evidence_type"] == "result"
    )
    assert explicit["child_chunk_ids"] == ["c1"]
    assert method["child_chunk_ids"] == ["c2"]
    assert result["child_chunk_ids"] == ["c3"]


def test_caption_context_prefers_same_page_before_window():
    chunks = [
        _chunk("c1", "d1", 1, page_numbers=[1]),
        _chunk("c2", "d1", 2, page_numbers=[2]),
        _chunk("c3", "d1", 3, page_numbers=[2], figure_caption=True),
        _chunk("c4", "d1", 4, page_numbers=[2]),
        _chunk("c5", "d1", 5, page_numbers=[2]),
        _chunk("c6", "d1", 6, page_numbers=[3]),
    ]
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=4)
    caption_parent = next(
        parent
        for parent in parents
        if parent["parent_type"] == "caption_context" and parent["anchor_chunk_id"] == "c3"
    )
    assert caption_parent["child_chunk_ids"] == ["c2", "c3", "c4", "c5"]


def test_parent_store_medium_interfaces(tmp_path: Path):
    chunks = [
        _chunk("c1", "d1", 1, section="Intro", section_path=["Intro", "Background"], page_numbers=[1]),
        _chunk("c2", "d1", 2, section="Intro", section_path=["Intro", "Background"], page_numbers=[1, 2], figure_caption=True),
        _chunk("c3", "d1", 3, section="Results", section_path=[], page_numbers=[2]),
    ]
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=5)
    chunk_path = tmp_path / "chunks.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    _write_jsonl(chunk_path, chunks)
    _write_jsonl(parent_path, parents)
    store = ParentStore.from_jsonl(parent_path, chunk_jsonl_path=chunk_path)

    assert store.get_parents_by_type("page")
    assert store.get_parents_for_doc("d1", "section_path")
    assert store.get_page_parent("d1", 2) is not None
    assert store.get_section_path_parents("d1")
    assert store.get_evidence_parents("d1", "figure")

    caption_expanded = store.expand_caption_context("c2")
    page_expanded = store.expand_page_context("c2")
    section_expanded = store.expand_section_path_context("c1")

    assert [chunk.chunk_id for chunk in caption_expanded] == ["c1", "c2", "c3"]
    assert [chunk.chunk_id for chunk in page_expanded] == ["c1", "c2"]
    assert [chunk.chunk_id for chunk in section_expanded] == ["c1", "c2"]


def test_empty_field_fallbacks():
    chunks = [_chunk("c1", "d1", 1, section="Methods", section_path=[], evidence_types=[])]
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=5)
    section_path_parent = next(parent for parent in parents if parent["parent_type"] == "section_path")
    evidence_parent = next(parent for parent in parents if parent["parent_type"] == "evidence_type_context")
    assert section_path_parent["section_path_key"] == "Methods"
    assert evidence_parent["evidence_type"] == "method"


def test_parent_id_stability_medium():
    assert make_parent_id("section_path", "doc_x", "doc_x::section_path::Intro > Background") == make_parent_id(
        "section_path", "doc_x", "doc_x::section_path::Intro > Background"
    )
