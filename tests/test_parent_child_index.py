from __future__ import annotations

import json
from pathlib import Path

from scripts.ingestion.build_parent_child_index import build_parent_child_records
from src.synbio_rag.application.rerank_common import _rerank_text
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.index.parent_store import ParentStore


def _chunk(chunk_id: str = "doc_1_sec01_chunk01") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc_1",
        "source_file": "doc_1.pdf",
        "title": "Synthetic biology paper",
        "section": "Results",
        "section_path": ["Results"],
        "chunk_index": 1,
        "token_count": 12,
        "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        "retrieval_text": "title: Synthetic biology paper\n\nalpha beta gamma",
        "quality_score": 1.0,
        "page_start": 2,
        "page_end": 2,
        "page_numbers": [2],
        "block_types": ["paragraph"],
        "block_ids": ["p2_b0001"],
        "source_block_ids": ["p2_b0001"],
        "evidence_types": ["paragraph"],
        "contains_table_caption": False,
        "contains_figure_caption": False,
        "contains_table_text": False,
        "contains_image": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_parent_child_records_splits_children_and_keeps_context_parent() -> None:
    parent_chunks, child_chunks, parent_index = build_parent_child_records(
        [_chunk()],
        child_size=5,
        child_overlap=1,
    )

    assert [item["index_role"] for item in parent_chunks] == ["parent"]
    assert len(child_chunks) == 3
    assert child_chunks[0]["chunk_id"] == "doc_1_sec01_chunk01::child001"
    assert child_chunks[0]["parent_chunk_id"] == "doc_1_sec01_chunk01"
    assert child_chunks[0]["child_start_token"] == 0
    assert child_chunks[1]["child_start_token"] == 4
    assert child_chunks[0]["retrieval_text"].startswith("title: Synthetic biology paper")
    assert any(parent["parent_type"] == "retrieval_parent" for parent in parent_index)
    assert any(parent["parent_type"] == "chunk_window" for parent in parent_index)


def test_structural_split_respects_paragraph_boundaries() -> None:
    chunk = _chunk()
    chunk["text"] = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota"
    chunk["source_block_metadata"] = [
        {"block_id": "b1", "source_block_id": "b1", "type": "paragraph", "page": 2},
        {"block_id": "b2", "source_block_id": "b2", "type": "paragraph", "page": 2},
        {"block_id": "b3", "source_block_id": "b3", "type": "paragraph", "page": 3},
    ]

    _parents, child_chunks, _index = build_parent_child_records(
        [chunk],
        child_size=7,
        child_overlap=1,
    )

    assert len(child_chunks) == 2
    assert child_chunks[0]["text"] == "alpha beta gamma\n\ndelta epsilon zeta"
    assert child_chunks[0]["source_block_ids"] == ["b1", "b2"]
    assert child_chunks[0]["child_split_strategy"] == "structure_block"
    assert child_chunks[1]["text"] == "eta theta iota"
    assert child_chunks[1]["page_numbers"] == [3]


def test_structural_split_keeps_table_caption_with_table_text() -> None:
    chunk = _chunk()
    chunk["text"] = "Table 1. Strains used\n\nstrain plasmid source value"
    chunk["source_block_metadata"] = [
        {"block_id": "cap", "source_block_id": "cap", "type": "table_caption", "page": 4},
        {"block_id": "tbl", "source_block_id": "tbl", "type": "table_text", "page": 4},
    ]

    _parents, child_chunks, _index = build_parent_child_records(
        [chunk],
        child_size=5,
        child_overlap=1,
    )

    assert len(child_chunks) == 1
    assert child_chunks[0]["contains_table_caption"] is True
    assert child_chunks[0]["contains_table_text"] is True
    assert child_chunks[0]["block_types"] == ["table_caption", "table_text"]
    assert child_chunks[0]["child_split_strategy"] == "evidence_block"


def test_structural_split_keeps_heading_with_following_body() -> None:
    chunk = _chunk()
    chunk["text"] = "## Results\n\nalpha beta gamma"
    chunk["source_block_metadata"] = [
        {"block_id": "h", "source_block_id": "h", "type": "section_heading", "page": 5},
        {"block_id": "p", "source_block_id": "p", "type": "paragraph", "page": 5},
    ]

    _parents, child_chunks, _index = build_parent_child_records(
        [chunk],
        child_size=10,
        child_overlap=1,
    )

    assert len(child_chunks) == 1
    assert child_chunks[0]["text"] == "## Results\n\nalpha beta gamma"
    assert child_chunks[0]["block_types"] == ["section_heading", "paragraph"]


def test_structural_split_keeps_unmatched_overlap_from_consuming_block_metadata() -> None:
    chunk = _chunk()
    chunk["text"] = "prior overlap text\n\n## Methods\n\nmethod body evidence"
    chunk["source_block_metadata"] = [
        {"block_id": "h", "source_block_id": "h", "type": "section_heading", "page": 5, "text_preview": "## Methods"},
        {"block_id": "p", "source_block_id": "p", "type": "paragraph", "page": 5, "text_preview": "method body evidence"},
    ]

    _parents, child_chunks, _index = build_parent_child_records(
        [chunk],
        child_size=20,
        child_overlap=1,
    )

    assert len(child_chunks) == 1
    assert child_chunks[0]["block_types"] == ["paragraph", "section_heading"]
    assert child_chunks[0]["source_block_ids"] == ["h", "p"]
    assert child_chunks[0]["source_block_metadata"][0]["type"] == "paragraph"
    assert child_chunks[0]["source_block_metadata"][1]["type"] == "section_heading"


def test_structural_split_treats_oversized_heading_metadata_as_body_window() -> None:
    chunk = _chunk()
    chunk["text"] = " ".join(["#"] + [f"word{i}" for i in range(90)])
    chunk["source_block_metadata"] = [
        {
            "block_id": "bad_heading",
            "source_block_id": "bad_heading",
            "type": "title",
            "page": 6,
            "text_preview": "# word0 word1",
        }
    ]

    _parents, child_chunks, _index = build_parent_child_records(
        [chunk],
        child_size=5,
        child_overlap=1,
    )

    assert len(child_chunks) > 1
    assert all(child["block_types"] == ["paragraph"] for child in child_chunks)
    assert all(child["child_split_strategy"] == "long_block_window" for child in child_chunks)


def test_parent_store_materializes_child_hit_to_parent_chunk(tmp_path: Path) -> None:
    parent_chunks, child_chunks, parent_index = build_parent_child_records(
        [_chunk()],
        child_size=5,
        child_overlap=1,
    )
    parent_path = tmp_path / "parent_chunks.jsonl"
    child_path = tmp_path / "child_chunks.jsonl"
    index_path = tmp_path / "parent_index.jsonl"
    _write_jsonl(parent_path, parent_chunks)
    _write_jsonl(child_path, child_chunks)
    _write_jsonl(index_path, parent_index)

    store = ParentStore.from_jsonl(
        index_path,
        chunk_jsonl_path=child_path,
        parent_chunk_jsonl_path=parent_path,
    )
    child = RetrievedChunk(
        chunk_id=child_chunks[1]["chunk_id"],
        doc_id="doc_1",
        source_file="doc_1.pdf",
        title="Synthetic biology paper",
        section="Results",
        text=child_chunks[1]["text"],
        vector_score=0.7,
        bm25_score=3.0,
        fusion_score=0.2,
        metadata={
            "parent_chunk_id": "doc_1_sec01_chunk01",
            "child_index": 2,
            "child_start_token": 4,
            "child_end_token": 9,
        },
    )

    materialized = store.materialize_parent_hits([child])

    assert len(materialized) == 1
    assert materialized[0].chunk_id == "doc_1_sec01_chunk01"
    assert materialized[0].text == parent_chunks[0]["text"]
    assert materialized[0].vector_score == 0.7
    assert materialized[0].metadata["parent_child_materialized"] is True
    assert materialized[0].metadata["matched_child_chunk_ids"] == [child.chunk_id]
    assert materialized[0].metadata["matched_child_snippets"][0]["chunk_id"] == child.chunk_id
    assert materialized[0].metadata["matched_child_snippets"][0]["text"] == child.text


def test_parent_store_collapses_multiple_child_hits_to_one_parent(tmp_path: Path) -> None:
    parent_chunks, child_chunks, parent_index = build_parent_child_records(
        [_chunk()],
        child_size=5,
        child_overlap=1,
    )
    parent_path = tmp_path / "parent_chunks.jsonl"
    child_path = tmp_path / "child_chunks.jsonl"
    index_path = tmp_path / "parent_index.jsonl"
    _write_jsonl(parent_path, parent_chunks)
    _write_jsonl(child_path, child_chunks)
    _write_jsonl(index_path, parent_index)
    store = ParentStore.from_jsonl(index_path, chunk_jsonl_path=child_path, parent_chunk_jsonl_path=parent_path)

    hits = [
        RetrievedChunk(
            chunk_id=child_chunks[0]["chunk_id"],
            doc_id="doc_1",
            source_file="doc_1.pdf",
            title="Synthetic biology paper",
            section="Results",
            text=child_chunks[0]["text"],
            bm25_score=1.0,
            metadata={"parent_chunk_id": "doc_1_sec01_chunk01"},
        ),
        RetrievedChunk(
            chunk_id=child_chunks[1]["chunk_id"],
            doc_id="doc_1",
            source_file="doc_1.pdf",
            title="Synthetic biology paper",
            section="Results",
            text=child_chunks[1]["text"],
            bm25_score=2.0,
            metadata={"parent_chunk_id": "doc_1_sec01_chunk01"},
        ),
    ]

    materialized = store.materialize_parent_hits(hits)

    assert [chunk.chunk_id for chunk in materialized] == ["doc_1_sec01_chunk01"]
    assert materialized[0].bm25_score == 2.0
    assert materialized[0].metadata["matched_child_chunk_ids"] == [
        child_chunks[0]["chunk_id"],
        child_chunks[1]["chunk_id"],
    ]
    assert [item["chunk_id"] for item in materialized[0].metadata["matched_child_snippets"]] == [
        child_chunks[0]["chunk_id"],
        child_chunks[1]["chunk_id"],
    ]


def test_rerank_text_uses_matched_child_snippets_before_parent_text() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc_1_sec01_chunk01",
        doc_id="doc_1",
        source_file="doc_1.pdf",
        title="Synthetic biology paper",
        section="Results",
        text=" ".join(f"parent{i}" for i in range(200)),
        metadata={
            "matched_child_snippets": [
                {
                    "chunk_id": "doc_1_sec01_chunk01::child002",
                    "text": "specific table evidence with 12 mg/L product titer",
                    "child_start_token": 80,
                    "child_end_token": 88,
                    "block_types": ["table_text"],
                    "evidence_types": ["table_text"],
                    "contains_table_text": True,
                }
            ]
        },
    )

    text = _rerank_text(chunk)

    assert "matched_child_evidence:" in text
    assert "doc_1_sec01_chunk01::child002" in text
    assert "[table text]" in text
    assert "specific table evidence with 12 mg/L product titer" in text
    assert "parent_context:" in text
    assert "parent0 parent1 parent2" not in text
