from __future__ import annotations

from src.synbio_rag.application.pipeline import _supplement_summary_sections
from src.synbio_rag.domain.schemas import RetrievedChunk


class FakeMilvusClient:
    def query(self, *, collection_name, filter, output_fields, limit):
        assert collection_name == "test_collection"
        assert "chunk_index" in output_fields
        if 'section == "Abstract"' not in filter:
            return []
        return [
            {
                "chunk_id": "doc_a_abstract",
                "doc_id": "doc_a",
                "source_file": "doc_a.pdf",
                "title": "Doc A",
                "section": "Abstract",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 7,
                "text": "This abstract has enough text to pass the summary supplement filter.",
            }
        ]


def test_supplement_summary_sections_stores_chunk_index_in_metadata():
    seed = RetrievedChunk(
        chunk_id="doc_a_body",
        doc_id="doc_a",
        source_file="doc_a.pdf",
        title="Doc A",
        section="Results",
        text="Seed body evidence with enough content.",
    )

    chunks, debug = _supplement_summary_sections(
        question="Summarize Doc A",
        seed_chunks=[seed],
        milvus_client=FakeMilvusClient(),
        collection_name="test_collection",
    )

    assert debug["used"] is True
    assert [chunk.chunk_id for chunk in chunks] == ["doc_a_body", "doc_a_abstract"]
    assert chunks[1].metadata["chunk_index"] == 7
