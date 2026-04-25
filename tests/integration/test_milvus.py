from __future__ import annotations

from pathlib import Path

import pytest
from pymilvus import MilvusClient

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder


@pytest.mark.integration
def test_milvus_roundtrip(tmp_path: Path) -> None:
    settings = Settings.from_env()
    model_path = Path(settings.kb.embedding_model_path)
    if not model_path.exists():
        pytest.skip(f"BGE model not found: {model_path}")

    embedder = BGEM3Embedder(str(model_path), dim=settings.kb.embedding_dim)
    texts = [
        "synthetic biology gene circuit design",
        "metabolic engineering for amino acid production",
        "CRISPR gene editing system in bacteria",
    ]
    embeddings = embedder.encode(texts)

    client = MilvusClient(str(tmp_path / "milvus_demo.db"))
    collection_name = "bge_test"
    client.create_collection(collection_name=collection_name, dimension=len(embeddings[0]))
    client.insert(
        collection_name=collection_name,
        data=[
            {
                "id": idx,
                "vector": vector,
                "text": text,
                "doc_type": "paper",
            }
            for idx, (text, vector) in enumerate(zip(texts, embeddings))
        ],
    )

    query_vector = embedder.encode(["gene regulation in bacteria"])[0]
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=2,
        output_fields=["text"],
    )

    assert results
    returned_texts = [hit["entity"]["text"] for hit in results[0]]
    assert any("bacteria" in text.lower() for text in returned_texts)
