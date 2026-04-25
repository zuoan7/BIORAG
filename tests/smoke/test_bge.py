from __future__ import annotations

from pathlib import Path

import pytest

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder


@pytest.mark.smoke
def test_bge_embedding_smoke() -> None:
    settings = Settings.from_env()
    model_path = Path(settings.kb.embedding_model_path)
    if not model_path.exists():
        pytest.skip(f"BGE model not found: {model_path}")

    embedder = BGEM3Embedder(str(model_path), dim=settings.kb.embedding_dim)
    vectors = embedder.encode(
        [
            "CRISPR gene editing in bacteria",
            "metabolic pathway engineering in yeast",
        ]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == settings.kb.embedding_dim
