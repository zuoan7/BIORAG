#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


DEFAULT_QUERIES = [
    {
        "question": "CRISPR gene editing in bacteria",
        "expected_terms": ["crispr", "bacteria"],
    },
    {
        "question": "metabolic engineering for amino acid production",
        "expected_terms": ["metabolic", "amino", "production"],
    },
    {
        "question": "gene circuit design in synthetic biology",
        "expected_terms": ["circuit", "synthetic", "biology"],
    },
]


def overlap_score(text: str, expected_terms: list[str]) -> int:
    lower = text.lower()
    return sum(1 for term in expected_terms if term.lower() in lower)


def main() -> None:
    settings = Settings.from_env()
    embedder = BGEM3Embedder(
        model_path=settings.kb.embedding_model_path,
        dim=settings.kb.embedding_dim,
    )
    dense = MilvusRetriever(settings.retrieval, embedder)
    bm25 = BM25Retriever(settings.retrieval, settings.kb, milvus_client=dense.client)
    hybrid = HybridRetriever(settings.retrieval, dense, bm25)
    collections = set(dense.client.list_collections())
    if settings.retrieval.collection_name not in collections:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": f"collection {settings.retrieval.collection_name} not found",
                    "hint": "run python scripts/ingestion/build_round1_kb.py first",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    report = []
    for item in DEFAULT_QUERIES:
        question = item["question"]
        expected_terms = item["expected_terms"]
        dense_hits = dense.search(question, limit=5, filters=QueryFilters())
        hybrid_hits = hybrid.search(question, limit=5, filters=QueryFilters())
        dense_best = max((overlap_score(hit.text, expected_terms) for hit in dense_hits), default=0)
        hybrid_best = max((overlap_score(hit.text, expected_terms) for hit in hybrid_hits), default=0)
        report.append(
            {
                "question": question,
                "dense_best_overlap": dense_best,
                "hybrid_best_overlap": hybrid_best,
                "dense_top_chunk_ids": [hit.chunk_id for hit in dense_hits[:3]],
                "hybrid_top_chunk_ids": [hit.chunk_id for hit in hybrid_hits[:3]],
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
