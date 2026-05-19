from __future__ import annotations

from src.synbio_rag.application.retrieval_query_planner import QueryVariant
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever


def _chunk(chunk_id: str, *, vector_score: float = 0.0, bm25_score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        source_file=f"doc_{chunk_id}.pdf",
        title="Title",
        section="Results",
        text="body",
        vector_score=vector_score,
        bm25_score=bm25_score,
        metadata={},
    )


class FakePlanner:
    def build(self, question, analysis, config, decomposition_query=None):
        return [QueryVariant(query="planned query", weight=1.0, kind="original")]

    def contains_cjk(self, text):
        return False


class FakeAliasPolicy:
    def expand(self, query, config):
        return f"{query} alias"


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def search(self, query, limit, filters=None):
        self.queries.append(query)
        return self.chunks[:limit]


class PassthroughPostProcessor:
    def apply_comparison_diversity(self, chunks, limit, analysis, config):
        return chunks[:limit]

    def apply_title_keyword_boost(self, chunks, question, config):
        return chunks

    def apply_structure_marker_boost(self, chunks, question, config):
        return chunks

    def apply_same_doc_body_expansion(
        self,
        *,
        diversified,
        dense_results,
        bm25_results,
        config,
        question="",
        bm25_retriever=None,
    ):
        return diversified

    def apply_source_floor(self, *, expanded, dense_results, sparse_results, config):
        return expanded


def test_hybrid_uses_injected_application_policies() -> None:
    config = RetrievalConfig()
    config.alias_expansion_enabled = True
    dense = FakeRetriever([_chunk("dense", vector_score=0.9)])
    bm25 = FakeRetriever([_chunk("bm25", bm25_score=9.0)])
    retriever = HybridRetriever(
        config=config,
        dense_retriever=dense,
        bm25_retriever=bm25,
        query_planner=FakePlanner(),
        alias_policy=FakeAliasPolicy(),
        postprocessor=PassthroughPostProcessor(),
    )

    result = retriever.search("raw question", limit=5)

    assert result
    assert dense.queries == ["planned query"]
    assert bm25.queries == ["planned query alias"]
    assert retriever.last_debug["query_variants"][0]["query"] == "planned query"
    assert "dense_hits" in retriever.last_debug
    assert "bm25_hits" in retriever.last_debug
    assert "rrf_hits" in retriever.last_debug
