from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.synbio_rag.application import pipeline as pipeline_mod
from src.synbio_rag.application.pipeline import (
    SynBioRAGPipeline,
    _build_query_rewrite_llm_client,
    _run_original_cn_fallback,
)
from src.synbio_rag.domain.config import RetrievalConfig, Settings
from src.synbio_rag.rewrite.query_rewrite_service import (
    QueryRewriteMode,
    QueryRewriteService,
    get_prompt_hash,
)


def _patch_pipeline_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class DenseRetriever(Dummy):
        client = object()

    class NeighborExpander(Dummy):
        _by_id = {}
        _positions = {}
        _doc_chunks = {}

        def _ensure_loaded(self):
            pass

    monkeypatch.setattr(pipeline_mod, "BGEM3Embedder", Dummy)
    monkeypatch.setattr(pipeline_mod, "QueryRouter", Dummy)
    monkeypatch.setattr(pipeline_mod, "MilvusRetriever", DenseRetriever)
    monkeypatch.setattr(pipeline_mod, "BM25Retriever", Dummy)
    monkeypatch.setattr(pipeline_mod, "HybridRetriever", Dummy)
    monkeypatch.setattr(pipeline_mod, "QwenReranker", Dummy)
    monkeypatch.setattr(pipeline_mod, "ChunkNeighborExpander", NeighborExpander)
    monkeypatch.setattr(pipeline_mod, "GenerationV2Service", Dummy)
    monkeypatch.setattr(pipeline_mod, "ParentContextExpander", Dummy)
    monkeypatch.setattr(pipeline_mod, "ConfidenceScorer", Dummy)


def _settings_with_rewrite(mode: str = "enabled") -> Settings:
    settings = Settings()
    settings.query_rewrite.mode = mode
    settings.llm.api_base = "https://qwen.example.invalid/compatible-mode/v1"
    settings.llm.api_key = "test-key"
    settings.generation.v2_use_qwen_synthesis = False
    settings.retrieval.parent_index_path = "/tmp/nonexistent-parent-index.jsonl"
    return settings


def test_query_rewrite_enabled_requires_llm_client_in_eval():
    settings = Settings()
    settings.query_rewrite.mode = "enabled"
    settings.query_rewrite.require_llm_for_eval = True
    settings.llm.api_base = ""
    settings.llm.api_key = ""

    with pytest.raises(RuntimeError, match="query_rewrite_llm_client_unavailable"):
        _build_query_rewrite_llm_client(settings)


def test_query_rewrite_enabled_injects_llm_client_when_available(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)
    pipeline = SynBioRAGPipeline(_settings_with_rewrite("enabled"))

    assert pipeline._rewrite_svc.llm_client is not None


def test_qwen_synthesis_false_does_not_disable_query_rewrite_client(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)
    settings = _settings_with_rewrite("enabled")
    settings.generation.v2_use_qwen_synthesis = False

    pipeline = SynBioRAGPipeline(settings)

    assert pipeline.settings.generation.v2_use_qwen_synthesis is False
    assert pipeline._rewrite_svc.llm_client is not None


def test_query_rewrite_off_does_not_require_llm_client(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)
    settings = _settings_with_rewrite("off")
    settings.query_rewrite.require_llm_for_eval = True
    settings.llm.api_base = ""
    settings.llm.api_key = ""

    pipeline = SynBioRAGPipeline(settings)

    assert pipeline._rewrite_svc.llm_client is None


def test_original_cn_fallback_can_trigger_after_successful_rewrite():
    config = RetrievalConfig()
    config.original_cn_fallback_enabled = True
    retrieved = [
        SimpleNamespace(chunk_id="seed", doc_id="doc_seed", metadata={}),
    ]
    fallback_chunk = SimpleNamespace(chunk_id="cn", doc_id="doc_cn", metadata={})

    class Pipeline:
        def _search_with_filter_fallback(self, **kwargs):
            return [fallback_chunk], {"selected": "original"}

    debug = _run_original_cn_fallback(
        question="这项研究优化了什么代谢瓶颈？",
        retrieval_question="What metabolic bottleneck did this study optimize?",
        rewrite_trace=SimpleNamespace(query_rewrite_mode="enabled"),
        retrieved=retrieved,
        analysis=SimpleNamespace(),
        filters=None,
        config=config,
        pipeline=Pipeline(),
    )

    assert debug["triggered"] is True
    assert debug["fallback_added_count"] == 1
    assert fallback_chunk.metadata["query_branch"] == "original_cn_fallback"


def test_query_rewrite_prompt_unchanged():
    assert get_prompt_hash() == "cd6b9faeb6a193c4"


def test_query_rewrite_service_uses_openai_compatible_client():
    class FakeClient:
        def chat_completion(self, **kwargs):
            return "What metabolic bottleneck did this study optimize?"

    svc = QueryRewriteService(
        mode=QueryRewriteMode.ENABLED,
        cache_enabled=False,
        llm_client=FakeClient(),
    )

    rewritten, trace = svc.rewrite("这项研究优化了什么代谢瓶颈？")

    assert rewritten == "What metabolic bottleneck did this study optimize?"
    assert trace.rewrite_llm_client_available is True
    assert trace.rewrite_fallback_used is False
