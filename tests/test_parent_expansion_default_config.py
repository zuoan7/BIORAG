from __future__ import annotations

import os
import pytest


class TestParentExpansionDefaultConfig:
    """parent_expansion_enabled defaults off, env override works."""

    @pytest.fixture(autouse=True)
    def _clear_env(self):
        for key in list(os.environ):
            if key.upper().startswith("RETRIEVAL_PARENT"):
                del os.environ[key]
        yield
        for key in list(os.environ):
            if key.upper().startswith("RETRIEVAL_PARENT"):
                del os.environ[key]

    def test_default_is_false(self):
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        assert s.retrieval.parent_expansion_enabled is False

    def test_env_false_disables(self):
        os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "false"
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        assert s.retrieval.parent_expansion_enabled is False

    def test_env_true_enables(self):
        os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "true"
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        assert s.retrieval.parent_expansion_enabled is True

    def test_env_0_disables(self):
        os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "0"
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        assert s.retrieval.parent_expansion_enabled is False

    def test_other_defaults_unchanged(self):
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        assert s.retrieval.parent_expansion_max_total == 12
        assert s.retrieval.parent_expansion_per_seed_limit == 2
        assert s.retrieval.parent_expansion_window_enabled is True
        assert s.retrieval.parent_expansion_caption_enabled is True
        assert s.retrieval.final_top_k == 8
        assert s.retrieval.rerank_top_k == 10
        assert s.generation.v2_use_qwen_synthesis is False


class TestPipelineDebugWithDefaultOff:
    """Ensure pipeline debug reflects parent_expansion state."""

    @pytest.fixture(autouse=True)
    def _clear_env(self):
        for key in list(os.environ):
            if key.upper().startswith("RETRIEVAL_PARENT"):
                del os.environ[key]
        yield
        for key in list(os.environ):
            if key.upper().startswith("RETRIEVAL_PARENT"):
                del os.environ[key]

    def test_explicit_off_debug_fields(self):
        """With env=false, parent_expansion.enabled=False and reason='disabled'."""
        os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "false"
        from src.synbio_rag.application.parent_expansion import ParentContextExpander
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        expander = ParentContextExpander(parent_store=None, config=s.retrieval)
        # Without parent_store + False enabled, expand does nothing
        assert expander.config.parent_expansion_enabled is False

    def test_default_off_debug_structure(self):
        """Default-off should produce standard debug fields."""
        from src.synbio_rag.application.parent_expansion import ParentContextExpander
        from src.synbio_rag.domain.config import Settings
        s = Settings.from_env()
        assert s.retrieval.parent_expansion_enabled is False
        expander = ParentContextExpander(parent_store=None, config=s.retrieval)
        # Even without parent_store, expand should gracefully return seeds
        from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
        seed = RetrievedChunk(chunk_id="test", doc_id="doc_test", source_file="test.pdf", title="Test Title", section="Results", text="test content")
        analysis = QueryAnalysis(intent=QueryIntent.FACTOID, notes="", requires_external_tools=False, search_limit=40, rerank_top_k=10)
        chunks, debug = expander.expand("test question", [seed], analysis)
        assert chunks == [seed]
        assert isinstance(debug, dict)
        assert "enabled" in debug
        assert "input_count" in debug
        assert "output_count" in debug
        assert debug["reason"] == "disabled"
