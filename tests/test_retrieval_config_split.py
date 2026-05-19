from __future__ import annotations

import os
import warnings
from dataclasses import asdict, fields
from unittest.mock import patch

import pytest

from src.synbio_rag.domain.config import (
    RetrievalConfig,
    RetrievalCoreConfig,
    Settings,
)


def _from_env_with(env: dict[str, str]):
    from src.synbio_rag.domain import config as config_mod

    with patch.dict(os.environ, env, clear=True), patch.object(
        config_mod, "dotenv_values", return_value={}
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Settings.from_env()


def test_retrieval_config_exposes_grouped_config_objects() -> None:
    config = RetrievalConfig()

    assert set(asdict(config)) == {
        "core",
        "comparison",
        "rerank",
        "evidence_boost",
        "same_doc",
        "alias_expansion",
        "context_expansion",
        "original_cn_fallback",
        "table_preview",
        "index_contract",
    }
    assert {field.name for field in fields(RetrievalConfig)} == set(asdict(config))


def test_legacy_attribute_reads_proxy_to_group_values() -> None:
    config = RetrievalConfig()

    assert config.final_top_k == config.core.final_top_k == 8
    assert config.rerank_top_k == config.rerank.rerank_top_k == 10
    assert config.parent_expansion_enabled is config.context_expansion.parent_expansion_enabled
    assert config.table_preview_enabled is config.table_preview.table_preview_enabled
    assert config.index_schema_version == config.index_contract.index_schema_version == "v2"


def test_legacy_constructor_kwargs_update_group_values() -> None:
    config = RetrievalConfig(
        final_top_k=2,
        rerank_top_k=7,
        parent_expansion_enabled=False,
        table_preview_units_path="/tmp/table-units.jsonl",
        index_type="HNSW",
        hnsw_ef=128,
    )

    assert config.core.final_top_k == 2
    assert config.rerank.rerank_top_k == 7
    assert config.context_expansion.parent_expansion_enabled is False
    assert config.table_preview.table_preview_units_path == "/tmp/table-units.jsonl"
    assert config.index_contract.index_type == "HNSW"
    assert config.index_contract.hnsw_ef == 128


def test_legacy_attribute_writes_update_group_values() -> None:
    config = RetrievalConfig()

    config.milvus_uri = "/tmp/papers.db"
    config.parent_expansion_enabled = False
    config.table_preview_merge_enabled = False
    config.nprobe = 32

    assert config.core.milvus_uri == "/tmp/papers.db"
    assert config.context_expansion.parent_expansion_enabled is False
    assert config.table_preview.table_preview_merge_enabled is False
    assert config.index_contract.nprobe == 32


def test_group_constructor_is_supported() -> None:
    config = RetrievalConfig(core=RetrievalCoreConfig(final_top_k=3))

    assert config.final_top_k == 3
    assert config.core.final_top_k == 3


def test_unknown_constructor_kwarg_still_fails() -> None:
    with pytest.raises(TypeError):
        RetrievalConfig(does_not_exist=True)


def test_settings_from_env_keeps_legacy_retrieval_overrides() -> None:
    settings = _from_env_with(
        {
            "RETRIEVAL_FINAL_TOP_K": "4",
            "RETRIEVAL_RERANK_TOP_K": "9",
            "RETRIEVAL_PARENT_EXPANSION_ENABLED": "false",
            "TABLE_PREVIEW_MERGE_ENABLED": "false",
            "RETRIEVAL_INDEX_TYPE": "HNSW",
            "RETRIEVAL_HNSW_EF": "256",
        }
    )

    assert settings.retrieval.final_top_k == 4
    assert settings.retrieval.core.final_top_k == 4
    assert settings.retrieval.rerank_top_k == 9
    assert settings.retrieval.rerank.rerank_top_k == 9
    assert settings.retrieval.parent_expansion_enabled is False
    assert settings.retrieval.context_expansion.parent_expansion_enabled is False
    assert settings.retrieval.table_preview_merge_enabled is False
    assert settings.retrieval.table_preview.table_preview_merge_enabled is False
    assert settings.retrieval.index_type == "HNSW"
    assert settings.retrieval.index_contract.hnsw_ef == 256
