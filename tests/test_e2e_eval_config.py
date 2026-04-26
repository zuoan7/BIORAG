from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluation.evaluate_e2e_small import (
    CONFIG_SCAN_PATHS,
    assert_expected_guarded_mode,
    resolve_effective_rerank_mode,
    scan_config_variable_names,
)
from src.synbio_rag.domain.config import Settings


def test_rerank_mode_env_reads_guarded_rank1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIORAG_RERANK_MODE", "guarded_rank1")
    monkeypatch.delenv("RETRIEVAL_RERANK_MODE", raising=False)

    assert resolve_effective_rerank_mode() == "guarded_rank1"
    assert assert_expected_guarded_mode("guarded_rank1") == "guarded_rank1"


def test_biorag_rerank_mode_overrides_retrieval_rerank_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIORAG_RERANK_MODE", "guarded_rank1")
    monkeypatch.setenv("RETRIEVAL_RERANK_MODE", "plain")

    settings = Settings.from_env()
    assert settings.retrieval.rerank_mode == "guarded_rank1"


def test_typo_variable_does_not_override_rank1_gap(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP", raising=False)
    monkeypatch.setenv("RETRIEVALANK1_MAX_SCORE_GAP", "0.91")

    settings = Settings.from_env()
    assert settings.retrieval.guarded_rank1_max_score_gap == pytest.approx(0.20)


def test_proper_rank1_gap_variable_is_honored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP", "0.37")

    settings = Settings.from_env()
    assert settings.retrieval.guarded_rank1_max_score_gap == pytest.approx(0.37)


def test_guarded_mode_assertion_fails_fast_for_plain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIORAG_RERANK_MODE", "plain")

    with pytest.raises(AssertionError, match="effective rerank mode is 'plain'"):
        assert_expected_guarded_mode("guarded_rank1")


def test_config_scan_finds_canonical_rank1_vars():
    result = scan_config_variable_names([Path(path) for path in CONFIG_SCAN_PATHS])
    assert "BIORAG_RERANK_MODE" in result.rerank_mode_vars
    assert "RETRIEVAL_RERANK_MODE" in result.rerank_mode_vars
    assert "RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP" in result.rank1_guard_vars
    assert "RETRIEVAL_GUARDED_RANK1_MIN_COMPLETENESS_GAIN" in result.rank1_guard_vars
    assert "RETRIEVALANK1_MAX_SCORE_GAP" not in result.typo_vars
