"""Tests for GENERATION_V2_PROFILE and neighbor promotion hard guard."""
from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from src.synbio_rag.domain.config import (
    GenerationConfig,
    _GENERATION_V2_PROFILES,
    _apply_profile,
    _enforce_forbidden_flags,
)


# ---------------------------------------------------------------------------
# _apply_profile unit tests
# ---------------------------------------------------------------------------

class TestApplyProfile:
    def _fresh(self) -> GenerationConfig:
        g = GenerationConfig()
        return g

    def test_stable_profile(self):
        g = self._fresh()
        _apply_profile(g, "stable")
        assert g.v2_use_qwen_synthesis is False
        assert g.v2_enable_comparison_coverage is False
        assert g.v2_enable_neighbor_audit is False

    def test_qwen_profile(self):
        g = self._fresh()
        _apply_profile(g, "qwen")
        assert g.v2_use_qwen_synthesis is True
        assert g.v2_enable_comparison_coverage is False
        assert g.v2_enable_neighbor_audit is False

    def test_comparison_profile(self):
        g = self._fresh()
        _apply_profile(g, "comparison")
        assert g.v2_use_qwen_synthesis is True
        assert g.v2_enable_comparison_coverage is True
        assert g.v2_enable_neighbor_audit is False

    def test_debug_profile(self):
        g = self._fresh()
        _apply_profile(g, "debug")
        assert g.v2_use_qwen_synthesis is True
        assert g.v2_enable_comparison_coverage is True
        assert g.v2_enable_neighbor_audit is True

    def test_profile_never_sets_neighbor_promotion(self):
        for profile in _GENERATION_V2_PROFILES:
            g = self._fresh()
            _apply_profile(g, profile)
            assert g.v2_enable_neighbor_promotion is False, f"profile={profile}"

    def test_profile_never_sets_neighbor_context_in_qwen(self):
        for profile in _GENERATION_V2_PROFILES:
            g = self._fresh()
            _apply_profile(g, profile)
            assert g.v2_include_neighbor_context_in_qwen is False, f"profile={profile}"


# ---------------------------------------------------------------------------
# _enforce_forbidden_flags unit tests
# ---------------------------------------------------------------------------

class TestEnforceForbiddenFlags:
    def test_neighbor_promotion_forced_false(self):
        g = GenerationConfig()
        g.v2_enable_neighbor_promotion = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _enforce_forbidden_flags(g)
        assert g.v2_enable_neighbor_promotion is False
        assert any("v2_enable_neighbor_promotion" in str(warning.message) for warning in w)

    def test_include_neighbor_context_in_qwen_forced_false(self):
        g = GenerationConfig()
        g.v2_include_neighbor_context_in_qwen = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _enforce_forbidden_flags(g)
        assert g.v2_include_neighbor_context_in_qwen is False
        assert any("v2_include_neighbor_context_in_qwen" in str(warning.message) for warning in w)

    def test_no_warning_when_flags_already_false(self):
        g = GenerationConfig()
        g.v2_enable_neighbor_promotion = False
        g.v2_include_neighbor_context_in_qwen = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _enforce_forbidden_flags(g)
        forbidden_warnings = [
            x for x in w
            if "hard guard" in str(x.message)
        ]
        assert len(forbidden_warnings) == 0


# ---------------------------------------------------------------------------
# Settings.from_env integration tests (env variable override)
# ---------------------------------------------------------------------------

class TestFromEnvProfileIntegration:
    """Test that profile + explicit env override works correctly."""

    def _build_minimal_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        base = {
            "GENERATION_VERSION": "v2",
            "GENERATION_V2_PROFILE": "stable",
        }
        if extra:
            base.update(extra)
        return base

    def _from_env_with(self, env: dict[str, str]) -> GenerationConfig:
        from src.synbio_rag.domain.config import Settings
        with patch.dict(os.environ, env, clear=False):
            s = Settings.from_env()
        return s.generation

    def test_stable_profile_defaults(self):
        gen = self._from_env_with(self._build_minimal_env())
        assert gen.v2_use_qwen_synthesis is False
        assert gen.v2_enable_comparison_coverage is False
        assert gen.v2_enable_neighbor_audit is False
        assert gen.v2_enable_neighbor_promotion is False

    def test_qwen_profile_sets_synthesis(self):
        gen = self._from_env_with(self._build_minimal_env({"GENERATION_V2_PROFILE": "qwen"}))
        assert gen.v2_use_qwen_synthesis is True
        assert gen.v2_enable_comparison_coverage is False

    def test_comparison_profile_sets_both(self):
        gen = self._from_env_with(self._build_minimal_env({"GENERATION_V2_PROFILE": "comparison"}))
        assert gen.v2_use_qwen_synthesis is True
        assert gen.v2_enable_comparison_coverage is True
        assert gen.v2_enable_neighbor_audit is False

    def test_debug_profile_sets_audit(self):
        gen = self._from_env_with(self._build_minimal_env({"GENERATION_V2_PROFILE": "debug"}))
        assert gen.v2_enable_neighbor_audit is True
        assert gen.v2_enable_neighbor_promotion is False

    def test_explicit_env_overrides_profile(self):
        """Explicit GENERATION_V2_USE_QWEN_SYNTHESIS=false overrides qwen profile."""
        env = self._build_minimal_env({
            "GENERATION_V2_PROFILE": "qwen",
            "GENERATION_V2_USE_QWEN_SYNTHESIS": "false",
        })
        gen = self._from_env_with(env)
        assert gen.v2_use_qwen_synthesis is False

    def test_explicit_env_cannot_enable_neighbor_promotion(self):
        """Hard guard prevents GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION=true from taking effect."""
        env = self._build_minimal_env({
            "GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION": "true",
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = self._from_env_with(env)
        assert gen.v2_enable_neighbor_promotion is False
        # hard guard warning should have been emitted
        assert any("hard guard" in str(warning.message) for warning in w)

    def test_explicit_env_cannot_enable_neighbor_context_in_qwen(self):
        env = self._build_minimal_env({
            "GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN": "true",
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = self._from_env_with(env)
        assert gen.v2_include_neighbor_context_in_qwen is False
        assert any("hard guard" in str(warning.message) for warning in w)

    def test_unknown_profile_falls_back_to_stable(self):
        env = self._build_minimal_env({"GENERATION_V2_PROFILE": "nonexistent_profile"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = self._from_env_with(env)
        assert gen.v2_profile == "stable"
        assert gen.v2_use_qwen_synthesis is False
        assert any("Unknown" in str(warning.message) for warning in w)

    def test_default_profile_is_stable_when_env_not_set(self):
        env = {"GENERATION_VERSION": "v2"}
        gen = self._from_env_with(env)
        assert gen.v2_profile == "stable"
        assert gen.v2_use_qwen_synthesis is False
