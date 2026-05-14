from __future__ import annotations

from src.synbio_rag.domain.config import Settings, TableEnhancementConfig


def test_table_enhancement_defaults_off() -> None:
    config = TableEnhancementConfig()
    assert config.enabled is False
    assert config.mode == "conservative_caption_nearby"
    assert config.window_after_caption == 5
    assert config.window_before_caption == 1
    assert config.max_associated_blocks_per_caption == 5
    assert config.min_confidence == "low"
    assert config.write_audit is True
    assert config.fail_on_schema_drift is True
    assert config.output_suffix == "table_enhanced"
    assert config.audit_root.endswith("reports/table_enhancement")
    assert config.dry_run is False


def test_settings_from_env_table_enhancement_override(monkeypatch) -> None:
    monkeypatch.setenv("TABLE_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("TABLE_ENHANCEMENT_MODE", "conservative_caption_nearby")
    monkeypatch.setenv("TABLE_ENHANCEMENT_WINDOW_AFTER_CAPTION", "3")
    monkeypatch.setenv("TABLE_ENHANCEMENT_WINDOW_BEFORE_CAPTION", "2")
    monkeypatch.setenv("TABLE_ENHANCEMENT_MAX_ASSOCIATED_BLOCKS_PER_CAPTION", "4")
    monkeypatch.setenv("TABLE_ENHANCEMENT_MIN_CONFIDENCE", "medium")
    monkeypatch.setenv("TABLE_ENHANCEMENT_WRITE_AUDIT", "false")
    monkeypatch.setenv("TABLE_ENHANCEMENT_FAIL_ON_SCHEMA_DRIFT", "false")
    monkeypatch.setenv("TABLE_ENHANCEMENT_OUTPUT_SUFFIX", "te_test")
    monkeypatch.setenv("TABLE_ENHANCEMENT_AUDIT_ROOT", "reports/table_enhancement_env_test")
    monkeypatch.setenv("TABLE_ENHANCEMENT_DRY_RUN", "true")

    settings = Settings.from_env()

    assert settings.table_enhancement.enabled is True
    assert settings.table_enhancement.window_after_caption == 3
    assert settings.table_enhancement.window_before_caption == 2
    assert settings.table_enhancement.max_associated_blocks_per_caption == 4
    assert settings.table_enhancement.min_confidence == "medium"
    assert settings.table_enhancement.write_audit is False
    assert settings.table_enhancement.fail_on_schema_drift is False
    assert settings.table_enhancement.output_suffix == "te_test"
    assert settings.table_enhancement.audit_root.endswith("reports/table_enhancement_env_test")
    assert settings.table_enhancement.dry_run is True
