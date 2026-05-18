from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluation import phase7r_build_table_index_production_proposal as phase7r_build
from scripts.evaluation import phase7r_validate_table_index_production_proposal as phase7r_validate


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/v7_phase7_table_index_production_proposal"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_index_production_proposal"
RESULTS_DIR = ROOT / "results/v7_phase7_table_index_production_proposal"


@pytest.fixture(scope="module", autouse=True)
def _ensure_phase7r_artifacts():
    phase7r_build.main()
    summary = phase7r_validate.write_validation_artifacts()
    assert summary["validation_status"] == "pass_with_warnings"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_template_json_is_parseable():
    manifest = json.loads(
        (DATA_DIR / "production_index_artifact_manifest_template.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["manifest_kind"] == "production_table_index_artifact_manifest_template"
    assert manifest["template_only"] is True
    assert manifest["production_artifact_created"] is False
    assert set(manifest["artifacts"]) >= phase7r_validate.MANIFEST_REQUIRED_ARTIFACTS
    assert set(manifest["build_metadata"]) >= phase7r_validate.MANIFEST_REQUIRED_BUILD_FIELDS


def test_promotion_gate_matrix_covers_required_gates():
    rows = _load_csv(DATA_DIR / "promotion_gate_matrix.csv")
    gate_ids = {row["gate_id"] for row in rows}
    assert gate_ids >= phase7r_validate.REQUIRED_PROMOTION_GATES
    assert phase7r_validate.promotion_gate_matrix_covers_required_blockers()[0] is True
    assert phase7r_validate.current_preview_units_fail_promotion_gate()[0] is True


def test_rollback_matrix_covers_hard_disable():
    rows = _load_csv(DATA_DIR / "rollback_scenario_matrix.csv")
    hard_disable = next(
        row for row in rows if row["scenario_id"] == "hard_disable_to_normal_only"
    )
    assert hard_disable["table_branch_executed_after_rollback"] == "false"
    assert hard_disable["table_support_selected_after_rollback"] == "false"
    assert hard_disable["table_citation_emitted_after_rollback"] == "false"
    assert hard_disable["answer_visible_table_evidence_after_rollback"] == "false"
    assert hard_disable["active_build_pointer_state"] == "disabled"
    assert hard_disable["normal_only_restored"] == "true"
    assert phase7r_validate.rollback_matrix_covers_hard_disable()[0] is True


def test_citation_readiness_blocks_csv_crop_formal_source():
    text = (REPORT_DIR / "citation_readiness_coupling.md").read_text(encoding="utf-8")
    assert "CSV/crop stay debug-only" in text
    assert "must not enter canonical_source or public Citation.source_file" in text
    assert phase7r_validate.citation_readiness_blocks_csv_crop_formal_source()[0] is True


def test_citation_readiness_blocks_preview_only():
    text = (REPORT_DIR / "citation_readiness_coupling.md").read_text(encoding="utf-8")
    assert "preview_only blocks formal citation" in text


def test_citation_readiness_blocks_production_ready_false():
    text = (REPORT_DIR / "citation_readiness_coupling.md").read_text(encoding="utf-8")
    assert "production_ready=false blocks formal citation" in text


def test_rollout_stage_matrix_default_does_not_allow_production():
    rows = _load_csv(DATA_DIR / "rollout_stage_matrix.csv")
    production = next(row for row in rows if row["stage"] == "production")
    canary_no_answer = next(row for row in rows if row["stage"] == "canary_no_answer")
    canary_answer_gated = next(row for row in rows if row["stage"] == "canary_answer_gated")
    for row in (production, canary_no_answer, canary_answer_gated):
        assert row["phase7r_execution_allowed"] == "false"
        assert row["default_state"] == "blocked"
    assert phase7r_validate.rollout_stage_matrix_blocks_production_by_default()[0] is True


def test_validator_results_pass():
    rows = _load_csv(RESULTS_DIR / "phase7r_validation_results.csv")
    assert rows
    assert all(row["check_pass"] == "true" for row in rows)
    report = (REPORT_DIR / "phase7r_validation_report.md").read_text(encoding="utf-8")
    assert "validation_status: `pass_with_warnings`" in report


def test_src_unmodified():
    proc = subprocess.run(
        ["git", "status", "--short", "--", "src"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""


def test_configs_unmodified():
    proc = subprocess.run(
        ["git", "status", "--short", "--", "configs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""
