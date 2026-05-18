from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluation import phase7s_canonical_source_dry_run as phase7s_canonical
from scripts.evaluation import phase7s_production_readiness_gate_dry_run as phase7s_gate
from scripts.evaluation import phase7s_validate_readiness_dry_run as phase7s_validate


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_production_readiness_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_production_readiness_dry_run"


@pytest.fixture(scope="module", autouse=True)
def _ensure_phase7s_artifacts():
    phase7s_canonical.run()
    phase7s_gate.run()
    summary = phase7s_validate.write_validation_artifacts()
    assert summary["validation_status"] == "pass_with_warnings"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_canonical_source_manifest_draft_is_parseable():
    rows = _load_jsonl(DATA_DIR / "canonical_source_manifest.draft.jsonl")
    assert len(rows) == 274
    required = {
        "table_index_unit_id",
        "doc_id",
        "table_id",
        "table_caption",
        "page",
        "candidate_id",
        "seed_id",
        "canonical_source_status",
        "canonical_source_confidence",
        "canonical_source_fields_available",
        "canonical_source_missing_fields",
        "formal_source_allowed",
        "debug_provenance_paths",
        "notes",
    }
    assert all(required.issubset(row) for row in rows)


def test_csv_crop_path_only_enters_debug_provenance():
    rows = _load_jsonl(DATA_DIR / "canonical_source_manifest.draft.jsonl")
    for row in rows:
        debug_paths = row["debug_provenance_paths"]
        assert set(debug_paths).issubset(
            {"source_csv_path", "source_pdf_crop_path", "source_markdown_path"}
        )
        formal_fields = {key: value for key, value in row.items() if key != "debug_provenance_paths"}
        serialized = json.dumps(formal_fields, ensure_ascii=False)
        for path_value in debug_paths.values():
            assert path_value not in serialized
        assert row["formal_source_allowed"] is False


def test_preview_only_is_blocked_by_production_gate():
    rows = _load_csv(RESULTS_DIR / "production_readiness_gate_dry_run.csv")
    assert len(rows) == 274
    assert all("index_unit_status_not_preview_only" in row["failed_gates"] for row in rows)
    assert all(row["gate_status"] == "blocked" for row in rows)


def test_production_ready_false_is_blocked_by_production_gate():
    rows = _load_csv(RESULTS_DIR / "production_readiness_gate_dry_run.csv")
    assert all(
        "production_ready_true_independent_gate" in row["failed_gates"] for row in rows
    )


def test_value_bboxes_false_blocks_value_level_citation():
    rows = _load_csv(RESULTS_DIR / "production_readiness_gate_dry_run.csv")
    assert all(
        "value_level_citation_disabled_due_to_value_bboxes_available=false"
        in row["blocker_reason"]
        for row in rows
    )


def test_blocker_category_is_not_empty():
    rows = _load_csv(RESULTS_DIR / "production_readiness_gate_dry_run.csv")
    assert all(row["blocker_category"] for row in rows)
    assert {row["blocker_category"] for row in rows} <= {
        "data_blocker",
        "schema_or_binder_blocker",
        "operational_blocker",
        "expected_preview_blocker",
        "not_evaluable",
    }


def test_validation_summary_is_pass_or_pass_with_warnings():
    rows = _load_csv(RESULTS_DIR / "phase7s_validation_summary.csv")
    status_row = next(row for row in rows if row["check_id"] == "validation_status")
    assert status_row["status"] in {"pass", "pass_with_warnings"}
    assert all(
        row["check_pass"] == "true"
        for row in rows
        if row["check_id"] != "validation_status"
    )


def test_src_configs_unmodified():
    proc = subprocess.run(
        ["git", "status", "--short", "--", "src", "configs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""
