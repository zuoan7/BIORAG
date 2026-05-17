from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_binder_prototype_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_binder_prototype_dry_run"
REPORT_DIR = ROOT / "reports/v7_phase7_table_citation_binder_prototype_dry_run"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_mapper_fixture_is_parseable_and_covers_required_cases():
    rows = _load_jsonl(DATA_DIR / "mapper_input_fixture.jsonl")
    fixture_types = {row["fixture_type"] for row in rows}
    assert {
        "table_level",
        "row_level",
        "cell_group_level",
        "csv_source_file_sanitized",
        "malformed_missing_table_id",
        "malformed_value_scope",
        "non_table_query_blocked",
        "normal_chunk_not_mapped",
    } <= fixture_types


def test_table_row_cell_group_map_successfully_as_debug_only():
    rows = _load_jsonl(RESULTS_DIR / "mapped_table_evidence_citations.jsonl")
    by_type = {row["fixture_type"]: row for row in rows}
    for fixture_type in ("table_level", "row_level", "cell_group_level"):
        assert by_type[fixture_type]["mapper_status"] == "mapped_with_warnings"
        assert by_type[fixture_type]["formal_citation_allowed"] is False
        assert by_type[fixture_type]["debug_provenance_only"] is True


def test_csv_crop_paths_do_not_enter_canonical_source():
    rows = _load_jsonl(RESULTS_DIR / "mapped_table_evidence_citations.jsonl")
    for row in rows:
        citation = row["schema_object"]
        source_file = citation["canonical_source"]["source_file"]
        provenance = citation["provenance_debug"]
        assert source_file != provenance["source_csv_path"]
        assert source_file != provenance["source_pdf_crop_path"]
        assert source_file != provenance["source_markdown_path"]
        if source_file:
            assert not source_file.endswith(".csv")
            assert not source_file.endswith(".png")


def test_value_level_claim_is_never_allowed_for_mapped_outputs():
    rows = _load_jsonl(RESULTS_DIR / "mapped_table_evidence_citations.jsonl")
    assert rows
    for row in rows:
        limitations = row["schema_object"]["limitations"]
        assert limitations["value_bboxes_available"] is False
        assert limitations["value_level_citation_claim_allowed"] is False


def test_preview_only_and_production_ready_false_block_formal_citation():
    rows = _load_jsonl(RESULTS_DIR / "mapped_table_evidence_citations.jsonl")
    for row in rows:
        limitations = row["schema_object"]["limitations"]
        assert limitations["production_ready"] is False
        assert limitations["index_unit_status"] == "preview_only"
        assert row["formal_citation_allowed"] is False


def test_malformed_and_policy_blocked_records_are_blocked():
    rows = _load_jsonl(RESULTS_DIR / "mapper_blocked_records.jsonl")
    by_id = {row["fixture_id"]: row for row in rows}
    assert "missing_table_id" in by_id["malformed_missing_table_id"]["block_reasons"]
    assert "citation_scope_value_forbidden" in by_id["malformed_value_scope"]["block_reasons"]
    assert (
        "non_table_query_blocks_table_citation"
        in by_id["non_table_query_table_candidate"]["block_reasons"]
    )
    assert "normal_chunk_not_table_evidence" in by_id["normal_chunk_not_mapped"]["block_reasons"]


def test_mapper_validation_results_pass_with_warnings():
    rows = _load_csv(RESULTS_DIR / "mapper_validation_results.csv")
    assert len(rows) == 8
    assert all(row["check_pass"] == "true" for row in rows)
    assert {row["record_kind"] for row in rows} == {"mapped", "blocked"}
    summary = (REPORT_DIR / "phase7q1_summary.md").read_text(encoding="utf-8")
    assert "validation_status: `pass_with_warnings`" in summary


def test_no_src_or_configs_modifications():
    proc = subprocess.run(
        ["git", "status", "--short", "--", "src", "configs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""
