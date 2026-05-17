from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from scripts.evaluation import phase7q_validate_citation_schema_examples as phase7q


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_schema_prototype"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_schema_prototype"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _examples_by_id():
    rows = _load_jsonl(DATA_DIR / "citation_prototype_examples.jsonl")
    return {row["example_id"]: row for row in rows}


def test_schema_json_is_parseable():
    schema = json.loads((DATA_DIR / "table_evidence_citation_schema.json").read_text())
    assert schema["title"] == "TableEvidenceCitation"
    assert schema["properties"]["citation_type"]["const"] == "table_evidence"


def test_table_row_and_cell_group_examples_are_parseable():
    examples = _examples_by_id()
    assert examples["phase7q_example_table_level"]["schema_object"]["citation_type"] == "table_evidence"
    assert examples["phase7q_example_row_level"]["schema_object"]["evidence_scope"]["citation_scope"] == "row"
    assert (
        examples["phase7q_example_cell_group_level"]["schema_object"]["evidence_scope"][
            "citation_scope"
        ]
        == "cell_group"
    )


def test_csv_and_crop_paths_do_not_enter_canonical_source_for_valid_examples():
    examples = _examples_by_id()
    for example_id in (
        "phase7q_example_table_level",
        "phase7q_example_row_level",
        "phase7q_example_cell_group_level",
    ):
        citation = examples[example_id]["schema_object"]
        source_file = citation["canonical_source"]["source_file"]
        provenance = citation["provenance_debug"]
        assert source_file != provenance["source_csv_path"]
        assert source_file != provenance["source_pdf_crop_path"]
        assert not source_file.endswith(".csv")
        assert not source_file.endswith(".png")


def test_citation_scope_disallows_value():
    schema = json.loads((DATA_DIR / "table_evidence_citation_schema.json").read_text())
    allowed = schema["properties"]["evidence_scope"]["properties"]["citation_scope"]["enum"]
    assert "value" not in allowed
    malformed = _examples_by_id()["phase7q_example_malformed_blocked"]["schema_object"]
    result = phase7q.validate_example(
        {
            "example_id": "x",
            "example_type": "malformed_blocked",
            "schema_object": malformed,
            "expected_validation_status": "blocked",
            "formal_citation_allowed": False,
            "debug_provenance_only": False,
            "example_context": {"query_type": "row_lookup", "retrieved_chunk_object_type": "table_index_unit"},
        }
    )
    assert "invalid_citation_scope" in result["block_reason"]


def test_value_level_claim_is_false_for_valid_examples():
    examples = _examples_by_id()
    for example_id in (
        "phase7q_example_table_level",
        "phase7q_example_row_level",
        "phase7q_example_cell_group_level",
    ):
        limitations = examples[example_id]["schema_object"]["limitations"]
        assert limitations["value_level_citation_claim_allowed"] is False
        assert limitations["value_bboxes_available"] is False


def test_production_ready_false_blocks_formal_citation():
    row = phase7q.validate_example(_examples_by_id()["phase7q_example_row_level"])
    assert row["formal_citation_allowed"] == "false"
    assert "production_ready_false_blocks_formal_citation" in row["warning_reason"]


def test_preview_only_blocks_formal_citation():
    row = phase7q.validate_example(_examples_by_id()["phase7q_example_table_level"])
    assert row["formal_citation_allowed"] == "false"
    assert "preview_only_blocks_formal_citation" in row["warning_reason"]


def test_malformed_example_is_blocked():
    row = phase7q.validate_example(_examples_by_id()["phase7q_example_malformed_blocked"])
    assert row["actual_validation_status"] == "blocked"
    assert row["check_pass"] == "true"
    assert "source_csv_path_in_canonical_source" in row["block_reason"]
    assert "value_level_citation_claim_allowed_not_false" in row["block_reason"]


def test_non_table_query_example_is_blocked():
    row = phase7q.validate_example(_examples_by_id()["phase7q_example_non_table_query_blocked"])
    assert row["actual_validation_status"] == "blocked"
    assert row["check_pass"] == "true"
    assert "non_table_query_blocks_table_citation" in row["block_reason"]


def test_schema_validation_results_are_present_and_pass():
    with (RESULTS_DIR / "schema_validation_results.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert all(row["check_pass"] == "true" for row in rows)
    assert {row["actual_validation_status"] for row in rows} == {"pass_with_warnings", "blocked"}


def test_does_not_modify_src_or_configs():
    proc = subprocess.run(
        ["git", "status", "--short", "--", "src", "configs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""
