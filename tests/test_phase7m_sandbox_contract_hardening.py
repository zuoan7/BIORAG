from __future__ import annotations

import csv
import json

from scripts.evaluation import phase7m_contract_regression_check as phase7m


def _redirect_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(phase7m, "OUTPUT_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(phase7m, "OUTPUT_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(phase7m, "OUTPUT_REPORT_DIR", tmp_path / "reports")


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_phase7m_generation_v2_contract_preserves_table_metadata(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)

    summary = phase7m.run_generation_v2_contract_smoke()

    assert summary["pass"] is True
    rows = _read_jsonl(tmp_path / "results/generation_v2_contract_results.jsonl")
    assert {row["table_unit_type"] for row in rows} == {
        "table_unit",
        "row_unit",
        "cell_group_unit",
    }
    assert all(row["contract_pass"] for row in rows)
    assert all(row["formal_citation_emitted"] is False for row in rows)
    assert all(row["answer_generated"] is False for row in rows)
    assert all(
        row["contract_checks"]["source_csv_path_preserved"]
        and row["contract_checks"]["source_pdf_crop_path_preserved"]
        for row in rows
    )


def test_phase7m_citation_guard_keeps_paths_out_of_formal_citations(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)

    summary = phase7m.run_citation_guard_smoke()

    assert summary["pass"] is True
    rows = _read_csv(tmp_path / "results/citation_guard_results.csv")
    assert rows
    assert all(row["formal_citation_emitted"] == "false" for row in rows)
    assert all(row["source_path_written_as_formal_citation"] == "false" for row in rows)
    assert all(row["value_level_citation_claim"] == "false" for row in rows)
    assert all(row["value_bboxes_available"] == "False" for row in rows)
    assert all(row["warning_level_binding_upgraded"] == "false" for row in rows)


def test_phase7m_policy_matrix_covers_all_required_query_types(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)

    summary = phase7m.run_policy_matrix_smoke()

    assert summary["pass"] is True
    rows = _read_csv(tmp_path / "results/policy_matrix_results.csv")
    assert {row["query_type"] for row in rows} == set(phase7m.REQUIRED_QUERY_TYPES)
    assert all(row["policy_assertion"] == "pass" for row in rows)
    non_table = next(row for row in rows if row["query_type"] == "non_table_query")
    assert non_table["table_branch_active_allowed"] == "false"
    assert non_table["support_pack_allowed"] == "false"


def test_phase7m_failure_path_hardening_covers_required_modes(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)

    summary = phase7m.run_failure_path_smoke()

    assert summary["pass"] is True
    assert summary["missing_modes"] == []
    rows = _read_csv(tmp_path / "results/failure_path_results.csv")
    assert {row["query_id"] for row in rows} >= set(phase7m.FOCUS_QUERY_IDS)
    assert all(row["acceptable_warning"] == "true" for row in rows)
    assert any("ambiguous_table_query" in row["covered_failure_modes"] for row in rows)
    assert any("non_table_query" in row["covered_failure_modes"] for row in rows)


def test_phase7m_baseline_and_rollback_guardrails(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)

    baseline = phase7m.run_phase7l_baseline_check()
    rollback = phase7m.run_rollback_regression_check()

    assert baseline["pass"] is True
    assert baseline["validation_status"] == "pass_with_warnings"
    assert baseline["bm25_queried"] is False
    assert baseline["milvus_accessed"] is False
    assert baseline["src_modified"] is False
    assert baseline["configs_modified"] is False
    assert rollback["pass"] is True


def test_phase7m_run_all_summary_is_pass_with_warnings(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)

    summary = phase7m.run_all()

    assert summary["validation_status"] == "pass_with_warnings"
    assert summary["recommend_next_step_phase7n"] is True
    assert summary["recommend_production"] is False
    assert summary["route_c_backlog_only"] is True
    assert (tmp_path / "reports/phase7m_summary.md").exists()
