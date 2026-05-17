from __future__ import annotations

from scripts.evaluation import phase7l_table_rag_smoke_common as smoke


def _redirect_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "OUTPUT_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(smoke, "OUTPUT_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(smoke, "OUTPUT_REPORT_DIR", tmp_path / "reports")


def test_table_unit_adapter_contract_for_eligible_units():
    units = smoke.load_eligible_units()

    assert len(units) == 274

    sample = smoke.adapt_table_unit(units[0])
    checks = smoke.validate_adapted_chunk(sample)

    assert all(checks.values())
    assert sample.chunk_id == f"table_unit::{units[0]['table_index_unit_id']}"
    assert sample.metadata["object_type"] == "table_index_unit"
    assert sample.metadata["production_ready"] is False
    assert sample.metadata["index_unit_status"] == "preview_only"
    assert sample.metadata["value_bboxes_available"] is False
    assert "[TABLE INDEX UNIT]" in sample.text


def test_sidecar_retriever_uses_local_eligible_units_only():
    units = smoke.load_eligible_units()
    queries = smoke.load_queries()
    query = next(row for row in queries if row["query_id"] == "phase7j_query_008")

    candidates = smoke.sidecar_search(query, units, top_k=5)

    assert candidates
    assert len(candidates) <= 5
    assert all(candidate.chunk.chunk_id.startswith("table_unit::") for candidate in candidates)
    assert all(candidate.guardrail_pass for candidate in candidates)
    assert {candidate.chunk.metadata["table_index_unit_id"] for candidate in candidates} <= {
        unit["table_index_unit_id"] for unit in units
    }


def test_shadow_mode_keeps_table_candidates_out_of_rerank_and_support(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)
    units = smoke.load_eligible_units()
    queries = smoke.load_queries()[:4]

    summary = smoke.run_shadow_mode_smoke(units, queries)

    assert summary["pass"] is True
    assert summary["table_candidates_in_rerank_count"] == 0
    assert summary["support_pack_table_count"] == 0
    assert summary["final_evidence_normal_only"] is True


def test_active_merge_can_add_preview_table_evidence_before_stub_rerank(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)
    units = smoke.load_eligible_units()
    queries = [
        row
        for row in smoke.load_queries()
        if row["query_id"] in {"phase7j_query_004", "phase7j_query_027"}
    ]

    summary = smoke.run_active_merge_smoke(units, queries)

    assert summary["pass"] is True
    assert summary["queries_with_table_support"] >= 1
    assert summary["all_table_candidates_production_ready_false"] is True
    assert summary["max_units_per_seed_checked"] is True
    assert summary["max_units_per_table_checked"] is True
    assert summary["weak_match_filtering_checked"] is True
    assert summary["row_cell_group_dedupe_checked"] is True


def test_rollback_restores_normal_only_path(monkeypatch, tmp_path):
    _redirect_outputs(monkeypatch, tmp_path)
    queries = smoke.load_queries()[:4]

    summary = smoke.run_rollback_smoke(queries=queries)

    assert summary["pass"] is True
    assert summary["table_branch_executed_count"] == 0
    assert summary["support_pack_table_count"] == 0
    assert summary["normal_only_restored"] is True
