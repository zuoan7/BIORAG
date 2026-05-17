import json
from functools import lru_cache

from scripts.extraction import construct_hybrid_table_gold_seed as gold


@lru_cache(maxsize=1)
def _built():
    inputs = gold.load_inputs()
    seeds, validation_rows, summary_rows = gold.build_phase7e_outputs(inputs)
    return inputs, seeds, validation_rows, summary_rows


def _seeds_by_id():
    _, seeds, _, _ = _built()
    return {seed["table_object_id"]: seed for seed in seeds}


def test_only_four_ready_candidates_enter_seed():
    _, seeds, _, _ = _built()

    assert [seed["table_object_id"] for seed in seeds] == gold.READY_CANDIDATE_IDS
    assert len(seeds) == 4


def test_rule_fix_grid_rejected_chunk_fallback_and_backlog_do_not_enter_seed():
    _, seeds, _, _ = _built()
    seed_ids = {seed["table_object_id"] for seed in seeds}
    forbidden = gold.RULE_FIX_IDS | gold.GRID_REJECTED_IDS | gold.CHUNK_FALLBACK_IDS | gold.BACKLOG_IDS

    assert seed_ids.isdisjoint(forbidden)


def test_each_seed_has_gold_seed_id():
    _, seeds, _, _ = _built()

    assert all(seed.get("gold_seed_id") for seed in seeds)


def test_each_seed_has_required_values_with_value_raw():
    _, seeds, _, _ = _built()

    for seed in seeds:
        assert seed["required_values"]
        assert all(value.get("value_raw") for value in seed["required_values"])


def test_confirmed_seed_has_rows_columns_and_cells():
    _, seeds, _, _ = _built()

    for seed in seeds:
        if seed["gold_seed_status"] == "confirmed_seed":
            assert seed["gold_rows"]
            assert seed["gold_columns"]
            assert seed["gold_cells"]


def test_confirmed_seed_has_no_unresolved_structural_blocker():
    _, seeds, _, _ = _built()

    for seed in seeds:
        if seed["gold_seed_status"] == "confirmed_seed":
            assert seed["remaining_blockers"] == []


def test_partial_seed_has_remaining_blockers():
    _, seeds, _, _ = _built()

    for seed in seeds:
        if seed["gold_seed_status"] == "partial_seed":
            assert seed["remaining_blockers"]


def test_value_bboxes_available_is_false():
    _, seeds, _, _ = _built()

    assert all(seed["value_bboxes_available"] is False for seed in seeds)
    assert all(cell.get("value_bbox") is None for seed in seeds for cell in seed["gold_cells"])


def test_source_span_granularity_is_not_value_level():
    _, seeds, _, _ = _built()

    assert all(seed["source_span_granularity"] != "value_level" for seed in seeds)
    assert all(span.get("granularity") != "value_level" for seed in seeds for span in seed["source_spans"])
    assert all(cell.get("source_span_granularity") != "value_level" for seed in seeds for cell in seed["gold_cells"])


def test_ready_for_gold_candidate_is_not_production_ready():
    _, seeds, _, _ = _built()
    payload = json.dumps(seeds, ensure_ascii=False)

    assert all(seed["production_scope"] == "not_production_ready" for seed in seeds)
    assert '"production_ready": true' not in payload


def test_gold_seed_is_not_official_benchmark():
    _, seeds, _, _ = _built()

    assert all(seed["benchmark_scope"] != "official_benchmark" for seed in seeds)
    assert all("not_official_benchmark" in seed["benchmark_scope"] for seed in seeds)


def test_doc0687_table2_confirmed_seed_has_required_logical_columns():
    seed = _seeds_by_id()["doc_0687__table_2__phase7c2_hybrid_02"]
    columns = {column["column_key"] for column in seed["gold_columns"]}

    assert seed["gold_seed_status"] == "confirmed_seed"
    assert set(gold.DOC0687_TABLE2_COLUMNS) <= columns
    assert any(value["logical_column"] == "YE/S" and value["value_raw"] == "0.35" for value in seed["required_values"])


def test_doc0687_table3_partial_seed_retains_asterisk_and_dagger():
    seed = _seeds_by_id()["doc_0687__table_3__phase7c2_hybrid_03"]
    values = [value["value_raw"] for value in seed["required_values"]]

    assert seed["gold_seed_status"] == "partial_seed"
    assert any("∗" in value for value in values)
    assert any("†" in value for value in values)


def test_doc0523_confirmed_seed_preserves_nd_and_unit_literals():
    seed = _seeds_by_id()["doc_0523__table_1__phase7c2_hybrid_01"]
    values = [value["value_raw"] for value in seed["required_values"]]

    assert seed["gold_seed_status"] == "confirmed_seed"
    assert "N.D." in values
    assert "g/L" in values
