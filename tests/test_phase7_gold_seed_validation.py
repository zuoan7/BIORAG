import copy
import json

from scripts.evaluation import validate_hybrid_extractor_against_gold_seed as validator


CONFIRMED_ID = validator.FORMAL_CONFIRMED_SEED_IDS[0]
PARTIAL_ID = validator.PARTIAL_EXPLORATORY_SEED_IDS[0]


def _seed(gold_seed_id=CONFIRMED_ID, status="confirmed_seed"):
    return {
        "gold_seed_id": gold_seed_id,
        "table_object_id": "doc_test__table_1",
        "doc_id": "doc_test",
        "table_id": "Table 1",
        "gold_seed_status": status,
        "gold_rows": [{"row_key": "row_1"}],
        "gold_columns": [{"column_key": "metric"}],
        "gold_cells": [
            {
                "gold_cell_id": "gold_cell_1",
                "row_key": "row_1",
                "logical_column": "metric",
                "value_raw": "1.0",
                "unit": "g/L",
                "footnote_refs": [],
                "source_span_ids": ["span_1"],
                "source_span_granularity": "cell_level",
                "value_bbox": None,
                "value_bbox_source": "not_available",
            }
        ],
        "required_values": [
            {
                "row_key": "row_1",
                "logical_column": "metric",
                "value_raw": "1.0",
                "unit": "g/L",
                "footnote_refs": [],
                "source_span_ids": ["span_1"],
            }
        ],
        "required_units": [
            {
                "unit_id": "unit_1",
                "scope": "metric",
                "unit_raw": "g/L",
                "binding_status": "bound_to_selected_cells",
            }
        ],
        "footnote_binding": {"binding_status": "not_applicable"},
        "reference_binding": {"binding_status": "not_applicable"},
        "literal_preservation": {"status": "pass"},
        "source_span_granularity": "cell_level",
        "value_bboxes_available": False,
        "construction_warnings": [],
    }


def _table_object():
    return {
        "table_object_id": "doc_test__table_1",
        "logical_rows": [{"row_key": "row_1"}],
        "logical_columns": ["metric"],
        "logical_cells": [
            {
                "logical_cell_id": "logical_cell_1",
                "row_key": "row_1",
                "logical_column": "metric",
                "value_raw": "1.0",
                "unit": "g/L",
                "footnote_refs": [],
                "source_span_ids": ["span_1"],
                "source_span_granularity": "cell_level",
                "value_bbox": None,
                "value_bbox_source": "not_available",
            }
        ],
        "source_span_granularity": "cell_level",
        "value_bboxes_available": False,
    }


def _payload(seeds, table_objects):
    return validator.build_validation_results(
        {
            "seeds": seeds,
            "confirmed_seed_ids": [CONFIRMED_ID],
            "partial_seed_ids": [PARTIAL_ID],
            "partial_seed_rows": [
                {
                    "gold_seed_id": PARTIAL_ID,
                    "remaining_blockers": "fixture_partial_only",
                }
            ],
            "table_objects": table_objects,
        }
    )


def _single_result(seed=None, table_object=None):
    seed = seed or _seed()
    table_object = table_object or _table_object()
    return _payload([seed], [table_object])["results"][0]


def test_confirmed_seed_enters_formal_confirmed():
    result = _single_result()

    assert result["validation_subset"] == "formal_confirmed"
    assert result["overall_validation_status"] in {"pass", "pass_with_warnings"}


def test_partial_seed_enters_exploratory_partial():
    partial_seed = _seed(PARTIAL_ID, "partial_seed")
    result = _single_result(partial_seed, _table_object())

    assert result["validation_subset"] == "exploratory_partial"


def test_partial_seed_does_not_enter_formal_overall():
    partial_seed = _seed(PARTIAL_ID, "partial_seed")
    payload = _payload([_seed(), partial_seed], [_table_object()])

    assert payload["formal_confirmed_overall"]["formal_seed_count"] == 1
    assert payload["partial_exploratory_count"] == 1


def test_missing_required_values_prevent_formal_pass():
    table_object = _table_object()
    table_object["logical_cells"][0]["value_raw"] = "2.0"
    result = _single_result(_seed(), table_object)

    assert result["required_value_coverage"] == "not_covered"
    assert result["overall_validation_status"] == "fail"


def test_missing_value_raw_prevents_formal_pass():
    table_object = _table_object()
    table_object["logical_cells"][0]["value_raw"] = None
    result = _single_result(_seed(), table_object)

    assert result["required_value_coverage"] == "not_covered"
    assert result["overall_validation_status"] == "fail"


def test_missing_source_span_prevents_formal_pass():
    table_object = _table_object()
    table_object["logical_cells"][0]["source_span_ids"] = []
    result = _single_result(_seed(), table_object)

    assert result["source_span_coverage"] == "not_covered"
    assert result["overall_validation_status"] == "fail"


def test_value_bboxes_available_false_is_not_failure():
    result = _single_result()

    assert result["bbox_provenance_check"] == "covered_with_warnings"
    assert result["overall_validation_status"] in {"pass", "pass_with_warnings"}


def test_source_span_granularity_must_not_be_value_level():
    seed = _seed()
    table_object = _table_object()
    seed["source_span_granularity"] = "value_level"
    table_object["source_span_granularity"] = "value_level"
    result = _single_result(seed, table_object)

    assert result["source_span_coverage"] == "not_covered"
    assert result["overall_validation_status"] == "fail"


def test_confirmed_seed_is_not_official_benchmark():
    payload = _payload([_seed()], [_table_object()])

    assert payload["validation_scope"] == "seed_level_offline_validation_not_official_benchmark"
    assert payload["validation_scope"] != "official_benchmark"


def test_validation_does_not_write_prod_ready_marker():
    marker = "production" + "_ready"
    seed = copy.deepcopy(_seed())
    seed["construction_warnings"] = [f"confirmed_seed_not_{marker}"]
    payload = _payload([seed], [_table_object()])

    assert marker not in json.dumps(payload, ensure_ascii=False)
