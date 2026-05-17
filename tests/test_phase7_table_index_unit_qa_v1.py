import json
from pathlib import Path

from scripts.extraction import build_phase7j_preview_subset as subset
from scripts.extraction import qa_table_index_units_v1 as qa


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_UNITS = ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units.preview.jsonl"
QA_UNITS = ROOT / "data/experiments/v7_phase7_table_index_unit_qa/table_index_units.qa.preview.jsonl"
ELIGIBLE_UNITS = ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
FORMAL_VALIDATION = ROOT / "results/v7_phase7_expanded_seed_validation/formal_seed_validation_results.csv"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path):
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _quality(unit):
    return unit["metadata"]["index_quality"]


def _metadata_without_quality(unit):
    metadata = dict(unit.get("metadata") or {})
    metadata.pop("index_quality", None)
    return metadata


def test_input_unit_count_is_414_and_formal_seed_count_is_15():
    original_units = _load_jsonl(ORIGINAL_UNITS)
    formal_rows = _load_csv(FORMAL_VALIDATION)

    assert len(original_units) == 414
    assert len({row["seed_id"] for row in formal_rows}) == 15


def test_qa_keeps_15_formal_seeds_and_all_414_units():
    qa_units = _load_jsonl(QA_UNITS)
    formal_ids = {row["seed_id"] for row in _load_csv(FORMAL_VALIDATION)}

    assert len(qa_units) == 414
    assert {unit["seed_id"] for unit in qa_units} == formal_ids


def test_original_phase7i_preview_is_not_overwritten():
    original_units = _load_jsonl(ORIGINAL_UNITS)

    assert ORIGINAL_UNITS != QA_UNITS
    assert all("index_quality" not in (unit.get("metadata") or {}) for unit in original_units)


def test_each_qa_unit_has_index_quality_metadata():
    qa_units = _load_jsonl(QA_UNITS)

    assert all(isinstance(_quality(unit), dict) for unit in qa_units)
    assert all("index_text_quality" in _quality(unit) for unit in qa_units)
    assert all("header_path_quality" in _quality(unit) for unit in qa_units)
    assert all("retrieval_ready" in _quality(unit) for unit in qa_units)


def test_retrieval_ready_is_not_production_ready():
    qa_units = _load_jsonl(QA_UNITS)
    ready_units = [unit for unit in qa_units if _quality(unit)["retrieval_ready"] is True]

    assert ready_units
    assert all(unit["guardrail"]["production_ready"] is False for unit in ready_units)
    assert any(_quality(unit)["retrieval_ready"] != unit["guardrail"]["production_ready"] for unit in qa_units)


def test_guardrail_flags_remain_false():
    qa_units = _load_jsonl(QA_UNITS)

    assert all(unit["guardrail"]["production_ready"] is False for unit in qa_units)
    assert all(unit["guardrail"]["is_official_benchmark_seed"] is False for unit in qa_units)
    assert all(unit["provenance"]["value_bboxes_available"] is False for unit in qa_units)


def test_p_value_parent_mismatch_causes_retrieval_ready_false():
    unit = {
        "table_index_unit_id": "fixture_pvalue_mismatch",
        "unit_type": "row_unit",
        "doc_id": "doc_fixture",
        "table_id": "Table 1",
        "content_text_for_embedding": "In doc_fixture Table 1, row A reports: Overall p-value / GOS=30.2 ± 22.1.",
        "metadata": {
            "row_label": "A",
            "header_path": [["Overall p-value", "GOS"]],
            "row_values": [
                {
                    "column_header": "Overall p-value / GOS",
                    "header_path": ["Overall p-value", "GOS"],
                    "value": "30.2 ± 22.1",
                }
            ],
        },
        "guardrail": {
            "index_unit_status": "preview_only",
            "production_ready": False,
            "is_official_benchmark_seed": False,
        },
        "provenance": {"value_bboxes_available": False, "source_span_granularity": "cell_level"},
    }

    quality, _ = qa.evaluate_unit(unit)

    assert "p_value_parent_mismatch" in quality["quality_flags"]
    assert quality["retrieval_ready"] is False


def test_header_path_contains_data_value_causes_retrieval_ready_false():
    unit = {
        "table_index_unit_id": "fixture_header_data",
        "unit_type": "row_unit",
        "doc_id": "doc_fixture",
        "table_id": "Table 1",
        "content_text_for_embedding": "In doc_fixture Table 1, row A reports: 20.7 ± 12.8=x; Metric=1.",
        "metadata": {
            "row_label": "A",
            "header_path": [["20.7 ± 12.8"], ["Metric"]],
            "row_values": [
                {"column_header": "20.7 ± 12.8", "header_path": ["20.7 ± 12.8"], "value": "x"},
                {"column_header": "Metric", "header_path": ["Metric"], "value": "1"},
            ],
        },
        "guardrail": {
            "index_unit_status": "preview_only",
            "production_ready": False,
            "is_official_benchmark_seed": False,
        },
        "provenance": {"value_bboxes_available": False, "source_span_granularity": "cell_level"},
    }

    quality, _ = qa.evaluate_unit(unit)

    assert "header_path_contains_data_value" in quality["quality_flags"]
    assert quality["retrieval_ready"] is False


def test_low_text_quality_and_header_fail_do_not_enter_eligible_subset():
    low_unit = {
        "table_index_unit_id": "fixture_low",
        "unit_type": "row_unit",
        "content_text_for_embedding": "weak",
        "metadata": {
            "index_quality": {
                "retrieval_ready": False,
                "index_text_quality": "low",
                "header_path_quality": "pass",
                "quality_flags": ["low_information_content"],
            }
        },
        "guardrail": {"production_ready": False, "index_unit_status": "preview_only"},
        "provenance": {"value_bboxes_available": False},
    }
    fail_unit = {
        "table_index_unit_id": "fixture_fail",
        "unit_type": "row_unit",
        "content_text_for_embedding": "enough content with two facts",
        "metadata": {
            "row_values": [
                {"column_header": "A", "value": "1"},
                {"column_header": "B", "value": "2"},
            ],
            "index_quality": {
                "retrieval_ready": False,
                "index_text_quality": "high",
                "header_path_quality": "fail",
                "quality_flags": ["header_path_contains_data_value"],
            },
        },
        "guardrail": {"production_ready": False, "index_unit_status": "preview_only"},
        "provenance": {"value_bboxes_available": False},
    }

    assert "index_text_quality_low" in subset.exclusion_reasons(low_unit, {})
    assert "header_path_quality_fail" in subset.exclusion_reasons(fail_unit, {})


def test_doc0261_table2_and_table3_gos_no_longer_under_overall_p_value():
    qa_units = _load_jsonl(QA_UNITS)
    targets = [
        unit
        for unit in qa_units
        if unit["doc_id"] == "doc_0261" and unit["table_id"] in {"Table 2", "Table 3"}
    ]

    assert targets
    for unit in targets:
        current_payload = json.dumps(_metadata_without_quality(unit), ensure_ascii=False)
        assert "Overall p-value / GOS" not in unit["content_text_for_embedding"]
        assert "Overall p-value / GOS" not in unit["content_markdown"]
        assert '"Overall p-value", "GOS"' not in current_payload
        assert '"Overall p-value","GOS"' not in current_payload


def test_retrieval_ready_false_units_do_not_enter_eligible_subset():
    qa_units = _load_jsonl(QA_UNITS)
    eligible_ids = {unit["table_index_unit_id"] for unit in _load_jsonl(ELIGIBLE_UNITS)}
    not_ready_ids = {
        unit["table_index_unit_id"] for unit in qa_units if _quality(unit)["retrieval_ready"] is False
    }

    assert not eligible_ids & not_ready_ids


def test_partial_reject_unreviewed_do_not_enter_qa_formal_set():
    qa_units = _load_jsonl(QA_UNITS)
    formal_ids = {row["seed_id"] for row in _load_csv(FORMAL_VALIDATION)}

    assert {unit["seed_id"] for unit in qa_units} == formal_ids
    assert all(unit["guardrail"]["seed_status"] == "confirmed_seed_with_warnings" for unit in qa_units)


def test_no_bm25_or_milvus_access_patterns_in_scripts_or_outputs():
    scripts = [
        ROOT / "scripts/extraction/qa_table_index_units_v1.py",
        ROOT / "scripts/extraction/build_phase7j_preview_subset.py",
        ROOT / "scripts/extraction/validate_table_index_unit_qa_v1.py",
    ]
    payload = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
    qa_units = _load_jsonl(QA_UNITS)
    output_payload = json.dumps(qa_units, ensure_ascii=False)

    assert "import pymilvus" not in payload
    assert "from pymilvus" not in payload
    assert "MilvusClient(" not in payload
    assert "bm25_index.json" not in payload
    assert "phase5f_official_clean_baseline/bm25" not in payload
    assert "import rank_bm25" not in payload
    assert "bm25_index.json" not in output_payload
    assert "milvus_id" not in output_payload


def test_no_embedding_or_retrieval_outputs_are_generated():
    qa_units = _load_jsonl(QA_UNITS)
    forbidden_keys = {
        "embedding",
        "embedding_vector",
        "vector",
        "retrieval_score",
        "rerank_score",
        "bm25_score",
        "milvus_id",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from keys(nested)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert all(not (set(keys(unit)) & forbidden_keys) for unit in qa_units)
