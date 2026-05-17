import csv
import json
from pathlib import Path

from scripts.evaluation import render_table_retrieval_evidence_cards as render_cards
from scripts.evaluation import run_table_retrieval_wiring_preview as preview


ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE_UNITS = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
EXCLUDED_UNITS = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/content_quality_excluded_units.csv"
)
QUERY_SET = ROOT / "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.jsonl"
RESULTS_CSV = (
    ROOT / "results/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_results.csv"
)
RESULTS_JSON = (
    ROOT / "results/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_results.json"
)
TOPK_JSONL = ROOT / "results/v7_phase7_table_retrieval_wiring_preview/topk_evidence_units.jsonl"
EVIDENCE_CARDS = (
    ROOT / "reports/v7_phase7_table_retrieval_wiring_preview/retrieval_evidence_cards.md"
)


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase7j_uses_only_phase7i1_eligible_units():
    eligible_units = _load_jsonl(ELIGIBLE_UNITS)
    queries = _load_jsonl(QUERY_SET)
    eligible_ids = {unit["table_index_unit_id"] for unit in eligible_units}

    assert len(eligible_units) == 274
    assert {query["query_source_unit_id"] for query in queries} <= eligible_ids
    assert {query["expected_table_index_unit_id"] for query in queries} <= eligible_ids


def test_excluded_units_do_not_enter_dry_run_results():
    excluded_ids = {row["table_index_unit_id"] for row in _load_csv(EXCLUDED_UNITS)}
    result_ids = {
        row["matched_table_index_unit_id"]
        for row in _load_csv(RESULTS_CSV)
        if row["matched_table_index_unit_id"]
    }

    assert len(excluded_ids) == 140
    assert not (result_ids & excluded_ids)


def test_query_count_and_required_query_types():
    queries = _load_jsonl(QUERY_SET)
    query_types = {query["query_type"] for query in queries}

    assert len(queries) >= 20
    assert len(queries) <= 50
    assert {"table_lookup", "row_lookup", "metric_lookup"} <= query_types


def test_lexical_scorer_has_no_bm25_milvus_embedding_or_model_access_patterns():
    source = (ROOT / "scripts/evaluation/run_table_retrieval_wiring_preview.py").read_text(
        encoding="utf-8"
    )

    assert "bm25_index.json" not in source.lower()
    assert "phase5f_official_clean_baseline/bm25" not in source
    assert "rank_bm25" not in source
    assert "pymilvus" not in source.lower()
    assert "MilvusClient" not in source
    assert "SentenceTransformer" not in source
    assert "embedding_vector" not in source
    assert "openai" not in source.lower()
    assert "qwen" not in source.lower()
    assert "ragas" not in source.lower()


def test_topk_results_all_come_from_eligible_units():
    eligible_ids = {unit["table_index_unit_id"] for unit in _load_jsonl(ELIGIBLE_UNITS)}
    result_rows = _load_csv(RESULTS_CSV)

    assert result_rows
    assert {
        row["matched_table_index_unit_id"]
        for row in result_rows
        if row["matched_table_index_unit_id"]
    } <= eligible_ids


def test_evidence_text_and_guardrail_flags_are_present():
    result_rows = _load_csv(RESULTS_CSV)
    evidence_rows = [row for row in result_rows if row["match_status"] != "no_match"]

    assert evidence_rows
    assert all(row["evidence_text"].strip() for row in result_rows)
    assert all(row["production_ready"] == "false" for row in evidence_rows)
    assert all(row["value_bboxes_available"] == "false" for row in evidence_rows)


def test_no_official_benchmark_flag_is_written():
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    assert payload["official_benchmark"] is False
    assert payload["formal_retrieval_evaluation"] is False
    assert payload["guardrail"]["official_benchmark_written"] is False


def test_evidence_cards_are_generated_and_renderable(tmp_path):
    assert EVIDENCE_CARDS.exists()
    assert "Phase7J 检索证据卡" in EVIDENCE_CARDS.read_text(encoding="utf-8")

    out = tmp_path / "cards.md"
    summary = render_cards.render_evidence_cards(topk_jsonl=TOPK_JSONL, report_path=out)

    assert summary["query_count"] >= 20
    assert out.exists()
    assert "期望目标" in out.read_text(encoding="utf-8")


def test_no_match_is_recorded_instead_of_silent_failure(tmp_path):
    unit = {
        "table_index_unit_id": "fixture_unit_001",
        "unit_type": "row_unit",
        "seed_id": "fixture_seed",
        "doc_id": "doc_fixture",
        "table_id": "Table F",
        "caption": "Fixture table.",
        "content_text_for_embedding": "alpha beta gamma",
        "metadata": {"row_label": "alpha", "row_values": []},
        "provenance": {
            "source_csv_path": "fixture.csv",
            "source_pdf_crop_path": "fixture.png",
            "value_bboxes_available": False,
        },
        "guardrail": {"production_ready": False},
    }
    query = {
        "query_id": "fixture_query_no_match",
        "query_text": "zzzzzz yyyyyy",
        "query_type": "row_lookup",
        "expected_seed_id": "other_seed",
        "expected_doc_id": "doc_other",
        "expected_table_id": "Table Other",
        "expected_unit_type": "row_unit",
        "expected_table_index_unit_id": "other_unit",
        "expected_row_label": "other",
    }
    units_path = tmp_path / "units.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    units_path.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    queries_path.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = preview.run_preview(
        eligible_jsonl=units_path,
        query_jsonl=queries_path,
        results_dir=tmp_path / "results",
    )

    assert summary["no_match_count"] == 1
    rows = _load_csv(tmp_path / "results/retrieval_wiring_preview_results.csv")
    assert rows[0]["match_status"] == "no_match"
    assert rows[0]["failure_reason"] == "lexical_score_zero"
