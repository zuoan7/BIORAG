import csv
import json
from pathlib import Path

from scripts.extraction import render_expanded_table_review_pack as render


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fixture_candidates():
    return [
        {
            "candidate_id": "phase7g_fixture_001",
            "table_object_id": "doc_fixture__table_1__phase7g_review",
            "doc_id": "doc_fixture",
            "table_id": "Table 1",
            "caption": "Table 1. Fixture numeric metric table.",
            "page": 1,
            "routing_status": "phase7g_pdfplumber_review",
            "auto_score": 0.91,
            "review_priority": "P0_quick_review",
            "suggested_decision": "likely_confirmed_candidate",
            "risk_tags": ["numeric_metric_table"],
            "table_type_tags": ["numeric_metric_table", "simple_row_column_table"],
            "markdown_path": "",
            "csv_path": "",
            "pdf_crop_path": "",
            "crop_status": "pending",
            "source_span_granularity": "cell_level",
            "value_bboxes_available": False,
            "cell_bboxes_available": True,
            "warnings": ["cell_bbox_not_value_bbox"],
            "rows": [["strain", "titer", "unit"], ["A", "1.2", "g/L"]],
        },
        {
            "candidate_id": "phase7g_fixture_auto_excluded",
            "table_object_id": "doc_fixture__bad",
            "doc_id": "doc_fixture",
            "table_id": "Table 2",
            "caption": "Figure mixed candidate.",
            "page": 2,
            "routing_status": "grid_rejected",
            "auto_score": 0.2,
            "review_priority": "auto_excluded",
            "suggested_decision": "reject_grid",
            "risk_tags": ["likely_false_positive"],
            "table_type_tags": ["general_table_signal"],
            "markdown_path": "",
            "csv_path": "",
            "pdf_crop_path": "",
            "crop_status": "auto_excluded",
            "source_span_granularity": "table_row_level",
            "value_bboxes_available": False,
            "cell_bboxes_available": False,
            "warnings": [],
            "rows": [],
        },
    ]


def _render_fixture(tmp_path: Path):
    output_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    _write_jsonl(output_dir / "candidate_pool_raw.jsonl", _fixture_candidates())
    result = render.render_review_pack(
        output_dir=output_dir,
        report_dir=report_dir,
        pdf_dir=tmp_path,
        review_target=1,
        export_crops_enabled=False,
    )
    return output_dir, report_dir, result


def _csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_review_pack_index_exists(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)

    assert (output_dir / "review_pack_index.csv").exists()


def test_review_labels_template_exists(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)

    assert (output_dir / "review_labels_template.csv").exists()


def test_each_review_candidate_has_candidate_id(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)
    rows = _csv_rows(output_dir / "review_pack_index.csv")

    assert rows
    assert all(row["candidate_id"] for row in rows)


def test_each_review_candidate_has_markdown_path(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)
    rows = _csv_rows(output_dir / "review_pack_index.csv")

    assert all(row["markdown_path"] for row in rows)
    assert (tmp_path / rows[0]["markdown_path"]).exists() or (output_dir / "markdown_cards").exists()


def test_each_review_candidate_has_csv_path(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)
    rows = _csv_rows(output_dir / "review_pack_index.csv")

    assert all(row["csv_path"] for row in rows)
    assert (output_dir / "csv_tables" / f"{rows[0]['candidate_id']}.csv").exists()


def test_review_decision_allowed_values_are_correct():
    assert render.REVIEW_DECISION_VALUES == [
        "accept_confirmed_seed_candidate",
        "accept_partial_seed_candidate",
        "reject_boundary",
        "reject_grid",
        "needs_rule_fix",
        "backlog",
        "skip",
    ]


def test_core_ok_allowed_values_are_correct():
    assert render.CORE_OK_VALUES == ["yes", "no", "unclear"]


def test_auto_excluded_does_not_enter_main_review_sheet(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)
    labels = _csv_rows(output_dir / "review_labels_template.csv")

    assert {row["candidate_id"] for row in labels} == {"phase7g_fixture_001"}


def test_does_not_generate_confirmed_seed(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)
    generated = {path.name for path in output_dir.rglob("*")}

    assert "confirmed_seed.jsonl" not in generated
    assert "partial_seed.csv" not in generated


def test_does_not_generate_production_ready(tmp_path):
    output_dir, report_dir, _ = _render_fixture(tmp_path)
    payload = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in list(output_dir.rglob("*")) + list(report_dir.rglob("*")) if path.is_file())

    assert "production_ready" not in payload


def test_value_bboxes_available_is_not_written_as_value_level_bbox(tmp_path):
    output_dir, _, _ = _render_fixture(tmp_path)
    rows = _csv_rows(output_dir / "review_pack_index.csv")

    assert all(row["value_bboxes_available"] == "false" for row in rows)
    assert all(row["source_span_granularity"] != "value_level" for row in rows)


def test_review_pack_source_does_not_read_bm25_or_milvus():
    sources = [
        Path("scripts/extraction/build_expanded_table_review_candidates.py").read_text(encoding="utf-8"),
        Path("scripts/extraction/render_expanded_table_review_pack.py").read_text(encoding="utf-8"),
        Path("scripts/extraction/export_table_review_crops.py").read_text(encoding="utf-8"),
    ]
    payload = "\n".join(sources).lower()

    assert "bm25_index.json" not in payload
    assert "phase5f_official_clean_baseline/bm25" not in payload
    assert "pymilvus" not in payload
    assert "milvusclient" not in payload
