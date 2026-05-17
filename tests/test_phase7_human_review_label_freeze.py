import csv
from pathlib import Path

from scripts.extraction import build_seed_drafts_from_review_labels as build
from scripts.extraction import freeze_human_table_review_labels as freeze


LABEL_FIELDS = [
    "candidate_id",
    "review_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "review_notes",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "risk_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
]

INDEX_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "routing_status",
    "auto_score",
    "review_priority",
    "suggested_decision",
    "risk_tags",
    "table_type_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
    "source_span_granularity",
    "value_bboxes_available",
    "cell_bboxes_available",
    "warnings_summary",
]

POOL_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "routing_status",
    "auto_score",
    "review_priority",
    "suggested_decision",
    "reason_for_selection",
    "risk_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
]


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _base_row(candidate_id, decision, boundary, grid, key_values):
    return {
        "candidate_id": candidate_id,
        "review_decision": decision,
        "boundary_ok": boundary,
        "grid_ok": grid,
        "key_values_ok": key_values,
        "unit_or_note_ok": "unchecked",
        "reference_ok": "unchecked",
        "review_notes": "",
        "table_object_id": f"{candidate_id}__object",
        "doc_id": "doc_fixture",
        "table_id": "Table 1",
        "caption": "Table 1. Fixture.",
        "page": "1",
        "review_priority": "P0_quick_review",
        "suggested_decision": "likely_confirmed_candidate",
        "risk_tags": "literal_reference_table",
        "markdown_path": f"markdown_cards/{candidate_id}.md",
        "csv_path": f"csv_tables/{candidate_id}.csv",
        "pdf_crop_path": f"pdf_crops/{candidate_id}.png",
        "crop_status": "ok",
    }


def _fixture_paths(tmp_path: Path):
    review_pack = tmp_path / "review_pack"
    labels = [
        _base_row("candidate_confirmed", " accept_confirmed_seed_candidate ", " yes ", "yes", "yes"),
        _base_row("candidate_partial", "accept_partial_seed_candidate", "no", "unclear", "yes"),
        _base_row("candidate_reject", "reject_boundary", "no", "no", "no"),
        _base_row("candidate_unreviewed", "", "", "", ""),
        _base_row("candidate_invalid", "not_a_decision", "yes", "yes", "yes"),
    ]
    index_rows = []
    pool_rows = []
    for row in labels:
        index_row = {field: row.get(field, "") for field in INDEX_FIELDS}
        index_row["routing_status"] = "phase7g_pdfplumber_review"
        index_row["auto_score"] = "0.9"
        index_row["table_type_tags"] = "numeric_metric_table"
        pool_row = {field: row.get(field, "") for field in POOL_FIELDS}
        pool_row["routing_status"] = "phase7g_pdfplumber_review"
        pool_row["auto_score"] = "0.9"
        pool_row["reason_for_selection"] = "fixture"
        index_rows.append(index_row)
        pool_rows.append(pool_row)
    _write_csv(review_pack / "review_labels_template.csv", labels, LABEL_FIELDS)
    _write_csv(review_pack / "review_pack_index.csv", index_rows, INDEX_FIELDS)
    _write_csv(review_pack / "candidate_pool_scored.csv", pool_rows, POOL_FIELDS)
    return {
        "labels": review_pack / "review_labels_template.csv",
        "index": review_pack / "review_pack_index.csv",
        "pool": review_pack / "candidate_pool_scored.csv",
        "output": tmp_path / "freeze",
        "report": tmp_path / "reports",
    }


def _freeze_and_build(tmp_path: Path):
    paths = _fixture_paths(tmp_path)
    original_text = paths["labels"].read_text(encoding="utf-8")
    freeze.freeze_review_labels(
        review_labels_path=paths["labels"],
        review_pack_index_path=paths["index"],
        candidate_pool_path=paths["pool"],
        output_dir=paths["output"],
    )
    build.build_seed_drafts(
        frozen_review_labels_path=paths["output"] / "frozen_review_labels.csv",
        review_pack_index_path=paths["index"],
        candidate_pool_path=paths["pool"],
        output_dir=paths["output"],
        report_dir=paths["report"],
    )
    return paths, original_text


def test_human_label_fields_strip_whitespace(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "frozen_review_labels.csv")
    confirmed = next(row for row in rows if row["candidate_id"] == "candidate_confirmed")

    assert confirmed["review_decision"] == "accept_confirmed_seed_candidate"
    assert confirmed["boundary_ok"] == "yes"
    assert "review_decision" in confirmed["whitespace_normalized_fields"]


def test_invalid_review_decision_enters_invalid_label(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "frozen_review_labels.csv")
    invalid = next(row for row in rows if row["candidate_id"] == "candidate_invalid")

    assert invalid["label_status"] == "invalid_label"
    assert invalid["invalid_fields"] == "review_decision"


def test_complete_human_review_row_is_identified(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "frozen_review_labels.csv")
    confirmed = next(row for row in rows if row["candidate_id"] == "candidate_confirmed")
    unreviewed = next(row for row in rows if row["candidate_id"] == "candidate_unreviewed")

    assert confirmed["core_fields_complete"] == "true"
    assert unreviewed["core_fields_complete"] == "false"


def test_unreviewed_candidate_does_not_enter_confirmed_seed_draft(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")

    assert "candidate_unreviewed" not in {row["candidate_id"] for row in rows}


def test_only_accept_confirmed_with_yes_yes_yes_enters_confirmed_seed_draft(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")

    assert {row["candidate_id"] for row in rows} == {"candidate_confirmed"}


def test_confirmed_seed_draft_is_not_confirmed_seed(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")
    generated_names = {path.name for path in paths["output"].glob("*")}

    assert rows[0]["seed_draft_status"] == "confirmed_seed_draft"
    assert rows[0]["seed_draft_status"] != "confirmed_seed"
    assert "confirmed_seed.csv" not in generated_names
    assert "confirmed_seed.jsonl" not in generated_names


def test_unchecked_unit_label_is_not_written_as_unit_confirmed(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")

    assert rows[0]["unit_or_note_ok"] == "unchecked"
    assert rows[0]["unit_binding_status"] == "unchecked"


def test_unchecked_reference_label_is_not_written_as_reference_confirmed(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")

    assert rows[0]["reference_ok"] == "unchecked"
    assert rows[0]["reference_binding_status"] == "unchecked"


def test_accept_partial_seed_candidate_does_not_enter_confirmed_seed_draft(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")
    partial_rows = _read_csv(paths["output"] / "partial_candidate_routing.csv")

    assert "candidate_partial" not in {row["candidate_id"] for row in rows}
    assert {row["candidate_id"] for row in partial_rows} == {"candidate_partial"}


def test_reject_boundary_does_not_enter_seed_draft(tmp_path):
    paths, _ = _freeze_and_build(tmp_path)
    draft_rows = _read_csv(paths["output"] / "confirmed_seed_draft_candidates.csv")
    reject_rows = _read_csv(paths["output"] / "reject_boundary_feedback.csv")

    assert "candidate_reject" not in {row["candidate_id"] for row in draft_rows}
    assert {row["candidate_id"] for row in reject_rows} == {"candidate_reject"}


def test_review_labels_template_source_file_is_not_overwritten(tmp_path):
    paths, original_text = _freeze_and_build(tmp_path)

    assert paths["labels"].read_text(encoding="utf-8") == original_text


def test_new_scripts_do_not_access_bm25_or_milvus_resources():
    sources = [
        Path("scripts/extraction/freeze_human_table_review_labels.py").read_text(encoding="utf-8"),
        Path("scripts/extraction/build_seed_drafts_from_review_labels.py").read_text(encoding="utf-8"),
        Path("scripts/extraction/analyze_human_review_label_errors.py").read_text(encoding="utf-8"),
    ]
    payload = "\n".join(sources).lower()

    assert "pymilvus" not in payload
    assert "milvusclient" not in payload
    assert "bm25_index.json" not in payload
    assert "phase5f_official_clean_baseline/bm25" not in payload
