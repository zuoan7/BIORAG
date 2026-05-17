import csv
import json
from pathlib import Path

from scripts.extraction import apply_bulk_binding_review_and_build_expanded_seed as bulk


DRAFT_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "unit_binding_status",
    "reference_binding_status",
    "requires_light_binding_review",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
    "risk_tags",
    "seed_draft_status",
    "seed_draft_notes",
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

PARTIAL_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "review_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "routing_status",
    "partial_evidence_usable",
    "recommended_next_action",
    "reason",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "risk_tags",
    "review_notes",
]

REJECT_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "suggested_decision",
    "review_priority",
    "auto_score",
    "risk_tags",
    "reason",
    "recommended_scoring_feedback",
]

UNREVIEWED_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "missing_core_fields",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "risk_tags",
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


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fixture(tmp_path: Path):
    freeze_dir = tmp_path / "freeze"
    review_pack_dir = tmp_path / "review_pack"
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "reports"
    freeze_report_dir = tmp_path / "freeze_reports"
    phase7f_summary = tmp_path / "phase7f_summary.md"

    draft_rows = []
    index_rows = []
    pool_rows = []
    followup_rows = []
    frozen_rows = []
    for index in range(1, 16):
        candidate_id = f"candidate_{index:03d}"
        markdown_path = review_pack_dir / "markdown_cards" / f"{candidate_id}.md"
        csv_path = review_pack_dir / "csv_tables" / f"{candidate_id}.csv"
        pdf_crop_path = review_pack_dir / "pdf_crops" / f"{candidate_id}.png"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_crop_path.parent.mkdir(parents=True, exist_ok=True)
        reference_text = " Source Study Ref " if index == 1 else ""
        markdown_path.write_text(
            f"# {candidate_id}\nTable {index}. Yield data {reference_text}\nvalue (g/L) ± SD\n",
            encoding="utf-8",
        )
        csv_path.write_text("row_index,col_001,col_002\n1,strain,titer (g/L)\n2,A,1.0 ± 0.1\n", encoding="utf-8")
        pdf_crop_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        row = {
            "candidate_id": candidate_id,
            "table_object_id": f"{candidate_id}__object",
            "doc_id": "doc_fixture",
            "table_id": f"Table {index}",
            "caption": f"Table {index}. Fixture yield table.",
            "page": str(index),
            "review_priority": "P0_quick_review",
            "suggested_decision": "likely_confirmed_candidate",
            "boundary_ok": "yes",
            "grid_ok": "yes",
            "key_values_ok": "yes",
            "unit_or_note_ok": "no" if index == 15 else "unchecked",
            "reference_ok": "no" if index == 15 else "unchecked",
            "unit_binding_status": "unchecked",
            "reference_binding_status": "unchecked",
            "requires_light_binding_review": "true",
            "markdown_path": str(markdown_path),
            "csv_path": str(csv_path),
            "pdf_crop_path": str(pdf_crop_path),
            "crop_status": "ok",
            "risk_tags": "unit_or_footnote_table",
            "seed_draft_status": "confirmed_seed_draft",
            "seed_draft_notes": "fixture",
        }
        draft_rows.append(row)
        followup_rows.append(
            {
                "candidate_id": candidate_id,
                "unit_or_note_ok": "unchecked",
                "reference_ok": "unchecked",
                "binding_notes": "",
                "table_object_id": row["table_object_id"],
                "doc_id": row["doc_id"],
                "table_id": row["table_id"],
                "caption": row["caption"],
                "markdown_path": row["markdown_path"],
                "csv_path": row["csv_path"],
                "pdf_crop_path": row["pdf_crop_path"],
                "risk_tags": row["risk_tags"],
            }
        )
        index_rows.append(
            {
                **{field: row.get(field, "") for field in INDEX_FIELDS},
                "routing_status": "phase7g_pdfplumber_review",
                "auto_score": "0.9",
                "table_type_tags": "numeric_metric_table",
                "source_span_granularity": "cell_level",
                "value_bboxes_available": "false",
                "cell_bboxes_available": "true",
                "warnings_summary": "none",
            }
        )
        pool_rows.append(
            {
                **{field: row.get(field, "") for field in POOL_FIELDS},
                "routing_status": "phase7g_pdfplumber_review",
                "auto_score": "0.9",
                "reason_for_selection": "fixture",
            }
        )
        frozen_rows.append(
            {
                **row,
                "review_decision": "accept_confirmed_seed_candidate",
                "core_fields_complete": "true",
                "label_status": "complete_review",
            }
        )

    partial_row = {
        "candidate_id": "candidate_partial",
        "table_object_id": "partial_object",
        "doc_id": "doc_fixture",
        "table_id": "Table P",
        "review_decision": "accept_partial_seed_candidate",
        "boundary_ok": "no",
        "grid_ok": "unclear",
        "key_values_ok": "yes",
        "unit_or_note_ok": "unchecked",
        "reference_ok": "unchecked",
        "routing_status": "needs_boundary_fix",
        "partial_evidence_usable": "true",
        "recommended_next_action": "fixture",
        "reason": "fixture",
        "markdown_path": "",
        "csv_path": "",
        "pdf_crop_path": "",
        "risk_tags": "",
        "review_notes": "",
    }
    reject_row = {
        "candidate_id": "candidate_reject",
        "table_object_id": "reject_object",
        "doc_id": "doc_fixture",
        "table_id": "Table R",
        "boundary_ok": "no",
        "grid_ok": "no",
        "key_values_ok": "no",
        "suggested_decision": "likely_partial_candidate",
        "review_priority": "P0_quick_review",
        "auto_score": "0.5",
        "risk_tags": "",
        "reason": "fixture",
        "recommended_scoring_feedback": "fixture",
    }
    unreviewed_row = {
        "candidate_id": "candidate_unreviewed",
        "table_object_id": "unreviewed_object",
        "doc_id": "doc_fixture",
        "table_id": "Table U",
        "caption": "Fixture",
        "page": "1",
        "review_priority": "P1_review",
        "suggested_decision": "likely_confirmed_candidate",
        "missing_core_fields": "review_decision",
        "markdown_path": "",
        "csv_path": "",
        "pdf_crop_path": "",
        "risk_tags": "",
    }

    _write_csv(freeze_dir / "confirmed_seed_draft_candidates.csv", draft_rows, DRAFT_FIELDS)
    _write_jsonl(freeze_dir / "confirmed_seed_draft_candidates.jsonl", draft_rows)
    _write_csv(freeze_dir / "frozen_review_labels.csv", frozen_rows, list(frozen_rows[0]))
    _write_csv(freeze_dir / "partial_candidate_routing.csv", [partial_row], PARTIAL_FIELDS)
    _write_csv(freeze_dir / "reject_boundary_feedback.csv", [reject_row], REJECT_FIELDS)
    _write_csv(freeze_dir / "unreviewed_candidates.csv", [unreviewed_row], UNREVIEWED_FIELDS)
    _write_csv(freeze_dir / "unit_reference_followup_template.csv", followup_rows, list(followup_rows[0]))
    _write_csv(review_pack_dir / "review_pack_index.csv", index_rows, INDEX_FIELDS)
    _write_csv(review_pack_dir / "candidate_pool_scored.csv", pool_rows, POOL_FIELDS)

    for name in [
        "human_label_audit_report.md",
        "seed_draft_routing_report.md",
        "partial_candidate_routing_report.md",
        "phase7g_1_summary.md",
    ]:
        (freeze_report_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (freeze_report_dir / name).write_text("# fixture\n", encoding="utf-8")
    phase7f_summary.write_text("# fixture phase7f\n", encoding="utf-8")

    bulk.build_expanded_seed(
        freeze_dir=freeze_dir,
        review_pack_dir=review_pack_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        freeze_report_dir=freeze_report_dir,
        phase7f_summary=phase7f_summary,
    )
    return output_dir, report_dir


def test_15_confirmed_seed_drafts_enter_confirmed_seed_with_warnings(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")

    assert len(seeds) == 15
    assert {seed["seed_status"] for seed in seeds} == {"confirmed_seed_with_warnings"}


def test_unchecked_unit_or_note_is_replaced_by_warning_or_not_applicable(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")
    replaced = [seed for seed in seeds if seed["candidate_id"] != "candidate_015"]

    assert all(seed["unit_or_note_ok"] in {"warning", "not_applicable"} for seed in replaced)
    assert all(seed["unit_or_note_ok"] != "unchecked" for seed in replaced)


def test_unchecked_reference_is_replaced_by_warning_or_not_applicable(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")
    replaced = [seed for seed in seeds if seed["candidate_id"] != "candidate_015"]

    assert all(seed["reference_ok"] in {"warning", "not_applicable"} for seed in replaced)
    assert all(seed["reference_ok"] != "unchecked" for seed in replaced)


def test_original_no_is_not_overwritten(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")
    no_seed = next(seed for seed in seeds if seed["candidate_id"] == "candidate_015")

    assert no_seed["unit_or_note_ok"] == "no"
    assert no_seed["reference_ok"] == "no"


def test_seed_status_is_not_fully_confirmed_or_official_benchmark(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")

    assert all(seed["seed_status"] != "fully_confirmed_seed" for seed in seeds)
    assert all(seed["seed_status"] != "official_benchmark_seed" for seed in seeds)


def test_production_ready_token_does_not_appear_in_seed_outputs(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")
    for seed in seeds:
        for path_field in ["markdown_path", "csv_path", "pdf_crop_path"]:
            seed.pop(path_field, None)
    payload = json.dumps(seeds, ensure_ascii=False)

    assert "production_ready" not in payload


def test_partial_reject_and_unreviewed_do_not_enter_seed(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")
    ids = {seed["candidate_id"] for seed in seeds}

    assert "candidate_partial" not in ids
    assert "candidate_reject" not in ids
    assert "candidate_unreviewed" not in ids


def test_partial_reject_and_unreviewed_are_carried_forward(tmp_path):
    output_dir, _ = _fixture(tmp_path)

    assert len(_read_csv(output_dir / "partial_candidate_routing_carried_forward.csv")) == 1
    assert len(_read_csv(output_dir / "reject_boundary_feedback_carried_forward.csv")) == 1
    assert len(_read_csv(output_dir / "unreviewed_candidates_carried_forward.csv")) == 1


def test_value_bboxes_available_is_not_forged_true(tmp_path):
    output_dir, _ = _fixture(tmp_path)
    seeds = _read_jsonl(output_dir / "expanded_table_seed.jsonl")

    assert {seed["value_bboxes_available"] for seed in seeds} == {"false"}


def test_bulk_binding_script_does_not_access_bm25_or_milvus_resources():
    source = Path("scripts/extraction/apply_bulk_binding_review_and_build_expanded_seed.py").read_text(encoding="utf-8")
    lowered = source.lower()

    assert "pymilvus" not in lowered
    assert "milvusclient" not in lowered
    assert "bm25_index.json" not in lowered
    assert "phase5f_official_clean_baseline/bm25" not in lowered
