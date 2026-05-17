import csv
import json
from pathlib import Path

from scripts.extraction import build_table_index_units_v1 as build
from scripts.extraction import validate_table_index_units_v1 as validate


TOP_LEVEL_FIELDS = {
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "caption",
    "content_text_for_embedding",
    "content_markdown",
    "metadata",
    "provenance",
    "guardrail",
}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fixture_inputs(tmp_path: Path):
    csv_dir = tmp_path / "review_pack" / "csv_tables"
    markdown_dir = tmp_path / "review_pack" / "markdown_cards"
    crop_dir = tmp_path / "review_pack" / "pdf_crops"
    csv_dir.mkdir(parents=True)
    markdown_dir.mkdir(parents=True)
    crop_dir.mkdir(parents=True)

    seeds = []
    formal_rows = []
    review_rows = []
    pool_rows = []
    for index in range(1, 16):
        candidate_id = f"phase7g_candidate_{index:03d}__doc_fixture__table_1"
        seed_id = f"phase7g2_expanded_seed_{index:03d}__{candidate_id}"
        csv_path = csv_dir / f"{candidate_id}.csv"
        markdown_path = markdown_dir / f"{candidate_id}.md"
        crop_path = crop_dir / f"{candidate_id}.png"
        csv_path.write_text(
            "row_index,col_001,col_002,col_003,col_004\n"
            "1,Strain,Titer (g/L),Yield (%),Reference\n"
            "2,S1,1.0,50,10\n"
            "3,S2,2.0,60,11\n",
            encoding="utf-8",
        )
        markdown_path.write_text(f"# {candidate_id}\n", encoding="utf-8")
        crop_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        common = {
            "candidate_id": candidate_id,
            "doc_id": f"doc_{index:04d}",
            "table_id": "Table 1",
            "caption": f"Table 1. Fixture table {index}.",
            "page": str(index),
            "markdown_path": str(markdown_path),
            "csv_path": str(csv_path),
            "pdf_crop_path": str(crop_path),
            "source_span_granularity": "cell_level",
            "value_bboxes_available": "false",
            "cell_bboxes_available": "true",
        }
        seed = {
            **common,
            "seed_id": seed_id,
            "table_object_id": f"doc_{index:04d}__table_1__fixture",
            "seed_status": "confirmed_seed_with_warnings",
            "unit_or_note_ok": "warning",
            "reference_ok": "warning",
            "binding_review_mode": "global_bulk_no_obvious_issue",
            "binding_review_limitation": "no_per_cell_binding_review",
            "seed_warnings": "bulk_binding_no_per_cell_review;value_bboxes_not_available",
        }
        seeds.append(seed)
        formal_rows.append(
            {
                **common,
                "seed_id": seed_id,
                "table_object_id": seed["table_object_id"],
                "binding_review_mode": "global_bulk_no_obvious_issue",
                "binding_review_limitation": "no_per_cell_binding_review",
                "unit_or_note_ok": "warning",
                "reference_ok": "warning",
                "warnings": "bulk_binding_no_per_cell_review;value_bboxes_not_available",
            }
        )
        review_rows.append({**common, "table_object_id": seed["table_object_id"]})
        pool_rows.append({**common, "table_object_id": seed["table_object_id"]})

    rejected_seed = {
        **seeds[0],
        "seed_id": "phase7g2_expanded_seed_reject__phase7g_candidate_reject",
        "candidate_id": "phase7g_candidate_reject",
        "seed_status": "reject",
    }
    seeds.append(rejected_seed)

    expanded_seed_path = tmp_path / "expanded_table_seed.jsonl"
    formal_path = tmp_path / "formal_seed_validation_results.csv"
    review_path = tmp_path / "review_pack_index.csv"
    pool_path = tmp_path / "candidate_pool_scored.csv"
    _write_jsonl(expanded_seed_path, seeds)
    _write_csv(formal_path, formal_rows, list(formal_rows[0].keys()))
    _write_csv(review_path, review_rows, list(review_rows[0].keys()))
    _write_csv(pool_path, pool_rows, list(pool_rows[0].keys()))
    return {
        "expanded_seed_path": expanded_seed_path,
        "formal_path": formal_path,
        "review_path": review_path,
        "pool_path": pool_path,
        "csv_dir": csv_dir,
        "formal_rows": formal_rows,
    }


def _build(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    result = build.build_table_index_units(
        expanded_seed_path=inputs["expanded_seed_path"],
        formal_validation_path=inputs["formal_path"],
        review_pack_index_path=inputs["review_path"],
        candidate_pool_path=inputs["pool_path"],
        csv_tables_dir=inputs["csv_dir"],
        output_dir=tmp_path / "output",
    )
    return inputs, result


def test_only_15_phase7h_formal_seeds_are_used(tmp_path):
    inputs, result = _build(tmp_path)
    formal_ids = {row["seed_id"] for row in inputs["formal_rows"]}

    assert result["preview_seed_count"] == 15
    assert {unit["seed_id"] for unit in result["units"]} == formal_ids


def test_each_seed_has_exactly_one_table_unit(tmp_path):
    _, result = _build(tmp_path)
    table_units = [unit for unit in result["units"] if unit["unit_type"] == "table_unit"]

    assert len(table_units) == 15
    assert all(
        sum(1 for unit in table_units if unit["seed_id"] == seed_id) == 1
        for seed_id in {unit["seed_id"] for unit in result["units"]}
    )


def test_each_seed_generates_row_units(tmp_path):
    _, result = _build(tmp_path)
    seed_ids = {unit["seed_id"] for unit in result["units"]}

    assert all(
        any(unit["seed_id"] == seed_id and unit["unit_type"] == "row_unit" for unit in result["units"])
        for seed_id in seed_ids
    )


def test_cell_group_units_do_not_claim_value_level_bbox(tmp_path):
    _, result = _build(tmp_path)
    cell_group_units = [unit for unit in result["units"] if unit["unit_type"] == "cell_group_unit"]

    assert cell_group_units
    assert all(not validate.has_value_bbox_claim(unit) for unit in cell_group_units)


def test_top_level_schema_contains_only_core_fields(tmp_path):
    _, result = _build(tmp_path)

    assert all(set(unit.keys()) == TOP_LEVEL_FIELDS for unit in result["units"])


def test_content_text_for_embedding_is_nonempty(tmp_path):
    _, result = _build(tmp_path)

    assert all(unit["content_text_for_embedding"].strip() for unit in result["units"])


def test_value_bboxes_available_is_false(tmp_path):
    _, result = _build(tmp_path)

    assert all(unit["provenance"]["value_bboxes_available"] is False for unit in result["units"])


def test_production_ready_is_false(tmp_path):
    _, result = _build(tmp_path)

    assert all(unit["guardrail"]["production_ready"] is False for unit in result["units"])


def test_is_official_benchmark_seed_is_false(tmp_path):
    _, result = _build(tmp_path)

    assert all(unit["guardrail"]["is_official_benchmark_seed"] is False for unit in result["units"])


def test_partial_reject_unreviewed_do_not_enter_units(tmp_path):
    _, result = _build(tmp_path)

    assert "phase7g_candidate_reject" not in {unit["candidate_id"] for unit in result["units"]}
    assert all(unit["guardrail"]["seed_status"] == "confirmed_seed_with_warnings" for unit in result["units"])


def test_scripts_do_not_access_bm25_or_milvus():
    build_source = Path(build.__file__).read_text(encoding="utf-8")
    sources = [build_source, Path(validate.__file__).read_text(encoding="utf-8")]
    payload = "\n".join(sources)

    assert "import pymilvus" not in payload
    assert "from pymilvus" not in payload
    assert "MilvusClient(" not in payload
    assert "bm25_index.json" not in build_source
    assert "phase5f_official_clean_baseline/bm25" not in build_source
    assert "import rank_bm25" not in payload


def test_units_do_not_contain_embedding_or_retrieval_outputs(tmp_path):
    _, result = _build(tmp_path)

    assert all(not validate.has_forbidden_index_field(unit) for unit in result["units"])
