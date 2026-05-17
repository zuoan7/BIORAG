import pytest

from scripts.extraction import review_hybrid_binding_candidates as review


def _candidate_rows():
    rows = []
    for hybrid_id in sorted(review.EXPECTED_BINDING_REVIEW_IDS):
        doc_id, _, table_part = hybrid_id.partition("__table_")
        table_number = table_part.split("__", 1)[0]
        rows.append(
            {
                "hybrid_table_object_id": hybrid_id,
                "doc_id": doc_id,
                "table_id": f"Table {table_number}",
                "final_case_action": "manual_review_binding",
                "review_evidence_summary": "fixture",
            }
        )
    return rows


def _hybrid_objects():
    objects = []
    for index, hybrid_id in enumerate(sorted(review.EXPECTED_BINDING_REVIEW_IDS), start=1):
        objects.append(
            {
                "table_object_id": hybrid_id,
                "doc_id": hybrid_id.split("__", 1)[0],
                "table_id": "fixture",
                "rows": [{"row_text": f"fixture row {index}"}],
                "hybrid_metadata": {
                    "original_chunk_table_object_id": f"chunk_{index}",
                    "pdfplumber_table_id": f"pdf_fixture_{index}",
                    "pdf_page": index,
                    "pdfplumber_strategy": "text",
                    "alignment_status": "matched",
                    "alignment_confidence": "high",
                    "source_span_granularity": "cell_level",
                    "cell_bboxes_available": True,
                    "value_bboxes_available": False,
                },
            }
        )
    return objects


def _raw_tables():
    tables = []
    for index in range(1, 6):
        tables.append(
            {
                "pdfplumber_table_id": f"pdf_fixture_{index}",
                "rows": [["header", "value"], ["row", str(index)]],
            }
        )
    return tables


def _alignment_rows():
    rows = []
    for index in range(1, 6):
        rows.append(
            {
                "chunk_table_object_id": f"chunk_{index}",
                "alignment_status": "matched",
                "alignment_confidence": "high",
                "layout_quality_status": "usable",
            }
        )
    return rows


def _review_rows():
    return review.build_review_rows(_candidate_rows(), _hybrid_objects(), _raw_tables(), _alignment_rows())


def test_only_processes_five_binding_review_candidates():
    rows = _review_rows()

    assert len(rows) == 5
    assert {row["hybrid_table_object_id"] for row in rows} == review.EXPECTED_BINDING_REVIEW_IDS

    extra = _candidate_rows() + [
        {
            "hybrid_table_object_id": "doc_extra__table_1__phase7c2_hybrid_99",
            "doc_id": "doc_extra",
            "table_id": "Table 1",
            "final_case_action": "manual_review_binding",
        }
    ]
    with pytest.raises(ValueError):
        review.build_review_rows(extra, _hybrid_objects(), _raw_tables(), _alignment_rows())


def test_ready_for_gold_candidate_is_not_confirmed_gold():
    rows = _review_rows()
    ready = [row for row in rows if row["final_binding_action"] == "ready_for_gold_candidate"]

    assert ready
    assert all(row["ready_for_gold_candidate_is_confirmed_gold"] is False for row in ready)
    assert all(row["confirmed_gold"] is False for row in rows)
    assert all(row["usable_hybrid_candidate_is_production_ready"] is False for row in rows)


def test_value_bbox_false_never_writes_value_level_provenance():
    rows = _review_rows()

    assert all(row["value_bboxes_available"] is False for row in rows)
    assert all(row["value_level_provenance_used"] is False for row in rows)
    assert all(row["bbox_provenance_level"] != "value_level" for row in rows)


def test_unit_visible_does_not_mean_unit_bound():
    rows = _review_rows()

    assert any(row["unit_visible"] and not row["unit_bound"] for row in rows)


def test_footnote_present_does_not_mean_footnote_bound():
    rows = _review_rows()

    assert any(row["footnote_present"] and not row["footnote_bound"] for row in rows)


def test_reference_visible_does_not_mean_row_level_reference_bound():
    rows = _review_rows()

    assert any(row["reference_visible"] and not row["row_level_reference_bound"] for row in rows)


def test_final_binding_action_uses_allowed_values():
    rows = _review_rows()

    assert {row["final_binding_action"] for row in rows} <= review.FINAL_BINDING_ACTIONS


def test_candidates_split_into_ready_rule_fix_and_fallback_backlog():
    rows = _review_rows()
    ready, rule_fix, fallback = review.split_rows(rows)

    assert {row["hybrid_table_object_id"] for row in ready} == {
        "doc_0468__table_2__phase7c2_hybrid_01",
        "doc_0687__table_3__phase7c2_hybrid_03",
    }
    assert {row["hybrid_table_object_id"] for row in rule_fix} == {
        "doc_0598__table_1__phase7c2_hybrid_01",
        "doc_0687__table_2__phase7c2_hybrid_02",
        "doc_0523__table_1__phase7c2_hybrid_01",
    }
    assert fallback == []
