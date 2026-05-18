from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.evaluation.phase7v_fast_validate import validate_phase7v_fast
from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.table_preview import (
    TablePreviewCandidate,
    adapt_table_preview_unit,
    apply_table_preview,
)
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk


ROOT = Path(__file__).resolve().parents[1]
UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)


class StaticProvider:
    def __init__(self, units: list[dict[str, Any]], scores: list[float]) -> None:
        self.called = False
        self.candidates = [
            TablePreviewCandidate(chunk=adapt_table_preview_unit(unit, score=score), score=score)
            for unit, score in zip(units, scores)
        ]
        self.last_debug = {"unit_count": len(self.candidates)}
        for rank, candidate in enumerate(self.candidates, start=1):
            candidate.rank = rank
            candidate.chunk.metadata["table_preview_rank"] = rank

    def search(self, question: str, *, top_k: int) -> list[TablePreviewCandidate]:
        self.called = True
        return self.candidates[:top_k]


class ForbiddenProvider:
    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("table preview loader should not run when disabled")


def _unit(unit_id: str, unit_type: str, *, row_label: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"page": "1", "header_path": [["A"]]}
    if row_label is not None:
        metadata["row_label"] = row_label
    return {
        "table_index_unit_id": unit_id,
        "unit_type": unit_type,
        "doc_id": "doc_test",
        "table_id": "Table 1",
        "caption": "[TABLE CAPTION] Table 1. Test table.",
        "content_text_for_embedding": f"In doc_test Table 1 {unit_type} {row_label or ''} reports value.",
        "metadata": metadata,
        "provenance": {
            "source_csv_path": "debug/table.csv",
            "source_pdf_crop_path": "debug/table.png",
            "value_bboxes_available": False,
            "cell_bboxes_available": True,
        },
        "guardrail": {
            "production_ready": False,
            "index_unit_status": "preview_only",
            "unit_or_note_ok": "warning",
            "reference_ok": "warning",
        },
        "seed_id": "seed_1",
        "candidate_id": "candidate_1",
    }


def _config(*, enabled: bool = True, type_aware: bool = True, max_merge: int = 1) -> RetrievalConfig:
    config = RetrievalConfig(
        table_preview_enabled=enabled,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_enabled=True,
        table_preview_merge_max_candidates=max_merge,
        table_preview_min_score=0.01,
    )
    if type_aware:
        setattr(config, "table_preview_type_aware_merge_enabled", True)
    return config


def _normal_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="normal::1",
        doc_id="normal_doc",
        source_file="normal.pdf",
        title="Normal",
        section="Abstract",
        text="Normal evidence.",
        metadata={"object_type": "normal_chunk"},
    )


def _merged_unit_ids(output: list[RetrievedChunk]) -> list[str]:
    return [
        chunk.metadata.get("table_index_unit_id", "")
        for chunk in output
        if chunk.metadata.get("object_type") == "table_index_unit"
    ]


def test_table_lookup_prefers_table_unit_over_higher_scored_row_unit():
    provider = StaticProvider(
        [_unit("row_1", "row_unit", row_label="A"), _unit("table_1", "table_unit")],
        [0.90, 0.75],
    )

    output, debug = apply_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(max_merge=1),
        provider=provider,
    )

    assert debug["merge_strategy"] == "type_aware_merge_v1"
    assert debug["query_route"] == "table_lookup"
    assert _merged_unit_ids(output) == ["table_1"]


def test_row_lookup_prefers_row_unit_over_cell_group_unit():
    provider = StaticProvider(
        [
            _unit("cell_1", "cell_group_unit", row_label="A"),
            _unit("row_1", "row_unit", row_label="A"),
        ],
        [0.80, 0.75],
    )

    output, debug = apply_table_preview(
        question="Find the row evidence for A in doc_test Table 1",
        retrieved=[_normal_chunk()],
        config=_config(max_merge=1),
        provider=provider,
    )

    assert debug["query_route"] == "row_lookup"
    assert _merged_unit_ids(output) == ["row_1"]


def test_metric_lookup_prefers_cell_group_unit_over_row_unit():
    provider = StaticProvider(
        [
            _unit("row_1", "row_unit", row_label="A"),
            _unit("cell_1", "cell_group_unit", row_label="A"),
        ],
        [0.80, 0.75],
    )

    output, debug = apply_table_preview(
        question="Find metric evidence for A: value in Table 1",
        retrieved=[_normal_chunk()],
        config=_config(max_merge=1),
        provider=provider,
    )

    assert debug["query_route"] == "metric_lookup"
    assert _merged_unit_ids(output) == ["cell_1"]


def test_non_table_query_does_not_merge_preview_candidates():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])

    output, debug = apply_table_preview(
        question="Summarize doc_test abstract and study motivation.",
        retrieved=[_normal_chunk()],
        config=_config(max_merge=1),
        provider=provider,
    )

    assert provider.called is True
    assert debug["mode"] == "merge_blocked"
    assert debug["reason"] == "non_table_query_guard"
    assert _merged_unit_ids(output) == []


def test_preview_metadata_is_preserved_after_type_aware_merge():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])

    output, _debug = apply_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(max_merge=1),
        provider=provider,
    )
    table_chunk = [chunk for chunk in output if chunk.metadata.get("object_type") == "table_index_unit"][0]

    assert table_chunk.metadata["object_type"] == "table_index_unit"
    assert table_chunk.metadata["table_preview"] is True
    assert table_chunk.metadata["index_unit_status"] == "preview_only"
    assert table_chunk.metadata["production_ready"] is False
    assert table_chunk.metadata["value_bboxes_available"] is False
    assert table_chunk.metadata["source_csv_path"].endswith(".csv")
    assert table_chunk.metadata["source_pdf_crop_path"].endswith(".png")


def test_citation_guard_blocks_preview_formal_citation_after_type_aware_merge():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])
    output, _debug = apply_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(max_merge=1),
        provider=provider,
    )
    chunk = [chunk for chunk in output if chunk.metadata.get("object_type") == "table_index_unit"][0]
    candidate = EvidenceCandidate(
        evidence_id="E1",
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        title=chunk.title,
        section=chunk.section,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score,
        rerank_score=chunk.rerank_score,
        fusion_score=chunk.fusion_score,
        metadata=dict(chunk.metadata),
        features={},
        reasons=["table_preview"],
    )
    support = [SupportItem("E1", candidate, 0.9, ["selected"])]
    binder = CitationBinder()

    candidates = binder.build_citation_candidates(support)
    answer, citations, debug = binder.bind("Preview evidence [E1].", support, citation_candidates=candidates)

    assert citations == []
    assert "[1]" not in answer
    assert debug["drop_reasons_by_evidence_id"]["E1"] == "table_preview_formal_citation_blocked"
    assert candidates[0].citation_eligible is False


def test_flag_off_does_not_load_table_units():
    output, debug = apply_table_preview(
        question="Which table reports Table 1?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=False),
        provider=ForbiddenProvider(),  # type: ignore[arg-type]
    )

    assert len(output) == 1
    assert debug["enabled"] is False
    assert debug["table_branch_executed"] is False


def test_phase7v_validation_summary_passes_or_warns(tmp_path):
    validation = validate_phase7v_fast(
        fixture_path=tmp_path / "ab_query_fixture.jsonl",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        tests_result="pytest validation smoke",
    )

    assert validation["validation_status"] in {"pass", "pass_with_warnings"}
    assert validation["ab_summary"]["type_aware_merge_v1"]["merge_expected_hit_at_5"] >= (
        validation["ab_summary"]["baseline_current"]["merge_expected_hit_at_5"]
    )
    assert validation["ab_summary"]["type_aware_merge_v1"]["non_table_preview_leak_count"] == 0
    assert validation["citation_guard_regression"]["formal_citation_count"] == 0
    assert validation["rollback_regression"]["pass"] is True
