from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.pipeline import _run_table_preview
from src.synbio_rag.application.table_preview import (
    TablePreviewCandidate,
    adapt_table_preview_unit,
    apply_table_preview,
)
from src.synbio_rag.domain.config import RetrievalConfig, Settings
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
    def __init__(self) -> None:
        self.called = False

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.called = True
        raise AssertionError("table preview provider must not run when disabled")


@pytest.fixture(autouse=True)
def _clear_table_preview_env():
    keys = [key for key in os.environ if key.startswith("TABLE_PREVIEW_")]
    old_values = {key: os.environ[key] for key in keys}
    for key in keys:
        del os.environ[key]
    yield
    for key in [key for key in os.environ if key.startswith("TABLE_PREVIEW_")]:
        del os.environ[key]
    os.environ.update(old_values)


def test_table_preview_merge_strategy_defaults_to_type_aware_merge_v1():
    settings = Settings.from_env()

    assert RetrievalConfig().table_preview_enabled is True
    assert RetrievalConfig().table_preview_merge_enabled is True
    assert RetrievalConfig().table_preview_merge_strategy == "type_aware_merge_v1"
    assert settings.retrieval.table_preview_enabled is True
    assert settings.retrieval.table_preview_merge_enabled is True
    assert settings.retrieval.table_preview_merge_strategy == "type_aware_merge_v1"


def test_table_preview_merge_strategy_env_sets_type_aware_merge_v1():
    os.environ["TABLE_PREVIEW_MERGE_STRATEGY"] = "type_aware_merge_v1"

    settings = Settings.from_env()

    assert settings.retrieval.table_preview_merge_strategy == "type_aware_merge_v1"


def test_flag_off_does_not_load_preview_units_through_pipeline_seam():
    provider = ForbiddenProvider()
    output, debug = _run_table_preview(
        question="Which table reports Table 1?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=False, merge_enabled=False),
        provider=provider,  # type: ignore[arg-type]
    )

    assert provider.called is False
    assert len(output) == 1
    assert debug["enabled"] is False
    assert debug["table_branch_executed"] is False


def test_shadow_mode_does_not_enter_rerank_input():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])

    output, debug = _run_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=False),
        provider=provider,  # type: ignore[arg-type]
    )

    assert provider.called is True
    assert debug["mode"] == "shadow"
    assert debug["merge_strategy"] == "type_aware_merge_v1"
    assert debug["table_candidates_in_rerank_input"] is False
    assert _preview_chunks(output) == []


def test_merge_mode_table_like_query_adds_preview_candidates():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])

    output, debug = _run_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )

    assert debug["mode"] == "merged_preview"
    assert debug["merge_strategy"] == "type_aware_merge_v1"
    assert debug["table_candidates_in_rerank_input"] is True
    assert len(_preview_chunks(output)) == 1


def test_table_lookup_prefers_table_unit():
    provider = StaticProvider(
        [_unit("row_1", "row_unit", row_label="A"), _unit("table_1", "table_unit")],
        [0.90, 0.75],
    )

    output, debug = apply_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True, max_merge=1),
        provider=provider,  # type: ignore[arg-type]
    )

    assert debug["query_route"] == "table_lookup"
    assert _merged_unit_types(output) == ["table_unit"]


def test_row_lookup_prefers_row_unit():
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
        config=_config(enabled=True, merge_enabled=True, max_merge=1),
        provider=provider,  # type: ignore[arg-type]
    )

    assert debug["query_route"] == "row_lookup"
    assert _merged_unit_types(output) == ["row_unit"]


def test_metric_lookup_prefers_cell_group_unit():
    provider = StaticProvider(
        [
            _unit("row_1", "row_unit", row_label="A"),
            _unit("cell_1", "cell_group_unit", row_label="A"),
        ],
        [0.80, 0.75],
    )

    output, debug = apply_table_preview(
        question="Find metric evidence for A value in Table 1",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True, max_merge=1),
        provider=provider,  # type: ignore[arg-type]
    )

    assert debug["query_route"] == "metric_lookup"
    assert _merged_unit_types(output) == ["cell_group_unit"]


def test_non_table_query_does_not_merge_preview_candidates():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])

    output, debug = apply_table_preview(
        question="Summarize doc_test abstract and study motivation.",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )

    assert provider.called is True
    assert debug["mode"] == "merge_blocked"
    assert debug["reason"] == "non_table_query_guard"
    assert debug["table_candidates_in_rerank_input"] is False
    assert _preview_chunks(output) == []


def test_preview_metadata_is_preserved():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])

    output, _debug = apply_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )
    chunk = _preview_chunks(output)[0]

    assert chunk.metadata["object_type"] == "table_index_unit"
    assert chunk.metadata["table_preview"] is True
    assert chunk.metadata["index_unit_status"] == "preview_only"
    assert chunk.metadata["production_ready"] is False
    assert chunk.metadata["value_bboxes_available"] is False
    assert chunk.metadata["source_csv_path"].endswith(".csv")
    assert chunk.metadata["source_pdf_crop_path"].endswith(".png")
    assert chunk.metadata["table_preview_merge_strategy"] == "type_aware_merge_v1"


def test_citation_guard_blocks_formal_preview_table_citation():
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])
    output, _debug = apply_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )
    chunk = _preview_chunks(output)[0]
    candidate = _evidence_candidate_from_chunk("E1", chunk)
    support = [SupportItem("E1", candidate, 0.9, ["selected_preview_table"])]
    binder = CitationBinder()

    candidates = binder.build_citation_candidates(support)
    answer, citations, debug = binder.bind(
        "Preview-only table evidence [E1].",
        support,
        citation_candidates=candidates,
    )

    assert citations == []
    assert "[1]" not in answer
    assert debug["drop_reasons_by_evidence_id"]["E1"] == "table_preview_formal_citation_blocked"
    assert candidates[0].citation_eligible is False
    assert chunk.metadata["source_csv_path"] not in [citation.source_file for citation in citations]
    assert chunk.metadata["source_pdf_crop_path"] not in [citation.source_file for citation in citations]


def test_configs_directory_unmodified():
    result = subprocess.run(
        ["git", "status", "--short", "--", "configs"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def _config(
    *,
    enabled: bool,
    merge_enabled: bool,
    max_merge: int = 5,
) -> RetrievalConfig:
    return RetrievalConfig(
        table_preview_enabled=enabled,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_enabled=merge_enabled,
        table_preview_merge_strategy="type_aware_merge_v1",
        table_preview_merge_max_candidates=max_merge,
        table_preview_min_score=0.01,
        table_preview_allow_formal_citation=False,
    )


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
            "source_markdown_path": "debug/table.md",
            "value_bboxes_available": False,
            "cell_bboxes_available": True,
        },
        "guardrail": {
            "production_ready": False,
            "index_unit_status": "preview_only",
            "unit_or_note_ok": "warning",
            "reference_ok": "warning",
        },
        "seed_id": f"seed_{unit_id}",
        "candidate_id": f"candidate_{unit_id}",
    }


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


def _preview_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.metadata.get("object_type") == "table_index_unit"]


def _merged_unit_types(chunks: list[RetrievedChunk]) -> list[str]:
    return [
        str(chunk.metadata.get("table_unit_type", ""))
        for chunk in _preview_chunks(chunks)
    ]


def _evidence_candidate_from_chunk(evidence_id: str, chunk: RetrievedChunk) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
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
        reasons=["phase7w_table_preview"],
    )
