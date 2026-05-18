from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.pipeline import _run_table_preview
from src.synbio_rag.application.table_preview import (
    TablePreviewCandidateProvider,
    adapt_table_preview_unit,
    apply_table_preview,
)
from src.synbio_rag.domain.config import RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import RetrievedChunk


ROOT = Path(__file__).resolve().parents[1]
UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)


def _normal_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="normal::1",
        doc_id="doc_0075",
        source_file="doc_0075.pdf",
        title="Normal evidence",
        section="Results",
        text="Normal retrieval evidence.",
        rerank_score=0.5,
        fusion_score=0.5,
    )


def _first_unit() -> dict:
    with UNITS_PATH.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


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


def test_table_preview_config_defaults_on():
    settings = Settings.from_env()

    assert settings.retrieval.table_preview_enabled is True
    assert settings.retrieval.table_preview_merge_enabled is True
    assert settings.retrieval.table_preview_merge_strategy == "type_aware_merge_v1"
    assert settings.retrieval.table_preview_allow_formal_citation is False


def test_table_preview_config_emergency_env_overrides():
    os.environ["TABLE_PREVIEW_ENABLED"] = "false"
    os.environ["TABLE_PREVIEW_MERGE_ENABLED"] = "false"
    os.environ["TABLE_PREVIEW_UNITS_PATH"] = str(UNITS_PATH)
    os.environ["TABLE_PREVIEW_MAX_CANDIDATES"] = "7"
    os.environ["TABLE_PREVIEW_MERGE_STRATEGY"] = "baseline_current"

    settings = Settings.from_env()

    assert settings.retrieval.table_preview_enabled is False
    assert settings.retrieval.table_preview_merge_enabled is False
    assert settings.retrieval.table_preview_merge_strategy == "baseline_current"
    assert settings.retrieval.table_preview_max_candidates == 7
    assert settings.retrieval.table_preview_units_path == str(UNITS_PATH)


def test_loader_reads_eligible_units_and_adapts_metadata():
    provider = TablePreviewCandidateProvider(str(UNITS_PATH))
    units = provider.load_units()
    chunk = adapt_table_preview_unit(units[0], score=0.42)

    assert len(units) == 274
    assert chunk.chunk_id.startswith("table_preview::")
    assert chunk.source_file == "table_preview_debug_only"
    assert chunk.metadata["object_type"] == "table_index_unit"
    assert chunk.metadata["table_preview"] is True
    assert chunk.metadata["index_unit_status"] == "preview_only"
    assert chunk.metadata["production_ready"] is False
    assert chunk.metadata["value_bboxes_available"] is False
    assert chunk.metadata["table_preview_allow_formal_citation"] is False
    assert chunk.metadata["source_csv_path"].endswith(".csv")
    assert chunk.metadata["source_pdf_crop_path"].endswith(".png")


def test_flag_off_does_not_load_or_mutate_candidates():
    class _ForbiddenProvider:
        def search(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("table preview loader should not run when disabled")

    retrieved = [_normal_chunk()]
    output, debug = apply_table_preview(
        question="Table 1 growth parameters",
        retrieved=retrieved,
        config=RetrievalConfig(table_preview_enabled=False),
        provider=_ForbiddenProvider(),  # type: ignore[arg-type]
    )

    assert output == retrieved
    assert debug["enabled"] is False
    assert debug["table_branch_executed"] is False


def test_shadow_mode_surfaces_debug_candidates_without_merge():
    config = RetrievalConfig(
        table_preview_enabled=True,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=5,
        table_preview_merge_enabled=False,
    )

    output, debug = apply_table_preview(
        question="Table 1 growth parameters in doc_0075",
        retrieved=[_normal_chunk()],
        config=config,
    )

    assert len(output) == 1
    assert debug["mode"] == "shadow"
    assert debug["candidate_count"] > 0
    assert debug["table_candidates_in_rerank_input"] is False


def test_merge_mode_adds_preview_candidates_only_for_table_like_query():
    config = RetrievalConfig(
        table_preview_enabled=True,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=5,
        table_preview_merge_enabled=True,
        table_preview_merge_max_candidates=2,
    )

    output, debug = apply_table_preview(
        question="Table 1 growth parameters in doc_0075",
        retrieved=[_normal_chunk()],
        config=config,
    )

    table_chunks = [chunk for chunk in output if chunk.metadata.get("object_type") == "table_index_unit"]
    assert debug["mode"] == "merged_preview"
    assert debug["merged_count"] > 0
    assert table_chunks
    assert all(chunk.metadata["index_unit_status"] == "preview_only" for chunk in table_chunks)


def test_non_table_query_guard_keeps_preview_out_of_rerank_input():
    config = RetrievalConfig(
        table_preview_enabled=True,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=5,
        table_preview_merge_enabled=True,
    )

    output, debug = apply_table_preview(
        question="doc_0075 abstract summary",
        retrieved=[_normal_chunk()],
        config=config,
    )

    assert len(output) == 1
    assert debug["mode"] == "merge_blocked"
    assert debug["reason"] == "non_table_query_guard"
    assert debug["table_candidates_in_rerank_input"] is False


def test_preview_table_support_is_not_formal_citation():
    unit = _first_unit()
    chunk = adapt_table_preview_unit(unit, score=0.9)
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
    assert debug["blocked_evidence_ids"] == ["E1"]
    assert debug["drop_reasons_by_evidence_id"]["E1"] == "table_preview_formal_citation_blocked"
    assert candidates[0].citation_eligible is False
