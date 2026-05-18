from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.pipeline import _run_table_preview
from src.synbio_rag.application.table_preview import TablePreviewCandidate, adapt_table_preview_unit
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk


PHASE_DIR = "v7_phase7_table_preview_mainchain_slim"
UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_pipeline_seam_smoke(
    *,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    citation_records: list[dict[str, Any]] = []

    records.append(_flag_off_record())
    records.append(_shadow_record())
    records.append(_merge_record())
    records.append(_priority_record("table_lookup", "Which table reports Table 1 test table?", "table_unit"))
    records.append(_priority_record("row_lookup", "Find the row evidence for A in doc_test Table 1", "row_unit"))
    records.append(_priority_record("metric_lookup", "Find metric evidence for A value in Table 1", "cell_group_unit"))
    records.append(_non_table_block_record())
    metadata_record, citation_record = _metadata_and_citation_records()
    records.append(metadata_record)
    records.append(citation_record)
    citation_records.append(citation_record)

    errors = [f"{row['scenario']}: {row['detail']}" for row in records if not row["pass"]]
    summary = {
        "validation_status": "pass" if not errors else "fail",
        "pass": not errors,
        "errors": errors,
        "scenario_count": len(records),
        "passed_count": sum(1 for row in records if row["pass"]),
        "pipeline_seam_smoke_results_path": str(results_dir / "pipeline_seam_smoke_results.csv"),
        "citation_guard_results_path": str(results_dir / "citation_guard_results.csv"),
        "guardrails": {
            "milvus_accessed": False,
            "official_bm25_accessed": False,
            "embedding_run": False,
            "qwen_or_llm_called": False,
            "production_table_index_built": False,
            "configs_modified": False,
        },
    }
    write_csv(
        results_dir / "pipeline_seam_smoke_results.csv",
        records,
        ["scenario", "pass", "detail", "mode", "reason", "merge_strategy", "query_route", "preview_count"],
    )
    write_csv(
        results_dir / "citation_guard_results.csv",
        citation_records,
        [
            "scenario",
            "pass",
            "detail",
            "chunk_id",
            "table_index_unit_id",
            "formal_table_citation_count",
            "drop_reason",
            "source_file",
            "csv_path_entered_source_file",
            "crop_path_entered_source_file",
            "production_ready",
        ],
    )
    write_json(results_dir / "pipeline_seam_smoke_summary.json", summary)
    write_report(summary, reports_dir / "pipeline_seam_smoke_report.md")
    return summary


def _flag_off_record() -> dict[str, Any]:
    provider = ForbiddenProvider()
    output, debug = _run_table_preview(
        question="Which table reports Table 1?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=False, merge_enabled=False),
        provider=provider,  # type: ignore[arg-type]
    )
    ok = (
        provider.called is False
        and debug["enabled"] is False
        and debug["table_branch_executed"] is False
        and len(_preview_chunks(output)) == 0
    )
    return _record("flag_off_no_provider_load", ok, debug, len(_preview_chunks(output)))


def _shadow_record() -> dict[str, Any]:
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])
    output, debug = _run_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=False),
        provider=provider,  # type: ignore[arg-type]
    )
    ok = (
        provider.called is True
        and debug["mode"] == "shadow"
        and debug["candidate_count"] == 1
        and debug["table_candidates_in_rerank_input"] is False
        and len(_preview_chunks(output)) == 0
    )
    return _record("shadow_debug_only", ok, debug, len(_preview_chunks(output)))


def _merge_record() -> dict[str, Any]:
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])
    output, debug = _run_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )
    ok = (
        debug["mode"] == "merged_preview"
        and debug["merge_strategy"] == "type_aware_merge_v1"
        and debug["table_candidates_in_rerank_input"] is True
        and len(_preview_chunks(output)) == 1
    )
    return _record("merge_table_query_adds_preview", ok, debug, len(_preview_chunks(output)))


def _priority_record(scenario: str, question: str, expected_unit_type: str) -> dict[str, Any]:
    units = {
        "table_lookup": [
            _unit("row_1", "row_unit", row_label="A"),
            _unit("table_1", "table_unit"),
        ],
        "row_lookup": [
            _unit("cell_1", "cell_group_unit", row_label="A"),
            _unit("row_1", "row_unit", row_label="A"),
        ],
        "metric_lookup": [
            _unit("row_1", "row_unit", row_label="A"),
            _unit("cell_1", "cell_group_unit", row_label="A"),
        ],
    }[scenario]
    scores = [0.90, 0.75] if scenario == "table_lookup" else [0.80, 0.75]
    provider = StaticProvider(units, scores)
    output, debug = _run_table_preview(
        question=question,
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True, max_merge=1),
        provider=provider,  # type: ignore[arg-type]
    )
    preview = _preview_chunks(output)
    actual_unit_type = preview[0].metadata.get("table_unit_type", "") if preview else ""
    ok = debug["query_route"] == scenario and actual_unit_type == expected_unit_type
    return _record(f"{scenario}_priority", ok, debug, len(preview), f"actual={actual_unit_type}")


def _non_table_block_record() -> dict[str, Any]:
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])
    output, debug = _run_table_preview(
        question="Summarize doc_test abstract and study motivation.",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )
    ok = (
        provider.called is True
        and debug["mode"] == "merge_blocked"
        and debug["reason"] == "non_table_query_guard"
        and debug["table_candidates_in_rerank_input"] is False
        and len(_preview_chunks(output)) == 0
    )
    return _record("non_table_query_hard_block", ok, debug, len(_preview_chunks(output)))


def _metadata_and_citation_records() -> tuple[dict[str, Any], dict[str, Any]]:
    provider = StaticProvider([_unit("table_1", "table_unit")], [0.9])
    output, debug = _run_table_preview(
        question="Which table reports Table 1 test table?",
        retrieved=[_normal_chunk()],
        config=_config(enabled=True, merge_enabled=True),
        provider=provider,  # type: ignore[arg-type]
    )
    preview = _preview_chunks(output)
    chunk = preview[0]
    metadata_ok = _preview_metadata_ok(chunk)
    metadata_record = _record("preview_metadata_preserved", metadata_ok, debug, len(preview))
    citation_record = _citation_guard_record(chunk)
    return metadata_record, citation_record


def _citation_guard_record(chunk: RetrievedChunk) -> dict[str, Any]:
    binder = CitationBinder()
    evidence_id = "E1"
    candidate = _evidence_candidate_from_chunk(evidence_id, chunk)
    support = [SupportItem(evidence_id, candidate, 0.9, ["selected_preview_table"])]
    candidates = binder.build_citation_candidates(support)
    _answer, citations, debug = binder.bind(
        "Preview-only table evidence [E1].",
        support,
        citation_candidates=candidates,
    )
    drop_reason = debug.get("drop_reasons_by_evidence_id", {}).get(evidence_id, "")
    source_files = [citation.source_file for citation in citations]
    csv_path = chunk.metadata.get("source_csv_path", "")
    crop_path = chunk.metadata.get("source_pdf_crop_path", "")
    ok = (
        len(citations) == 0
        and drop_reason == "table_preview_formal_citation_blocked"
        and csv_path not in source_files
        and crop_path not in source_files
        and chunk.metadata.get("production_ready") is False
    )
    return {
        "scenario": "citation_guard_blocks_formal_table_citation",
        "pass": ok,
        "detail": f"drop_reason={drop_reason}",
        "mode": "",
        "reason": drop_reason,
        "merge_strategy": "type_aware_merge_v1",
        "query_route": "",
        "preview_count": 1,
        "chunk_id": chunk.chunk_id,
        "table_index_unit_id": chunk.metadata.get("table_index_unit_id", ""),
        "formal_table_citation_count": len(citations),
        "drop_reason": drop_reason,
        "source_file": chunk.source_file,
        "csv_path_entered_source_file": csv_path in source_files,
        "crop_path_entered_source_file": crop_path in source_files,
        "production_ready": chunk.metadata.get("production_ready"),
    }


def _record(
    scenario: str,
    ok: bool,
    debug: dict[str, Any],
    preview_count: int,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "pass": ok,
        "detail": detail or debug.get("reason", ""),
        "mode": debug.get("mode", ""),
        "reason": debug.get("reason", ""),
        "merge_strategy": debug.get("merge_strategy", ""),
        "query_route": debug.get("query_route", ""),
        "preview_count": preview_count,
    }


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


def _preview_metadata_ok(chunk: RetrievedChunk) -> bool:
    metadata = chunk.metadata
    return (
        metadata.get("object_type") == "table_index_unit"
        and metadata.get("table_preview") is True
        and metadata.get("index_unit_status") == "preview_only"
        and metadata.get("production_ready") is False
        and metadata.get("value_bboxes_available") is False
        and metadata.get("table_preview_allow_formal_citation") is False
        and metadata.get("citation_formal_allowed") is False
        and bool(metadata.get("source_csv_path"))
        and bool(metadata.get("source_pdf_crop_path"))
    )


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


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7W-slim Pipeline Seam Smoke",
        "",
        "This smoke used fake normal chunks, fake preview provider, stub rerank/support behavior, and CitationBinder only.",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- scenario_count: {summary['scenario_count']}",
        f"- passed_count: {summary['passed_count']}",
        f"- errors: {summary['errors']}",
        "",
        "## Guardrails",
        "",
        "- Milvus accessed: no",
        "- Official BM25 accessed: no",
        "- Embedding run: no",
        "- Qwen / LLM / RAGAS / OCR / VLM run: no",
        "- Production table index built: no",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7W-slim pipeline seam smoke.")
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    args = parser.parse_args()
    summary = run_pipeline_seam_smoke(results_dir=args.results_dir, reports_dir=args.reports_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
