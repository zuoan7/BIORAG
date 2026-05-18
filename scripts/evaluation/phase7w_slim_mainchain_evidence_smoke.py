from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.table_preview import apply_table_preview
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk


PHASE_DIR = "v7_phase7_table_preview_mainchain_slim"
UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
PHASE7U_FIXTURE_PATH = ROOT / "data/experiments/v7_phase7_table_preview_eval_smoke/query_fixture.jsonl"
FIXTURE_PATH = ROOT / f"data/experiments/{PHASE_DIR}/query_fixture.jsonl"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"

CORE_TABLE_QUERY_TYPES = {"table_lookup", "row_lookup", "metric_lookup"}
TABLE_QUERY_TYPES = CORE_TABLE_QUERY_TYPES | {"source_or_reference_lookup"}
FIELDNAMES = [
    "query_id",
    "query_type",
    "mode",
    "table_preview_enabled",
    "merge_strategy",
    "table_candidate_count",
    "merged_preview_count",
    "rerank_input_preview_count",
    "support_preview_count",
    "expected_table_hit",
    "expected_unit_type_hit",
    "non_table_preview_leak",
    "formal_table_citation_count",
    "debug_csv_crop_path_visible_only",
    "flag_off_restored",
    "evidence_improvement_label",
    "query_route",
    "reason",
    "expected_table_index_unit_id",
    "expected_unit_type",
    "merged_table_index_unit_ids",
    "merged_unit_types",
    "support_table_index_unit_ids",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_preview_units(units_path: Path = UNITS_PATH) -> list[dict[str, Any]]:
    return load_jsonl(units_path)


def build_query_fixture(
    *,
    source_fixture_path: Path = PHASE7U_FIXTURE_PATH,
    output_path: Path = FIXTURE_PATH,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    source_rows = load_jsonl(source_fixture_path)
    selected: list[dict[str, Any]] = []
    quotas = {"table_lookup": 3, "row_lookup": 3, "metric_lookup": 3}
    for query_type, quota in quotas.items():
        selected.extend([row for row in source_rows if row.get("query_type") == query_type][:quota])

    controls = [row for row in source_rows if row.get("query_type") == "non_table_control"]
    controls.extend(_extra_non_table_controls(start=len(controls) + 1))
    selected.extend(controls[:5])

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        query_type = row.get("query_type", "non_table_control")
        rows.append(
            {
                "query_id": f"phase7w_slim_query_{idx:03d}",
                "source_query_id": row.get("query_id") or row.get("source_query_id", ""),
                "query_text": row["query_text"],
                "query_type": query_type,
                "expected_doc_id": row.get("expected_doc_id", ""),
                "expected_table_id": row.get("expected_table_id", ""),
                "expected_table_index_unit_id": row.get("expected_table_index_unit_id", ""),
                "expected_unit_type": row.get("expected_unit_type", "none"),
                "expected_row_label": row.get("expected_row_label", ""),
                "query_notes": "phase7w slim main-chain evidence smoke fixture",
            }
        )

    write_jsonl(output_path, rows)
    summary = validate_fixture_payload(rows=rows, units=load_preview_units(), fixture_path=output_path)
    summary["source_fixture_path"] = str(source_fixture_path)
    if summary_path is None:
        summary_path = RESULTS_DIR / "query_fixture_summary.json"
    write_json(summary_path, summary)
    return summary


def validate_fixture_payload(
    *,
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    fixture_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(row.get("query_type", "") for row in rows)
    unit_ids = {str(unit.get("table_index_unit_id", "")) for unit in units}
    if not 10 <= len(rows) <= 15:
        errors.append(f"expected 10-15 fixture queries, got {len(rows)}")
    for query_type in CORE_TABLE_QUERY_TYPES:
        if counts.get(query_type, 0) != 3:
            errors.append(f"expected 3 {query_type} queries, got {counts.get(query_type, 0)}")
    if counts.get("non_table_control", 0) != 5:
        errors.append(f"expected 5 non_table_control queries, got {counts.get('non_table_control', 0)}")
    if counts.get("source_or_reference_lookup", 0) > 1:
        errors.append("expected at most 1 source_or_reference_lookup query")
    for row in rows:
        if row.get("query_type") in TABLE_QUERY_TYPES:
            expected_unit_id = str(row.get("expected_table_index_unit_id", ""))
            if expected_unit_id not in unit_ids:
                errors.append(f"{row.get('query_id')} expected unit not found in preview units")
            if row.get("expected_unit_type") not in {"table_unit", "row_unit", "cell_group_unit"}:
                errors.append(f"{row.get('query_id')} has invalid expected_unit_type")
        elif row.get("query_type") == "non_table_control":
            if row.get("expected_unit_type") != "none":
                errors.append(f"{row.get('query_id')} non-table control must use expected_unit_type=none")
        else:
            errors.append(f"{row.get('query_id')} unknown query_type={row.get('query_type')!r}")
    return {
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "preview_unit_count": len(units),
        "query_count": len(rows),
        "query_type_counts": dict(sorted(counts.items())),
        "table_like_query_count": sum(counts.get(query_type, 0) for query_type in TABLE_QUERY_TYPES),
        "non_table_control_count": counts.get("non_table_control", 0),
    }


def ensure_fixture(fixture_path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    if not fixture_path.exists():
        build_query_fixture(output_path=fixture_path)
    return load_jsonl(fixture_path)


def preview_config(*, enabled: bool, merge_enabled: bool, strategy: str) -> RetrievalConfig:
    return RetrievalConfig(
        table_preview_enabled=enabled,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_enabled=merge_enabled,
        table_preview_merge_strategy=strategy,
        table_preview_merge_max_candidates=5,
        table_preview_min_score=0.05,
        table_preview_allow_formal_citation=False,
        rerank_score_floor_ratio=0.0,
    )


def normal_retrieved(query_id: str, expected_doc_id: str = "") -> list[RetrievedChunk]:
    doc_id = expected_doc_id or "normal_doc"
    return [
        RetrievedChunk(
            chunk_id=f"normal::{query_id}",
            doc_id=doc_id,
            source_file=f"{doc_id or 'normal_doc'}.pdf",
            title="Normal retrieval stub",
            section="Abstract",
            text="Normal retrieval evidence stub for Phase7W-slim main-chain smoke.",
            vector_score=0.2,
            bm25_score=0.0,
            rerank_score=0.0,
            fusion_score=0.2,
            metadata={"object_type": "normal_chunk", "phase7w_stub_normal": True},
        )
    ]


def preview_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.metadata.get("object_type") == "table_index_unit"]


def run_mainchain_evidence_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    queries = ensure_fixture(fixture_path)
    records: list[dict[str, Any]] = []
    by_query: dict[str, dict[str, dict[str, Any]]] = {}

    for query in queries:
        normal_record = _run_mode(query=query, mode="normal_only")
        preview_record = _run_mode(query=query, mode="table_preview_type_aware")
        by_query[query["query_id"]] = {
            "normal_only": normal_record,
            "table_preview_type_aware": preview_record,
        }
        records.extend([normal_record, preview_record])

    for query in queries:
        pair = by_query[query["query_id"]]
        normal_hit = bool(pair["normal_only"]["expected_table_hit"])
        preview_hit = bool(pair["table_preview_type_aware"]["expected_table_hit"])
        pair["normal_only"]["evidence_improvement_label"] = "not_applicable"
        if query["query_type"] not in TABLE_QUERY_TYPES:
            label = "not_applicable"
        elif preview_hit and not normal_hit:
            label = "preview_better"
        elif preview_hit == normal_hit:
            label = "preview_same"
        else:
            label = "preview_worse"
        pair["table_preview_type_aware"]["evidence_improvement_label"] = label

    preview_records = [row for row in records if row["mode"] == "table_preview_type_aware"]
    normal_records = [row for row in records if row["mode"] == "normal_only"]
    table_preview_records = [row for row in preview_records if row["query_type"] in TABLE_QUERY_TYPES]
    non_table_preview_records = [row for row in preview_records if row["query_type"] == "non_table_control"]
    preview_chunk_rows = [row for row in preview_records if int(row["rerank_input_preview_count"]) > 0]
    metadata_ok_count = sum(1 for row in preview_records if row.get("preview_metadata_preserved", True))
    merge_count = sum(1 for row in table_preview_records if int(row["merged_preview_count"]) > 0)
    non_table_leak_count = sum(1 for row in non_table_preview_records if row["non_table_preview_leak"])
    formal_citation_count = sum(int(row["formal_table_citation_count"]) for row in preview_records)
    debug_path_safe = all(bool(row["debug_csv_crop_path_visible_only"]) for row in preview_records)
    flag_off_restored = all(bool(row["flag_off_restored"]) for row in normal_records + preview_records)
    pass_conditions = {
        "table_like_query_preview_merge_rate_ge_90": _rate(merge_count, len(table_preview_records)) >= 0.9,
        "non_table_preview_leak_zero": non_table_leak_count == 0,
        "formal_table_citation_count_zero": formal_citation_count == 0,
        "preview_metadata_preserved": metadata_ok_count == len(preview_records),
        "debug_csv_crop_not_formal_source": debug_path_safe,
        "flag_off_restored": flag_off_restored,
        "pipeline_seam_not_crashed": True,
    }
    errors = [name for name, ok in pass_conditions.items() if not ok]
    summary = {
        "validation_status": "pass" if not errors else "fail",
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "query_count": len(queries),
        "record_count": len(records),
        "table_like_query_count": len(table_preview_records),
        "non_table_control_count": len(non_table_preview_records),
        "table_like_query_preview_merge_rate": _rate(merge_count, len(table_preview_records)),
        "non_table_preview_leak_count": non_table_leak_count,
        "formal_table_citation_count": formal_citation_count,
        "metadata_preservation_rate": _rate(metadata_ok_count, len(preview_records)),
        "debug_csv_crop_path_visible_only": debug_path_safe,
        "flag_off_restored": flag_off_restored,
        "preview_chunk_record_count": len(preview_chunk_rows),
        "evidence_improvement_counts": dict(
            Counter(row["evidence_improvement_label"] for row in preview_records)
        ),
        "pass_conditions": pass_conditions,
        "guardrails": _guardrails(),
        "records_path": str(results_dir / "mainchain_evidence_ab_results.csv"),
    }
    write_csv(results_dir / "mainchain_evidence_ab_results.csv", records, FIELDNAMES)
    write_json(results_dir / "mainchain_evidence_ab_summary.json", summary)
    write_report(summary, reports_dir / "mainchain_evidence_smoke_report.md")
    return summary


def _run_mode(*, query: dict[str, Any], mode: str) -> dict[str, Any]:
    enabled = mode == "table_preview_type_aware"
    config = preview_config(
        enabled=enabled,
        merge_enabled=enabled,
        strategy="type_aware_merge_v1" if enabled else "baseline_current",
    )
    input_chunks = normal_retrieved(query["query_id"], query.get("expected_doc_id", ""))
    output, debug = apply_table_preview(
        question=query["query_text"],
        retrieved=input_chunks,
        config=config,
    )
    reranked = stub_rerank(output)
    support = stub_support_selector(reranked)
    table_output = preview_chunks(output)
    support_preview = preview_chunks(support)
    expected_unit_id = query.get("expected_table_index_unit_id", "")
    expected_unit_type = query.get("expected_unit_type", "")
    merged_ids = [chunk.metadata.get("table_index_unit_id", "") for chunk in table_output]
    merged_unit_types = [chunk.metadata.get("table_unit_type", "") for chunk in table_output]
    citation_guard = run_citation_guard(table_output[:3])
    restored_output, restored_debug = apply_table_preview(
        question=query["query_text"],
        retrieved=input_chunks,
        config=preview_config(enabled=False, merge_enabled=False, strategy="baseline_current"),
    )
    flag_off_restored = (
        restored_debug.get("enabled") is False
        and [chunk.chunk_id for chunk in restored_output] == [chunk.chunk_id for chunk in input_chunks]
        and not preview_chunks(restored_output)
    )
    non_table_leak = query["query_type"] == "non_table_control" and len(table_output) > 0
    return {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "mode": mode,
        "table_preview_enabled": enabled,
        "merge_strategy": debug.get("merge_strategy", ""),
        "table_candidate_count": debug.get("candidate_count", 0),
        "merged_preview_count": debug.get("merged_count", 0),
        "rerank_input_preview_count": len(preview_chunks(reranked)),
        "support_preview_count": len(support_preview),
        "expected_table_hit": bool(expected_unit_id and expected_unit_id in merged_ids),
        "expected_unit_type_hit": bool(
            expected_unit_type != "none" and expected_unit_type in merged_unit_types
        ),
        "non_table_preview_leak": non_table_leak,
        "formal_table_citation_count": citation_guard["formal_table_citation_count"],
        "debug_csv_crop_path_visible_only": citation_guard["debug_csv_crop_path_visible_only"],
        "flag_off_restored": flag_off_restored,
        "evidence_improvement_label": "not_applicable",
        "query_route": debug.get("query_route", ""),
        "reason": debug.get("reason", ""),
        "expected_table_index_unit_id": expected_unit_id,
        "expected_unit_type": expected_unit_type,
        "merged_table_index_unit_ids": ";".join(str(value) for value in merged_ids),
        "merged_unit_types": ";".join(str(value) for value in merged_unit_types),
        "support_table_index_unit_ids": ";".join(
            str(chunk.metadata.get("table_index_unit_id", "")) for chunk in support_preview
        ),
        "preview_metadata_preserved": all(_preview_metadata_ok(chunk) for chunk in table_output),
    }


def stub_rerank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    for rank, chunk in enumerate(chunks, start=1):
        chunk.rerank_score = 1.0 / rank
        chunk.metadata["rerank_rank"] = rank
    return list(chunks)


def stub_support_selector(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    preview = preview_chunks(chunks)
    return preview[:2] if preview else chunks[:1]


def run_citation_guard(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    binder = CitationBinder()
    formal_count = 0
    debug_path_visible_only = True
    for idx, chunk in enumerate(chunks, start=1):
        evidence_id = f"E{idx}"
        candidate = evidence_candidate_from_chunk(evidence_id, chunk)
        support = [SupportItem(evidence_id, candidate, 0.9, ["selected_preview_table"])]
        candidates = binder.build_citation_candidates(support)
        _answer, citations, debug = binder.bind(
            f"Preview-only table evidence [{evidence_id}].",
            support,
            citation_candidates=candidates,
        )
        formal_count += len(citations)
        debug_paths = {
            chunk.metadata.get("source_csv_path"),
            chunk.metadata.get("source_pdf_crop_path"),
            chunk.metadata.get("source_markdown_path"),
        }
        if any(citation.source_file in debug_paths for citation in citations):
            debug_path_visible_only = False
        if debug.get("drop_reasons_by_evidence_id", {}).get(evidence_id) != (
            "table_preview_formal_citation_blocked"
        ):
            debug_path_visible_only = False
    return {
        "formal_table_citation_count": formal_count,
        "debug_csv_crop_path_visible_only": debug_path_visible_only,
    }


def evidence_candidate_from_chunk(evidence_id: str, chunk: RetrievedChunk) -> EvidenceCandidate:
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
        and bool(metadata.get("table_index_unit_id"))
        and bool(metadata.get("source_csv_path"))
        and bool(metadata.get("source_pdf_crop_path"))
    )


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7W-slim Main-Chain Evidence A/B Smoke",
        "",
        "This smoke used fake normal retrieval, local preview JSONL, stub rerank, and stub support selection only.",
        "",
        "## Metrics",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- query_count: {summary['query_count']}",
        f"- table_like_query_preview_merge_rate: {summary['table_like_query_preview_merge_rate']:.2%}",
        f"- non_table_preview_leak_count: {summary['non_table_preview_leak_count']}",
        f"- formal_table_citation_count: {summary['formal_table_citation_count']}",
        f"- metadata_preservation_rate: {summary['metadata_preservation_rate']:.2%}",
        f"- debug_csv_crop_path_visible_only: {summary['debug_csv_crop_path_visible_only']}",
        f"- flag_off_restored: {summary['flag_off_restored']}",
        f"- evidence_improvement_counts: {summary['evidence_improvement_counts']}",
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


def _extra_non_table_controls(start: int) -> list[dict[str, str]]:
    controls = [
        ("Summarize doc_0076 study design and main biological system.", "doc_0076"),
        ("Explain the stated motivation for doc_0600 induction study.", "doc_0600"),
    ]
    rows = []
    for offset, (query_text, doc_id) in enumerate(controls, start=start):
        rows.append(
            {
                "query_id": f"phase7w_extra_control_{offset:03d}",
                "query_text": query_text,
                "query_type": "non_table_control",
                "expected_doc_id": doc_id,
                "expected_table_id": "",
                "expected_table_index_unit_id": "",
                "expected_unit_type": "none",
                "expected_row_label": "",
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 6)


def _guardrails() -> dict[str, bool]:
    return {
        "milvus_accessed": False,
        "official_bm25_accessed": False,
        "embedding_run": False,
        "qwen_or_llm_called": False,
        "ragas_ocr_vlm_called": False,
        "production_table_index_built": False,
        "preview_units_upgraded": False,
        "formal_table_citation_generated": False,
        "canonical_source_resolution": False,
        "ingestion_pipeline_modified": False,
        "route_c_implemented": False,
    }


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7W-slim main-chain evidence A/B smoke.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    args = parser.parse_args()
    if not args.fixture_path.exists():
        build_query_fixture(
            output_path=args.fixture_path,
            summary_path=args.results_dir / "query_fixture_summary.json",
        )
    summary = run_mainchain_evidence_smoke(
        fixture_path=args.fixture_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
