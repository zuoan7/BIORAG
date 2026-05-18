from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7x_final_build_query_set import (
    QUERY_SET_PATH,
    TABLE_QUERY_TYPES,
    UNITS_PATH,
    ensure_query_set,
)
from src.synbio_rag.application.generation_v2 import GenerationV2Service
from src.synbio_rag.application.table_preview import apply_table_preview
from src.synbio_rag.domain.config import GenerationConfig, ModelEndpointConfig, RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RAGResponse, RetrievedChunk


PHASE_DIR = "v7_phase7_table_preview_final_acceptance"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"
AB_RESULTS_PATH = RESULTS_DIR / "mainchain_ab_results.csv"
AB_SUMMARY_PATH = RESULTS_DIR / "mainchain_ab_summary.json"
AB_REPORT_PATH = REPORTS_DIR / "mainchain_ab_acceptance_report.md"

MODES = ("normal_only", "table_preview_default_on")
FIELDNAMES = [
    "query_id",
    "query_type",
    "mode",
    "backend_mode",
    "mode_status",
    "error",
    "table_preview_enabled",
    "merge_strategy",
    "support_contains_table_preview",
    "table_preview_support_count",
    "expected_table_hit",
    "expected_unit_type_hit",
    "evidence_improvement_label",
    "non_table_preview_leak",
    "formal_table_citation_count",
    "csv_crop_in_formal_citation",
    "answer_generated",
    "answer_uses_table_evidence",
    "answer_improvement_label",
    "table_candidate_count",
    "table_preview_merged_count",
    "table_candidates_in_rerank_input",
    "expected_table_index_unit_id",
    "expected_unit_type",
    "merged_table_index_unit_ids",
    "support_table_index_unit_ids",
    "support_table_unit_types",
    "support_table_doc_ids",
    "support_table_ids",
    "support_row_labels",
    "answer_text_preview",
    "support_text_preview",
    "citation_source_files",
    "metadata_preserved",
    "flag_off_restored",
]


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


def run_mainchain_ab_acceptance(
    *,
    query_set_path: Path = QUERY_SET_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    backend_mode: str = "real",
) -> dict[str, Any]:
    queries = ensure_query_set(query_set_path)
    records: list[dict[str, Any]] = []
    pipeline = None
    real_backend_status = "not_attempted"
    real_backend_error = ""

    if backend_mode == "real":
        try:
            from src.synbio_rag.application.pipeline import SynBioRAGPipeline

            settings = Settings.from_env()
            settings.retrieval.table_preview_enabled = True
            settings.retrieval.table_preview_merge_enabled = True
            settings.retrieval.table_preview_merge_strategy = "type_aware_merge_v1"
            pipeline = SynBioRAGPipeline(settings)
            real_backend_status = "pass"
        except Exception as exc:  # pragma: no cover - environment dependent
            real_backend_status = "blocked"
            real_backend_error = f"{type(exc).__name__}: {exc}"
            summary = _blocked_summary(
                queries=queries,
                backend_mode=backend_mode,
                error=real_backend_error,
            )
            write_csv(results_dir / "mainchain_ab_results.csv", [], FIELDNAMES)
            write_json(results_dir / "mainchain_ab_summary.json", summary)
            write_ab_report(summary, reports_dir / "mainchain_ab_acceptance_report.md")
            return summary
    elif backend_mode != "seam":
        raise ValueError("backend_mode must be 'real' or 'seam'")

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for query in queries:
        pairs[query["query_id"]] = {}
        for mode in MODES:
            if backend_mode == "real":
                record = _run_real_mode(pipeline, query, mode)  # type: ignore[arg-type]
            else:
                record = _run_seam_mode(query, mode)
            pairs[query["query_id"]][mode] = record
            records.append(record)

    _label_pairs(queries, pairs)
    summary = _summarize(
        queries=queries,
        records=records,
        backend_mode=backend_mode,
        real_backend_status=real_backend_status,
        real_backend_error=real_backend_error,
        results_path=results_dir / "mainchain_ab_results.csv",
    )
    write_csv(results_dir / "mainchain_ab_results.csv", records, FIELDNAMES)
    write_json(results_dir / "mainchain_ab_summary.json", summary)
    write_ab_report(summary, reports_dir / "mainchain_ab_acceptance_report.md")
    return summary


def _run_real_mode(pipeline: Any, query: dict[str, Any], mode: str) -> dict[str, Any]:
    retrieval = pipeline.settings.retrieval
    old_enabled = retrieval.table_preview_enabled
    old_merge_enabled = retrieval.table_preview_merge_enabled
    old_strategy = retrieval.table_preview_merge_strategy
    try:
        if mode == "normal_only":
            retrieval.table_preview_enabled = False
            retrieval.table_preview_merge_enabled = False
            retrieval.table_preview_merge_strategy = "type_aware_merge_v1"
        else:
            retrieval.table_preview_enabled = True
            retrieval.table_preview_merge_enabled = True
            retrieval.table_preview_merge_strategy = "type_aware_merge_v1"
        response = pipeline.answer(query["query_text"])
        return _record_from_response(query=query, mode=mode, backend_mode="real", response=response)
    except Exception as exc:  # pragma: no cover - environment dependent
        return _error_record(
            query=query,
            mode=mode,
            backend_mode="real",
            error=f"{type(exc).__name__}: {exc}",
            details=traceback.format_exc(limit=4),
        )
    finally:
        retrieval.table_preview_enabled = old_enabled
        retrieval.table_preview_merge_enabled = old_merge_enabled
        retrieval.table_preview_merge_strategy = old_strategy


def _run_seam_mode(query: dict[str, Any], mode: str) -> dict[str, Any]:
    enabled = mode == "table_preview_default_on"
    config = _default_on_preview_config() if enabled else _flag_off_preview_config()
    input_chunks = _normal_retrieved(query)
    output, debug = apply_table_preview(
        question=query["query_text"],
        retrieved=input_chunks,
        config=config,
    )
    reranked = _stub_rerank(output)
    support_chunks = _select_support_chunks(query, reranked)
    result = _generate_extract_answer(query["query_text"], support_chunks)
    response = RAGResponse(
        answer=result.answer,
        confidence=1.0 if support_chunks else 0.0,
        route=QueryIntent.FACTOID,
        citations=result.citations,
        used_external_tool=False,
        tool_name=None,
        tool_result=None,
        debug={
            "table_preview": debug,
            "generation_v2": result.debug,
        },
    )
    return _record_from_response(query=query, mode=mode, backend_mode="seam", response=response)


def _record_from_response(
    *,
    query: dict[str, Any],
    mode: str,
    backend_mode: str,
    response: RAGResponse,
) -> dict[str, Any]:
    table_debug = response.debug.get("table_preview", {}) or {}
    generation_debug = response.debug.get("generation_v2", {}) or {}
    support_chunk_ids = _selected_support_chunk_ids(generation_debug)
    candidates = generation_debug.get("candidates", []) or []
    support_table_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("chunk_id") in support_chunk_ids
        and (candidate.get("metadata") or {}).get("object_type") == "table_index_unit"
    ]
    support_unit_ids = [
        str((candidate.get("metadata") or {}).get("table_index_unit_id", ""))
        for candidate in support_table_candidates
    ]
    support_unit_types = [
        str((candidate.get("metadata") or {}).get("table_unit_type", ""))
        for candidate in support_table_candidates
    ]
    merged_unit_ids = [str(value) for value in table_debug.get("merged_table_index_unit_ids", [])]
    expected_unit_id = str(query.get("expected_table_index_unit_id", ""))
    expected_unit_type = str(query.get("expected_unit_type", ""))
    citation_source_files = [citation.source_file for citation in response.citations]
    formal_table_citation_count = sum(
        1
        for citation in response.citations
        if citation.chunk_id.startswith("table_preview::")
        or citation.source_file == "table_preview_debug_only"
    )
    csv_crop_leak = any(_looks_like_debug_citation_source(source) for source in citation_source_files)
    answer_generated = bool((response.answer or "").strip())
    answer_uses_table = _answer_uses_table_evidence(response.answer, support_table_candidates)
    non_table_leak = (
        query.get("query_type") == "non_table_control"
        and (
            bool(table_debug.get("table_candidates_in_rerank_input"))
            or bool(support_table_candidates)
        )
    )
    metadata_preserved = all(
        _candidate_metadata_preserved(candidate) for candidate in support_table_candidates
    )
    if support_table_candidates:
        support_text = " | ".join(
            _one_line(str(candidate.get("text", "")))[:220] for candidate in support_table_candidates[:2]
        )
    else:
        support_text = _normal_support_preview(candidates, support_chunk_ids)
    return {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "mode": mode,
        "backend_mode": backend_mode,
        "mode_status": "pass",
        "error": "",
        "table_preview_enabled": bool(table_debug.get("enabled", False)),
        "merge_strategy": table_debug.get("merge_strategy", ""),
        "support_contains_table_preview": bool(support_table_candidates),
        "table_preview_support_count": len(support_table_candidates),
        "expected_table_hit": bool(
            expected_unit_id and expected_unit_id in set(merged_unit_ids + support_unit_ids)
        ),
        "expected_unit_type_hit": bool(
            expected_unit_type != "none" and expected_unit_type in support_unit_types
        )
        or bool(expected_unit_type != "none" and expected_unit_type in table_debug.get("merged_unit_types", [])),
        "evidence_improvement_label": "not_applicable",
        "non_table_preview_leak": non_table_leak,
        "formal_table_citation_count": formal_table_citation_count,
        "csv_crop_in_formal_citation": csv_crop_leak,
        "answer_generated": answer_generated,
        "answer_uses_table_evidence": answer_uses_table,
        "answer_improvement_label": "skipped",
        "table_candidate_count": int(table_debug.get("candidate_count", 0) or 0),
        "table_preview_merged_count": int(table_debug.get("merged_count", 0) or 0),
        "table_candidates_in_rerank_input": bool(table_debug.get("table_candidates_in_rerank_input")),
        "expected_table_index_unit_id": expected_unit_id,
        "expected_unit_type": expected_unit_type,
        "merged_table_index_unit_ids": ";".join(merged_unit_ids),
        "support_table_index_unit_ids": ";".join(support_unit_ids),
        "support_table_unit_types": ";".join(support_unit_types),
        "support_table_doc_ids": ";".join(
            str((candidate.get("metadata") or {}).get("doc_id", "")) for candidate in support_table_candidates
        ),
        "support_table_ids": ";".join(
            str((candidate.get("metadata") or {}).get("table_id", "")) for candidate in support_table_candidates
        ),
        "support_row_labels": ";".join(
            str((candidate.get("metadata") or {}).get("row_label", "")) for candidate in support_table_candidates
        ),
        "answer_text_preview": _one_line(response.answer)[:600],
        "support_text_preview": support_text,
        "citation_source_files": ";".join(citation_source_files),
        "metadata_preserved": metadata_preserved,
        "flag_off_restored": (
            mode == "normal_only"
            and table_debug.get("enabled") is False
            and not bool(table_debug.get("table_candidates_in_rerank_input"))
            and not bool(support_table_candidates)
        ),
    }


def _label_pairs(queries: list[dict[str, Any]], pairs: dict[str, dict[str, dict[str, Any]]]) -> None:
    for query in queries:
        pair = pairs[query["query_id"]]
        normal = pair["normal_only"]
        preview = pair["table_preview_default_on"]
        normal["evidence_improvement_label"] = "not_applicable"
        normal["answer_improvement_label"] = "skipped"
        if query["query_type"] not in TABLE_QUERY_TYPES:
            preview["evidence_improvement_label"] = "not_applicable"
            preview["answer_improvement_label"] = "skipped"
            continue

        normal_hit = _truthy(normal["expected_table_hit"])
        preview_hit = _truthy(preview["expected_table_hit"])
        if preview_hit and not normal_hit:
            evidence_label = "preview_better"
        elif preview_hit == normal_hit:
            evidence_label = "preview_same"
        else:
            evidence_label = "preview_worse"
        preview["evidence_improvement_label"] = evidence_label

        normal_uses_table = _truthy(normal["answer_uses_table_evidence"])
        preview_uses_table = _truthy(preview["answer_uses_table_evidence"])
        if preview_uses_table and not normal_uses_table:
            answer_label = "preview_better"
        elif preview_uses_table == normal_uses_table:
            answer_label = "preview_same"
        else:
            answer_label = "preview_worse"
        preview["answer_improvement_label"] = answer_label


def _summarize(
    *,
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    backend_mode: str,
    real_backend_status: str,
    real_backend_error: str,
    results_path: Path,
) -> dict[str, Any]:
    preview_records = [row for row in records if row["mode"] == "table_preview_default_on"]
    normal_records = [row for row in records if row["mode"] == "normal_only"]
    table_preview_records = [row for row in preview_records if row["query_type"] in TABLE_QUERY_TYPES]
    non_table_preview_records = [row for row in preview_records if row["query_type"] == "non_table_control"]
    errors = [row["error"] for row in records if row.get("mode_status") != "pass"]
    support_count = sum(1 for row in table_preview_records if _truthy(row["support_contains_table_preview"]))
    merge_count = sum(1 for row in table_preview_records if int(row["table_preview_merged_count"]) > 0)
    better_same_count = sum(
        1
        for row in table_preview_records
        if row["evidence_improvement_label"] in {"preview_better", "preview_same"}
    )
    formal_count = sum(int(row["formal_table_citation_count"]) for row in preview_records)
    csv_crop_leak_count = sum(1 for row in preview_records if _truthy(row["csv_crop_in_formal_citation"]))
    non_table_leak_count = sum(1 for row in non_table_preview_records if _truthy(row["non_table_preview_leak"]))
    metadata_ok_count = sum(1 for row in preview_records if _truthy(row["metadata_preserved"]))
    flag_off_restored = all(_truthy(row["flag_off_restored"]) for row in normal_records)
    answer_label_counts = Counter(row["answer_improvement_label"] for row in table_preview_records)
    evidence_label_counts = Counter(row["evidence_improvement_label"] for row in table_preview_records)

    pass_conditions = {
        "real_backend_not_blocked": real_backend_status != "blocked",
        "mode_errors_zero": not errors,
        "table_like_preview_support_rate_ge_80": _rate(support_count, len(table_preview_records)) >= 0.8,
        "table_like_preview_merge_rate_ge_80": _rate(merge_count, len(table_preview_records)) >= 0.8,
        "evidence_better_or_same_rate_ge_90": _rate(better_same_count, len(table_preview_records)) >= 0.9,
        "non_table_preview_leak_zero": non_table_leak_count == 0,
        "formal_table_citation_count_zero": formal_count == 0,
        "csv_crop_formal_citation_leak_zero": csv_crop_leak_count == 0,
        "metadata_preservation_100": metadata_ok_count == len(preview_records),
        "flag_off_restored": flag_off_restored,
    }
    failed_conditions = [name for name, ok in pass_conditions.items() if not ok]
    status = "pass" if not failed_conditions else "fail"
    if real_backend_status == "blocked":
        status = "blocked"
    return {
        "validation_status": status,
        "pass": status == "pass",
        "errors": errors + failed_conditions,
        "backend_mode": backend_mode,
        "real_backend_status": real_backend_status,
        "real_backend_error": real_backend_error,
        "query_count": len(queries),
        "record_count": len(records),
        "table_like_query_count": len(table_preview_records),
        "non_table_control_count": len(non_table_preview_records),
        "table_like_preview_support_rate": _rate(support_count, len(table_preview_records)),
        "table_like_preview_merge_rate": _rate(merge_count, len(table_preview_records)),
        "evidence_better_or_same_rate": _rate(better_same_count, len(table_preview_records)),
        "evidence_improvement_counts": dict(evidence_label_counts),
        "answer_improvement_counts": dict(answer_label_counts),
        "non_table_preview_leak_count": non_table_leak_count,
        "formal_table_citation_count": formal_count,
        "csv_crop_formal_citation_leak_count": csv_crop_leak_count,
        "metadata_preservation_rate": _rate(metadata_ok_count, len(preview_records)),
        "flag_off_restored": flag_off_restored,
        "pass_conditions": pass_conditions,
        "records_path": str(results_path),
    }


def _blocked_summary(
    *,
    queries: list[dict[str, Any]],
    backend_mode: str,
    error: str,
) -> dict[str, Any]:
    return {
        "validation_status": "blocked",
        "pass": False,
        "errors": ["real_mainchain_backend_unavailable"],
        "backend_mode": backend_mode,
        "real_backend_status": "blocked",
        "real_backend_error": error,
        "query_count": len(queries),
        "record_count": 0,
        "table_like_query_count": sum(1 for row in queries if row["query_type"] in TABLE_QUERY_TYPES),
        "non_table_control_count": sum(1 for row in queries if row["query_type"] == "non_table_control"),
        "table_like_preview_support_rate": 0.0,
        "table_like_preview_merge_rate": 0.0,
        "evidence_better_or_same_rate": 0.0,
        "evidence_improvement_counts": {},
        "answer_improvement_counts": {},
        "non_table_preview_leak_count": 0,
        "formal_table_citation_count": 0,
        "csv_crop_formal_citation_leak_count": 0,
        "metadata_preservation_rate": 0.0,
        "flag_off_restored": False,
        "pass_conditions": {"real_backend_not_blocked": False},
        "records_path": str(AB_RESULTS_PATH),
    }


def write_ab_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase7X Final Main-Chain A/B Acceptance",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- backend_mode: {summary['backend_mode']}",
        f"- real_backend_status: {summary['real_backend_status']}",
        f"- query_count: {summary['query_count']}",
        f"- table_like_query_count: {summary['table_like_query_count']}",
        f"- non_table_control_count: {summary['non_table_control_count']}",
        f"- table_like_preview_support_rate: {summary['table_like_preview_support_rate']:.2%}",
        f"- table_like_preview_merge_rate: {summary['table_like_preview_merge_rate']:.2%}",
        f"- evidence_better_or_same_rate: {summary['evidence_better_or_same_rate']:.2%}",
        f"- evidence_improvement_counts: {summary['evidence_improvement_counts']}",
        f"- answer_improvement_counts: {summary['answer_improvement_counts']}",
        f"- non_table_preview_leak_count: {summary['non_table_preview_leak_count']}",
        f"- formal_table_citation_count: {summary['formal_table_citation_count']}",
        f"- csv_crop_formal_citation_leak_count: {summary['csv_crop_formal_citation_leak_count']}",
        f"- metadata_preservation_rate: {summary['metadata_preservation_rate']:.2%}",
        f"- flag_off_restored: {summary['flag_off_restored']}",
    ]
    if summary.get("real_backend_error"):
        lines.extend(["", "## Backend Blocker", "", f"- {summary['real_backend_error']}"])
    if summary.get("errors"):
        lines.extend(["", "## Errors", *[f"- {error}" for error in summary["errors"]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_on_preview_config() -> RetrievalConfig:
    return RetrievalConfig(
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_max_candidates=5,
        table_preview_min_score=0.05,
        rerank_score_floor_ratio=0.0,
    )


def _flag_off_preview_config() -> RetrievalConfig:
    return RetrievalConfig(
        table_preview_enabled=False,
        table_preview_merge_enabled=False,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_max_candidates=5,
        table_preview_min_score=0.05,
        rerank_score_floor_ratio=0.0,
    )


def _normal_retrieved(query: dict[str, Any]) -> list[RetrievedChunk]:
    doc_id = query.get("expected_doc_id") or "normal_doc"
    return [
        RetrievedChunk(
            chunk_id=f"normal::{query['query_id']}",
            doc_id=doc_id,
            source_file=f"{doc_id}.pdf",
            title="Normal retrieval evidence",
            section="Abstract",
            text=(
                "Normal main-chain retrieval evidence placeholder for Phase7X final "
                "acceptance seam mode."
            ),
            vector_score=0.2,
            bm25_score=0.0,
            rerank_score=0.0,
            fusion_score=0.2,
            metadata={"object_type": "normal_chunk", "phase7x_seam_normal": True},
        )
    ]


def _stub_rerank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    output = list(chunks)
    for rank, chunk in enumerate(output, start=1):
        chunk.rerank_score = 1.0 / rank
        chunk.metadata["rerank_rank"] = rank
    return output


def _select_support_chunks(query: dict[str, Any], chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    preview = [chunk for chunk in chunks if chunk.metadata.get("object_type") == "table_index_unit"]
    if query["query_type"] in TABLE_QUERY_TYPES and preview:
        return preview[:3]
    return chunks[:1]


def _generate_extract_answer(question: str, support_chunks: list[RetrievedChunk]):
    service = GenerationV2Service(ModelEndpointConfig(provider="stub", model_name="local-extractive"))
    config = GenerationConfig(
        v2_use_qwen_synthesis=False,
        v2_require_citation=False,
        v2_min_support_score=0.0,
        v2_max_support_factoid=3,
    )
    analysis = QueryAnalysis(
        intent=QueryIntent.FACTOID,
        requires_external_tools=False,
        search_limit=5,
        rerank_top_k=5,
        notes="phase7x_final_answer_acceptance",
    )
    return service.run(question=question, analysis=analysis, seed_chunks=support_chunks, config=config)


def _selected_support_chunk_ids(generation_debug: dict[str, Any]) -> set[str]:
    selected = generation_debug.get("support_selection_debug", {}).get("selected_evidence_ids", [])
    candidates = generation_debug.get("candidates", []) or []
    by_eid = {candidate.get("evidence_id"): candidate.get("chunk_id") for candidate in candidates}
    ids = {str(by_eid[eid]) for eid in selected if eid in by_eid}
    lifecycle_ids = (
        generation_debug.get("evidence_lifecycle_debug", {})
        .get("selected_support", {})
        .get("kept_chunk_ids", [])
    )
    ids.update(str(chunk_id) for chunk_id in lifecycle_ids)
    return {chunk_id for chunk_id in ids if chunk_id}


def _normal_support_preview(candidates: list[dict[str, Any]], support_chunk_ids: set[str]) -> str:
    for candidate in candidates:
        if candidate.get("chunk_id") in support_chunk_ids:
            return _one_line(str(candidate.get("text", "")))[:220]
    return ""


def _candidate_metadata_preserved(candidate: dict[str, Any]) -> bool:
    metadata = candidate.get("metadata") or {}
    return (
        metadata.get("object_type") == "table_index_unit"
        and metadata.get("table_preview") is True
        and metadata.get("index_unit_status") == "preview_only"
        and metadata.get("production_ready") is False
        and metadata.get("value_bboxes_available") is False
        and metadata.get("table_preview_allow_formal_citation") is False
        and metadata.get("citation_formal_allowed") is False
        and bool(metadata.get("table_index_unit_id"))
    )


def _answer_uses_table_evidence(answer: str, support_table_candidates: list[dict[str, Any]]) -> bool:
    if not support_table_candidates:
        return False
    answer_text = answer or ""
    if "证据不足" in answer_text or "无法基于已检索证据回答" in answer_text:
        return False
    lowered = answer_text.lower()
    if "[table" in lowered or "table_preview::" in lowered:
        return True
    for candidate in support_table_candidates:
        metadata = candidate.get("metadata") or {}
        for key in ("table_id", "row_label", "caption"):
            value = str(metadata.get(key) or "").strip()
            if value and value.lower() in lowered:
                return True
        unit_type = str(metadata.get("table_unit_type") or "").strip()
        if unit_type and unit_type in lowered:
            return True
    return False


def _error_record(
    *,
    query: dict[str, Any],
    mode: str,
    backend_mode: str,
    error: str,
    details: str,
) -> dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "mode": mode,
        "backend_mode": backend_mode,
        "mode_status": "error",
        "error": f"{error}; {details}",
        "table_preview_enabled": mode == "table_preview_default_on",
        "merge_strategy": "type_aware_merge_v1",
        "support_contains_table_preview": False,
        "table_preview_support_count": 0,
        "expected_table_hit": False,
        "expected_unit_type_hit": False,
        "evidence_improvement_label": "not_applicable",
        "non_table_preview_leak": False,
        "formal_table_citation_count": 0,
        "csv_crop_in_formal_citation": False,
        "answer_generated": False,
        "answer_uses_table_evidence": False,
        "answer_improvement_label": "skipped",
        "table_candidate_count": 0,
        "table_preview_merged_count": 0,
        "table_candidates_in_rerank_input": False,
        "expected_table_index_unit_id": query.get("expected_table_index_unit_id", ""),
        "expected_unit_type": query.get("expected_unit_type", ""),
        "merged_table_index_unit_ids": "",
        "support_table_index_unit_ids": "",
        "support_table_unit_types": "",
        "support_table_doc_ids": "",
        "support_table_ids": "",
        "support_row_labels": "",
        "answer_text_preview": "",
        "support_text_preview": "",
        "citation_source_files": "",
        "metadata_preserved": False,
        "flag_off_restored": mode == "normal_only",
    }


def _looks_like_debug_citation_source(source_file: str) -> bool:
    lowered = (source_file or "").lower()
    return lowered.endswith((".csv", ".png", ".jpg", ".jpeg", ".md"))


def _one_line(value: str) -> str:
    return " ".join((value or "").split())


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7X final main-chain A/B acceptance.")
    parser.add_argument("--query-set-path", type=_path_arg, default=QUERY_SET_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    parser.add_argument("--backend-mode", choices=("real", "seam"), default="real")
    args = parser.parse_args()
    summary = run_mainchain_ab_acceptance(
        query_set_path=args.query_set_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        backend_mode=args.backend_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
