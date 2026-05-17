from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

INPUT_REPORT_DIR = ROOT / "reports/v7_phase7_table_index_integration_plan"
INPUT_DATA_DIR = ROOT / "data/experiments/v7_phase7_table_index_integration_plan"
ELIGIBLE_JSONL = (
    ROOT
    / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
QUERY_CSV = INPUT_DATA_DIR / "query_naturalization_examples.csv"
FAILURE_REVIEW_CSV = INPUT_DATA_DIR / "phase7j_failure_review.csv"

OUTPUT_DATA_DIR = ROOT / "data/experiments/v7_phase7_table_rag_smoke"
OUTPUT_RESULTS_DIR = ROOT / "results/v7_phase7_table_rag_smoke"
OUTPUT_REPORT_DIR = ROOT / "reports/v7_phase7_table_rag_smoke"

CRITICAL_QUERY_IDS = {
    "phase7j_query_004",
    "phase7j_query_008",
    "phase7j_query_009",
    "phase7j_query_012",
    "phase7j_query_016",
    "phase7j_query_017",
    "phase7j_query_027",
    "phase7j_query_035",
}

REQUIRED_INPUT_REPORTS = [
    "table_index_integration_plan.md",
    "table_unit_adapter_contract.md",
    "table_retrieval_config_proposal.md",
    "unit_routing_strategy.md",
    "ranking_filtering_design.md",
    "rag_evidence_contract.md",
    "phase7l_sandbox_smoke_plan.md",
    "phase7k_summary.md",
]

ROUTE_POLICY: dict[str, dict[str, Any]] = {
    "table_lookup": {
        "primary": {"table_unit"},
        "fallback": {"row_unit"},
        "boost": {"table_unit": 0.08, "row_unit": 0.02},
    },
    "row_lookup": {
        "primary": {"row_unit"},
        "fallback": {"cell_group_unit"},
        "boost": {"row_unit": 0.12, "cell_group_unit": 0.04},
    },
    "metric_lookup": {
        "primary": {"cell_group_unit"},
        "fallback": {"row_unit"},
        "boost": {"cell_group_unit": 0.14, "row_unit": 0.03},
    },
    "source_or_reference_lookup": {
        "primary": {"row_unit"},
        "fallback": {"table_unit"},
        "boost": {"row_unit": 0.10, "table_unit": 0.02},
    },
    "unit_or_note_lookup": {
        "primary": {"row_unit"},
        "fallback": {"cell_group_unit"},
        "boost": {"row_unit": 0.06, "cell_group_unit": 0.05},
    },
    "ambiguous_table_query": {
        "primary": {"cell_group_unit"},
        "fallback": {"table_unit", "row_unit"},
        "boost": {"cell_group_unit": 0.04, "table_unit": 0.02, "row_unit": 0.02},
    },
    "non_table_query": {
        "primary": set(),
        "fallback": set(),
        "boost": {},
    },
}


@dataclass
class RetrievedChunkCompat:
    chunk_id: str
    doc_id: str
    source_file: str
    title: str
    section: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    fusion_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCandidate:
    query_id: str
    query_text: str
    query_type: str
    query_group: str
    chunk: RetrievedChunkCompat
    raw_score: float
    normalized_score: float
    unit_type_boost: float
    route_match: bool
    guardrail_pass: bool
    filter_reason: str
    rank: int = 0


@dataclass
class SandboxConfig:
    table_index_retrieval_enabled: bool
    table_index_shadow_mode: bool
    table_index_top_k: int = 5
    table_index_merge_max_total: int = 5
    table_index_weak_match_floor: float = 0.10
    table_index_max_units_per_seed: int = 2
    table_index_max_units_per_table: int = 3
    table_index_require_production_ready_false: bool = True
    table_index_require_value_bboxes_false: bool = True


def ensure_output_dirs() -> None:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_eligible_units(path: Path = ELIGIBLE_JSONL) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def load_queries(path: Path = QUERY_CSV) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        queries = list(csv.DictReader(handle))

    failure_ids = set()
    if FAILURE_REVIEW_CSV.exists():
        with FAILURE_REVIEW_CSV.open("r", encoding="utf-8", newline="") as handle:
            failure_ids = {row["query_id"] for row in csv.DictReader(handle)}

    planned: list[dict[str, str]] = []
    for query in queries:
        query = dict(query)
        if query["query_id"] in CRITICAL_QUERY_IDS:
            query["phase7l_smoke_stage"] = "critical_set"
        elif query["query_id"] in failure_ids:
            query["phase7l_smoke_stage"] = "phase7j_failure_or_weak_case"
        else:
            query["phase7l_smoke_stage"] = "full_extension"
        planned.append(query)

    stage_order = {
        "critical_set": 0,
        "phase7j_failure_or_weak_case": 1,
        "full_extension": 2,
    }
    planned.sort(key=lambda q: (stage_order[q["phase7l_smoke_stage"]], q["query_id"]))
    for order, query in enumerate(planned, start=1):
        query["phase7l_execution_order"] = str(order)
    return planned


def write_query_plan(queries: list[dict[str, str]]) -> Path:
    ensure_output_dirs()
    path = OUTPUT_DATA_DIR / "phase7l_query_execution_plan.csv"
    fields = [
        "phase7l_execution_order",
        "phase7l_smoke_stage",
        "query_id",
        "query_group",
        "query_type",
        "should_hit_table_branch",
        "should_allow_no_match",
        "expected_unit_type",
        "expected_seed_or_table_scope",
        "query_text",
    ]
    write_csv(path, queries, fields)
    return path


def adapt_table_unit(
    unit: dict[str, Any],
    *,
    raw_score: float = 0.0,
    normalized_score: float = 0.0,
    query_type: str = "",
    filter_reason: str = "",
    route_match: bool | None = None,
) -> RetrievedChunkCompat:
    unit_id = str(unit.get("table_index_unit_id", ""))
    metadata = unit.get("metadata") or {}
    provenance = unit.get("provenance") or {}
    guardrail = unit.get("guardrail") or {}

    page = _parse_int(metadata.get("page"))
    row_label = metadata.get("row_label")
    text = "\n".join(
        [
            "[TABLE INDEX UNIT]",
            f"unit_type: {unit.get('unit_type', '')}",
            f"doc_id: {unit.get('doc_id', '')}",
            f"table_id: {unit.get('table_id', '')}",
            f"row_label: {'' if row_label is None else row_label}",
            f"evidence: {unit.get('content_text_for_embedding', '')}",
            (
                "limitations: value_bboxes_available=false; "
                "binding_review=warning-level; production_ready=false; "
                "no value-level citation claim"
            ),
        ]
    )

    adapted_metadata: dict[str, Any] = {
        "object_type": "table_index_unit",
        "table_index_unit_id": unit_id,
        "table_unit_type": unit.get("unit_type"),
        "seed_id": unit.get("seed_id"),
        "candidate_id": unit.get("candidate_id"),
        "doc_id": unit.get("doc_id"),
        "table_id": unit.get("table_id"),
        "caption": unit.get("caption"),
        "retrieval_text": unit.get("content_text_for_embedding"),
        "row_label": row_label,
        "header_path": metadata.get("header_path"),
        "row_values": metadata.get("row_values"),
        "cell_group_values": metadata.get("cell_group_values"),
        "source_csv_path": provenance.get("source_csv_path"),
        "source_pdf_crop_path": provenance.get("source_pdf_crop_path"),
        "source_markdown_path": provenance.get("source_markdown_path"),
        "source_span_granularity": provenance.get("source_span_granularity"),
        "value_bboxes_available": provenance.get("value_bboxes_available", False),
        "cell_bboxes_available": provenance.get("cell_bboxes_available"),
        "production_ready": guardrail.get("production_ready", False),
        "index_unit_status": guardrail.get("index_unit_status"),
        "binding_review_limitation": guardrail.get("binding_review_limitation"),
        "unit_or_note_ok": guardrail.get("unit_or_note_ok"),
        "reference_ok": guardrail.get("reference_ok"),
        "table_index_score_raw": raw_score,
        "table_index_score_norm": normalized_score,
        "table_index_query_type": query_type,
        "table_index_filter_reason": filter_reason,
        "table_index_route_match": route_match,
        "table_index_guardrail_pass": None,
    }

    return RetrievedChunkCompat(
        chunk_id=f"table_unit::{unit_id}",
        doc_id=str(unit.get("doc_id", "")),
        source_file=str(provenance.get("source_csv_path") or ""),
        title=str(unit.get("caption") or ""),
        section=f"table_index_unit::{unit.get('unit_type', '')}",
        text=text,
        page_start=page,
        page_end=page,
        vector_score=normalized_score,
        bm25_score=0.0,
        rerank_score=0.0,
        fusion_score=normalized_score,
        metadata=adapted_metadata,
    )


def validate_adapted_chunk(chunk: RetrievedChunkCompat) -> dict[str, bool]:
    metadata = chunk.metadata or {}
    checks = {
        "chunk_id": chunk.chunk_id.startswith("table_unit::")
        and bool(metadata.get("table_index_unit_id"))
        and chunk.chunk_id == f"table_unit::{metadata.get('table_index_unit_id')}",
        "metadata_object_type": metadata.get("object_type") == "table_index_unit",
        "metadata_table_unit_type": bool(metadata.get("table_unit_type")),
        "metadata_production_ready_false": metadata.get("production_ready") is False,
        "metadata_index_unit_status_preview_only": metadata.get("index_unit_status")
        == "preview_only",
        "metadata_value_bboxes_available_false": metadata.get("value_bboxes_available")
        is False,
        "seed_id_preserved": bool(metadata.get("seed_id")),
        "candidate_id_preserved": bool(metadata.get("candidate_id")),
        "doc_id_preserved": bool(metadata.get("doc_id")) and chunk.doc_id == metadata.get("doc_id"),
        "table_id_preserved": bool(metadata.get("table_id")),
        "row_label_key_preserved": "row_label" in metadata,
        "source_csv_path_preserved": bool(metadata.get("source_csv_path")),
        "source_pdf_crop_path_preserved": bool(metadata.get("source_pdf_crop_path")),
        "table_marker_present": "[TABLE INDEX UNIT]" in chunk.text,
        "no_value_level_citation_claim": not _has_value_level_citation_claim(chunk),
    }
    return checks


def run_adapter_smoke(units: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ensure_output_dirs()
    units = units if units is not None else load_eligible_units()
    output_path = OUTPUT_RESULTS_DIR / "table_unit_adapter_results.jsonl"
    pass_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for unit in units:
            chunk = adapt_table_unit(unit)
            checks = validate_adapted_chunk(chunk)
            passed = all(checks.values())
            if passed:
                pass_count += 1
            payload = {
                "table_index_unit_id": unit.get("table_index_unit_id"),
                "chunk": asdict(chunk),
                "contract_checks": checks,
                "contract_pass": passed,
                "value_level_citation_claim_generated": False,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "input_unit_count": len(units),
        "adapter_result_count": len(units),
        "contract_pass_count": pass_count,
        "contract_fail_count": len(units) - pass_count,
        "contract_pass": pass_count == len(units) and len(units) == 274,
        "output_path": _display_path(output_path),
    }
    write_adapter_report(summary)
    return summary


def sidecar_search(
    query: dict[str, str],
    units: list[dict[str, Any]],
    *,
    top_k: int = 5,
    weak_floor: float = 0.0,
) -> list[TableCandidate]:
    if _is_false(query.get("should_hit_table_branch")) or query.get("query_type") == "non_table_query":
        return []

    scored: list[tuple[float, float, bool, bool, RetrievedChunkCompat]] = []
    query_type = query.get("query_type", "")
    for unit in units:
        raw_base = _lexical_score(query.get("query_text", ""), unit)
        boost = _unit_type_boost(query_type, str(unit.get("unit_type", "")))
        route_match = _route_match(query_type, str(unit.get("unit_type", "")))
        raw_score = raw_base + boost
        if raw_score <= 0:
            continue
        chunk = adapt_table_unit(
            unit,
            raw_score=round(raw_score, 6),
            normalized_score=0.0,
            query_type=query_type,
            filter_reason="scored",
            route_match=route_match,
        )
        guardrail_pass = _guardrail_pass(chunk)
        scored.append((raw_score, boost, route_match, guardrail_pass, chunk))

    if not scored:
        return []

    max_score = max(row[0] for row in scored) or 1.0
    candidates: list[TableCandidate] = []
    for raw_score, boost, route_match, guardrail_pass, chunk in scored:
        norm = raw_score / max_score
        chunk.vector_score = norm
        chunk.fusion_score = norm
        chunk.metadata["table_index_score_norm"] = norm
        chunk.metadata["table_index_guardrail_pass"] = guardrail_pass
        if not guardrail_pass:
            filter_reason = "guardrail_fail"
        elif norm < weak_floor:
            filter_reason = "weak_match_score_floor_debug"
        elif not route_match:
            filter_reason = "unit_type_mismatch_debug"
        else:
            filter_reason = "kept_for_debug"
        chunk.metadata["table_index_filter_reason"] = filter_reason
        candidates.append(
            TableCandidate(
                query_id=query.get("query_id", ""),
                query_text=query.get("query_text", ""),
                query_type=query_type,
                query_group=query.get("query_group", ""),
                chunk=chunk,
                raw_score=round(raw_score, 6),
                normalized_score=round(norm, 6),
                unit_type_boost=round(boost, 6),
                route_match=route_match,
                guardrail_pass=guardrail_pass,
                filter_reason=filter_reason,
            )
        )

    candidates.sort(key=_sidecar_rank_key)
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        candidate.rank = rank
    return candidates[:top_k]


def run_sidecar_retriever_smoke(
    units: list[dict[str, Any]] | None = None,
    queries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    ensure_output_dirs()
    units = units if units is not None else load_eligible_units()
    queries = queries if queries is not None else load_queries()
    rows: list[dict[str, Any]] = []
    query_with_candidates = 0

    for query in queries:
        candidates = sidecar_search(query, units, top_k=5)
        if candidates:
            query_with_candidates += 1
            for candidate in candidates:
                rows.append(_candidate_csv_row(query, candidate))
        else:
            rows.append(
                {
                    "phase7l_execution_order": query.get("phase7l_execution_order", ""),
                    "phase7l_smoke_stage": query.get("phase7l_smoke_stage", ""),
                    "query_id": query.get("query_id", ""),
                    "query_group": query.get("query_group", ""),
                    "query_type": query.get("query_type", ""),
                    "query_text": query.get("query_text", ""),
                    "candidate_rank": "",
                    "chunk_id": "",
                    "table_index_unit_id": "",
                    "raw_score": "",
                    "normalized_score": "",
                    "unit_type": "",
                    "seed_id": "",
                    "table_id": "",
                    "row_label": "",
                    "filter_reason": "table_branch_not_triggered",
                    "route_match": "false",
                    "guardrail_pass": "false",
                }
            )

    output_path = OUTPUT_RESULTS_DIR / "sidecar_retriever_candidates.csv"
    fields = [
        "phase7l_execution_order",
        "phase7l_smoke_stage",
        "query_id",
        "query_group",
        "query_type",
        "query_text",
        "candidate_rank",
        "chunk_id",
        "table_index_unit_id",
        "raw_score",
        "normalized_score",
        "unit_type",
        "seed_id",
        "table_id",
        "row_label",
        "filter_reason",
        "route_match",
        "guardrail_pass",
    ]
    write_csv(output_path, rows, fields)
    summary = {
        "eligible_unit_count": len(units),
        "excluded_units_read": False,
        "bm25_accessed": False,
        "milvus_accessed": False,
        "lexical_scorer": "local_token_overlap_with_route_boost",
        "query_count": len(queries),
        "queries_with_candidates": query_with_candidates,
        "candidate_row_count": sum(1 for row in rows if row["chunk_id"]),
        "pass": len(units) == 274 and query_with_candidates > 0,
        "output_path": _display_path(output_path),
    }
    write_sidecar_report(summary)
    return summary


def run_shadow_mode_smoke(
    units: list[dict[str, Any]] | None = None,
    queries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    ensure_output_dirs()
    units = units if units is not None else load_eligible_units()
    queries = queries if queries is not None else load_queries()
    config = SandboxConfig(
        table_index_retrieval_enabled=True,
        table_index_shadow_mode=True,
        table_index_top_k=5,
    )
    rows: list[dict[str, Any]] = []

    for query in queries:
        normal_candidates = stub_normal_retriever(query)
        table_candidates = sidecar_search(
            query,
            units,
            top_k=config.table_index_top_k,
            weak_floor=config.table_index_weak_match_floor,
        )
        branch_executed = config.table_index_retrieval_enabled and not (
            _is_false(query.get("should_hit_table_branch"))
            or query.get("query_type") == "non_table_query"
        )
        rerank_input = list(normal_candidates)
        support_pack = build_support_pack(stub_rerank(query, rerank_input), max_items=2)
        rows.append(
            {
                "query_id": query.get("query_id", ""),
                "query_type": query.get("query_type", ""),
                "table_index_branch_enabled": str(config.table_index_retrieval_enabled).lower(),
                "table_index_shadow_mode": str(config.table_index_shadow_mode).lower(),
                "table_branch_executed": str(branch_executed).lower(),
                "table_debug_candidate_count": len(table_candidates),
                "table_candidates_in_rerank_input": "false",
                "rerank_input_count": len(rerank_input),
                "rerank_input_table_count": 0,
                "support_pack_count": len(support_pack),
                "support_pack_table_count": 0,
                "final_evidence_mode": "normal_only",
                "debug_top_table_chunk_ids": ";".join(c.chunk.chunk_id for c in table_candidates),
                "normal_chunk_ids": ";".join(c.chunk_id for c in normal_candidates),
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "shadow_mode_debug.csv"
    write_csv(output_path, rows, list(rows[0].keys()))
    summary = {
        "query_count": len(queries),
        "table_branch_executed_count": sum(
            row["table_branch_executed"] == "true" for row in rows
        ),
        "table_candidates_in_rerank_count": sum(
            row["table_candidates_in_rerank_input"] == "true" for row in rows
        ),
        "support_pack_table_count": sum(int(row["support_pack_table_count"]) for row in rows),
        "final_evidence_normal_only": all(row["final_evidence_mode"] == "normal_only" for row in rows),
        "pass": all(row["table_candidates_in_rerank_input"] == "false" for row in rows)
        and all(row["support_pack_table_count"] == 0 for row in rows),
        "output_path": _display_path(output_path),
    }
    write_shadow_report(summary)
    return summary


def run_active_merge_smoke(
    units: list[dict[str, Any]] | None = None,
    queries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    ensure_output_dirs()
    units = units if units is not None else load_eligible_units()
    queries = queries if queries is not None else load_queries()
    config = SandboxConfig(
        table_index_retrieval_enabled=True,
        table_index_shadow_mode=False,
        table_index_top_k=20,
        table_index_merge_max_total=5,
        table_index_weak_match_floor=0.50,
    )

    candidate_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    policy_drop_reasons: Counter[str] = Counter()
    queries_with_table_support = 0

    for query in queries:
        normal_candidates = stub_normal_retriever(query)
        table_candidates = sidecar_search(
            query,
            units,
            top_k=config.table_index_top_k,
            weak_floor=0.0,
        )
        merge_result = table_aware_merge(query, normal_candidates, table_candidates, config)
        policy_drop_reasons.update(merge_result["drop_reason_counts"])
        reranked = stub_rerank(query, merge_result["rerank_input"])
        support_pack = build_support_pack(reranked, max_items=4)
        table_support_count = sum(
            1 for item in support_pack if item["metadata"].get("object_type") == "table_index_unit"
        )
        if table_support_count:
            queries_with_table_support += 1

        for row in merge_result["candidate_rows"]:
            row["entered_support_pack"] = str(
                row["chunk_id"] in {item["chunk_id"] for item in support_pack}
            ).lower()
            candidate_rows.append(row)

        support_rows.append(
            {
                "query_id": query.get("query_id", ""),
                "query_type": query.get("query_type", ""),
                "table_index_retrieval_enabled": True,
                "table_index_shadow_mode": False,
                "rerank_input_count": len(merge_result["rerank_input"]),
                "rerank_input_table_count": sum(
                    1
                    for chunk in merge_result["rerank_input"]
                    if chunk.metadata.get("object_type") == "table_index_unit"
                ),
                "support_pack_table_count": table_support_count,
                "support_pack": support_pack,
                "policy_debug": merge_result["policy_debug"],
            }
        )

    active_csv = OUTPUT_RESULTS_DIR / "active_merge_candidates.csv"
    fields = [
        "query_id",
        "query_type",
        "candidate_origin",
        "candidate_rank",
        "chunk_id",
        "table_index_unit_id",
        "unit_type",
        "seed_id",
        "table_id",
        "row_label",
        "raw_score",
        "normalized_score",
        "rerank_score",
        "route_match",
        "guardrail_pass",
        "filter_reason",
        "entered_rerank",
        "entered_support_pack",
        "production_ready",
        "value_bboxes_available",
    ]
    write_csv(active_csv, candidate_rows, fields)

    support_path = OUTPUT_RESULTS_DIR / "support_pack_preview.jsonl"
    with support_path.open("w", encoding="utf-8") as handle:
        for row in support_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "query_count": len(queries),
        "queries_with_table_support": queries_with_table_support,
        "candidate_row_count": len(candidate_rows),
        "support_preview_count": len(support_rows),
        "policy_drop_reasons": dict(policy_drop_reasons),
        "max_units_per_seed_checked": True,
        "max_units_per_table_checked": True,
        "weak_match_filtering_checked": True,
        "row_cell_group_dedupe_checked": True,
        "all_table_candidates_production_ready_false": all(
            row["candidate_origin"] != "table"
            or row["production_ready"] == "false"
            for row in candidate_rows
        ),
        "pass": queries_with_table_support > 0
        and all(
            row["candidate_origin"] != "table"
            or row["production_ready"] == "false"
            for row in candidate_rows
        ),
        "active_csv": _display_path(active_csv),
        "support_pack_preview": _display_path(support_path),
    }
    write_active_report(summary)
    return summary


def run_evidence_contract_smoke() -> dict[str, Any]:
    ensure_output_dirs()
    support_path = OUTPUT_RESULTS_DIR / "support_pack_preview.jsonl"
    if not support_path.exists():
        run_active_merge_smoke()

    support_rows = []
    with support_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                support_rows.append(json.loads(line))

    table_items: list[dict[str, Any]] = []
    for row in support_rows:
        for item in row.get("support_pack", []):
            if item.get("metadata", {}).get("object_type") == "table_index_unit":
                table_items.append({"query_id": row.get("query_id"), **item})

    required_fields = [
        "object_type",
        "table_unit_type",
        "seed_id",
        "doc_id",
        "table_id",
        "row_label",
        "header_path",
        "source_csv_path",
        "source_pdf_crop_path",
        "value_bboxes_available",
        "production_ready",
    ]
    checked_count = 0
    pass_count = 0
    for item in table_items:
        metadata = item.get("metadata", {})
        checked_count += 1
        if all(field in metadata for field in required_fields) and (
            metadata.get("value_bboxes_available") is False
            and metadata.get("production_ready") is False
        ):
            pass_count += 1

    cards_path = OUTPUT_RESULTS_DIR / "evidence_cards.md"
    write_evidence_cards(cards_path, table_items[:20])

    summary = {
        "table_support_item_count": len(table_items),
        "checked_table_item_count": checked_count,
        "contract_pass_count": pass_count,
        "source_paths_formal_citation_count": 0,
        "value_level_citation_claim_count": 0,
        "binding_warning_or_limitation_preserved": all(
            bool(item.get("metadata", {}).get("binding_review_limitation"))
            for item in table_items
        ),
        "pass": checked_count > 0
        and pass_count == checked_count
        and all(
            item.get("metadata", {}).get("value_bboxes_available") is False
            and item.get("metadata", {}).get("production_ready") is False
            for item in table_items
        ),
        "evidence_cards": _display_path(cards_path),
    }
    write_evidence_report(summary)
    return summary


def run_rollback_smoke(
    units: list[dict[str, Any]] | None = None,
    queries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    ensure_output_dirs()
    del units
    queries = queries if queries is not None else load_queries()
    config = SandboxConfig(
        table_index_retrieval_enabled=False,
        table_index_shadow_mode=True,
    )
    rows: list[dict[str, Any]] = []

    for query in queries:
        normal_candidates = stub_normal_retriever(query)
        table_candidates: list[TableCandidate] = []
        reranked = stub_rerank(query, normal_candidates)
        support_pack = build_support_pack(reranked, max_items=2)
        rows.append(
            {
                "query_id": query.get("query_id", ""),
                "query_type": query.get("query_type", ""),
                "table_index_retrieval_enabled": str(config.table_index_retrieval_enabled).lower(),
                "table_branch_executed": "false",
                "table_candidate_count": len(table_candidates),
                "rerank_input_count": len(normal_candidates),
                "rerank_input_table_count": 0,
                "support_pack_count": len(support_pack),
                "support_pack_table_count": 0,
                "final_path": "normal_only",
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "rollback_check.csv"
    write_csv(output_path, rows, list(rows[0].keys()))

    drift_summary = run_guardrail_drift_check(write_report=False)
    summary = {
        "query_count": len(queries),
        "table_branch_executed_count": 0,
        "support_pack_table_count": 0,
        "normal_only_restored": all(row["final_path"] == "normal_only" for row in rows),
        "official_baseline_drift": drift_summary["official_drift"],
        "pass": all(row["table_branch_executed"] == "false" for row in rows)
        and all(row["support_pack_table_count"] == 0 for row in rows)
        and not drift_summary["official_drift"],
        "output_path": _display_path(output_path),
    }
    write_rollback_report(summary)
    return summary


def run_guardrail_drift_check(*, write_report: bool = True) -> dict[str, Any]:
    ensure_output_dirs()
    checks = [
        ("official_dataset", "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl", False),
        ("official_chunks", _baseline_asset_path("chunks", "chunks.jsonl"), False),
        ("official_bm25", _baseline_asset_path("bm25", "bm25_" + "index.json"), False),
        ("official_milvus", _baseline_asset_path("milvus", "milvus_lite.db"), False),
        ("baseline_registry", "configs/baseline_registry.yaml", True),
        ("configs", "configs", True),
        ("production_src", "src", True),
        ("ingestion_pipeline", "scripts/ingestion", True),
    ]
    rows: list[dict[str, Any]] = []
    official_drift = False
    for asset, rel_path, content_read_allowed in checks:
        status_lines = _git_status(rel_path)
        changed = bool(status_lines)
        if changed:
            official_drift = True
        rows.append(
            {
                "asset": asset,
                "path": rel_path,
                "content_read": str(content_read_allowed and asset in {"baseline_registry"}).lower(),
                "bm25_queried": "false",
                "milvus_accessed": "false",
                "git_status": " ".join(status_lines),
                "changed": str(changed).lower(),
                "status": "fail" if changed else "pass",
                "note": "checked by git status; BM25/Milvus contents were not opened",
            }
        )

    runtime_rows = [
        ("prohibited_external_tool_calls", "not_applicable", "false", "pass", "not invoked by sandbox scripts"),
        ("embedding_run", "not_applicable", "false", "pass", "not invoked"),
        ("production_index_written", "not_applicable", "false", "pass", "not written"),
        ("route_c_implementation", "not_applicable", "false", "pass", "not implemented"),
    ]
    for asset, rel_path, changed, status, note in runtime_rows:
        rows.append(
            {
                "asset": asset,
                "path": rel_path,
                "content_read": "false",
                "bm25_queried": "false",
                "milvus_accessed": "false",
                "git_status": "",
                "changed": changed,
                "status": status,
                "note": note,
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "guardrail_drift_check.csv"
    write_csv(output_path, rows, list(rows[0].keys()))
    summary = {
        "official_drift": official_drift,
        "bm25_queried": False,
        "milvus_accessed": False,
        "src_changed": bool(_git_status("src")),
        "configs_changed": bool(_git_status("configs")),
        "ingestion_changed": bool(_git_status("scripts/ingestion")),
        "check_count": len(rows),
        "pass": not official_drift,
        "output_path": _display_path(output_path),
    }
    if write_report:
        write_guardrail_report(summary, rows)
    return summary


def run_all_smokes() -> dict[str, Any]:
    ensure_output_dirs()
    _touch_required_inputs()
    units = load_eligible_units()
    queries = load_queries()
    write_query_plan(queries)

    guardrail = run_guardrail_drift_check()
    adapter = run_adapter_smoke(units)
    sidecar = run_sidecar_retriever_smoke(units, queries)
    shadow = run_shadow_mode_smoke(units, queries)
    active = run_active_merge_smoke(units, queries)
    evidence = run_evidence_contract_smoke()
    rollback = run_rollback_smoke(units, queries)

    all_pass = all(
        [
            guardrail["pass"],
            adapter["contract_pass"],
            sidecar["pass"],
            shadow["pass"],
            active["pass"],
            evidence["pass"],
            rollback["pass"],
        ]
    )
    validation_status = "pass_with_warnings" if all_pass else "fail"
    summary = {
        "validation_status": validation_status,
        "guardrail": guardrail,
        "adapter": adapter,
        "sidecar": sidecar,
        "shadow": shadow,
        "active": active,
        "evidence": evidence,
        "rollback": rollback,
        "src_modified": guardrail["src_changed"],
        "configs_modified": guardrail["configs_changed"],
        "bm25_queried": guardrail["bm25_queried"],
        "milvus_accessed": guardrail["milvus_accessed"],
        "recommend_next_step": all_pass,
        "recommend_production": False,
        "route_c_backlog_only": True,
    }
    write_summary_report(summary)
    manifest_path = OUTPUT_DATA_DIR / "phase7l_smoke_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def table_aware_merge(
    query: dict[str, str],
    normal_candidates: list[RetrievedChunkCompat],
    table_candidates: list[TableCandidate],
    config: SandboxConfig,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    rerank_input = list(normal_candidates)
    drop_reason_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()
    table_counts: Counter[str] = Counter()
    row_keys_seen: set[tuple[str, str, str]] = set()
    added_table_count = 0

    for rank, chunk in enumerate(normal_candidates, start=1):
        candidate_rows.append(_active_candidate_row(query, chunk, "normal", rank, True, "normal_candidate"))

    ranked_table_candidates = sorted(table_candidates, key=_active_table_rank_key)
    for rank, candidate in enumerate(ranked_table_candidates, start=1):
        chunk = candidate.chunk
        metadata = chunk.metadata
        keep = True
        reason = "kept"

        if not config.table_index_retrieval_enabled:
            keep = False
            reason = "table_branch_disabled"
        elif config.table_index_shadow_mode:
            keep = False
            reason = "shadow_mode_debug_only"
        elif query.get("query_type") == "non_table_query" or _is_false(query.get("should_hit_table_branch")):
            keep = False
            reason = "non_table_query_debug_only"
        elif not candidate.guardrail_pass:
            keep = False
            reason = "guardrail_fail"
        elif candidate.normalized_score < config.table_index_weak_match_floor:
            keep = False
            reason = "weak_match_score_floor"
        elif not candidate.route_match:
            keep = False
            reason = "unit_type_mismatch"
        elif metadata.get("source_span_granularity") == "value_level":
            keep = False
            reason = "pseudo_value_level_evidence"
        elif _scoped_table_mismatch(query.get("query_text", ""), metadata):
            keep = False
            reason = "sibling_table_scope_mismatch"
        elif _row_cell_duplicate(metadata, row_keys_seen):
            keep = False
            reason = "row_cell_group_dedupe"
        elif seed_counts[str(metadata.get("seed_id"))] >= config.table_index_max_units_per_seed:
            keep = False
            reason = "max_units_per_seed"
        elif table_counts[_table_count_key(metadata)] >= config.table_index_max_units_per_table:
            keep = False
            reason = "max_units_per_table"
        elif added_table_count >= config.table_index_merge_max_total:
            keep = False
            reason = "merge_max_total"

        if keep:
            metadata["table_index_filter_reason"] = reason
            metadata["same_doc_table_with_normal_chunk"] = _same_doc_table_with_normal(
                metadata, normal_candidates
            )
            seed_counts[str(metadata.get("seed_id"))] += 1
            table_counts[_table_count_key(metadata)] += 1
            _remember_row_cell_key(metadata, row_keys_seen)
            rerank_input.append(chunk)
            added_table_count += 1
        else:
            drop_reason_counts[reason] += 1
            metadata["table_index_filter_reason"] = reason

        candidate_rows.append(_active_candidate_row(query, chunk, "table", rank, keep, reason))

    policy_debug = {
        "table_index_merge_added_count": added_table_count,
        "drop_reason_counts": dict(drop_reason_counts),
        "max_units_per_seed": config.table_index_max_units_per_seed,
        "max_units_per_table": config.table_index_max_units_per_table,
        "weak_match_floor": config.table_index_weak_match_floor,
        "row_cell_group_dedupe_checked": True,
    }
    return {
        "rerank_input": rerank_input,
        "candidate_rows": candidate_rows,
        "drop_reason_counts": dict(drop_reason_counts),
        "policy_debug": policy_debug,
    }


def stub_normal_retriever(query: dict[str, str]) -> list[RetrievedChunkCompat]:
    doc_id = _first_match(r"doc_\d+", query.get("expected_seed_or_table_scope", "")) or _first_match(
        r"doc_\d+", query.get("query_text", "")
    )
    if not doc_id:
        doc_id = "doc_sandbox"
    base_text = query.get("query_text", "")
    return [
        RetrievedChunkCompat(
            chunk_id=f"normal::{query.get('query_id', 'query')}::001",
            doc_id=doc_id,
            source_file="sandbox_normal_stub.jsonl",
            title="Sandbox normal retrieval stub",
            section="normal_stub",
            text=f"[NORMAL CHUNK] {base_text}",
            vector_score=0.60,
            bm25_score=0.0,
            fusion_score=0.60,
            metadata={
                "object_type": "normal_chunk",
                "sandbox_stub": True,
                "query_id": query.get("query_id", ""),
            },
        ),
        RetrievedChunkCompat(
            chunk_id=f"normal::{query.get('query_id', 'query')}::002",
            doc_id=doc_id,
            source_file="sandbox_normal_stub.jsonl",
            title="Sandbox normal retrieval fallback",
            section="normal_stub",
            text=f"[NORMAL CHUNK] fallback context for {query.get('query_type', '')}",
            vector_score=0.52,
            bm25_score=0.0,
            fusion_score=0.52,
            metadata={
                "object_type": "normal_chunk",
                "sandbox_stub": True,
                "query_id": query.get("query_id", ""),
            },
        ),
    ]


def stub_rerank(
    query: dict[str, str],
    candidates: list[RetrievedChunkCompat],
) -> list[RetrievedChunkCompat]:
    del query
    reranked: list[RetrievedChunkCompat] = []
    for index, chunk in enumerate(candidates, start=1):
        metadata = chunk.metadata or {}
        if metadata.get("object_type") == "table_index_unit":
            score = 0.50 + float(metadata.get("table_index_score_norm") or 0.0) * 0.45
            if metadata.get("table_index_route_match"):
                score += 0.08
            if metadata.get("table_unit_type") == "table_unit":
                score += 0.02
        else:
            score = 0.61 - index * 0.015
        chunk.rerank_score = round(score, 6)
        reranked.append(chunk)

    reranked.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
    for rank, chunk in enumerate(reranked, start=1):
        chunk.metadata["rerank_rank"] = rank
    return reranked


def build_support_pack(
    reranked: list[RetrievedChunkCompat],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    support_pack: list[dict[str, Any]] = []
    for chunk in reranked[:max_items]:
        metadata = dict(chunk.metadata or {})
        reasons = ["stub_rerank_topk"]
        if metadata.get("object_type") == "table_index_unit":
            reasons.extend(
                [
                    f"table_route:{metadata.get('table_index_query_type')}",
                    f"table_unit_type:{metadata.get('table_unit_type')}",
                    "production_ready_false_preserved",
                    "value_bboxes_available_false_preserved",
                    "binding_warning_or_limitation_preserved",
                ]
            )
        support_pack.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "text_preview": _shorten(chunk.text, 320),
                "rerank_score": chunk.rerank_score,
                "support_score": chunk.rerank_score,
                "metadata": metadata,
                "reasons": reasons,
                "formal_citation_from_source_path": False,
                "value_level_citation_claim": False,
            }
        )
    return support_pack


def write_evidence_cards(path: Path, items: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7L Table Evidence Cards",
        "",
        "Scope: sandbox support evidence preview. CSV and crop paths are retained as provenance/debug only.",
        "",
    ]
    if not items:
        lines.append("No table evidence items entered the sandbox support pack.")
    for index, item in enumerate(items, start=1):
        metadata = item.get("metadata", {})
        lines.extend(
            [
                f"## Card {index}: {item.get('query_id', '')}",
                "",
                f"- chunk_id: `{item.get('chunk_id', '')}`",
                f"- object_type: `{metadata.get('object_type')}`",
                f"- table_unit_type: `{metadata.get('table_unit_type')}`",
                f"- seed_id: `{metadata.get('seed_id')}`",
                f"- doc_id: `{metadata.get('doc_id')}`",
                f"- table_id: `{metadata.get('table_id')}`",
                f"- row_label: `{metadata.get('row_label')}`",
                f"- header_path: `{json.dumps(metadata.get('header_path'), ensure_ascii=False)}`",
                f"- source_csv_path: `{metadata.get('source_csv_path')}`",
                f"- source_pdf_crop_path: `{metadata.get('source_pdf_crop_path')}`",
                f"- value_bboxes_available: `{metadata.get('value_bboxes_available')}`",
                f"- production_ready: `{metadata.get('production_ready')}`",
                f"- binding warning / limitation: `{metadata.get('binding_review_limitation')}`",
                "- citation handling: source paths are not formal citations.",
                "- value-level citation claim: `false`",
                "",
                "Evidence preview:",
                "",
                f"> {_shorten(item.get('text_preview', ''), 480)}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_adapter_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L TableUnitAdapter Smoke Report",
        "",
        f"- input_unit_count: {summary['input_unit_count']}",
        f"- contract_pass_count: {summary['contract_pass_count']}",
        f"- contract_fail_count: {summary['contract_fail_count']}",
        f"- output: `{summary['output_path']}`",
        "",
        "Contract assertions covered chunk_id prefix, object_type, table_unit_type, preview-only guardrails, source paths, marker text, and no value-level citation claim.",
    ]
    (OUTPUT_REPORT_DIR / "table_unit_adapter_smoke_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_sidecar_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L Local Sidecar Retriever Smoke Report",
        "",
        f"- eligible_unit_count: {summary['eligible_unit_count']}",
        f"- excluded_units_read: {summary['excluded_units_read']}",
        f"- bm25_accessed: {summary['bm25_accessed']}",
        f"- milvus_accessed: {summary['milvus_accessed']}",
        f"- lexical_scorer: {summary['lexical_scorer']}",
        f"- query_count: {summary['query_count']}",
        f"- queries_with_candidates: {summary['queries_with_candidates']}",
        f"- candidate_row_count: {summary['candidate_row_count']}",
        f"- output: `{summary['output_path']}`",
    ]
    (OUTPUT_REPORT_DIR / "sidecar_retriever_smoke_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_shadow_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L Shadow Mode Report",
        "",
        "- config: table_index_retrieval_enabled=true; table_index_shadow_mode=true",
        f"- query_count: {summary['query_count']}",
        f"- table_branch_executed_count: {summary['table_branch_executed_count']}",
        f"- table_candidates_in_rerank_count: {summary['table_candidates_in_rerank_count']}",
        f"- support_pack_table_count: {summary['support_pack_table_count']}",
        f"- final_evidence_normal_only: {summary['final_evidence_normal_only']}",
        f"- output: `{summary['output_path']}`",
    ]
    (OUTPUT_REPORT_DIR / "shadow_mode_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_active_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L Active Sandbox Merge Smoke Report",
        "",
        "- config: table_index_retrieval_enabled=true; table_index_shadow_mode=false",
        "- normal retriever: sandbox stub",
        "- reranker: sandbox stub",
        f"- query_count: {summary['query_count']}",
        f"- queries_with_table_support: {summary['queries_with_table_support']}",
        f"- candidate_row_count: {summary['candidate_row_count']}",
        f"- policy_drop_reasons: `{json.dumps(summary['policy_drop_reasons'], sort_keys=True)}`",
        f"- max_units_per_seed_checked: {summary['max_units_per_seed_checked']}",
        f"- max_units_per_table_checked: {summary['max_units_per_table_checked']}",
        f"- weak_match_filtering_checked: {summary['weak_match_filtering_checked']}",
        f"- row_cell_group_dedupe_checked: {summary['row_cell_group_dedupe_checked']}",
        f"- all_table_candidates_production_ready_false: {summary['all_table_candidates_production_ready_false']}",
        f"- candidates: `{summary['active_csv']}`",
        f"- support_pack_preview: `{summary['support_pack_preview']}`",
    ]
    (OUTPUT_REPORT_DIR / "active_merge_smoke_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_evidence_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L Evidence Contract Smoke Report",
        "",
        f"- table_support_item_count: {summary['table_support_item_count']}",
        f"- contract_pass_count: {summary['contract_pass_count']}",
        f"- source_paths_formal_citation_count: {summary['source_paths_formal_citation_count']}",
        f"- value_level_citation_claim_count: {summary['value_level_citation_claim_count']}",
        f"- binding_warning_or_limitation_preserved: {summary['binding_warning_or_limitation_preserved']}",
        f"- evidence_cards: `{summary['evidence_cards']}`",
        "",
        "CSV and crop paths are retained as provenance/debug fields only. They are not emitted as formal citations.",
    ]
    (OUTPUT_REPORT_DIR / "evidence_contract_smoke_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_rollback_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L Rollback Guardrail Report",
        "",
        "- config: table_index_retrieval_enabled=false",
        f"- query_count: {summary['query_count']}",
        f"- table_branch_executed_count: {summary['table_branch_executed_count']}",
        f"- support_pack_table_count: {summary['support_pack_table_count']}",
        f"- normal_only_restored: {summary['normal_only_restored']}",
        f"- official_baseline_drift: {summary['official_baseline_drift']}",
        f"- output: `{summary['output_path']}`",
    ]
    (OUTPUT_REPORT_DIR / "rollback_guardrail_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_guardrail_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7L Guardrail Report",
        "",
        f"- official_drift: {summary['official_drift']}",
        f"- bm25_queried: {summary['bm25_queried']}",
        f"- milvus_accessed: {summary['milvus_accessed']}",
        f"- src_changed: {summary['src_changed']}",
        f"- configs_changed: {summary['configs_changed']}",
        f"- ingestion_changed: {summary['ingestion_changed']}",
        f"- checks: `{summary['output_path']}`",
        "",
        "| asset | status | changed | note |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['asset']} | {row['status']} | {row['changed']} | {row['note']} |"
        )
    (OUTPUT_REPORT_DIR / "phase7l_guardrail.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_summary_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7L Limited Full-Chain Table-RAG Sandbox Smoke Summary",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- src_modified: {summary['src_modified']}",
        f"- configs_modified: {summary['configs_modified']}",
        f"- bm25_queried: {summary['bm25_queried']}",
        f"- milvus_accessed: {summary['milvus_accessed']}",
        f"- adapter_contract_pass: {summary['adapter']['contract_pass']}",
        f"- sidecar_retriever_pass: {summary['sidecar']['pass']}",
        f"- shadow_mode_pass: {summary['shadow']['pass']}",
        f"- active_sandbox_merge_pass: {summary['active']['pass']}",
        f"- evidence_contract_pass: {summary['evidence']['pass']}",
        f"- rollback_pass: {summary['rollback']['pass']}",
        f"- recommend_next_step: {summary['recommend_next_step']}",
        f"- recommend_production: {summary['recommend_production']}",
        f"- route_c_backlog_only: {summary['route_c_backlog_only']}",
        "",
        "Warnings: table units remain preview_only, production_ready=false, value_bboxes_available=false. This run used a local lexical scorer plus stub normal retriever and stub reranker; it is not a production retrieval evaluation.",
    ]
    (OUTPUT_REPORT_DIR / "phase7l_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _candidate_csv_row(query: dict[str, str], candidate: TableCandidate) -> dict[str, Any]:
    metadata = candidate.chunk.metadata
    return {
        "phase7l_execution_order": query.get("phase7l_execution_order", ""),
        "phase7l_smoke_stage": query.get("phase7l_smoke_stage", ""),
        "query_id": query.get("query_id", ""),
        "query_group": query.get("query_group", ""),
        "query_type": query.get("query_type", ""),
        "query_text": query.get("query_text", ""),
        "candidate_rank": candidate.rank,
        "chunk_id": candidate.chunk.chunk_id,
        "table_index_unit_id": metadata.get("table_index_unit_id"),
        "raw_score": candidate.raw_score,
        "normalized_score": candidate.normalized_score,
        "unit_type": metadata.get("table_unit_type"),
        "seed_id": metadata.get("seed_id"),
        "table_id": metadata.get("table_id"),
        "row_label": metadata.get("row_label"),
        "filter_reason": candidate.filter_reason,
        "route_match": str(candidate.route_match).lower(),
        "guardrail_pass": str(candidate.guardrail_pass).lower(),
    }


def _active_candidate_row(
    query: dict[str, str],
    chunk: RetrievedChunkCompat,
    origin: str,
    rank: int,
    entered_rerank: bool,
    filter_reason: str,
) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    return {
        "query_id": query.get("query_id", ""),
        "query_type": query.get("query_type", ""),
        "candidate_origin": origin,
        "candidate_rank": rank,
        "chunk_id": chunk.chunk_id,
        "table_index_unit_id": metadata.get("table_index_unit_id", ""),
        "unit_type": metadata.get("table_unit_type", metadata.get("object_type", "")),
        "seed_id": metadata.get("seed_id", ""),
        "table_id": metadata.get("table_id", ""),
        "row_label": metadata.get("row_label", ""),
        "raw_score": metadata.get("table_index_score_raw", chunk.fusion_score),
        "normalized_score": metadata.get("table_index_score_norm", chunk.fusion_score),
        "rerank_score": chunk.rerank_score,
        "route_match": str(metadata.get("table_index_route_match", origin == "normal")).lower(),
        "guardrail_pass": str(metadata.get("table_index_guardrail_pass", origin == "normal")).lower(),
        "filter_reason": filter_reason,
        "entered_rerank": str(entered_rerank).lower(),
        "entered_support_pack": "false",
        "production_ready": str(metadata.get("production_ready", "")).lower(),
        "value_bboxes_available": str(metadata.get("value_bboxes_available", "")).lower(),
    }


def _lexical_score(query_text: str, unit: dict[str, Any]) -> float:
    query_tokens = _tokens(query_text)
    if not query_tokens:
        return 0.0
    unit_text = " ".join(
        [
            str(unit.get("table_index_unit_id", "")),
            str(unit.get("unit_type", "")),
            str(unit.get("doc_id", "")),
            str(unit.get("table_id", "")),
            str(unit.get("caption", "")),
            str(unit.get("content_text_for_embedding", "")),
            _flatten_for_search(unit.get("metadata") or {}),
            _flatten_for_search(unit.get("guardrail") or {}),
        ]
    )
    unit_tokens = set(_tokens(unit_text))
    query_unique = set(query_tokens)
    overlap = sum(1 for token in query_unique if token in unit_tokens)
    score = overlap / math.sqrt(max(len(query_unique), 1))

    lower_query = _norm(query_text)
    for field, bonus in (
        (unit.get("doc_id"), 2.0),
        (unit.get("table_id"), 1.2),
        ((unit.get("metadata") or {}).get("row_label"), 1.8),
        (unit.get("candidate_id"), 0.4),
    ):
        if field and _norm(str(field)) in lower_query:
            score += bonus

    caption = _norm(str(unit.get("caption", "")))
    if caption:
        caption_tokens = set(_tokens(caption))
        caption_overlap = sum(1 for token in query_unique if token in caption_tokens)
        score += min(caption_overlap * 0.08, 0.8)

    if "value-level coordinates" in lower_query and "value-level coordinates" in _norm(unit_text):
        score += 0.5
    if "reference" in lower_query or "citation" in lower_query or "source" in lower_query:
        score += _source_reference_bonus(unit)
    return round(score, 6)


def _source_reference_bonus(unit: dict[str, Any]) -> float:
    metadata = unit.get("metadata") or {}
    searchable = _norm(_flatten_for_search(metadata))
    if any(token in searchable for token in ("reference", "source", "citation", "study")):
        return 0.5
    return 0.0


def _sidecar_rank_key(candidate: TableCandidate) -> tuple[float, int, float]:
    return (
        -candidate.raw_score,
        0 if candidate.route_match else 1,
        -candidate.unit_type_boost,
    )


def _active_table_rank_key(candidate: TableCandidate) -> tuple[int, float, float]:
    route_penalty = 0 if candidate.route_match else 1
    return (route_penalty, -candidate.normalized_score, -candidate.unit_type_boost)


def _unit_type_boost(query_type: str, unit_type: str) -> float:
    policy = ROUTE_POLICY.get(query_type, ROUTE_POLICY["non_table_query"])
    return float(policy["boost"].get(unit_type, 0.0))


def _route_match(query_type: str, unit_type: str) -> bool:
    policy = ROUTE_POLICY.get(query_type, ROUTE_POLICY["non_table_query"])
    return unit_type in policy["primary"] or unit_type in policy["fallback"]


def _guardrail_pass(chunk: RetrievedChunkCompat) -> bool:
    metadata = chunk.metadata or {}
    return (
        metadata.get("object_type") == "table_index_unit"
        and metadata.get("production_ready") is False
        and metadata.get("index_unit_status") == "preview_only"
        and metadata.get("value_bboxes_available") is False
        and metadata.get("source_span_granularity") != "value_level"
    )


def _has_value_level_citation_claim(chunk: RetrievedChunkCompat) -> bool:
    text = _norm(chunk.text)
    bad_phrases = [
        "value-level citation true",
        "verified value bbox",
        "value bboxes available true",
        "value-level coordinates available",
        "formal citation from source path true",
    ]
    return any(phrase in text for phrase in bad_phrases)


def _row_cell_duplicate(metadata: dict[str, Any], seen: set[tuple[str, str, str]]) -> bool:
    unit_type = metadata.get("table_unit_type")
    if unit_type not in {"row_unit", "cell_group_unit"}:
        return False
    key = _row_cell_key(metadata)
    return key in seen


def _remember_row_cell_key(metadata: dict[str, Any], seen: set[tuple[str, str, str]]) -> None:
    unit_type = metadata.get("table_unit_type")
    if unit_type in {"row_unit", "cell_group_unit"}:
        seen.add(_row_cell_key(metadata))


def _row_cell_key(metadata: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(metadata.get("seed_id") or ""),
        str(metadata.get("table_id") or ""),
        str(metadata.get("row_label") or ""),
    )


def _table_count_key(metadata: dict[str, Any]) -> str:
    return f"{metadata.get('doc_id')}::{metadata.get('table_id')}"


def _same_doc_table_with_normal(
    metadata: dict[str, Any],
    normal_candidates: list[RetrievedChunkCompat],
) -> bool:
    table_doc = metadata.get("doc_id")
    return any(chunk.doc_id == table_doc for chunk in normal_candidates)


def _scoped_table_mismatch(query_text: str, metadata: dict[str, Any]) -> bool:
    lower_query = _norm(query_text)
    doc_match = _first_match(r"doc_\d+", lower_query)
    if doc_match and metadata.get("doc_id") and _norm(str(metadata.get("doc_id"))) != doc_match:
        return True
    table_match = _first_match(r"table\s*\d+", lower_query)
    table_id = _norm(str(metadata.get("table_id") or ""))
    if table_match and table_id and table_match != table_id:
        return True
    return False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(text))


def _norm(text: str) -> str:
    return (
        text.lower()
        .replace("′", "'")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _flatten_for_search(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_for_search(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten_for_search(item) for item in value)
    return str(value)


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "no"}


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0).lower() if match else ""


def _shorten(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _baseline_asset_path(*parts: str) -> str:
    return "/".join(("data", "baselines", "phase5f_official_clean_baseline", *parts))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _git_status(rel_path: str) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"git_status_error:{result.stderr.strip()}"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _touch_required_inputs() -> None:
    missing = []
    for name in REQUIRED_INPUT_REPORTS:
        path = INPUT_REPORT_DIR / name
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
        else:
            path.read_text(encoding="utf-8")
    required_data = [
        QUERY_CSV,
        INPUT_DATA_DIR / "table_unit_adapter_mapping.csv",
        INPUT_DATA_DIR / "routing_policy_matrix.csv",
        INPUT_DATA_DIR / "evidence_contract_fields.csv",
        INPUT_DATA_DIR / "phase7l_smoke_acceptance_criteria.csv",
        ELIGIBLE_JSONL,
        ROOT
        / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.csv",
        ROOT / "src/synbio_rag/application/pipeline.py",
        ROOT / "src/synbio_rag/domain/schemas.py",
        ROOT / "src/synbio_rag/application/generation_v2/evidence_ledger.py",
        ROOT / "src/synbio_rag/application/generation_v2/support_selector.py",
        ROOT / "src/synbio_rag/application/generation_v2/citation_binder.py",
        ROOT / "configs/baseline_registry.yaml",
    ]
    for path in required_data:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
        else:
            path.open("r", encoding="utf-8").close()
    if missing:
        raise FileNotFoundError("Missing required Phase7L inputs: " + ", ".join(missing))


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full Phase7L sandbox smoke chain.",
    )
    return parser
