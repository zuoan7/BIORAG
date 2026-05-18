from __future__ import annotations

import argparse
import csv
import importlib.util
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


PHASE_DIR = "v7_phase7_table_preview_eval_smoke"
UNITS_PATH = (
    ROOT
    / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
SOURCE_QUERY_SET = (
    ROOT / "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.jsonl"
)
FIXTURE_PATH = ROOT / f"data/experiments/{PHASE_DIR}/query_fixture.jsonl"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"
DEFAULT_RERANKER_MODEL_PATH = ROOT / "models/BAAI/bge-reranker-v2-m3"

TABLE_QUERY_TYPES = {"table_lookup", "row_lookup", "metric_lookup"}
REQUIRED_QUERY_TYPES = TABLE_QUERY_TYPES | {"non_table_control"}
REQUIRED_QUERY_FIELDS = {
    "query_id",
    "query_text",
    "query_type",
    "expected_table_id",
    "expected_unit_type",
}


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
    source_query_set: Path = SOURCE_QUERY_SET,
    units_path: Path = UNITS_PATH,
    output_path: Path = FIXTURE_PATH,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    units = load_preview_units(units_path)
    source_queries = load_jsonl(source_query_set)
    selected: list[dict[str, Any]] = []
    quotas = {"table_lookup": 4, "row_lookup": 3, "metric_lookup": 3}

    for query_type, quota in quotas.items():
        candidates = [row for row in source_queries if row.get("query_type") == query_type]
        selected.extend(candidates[:quota])

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        rows.append(
            {
                "query_id": f"phase7u_query_{idx:03d}",
                "source_query_id": row.get("query_id", ""),
                "query_text": row["query_text"],
                "query_type": row["query_type"],
                "expected_doc_id": row.get("expected_doc_id", ""),
                "expected_table_id": row.get("expected_table_id", ""),
                "expected_table_index_unit_id": row.get("expected_table_index_unit_id", ""),
                "expected_unit_type": row.get("expected_unit_type", ""),
                "expected_row_label": row.get("expected_row_label", ""),
                "query_notes": "phase7u runtime preview smoke fixture",
            }
        )

    non_table_controls = [
        {
            "query_text": "Summarize doc_0075 abstract and study motivation.",
            "expected_doc_id": "doc_0075",
        },
        {
            "query_text": "Explain the background objective for doc_0261 infant microbiota study.",
            "expected_doc_id": "doc_0261",
        },
        {
            "query_text": "What organism context is discussed in doc_0066 protein localization work?",
            "expected_doc_id": "doc_0066",
        },
    ]
    for offset, control in enumerate(non_table_controls, start=len(rows) + 1):
        rows.append(
            {
                "query_id": f"phase7u_query_{offset:03d}",
                "source_query_id": "",
                "query_text": control["query_text"],
                "query_type": "non_table_control",
                "expected_doc_id": control["expected_doc_id"],
                "expected_table_id": "",
                "expected_table_index_unit_id": "",
                "expected_unit_type": "none",
                "expected_row_label": "",
                "query_notes": "non-table control with doc_id overlap for merge guard smoke",
            }
        )

    write_jsonl(output_path, rows)
    summary = validate_query_fixture_payload(rows=rows, units=units, fixture_path=output_path)
    summary["source_query_set"] = str(source_query_set)
    summary["units_path"] = str(units_path)
    if summary_path is None:
        summary_path = RESULTS_DIR / "query_fixture_summary.json"
    write_json(summary_path, summary)
    return summary


def validate_query_fixture_payload(
    *,
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    fixture_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    unit_ids = {str(unit.get("table_index_unit_id", "")) for unit in units}
    counts = Counter(row.get("query_type", "") for row in rows)

    if len(units) != 274:
        errors.append(f"expected 274 preview units, got {len(units)}")
    if not 10 <= len(rows) <= 20:
        errors.append(f"expected 10-20 fixture queries, got {len(rows)}")
    missing_types = REQUIRED_QUERY_TYPES - set(counts)
    if missing_types:
        errors.append(f"missing query types: {sorted(missing_types)}")
    if counts.get("non_table_control", 0) < 2:
        errors.append("expected at least 2 non_table_control queries")

    for row in rows:
        missing_fields = REQUIRED_QUERY_FIELDS - set(row)
        if missing_fields:
            errors.append(f"{row.get('query_id', '<missing_id>')} missing fields {sorted(missing_fields)}")
        if row.get("query_type") in TABLE_QUERY_TYPES:
            expected_unit_id = str(row.get("expected_table_index_unit_id", ""))
            if expected_unit_id not in unit_ids:
                errors.append(f"{row.get('query_id')} expected unit not in preview units")
            if not row.get("expected_table_id"):
                errors.append(f"{row.get('query_id')} missing expected_table_id")
            if row.get("expected_unit_type") not in {"table_unit", "row_unit", "cell_group_unit"}:
                errors.append(f"{row.get('query_id')} has invalid expected_unit_type")
        elif row.get("query_type") == "non_table_control":
            if row.get("expected_unit_type") != "none":
                errors.append(f"{row.get('query_id')} non-table control must use expected_unit_type=none")

    return {
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "preview_unit_count": len(units),
        "query_count": len(rows),
        "query_type_counts": dict(sorted(counts.items())),
        "required_query_types": sorted(REQUIRED_QUERY_TYPES),
        "non_table_control_count": counts.get("non_table_control", 0),
    }


def ensure_fixture(fixture_path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    if not fixture_path.exists():
        build_query_fixture(output_path=fixture_path)
    return load_jsonl(fixture_path)


def preview_config(*, enabled: bool, merge_enabled: bool) -> RetrievalConfig:
    return RetrievalConfig(
        table_preview_enabled=enabled,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_enabled=merge_enabled,
        table_preview_merge_max_candidates=5,
        table_preview_min_score=0.05,
        table_preview_allow_formal_citation=False,
        rerank_score_floor_ratio=0.0,
    )


def normal_retrieved(query_id: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"normal::{query_id}",
            doc_id="normal_doc",
            source_file="normal_only_stub.pdf",
            title="Normal retrieval stub",
            section="Abstract",
            text="Normal retrieval evidence stub used only for Phase7U runtime preview smoke.",
            vector_score=0.2,
            bm25_score=0.0,
            rerank_score=0.0,
            fusion_score=0.2,
            metadata={"object_type": "normal_chunk", "phase7u_stub_normal": True},
        )
    ]


def preview_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.metadata.get("object_type") == "table_index_unit"]


def run_shadow_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    queries = ensure_fixture(fixture_path)
    config = preview_config(enabled=True, merge_enabled=False)
    records: list[dict[str, Any]] = []

    for query in queries:
        output, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=config,
        )
        expected_unit_id = query.get("expected_table_index_unit_id", "")
        candidate_ids = debug.get("candidate_table_index_unit_ids", [])
        is_table_query = query["query_type"] in TABLE_QUERY_TYPES
        records.append(
            {
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "is_table_query": is_table_query,
                "mode": debug.get("mode", ""),
                "reason": debug.get("reason", ""),
                "input_count": debug.get("input_count", 0),
                "output_count": debug.get("output_count", 0),
                "candidate_count": debug.get("candidate_count", 0),
                "expected_table_index_unit_id": expected_unit_id,
                "expected_candidate_seen": bool(expected_unit_id and expected_unit_id in candidate_ids),
                "table_candidates_in_rerank_input": debug.get("table_candidates_in_rerank_input", False),
                "preview_output_count": len(preview_chunks(output)),
                "top_candidate_ids": candidate_ids[:5],
            }
        )

    table_records = [row for row in records if row["is_table_query"]]
    errors = []
    for row in table_records:
        if row["mode"] != "shadow":
            errors.append(f"{row['query_id']} not in shadow mode")
        if int(row["candidate_count"]) <= 0:
            errors.append(f"{row['query_id']} has no table candidates")
        if row["table_candidates_in_rerank_input"]:
            errors.append(f"{row['query_id']} leaked candidates into rerank input in shadow mode")
        if int(row["preview_output_count"]) != 0:
            errors.append(f"{row['query_id']} returned preview chunks in shadow mode")

    summary = {
        "smoke": "shadow",
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "config": {
            "TABLE_PREVIEW_ENABLED": True,
            "TABLE_PREVIEW_MERGE_ENABLED": False,
            "GENERATION_V2_USE_QWEN_SYNTHESIS": False,
        },
        "query_count": len(records),
        "table_query_count": len(table_records),
        "table_query_candidate_count_positive": sum(
            1 for row in table_records if int(row["candidate_count"]) > 0
        ),
        "expected_hit_at_20": sum(1 for row in table_records if row["expected_candidate_seen"]),
        "records_path": str(output_dir / "shadow_smoke_records.jsonl"),
    }
    write_jsonl(output_dir / "shadow_smoke_records.jsonl", records)
    write_json(output_dir / "shadow_smoke_summary.json", summary)
    return summary


def run_merge_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    queries = ensure_fixture(fixture_path)
    config = preview_config(enabled=True, merge_enabled=True)
    records: list[dict[str, Any]] = []

    for query in queries:
        output, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=config,
        )
        table_output = preview_chunks(output)
        expected_unit_id = query.get("expected_table_index_unit_id", "")
        merged_ids = [chunk.metadata.get("table_index_unit_id", "") for chunk in table_output]
        is_table_query = query["query_type"] in TABLE_QUERY_TYPES
        records.append(
            {
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "is_table_query": is_table_query,
                "mode": debug.get("mode", ""),
                "reason": debug.get("reason", ""),
                "candidate_count": debug.get("candidate_count", 0),
                "merged_count": debug.get("merged_count", 0),
                "table_candidates_in_rerank_input": debug.get("table_candidates_in_rerank_input", False),
                "preview_output_count": len(table_output),
                "expected_table_index_unit_id": expected_unit_id,
                "expected_candidate_merged": bool(expected_unit_id and expected_unit_id in merged_ids),
                "merged_table_index_unit_ids": merged_ids,
            }
        )

    errors: list[str] = []
    for row in records:
        if row["is_table_query"]:
            if row["mode"] != "merged_preview":
                errors.append(f"{row['query_id']} table-like query did not enter merged_preview")
            if int(row["merged_count"]) <= 0:
                errors.append(f"{row['query_id']} did not merge preview candidates")
            if not row["table_candidates_in_rerank_input"]:
                errors.append(f"{row['query_id']} did not mark rerank input as containing preview candidates")
        else:
            if row["mode"] != "merge_blocked":
                errors.append(f"{row['query_id']} non-table control was not blocked")
            if int(row["preview_output_count"]) != 0:
                errors.append(f"{row['query_id']} non-table control leaked preview chunks")
            if row["table_candidates_in_rerank_input"]:
                errors.append(f"{row['query_id']} non-table control marked preview rerank input")

    table_records = [row for row in records if row["is_table_query"]]
    non_table_records = [row for row in records if not row["is_table_query"]]
    summary = {
        "smoke": "merge",
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "config": {
            "TABLE_PREVIEW_ENABLED": True,
            "TABLE_PREVIEW_MERGE_ENABLED": True,
            "GENERATION_V2_USE_QWEN_SYNTHESIS": False,
        },
        "query_count": len(records),
        "table_query_count": len(table_records),
        "non_table_control_count": len(non_table_records),
        "merged_table_query_count": sum(1 for row in table_records if int(row["merged_count"]) > 0),
        "blocked_non_table_count": sum(1 for row in non_table_records if row["mode"] == "merge_blocked"),
        "expected_merged_at_5": sum(1 for row in table_records if row["expected_candidate_merged"]),
        "records_path": str(output_dir / "merge_smoke_records.jsonl"),
    }
    write_jsonl(output_dir / "merge_smoke_records.jsonl", records)
    write_json(output_dir / "merge_smoke_summary.json", summary)
    return summary


class ForbiddenPreviewProvider:
    def __init__(self) -> None:
        self.called = False

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.called = True
        raise AssertionError("preview provider must not run when TABLE_PREVIEW_ENABLED=false")


def run_rollback_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    queries = ensure_fixture(fixture_path)
    config = preview_config(enabled=False, merge_enabled=False)
    provider = ForbiddenPreviewProvider()
    records: list[dict[str, Any]] = []

    for query in queries:
        retrieved = normal_retrieved(query["query_id"])
        output, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=retrieved,
            config=config,
            provider=provider,  # type: ignore[arg-type]
        )
        records.append(
            {
                "query_id": query["query_id"],
                "mode": debug.get("mode", ""),
                "reason": debug.get("reason", ""),
                "enabled": debug.get("enabled", None),
                "table_branch_executed": debug.get("table_branch_executed", None),
                "input_chunk_ids": [chunk.chunk_id for chunk in retrieved],
                "output_chunk_ids": [chunk.chunk_id for chunk in output],
                "preview_output_count": len(preview_chunks(output)),
            }
        )

    errors: list[str] = []
    if provider.called:
        errors.append("preview provider was called while table preview flag was off")
    for row in records:
        if row["enabled"] is not False:
            errors.append(f"{row['query_id']} did not report enabled=false")
        if row["table_branch_executed"] is not False:
            errors.append(f"{row['query_id']} executed table branch while disabled")
        if row["input_chunk_ids"] != row["output_chunk_ids"]:
            errors.append(f"{row['query_id']} changed normal-only retrieval output")
        if int(row["preview_output_count"]) != 0:
            errors.append(f"{row['query_id']} emitted preview chunks while disabled")

    summary = {
        "smoke": "rollback",
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "config": {
            "TABLE_PREVIEW_ENABLED": False,
            "TABLE_PREVIEW_MERGE_ENABLED": False,
            "GENERATION_V2_USE_QWEN_SYNTHESIS": False,
        },
        "query_count": len(records),
        "provider_called": provider.called,
        "records_path": str(output_dir / "rollback_smoke_records.jsonl"),
    }
    write_jsonl(output_dir / "rollback_smoke_records.jsonl", records)
    write_json(output_dir / "rollback_smoke_summary.json", summary)
    return summary


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
        reasons=["phase7u_table_preview"],
    )


def run_citation_guard_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    queries = [query for query in ensure_fixture(fixture_path) if query["query_type"] in TABLE_QUERY_TYPES]
    config = preview_config(enabled=True, merge_enabled=True)
    binder = CitationBinder()
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    if not queries:
        errors.append("no table-oriented queries in fixture")
    else:
        output, debug = apply_table_preview(
            question=queries[0]["query_text"],
            retrieved=normal_retrieved(queries[0]["query_id"]),
            config=config,
        )
        table_chunks = preview_chunks(output)[:3]
        if not table_chunks:
            errors.append("merge smoke did not produce preview table chunks for citation guard")
        for idx, chunk in enumerate(table_chunks, start=1):
            evidence_id = f"E{idx}"
            candidate = evidence_candidate_from_chunk(evidence_id, chunk)
            support = [SupportItem(evidence_id, candidate, 0.9, ["selected_preview_table"])]
            candidates = binder.build_citation_candidates(support)
            answer, citations, citation_debug = binder.bind(
                f"Preview-only table evidence [{evidence_id}].",
                support,
                citation_candidates=candidates,
            )
            blocked = citation_debug.get("blocked_evidence_ids", [])
            drop_reasons = citation_debug.get("drop_reasons_by_evidence_id", {})
            record = {
                "query_id": queries[0]["query_id"],
                "chunk_id": chunk.chunk_id,
                "table_index_unit_id": chunk.metadata.get("table_index_unit_id", ""),
                "citation_count": len(citations),
                "answer_contains_formal_marker": "[1]" in answer,
                "blocked_evidence_ids": blocked,
                "drop_reason": drop_reasons.get(evidence_id, ""),
                "candidate_eligible": candidates[0].citation_eligible,
                "table_preview_citation_block_reasons": candidate.metadata.get(
                    "table_preview_citation_block_reasons", []
                ),
                "source_debug_mode": debug.get("mode", ""),
            }
            records.append(record)
            if record["citation_count"] != 0:
                errors.append(f"{chunk.chunk_id} produced formal citation")
            if record["answer_contains_formal_marker"]:
                errors.append(f"{chunk.chunk_id} kept formal citation marker")
            if evidence_id not in blocked:
                errors.append(f"{chunk.chunk_id} did not report blocked evidence id")
            if record["drop_reason"] != "table_preview_formal_citation_blocked":
                errors.append(f"{chunk.chunk_id} drop reason was {record['drop_reason']!r}")
            if record["candidate_eligible"] is not False:
                errors.append(f"{chunk.chunk_id} citation candidate remained eligible")

    summary = {
        "smoke": "citation_guard",
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "record_count": len(records),
        "formal_citation_count": sum(int(row["citation_count"]) for row in records),
        "records_path": str(output_dir / "citation_guard_smoke_records.jsonl"),
    }
    write_jsonl(output_dir / "citation_guard_smoke_records.jsonl", records)
    write_json(output_dir / "citation_guard_smoke_summary.json", summary)
    return summary


def local_reranker_status(model_path: Path = DEFAULT_RERANKER_MODEL_PATH) -> dict[str, Any]:
    return {
        "model_path": str(model_path),
        "model_dir_exists": model_path.exists(),
        "weight_file_exists": (model_path / "model.safetensors").exists(),
        "flag_embedding_importable": importlib.util.find_spec("FlagEmbedding") is not None,
    }


def run_rerank_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_dir: Path = RESULTS_DIR,
    model_path: Path = DEFAULT_RERANKER_MODEL_PATH,
    max_queries: int = 3,
) -> dict[str, Any]:
    status = local_reranker_status(model_path)
    status["local_reranker_available"] = all(
        [
            status["model_dir_exists"],
            status["weight_file_exists"],
            status["flag_embedding_importable"],
        ]
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    if not status["local_reranker_available"]:
        summary = {
            "smoke": "rerank",
            "status": "skipped",
            "pass": True,
            "errors": [],
            "reason": "local reranker is not available",
            **status,
            "records_path": str(output_dir / "rerank_smoke_records.jsonl"),
        }
        write_jsonl(output_dir / "rerank_smoke_records.jsonl", records)
        write_json(output_dir / "rerank_smoke_summary.json", summary)
        return summary

    config = preview_config(enabled=True, merge_enabled=True)
    try:
        from src.synbio_rag.application.rerank_service import QwenReranker

        reranker = QwenReranker(
            api_base="",
            api_key="",
            model_path=str(model_path),
            service_url="",
            batch_size=4,
            use_fp16=False,
            retrieval_config=config,
        )
    except Exception as exc:  # pragma: no cover - depends on local model runtime
        errors.append(f"failed to initialize local reranker: {exc}")
        reranker = None

    if reranker is None or reranker.local_reranker is None:
        summary = {
            "smoke": "rerank",
            "status": "skipped",
            "pass": True,
            "errors": errors,
            "reason": "local reranker could not be initialized",
            **status,
            "records_path": str(output_dir / "rerank_smoke_records.jsonl"),
        }
        write_jsonl(output_dir / "rerank_smoke_records.jsonl", records)
        write_json(output_dir / "rerank_smoke_summary.json", summary)
        return summary

    queries = [query for query in ensure_fixture(fixture_path) if query["query_type"] in TABLE_QUERY_TYPES][
        :max_queries
    ]
    for query in queries:
        merged, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=config,
        )
        input_preview = preview_chunks(merged)
        try:
            reranked = reranker.rerank(
                query["query_text"],
                merged,
                top_k=len(merged),
                analysis=None,
                mode="plain",
            )
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            errors.append(f"{query['query_id']} rerank failed: {exc}")
            reranked = []
        output_preview = preview_chunks(reranked)
        input_ids = [chunk.chunk_id for chunk in input_preview]
        output_ids = [chunk.chunk_id for chunk in output_preview]
        metadata_ok = all(
            chunk.metadata.get("object_type") == "table_index_unit"
            and chunk.metadata.get("table_preview") is True
            and bool(chunk.metadata.get("table_index_unit_id"))
            and chunk.metadata.get("index_unit_status") == "preview_only"
            for chunk in output_preview
        )
        record = {
            "query_id": query["query_id"],
            "query_type": query["query_type"],
            "merge_mode": debug.get("mode", ""),
            "rerank_input_count": len(merged),
            "preview_input_count": len(input_preview),
            "rerank_output_count": len(reranked),
            "preview_output_count": len(output_preview),
            "preview_input_chunk_ids": input_ids,
            "preview_output_chunk_ids": output_ids,
            "preview_metadata_preserved": metadata_ok,
            "reranker_debug_mode": reranker.last_debug.get("mode", ""),
        }
        records.append(record)
        if not input_preview:
            errors.append(f"{query['query_id']} had no preview chunks in rerank input")
        if set(input_ids) - set(output_ids):
            errors.append(f"{query['query_id']} lost preview chunks after local rerank")
        if not metadata_ok:
            errors.append(f"{query['query_id']} lost preview metadata after local rerank")

    summary = {
        "smoke": "rerank",
        "status": "passed" if not errors else "failed",
        "pass": not errors,
        "errors": errors,
        **status,
        "query_count": len(records),
        "records_path": str(output_dir / "rerank_smoke_records.jsonl"),
    }
    write_jsonl(output_dir / "rerank_smoke_records.jsonl", records)
    write_json(output_dir / "rerank_smoke_summary.json", summary)
    return summary


def validate_preview_eval_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    rows = ensure_fixture(fixture_path)
    fixture_summary = validate_query_fixture_payload(
        rows=rows,
        units=load_preview_units(UNITS_PATH),
        fixture_path=fixture_path,
    )
    summary_files = {
        "shadow": results_dir / "shadow_smoke_summary.json",
        "merge": results_dir / "merge_smoke_summary.json",
        "rollback": results_dir / "rollback_smoke_summary.json",
        "citation_guard": results_dir / "citation_guard_smoke_summary.json",
        "rerank": results_dir / "rerank_smoke_summary.json",
    }
    loaded: dict[str, dict[str, Any]] = {}
    errors = list(fixture_summary["errors"])

    for name, path in summary_files.items():
        if not path.exists():
            errors.append(f"missing {name} summary: {path}")
            continue
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))

    for name in ("shadow", "merge", "rollback", "citation_guard"):
        if name in loaded and loaded[name].get("pass") is not True:
            errors.append(f"{name} smoke failed: {loaded[name].get('errors', [])}")

    rerank_summary = loaded.get("rerank")
    if rerank_summary:
        if rerank_summary.get("local_reranker_available") and rerank_summary.get("status") != "passed":
            errors.append(f"local reranker available but rerank smoke did not pass: {rerank_summary}")
        if rerank_summary.get("pass") is not True:
            errors.append(f"rerank smoke failed: {rerank_summary.get('errors', [])}")

    validation = {
        "pass": not errors,
        "errors": errors,
        "fixture_summary": fixture_summary,
        "smoke_summaries": loaded,
        "guardrails": {
            "production_table_index_built": False,
            "preview_units_upgraded": False,
            "formal_table_citation_generated": False,
            "canonical_source_resolution": False,
            "llm_or_ragas_called": False,
            "milvus_accessed": False,
            "official_bm25_accessed": False,
            "embedding_run": False,
            "ingestion_pipeline_modified": False,
        },
    }
    write_json(results_dir / "validation_summary.json", validation)
    write_report(validation, reports_dir / "preview_eval_smoke_report.md")
    write_csv(
        results_dir / "validation_status.csv",
        [
            {"check": "fixture", "pass": fixture_summary["pass"]},
            *[
                {"check": name, "pass": summary.get("pass", False), "status": summary.get("status", "")}
                for name, summary in loaded.items()
            ],
            {"check": "overall", "pass": validation["pass"]},
        ],
        ["check", "pass", "status"],
    )
    return validation


def write_report(validation: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = validation.get("smoke_summaries", {})
    fixture = validation.get("fixture_summary", {})
    lines = [
        "# Phase7U Preview Retrieval + Rerank Evaluation Smoke",
        "",
        "This is a smoke / preview evaluation only. It does not build a production table index,",
        "does not promote preview units, and does not generate formal table citations.",
        "",
        "## Inputs",
        "",
        f"- Preview units: {fixture.get('preview_unit_count', 0)}",
        f"- Query fixture: {fixture.get('query_count', 0)} queries",
        f"- Query types: {fixture.get('query_type_counts', {})}",
        "",
        "## Smoke Results",
        "",
    ]
    for name in ("shadow", "merge", "rollback", "citation_guard", "rerank"):
        summary = summaries.get(name, {})
        lines.append(
            f"- {name}: pass={summary.get('pass')} status={summary.get('status', '')} "
            f"errors={len(summary.get('errors', []))}"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Production table index built: false",
            "- Preview units upgraded: false",
            "- Formal table citation generated: false",
            "- Canonical source resolution: false",
            "- LLM / RAGAS / OCR / VLM called: false",
            "- Milvus accessed: false",
            "- Official BM25 accessed: false",
            "- Embedding run: false",
            "- Ingestion pipeline modified: false",
            "",
            "## Decision",
            "",
            f"- Overall pass: {validation.get('pass')}",
        ]
    )
    if validation.get("errors"):
        lines.append(f"- Errors: {validation['errors']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7U shadow preview smoke.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--output-dir", type=_path_arg, default=RESULTS_DIR)
    args = parser.parse_args()
    summary = run_shadow_smoke(fixture_path=args.fixture_path, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
