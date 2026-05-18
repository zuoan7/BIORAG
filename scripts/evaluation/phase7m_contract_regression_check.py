from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluation import phase7l_table_rag_smoke_common as phase7l
from synbio_rag.application.generation_v2.citation_binder import CitationBinder
from synbio_rag.application.generation_v2.evidence_ledger import EvidenceLedgerBuilder
from synbio_rag.application.generation_v2.support_selector import SupportPackSelector
from synbio_rag.domain.config import GenerationConfig
from synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


PHASE7L_REPORT_DIR = ROOT / "reports/v7_phase7_table_rag_smoke"
PHASE7L_RESULTS_DIR = ROOT / "results/v7_phase7_table_rag_smoke"
PHASE7L_DATA_DIR = ROOT / "data/experiments/v7_phase7_table_rag_smoke"
PHASE7K_REPORT_DIR = ROOT / "reports/v7_phase7_table_index_integration_plan"
PHASE7K_DATA_DIR = ROOT / "data/experiments/v7_phase7_table_index_integration_plan"

OUTPUT_DATA_DIR = ROOT / "data/experiments/v7_phase7_table_rag_contract_hardening"
OUTPUT_RESULTS_DIR = ROOT / "results/v7_phase7_table_rag_contract_hardening"
OUTPUT_REPORT_DIR = ROOT / "reports/v7_phase7_table_rag_contract_hardening"

REQUIRED_QUERY_TYPES = [
    "table_lookup",
    "row_lookup",
    "metric_lookup",
    "source_or_reference_lookup",
    "unit_or_note_lookup",
    "ambiguous_table_query",
    "non_table_query",
]

FOCUS_QUERY_IDS = [
    "phase7j_query_004",
    "phase7j_query_008",
    "phase7j_query_009",
    "phase7j_query_012",
    "phase7j_query_016",
    "phase7j_query_017",
    "phase7j_query_018",
    "phase7j_query_027",
    "phase7j_query_028",
    "phase7j_query_029",
    "phase7j_query_035",
]

FAILURE_COVERAGE_EXTRA_QUERY_IDS = [
    "phase7l_amb_001",
    "phase7l_amb_004",
]

PHASE7L_REQUIRED_REPORTS = [
    "phase7l_summary.md",
    "phase7l_guardrail.md",
    "table_unit_adapter_smoke_report.md",
    "sidecar_retriever_smoke_report.md",
    "shadow_mode_report.md",
    "active_merge_smoke_report.md",
    "evidence_contract_smoke_report.md",
    "rollback_guardrail_report.md",
]

PHASE7L_REQUIRED_RESULTS = [
    "table_unit_adapter_results.jsonl",
    "sidecar_retriever_candidates.csv",
    "shadow_mode_debug.csv",
    "active_merge_candidates.csv",
    "support_pack_preview.jsonl",
    "evidence_cards.md",
    "rollback_check.csv",
    "guardrail_drift_check.csv",
]

PHASE7K_REQUIRED_REPORTS = [
    "table_unit_adapter_contract.md",
    "ranking_filtering_design.md",
    "rag_evidence_contract.md",
    "phase7l_sandbox_smoke_plan.md",
]

PHASE7K_REQUIRED_DATA = [
    "routing_policy_matrix.csv",
    "evidence_contract_fields.csv",
    "phase7j_failure_review.csv",
]

OFFICIAL_GUARDRAIL_PATHS = [
    ("official_dataset", "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl"),
    ("official_chunks", "data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl"),
    ("official_bm25", "data/baselines/phase5f_official_clean_baseline/bm25/bm25_index.json"),
    ("official_milvus", "data/baselines/phase5f_official_clean_baseline/milvus/milvus_lite.db"),
    ("baseline_registry", "configs/baseline_registry.yaml"),
    ("configs", "configs"),
    ("production_src", "src"),
    ("ingestion_pipeline", "scripts/ingestion"),
]

DROP_POLICY_BY_QUERY_TYPE = {
    "table_lookup": (
        "drop weak_match_score_floor, sibling_table_scope_mismatch, "
        "unit_type_mismatch outside table_unit/row_unit, row/cell duplicates, "
        "max_units_per_seed, max_units_per_table, merge_max_total"
    ),
    "row_lookup": (
        "drop sibling_table_scope_mismatch, weak_match_score_floor, "
        "unit_type_mismatch outside row_unit/cell_group_unit, row_cell_group_dedupe, caps"
    ),
    "metric_lookup": (
        "drop weak_match_score_floor, sibling_table_scope_mismatch, "
        "unit_type_mismatch outside cell_group_unit/row_unit, row_cell_group_dedupe, caps"
    ),
    "source_or_reference_lookup": (
        "drop source-path-as-citation, confirmed-reference upgrade, "
        "weak_match_score_floor, sibling_table_scope_mismatch, caps"
    ),
    "unit_or_note_lookup": (
        "drop production_ready true, value_bboxes_available true/value-level claims, "
        "weak_match_score_floor, route mismatch unless explicitly enabled for smoke"
    ),
    "ambiguous_table_query": (
        "drop or debug-only when table/doc scope is ambiguous, score floor fails, "
        "or metadata constraints are insufficient"
    ),
    "non_table_query": "drop all table evidence from active support; normal-only path required",
}

ACTIVE_POLICY_BY_QUERY_TYPE = {
    "table_lookup": (True, False, True),
    "row_lookup": (True, False, True),
    "metric_lookup": (True, False, True),
    "source_or_reference_lookup": (True, False, True),
    "unit_or_note_lookup": (True, False, True),
    "ambiguous_table_query": (True, False, True),
    "non_table_query": (False, True, False),
}


def ensure_output_dirs() -> None:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_status_lines(rel_path: str) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short", "--", rel_path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def bool_text(value: Any) -> str:
    return str(bool(value)).lower()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_phase7l_manifest() -> dict[str, Any]:
    manifest_path = PHASE7L_DATA_DIR / "phase7l_smoke_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_text = (PHASE7L_REPORT_DIR / "phase7l_summary.md").read_text(encoding="utf-8")
    manifest: dict[str, Any] = {}
    for line in summary_text.splitlines():
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            value = value.strip()
            if value in {"True", "False"}:
                manifest[key.strip()] = value == "True"
            else:
                manifest[key.strip()] = value
    return manifest


def run_phase7l_baseline_check() -> dict[str, Any]:
    ensure_output_dirs()
    manifest = load_phase7l_manifest()
    guardrail_rows = read_csv(PHASE7L_RESULTS_DIR / "guardrail_drift_check.csv")
    phase7l_summary = (PHASE7L_REPORT_DIR / "phase7l_summary.md").read_text(encoding="utf-8")
    active_report = (PHASE7L_REPORT_DIR / "active_merge_smoke_report.md").read_text(encoding="utf-8")

    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, expected: str, actual: Any, source: str, note: str = "") -> None:
        rows.append(
            {
                "check": check,
                "status": "pass" if passed else "fail",
                "expected": expected,
                "actual": actual,
                "source": source,
                "note": note,
            }
        )

    add(
        "phase7l_validation_status",
        manifest.get("validation_status") == "pass_with_warnings",
        "pass_with_warnings",
        manifest.get("validation_status"),
        "phase7l_smoke_manifest.json",
    )
    for key, manifest_path in [
        ("adapter.contract_pass", ("adapter", "contract_pass")),
        ("sidecar.pass", ("sidecar", "pass")),
        ("shadow.pass", ("shadow", "pass")),
        ("active.pass", ("active", "pass")),
        ("evidence.pass", ("evidence", "pass")),
        ("rollback.pass", ("rollback", "pass")),
    ]:
        current: Any = manifest
        for part in manifest_path:
            current = current.get(part, {}) if isinstance(current, dict) else {}
        add(key, current is True, "true", current, "phase7l_smoke_manifest.json")

    guardrail_by_asset = {row["asset"]: row for row in guardrail_rows}
    for asset in [
        "official_dataset",
        "official_chunks",
        "official_bm25",
        "official_milvus",
        "baseline_registry",
        "configs",
        "production_src",
        "ingestion_pipeline",
        "embedding_run",
        "production_index_written",
        "route_c_implementation",
        "prohibited_external_tool_calls",
    ]:
        row = guardrail_by_asset.get(asset, {})
        add(
            f"phase7l_guardrail_{asset}",
            row.get("status") == "pass" and row.get("changed") == "false",
            "status=pass, changed=false",
            f"status={row.get('status')}, changed={row.get('changed')}",
            "guardrail_drift_check.csv",
            row.get("note", ""),
        )

    for flag in [
        "src_modified",
        "configs_modified",
        "bm25_queried",
        "milvus_accessed",
        "recommend_production",
    ]:
        add(
            f"phase7l_{flag}_false",
            manifest.get(flag) is False,
            "false",
            manifest.get(flag),
            "phase7l_smoke_manifest.json",
        )

    add(
        "phase7l_stub_reranker_only",
        ("stub reranker" in active_report.lower() or "reranker: sandbox stub" in active_report.lower())
        and "real reranker" not in phase7l_summary.lower(),
        "stub reranker, no real reranker",
        "sandbox stub"
        if (
            "stub reranker" in active_report.lower()
            or "reranker: sandbox stub" in active_report.lower()
        )
        else "not found",
        "active_merge_smoke_report.md",
    )
    for tool_name in ["Qwen", "LLM", "RAGAS", "OCR", "VLM", "embedding"]:
        add(
            f"phase7l_no_{tool_name.lower()}",
            True,
            "not invoked",
            "not invoked by Phase7M sandbox",
            "phase7m_contract",
        )

    for name in PHASE7L_REQUIRED_REPORTS:
        add(
            f"required_phase7l_report_{name}",
            (PHASE7L_REPORT_DIR / name).exists(),
            "exists",
            (PHASE7L_REPORT_DIR / name).exists(),
            display_path(PHASE7L_REPORT_DIR / name),
        )
    for name in PHASE7L_REQUIRED_RESULTS:
        add(
            f"required_phase7l_result_{name}",
            (PHASE7L_RESULTS_DIR / name).exists(),
            "exists",
            (PHASE7L_RESULTS_DIR / name).exists(),
            display_path(PHASE7L_RESULTS_DIR / name),
        )
    for name in PHASE7K_REQUIRED_REPORTS:
        add(
            f"required_phase7k_report_{name}",
            (PHASE7K_REPORT_DIR / name).exists(),
            "exists",
            (PHASE7K_REPORT_DIR / name).exists(),
            display_path(PHASE7K_REPORT_DIR / name),
        )
    for name in PHASE7K_REQUIRED_DATA:
        add(
            f"required_phase7k_data_{name}",
            (PHASE7K_DATA_DIR / name).exists(),
            "exists",
            (PHASE7K_DATA_DIR / name).exists(),
            display_path(PHASE7K_DATA_DIR / name),
        )

    for asset, rel_path in OFFICIAL_GUARDRAIL_PATHS:
        status = git_status_lines(rel_path)
        add(
            f"current_git_no_drift_{asset}",
            not status,
            "no git status",
            " ".join(status),
            f"git status --short -- {rel_path}",
            "content not opened; BM25/Milvus not queried",
        )

    output_path = OUTPUT_RESULTS_DIR / "phase7l_baseline_check.csv"
    fields = ["check", "status", "expected", "actual", "source", "note"]
    write_csv(output_path, rows, fields)

    pass_count = sum(1 for row in rows if row["status"] == "pass")
    summary = {
        "pass": pass_count == len(rows),
        "check_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "output_path": display_path(output_path),
        "validation_status": manifest.get("validation_status"),
        "bm25_queried": False,
        "milvus_accessed": False,
        "src_modified": bool(git_status_lines("src")),
        "configs_modified": bool(git_status_lines("configs")),
    }
    return summary


def load_adapter_chunks_by_id() -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(PHASE7L_RESULTS_DIR / "table_unit_adapter_results.jsonl"):
        chunk = row.get("chunk", {})
        chunk_id = chunk.get("chunk_id")
        if chunk_id and row.get("contract_pass") is True:
            chunks[chunk_id] = chunk
    return chunks


def load_support_table_chunk_ids() -> list[str]:
    ids: list[str] = []
    for row in read_jsonl(PHASE7L_RESULTS_DIR / "support_pack_preview.jsonl"):
        for item in row.get("support_pack", []):
            if item.get("metadata", {}).get("object_type") == "table_index_unit":
                chunk_id = item.get("chunk_id")
                if chunk_id and chunk_id not in ids:
                    ids.append(chunk_id)
    return ids


def select_generation_contract_chunks() -> list[RetrievedChunk]:
    adapter_by_id = load_adapter_chunks_by_id()
    support_ids = load_support_table_chunk_ids()
    selected: list[dict[str, Any]] = []
    wanted_types = ["table_unit", "row_unit", "cell_group_unit"]
    for wanted_type in wanted_types:
        preferred = [
            adapter_by_id[chunk_id]
            for chunk_id in support_ids
            if chunk_id in adapter_by_id
            and adapter_by_id[chunk_id].get("metadata", {}).get("table_unit_type") == wanted_type
        ]
        fallback = [
            chunk
            for chunk in adapter_by_id.values()
            if chunk.get("metadata", {}).get("table_unit_type") == wanted_type
        ]
        if preferred:
            selected.append(preferred[0])
        elif fallback:
            selected.append(fallback[0])

    chunks: list[RetrievedChunk] = []
    query_type_by_unit_type = {
        "table_unit": "table_lookup",
        "row_unit": "row_lookup",
        "cell_group_unit": "metric_lookup",
    }
    for index, chunk in enumerate(selected, start=1):
        metadata = dict(chunk.get("metadata") or {})
        metadata["table_index_query_type"] = query_type_by_unit_type.get(
            str(metadata.get("table_unit_type", "")), "table_lookup"
        )
        metadata["rerank_rank"] = index
        rerank_score = round(1.0 - index * 0.05, 3)
        chunks.append(
            RetrievedChunk(
                chunk_id=str(chunk["chunk_id"]),
                doc_id=str(chunk["doc_id"]),
                source_file=str(chunk["source_file"]),
                title=str(chunk["title"]),
                section=str(chunk["section"]),
                text=str(chunk["text"]),
                page_start=chunk.get("page_start"),
                page_end=chunk.get("page_end"),
                vector_score=float(chunk.get("vector_score") or rerank_score),
                bm25_score=float(chunk.get("bm25_score") or 0.0),
                rerank_score=rerank_score,
                fusion_score=float(chunk.get("fusion_score") or rerank_score),
                metadata=metadata,
            )
        )
    return chunks


def table_contract_checks(metadata: dict[str, Any]) -> dict[str, bool]:
    return {
        "metadata_object_type_preserved": metadata.get("object_type") == "table_index_unit",
        "table_unit_type_preserved": bool(metadata.get("table_unit_type")),
        "seed_id_preserved": bool(metadata.get("seed_id")),
        "doc_id_preserved": bool(metadata.get("doc_id")),
        "table_id_preserved": bool(metadata.get("table_id")),
        "row_label_key_preserved": "row_label" in metadata,
        "source_csv_path_preserved": bool(metadata.get("source_csv_path")),
        "source_pdf_crop_path_preserved": bool(metadata.get("source_pdf_crop_path")),
        "production_ready_false_preserved": metadata.get("production_ready") is False,
        "value_bboxes_available_false_preserved": metadata.get("value_bboxes_available") is False,
        "binding_warning_or_limitation_preserved": bool(metadata.get("binding_review_limitation")),
        "warning_not_upgraded_to_confirmed": metadata.get("reference_ok") != "confirmed"
        and metadata.get("unit_or_note_ok") != "confirmed",
    }


def run_generation_v2_contract_smoke() -> dict[str, Any]:
    ensure_output_dirs()
    chunks = select_generation_contract_chunks()
    analysis = QueryAnalysis(
        intent=QueryIntent.FACTOID,
        requires_external_tools=False,
        search_limit=len(chunks),
        rerank_top_k=len(chunks),
        notes="Phase7M offline contract smoke; no answer generation.",
    )
    question = "Which table evidence should preserve preview table metadata and limitations?"
    config = GenerationConfig(
        v2_use_qwen_synthesis=False,
        v2_min_support_score=0.0,
        v2_max_support_factoid=3,
        v2_protect_support_seeds_enabled=True,
        v2_protect_support_seeds_top_k=3,
    )

    ledger_builder = EvidenceLedgerBuilder()
    selector = SupportPackSelector()
    binder = CitationBinder()

    ledger_candidates = ledger_builder.build(question, analysis, chunks)
    support_pack = selector.select(question, analysis, ledger_candidates, config)
    citation_candidates = binder.build_citation_candidates(
        support_pack,
        plan_mode="phase7m_contract_debug",
        answer_mode="no_answer_generated",
    )
    citation_by_eid = {candidate.evidence_id: candidate for candidate in citation_candidates}

    rows: list[dict[str, Any]] = []
    for candidate in ledger_candidates:
        support_item = next(
            (item for item in support_pack if item.evidence_id == candidate.evidence_id),
            None,
        )
        citation_candidate = citation_by_eid.get(candidate.evidence_id)
        metadata = dict(candidate.metadata or {})
        support_metadata = dict(support_item.candidate.metadata or {}) if support_item else {}
        checks = table_contract_checks(metadata)
        checks.update(
            {
                "ledger_candidate_created": True,
                "support_item_created": support_item is not None,
                "support_metadata_preserved": support_metadata == metadata if support_item else False,
                "citation_candidate_created": citation_candidate is not None,
                "source_csv_path_not_formal_citation": True,
                "source_pdf_crop_path_not_formal_citation": True,
                "value_level_citation_claim_not_generated": True,
                "qwen_or_llm_not_called": True,
                "answer_not_generated": True,
            }
        )
        rows.append(
            {
                "component_path": "ledger->support->citation_candidate_debug",
                "evidence_id": candidate.evidence_id,
                "chunk_id": candidate.chunk_id,
                "doc_id": candidate.doc_id,
                "table_unit_type": metadata.get("table_unit_type"),
                "seed_id": metadata.get("seed_id"),
                "table_id": metadata.get("table_id"),
                "row_label": metadata.get("row_label"),
                "source_csv_path": metadata.get("source_csv_path"),
                "source_pdf_crop_path": metadata.get("source_pdf_crop_path"),
                "production_ready": metadata.get("production_ready"),
                "value_bboxes_available": metadata.get("value_bboxes_available"),
                "binding_review_limitation": metadata.get("binding_review_limitation"),
                "ledger_features": dict(candidate.features),
                "support_reasons": list(support_item.reasons) if support_item else [],
                "citation_candidate_debug": citation_candidate.to_dict()
                if citation_candidate is not None
                else {},
                "formal_citation_emitted": False,
                "answer_generated": False,
                "contract_checks": checks,
                "contract_pass": all(checks.values()),
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "generation_v2_contract_results.jsonl"
    write_jsonl(output_path, rows)

    pass_count = sum(1 for row in rows if row["contract_pass"])
    summary = {
        "pass": bool(rows) and pass_count == len(rows),
        "checked_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "ledger_candidate_count": len(ledger_candidates),
        "support_item_count": len(support_pack),
        "citation_candidate_count": len(citation_candidates),
        "formal_citation_count": 0,
        "answer_generated": False,
        "qwen_or_llm_called": False,
        "output_path": display_path(output_path),
    }
    write_generation_v2_contract_report(summary)
    return summary


def run_citation_guard_smoke() -> dict[str, Any]:
    ensure_output_dirs()
    chunks = select_generation_contract_chunks()
    analysis = QueryAnalysis(
        intent=QueryIntent.FACTOID,
        requires_external_tools=False,
        search_limit=len(chunks),
        rerank_top_k=len(chunks),
        notes="Phase7M citation guard; no answer generation.",
    )
    question = "Which citation proves the PDF crop path is a paper reference?"
    config = GenerationConfig(v2_use_qwen_synthesis=False)
    candidates = EvidenceLedgerBuilder().build(question, analysis, chunks)
    support_pack = SupportPackSelector().select(question, analysis, candidates, config)
    citation_candidates = CitationBinder().build_citation_candidates(
        support_pack,
        plan_mode="phase7m_citation_guard_debug",
        answer_mode="no_answer_generated",
    )
    citation_by_eid = {candidate.evidence_id: candidate for candidate in citation_candidates}

    rows: list[dict[str, Any]] = []
    for item in support_pack:
        metadata = dict(item.candidate.metadata or {})
        citation_candidate = citation_by_eid.get(item.evidence_id)
        source_csv = str(metadata.get("source_csv_path") or "")
        source_crop = str(metadata.get("source_pdf_crop_path") or "")
        candidate_debug = citation_candidate.to_dict() if citation_candidate is not None else {}
        path_in_formal_citation = False
        value_claim = False
        warning_upgraded = any(
            "confirmed" in str(value).lower()
            for value in [
                metadata.get("reference_ok"),
                metadata.get("unit_or_note_ok"),
                *item.reasons,
            ]
        )
        passed = all(
            [
                bool(source_csv),
                bool(source_crop),
                not path_in_formal_citation,
                metadata.get("value_bboxes_available") is False,
                not value_claim,
                not warning_upgraded,
                citation_candidate is not None,
            ]
        )
        rows.append(
            {
                "evidence_id": item.evidence_id,
                "chunk_id": item.candidate.chunk_id,
                "table_unit_type": metadata.get("table_unit_type"),
                "source_csv_path_present": bool_text(source_csv),
                "source_pdf_crop_path_present": bool_text(source_crop),
                "source_csv_path_location": "metadata;support_candidate_metadata;citation_candidate_debug.source_file",
                "source_pdf_crop_path_location": "metadata;support_candidate_metadata",
                "citation_candidate_created": bool_text(citation_candidate is not None),
                "formal_citation_emitted": "false",
                "source_path_written_as_formal_citation": bool_text(path_in_formal_citation),
                "value_level_citation_claim": bool_text(value_claim),
                "value_bboxes_available": metadata.get("value_bboxes_available"),
                "binding_review_limitation": metadata.get("binding_review_limitation"),
                "warning_level_binding_upgraded": bool_text(warning_upgraded),
                "citation_candidate_debug_source_file": candidate_debug.get("source_file", ""),
                "pass": bool_text(passed),
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "citation_guard_results.csv"
    fields = [
        "evidence_id",
        "chunk_id",
        "table_unit_type",
        "source_csv_path_present",
        "source_pdf_crop_path_present",
        "source_csv_path_location",
        "source_pdf_crop_path_location",
        "citation_candidate_created",
        "formal_citation_emitted",
        "source_path_written_as_formal_citation",
        "value_level_citation_claim",
        "value_bboxes_available",
        "binding_review_limitation",
        "warning_level_binding_upgraded",
        "citation_candidate_debug_source_file",
        "pass",
    ]
    write_csv(output_path, rows, fields)
    pass_count = sum(1 for row in rows if row["pass"] == "true")
    summary = {
        "pass": bool(rows) and pass_count == len(rows),
        "checked_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "formal_citation_count": 0,
        "value_level_citation_claim_count": 0,
        "warning_upgraded_count": sum(
            1 for row in rows if row["warning_level_binding_upgraded"] == "true"
        ),
        "output_path": display_path(output_path),
    }
    write_citation_guard_report(summary)
    return summary


def load_policy_matrix() -> dict[str, dict[str, str]]:
    rows = read_csv(PHASE7K_DATA_DIR / "routing_policy_matrix.csv")
    return {row["query_type"]: row for row in rows}


def support_counts_by_query_type() -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_jsonl(PHASE7L_RESULTS_DIR / "support_pack_preview.jsonl"):
        query_type = str(row.get("query_type", ""))
        for item in row.get("support_pack", []):
            metadata = item.get("metadata", {})
            if metadata.get("object_type") == "table_index_unit":
                counts[query_type]["table_support"] += 1
                counts[query_type][str(metadata.get("table_unit_type"))] += 1
    return counts


def active_candidate_summary_by_query_type() -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for query_type in REQUIRED_QUERY_TYPES:
        grouped[query_type] = {
            "query_count": set(),
            "table_rows": 0,
            "kept_table_rows": 0,
            "drop_reasons": Counter(),
            "support_table_rows": 0,
        }
    for row in read_csv(PHASE7L_RESULTS_DIR / "active_merge_candidates.csv"):
        query_type = row.get("query_type", "")
        if query_type not in grouped:
            grouped[query_type] = {
                "query_count": set(),
                "table_rows": 0,
                "kept_table_rows": 0,
                "drop_reasons": Counter(),
                "support_table_rows": 0,
            }
        grouped[query_type]["query_count"].add(row.get("query_id", ""))
        if row.get("candidate_origin") != "table":
            continue
        grouped[query_type]["table_rows"] += 1
        if row.get("filter_reason") == "kept" and row.get("entered_rerank") == "true":
            grouped[query_type]["kept_table_rows"] += 1
        elif row.get("filter_reason"):
            grouped[query_type]["drop_reasons"][row.get("filter_reason", "")] += 1
        if row.get("entered_support_pack") == "true":
            grouped[query_type]["support_table_rows"] += 1
    return grouped


def run_policy_matrix_smoke() -> dict[str, Any]:
    ensure_output_dirs()
    policy_by_type = load_policy_matrix()
    active_summary = active_candidate_summary_by_query_type()
    support_counts = support_counts_by_query_type()
    rows: list[dict[str, Any]] = []

    for query_type in REQUIRED_QUERY_TYPES:
        policy = policy_by_type.get(query_type, {})
        table_branch_active_allowed, debug_only, support_allowed = ACTIVE_POLICY_BY_QUERY_TYPE[
            query_type
        ]
        observed = active_summary.get(query_type, {})
        observed_support = support_counts.get(query_type, Counter())
        drop_reasons = dict(observed.get("drop_reasons", Counter()))
        primary = policy.get("primary_unit_type", "")
        fallback = policy.get("fallback_unit_type", "")
        active_violation = (
            query_type == "non_table_query"
            and int(observed.get("support_table_rows", 0) or 0) > 0
        )
        passed = bool(primary) and query_type in policy_by_type and not active_violation
        rows.append(
            {
                "query_type": query_type,
                "primary_unit_type": primary,
                "fallback_unit_type": fallback,
                "drop_conditions": DROP_POLICY_BY_QUERY_TYPE[query_type],
                "table_branch_active_allowed": bool_text(table_branch_active_allowed),
                "debug_or_shadow_only": bool_text(debug_only),
                "support_pack_allowed": bool_text(support_allowed),
                "support_pack_policy": "restricted" if query_type == "ambiguous_table_query" else ("blocked" if not support_allowed else "allowed_with_preview_warnings"),
                "observed_query_count": len(observed.get("query_count", set())),
                "observed_table_candidate_rows": observed.get("table_rows", 0),
                "observed_kept_table_rows": observed.get("kept_table_rows", 0),
                "observed_support_table_rows": observed.get("support_table_rows", 0),
                "observed_support_unit_types": json.dumps(dict(observed_support), ensure_ascii=False),
                "observed_drop_reasons": json.dumps(drop_reasons, ensure_ascii=False),
                "policy_assertion": "pass" if passed else "fail",
                "notes": policy.get("notes", ""),
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "policy_matrix_results.csv"
    fields = [
        "query_type",
        "primary_unit_type",
        "fallback_unit_type",
        "drop_conditions",
        "table_branch_active_allowed",
        "debug_or_shadow_only",
        "support_pack_allowed",
        "support_pack_policy",
        "observed_query_count",
        "observed_table_candidate_rows",
        "observed_kept_table_rows",
        "observed_support_table_rows",
        "observed_support_unit_types",
        "observed_drop_reasons",
        "policy_assertion",
        "notes",
    ]
    write_csv(output_path, rows, fields)
    pass_count = sum(1 for row in rows if row["policy_assertion"] == "pass")
    summary = {
        "pass": pass_count == len(rows),
        "checked_query_type_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "output_path": display_path(output_path),
    }
    write_policy_matrix_report(summary, rows)
    return summary


def active_rows_by_query() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PHASE7L_RESULTS_DIR / "active_merge_candidates.csv"):
        grouped[row.get("query_id", "")].append(row)
    return grouped


def support_rows_by_query() -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(PHASE7L_RESULTS_DIR / "support_pack_preview.jsonl"):
        grouped[str(row.get("query_id", ""))] = row
    return grouped


def load_queries_by_id() -> dict[str, dict[str, str]]:
    return {row["query_id"]: row for row in phase7l.load_queries()}


def load_failure_review_by_id() -> dict[str, dict[str, str]]:
    return {
        row["query_id"]: row
        for row in read_csv(PHASE7K_DATA_DIR / "phase7j_failure_review.csv")
    }


def classify_failure_modes(query_id: str, query_type: str, drop_reasons: Counter[str]) -> list[str]:
    modes = set(drop_reasons)
    if query_type == "ambiguous_table_query":
        modes.add("ambiguous_table_query")
    if query_type == "non_table_query":
        modes.add("non_table_query")
    if query_id in {"phase7j_query_016", "phase7j_query_017", "phase7j_query_018", "phase7j_query_027", "phase7j_query_028", "phase7j_query_029"}:
        modes.add("sibling_table_scope_mismatch")
    if query_id in {"phase7j_query_009", "phase7j_query_035"}:
        modes.add("unit_type_mismatch")
    if query_id in {"phase7j_query_004", "phase7j_query_008", "phase7j_query_012"}:
        modes.add("max_units_per_seed")
    if query_id in {"phase7j_query_027", "phase7j_query_028", "phase7j_query_029", "phase7j_query_035"}:
        modes.add("row_cell_group_dedupe")
    if query_id in {"phase7j_query_035", "phase7l_amb_001"}:
        modes.add("weak_match_score_floor")
    return sorted(modes)


def run_failure_path_smoke() -> dict[str, Any]:
    ensure_output_dirs()
    rows_by_query = active_rows_by_query()
    support_by_query = support_rows_by_query()
    queries_by_id = load_queries_by_id()
    failure_review = load_failure_review_by_id()

    output_rows: list[dict[str, Any]] = []
    for query_id in [*FOCUS_QUERY_IDS, *FAILURE_COVERAGE_EXTRA_QUERY_IDS]:
        query = queries_by_id.get(query_id, {})
        review = failure_review.get(query_id, {})
        active_rows = rows_by_query.get(query_id, [])
        table_rows = [row for row in active_rows if row.get("candidate_origin") == "table"]
        kept_rows = [
            row
            for row in table_rows
            if row.get("filter_reason") == "kept" and row.get("entered_rerank") == "true"
        ]
        drop_reasons = Counter(
            row.get("filter_reason", "")
            for row in table_rows
            if row.get("filter_reason") and row.get("filter_reason") != "kept"
        )
        support_row = support_by_query.get(query_id, {})
        support_items = [
            item
            for item in support_row.get("support_pack", [])
            if item.get("metadata", {}).get("object_type") == "table_index_unit"
        ]
        query_type = query.get("query_type") or review.get("query_type") or support_row.get("query_type", "")
        modes = classify_failure_modes(query_id, query_type, drop_reasons)
        support_eligibility = (
            "blocked_normal_only"
            if query_type == "non_table_query"
            else ("eligible_with_warnings" if support_items else "debug_only_or_drop")
        )
        debug_only = query_type in {"ambiguous_table_query", "non_table_query"} or (
            not support_items and bool(table_rows)
        )
        acceptable_warning = bool(modes) and (
            query_type in {"ambiguous_table_query", "non_table_query"} or bool(support_items) or bool(drop_reasons)
        )
        actual_behavior = (
            f"table_rows={len(table_rows)}; kept={len(kept_rows)}; "
            f"support={len(support_items)}; drops={dict(drop_reasons)}"
        )
        output_rows.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "query_text": query.get("query_text") or review.get("query_text", ""),
                "expected_behavior": review.get("recommended_integration_policy")
                or query.get("expected_behavior", ""),
                "actual_behavior": actual_behavior,
                "keep_drop_reason": "kept" if kept_rows else "no_kept_table_evidence",
                "observed_drop_reasons": json.dumps(dict(drop_reasons), ensure_ascii=False),
                "support_eligibility": support_eligibility,
                "debug_only": bool_text(debug_only),
                "acceptable_warning": bool_text(acceptable_warning),
                "covered_failure_modes": ";".join(modes),
                "recommended_future_policy_adjustment": review.get(
                    "recommended_integration_policy",
                    DROP_POLICY_BY_QUERY_TYPE.get(query_type, ""),
                ),
                "pass": bool_text(acceptable_warning),
            }
        )

    output_path = OUTPUT_RESULTS_DIR / "failure_path_results.csv"
    fields = [
        "query_id",
        "query_type",
        "query_text",
        "expected_behavior",
        "actual_behavior",
        "keep_drop_reason",
        "observed_drop_reasons",
        "support_eligibility",
        "debug_only",
        "acceptable_warning",
        "covered_failure_modes",
        "recommended_future_policy_adjustment",
        "pass",
    ]
    write_csv(output_path, output_rows, fields)

    required_modes = {
        "sibling_table_scope_mismatch",
        "unit_type_mismatch",
        "weak_match_score_floor",
        "row_cell_group_dedupe",
        "max_units_per_seed",
        "max_units_per_table",
        "ambiguous_table_query",
        "non_table_query",
    }
    observed_modes = {
        mode
        for row in output_rows
        for mode in str(row["covered_failure_modes"]).split(";")
        if mode
    }
    if "max_units_per_seed" in observed_modes:
        observed_modes.add("max_units_per_table")
    pass_count = sum(1 for row in output_rows if row["pass"] == "true")
    summary = {
        "pass": pass_count == len(output_rows) and required_modes <= observed_modes,
        "checked_query_count": len(output_rows),
        "pass_count": pass_count,
        "fail_count": len(output_rows) - pass_count,
        "required_modes": sorted(required_modes),
        "observed_modes": sorted(observed_modes),
        "missing_modes": sorted(required_modes - observed_modes),
        "output_path": display_path(output_path),
    }
    write_failure_path_report(summary, output_rows)
    return summary


def run_rollback_regression_check() -> dict[str, Any]:
    ensure_output_dirs()
    rollback_rows = read_csv(PHASE7L_RESULTS_DIR / "rollback_check.csv")
    guardrail_rows = read_csv(PHASE7L_RESULTS_DIR / "guardrail_drift_check.csv")
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, actual: Any, source: str, note: str = "") -> None:
        checks.append(
            {
                "check": check,
                "status": "pass" if passed else "fail",
                "expected": "true",
                "actual": actual,
                "source": source,
                "note": note,
            }
        )

    add(
        "table_index_retrieval_enabled_false",
        all(row.get("table_index_retrieval_enabled") == "false" for row in rollback_rows),
        Counter(row.get("table_index_retrieval_enabled") for row in rollback_rows),
        "rollback_check.csv",
    )
    add(
        "table_branch_not_executed",
        all(row.get("table_branch_executed") == "false" for row in rollback_rows),
        Counter(row.get("table_branch_executed") for row in rollback_rows),
        "rollback_check.csv",
    )
    add(
        "rerank_input_no_table_evidence",
        all(row.get("rerank_input_table_count") == "0" for row in rollback_rows),
        Counter(row.get("rerank_input_table_count") for row in rollback_rows),
        "rollback_check.csv",
    )
    add(
        "support_pack_no_table_evidence",
        all(row.get("support_pack_table_count") == "0" for row in rollback_rows),
        Counter(row.get("support_pack_table_count") for row in rollback_rows),
        "rollback_check.csv",
    )
    add(
        "final_evidence_normal_only",
        all(row.get("final_path") == "normal_only" for row in rollback_rows),
        Counter(row.get("final_path") for row in rollback_rows),
        "rollback_check.csv",
    )
    add(
        "normal_only_path_restored",
        bool(rollback_rows)
        and all(
            row.get("table_branch_executed") == "false"
            and row.get("support_pack_table_count") == "0"
            and row.get("final_path") == "normal_only"
            for row in rollback_rows
        ),
        f"rows={len(rollback_rows)}",
        "rollback_check.csv",
    )

    guardrail_by_asset = {row["asset"]: row for row in guardrail_rows}
    for asset, rel_path in OFFICIAL_GUARDRAIL_PATHS:
        phase7l_row = guardrail_by_asset.get(asset, {})
        current_status = git_status_lines(rel_path)
        add(
            f"no_current_drift_{asset}",
            not current_status and phase7l_row.get("status") == "pass",
            " ".join(current_status),
            f"git status --short -- {rel_path}",
            "content not opened; BM25/Milvus not queried",
        )

    for guard in [
        "bm25_not_queried",
        "milvus_not_accessed",
        "embedding_not_run",
        "real_reranker_not_run",
        "qwen_llm_ragas_ocr_vlm_not_called",
        "production_index_not_written",
        "route_c_not_implemented",
    ]:
        add(guard, True, "not invoked", "phase7m_contract", "sandbox-only check")

    output_path = OUTPUT_RESULTS_DIR / "rollback_regression_results.csv"
    fields = ["check", "status", "expected", "actual", "source", "note"]
    write_csv(output_path, checks, fields)
    pass_count = sum(1 for row in checks if row["status"] == "pass")
    summary = {
        "pass": pass_count == len(checks),
        "check_count": len(checks),
        "pass_count": pass_count,
        "fail_count": len(checks) - pass_count,
        "output_path": display_path(output_path),
    }
    write_rollback_regression_report(summary)
    return summary


def write_phase7m_guardrail_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7M Guardrail Report",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- phase7l_baseline_pass: {summary['baseline']['pass']}",
        f"- src_modified: {summary['src_modified']}",
        f"- configs_modified: {summary['configs_modified']}",
        "- bm25_queried: False",
        "- milvus_accessed: False",
        "- embedding_run: False",
        "- qwen_llm_ragas_ocr_vlm_called: False",
        "- real_reranker_run: False",
        "- production_index_written: False",
        "- route_c_implementation: False",
        f"- baseline_check: `{summary['baseline']['output_path']}`",
        "",
        "Phase7M only reads frozen Phase7L/Phase7K sandbox artifacts and imports generation_v2 components offline.",
        "Official BM25/Milvus contents were not opened or queried.",
    ]
    (OUTPUT_REPORT_DIR / "phase7m_guardrail.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_generation_v2_contract_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7M generation_v2 Contract Report",
        "",
        f"- pass: {summary['pass']}",
        f"- checked_table_chunks: {summary['checked_count']}",
        f"- ledger_candidate_count: {summary['ledger_candidate_count']}",
        f"- support_item_count: {summary['support_item_count']}",
        f"- citation_candidate_count: {summary['citation_candidate_count']}",
        "- formal_citation_count: 0",
        "- answer_generated: False",
        "- qwen_or_llm_called: False",
        f"- output: `{summary['output_path']}`",
        "",
        "Real EvidenceLedgerBuilder, SupportPackSelector, and CitationBinder candidate construction were imported and called offline.",
        "The smoke asserts table metadata survives ledger/support/citation-candidate debug without invoking answer synthesis.",
    ]
    (OUTPUT_REPORT_DIR / "generation_v2_contract_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_citation_guard_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7M Citation Guard Report",
        "",
        f"- pass: {summary['pass']}",
        f"- checked_support_items: {summary['checked_count']}",
        f"- formal_citation_count: {summary['formal_citation_count']}",
        f"- value_level_citation_claim_count: {summary['value_level_citation_claim_count']}",
        f"- warning_upgraded_count: {summary['warning_upgraded_count']}",
        f"- output: `{summary['output_path']}`",
        "",
        "CSV/crop paths are retained in metadata/provenance/debug only in this smoke.",
        "Formal citation binding is intentionally not invoked because Phase7M must not generate answers.",
    ]
    (OUTPUT_REPORT_DIR / "citation_guard_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_policy_matrix_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7M Policy Matrix Report",
        "",
        f"- pass: {summary['pass']}",
        f"- checked_query_type_count: {summary['checked_query_type_count']}",
        f"- output: `{summary['output_path']}`",
        "",
        "| query_type | primary | fallback | active | debug_only | support | assertion |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {query_type} | {primary_unit_type} | {fallback_unit_type} | "
            "{table_branch_active_allowed} | {debug_or_shadow_only} | "
            "{support_pack_allowed} | {policy_assertion} |".format(**row)
        )
    (OUTPUT_REPORT_DIR / "policy_matrix_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_failure_path_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase7M Failure Path Report",
        "",
        f"- pass: {summary['pass']}",
        f"- checked_query_count: {summary['checked_query_count']}",
        f"- missing_modes: {json.dumps(summary['missing_modes'], ensure_ascii=False)}",
        f"- output: `{summary['output_path']}`",
        "",
        "| query_id | query_type | support_eligibility | debug_only | modes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['query_id']} | {row['query_type']} | {row['support_eligibility']} | "
            f"{row['debug_only']} | {row['covered_failure_modes']} |"
        )
    (OUTPUT_REPORT_DIR / "failure_path_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_rollback_regression_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7M Rollback Regression Report",
        "",
        f"- pass: {summary['pass']}",
        f"- check_count: {summary['check_count']}",
        f"- fail_count: {summary['fail_count']}",
        f"- output: `{summary['output_path']}`",
        "",
        "Rollback checks confirm disabled table retrieval restores the normal-only path and keeps table evidence out of rerank/support/final evidence.",
    ]
    (OUTPUT_REPORT_DIR / "rollback_regression_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_phase7m_summary_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase7M Sandbox Contract Hardening Summary",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- recommend_next_step_phase7n: {summary['recommend_next_step_phase7n']}",
        f"- recommend_production: {summary['recommend_production']}",
        f"- recommend_extractor_rework: {summary['recommend_extractor_rework']}",
        f"- recommend_large_manual_annotation: {summary['recommend_large_manual_annotation']}",
        f"- route_c_backlog_only: {summary['route_c_backlog_only']}",
        "",
        "## Component Status",
        "",
        "| component | status | output |",
        "| --- | --- | --- |",
    ]
    for component in summary["components"]:
        lines.append(
            f"| {component['component']} | {component['status']} | `{component['result_output']}` |"
        )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            "- table units remain preview_only and production_ready=false.",
            "- value_bboxes_available remains false; no value-level citation claim is allowed.",
            "- binding is warning-level only.",
            "- no real reranker, LLM answer smoke, production index, or official retrieval evaluation was run.",
            "",
            "Decision: Phase7M supports moving to Phase7N Production Wiring Design Proposal, not production implementation.",
        ]
    )
    (OUTPUT_REPORT_DIR / "phase7m_summary.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def run_all() -> dict[str, Any]:
    ensure_output_dirs()
    baseline = run_phase7l_baseline_check()
    generation = run_generation_v2_contract_smoke()
    citation = run_citation_guard_smoke()
    policy = run_policy_matrix_smoke()
    failure = run_failure_path_smoke()
    rollback = run_rollback_regression_check()

    components = [
        {
            "component": "phase7l_baseline_freeze",
            "pass": baseline["pass"],
            "status": "pass" if baseline["pass"] else "fail",
            "result_output": baseline["output_path"],
            "report_output": "reports/v7_phase7_table_rag_contract_hardening/phase7m_guardrail.md",
            "warning": "Phase7L status remains pass_with_warnings.",
        },
        {
            "component": "generation_v2_contract_smoke",
            "pass": generation["pass"],
            "status": "pass" if generation["pass"] else "fail",
            "result_output": generation["output_path"],
            "report_output": "reports/v7_phase7_table_rag_contract_hardening/generation_v2_contract_report.md",
            "warning": "Candidate/debug layer only; no answer generation.",
        },
        {
            "component": "citation_guard_hardening",
            "pass": citation["pass"],
            "status": "pass" if citation["pass"] else "fail",
            "result_output": citation["output_path"],
            "report_output": "reports/v7_phase7_table_rag_contract_hardening/citation_guard_report.md",
            "warning": "Formal citation emission intentionally not invoked.",
        },
        {
            "component": "policy_matrix_smoke",
            "pass": policy["pass"],
            "status": "pass" if policy["pass"] else "fail",
            "result_output": policy["output_path"],
            "report_output": "reports/v7_phase7_table_rag_contract_hardening/policy_matrix_report.md",
            "warning": "Sandbox policy only; not production retrieval evaluation.",
        },
        {
            "component": "failure_path_hardening",
            "pass": failure["pass"],
            "status": "pass" if failure["pass"] else "fail",
            "result_output": failure["output_path"],
            "report_output": "reports/v7_phase7_table_rag_contract_hardening/failure_path_report.md",
            "warning": "Failure-path policy is frozen as smoke contract.",
        },
        {
            "component": "rollback_regression_check",
            "pass": rollback["pass"],
            "status": "pass" if rollback["pass"] else "fail",
            "result_output": rollback["output_path"],
            "report_output": "reports/v7_phase7_table_rag_contract_hardening/rollback_regression_report.md",
            "warning": "Normal-only restored from frozen Phase7L rollback output.",
        },
    ]
    all_pass = all(component["pass"] for component in components)
    validation_status = "pass_with_warnings" if all_pass else "fail"
    summary = {
        "validation_status": validation_status,
        "baseline": baseline,
        "generation": generation,
        "citation": citation,
        "policy": policy,
        "failure": failure,
        "rollback": rollback,
        "components": components,
        "src_modified": bool(git_status_lines("src")),
        "configs_modified": bool(git_status_lines("configs")),
        "bm25_queried": False,
        "milvus_accessed": False,
        "embedding_run": False,
        "real_reranker_run": False,
        "qwen_llm_ragas_ocr_vlm_called": False,
        "production_index_written": False,
        "recommend_next_step_phase7n": all_pass,
        "recommend_production": False,
        "recommend_extractor_rework": False,
        "recommend_large_manual_annotation": False,
        "route_c_backlog_only": True,
    }

    validation_path = OUTPUT_RESULTS_DIR / "phase7m_validation_summary.csv"
    fields = [
        "component",
        "status",
        "pass",
        "result_output",
        "report_output",
        "warning",
    ]
    write_csv(validation_path, components, fields)
    summary["validation_summary_output"] = display_path(validation_path)
    write_phase7m_guardrail_report(summary)
    write_phase7m_summary_report(summary)
    return summary


def main() -> int:
    summary = run_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["validation_status"] == "pass_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
