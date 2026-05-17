#!/usr/bin/env python3
"""Build Phase7Q-1 table citation mapper dry-run fixtures."""

from __future__ import annotations

import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

PHASE7Q_DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_schema_prototype"
PHASE7M_RESULTS = (
    ROOT / "results/v7_phase7_table_rag_contract_hardening/generation_v2_contract_results.jsonl"
)
PHASE7M_CITATION_GUARD = (
    ROOT / "results/v7_phase7_table_rag_contract_hardening/citation_guard_results.csv"
)
PHASE7L_SUPPORT_PACK = ROOT / "results/v7_phase7_table_rag_smoke/support_pack_preview.jsonl"
PHASE7P_FIXTURE = (
    ROOT
    / "data/experiments/v7_phase7_table_rag_reranker_compatibility_smoke/reranker_input_fixture.csv"
)
PHASE7P_RESULTS = (
    ROOT / "results/v7_phase7_table_rag_reranker_compatibility_smoke/reranker_smoke_results.csv"
)

REPORT_DIR = ROOT / "reports/v7_phase7_table_citation_binder_prototype_dry_run"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_binder_prototype_dry_run"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_binder_prototype_dry_run"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_support_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for item in row.get("support_pack") or []:
                chunk_id = item.get("chunk_id")
                if chunk_id and chunk_id not in index:
                    index[chunk_id] = item
    return index


def first_csv_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def strip_table_caption_prefix(value: str | None) -> str | None:
    if not value:
        return value
    prefix = "[TABLE CAPTION]"
    if value.startswith(prefix):
        return value[len(prefix) :].strip()
    return value


def flatten_header_path(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    flattened: list[str] = []
    for item in value:
        if isinstance(item, list):
            flattened.append(" / ".join(str(part) for part in item if part is not None))
        elif item is not None:
            flattened.append(str(item))
    return [item for item in flattened if item]


def table_index_unit_id_from_chunk(chunk_id: str) -> str | None:
    prefix = "table_unit::"
    if chunk_id.startswith(prefix):
        return chunk_id[len(prefix) :]
    return None


def derive_candidate_id(seed_id: str | None, source_csv_path: str | None) -> str | None:
    if source_csv_path:
        return Path(source_csv_path).stem
    if seed_id and "__" in seed_id:
        return seed_id.split("__", 1)[1]
    return None


def query_type_for_unit(table_unit_type: str) -> str:
    if table_unit_type == "table_unit":
        return "table_lookup"
    if table_unit_type == "row_unit":
        return "row_lookup"
    return "metric_lookup"


def fixture_from_phase7m_row(
    row: dict[str, Any],
    support_index: dict[str, dict[str, Any]],
    fixture_id: str,
    fixture_type: str,
) -> dict[str, Any]:
    candidate = row["citation_candidate_debug"]
    chunk_id = candidate["chunk_id"]
    support_item = support_index.get(chunk_id) or {}
    support_metadata = support_item.get("metadata") or {}
    table_unit_type = row.get("table_unit_type") or support_metadata.get("table_unit_type")
    source_csv_path = row.get("source_csv_path") or support_metadata.get("source_csv_path")
    source_pdf_crop_path = row.get("source_pdf_crop_path") or support_metadata.get(
        "source_pdf_crop_path"
    )
    source_markdown_path = support_metadata.get("source_markdown_path")
    seed_id = row.get("seed_id") or support_metadata.get("seed_id")
    candidate_id = support_metadata.get("candidate_id") or derive_candidate_id(
        seed_id, source_csv_path
    )
    caption = support_metadata.get("caption") or strip_table_caption_prefix(candidate.get("title"))
    metadata = {
        "object_type": "table_index_unit",
        "table_index_unit_id": support_metadata.get("table_index_unit_id")
        or table_index_unit_id_from_chunk(chunk_id),
        "table_unit_type": table_unit_type,
        "seed_id": seed_id,
        "candidate_id": candidate_id,
        "doc_id": row.get("doc_id") or candidate.get("doc_id"),
        "table_id": row.get("table_id") or support_metadata.get("table_id"),
        "caption": strip_table_caption_prefix(caption),
        "retrieval_text": support_metadata.get("retrieval_text") or candidate.get("text"),
        "row_label": row.get("row_label") if "row_label" in row else support_metadata.get("row_label"),
        "header_path": flatten_header_path(support_metadata.get("header_path")),
        "source_csv_path": source_csv_path,
        "source_pdf_crop_path": source_pdf_crop_path,
        "source_markdown_path": source_markdown_path,
        "source_span_granularity": support_metadata.get("source_span_granularity")
        or ("table" if table_unit_type == "table_unit" else "table_row_level"),
        "value_bboxes_available": row.get("value_bboxes_available", False),
        "cell_bboxes_available": support_metadata.get("cell_bboxes_available"),
        "production_ready": row.get("production_ready", False),
        "index_unit_status": support_metadata.get("index_unit_status", "preview_only"),
        "binding_review_limitation": row.get("binding_review_limitation")
        or support_metadata.get("binding_review_limitation"),
    }
    return {
        "fixture_id": fixture_id,
        "fixture_type": fixture_type,
        "source_artifacts": [
            rel(PHASE7M_RESULTS),
            rel(PHASE7L_SUPPORT_PACK),
        ],
        "query_id": f"phase7q1_{fixture_id}",
        "query_type": query_type_for_unit(str(table_unit_type)),
        "expected_mapper_status": "mapped_with_warnings",
        "expected_block_reason_contains": "",
        "expected_formal_citation_allowed": False,
        "expected_debug_provenance_only": True,
        "retrieved_chunk": {
            "chunk_id": chunk_id,
            "doc_id": candidate.get("doc_id"),
            "source_file": candidate.get("source_file"),
            "title": candidate.get("title"),
            "section": candidate.get("section"),
            "text": candidate.get("text"),
            "page_start": candidate.get("page_start"),
            "page_end": candidate.get("page_end"),
            "metadata": metadata,
        },
    }


def build_fixtures() -> list[dict[str, Any]]:
    phase7m_rows = load_jsonl(PHASE7M_RESULTS)
    support_index = build_support_index(PHASE7L_SUPPORT_PACK)
    by_unit = {row["table_unit_type"]: row for row in phase7m_rows}
    fixtures = [
        fixture_from_phase7m_row(
            by_unit["table_unit"],
            support_index,
            "table_level_from_phase7m",
            "table_level",
        ),
        fixture_from_phase7m_row(
            by_unit["row_unit"],
            support_index,
            "row_level_from_phase7m",
            "row_level",
        ),
        fixture_from_phase7m_row(
            by_unit["cell_group_unit"],
            support_index,
            "cell_group_from_phase7m",
            "cell_group_level",
        ),
    ]

    csv_sanitized = deepcopy(fixtures[1])
    csv_sanitized["fixture_id"] = "csv_source_file_sanitized"
    csv_sanitized["fixture_type"] = "csv_source_file_sanitized"
    csv_sanitized["query_id"] = "phase7q1_csv_source_file_sanitized"
    fixtures.append(csv_sanitized)

    missing_table_id = deepcopy(fixtures[1])
    missing_table_id["fixture_id"] = "malformed_missing_table_id"
    missing_table_id["fixture_type"] = "malformed_missing_table_id"
    missing_table_id["query_id"] = "phase7q1_malformed_missing_table_id"
    missing_table_id["expected_mapper_status"] = "blocked"
    missing_table_id["expected_block_reason_contains"] = "missing_table_id"
    missing_table_id["expected_debug_provenance_only"] = False
    missing_table_id["retrieved_chunk"]["metadata"]["table_id"] = ""
    fixtures.append(missing_table_id)

    value_scope = deepcopy(fixtures[1])
    value_scope["fixture_id"] = "malformed_value_scope"
    value_scope["fixture_type"] = "malformed_value_scope"
    value_scope["query_id"] = "phase7q1_malformed_value_scope"
    value_scope["expected_mapper_status"] = "blocked"
    value_scope["expected_block_reason_contains"] = "citation_scope_value_forbidden"
    value_scope["expected_debug_provenance_only"] = False
    value_scope["retrieved_chunk"]["metadata"]["forced_citation_scope"] = "value"
    fixtures.append(value_scope)

    non_table_query = deepcopy(fixtures[0])
    non_table_query["fixture_id"] = "non_table_query_table_candidate"
    non_table_query["fixture_type"] = "non_table_query_blocked"
    non_table_query["source_artifacts"] = [
        rel(PHASE7M_RESULTS),
        rel(PHASE7P_FIXTURE),
        rel(PHASE7P_RESULTS),
    ]
    non_table_query["query_id"] = "phase7p_non_table_001"
    non_table_query["query_type"] = "non_table_query"
    non_table_query["expected_mapper_status"] = "blocked"
    non_table_query["expected_block_reason_contains"] = "non_table_query_blocks_table_citation"
    non_table_query["expected_debug_provenance_only"] = False
    fixtures.append(non_table_query)

    phase7p_rows = load_csv(PHASE7P_FIXTURE)
    normal_row = first_csv_row(
        phase7p_rows,
        query_id="phase7p_non_table_001",
        candidate_type="normal",
    ) or first_csv_row(phase7p_rows, candidate_type="normal")
    if normal_row is None:
        raise RuntimeError("Phase7P normal fixture row not found")
    fixtures.append(
        {
            "fixture_id": "normal_chunk_not_mapped",
            "fixture_type": "normal_chunk_not_mapped",
            "source_artifacts": [rel(PHASE7P_FIXTURE)],
            "query_id": normal_row.get("query_id") or "phase7q1_normal_chunk",
            "query_type": normal_row.get("query_type") or "non_table_query",
            "expected_mapper_status": "blocked",
            "expected_block_reason_contains": "normal_chunk_not_table_evidence",
            "expected_formal_citation_allowed": False,
            "expected_debug_provenance_only": False,
            "retrieved_chunk": {
                "chunk_id": normal_row.get("chunk_id"),
                "doc_id": normal_row.get("doc_id"),
                "source_file": f"papers/{normal_row.get('doc_id')}.pdf",
                "title": "Synthetic normal fixture from Phase7P",
                "section": "normal_stub",
                "text": normal_row.get("question"),
                "page_start": None,
                "page_end": None,
                "metadata": {
                    "object_type": "normal_chunk",
                    "production_ready": normal_row.get("production_ready") == "True",
                    "index_unit_status": normal_row.get("index_unit_status"),
                },
            },
        }
    )
    return fixtures


def artifact_manifest(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = [
        (PHASE7Q_DATA_DIR / "table_evidence_citation_schema.json", "Phase7Q schema input"),
        (PHASE7Q_DATA_DIR / "citation_mapping_matrix.csv", "Phase7Q mapping matrix input"),
        (PHASE7M_RESULTS, "Phase7M candidate/debug rows"),
        (PHASE7M_CITATION_GUARD, "Phase7M citation guard reference"),
        (PHASE7L_SUPPORT_PACK, "Phase7L support metadata/header-path reference"),
        (PHASE7P_FIXTURE, "Phase7P fixture reference for non-table/normal cases"),
        (PHASE7P_RESULTS, "Phase7P reranker result reference; not rerun"),
    ]
    used = Counter(path for fixture in fixtures for path in fixture.get("source_artifacts", []))
    rows: list[dict[str, Any]] = []
    for path, role in paths:
        count = count_jsonl(path) if path.suffix == ".jsonl" else len(load_csv(path)) if path.suffix == ".csv" and path.exists() else (1 if path.exists() else 0)
        rows.append(
            {
                "artifact_path": rel(path),
                "exists": path.exists(),
                "record_count": count,
                "role": role,
                "used_by_fixture_count": used.get(rel(path), 0),
                "read_only": True,
            }
        )
    return rows


def fixture_summary_rows(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        chunk = fixture["retrieved_chunk"]
        metadata = chunk.get("metadata") or {}
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "fixture_type": fixture["fixture_type"],
                "query_type": fixture["query_type"],
                "chunk_id": chunk.get("chunk_id"),
                "object_type": metadata.get("object_type"),
                "table_unit_type": metadata.get("table_unit_type", ""),
                "table_id": metadata.get("table_id", ""),
                "expected_mapper_status": fixture["expected_mapper_status"],
                "expected_block_reason_contains": fixture["expected_block_reason_contains"],
                "expected_formal_citation_allowed": fixture["expected_formal_citation_allowed"],
                "expected_debug_provenance_only": fixture["expected_debug_provenance_only"],
            }
        )
    return rows


def render_guardrail() -> str:
    return """# Phase7Q-1 Guardrail

Phase7Q-1 is a table citation mapper prototype dry-run. It converts already-existing table candidate/debug artifacts into `TableEvidenceCitation` prototype objects or blocked records.

Boundaries:

- Do not modify `src/`, `configs/`, the ingestion pipeline, the current `Citation` dataclass, or production `CitationBinder`.
- Do not generate answers or formal production citations.
- Do not promote preview table units into production evidence.
- Keep `source_csv_path`, `source_pdf_crop_path`, and markdown cards in debug provenance only.
- Do not call Qwen, LLMs, RAGAS, OCR, VLM, embedding, reranker, Milvus, or official BM25.
- Route C remains backlog.

Allowed behavior is limited to read-only artifact inspection and new Phase7Q-1 reports, data fixtures, results, scripts, and tests."""


def render_manifest_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase7Q-1 Input Artifact Manifest",
        "",
        "All inputs are read-only artifacts from Phase7Q, Phase7M, Phase7L, and Phase7P. Phase7P reranker outputs are used only as static files; the reranker is not run.",
        "",
        "| artifact_path | exists | record_count | role | used_by_fixture_count |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['artifact_path']}` | {row['exists']} | {row['record_count']} | {row['role']} | {row['used_by_fixture_count']} |"
        )
    return "\n".join(lines)


def render_mapper_contract() -> str:
    return """# Phase7Q-1 Mapper Contract

## Input

The mapper consumes a dry-run fixture row with:

- `query_id` and `query_type`
- expected mapper status
- a table-adapted `retrieved_chunk` object with `chunk_id`, `doc_id`, `source_file`, text/page fields, and `metadata`

The fixture is built from existing Phase7M/7L/7P artifacts. It does not query retrieval stores or rerun ranking.

## Output

Each input becomes either:

- a `TableEvidenceCitation` prototype object with `mapper_status=mapped_with_warnings`; or
- a blocked record with `mapper_status=blocked` and structured `block_reasons`.

No output is a production citation. All mapped records remain formal-citation blocked because Phase7 table units are still `production_ready=false` and `index_unit_status=preview_only`.

## Mapping Rules

- `canonical_source` is formal source only. If `RetrievedChunk.source_file` is a CSV or crop path, the mapper does not copy it into `canonical_source.source_file`.
- `provenance_debug` receives CSV/crop/markdown paths and table-index trace ids.
- `table_unit_type` maps to citation scope: `table_unit -> table`, `row_unit -> row`, `cell_group_unit -> cell_group`.
- `citation_scope=value` is forbidden.
- `query_type=non_table_query` blocks table citation.
- `object_type != table_index_unit` blocks table citation.
- Missing `doc_id`, `table_id`, `table_unit_type`, or text blocks mapping.
- `production_ready=false`, `index_unit_status=preview_only`, `value_bboxes_available=false`, and warning-level binding are surfaced as limitations and warnings.

## Block Modes

- `normal_chunk_not_table_evidence`
- `non_table_query_blocks_table_citation`
- `missing_doc_id`
- `missing_table_id`
- `invalid_table_unit_type`
- `missing_quote_text`
- `citation_scope_value_forbidden`
- `invalid_citation_scope`

## Warning Modes

- `canonical_source_file_unresolved`
- `production_ready_false_blocks_formal_citation`
- `preview_only_blocks_formal_citation`
- `value_bboxes_unavailable`
- `binding_warning_level`"""


def render_fixture_report(fixtures: list[dict[str, Any]]) -> str:
    rows = fixture_summary_rows(fixtures)
    lines = [
        "# Phase7Q-1 Mapper Input Fixture",
        "",
        f"- fixture_count: {len(fixtures)}",
        f"- mapped_expected_count: {sum(1 for row in rows if row['expected_mapper_status'] == 'mapped_with_warnings')}",
        f"- blocked_expected_count: {sum(1 for row in rows if row['expected_mapper_status'] == 'blocked')}",
        "",
        "| fixture_id | fixture_type | query_type | object_type | table_unit_type | expected_mapper_status | expected_block_reason_contains |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['fixture_id']}` | {row['fixture_type']} | {row['query_type']} | {row['object_type']} | {row['table_unit_type']} | {row['expected_mapper_status']} | {row['expected_block_reason_contains'] or '-'} |"
        )
    return "\n".join(lines)


def build_fixture_artifacts() -> dict[str, Any]:
    ensure_dirs()
    fixtures = build_fixtures()
    manifest_rows = artifact_manifest(fixtures)
    summary_rows = fixture_summary_rows(fixtures)

    write_jsonl(DATA_DIR / "mapper_input_fixture.jsonl", fixtures)
    write_csv(
        DATA_DIR / "input_artifact_manifest.csv",
        manifest_rows,
        [
            "artifact_path",
            "exists",
            "record_count",
            "role",
            "used_by_fixture_count",
            "read_only",
        ],
    )
    write_csv(
        DATA_DIR / "mapper_input_fixture_summary.csv",
        summary_rows,
        [
            "fixture_id",
            "fixture_type",
            "query_type",
            "chunk_id",
            "object_type",
            "table_unit_type",
            "table_id",
            "expected_mapper_status",
            "expected_block_reason_contains",
            "expected_formal_citation_allowed",
            "expected_debug_provenance_only",
        ],
    )
    write_text(REPORT_DIR / "phase7q1_guardrail.md", render_guardrail())
    write_text(REPORT_DIR / "input_artifact_manifest.md", render_manifest_report(manifest_rows))
    write_text(REPORT_DIR / "mapper_contract.md", render_mapper_contract())
    write_text(REPORT_DIR / "mapper_input_fixture_report.md", render_fixture_report(fixtures))
    return {
        "fixture_count": len(fixtures),
        "mapped_expected_count": sum(
            1 for fixture in fixtures if fixture["expected_mapper_status"] == "mapped_with_warnings"
        ),
        "blocked_expected_count": sum(
            1 for fixture in fixtures if fixture["expected_mapper_status"] == "blocked"
        ),
    }


def main() -> int:
    summary = build_fixture_artifacts()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
