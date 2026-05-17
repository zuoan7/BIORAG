#!/usr/bin/env python3
"""Build Phase7Q table citation schema prototype artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7q_validate_citation_schema_examples import (
    write_validation_artifacts,
)

REPORT_DIR = ROOT / "reports/v7_phase7_table_citation_schema_prototype"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_citation_schema_prototype"
RESULTS_DIR = ROOT / "results/v7_phase7_table_citation_schema_prototype"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def table_evidence_citation_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://biorag.local/schemas/phase7q/table_evidence_citation.schema.json",
        "title": "TableEvidenceCitation",
        "description": "Phase7Q prototype schema for typed table evidence citations. This is not a production Citation replacement.",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "citation_type",
            "citation_id",
            "doc_id",
            "canonical_source",
            "table_scope",
            "evidence_scope",
            "quote",
            "provenance_debug",
            "limitations",
        ],
        "properties": {
            "citation_type": {"const": "table_evidence"},
            "citation_id": {"type": "string", "minLength": 1},
            "doc_id": {"type": "string", "minLength": 1},
            "canonical_source": {
                "type": "object",
                "additionalProperties": False,
                "required": ["paper_title", "source_file", "doi", "pmid"],
                "properties": {
                    "paper_title": {"type": ["string", "null"]},
                    "source_file": {
                        "type": ["string", "null"],
                        "description": "Formal canonical paper source only. Debug CSV or crop paths are forbidden here.",
                    },
                    "doi": {"type": ["string", "null"]},
                    "pmid": {"type": ["string", "null"]},
                },
            },
            "table_scope": {
                "type": "object",
                "additionalProperties": False,
                "required": ["table_id", "table_caption", "page_start", "page_end"],
                "properties": {
                    "table_id": {"type": "string", "minLength": 1},
                    "table_caption": {"type": ["string", "null"]},
                    "page_start": {"type": ["integer", "null"]},
                    "page_end": {"type": ["integer", "null"]},
                },
            },
            "evidence_scope": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "table_unit_type",
                    "citation_scope",
                    "row_label",
                    "header_path",
                    "source_span_granularity",
                ],
                "properties": {
                    "table_unit_type": {
                        "enum": ["table_unit", "row_unit", "cell_group_unit"]
                    },
                    "citation_scope": {
                        "enum": ["table", "row", "cell_group"],
                        "description": "The prototype intentionally excludes value-level citation scope.",
                    },
                    "row_label": {"type": ["string", "null"]},
                    "header_path": {"type": "array", "items": {"type": "string"}},
                    "source_span_granularity": {
                        "enum": [
                            "table",
                            "row",
                            "cell_group",
                            "cell_level",
                            "table_row_level",
                        ]
                    },
                },
            },
            "quote": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "quote_scope"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "quote_scope": {
                        "enum": [
                            "table_summary",
                            "row_summary",
                            "cell_group_summary",
                        ]
                    },
                },
            },
            "provenance_debug": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_csv_path",
                    "source_pdf_crop_path",
                    "source_markdown_path",
                    "table_index_unit_id",
                    "seed_id",
                    "candidate_id",
                ],
                "properties": {
                    "source_csv_path": {"type": ["string", "null"]},
                    "source_pdf_crop_path": {"type": ["string", "null"]},
                    "source_markdown_path": {"type": ["string", "null"]},
                    "table_index_unit_id": {"type": ["string", "null"]},
                    "seed_id": {"type": ["string", "null"]},
                    "candidate_id": {"type": ["string", "null"]},
                },
            },
            "limitations": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "production_ready",
                    "index_unit_status",
                    "value_bboxes_available",
                    "cell_bboxes_available",
                    "binding_review_level",
                    "bbox_verification_level",
                    "value_level_citation_claim_allowed",
                ],
                "properties": {
                    "production_ready": {"const": False},
                    "index_unit_status": {"const": "preview_only"},
                    "value_bboxes_available": {"const": False},
                    "cell_bboxes_available": {"type": ["boolean", "null"]},
                    "binding_review_level": {
                        "enum": ["warning", "reviewed", "verified"]
                    },
                    "bbox_verification_level": {
                        "enum": ["none", "table", "cell", "value"]
                    },
                    "value_level_citation_claim_allowed": {"const": False},
                },
            },
        },
    }


def citation_object(
    *,
    citation_id: str,
    doc_id: str,
    source_file: str,
    paper_title: str,
    table_id: str,
    table_caption: str,
    page: int,
    table_unit_type: str,
    citation_scope: str,
    row_label: str | None,
    header_path: list[str],
    source_span_granularity: str,
    quote_text: str,
    quote_scope: str,
    table_index_unit_id: str,
    seed_id: str,
    candidate_id: str,
    csv_path: str,
    crop_path: str,
    markdown_path: str,
    cell_bboxes_available: bool | None = True,
) -> dict[str, Any]:
    return {
        "citation_type": "table_evidence",
        "citation_id": citation_id,
        "doc_id": doc_id,
        "canonical_source": {
            "paper_title": paper_title,
            "source_file": source_file,
            "doi": None,
            "pmid": None,
        },
        "table_scope": {
            "table_id": table_id,
            "table_caption": table_caption,
            "page_start": page,
            "page_end": page,
        },
        "evidence_scope": {
            "table_unit_type": table_unit_type,
            "citation_scope": citation_scope,
            "row_label": row_label,
            "header_path": header_path,
            "source_span_granularity": source_span_granularity,
        },
        "quote": {
            "text": quote_text,
            "quote_scope": quote_scope,
        },
        "provenance_debug": {
            "source_csv_path": csv_path,
            "source_pdf_crop_path": crop_path,
            "source_markdown_path": markdown_path,
            "table_index_unit_id": table_index_unit_id,
            "seed_id": seed_id,
            "candidate_id": candidate_id,
        },
        "limitations": {
            "production_ready": False,
            "index_unit_status": "preview_only",
            "value_bboxes_available": False,
            "cell_bboxes_available": cell_bboxes_available,
            "binding_review_level": "warning",
            "bbox_verification_level": "table",
            "value_level_citation_claim_allowed": False,
        },
    }


def prototype_examples() -> list[dict[str, Any]]:
    common_csv = "data/experiments/v7_phase7_expanded_table_review_pack/csv_tables/phase7g_candidate_241__doc_0076__table_2__p008.csv"
    common_crop = "data/experiments/v7_phase7_expanded_table_review_pack/pdf_crops/phase7g_candidate_241__doc_0076__table_2__p008.png"
    common_md = "data/experiments/v7_phase7_expanded_table_review_pack/markdown_cards/phase7g_candidate_241__doc_0076__table_2__p008.md"
    table = citation_object(
        citation_id="phase7q_table_001",
        doc_id="doc_0076",
        source_file="papers/doc_0076.pdf",
        paper_title="doc_0076 canonical paper title unresolved in prototype",
        table_id="Table 2",
        table_caption="Table 2. Primers and plasmids used for this study.",
        page=8,
        table_unit_type="table_unit",
        citation_scope="table",
        row_label=None,
        header_path=[
            "Strain type (copy)",
            "Induction carbon source",
            "Product in culture supernatant (g/L)",
            "Reference",
        ],
        source_span_granularity="table",
        quote_text="Table 2 summarizes primers and plasmids and lists main columns for strain type, induction carbon source, product concentration, and reference.",
        quote_scope="table_summary",
        table_index_unit_id="phase7i_preview_004__table_unit",
        seed_id="phase7g2_expanded_seed_004__phase7g_candidate_241__doc_0076__table_2__p008",
        candidate_id="phase7g_candidate_241__doc_0076__table_2__p008",
        csv_path=common_csv,
        crop_path=common_crop,
        markdown_path=common_md,
    )
    row = citation_object(
        citation_id="phase7q_row_001",
        doc_id="doc_0076",
        source_file="papers/doc_0076.pdf",
        paper_title="doc_0076 canonical paper title unresolved in prototype",
        table_id="Table 2",
        table_caption="Table 2. Primers and plasmids used for this study.",
        page=8,
        table_unit_type="row_unit",
        citation_scope="row",
        row_label="Mut+/s (6-8)",
        header_path=[
            "Strain type (copy)",
            "Induction carbon source",
            "Product in culture supernatant (g/L)",
            "Reference",
        ],
        source_span_granularity="table_row_level",
        quote_text='Row "Mut+/s (6-8)" reports methanol induction, product in culture supernatant of 1.50 g/L, and reference 50.',
        quote_scope="row_summary",
        table_index_unit_id="phase7i_preview_004__row_unit_001",
        seed_id="phase7g2_expanded_seed_004__phase7g_candidate_241__doc_0076__table_2__p008",
        candidate_id="phase7g_candidate_241__doc_0076__table_2__p008",
        csv_path=common_csv,
        crop_path=common_crop,
        markdown_path=common_md,
    )
    cell_group = citation_object(
        citation_id="phase7q_cell_group_001",
        doc_id="doc_0261",
        source_file="papers/doc_0261.pdf",
        paper_title="doc_0261 canonical paper title unresolved in prototype",
        table_id="Table 2",
        table_caption="Table 2. Phylum and genus-level microbiota changes across treatment groups.",
        page=6,
        table_unit_type="cell_group_unit",
        citation_scope="cell_group",
        row_label="Verrucomicrobia",
        header_path=[
            "Abundance, % (mean +/- SD) / Control",
            "Abundance, % (mean +/- SD) / 2-FL",
            "Abundance, % (mean +/- SD) / Lactose",
            "Abundance, % (mean +/- SD) / GOS",
            "Overall p-value / FDR adjusted",
        ],
        source_span_granularity="cell_group",
        quote_text='Row "Verrucomicrobia" has a selected value group covering treatment abundance columns and the overall adjusted p-value.',
        quote_scope="cell_group_summary",
        table_index_unit_id="phase7i_preview_005__cell_group_unit_020_001",
        seed_id="phase7g2_expanded_seed_005__phase7g_candidate_294__doc_0261__table_2__p006",
        candidate_id="phase7g_candidate_294__doc_0261__table_2__p006",
        csv_path="data/experiments/v7_phase7_expanded_table_review_pack/csv_tables/phase7g_candidate_294__doc_0261__table_2__p006.csv",
        crop_path="data/experiments/v7_phase7_expanded_table_review_pack/pdf_crops/phase7g_candidate_294__doc_0261__table_2__p006.png",
        markdown_path="data/experiments/v7_phase7_expanded_table_review_pack/markdown_cards/phase7g_candidate_294__doc_0261__table_2__p006.md",
    )
    malformed = citation_object(
        citation_id="phase7q_malformed_001",
        doc_id="doc_0076",
        source_file=common_csv,
        paper_title="doc_0076 canonical paper title unresolved in prototype",
        table_id="Table 2",
        table_caption="Table 2. Primers and plasmids used for this study.",
        page=8,
        table_unit_type="row_unit",
        citation_scope="value",
        row_label="Mut+/s (6-8)",
        header_path=["Product in culture supernatant (g/L)"],
        source_span_granularity="cell_level",
        quote_text="Malformed example attempts to cite an individual value as formal evidence.",
        quote_scope="row_summary",
        table_index_unit_id="phase7i_preview_004__row_unit_001",
        seed_id="phase7g2_expanded_seed_004__phase7g_candidate_241__doc_0076__table_2__p008",
        candidate_id="phase7g_candidate_241__doc_0076__table_2__p008",
        csv_path=common_csv,
        crop_path=common_crop,
        markdown_path=common_md,
    )
    malformed["limitations"]["value_level_citation_claim_allowed"] = True

    non_table = citation_object(
        citation_id="phase7q_non_table_query_001",
        doc_id="doc_0076",
        source_file="papers/doc_0076.pdf",
        paper_title="doc_0076 canonical paper title unresolved in prototype",
        table_id="Table 2",
        table_caption="Table 2. Primers and plasmids used for this study.",
        page=8,
        table_unit_type="table_unit",
        citation_scope="table",
        row_label=None,
        header_path=["Strain type (copy)", "Induction carbon source"],
        source_span_granularity="table",
        quote_text="This structurally valid object is blocked because the query context is non_table_query.",
        quote_scope="table_summary",
        table_index_unit_id="phase7i_preview_004__table_unit",
        seed_id="phase7g2_expanded_seed_004__phase7g_candidate_241__doc_0076__table_2__p008",
        candidate_id="phase7g_candidate_241__doc_0076__table_2__p008",
        csv_path=common_csv,
        crop_path=common_crop,
        markdown_path=common_md,
    )

    warning_reason = (
        "production_ready=false; index_unit_status=preview_only; "
        "value_bboxes_available=false; binding_review_level=warning"
    )
    return [
        {
            "example_id": "phase7q_example_table_level",
            "example_type": "table_level",
            "example_context": {
                "query_type": "table_lookup",
                "retrieved_chunk_object_type": "table_index_unit",
            },
            "schema_object": table,
            "expected_validation_status": "pass_with_warnings",
            "block_reason": "",
            "warning_reason": warning_reason,
            "formal_citation_allowed": False,
            "debug_provenance_only": True,
        },
        {
            "example_id": "phase7q_example_row_level",
            "example_type": "row_level",
            "example_context": {
                "query_type": "row_lookup",
                "retrieved_chunk_object_type": "table_index_unit",
            },
            "schema_object": row,
            "expected_validation_status": "pass_with_warnings",
            "block_reason": "",
            "warning_reason": warning_reason,
            "formal_citation_allowed": False,
            "debug_provenance_only": True,
        },
        {
            "example_id": "phase7q_example_cell_group_level",
            "example_type": "cell_group_level",
            "example_context": {
                "query_type": "metric_lookup",
                "retrieved_chunk_object_type": "table_index_unit",
            },
            "schema_object": cell_group,
            "expected_validation_status": "pass_with_warnings",
            "block_reason": "",
            "warning_reason": warning_reason,
            "formal_citation_allowed": False,
            "debug_provenance_only": True,
        },
        {
            "example_id": "phase7q_example_malformed_blocked",
            "example_type": "malformed_blocked",
            "example_context": {
                "query_type": "row_lookup",
                "retrieved_chunk_object_type": "table_index_unit",
            },
            "schema_object": malformed,
            "expected_validation_status": "blocked",
            "block_reason": "citation_scope=value; canonical_source.source_file is CSV debug path; value-level claim requested",
            "warning_reason": "",
            "formal_citation_allowed": False,
            "debug_provenance_only": False,
        },
        {
            "example_id": "phase7q_example_non_table_query_blocked",
            "example_type": "non_table_query_blocked",
            "example_context": {
                "query_type": "non_table_query",
                "retrieved_chunk_object_type": "table_index_unit",
            },
            "schema_object": non_table,
            "expected_validation_status": "blocked",
            "block_reason": "non_table_query blocks table evidence citation",
            "warning_reason": warning_reason,
            "formal_citation_allowed": False,
            "debug_provenance_only": False,
        },
    ]


def mapping_rows() -> list[dict[str, str]]:
    return [
        {
            "target_field": "doc_id",
            "source_field": "RetrievedChunk.doc_id or metadata.doc_id",
            "source_object": "RetrievedChunk/CitationCandidate",
            "citation_layer": "formal",
            "required": "yes",
            "mapping_rule": "Copy document identity; mismatch between chunk doc_id and metadata doc_id blocks.",
        },
        {
            "target_field": "canonical_source.paper_title",
            "source_field": "RetrievedChunk.title after table-caption prefix stripping or paper metadata title",
            "source_object": "RetrievedChunk/CitationCandidate",
            "citation_layer": "formal",
            "required": "no",
            "mapping_rule": "Use canonical paper title when available; table caption alone is not a paper title.",
        },
        {
            "target_field": "canonical_source.source_file",
            "source_field": "RetrievedChunk.source_file only if it is canonical paper source",
            "source_object": "RetrievedChunk/CitationCandidate",
            "citation_layer": "formal",
            "required": "no",
            "mapping_rule": "Must not copy source_csv_path or source_pdf_crop_path; debug paths block formal mapping.",
        },
        {
            "target_field": "table_scope.table_id",
            "source_field": "metadata.table_id",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "formal",
            "required": "yes",
            "mapping_rule": "Copy table id; missing value blocks table citation.",
        },
        {
            "target_field": "table_scope.table_caption",
            "source_field": "metadata.caption or RetrievedChunk.title",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "formal",
            "required": "no",
            "mapping_rule": "Use caption for table scope, not paper identity.",
        },
        {
            "target_field": "evidence_scope.row_label",
            "source_field": "metadata.row_label",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "formal",
            "required": "scope-dependent",
            "mapping_rule": "Required for row and cell_group scopes; null allowed for table scope.",
        },
        {
            "target_field": "evidence_scope.header_path",
            "source_field": "metadata.header_path",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "formal",
            "required": "yes",
            "mapping_rule": "Flatten selected header hierarchy into string array for the cited table/row/cell group.",
        },
        {
            "target_field": "table_scope.page_start",
            "source_field": "RetrievedChunk.page_start",
            "source_object": "RetrievedChunk/CitationCandidate",
            "citation_layer": "formal",
            "required": "no",
            "mapping_rule": "Copy page start when available; null otherwise.",
        },
        {
            "target_field": "table_scope.page_end",
            "source_field": "RetrievedChunk.page_end",
            "source_object": "RetrievedChunk/CitationCandidate",
            "citation_layer": "formal",
            "required": "no",
            "mapping_rule": "Copy page end when available; null otherwise.",
        },
        {
            "target_field": "evidence_scope.table_unit_type",
            "source_field": "metadata.table_unit_type",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "formal guard",
            "required": "yes",
            "mapping_rule": "Allowed values are table_unit, row_unit, cell_group_unit.",
        },
        {
            "target_field": "evidence_scope.citation_scope",
            "source_field": "derived from metadata.table_unit_type and query type",
            "source_object": "CitationCandidate/query policy",
            "citation_layer": "formal guard",
            "required": "yes",
            "mapping_rule": "table_unit -> table; row_unit -> row; cell_group_unit -> cell_group; value is forbidden.",
        },
        {
            "target_field": "quote.text",
            "source_field": "RetrievedChunk.text or metadata.retrieval_text",
            "source_object": "RetrievedChunk/CitationCandidate",
            "citation_layer": "formal quote",
            "required": "yes",
            "mapping_rule": "Use bounded table/row/cell-group summary text; no generated answer text.",
        },
        {
            "target_field": "provenance_debug.source_csv_path",
            "source_field": "metadata.source_csv_path",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "debug",
            "required": "no",
            "mapping_rule": "Copy only into provenance_debug; never into canonical_source.",
        },
        {
            "target_field": "provenance_debug.source_pdf_crop_path",
            "source_field": "metadata.source_pdf_crop_path",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "debug",
            "required": "no",
            "mapping_rule": "Copy only into provenance_debug; never into canonical_source.",
        },
        {
            "target_field": "provenance_debug.table_index_unit_id",
            "source_field": "metadata.table_index_unit_id or chunk_id suffix",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "debug",
            "required": "no",
            "mapping_rule": "Trace prototype table unit identity.",
        },
        {
            "target_field": "provenance_debug.seed_id",
            "source_field": "metadata.seed_id",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "debug",
            "required": "no",
            "mapping_rule": "Trace seed grouping only.",
        },
        {
            "target_field": "provenance_debug.candidate_id",
            "source_field": "metadata.candidate_id",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "debug",
            "required": "no",
            "mapping_rule": "Trace extraction candidate only.",
        },
        {
            "target_field": "limitations.production_ready",
            "source_field": "metadata.production_ready",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "limitation guard",
            "required": "yes",
            "mapping_rule": "Phase7Q requires false; false blocks production formal citation.",
        },
        {
            "target_field": "limitations.index_unit_status",
            "source_field": "metadata.index_unit_status",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "limitation guard",
            "required": "yes",
            "mapping_rule": "Phase7Q requires preview_only; preview_only blocks production formal citation.",
        },
        {
            "target_field": "limitations.value_bboxes_available",
            "source_field": "metadata.value_bboxes_available",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "limitation guard",
            "required": "yes",
            "mapping_rule": "False forces value_level_citation_claim_allowed=false.",
        },
        {
            "target_field": "limitations.binding_review_level",
            "source_field": "metadata.binding_review_limitation/reference_ok/unit_or_note_ok",
            "source_object": "RetrievedChunk.metadata",
            "citation_layer": "limitation guard",
            "required": "yes",
            "mapping_rule": "Current warning-level binding maps to warning and must be surfaced.",
        },
    ]


def guard_delta_rows() -> list[dict[str, str]]:
    return [
        {
            "guard": "formal_source_debug_provenance_separation",
            "current_binder_gap": "Citation.source_file can hold a technical path.",
            "schema_delta": "canonical_source is formal; provenance_debug holds CSV/crop/markdown paths.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "csv_crop_path_formal_block",
            "current_binder_gap": "CSV path may appear in CitationCandidate.source_file debug.",
            "schema_delta": "Validator blocks CSV/crop equality or file extension in canonical_source.source_file.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "no_value_level_claim",
            "current_binder_gap": "No typed citation_scope exists.",
            "schema_delta": "citation_scope enum excludes value; value claim flag must be false.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "preview_only_production_ready_surface",
            "current_binder_gap": "Production readiness is only metadata/debug.",
            "schema_delta": "limitations explicitly exposes production_ready=false and index_unit_status=preview_only.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "binding_warning_surface",
            "current_binder_gap": "Warning-level binding is not present in Citation.",
            "schema_delta": "limitations.binding_review_level records warning/reviewed/verified.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "citation_scope_restriction",
            "current_binder_gap": "Citation has no table, row, or cell_group scope.",
            "schema_delta": "evidence_scope.citation_scope is table/row/cell_group only.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "malformed_metadata_block",
            "current_binder_gap": "Binder only checks chunk_id/doc_id/source_file/text presence.",
            "schema_delta": "Validator blocks missing fields, invalid scope, debug path in formal source, and value claims.",
            "phase7q_status": "prototype_guard",
        },
        {
            "guard": "non_table_query_block",
            "current_binder_gap": "CitationBinder itself has no query-type table evidence block.",
            "schema_delta": "Example context blocks table citation under non_table_query.",
            "phase7q_status": "prototype_guard",
        },
    ]


def render_guardrail() -> str:
    return """# Phase7Q Guardrail

Phase7Q only designs and prototypes a typed table citation schema. It does not implement production citation binding.

Hard boundaries:

- Do not modify the current `Citation` dataclass.
- Do not modify `CitationBinder` production behavior.
- Do not modify `src/`, `configs/`, or the ingestion pipeline.
- Do not generate an answer.
- Do not generate formal production citations.
- Do not promote preview table units into production evidence.
- Do not put `source_csv_path` or `source_pdf_crop_path` into a formal citation source.
- Do not call Qwen, any LLM, RAGAS, OCR, or VLM.
- Do not access Milvus, query official BM25, run embeddings, run a reranker, or build a production table index.
- Route C remains backlog.

Allowed outputs are limited to schema prototype reports, structured prototype files, offline validation results, and prototype tests under the Phase7Q paths."""


def render_gap_analysis() -> str:
    return """# Current Citation Gap Analysis

The current `Citation` dataclass in `src/synbio_rag/domain/schemas.py` has a normal text-chunk shape: `chunk_id`, `doc_id`, `title`, `source_file`, `section`, page range, score, and quote. `CitationBinder` builds this object from `CitationCandidate.source_file` after checking only basic text and metadata presence.

For table evidence, that shape is unsafe:

- `source_file` can confuse canonical paper source with debug paths such as CSV tables or PDF crops.
- There is no `citation_type`, so downstream code cannot distinguish table evidence from ordinary text evidence.
- There is no `citation_scope`, so table, row, cell-group, and forbidden value-level claims are indistinguishable.
- There is no table-specific scope: no `table_id`, `row_label`, selected `header_path`, or table page scope tied to the cited table.
- There is no `limitations` object to surface `production_ready=false`, `preview_only`, `value_bboxes_available=false`, or warning-level binding.
- There is no debug provenance layer separate from formal citation source.
- There is no way to express `value_bboxes_available=false` while still allowing a row or cell-group summary citation.
- There is no binding warning level in the public citation.
- There is no typed guard that blocks `preview_only` or `production_ready=false` table units from becoming formal citations.

Phase7M showed that debug metadata can flow through ledger/support/citation-candidate construction, but no formal citation was emitted. Phase7N and Phase7O kept CSV/crop paths debug-only and identified typed table citation as a blocker. Phase7P showed reranker score cannot be used as a production safety signal, so citation safety must live in schema and guard logic, not ranking."""


def render_schema_doc() -> str:
    return """# TableEvidenceCitation Schema Prototype

`TableEvidenceCitation` is a typed prototype for table evidence citation. It is not a production replacement for the current `Citation` dataclass.

## Required Shape

The structured schema is stored at:

- `data/experiments/v7_phase7_table_citation_schema_prototype/table_evidence_citation_schema.json`

Top-level fields:

- `citation_type`: constant `table_evidence`.
- `citation_id`: prototype citation id.
- `doc_id`: canonical document id.
- `canonical_source`: formal citation source with paper title, canonical source file, DOI, and PMID.
- `table_scope`: table id, caption, and page range.
- `evidence_scope`: table unit type, formal citation scope, row label, header path, and source-span granularity.
- `quote`: bounded table/row/cell-group quote text.
- `provenance_debug`: CSV/crop/markdown/unit/seed/candidate traceability only.
- `limitations`: explicit readiness, bbox, binding, and value-claim limits.

## Formal Source Rule

`canonical_source` is the only formal citation source. `source_csv_path`, `source_pdf_crop_path`, and `source_markdown_path` belong only in `provenance_debug`.

`canonical_source.source_file` must not equal a CSV path, a PDF crop path, or any debug artifact path. If only debug paths are available, the prototype citation can be retained for debug but must not become a formal production citation.

## Scope Rule

Allowed `citation_scope` values are:

- `table`
- `row`
- `cell_group`

`value` is intentionally not allowed. While `value_bboxes_available=false`, `value_level_citation_claim_allowed` must remain `false`.

## Limitation Rule

Phase7Q keeps all preview table units non-production:

- `production_ready=false`
- `index_unit_status=preview_only`
- `value_bboxes_available=false`
- `binding_review_level=warning`
- `value_level_citation_claim_allowed=false`

These limitations must be visible to downstream consumers instead of being hidden in debug metadata."""


def render_mapping_doc(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Citation Mapping From RetrievedChunk",
        "",
        "This matrix defines a prototype mapping from table-adapted `RetrievedChunk.metadata` and `CitationCandidate` fields into `TableEvidenceCitation`.",
        "",
        "Formal citation fields identify the paper, table, page, and evidence scope. Debug fields preserve CSV/crop/markdown and table-index traceability only.",
        "",
        "| target_field | source_field | citation_layer | mapping_rule |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_field']}` | `{row['source_field']}` | {row['citation_layer']} | {row['mapping_rule']} |"
        )
    lines.extend(
        [
            "",
            "Mapping blocks:",
            "",
            "- If `metadata.doc_id` conflicts with `RetrievedChunk.doc_id`, block.",
            "- If `source_csv_path` or `source_pdf_crop_path` would enter `canonical_source.source_file`, block.",
            "- If `citation_scope=value`, block.",
            "- If `production_ready=false` or `index_unit_status=preview_only`, block production formal citation and allow debug provenance only for otherwise valid examples.",
            "- If query context is `non_table_query`, block table citation even if a reranker ranks table evidence highly.",
        ]
    )
    return "\n".join(lines)


def render_examples_doc(examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Citation Prototype Examples",
        "",
        "The JSONL examples are stored at:",
        "",
        "- `data/experiments/v7_phase7_table_citation_schema_prototype/citation_prototype_examples.jsonl`",
        "",
        "Each example contains a `schema_object`, expected validation status, block or warning reason, formal citation allowance, and debug-provenance-only flag.",
        "",
        "| example_id | type | expected_validation_status | formal_citation_allowed | debug_provenance_only | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for example in examples:
        reason = example.get("block_reason") or example.get("warning_reason") or "-"
        lines.append(
            f"| `{example['example_id']}` | {example['example_type']} | {example['expected_validation_status']} | {example['formal_citation_allowed']} | {example['debug_provenance_only']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "The first three examples are structurally valid table, row, and cell-group citations, but they remain `pass_with_warnings` because current preview units are not production-ready formal citations.",
            "",
            "The malformed example is blocked for value scope, CSV-as-formal-source, and value-level claim attempt. The non-table-query example is blocked by query context.",
        ]
    )
    return "\n".join(lines)


def render_guard_delta_doc(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Citation Guard Delta",
        "",
        "The prototype schema adds guard surface that the current `CitationBinder` public citation cannot express.",
        "",
        "| guard | current_binder_gap | schema_delta | phase7q_status |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['guard']} | {row['current_binder_gap']} | {row['schema_delta']} | {row['phase7q_status']} |"
        )
    lines.extend(
        [
            "",
            "This is still a prototype delta. It does not change production binding behavior and does not enable formal table citation.",
        ]
    )
    return "\n".join(lines)


def render_summary(validation_summary: dict[str, Any]) -> str:
    generated_reports = [
        "phase7q_guardrail.md",
        "current_citation_gap_analysis.md",
        "table_evidence_citation_schema.md",
        "citation_mapping_from_retrieved_chunk.md",
        "citation_prototype_examples.md",
        "citation_guard_delta.md",
        "schema_validation_report.md",
        "phase7q_summary.md",
    ]
    generated_data = [
        "table_evidence_citation_schema.json",
        "citation_prototype_examples.jsonl",
        "citation_mapping_matrix.csv",
        "citation_guard_delta_matrix.csv",
    ]
    generated_results = ["schema_validation_results.csv"]
    return f"""# Phase7Q Summary

## 1. Generated Files

Reports:

{chr(10).join(f'- `reports/v7_phase7_table_citation_schema_prototype/{name}`' for name in generated_reports)}

Structured files:

{chr(10).join(f'- `data/experiments/v7_phase7_table_citation_schema_prototype/{name}`' for name in generated_data)}

Results:

{chr(10).join(f'- `results/v7_phase7_table_citation_schema_prototype/{name}`' for name in generated_results)}

Scripts/tests:

- `scripts/evaluation/phase7q_build_table_citation_schema.py`
- `scripts/evaluation/phase7q_validate_citation_schema_examples.py`
- `tests/test_phase7q_table_citation_schema_prototype.py`

## 2. Source And Config Guardrails

- Modified `src/`: no.
- Modified `configs/`: no.
- Accessed Milvus / queried official BM25: no.
- Called LLM / Qwen / RAGAS / OCR / VLM: no.
- Ran embedding / reranker: no.
- Built production table index: no.
- Generated answer: no.
- Generated formal production citation: no.

## 3. Current Citation Gap Analysis

Current `Citation` cannot safely encode table evidence because it has no typed citation kind, table/row/cell-group scope, limitation layer, or separation between canonical source and debug provenance. `CitationBinder` can preserve debug candidates but cannot make CSV/crop paths safe as formal citations.

## 4. TableEvidenceCitation Schema Conclusion

The prototype separates `canonical_source` from `provenance_debug`, exposes `table_scope`, `evidence_scope`, and `limitations`, excludes `citation_scope=value`, and forces `value_level_citation_claim_allowed=false`.

## 5. Mapping Matrix Conclusion

The mapping matrix defines which fields are formal citation fields and which are debug-only. CSV/crop/markdown paths map only into `provenance_debug`.

## 6. Prototype Examples

- total examples: 5
- table-level valid-with-warnings: 1
- row-level valid-with-warnings: 1
- cell-group-level valid-with-warnings: 1
- malformed blocked: 1
- non-table-query blocked: 1

## 7. Citation Guard Delta

The schema adds prototype guards for formal/debug separation, CSV/crop formal source blocking, no value-level claim, preview/production limitations, binding warning surfacing, legal citation scope, malformed metadata blocking, and non-table query blocking.

## 8. Schema Validation Result

- validation_status: `{validation_summary['validation_status']}`
- example_count: {validation_summary['example_count']}
- pass_count: {validation_summary['pass_count']}
- blocked_count: {validation_summary['blocked_count']}
- pass_with_warnings_count: {validation_summary['pass_with_warnings_count']}

## 9. Decision

- validation_status: `pass_with_warnings`
- Recommend entering Phase7R: yes, if the next goal is production index build/promote/rollback proposal.
- Conservative alternative: Phase7Q-1 citation binder prototype dry-run / no production binding.
- Recommend production: no.
- Recommend extractor rework: no.
- Recommend continued large manual annotation: no.
- Route C remains backlog: yes.

Warnings remain: schema is prototype only, not wired into production binder; table units remain `preview_only`, `production_ready=false`, `value_bboxes_available=false`, and warning-level binding; no LLM answer smoke, production index, or formal retrieval evaluation has run."""


def build_phase7q_artifacts() -> dict[str, Any]:
    ensure_dirs()
    schema = table_evidence_citation_schema()
    examples = prototype_examples()
    mappings = mapping_rows()
    deltas = guard_delta_rows()

    write_json(DATA_DIR / "table_evidence_citation_schema.json", schema)
    write_jsonl(DATA_DIR / "citation_prototype_examples.jsonl", examples)
    write_csv(
        DATA_DIR / "citation_mapping_matrix.csv",
        mappings,
        [
            "target_field",
            "source_field",
            "source_object",
            "citation_layer",
            "required",
            "mapping_rule",
        ],
    )
    write_csv(
        DATA_DIR / "citation_guard_delta_matrix.csv",
        deltas,
        ["guard", "current_binder_gap", "schema_delta", "phase7q_status"],
    )

    write_text(REPORT_DIR / "phase7q_guardrail.md", render_guardrail())
    write_text(REPORT_DIR / "current_citation_gap_analysis.md", render_gap_analysis())
    write_text(REPORT_DIR / "table_evidence_citation_schema.md", render_schema_doc())
    write_text(
        REPORT_DIR / "citation_mapping_from_retrieved_chunk.md",
        render_mapping_doc(mappings),
    )
    write_text(REPORT_DIR / "citation_prototype_examples.md", render_examples_doc(examples))
    write_text(REPORT_DIR / "citation_guard_delta.md", render_guard_delta_doc(deltas))

    validation_summary = write_validation_artifacts(
        DATA_DIR / "citation_prototype_examples.jsonl",
        RESULTS_DIR / "schema_validation_results.csv",
        REPORT_DIR / "schema_validation_report.md",
    )
    write_text(REPORT_DIR / "phase7q_summary.md", render_summary(validation_summary))
    return validation_summary


def main() -> int:
    summary = build_phase7q_artifacts()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["validation_status"] == "pass_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
