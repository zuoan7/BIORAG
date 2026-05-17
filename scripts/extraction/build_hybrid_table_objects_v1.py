#!/usr/bin/env python3
"""Build hybrid table_object_v1 artifacts from chunk and pdfplumber tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_TABLE_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_objects.jsonl"
)
DEFAULT_PDFPLUMBER_RAW_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/pdfplumber_tables.raw.jsonl"
)
DEFAULT_ALIGNMENT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/chunk_pdfplumber_alignment.csv"
)
DEFAULT_SCHEMA_PATH = ROOT / "schemas/table_object_v1.yaml"
DEFAULT_OUTPUT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/hybrid_table_objects.jsonl"
)

SOURCE_SPAN_LIMITATION = (
    "pdfplumber cell bbox only supports cell-level layout provenance; value-level token bbox is absent "
    "and is not inferred."
)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
HYBRID_V2_RULE_FIX_WARNINGS = {
    "split_cell_warning",
    "merged_cell_warning",
    "row_continuation_warning",
    "column_alignment_inconsistent",
    "cell_grid_needs_rule_fix",
    "metric_level_cell_gap",
    "numeric_column_order_uncertain",
    "missing_metric_cell_warning",
    "metric_column_group_uncertain",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def compact_id(value: str) -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def parse_json_value(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_alignment(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_has_header_signal(row: list[str]) -> bool:
    text = normalize_space(" ".join(row))
    non_empty = [cell for cell in row if cell]
    if len(non_empty) < 2:
        return False
    numeric_count = len(NUMERIC_RE.findall(text))
    return numeric_count <= max(1, len(non_empty) // 2)


def cell_bbox_available(pdf: dict[str, Any] | None) -> bool:
    return bool(pdf and pdf.get("cell_bboxes_available") and pdf.get("cell_bboxes"))


def pdf_layout_status(pdf: dict[str, Any] | None) -> str:
    if not pdf:
        return "not_evaluable"
    if pdf.get("layout_quality_status"):
        return pdf["layout_quality_status"]
    warnings = set(pdf.get("extraction_warnings") or [])
    if "suspected_false_positive_layout" in warnings:
        return "likely_false_positive"
    if pdf.get("row_count", 0) == 0 or pdf.get("column_count", 0) == 0:
        return "failed"
    if pdf.get("extraction_confidence") == "high" and pdf.get("cell_bboxes_available"):
        return "usable"
    return "weak"


def stable_cell_bbox_available(pdf: dict[str, Any] | None, alignment: dict[str, str]) -> bool:
    if not cell_bbox_available(pdf):
        return False
    if alignment.get("alignment_status") != "matched":
        return False
    if alignment.get("alignment_confidence") not in {"high", "medium"}:
        return False
    if pdf_layout_status(pdf) != "usable":
        return False
    if float((pdf or {}).get("cell_bbox_coverage") or 0.0) < 0.65:
        return False
    return True


def pdf_rows_usable(pdf: dict[str, Any] | None, alignment: dict[str, str]) -> bool:
    if not pdf:
        return False
    if alignment.get("alignment_status") != "matched":
        return False
    if alignment.get("alignment_confidence") not in {"high", "medium"}:
        return False
    if pdf_layout_status(pdf) != "usable":
        return False
    if pdf.get("row_count", 0) < 2 or pdf.get("column_count", 0) < 2:
        return False
    if pdf.get("non_empty_cell_count", 0) < 2:
        return False
    return True


def normalized_source_span_granularity(value: str | None) -> str:
    if value in {"table_level", "table_row_level", "row_level", "cell_level", "mixed_or_unclear"}:
        return value
    if value == "value_level":
        return "mixed_or_unclear"
    return value or "table_row_level"


def normalize_hybrid_v2_source_span_granularity(value: str | None) -> str:
    """Keep Phase7D v2 from promoting absent token bboxes to value-level provenance."""

    return normalized_source_span_granularity(value)


def make_hybrid_id(chunk_obj: dict[str, Any]) -> str:
    original = chunk_obj.get("table_object_id") or ""
    if "__phase7b2_" in original:
        return original.replace("__phase7b2_", "__phase7c2_hybrid_")
    return f"{chunk_obj.get('doc_id')}__{compact_id(chunk_obj.get('table_id') or 'table')}__phase7c2_hybrid"


def make_columns(hybrid_id: str, rows: list[list[str]], has_header: bool) -> list[dict[str, Any]]:
    column_count = max((len(row) for row in rows), default=0)
    header_row = rows[0] if has_header and rows else []
    columns = []
    for col_index in range(1, column_count + 1):
        header = header_row[col_index - 1] if col_index - 1 < len(header_row) else ""
        columns.append(
            {
                "column_id": f"{hybrid_id}__col_{col_index:03d}",
                "column_index": col_index,
                "header": header or f"pdfplumber_col_{col_index:03d}",
                "unit": None,
                "header_path": [header] if header else [],
                "source_span_ids": [],
                "warnings": [] if header else ["pdfplumber_header_empty_or_inferred"],
            }
        )
    return columns


def make_rows_cells_source_spans(
    hybrid_id: str,
    chunk_obj: dict[str, Any],
    pdf: dict[str, Any],
    use_cell_level: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pdf_rows = pdf.get("rows") or []
    has_header = row_has_header_signal(pdf_rows[0]) if pdf_rows else False
    data_rows = pdf_rows[1:] if has_header else pdf_rows
    columns = make_columns(hybrid_id, pdf_rows, has_header)
    pdf_cells = {
        (cell.get("row_index"), cell.get("column_index")): cell for cell in (pdf.get("cells") or [])
    }
    offset = 1 if has_header else 0
    source_spans: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    first_chunk_id = (chunk_obj.get("chunk_ids") or [""])[0]
    block_id = f"pdfplumber:{pdf.get('pdfplumber_table_id')}"
    for row_index, row in enumerate(data_rows, start=1):
        row_id = f"{hybrid_id}__row_{row_index:03d}"
        row_text = normalize_space(" | ".join(row))
        row_span_ids: list[str] = []
        rows.append(
            {
                "row_id": row_id,
                "row_index": row_index,
                "row_label": next((cell for cell in row if cell), None),
                "row_text": row_text,
                "source_span_ids": row_span_ids,
                "warnings": [],
            }
        )
        for col_index, column in enumerate(columns, start=1):
            pdf_row_index = row_index + offset
            pdf_cell = pdf_cells.get((pdf_row_index, col_index), {})
            value = normalize_space(
                pdf_cell.get("text")
                if pdf_cell
                else (row[col_index - 1] if col_index - 1 < len(row) else "")
            )
            span_id = f"{hybrid_id}__span_r{row_index:03d}c{col_index:03d}"
            bbox = pdf_cell.get("bbox") if pdf_cell else None
            span_granularity = "cell_level" if use_cell_level and bbox else "table_row_level"
            source_span = {
                "source_span_id": span_id,
                "doc_id": chunk_obj.get("doc_id"),
                "chunk_id": first_chunk_id,
                "block_id": block_id,
                "page": pdf.get("page_number"),
                "span_text": value,
                "granularity": span_granularity,
                "bbox": bbox if use_cell_level and bbox else None,
            }
            source_spans.append(source_span)
            row_span_ids.append(span_id)
            cells.append(
                {
                    "cell_id": f"{hybrid_id}__cell_{len(cells) + 1:04d}",
                    "row_id": row_id,
                    "column_id": column["column_id"],
                    "value_raw": value,
                    "value_normalized": None,
                    "unit": None,
                    "literal_marker": None,
                    "footnote_refs": [],
                    "reference_refs": [],
                    "source_span_ids": [span_id],
                    "warnings": [],
                    "cell_bbox": bbox if use_cell_level and bbox else None,
                    "cell_bbox_source": "pdfplumber_cell" if use_cell_level and bbox else None,
                    "no_value_level_bbox": True,
                }
            )
    return columns, rows, cells, source_spans


def warning_union(*warning_lists: list[str]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for warning_list in warning_lists:
        for warning in warning_list or []:
            if warning not in seen:
                seen.add(warning)
                warnings.append(warning)
    return warnings


def build_hybrid_metadata(
    chunk_obj: dict[str, Any],
    pdf: dict[str, Any] | None,
    alignment: dict[str, str],
    extraction_method: str,
    source_span_granularity: str,
) -> dict[str, Any]:
    pdf_table_id = alignment.get("pdfplumber_table_id") or None
    return {
        "extraction_method": extraction_method,
        "original_chunk_table_object_id": chunk_obj.get("table_object_id"),
        "pdfplumber_table_id": pdf_table_id,
        "alignment_status": alignment.get("alignment_status") or "not_evaluable",
        "alignment_confidence": alignment.get("alignment_confidence") or "none",
        "pdf_page": (pdf or {}).get("page_number"),
        "pdf_table_bbox": (pdf or {}).get("bbox") or parse_json_value(alignment.get("pdf_table_bbox", "")),
        "pdfplumber_strategy": (pdf or {}).get("strategy") or alignment.get("pdf_strategy") or "unknown",
        "cell_bboxes_available": cell_bbox_available(pdf),
        "value_bboxes_available": False,
        "source_span_granularity": source_span_granularity,
        "source_span_limitation": SOURCE_SPAN_LIMITATION,
    }


def build_hybrid_object(
    chunk_obj: dict[str, Any],
    pdf: dict[str, Any] | None,
    alignment: dict[str, str],
) -> dict[str, Any]:
    hybrid_id = make_hybrid_id(chunk_obj)
    use_pdf_rows = pdf_rows_usable(pdf, alignment)
    use_cell_level = stable_cell_bbox_available(pdf, alignment) if use_pdf_rows else False
    extraction_method = "hybrid_pdfplumber_chunk" if use_pdf_rows else "chunk_fallback"
    source_span_granularity = (
        "cell_level"
        if use_pdf_rows and use_cell_level
        else normalized_source_span_granularity(chunk_obj.get("source_span_granularity"))
    )

    obj = dict(chunk_obj)
    obj["table_object_id"] = hybrid_id
    obj["phase"] = "v7_phase7C_2_pdfplumber_pilot"
    obj["schema_name"] = "table_object_v1"
    obj["schema_version"] = "v1"
    obj["extraction_method"] = extraction_method
    obj["source_span_granularity"] = source_span_granularity
    obj["source_span_limitation"] = SOURCE_SPAN_LIMITATION
    obj["no_value_level_bbox"] = True
    obj["cell_bboxes_available"] = cell_bbox_available(pdf)
    obj["value_bboxes_available"] = False
    obj["validation_status"] = "manual_review"

    if use_pdf_rows and pdf:
        columns, rows, cells, source_spans = make_rows_cells_source_spans(
            hybrid_id, chunk_obj, pdf, use_cell_level
        )
        obj["columns"] = columns
        obj["rows"] = rows
        obj["cells"] = cells
        obj["source_spans"] = source_spans

    hybrid_warnings = ["value_level_bbox_absent", "cell_bbox_not_value_bbox"]
    if not use_pdf_rows:
        hybrid_warnings.append("hybrid_used_chunk_fallback")
    if alignment.get("alignment_status") == "page_only_match":
        hybrid_warnings.append("page_only_alignment_manual_review")
    if alignment.get("alignment_confidence") == "low":
        hybrid_warnings.append("pdfplumber_alignment_low_confidence")
    if alignment.get("alignment_status") in {"conflict", "multiple_pdf_tables", "no_pdf_table_found"}:
        hybrid_warnings.append(f"pdfplumber_alignment_{alignment.get('alignment_status')}")
    if pdf and pdf_layout_status(pdf) != "usable":
        hybrid_warnings.append("pdfplumber_low_layout_quality")
    if cell_bbox_available(pdf):
        hybrid_warnings.append("hybrid_cell_bbox_available")
    else:
        hybrid_warnings.append("pdfplumber_cell_bbox_missing")

    obj["warnings"] = warning_union(list(chunk_obj.get("warnings") or []), hybrid_warnings)
    obj["hybrid_metadata"] = build_hybrid_metadata(
        chunk_obj,
        pdf,
        alignment,
        extraction_method,
        source_span_granularity,
    )
    obj["notes"] = list(chunk_obj.get("notes") or []) + [
        "Phase7C-2 hybrid pilot: detailed alignment/layout diagnostics are stored in sidecars.",
        "pdfplumber cell bbox is layout-cell provenance only; value-level bbox is absent.",
    ]
    return obj


def run(args: argparse.Namespace) -> None:
    if not args.schema.exists():
        raise SystemExit(f"schema missing: {args.schema}")
    chunk_objects = load_jsonl(args.chunk_table_objects)
    pdf_tables = load_jsonl(args.pdfplumber_raw)
    alignments = load_alignment(args.alignment)
    chunks_by_id = {obj.get("table_object_id"): obj for obj in chunk_objects}
    pdf_by_id = {pdf.get("pdfplumber_table_id"): pdf for pdf in pdf_tables}
    hybrids: list[dict[str, Any]] = []
    for alignment in alignments:
        chunk_obj = chunks_by_id.get(alignment.get("chunk_table_object_id"))
        if not chunk_obj:
            continue
        pdf = pdf_by_id.get(alignment.get("pdfplumber_table_id", ""))
        hybrids.append(build_hybrid_object(chunk_obj, pdf, alignment))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for obj in hybrids:
            handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "hybrid_table_objects": len(hybrids),
                "output": rel(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid table_object_v1 JSONL.")
    parser.add_argument("--chunk-table-objects", type=Path, default=DEFAULT_CHUNK_TABLE_OBJECTS_PATH)
    parser.add_argument("--pdfplumber-raw", type=Path, default=DEFAULT_PDFPLUMBER_RAW_PATH)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    args.chunk_table_objects = resolve_path(args.chunk_table_objects)
    args.pdfplumber_raw = resolve_path(args.pdfplumber_raw)
    args.alignment = resolve_path(args.alignment)
    args.schema = resolve_path(args.schema)
    args.output = resolve_path(args.output)
    return args


if __name__ == "__main__":
    run(parse_args())
