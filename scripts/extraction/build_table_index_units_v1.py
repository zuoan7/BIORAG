#!/usr/bin/env python3
"""Build Phase7I table_index_unit_v1 preview artifacts from Phase7H formal seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPANDED_SEED_PATH = (
    ROOT / "data/experiments/v7_phase7_expanded_seed_from_human_review/expanded_table_seed.jsonl"
)
DEFAULT_FORMAL_VALIDATION_PATH = (
    ROOT / "results/v7_phase7_expanded_seed_validation/formal_seed_validation_results.csv"
)
DEFAULT_REVIEW_PACK_INDEX_PATH = (
    ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/review_pack_index.csv"
)
DEFAULT_CANDIDATE_POOL_PATH = (
    ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/candidate_pool_scored.csv"
)
DEFAULT_CSV_TABLES_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/csv_tables"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_table_index_unit_design"

TOP_LEVEL_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "caption",
    "content_text_for_embedding",
    "content_markdown",
    "metadata",
    "provenance",
    "guardrail",
]
UNIT_TYPES = ("table_unit", "row_unit", "cell_group_unit")
HEADER_STRUCTURE_TYPES = {
    "single_header",
    "multirow_header",
    "spanning_header",
    "uncertain_header",
}
CSV_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "row_index",
    "row_label",
    "content_text_for_embedding",
    "index_unit_status",
    "production_ready",
    "is_official_benchmark_seed",
    "value_bboxes_available",
    "cell_bboxes_available",
    "source_span_granularity",
]
STATS_FIELDS = [
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "table_unit_count",
    "row_unit_count",
    "cell_group_unit_count",
    "total_unit_count",
    "csv_row_count",
    "csv_data_row_count",
    "csv_header_row_count",
    "csv_column_count",
    "header_structure_type",
    "cell_group_skip_reason",
    "validation_status",
    "seed_warnings",
]
HEADER_TERMS = {
    "abundance",
    "atmosphere",
    "bacteria",
    "carbon",
    "cells",
    "condition",
    "control",
    "copy",
    "distribution",
    "donor",
    "energy",
    "enzyme",
    "fucose",
    "gene",
    "genbank",
    "genome",
    "genotype",
    "gos",
    "growth",
    "hits",
    "induction",
    "interaction",
    "lactose",
    "maintenance",
    "medium",
    "organism",
    "overall",
    "p-value",
    "pfam",
    "positive",
    "precursor",
    "primer",
    "product",
    "protein",
    "reference",
    "role",
    "sequence",
    "sequenced",
    "size",
    "source",
    "strain",
    "synthesis",
    "taxon",
    "topology",
    "vessel",
    "wild type",
}


def header_term_present(text: str) -> bool:
    for term in HEADER_TERMS:
        if term == "bacteria":
            if re.search(r"\bbacteria\b|bacterial", text):
                return True
            continue
        if term in text:
            return True
    return False


@dataclass(frozen=True)
class ParsedTable:
    rows: list[list[str]]
    header_rows: list[list[str]]
    data_rows: list[list[str]]
    column_headers: list[str]
    header_paths: list[list[str]]
    header_structure_type: str


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def split_semicolon(value: Any) -> list[str]:
    text = normalize(value)
    if not text or text == "none":
        return []
    return [item for item in text.split(";") if item]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def parse_row_index(value: str, fallback: int) -> int | str:
    text = normalize(value)
    return int(text) if text.isdigit() else fallback


def pad_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def read_csv_table(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return pad_rows([[normalize(cell) for cell in row] for row in csv.reader(handle)])


def data_cells(row: list[str]) -> list[str]:
    return row[1:] if row else []


def nonempty_cells(values: list[str]) -> list[str]:
    return [value for value in values if normalize(value)]


def numericish(value: str) -> bool:
    text = normalize(value).lower()
    if not text:
        return False
    return bool(re.search(r"(\d|±|−|<=|>=|<|>|\bnd\b|\+)", text))


def is_empty_data_row(row: list[str]) -> bool:
    return not nonempty_cells(data_cells(row))


def is_preheader_noise(values: list[str]) -> bool:
    filled = nonempty_cells(values)
    if len(filled) != 1:
        return False
    text = filled[0].strip().lower()
    return bool(re.fullmatch(r"[a-z]\*?|[a-z]\d?", text))


def is_title_noise(values: list[str]) -> bool:
    filled = nonempty_cells(values)
    if not filled:
        return True
    joined = " ".join(filled).lower()
    if re.search(r"\btable\s*\d", joined):
        return True
    if joined.startswith("ble ") and re.search(r"\d", joined):
        return True
    if "screening of" in joined and "abilities" in joined:
        return True
    return False


def looks_like_header(values: list[str], existing_header_rows: int) -> bool:
    filled = nonempty_cells(values)
    if not filled:
        return False
    lowered = " ".join(filled).lower()
    if header_term_present(lowered):
        return True
    numeric_count = sum(1 for value in filled if numericish(value))
    if existing_header_rows > 0 and values and not normalize(values[0]):
        if len(filled) >= 2 and numeric_count <= 1:
            return True
    if numeric_count == 0 and len(filled) >= 2:
        avg_len = sum(len(value) for value in filled) / len(filled)
        return avg_len <= 28
    return False


def split_header_and_data(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    body = rows[1:] if rows and rows[0] and rows[0][0].lower() == "row_index" else rows
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []
    started_data = False

    for row in body:
        values = data_cells(row)
        if is_empty_data_row(row):
            continue
        if not started_data:
            if is_preheader_noise(values) or is_title_noise(values):
                continue
            if not header_rows:
                header_rows.append(row)
                continue
            if looks_like_header(values, len(header_rows)) and len(header_rows) < 4:
                header_rows.append(row)
                continue
            started_data = True
        if started_data:
            data_rows.append(row)

    if not data_rows and header_rows:
        data_rows = header_rows[1:]
        header_rows = header_rows[:1]
    return header_rows, data_rows


def header_paths_from_rows(header_rows: list[list[str]], width: int) -> list[list[str]]:
    paths: list[list[str]] = [[] for _ in range(width)]
    for row in header_rows:
        values = data_cells(row)
        values = values + [""] * (width - len(values))
        last_seen = ""
        for index, raw_cell in enumerate(values[:width]):
            cell = normalize(raw_cell)
            if cell:
                last_seen = cell
            elif last_seen:
                cell = last_seen
            if cell and (not paths[index] or paths[index][-1] != cell):
                paths[index].append(cell)
    for index, path in enumerate(paths):
        if not path:
            path.append(f"col_{index + 1:03d}")
    return paths


def header_structure_type(header_rows: list[list[str]], header_paths: list[list[str]]) -> str:
    if not header_rows:
        return "uncertain_header"
    if len(header_rows) == 1:
        return "single_header"
    if any(len(path) > 1 for path in header_paths):
        return "spanning_header"
    return "multirow_header"


def parse_table(path: Path) -> ParsedTable:
    rows = read_csv_table(path)
    width = max((len(data_cells(row)) for row in rows), default=0)
    header_rows, data_rows = split_header_and_data(rows)
    header_paths = header_paths_from_rows(header_rows, width)
    structure_type = header_structure_type(header_rows, header_paths)
    if structure_type not in HEADER_STRUCTURE_TYPES:
        structure_type = "uncertain_header"
    return ParsedTable(
        rows=rows,
        header_rows=header_rows,
        data_rows=data_rows,
        column_headers=[" / ".join(path) for path in header_paths],
        header_paths=header_paths,
        header_structure_type=structure_type,
    )


def row_label(values: list[str]) -> str:
    for value in values:
        if normalize(value):
            return normalize(value)
    return "unlabeled_row"


def row_values_for(row: list[str], table: ParsedTable) -> list[dict[str, Any]]:
    values = data_cells(row)
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        value = normalize(value)
        if not value:
            continue
        header_path = table.header_paths[index] if index < len(table.header_paths) else [f"col_{index + 1:03d}"]
        result.append(
            {
                "column_index": index + 1,
                "column_header": " / ".join(header_path),
                "header_path": header_path,
                "value": value,
            }
        )
    return result


def table_shape(table: ParsedTable) -> dict[str, int]:
    return {
        "csv_rows": len(table.rows),
        "data_rows": len(table.data_rows),
        "columns": len(table.header_paths),
    }


def clean_caption(caption: str) -> str:
    return normalize(caption).replace("[TABLE CAPTION]", "").strip()


def summarize_caption(caption: str) -> str:
    text = clean_caption(caption)
    return text[:260] + ("..." if len(text) > 260 else "")


def common_provenance(seed: dict[str, Any], formal: dict[str, str], review: dict[str, str]) -> dict[str, Any]:
    source_csv_path = formal.get("csv_path") or seed.get("csv_path") or review.get("csv_path") or ""
    source_markdown_path = formal.get("markdown_path") or seed.get("markdown_path") or review.get("markdown_path") or ""
    source_pdf_crop_path = formal.get("pdf_crop_path") or seed.get("pdf_crop_path") or review.get("pdf_crop_path") or ""
    return {
        "source_csv_path": source_csv_path,
        "source_markdown_path": source_markdown_path,
        "source_pdf_crop_path": source_pdf_crop_path,
        "source_span_granularity": formal.get("source_span_granularity")
        or seed.get("source_span_granularity")
        or review.get("source_span_granularity")
        or "cell_level",
        "value_bboxes_available": False,
        "cell_bboxes_available": as_bool(
            formal.get("cell_bboxes_available")
            or seed.get("cell_bboxes_available")
            or review.get("cell_bboxes_available")
        ),
    }


def common_guardrail(seed: dict[str, Any], formal: dict[str, str]) -> dict[str, Any]:
    return {
        "seed_status": seed.get("seed_status") or "confirmed_seed_with_warnings",
        "binding_review_mode": formal.get("binding_review_mode") or seed.get("binding_review_mode") or "",
        "binding_review_limitation": formal.get("binding_review_limitation")
        or seed.get("binding_review_limitation")
        or "",
        "unit_or_note_ok": formal.get("unit_or_note_ok") or seed.get("unit_or_note_ok") or "",
        "reference_ok": formal.get("reference_ok") or seed.get("reference_ok") or "",
        "seed_warnings": split_semicolon(formal.get("warnings") or seed.get("seed_warnings")),
        "index_unit_status": "preview_only",
        "is_official_benchmark_seed": False,
        "production_ready": False,
    }


def make_unit(
    unit_id: str,
    unit_type: str,
    seed: dict[str, Any],
    formal: dict[str, str],
    review: dict[str, str],
    content_text: str,
    content_markdown: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "table_index_unit_id": unit_id,
        "unit_type": unit_type,
        "seed_id": seed.get("seed_id"),
        "candidate_id": seed.get("candidate_id"),
        "doc_id": seed.get("doc_id"),
        "table_id": seed.get("table_id"),
        "caption": seed.get("caption") or review.get("caption") or "",
        "content_text_for_embedding": normalize(content_text),
        "content_markdown": content_markdown.strip(),
        "metadata": metadata,
        "provenance": common_provenance(seed, formal, review),
        "guardrail": common_guardrail(seed, formal),
    }


def make_table_unit(
    ordinal: int,
    seed: dict[str, Any],
    formal: dict[str, str],
    review: dict[str, str],
    table: ParsedTable,
) -> dict[str, Any]:
    caption = seed.get("caption") or review.get("caption") or ""
    headers = table.column_headers[:12]
    warning_text = (
        "Binding review remains warning-level; source spans are not value-level; "
        "value-level coordinates are unavailable."
    )
    text = (
        f"In {seed.get('doc_id')} {seed.get('table_id')}, the table caption is: "
        f"{summarize_caption(caption)}. Table topic summary: {summarize_caption(caption)}. "
        f"Main column headers include: {', '.join(headers)}. {warning_text}"
    )
    markdown = "\n".join(
        [
            f"### table_unit: {seed.get('doc_id')} / {seed.get('table_id')}",
            "",
            f"- caption: {clean_caption(caption)}",
            f"- data rows: {len(table.data_rows)}",
            f"- columns: {len(table.header_paths)}",
            f"- header structure: `{table.header_structure_type}`",
            f"- warning limitation: {warning_text}",
        ]
    )
    metadata = {
        "page": seed.get("page") or review.get("page") or formal.get("page"),
        "column_headers": table.column_headers,
        "header_path": table.header_paths,
        "table_shape": table_shape(table),
        "header_structure_type": table.header_structure_type,
    }
    return make_unit(
        f"phase7i_preview_{ordinal:03d}__table_unit",
        "table_unit",
        seed,
        formal,
        review,
        text,
        markdown,
        metadata,
    )


def make_row_unit(
    ordinal: int,
    row_number: int,
    source_row: list[str],
    seed: dict[str, Any],
    formal: dict[str, str],
    review: dict[str, str],
    table: ParsedTable,
) -> dict[str, Any]:
    values = data_cells(source_row)
    label = row_label(values)
    row_values = row_values_for(source_row, table)
    facts = [f"{item['column_header']}={item['value']}" for item in row_values[:12]]
    row_index = parse_row_index(source_row[0] if source_row else "", row_number)
    warning_text = "Binding notes are warning-level only; value-level coordinates are not claimed."
    text = (
        f"In {seed.get('doc_id')} {seed.get('table_id')}, row \"{label}\" reports: "
        f"{'; '.join(facts)}. {warning_text}"
    )
    markdown = "\n".join(
        [
            f"### row_unit: row {row_index} / {label}",
            "",
            f"- context: `{seed.get('doc_id')}` / `{seed.get('table_id')}`",
            f"- values: {'; '.join(facts[:8])}",
            f"- warning limitation: {warning_text}",
        ]
    )
    metadata = {
        "page": seed.get("page") or review.get("page") or formal.get("page"),
        "row_index": row_index,
        "row_label": label,
        "column_headers": table.column_headers,
        "header_path": [item["header_path"] for item in row_values],
        "row_values": row_values,
        "table_shape": table_shape(table),
        "header_structure_type": table.header_structure_type,
    }
    return make_unit(
        f"phase7i_preview_{ordinal:03d}__row_unit_{row_number:03d}",
        "row_unit",
        seed,
        formal,
        review,
        text,
        markdown,
        metadata,
    )


def selected_cell_group_values(row_values: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in row_values
        if normalize(item.get("value")) and normalize(item.get("value")) != normalize(label)
    ]
    numeric_candidates = [item for item in candidates if numericish(item.get("value", ""))]
    if len(numeric_candidates) >= 3:
        return numeric_candidates[:8]
    if len(candidates) >= 4 and numeric_candidates:
        return candidates[:8]
    return []


def make_cell_group_unit(
    ordinal: int,
    row_number: int,
    group_number: int,
    seed: dict[str, Any],
    formal: dict[str, str],
    review: dict[str, str],
    table: ParsedTable,
    row_unit: dict[str, Any],
    selected_values: list[dict[str, Any]],
) -> dict[str, Any]:
    label = row_unit["metadata"]["row_label"]
    row_index = row_unit["metadata"]["row_index"]
    facts = [f"{item['column_header']}={item['value']}" for item in selected_values]
    warning_text = (
        "This is a row-level value group, not independent value-level evidence; "
        "value-level coordinates are not claimed."
    )
    text = (
        f"In {seed.get('doc_id')} {seed.get('table_id')}, row \"{label}\" has selected key values: "
        f"{'; '.join(facts)}. {warning_text}"
    )
    markdown = "\n".join(
        [
            f"### cell_group_unit: row {row_index} / {label}",
            "",
            f"- selected key values: {'; '.join(facts)}",
            f"- provenance limitation: {warning_text}",
        ]
    )
    metadata = {
        "page": seed.get("page") or review.get("page") or formal.get("page"),
        "row_index": row_index,
        "row_label": label,
        "column_headers": table.column_headers,
        "header_path": [item["header_path"] for item in selected_values],
        "cell_group_values": selected_values,
        "table_shape": table_shape(table),
        "header_structure_type": table.header_structure_type,
    }
    return make_unit(
        f"phase7i_preview_{ordinal:03d}__cell_group_unit_{row_number:03d}_{group_number:03d}",
        "cell_group_unit",
        seed,
        formal,
        review,
        text,
        markdown,
        metadata,
    )


def resolve_seed_csv_path(
    seed: dict[str, Any],
    formal: dict[str, str],
    review: dict[str, str],
    csv_tables_dir: Path,
) -> Path:
    candidates = [
        formal.get("csv_path"),
        seed.get("csv_path"),
        review.get("csv_path"),
        str(csv_tables_dir / f"{seed.get('candidate_id')}.csv"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve_path(Path(candidate))
        if path.exists():
            return path
    return resolve_path(Path(candidates[-1]))


def validation_status_for_seed(seed: dict[str, Any], formal: dict[str, str]) -> str:
    warnings = split_semicolon(formal.get("warnings") or seed.get("seed_warnings"))
    return "pass_with_warnings" if warnings else "pass"


def build_units_for_seed(
    ordinal: int,
    seed: dict[str, Any],
    formal: dict[str, str],
    review: dict[str, str],
    csv_tables_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = resolve_seed_csv_path(seed, formal, review, csv_tables_dir)
    table = parse_table(csv_path)
    units = [make_table_unit(ordinal, seed, formal, review, table)]
    row_units: list[dict[str, Any]] = []
    cell_group_units: list[dict[str, Any]] = []
    skipped_cell_groups = 0

    for row_number, source_row in enumerate(table.data_rows, start=1):
        unit = make_row_unit(ordinal, row_number, source_row, seed, formal, review, table)
        row_units.append(unit)
        units.append(unit)
        selected = selected_cell_group_values(unit["metadata"].get("row_values") or [], unit["metadata"]["row_label"])
        if selected:
            cell_group = make_cell_group_unit(
                ordinal,
                row_number,
                1,
                seed,
                formal,
                review,
                table,
                unit,
                selected,
            )
            cell_group_units.append(cell_group)
            units.append(cell_group)
        else:
            skipped_cell_groups += 1

    if not table.data_rows:
        skip_reason = "no_csv_data_rows"
    elif not cell_group_units:
        skip_reason = "no_rows_with_three_or_more_key_values"
    elif skipped_cell_groups:
        skip_reason = f"{skipped_cell_groups}_rows_without_key_value_group"
    else:
        skip_reason = "none"

    stats = {
        "seed_id": seed.get("seed_id"),
        "candidate_id": seed.get("candidate_id"),
        "doc_id": seed.get("doc_id"),
        "table_id": seed.get("table_id"),
        "table_unit_count": 1,
        "row_unit_count": len(row_units),
        "cell_group_unit_count": len(cell_group_units),
        "total_unit_count": len(units),
        "csv_row_count": len(table.rows),
        "csv_data_row_count": len(table.data_rows),
        "csv_header_row_count": len(table.header_rows),
        "csv_column_count": len(table.header_paths),
        "header_structure_type": table.header_structure_type,
        "cell_group_skip_reason": skip_reason,
        "validation_status": validation_status_for_seed(seed, formal),
        "seed_warnings": ";".join(split_semicolon(formal.get("warnings") or seed.get("seed_warnings"))),
    }
    return units, stats


def csv_row_for_unit(unit: dict[str, Any]) -> dict[str, Any]:
    metadata = unit.get("metadata") or {}
    provenance = unit.get("provenance") or {}
    guardrail = unit.get("guardrail") or {}
    return {
        "table_index_unit_id": unit.get("table_index_unit_id"),
        "unit_type": unit.get("unit_type"),
        "seed_id": unit.get("seed_id"),
        "candidate_id": unit.get("candidate_id"),
        "doc_id": unit.get("doc_id"),
        "table_id": unit.get("table_id"),
        "row_index": metadata.get("row_index", ""),
        "row_label": metadata.get("row_label", ""),
        "content_text_for_embedding": unit.get("content_text_for_embedding"),
        "index_unit_status": guardrail.get("index_unit_status"),
        "production_ready": str(guardrail.get("production_ready")).lower(),
        "is_official_benchmark_seed": str(guardrail.get("is_official_benchmark_seed")).lower(),
        "value_bboxes_available": str(provenance.get("value_bboxes_available")).lower(),
        "cell_bboxes_available": str(provenance.get("cell_bboxes_available")).lower(),
        "source_span_granularity": provenance.get("source_span_granularity"),
    }


def write_outputs(output_dir: Path, units: list[dict[str, Any]], stats: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "table_index_units.preview.jsonl", units)
    write_csv(output_dir / "table_index_units.preview.csv", [csv_row_for_unit(unit) for unit in units], CSV_FIELDS)
    write_jsonl(output_dir / "table_unit_preview.jsonl", [unit for unit in units if unit["unit_type"] == "table_unit"])
    write_jsonl(output_dir / "row_unit_preview.jsonl", [unit for unit in units if unit["unit_type"] == "row_unit"])
    write_jsonl(
        output_dir / "cell_group_unit_preview.jsonl",
        [unit for unit in units if unit["unit_type"] == "cell_group_unit"],
    )
    write_csv(output_dir / "table_index_unit_stats.csv", stats, STATS_FIELDS)


def build_table_index_units(
    expanded_seed_path: Path = DEFAULT_EXPANDED_SEED_PATH,
    formal_validation_path: Path = DEFAULT_FORMAL_VALIDATION_PATH,
    review_pack_index_path: Path = DEFAULT_REVIEW_PACK_INDEX_PATH,
    candidate_pool_path: Path = DEFAULT_CANDIDATE_POOL_PATH,
    csv_tables_dir: Path = DEFAULT_CSV_TABLES_DIR,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    expanded_seed_path = resolve_path(expanded_seed_path)
    formal_validation_path = resolve_path(formal_validation_path)
    review_pack_index_path = resolve_path(review_pack_index_path)
    candidate_pool_path = resolve_path(candidate_pool_path)
    csv_tables_dir = resolve_path(csv_tables_dir)
    output_dir = resolve_path(output_dir) if output_dir is not None else None

    formal_rows = load_csv(formal_validation_path)
    formal_by_seed = {row["seed_id"]: row for row in formal_rows}
    formal_order = [row["seed_id"] for row in formal_rows]
    review_by_candidate = {row["candidate_id"]: row for row in load_csv(review_pack_index_path)}
    pool_by_candidate = {row["candidate_id"]: row for row in load_csv(candidate_pool_path)}
    seed_by_id = {row["seed_id"]: row for row in load_jsonl(expanded_seed_path)}

    units: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for ordinal, seed_id in enumerate(formal_order, start=1):
        seed = seed_by_id[seed_id]
        if seed.get("seed_status") != "confirmed_seed_with_warnings":
            continue
        candidate_id = seed.get("candidate_id")
        review = {**pool_by_candidate.get(candidate_id, {}), **review_by_candidate.get(candidate_id, {})}
        seed_units, seed_stats = build_units_for_seed(
            ordinal,
            seed,
            formal_by_seed[seed_id],
            review,
            csv_tables_dir,
        )
        units.extend(seed_units)
        stats.append(seed_stats)

    if output_dir is not None:
        write_outputs(output_dir, units, stats)

    unit_counts = Counter(unit["unit_type"] for unit in units)
    return {
        "formal_seed_count": len(formal_rows),
        "preview_seed_count": len({unit["seed_id"] for unit in units}),
        "unit_count": len(units),
        "unit_type_counts": dict(unit_counts),
        "units": units,
        "stats": stats,
        "output_dir": rel(output_dir) if output_dir is not None else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase7I table_index_unit_v1 preview artifacts.")
    parser.add_argument("--expanded-seed", type=Path, default=DEFAULT_EXPANDED_SEED_PATH)
    parser.add_argument("--formal-validation", type=Path, default=DEFAULT_FORMAL_VALIDATION_PATH)
    parser.add_argument("--review-pack-index", type=Path, default=DEFAULT_REVIEW_PACK_INDEX_PATH)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL_PATH)
    parser.add_argument("--csv-tables-dir", type=Path, default=DEFAULT_CSV_TABLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_table_index_units(
        expanded_seed_path=args.expanded_seed,
        formal_validation_path=args.formal_validation,
        review_pack_index_path=args.review_pack_index,
        candidate_pool_path=args.candidate_pool,
        csv_tables_dir=args.csv_tables_dir,
        output_dir=args.output_dir,
    )
    summary = {
        "formal_seed_count": result["formal_seed_count"],
        "preview_seed_count": result["preview_seed_count"],
        "unit_count": result["unit_count"],
        "unit_type_counts": result["unit_type_counts"],
        "output_dir": result["output_dir"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
