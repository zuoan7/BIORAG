#!/usr/bin/env python3
"""Build derived index unit preview from Phase7A table_objects.

The output is a preview for future index design only. It does not write Milvus,
build BM25, run embedding, or run retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_OBJECTS_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_objects.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_index_units.preview.jsonl"
TABLE_OBJECTS_PATH = DEFAULT_TABLE_OBJECTS_PATH
OUTPUT_PATH = DEFAULT_OUTPUT_PATH
PHASE_LABEL = "v7_phase7A_table_extraction_mvp"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def cells_by_row(obj: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for cell in obj.get("cells", []):
        result.setdefault(cell.get("row_id"), []).append(cell)
    return result


def columns_by_id(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {column.get("column_id"): column for column in obj.get("columns", [])}


def unit(
    unit_id: str,
    unit_type: str,
    obj: dict[str, Any],
    text: str,
    row_ids: list[str] | None = None,
    cell_ids: list[str] | None = None,
    source_span_ids: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "doc_id": obj.get("doc_id"),
        "table_object_id": obj.get("table_object_id"),
        "row_ids": row_ids or [],
        "cell_ids": cell_ids or [],
        "text": normalize(text),
        "metadata": {
            "table_id": obj.get("table_id"),
            "source_file": obj.get("source_file"),
            "page": obj.get("page"),
            "validation_status": obj.get("validation_status"),
            "caution": obj.get("validation_status") == "partial",
            "source_span_granularity": obj.get("source_span_granularity"),
            "source_span_limitation": obj.get("source_span_limitation"),
            "phase": PHASE_LABEL,
            "preview_only": True,
            "json_is_source_of_truth": True,
            "not_vectorized": True,
            "not_written_to_milvus_or_bm25": True,
        },
        "source_span_ids": source_span_ids or [],
        "warnings": warnings or [],
    }


def build_units(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if obj.get("validation_status") == "fail":
        return []
    records: list[dict[str, Any]] = []
    table_object_id = obj.get("table_object_id")
    columns = columns_by_id(obj)
    by_row = cells_by_row(obj)
    all_source_span_ids = [span.get("source_span_id") for span in obj.get("source_spans", []) if span.get("source_span_id")]

    column_names = ", ".join(normalize(column.get("header")) for column in obj.get("columns", [])[:12])
    records.append(
        unit(
            f"{table_object_id}__unit_table_summary",
            "table_summary",
            obj,
            f"{obj.get('doc_id')} {obj.get('table_id')} 的表格 caption 是：{obj.get('caption')}。主要列包括：{column_names}。validation_status={obj.get('validation_status')}。本对象的 source_span 粒度为 {obj.get('source_span_granularity')}；限制是 {obj.get('source_span_limitation')}。warnings 包括 {', '.join(obj.get('warnings') or [])}。",
            source_span_ids=all_source_span_ids[:12],
            warnings=obj.get("warnings") or [],
        )
    )
    records.append(
        unit(
            f"{table_object_id}__unit_caption_context",
            "caption_context",
            obj,
            f"{obj.get('doc_id')} {obj.get('table_id')} caption/context：{obj.get('caption')}。section_path：{' > '.join(obj.get('section_path') or [])}。",
            source_span_ids=all_source_span_ids[:6],
            warnings=obj.get("warnings") or [],
        )
    )

    for row in obj.get("rows", []):
        row_cells = by_row.get(row.get("row_id"), [])
        facts = []
        for cell in row_cells[:10]:
            col = columns.get(cell.get("column_id"), {})
            unit_text = f" {cell.get('unit')}" if cell.get("unit") else ""
            facts.append(f"{normalize(col.get('header'))} = {normalize(cell.get('value_raw'))}{unit_text}")
        records.append(
            unit(
                f"{row.get('row_id')}__unit_row_fact",
                "row_fact",
                obj,
                f"{obj.get('table_id')} 中 row {row.get('row_label') or row.get('row_index')} 的结构化值：{'; '.join(facts)}。",
                row_ids=[row.get("row_id")],
                cell_ids=[cell.get("cell_id") for cell in row_cells],
                source_span_ids=row.get("source_span_ids") or [],
                warnings=list({warning for cell in row_cells for warning in (cell.get('warnings') or [])}) + (row.get("warnings") or []),
            )
        )
        for cell in row_cells[:10]:
            col = columns.get(cell.get("column_id"), {})
            records.append(
                unit(
                    f"{cell.get('cell_id')}__unit_cell_fact",
                    "cell_fact",
                    obj,
                    f"{obj.get('table_id')} 中 row {row.get('row_label') or row.get('row_index')}、column {col.get('header')} 的 value_raw 是 {cell.get('value_raw')}。unit={cell.get('unit') or '未绑定'}；literal_marker={cell.get('literal_marker') or '无'}。",
                    row_ids=[row.get("row_id")],
                    cell_ids=[cell.get("cell_id")],
                    source_span_ids=cell.get("source_span_ids") or [],
                    warnings=cell.get("warnings") or [],
                )
            )
    return records


def configure_paths(table_objects_path: Path, output_path: Path, phase_label: str) -> None:
    global TABLE_OBJECTS_PATH, OUTPUT_PATH, PHASE_LABEL
    TABLE_OBJECTS_PATH = table_objects_path if table_objects_path.is_absolute() else ROOT / table_objects_path
    OUTPUT_PATH = output_path if output_path.is_absolute() else ROOT / output_path
    PHASE_LABEL = phase_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build derived natural-language table index unit preview.")
    parser.add_argument("--table-objects", type=Path, default=DEFAULT_TABLE_OBJECTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--phase-label", default="v7_phase7A_table_extraction_mvp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.table_objects, args.output, args.phase_label)
    objects = load_jsonl(TABLE_OBJECTS_PATH)
    records: list[dict[str, Any]] = []
    for obj in objects:
        records.extend(build_units(obj))
    write_jsonl(OUTPUT_PATH, records)
    print(json.dumps({"table_objects": len(objects), "index_units": len(records), "output": str(OUTPUT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
