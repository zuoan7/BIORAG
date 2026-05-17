#!/usr/bin/env python3
"""Render Phase7A table_objects JSONL as a Markdown review view."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_OBJECTS_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_objects.jsonl"
DEFAULT_VALIDATION_CSV_PATH = ROOT / "reports/v7_phase7_table_extraction_mvp/table_object_validation_summary.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_objects_review.md"
TABLE_OBJECTS_PATH = DEFAULT_TABLE_OBJECTS_PATH
VALIDATION_CSV_PATH = DEFAULT_VALIDATION_CSV_PATH
OUTPUT_PATH = DEFAULT_OUTPUT_PATH
PHASE_LABEL = "Phase7A"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_validation(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["table_object_id"]: row for row in csv.DictReader(handle)}


def cell_map(obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for cell in obj.get("cells", []):
        result[(cell.get("row_id"), cell.get("column_id"))] = cell
    return result


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return " ".join(text.split())


def render_preview_table(obj: dict[str, Any], max_rows: int = 8, max_cols: int = 8) -> list[str]:
    columns = obj.get("columns", [])[:max_cols]
    rows = obj.get("rows", [])[:max_rows]
    if not columns or not rows:
        return ["_无法生成表格预览：columns 或 rows 为空。_"]

    lookup = cell_map(obj)
    header = ["row"] + [md_escape(col.get("header")) for col in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        row_values = [md_escape(row.get("row_label") or row.get("row_id"))]
        for col in columns:
            cell = lookup.get((row.get("row_id"), col.get("column_id")))
            row_values.append(md_escape(cell.get("value_raw") if cell else ""))
        lines.append("| " + " | ".join(row_values) + " |")
    if len(obj.get("rows", [])) > max_rows or len(obj.get("columns", [])) > max_cols:
        lines.append("")
        lines.append(f"_预览已截断：显示前 {max_rows} 行、前 {max_cols} 列。_")
    return lines


def render_object(obj: dict[str, Any], validation: dict[str, str]) -> list[str]:
    status = validation.get("validation_status") or obj.get("validation_status") or "unknown"
    warnings = obj.get("warnings") or []
    blocking = obj.get("blocking_warnings") or []
    nonblocking = obj.get("nonblocking_warnings") or []
    if validation.get("blocking_warnings") and validation["blocking_warnings"] != "none":
        blocking = validation["blocking_warnings"].split(";")
    if validation.get("nonblocking_warnings") and validation["nonblocking_warnings"] != "none":
        nonblocking = validation["nonblocking_warnings"].split(";")
    lines = [
        f"## {obj.get('table_object_id')}",
        "",
        f"- doc_id：`{obj.get('doc_id')}`",
        f"- source_file：`{obj.get('source_file')}`",
        f"- table_id：`{obj.get('table_id')}`",
        f"- page：`{obj.get('page')}`",
        f"- validation_status：`{status}`",
        f"- blocking_warnings：`{', '.join(blocking) if blocking else 'none'}`",
        f"- nonblocking_warnings：`{', '.join(nonblocking) if nonblocking else 'none'}`",
        f"- candidate_status：`{obj.get('candidate_status')}`",
        f"- boundary_status：`{obj.get('boundary_status')}`",
        f"- merge_status：`{obj.get('merge_status')}`",
        f"- source_span_granularity：`{obj.get('source_span_granularity')}`",
        f"- source_span_limitation：{md_escape(obj.get('source_span_limitation'))}",
        f"- extraction_method：`{obj.get('extraction_method')}`",
        f"- extraction_confidence：`{obj.get('extraction_confidence')}`",
        "",
        "### Caption",
        "",
        md_escape(obj.get("caption")) or "_无 caption_",
        "",
        "### Block IDs",
        "",
        f"- caption_block_ids：`{', '.join(obj.get('caption_block_ids') or [])}`",
        f"- header_block_ids：`{', '.join(obj.get('header_block_ids') or [])}`",
        f"- body_block_ids：`{', '.join(obj.get('body_block_ids') or [])}`",
        f"- source_block_ids：`{', '.join(obj.get('source_block_ids') or [])}`",
        f"- chunk_ids：`{', '.join(obj.get('chunk_ids') or [])}`",
        "",
        "### Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- 无")
    lines.extend(["", "### Markdown Table Preview", ""])
    lines.extend(render_preview_table(obj))
    lines.extend(
        [
            "",
            "### Source Span Limitation",
            "",
            "本轮 source_span 来自 official chunks 的 chunk/block/page/text preview。若粒度为 `table_row_level` 或 `row_level`，只能说明行级或 block 级可追溯；本轮不提供也不伪造 value-level bbox。",
            "",
            "### Review Notes",
            "",
            f"- candidate_status_reason：{md_escape(obj.get('candidate_status_reason')) or '无'}",
            f"- continued_parts：{md_escape(obj.get('continued_parts')) or '无'}",
            f"- merged_from_table_object_ids：`{', '.join(obj.get('merged_from_table_object_ids') or []) or 'none'}`",
            "",
            "### Notes",
            "",
        ]
    )
    notes = obj.get("notes") or []
    if validation.get("notes"):
        notes.append(f"validation: {validation['notes']}")
    if notes:
        lines.extend(f"- {md_escape(note)}" for note in notes)
    else:
        lines.append("- 无")
    lines.append("")
    return lines


def configure_paths(
    table_objects_path: Path,
    validation_csv_path: Path,
    output_path: Path,
    phase_label: str,
) -> None:
    global TABLE_OBJECTS_PATH, VALIDATION_CSV_PATH, OUTPUT_PATH, PHASE_LABEL
    TABLE_OBJECTS_PATH = table_objects_path if table_objects_path.is_absolute() else ROOT / table_objects_path
    VALIDATION_CSV_PATH = validation_csv_path if validation_csv_path.is_absolute() else ROOT / validation_csv_path
    OUTPUT_PATH = output_path if output_path.is_absolute() else ROOT / output_path
    PHASE_LABEL = phase_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render table_objects JSONL as Markdown review cards.")
    parser.add_argument("--table-objects", type=Path, default=DEFAULT_TABLE_OBJECTS_PATH)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--phase-label", default="Phase7A")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.table_objects, args.validation_csv, args.output, args.phase_label)
    objects = load_jsonl(TABLE_OBJECTS_PATH)
    validations = load_validation(VALIDATION_CSV_PATH)
    lines = [
        f"# {PHASE_LABEL} table_objects 审阅视图",
        "",
        "本文件是从 `table_objects.jsonl` 派生的人工审阅视图，不是主结构化格式。JSONL 才是本轮 source of truth。",
        "",
        "本轮不直接 embedding 完整 JSON；后续如需索引，应从 table_object 派生 table/row/cell/caption_context 等自然语言 index units。",
        "",
    ]
    for obj in objects:
        lines.extend(render_object(obj, validations.get(obj.get("table_object_id"), {})))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"table_objects": len(objects), "output": str(OUTPUT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
