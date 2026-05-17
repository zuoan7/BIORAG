#!/usr/bin/env python3
"""Render hybrid table_objects as a Markdown review view."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYBRID_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/hybrid_table_objects.jsonl"
)
DEFAULT_VALIDATION_CSV_PATH = (
    ROOT / "reports/v7_phase7_pdfplumber_pilot/hybrid_table_object_validation_summary.csv"
)
DEFAULT_OUTPUT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/hybrid_table_objects_review.md"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_validation(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["hybrid_table_object_id"]: row for row in csv.DictReader(handle)}


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return " ".join(text.split())


def metadata(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("hybrid_metadata") or {}


def bool_string(value: Any) -> str:
    return str(bool(value)).lower()


def cell_map(obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in obj.get("cells") or []:
        result[(cell.get("row_id"), cell.get("column_id"))] = cell
    return result


def render_preview_table(obj: dict[str, Any], max_rows: int = 8, max_cols: int = 8) -> list[str]:
    columns = (obj.get("columns") or [])[:max_cols]
    rows = (obj.get("rows") or [])[:max_rows]
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
    if len(obj.get("rows") or []) > max_rows or len(obj.get("columns") or []) > max_cols:
        lines.extend(["", f"_预览已截断：显示前 {max_rows} 行、前 {max_cols} 列。_"])
    return lines


def render_object(obj: dict[str, Any], validation: dict[str, str]) -> list[str]:
    meta = metadata(obj)
    status = validation.get("hybrid_validation_status") or obj.get("validation_status") or "unknown"
    warnings = obj.get("warnings") or []
    source_span_granularity = (
        validation.get("source_span_granularity")
        or meta.get("source_span_granularity")
        or obj.get("source_span_granularity")
    )
    cell_bboxes_available = validation.get("cell_bboxes_available") or bool_string(
        meta.get("cell_bboxes_available")
    )
    value_bboxes_available = validation.get("value_bboxes_available") or bool_string(
        meta.get("value_bboxes_available")
    )
    lines = [
        f"## {obj.get('table_object_id')}",
        "",
        f"- hybrid_table_object_id：`{obj.get('table_object_id')}`",
        f"- original_chunk_table_object_id：`{validation.get('original_chunk_table_object_id') or meta.get('original_chunk_table_object_id')}`",
        f"- pdfplumber_table_id：`{validation.get('pdfplumber_table_id') or meta.get('pdfplumber_table_id')}`",
        f"- doc_id：`{obj.get('doc_id')}`",
        f"- table_id：`{obj.get('table_id')}`",
        f"- hybrid_validation_status：`{status}`",
        f"- primary_failure_stage：`{validation.get('primary_failure_stage') or 'unknown'}`",
        f"- manual_review_reason：`{validation.get('manual_review_reason') or 'unknown'}`",
        f"- recommended_next_action：`{validation.get('recommended_next_action') or 'unknown'}`",
        f"- alignment_status：`{validation.get('alignment_status') or meta.get('alignment_status')}`",
        f"- alignment_confidence：`{validation.get('alignment_confidence') or meta.get('alignment_confidence')}`",
        f"- layout_quality_status：`{validation.get('layout_quality_status') or 'unknown'}`",
        f"- extraction_method：`{validation.get('extraction_method') or meta.get('extraction_method') or obj.get('extraction_method')}`",
        f"- source_span_granularity：`{source_span_granularity}`",
        f"- cell_bboxes_available：`{cell_bboxes_available}`",
        f"- value_bboxes_available：`{value_bboxes_available}`",
        f"- source_span_limitation：`{meta.get('source_span_limitation') or obj.get('source_span_limitation')}`",
        f"- blocking_warnings：`{validation.get('blocking_warnings') or 'none'}`",
        f"- nonblocking_warnings：`{validation.get('nonblocking_warnings') or 'none'}`",
        "",
        "### Caption",
        "",
        md_escape(obj.get("caption")) or "_无 caption_",
        "",
        "### Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- 无")
    lines.extend(["", "### Table Preview", ""])
    lines.extend(render_preview_table(obj))
    lines.extend(
        [
            "",
            "### Validation Notes",
            "",
            validation.get("notes") or "无。",
            "",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> None:
    objects = load_jsonl(args.hybrid_objects)
    validations = load_validation(args.validation_csv)
    lines = [
        "# Phase7C-2 hybrid table_objects 审阅视图",
        "",
        "本文件是从 `hybrid_table_objects.jsonl` 和 validation CSV 派生的人工审阅视图，不是主结构化格式。JSONL 才是本轮 source of truth。",
        "",
        "本轮不直接 embedding 完整 JSON，不写 Milvus，不建 BM25，不接 ingestion。pdfplumber cell bbox 不是 value-level bbox。",
        "",
    ]
    for obj in objects:
        lines.extend(render_object(obj, validations.get(obj.get("table_object_id"), {})))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"hybrid_table_objects": len(objects), "output": rel(args.output)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render hybrid table_objects review Markdown.")
    parser.add_argument("--hybrid-objects", type=Path, default=DEFAULT_HYBRID_OBJECTS_PATH)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    args.hybrid_objects = resolve_path(args.hybrid_objects)
    args.validation_csv = resolve_path(args.validation_csv)
    args.output = resolve_path(args.output)
    return args


if __name__ == "__main__":
    run(parse_args())
