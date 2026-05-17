#!/usr/bin/env python3
"""Render Phase7I table index unit preview as a compact Markdown review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units.preview.jsonl"
)
DEFAULT_STATS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_unit_stats.csv"
)
DEFAULT_OUTPUT_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units_review.md"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\n", " ").replace("|", "\\|").split())


def truncate(value: Any, limit: int = 360) -> str:
    text = md_escape(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def unit_counts(units: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "table_unit": sum(1 for unit in units if unit.get("unit_type") == "table_unit"),
        "row_unit": sum(1 for unit in units if unit.get("unit_type") == "row_unit"),
        "cell_group_unit": sum(1 for unit in units if unit.get("unit_type") == "cell_group_unit"),
    }


def first_unit(units: list[dict[str, Any]], unit_type: str) -> dict[str, Any] | None:
    for unit in units:
        if unit.get("unit_type") == unit_type:
            return unit
    return None


def provenance_limitation(unit: dict[str, Any]) -> str:
    provenance = unit.get("provenance") or {}
    return (
        f"source_span_granularity={provenance.get('source_span_granularity')}; "
        f"value_bboxes_available={str(provenance.get('value_bboxes_available')).lower()}; "
        f"cell_bboxes_available={str(provenance.get('cell_bboxes_available')).lower()}."
    )


def guardrail_warnings(unit: dict[str, Any]) -> str:
    guardrail = unit.get("guardrail") or {}
    warnings = guardrail.get("seed_warnings") or []
    return ";".join(warnings) if warnings else "none"


def render_seed(seed_id: str, units: list[dict[str, Any]], stats: dict[str, str]) -> list[str]:
    table_unit = first_unit(units, "table_unit") or units[0]
    row_unit = first_unit(units, "row_unit")
    cell_group_unit = first_unit(units, "cell_group_unit")
    counts = unit_counts(units)
    lines = [
        f"## {seed_id}",
        "",
        f"- seed_id：`{seed_id}`",
        f"- doc_id：`{table_unit.get('doc_id')}`",
        f"- table_id：`{table_unit.get('table_id')}`",
        f"- caption：{truncate(table_unit.get('caption'), 300)}",
        f"- table_unit：`{counts['table_unit']}`",
        f"- row_unit 数量：`{counts['row_unit']}`",
        f"- cell_group_unit 数量：`{counts['cell_group_unit']}`",
        f"- header_structure_type：`{stats.get('header_structure_type', '')}`",
        f"- provenance limitation：{provenance_limitation(table_unit)}",
        f"- guardrail warnings：`{guardrail_warnings(table_unit)}`",
        "",
        "### table_unit 摘要",
        "",
        truncate(table_unit.get("content_text_for_embedding"), 520),
        "",
        "### 示例 row_unit",
        "",
    ]
    if row_unit:
        lines.extend(
            [
                f"- unit_id：`{row_unit.get('table_index_unit_id')}`",
                f"- row_index：`{(row_unit.get('metadata') or {}).get('row_index')}`",
                f"- row_label：`{md_escape((row_unit.get('metadata') or {}).get('row_label'))}`",
                f"- text：{truncate(row_unit.get('content_text_for_embedding'), 520)}",
            ]
        )
    else:
        lines.append("_无 row_unit；CSV 无可用数据行或 validation 应标记 fail。_")

    lines.extend(["", "### 示例 cell_group_unit", ""])
    if cell_group_unit:
        metadata = cell_group_unit.get("metadata") or {}
        lines.extend(
            [
                f"- unit_id：`{cell_group_unit.get('table_index_unit_id')}`",
                f"- row_label：`{md_escape(metadata.get('row_label'))}`",
                f"- text：{truncate(cell_group_unit.get('content_text_for_embedding'), 520)}",
            ]
        )
    else:
        lines.append(f"_未生成 cell_group_unit；原因：`{stats.get('cell_group_skip_reason', 'unknown')}`。_")
    lines.append("")
    return lines


def render_review(units_path: Path, stats_path: Path, output_path: Path) -> dict[str, Any]:
    units_path = resolve_path(units_path)
    stats_path = resolve_path(stats_path)
    output_path = resolve_path(output_path)
    units = load_jsonl(units_path)
    stats_rows = load_csv(stats_path)
    stats_by_seed = {row["seed_id"]: row for row in stats_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[unit["seed_id"]].append(unit)

    lines = [
        "# Phase7I 表格索引单元审阅摘要",
        "",
        "本文件是 Phase7I preview units 的人工审阅摘要，不是 retrieval evaluation，不是 embedding 输出，也不是 production index。",
        "",
        f"- seed 数量：`{len(grouped)}`",
        f"- unit 总数：`{len(units)}`",
        "",
    ]
    for seed_id in stats_by_seed:
        lines.extend(render_seed(seed_id, grouped.get(seed_id, []), stats_by_seed[seed_id]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"seed_count": len(grouped), "unit_count": len(units), "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase7I table index unit review Markdown.")
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS_PATH)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_review(args.units, args.stats, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
