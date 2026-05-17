#!/usr/bin/env python3
"""Validate Phase7I table_index_unit_v1 preview artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units.preview.jsonl"
)
DEFAULT_STATS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_unit_stats.csv"
)
DEFAULT_FORMAL_VALIDATION_PATH = (
    ROOT / "results/v7_phase7_expanded_seed_validation/formal_seed_validation_results.csv"
)
DEFAULT_REPORT_PATH = (
    ROOT / "reports/v7_phase7_table_index_unit_design/table_index_unit_validation_report.md"
)

TOP_LEVEL_ALLOWED_FIELDS = {
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
}
REQUIRED_FIELDS = TOP_LEVEL_ALLOWED_FIELDS
UNIT_TYPES = {"table_unit", "row_unit", "cell_group_unit"}
FORBIDDEN_VALUE_BBOX_KEYS = {
    "value_bbox",
    "value_bboxes",
    "value_level_bbox",
    "value_level_bboxes",
    "bbox",
    "bboxes",
}
ALLOWED_BBOX_KEYS = {"value_bboxes_available", "cell_bboxes_available"}
FORBIDDEN_INDEX_KEYS = {
    "embedding",
    "embedding_vector",
    "vector",
    "bm25_id",
    "bm25_score",
    "milvus_id",
    "retrieval_score",
    "rerank_score",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
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


def add_check(checks: list[dict[str, str]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def grouped_by_seed(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[unit.get("seed_id", "")].append(unit)
    return grouped


def count_by_type(units: list[dict[str, Any]], unit_type: str) -> int:
    return sum(1 for unit in units if unit.get("unit_type") == unit_type)


def recursive_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(recursive_keys(nested))
    elif isinstance(value, list):
        for item in value:
            keys.extend(recursive_keys(item))
    return keys


def has_value_bbox_claim(unit: dict[str, Any]) -> bool:
    for key in recursive_keys(unit):
        if key in ALLOWED_BBOX_KEYS:
            continue
        if key in FORBIDDEN_VALUE_BBOX_KEYS:
            return True
    provenance = unit.get("provenance") or {}
    if provenance.get("value_bboxes_available") is not False:
        return True
    if provenance.get("source_span_granularity") == "value_level":
        return True
    return False


def has_forbidden_index_field(unit: dict[str, Any]) -> bool:
    return any(key in FORBIDDEN_INDEX_KEYS for key in recursive_keys(unit))


def validation_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "pass_with_warnings" in statuses:
        return "pass_with_warnings"
    return "pass"


def validate_units(
    units: list[dict[str, Any]],
    stats_rows: list[dict[str, str]],
    formal_rows: list[dict[str, str]],
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    formal_ids = {row["seed_id"] for row in formal_rows}
    unit_seed_ids = {unit.get("seed_id") for unit in units}
    grouped = grouped_by_seed(units)

    if len(formal_ids) == 15 and unit_seed_ids == formal_ids:
        add_check(checks, "formal_seed_scope", "pass", "preview units 只包含 Phase7H formal 15 条 seed。")
    else:
        add_check(
            checks,
            "formal_seed_scope",
            "fail",
            f"formal={len(formal_ids)}；unit_seed={len(unit_seed_ids)}；差异={sorted((formal_ids ^ unit_seed_ids))[:5]}",
        )

    table_failures = [
        seed_id for seed_id in formal_ids if count_by_type(grouped.get(seed_id, []), "table_unit") != 1
    ]
    add_check(
        checks,
        "one_table_unit_per_seed",
        "pass" if not table_failures else "fail",
        "每条 seed 有且仅有 1 个 table_unit。" if not table_failures else f"异常 seed={table_failures}",
    )

    row_failures = [seed_id for seed_id in formal_ids if count_by_type(grouped.get(seed_id, []), "row_unit") < 1]
    add_check(
        checks,
        "at_least_one_row_unit_per_seed",
        "pass" if not row_failures else "fail",
        "每条 seed 至少有 1 个 row_unit。" if not row_failures else f"缺少 row_unit seed={row_failures}",
    )

    zero_cell_group = [
        row["seed_id"] for row in stats_rows if row.get("cell_group_unit_count") in {"", "0"}
    ]
    if zero_cell_group:
        add_check(
            checks,
            "cell_group_unit_count",
            "pass_with_warnings",
            f"cell_group_unit 允许为 0；未生成的 seed 数量={len(zero_cell_group)}。",
        )
    else:
        add_check(checks, "cell_group_unit_count", "pass", "所有 seed 均生成至少 1 个 cell_group_unit。")

    missing_ids = [unit for unit in units if not unit.get("table_index_unit_id")]
    add_check(
        checks,
        "table_index_unit_id_present",
        "pass" if not missing_ids else "fail",
        "所有 unit 都有 table_index_unit_id。" if not missing_ids else f"缺失数量={len(missing_ids)}",
    )

    invalid_types = [unit.get("table_index_unit_id") for unit in units if unit.get("unit_type") not in UNIT_TYPES]
    add_check(
        checks,
        "unit_type_present_and_valid",
        "pass" if not invalid_types else "fail",
        "所有 unit_type 均有效。" if not invalid_types else f"异常 unit={invalid_types[:5]}",
    )

    empty_content = [
        unit.get("table_index_unit_id") for unit in units if not str(unit.get("content_text_for_embedding", "")).strip()
    ]
    add_check(
        checks,
        "content_text_for_embedding_nonempty",
        "pass" if not empty_content else "fail",
        "所有 content_text_for_embedding 非空。" if not empty_content else f"空文本 unit={empty_content[:5]}",
    )

    missing_subobjects = [
        unit.get("table_index_unit_id")
        for unit in units
        if not isinstance(unit.get("metadata"), dict)
        or not isinstance(unit.get("provenance"), dict)
        or not isinstance(unit.get("guardrail"), dict)
    ]
    add_check(
        checks,
        "metadata_provenance_guardrail_present",
        "pass" if not missing_subobjects else "fail",
        "所有 unit 均包含 metadata / provenance / guardrail。"
        if not missing_subobjects
        else f"缺失子对象 unit={missing_subobjects[:5]}",
    )

    extra_fields = [
        unit.get("table_index_unit_id")
        for unit in units
        if set(unit.keys()) - TOP_LEVEL_ALLOWED_FIELDS
    ]
    missing_required = [
        unit.get("table_index_unit_id")
        for unit in units
        if REQUIRED_FIELDS - set(unit.keys())
    ]
    add_check(
        checks,
        "top_level_schema_minimal",
        "pass" if not extra_fields and not missing_required else "fail",
        "顶层字段保持在 schema 核心字段内。"
        if not extra_fields and not missing_required
        else f"extra={extra_fields[:3]} missing={missing_required[:3]}",
    )

    value_bbox_bad = [unit.get("table_index_unit_id") for unit in units if has_value_bbox_claim(unit)]
    add_check(
        checks,
        "no_value_level_bbox_claim",
        "pass" if not value_bbox_bad else "fail",
        "未出现 value-level bbox claim，且 value_bboxes_available=false。"
        if not value_bbox_bad
        else f"异常 unit={value_bbox_bad[:5]}",
    )

    bad_preview_status = [
        unit.get("table_index_unit_id")
        for unit in units
        if (unit.get("guardrail") or {}).get("index_unit_status") != "preview_only"
    ]
    add_check(
        checks,
        "index_unit_status_preview_only",
        "pass" if not bad_preview_status else "fail",
        "所有 unit 均为 preview_only。" if not bad_preview_status else f"异常 unit={bad_preview_status[:5]}",
    )

    production_ready = [
        unit.get("table_index_unit_id")
        for unit in units
        if (unit.get("guardrail") or {}).get("production_ready") is not False
    ]
    add_check(
        checks,
        "production_ready_false",
        "pass" if not production_ready else "fail",
        "所有 unit 的 production_ready=false。" if not production_ready else f"异常 unit={production_ready[:5]}",
    )

    official_seed = [
        unit.get("table_index_unit_id")
        for unit in units
        if (unit.get("guardrail") or {}).get("is_official_benchmark_seed") is not False
    ]
    add_check(
        checks,
        "official_benchmark_seed_false",
        "pass" if not official_seed else "fail",
        "所有 unit 的 is_official_benchmark_seed=false。" if not official_seed else f"异常 unit={official_seed[:5]}",
    )

    bad_seed_status = [
        unit.get("table_index_unit_id")
        for unit in units
        if (unit.get("guardrail") or {}).get("seed_status") != "confirmed_seed_with_warnings"
    ]
    add_check(
        checks,
        "partial_reject_unreviewed_excluded",
        "pass" if not bad_seed_status else "fail",
        "partial / reject / unreviewed 未进入 preview units。"
        if not bad_seed_status
        else f"异常 unit={bad_seed_status[:5]}",
    )

    forbidden_index_units = [unit.get("table_index_unit_id") for unit in units if has_forbidden_index_field(unit)]
    add_check(
        checks,
        "no_embedding_retrieval_fields",
        "pass" if not forbidden_index_units else "fail",
        "未生成 embedding/vector/retrieval/rerank 字段。"
        if not forbidden_index_units
        else f"异常 unit={forbidden_index_units[:5]}",
    )

    serialized = json.dumps(units, ensure_ascii=False).lower()
    if "bm25_index.json" in serialized or "pymilvus" in serialized or "milvusclient" in serialized:
        add_check(checks, "no_bm25_or_milvus_access", "fail", "unit payload 出现 BM25/Milvus 访问痕迹。")
    else:
        add_check(checks, "no_bm25_or_milvus_access", "pass", "未出现 BM25/Milvus 访问或索引字段。")

    inherited_warnings = sum(
        1 for unit in units if (unit.get("guardrail") or {}).get("seed_warnings")
    )
    add_check(
        checks,
        "seed_warnings_inherited",
        "pass_with_warnings" if inherited_warnings else "pass",
        f"继承 warning 的 unit 数量={inherited_warnings}。",
    )

    overall = validation_status(checks)
    return {
        "validation_status": overall,
        "checks": checks,
        "unit_type_counts": dict(Counter(unit.get("unit_type") for unit in units)),
        "seed_count": len(grouped),
        "unit_count": len(units),
    }


def render_report(result: dict[str, Any], stats_rows: list[dict[str, str]], report_path: Path) -> None:
    status_counts = Counter(check["status"] for check in result["checks"])
    lines = [
        "# 表格索引单元验证报告",
        "",
        "## 1. validation_status",
        "",
        f"- overall_validation_status：`{result['validation_status']}`",
        f"- seed 数量：`{result['seed_count']}`",
        f"- unit 总数：`{result['unit_count']}`",
        f"- table_unit：`{result['unit_type_counts'].get('table_unit', 0)}`",
        f"- row_unit：`{result['unit_type_counts'].get('row_unit', 0)}`",
        f"- cell_group_unit：`{result['unit_type_counts'].get('cell_group_unit', 0)}`",
        "",
        "## 2. validation_status 统计",
        "",
    ]
    for status in ["pass", "pass_with_warnings", "fail"]:
        lines.append(f"- `{status}`：{status_counts.get(status, 0)}")
    lines.extend(
        [
            "",
            "## 3. 检查项",
            "",
            "| check | status | detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in result["checks"]:
        detail = check["detail"].replace("|", "\\|")
        lines.append(f"| `{check['name']}` | `{check['status']}` | {detail} |")

    lines.extend(
        [
            "",
            "## 4. 每条 seed 的 unit 统计",
            "",
            "| seed_id | table_unit | row_unit | cell_group_unit | validation_status | cell_group_skip_reason |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in stats_rows:
        lines.append(
            "| `{seed_id}` | {table_unit_count} | {row_unit_count} | {cell_group_unit_count} | `{validation_status}` | `{cell_group_skip_reason}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 5. Guardrail 结论",
            "",
            "- 本轮只使用 Phase7H formal 15 条 seed。",
            "- 所有 unit 均为 `preview_only`。",
            "- 所有 unit 均保持 `production_ready=false` 与 `is_official_benchmark_seed=false`。",
            "- 所有 provenance 均保持 `value_bboxes_available=false`；cell bbox 未被写成 value bbox。",
            "- 本轮未读取或查询 BM25 index，未访问 Milvus，未运行 embedding、rerank 或 retrieval。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def validate_table_index_units(
    units_path: Path = DEFAULT_UNITS_PATH,
    stats_path: Path = DEFAULT_STATS_PATH,
    formal_validation_path: Path = DEFAULT_FORMAL_VALIDATION_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    units_path = resolve_path(units_path)
    stats_path = resolve_path(stats_path)
    formal_validation_path = resolve_path(formal_validation_path)
    report_path = resolve_path(report_path)
    units = load_jsonl(units_path)
    stats_rows = load_csv(stats_path)
    formal_rows = load_csv(formal_validation_path)
    result = validate_units(units, stats_rows, formal_rows)
    render_report(result, stats_rows, report_path)
    result["report"] = rel(report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase7I table_index_unit_v1 preview artifacts.")
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS_PATH)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--formal-validation", type=Path, default=DEFAULT_FORMAL_VALIDATION_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_table_index_units(
        units_path=args.units,
        stats_path=args.stats,
        formal_validation_path=args.formal_validation,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "validation_status": result["validation_status"],
                "seed_count": result["seed_count"],
                "unit_count": result["unit_count"],
                "unit_type_counts": result["unit_type_counts"],
                "report": result["report"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
