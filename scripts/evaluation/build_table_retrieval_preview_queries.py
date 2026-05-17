#!/usr/bin/env python3
"""Build Phase7J offline table retrieval preview queries."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELIGIBLE_JSONL = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
DEFAULT_ELIGIBLE_CSV = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_table_retrieval_wiring_preview"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_table_retrieval_wiring_preview"

QUERY_FIELDS = [
    "query_id",
    "query_text",
    "query_type",
    "expected_seed_id",
    "expected_doc_id",
    "expected_table_id",
    "expected_unit_type",
    "expected_table_index_unit_id",
    "expected_row_label",
    "expected_keywords",
    "query_source_unit_id",
    "query_notes",
]

EXPECTATION_FIELDS = [
    "query_id",
    "expected_seed_id",
    "expected_doc_id",
    "expected_table_id",
    "expected_unit_type",
    "expected_table_index_unit_id",
    "expected_row_label",
    "query_source_unit_id",
    "expectation_basis",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


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


def clean_caption(caption: str) -> str:
    text = normalize(caption).replace("[TABLE CAPTION]", "").strip()
    return re.sub(r"\s+", " ", text)


def compact_text(text: str, limit: int = 140) -> str:
    text = normalize(text)
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def path_to_text(path: Any) -> str:
    if isinstance(path, list):
        return " / ".join(normalize(item) for item in path if normalize(item))
    return normalize(path)


def value_items(unit: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = unit.get("metadata") or {}
    values = metadata.get("cell_group_values") or metadata.get("row_values") or []
    return [item for item in values if isinstance(item, dict)]


def header_texts(unit: dict[str, Any]) -> list[str]:
    metadata = unit.get("metadata") or {}
    values = value_items(unit)
    headers = [normalize(item.get("column_header")) for item in values]
    if not headers:
        headers = [path_to_text(path) for path in metadata.get("header_path") or []]
    return [header for header in headers if header]


def keyword_text(unit: dict[str, Any], extra: list[str] | None = None) -> str:
    metadata = unit.get("metadata") or {}
    parts = [
        unit.get("doc_id", ""),
        unit.get("table_id", ""),
        metadata.get("row_label", ""),
        clean_caption(unit.get("caption", "")),
    ]
    parts.extend(header_texts(unit)[:3])
    if extra:
        parts.extend(extra)
    keywords: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*|[\u4e00-\u9fff]+", normalize(part)):
            key = token.lower()
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            keywords.append(token)
            if len(keywords) >= 10:
                return ";".join(keywords)
    return ";".join(keywords)


def row_label(unit: dict[str, Any]) -> str:
    return normalize((unit.get("metadata") or {}).get("row_label"))


def first_metric_pair(unit: dict[str, Any]) -> tuple[str, str]:
    values = [
        item
        for item in value_items(unit)
        if normalize(item.get("column_header")) and normalize(item.get("value"))
    ]
    if not values:
        return "", ""
    first = values[0]
    second = values[1] if len(values) > 1 else values[0]
    return normalize(first.get("column_header")), normalize(second.get("column_header"))


def make_query(
    number: int,
    unit: dict[str, Any],
    query_text: str,
    query_type: str,
    notes: str,
    expected_unit_type: str | None = None,
    keyword_extra: list[str] | None = None,
) -> dict[str, Any]:
    metadata = unit.get("metadata") or {}
    return {
        "query_id": f"phase7j_query_{number:03d}",
        "query_text": normalize(query_text),
        "query_type": query_type,
        "expected_seed_id": unit.get("seed_id", ""),
        "expected_doc_id": unit.get("doc_id", ""),
        "expected_table_id": unit.get("table_id", ""),
        "expected_unit_type": expected_unit_type or unit.get("unit_type", ""),
        "expected_table_index_unit_id": unit.get("table_index_unit_id", ""),
        "expected_row_label": metadata.get("row_label", "") or "",
        "expected_keywords": keyword_text(unit, keyword_extra),
        "query_source_unit_id": unit.get("table_index_unit_id", ""),
        "query_notes": notes,
    }


def every_nth(units: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(units) <= limit:
        return units
    step = len(units) / limit
    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for index in range(limit):
        unit = units[int(index * step)]
        unit_id = unit["table_index_unit_id"]
        if unit_id not in used:
            picked.append(unit)
            used.add(unit_id)
    for unit in units:
        if len(picked) >= limit:
            break
        unit_id = unit["table_index_unit_id"]
        if unit_id not in used:
            picked.append(unit)
            used.add(unit_id)
    return picked


def build_queries(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    query_no = 1

    table_units = sorted(
        [unit for unit in units if unit.get("unit_type") == "table_unit"],
        key=lambda item: item.get("table_index_unit_id", ""),
    )
    for index, unit in enumerate(table_units[:12]):
        caption = compact_text(clean_caption(unit.get("caption", "")), 150)
        if index % 3 == 0:
            text = f"Find the table in {unit['doc_id']} {unit['table_id']} about {caption}"
            notes = "table_lookup，包含 doc_id/table_id，用于表级 wiring sanity。"
        else:
            text = f"Which table reports {caption}"
            notes = "table_lookup，不包含 doc_id，用于模拟更自然的表主题检索。"
        queries.append(make_query(query_no, unit, text, "table_lookup", notes, "table_unit"))
        query_no += 1

    row_units = [
        unit
        for unit in units
        if unit.get("unit_type") == "row_unit" and row_label(unit) and len(value_items(unit)) >= 2
    ]
    row_units = sorted(row_units, key=lambda item: (item.get("seed_id", ""), item.get("table_index_unit_id", "")))
    for index, unit in enumerate(every_nth(row_units, 12)):
        label = row_label(unit)
        metric, _ = first_metric_pair(unit)
        if index % 2 == 0:
            text = (
                f"Find the row evidence for {label} in {unit['doc_id']} {unit['table_id']} "
                f"covering {metric}"
            )
        else:
            text = f"Which table row reports {metric} for {label}"
        queries.append(
            make_query(
                query_no,
                unit,
                text,
                "row_lookup",
                "row_lookup，从 eligible row_unit 自动派生。",
                "row_unit",
                [label, metric],
            )
        )
        query_no += 1

    cell_units = [
        unit
        for unit in units
        if unit.get("unit_type") == "cell_group_unit" and row_label(unit) and len(value_items(unit)) >= 2
    ]
    cell_units = sorted(cell_units, key=lambda item: (item.get("seed_id", ""), item.get("table_index_unit_id", "")))
    for index, unit in enumerate(every_nth(cell_units, 8)):
        label = row_label(unit)
        metric_a, metric_b = first_metric_pair(unit)
        if index % 2 == 0:
            text = (
                f"Find metric evidence for {label}: {metric_a} and {metric_b} "
                f"in {unit['table_id']}"
            )
        else:
            text = f"What structured table evidence lists {metric_a} and {metric_b} for {label}"
        queries.append(
            make_query(
                query_no,
                unit,
                text,
                "metric_lookup",
                "metric_lookup，从 eligible cell_group_unit 自动派生。",
                "cell_group_unit",
                [label, metric_a, metric_b],
            )
        )
        query_no += 1

    reference_units = [
        unit
        for unit in row_units
        if any("reference" in normalize(item.get("column_header")).lower() for item in value_items(unit))
    ]
    for unit in every_nth(reference_units, 2):
        label = row_label(unit)
        text = f"Find the table row that gives the reference for {label} in {unit['table_id']}"
        queries.append(
            make_query(
                query_no,
                unit,
                text,
                "source_or_reference_lookup",
                "source_or_reference_lookup，使用 Reference 列生成。",
                "row_unit",
                [label, "Reference"],
            )
        )
        query_no += 1

    note_units = [
        unit
        for unit in units
        if row_label(unit)
        and (
            "warning-level" in normalize(unit.get("content_text_for_embedding")).lower()
            or "not claimed" in normalize(unit.get("content_text_for_embedding")).lower()
        )
    ]
    for unit in every_nth(note_units, 2):
        label = row_label(unit)
        text = (
            f"Find note-aware table evidence for {label} where value-level coordinates are not claimed"
        )
        queries.append(
            make_query(
                query_no,
                unit,
                text,
                "unit_or_note_lookup",
                "unit_or_note_lookup，用于检查 limitation/evidence 展示，不评价答案。",
                unit.get("unit_type", ""),
                [label, "value-level coordinates are not claimed"],
            )
        )
        query_no += 1

    return queries


def render_query_design_report(
    report_path: Path,
    units: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    eligible_jsonl: Path,
    eligible_csv: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    query_type_counts = Counter(query["query_type"] for query in queries)
    expected_type_counts = Counter(query["expected_unit_type"] for query in queries)
    without_doc_id = sum(1 for query in queries if query["expected_doc_id"] not in query["query_text"])
    lines = [
        "# Phase7J 查询预览集设计报告",
        "",
        "## 定位",
        "",
        "本报告说明 Phase7J offline table retrieval wiring preview 的 query preview set 构造方式。本轮只从 Phase7I-1 eligible units 自动派生检索预览查询，不构造用户答案，不调用模型，不运行 embedding、rerank、BM25 或 Milvus。",
        "",
        "## 输入",
        "",
        f"- eligible JSONL：`{rel(eligible_jsonl)}`",
        f"- eligible CSV：`{rel(eligible_csv)}`",
        f"- eligible unit 数量：`{len(units)}`",
        "",
        "## Query 构造原则",
        "",
        "- 从 274 个 Phase7I-1 eligible units 自动生成。",
        "- query_text 不包含内部 table_index_unit_id。",
        "- 每条 query 记录 expected seed/doc/table/unit_type/unit_id/row_label。",
        "- 保留一部分不含 doc_id 的 query，用于模拟真实检索入口。",
        "- 仅用于 lexical dry-run wiring sanity，不生成答案，不形成 benchmark。",
        "",
        "## Query 数量与分布",
        "",
        f"- query 总数：`{len(queries)}`",
        f"- 不含 doc_id 的 query 数量：`{without_doc_id}`",
        "",
        "| query_type | count |",
        "| --- | ---: |",
    ]
    for query_type, count in sorted(query_type_counts.items()):
        lines.append(f"| `{query_type}` | {count} |")
    lines.extend(["", "| expected_unit_type | count |", "| --- | ---: |"])
    for unit_type, count in sorted(expected_type_counts.items()):
        lines.append(f"| `{unit_type}` | {count} |")
    lines.extend(
        [
            "",
            "## 阶段限制",
            "",
            "- 本 query set 不是正式 retrieval evaluation。",
            "- 本 query set 不是 benchmark。",
            "- 本 query set 不比较 flat chunks。",
            "- 本 query set 不接 production，不读取 BM25，不访问 Milvus。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_query_preview_set(
    eligible_jsonl: Path = DEFAULT_ELIGIBLE_JSONL,
    eligible_csv: Path = DEFAULT_ELIGIBLE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    eligible_jsonl = resolve_path(eligible_jsonl)
    eligible_csv = resolve_path(eligible_csv)
    output_dir = resolve_path(output_dir)
    report_dir = resolve_path(report_dir)

    units = load_jsonl(eligible_jsonl)
    csv_rows = load_csv(eligible_csv)
    csv_ids = {row["table_index_unit_id"] for row in csv_rows}
    jsonl_ids = {unit["table_index_unit_id"] for unit in units}
    if len(units) != 274:
        raise ValueError(f"expected 274 eligible units, got {len(units)}")
    if csv_ids != jsonl_ids:
        raise ValueError("eligible JSONL and CSV unit ids do not match")

    queries = build_queries(units)
    if not 20 <= len(queries) <= 50:
        raise ValueError(f"query count must be 20-50, got {len(queries)}")
    if any(query["expected_table_index_unit_id"] in query["query_text"] for query in queries):
        raise ValueError("query_text must not contain internal table_index_unit_id")

    expectations = [
        {
            "query_id": query["query_id"],
            "expected_seed_id": query["expected_seed_id"],
            "expected_doc_id": query["expected_doc_id"],
            "expected_table_id": query["expected_table_id"],
            "expected_unit_type": query["expected_unit_type"],
            "expected_table_index_unit_id": query["expected_table_index_unit_id"],
            "expected_row_label": query["expected_row_label"],
            "query_source_unit_id": query["query_source_unit_id"],
            "expectation_basis": "auto_derived_from_phase7i1_eligible_unit",
        }
        for query in queries
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "query_set.preview.csv", queries, QUERY_FIELDS)
    write_jsonl(output_dir / "query_set.preview.jsonl", queries)
    write_csv(output_dir / "query_unit_expectations.csv", expectations, EXPECTATION_FIELDS)
    render_query_design_report(
        report_dir / "query_set_design_report.md",
        units,
        queries,
        eligible_jsonl,
        eligible_csv,
    )

    return {
        "eligible_unit_count": len(units),
        "query_count": len(queries),
        "query_type_distribution": dict(Counter(query["query_type"] for query in queries)),
        "expected_unit_type_distribution": dict(
            Counter(query["expected_unit_type"] for query in queries)
        ),
        "output_dir": rel(output_dir),
        "report_path": rel(report_dir / "query_set_design_report.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-jsonl", type=Path, default=DEFAULT_ELIGIBLE_JSONL)
    parser.add_argument("--eligible-csv", type=Path, default=DEFAULT_ELIGIBLE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_query_preview_set(
        eligible_jsonl=args.eligible_jsonl,
        eligible_csv=args.eligible_csv,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
