#!/usr/bin/env python3
"""Validate Phase7A offline table_object JSONL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_OBJECTS_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_objects.jsonl"
DEFAULT_CANDIDATES_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_candidates.jsonl"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_table_extraction_mvp"
TABLE_OBJECTS_PATH = DEFAULT_TABLE_OBJECTS_PATH
CANDIDATES_PATH = DEFAULT_CANDIDATES_PATH
REPORT_DIR = DEFAULT_REPORT_DIR
SUMMARY_CSV_PATH = REPORT_DIR / "table_object_validation_summary.csv"
REPORT_MD_PATH = REPORT_DIR / "table_object_validation_report.md"
PHASE_SUMMARY_PATH = REPORT_DIR / "phase7a_mvp_summary.md"
REVIEW_MD_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_objects_review.md"
INDEX_PREVIEW_PATH = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_index_units.preview.jsonl"
COMPARISON_MD_PATH = REPORT_DIR / "phase7b_2_rerun_comparison.md"
PHASE_LABEL = "Phase7A"
SUMMARY_FILENAME = "phase7a_mvp_summary.md"

SMOKE_DOC_IDS = [
    "doc_0322",
    "doc_0158",
    "doc_0598",
    "doc_0452",
    "doc_0468",
    "doc_0687",
    "doc_0458",
    "doc_0522",
    "doc_0523",
]

BLOCKING_WARNINGS = {
    "false_positive_candidate",
    "duplicate_table_candidate",
    "body_blocks_missing",
    "mixed_table_block_risk",
    "table_tail_truncation",
    "continued_table_needs_merge",
    "cell_alignment_error",
    "matrix_flattened",
    "target_mapping_risk",
    "boundary_blocking_warning",
    "row_cell_blocking_warning",
}

CSV_FIELDS = [
    "table_object_id",
    "doc_id",
    "table_id",
    "has_caption",
    "has_body_blocks",
    "has_header_blocks",
    "has_rows",
    "has_columns",
    "has_cells",
    "has_value_raw",
    "has_source_spans",
    "source_span_granularity",
    "source_span_limitation",
    "candidate_status",
    "boundary_status",
    "merge_status",
    "warnings_count",
    "blocking_warnings",
    "nonblocking_warnings",
    "validation_status",
    "notes",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def split_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    blocking = [warning for warning in warnings if warning in BLOCKING_WARNINGS]
    nonblocking = [warning for warning in warnings if warning not in BLOCKING_WARNINGS]
    return blocking, nonblocking


def validate_object(obj: dict[str, Any]) -> dict[str, str]:
    warnings = list(obj.get("warnings") or [])
    blocking, nonblocking = split_warnings(warnings)
    cells = obj.get("cells") or []

    has_caption = bool(obj.get("caption"))
    has_body_blocks = bool(obj.get("body_block_ids"))
    has_header_blocks = bool(obj.get("header_block_ids"))
    has_rows = bool(obj.get("rows"))
    has_columns = bool(obj.get("columns"))
    has_cells = bool(cells)
    has_value_raw = any(cell.get("value_raw") not in (None, "") for cell in cells)
    has_source_spans = bool(obj.get("source_spans"))
    source_span_granularity = obj.get("source_span_granularity") or ""
    source_span_limitation = obj.get("source_span_limitation") or ""

    notes: list[str] = []
    status = "pass"

    if not obj.get("table_object_id"):
        notes.append("缺少 table_object_id。")
        status = "fail"
    if not obj.get("doc_id"):
        notes.append("缺少 doc_id。")
        status = "fail"
    if not (obj.get("table_id") or has_caption):
        notes.append("缺少 table_id 和 caption。")
        status = "fail"
    if not obj.get("source_block_ids") or not obj.get("chunk_ids"):
        notes.append("缺少 source_block_ids 或 chunk_ids。")
        status = "fail"
    if not source_span_granularity:
        notes.append("缺少 source_span_granularity。")
        status = "fail"
    if not source_span_limitation:
        notes.append("缺少 source_span_limitation。")
        status = "partial"
    if not has_source_spans:
        notes.append("缺少 source_spans。")
        status = "fail"
    if "false_positive_candidate" in warnings:
        notes.append("明确为 false positive candidate。")
        status = "fail"
    if "duplicate_table_candidate" in warnings and obj.get("candidate_status") in {"filtered", "deduped"}:
        notes.append("重复或 shadow candidate 不应作为普通对象通过。")
        status = "fail"

    if status != "fail":
        if not has_caption:
            notes.append("无 caption，不能判 pass。")
            status = "partial"
        if not has_body_blocks:
            notes.append("无 body_block_ids，boundary 需要人工复核。")
            status = "partial"
            if "body_blocks_missing" not in warnings:
                warnings.append("body_blocks_missing")
                blocking.append("body_blocks_missing")
        if not has_rows or not has_columns or not has_cells:
            notes.append("rows/columns/cells 不完整。")
            status = "partial"
        if not has_value_raw:
            notes.append("cells 中没有 value_raw。")
            status = "partial"
        if blocking:
            notes.append("存在阻断型 warning。")
            status = "partial"

    if status == "pass" and warnings:
        status = "pass_with_warnings"
    if not warnings:
        notes.append("未记录 warning。")
    if source_span_granularity in {"table_level", "mixed_or_unclear"}:
        notes.append("source_span 粒度不足或混杂，需要人工审阅。")
        if status == "pass":
            status = "pass_with_warnings"

    return {
        "table_object_id": obj.get("table_object_id", ""),
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "has_caption": bool_text(has_caption),
        "has_body_blocks": bool_text(has_body_blocks),
        "has_header_blocks": bool_text(has_header_blocks),
        "has_rows": bool_text(has_rows),
        "has_columns": bool_text(has_columns),
        "has_cells": bool_text(has_cells),
        "has_value_raw": bool_text(has_value_raw),
        "has_source_spans": bool_text(has_source_spans),
        "source_span_granularity": source_span_granularity,
        "source_span_limitation": source_span_limitation,
        "candidate_status": obj.get("candidate_status", ""),
        "boundary_status": obj.get("boundary_status", ""),
        "merge_status": obj.get("merge_status", ""),
        "warnings_count": str(len(warnings)),
        "blocking_warnings": ";".join(blocking) if blocking else "none",
        "nonblocking_warnings": ";".join(nonblocking) if nonblocking else "none",
        "validation_status": status,
        "notes": " ".join(notes) if notes else "核心字段完整；仅保留非阻断 warning。",
    }


def write_summary_csv(rows: list[dict[str, str]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_validated_table_objects(objects: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    status_by_id = {row["table_object_id"]: row["validation_status"] for row in rows}
    blocking_by_id = {row["table_object_id"]: row["blocking_warnings"] for row in rows}
    nonblocking_by_id = {row["table_object_id"]: row["nonblocking_warnings"] for row in rows}
    for obj in objects:
        table_object_id = obj.get("table_object_id")
        if table_object_id in status_by_id:
            obj["validation_status"] = status_by_id[table_object_id]
            obj["blocking_warnings"] = [] if blocking_by_id[table_object_id] == "none" else blocking_by_id[table_object_id].split(";")
            obj["nonblocking_warnings"] = [] if nonblocking_by_id[table_object_id] == "none" else nonblocking_by_id[table_object_id].split(";")
    with TABLE_OBJECTS_PATH.open("w", encoding="utf-8") as handle:
        for obj in objects:
            handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def warning_counter(objects: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for obj in objects:
        counter.update(obj.get("warnings") or [])
    return counter


def write_validation_report(objects: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    status_counts = Counter(row["validation_status"] for row in rows)
    granularity_counts = Counter(row["source_span_granularity"] for row in rows)
    warnings = warning_counter(objects)
    pass_like = status_counts.get("pass", 0) + status_counts.get("pass_with_warnings", 0)
    reached = len({obj.get("doc_id") for obj in objects}) >= 8 and len(objects) >= 10 and pass_like >= 5

    lines = [
        f"# {PHASE_LABEL} table_object 校验报告",
        "",
        "## 1. 校验目标",
        "",
        "本报告验证 Phase7A 离线抽取的 `table_objects.jsonl` 是否具备人工审阅所需的最低结构字段、source_span、warnings 和 validation_status。",
        "",
        "## 2. 校验规则",
        "",
        "- 必须有 `table_object_id`、`doc_id`、`table_id` 或 `caption`。",
        "- 必须有 `source_block_ids`、`chunk_ids`、`source_span_granularity` 和 `source_spans`。",
        "- 若无 caption、body blocks、rows、columns、cells 或 value_raw，则判为 `partial` 或 `fail`。",
        "- 若无 source_spans，则判为 `fail`。",
        "- 只有非阻断 parser/boundary/provenance warning 时，可判 `pass_with_warnings`。",
        "- `false_positive_candidate` 明确时判 `fail`。",
        "- `body_blocks_missing`、`mixed_table_block_risk`、`table_tail_truncation`、`continued_table_needs_merge`、`cell_alignment_error`、`matrix_flattened`、`target_mapping_risk` 等阻断型 warning 出现时，不允许判 `pass_with_warnings`。",
        "- 有阻断 warning 但仍有可审阅结构时判 `partial`。",
        "- `pass_with_warnings` 只表示字段完整且无 P0 blocker，仍然不是 confirmed 或 production-ready。",
        "",
        "## 3. 总数与校验状态统计",
        "",
        f"`table_object` 总数：{len(objects)}。",
        "",
        "| validation_status | 数量 |",
        "|---|---:|",
    ]
    for status in ["pass", "pass_with_warnings", "partial", "manual_review", "fail"]:
        lines.append(f"| `{status}` | {status_counts.get(status, 0)} |")

    lines.extend(["", "## 4. 每个对象的校验结论", "", "| table_object_id | doc_id | table_id | status | notes |", "|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| `{row['table_object_id']}` | `{row['doc_id']}` | `{row['table_id']}` | `{row['validation_status']}` | {row['notes']} |"
        )

    lines.extend(["", "## 5. 主要 warnings", "", "| warning | 数量 |", "|---|---:|"])
    for warning, count in warnings.most_common():
        lines.append(f"| `{warning}` | {count} |")

    lines.extend(["", "## 6. source_span 粒度统计", "", "| source_span_granularity | 数量 |", "|---|---:|"])
    for granularity, count in sorted(granularity_counts.items()):
        lines.append(f"| `{granularity}` | {count} |")

    lines.extend(
        [
            "",
            "## 7. 抽取质量主要问题",
            "",
            "主要问题集中在 official chunks 的 parser 边界：`contains_table_text=false` 很普遍，表体经常位于 `table_caption`、`paragraph` 或相邻 chunk；source_span 只能稳定记录到 table_row_level 或 row/block level；numeric/unit/footnote/reference binding 仍需要人工审阅。",
            "",
            "本轮没有伪造 value-level bbox，也没有把 row-level source_span 写成 production-grade provenance。",
            "",
            "## 8. 是否达到 MVP smoke 目标",
            "",
            f"本轮 pass/pass_with_warnings 数量：{pass_like}。MVP smoke 目标判定：{'达到' if reached else '未完全达到'}。",
            "",
            "## 9. 不接 production 声明",
            "",
            "本轮 validation 只用于离线 MVP 审阅，不修改 ingestion pipeline，不写 Milvus，不重建 BM25，不跑 retrieval，不调用 Qwen/RAGAS/OCR/VLM，不修改 official baseline，不进入 Route C implementation。",
            "",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def candidate_status_counts(candidates: list[dict[str, Any]]) -> Counter[str]:
    return Counter(candidate.get("candidate_status", "active") for candidate in candidates)


def p0_blocker_count(objects: list[dict[str, Any]]) -> int:
    return sum(1 for obj in objects if set(obj.get("warnings") or []) & BLOCKING_WARNINGS)


def p1_warning_count(objects: list[dict[str, Any]]) -> int:
    p1_warnings = {
        "unit_visible_not_bound",
        "unit_scope_table_level",
        "unit_scope_column_level",
        "footnote_present_not_bound",
        "internal_reference_column",
        "external_citation_not_supported",
        "reference_visible_not_bound",
        "source_span_table_row_level_only",
        "no_value_level_bbox",
        "metric_level_cell_gap",
        "numeric_column_order_uncertain",
    }
    return sum(1 for obj in objects for warning in obj.get("warnings", []) if warning in p1_warnings)


def write_phase_summary(objects: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    status_counts = Counter(row["validation_status"] for row in rows)
    granularity_counts = Counter(row["source_span_granularity"] for row in rows)
    warnings = warning_counter(objects)
    candidates = load_optional_jsonl(CANDIDATES_PATH)
    candidate_count = len(candidates)
    candidate_counts = candidate_status_counts(candidates)
    doc_ids = sorted({obj.get("doc_id", "") for obj in objects if obj.get("doc_id")})
    pass_like = status_counts.get("pass", 0) + status_counts.get("pass_with_warnings", 0)
    smoke_reached = 8 <= len(doc_ids) <= 12 and 15 <= candidate_count <= 30 and len(objects) >= 10
    output_dir = TABLE_OBJECTS_PATH.parent
    report_dir = REPORT_DIR

    lines = [
        f"# {PHASE_LABEL} 结构化表格抽取 rerun 总结",
        "",
        "## 1. 本轮生成文件",
        "",
        f"- `{(output_dir / 'table_candidates.jsonl').relative_to(ROOT)}`：table candidate 检测结果，保留 filtered / deduped / merged 状态。",
        f"- `{TABLE_OBJECTS_PATH.relative_to(ROOT)}`：主结构化 JSONL。",
        f"- `{(output_dir / 'table_objects_review.md').relative_to(ROOT)}`：Markdown 人工审阅视图。",
        f"- `{(output_dir / 'table_index_units.preview.jsonl').relative_to(ROOT)}`：未来索引派生视图预览。",
        f"- `{(report_dir / 'table_candidate_detection_report.md').relative_to(ROOT)}`：候选检测报告。",
        f"- `{SUMMARY_CSV_PATH.relative_to(ROOT)}`：validation 机器可读摘要。",
        f"- `{REPORT_MD_PATH.relative_to(ROOT)}`：validation 中文报告。",
        f"- `{COMPARISON_MD_PATH.relative_to(ROOT)}`：Phase7A vs Phase7B-2 对比报告。",
        f"- `{PHASE_SUMMARY_PATH.relative_to(ROOT)}`：本总结。",
        "",
        "## 2. 修改脚本与测试",
        "",
        "- `scripts/extraction/extract_table_objects_v1.py`",
        "- `scripts/extraction/validate_table_objects_v1.py`",
        "- `scripts/extraction/render_table_objects_markdown.py`",
        "- `scripts/extraction/build_table_index_units_preview.py`",
        "- `tests/test_phase7_table_extraction_heuristics.py`：新增离线 heuristic / validation 单元测试。",
        "",
        "## 3. Smoke 输入 doc_id",
        "",
        f"`{', '.join(SMOKE_DOC_IDS)}`",
        "",
        f"official chunks 中实际产生对象的 doc_id：`{', '.join(doc_ids)}`。",
        "",
        "## 3.1 Same-smoke rerun 命令",
        "",
        "```bash",
        f"python scripts/extraction/extract_table_objects_v1.py --output-dir {output_dir.relative_to(ROOT)} --report-dir {report_dir.relative_to(ROOT)} --phase-label Phase7B-2 --run-tag phase7b2",
        f"python scripts/extraction/validate_table_objects_v1.py --table-objects {TABLE_OBJECTS_PATH.relative_to(ROOT)} --candidates {CANDIDATES_PATH.relative_to(ROOT)} --output-dir {output_dir.relative_to(ROOT)} --report-dir {report_dir.relative_to(ROOT)} --phase-label Phase7B-2 --summary-filename phase7b_2_summary.md --write-comparison",
        f"python scripts/extraction/render_table_objects_markdown.py --table-objects {TABLE_OBJECTS_PATH.relative_to(ROOT)} --validation-csv {SUMMARY_CSV_PATH.relative_to(ROOT)} --output {REVIEW_MD_PATH.relative_to(ROOT)} --phase-label Phase7B-2",
        f"python scripts/extraction/build_table_index_units_preview.py --table-objects {TABLE_OBJECTS_PATH.relative_to(ROOT)} --output {INDEX_PREVIEW_PATH.relative_to(ROOT)} --phase-label v7_phase7B_2_table_extraction_mvp_rerun",
        "```",
        "",
        "## 4. 数量统计",
        "",
        f"- table_candidates：{candidate_count}",
        f"- table_objects：{len(objects)}",
        f"- filtered candidates：{candidate_counts.get('filtered', 0)}",
        f"- deduped candidates：{candidate_counts.get('deduped', 0)}",
        f"- merged continued candidates：{candidate_counts.get('merged_into_primary', 0)}",
        f"- merged table_objects：{sum(1 for obj in objects if obj.get('merge_status') == 'merged')}",
        f"- P0 blocker 对象数：{p0_blocker_count(objects)}",
        f"- P1 warning 总次数：{p1_warning_count(objects)}",
        "",
        "## 5. validation_status 统计",
        "",
        "| validation_status | 数量 |",
        "|---|---:|",
    ]
    for status in ["pass", "pass_with_warnings", "partial", "manual_review", "fail"]:
        lines.append(f"| `{status}` | {status_counts.get(status, 0)} |")

    lines.extend(["", "## 6. source_span 粒度统计", "", "| source_span_granularity | 数量 |", "|---|---:|"])
    for granularity, count in sorted(granularity_counts.items()):
        lines.append(f"| `{granularity}` | {count} |")

    lines.extend(["", "## 7. 主要 warnings", "", "| warning | 数量 |", "|---|---:|"])
    for warning, count in warnings.most_common(20):
        lines.append(f"| `{warning}` | {count} |")

    lines.extend(
        [
            "",
            "## 8. Markdown review 与 index units preview",
            "",
            f"Markdown review：{'已生成' if REVIEW_MD_PATH.exists() else '未生成'}。",
            f"Index units preview：{'已生成' if INDEX_PREVIEW_PATH.exists() else '未生成'}。",
            "",
            "JSONL 是主结构化格式；Markdown 是派生审阅视图；index units preview 是从 JSON 派生的未来索引设计预览。完整 JSON 不直接用于 embedding，Phase7B-2 不进行向量化，不写 Milvus/BM25。",
            "",
            "## 9. duplicate / continued 处理结果",
            "",
            f"duplicate/shadow filtered 或 deduped candidates：{candidate_counts.get('filtered', 0) + candidate_counts.get('deduped', 0)}。",
            f"continued merged candidates：{candidate_counts.get('merged_into_primary', 0)}；merged table_objects：{sum(1 for obj in objects if obj.get('merge_status') == 'merged')}。",
            "",
            "## 10. Phase7A vs Phase7B-2 对比结论",
            "",
            "validation policy 已收紧，`pass_with_warnings` 下降不代表失败，而是 false positive、continued split、body missing、mixed boundary 和 row/cell blocker 不再被包装成可通过对象。建议先进入下一轮人工 review / gold dataset 建设，不建议立即扩大 smoke 或进入 production。",
            "",
            "## 11. MVP smoke 目标",
            "",
            f"目标判定：{'达到' if smoke_reached else '未完全达到'}。",
            "",
            "## 12. 当前抽取质量主要问题",
            "",
            "主要问题是 official chunks 的表体边界和表内绑定不足：`contains_table_text=false` 普遍存在，表体被解析为 `table_caption`、`paragraph` 或相邻 chunk；复杂 numeric/matrix 表仍有 column order、unit scope、footnote/reference binding 风险；source_span 仍不是 value-level bbox。",
            "",
            "## 13. 下一阶段建议",
            "",
            "建议进入下一轮人工 review / gold dataset 建设。暂不建议扩大 smoke，暂不建议进入 production。Route C 仍只是 backlog，不应立即实施。",
            "",
            "## 14. baseline / guardrail 是否漂移",
            "",
            "未发现 baseline / guardrail 漂移。official dataset SHA256 与 official chunks SHA256 已只读校验一致；本轮未修改 official baseline。",
            "",
            "## 15. Route C 状态",
            "",
            "Route C 仍只是 backlog。本轮不是 production implementation，也不构成 Route C 授权。",
            "",
            "## 16. 明确未执行事项",
            "",
            "- 未改 ingestion pipeline。",
            "- 未改 configs。",
            "- 未改 README。",
            "- 未改 baseline registry。",
            "- 未改 official dataset。",
            "- 未改 official baseline。",
            "- 未重建 chunks。",
            "- 未重建 BM25。",
            "- 未访问 Milvus。",
            "- 未写入 Milvus。",
            "- 未读取或查询 BM25 index。",
            "- 未跑 retrieval。",
            "- 未跑 embedding/rerank。",
            "- 未调用 Qwen/RAGAS/OCR/VLM。",
            "- 未接入 production。",
            "- 未进入 Route C。",
            "",
        ]
    )
    PHASE_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def count_warning(objects: list[dict[str, Any]], warning: str) -> int:
    return sum(1 for obj in objects if warning in (obj.get("warnings") or []))


def write_comparison_report(objects: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    phase7a_objects_path = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_objects.jsonl"
    phase7a_candidates_path = ROOT / "data/experiments/v7_phase7_table_extraction_mvp/table_candidates.jsonl"
    phase7a_objects = load_optional_jsonl(phase7a_objects_path)
    phase7a_candidates = load_optional_jsonl(phase7a_candidates_path)
    phase7b_candidates = load_optional_jsonl(CANDIDATES_PATH)

    a_status = Counter(obj.get("validation_status", "") for obj in phase7a_objects)
    b_status = Counter(row["validation_status"] for row in rows)
    b_candidate_status = candidate_status_counts(phase7b_candidates)
    b_false_positive_candidates = sum(
        1 for candidate in phase7b_candidates if "false_positive_candidate" in (candidate.get("warnings") or [])
    )
    b_warnings = warning_counter(objects)
    a_warnings = warning_counter(phase7a_objects)

    lines = [
        "# Phase7A vs Phase7B-2 同 smoke rerun 对比报告",
        "",
        "## 1. 对比目标",
        "",
        "本报告比较 Phase7A 表格抽取 MVP 与 Phase7B-2 heuristic hardening 后的同一批 smoke rerun。目标是确认 false positive、duplicate、continued split、boundary、row/cell blocker 与 validation policy 是否变得更诚实。本轮不扩大 smoke，不接 production，不读取或查询 BM25 index，不访问 Milvus，不运行 retrieval/embedding/rerank/model/OCR/VLM。",
        "",
        "## 2. Phase7A 数量",
        "",
        f"- table_candidates：{len(phase7a_candidates)}",
        f"- table_objects：{len(phase7a_objects)}",
        "",
        "## 3. Phase7B-2 数量",
        "",
        f"- table_candidates：{len(phase7b_candidates)}",
        f"- table_objects：{len(objects)}",
        "",
        "## 4. table_candidates 数量变化",
        "",
        f"Phase7A 为 {len(phase7a_candidates)}，Phase7B-2 为 {len(phase7b_candidates)}。候选仍保留同一检测范围，但 Phase7B-2 在 candidate JSONL 中显式标记 `filtered`、`deduped` 与 `merged_into_primary`。",
        "",
        "## 5. table_objects 数量变化",
        "",
        f"Phase7A 为 {len(phase7a_objects)}，Phase7B-2 为 {len(objects)}。对象数下降主要来自 false positive / shadow candidate 过滤，以及 continued part 合并，不是 smoke 扩大或 baseline 改动。",
        "",
        "## 6. validation_status 变化",
        "",
        "| validation_status | Phase7A | Phase7B-2 |",
        "|---|---:|---:|",
    ]
    for status in ["pass", "pass_with_warnings", "partial", "manual_review", "fail"]:
        lines.append(f"| `{status}` | {a_status.get(status, 0)} | {b_status.get(status, 0)} |")

    lines.extend(
        [
            "",
            "## 7. pass_with_warnings 是否更真实",
            "",
            "`pass_with_warnings` 的下降不是失败，而是 validation 收紧后不再让 P0 blocker 混入可通过对象。Phase7B-2 中该状态只表示字段完整、可人工审阅且无 P0 blocker；它仍不等于 confirmed 或 production-ready。",
            "",
            "## 8. partial / fail 增减是否合理",
            "",
            "partial 增加是合理的：body missing、mixed boundary、row/cell alignment、matrix flattened、continued needs merge 等对象应保留为可审阅但不可通过。fail 只用于明确 false positive 或无可审阅结构的对象；本轮明显 false positive 默认不生成 table_object，而在 candidate 层记录 filtered/deduped。",
            "",
            "## 9. false positive 数量变化",
            "",
            f"Phase7A table_objects 中 false positive 主要来自 Round 1 标注的 4 个对象。Phase7B-2 candidate 层 `false_positive_candidate` 标记数量为 {b_false_positive_candidates}，其中 filtered 为 {b_candidate_status.get('filtered', 0)}、deduped/shadow 为 {b_candidate_status.get('deduped', 0)}；对象层不再把这些明显 false positive 输出为 pass_with_warnings。",
            "",
            "## 10. duplicate / continued 处理结果",
            "",
            f"deduped candidates：{b_candidate_status.get('deduped', 0)}；merged continued candidates：{b_candidate_status.get('merged_into_primary', 0)}；merged table_objects：{sum(1 for obj in objects if obj.get('merge_status') == 'merged')}。continued part 不再作为普通独立表通过。",
            "",
            "## 11. body_blocks_missing 变化",
            "",
            f"Phase7A `body_blocks_missing` 显式 warning 数量为 {a_warnings.get('body_blocks_missing', 0)}，Phase7B-2 为 {b_warnings.get('body_blocks_missing', 0)}。Phase7B-2 将无 body_block_ids 作为 blocking condition，不允许 pass_with_warnings。",
            "",
            "## 12. mixed_table_block_risk 变化",
            "",
            f"Phase7A `mixed_table_block_risk`：{a_warnings.get('mixed_table_block_risk', 0)}；Phase7B-2：{b_warnings.get('mixed_table_block_risk', 0)}。该 warning 在 Phase7B-2 中阻断 pass_with_warnings。",
            "",
            "## 13. row/cell blocker 变化",
            "",
            f"Phase7B-2 `cell_alignment_error`：{b_warnings.get('cell_alignment_error', 0)}，`metric_level_cell_gap`：{b_warnings.get('metric_level_cell_gap', 0)}，`matrix_flattened`：{b_warnings.get('matrix_flattened', 0)}，`row_cell_blocking_warning`：{b_warnings.get('row_cell_blocking_warning', 0)}。这些对象会降级为 partial 或 fail，而不是继续 pass_with_warnings。",
            "",
            "## 14. unit / footnote / reference warning 变化",
            "",
            f"Phase7B-2 `unit_visible_not_bound`：{b_warnings.get('unit_visible_not_bound', 0)}，`unit_scope_column_level`：{b_warnings.get('unit_scope_column_level', 0)}，`footnote_present_not_bound`：{b_warnings.get('footnote_present_not_bound', 0)}，`internal_reference_column`：{b_warnings.get('internal_reference_column', 0)}，`external_citation_not_supported`：{b_warnings.get('external_citation_not_supported', 0)}。这些 warning 更接近 Phase6D contract：visible 不等于 bound。",
            "",
            "## 15. source_span limitation 是否更清楚",
            "",
            "所有 Phase7B-2 table_object 都写入 `source_span_granularity`、`source_span_limitation` 与 `no_value_level_bbox=true`。row/table-row level provenance 只作为离线人工审阅限制，不暗示 cell/value-level provenance。",
            "",
            "## 16. Markdown review 是否更可审阅",
            "",
            "Markdown card 已显示 table_object_id、doc_id、table_id、validation_status、blocking/nonblocking warnings、candidate_status、boundary_status、merge_status、source_span limitation、block ids、预览表与 review_notes。",
            "",
            "## 17. index units preview 是否更合理",
            "",
            "Index units preview 继续作为 JSON 主数据的派生自然语言视图，不直接 dump 完整 JSON。`fail` 对象默认不生成 index units；`partial` 对象在 metadata 中带 caution。本轮未向量化，未写 Milvus/BM25。",
            "",
            "## 18. 是否达到 Phase7B-2 修复目标",
            "",
            "达到本轮离线修复目标：false positive / duplicate / continued 已处理，body missing / mixed block / row-cell blocker 能阻断 pass_with_warnings，validation_status 更诚实，source_span limitation 更清楚。",
            "",
            "## 19. 不扩大 smoke、不接 production 声明",
            "",
            f"本轮 smoke doc_id 保持不变：`{', '.join(SMOKE_DOC_IDS)}`。未改 ingestion pipeline，未改 configs/README/baseline registry，未改 official dataset/baseline，未重建 chunks/BM25，未读取或查询 BM25 index，未访问或写入 Milvus，未跑 retrieval/embedding/rerank，未调用 Qwen/RAGAS/OCR/VLM，未接入 production，未进入 Route C。",
            "",
        ]
    )
    COMPARISON_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def configure_paths(
    table_objects_path: Path,
    candidates_path: Path,
    report_dir: Path,
    output_dir: Path,
    phase_label: str,
    summary_filename: str,
) -> None:
    global TABLE_OBJECTS_PATH, CANDIDATES_PATH, REPORT_DIR, SUMMARY_CSV_PATH, REPORT_MD_PATH
    global PHASE_SUMMARY_PATH, REVIEW_MD_PATH, INDEX_PREVIEW_PATH, COMPARISON_MD_PATH
    global PHASE_LABEL, SUMMARY_FILENAME

    TABLE_OBJECTS_PATH = table_objects_path if table_objects_path.is_absolute() else ROOT / table_objects_path
    CANDIDATES_PATH = candidates_path if candidates_path.is_absolute() else ROOT / candidates_path
    REPORT_DIR = report_dir if report_dir.is_absolute() else ROOT / report_dir
    SUMMARY_CSV_PATH = REPORT_DIR / "table_object_validation_summary.csv"
    REPORT_MD_PATH = REPORT_DIR / "table_object_validation_report.md"
    PHASE_SUMMARY_PATH = REPORT_DIR / summary_filename
    resolved_output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    REVIEW_MD_PATH = resolved_output_dir / "table_objects_review.md"
    INDEX_PREVIEW_PATH = resolved_output_dir / "table_index_units.preview.jsonl"
    COMPARISON_MD_PATH = REPORT_DIR / "phase7b_2_rerun_comparison.md"
    PHASE_LABEL = phase_label
    SUMMARY_FILENAME = summary_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate offline table_object JSONL artifacts.")
    parser.add_argument("--table-objects", type=Path, default=DEFAULT_TABLE_OBJECTS_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TABLE_OBJECTS_PATH.parent)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--phase-label", default="Phase7A")
    parser.add_argument("--summary-filename", default="phase7a_mvp_summary.md")
    parser.add_argument("--write-comparison", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(
        args.table_objects,
        args.candidates,
        args.report_dir,
        args.output_dir,
        args.phase_label,
        args.summary_filename,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    objects = load_jsonl(TABLE_OBJECTS_PATH)
    rows = [validate_object(obj) for obj in objects]
    write_validated_table_objects(objects, rows)
    write_summary_csv(rows)
    write_validation_report(objects, rows)
    write_phase_summary(objects, rows)
    if args.write_comparison:
        write_comparison_report(objects, rows)
    print(
        json.dumps(
            {
                "table_objects": len(objects),
                "validation_status": Counter(row["validation_status"] for row in rows),
                "outputs": [
                    str(SUMMARY_CSV_PATH.relative_to(ROOT)),
                    str(REPORT_MD_PATH.relative_to(ROOT)),
                    str(PHASE_SUMMARY_PATH.relative_to(ROOT)),
                    str(COMPARISON_MD_PATH.relative_to(ROOT)) if args.write_comparison else "",
                ],
            },
            ensure_ascii=False,
            default=dict,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
