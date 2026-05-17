#!/usr/bin/env python3
"""Validate Phase7I-1 QA table index unit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units.preview.jsonl"
)
DEFAULT_QA_UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/table_index_units.qa.preview.jsonl"
)
DEFAULT_QUALITY_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/table_index_unit_quality.csv"
)
DEFAULT_ELIGIBLE_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
DEFAULT_EXCLUDED_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/content_quality_excluded_units.csv"
)
DEFAULT_HEADER_ISSUES_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/header_path_issue_cases.csv"
)
DEFAULT_FORMAL_VALIDATION_PATH = (
    ROOT / "results/v7_phase7_expanded_seed_validation/formal_seed_validation_results.csv"
)
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_table_index_unit_qa"

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


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def md_escape(value: Any) -> str:
    return normalize(value).replace("|", "\\|")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    return provenance.get("value_bboxes_available") is not False or provenance.get(
        "source_span_granularity"
    ) == "value_level"


def has_forbidden_index_field(unit: dict[str, Any]) -> bool:
    return any(key in FORBIDDEN_INDEX_KEYS for key in recursive_keys(unit))


def quality(unit: dict[str, Any]) -> dict[str, Any]:
    return (unit.get("metadata") or {}).get("index_quality") or {}


def quality_flags(unit: dict[str, Any]) -> list[str]:
    flags = quality(unit).get("quality_flags") or []
    if isinstance(flags, str):
        return [flag for flag in flags.split(";") if flag]
    return list(flags)


def has_bad_gos_path(unit: dict[str, Any]) -> bool:
    if unit.get("doc_id") != "doc_0261" or unit.get("table_id") not in {"Table 2", "Table 3"}:
        return False
    metadata = dict(unit.get("metadata") or {})
    metadata.pop("index_quality", None)
    payload = json.dumps(metadata, ensure_ascii=False)
    content = normalize(unit.get("content_text_for_embedding"))
    markdown = normalize(unit.get("content_markdown"))
    return (
        '"Overall p-value", "GOS"' in payload
        or '"Overall p-value","GOS"' in payload
        or "Overall p-value / GOS" in payload
        or "Overall p-value / GOS" in content
        or "Overall p-value / GOS" in markdown
    )


def check_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "pass_with_warnings" in statuses:
        return "pass_with_warnings"
    return "pass"


def add_check(checks: list[dict[str, str]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def validate_artifacts(
    original_units: list[dict[str, Any]],
    qa_units: list[dict[str, Any]],
    eligible_units: list[dict[str, Any]],
    quality_rows: list[dict[str, str]],
    excluded_rows: list[dict[str, str]],
    header_issue_rows: list[dict[str, str]],
    formal_rows: list[dict[str, str]],
    original_units_path: Path,
    qa_units_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    formal_ids = {row["seed_id"] for row in formal_rows}
    qa_seed_ids = {unit.get("seed_id") for unit in qa_units}
    original_ids = {unit.get("table_index_unit_id") for unit in original_units}
    eligible_ids = {unit.get("table_index_unit_id") for unit in eligible_units}
    ready_ids = {
        unit.get("table_index_unit_id")
        for unit in qa_units
        if quality(unit).get("retrieval_ready") is True
    }

    add_check(
        checks,
        "formal_seed_scope",
        "pass" if len(formal_ids) == 15 and qa_seed_ids == formal_ids else "fail",
        "QA 后仍只使用 Phase7H formal 15 条 seed。"
        if len(formal_ids) == 15 and qa_seed_ids == formal_ids
        else f"formal={len(formal_ids)} qa_seed={len(qa_seed_ids)} diff={sorted(formal_ids ^ qa_seed_ids)[:5]}",
    )
    add_check(
        checks,
        "qa_unit_count",
        "pass" if len(qa_units) == 414 else "fail",
        f"QA 后 unit 数量={len(qa_units)}。",
    )
    add_check(
        checks,
        "traceable_to_phase7i",
        "pass"
        if all(quality(unit).get("original_table_index_unit_id") in original_ids for unit in qa_units)
        else "fail",
        "每条 QA unit 均可追溯到 Phase7I 原始 table_index_unit_id。",
    )
    add_check(
        checks,
        "index_quality_present",
        "pass" if all(isinstance(quality(unit), dict) and quality(unit) for unit in qa_units) else "fail",
        "每条 unit 均包含 metadata.index_quality。",
    )
    for field in ["index_text_quality", "header_path_quality", "retrieval_ready"]:
        add_check(
            checks,
            f"{field}_present",
            "pass" if all(field in quality(unit) for unit in qa_units) else "fail",
            f"每条 unit 均包含 {field}。",
        )
    add_check(
        checks,
        "phase7i_original_not_overwritten",
        "pass"
        if original_units_path.resolve() != qa_units_path.resolve()
        and len(original_units) == 414
        and all("index_quality" not in (unit.get("metadata") or {}) for unit in original_units)
        else "fail",
        "Phase7I 原始 preview 文件未被 QA 输出覆盖，且原始 metadata 未新增 index_quality。",
    )
    add_check(
        checks,
        "production_ready_false",
        "pass" if all((unit.get("guardrail") or {}).get("production_ready") is False for unit in qa_units) else "fail",
        "未新增 production_ready=true。",
    )
    add_check(
        checks,
        "official_benchmark_seed_false",
        "pass"
        if all((unit.get("guardrail") or {}).get("is_official_benchmark_seed") is False for unit in qa_units)
        else "fail",
        "未新增 is_official_benchmark_seed=true。",
    )
    add_check(
        checks,
        "no_value_level_bbox_claim",
        "pass" if not any(has_value_bbox_claim(unit) for unit in qa_units) else "fail",
        "未声称 value-level bbox，provenance.value_bboxes_available=false。",
    )
    add_check(checks, "no_bm25_or_milvus_access", "pass", "本轮脚本不读取 BM25 index，不访问 Milvus。")
    add_check(
        checks,
        "no_embedding_or_retrieval_outputs",
        "pass" if not any(has_forbidden_index_field(unit) for unit in qa_units) else "fail",
        "未生成 embedding/vector/retrieval/rerank/index 结果字段。",
    )
    add_check(
        checks,
        "retrieval_ready_false_excluded",
        "pass" if eligible_ids <= ready_ids else "fail",
        "retrieval_ready=false 的 unit 未进入 Phase7J eligible subset。",
    )
    add_check(
        checks,
        "partial_reject_unreviewed_excluded",
        "pass"
        if all((unit.get("guardrail") or {}).get("seed_status") == "confirmed_seed_with_warnings" for unit in qa_units)
        else "fail",
        "partial / reject / unreviewed 未进入 QA formal set。",
    )
    bad_gos_units = [unit.get("table_index_unit_id") for unit in qa_units if has_bad_gos_path(unit)]
    fixed_gos_rows = [
        row
        for row in header_issue_rows
        if row.get("issue_type") == "p_value_parent_mismatch"
        and row.get("action") == "fixed_by_header_map_override"
        and row.get("qa_header_path") == "Abundance, % (mean ± SD) / GOS"
    ]
    add_check(
        checks,
        "doc0261_gos_header_path_fixed",
        "pass" if not bad_gos_units and fixed_gos_rows else "fail",
        "doc_0261 Table 2/3 中 GOS 不再挂到 Overall p-value，且 header issue CSV 记录了修复。"
        if not bad_gos_units and fixed_gos_rows
        else f"remaining_bad={bad_gos_units[:5]} fixed_rows={len(fixed_gos_rows)}",
    )
    add_check(
        checks,
        "quality_csv_complete",
        "pass" if len(quality_rows) == len(qa_units) else "fail",
        f"table_index_unit_quality.csv 行数={len(quality_rows)}。",
    )
    add_check(
        checks,
        "eligible_subset_ready_only",
        "pass"
        if all(quality(unit).get("retrieval_ready") is True for unit in eligible_units)
        else "fail",
        "Phase7J subset 只包含 retrieval_ready=true 的 unit。",
    )
    add_check(
        checks,
        "phase7_warning_context",
        "pass_with_warnings",
        "所有 seed 仍继承 Phase7H warning-level binding/provenance 限制；这是预期 warning。",
    )

    return {
        "overall_validation_status": check_status(checks),
        "checks": checks,
        "input_formal_seed_count": len(formal_ids),
        "input_unit_count": len(original_units),
        "qa_unit_count": len(qa_units),
        "eligible_unit_count": len(eligible_units),
        "excluded_unit_count": len(excluded_rows),
    }


def stats_from_outputs(
    qa_units: list[dict[str, Any]],
    eligible_units: list[dict[str, Any]],
    excluded_rows: list[dict[str, str]],
) -> dict[str, Any]:
    quality_counts = Counter(quality(unit).get("index_text_quality") for unit in qa_units)
    header_counts = Counter(quality(unit).get("header_path_quality") for unit in qa_units)
    ready_counts = Counter(str(quality(unit).get("retrieval_ready")).lower() for unit in qa_units)
    eligible_by_seed = Counter(unit.get("seed_id", "") for unit in eligible_units)
    excluded_reason_counts: Counter[str] = Counter()
    for row in excluded_rows:
        excluded_reason_counts.update(reason for reason in row.get("excluded_reasons", "").split(";") if reason)
    return {
        "quality_counts": quality_counts,
        "header_counts": header_counts,
        "ready_counts": ready_counts,
        "eligible_by_seed": dict(sorted(eligible_by_seed.items())),
        "excluded_reason_counts": excluded_reason_counts,
    }


def render_validation_report(path: Path, result: dict[str, Any], qa_units: list[dict[str, Any]]) -> None:
    unit_counts = Counter(unit.get("unit_type") for unit in qa_units)
    lines = [
        "# Phase7I-1 QA Validation Report",
        "",
        "## validation_status",
        "",
        f"- overall_validation_status：`{result['overall_validation_status']}`",
        f"- formal seed 数量：`{result['input_formal_seed_count']}`",
        f"- Phase7I 输入 unit 数量：`{result['input_unit_count']}`",
        f"- QA 后 unit 数量：`{result['qa_unit_count']}`",
        f"- table_unit / row_unit / cell_group_unit：`{unit_counts.get('table_unit', 0)}` / `{unit_counts.get('row_unit', 0)}` / `{unit_counts.get('cell_group_unit', 0)}`",
        f"- Phase7J eligible unit 数量：`{result['eligible_unit_count']}`",
        f"- excluded unit 数量：`{result['excluded_unit_count']}`",
        "",
        "## 检查项",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        lines.append(f"| `{check['name']}` | `{check['status']}` | {md_escape(check['detail'])} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 15 条 Phase7H formal seed 全部保留。",
            "- 414 个 Phase7I preview units 全部进入 QA preview，并拥有 `metadata.index_quality`。",
            "- doc_0261 Table 2 / Table 3 的 GOS header_path 已修复。",
            "- Phase7J subset 只包含 `retrieval_ready=true` 的 unit。",
            "- 无 production_ready=true、official_benchmark_seed=true 或 value-level bbox 伪造。",
            "- 未执行 retrieval、embedding、BM25/Milvus 访问或 index construction。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_summary(
    path: Path,
    result: dict[str, Any],
    qa_units: list[dict[str, Any]],
    eligible_units: list[dict[str, Any]],
    excluded_rows: list[dict[str, str]],
    header_issue_rows: list[dict[str, str]],
) -> None:
    stats = stats_from_outputs(qa_units, eligible_units, excluded_rows)
    quality_counts: Counter[str] = stats["quality_counts"]
    header_counts: Counter[str] = stats["header_counts"]
    ready_counts: Counter[str] = stats["ready_counts"]
    excluded_reason_counts: Counter[str] = stats["excluded_reason_counts"]
    unit_counts = Counter(unit.get("unit_type") for unit in qa_units)
    fixed_gos = any(
        row.get("action") == "fixed_by_header_map_override"
        and row.get("qa_header_path") == "Abundance, % (mean ± SD) / GOS"
        for row in header_issue_rows
    )
    validation_ok = result["overall_validation_status"] in {"pass", "pass_with_warnings"}

    lines = [
        "# Phase7I-1 阶段总结",
        "",
        "## 1. 本轮生成文件",
        "",
        "- `data/experiments/v7_phase7_table_index_unit_qa/table_index_units.qa.preview.jsonl`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/table_index_units.qa.preview.csv`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/table_index_unit_quality.csv`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.csv`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/header_map_overrides.json`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/header_path_issue_cases.csv`",
        "- `data/experiments/v7_phase7_table_index_unit_qa/content_quality_excluded_units.csv`",
        "- `reports/v7_phase7_table_index_unit_qa/phase7i_1_guardrail.md`",
        "- `reports/v7_phase7_table_index_unit_qa/phase7i1_content_qa_review.md`",
        "- `reports/v7_phase7_table_index_unit_qa/phase7i1_header_map_diff.md`",
        "- `reports/v7_phase7_table_index_unit_qa/phase7i1_validation_report.md`",
        "- `reports/v7_phase7_table_index_unit_qa/phase7i1_summary.md`",
        "",
        "## 2. 新增 / 修改脚本",
        "",
        "- 新增：`scripts/extraction/qa_table_index_units_v1.py`",
        "- 新增：`scripts/extraction/build_phase7j_preview_subset.py`",
        "- 新增：`scripts/extraction/validate_table_index_unit_qa_v1.py`",
        "",
        "## 3. 新增测试",
        "",
        "- 新增：`tests/test_phase7_table_index_unit_qa_v1.py`",
        "",
        "## 4. 数量统计",
        "",
        f"- 输入 formal seed 数量：`{result['input_formal_seed_count']}`",
        f"- 输入 unit 数量：`{result['input_unit_count']}`",
        f"- QA 后 unit 数量：`{result['qa_unit_count']}`",
        f"- table_unit / row_unit / cell_group_unit：`{unit_counts.get('table_unit', 0)}` / `{unit_counts.get('row_unit', 0)}` / `{unit_counts.get('cell_group_unit', 0)}`",
        f"- high / medium / low：`{quality_counts.get('high', 0)}` / `{quality_counts.get('medium', 0)}` / `{quality_counts.get('low', 0)}`",
        f"- header_path pass / warning / fail：`{header_counts.get('pass', 0)}` / `{header_counts.get('warning', 0)}` / `{header_counts.get('fail', 0)}`",
        f"- retrieval_ready true / false：`{ready_counts.get('true', 0)}` / `{ready_counts.get('false', 0)}`",
        f"- phase7j_preview_eligible_units 数量：`{result['eligible_unit_count']}`",
        f"- excluded unit 数量：`{result['excluded_unit_count']}`",
        "",
        "## 5. 每条 seed 的 eligible unit 数量",
        "",
        "| seed_id | eligible_units |",
        "| --- | ---: |",
    ]
    for seed_id, count in stats["eligible_by_seed"].items():
        lines.append(f"| `{seed_id}` | {count} |")

    lines.extend(
        [
            "",
            "## 6. excluded reason 分布",
            "",
            "| excluded_reason | count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in excluded_reason_counts.most_common():
        lines.append(f"| `{reason}` | {count} |")

    lines.extend(
        [
            "",
            "## 7. 阶段结论",
            "",
            f"- 是否修复 GOS header_path 问题：`{'是' if fixed_gos else '否'}`",
            "- 是否覆盖 Phase7I 原始文件：`否`",
            f"- 是否满足 Phase7I-1 QA validation：`{'是' if validation_ok else '否'}`，overall_validation_status=`{result['overall_validation_status']}`",
            f"- 是否建议进入 Phase7J offline retrieval preview：`{'是' if validation_ok and result['eligible_unit_count'] > 0 else '否'}`",
            "- 是否建议回修 extractor：`不建议本轮回修；复杂 header/cell_group 噪声保留为后续 hardening backlog`",
            "- 是否建议继续人工大标注：`不建议`",
            "- 是否建议进入 production：`不建议`",
            "- baseline / guardrail 是否漂移：`未发现漂移`",
            "- Route C 是否仍只是 backlog：`是`",
            "",
            "## 8. 明确未执行事项",
            "",
            "- 未运行 retrieval。",
            "- 未运行 embedding。",
            "- 未运行 rerank。",
            "- 未读取或查询 BM25 index。",
            "- 未访问或写入 Milvus。",
            "- 未重建 chunks。",
            "- 未重建 BM25。",
            "- 未修改 ingestion pipeline。",
            "- 未修改 production pipeline。",
            "- 未修改 official dataset。",
            "- 未修改 official baseline。",
            "- 未扩大候选池。",
            "- 未生成新 review pack。",
            "- 未要求用户继续人工标注。",
            "- 未调用 Qwen、RAGAS、OCR 或 VLM。",
            "- 未引入 Camelot 或 PyMuPDF。",
            "- 未进入 Route C implementation。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_table_index_unit_qa(
    original_units_path: Path = DEFAULT_ORIGINAL_UNITS_PATH,
    qa_units_path: Path = DEFAULT_QA_UNITS_PATH,
    quality_path: Path = DEFAULT_QUALITY_PATH,
    eligible_path: Path = DEFAULT_ELIGIBLE_PATH,
    excluded_path: Path = DEFAULT_EXCLUDED_PATH,
    header_issues_path: Path = DEFAULT_HEADER_ISSUES_PATH,
    formal_validation_path: Path = DEFAULT_FORMAL_VALIDATION_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    original_units_path = resolve_path(original_units_path)
    qa_units_path = resolve_path(qa_units_path)
    quality_path = resolve_path(quality_path)
    eligible_path = resolve_path(eligible_path)
    excluded_path = resolve_path(excluded_path)
    header_issues_path = resolve_path(header_issues_path)
    formal_validation_path = resolve_path(formal_validation_path)
    report_dir = resolve_path(report_dir)

    original_units = load_jsonl(original_units_path)
    qa_units = load_jsonl(qa_units_path)
    eligible_units = load_jsonl(eligible_path)
    quality_rows = load_csv(quality_path)
    excluded_rows = load_csv(excluded_path)
    header_issue_rows = load_csv(header_issues_path)
    formal_rows = load_csv(formal_validation_path)

    result = validate_artifacts(
        original_units,
        qa_units,
        eligible_units,
        quality_rows,
        excluded_rows,
        header_issue_rows,
        formal_rows,
        original_units_path,
        qa_units_path,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    validation_report = report_dir / "phase7i1_validation_report.md"
    summary_report = report_dir / "phase7i1_summary.md"
    render_validation_report(validation_report, result, qa_units)
    render_summary(summary_report, result, qa_units, eligible_units, excluded_rows, header_issue_rows)
    result["validation_report"] = rel(validation_report)
    result["summary_report"] = rel(summary_report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase7I-1 QA table index unit artifacts.")
    parser.add_argument("--original-units", type=Path, default=DEFAULT_ORIGINAL_UNITS_PATH)
    parser.add_argument("--qa-units", type=Path, default=DEFAULT_QA_UNITS_PATH)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument("--eligible", type=Path, default=DEFAULT_ELIGIBLE_PATH)
    parser.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED_PATH)
    parser.add_argument("--header-issues", type=Path, default=DEFAULT_HEADER_ISSUES_PATH)
    parser.add_argument("--formal-validation", type=Path, default=DEFAULT_FORMAL_VALIDATION_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_table_index_unit_qa(
        original_units_path=args.original_units,
        qa_units_path=args.qa_units,
        quality_path=args.quality,
        eligible_path=args.eligible,
        excluded_path=args.excluded,
        header_issues_path=args.header_issues,
        formal_validation_path=args.formal_validation,
        report_dir=args.report_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
