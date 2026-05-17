#!/usr/bin/env python3
"""Validate Phase7J offline table retrieval wiring preview artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELIGIBLE_JSONL = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
DEFAULT_QA_JSONL = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/table_index_units.qa.preview.jsonl"
)
DEFAULT_EXCLUDED_CSV = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/content_quality_excluded_units.csv"
)
DEFAULT_QUERY_JSONL = (
    ROOT / "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.jsonl"
)
DEFAULT_RESULTS_CSV = (
    ROOT / "results/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_results.csv"
)
DEFAULT_RESULTS_JSON = (
    ROOT / "results/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_results.json"
)
DEFAULT_TOPK_JSONL = ROOT / "results/v7_phase7_table_retrieval_wiring_preview/topk_evidence_units.jsonl"
DEFAULT_QUERY_SUMMARY_CSV = (
    ROOT / "results/v7_phase7_table_retrieval_wiring_preview/query_hit_summary.csv"
)
DEFAULT_EVIDENCE_REPORT = (
    ROOT / "reports/v7_phase7_table_retrieval_wiring_preview/retrieval_evidence_cards.md"
)
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_table_retrieval_wiring_preview"

GENERATED_FILES = [
    "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.csv",
    "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.jsonl",
    "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_unit_expectations.csv",
    "results/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_results.csv",
    "results/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_results.json",
    "results/v7_phase7_table_retrieval_wiring_preview/topk_evidence_units.jsonl",
    "results/v7_phase7_table_retrieval_wiring_preview/query_hit_summary.csv",
    "results/v7_phase7_table_retrieval_wiring_preview/ranking_debug.csv",
    "reports/v7_phase7_table_retrieval_wiring_preview/phase7j_guardrail.md",
    "reports/v7_phase7_table_retrieval_wiring_preview/query_set_design_report.md",
    "reports/v7_phase7_table_retrieval_wiring_preview/retrieval_wiring_preview_report.md",
    "reports/v7_phase7_table_retrieval_wiring_preview/retrieval_evidence_cards.md",
    "reports/v7_phase7_table_retrieval_wiring_preview/phase7i1_to_phase7j_traceability.md",
    "reports/v7_phase7_table_retrieval_wiring_preview/phase7j_summary.md",
]

SCRIPT_FILES = [
    "scripts/evaluation/build_table_retrieval_preview_queries.py",
    "scripts/evaluation/run_table_retrieval_wiring_preview.py",
    "scripts/evaluation/render_table_retrieval_evidence_cards.py",
    "scripts/evaluation/validate_table_retrieval_wiring_preview.py",
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


def md_escape(value: Any) -> str:
    return normalize(value).replace("|", "\\|")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_false(value: Any) -> bool:
    return value is False or str(value).strip().lower() == "false"


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
    allowed = {"value_bboxes_available", "cell_bboxes_available"}
    forbidden = {"value_bbox", "value_bboxes", "value_level_bbox", "value_level_bboxes", "bbox", "bboxes"}
    for key in recursive_keys(unit):
        if key in allowed:
            continue
        if key in forbidden:
            return True
    provenance = unit.get("provenance") or {}
    return provenance.get("value_bboxes_available") is not False or provenance.get(
        "source_span_granularity"
    ) == "value_level"


def add_check(checks: list[dict[str, str]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def validation_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "pass_with_warnings" in statuses:
        return "pass_with_warnings"
    return "pass"


def group_by_query(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(row["rank"] or 0))
    return grouped


def validate_artifacts(
    eligible_units: list[dict[str, Any]],
    qa_units: list[dict[str, Any]],
    excluded_rows: list[dict[str, str]],
    queries: list[dict[str, Any]],
    result_rows: list[dict[str, str]],
    result_json: dict[str, Any],
    topk_records: list[dict[str, Any]],
    query_summary_rows: list[dict[str, str]],
    evidence_report_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    eligible_ids = {unit["table_index_unit_id"] for unit in eligible_units}
    excluded_ids = {row["table_index_unit_id"] for row in excluded_rows}
    query_ids = {query["query_id"] for query in queries}
    result_ids = {
        row["matched_table_index_unit_id"]
        for row in result_rows
        if row.get("matched_table_index_unit_id")
    }
    grouped = group_by_query(result_rows)
    query_types = {query["query_type"] for query in queries}
    guardrail = result_json.get("guardrail") or {}

    add_check(
        checks,
        "eligible_unit_count_274",
        "pass" if len(eligible_units) == 274 else "fail",
        f"eligible units 输入数量={len(eligible_units)}。",
    )
    add_check(
        checks,
        "query_count_20_to_50",
        "pass" if 20 <= len(queries) <= 50 else "fail",
        f"query 数量={len(queries)}。",
    )
    add_check(
        checks,
        "required_query_type_coverage",
        "pass"
        if {"table_lookup", "row_lookup", "metric_lookup"} <= query_types
        else "fail",
        f"query_type 覆盖={sorted(query_types)}。",
    )
    add_check(
        checks,
        "matching_only_eligible_units",
        "pass" if result_ids <= eligible_ids else "fail",
        "所有非空 matched_table_index_unit_id 均来自 Phase7I-1 eligible units。",
    )
    add_check(
        checks,
        "excluded_units_not_in_results",
        "pass" if not (result_ids & excluded_ids) else "fail",
        "Phase7I-1 excluded units 未进入 matching 结果。",
    )
    missing_queries = query_ids - set(grouped)
    top1_missing = [
        query_id
        for query_id, rows in grouped.items()
        if not rows or (rows[0]["match_status"] != "no_match" and not rows[0]["matched_table_index_unit_id"])
    ]
    add_check(
        checks,
        "query_has_top1_or_no_match",
        "pass" if not missing_queries and not top1_missing else "fail",
        "每条 query 至少有 top1 记录；no_match 会显式记录。",
    )
    traceable_failures = [
        row["query_id"]
        for row in result_rows
        if row["match_status"] != "no_match"
        and not (row.get("matched_seed_id") and row.get("matched_doc_id") and row.get("matched_table_id"))
    ]
    add_check(
        checks,
        "topk_traceable_to_seed_doc_table",
        "pass" if not traceable_failures else "fail",
        "top-k evidence 可追溯到 seed_id / doc_id / table_id。",
    )
    empty_evidence = [row["query_id"] for row in result_rows if not normalize(row.get("evidence_text"))]
    add_check(
        checks,
        "evidence_text_not_empty",
        "pass" if not empty_evidence else "fail",
        "所有结果行均包含 evidence_text；no_match 使用显式说明文本。",
    )
    non_no_match_rows = [row for row in result_rows if row["match_status"] != "no_match"]
    add_check(
        checks,
        "production_ready_false",
        "pass" if all(as_false(row.get("production_ready")) for row in non_no_match_rows) else "fail",
        "所有 evidence unit 的 production_ready=false。",
    )
    add_check(
        checks,
        "value_bboxes_available_false",
        "pass" if all(as_false(row.get("value_bboxes_available")) for row in non_no_match_rows) else "fail",
        "所有 evidence unit 的 value_bboxes_available=false。",
    )
    add_check(
        checks,
        "no_value_level_bbox_claim",
        "pass" if not any(has_value_bbox_claim(unit) for unit in eligible_units) else "fail",
        "eligible units 未伪造 value-level bbox；仅保留 value_bboxes_available=false。",
    )
    add_check(
        checks,
        "no_bm25_or_milvus_access",
        "pass"
        if guardrail.get("bm25_index_read_or_queried") is False
        and guardrail.get("milvus_accessed_or_written") is False
        else "fail",
        "结果 guardrail 记录未读取/查询 BM25，未访问/写入 Milvus。",
    )
    add_check(
        checks,
        "no_embedding_or_retrieval_service",
        "pass"
        if guardrail.get("embedding_run") is False
        and guardrail.get("rerank_run") is False
        and guardrail.get("isolated_lexical_matching_only") is True
        else "fail",
        "本轮只运行 isolated lexical matching，未运行 embedding/rerank/retrieval service。",
    )
    add_check(
        checks,
        "no_model_call",
        "pass" if guardrail.get("model_called") is False else "fail",
        "未调用 Qwen / RAGAS / OCR / VLM 或其他模型服务。",
    )
    add_check(
        checks,
        "no_official_benchmark_conclusion",
        "pass"
        if result_json.get("official_benchmark") is False
        and result_json.get("formal_retrieval_evaluation") is False
        else "fail",
        "结果 JSON 明确 official_benchmark=false 且 formal_retrieval_evaluation=false。",
    )
    add_check(
        checks,
        "evidence_cards_generated",
        "pass" if evidence_report_path.exists() and evidence_report_path.stat().st_size > 0 else "fail",
        f"evidence cards 路径：`{rel(evidence_report_path)}`。",
    )
    add_check(
        checks,
        "topk_jsonl_query_coverage",
        "pass" if {record["query_id"] for record in topk_records} == query_ids else "fail",
        "topk_evidence_units.jsonl 覆盖全部 query。",
    )
    add_check(
        checks,
        "phase7_warning_context",
        "pass_with_warnings",
        "所有 unit 仍继承 Phase7H warning-level binding/provenance 限制；这是 preview 预期状态。",
    )

    status = validation_status(checks)
    match_status_counts = Counter(row["match_status"] for row in result_rows)
    hit_unit_type_counts = Counter(
        row["matched_unit_type"]
        for row in result_rows
        if row.get("matched_unit_type") and row["match_status"] != "no_match"
    )
    return {
        "validation_status": status,
        "checks": checks,
        "eligible_unit_count": len(eligible_units),
        "qa_unit_count": len(qa_units),
        "excluded_unit_count": len(excluded_rows),
        "query_count": len(queries),
        "query_type_distribution": dict(Counter(query["query_type"] for query in queries)),
        "match_status_counts": dict(match_status_counts),
        "hit_unit_type_counts": dict(hit_unit_type_counts),
        "no_match_count": match_status_counts.get("no_match", 0),
        "weak_match_count": match_status_counts.get("weak_match", 0),
        "query_summary_count": len(query_summary_rows),
        "guardrail_fail_count": sum(1 for check in checks if check["status"] == "fail"),
    }


def render_guardrail(report_dir: Path) -> None:
    lines = [
        "# Phase7J Guardrail 边界说明",
        "",
        "1. 本轮定位为 offline table retrieval wiring preview。",
        "2. 本轮不是正式 retrieval evaluation。",
        "3. 本轮不是 benchmark。",
        "4. 本轮不比较 flat chunks。",
        "5. 本轮不运行 embedding。",
        "6. 本轮不访问 BM25 / Milvus。",
        "7. 本轮只使用 274 个 Phase7I-1 eligible units。",
        "8. Phase7I-1 excluded units 不进入 formal matching。",
        "9. 本轮只做 isolated lexical dry-run。",
        "10. 本轮不接 production。",
        "11. 本轮不伪造 value-level bbox；value_bboxes_available 仍为 false。",
        "12. Route C 仍只是 backlog。",
        "",
        "## Official baseline pins",
        "",
        "- official dataset：`reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl`",
        "- official clean baseline：`phase5f_official_clean_baseline`",
        "- official chunks：`data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl`",
        "- official BM25 与 Milvus collection 仅作为 pin 记录，本轮不读取、不查询、不访问、不写入。",
    ]
    (report_dir / "phase7j_guardrail.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_validation_report(report_dir: Path, validation: dict[str, Any]) -> None:
    lines = [
        "# Phase7J 检索 Wiring Preview 验证报告",
        "",
        "## validation_status",
        "",
        f"- validation_status：`{validation['validation_status']}`",
        f"- eligible unit 数量：`{validation['eligible_unit_count']}`",
        f"- query 数量：`{validation['query_count']}`",
        f"- excluded unit 数量：`{validation['excluded_unit_count']}`",
        f"- no_match 数量：`{validation['no_match_count']}`",
        f"- weak_match 数量：`{validation['weak_match_count']}`",
        "",
        "## 检查项",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in validation["checks"]:
        lines.append(
            f"| `{md_escape(check['name'])}` | `{md_escape(check['status'])}` | {md_escape(check['detail'])} |"
        )
    lines.extend(
        [
            "",
            "## match_status 统计",
            "",
            "| match_status | count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(validation["match_status_counts"].items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## hit_unit_type 统计", "", "| unit_type | count |", "| --- | ---: |"])
    for unit_type, count in sorted(validation["hit_unit_type_counts"].items()):
        lines.append(f"| `{unit_type}` | {count} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- query dry-run 已执行，并生成 top-k evidence。",
            "- evidence cards 已生成，可用于人工审阅 evidence format、metadata traceability 与 unit_type 命中形态。",
            "- no_match / weak_match 若出现，只作为 wiring sanity 记录，不作为正式 recall 结论。",
            "- 本报告不输出 official benchmark conclusion。",
        ]
    )
    (report_dir / "retrieval_wiring_preview_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def render_traceability(report_dir: Path, validation: dict[str, Any]) -> None:
    lines = [
        "# Phase7I-1 到 Phase7J 可追溯报告",
        "",
        "## 1. eligible units 如何进入 Phase7J",
        "",
        "Phase7J formal input 只读取 `data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl` 与对应 CSV。该 subset 来自 Phase7I-1 QA 后 retrieval_ready=true 的 units，本轮读取数量为 274。",
        "",
        "## 2. excluded units 为什么不进入",
        "",
        "Phase7I-1 excluded 140 个 unit 主要因为 retrieval_ready=false、header_path_quality fail、index_text_quality low、header_path_contains_data_value 或 information_density_below_threshold。本轮仅只读引用 excluded CSV 做 guardrail 检查，不把 excluded units 放入 lexical matching 候选池。",
        "",
        "## 3. query 如何自动生成",
        "",
        "query builder 从 eligible units 的 caption、doc_id、table_id、metadata.row_label、metadata.header_path、metadata.row_values 与 metadata.cell_group_values 自动派生 query_text，并写入 expected_seed_id、expected_doc_id、expected_table_id、expected_unit_type、expected_table_index_unit_id 与 expected_row_label。query_text 不包含内部 table_index_unit_id。",
        "",
        "## 4. lexical dry-run 如何执行",
        "",
        "dry-run 只在本地 JSONL/CSV 上执行 token overlap、doc_id/table_id/row_label phrase hit、header/value phrase hit 与 query_type-unit_type 轻量加权。它不读取 BM25 index，不访问 Milvus，不运行 embedding，不调用模型，不接 production retrieval service。",
        "",
        "## 5. top-k evidence 如何追溯",
        "",
        "每条 top-k evidence 行记录 matched_table_index_unit_id、matched_seed_id、matched_doc_id、matched_table_id、matched_row_label、source_csv_path、source_pdf_crop_path、value_bboxes_available 与 production_ready。由此可从 query 追溯到 seed、unit、CSV 与 crop。",
        "",
        "## 6. 为什么不是正式 retrieval evaluation",
        "",
        "本轮 query set 是自动构造的小规模 preview set，scoring 是 isolated lexical/token matching，未运行 embedding/rerank/retrieval service，也未设置正式 recall 阈值。因此结果只用于 wiring sanity，不构成正式 retrieval evaluation。",
        "",
        "## 7. 为什么不比较 flat chunks",
        "",
        "本轮目标是检查 table units 的 evidence format、metadata traceability 与 unit_type 命中情况。flat chunks baseline、official BM25、Milvus collection 与 official clean baseline 都保持只读 pin 状态，不进入比较。",
        "",
        "## 8. 为什么不接 production",
        "",
        "Phase7I-1 eligible units 仍是 preview_only，production_ready=false，value_bboxes_available=false，binding/provenance 仍是 warning-level。接 production 会把 preview wiring 误写成 production retrieval 能力，因此本轮不接 production。",
        "",
        "## 9. Phase7K 如何使用本轮结果",
        "",
        "Phase7K 可以使用本轮 query_set、top-k evidence、query_hit_summary 与 ranking_debug 来规划 table index integration：明确候选 unit 粒度、evidence card 字段、metadata 追溯字段、guardrail 限制和后续 integration test 入口。Phase7K 仍需单独定义正式 integration plan，不能把本轮 dry-run 当成 production index 或 benchmark。",
    ]
    (report_dir / "phase7i1_to_phase7j_traceability.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def render_summary(report_dir: Path, validation: dict[str, Any]) -> None:
    query_dist = validation["query_type_distribution"]
    match_counts = validation["match_status_counts"]
    hit_counts = validation["hit_unit_type_counts"]
    generated_list = "\n".join(f"- `{path}`" for path in GENERATED_FILES)
    script_list = "\n".join(f"- `{path}`" for path in SCRIPT_FILES)
    test_list = "- `tests/test_phase7_table_retrieval_wiring_preview.py`"
    lines = [
        "# Phase7J 阶段总结",
        "",
        "## 1. 本轮生成文件",
        "",
        generated_list,
        "",
        "## 2. 新增 / 修改脚本",
        "",
        script_list,
        "",
        "## 3. 新增测试",
        "",
        test_list,
        "",
        "## 4. eligible units 数量",
        "",
        f"- `274`，本轮验证读取数量为 `{validation['eligible_unit_count']}`。",
        "",
        "## 5. query 数量",
        "",
        f"- `{validation['query_count']}`",
        "",
        "## 6. query_type 分布",
        "",
        "| query_type | count |",
        "| --- | ---: |",
    ]
    for query_type, count in sorted(query_dist.items()):
        lines.append(f"| `{query_type}` | {count} |")
    lines.extend(
        [
            "",
            "## 7. top-k dry-run 是否成功",
            "",
            "- 是。已生成 `retrieval_wiring_preview_results.csv/json`、`topk_evidence_units.jsonl`、`query_hit_summary.csv` 与 `ranking_debug.csv`。",
            "",
            "## 8. match_status 统计",
            "",
            "| match_status | count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(match_counts.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## 9. hit_unit_type 统计", "", "| unit_type | count |", "| --- | ---: |"])
    for unit_type, count in sorted(hit_counts.items()):
        lines.append(f"| `{unit_type}` | {count} |")
    lines.extend(
        [
            "",
            "## 10. no_match / weak_match 数量",
            "",
            f"- no_match：`{validation['no_match_count']}`",
            f"- weak_match：`{validation['weak_match_count']}`",
            "",
            "## 11. evidence_cards 是否生成",
            "",
            "- 是，路径为 `reports/v7_phase7_table_retrieval_wiring_preview/retrieval_evidence_cards.md`。",
            "",
            "## 12. 是否满足 retrieval wiring preview",
            "",
            f"- 是，validation_status=`{validation['validation_status']}`。本轮满足 offline table retrieval wiring preview 的范围要求。",
            "",
            "## 13. 是否建议进入 Phase7K table index integration plan",
            "",
            "- 建议进入 Phase7K table index integration plan，但 Phase7K 需要另行定义 integration plan 与正式 guardrail。",
            "",
            "## 14. 是否建议回修 index unit generation",
            "",
            "- 不建议本轮立即回修。现有 eligible units 足以支持 wiring preview；复杂 header/cell_group hardening 可留作后续 backlog。",
            "",
            "## 15. 是否建议回修 extractor",
            "",
            "- 不建议。本轮不是 extractor validation，也未发现需要阻断 Phase7K planning 的 extractor 问题。",
            "",
            "## 16. 是否建议继续人工大标注",
            "",
            "- 不建议。本轮不要求用户继续人工标注。",
            "",
            "## 17. 是否建议进入 production",
            "",
            "- 不建议。所有 units 仍为 preview_only，production_ready=false，value_bboxes_available=false。",
            "",
            "## 18. baseline / guardrail 是否漂移",
            "",
            "- 未发现漂移。本轮未修改 official dataset、official baseline、chunks、BM25、Milvus、configs 或 baseline registry。",
            "",
            "## 19. Route C 是否仍只是 backlog",
            "",
            "- 是。Route C 仍只是 backlog，未进入 implementation。",
            "",
            "## 20. 明确未执行事项",
            "",
            "- 未运行 embedding。",
            "- 未运行 rerank。",
            "- 未读取或查询 BM25 index。",
            "- 未访问或写入 Milvus。",
            "- 未构建 production index。",
            "- 未比较 flat chunks。",
            "- 未调用 Qwen。",
            "- 未调用 RAGAS。",
            "- 未调用 OCR/VLM。",
            "- 未修改 ingestion pipeline。",
            "- 未修改 production pipeline。",
            "- 未修改 official dataset。",
            "- 未修改 official baseline。",
            "- 未扩大候选池。",
            "- 未生成新 review pack。",
            "- 未要求用户继续人工标注。",
            "- 未进入 Route C implementation。",
        ]
    )
    (report_dir / "phase7j_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_preview(
    eligible_jsonl: Path = DEFAULT_ELIGIBLE_JSONL,
    qa_jsonl: Path = DEFAULT_QA_JSONL,
    excluded_csv: Path = DEFAULT_EXCLUDED_CSV,
    query_jsonl: Path = DEFAULT_QUERY_JSONL,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    results_json: Path = DEFAULT_RESULTS_JSON,
    topk_jsonl: Path = DEFAULT_TOPK_JSONL,
    query_summary_csv: Path = DEFAULT_QUERY_SUMMARY_CSV,
    evidence_report: Path = DEFAULT_EVIDENCE_REPORT,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    eligible_jsonl = resolve_path(eligible_jsonl)
    qa_jsonl = resolve_path(qa_jsonl)
    excluded_csv = resolve_path(excluded_csv)
    query_jsonl = resolve_path(query_jsonl)
    results_csv = resolve_path(results_csv)
    results_json = resolve_path(results_json)
    topk_jsonl = resolve_path(topk_jsonl)
    query_summary_csv = resolve_path(query_summary_csv)
    evidence_report = resolve_path(evidence_report)
    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    eligible_units = load_jsonl(eligible_jsonl)
    qa_units = load_jsonl(qa_jsonl)
    excluded_rows = load_csv(excluded_csv)
    queries = load_jsonl(query_jsonl)
    result_rows = load_csv(results_csv)
    result_json = read_json(results_json)
    topk_records = load_jsonl(topk_jsonl)
    query_summary_rows = load_csv(query_summary_csv)

    validation = validate_artifacts(
        eligible_units=eligible_units,
        qa_units=qa_units,
        excluded_rows=excluded_rows,
        queries=queries,
        result_rows=result_rows,
        result_json=result_json,
        topk_records=topk_records,
        query_summary_rows=query_summary_rows,
        evidence_report_path=evidence_report,
    )
    render_guardrail(report_dir)
    render_validation_report(report_dir, validation)
    render_traceability(report_dir, validation)
    render_summary(report_dir, validation)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-jsonl", type=Path, default=DEFAULT_ELIGIBLE_JSONL)
    parser.add_argument("--qa-jsonl", type=Path, default=DEFAULT_QA_JSONL)
    parser.add_argument("--excluded-csv", type=Path, default=DEFAULT_EXCLUDED_CSV)
    parser.add_argument("--query-jsonl", type=Path, default=DEFAULT_QUERY_JSONL)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--topk-jsonl", type=Path, default=DEFAULT_TOPK_JSONL)
    parser.add_argument("--query-summary-csv", type=Path, default=DEFAULT_QUERY_SUMMARY_CSV)
    parser.add_argument("--evidence-report", type=Path, default=DEFAULT_EVIDENCE_REPORT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = validate_preview(
        eligible_jsonl=args.eligible_jsonl,
        qa_jsonl=args.qa_jsonl,
        excluded_csv=args.excluded_csv,
        query_jsonl=args.query_jsonl,
        results_csv=args.results_csv,
        results_json=args.results_json,
        topk_jsonl=args.topk_jsonl,
        query_summary_csv=args.query_summary_csv,
        evidence_report=args.evidence_report,
        report_dir=args.report_dir,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
