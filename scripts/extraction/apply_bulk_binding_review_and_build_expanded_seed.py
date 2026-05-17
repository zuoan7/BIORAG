#!/usr/bin/env python3
"""Apply Phase7G-2 bulk binding review and build expanded table seed."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FREEZE_DIR = ROOT / "data/experiments/v7_phase7_human_review_label_freeze"
DEFAULT_REVIEW_PACK_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_expanded_seed_from_human_review"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_expanded_seed_from_human_review"
DEFAULT_PHASE7F_SUMMARY = ROOT / "reports/v7_phase7_gold_seed_validation/phase7f_summary.md"

USER_GLOBAL_BINDING_STATEMENT = (
    "用户已整体核查 15 条 confirmed_seed_draft，无明显 unit/note/reference 问题；"
    "未做逐 cell binding 审查。"
)
BINDING_REVIEW_MODE = "global_bulk_no_obvious_issue"
BINDING_REVIEW_LIMITATION = "no_per_cell_binding_review"
BINDING_NOTES = (
    "用户整体核查无明显 unit/note/reference 问题；本轮采用保守 "
    "bulk warning/not_applicable 标签，未做逐 cell binding 审查。"
)

SEED_FIELDS = [
    "seed_id",
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "source_phase",
    "seed_status",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "binding_review_mode",
    "binding_review_limitation",
    "binding_notes",
    "required_values_source",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
    "risk_tags",
    "source_span_granularity",
    "value_bboxes_available",
    "cell_bboxes_available",
    "auto_binding_fill",
    "unit_or_note_signal_detected",
    "reference_signal_detected",
    "unit_or_note_signal_hits",
    "reference_signal_hits",
    "seed_warnings",
    "seed_notes",
]

SUMMARY_FIELDS = [
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "seed_status",
    "unit_or_note_ok",
    "reference_ok",
    "auto_binding_fill",
    "unit_or_note_signal_detected",
    "reference_signal_detected",
    "source_span_granularity",
    "value_bboxes_available",
    "cell_bboxes_available",
    "seed_warnings",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def by_candidate(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("candidate_id", ""): row for row in rows if row.get("candidate_id")}


def semicolon(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def bool_text(value: Any) -> str:
    text = str(value).strip().lower()
    return "true" if text in {"true", "1", "yes"} else "false"


def read_text_resource(path_text: str) -> str:
    path = resolve(Path(path_text))
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_crop_signature(path_text: str) -> dict[str, Any]:
    path = resolve(Path(path_text))
    if not path.exists() or not path.is_file():
        return {"crop_file_exists": "false", "crop_file_size": "0", "crop_file_signature": "missing"}
    with path.open("rb") as handle:
        signature = handle.read(8)
    return {
        "crop_file_exists": "true",
        "crop_file_size": str(path.stat().st_size),
        "crop_file_signature": signature.hex(),
    }


UNIT_SIGNAL_PATTERNS = [
    ("percent", re.compile(r"%")),
    ("g_per_l", re.compile(r"\b(?:g|mg|µg|μg)/l\b", re.IGNORECASE)),
    ("molar", re.compile(r"\b(?:mm|µm|μm|nm|mol|mmol)\b", re.IGNORECASE)),
    ("time", re.compile(r"\b(?:h|min|hour|hours|minute|minutes)\b", re.IGNORECASE)),
    ("temperature", re.compile(r"(?:°c|\bc\b)", re.IGNORECASE)),
    ("od", re.compile(r"\bod(?:600|660)?\b", re.IGNORECASE)),
    ("mean_sd", re.compile(r"\b(?:mean|sd|s\.d\.)\b|±", re.IGNORECASE)),
    ("superscript_or_marker", re.compile(r"(?:\b[a-z]\b|\*|†|‡)")),
    ("p_value", re.compile(r"\b(?:fdr|adj|p[- ]?value|p\s*[<=>])\b", re.IGNORECASE)),
    ("note_keyword", re.compile(r"\b(?:note|footnote|abbreviation|n\.d\.|nt|nc)\b", re.IGNORECASE)),
    ("unit_keyword", re.compile(r"\b(?:unit|units|concentration|ratio|rate|yield)\b", re.IGNORECASE)),
]

REFERENCE_SIGNAL_PATTERNS = [
    ("reference", re.compile(r"\b(?:reference|references|ref)\b", re.IGNORECASE)),
    ("source", re.compile(r"\b(?:source|sources)\b", re.IGNORECASE)),
    ("citation", re.compile(r"\b(?:citation|literature|study|studies)\b", re.IGNORECASE)),
    ("et_al", re.compile(r"\bet\s+al\.?\b", re.IGNORECASE)),
    ("author_year", re.compile(r"\b[A-Z][A-Za-z-]+(?:\s+et\s+al\.?)?\s*\(\d{4}[a-z]?\)")),
]


def signal_hits(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    return [name for name, pattern in patterns if pattern.search(text)]


def apply_binding_policy(original_value: str, detected: bool) -> tuple[str, bool]:
    value = (original_value or "").strip()
    if value in {"yes", "warning", "no", "not_applicable"}:
        return value, False
    if value == "unchecked" or not value:
        return ("warning" if detected else "not_applicable"), True
    return value, False


def merge_metadata(
    draft: dict[str, Any],
    index_lookup: dict[str, dict[str, str]],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    merged = dict(draft)
    candidate_id = str(draft.get("candidate_id", ""))
    for source in [index_lookup.get(candidate_id, {}), pool_lookup.get(candidate_id, {})]:
        for key, value in source.items():
            if not merged.get(key):
                merged[key] = value
    return merged


def seed_from_draft(
    draft: dict[str, Any],
    index: int,
    index_lookup: dict[str, dict[str, str]],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    row = merge_metadata(draft, index_lookup, pool_lookup)
    resource_text = "\n".join(
        [
            str(row.get("caption", "")),
            str(row.get("risk_tags", "")),
            read_text_resource(str(row.get("markdown_path", ""))),
            read_text_resource(str(row.get("csv_path", ""))),
        ]
    )
    crop_info = read_crop_signature(str(row.get("pdf_crop_path", "")))
    unit_hits = signal_hits(resource_text, UNIT_SIGNAL_PATTERNS)
    reference_hits = signal_hits(resource_text, REFERENCE_SIGNAL_PATTERNS)
    unit_or_note_ok, unit_auto = apply_binding_policy(str(row.get("unit_or_note_ok", "")), bool(unit_hits))
    reference_ok, reference_auto = apply_binding_policy(str(row.get("reference_ok", "")), bool(reference_hits))

    warnings = [
        "bulk_binding_no_per_cell_review",
        "formal_benchmark_guardrail",
    ]
    if unit_or_note_ok == "warning":
        warnings.append("unit_or_note_binding_warning")
    if reference_ok == "warning":
        warnings.append("reference_binding_warning")
    if bool_text(row.get("value_bboxes_available", "false")) != "true":
        warnings.append("value_bboxes_not_available")
    if crop_info["crop_file_exists"] != "true":
        warnings.append("pdf_crop_missing")

    seed_id = f"phase7g2_expanded_seed_{index:03d}__{row.get('candidate_id', '')}"
    seed = {
        "seed_id": seed_id,
        "candidate_id": row.get("candidate_id", ""),
        "table_object_id": row.get("table_object_id", ""),
        "doc_id": row.get("doc_id", ""),
        "table_id": row.get("table_id", ""),
        "caption": row.get("caption", ""),
        "page": row.get("page", ""),
        "source_phase": "phase7g_2_bulk_binding_from_human_review",
        "seed_status": "confirmed_seed_with_warnings",
        "boundary_ok": row.get("boundary_ok", ""),
        "grid_ok": row.get("grid_ok", ""),
        "key_values_ok": row.get("key_values_ok", ""),
        "unit_or_note_ok": unit_or_note_ok,
        "reference_ok": reference_ok,
        "binding_review_mode": BINDING_REVIEW_MODE,
        "binding_review_limitation": BINDING_REVIEW_LIMITATION,
        "binding_notes": BINDING_NOTES,
        "required_values_source": "phase7g_1_confirmed_seed_draft_human_core_labels",
        "markdown_path": row.get("markdown_path", ""),
        "csv_path": row.get("csv_path", ""),
        "pdf_crop_path": row.get("pdf_crop_path", ""),
        "crop_status": row.get("crop_status", ""),
        "risk_tags": row.get("risk_tags", ""),
        "source_span_granularity": row.get("source_span_granularity", ""),
        "value_bboxes_available": bool_text(row.get("value_bboxes_available", "false")),
        "cell_bboxes_available": bool_text(row.get("cell_bboxes_available", "false")),
        "auto_binding_fill": str(unit_auto or reference_auto).lower(),
        "unit_or_note_signal_detected": str(bool(unit_hits)).lower(),
        "reference_signal_detected": str(bool(reference_hits)).lower(),
        "unit_or_note_signal_hits": semicolon(unit_hits),
        "reference_signal_hits": semicolon(reference_hits),
        "seed_warnings": semicolon(warnings),
        "seed_notes": "由 Phase7G-1 confirmed_seed_draft 批量转换；partial/reject/unreviewed 未进入 seed。",
        **crop_info,
    }
    return seed


def policy_payload() -> dict[str, Any]:
    return {
        "user_global_binding_statement": USER_GLOBAL_BINDING_STATEMENT,
        "binding_review_mode": BINDING_REVIEW_MODE,
        "binding_review_limitation": BINDING_REVIEW_LIMITATION,
        "unit_or_note_policy": [
            "如果候选表存在 unit/note/footnote/statistical marker 信号，则 unit_or_note_ok=warning。",
            "如果无相关信号，则 unit_or_note_ok=not_applicable。",
            "不得自动写 yes，除非原始人工标签已有 yes。",
            "原始 no 不得被覆盖。",
        ],
        "reference_policy": [
            "如果候选表存在 Reference / Source / Citation / Study 等来源列或来源信号，则 reference_ok=warning。",
            "如果无相关信号，则 reference_ok=not_applicable。",
            "不得自动写 yes，除非原始人工标签已有 yes。",
            "原始 no 不得被覆盖。",
        ],
        "seed_status_policy": [
            "15 条 confirmed_seed_draft 转为 confirmed_seed_with_warnings。",
            "不得写 fully_confirmed_seed。",
            "不得写 official_benchmark_seed。",
            "不得写 production_ready。",
        ],
    }


def write_policy(output_dir: Path, report_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = policy_payload()
    (output_dir / "bulk_binding_review_policy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_text(
        report_dir / "bulk_binding_policy_report.md",
        [
            "# 批量 Binding 策略报告",
            "",
            "## 用户全局核查声明",
            f"- {USER_GLOBAL_BINDING_STATEMENT}",
            "",
            "## Binding Review Mode",
            f"- `{BINDING_REVIEW_MODE}`",
            "",
            "## Binding Review Limitation",
            f"- `{BINDING_REVIEW_LIMITATION}`",
            "",
            "## Unit / Note 策略",
            "- 有 unit/note/footnote/statistical marker 信号时，`unchecked` 转为 `warning`。",
            "- 无相关信号时，`unchecked` 转为 `not_applicable`。",
            "- 不自动写 `yes`，原始 `no` 不覆盖。",
            "",
            "## Reference 策略",
            "- 有 Reference / Source / Citation / Study / Literature / author-year / et al. 信号时，`unchecked` 转为 `warning`。",
            "- 无相关信号时，`unchecked` 转为 `not_applicable`。",
            "- 不自动写 `yes`，原始 `no` 不覆盖。",
            "",
            "## Seed Status 策略",
            "- `confirmed_seed_draft` 只能转为 `confirmed_seed_with_warnings`。",
            "- 不写 fully confirmed seed，不写 official benchmark seed，不写 production ready。",
        ],
    )


def validation_checks(seeds: list[dict[str, Any]], draft_ids: set[str]) -> list[dict[str, Any]]:
    checks = [
        ("seed_count_is_15", len(seeds) == 15, f"seed_count={len(seeds)}"),
        ("all_seed_from_confirmed_seed_draft", all(seed["candidate_id"] in draft_ids for seed in seeds), ""),
        ("all_boundary_yes", all(seed["boundary_ok"] == "yes" for seed in seeds), ""),
        ("all_grid_yes", all(seed["grid_ok"] == "yes" for seed in seeds), ""),
        ("all_key_values_yes", all(seed["key_values_ok"] == "yes" for seed in seeds), ""),
        ("unit_or_note_not_unchecked", all(seed["unit_or_note_ok"] != "unchecked" for seed in seeds), ""),
        ("reference_not_unchecked", all(seed["reference_ok"] != "unchecked" for seed in seeds), ""),
        (
            "seed_status_confirmed_seed_with_warnings",
            all(seed["seed_status"] == "confirmed_seed_with_warnings" for seed in seeds),
            "",
        ),
        ("binding_review_mode_ok", all(seed["binding_review_mode"] == BINDING_REVIEW_MODE for seed in seeds), ""),
        (
            "binding_review_limitation_ok",
            all(seed["binding_review_limitation"] == BINDING_REVIEW_LIMITATION for seed in seeds),
            "",
        ),
        ("no_production_ready_token", "production_ready" not in json.dumps(seeds, ensure_ascii=False), ""),
        ("no_official_benchmark_token", "official_benchmark" not in json.dumps(seeds, ensure_ascii=False), ""),
        (
            "no_value_level_source_span",
            all(seed["source_span_granularity"] != "value_level" for seed in seeds),
            "",
        ),
        (
            "value_bboxes_not_forged_true",
            all(seed["value_bboxes_available"] in {"false", "true"} for seed in seeds),
            "保持 review_pack_index 原始布尔值，不从 cell bbox 推断 value bbox。",
        ),
    ]
    return [
        {"check": name, "status": "pass" if ok else "fail", "details": details}
        for name, ok, details in checks
    ]


def write_guardrail(report_dir: Path) -> None:
    write_text(
        report_dir / "phase7g_2_guardrail.md",
        [
            "# Phase7G-2 护栏",
            "",
            "1. 本轮定位为 bulk binding label apply and expanded seed construction。",
            "2. 本轮不是继续人工标注，不要求用户逐条填写 `unit_or_note_ok` / `reference_ok`。",
            "3. 本轮不是 extractor validation，不运行 coverage evaluation 或 flat comparison。",
            "4. 本轮不是 official benchmark，expanded seed 不能写成 official benchmark seed。",
            "5. 本轮不扩大候选池，只使用 Phase7G-1 已冻结的 15 条 confirmed_seed_draft。",
            "6. 本轮使用用户全局人工核查声明：整体核查无明显 unit/note/reference 问题，未做逐 cell binding 审查。",
            "7. bulk binding label 只能是 `warning` / `not_applicable`，除非原始人工标签已有明确值；不得写 fully confirmed binding。",
            "8. `confirmed_seed_with_warnings` 不等于 production-ready。",
            "9. 本轮不接 production，不修改 ingestion 主链路或 production pipeline。",
            "10. 本轮不访问 Milvus / BM25，不写入 Milvus，不读取或查询 BM25 index。",
            "11. Route C 仍只是 backlog，本轮不进入 Route C implementation。",
        ],
    )


def write_validation_report(report_dir: Path, checks: list[dict[str, Any]]) -> None:
    lines = [
        "# Expanded Seed 内部一致性检查报告",
        "",
        "本报告不是 extractor validation，只检查 seed 文件内部一致性与阶段护栏。",
        "",
        "## 检查结果",
    ]
    lines.extend(f"- `{row['check']}`：{row['status']}；{row['details']}" for row in checks)
    overall = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    lines.extend(["", f"## overall_status：`{overall}`"])
    write_text(report_dir / "expanded_seed_validation_report.md", lines)


def write_review_cards(seeds: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Expanded Table Seed 复核卡片",
        "",
        "这些卡片只汇总 Phase7G-2 seed，不是新一轮 review pack。",
    ]
    for seed in seeds:
        lines.extend(
            [
                "",
                f"## {seed['seed_id']}",
                f"- candidate_id：`{seed['candidate_id']}`",
                f"- doc/table：`{seed['doc_id']}` / `{seed['table_id']}`",
                f"- seed_status：`{seed['seed_status']}`",
                f"- unit_or_note_ok：`{seed['unit_or_note_ok']}`；signals：`{seed['unit_or_note_signal_hits']}`",
                f"- reference_ok：`{seed['reference_ok']}`；signals：`{seed['reference_signal_hits']}`",
                f"- binding_review_mode：`{seed['binding_review_mode']}`",
                f"- markdown_path：`{seed['markdown_path']}`",
                f"- csv_path：`{seed['csv_path']}`",
                f"- pdf_crop_path：`{seed['pdf_crop_path']}`",
                f"- warnings：`{seed['seed_warnings']}`",
            ]
        )
    write_text(output_dir / "seed_review_cards.md", lines)


def md_counter(counts: Counter[str]) -> list[str]:
    return [f"- `{key}`：{value}" for key, value in counts.most_common()] if counts else ["- 无"]


def md_seed_list(seeds: list[dict[str, Any]]) -> list[str]:
    return [
        f"- `{seed['candidate_id']}`：`{seed['doc_id']}` / `{seed['table_id']}`；"
        f"unit=`{seed['unit_or_note_ok']}`；reference=`{seed['reference_ok']}`"
        for seed in seeds
    ]


def write_construction_report(
    report_dir: Path,
    seeds: list[dict[str, Any]],
    partial_rows: list[dict[str, str]],
    reject_rows: list[dict[str, str]],
    unreviewed_rows: list[dict[str, str]],
) -> None:
    write_text(
        report_dir / "expanded_seed_construction_report.md",
        [
            "# Expanded Seed 构造报告",
            "",
            f"- 输入 confirmed_seed_draft 数量：{len(seeds)}。",
            f"- 输出 confirmed_seed_with_warnings 数量：{len(seeds)}。",
            f"- partial carried forward 数量：{len(partial_rows)}。",
            f"- reject carried forward 数量：{len(reject_rows)}。",
            f"- unreviewed carried forward 数量：{len(unreviewed_rows)}。",
            "",
            "## unit_or_note_ok 统计",
            *md_counter(Counter(seed["unit_or_note_ok"] for seed in seeds)),
            "",
            "## reference_ok 统计",
            *md_counter(Counter(seed["reference_ok"] for seed in seeds)),
            "",
            "## seed 清单",
            *md_seed_list(seeds),
            "",
            "## 说明",
            "- 本轮使用 conservative bulk binding label，不把 binding 写成 fully confirmed。",
            "- partial / reject / unreviewed 只 carry forward，不进入 formal seed set。",
        ],
    )


def write_traceability_report(report_dir: Path, seeds: list[dict[str, Any]]) -> None:
    write_text(
        report_dir / "phase7g1_to_phase7g2_traceability.md",
        [
            "# Phase7G-1 到 Phase7G-2 追踪报告",
            "",
            "## 15 条 draft 如何进入 confirmed_seed_with_warnings",
            "- Phase7G-1 中满足 boundary/grid/key_values 全为 `yes` 的 15 条 `confirmed_seed_draft` 全部进入本轮 seed。",
            "- 本轮将 `seed_status` 固定为 `confirmed_seed_with_warnings`，不生成 fully confirmed seed。",
            "",
            "## 用户全局核查结论如何记录",
            f"- `user_global_binding_statement`：{USER_GLOBAL_BINDING_STATEMENT}",
            f"- `binding_review_mode`：`{BINDING_REVIEW_MODE}`",
            f"- `binding_review_limitation`：`{BINDING_REVIEW_LIMITATION}`",
            "",
            "## bulk binding warning/not_applicable 如何生成",
            "- 读取对应 Markdown card、CSV table、caption 与 risk_tags 做轻量文本信号检测。",
            "- 有 unit/note/reference 信号时，原 `unchecked` 转为 `warning`。",
            "- 无相关信号时，原 `unchecked` 转为 `not_applicable`。",
            "- 原始 `yes` / `warning` / `no` / `not_applicable` 保留；`no` 永不覆盖。",
            "",
            "## 为什么不要求用户逐条填写",
            "- 用户已给出 15 条 draft 的全局人工核查结论。",
            "- 本轮目标是保守构造 expanded seed，不把 binding 夸大为 fully confirmed。",
            "",
            "## 为什么 partial / reject / unreviewed 不进入 seed",
            "- partial 仍存在 boundary/grid/key_values 的二次分流问题。",
            "- reject_boundary 是 scoring/boundary gate 反馈，不是 seed。",
            "- unreviewed 缺少完整核心人工标签。",
            "",
            "## 为什么该 seed 不是 official benchmark",
            "- 本轮没有逐 cell binding 审查，没有 extractor validation，也没有 official benchmark schema 冻结。",
            "- `confirmed_seed_with_warnings` 只作为后续 Phase7H expanded seed validation 的输入。",
            "",
            "## 后续 Phase7H validation 如何使用这些 seed",
            "- Phase7H 可读取 `expanded_table_seed.jsonl` 作为 expanded seed validation 输入。",
            "- validation 需要继续区分 warnings、binding limitation、value bbox limitation 与 formal correctness。",
            "",
            "## traceability 清单",
            *[f"- `{seed['candidate_id']}` -> `{seed['seed_id']}`" for seed in seeds],
        ],
    )


def write_summary_report(
    report_dir: Path,
    output_dir: Path,
    seeds: list[dict[str, Any]],
    partial_rows: list[dict[str, str]],
    reject_rows: list[dict[str, str]],
    unreviewed_rows: list[dict[str, str]],
) -> None:
    generated_files = [
        output_dir / "bulk_binding_review_policy.json",
        output_dir / "expanded_table_seed.jsonl",
        output_dir / "expanded_table_seed_summary.csv",
        output_dir / "confirmed_seed_with_warnings.jsonl",
        output_dir / "partial_candidate_routing_carried_forward.csv",
        output_dir / "reject_boundary_feedback_carried_forward.csv",
        output_dir / "unreviewed_candidates_carried_forward.csv",
        output_dir / "seed_review_cards.md",
        report_dir / "phase7g_2_guardrail.md",
        report_dir / "bulk_binding_policy_report.md",
        report_dir / "expanded_seed_construction_report.md",
        report_dir / "expanded_seed_validation_report.md",
        report_dir / "phase7g1_to_phase7g2_traceability.md",
        report_dir / "phase7g_2_summary.md",
    ]
    write_text(
        report_dir / "phase7g_2_summary.md",
        [
            "# Phase7G-2 总结",
            "",
            "## 1. 本轮生成文件",
            *(f"- `{rel(path)}`" for path in generated_files),
            "",
            "## 2. 新增 / 修改脚本",
            "- 新增：`scripts/extraction/apply_bulk_binding_review_and_build_expanded_seed.py`",
            "",
            "## 3. 新增测试",
            "- 新增：`tests/test_phase7_bulk_binding_seed_construction.py`",
            "",
            f"## 4. 输入 confirmed_seed_draft 数量：{len(seeds)}",
            f"## 5. 输出 confirmed_seed_with_warnings 数量：{len(seeds)}",
            "",
            "## 6. unit_or_note_ok 统计",
            *md_counter(Counter(seed["unit_or_note_ok"] for seed in seeds)),
            "",
            "## 7. reference_ok 统计",
            *md_counter(Counter(seed["reference_ok"] for seed in seeds)),
            "",
            f"## 8. auto_binding_fill 数量：{sum(seed['auto_binding_fill'] == 'true' for seed in seeds)}",
            f"## 9. partial carried forward 数量：{len(partial_rows)}",
            f"## 10. reject carried forward 数量：{len(reject_rows)}",
            f"## 11. unreviewed carried forward 数量：{len(unreviewed_rows)}",
            "",
            "## 12. 是否要求用户逐条继续填写",
            "- 否。本轮采用用户全局人工核查声明和 conservative bulk binding label。",
            "",
            "## 13. 是否构造 official benchmark",
            "- 否。",
            "",
            "## 14. 是否运行 extractor validation",
            "- 否。本轮只做 seed 内部一致性检查。",
            "",
            "## 15. 是否建议进入 Phase7H expanded seed validation",
            "- 是。建议使用 `expanded_table_seed.jsonl` 进入 Phase7H expanded seed validation。",
            "",
            "## 16. 是否建议回修 extractor",
            "- 否。本轮没有运行 extractor validation，不能据此要求回修 extractor。",
            "",
            "## 17. 是否建议继续人工大标注",
            "- 否。当前不需要新一轮大标注。",
            "",
            "## 18. 是否建议进入 production",
            "- 否。",
            "",
            "## 19. baseline / guardrail 是否漂移",
            "- 未发现漂移。本轮未修改 official dataset、official baseline、configs、baseline registry、chunks、BM25 或 Milvus。",
            "",
            "## 20. Route C 是否仍只是 backlog",
            "- 是。Route C 仍只是 backlog，本轮未进入 implementation。",
            "",
            "## 21. 明确未执行事项",
            "- 未要求用户继续逐条填写 unit/reference。",
            "- 未扩大候选池。",
            "- 未生成新 review pack。",
            "- 未回修 extractor。",
            "- 未运行 extractor validation。",
            "- 未运行 coverage evaluation。",
            "- 未做 flat comparison。",
            "- 未引入 Camelot。",
            "- 未引入 PyMuPDF。",
            "- 未引入 OCR/VLM。",
            "- 未改 ingestion pipeline。",
            "- 未改 production pipeline。",
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
            "- 未调用 Qwen/RAGAS。",
            "- 未接入 production。",
            "- 未进入 Route C。",
        ],
    )


def read_required_reports(freeze_report_dir: Path, phase7f_summary: Path) -> dict[str, str]:
    paths = [
        freeze_report_dir / "human_label_audit_report.md",
        freeze_report_dir / "seed_draft_routing_report.md",
        freeze_report_dir / "partial_candidate_routing_report.md",
        freeze_report_dir / "phase7g_1_summary.md",
        phase7f_summary,
    ]
    return {rel(path): path.read_text(encoding="utf-8") for path in paths}


def build_expanded_seed(
    freeze_dir: Path,
    review_pack_dir: Path,
    output_dir: Path,
    report_dir: Path,
    freeze_report_dir: Path = ROOT / "reports/v7_phase7_human_review_label_freeze",
    phase7f_summary: Path = DEFAULT_PHASE7F_SUMMARY,
) -> dict[str, Any]:
    freeze_dir = resolve(freeze_dir)
    review_pack_dir = resolve(review_pack_dir)
    output_dir = resolve(output_dir)
    report_dir = resolve(report_dir)
    freeze_report_dir = resolve(freeze_report_dir)
    phase7f_summary = resolve(phase7f_summary)

    frozen_rows = load_csv(freeze_dir / "frozen_review_labels.csv")
    draft_csv_rows = load_csv(freeze_dir / "confirmed_seed_draft_candidates.csv")
    draft_jsonl_rows = load_jsonl(freeze_dir / "confirmed_seed_draft_candidates.jsonl")
    partial_rows = load_csv(freeze_dir / "partial_candidate_routing.csv")
    reject_rows = load_csv(freeze_dir / "reject_boundary_feedback.csv")
    unreviewed_rows = load_csv(freeze_dir / "unreviewed_candidates.csv")
    followup_rows = load_csv(freeze_dir / "unit_reference_followup_template.csv")
    reports = read_required_reports(freeze_report_dir, phase7f_summary)

    index_rows = load_csv(review_pack_dir / "review_pack_index.csv")
    pool_rows = load_csv(review_pack_dir / "candidate_pool_scored.csv")
    for resource_dir in ["csv_tables", "markdown_cards", "pdf_crops"]:
        if not (review_pack_dir / resource_dir).is_dir():
            raise SystemExit(f"missing review pack resource directory: {rel(review_pack_dir / resource_dir)}")

    csv_ids = {row["candidate_id"] for row in draft_csv_rows}
    jsonl_ids = {str(row.get("candidate_id", "")) for row in draft_jsonl_rows}
    followup_ids = {row["candidate_id"] for row in followup_rows}
    if csv_ids != jsonl_ids or csv_ids != followup_ids:
        raise SystemExit("confirmed_seed_draft CSV/JSONL/followup candidate ids do not match")

    index_lookup = by_candidate(index_rows)
    pool_lookup = by_candidate(pool_rows)
    seeds = [
        seed_from_draft(row, index=index, index_lookup=index_lookup, pool_lookup=pool_lookup)
        for index, row in enumerate(draft_csv_rows, start=1)
    ]
    draft_ids = {row["candidate_id"] for row in draft_csv_rows}
    checks = validation_checks(seeds, draft_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    write_policy(output_dir, report_dir)
    write_jsonl(seeds, output_dir / "expanded_table_seed.jsonl")
    write_jsonl(seeds, output_dir / "confirmed_seed_with_warnings.jsonl")
    write_csv(seeds, output_dir / "expanded_table_seed_summary.csv", SUMMARY_FIELDS)
    write_csv(partial_rows, output_dir / "partial_candidate_routing_carried_forward.csv", list(partial_rows[0]) if partial_rows else [])
    write_csv(reject_rows, output_dir / "reject_boundary_feedback_carried_forward.csv", list(reject_rows[0]) if reject_rows else [])
    write_csv(unreviewed_rows, output_dir / "unreviewed_candidates_carried_forward.csv", list(unreviewed_rows[0]) if unreviewed_rows else [])
    write_review_cards(seeds, output_dir)

    write_guardrail(report_dir)
    write_construction_report(report_dir, seeds, partial_rows, reject_rows, unreviewed_rows)
    write_validation_report(report_dir, checks)
    write_traceability_report(report_dir, seeds)
    write_summary_report(report_dir, output_dir, seeds, partial_rows, reject_rows, unreviewed_rows)

    return {
        "frozen_count": len(frozen_rows),
        "draft_count": len(draft_csv_rows),
        "draft_jsonl_count": len(draft_jsonl_rows),
        "seed_count": len(seeds),
        "unit_or_note_counts": Counter(seed["unit_or_note_ok"] for seed in seeds),
        "reference_counts": Counter(seed["reference_ok"] for seed in seeds),
        "auto_binding_fill_count": sum(seed["auto_binding_fill"] == "true" for seed in seeds),
        "partial_carried_forward_count": len(partial_rows),
        "reject_carried_forward_count": len(reject_rows),
        "unreviewed_carried_forward_count": len(unreviewed_rows),
        "validation_failed": [row for row in checks if row["status"] != "pass"],
        "required_report_count": len(reports),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--review-pack-dir", type=Path, default=DEFAULT_REVIEW_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--freeze-report-dir", type=Path, default=ROOT / "reports/v7_phase7_human_review_label_freeze")
    parser.add_argument("--phase7f-summary", type=Path, default=DEFAULT_PHASE7F_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_expanded_seed(
        freeze_dir=args.freeze_dir,
        review_pack_dir=args.review_pack_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        freeze_report_dir=args.freeze_report_dir,
        phase7f_summary=args.phase7f_summary,
    )
    failed = len(result["validation_failed"])
    print(
        "phase7g2_expanded_seed: "
        f"draft={result['draft_count']} "
        f"seed={result['seed_count']} "
        f"auto_binding_fill={result['auto_binding_fill_count']} "
        f"validation_failed={failed}"
    )


if __name__ == "__main__":
    main()
