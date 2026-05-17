#!/usr/bin/env python3
"""Review Phase7C-4 hybrid binding candidates.

This offline review reads Phase7C-3/Phase7C-2 artifacts and Phase6D
contracts, then writes the restricted binding-review outputs for the five
manual_review_binding candidates only. It does not run extraction, retrieval,
BM25, Milvus, OCR, VLM, or model calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

PHASE7C3_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_gate_hardening"
PHASE7C3_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_gate_hardening"
PHASE7C2_DATA_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_pilot_v2"
PHASE6D_REPORT_DIR = ROOT / "reports/v7_phase6d_table_contract_refinement"

DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_binding_review"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_binding_review"

DEFAULT_BINDING_CANDIDATES = PHASE7C3_DATA_DIR / "hybrid_candidates_for_binding_review.jsonl"
DEFAULT_HYBRID_OBJECTS = PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl"
DEFAULT_RAW_TABLES = PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl"
DEFAULT_ALIGNMENT = PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv"

REQUIRED_PHASE7C3_INPUTS = [
    PHASE7C3_DATA_DIR / "hybrid_case_decisions.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_case_decision_summary.csv",
    PHASE7C3_DATA_DIR / "hybrid_table_objects_gated.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_candidates_for_binding_review.jsonl",
    PHASE7C3_DATA_DIR / "hybrid_table_objects_gated_review.md",
    PHASE7C3_REPORT_DIR / "source_review_decision_report.md",
    PHASE7C3_REPORT_DIR / "hybrid_validation_gated_summary.csv",
    PHASE7C3_REPORT_DIR / "hybrid_validation_gated_report.md",
    PHASE7C3_REPORT_DIR / "phase7c2_vs_phase7c3_comparison.md",
    PHASE7C3_REPORT_DIR / "phase7c_3_summary.md",
]

REQUIRED_PHASE7C2_INPUTS = [
    PHASE7C2_DATA_DIR / "hybrid_table_objects.jsonl",
    PHASE7C2_DATA_DIR / "pdfplumber_tables.raw_v2.jsonl",
    PHASE7C2_DATA_DIR / "chunk_pdfplumber_alignment_v2.csv",
]

REQUIRED_PHASE6D_INPUTS = [
    PHASE6D_REPORT_DIR / "phase6d_refine_round1_summary.md",
    PHASE6D_REPORT_DIR / "numeric_unit_footnote_contract.md",
    PHASE6D_REPORT_DIR / "numeric_unit_footnote_rules.csv",
    PHASE6D_REPORT_DIR / "matrix_superscript_literal_contract.md",
    PHASE6D_REPORT_DIR / "matrix_superscript_literal_rules.csv",
    PHASE6D_REPORT_DIR / "source_span_granularity_contract.md",
    PHASE6D_REPORT_DIR / "source_span_granularity_rules.csv",
    PHASE6D_REPORT_DIR / "partial_to_confirmed_decision_guide.md",
    PHASE6D_REPORT_DIR / "partial_to_confirmed_rules.csv",
]

EXPECTED_BINDING_REVIEW_IDS = {
    "doc_0598__table_1__phase7c2_hybrid_01",
    "doc_0468__table_2__phase7c2_hybrid_01",
    "doc_0687__table_2__phase7c2_hybrid_02",
    "doc_0687__table_3__phase7c2_hybrid_03",
    "doc_0523__table_1__phase7c2_hybrid_01",
}

STATUS_VALUES = {"pass", "pass_with_warnings", "uncertain", "fail", "not_applicable"}
BINDING_REVIEW_STATUSES = {
    "usable_hybrid_candidate",
    "hybrid_candidate_with_binding_warnings",
    "hybrid_needs_rule_fix",
    "chunk_fallback",
    "exclude_or_backlog",
}
FINAL_BINDING_ACTIONS = {
    "ready_for_gold_candidate",
    "keep_for_manual_binding_review",
    "needs_pdfplumber_rule_fix",
    "use_chunk_fallback",
    "backlog",
}

READY_ACTION = "ready_for_gold_candidate"
RULE_FIX_ACTION = "needs_pdfplumber_rule_fix"

SUMMARY_FIELDS = [
    "hybrid_table_object_id",
    "doc_id",
    "table_id",
    "row_grid_status",
    "column_grid_status",
    "cell_grid_status",
    "value_placement_status",
    "unit_binding_status",
    "footnote_binding_status",
    "reference_binding_status",
    "literal_preservation_status",
    "bbox_provenance_status",
    "binding_review_status",
    "final_binding_action",
    "ready_for_gold_candidate_is_confirmed_gold",
    "usable_hybrid_candidate_is_production_ready",
    "value_bboxes_available",
    "cell_bboxes_available",
    "source_span_granularity",
    "key_warnings",
]

RULE_FIX_FIELDS = [
    "hybrid_table_object_id",
    "doc_id",
    "table_id",
    "binding_review_status",
    "final_binding_action",
    "rule_fix_scope",
    "key_warnings",
    "review_notes",
]

FALLBACK_FIELDS = [
    "hybrid_table_object_id",
    "doc_id",
    "table_id",
    "binding_review_status",
    "final_binding_action",
    "reason",
    "key_warnings",
]


@dataclass(frozen=True)
class ReviewProfile:
    hybrid_table_object_id: str
    row_grid_status: str
    column_grid_status: str
    cell_grid_status: str
    value_placement_status: str
    unit_binding_status: str
    footnote_binding_status: str
    reference_binding_status: str
    literal_preservation_status: str
    bbox_provenance_status: str
    binding_review_status: str
    final_binding_action: str
    unit_visible: bool
    unit_bound: bool
    footnote_present: bool
    footnote_bound: bool
    reference_visible: bool
    row_level_reference_bound: bool
    literal_tokens: list[str]
    rule_fix_scope: str
    key_warnings: list[str]
    review_notes: str
    gold_readiness_rationale: str


REVIEW_PROFILES = {
    "doc_0598__table_1__phase7c2_hybrid_01": ReviewProfile(
        hybrid_table_object_id="doc_0598__table_1__phase7c2_hybrid_01",
        row_grid_status="uncertain",
        column_grid_status="fail",
        cell_grid_status="fail",
        value_placement_status="uncertain",
        unit_binding_status="not_applicable",
        footnote_binding_status="not_applicable",
        reference_binding_status="not_applicable",
        literal_preservation_status="uncertain",
        bbox_provenance_status="uncertain",
        binding_review_status="hybrid_needs_rule_fix",
        final_binding_action=RULE_FIX_ACTION,
        unit_visible=False,
        unit_bound=False,
        footnote_present=False,
        footnote_bound=False,
        reference_visible=False,
        row_level_reference_bound=False,
        literal_tokens=["mvaEF", "mvaERadhE1", "CmF", "CmR"],
        rule_fix_scope="primer/name/sequence/location 列需要重新合并；长 primer row 需要稳定 row continuation 规则。",
        key_warnings=[
            "Location 被拆成 Lo/cation",
            "primer name 与 sequence 被合并在同一列",
            "长 primer row 跨多行折返",
            "hybrid object 仍是 chunk_fallback/table_row_level",
            "cell bbox 不能证明 logical value cell",
            "value bbox 不存在",
        ],
        review_notes=(
            "raw pdfplumber cell bbox 可定位页面格子，但当前 logical grid 不能稳定表达 Primer name、"
            "Primer sequence、Location 三列；不适合直接进入 gold construction。"
        ),
        gold_readiness_rationale="不进入 ready；需要先修复 column split 与 row continuation。",
    ),
    "doc_0468__table_2__phase7c2_hybrid_01": ReviewProfile(
        hybrid_table_object_id="doc_0468__table_2__phase7c2_hybrid_01",
        row_grid_status="pass_with_warnings",
        column_grid_status="pass_with_warnings",
        cell_grid_status="pass_with_warnings",
        value_placement_status="pass_with_warnings",
        unit_binding_status="pass_with_warnings",
        footnote_binding_status="pass_with_warnings",
        reference_binding_status="pass_with_warnings",
        literal_preservation_status="pass_with_warnings",
        bbox_provenance_status="pass_with_warnings",
        binding_review_status="hybrid_candidate_with_binding_warnings",
        final_binding_action=READY_ACTION,
        unit_visible=True,
        unit_bound=False,
        footnote_present=True,
        footnote_bound=False,
        reference_visible=True,
        row_level_reference_bound=True,
        literal_tokens=["DSM20083", "Bi-07", "Bl-04", "Bado_TS"],
        rule_fix_scope="无本轮阻断 rule fix；gold construction 只能在已审阅行内重组 split designation。",
        key_warnings=[
            "strain designation 被拆成 DS/M20083 等相邻 cell",
            "T*/atmosphere* footnote 只可保留 warning",
            "unit/temperature 可见但未形成 per-cell unit binding",
            "source 字段是表内 source column，不是外部 citation provenance",
            "value bbox 不存在",
        ],
        review_notes=(
            "表体行与 source/probiotic/medium/T/atmosphere/abbreviation 列整体稳定；"
            "designation split 可在后续小规模 gold construction 中作为人工绑定项处理。"
        ),
        gold_readiness_rationale="可进入 ready_for_gold_candidate，但仅限小规模人工 gold construction，不是 confirmed gold。",
    ),
    "doc_0687__table_2__phase7c2_hybrid_02": ReviewProfile(
        hybrid_table_object_id="doc_0687__table_2__phase7c2_hybrid_02",
        row_grid_status="pass_with_warnings",
        column_grid_status="uncertain",
        cell_grid_status="fail",
        value_placement_status="fail",
        unit_binding_status="uncertain",
        footnote_binding_status="pass_with_warnings",
        reference_binding_status="uncertain",
        literal_preservation_status="pass_with_warnings",
        bbox_provenance_status="uncertain",
        binding_review_status="hybrid_needs_rule_fix",
        final_binding_action=RULE_FIX_ACTION,
        unit_visible=True,
        unit_bound=False,
        footnote_present=True,
        footnote_bound=False,
        reference_visible=True,
        row_level_reference_bound=False,
        literal_tokens=["YE/S", "qethanol", "qxylose", "qarabinose"],
        rule_fix_scope="numeric 小数拆列、qxylose/reference split 与 metric-level cell reconstruction 需要规则修复。",
        key_warnings=[
            "YE/S 小数被拆成 0/.35 等多 cell",
            "qxylose header 被拆成 q/xylose",
            "Reference 被拆成 Refere/nce，行级 reference 不能稳定绑定",
            "unit visible 但未绑定到完整 metric cell",
            "metric-level cell gap 仍存在",
            "value bbox 不存在",
        ],
        review_notes=(
            "row 顺序基本可读，但 numeric column identity 依赖拆列重组；cell bbox 只能证明拆开的 cell，"
            "不能证明完整 metric value placement。"
        ),
        gold_readiness_rationale="不进入 ready；适合后续 pdfplumber rule fix 后重审。",
    ),
    "doc_0687__table_3__phase7c2_hybrid_03": ReviewProfile(
        hybrid_table_object_id="doc_0687__table_3__phase7c2_hybrid_03",
        row_grid_status="pass_with_warnings",
        column_grid_status="pass_with_warnings",
        cell_grid_status="pass_with_warnings",
        value_placement_status="pass_with_warnings",
        unit_binding_status="pass_with_warnings",
        footnote_binding_status="pass_with_warnings",
        reference_binding_status="pass_with_warnings",
        literal_preservation_status="pass_with_warnings",
        bbox_provenance_status="pass_with_warnings",
        binding_review_status="hybrid_candidate_with_binding_warnings",
        final_binding_action=READY_ACTION,
        unit_visible=True,
        unit_bound=True,
        footnote_present=True,
        footnote_bound=True,
        reference_visible=True,
        row_level_reference_bound=True,
        literal_tokens=["YE/S", "qglucose", "qethanol", "qxylose", "Reference"],
        rule_fix_scope="无本轮阻断 rule fix；gold construction 必须保留 asterisk/dagger 与 multi-row reference warning。",
        key_warnings=[
            "caption/footnote 文本被拆成多行",
            "asterisk/dagger rule 可见但仍需人工记录 applicability",
            "部分 reference 跨 row continuation",
            "cell bbox 仅支持 cell-level provenance",
            "value bbox 不存在",
        ],
        review_notes=(
            "9 列数据区从 S. cerevisiae strain 到 Reference 基本稳定；YE/S、qglucose、qethanol、"
            "qxylose 的数值在对应 cell 中保留，marker 也随 value_raw 保留。"
        ),
        gold_readiness_rationale="可进入 ready_for_gold_candidate，但后续仍需单独授权构造 confirmed gold。",
    ),
    "doc_0523__table_1__phase7c2_hybrid_01": ReviewProfile(
        hybrid_table_object_id="doc_0523__table_1__phase7c2_hybrid_01",
        row_grid_status="pass_with_warnings",
        column_grid_status="uncertain",
        cell_grid_status="uncertain",
        value_placement_status="uncertain",
        unit_binding_status="uncertain",
        footnote_binding_status="not_applicable",
        reference_binding_status="uncertain",
        literal_preservation_status="pass",
        bbox_provenance_status="uncertain",
        binding_review_status="hybrid_needs_rule_fix",
        final_binding_action=RULE_FIX_ACTION,
        unit_visible=True,
        unit_bound=False,
        footnote_present=False,
        footnote_bound=False,
        reference_visible=True,
        row_level_reference_bound=False,
        literal_tokens=["LNT II", "LNT", "N.D.", "g/L"],
        rule_fix_scope="medium/culture 与 LNT II titer 共享 cell，ref this/study 跨行，table tail 混入正文，需要规则修复。",
        key_warnings=[
            "medium/cultureconditions 与 LNT II titer 合并在 col_002",
            "g/L unit 可见但无法绑定到独立 LNT II value cell",
            "row-level reference this/study 跨行",
            "N.D. literal 已保留但 cell identity 仍不稳",
            "表尾混入正文",
            "value bbox 不存在",
        ],
        review_notes=(
            "LNT II/LNT/N.D. 值可见，且 N.D. 原文保留；但 titer value 与 medium/culture 文本共 cell，"
            "cell-level bbox 不足以证明 value-level placement。"
        ),
        gold_readiness_rationale="不进入 ready；需要先拆出 titer/value/ref 的 logical cells。",
    ),
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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def semi(values: list[str]) -> str:
    return "; ".join(values)


def read_input_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(rel(path))
            continue
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            record_count = sum(1 for line in text.splitlines() if line.strip())
        elif suffix == ".csv":
            record_count = max(0, len(text.splitlines()) - 1)
        else:
            record_count = 0
        inventory.append(
            {
                "path": rel(path),
                "line_count": len(text.splitlines()),
                "record_count": record_count,
                "bytes": len(text.encode("utf-8")),
            }
        )
    if missing:
        raise FileNotFoundError("缺少必读输入：" + "; ".join(missing))
    return inventory


def object_id(obj: dict[str, Any]) -> str:
    return str(obj.get("table_object_id") or obj.get("hybrid_table_object_id") or "")


def raw_table_id(obj: dict[str, Any]) -> str:
    return str(obj.get("pdfplumber_table_id") or obj.get("table_id") or obj.get("id") or "")


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def row_texts(obj: dict[str, Any], limit: int = 8) -> list[str]:
    rows = obj.get("rows") or []
    result: list[str] = []
    for row in rows[:limit]:
        if isinstance(row, dict):
            result.append(normalize_text(row.get("row_text") or row.get("row_label") or ""))
        elif isinstance(row, list):
            result.append(" | ".join(normalize_text(cell) for cell in row))
    return result


def table_preview_markdown(rows: list[str], limit: int = 8) -> str:
    lines = ["| # | preview |", "|---:|---|"]
    for index, row in enumerate(rows[:limit], start=1):
        preview = row[:220].replace("|", "\\|")
        lines.append(f"| {index} | {preview} |")
    if len(lines) == 2:
        lines.append("| 1 |  |")
    return "\n".join(lines)


def validate_profile(profile: ReviewProfile) -> None:
    fields = [
        profile.row_grid_status,
        profile.column_grid_status,
        profile.cell_grid_status,
        profile.value_placement_status,
        profile.unit_binding_status,
        profile.footnote_binding_status,
        profile.reference_binding_status,
        profile.literal_preservation_status,
        profile.bbox_provenance_status,
    ]
    invalid_status = [status for status in fields if status not in STATUS_VALUES]
    if invalid_status:
        raise ValueError(f"{profile.hybrid_table_object_id} 存在非法 status: {invalid_status}")
    if profile.binding_review_status not in BINDING_REVIEW_STATUSES:
        raise ValueError(f"{profile.hybrid_table_object_id} binding_review_status 非法")
    if profile.final_binding_action not in FINAL_BINDING_ACTIONS:
        raise ValueError(f"{profile.hybrid_table_object_id} final_binding_action 非法")
    if profile.final_binding_action == READY_ACTION:
        grid_statuses = [profile.row_grid_status, profile.column_grid_status, profile.cell_grid_status]
        if any(status not in {"pass", "pass_with_warnings"} for status in grid_statuses):
            raise ValueError(f"{profile.hybrid_table_object_id} ready candidate grid status 不满足要求")
        if profile.value_placement_status not in {"pass", "pass_with_warnings"}:
            raise ValueError(f"{profile.hybrid_table_object_id} ready candidate value placement 不满足要求")
        if profile.bbox_provenance_status not in {"pass", "pass_with_warnings"}:
            raise ValueError(f"{profile.hybrid_table_object_id} ready candidate bbox provenance 不满足要求")


def validate_candidates(candidates: list[dict[str, Any]]) -> None:
    ids = {row.get("hybrid_table_object_id", "") for row in candidates}
    if ids != EXPECTED_BINDING_REVIEW_IDS:
        raise ValueError(
            "本轮只能处理 5 个 binding review candidates："
            f"expected={sorted(EXPECTED_BINDING_REVIEW_IDS)} actual={sorted(ids)}"
        )
    if len(candidates) != 5:
        raise ValueError(f"binding review candidate 数量应为 5，实际为 {len(candidates)}")
    for row in candidates:
        if row.get("final_case_action") != "manual_review_binding":
            raise ValueError(f"{row.get('hybrid_table_object_id')} 不是 manual_review_binding")


def build_review_rows(
    candidates: list[dict[str, Any]],
    hybrid_objects: list[dict[str, Any]],
    raw_tables: list[dict[str, Any]],
    alignment_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    validate_candidates(candidates)
    for profile in REVIEW_PROFILES.values():
        validate_profile(profile)

    object_by_id = {object_id(obj): obj for obj in hybrid_objects}
    raw_by_id = {raw_table_id(obj): obj for obj in raw_tables}
    alignment_by_chunk_id = {row.get("chunk_table_object_id", ""): row for row in alignment_rows}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        hybrid_id = candidate["hybrid_table_object_id"]
        profile = REVIEW_PROFILES[hybrid_id]
        obj = object_by_id.get(hybrid_id)
        if obj is None:
            raise ValueError(f"缺少 hybrid_table_object: {hybrid_id}")
        metadata = obj.get("hybrid_metadata") or {}
        pdfplumber_table_id = metadata.get("pdfplumber_table_id", "")
        raw = raw_by_id.get(pdfplumber_table_id, {})
        alignment = alignment_by_chunk_id.get(metadata.get("original_chunk_table_object_id", ""), {})
        value_bboxes_available = bool(metadata.get("value_bboxes_available", False))
        if value_bboxes_available:
            raise ValueError(f"{hybrid_id} value_bboxes_available 预期应为 false")
        raw_preview = row_texts(raw, limit=10) or row_texts(obj, limit=10)
        source_span_granularity = metadata.get("source_span_granularity", "")
        row = {
            "hybrid_table_object_id": hybrid_id,
            "doc_id": candidate.get("doc_id", obj.get("doc_id", "")),
            "table_id": candidate.get("table_id", obj.get("table_id", "")),
            "pdfplumber_table_id": pdfplumber_table_id,
            "pdf_page": metadata.get("pdf_page", ""),
            "pdfplumber_strategy": metadata.get("pdfplumber_strategy", ""),
            "alignment_status": metadata.get("alignment_status", alignment.get("alignment_status", "")),
            "alignment_confidence": metadata.get("alignment_confidence", alignment.get("alignment_confidence", "")),
            "layout_quality_status": alignment.get("layout_quality_status", ""),
            "source_span_granularity": source_span_granularity,
            "cell_bboxes_available": bool(metadata.get("cell_bboxes_available", False)),
            "value_bboxes_available": False,
            "value_level_provenance_used": False,
            "bbox_provenance_level": "cell_level_only" if metadata.get("cell_bboxes_available") else "no_cell_bbox",
            "bbox_provenance_limitation": (
                "pdfplumber cell bbox 只能作为 cell-level layout provenance；"
                "value-level bbox 不存在，且本轮不推断、不伪造。"
            ),
            "row_grid_status": profile.row_grid_status,
            "column_grid_status": profile.column_grid_status,
            "cell_grid_status": profile.cell_grid_status,
            "value_placement_status": profile.value_placement_status,
            "unit_binding_status": profile.unit_binding_status,
            "footnote_binding_status": profile.footnote_binding_status,
            "reference_binding_status": profile.reference_binding_status,
            "literal_preservation_status": profile.literal_preservation_status,
            "bbox_provenance_status": profile.bbox_provenance_status,
            "binding_review_status": profile.binding_review_status,
            "final_binding_action": profile.final_binding_action,
            "ready_for_gold_candidate_is_confirmed_gold": False,
            "usable_hybrid_candidate_is_production_ready": False,
            "confirmed_gold": False,
            "production_ready": False,
            "unit_visible": profile.unit_visible,
            "unit_bound": profile.unit_bound,
            "footnote_present": profile.footnote_present,
            "footnote_bound": profile.footnote_bound,
            "reference_visible": profile.reference_visible,
            "row_level_reference_bound": profile.row_level_reference_bound,
            "literal_tokens_observed": profile.literal_tokens,
            "key_warnings": profile.key_warnings,
            "review_notes": profile.review_notes,
            "rule_fix_scope": profile.rule_fix_scope,
            "gold_readiness_rationale": profile.gold_readiness_rationale,
            "phase7c3_review_evidence_summary": candidate.get("review_evidence_summary", ""),
            "table_preview_rows": raw_preview,
        }
        rows.append(row)

    validate_review_rows(rows)
    return rows


def validate_review_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 5:
        raise ValueError(f"binding review 输出数量应为 5，实际为 {len(rows)}")
    ids = {row["hybrid_table_object_id"] for row in rows}
    if ids != EXPECTED_BINDING_REVIEW_IDS:
        raise ValueError("binding review 输出 id 集合漂移")
    for row in rows:
        if row["final_binding_action"] not in FINAL_BINDING_ACTIONS:
            raise ValueError(f"{row['hybrid_table_object_id']} final_binding_action 非法")
        if row["binding_review_status"] not in BINDING_REVIEW_STATUSES:
            raise ValueError(f"{row['hybrid_table_object_id']} binding_review_status 非法")
        if row["ready_for_gold_candidate_is_confirmed_gold"]:
            raise ValueError("ready_for_gold_candidate 不得等于 confirmed gold")
        if row["usable_hybrid_candidate_is_production_ready"]:
            raise ValueError("usable_hybrid_candidate 不得等于 production-ready")
        if row["value_bboxes_available"] or row["value_level_provenance_used"]:
            raise ValueError("value bbox 不存在时不得写 value-level provenance")
        if row["bbox_provenance_level"] == "value_level":
            raise ValueError("本轮不得写 value_level bbox provenance")


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ready = [row for row in rows if row["final_binding_action"] == READY_ACTION]
    rule_fix = [row for row in rows if row["final_binding_action"] == RULE_FIX_ACTION]
    fallback = [
        row
        for row in rows
        if row["final_binding_action"] in {"use_chunk_fallback", "backlog", "keep_for_manual_binding_review"}
    ]
    return ready, rule_fix, fallback


def counter_table(rows: list[dict[str, Any]], field: str) -> str:
    counts = Counter(row[field] for row in rows)
    lines = ["| 值 | 数量 |", "|---|---:|"]
    for key, count in sorted(counts.items()):
        lines.append(f"| `{key}` | {count} |")
    return "\n".join(lines)


def bullet_ids(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- 无"
    return "\n".join(f"- `{row['hybrid_table_object_id']}`" for row in rows)


def write_guardrail(report_dir: Path) -> None:
    text = """# Phase7C-4 Guardrail

## 1. 本轮定位

本轮是 Hybrid Binding Review for Keep Candidates，只对 Phase7C-3 选出的 5 个 `manual_review_binding` hybrid candidate 做受限 binding review。

## 2. 范围限制

- 只审阅 5 个 binding review candidate。
- 不扩大 smoke。
- 不处理 grid rejected / fallback / backlog case。
- 不重新打开 Phase7C-3 已分流的其他 11 个 case。

## 3. 禁止事项

- 不引入 Camelot / PyMuPDF。
- 不重跑 pdfplumber extraction。
- 不调整 pdfplumber 策略。
- 不构造 confirmed gold。
- 不运行 coverage evaluation。
- 不接入 production。
- 不访问 Milvus / BM25。
- 不读取或查询 BM25 index。
- 不运行 retrieval / embedding / rerank / model。
- 不调用 Qwen / RAGAS / OCR / VLM。
- 不伪造 value-level bbox。
- 不把 cell bbox 写成 value-level bbox。

## 4. Route C

Route C 仍只是 backlog。本轮不进入 Route C implementation，也不把 Camelot / PyMuPDF 写成下一步立即实施。
"""
    (report_dir / "phase7c_4_guardrail.md").write_text(text, encoding="utf-8")


def write_review_report(rows: list[dict[str, Any]], report_dir: Path, input_inventory: list[dict[str, Any]]) -> None:
    ready, rule_fix, fallback = split_rows(rows)
    lines = [
        "# Hybrid Binding Review 报告",
        "",
        "## 1. 目标",
        "",
        "本报告只审阅 Phase7C-3 进入 `manual_review_binding` 的 5 个 hybrid candidate。结论用于判断是否可进入后续小规模 gold construction 候选池，不构造 confirmed gold，不写 production-ready。",
        "",
        "## 2. 只读输入",
        "",
        "| path | records | lines |",
        "|---|---:|---:|",
    ]
    for item in input_inventory:
        lines.append(f"| `{item['path']}` | {item['record_count']} | {item['line_count']} |")
    lines.extend(
        [
            "",
            "## 3. 综合分流",
            "",
            f"- ready_for_gold_candidate：{len(ready)}",
            f"- needs_pdfplumber_rule_fix：{len(rule_fix)}",
            f"- fallback / backlog / keep_for_manual_binding_review：{len(fallback)}",
            "",
            "ready_for_gold_candidate 不等于 confirmed gold；usable_hybrid_candidate 不等于 production-ready。",
            "",
            "## 4. 状态统计",
            "",
            "### row_grid_status",
            counter_table(rows, "row_grid_status"),
            "",
            "### column_grid_status",
            counter_table(rows, "column_grid_status"),
            "",
            "### cell_grid_status",
            counter_table(rows, "cell_grid_status"),
            "",
            "### value_placement_status",
            counter_table(rows, "value_placement_status"),
            "",
            "### unit_binding_status",
            counter_table(rows, "unit_binding_status"),
            "",
            "### footnote_binding_status",
            counter_table(rows, "footnote_binding_status"),
            "",
            "### reference_binding_status",
            counter_table(rows, "reference_binding_status"),
            "",
            "### literal_preservation_status",
            counter_table(rows, "literal_preservation_status"),
            "",
            "### bbox_provenance_status",
            counter_table(rows, "bbox_provenance_status"),
            "",
            "### binding_review_status",
            counter_table(rows, "binding_review_status"),
            "",
            "### final_binding_action",
            counter_table(rows, "final_binding_action"),
            "",
            "## 5. 逐案结论",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['hybrid_table_object_id']}",
                "",
                f"- doc_id：`{row['doc_id']}`",
                f"- table_id：`{row['table_id']}`",
                f"- binding_review_status：`{row['binding_review_status']}`",
                f"- final_binding_action：`{row['final_binding_action']}`",
                f"- row/column/cell：`{row['row_grid_status']}` / `{row['column_grid_status']}` / `{row['cell_grid_status']}`",
                f"- value/unit/footnote/reference：`{row['value_placement_status']}` / `{row['unit_binding_status']}` / `{row['footnote_binding_status']}` / `{row['reference_binding_status']}`",
                f"- bbox provenance：`{row['bbox_provenance_status']}`；value_bboxes_available=`false`",
                f"- key warnings：{semi(row['key_warnings'])}",
                f"- review notes：{row['review_notes']}",
                f"- gold readiness：{row['gold_readiness_rationale']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 口径说明",
            "",
            "- unit visible 不等于 unit bound。",
            "- footnote present 不等于 footnote bound。",
            "- reference visible 不等于 row-level reference bound。",
            "- cell bbox 不等于 value bbox。",
            "- value_bboxes_available 预期仍为 false。",
            "- 本轮不伪造 value-level bbox，不把任何候选写成 confirmed gold。",
        ]
    )
    (report_dir / "hybrid_binding_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cards(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Hybrid Binding Review Cards",
        "",
        "本文件只展示 5 个 `manual_review_binding` candidate 的受限 binding review card。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['hybrid_table_object_id']}",
                "",
                f"- hybrid_table_object_id：`{row['hybrid_table_object_id']}`",
                f"- doc_id：`{row['doc_id']}`",
                f"- table_id：`{row['table_id']}`",
                f"- binding_review_status：`{row['binding_review_status']}`",
                f"- final_binding_action：`{row['final_binding_action']}`",
                f"- row_grid_status：`{row['row_grid_status']}`",
                f"- column_grid_status：`{row['column_grid_status']}`",
                f"- cell_grid_status：`{row['cell_grid_status']}`",
                f"- value_placement_status：`{row['value_placement_status']}`",
                f"- unit_binding_status：`{row['unit_binding_status']}`",
                f"- footnote_binding_status：`{row['footnote_binding_status']}`",
                f"- reference_binding_status：`{row['reference_binding_status']}`",
                f"- literal_preservation_status：`{row['literal_preservation_status']}`",
                f"- bbox_provenance_status：`{row['bbox_provenance_status']}`",
                f"- key warnings：{semi(row['key_warnings'])}",
                f"- review notes：{row['review_notes']}",
                "",
                "### table preview",
                "",
                table_preview_markdown(row["table_preview_rows"]),
                "",
            ]
        )
    (output_dir / "hybrid_binding_review_cards.md").write_text("\n".join(lines), encoding="utf-8")


def write_gold_decision(rows: list[dict[str, Any]], report_dir: Path) -> None:
    ready, rule_fix, fallback = split_rows(rows)
    lines = [
        "# Gold Readiness Decision",
        "",
        "## 1. 是否有 candidate 可进入后续 gold construction",
        "",
        "有。共有 2 个 candidate 可进入后续小规模 gold construction 候选池，但这不等于 confirmed gold，也不授权开始 gold construction。",
        "",
        "## 2. 可进入清单",
        "",
        bullet_ids(ready),
        "",
        "## 3. 仍需 rule fix 的清单",
        "",
        bullet_ids(rule_fix),
        "",
        "## 4. fallback / backlog 清单",
        "",
        bullet_ids(fallback),
        "",
        "## 5. 为什么 ready_for_gold_candidate 不等于 confirmed gold",
        "",
        "ready_for_gold_candidate 只表示当前 row/column/cell grid、value placement、unit/footnote/reference 风险足以支撑后续人工构造 gold 的候选入口。confirmed gold 仍需要单独构造 required cells、required units、reference/footnote binding、source_span 与人工审阅记录，并通过后续阶段的独立验收。",
        "",
        "## 6. 后续 gold construction 限制",
        "",
        "如后续进入 gold construction，应只限制在本轮 ready 清单内：`doc_0468__table_2__phase7c2_hybrid_01` 与 `doc_0687__table_3__phase7c2_hybrid_03`。不得把 rule_fix case、fallback case、backlog case 混入。",
        "",
        "## 7. 授权边界",
        "",
        "后续 gold construction 仍需用户单独授权。本轮不构造 confirmed gold。",
        "",
        "## 8. 为什么不建议扩大 smoke",
        "",
        "本轮仍发现 split cell、merged cell、metric-level gap、unit visible not bound、footnote present not bound、reference visible not bound 等绑定风险。扩大 smoke 会把 parser/layout 风险扩散到更多 case，当前更应先固定小候选池和 rule fix backlog。",
        "",
        "## 9. 为什么不建议 production",
        "",
        "value-level bbox 不存在，pdfplumber cell bbox 只能提供 cell-level layout provenance；部分 case 仍需规则修复，ready 候选也只是离线 gold construction 候选，不具备 production extraction quality gate、rollback、monitoring 或 retrieval integration。",
        "",
        "## 10. Route C",
        "",
        "Route C 仍只是 backlog。本轮不建议立即实施 Camelot / PyMuPDF / OCR / VLM 路线。",
    ]
    (report_dir / "gold_readiness_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], output_dir: Path, report_dir: Path) -> None:
    ready, rule_fix, fallback = split_rows(rows)
    generated_files = [
        output_dir / "hybrid_binding_review.jsonl",
        output_dir / "hybrid_binding_review_summary.csv",
        output_dir / "hybrid_candidates_ready_for_gold.jsonl",
        output_dir / "hybrid_candidates_needing_rule_fix.csv",
        output_dir / "hybrid_candidates_fallback_or_backlog.csv",
        output_dir / "hybrid_binding_review_cards.md",
        report_dir / "phase7c_4_guardrail.md",
        report_dir / "hybrid_binding_review_report.md",
        report_dir / "gold_readiness_decision.md",
        report_dir / "phase7c_4_summary.md",
    ]
    lines = [
        "# Phase7C-4 总结",
        "",
        "## 1. 本轮生成文件",
        "",
    ]
    lines.extend(f"- `{rel(path)}`" for path in generated_files)
    lines.extend(
        [
            "",
            "## 2. 新增 / 修改脚本",
            "",
            "- 新增脚本：`scripts/extraction/review_hybrid_binding_candidates.py`",
            "- 未修改 ingestion 主链路、production pipeline、configs、README、baseline registry。",
            "",
            "## 3. 是否新增测试",
            "",
            "- 新增测试：`tests/test_phase7_hybrid_binding_review.py`",
            "",
            "## 4. 处理 candidate 数量",
            "",
            f"- 处理数量：{len(rows)}",
            "- 只处理 Phase7C-3 的 5 个 `manual_review_binding` candidate。",
            "",
            "## 5. 状态统计",
            "",
            "### row_grid_status",
            counter_table(rows, "row_grid_status"),
            "",
            "### column_grid_status",
            counter_table(rows, "column_grid_status"),
            "",
            "### cell_grid_status",
            counter_table(rows, "cell_grid_status"),
            "",
            "### value_placement_status",
            counter_table(rows, "value_placement_status"),
            "",
            "### unit_binding_status",
            counter_table(rows, "unit_binding_status"),
            "",
            "### footnote_binding_status",
            counter_table(rows, "footnote_binding_status"),
            "",
            "### reference_binding_status",
            counter_table(rows, "reference_binding_status"),
            "",
            "### bbox_provenance_status",
            counter_table(rows, "bbox_provenance_status"),
            "",
            "### binding_review_status",
            counter_table(rows, "binding_review_status"),
            "",
            "### final_binding_action",
            counter_table(rows, "final_binding_action"),
            "",
            "## 6. 清单",
            "",
            "### ready_for_gold_candidate",
            bullet_ids(ready),
            "",
            "### needs_pdfplumber_rule_fix",
            bullet_ids(rule_fix),
            "",
            "### fallback / backlog",
            bullet_ids(fallback),
            "",
            "## 7. 建议",
            "",
            "- 是否建议进入 gold construction：建议后续可单独授权小规模 gold construction，但只限 ready 清单，不等于本轮已构造 confirmed gold。",
            "- 是否建议继续 pdfplumber 主线：建议继续离线 hardening 主线。",
            "- 是否建议调 pdfplumber 策略：本轮不建议；先保留为 rule fix backlog。",
            "- 是否建议扩大 smoke：不建议。",
            "- 是否建议引入 Camelot / PyMuPDF：不建议立即引入。",
            "- 是否建议进入 production：不建议。",
            "- baseline / guardrail 是否漂移：未发现漂移。",
            "- Route C 是否仍只是 backlog：是。",
            "",
            "## 8. 明确未执行事项",
            "",
            "- 未扩大 smoke。",
            "- 未处理 5 个 binding review candidate 之外的 case。",
            "- 未引入 Camelot。",
            "- 未引入 PyMuPDF。",
            "- 未重跑 pdfplumber extraction。",
            "- 未调 pdfplumber 策略。",
            "- 未构造 confirmed gold。",
            "- 未运行 coverage。",
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
        ]
    )
    (report_dir / "phase7c_4_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, report_dir: Path, input_inventory: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ready, rule_fix, fallback = split_rows(rows)

    write_jsonl(rows, output_dir / "hybrid_binding_review.jsonl")
    write_csv(
        [
            {
                **row,
                "key_warnings": semi(row["key_warnings"]),
                "literal_tokens_observed": semi(row["literal_tokens_observed"]),
            }
            for row in rows
        ],
        output_dir / "hybrid_binding_review_summary.csv",
        SUMMARY_FIELDS,
    )
    write_jsonl(ready, output_dir / "hybrid_candidates_ready_for_gold.jsonl")
    write_csv(
        [
            {
                **row,
                "key_warnings": semi(row["key_warnings"]),
            }
            for row in rule_fix
        ],
        output_dir / "hybrid_candidates_needing_rule_fix.csv",
        RULE_FIX_FIELDS,
    )
    write_csv(
        [
            {
                **row,
                "reason": row["gold_readiness_rationale"],
                "key_warnings": semi(row["key_warnings"]),
            }
            for row in fallback
        ],
        output_dir / "hybrid_candidates_fallback_or_backlog.csv",
        FALLBACK_FIELDS,
    )
    write_cards(rows, output_dir)
    write_guardrail(report_dir)
    write_review_report(rows, report_dir, input_inventory)
    write_gold_decision(rows, report_dir)
    write_summary(rows, output_dir, report_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review Phase7C-4 hybrid binding candidates.")
    parser.add_argument("--binding-candidates", type=Path, default=DEFAULT_BINDING_CANDIDATES)
    parser.add_argument("--hybrid-objects", type=Path, default=DEFAULT_HYBRID_OBJECTS)
    parser.add_argument("--raw-tables", type=Path, default=DEFAULT_RAW_TABLES)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    binding_candidates_path = resolve_path(args.binding_candidates)
    hybrid_objects_path = resolve_path(args.hybrid_objects)
    raw_tables_path = resolve_path(args.raw_tables)
    alignment_path = resolve_path(args.alignment)
    output_dir = resolve_path(args.output_dir)
    report_dir = resolve_path(args.report_dir)

    required = REQUIRED_PHASE7C3_INPUTS + REQUIRED_PHASE7C2_INPUTS + REQUIRED_PHASE6D_INPUTS
    input_inventory = read_input_inventory(required)
    candidates = load_jsonl(binding_candidates_path)
    hybrid_objects = load_jsonl(hybrid_objects_path)
    raw_tables = load_jsonl(raw_tables_path)
    alignment_rows = load_csv(alignment_path)
    rows = build_review_rows(candidates, hybrid_objects, raw_tables, alignment_rows)
    write_outputs(rows, output_dir, report_dir, input_inventory)
    return rows


def main() -> None:
    parser = build_arg_parser()
    rows = run(parser.parse_args())
    action_counts = Counter(row["final_binding_action"] for row in rows)
    print(f"reviewed_candidates={len(rows)}")
    print("final_binding_action=" + json.dumps(action_counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
