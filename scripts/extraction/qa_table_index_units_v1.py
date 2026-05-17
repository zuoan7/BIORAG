#!/usr/bin/env python3
"""Build Phase7I-1 QA preview artifacts for table index units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
QA_RULE_VERSION = "phase7i1_table_index_unit_qa_v1"

DEFAULT_PHASE7I_UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units.preview.jsonl"
)
DEFAULT_PHASE7I_UNITS_CSV_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_units.preview.csv"
)
DEFAULT_PHASE7I_STATS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_index_unit_stats.csv"
)
DEFAULT_TABLE_UNIT_PATH = ROOT / "data/experiments/v7_phase7_table_index_unit_design/table_unit_preview.jsonl"
DEFAULT_ROW_UNIT_PATH = ROOT / "data/experiments/v7_phase7_table_index_unit_design/row_unit_preview.jsonl"
DEFAULT_CELL_GROUP_UNIT_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_design/cell_group_unit_preview.jsonl"
)
DEFAULT_FORMAL_VALIDATION_PATH = (
    ROOT / "results/v7_phase7_expanded_seed_validation/formal_seed_validation_results.csv"
)
DEFAULT_REVIEW_PACK_INDEX_PATH = (
    ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/review_pack_index.csv"
)
DEFAULT_CANDIDATE_POOL_PATH = (
    ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/candidate_pool_scored.csv"
)
DEFAULT_CSV_TABLES_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/csv_tables"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_table_index_unit_qa"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_table_index_unit_qa"

EXPECTED_TOTAL_UNITS = 414
EXPECTED_UNIT_TYPE_COUNTS = {
    "table_unit": 15,
    "row_unit": 254,
    "cell_group_unit": 145,
}

QA_PREVIEW_CSV_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "row_index",
    "row_label",
    "content_text_for_embedding",
    "index_text_quality",
    "header_path_quality",
    "retrieval_ready",
    "quality_flags",
    "content_text_changed",
    "header_path_changed",
    "index_unit_status",
    "production_ready",
    "is_official_benchmark_seed",
    "value_bboxes_available",
    "cell_bboxes_available",
    "source_span_granularity",
]

QUALITY_CSV_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "doc_id",
    "table_id",
    "index_text_quality",
    "header_path_quality",
    "retrieval_ready",
    "quality_flags",
    "quality_notes",
    "original_content_text_preview",
    "qa_content_text_preview",
]

HEADER_ISSUE_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "doc_id",
    "table_id",
    "row_index",
    "row_label",
    "issue_type",
    "original_header_path",
    "qa_header_path",
    "value",
    "action",
    "quality_flag",
    "note",
]

DOC0261_TARGET_TABLES = {"Table 2", "Table 3"}
DOC0261_FIXED_HEADER_PATHS = [
    ["Taxon"],
    ["Abundance, % (mean ± SD)", "Control"],
    ["Abundance, % (mean ± SD)", "2′-FL"],
    ["Abundance, % (mean ± SD)", "Lactose"],
    ["Abundance, % (mean ± SD)", "GOS"],
    ["Overall p-value", "(FDR adj)1"],
]
BAD_GOS_PATH = ["Overall p-value", "GOS"]
FIXED_GOS_PATH = ["Abundance, % (mean ± SD)", "GOS"]

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_table(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [[normalize(cell) for cell in row] for row in csv.reader(handle)]


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preview(text: Any, limit: int = 280) -> str:
    value = normalize(text)
    return value[:limit] + ("..." if len(value) > limit else "")


def path_to_text(path: Any) -> str:
    if isinstance(path, list):
        return " / ".join(normalize(item) for item in path if normalize(item))
    return normalize(path)


def same_path(left: Any, right: list[str]) -> bool:
    return isinstance(left, list) and [normalize(item) for item in left] == right


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


def is_doc0261_override_scope(unit: dict[str, Any]) -> bool:
    return unit.get("doc_id") == "doc_0261" and unit.get("table_id") in DOC0261_TARGET_TABLES


def corrected_header_path(path: Any, unit: dict[str, Any]) -> tuple[Any, bool]:
    if is_doc0261_override_scope(unit) and same_path(path, BAD_GOS_PATH):
        return FIXED_GOS_PATH.copy(), True
    return path, False


def update_value_header(value: dict[str, Any], unit: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    updated = deepcopy(value)
    new_path, changed = corrected_header_path(updated.get("header_path"), unit)
    if changed:
        updated["header_path"] = new_path
        updated["column_header"] = path_to_text(new_path)
    return updated, changed


def update_header_path_list(paths: Any, unit: dict[str, Any]) -> tuple[Any, bool]:
    if not isinstance(paths, list):
        return paths, False
    changed = False
    updated: list[Any] = []
    for path in paths:
        new_path, path_changed = corrected_header_path(path, unit)
        changed = changed or path_changed
        updated.append(new_path)
    return updated, changed


def update_doc0261_header_map(unit: dict[str, Any]) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
    updated = deepcopy(unit)
    issues: list[dict[str, Any]] = []
    if not is_doc0261_override_scope(updated):
        return updated, False, issues

    metadata = updated.get("metadata") or {}
    changed = False

    original_paths = metadata.get("header_path")
    if isinstance(original_paths, list):
        new_paths, paths_changed = update_header_path_list(original_paths, updated)
        if paths_changed:
            metadata["header_path"] = new_paths
            changed = True

    for value_key in ["row_values", "cell_group_values"]:
        values = metadata.get(value_key)
        if not isinstance(values, list):
            continue
        new_values = []
        for value in values:
            if not isinstance(value, dict):
                new_values.append(value)
                continue
            before_path = value.get("header_path")
            fixed_value, value_changed = update_value_header(value, updated)
            if value_changed:
                changed = True
                issues.append(
                    header_issue_row(
                        updated,
                        issue_type="p_value_parent_mismatch",
                        original_header_path=before_path,
                        qa_header_path=fixed_value.get("header_path"),
                        value=fixed_value.get("value", ""),
                        action="fixed_by_header_map_override",
                        quality_flag="",
                        note="doc_0261 Table 2/3 的 GOS 值已从 Overall p-value group 修正到 Abundance group。",
                    )
                )
            new_values.append(fixed_value)
        metadata[value_key] = new_values

    if changed:
        metadata["column_headers"] = [path_to_text(path) for path in DOC0261_FIXED_HEADER_PATHS]
        if updated["unit_type"] == "table_unit":
            metadata["header_path"] = deepcopy(DOC0261_FIXED_HEADER_PATHS)
            issues.append(
                header_issue_row(
                    updated,
                    issue_type="p_value_parent_mismatch",
                    original_header_path=BAD_GOS_PATH,
                    qa_header_path=FIXED_GOS_PATH,
                    value="",
                    action="fixed_by_header_map_override",
                    quality_flag="",
                    note="table_unit header_path 已使用 doc_0261 override 规则修正。",
                )
            )
        updated["metadata"] = metadata
        updated["content_text_for_embedding"] = rebuild_content_text(updated)
        updated["content_markdown"] = rebuild_content_markdown(updated)

    return updated, changed, issues


def facts_from_values(values: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    selected = values if limit is None else values[:limit]
    return [
        f"{normalize(item.get('column_header'))}={normalize(item.get('value'))}"
        for item in selected
        if normalize(item.get("column_header")) and normalize(item.get("value"))
    ]


def warning_text_for(unit_type: str) -> str:
    if unit_type == "table_unit":
        return (
            "Binding review remains warning-level; source spans are not value-level; "
            "value-level coordinates are unavailable."
        )
    if unit_type == "cell_group_unit":
        return (
            "This is a row-level value group, not independent value-level evidence; "
            "value-level coordinates are not claimed."
        )
    return "Binding notes are warning-level only; value-level coordinates are not claimed."


def clean_caption(caption: str) -> str:
    return normalize(caption).replace("[TABLE CAPTION]", "").strip()


def rebuild_content_text(unit: dict[str, Any]) -> str:
    metadata = unit.get("metadata") or {}
    warning = warning_text_for(unit.get("unit_type", ""))
    if unit.get("unit_type") == "table_unit":
        headers = [path_to_text(path) for path in metadata.get("header_path", [])][:12]
        caption = clean_caption(unit.get("caption", ""))
        topic = preview(caption, 260)
        return normalize(
            f"In {unit.get('doc_id')} {unit.get('table_id')}, the table caption is: {topic}. "
            f"Table topic summary: {topic}. Main column headers include: {', '.join(headers)}. {warning}"
        )
    if unit.get("unit_type") == "cell_group_unit":
        facts = facts_from_values(metadata.get("cell_group_values") or [])
        return normalize(
            f"In {unit.get('doc_id')} {unit.get('table_id')}, row \"{metadata.get('row_label', '')}\" "
            f"has selected key values: {'; '.join(facts)}. {warning}"
        )
    facts = facts_from_values(metadata.get("row_values") or [], limit=12)
    return normalize(
        f"In {unit.get('doc_id')} {unit.get('table_id')}, row \"{metadata.get('row_label', '')}\" "
        f"reports: {'; '.join(facts)}. {warning}"
    )


def rebuild_content_markdown(unit: dict[str, Any]) -> str:
    metadata = unit.get("metadata") or {}
    warning = warning_text_for(unit.get("unit_type", ""))
    if unit.get("unit_type") == "table_unit":
        return "\n".join(
            [
                f"### table_unit: {unit.get('doc_id')} / {unit.get('table_id')}",
                "",
                f"- caption: {clean_caption(unit.get('caption', ''))}",
                f"- data rows: {(metadata.get('table_shape') or {}).get('data_rows', '')}",
                f"- columns: {(metadata.get('table_shape') or {}).get('columns', '')}",
                f"- header structure: `{metadata.get('header_structure_type', '')}`",
                f"- warning limitation: {warning}",
            ]
        )
    if unit.get("unit_type") == "cell_group_unit":
        facts = facts_from_values(metadata.get("cell_group_values") or [])
        return "\n".join(
            [
                f"### cell_group_unit: row {metadata.get('row_index', '')} / {metadata.get('row_label', '')}",
                "",
                f"- selected key values: {'; '.join(facts)}",
                f"- provenance limitation: {warning}",
            ]
        )
    facts = facts_from_values(metadata.get("row_values") or [], limit=8)
    return "\n".join(
        [
            f"### row_unit: row {metadata.get('row_index', '')} / {metadata.get('row_label', '')}",
            "",
            f"- context: `{unit.get('doc_id')}` / `{unit.get('table_id')}`",
            f"- values: {'; '.join(facts)}",
            f"- warning limitation: {warning}",
        ]
    )


def data_value_like(text: str) -> bool:
    value = normalize(text)
    if not value:
        return False
    lower = value.lower()
    if re.search(r"\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?", value):
        return True
    if re.fullmatch(r"[<>]=?\s*\d+(?:\.\d+)?(?:[a-z])?", lower):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?(?:[a-z])?", lower):
        return True
    return False


def header_path_contains_data_value(path: Any) -> bool:
    if not isinstance(path, list):
        return data_value_like(path_to_text(path))
    return any(data_value_like(str(part)) for part in path)


def p_value_header_component(text: str) -> bool:
    lower = normalize(text).lower()
    return "p-value" in lower or "p value" in lower or lower in {"p", "pval", "p-value"}


def p_value_leaf_ok(text: str) -> bool:
    lower = normalize(text).lower()
    return bool(re.search(r"p[- ]?value|fdr|adj|adjusted|q[- ]?value", lower))


def p_value_like_value(text: str) -> bool:
    value = normalize(text).lower()
    if not value:
        return False
    if "±" in value:
        return False
    if re.fullmatch(r"[<>]=?\s*\d+(?:\.\d+)?", value):
        return True
    try:
        numeric = float(value)
    except ValueError:
        return False
    return 0 <= numeric <= 1


def p_value_parent_mismatch(path: Any, value: Any = "") -> bool:
    if not isinstance(path, list) or not path:
        return False
    if not any(p_value_header_component(part) for part in path[:-1]):
        return False
    leaf = normalize(path[-1])
    if p_value_leaf_ok(leaf):
        return False
    return not p_value_like_value(value)


def header_issue_row(
    unit: dict[str, Any],
    issue_type: str,
    original_header_path: Any,
    qa_header_path: Any,
    value: Any,
    action: str,
    quality_flag: str,
    note: str,
) -> dict[str, Any]:
    metadata = unit.get("metadata") or {}
    return {
        "table_index_unit_id": unit.get("table_index_unit_id", ""),
        "unit_type": unit.get("unit_type", ""),
        "seed_id": unit.get("seed_id", ""),
        "doc_id": unit.get("doc_id", ""),
        "table_id": unit.get("table_id", ""),
        "row_index": metadata.get("row_index", ""),
        "row_label": metadata.get("row_label", ""),
        "issue_type": issue_type,
        "original_header_path": path_to_text(original_header_path),
        "qa_header_path": path_to_text(qa_header_path),
        "value": value,
        "action": action,
        "quality_flag": quality_flag,
        "note": note,
    }


def iter_value_paths(unit: dict[str, Any]) -> list[tuple[Any, str]]:
    metadata = unit.get("metadata") or {}
    paths: list[tuple[Any, str]] = []
    values = metadata.get("row_values") or metadata.get("cell_group_values") or []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                paths.append((item.get("header_path"), normalize(item.get("value"))))
    header_path = metadata.get("header_path")
    if isinstance(header_path, list):
        if header_path and all(isinstance(item, list) for item in header_path):
            paths.extend((path, "") for path in header_path)
        elif header_path:
            paths.append((header_path, ""))
    return paths


def row_value_count(unit: dict[str, Any]) -> int:
    metadata = unit.get("metadata") or {}
    values = metadata.get("row_values") or metadata.get("cell_group_values") or []
    return sum(1 for item in values if isinstance(item, dict) and normalize(item.get("value")))


def row_label_quality_flags(unit: dict[str, Any]) -> tuple[list[str], list[str]]:
    if unit.get("unit_type") == "table_unit":
        return [], []
    label = normalize((unit.get("metadata") or {}).get("row_label"))
    if not label or label.lower() in {"unlabeled_row", "row", "none", "na", "n/a", "-"}:
        return ["empty_or_generic_row_label"], ["row_label 为空或过于泛化。"]
    if re.fullmatch(r"[\W_]+", label):
        return ["weak_row_label"], ["row_label 主要由符号组成。"]
    if len(label) <= 1:
        return ["weak_row_label"], ["row_label 过短，行级语义较弱。"]
    return [], []


def low_information(unit: dict[str, Any]) -> bool:
    text = normalize(unit.get("content_text_for_embedding"))
    if not text:
        return True
    if unit.get("unit_type") == "table_unit":
        headers = (unit.get("metadata") or {}).get("column_headers") or []
        return len(headers) <= 1 and len(clean_caption(unit.get("caption", ""))) < 40
    value_count = row_value_count(unit)
    metadata = unit.get("metadata") or {}
    row_values = metadata.get("row_values") or metadata.get("cell_group_values") or []
    non_label_values = 0
    for item in row_values:
        if not isinstance(item, dict):
            continue
        value = normalize(item.get("value"))
        column = normalize(item.get("column_header")).lower()
        if value and value != normalize(metadata.get("row_label")) and column not in {"taxon", "strain"}:
            non_label_values += 1
    return value_count <= 1 or non_label_values == 0


def caption_noise_flags(unit: dict[str, Any]) -> tuple[list[str], list[str]]:
    flags: list[str] = []
    notes: list[str] = []
    caption = normalize(unit.get("caption"))
    lower = caption.lower()
    if "continued" in lower:
        flags.append("continued_table_noise")
        notes.append("caption 或 table title 含 continued 信号。")
    if len(caption) > 360 or "##" in caption or "materials and methods" in lower:
        flags.append("caption_noise_heavy")
        notes.append("caption/title 中正文或 markdown 噪声较重。")
    return flags, notes


def evaluate_unit(unit: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flags: list[str] = []
    notes: list[str] = []
    issue_rows: list[dict[str, Any]] = []

    for path, value in iter_value_paths(unit):
        if header_path_contains_data_value(path):
            flags.append("header_path_contains_data_value")
            issue_rows.append(
                header_issue_row(
                    unit,
                    issue_type="header_path_contains_data_value",
                    original_header_path=path,
                    qa_header_path=path,
                    value=value,
                    action="marked_for_exclusion",
                    quality_flag="header_path_contains_data_value",
                    note="header_path 中疑似混入数值型数据。",
                )
            )
        if p_value_parent_mismatch(path, value):
            flags.append("p_value_parent_mismatch")
            issue_rows.append(
                header_issue_row(
                    unit,
                    issue_type="p_value_parent_mismatch",
                    original_header_path=path,
                    qa_header_path=path,
                    value=value,
                    action="marked_for_exclusion",
                    quality_flag="p_value_parent_mismatch",
                    note="p-value 父级下仍挂载非 p-value 子列。",
                )
            )

    metadata = unit.get("metadata") or {}
    if metadata.get("header_structure_type") == "uncertain_header":
        flags.append("header_path_uncertain")
        notes.append("header_structure_type=uncertain_header。")

    label_flags, label_notes = row_label_quality_flags(unit)
    flags.extend(label_flags)
    notes.extend(label_notes)

    if low_information(unit):
        flags.append("low_information_content")
        notes.append("content_text_for_embedding 信息量不足。")

    caption_flags, caption_notes = caption_noise_flags(unit)
    flags.extend(caption_flags)
    notes.extend(caption_notes)

    if len(normalize(unit.get("content_text_for_embedding"))) > 1400:
        flags.append("overlong_content_text")
        notes.append("content_text_for_embedding 过长。")

    if has_value_bbox_claim(unit):
        flags.append("value_level_bbox_claim_detected")
        notes.append("检测到 value-level bbox claim 或 value_bboxes_available 非 false。")

    guardrail = unit.get("guardrail") or {}
    if guardrail.get("production_ready") is not False:
        flags.append("production_ready_claim_detected")
        notes.append("guardrail.production_ready 不是 false。")
    if guardrail.get("is_official_benchmark_seed") is not False:
        flags.append("official_benchmark_seed_claim_detected")
        notes.append("guardrail.is_official_benchmark_seed 不是 false。")

    if has_forbidden_index_field(unit):
        flags.append("forbidden_index_field_detected")
        notes.append("检测到 embedding/retrieval/index 结果字段。")

    unique_flags = sorted(set(flags))
    if "p_value_parent_mismatch" in unique_flags or "header_path_contains_data_value" in unique_flags:
        header_quality = "fail"
    elif "header_path_uncertain" in unique_flags:
        header_quality = "warning"
    else:
        header_quality = "pass"

    if "low_information_content" in unique_flags or not normalize(unit.get("content_text_for_embedding")):
        text_quality = "low"
    elif any(flag in unique_flags for flag in ["caption_noise_heavy", "continued_table_noise", "overlong_content_text"]):
        text_quality = "medium"
    elif unit.get("unit_type") != "table_unit" and row_value_count(unit) < 3:
        text_quality = "medium"
    else:
        text_quality = "high"

    retrieval_ready = (
        text_quality != "low"
        and header_quality != "fail"
        and normalize(unit.get("content_text_for_embedding")) != ""
        and guardrail.get("production_ready") is False
        and guardrail.get("index_unit_status") == "preview_only"
        and guardrail.get("is_official_benchmark_seed") is False
        and (unit.get("provenance") or {}).get("value_bboxes_available") is False
        and "value_level_bbox_claim_detected" not in unique_flags
        and "forbidden_index_field_detected" not in unique_flags
    )
    if unit.get("unit_type") == "cell_group_unit" and any(
        flag in unique_flags
        for flag in ["header_path_uncertain", "low_information_content", "p_value_parent_mismatch"]
    ):
        retrieval_ready = False

    quality = {
        "index_text_quality": text_quality,
        "header_path_quality": header_quality,
        "retrieval_ready": retrieval_ready,
        "quality_flags": unique_flags,
        "quality_notes": notes,
    }
    return quality, issue_rows


def attach_duplicate_flags(units: list[dict[str, Any]], qualities: dict[str, dict[str, Any]]) -> None:
    normalized_hashes = Counter(
        sha256_text(normalize(unit.get("content_text_for_embedding")).lower()) for unit in units
    )
    for unit in units:
        content_hash = sha256_text(normalize(unit.get("content_text_for_embedding")).lower())
        if normalized_hashes[content_hash] <= 1:
            continue
        quality = qualities[unit["table_index_unit_id"]]
        flags = set(quality["quality_flags"])
        flags.add("duplicate_or_near_duplicate_unit")
        quality["quality_flags"] = sorted(flags)
        quality["quality_notes"].append("content_text_for_embedding 与其他 unit 重复或近似重复。")


def attach_excessive_cell_group_flags(units: list[dict[str, Any]], qualities: dict[str, dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        if unit.get("unit_type") == "cell_group_unit":
            grouped[(unit.get("seed_id", ""), unit.get("table_id", ""))].append(unit)
    for key, group in grouped.items():
        low_quality = [
            unit
            for unit in group
            if qualities[unit["table_index_unit_id"]]["index_text_quality"] == "low"
            or qualities[unit["table_index_unit_id"]]["header_path_quality"] == "fail"
        ]
        if len(group) < 10 or len(low_quality) < 10 or len(low_quality) / len(group) < 0.4:
            continue
        for unit in group:
            quality = qualities[unit["table_index_unit_id"]]
            flags = set(quality["quality_flags"])
            flags.add("excessive_low_quality_cell_group_units")
            quality["quality_flags"] = sorted(flags)
            quality["quality_notes"].append(
                f"同表 cell_group_unit 低质量比例较高：{len(low_quality)}/{len(group)}。"
            )


def qa_csv_row(unit: dict[str, Any]) -> dict[str, Any]:
    metadata = unit.get("metadata") or {}
    guardrail = unit.get("guardrail") or {}
    provenance = unit.get("provenance") or {}
    quality = metadata.get("index_quality") or {}
    return {
        "table_index_unit_id": unit.get("table_index_unit_id", ""),
        "unit_type": unit.get("unit_type", ""),
        "seed_id": unit.get("seed_id", ""),
        "candidate_id": unit.get("candidate_id", ""),
        "doc_id": unit.get("doc_id", ""),
        "table_id": unit.get("table_id", ""),
        "row_index": metadata.get("row_index", ""),
        "row_label": metadata.get("row_label", ""),
        "content_text_for_embedding": unit.get("content_text_for_embedding", ""),
        "index_text_quality": quality.get("index_text_quality", ""),
        "header_path_quality": quality.get("header_path_quality", ""),
        "retrieval_ready": str(quality.get("retrieval_ready")).lower(),
        "quality_flags": ";".join(quality.get("quality_flags") or []),
        "content_text_changed": str(quality.get("content_text_changed")).lower(),
        "header_path_changed": str(quality.get("header_path_changed")).lower(),
        "index_unit_status": guardrail.get("index_unit_status", ""),
        "production_ready": str(guardrail.get("production_ready")).lower(),
        "is_official_benchmark_seed": str(guardrail.get("is_official_benchmark_seed")).lower(),
        "value_bboxes_available": str(provenance.get("value_bboxes_available")).lower(),
        "cell_bboxes_available": str(provenance.get("cell_bboxes_available")).lower(),
        "source_span_granularity": provenance.get("source_span_granularity", ""),
    }


def quality_csv_row(unit: dict[str, Any]) -> dict[str, Any]:
    quality = (unit.get("metadata") or {}).get("index_quality") or {}
    return {
        "table_index_unit_id": unit.get("table_index_unit_id", ""),
        "unit_type": unit.get("unit_type", ""),
        "seed_id": unit.get("seed_id", ""),
        "doc_id": unit.get("doc_id", ""),
        "table_id": unit.get("table_id", ""),
        "index_text_quality": quality.get("index_text_quality", ""),
        "header_path_quality": quality.get("header_path_quality", ""),
        "retrieval_ready": str(quality.get("retrieval_ready")).lower(),
        "quality_flags": ";".join(quality.get("quality_flags") or []),
        "quality_notes": ";".join(quality.get("quality_notes") or []),
        "original_content_text_preview": quality.get("original_content_text_preview", ""),
        "qa_content_text_preview": preview(unit.get("content_text_for_embedding")),
    }


def input_boundary_checks(
    units: list[dict[str, Any]],
    formal_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    table_units: list[dict[str, Any]],
    row_units: list[dict[str, Any]],
    cell_group_units: list[dict[str, Any]],
    csv_tables_dir: Path,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    formal_ids = {row["seed_id"] for row in formal_rows}
    unit_seed_ids = {unit.get("seed_id") for unit in units}
    type_counts = Counter(unit.get("unit_type") for unit in units)
    review_candidates = {row.get("candidate_id") for row in review_rows}
    pool_by_candidate = {row.get("candidate_id"): row for row in candidate_rows}

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add(
        "phase7i_unit_total",
        "pass" if len(units) == EXPECTED_TOTAL_UNITS else "pass_with_warnings",
        f"Phase7I 输入 unit 数量={len(units)}，期望={EXPECTED_TOTAL_UNITS}。",
    )
    for unit_type, expected in EXPECTED_UNIT_TYPE_COUNTS.items():
        add(
            f"{unit_type}_count",
            "pass" if type_counts.get(unit_type, 0) == expected else "pass_with_warnings",
            f"{unit_type}={type_counts.get(unit_type, 0)}，期望={expected}。",
        )
    add(
        "split_preview_files_count",
        "pass"
        if len(table_units) == 15 and len(row_units) == 254 and len(cell_group_units) == 145
        else "pass_with_warnings",
        f"table/row/cell_group split={len(table_units)}/{len(row_units)}/{len(cell_group_units)}。",
    )
    add(
        "formal_seed_count",
        "pass" if len(formal_ids) == 15 else "fail",
        f"Phase7H formal seed 数量={len(formal_ids)}。",
    )
    add(
        "formal_seed_scope",
        "pass" if unit_seed_ids == formal_ids else "fail",
        "unit seed 集合与 Phase7H formal seed 完全一致。"
        if unit_seed_ids == formal_ids
        else f"差异={sorted(unit_seed_ids ^ formal_ids)[:5]}",
    )

    bad_candidates = [
        unit.get("candidate_id")
        for unit in units
        if unit.get("candidate_id") not in review_candidates
        or (pool_by_candidate.get(unit.get("candidate_id")) or {}).get("review_priority") == "auto_excluded"
    ]
    add(
        "review_pack_scope",
        "pass" if not bad_candidates else "fail",
        "所有 unit 的 candidate 均来自 Phase7G review pack formal 范围，未混入 auto_excluded。"
        if not bad_candidates
        else f"异常 candidate={sorted(set(bad_candidates))[:5]}",
    )

    bad_seed_status = [
        unit.get("table_index_unit_id")
        for unit in units
        if (unit.get("guardrail") or {}).get("seed_status") != "confirmed_seed_with_warnings"
    ]
    add(
        "partial_reject_unreviewed_excluded",
        "pass" if not bad_seed_status else "fail",
        "partial / reject / unreviewed 未进入 QA formal set。"
        if not bad_seed_status
        else f"异常 unit={bad_seed_status[:5]}",
    )

    add(
        "preview_only_guardrail",
        "pass"
        if all((unit.get("guardrail") or {}).get("index_unit_status") == "preview_only" for unit in units)
        else "fail",
        "所有 unit guardrail.index_unit_status=preview_only。",
    )
    add(
        "production_ready_false",
        "pass" if all((unit.get("guardrail") or {}).get("production_ready") is False for unit in units) else "fail",
        "所有 unit guardrail.production_ready=false。",
    )
    add(
        "official_benchmark_seed_false",
        "pass"
        if all((unit.get("guardrail") or {}).get("is_official_benchmark_seed") is False for unit in units)
        else "fail",
        "所有 unit guardrail.is_official_benchmark_seed=false。",
    )
    add(
        "value_bboxes_available_false",
        "pass"
        if all((unit.get("provenance") or {}).get("value_bboxes_available") is False for unit in units)
        else "fail",
        "所有 unit provenance.value_bboxes_available=false。",
    )
    source_csv_paths = {
        resolve_path(Path((unit.get("provenance") or {}).get("source_csv_path", "")))
        for unit in units
        if (unit.get("provenance") or {}).get("source_csv_path")
    }
    missing_source_csv = [path for path in source_csv_paths if not path.exists()]
    outside_csv_dir = [
        path
        for path in source_csv_paths
        if csv_tables_dir.resolve() not in path.resolve().parents and path.resolve() != csv_tables_dir.resolve()
    ]
    for path in source_csv_paths:
        if path.exists():
            read_csv_table(path)
    add(
        "phase7g_csv_tables_readonly_reference",
        "pass" if not missing_source_csv and not outside_csv_dir else "fail",
        f"已只读引用 Phase7G CSV artifact，source_csv 数量={len(source_csv_paths)}。"
        if not missing_source_csv and not outside_csv_dir
        else f"missing={len(missing_source_csv)} outside={len(outside_csv_dir)}",
    )
    add("no_bm25_or_milvus_access", "pass", "本脚本不读取 BM25 index，不访问 Milvus。")
    return checks


def build_header_overrides() -> dict[str, Any]:
    rules = []
    for table_id in ["Table 2", "Table 3"]:
        rules.append(
            {
                "doc_id": "doc_0261",
                "table_id": table_id,
                "override_reason": "multirow spanning header: GOS belongs to Abundance group, not Overall p-value.",
                "column_header_paths": deepcopy(DOC0261_FIXED_HEADER_PATHS),
                "forbidden_header_paths": [BAD_GOS_PATH],
                "override_by_column_index": {
                    "1": ["Taxon"],
                    "2": ["Abundance, % (mean ± SD)", "Control"],
                    "3": ["Abundance, % (mean ± SD)", "2′-FL"],
                    "4": ["Abundance, % (mean ± SD)", "Lactose"],
                    "5": ["Abundance, % (mean ± SD)", "GOS"],
                    "6": ["Overall p-value", "(FDR adj)1"],
                },
            }
        )
    return {
        "qa_rule_version": QA_RULE_VERSION,
        "scope": "Phase7I-1 QA header-map hardening",
        "rules": rules,
        "non_goals": [
            "no_pdfplumber_extractor_fix",
            "no_chunk_rebuild",
            "no_bm25_or_milvus_access",
            "no_retrieval_or_embedding",
        ],
    }


def run_qa(
    phase7i_units_path: Path = DEFAULT_PHASE7I_UNITS_PATH,
    phase7i_units_csv_path: Path = DEFAULT_PHASE7I_UNITS_CSV_PATH,
    phase7i_stats_path: Path = DEFAULT_PHASE7I_STATS_PATH,
    table_unit_path: Path = DEFAULT_TABLE_UNIT_PATH,
    row_unit_path: Path = DEFAULT_ROW_UNIT_PATH,
    cell_group_unit_path: Path = DEFAULT_CELL_GROUP_UNIT_PATH,
    formal_validation_path: Path = DEFAULT_FORMAL_VALIDATION_PATH,
    review_pack_index_path: Path = DEFAULT_REVIEW_PACK_INDEX_PATH,
    candidate_pool_path: Path = DEFAULT_CANDIDATE_POOL_PATH,
    csv_tables_dir: Path = DEFAULT_CSV_TABLES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    phase7i_units_path = resolve_path(phase7i_units_path)
    phase7i_units_csv_path = resolve_path(phase7i_units_csv_path)
    phase7i_stats_path = resolve_path(phase7i_stats_path)
    table_unit_path = resolve_path(table_unit_path)
    row_unit_path = resolve_path(row_unit_path)
    cell_group_unit_path = resolve_path(cell_group_unit_path)
    formal_validation_path = resolve_path(formal_validation_path)
    review_pack_index_path = resolve_path(review_pack_index_path)
    candidate_pool_path = resolve_path(candidate_pool_path)
    csv_tables_dir = resolve_path(csv_tables_dir)
    output_dir = resolve_path(output_dir)
    report_dir = resolve_path(report_dir)

    units = load_jsonl(phase7i_units_path)
    table_units = load_jsonl(table_unit_path)
    row_units = load_jsonl(row_unit_path)
    cell_group_units = load_jsonl(cell_group_unit_path)
    phase7i_csv_rows = load_csv(phase7i_units_csv_path)
    phase7i_stats_rows = load_csv(phase7i_stats_path)
    formal_rows = load_csv(formal_validation_path)
    review_rows = load_csv(review_pack_index_path)
    candidate_rows = load_csv(candidate_pool_path)

    boundary_checks = input_boundary_checks(
        units,
        formal_rows,
        review_rows,
        candidate_rows,
        table_units,
        row_units,
        cell_group_units,
        csv_tables_dir,
    )

    overrides = build_header_overrides()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "header_map_overrides.json").write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    qa_units: list[dict[str, Any]] = []
    header_issue_rows: list[dict[str, Any]] = []
    base_quality: dict[str, dict[str, Any]] = {}

    for original in units:
        original_text = normalize(original.get("content_text_for_embedding"))
        updated, header_changed, fixed_issues = update_doc0261_header_map(original)
        header_issue_rows.extend(fixed_issues)
        content_changed = normalize(updated.get("content_text_for_embedding")) != original_text
        quality, marked_issues = evaluate_unit(updated)
        header_issue_rows.extend(marked_issues)
        quality.update(
            {
                "original_table_index_unit_id": original.get("table_index_unit_id"),
                "original_content_text_hash": sha256_text(original_text),
                "original_content_text_preview": preview(original_text),
                "content_text_changed": content_changed,
                "header_path_changed": header_changed,
                "qa_rule_version": QA_RULE_VERSION,
            }
        )
        updated.setdefault("metadata", {})["index_quality"] = quality
        qa_units.append(updated)
        base_quality[updated["table_index_unit_id"]] = quality

    attach_duplicate_flags(qa_units, base_quality)
    attach_excessive_cell_group_flags(qa_units, base_quality)

    # Recompute retrieval_ready after table-level flags have been added only where the hard gates changed.
    for unit in qa_units:
        quality = base_quality[unit["table_index_unit_id"]]
        hard_flags = set(quality["quality_flags"])
        if any(
            flag in hard_flags
            for flag in [
                "p_value_parent_mismatch",
                "header_path_contains_data_value",
                "low_information_content",
                "value_level_bbox_claim_detected",
                "forbidden_index_field_detected",
            ]
        ):
            quality["retrieval_ready"] = False
        unit["metadata"]["index_quality"] = quality

    write_jsonl(output_dir / "table_index_units.qa.preview.jsonl", qa_units)
    write_csv(output_dir / "table_index_units.qa.preview.csv", [qa_csv_row(unit) for unit in qa_units], QA_PREVIEW_CSV_FIELDS)
    write_csv(output_dir / "table_index_unit_quality.csv", [quality_csv_row(unit) for unit in qa_units], QUALITY_CSV_FIELDS)
    write_csv(output_dir / "header_path_issue_cases.csv", header_issue_rows, HEADER_ISSUE_FIELDS)

    render_guardrail_report(report_dir / "phase7i_1_guardrail.md")
    render_content_qa_report(
        report_dir / "phase7i1_content_qa_review.md",
        qa_units,
        boundary_checks,
        phase7i_csv_rows,
        phase7i_stats_rows,
    )
    render_header_diff_report(report_dir / "phase7i1_header_map_diff.md", qa_units, header_issue_rows)

    quality_counts = Counter((unit["metadata"]["index_quality"]["index_text_quality"]) for unit in qa_units)
    header_counts = Counter((unit["metadata"]["index_quality"]["header_path_quality"]) for unit in qa_units)
    ready_counts = Counter(str(unit["metadata"]["index_quality"]["retrieval_ready"]).lower() for unit in qa_units)

    return {
        "input_unit_count": len(units),
        "qa_unit_count": len(qa_units),
        "formal_seed_count": len({row["seed_id"] for row in formal_rows}),
        "unit_type_counts": dict(Counter(unit["unit_type"] for unit in qa_units)),
        "index_text_quality_counts": dict(quality_counts),
        "header_path_quality_counts": dict(header_counts),
        "retrieval_ready_counts": dict(ready_counts),
        "header_issue_count": len(header_issue_rows),
        "boundary_checks": boundary_checks,
        "output_dir": rel(output_dir),
        "report_dir": rel(report_dir),
    }


def render_guardrail_report(path: Path) -> None:
    lines = [
        "# Phase7I-1 Guardrail",
        "",
        "## 定位",
        "",
        "1. 本轮定位为 table index unit QA and header-map hardening。",
        "2. 本轮是 Phase7I 到 Phase7J 之间的质量闸门。",
        "3. 本轮不是 retrieval。",
        "4. 本轮不是 embedding。",
        "5. 本轮不是 index construction。",
        "6. 本轮不是 production。",
        "7. 本轮不扩大候选池。",
        "8. 本轮不覆盖 Phase7I 原始产物。",
        "9. 本轮只使用 Phase7H formal 15 条 seed 和 Phase7I 414 个 preview units。",
        "10. `retrieval_ready=true` 只表示可进入 Phase7J offline retrieval preview。",
        "11. `retrieval_ready` 不等于 `production_ready`。",
        "12. 本轮不访问 Milvus / BM25。",
        "13. 本轮不伪造 value-level bbox。",
        "14. Route C 仍只是 backlog。",
        "",
        "## 边界",
        "",
        "- QA metadata 仅写入 `metadata.index_quality`，不扩张顶层 schema。",
        "- Phase7I 原始 `table_index_units.preview.*` 保留为只读输入。",
        "- QA 版 preview 仍保持 `guardrail.index_unit_status=preview_only`、`production_ready=false`、`is_official_benchmark_seed=false`。",
        "- `provenance.value_bboxes_available=false` 继续继承；cell bbox 不被提升为 value bbox。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_content_qa_report(
    path: Path,
    qa_units: list[dict[str, Any]],
    boundary_checks: list[dict[str, str]],
    phase7i_csv_rows: list[dict[str, str]],
    phase7i_stats_rows: list[dict[str, str]],
) -> None:
    quality_rows = [quality_csv_row(unit) for unit in qa_units]
    quality_counts = Counter(row["index_text_quality"] for row in quality_rows)
    header_counts = Counter(row["header_path_quality"] for row in quality_rows)
    ready_counts = Counter(row["retrieval_ready"] for row in quality_rows)
    flag_counts: Counter[str] = Counter()
    for unit in qa_units:
        flag_counts.update((unit["metadata"]["index_quality"].get("quality_flags") or []))

    lines = [
        "# Phase7I-1 内容 QA Review",
        "",
        "## 输入边界",
        "",
        f"- Phase7I preview CSV 行数：`{len(phase7i_csv_rows)}`",
        f"- Phase7I stats seed 行数：`{len(phase7i_stats_rows)}`",
        f"- QA 后 unit 数量：`{len(qa_units)}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in boundary_checks:
        detail = check["detail"].replace("|", "\\|")
        lines.append(f"| `{check['name']}` | `{check['status']}` | {detail} |")

    lines.extend(
        [
            "",
            "## 质量统计",
            "",
            f"- index_text_quality high / medium / low：`{quality_counts.get('high', 0)}` / `{quality_counts.get('medium', 0)}` / `{quality_counts.get('low', 0)}`",
            f"- header_path_quality pass / warning / fail：`{header_counts.get('pass', 0)}` / `{header_counts.get('warning', 0)}` / `{header_counts.get('fail', 0)}`",
            f"- retrieval_ready true / false：`{ready_counts.get('true', 0)}` / `{ready_counts.get('false', 0)}`",
            "",
            "## quality_flags 分布",
            "",
            "| quality_flag | count |",
            "| --- | ---: |",
        ]
    )
    for flag, count in flag_counts.most_common():
        lines.append(f"| `{flag}` | {count} |")

    flagged = [
        unit
        for unit in qa_units
        if unit["metadata"]["index_quality"].get("quality_flags")
        or not unit["metadata"]["index_quality"].get("retrieval_ready")
    ][:30]
    lines.extend(
        [
            "",
            "## 典型被标记 unit",
            "",
            "| unit_id | unit_type | doc_id | table_id | row_label | flags | notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for unit in flagged:
        metadata = unit.get("metadata") or {}
        quality = metadata.get("index_quality") or {}
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                unit.get("table_index_unit_id"),
                unit.get("unit_type"),
                unit.get("doc_id"),
                unit.get("table_id"),
                metadata.get("row_label", ""),
                ";".join(quality.get("quality_flags") or []),
                ";".join(quality.get("quality_notes") or []).replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- QA preview 保留全部 Phase7I 414 个 unit。",
            "- 低质量或 header fail 的 unit 保留在 QA preview，但不会自动进入 Phase7J eligible subset。",
            "- `retrieval_ready` 只作为 Phase7J offline retrieval preview 的输入资格，不代表 production_ready。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_header_diff_report(
    path: Path,
    qa_units: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
) -> None:
    fixed_rows = [row for row in issue_rows if row["action"] == "fixed_by_header_map_override"]
    remaining_bad_gos = [
        unit.get("table_index_unit_id")
        for unit in qa_units
        if is_doc0261_override_scope(unit) and unit_contains_current_bad_gos(unit)
    ]
    lines = [
        "# Phase7I-1 Header Map Diff",
        "",
        "## doc_0261 Table 2 / Table 3 override",
        "",
        "| column | Phase7I header_path | Phase7I-1 QA header_path |",
        "| ---: | --- | --- |",
        "| 1 | `Taxon` | `Taxon` |",
        "| 2 | `Abundance, % (mean ± SD) / Control` | `Abundance, % (mean ± SD) / Control` |",
        "| 3 | `Abundance, % (mean ± SD) / 2′-FL` | `Abundance, % (mean ± SD) / 2′-FL` |",
        "| 4 | `Abundance, % (mean ± SD) / Lactose` | `Abundance, % (mean ± SD) / Lactose` |",
        "| 5 | `Overall p-value / GOS` | `Abundance, % (mean ± SD) / GOS` |",
        "| 6 | `Overall p-value / (FDR adj)1` | `Overall p-value / (FDR adj)1` |",
        "",
        "## 修复统计",
        "",
        f"- fixed header_path issue rows：`{len(fixed_rows)}`",
        f"- QA 后仍包含 `Overall p-value / GOS` 的 doc_0261 Table 2/3 unit 数量：`{len(remaining_bad_gos)}`",
        "",
        "## 修复样例",
        "",
        "| unit_id | unit_type | table_id | row_label | before | after | value |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in fixed_rows[:20]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["table_index_unit_id"],
                row["unit_type"],
                row["table_id"],
                row["row_label"],
                row["original_header_path"],
                row["qa_header_path"],
                row["value"],
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- doc_0261 Table 2 / Table 3 的 GOS 列已归入 `Abundance, % (mean ± SD)` group。",
            "- `Overall p-value` 只保留 `(FDR adj)1` 子列。",
            "- 修复仅写入 QA 版 preview，不覆盖 Phase7I 原始 preview。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unit_contains_current_bad_gos(unit: dict[str, Any]) -> bool:
    metadata = deepcopy(unit.get("metadata") or {})
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase7I-1 table index unit QA artifacts.")
    parser.add_argument("--phase7i-units", type=Path, default=DEFAULT_PHASE7I_UNITS_PATH)
    parser.add_argument("--phase7i-units-csv", type=Path, default=DEFAULT_PHASE7I_UNITS_CSV_PATH)
    parser.add_argument("--phase7i-stats", type=Path, default=DEFAULT_PHASE7I_STATS_PATH)
    parser.add_argument("--table-units", type=Path, default=DEFAULT_TABLE_UNIT_PATH)
    parser.add_argument("--row-units", type=Path, default=DEFAULT_ROW_UNIT_PATH)
    parser.add_argument("--cell-group-units", type=Path, default=DEFAULT_CELL_GROUP_UNIT_PATH)
    parser.add_argument("--formal-validation", type=Path, default=DEFAULT_FORMAL_VALIDATION_PATH)
    parser.add_argument("--review-pack-index", type=Path, default=DEFAULT_REVIEW_PACK_INDEX_PATH)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL_PATH)
    parser.add_argument("--csv-tables-dir", type=Path, default=DEFAULT_CSV_TABLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_qa(
        phase7i_units_path=args.phase7i_units,
        phase7i_units_csv_path=args.phase7i_units_csv,
        phase7i_stats_path=args.phase7i_stats,
        table_unit_path=args.table_units,
        row_unit_path=args.row_units,
        cell_group_unit_path=args.cell_group_units,
        formal_validation_path=args.formal_validation,
        review_pack_index_path=args.review_pack_index,
        candidate_pool_path=args.candidate_pool,
        csv_tables_dir=args.csv_tables_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
