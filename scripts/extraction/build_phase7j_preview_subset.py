#!/usr/bin/env python3
"""Build the Phase7J offline retrieval preview eligible subset from QA units."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/table_index_units.qa.preview.jsonl"
)
DEFAULT_QUALITY_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/table_index_unit_quality.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_table_index_unit_qa"

ELIGIBLE_CSV_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "row_index",
    "row_label",
    "index_text_quality",
    "header_path_quality",
    "retrieval_ready",
    "quality_flags",
    "content_text_for_embedding",
    "index_unit_status",
    "production_ready",
    "value_bboxes_available",
]

EXCLUDED_CSV_FIELDS = [
    "table_index_unit_id",
    "unit_type",
    "seed_id",
    "doc_id",
    "table_id",
    "row_index",
    "row_label",
    "index_text_quality",
    "header_path_quality",
    "retrieval_ready",
    "quality_flags",
    "excluded_reasons",
    "content_text_preview",
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


def preview(text: Any, limit: int = 260) -> str:
    value = normalize(text)
    return value[:limit] + ("..." if len(value) > limit else "")


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


def parse_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def flags_for(unit: dict[str, Any]) -> list[str]:
    quality = (unit.get("metadata") or {}).get("index_quality") or {}
    flags = quality.get("quality_flags") or []
    if isinstance(flags, str):
        return [flag for flag in flags.split(";") if flag]
    return list(flags)


def quality_for(unit: dict[str, Any], quality_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    inline = (unit.get("metadata") or {}).get("index_quality") or {}
    csv_row = quality_by_id.get(unit.get("table_index_unit_id"), {})
    return {
        "index_text_quality": inline.get("index_text_quality") or csv_row.get("index_text_quality", ""),
        "header_path_quality": inline.get("header_path_quality") or csv_row.get("header_path_quality", ""),
        "retrieval_ready": inline.get("retrieval_ready")
        if "retrieval_ready" in inline
        else parse_bool(csv_row.get("retrieval_ready", "")),
        "quality_flags": flags_for(unit) or [flag for flag in csv_row.get("quality_flags", "").split(";") if flag],
    }


def information_density_ok(unit: dict[str, Any]) -> bool:
    text = normalize(unit.get("content_text_for_embedding"))
    if not text:
        return False
    metadata = unit.get("metadata") or {}
    if unit.get("unit_type") == "table_unit":
        return len(metadata.get("column_headers") or []) > 1 or len(text) >= 80
    values = metadata.get("row_values") or metadata.get("cell_group_values") or []
    nonempty_values = [
        item
        for item in values
        if isinstance(item, dict)
        and normalize(item.get("value"))
        and normalize(item.get("column_header"))
    ]
    return len(nonempty_values) >= 2


def exclusion_reasons(unit: dict[str, Any], quality_by_id: dict[str, dict[str, str]]) -> list[str]:
    quality = quality_for(unit, quality_by_id)
    flags = set(quality["quality_flags"])
    guardrail = unit.get("guardrail") or {}
    provenance = unit.get("provenance") or {}
    reasons: list[str] = []

    if quality["retrieval_ready"] is not True:
        reasons.append("retrieval_ready_false")
    if quality["header_path_quality"] == "fail":
        reasons.append("header_path_quality_fail")
    if quality["index_text_quality"] == "low":
        reasons.append("index_text_quality_low")
    if "p_value_parent_mismatch" in flags:
        reasons.append("p_value_parent_mismatch")
    if "header_path_contains_data_value" in flags:
        reasons.append("header_path_contains_data_value")
    if not normalize(unit.get("content_text_for_embedding")):
        reasons.append("empty_content_text_for_embedding")
    if not information_density_ok(unit):
        reasons.append("information_density_below_threshold")
    if guardrail.get("production_ready") is not False:
        reasons.append("production_ready_not_false")
    if guardrail.get("index_unit_status") != "preview_only":
        reasons.append("index_unit_status_not_preview_only")
    if provenance.get("value_bboxes_available") is not False:
        reasons.append("value_bboxes_available_not_false")
    if "value_level_bbox_claim_detected" in flags:
        reasons.append("value_level_bbox_claim_detected")
    if unit.get("unit_type") == "cell_group_unit" and flags.intersection(
        {"header_path_uncertain", "low_information_content", "p_value_parent_mismatch"}
    ):
        reasons.append("cell_group_strict_quality_gate")
    return sorted(set(reasons))


def unit_csv_row(unit: dict[str, Any], quality_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    metadata = unit.get("metadata") or {}
    quality = quality_for(unit, quality_by_id)
    guardrail = unit.get("guardrail") or {}
    provenance = unit.get("provenance") or {}
    return {
        "table_index_unit_id": unit.get("table_index_unit_id", ""),
        "unit_type": unit.get("unit_type", ""),
        "seed_id": unit.get("seed_id", ""),
        "candidate_id": unit.get("candidate_id", ""),
        "doc_id": unit.get("doc_id", ""),
        "table_id": unit.get("table_id", ""),
        "row_index": metadata.get("row_index", ""),
        "row_label": metadata.get("row_label", ""),
        "index_text_quality": quality["index_text_quality"],
        "header_path_quality": quality["header_path_quality"],
        "retrieval_ready": str(quality["retrieval_ready"]).lower(),
        "quality_flags": ";".join(quality["quality_flags"]),
        "content_text_for_embedding": unit.get("content_text_for_embedding", ""),
        "index_unit_status": guardrail.get("index_unit_status", ""),
        "production_ready": str(guardrail.get("production_ready")).lower(),
        "value_bboxes_available": str(provenance.get("value_bboxes_available")).lower(),
    }


def excluded_csv_row(
    unit: dict[str, Any],
    quality_by_id: dict[str, dict[str, str]],
    reasons: list[str],
) -> dict[str, Any]:
    row = unit_csv_row(unit, quality_by_id)
    return {
        "table_index_unit_id": row["table_index_unit_id"],
        "unit_type": row["unit_type"],
        "seed_id": row["seed_id"],
        "doc_id": row["doc_id"],
        "table_id": row["table_id"],
        "row_index": row["row_index"],
        "row_label": row["row_label"],
        "index_text_quality": row["index_text_quality"],
        "header_path_quality": row["header_path_quality"],
        "retrieval_ready": row["retrieval_ready"],
        "quality_flags": row["quality_flags"],
        "excluded_reasons": ";".join(reasons),
        "content_text_preview": preview(unit.get("content_text_for_embedding")),
    }


def build_phase7j_preview_subset(
    qa_units_path: Path = DEFAULT_QA_UNITS_PATH,
    quality_path: Path = DEFAULT_QUALITY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    qa_units_path = resolve_path(qa_units_path)
    quality_path = resolve_path(quality_path)
    output_dir = resolve_path(output_dir)

    units = load_jsonl(qa_units_path)
    quality_rows = load_csv(quality_path)
    quality_by_id = {row["table_index_unit_id"]: row for row in quality_rows}

    eligible_units: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()

    for unit in units:
        reasons = exclusion_reasons(unit, quality_by_id)
        if reasons:
            excluded_rows.append(excluded_csv_row(unit, quality_by_id, reasons))
            reason_counter.update(reasons)
        else:
            eligible_units.append(unit)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "phase7j_preview_eligible_units.jsonl", eligible_units)
    write_csv(
        output_dir / "phase7j_preview_eligible_units.csv",
        [unit_csv_row(unit, quality_by_id) for unit in eligible_units],
        ELIGIBLE_CSV_FIELDS,
    )
    write_csv(output_dir / "content_quality_excluded_units.csv", excluded_rows, EXCLUDED_CSV_FIELDS)

    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "excluded": 0})
    for unit in eligible_units:
        by_type[unit.get("unit_type", "")]["eligible"] += 1
    for row in excluded_rows:
        by_type[row.get("unit_type", "")]["excluded"] += 1

    by_seed = Counter(unit.get("seed_id", "") for unit in eligible_units)
    return {
        "input_unit_count": len(units),
        "eligible_unit_count": len(eligible_units),
        "excluded_unit_count": len(excluded_rows),
        "eligible_excluded_by_unit_type": dict(by_type),
        "eligible_by_seed": dict(sorted(by_seed.items())),
        "excluded_reason_distribution": dict(reason_counter.most_common()),
        "output_dir": rel(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase7J offline retrieval preview eligible subset.")
    parser.add_argument("--qa-units", type=Path, default=DEFAULT_QA_UNITS_PATH)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_phase7j_preview_subset(
        qa_units_path=args.qa_units,
        quality_path=args.quality,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
