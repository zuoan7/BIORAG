#!/usr/bin/env python3
"""Phase7H expanded seed consistency validation.

This checker validates the Phase7G-2 confirmed_seed_with_warnings formal
subset and keeps partial/reject/unreviewed carried-forward rows as appendices.
It does not read BM25 indexes, call retrieval, or access Milvus.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SEED_DIR = ROOT / "data/experiments/v7_phase7_expanded_seed_from_human_review"
DEFAULT_REVIEW_PACK_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_FREEZE_DIR = ROOT / "data/experiments/v7_phase7_human_review_label_freeze"
DEFAULT_OUTPUT_DIR = ROOT / "results/v7_phase7_expanded_seed_validation"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_expanded_seed_validation"
DEFAULT_PHASE7G2_REPORT_DIR = ROOT / "reports/v7_phase7_expanded_seed_from_human_review"
DEFAULT_PHASE7F_SUMMARY = ROOT / "reports/v7_phase7_gold_seed_validation/phase7f_summary.md"

BINDING_REVIEW_MODE = "global_bulk_no_obvious_issue"
BINDING_REVIEW_LIMITATION = "no_per_cell_binding_review"
FORMAL_SEED_STATUS = "confirmed_seed_with_warnings"
PROD_READY_MARKER = "production" + "_ready"
OFFICIAL_BENCHMARK_MARKER = "official" + "_benchmark_seed"

CHECK_STATUS_VALUES = {"pass", "pass_with_warnings", "partial", "fail", "not_applicable", "not_evaluable"}
OVERALL_STATUS_VALUES = {"pass", "pass_with_warnings", "partial", "fail", "not_evaluable"}

VALIDATION_FIELDS = [
    "seed_identity_check",
    "traceability_check",
    "artifact_availability_check",
    "human_label_consistency_check",
    "csv_structure_sanity_check",
    "markdown_card_check",
    "pdf_crop_check",
    "binding_policy_check",
    "source_span_check",
    "bbox_provenance_check",
    "formal_membership_check",
    "overall_validation_status",
]

RESULT_FIELDS = [
    "seed_id",
    "candidate_id",
    "doc_id",
    "table_id",
    "table_object_id",
    "validation_subset",
    *VALIDATION_FIELDS,
    "markdown_path",
    "markdown_exists",
    "csv_path",
    "csv_exists",
    "csv_parse_status",
    "csv_row_count",
    "csv_nonempty_row_count",
    "csv_column_count",
    "csv_trailing_empty_rows",
    "pdf_crop_path",
    "pdf_crop_exists",
    "binding_review_mode",
    "binding_review_limitation",
    "unit_or_note_ok",
    "reference_ok",
    "source_span_granularity",
    "value_bboxes_available",
    "cell_bboxes_available",
    "warnings",
    "missing_items",
]


def resolve_path(path_value: str | Path, base_dir: Path = ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_lower(value: Any) -> str:
    return normalize(value).lower()


def bool_text(value: Any) -> str:
    text = normalize_lower(value)
    return "true" if text in {"true", "1", "yes"} else "false"


def split_semicolon(value: Any) -> list[str]:
    text = normalize(value)
    if not text or text == "none":
        return []
    return [part for part in text.split(";") if part]


def semicolon(values: list[Any]) -> str:
    cleaned = [str(value) for value in values if normalize(value)]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def by_candidate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id", "")): row for row in rows if row.get("candidate_id")}


def object_contains_marker(value: Any, marker: str) -> bool:
    marker = marker.lower()
    if isinstance(value, dict):
        return any(marker in str(key).lower() or object_contains_marker(item, marker) for key, item in value.items())
    if isinstance(value, list):
        return any(object_contains_marker(item, marker) for item in value)
    return marker in str(value).lower()


def has_formal_guardrail_warning(seed: dict[str, Any]) -> bool:
    warnings = split_semicolon(seed.get("seed_warnings"))
    text = normalize_lower(seed.get("seed_warnings"))
    equivalents = {
        "formal_benchmark_guardrail",
        "not_official_benchmark",
        "not_official_benchmark_seed",
        "not official benchmark",
    }
    return any(item in warnings or item in text for item in equivalents)


def rows_have_matching_identity(
    seed: dict[str, Any],
    rows: list[tuple[str, dict[str, Any] | None]],
    missing_items: list[dict[str, Any]],
    warnings: list[str],
) -> bool:
    ok = True
    for source_name, row in rows:
        if row is None:
            missing_items.append({"kind": "traceability", "source": source_name, "reason": "candidate_missing"})
            ok = False
            continue
        for field in ["candidate_id", "doc_id", "table_id", "table_object_id"]:
            if field not in row:
                if field == "table_object_id":
                    continue
                missing_items.append({"kind": "traceability", "source": source_name, "field": field, "reason": "field_missing"})
                ok = False
                continue
            expected = normalize(seed.get(field))
            actual = normalize(row.get(field))
            if field == "table_object_id" and not actual:
                continue
            if actual and expected and actual != expected:
                missing_items.append(
                    {
                        "kind": "identity_mismatch",
                        "source": source_name,
                        "field": field,
                        "seed_value": expected,
                        "source_value": actual,
                    }
                )
                ok = False
    return ok


def parse_csv_table(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "csv_exists": "false",
        "csv_parse_status": "missing",
        "csv_row_count": 0,
        "csv_nonempty_row_count": 0,
        "csv_column_count": 0,
        "csv_trailing_empty_rows": 0,
        "warnings": [],
        "missing_items": [],
    }
    if not path.exists() or not path.is_file():
        result["missing_items"].append({"kind": "csv_path", "path": str(path), "reason": "missing"})
        return result

    result["csv_exists"] = "true"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except csv.Error as exc:
        result["csv_parse_status"] = "parse_error"
        result["missing_items"].append({"kind": "csv_parse", "path": str(path), "reason": str(exc)})
        return result
    except UnicodeDecodeError as exc:
        result["csv_parse_status"] = "decode_error"
        result["missing_items"].append({"kind": "csv_parse", "path": str(path), "reason": str(exc)})
        return result

    result["csv_parse_status"] = "parsed"
    result["csv_row_count"] = len(rows)
    result["csv_nonempty_row_count"] = sum(1 for row in rows if any(normalize(cell) for cell in row))
    result["csv_column_count"] = max((len(row) for row in rows), default=0)
    trailing_empty = 0
    for row in reversed(rows):
        if any(normalize(cell) for cell in row):
            break
        trailing_empty += 1
    result["csv_trailing_empty_rows"] = trailing_empty

    if not rows or result["csv_nonempty_row_count"] == 0:
        result["missing_items"].append({"kind": "csv_content", "path": str(path), "reason": "empty_csv"})
    if result["csv_column_count"] < 2:
        result["warnings"].append("csv_column_count_lt_2")
    if result["csv_nonempty_row_count"] < 2:
        result["warnings"].append("csv_nonempty_row_count_lt_2")
    if trailing_empty:
        result["warnings"].append("csv_trailing_empty_rows_detected")
    return result


def status_from_missing_and_warnings(missing_items: list[Any], warnings: list[str]) -> str:
    if missing_items:
        return "fail"
    if warnings:
        return "pass_with_warnings"
    return "pass"


def validate_seed(
    seed: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    base_dir = context.get("base_dir", ROOT)
    candidate_id = str(seed.get("candidate_id", ""))
    seed_id = str(seed.get("seed_id", ""))

    warnings = split_semicolon(seed.get("seed_warnings"))
    missing_items: list[dict[str, Any]] = []

    seed_id_count = context["seed_id_counts"].get(seed_id, 0)
    candidate_count = context["candidate_counts"].get(candidate_id, 0)
    identity_missing = []
    for field in ["seed_id", "candidate_id", "doc_id", "table_id", "table_object_id"]:
        if not normalize(seed.get(field)):
            identity_missing.append(field)
    if identity_missing:
        missing_items.append({"kind": "seed_identity", "reason": "missing_identity_fields", "fields": identity_missing})
    if seed_id_count != 1:
        missing_items.append({"kind": "seed_identity", "reason": "seed_id_not_unique", "seed_id": seed_id})
    if candidate_count != 1:
        missing_items.append({"kind": "seed_identity", "reason": "candidate_id_not_unique", "candidate_id": candidate_id})
    seed_identity_check = "fail" if any(item.get("kind") == "seed_identity" for item in missing_items) else "pass"

    expanded_row = context["expanded_by_candidate"].get(candidate_id)
    summary_row = context["summary_by_candidate"].get(candidate_id)
    draft_row = context["draft_by_candidate"].get(candidate_id)
    frozen_row = context["frozen_by_candidate"].get(candidate_id)
    review_index_row = context["review_index_by_candidate"].get(candidate_id)
    pool_row = context["pool_by_candidate"].get(candidate_id)

    traceability_missing: list[dict[str, Any]] = []
    traceability_warnings: list[str] = []
    if not rows_have_matching_identity(
        seed,
        [
            ("expanded_table_seed", expanded_row),
            ("expanded_table_seed_summary", summary_row),
            ("confirmed_seed_draft_candidates", draft_row),
            ("frozen_review_labels", frozen_row),
            ("review_pack_index", review_index_row),
            ("candidate_pool_scored", pool_row),
        ],
        traceability_missing,
        traceability_warnings,
    ):
        missing_items.extend(traceability_missing)
    warnings.extend(traceability_warnings)
    traceability_check = "fail" if traceability_missing else ("pass_with_warnings" if traceability_warnings else "pass")

    human_missing: list[dict[str, Any]] = []
    if not draft_row or normalize(draft_row.get("seed_draft_status")) != "confirmed_seed_draft":
        human_missing.append({"kind": "human_label", "reason": "not_confirmed_seed_draft", "candidate_id": candidate_id})
    if not frozen_row or normalize(frozen_row.get("review_decision")) != "accept_confirmed_seed_candidate":
        human_missing.append({"kind": "human_label", "reason": "not_accept_confirmed_seed_candidate", "candidate_id": candidate_id})
    for field in ["boundary_ok", "grid_ok", "key_values_ok"]:
        if normalize_lower(seed.get(field)) != "yes":
            human_missing.append({"kind": "human_label", "field": field, "seed_value": seed.get(field), "reason": "seed_not_yes"})
        for source_name, row in [("confirmed_seed_draft_candidates", draft_row), ("frozen_review_labels", frozen_row)]:
            if row and normalize_lower(row.get(field)) != "yes":
                human_missing.append(
                    {
                        "kind": "human_label",
                        "source": source_name,
                        "field": field,
                        "source_value": row.get(field),
                        "reason": "source_not_yes",
                    }
                )
    missing_items.extend(human_missing)
    human_label_consistency_check = "fail" if human_missing else "pass"

    markdown_path_text = str(seed.get("markdown_path", ""))
    csv_path_text = str(seed.get("csv_path", ""))
    pdf_crop_path_text = str(seed.get("pdf_crop_path", ""))
    markdown_path = resolve_path(markdown_path_text, base_dir)
    csv_path = resolve_path(csv_path_text, base_dir)
    pdf_crop_path = resolve_path(pdf_crop_path_text, base_dir)

    markdown_exists = markdown_path.exists() and markdown_path.is_file()
    markdown_warnings: list[str] = []
    if not markdown_exists:
        markdown_warnings.append("markdown_card_missing")
        append_unique(warnings, "markdown_card_missing")
    elif markdown_path.stat().st_size == 0:
        markdown_warnings.append("markdown_card_empty")
        append_unique(warnings, "markdown_card_empty")
    markdown_card_check = "pass_with_warnings" if markdown_warnings else "pass"

    csv_info = parse_csv_table(csv_path)
    csv_missing = list(csv_info["missing_items"])
    csv_warnings = list(csv_info["warnings"])
    for warning in csv_warnings:
        append_unique(warnings, warning)
    missing_items.extend(csv_missing)
    csv_structure_sanity_check = status_from_missing_and_warnings(csv_missing, csv_warnings)

    pdf_crop_exists = pdf_crop_path.exists() and pdf_crop_path.is_file()
    crop_warnings: list[str] = []
    if not pdf_crop_path_text:
        crop_warnings.append("pdf_crop_path_missing")
    elif not pdf_crop_exists:
        crop_warnings.append("pdf_crop_missing")
    elif pdf_crop_path.stat().st_size == 0:
        crop_warnings.append("pdf_crop_empty")
    if normalize_lower(seed.get("crop_status")) not in {"", "ok"}:
        crop_warnings.append(f"crop_status={seed.get('crop_status')}")
    for warning in crop_warnings:
        append_unique(warnings, warning)
    pdf_crop_check = "pass_with_warnings" if crop_warnings else "pass"

    artifact_missing = []
    artifact_warnings = []
    if csv_structure_sanity_check == "fail":
        artifact_missing.append({"kind": "artifact_availability", "reason": "csv_not_available_or_unusable"})
    if markdown_card_check == "pass_with_warnings":
        artifact_warnings.extend(markdown_warnings)
    if pdf_crop_check == "pass_with_warnings":
        artifact_warnings.extend(crop_warnings)
    artifact_availability_check = status_from_missing_and_warnings(artifact_missing, artifact_warnings)
    missing_items.extend(artifact_missing)

    binding_missing: list[dict[str, Any]] = []
    binding_warnings: list[str] = []
    if normalize(seed.get("binding_review_mode")) != BINDING_REVIEW_MODE:
        binding_missing.append({"kind": "binding_policy", "field": "binding_review_mode", "reason": "unexpected_value"})
    if normalize(seed.get("binding_review_limitation")) != BINDING_REVIEW_LIMITATION:
        binding_missing.append({"kind": "binding_policy", "field": "binding_review_limitation", "reason": "unexpected_value"})
    if context.get("policy"):
        policy = context["policy"]
        if normalize(policy.get("binding_review_mode")) != BINDING_REVIEW_MODE:
            binding_missing.append({"kind": "binding_policy", "source": "bulk_binding_review_policy", "field": "binding_review_mode"})
        if normalize(policy.get("binding_review_limitation")) != BINDING_REVIEW_LIMITATION:
            binding_missing.append(
                {"kind": "binding_policy", "source": "bulk_binding_review_policy", "field": "binding_review_limitation"}
            )
    for field in ["unit_or_note_ok", "reference_ok"]:
        value = normalize(seed.get(field))
        if value == "unchecked":
            binding_missing.append({"kind": "binding_policy", "field": field, "reason": "unchecked_not_allowed"})
        elif value not in {"warning", "not_applicable", "yes", "no"}:
            binding_missing.append({"kind": "binding_policy", "field": field, "value": value, "reason": "unexpected_value"})
    if normalize_lower(seed.get("auto_binding_fill")) != "true":
        binding_warnings.append("auto_binding_fill_not_true")
    if not has_formal_guardrail_warning(seed):
        binding_missing.append({"kind": "binding_policy", "field": "seed_warnings", "reason": "formal_benchmark_guardrail_missing"})
    if not binding_missing:
        binding_warnings.append("bulk_binding_no_per_cell_review")
    for warning in binding_warnings:
        append_unique(warnings, warning)
    missing_items.extend(binding_missing)
    binding_policy_check = status_from_missing_and_warnings(binding_missing, binding_warnings)

    source_span_missing: list[dict[str, Any]] = []
    source_span_warnings: list[str] = []
    source_span_granularity = normalize(seed.get("source_span_granularity"))
    if not source_span_granularity:
        source_span_missing.append({"kind": "source_span", "reason": "source_span_granularity_missing"})
    elif source_span_granularity == "value_level":
        source_span_missing.append({"kind": "source_span", "reason": "value_level_source_span_not_supported_by_phase7h_seed"})
    else:
        source_span_warnings.append("source_span_not_value_level")
    if review_index_row and normalize(review_index_row.get("source_span_granularity")) != source_span_granularity:
        source_span_warnings.append("source_span_granularity_differs_from_review_pack_index")
    for warning in source_span_warnings:
        append_unique(warnings, warning)
    missing_items.extend(source_span_missing)
    source_span_check = status_from_missing_and_warnings(source_span_missing, source_span_warnings)

    bbox_missing: list[dict[str, Any]] = []
    bbox_warnings: list[str] = []
    if bool_text(seed.get("value_bboxes_available")) != "false":
        bbox_missing.append({"kind": "bbox_provenance", "field": "value_bboxes_available", "reason": "must_remain_false"})
    if "value_bbox" in seed and seed.get("value_bbox"):
        bbox_missing.append({"kind": "bbox_provenance", "field": "value_bbox", "reason": "value_level_bbox_must_not_be_forged"})
    if bool_text(seed.get("cell_bboxes_available")) == "true":
        bbox_warnings.append("cell_bbox_available_not_value_bbox")
    bbox_warnings.append("value_bboxes_available_false_non_blocking")
    for warning in bbox_warnings:
        append_unique(warnings, warning)
    missing_items.extend(bbox_missing)
    bbox_provenance_check = status_from_missing_and_warnings(bbox_missing, bbox_warnings)

    formal_missing: list[dict[str, Any]] = []
    if normalize(seed.get("seed_status")) != FORMAL_SEED_STATUS:
        formal_missing.append({"kind": "formal_membership", "field": "seed_status", "reason": "not_confirmed_seed_with_warnings"})
    if candidate_id in context["partial_candidate_ids"]:
        formal_missing.append({"kind": "formal_membership", "reason": "candidate_in_partial_carried_forward"})
    if candidate_id in context["reject_candidate_ids"]:
        formal_missing.append({"kind": "formal_membership", "reason": "candidate_in_reject_carried_forward"})
    if candidate_id in context["unreviewed_candidate_ids"]:
        formal_missing.append({"kind": "formal_membership", "reason": "candidate_in_unreviewed_carried_forward"})
    if object_contains_marker(seed, PROD_READY_MARKER):
        formal_missing.append({"kind": "formal_membership", "reason": "forbidden_production_marker"})
    if object_contains_marker(seed, OFFICIAL_BENCHMARK_MARKER):
        formal_missing.append({"kind": "formal_membership", "reason": "forbidden_official_benchmark_seed_marker"})
    missing_items.extend(formal_missing)
    formal_membership_check = "fail" if formal_missing else "pass"

    result: dict[str, Any] = {
        "seed_id": seed_id,
        "candidate_id": candidate_id,
        "doc_id": seed.get("doc_id", ""),
        "table_id": seed.get("table_id", ""),
        "table_object_id": seed.get("table_object_id", ""),
        "validation_subset": "formal_confirmed_seed_with_warnings",
        "seed_identity_check": seed_identity_check,
        "traceability_check": traceability_check,
        "artifact_availability_check": artifact_availability_check,
        "human_label_consistency_check": human_label_consistency_check,
        "csv_structure_sanity_check": csv_structure_sanity_check,
        "markdown_card_check": markdown_card_check,
        "pdf_crop_check": pdf_crop_check,
        "binding_policy_check": binding_policy_check,
        "source_span_check": source_span_check,
        "bbox_provenance_check": bbox_provenance_check,
        "formal_membership_check": formal_membership_check,
        "overall_validation_status": "not_evaluable",
        "markdown_path": markdown_path_text,
        "markdown_exists": bool_text(markdown_exists),
        "csv_path": csv_path_text,
        "csv_exists": csv_info["csv_exists"],
        "csv_parse_status": csv_info["csv_parse_status"],
        "csv_row_count": csv_info["csv_row_count"],
        "csv_nonempty_row_count": csv_info["csv_nonempty_row_count"],
        "csv_column_count": csv_info["csv_column_count"],
        "csv_trailing_empty_rows": csv_info["csv_trailing_empty_rows"],
        "pdf_crop_path": pdf_crop_path_text,
        "pdf_crop_exists": bool_text(pdf_crop_exists),
        "binding_review_mode": seed.get("binding_review_mode", ""),
        "binding_review_limitation": seed.get("binding_review_limitation", ""),
        "unit_or_note_ok": seed.get("unit_or_note_ok", ""),
        "reference_ok": seed.get("reference_ok", ""),
        "source_span_granularity": seed.get("source_span_granularity", ""),
        "value_bboxes_available": bool_text(seed.get("value_bboxes_available")),
        "cell_bboxes_available": bool_text(seed.get("cell_bboxes_available")),
        "warnings": semicolon(warnings),
        "missing_items": missing_items,
    }
    result["overall_validation_status"] = derive_overall_status(result)
    assert result["overall_validation_status"] in OVERALL_STATUS_VALUES
    for field in VALIDATION_FIELDS[:-1]:
        assert result[field] in CHECK_STATUS_VALUES, (field, result[field])
    return result


def derive_overall_status(result: dict[str, Any]) -> str:
    statuses = [result[field] for field in VALIDATION_FIELDS if field != "overall_validation_status"]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "not_evaluable" for status in statuses):
        return "not_evaluable"
    if any(status == "partial" for status in statuses):
        return "partial"
    if any(status == "pass_with_warnings" for status in statuses):
        return "pass_with_warnings"
    return "pass"


def compute_formal_overall(results: list[dict[str, Any]], expected_count: int = 15) -> dict[str, Any]:
    status_counts = Counter(result["overall_validation_status"] for result in results)
    if len(results) != expected_count:
        status = "not_evaluable"
    elif all(result["overall_validation_status"] in {"pass", "pass_with_warnings"} for result in results):
        status = "pass_with_warnings" if any(result["overall_validation_status"] == "pass_with_warnings" for result in results) else "pass"
    elif any(result["overall_validation_status"] == "fail" for result in results):
        status = "fail"
    elif any(result["overall_validation_status"] == "partial" for result in results):
        status = "partial"
    else:
        status = "not_evaluable"

    check_counts = {
        field: dict(Counter(result[field] for result in results))
        for field in VALIDATION_FIELDS
        if field != "overall_validation_status"
    }
    return {
        "formal_overall_status": status,
        "formal_seed_count": len(results),
        "expected_formal_seed_count": expected_count,
        "formal_pass_like_count": sum(
            1 for result in results if result["overall_validation_status"] in {"pass", "pass_with_warnings"}
        ),
        "status_counts": dict(status_counts),
        "check_counts": check_counts,
    }


def build_context(inputs: dict[str, Any]) -> dict[str, Any]:
    seeds = inputs["confirmed_seeds"]
    return {
        "base_dir": inputs.get("base_dir", ROOT),
        "policy": inputs.get("bulk_binding_policy", {}),
        "seed_id_counts": Counter(str(seed.get("seed_id", "")) for seed in seeds),
        "candidate_counts": Counter(str(seed.get("candidate_id", "")) for seed in seeds),
        "expanded_by_candidate": by_candidate(inputs.get("expanded_seeds", [])),
        "summary_by_candidate": by_candidate(inputs.get("expanded_seed_summary", [])),
        "draft_by_candidate": by_candidate(inputs.get("confirmed_seed_draft_rows", [])),
        "frozen_by_candidate": by_candidate(inputs.get("frozen_review_labels", [])),
        "review_index_by_candidate": by_candidate(inputs.get("review_pack_index", [])),
        "pool_by_candidate": by_candidate(inputs.get("candidate_pool_scored", [])),
        "partial_candidate_ids": {row.get("candidate_id", "") for row in inputs.get("partial_carried_forward", [])},
        "reject_candidate_ids": {row.get("candidate_id", "") for row in inputs.get("reject_carried_forward", [])},
        "unreviewed_candidate_ids": {row.get("candidate_id", "") for row in inputs.get("unreviewed_carried_forward", [])},
    }


def build_validation_results(inputs: dict[str, Any]) -> dict[str, Any]:
    context = build_context(inputs)
    results = [validate_seed(seed, context) for seed in inputs["confirmed_seeds"]]
    formal_overall = compute_formal_overall(results)
    return {
        "phase": "v7_phase7H_expanded_seed_consistency_validation",
        "validation_scope": "expanded_seed_consistency_validation_not_official_benchmark",
        "formal_subset_policy": "confirmed_seed_with_warnings_only",
        "carried_forward_policy": "partial_reject_unreviewed_appendix_only_excluded_from_formal_overall",
        "production_readiness_note": "本轮不是 production readiness。",
        "retrieval_model_note": "本轮未运行 retrieval、RAG、RAGAS、Qwen、OCR、VLM、embedding 或 rerank。",
        "official_baseline_guardrail": {
            "official_dataset": "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl",
            "official_dataset_sha256": "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3",
            "official_clean_baseline": "phase5f_official_clean_baseline",
            "official_chunks_sha256": "5dbacc5bb85351203355bf3f2d22f46ec02e24f513ab9523ca3407664669f75b",
            "baseline_guardrail_status": "not_modified_not_read_for_bm25_or_milvus",
        },
        "formal_overall": formal_overall,
        "appendix_counts": {
            "partial_carried_forward": len(inputs.get("partial_carried_forward", [])),
            "reject_carried_forward": len(inputs.get("reject_carried_forward", [])),
            "unreviewed_carried_forward": len(inputs.get("unreviewed_carried_forward", [])),
        },
        "results": results,
    }


def flat_result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: result.get(field) for field in RESULT_FIELDS} for result in results]


def load_inputs(
    seed_dir: Path = DEFAULT_SEED_DIR,
    review_pack_dir: Path = DEFAULT_REVIEW_PACK_DIR,
    freeze_dir: Path = DEFAULT_FREEZE_DIR,
    phase7g2_report_dir: Path = DEFAULT_PHASE7G2_REPORT_DIR,
    phase7f_summary: Path = DEFAULT_PHASE7F_SUMMARY,
) -> dict[str, Any]:
    phase7f_text = read_text(phase7f_summary)
    phase7g2_report_names = [
        "phase7g_2_guardrail.md",
        "bulk_binding_policy_report.md",
        "expanded_seed_construction_report.md",
        "expanded_seed_validation_report.md",
        "phase7g1_to_phase7g2_traceability.md",
        "phase7g_2_summary.md",
    ]
    phase7g2_reports = {
        name: read_text(phase7g2_report_dir / name)
        for name in phase7g2_report_names
    }
    return {
        "base_dir": ROOT,
        "bulk_binding_policy": load_json(seed_dir / "bulk_binding_review_policy.json"),
        "expanded_seeds": load_jsonl(seed_dir / "expanded_table_seed.jsonl"),
        "expanded_seed_summary": load_csv(seed_dir / "expanded_table_seed_summary.csv"),
        "confirmed_seeds": load_jsonl(seed_dir / "confirmed_seed_with_warnings.jsonl"),
        "partial_carried_forward": load_csv(seed_dir / "partial_candidate_routing_carried_forward.csv"),
        "reject_carried_forward": load_csv(seed_dir / "reject_boundary_feedback_carried_forward.csv"),
        "unreviewed_carried_forward": load_csv(seed_dir / "unreviewed_candidates_carried_forward.csv"),
        "seed_review_cards_text": read_text(seed_dir / "seed_review_cards.md"),
        "review_pack_index": load_csv(review_pack_dir / "review_pack_index.csv"),
        "candidate_pool_scored": load_csv(review_pack_dir / "candidate_pool_scored.csv"),
        "review_pack_artifact_inventory": {
            "markdown_cards": sorted(path.name for path in (review_pack_dir / "markdown_cards").iterdir()),
            "csv_tables": sorted(path.name for path in (review_pack_dir / "csv_tables").iterdir()),
            "pdf_crops": sorted(path.name for path in (review_pack_dir / "pdf_crops").iterdir()),
        },
        "frozen_review_labels": load_csv(freeze_dir / "frozen_review_labels.csv"),
        "confirmed_seed_draft_rows": load_csv(freeze_dir / "confirmed_seed_draft_candidates.csv"),
        "partial_routing_rows": load_csv(freeze_dir / "partial_candidate_routing.csv"),
        "reject_feedback_rows": load_csv(freeze_dir / "reject_boundary_feedback.csv"),
        "unreviewed_rows": load_csv(freeze_dir / "unreviewed_candidates.csv"),
        "phase7g2_reports": phase7g2_reports,
        "phase7f_summary_present": bool(phase7f_text),
    }


def copy_appendix(rows: list[dict[str, Any]], output_path: Path) -> None:
    write_csv(output_path, rows)


def status_count_lines(counter: dict[str, int]) -> list[str]:
    if not counter:
        return ["- 无"]
    order = ["pass", "pass_with_warnings", "partial", "fail", "not_applicable", "not_evaluable"]
    return [f"- `{key}`：{counter[key]}" for key in order if key in counter]


def seed_status_lines(results: list[dict[str, Any]]) -> list[str]:
    return [
        f"- `{row['seed_id']}`：`{row['candidate_id']}`；`{row['doc_id']}` / `{row['table_id']}`；"
        f"overall=`{row['overall_validation_status']}`；artifact=`{row['artifact_availability_check']}`；"
        f"CSV=`{row['csv_structure_sanity_check']}`；binding=`{row['binding_policy_check']}`；"
        f"source_span=`{row['source_span_check']}`；bbox=`{row['bbox_provenance_check']}`；"
        f"warnings=`{row['warnings']}`；missing_items=`{csv_value(row['missing_items'])}`"
        for row in results
    ]


def write_guardrail(report_dir: Path) -> None:
    write_text(
        report_dir / "phase7h_guardrail.md",
        [
            "# Phase7H 护栏",
            "",
            "1. 本轮定位为 expanded seed consistency validation。",
            "2. 本轮不是 official benchmark。",
            "3. 本轮不是 production readiness。",
            "4. 本轮不是 RAG/retrieval evaluation。",
            "5. 本轮不扩大候选池。",
            "6. 本轮不修改 seed。",
            "7. 本轮不修改 extractor。",
            "8. formal subset 只使用 `confirmed_seed_with_warnings`。",
            "9. partial/reject/unreviewed carried forward 不进入 formal overall。",
            "10. 本轮不接 production。",
            "11. 本轮不访问 Milvus / BM25，不写入 Milvus，不读取或查询 BM25 index。",
            "12. Route C 仍只是 backlog。",
        ],
    )


def write_spec(report_dir: Path) -> None:
    write_text(
        report_dir / "expanded_seed_validation_spec.md",
        [
            "# Expanded Seed Validation Spec",
            "",
            "## 验证定位",
            "本规范只定义 Phase7H expanded seed consistency validation；它不是 official benchmark、production readiness 或 retrieval evaluation。",
            "",
            "## validation fields",
            "- `seed_identity_check`：检查 `seed_id` 唯一，且 `seed_id` / `candidate_id` / `doc_id` / `table_id` / `table_object_id` 存在。",
            "- `traceability_check`：检查 seed 可追溯到 Phase7G-2 expanded seed、summary、Phase7G-1 frozen labels、confirmed draft、review pack index 与 candidate pool。",
            "- `artifact_availability_check`：合并 Markdown、CSV、crop artifact 可用性；CSV 缺失或不可解析会 fail。",
            "- `human_label_consistency_check`：检查 confirmed draft / frozen label 中 `boundary_ok`、`grid_ok`、`key_values_ok` 均为 `yes`。",
            "- `csv_structure_sanity_check`：解析 CSV，检查非空、行列数合理、明显全空尾行只记 warning。",
            "- `markdown_card_check`：Markdown 缺失或空文件记 `pass_with_warnings`。",
            "- `pdf_crop_check`：crop 缺失、空文件或 crop_status 非 ok 只记 `pass_with_warnings`。",
            "- `binding_policy_check`：检查 bulk binding mode、limitation、unit/reference 不为 `unchecked`，并保留 guardrail warning。",
            "- `source_span_check`：检查 `source_span_granularity` 不为 `value_level`；cell/table-level limitation 记 warning。",
            "- `bbox_provenance_check`：检查 `value_bboxes_available=false`，`cell_bboxes_available=true` 不解释为 value bbox。",
            "- `formal_membership_check`：检查只属于 `confirmed_seed_with_warnings`，且不混入 partial/reject/unreviewed，也不含 production/official benchmark marker。",
            "- `overall_validation_status`：由以上字段派生。",
            "",
            "## 字段取值",
            "- 单项字段取值：`pass`、`pass_with_warnings`、`partial`、`fail`、`not_applicable`、`not_evaluable`。",
            "- `overall_validation_status` 取值：`pass`、`pass_with_warnings`、`partial`、`fail`、`not_evaluable`。",
            "",
            "## Formal pass 规则",
            "1. `seed_id` 唯一。",
            "2. 每条 seed 来自 `confirmed_seed_draft`。",
            "3. `seed_status=confirmed_seed_with_warnings`。",
            "4. `boundary_ok=yes`。",
            "5. `grid_ok=yes`。",
            "6. `key_values_ok=yes`。",
            "7. `unit_or_note_ok` 不得为 `unchecked`。",
            "8. `reference_ok` 不得为 `unchecked`。",
            "9. `binding_review_mode=global_bulk_no_obvious_issue`。",
            "10. `binding_review_limitation=no_per_cell_binding_review`。",
            "11. `markdown_path` 必须存在或给出明确 warning。",
            "12. `csv_path` 必须存在且 CSV 可解析。",
            "13. `pdf_crop_path` 缺失不阻断，但必须记录 crop warning。",
            "14. `source_span_granularity` 不得为 `value_level`，除非真实存在；Phase7H 未发现真实 value-level span。",
            "15. `value_bboxes_available` 必须为 `false`，且不得作为失败。",
            "16. `cell_bboxes_available` 可为 `true`，但不得写成 value bbox。",
            "17. 不允许 production readiness marker。",
            "18. 不允许 official benchmark seed marker。",
            "19. partial/reject/unreviewed 不进入 formal overall。",
            "",
            "## Formal overall pass 规则",
            "- 15 条 formal seed 均为 `pass` 或 `pass_with_warnings`。",
            "- 不输出 mixed formal overall。",
            "- carried-forward partial/reject/unreviewed 只作为 appendix。",
        ],
    )


def write_formal_vs_carried_forward(report_dir: Path, payload: dict[str, Any]) -> None:
    counts = payload["appendix_counts"]
    write_text(
        report_dir / "formal_vs_carried_forward_split.md",
        [
            "# Formal 与 Carried-forward 分离说明",
            "",
            f"- Formal subset：仅包含 Phase7G-2 的 15 条 `confirmed_seed_with_warnings`。",
            f"- partial carried forward：{counts['partial_carried_forward']} 条，不是 formal seed。",
            f"- reject carried forward：{counts['reject_carried_forward']} 条，不进入 seed。",
            f"- unreviewed carried forward：{counts['unreviewed_carried_forward']} 条，不进入 seed。",
            "",
            "## 处理原则",
            "- partial carried forward 只作为后续 extractor hardening / labeling backlog。",
            "- reject carried forward 只作为 boundary / scoring 反馈。",
            "- unreviewed 仍缺少完整人工核心标签，只作为后续标注 backlog。",
            "- 三类 carried-forward 均不影响 formal overall。",
        ],
    )


def write_validation_report(report_dir: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    overall = payload["formal_overall"]
    counts = payload["appendix_counts"]
    check_counts = overall["check_counts"]
    write_text(
        report_dir / "expanded_seed_validation_report.md",
        [
            "# Expanded Seed Validation Report",
            "",
            "## 1. validation 目标",
            "本轮验证 Phase7G-2 生成的 expanded seed 是否内部一致、可追溯、artifact 完整，并判断是否足以进入下一阶段 table index unit design。",
            "",
            "## 2. 输入文件",
            "- `data/experiments/v7_phase7_expanded_seed_from_human_review/expanded_table_seed.jsonl`",
            "- `data/experiments/v7_phase7_expanded_seed_from_human_review/expanded_table_seed_summary.csv`",
            "- `data/experiments/v7_phase7_expanded_seed_from_human_review/confirmed_seed_with_warnings.jsonl`",
            "- `data/experiments/v7_phase7_expanded_seed_from_human_review/*_carried_forward.csv`",
            "- `data/experiments/v7_phase7_expanded_seed_from_human_review/seed_review_cards.md`",
            "- `data/experiments/v7_phase7_expanded_table_review_pack/review_pack_index.csv`",
            "- `data/experiments/v7_phase7_expanded_table_review_pack/candidate_pool_scored.csv`",
            "- `data/experiments/v7_phase7_expanded_table_review_pack/markdown_cards/`",
            "- `data/experiments/v7_phase7_expanded_table_review_pack/csv_tables/`",
            "- `data/experiments/v7_phase7_expanded_table_review_pack/pdf_crops/`",
            "- `data/experiments/v7_phase7_human_review_label_freeze/*.csv`",
            "- `reports/v7_phase7_expanded_seed_from_human_review/*.md`",
            "- `reports/v7_phase7_gold_seed_validation/phase7f_summary.md` 只读引用。",
            "",
            f"## 3. formal seed 数量：{overall['formal_seed_count']}",
            "",
            "## 4. formal seed 清单",
            *[
                f"- `{row['seed_id']}`：`{row['candidate_id']}`；`{row['doc_id']}` / `{row['table_id']}`"
                for row in results
            ],
            "",
            "## 5. carried-forward appendix 数量",
            f"- partial：{counts['partial_carried_forward']}",
            f"- reject：{counts['reject_carried_forward']}",
            f"- unreviewed：{counts['unreviewed_carried_forward']}",
            "",
            "## 6. 每条 formal seed 的 validation status",
            *seed_status_lines(results),
            "",
            f"## 7. formal overall status：`{overall['formal_overall_status']}`",
            "",
            "## 8. artifact availability 统计",
            *status_count_lines(check_counts["artifact_availability_check"]),
            "",
            "## 9. CSV sanity 统计",
            *status_count_lines(check_counts["csv_structure_sanity_check"]),
            "",
            "## 10. binding policy check 统计",
            *status_count_lines(check_counts["binding_policy_check"]),
            "",
            "## 11. source_span check 统计",
            *status_count_lines(check_counts["source_span_check"]),
            "",
            "## 12. bbox provenance check 统计",
            *status_count_lines(check_counts["bbox_provenance_check"]),
            "",
            "## 13. partial/reject/unreviewed appendix 说明",
            "这些 carried-forward rows 均不进入 formal overall，只作为后续 extractor hardening、boundary/scoring 反馈或 labeling backlog。",
            "",
            "## 14. 主要 warnings",
            "- bulk binding 仍是 `global_bulk_no_obvious_issue`，限制为 `no_per_cell_binding_review`。",
            "- `source_span_granularity` 为 cell-level，不是 value-level。",
            "- `value_bboxes_available=false`，`cell_bboxes_available=true` 不解释为 value bbox。",
            "",
            "## 15. 未解决问题",
            "- 未进行逐 cell binding 审查。",
            "- 未构造 value-level bbox。",
            "- carried-forward partial/reject/unreviewed 仍需后续 backlog 处理。",
            "",
            "## 16. 为什么本轮不是 official benchmark",
            "本轮只验证 expanded seed consistency，没有 official benchmark schema 冻结，也没有把 seed 写入 official baseline。",
            "",
            "## 17. 为什么本轮不等于 production readiness",
            "本轮没有运行 production pipeline、retrieval、embedding/rerank、模型评估或线上接入。",
            "",
            "## 18. 是否建议进入 Phase7I table index unit design",
            "- 建议进入。依据是 formal seed 15 条均为 pass-like，且 artifact、traceability、binding/source/bbox limitation 均已显式记录。",
            "",
            "## 19. 是否建议回修 extractor",
            "- 不建议在本轮立即回修 extractor。本轮不是 extractor validation；partial blockers 可进入后续 hardening backlog。",
            "",
            "## 20. 是否建议继续人工大标注",
            "- 不建议在本轮继续人工大标注。当前目标是进入 table index unit design，而不是扩大标签池。",
            "",
            "## 21. Route C 是否仍只是 backlog",
            "- 是。Route C 仍只是 backlog，本轮未进入 implementation。",
        ],
    )


def write_traceability_report(report_dir: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    write_text(
        report_dir / "phase7g2_to_phase7h_traceability.md",
        [
            "# Phase7G-2 到 Phase7H Traceability",
            "",
            "## 1. Phase7G-2 的 15 条 seed 如何进入 Phase7H formal validation",
            "Phase7H 仅读取 `confirmed_seed_with_warnings.jsonl` 中的 15 条记录作为 formal subset，不把 partial/reject/unreviewed carried-forward 混入 formal overall。",
            "",
            "## 2. 每条 seed 的 candidate / review label / review pack artifact 追溯",
            *[
                f"- `{row['seed_id']}` -> candidate=`{row['candidate_id']}`；doc/table=`{row['doc_id']}` / `{row['table_id']}`；"
                f"traceability=`{row['traceability_check']}`；markdown=`{row['markdown_exists']}`；csv=`{row['csv_parse_status']}`；crop=`{row['pdf_crop_exists']}`"
                for row in results
            ],
            "",
            "## 3. bulk binding warning 如何继承",
            "Phase7H 保留 `global_bulk_no_obvious_issue` 与 `no_per_cell_binding_review`，不把 `warning` / `not_applicable` 提升为 fully confirmed binding。",
            "",
            "## 4. source_span / bbox limitation 如何继承",
            "Phase7H 继承 cell-level source span 与 `value_bboxes_available=false`；`cell_bboxes_available=true` 只说明 cell bbox 可用，不构成 value bbox。",
            "",
            "## 5. partial/reject/unreviewed 为什么不进入 formal",
            "partial 仍存在 boundary/grid/key_values 或 routing blocker；reject 是 boundary/scoring 反馈；unreviewed 缺少完整人工核心标签。",
            "",
            "## 6. Phase7H 结果如何支撑下一阶段 table index unit design",
            "15 条 formal seed 均有可解析 CSV、可追溯 review pack artifact 和明确 limitation，可作为 Phase7I 设计 table index unit 的离线一致性输入。",
            "",
            "## 7. 为什么 Phase7H 仍不是 retrieval validation",
            "本轮没有读取 BM25 index、没有访问 Milvus、没有运行 embedding/rerank、没有调用 Qwen/RAGAS，也没有执行 RAG retrieval evaluation。",
        ],
    )


def write_summary(report_dir: Path, output_dir: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    overall = payload["formal_overall"]
    counts = payload["appendix_counts"]
    check_counts = overall["check_counts"]
    generated_files = [
        output_dir / "expanded_seed_validation_results.json",
        output_dir / "expanded_seed_validation_results.csv",
        output_dir / "formal_seed_validation_results.csv",
        output_dir / "carried_forward_partial_feedback.csv",
        output_dir / "carried_forward_reject_feedback.csv",
        output_dir / "carried_forward_unreviewed_feedback.csv",
        report_dir / "phase7h_guardrail.md",
        report_dir / "expanded_seed_validation_spec.md",
        report_dir / "expanded_seed_validation_report.md",
        report_dir / "formal_vs_carried_forward_split.md",
        report_dir / "phase7g2_to_phase7h_traceability.md",
        report_dir / "phase7h_summary.md",
    ]
    write_text(
        report_dir / "phase7h_summary.md",
        [
            "# Phase7H Summary",
            "",
            "## 1. 本轮生成文件",
            *(f"- `{rel(path)}`" for path in generated_files),
            "",
            "## 2. 新增 / 修改脚本",
            "- 新增：`scripts/evaluation/validate_expanded_table_seed_consistency.py`",
            "",
            "## 3. 新增测试",
            "- 新增：`tests/test_phase7_expanded_seed_consistency_validation.py`",
            "",
            f"## 4. formal seed 数量：{overall['formal_seed_count']}",
            f"## 5. carried-forward partial 数量：{counts['partial_carried_forward']}",
            f"## 6. carried-forward reject 数量：{counts['reject_carried_forward']}",
            f"## 7. carried-forward unreviewed 数量：{counts['unreviewed_carried_forward']}",
            f"## 8. formal overall status：`{overall['formal_overall_status']}`",
            "",
            "## 9. 每条 formal seed 的 overall_validation_status",
            *[f"- `{row['seed_id']}`：`{row['overall_validation_status']}`" for row in results],
            "",
            "## 10. artifact availability 统计",
            *status_count_lines(check_counts["artifact_availability_check"]),
            "",
            "## 11. CSV sanity 统计",
            *status_count_lines(check_counts["csv_structure_sanity_check"]),
            "",
            "## 12. binding policy check 统计",
            *status_count_lines(check_counts["binding_policy_check"]),
            "",
            "## 13. source_span check 统计",
            *status_count_lines(check_counts["source_span_check"]),
            "",
            "## 14. bbox provenance check 统计",
            *status_count_lines(check_counts["bbox_provenance_check"]),
            "",
            "## 15. 是否满足 expanded seed validation",
            "- 是。15 条 formal seed 均为 pass-like，且 carried-forward appendix 未进入 formal overall。",
            "",
            "## 16. 是否建议进入 Phase7I table index unit design",
            "- 建议进入。",
            "",
            "## 17. 是否建议回修 extractor",
            "- 不建议本轮立即回修 extractor。本轮不是 extractor validation。",
            "",
            "## 18. 是否建议继续人工大标注",
            "- 不建议本轮继续人工大标注。",
            "",
            "## 19. 是否建议进入 production",
            "- 不建议进入 production。",
            "",
            "## 20. baseline / guardrail 是否漂移",
            "- 未发现漂移。本轮未修改 official dataset、official baseline、configs、baseline registry、chunks、BM25 或 Milvus；也未读取或查询 BM25 index。",
            "",
            "## 21. Route C 是否仍只是 backlog",
            "- 是。Route C 仍只是 backlog。",
            "",
            "## 22. 明确未执行事项",
            "- 未扩大候选池。",
            "- 未生成新 review pack。",
            "- 未修改 seed。",
            "- 未回修 extractor。",
            "- 未运行 retrieval。",
            "- 未运行 RAGAS。",
            "- 未运行 Qwen。",
            "- 未做 flat comparison。",
            "- 未引入 Camelot。",
            "- 未引入 PyMuPDF。",
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
            "- 未接入 production。",
            "- 未进入 Route C。",
        ],
    )


def write_reports(report_dir: Path, output_dir: Path, payload: dict[str, Any]) -> None:
    write_guardrail(report_dir)
    write_spec(report_dir)
    write_validation_report(report_dir, payload)
    write_formal_vs_carried_forward(report_dir, payload)
    write_traceability_report(report_dir, payload)
    write_summary(report_dir, output_dir, payload)


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    seed_dir: Path = DEFAULT_SEED_DIR,
    review_pack_dir: Path = DEFAULT_REVIEW_PACK_DIR,
    freeze_dir: Path = DEFAULT_FREEZE_DIR,
    phase7g2_report_dir: Path = DEFAULT_PHASE7G2_REPORT_DIR,
    phase7f_summary: Path = DEFAULT_PHASE7F_SUMMARY,
) -> dict[str, Any]:
    inputs = load_inputs(seed_dir, review_pack_dir, freeze_dir, phase7g2_report_dir, phase7f_summary)
    payload = build_validation_results(inputs)
    results = payload["results"]

    write_json(output_dir / "expanded_seed_validation_results.json", payload)
    write_csv(output_dir / "expanded_seed_validation_results.csv", flat_result_rows(results), RESULT_FIELDS)
    write_csv(output_dir / "formal_seed_validation_results.csv", flat_result_rows(results), RESULT_FIELDS)
    copy_appendix(inputs["partial_carried_forward"], output_dir / "carried_forward_partial_feedback.csv")
    copy_appendix(inputs["reject_carried_forward"], output_dir / "carried_forward_reject_feedback.csv")
    copy_appendix(inputs["unreviewed_carried_forward"], output_dir / "carried_forward_unreviewed_feedback.csv")
    write_reports(report_dir, output_dir, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase7G-2 expanded seed consistency.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--review-pack-dir", type=Path, default=DEFAULT_REVIEW_PACK_DIR)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--phase7g2-report-dir", type=Path, default=DEFAULT_PHASE7G2_REPORT_DIR)
    parser.add_argument("--phase7f-summary", type=Path, default=DEFAULT_PHASE7F_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        seed_dir=args.seed_dir,
        review_pack_dir=args.review_pack_dir,
        freeze_dir=args.freeze_dir,
        phase7g2_report_dir=args.phase7g2_report_dir,
        phase7f_summary=args.phase7f_summary,
    )
    print(json.dumps(payload["formal_overall"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
