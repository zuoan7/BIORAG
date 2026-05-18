#!/usr/bin/env python3
"""Phase 5F-4B official clean baseline asset promotion audit.

This script is intentionally read-only for retrieval, cleaning, dataset, and
index assets. It only writes audit reports under
reports/phase5f4_asset_promotion_audit/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "phase5f4_asset_promotion_audit"

EXPECTED_DATASET_SHA256 = (
    "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
)
EXPECTED_DATASET_TOTAL = 90
EXPECTED_QUERY_TYPE_DISTRIBUTION = {
    "table_content": 31,
    "caption_level_table": 9,
    "figure_caption": 20,
    "normal_control": 30,
}
EXPECTED_BASELINE_CHUNK_COUNT = 15802
EXPECTED_BASELINE_DOC_COUNT = 705
EXPECTED_MISSING_DOC_IDS = ["doc_0314", "doc_0427", "doc_0526"]
EXPECTED_VECTOR_DIM = 1024

DATASET_PATH = (
    ROOT / "reports" / "phase5f_eval_semantic_enhancement_v2"
    / "strict_main_eval_set_v2.jsonl"
)

PRIORITY_FILES = [
    "reports/phase5f4_index_asset_normalization/summary.md",
    "reports/phase5f4_index_asset_normalization/official_baseline_decision.md",
    "reports/phase5f4_index_asset_normalization/official_baseline_decision.json",
    "reports/phase5f4_index_asset_normalization/current_default_asset_audit.md",
    "reports/phase5f4_index_asset_normalization/current_default_asset_audit.json",
    "reports/phase5f4_index_asset_normalization/phase5c5_baseline_full_asset_audit.md",
    "reports/phase5f4_index_asset_normalization/phase5c5_baseline_full_asset_audit.json",
    "reports/phase5f4_index_asset_normalization/phase5c5_enhanced_full_asset_audit.md",
    "reports/phase5f4_index_asset_normalization/phase5c5_enhanced_full_asset_audit.json",
    "reports/phase5f4_index_asset_normalization/target_coverage_matrix.csv",
    "reports/phase5f4_index_asset_normalization/target_coverage_summary.md",
    "reports/phase5f4_index_asset_normalization/next_phase_plan.md",
    "reports/phase5f4_clean_main_baseline/dataset_freeze.md",
    "reports/phase5f4_clean_main_baseline/dataset_manifest.json",
    "reports/phase5f4_clean_main_baseline/dataset_hash.txt",
    "reports/phase5f4_clean_main_baseline/retrieval_asset_inventory.md",
    "reports/phase5f4_clean_main_baseline/retrieval_asset_inventory.json",
    "reports/phase5f4_clean_main_baseline/main_results.json",
    "reports/phase5f4_clean_main_baseline/variant_comparison.md",
    "reports/phase5f4_clean_main_baseline/variant_comparison.json",
    "reports/phase5f4_clean_main_baseline/per_sample_results.jsonl",
    "reports/phase5f4_clean_main_baseline/summary.md",
    "reports/phase5f4_failure_review/summary.md",
    "reports/phase5f4_failure_review/repair_trigger_decision.md",
    "reports/phase5f4_failure_review/variant_difference_review.md",
    "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl",
    "reports/phase5c5_full_retrieval_ab/index_build_summary.md",
    "reports/phase5c5_full_retrieval_ab/index_build_stats.json",
    "reports/phase5c5_full_retrieval_ab/chunk_ab_summary.md",
    "reports/phase5c5_full_retrieval_ab/chunk_ab_stats.json",
    "reports/phase5c5_full_retrieval_ab/summary.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def is_tmp_path(path: str | Path | None) -> bool:
    if not path:
        return False
    try:
        resolved = Path(str(path)).resolve()
    except OSError:
        resolved = Path(str(path))
    return str(resolved).startswith("/tmp/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json_load(path: Path) -> tuple[Any | None, str]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh), ""
    except Exception as exc:  # noqa: BLE001 - audit must record failures.
        return None, f"{type(exc).__name__}: {exc}"


def safe_text_read(path: Path, limit: int | None = None) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if limit is not None:
            return text[:limit], ""
        return text, ""
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"


def write_json(name: str, payload: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=json_default)
        fh.write("\n")


def write_md(name: str, lines: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "name"):
        return value.name
    return str(value)


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def parse_block_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[;,]\s*|\s+\|\s+", value)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.extend(parse_block_ids(item))
            elif item is not None:
                out.append(str(item))
        return [p for p in out if p]
    return [str(value)]


def compact_list(values: list[Any], max_items: int = 20) -> list[Any]:
    return values[:max_items]


def file_record(path: Path, loaded: bool, error: str = "") -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "path": rel(path),
        "exists": exists,
        "loaded": bool(loaded and exists and not error),
        "error": error,
    }
    if exists:
        stat = path.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).replace(microsecond=0).isoformat(),
            }
        )
    return record


def infer_source_info(path: str, data: Any, text: str) -> list[str]:
    info: list[str] = []
    blob = text
    if data is not None:
        blob = json.dumps(data, ensure_ascii=False, default=json_default)
    lower_path = path.lower()
    lower_blob = blob.lower()

    if "phase5c5_baseline_full_asset_audit" in lower_path:
        info.extend(
            [
                "phase5c5_baseline_full chunks path/count",
                "phase5c5_baseline_full BM25 cache path/count",
                "phase5c5_baseline_full Milvus URI/collection/row count",
                "non-enhanced baseline suitability evidence",
            ]
        )
    if "retrieval_asset_inventory" in lower_path:
        info.extend(["retrieval asset inventory for current/default/full variants"])
    if "official_baseline_decision" in lower_path:
        info.extend(["official baseline candidate decision and risk notes"])
    if "index_build_stats" in lower_path or "index_build_summary" in lower_path:
        info.extend(["Phase 5C-5 build counts and build timing"])
    if "variant_comparison" in lower_path:
        info.extend(["baseline vs enhanced retrieval metrics"])
    if "main_results" in lower_path or "per_sample_results" in lower_path:
        info.extend(["strict_main_eval_set_v2 retrieval results"])
    if "dataset" in lower_path or "strict_main_eval_set_v2" in lower_path:
        info.extend(["frozen dataset hash/distribution/schema evidence"])
    if "target_coverage" in lower_path:
        info.extend(["target document and stable block coverage evidence"])
    if "failure_review" in lower_path:
        info.extend(["failure review context; not a promotion input"])
    if "/tmp/" in lower_blob:
        info.append("contains /tmp asset paths")
    if "bm25" in lower_blob:
        info.append("mentions BM25 asset")
    if "milvus" in lower_blob or "collection" in lower_blob:
        info.append("mentions Milvus collection")
    if "chunks" in lower_blob:
        info.append("mentions chunks asset")
    return sorted(set(info)) or ["read for audit context"]


def load_priority_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    sources: dict[str, Any] = {}
    loaded_data: dict[str, Any] = {}

    for rel_path in PRIORITY_FILES:
        path = ROOT / rel_path
        if not path.exists():
            sources[rel_path] = file_record(path, loaded=False, error="missing")
            sources[rel_path]["provided_info"] = []
            continue

        loaded = False
        error = ""
        data: Any | None = None
        text = ""
        suffix = path.suffix.lower()
        if suffix == ".json":
            data, error = safe_json_load(path)
            loaded = data is not None
            if loaded:
                loaded_data[rel_path] = data
        elif suffix in {".md", ".txt", ".csv", ".jsonl"}:
            # Full read is fine for the listed files; none are index-sized.
            text, error = safe_text_read(path)
            loaded = not error
            if loaded:
                loaded_data[rel_path] = text
        else:
            text, error = safe_text_read(path, limit=20000)
            loaded = not error
            if loaded:
                loaded_data[rel_path] = text

        sources[rel_path] = file_record(path, loaded=loaded, error=error)
        sources[rel_path]["provided_info"] = infer_source_info(rel_path, data, text)

    return sources, loaded_data


def choose_baseline_paths(loaded: dict[str, Any]) -> dict[str, Any]:
    baseline_audit = loaded.get(
        "reports/phase5f4_index_asset_normalization/"
        "phase5c5_baseline_full_asset_audit.json",
        {},
    )
    inventory = loaded.get(
        "reports/phase5f4_clean_main_baseline/retrieval_asset_inventory.json",
        {},
    )
    index_stats = loaded.get(
        "reports/phase5c5_full_retrieval_ab/index_build_stats.json", {}
    )

    asset_from_inventory = (
        inventory.get("assets", {}).get("phase5c5_baseline_full", {})
        if isinstance(inventory, dict)
        else {}
    )
    baseline_stats = (
        index_stats.get("baseline", {}) if isinstance(index_stats, dict) else {}
    )

    chunks_path = (
        baseline_audit.get("chunks_path")
        or baseline_audit.get("chunks", {}).get("path")
        or asset_from_inventory.get("chunks_path")
        or baseline_stats.get("chunks_jsonl")
    )
    bm25_path = (
        baseline_audit.get("bm25_cache_path")
        or baseline_audit.get("bm25_cache", {}).get("path")
        or asset_from_inventory.get("bm25_cache_path")
        or baseline_stats.get("bm25_path")
    )
    milvus_uri = (
        baseline_audit.get("milvus_uri")
        or baseline_audit.get("milvus", {}).get("uri")
        or asset_from_inventory.get("milvus_uri")
        or baseline_stats.get("milvus_uri")
    )
    collection = (
        baseline_audit.get("milvus_collection")
        or baseline_audit.get("milvus", {}).get("collection")
        or asset_from_inventory.get("collection")
        or baseline_stats.get("collection")
    )

    return {
        "chunks_path": chunks_path,
        "bm25_path": bm25_path,
        "milvus_uri": milvus_uri,
        "milvus_collection": collection,
        "baseline_audit": baseline_audit,
        "inventory_asset": asset_from_inventory,
        "index_build_stats": baseline_stats,
    }


def scan_dataset() -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": rel(DATASET_PATH),
        "exists": DATASET_PATH.exists(),
        "expected_sha256": EXPECTED_DATASET_SHA256,
    }
    if not DATASET_PATH.exists():
        result.update(
            {
                "sha256": None,
                "sha256_matches_expected": False,
                "blocker": True,
                "error": "dataset missing",
                "samples": [],
            }
        )
        return result

    sha = sha256_file(DATASET_PATH)
    samples: list[dict[str, Any]] = []
    query_types: Counter[str] = Counter()
    include_values: Counter[str] = Counter()
    hard_rule_values: Counter[str] = Counter()
    nonempty_stable_ids = 0
    diagnostic_or_lexical: list[str] = []
    parse_errors: list[str] = []

    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"line {line_no}: {exc}")
                continue
            samples.append(sample)
            query_type = str(sample.get("query_type") or "")
            query_types[query_type] += 1
            include_values[str(sample.get("include_in_main_denominator"))] += 1
            hard_rule_values[str(sample.get("hard_rule_passed"))] += 1
            if parse_block_ids(sample.get("stable_target_block_ids")):
                nonempty_stable_ids += 1
            sample_id = str(sample.get("sample_id") or "")
            ability_scope = str(sample.get("ability_scope") or "").lower()
            if (
                query_type in {"diagnostic", "lexical_stress"}
                or "diagnostic" in sample_id.lower()
                or "lexical" in sample_id.lower()
                or "diagnostic" in ability_scope
                or "lexical" in ability_scope
            ):
                diagnostic_or_lexical.append(sample_id)

    include_all_true = all(
        normalize_bool(s.get("include_in_main_denominator")) is True for s in samples
    )
    hard_rule_all_true = all(
        normalize_bool(s.get("hard_rule_passed")) is True for s in samples
    )
    distribution = dict(sorted(query_types.items()))
    distribution_matches = distribution == dict(sorted(EXPECTED_QUERY_TYPE_DISTRIBUTION.items()))
    sha_matches = sha == EXPECTED_DATASET_SHA256
    total_matches = len(samples) == EXPECTED_DATASET_TOTAL
    blocker = not sha_matches

    result.update(
        {
            "sha256": sha,
            "sha256_matches_expected": sha_matches,
            "total_samples": len(samples),
            "total_matches_expected": total_matches,
            "query_type_distribution": distribution,
            "query_type_distribution_matches_expected": distribution_matches,
            "stable_target_block_ids_nonempty_count": nonempty_stable_ids,
            "stable_target_block_ids_nonempty_rate": (
                nonempty_stable_ids / len(samples) if samples else 0.0
            ),
            "include_in_main_denominator_counts": dict(include_values),
            "include_in_main_denominator_all_true": include_all_true,
            "hard_rule_passed_counts": dict(hard_rule_values),
            "hard_rule_passed_all_true": hard_rule_all_true,
            "diagnostic_or_lexical_stress_mixed_count": len(diagnostic_or_lexical),
            "diagnostic_or_lexical_stress_sample_ids": diagnostic_or_lexical,
            "parse_errors": parse_errors,
            "blocker": blocker,
            "samples": samples,
        }
    )
    return result


def gather_chunk_block_ids(chunk: dict[str, Any]) -> set[str]:
    block_ids = set(parse_block_ids(chunk.get("source_block_ids")))
    block_ids.update(parse_block_ids(chunk.get("block_ids")))
    sbm = chunk.get("source_block_metadata") or []
    if isinstance(sbm, list):
        for item in sbm:
            if not isinstance(item, dict):
                continue
            for key in ("block_id", "source_block_id", "id"):
                block_ids.update(parse_block_ids(item.get(key)))
    return {b for b in block_ids if b}


def has_table_enhancement_metadata(chunk: dict[str, Any]) -> bool:
    top_level_keys = {
        "table_related",
        "table_related_type",
        "table_enhancement_enabled",
        "phase5c1_pilot",
        "associated_table_caption_block_id",
        "associated_table_caption_text",
        "associated_table_id",
        "table_object",
    }
    for key in top_level_keys:
        if key in chunk and chunk.get(key):
            return True

    sbm = chunk.get("source_block_metadata") or []
    if isinstance(sbm, list):
        for item in sbm:
            if not isinstance(item, dict):
                continue
            if item.get("table_related") is True:
                return True
            if item.get("table_enhancement_enabled") is True:
                return True
            if item.get("phase5c1_pilot") is True:
                return True
            for key in (
                "table_related_type",
                "associated_table_caption_block_id",
                "associated_table_caption_text",
                "associated_table_id",
            ):
                if item.get(key):
                    return True
    return False


def has_caption_cleanup_metadata(chunk: dict[str, Any]) -> bool:
    keys = set(chunk.keys())
    sbm = chunk.get("source_block_metadata") or []
    if isinstance(sbm, list):
        for item in sbm:
            if isinstance(item, dict):
                keys.update(item.keys())
                for value in item.values():
                    if isinstance(value, str) and (
                        "caption_cleanup" in value.lower() or "phase5d" in value.lower()
                    ):
                        return True
    for key in keys:
        lowered = str(key).lower()
        if "caption_cleanup" in lowered or "phase5d" in lowered:
            return True
    return False


def metadata_json_length_estimate(chunk: dict[str, Any]) -> int:
    sbm = chunk.get("source_block_metadata") or []
    if isinstance(sbm, list):
        sbm_summary = [
            {
                k: v
                for k, v in (item if isinstance(item, dict) else {}).items()
                if k in ("block_id", "type", "page", "bbox", "text_length")
            }
            for item in sbm[:20]
        ]
    else:
        sbm_summary = []
    payload = {
        "section_path": chunk.get("section_path") or [],
        "block_types": chunk.get("block_types") or [],
        "source_block_ids": chunk.get("source_block_ids") or [],
        "block_ids": chunk.get("block_ids") or [],
        "evidence_types": chunk.get("evidence_types") or [],
        "page_numbers": chunk.get("page_numbers") or [],
        "layout_columns": chunk.get("layout_columns") or 1,
        "reading_order_span": chunk.get("reading_order_span") or [],
        "bbox_span": chunk.get("bbox_span") or [],
        "excluded_block_counts": chunk.get("excluded_block_counts") or {},
        "contains_references": chunk.get("contains_references") or False,
        "contains_metadata": chunk.get("contains_metadata") or False,
        "contains_noise": chunk.get("contains_noise") or False,
        "parser_stage": chunk.get("parser_stage") or "",
        "source_block_metadata": sbm_summary,
        "table_id": chunk.get("table_id") or "",
        "figure_id": chunk.get("figure_id") or "",
        "image_id": chunk.get("image_id") or "",
        "object_id": chunk.get("object_id") or "",
        "object_type": chunk.get("object_type") or "",
        "table_caption": chunk.get("table_caption") or "",
        "table_markdown": chunk.get("table_markdown") or "",
        "table_csv_path": chunk.get("table_csv_path") or "",
        "table_json_path": chunk.get("table_json_path") or "",
        "figure_caption": chunk.get("figure_caption") or "",
        "image_path": chunk.get("image_path") or "",
        "image_hash": chunk.get("image_hash") or "",
        "ocr_text": chunk.get("ocr_text") or "",
        "visual_summary": chunk.get("visual_summary") or "",
        "asset_uri": chunk.get("asset_uri") or "",
    }
    return len(json.dumps(payload, ensure_ascii=False))


def scan_chunks(chunks_path: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": chunks_path,
        "exists": False,
        "readable": False,
        "is_tmp_path": is_tmp_path(chunks_path),
    }
    if not chunks_path:
        result["error"] = "chunks path not found"
        return result
    path = Path(chunks_path)
    result["exists"] = path.exists()
    if not path.exists():
        result["error"] = "chunks file missing"
        return result

    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)

    chunk_ids: set[str] = set()
    duplicate_chunk_ids: list[str] = []
    doc_ids: set[str] = set()
    empty_doc_id_count = 0
    missing_retrieval_text_field_count = 0
    missing_text_field_count = 0
    empty_retrieval_text_count = 0
    content_shape_counts: Counter[str] = Counter()
    parser_stage_counts: Counter[str] = Counter()
    table_enhancement_metadata_count = 0
    caption_cleanup_metadata_count = 0
    retrieval_text_varchar_risk_count = 0
    text_varchar_risk_count = 0
    metadata_json_varchar_risk_count = 0
    token_count_over_4096_count = 0
    max_retrieval_text_chars = 0
    max_text_chars = 0
    max_metadata_json_chars = 0
    max_token_count = 0
    first_field_set: list[str] = []
    field_sets: Counter[str] = Counter()
    parse_errors: list[str] = []
    doc_to_chunks: dict[str, list[str]] = defaultdict(list)
    chunk_retrieval_nonempty: dict[str, bool] = {}
    doc_block_to_chunks: dict[tuple[str, str], list[str]] = defaultdict(list)
    block_to_chunks: dict[str, list[str]] = defaultdict(list)

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_no}: {exc}")
                    continue

                if not first_field_set:
                    first_field_set = sorted(chunk.keys())
                field_sets["|".join(sorted(chunk.keys()))] += 1

                chunk_id = str(chunk.get("chunk_id") or "")
                doc_id = str(chunk.get("doc_id") or "")
                if chunk_id in chunk_ids:
                    duplicate_chunk_ids.append(chunk_id)
                if chunk_id:
                    chunk_ids.add(chunk_id)
                if doc_id:
                    doc_ids.add(doc_id)
                    if chunk_id:
                        doc_to_chunks[doc_id].append(chunk_id)
                else:
                    empty_doc_id_count += 1

                if "retrieval_text" not in chunk:
                    missing_retrieval_text_field_count += 1
                if "text" not in chunk:
                    missing_text_field_count += 1
                retrieval_text = str(chunk.get("retrieval_text") or "")
                text = str(chunk.get("text") or "")
                nonempty_retrieval = bool(retrieval_text.strip())
                if not nonempty_retrieval:
                    empty_retrieval_text_count += 1
                if chunk_id:
                    chunk_retrieval_nonempty[chunk_id] = nonempty_retrieval
                max_retrieval_text_chars = max(
                    max_retrieval_text_chars, len(retrieval_text)
                )
                max_text_chars = max(max_text_chars, len(text))
                if len(retrieval_text) > 16384:
                    retrieval_text_varchar_risk_count += 1
                if len(text) > 16384:
                    text_varchar_risk_count += 1
                metadata_len = metadata_json_length_estimate(chunk)
                max_metadata_json_chars = max(max_metadata_json_chars, metadata_len)
                if metadata_len > 8192:
                    metadata_json_varchar_risk_count += 1

                token_count = chunk.get("token_count")
                if isinstance(token_count, (int, float)):
                    max_token_count = max(max_token_count, int(token_count))
                    if token_count > 4096:
                        token_count_over_4096_count += 1

                parser_stage = str(chunk.get("parser_stage") or "")
                if parser_stage:
                    parser_stage_counts[parser_stage] += 1
                if chunk.get("contains_table_text"):
                    content_shape_counts["table_text"] += 1
                elif chunk.get("contains_table_caption"):
                    content_shape_counts["table_caption"] += 1
                elif chunk.get("contains_figure_caption"):
                    content_shape_counts["figure_caption"] += 1
                elif chunk.get("contains_image"):
                    content_shape_counts["image_related"] += 1
                elif chunk.get("contains_references"):
                    content_shape_counts["references"] += 1
                else:
                    content_shape_counts["body_or_other"] += 1

                if has_table_enhancement_metadata(chunk):
                    table_enhancement_metadata_count += 1
                if has_caption_cleanup_metadata(chunk):
                    caption_cleanup_metadata_count += 1

                if chunk_id:
                    for block_id in gather_chunk_block_ids(chunk):
                        block_to_chunks[block_id].append(chunk_id)
                        if doc_id:
                            doc_block_to_chunks[(doc_id, block_id)].append(chunk_id)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    expected_docs = {f"doc_{i:04d}" for i in range(1, 709)}
    missing_expected_doc_ids = sorted(expected_docs - doc_ids)
    extra_doc_ids = sorted(doc_ids - expected_docs)
    result.update(
        {
            "readable": True,
            "chunk_count": len(chunk_ids),
            "line_count": len(chunk_ids) + len(duplicate_chunk_ids),
            "chunk_id_set": chunk_ids,
            "duplicate_chunk_id_count": len(duplicate_chunk_ids),
            "duplicate_chunk_ids_sample": compact_list(sorted(duplicate_chunk_ids)),
            "doc_id_count": len(doc_ids),
            "doc_ids": doc_ids,
            "empty_doc_id_count": empty_doc_id_count,
            "missing_expected_doc_ids": missing_expected_doc_ids,
            "extra_doc_ids": extra_doc_ids,
            "missing_retrieval_text_field_count": missing_retrieval_text_field_count,
            "missing_text_field_count": missing_text_field_count,
            "empty_retrieval_text_count": empty_retrieval_text_count,
            "max_retrieval_text_chars": max_retrieval_text_chars,
            "max_text_chars": max_text_chars,
            "max_metadata_json_chars_estimate": max_metadata_json_chars,
            "retrieval_text_varchar_risk_count": retrieval_text_varchar_risk_count,
            "text_varchar_risk_count": text_varchar_risk_count,
            "metadata_json_varchar_risk_count_estimate": metadata_json_varchar_risk_count,
            "token_count_over_4096_count": token_count_over_4096_count,
            "max_token_count": max_token_count,
            "content_shape_counts": dict(content_shape_counts),
            "parser_stage_counts": dict(parser_stage_counts),
            "table_enhancement_metadata_chunk_count": table_enhancement_metadata_count,
            "table_related_association_chunk_count": table_enhancement_metadata_count,
            "caption_cleanup_metadata_chunk_count": caption_cleanup_metadata_count,
            "field_set_count": len(field_sets),
            "first_field_set": first_field_set,
            "field_sets": [
                {"field_set": key, "count": count}
                for key, count in field_sets.most_common(10)
            ],
            "parse_errors": parse_errors,
            "is_baseline_full_candidate": len(chunk_ids) == EXPECTED_BASELINE_CHUNK_COUNT,
            "is_non_enhanced_baseline": table_enhancement_metadata_count == 0
            and caption_cleanup_metadata_count == 0,
            "doc_to_chunks": doc_to_chunks,
            "doc_block_to_chunks": doc_block_to_chunks,
            "block_to_chunks": block_to_chunks,
            "chunk_retrieval_nonempty": chunk_retrieval_nonempty,
        }
    )
    return result


def scan_bm25(bm25_path: str | None, chunk_ids: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": bm25_path,
        "exists": False,
        "readable": False,
        "is_tmp_path": is_tmp_path(bm25_path),
    }
    if not bm25_path:
        result["error"] = "BM25 path not found"
        return result
    path = Path(bm25_path)
    result["exists"] = path.exists()
    if not path.exists():
        result["error"] = "BM25 cache missing"
        return result

    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    data, error = safe_json_load(path)
    if error or data is None:
        result["error"] = error
        return result

    records: list[Any] = []
    if isinstance(data, dict):
        raw_records = data.get("records")
        if isinstance(raw_records, list):
            records = raw_records
        result["top_level_keys"] = sorted(str(k) for k in data.keys())
        result["avgdl"] = data.get("avgdl")
    elif isinstance(data, list):
        records = data
        result["top_level_keys"] = []
    else:
        result["error"] = f"unsupported BM25 JSON type: {type(data).__name__}"
        return result

    bm25_ids: set[str] = set()
    duplicate_ids: list[str] = []
    missing_chunk_id_records = 0
    for record in records:
        cid = ""
        if isinstance(record, dict):
            cid = str(record.get("chunk_id") or "")
        if not cid:
            missing_chunk_id_records += 1
            continue
        if cid in bm25_ids:
            duplicate_ids.append(cid)
        bm25_ids.add(cid)

    records_missing_from_chunks = sorted(bm25_ids - chunk_ids)
    chunks_missing_from_records = sorted(chunk_ids - bm25_ids)
    result.update(
        {
            "readable": True,
            "record_count": len(records),
            "chunk_id_set": bm25_ids,
            "missing_chunk_id_records": missing_chunk_id_records,
            "duplicate_chunk_id_count": len(duplicate_ids),
            "duplicate_chunk_ids_sample": compact_list(sorted(duplicate_ids)),
            "chunk_id_set_matches_chunks": (
                bool(chunk_ids) and bm25_ids == chunk_ids
            ),
            "records_missing_from_chunks_count": len(records_missing_from_chunks),
            "records_missing_from_chunks_sample": compact_list(records_missing_from_chunks),
            "chunks_missing_from_records_count": len(chunks_missing_from_records),
            "chunks_missing_from_records_sample": compact_list(chunks_missing_from_records),
        }
    )
    return result


def milvus_field_type(field: dict[str, Any]) -> str:
    value = field.get("type")
    if hasattr(value, "name"):
        return value.name
    return str(value)


def scan_milvus(
    milvus_uri: str | None, collection_name: str | None, chunk_ids: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "uri": milvus_uri,
        "collection": collection_name,
        "db_exists": False,
        "collection_available": False,
        "queryable": False,
        "is_tmp_path": is_tmp_path(milvus_uri),
        "depends_on_tmp_file_path": is_tmp_path(milvus_uri),
    }
    if not milvus_uri or not collection_name:
        result["error"] = "Milvus URI or collection name not found"
        return result

    uri_path = Path(milvus_uri)
    result["db_exists"] = uri_path.exists()
    if uri_path.exists():
        result["size_bytes"] = uri_path.stat().st_size

    try:
        from pymilvus import MilvusClient
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"pymilvus import failed: {type(exc).__name__}: {exc}"
        return result

    try:
        client = MilvusClient(milvus_uri)
        collections = client.list_collections()
        result["collections"] = collections
        result["collection_available"] = collection_name in collections
        result["can_add_alias_api_available"] = hasattr(client, "create_alias")
        if not result["collection_available"]:
            result["error"] = "collection not found"
            return result

        stats = client.get_collection_stats(collection_name)
        row_count_raw = stats.get("row_count")
        try:
            row_count = int(row_count_raw)
        except (TypeError, ValueError):
            row_count = row_count_raw
        desc = client.describe_collection(collection_name)

        fields_out: list[dict[str, Any]] = []
        vector_dimension = None
        primary_key_field = None
        field_names: set[str] = set()
        for field in desc.get("fields", []):
            name = str(field.get("name") or "")
            field_names.add(name)
            params = field.get("params") or {}
            ftype = milvus_field_type(field)
            if field.get("is_primary"):
                primary_key_field = name
            if ftype == "FLOAT_VECTOR" or "FLOAT_VECTOR" in ftype:
                vector_dimension = params.get("dim")
            fields_out.append(
                {
                    "name": name,
                    "type": ftype,
                    "params": params,
                    "is_primary": bool(field.get("is_primary")),
                    "description": field.get("description", ""),
                }
            )

        required_fields = {
            "chunk_id",
            "doc_id",
            "text",
            "retrieval_text",
            "metadata_json",
            "embedding",
        }
        schema_compatible = (
            required_fields.issubset(field_names)
            and primary_key_field == "chunk_id"
            and vector_dimension == EXPECTED_VECTOR_DIM
        )

        sample_rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["chunk_id", "doc_id", "retrieval_text"],
            limit=5,
        )
        result["queryable"] = True
        result["sample_query_count"] = len(sample_rows)
        result["sample_query_rows"] = [
            {
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "retrieval_text_nonempty": bool(
                    str(row.get("retrieval_text") or "").strip()
                ),
            }
            for row in sample_rows
        ]

        milvus_ids: set[str] | None = None
        id_query_error = ""
        if isinstance(row_count, int) and row_count <= 16384:
            try:
                id_rows = client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["chunk_id"],
                    limit=max(row_count, 1),
                )
                milvus_ids = {
                    str(row.get("chunk_id") or "")
                    for row in id_rows
                    if str(row.get("chunk_id") or "")
                }
            except Exception as exc:  # noqa: BLE001
                id_query_error = f"{type(exc).__name__}: {exc}"
        else:
            id_query_error = "row_count exceeds Milvus Lite query limit for one-shot set check"

        result.update(
            {
                "row_count": row_count,
                "stats": stats,
                "aliases": desc.get("aliases", []),
                "schema": {
                    "collection_name": desc.get("collection_name"),
                    "description": desc.get("description"),
                    "auto_id": desc.get("auto_id"),
                    "enable_dynamic_field": desc.get("enable_dynamic_field"),
                    "fields": fields_out,
                },
                "field_names": sorted(field_names),
                "primary_key_field": primary_key_field,
                "vector_dimension": vector_dimension,
                "schema_compatible_with_current_retrieval_code": schema_compatible,
                "collection_persistent": not is_tmp_path(milvus_uri),
                "durable_reference_available": not is_tmp_path(milvus_uri),
                "id_query_error": id_query_error,
            }
        )
        if milvus_ids is not None:
            missing_from_milvus = sorted(chunk_ids - milvus_ids)
            extra_in_milvus = sorted(milvus_ids - chunk_ids)
            result.update(
                {
                    "chunk_id_set": milvus_ids,
                    "chunk_id_set_query_count": len(milvus_ids),
                    "chunk_id_set_matches_chunks": milvus_ids == chunk_ids,
                    "chunks_missing_from_milvus_count": len(missing_from_milvus),
                    "chunks_missing_from_milvus_sample": compact_list(missing_from_milvus),
                    "milvus_ids_missing_from_chunks_count": len(extra_in_milvus),
                    "milvus_ids_missing_from_chunks_sample": compact_list(extra_in_milvus),
                }
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_retrieval_results(loaded: dict[str, Any]) -> dict[str, Any]:
    variant_comparison = loaded.get(
        "reports/phase5f4_clean_main_baseline/variant_comparison.json", {}
    )
    main_results = loaded.get(
        "reports/phase5f4_clean_main_baseline/main_results.json", {}
    )
    per_sample_text = loaded.get(
        "reports/phase5f4_clean_main_baseline/per_sample_results.jsonl", ""
    )
    per_sample_count = 0
    baseline_per_sample_count = 0
    baseline_stable_hits_at_10 = 0
    per_sample_parse_errors: list[str] = []

    if isinstance(per_sample_text, str):
        for line_no, line in enumerate(per_sample_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                per_sample_parse_errors.append(f"line {line_no}: {exc}")
                continue
            per_sample_count += 1
            if row.get("index_variant") == "phase5c5_baseline_full":
                baseline_per_sample_count += 1
                if normalize_bool(row.get("stable_block_hit_at_10")) is True:
                    baseline_stable_hits_at_10 += 1

    overall = {}
    by_query_type = {}
    if isinstance(variant_comparison, dict):
        overall = variant_comparison.get("overall", {})
        by_query_type = variant_comparison.get("by_query_type", {})
    if not overall and isinstance(main_results, dict):
        metrics = main_results.get("metrics", {})
        baseline = metrics.get("phase5c5_baseline_full", {})
        overall = {"baseline": baseline.get("overall", {})}
        by_query_type = baseline.get("by_query_type", {})

    baseline_overall = overall.get("baseline", {}) if isinstance(overall, dict) else {}
    return {
        "phase5f4_variant_comparison_found": bool(variant_comparison),
        "phase5f4_main_results_found": bool(main_results),
        "per_sample_results_found": bool(per_sample_text),
        "per_sample_rows_total": per_sample_count,
        "baseline_per_sample_rows": baseline_per_sample_count,
        "baseline_per_sample_stable_hit_at_10_count": baseline_stable_hits_at_10,
        "baseline_overall": baseline_overall,
        "baseline_stable_block_hit_at_10": baseline_overall.get("stable_block_hit_at_10"),
        "baseline_doc_hit_at_10": baseline_overall.get("doc_hit_at_10"),
        "by_query_type": by_query_type,
        "parse_errors": per_sample_parse_errors,
    }


def build_target_coverage(
    dataset: dict[str, Any], chunks: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = dataset.get("samples") or []
    doc_ids: set[str] = chunks.get("doc_ids") or set()
    doc_block_to_chunks: dict[tuple[str, str], list[str]] = (
        chunks.get("doc_block_to_chunks") or {}
    )
    chunk_retrieval_nonempty: dict[str, bool] = (
        chunks.get("chunk_retrieval_nonempty") or {}
    )

    rows: list[dict[str, Any]] = []
    query_type_summary: dict[str, Counter[str]] = defaultdict(Counter)
    target_doc_present_count = 0
    stable_any_count = 0
    stable_all_count = 0
    stable_retrievable_count = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        query_type = str(sample.get("query_type") or "")
        target_doc_id = str(sample.get("target_doc_id") or "")
        stable_ids = parse_block_ids(sample.get("stable_target_block_ids"))
        target_doc_present = target_doc_id in doc_ids
        if target_doc_present:
            target_doc_present_count += 1

        present_block_ids: list[str] = []
        missing_block_ids: list[str] = []
        target_chunk_ids: list[str] = []
        for block_id in stable_ids:
            chunk_ids = doc_block_to_chunks.get((target_doc_id, block_id), [])
            if chunk_ids:
                present_block_ids.append(block_id)
                target_chunk_ids.extend(chunk_ids)
            else:
                missing_block_ids.append(block_id)

        target_chunk_ids = sorted(set(target_chunk_ids))
        stable_any = bool(present_block_ids)
        stable_all = bool(stable_ids) and not missing_block_ids
        retrievable = stable_any and any(
            chunk_retrieval_nonempty.get(cid, False) for cid in target_chunk_ids
        )
        if stable_any:
            stable_any_count += 1
        if stable_all:
            stable_all_count += 1
        if retrievable:
            stable_retrievable_count += 1

        row = {
            "sample_id": sample_id,
            "query_type": query_type,
            "target_doc_id": target_doc_id,
            "target_doc_present": target_doc_present,
            "stable_target_block_ids": ";".join(stable_ids),
            "present_stable_block_ids": ";".join(sorted(set(present_block_ids))),
            "missing_stable_block_ids": ";".join(sorted(set(missing_block_ids))),
            "stable_block_any_present": stable_any,
            "stable_block_all_present": stable_all,
            "target_block_chunk_ids": ";".join(target_chunk_ids),
            "target_block_retrieval_text_nonempty": retrievable,
            "target_doc_present_but_stable_block_missing": (
                target_doc_present and not stable_any
            ),
            "stable_block_present_but_unretrievable": stable_any and not retrievable,
        }
        rows.append(row)

        query_type_summary[query_type]["total"] += 1
        query_type_summary[query_type]["target_doc_present"] += int(target_doc_present)
        query_type_summary[query_type]["stable_any_present"] += int(stable_any)
        query_type_summary[query_type]["stable_all_present"] += int(stable_all)
        query_type_summary[query_type]["stable_retrievable"] += int(retrievable)

    total = len(samples)
    summary = {
        "total_samples": total,
        "target_doc_coverage_count": target_doc_present_count,
        "target_doc_coverage_expected": total,
        "stable_block_any_coverage_count": stable_any_count,
        "stable_block_all_coverage_count": stable_all_count,
        "stable_block_retrievable_count": stable_retrievable_count,
        "target_doc_coverage_pass": target_doc_present_count == total,
        "stable_block_coverage_pass": stable_any_count == total,
        "stable_block_all_ids_present_pass": stable_all_count == total,
        "stable_block_retrievable_pass": stable_retrievable_count == total,
        "query_type_summary": {
            key: dict(counter) for key, counter in sorted(query_type_summary.items())
        },
        "target_doc_present_but_stable_block_missing_count": sum(
            1 for row in rows if row["target_doc_present_but_stable_block_missing"]
        ),
        "stable_block_present_but_unretrievable_count": sum(
            1 for row in rows if row["stable_block_present_but_unretrievable"]
        ),
        "missing_rows": [
            row
            for row in rows
            if not row["target_doc_present"]
            or not row["stable_block_any_present"]
            or not row["target_block_retrieval_text_nonempty"]
        ],
        "partial_stable_block_missing_rows": [
            row
            for row in rows
            if row["stable_block_any_present"] and not row["stable_block_all_present"]
        ],
    }
    return rows, summary


def write_target_coverage_csv(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "baseline_full_target_coverage.csv"
    fieldnames = [
        "sample_id",
        "query_type",
        "target_doc_id",
        "target_doc_present",
        "stable_target_block_ids",
        "present_stable_block_ids",
        "missing_stable_block_ids",
        "stable_block_any_present",
        "stable_block_all_present",
        "target_block_chunk_ids",
        "target_block_retrieval_text_nonempty",
        "target_doc_present_but_stable_block_missing",
        "stable_block_present_but_unretrievable",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_consistency(
    chunks: dict[str, Any],
    bm25: dict[str, Any],
    milvus: dict[str, Any],
    build_stats: dict[str, Any],
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []

    if not chunks.get("readable"):
        blocking.append("chunks file is missing or unreadable")
    if chunks.get("chunk_count") != EXPECTED_BASELINE_CHUNK_COUNT:
        blocking.append(
            f"chunks count is {chunks.get('chunk_count')}, expected {EXPECTED_BASELINE_CHUNK_COUNT}"
        )
    if chunks.get("duplicate_chunk_id_count"):
        blocking.append("chunks contain duplicate chunk_id values")
    if chunks.get("empty_doc_id_count"):
        blocking.append("chunks contain empty doc_id values")
    if chunks.get("doc_id_count") != EXPECTED_BASELINE_DOC_COUNT:
        blocking.append(
            f"chunks doc_id count is {chunks.get('doc_id_count')}, expected {EXPECTED_BASELINE_DOC_COUNT}"
        )
    if chunks.get("missing_expected_doc_ids") != EXPECTED_MISSING_DOC_IDS:
        warnings.append(
            "missing doc_id set differs from expected doc_0314/doc_0427/doc_0526"
        )
    if chunks.get("empty_retrieval_text_count"):
        blocking.append("chunks contain empty retrieval_text values")
    if chunks.get("table_enhancement_metadata_chunk_count"):
        blocking.append("chunks contain table enhancement/table_related metadata")
    if chunks.get("caption_cleanup_metadata_chunk_count"):
        blocking.append("chunks contain caption cleanup experimental metadata")
    if chunks.get("retrieval_text_varchar_risk_count"):
        blocking.append("retrieval_text VARCHAR length risk detected")
    if chunks.get("text_varchar_risk_count"):
        blocking.append("text VARCHAR length risk detected")
    if chunks.get("metadata_json_varchar_risk_count_estimate"):
        warnings.append("metadata_json VARCHAR estimate exceeds 8192 for some chunks")

    if not bm25.get("readable"):
        blocking.append("BM25 cache is missing or unreadable")
    if bm25.get("record_count") != EXPECTED_BASELINE_CHUNK_COUNT:
        blocking.append(
            f"BM25 record count is {bm25.get('record_count')}, expected {EXPECTED_BASELINE_CHUNK_COUNT}"
        )
    if bm25.get("chunk_id_set_matches_chunks") is not True:
        blocking.append("BM25 chunk_id set does not equal chunks chunk_id set")

    if not milvus.get("collection_available"):
        blocking.append("Milvus collection is missing")
    if milvus.get("row_count") != EXPECTED_BASELINE_CHUNK_COUNT:
        blocking.append(
            f"Milvus row count is {milvus.get('row_count')}, expected {EXPECTED_BASELINE_CHUNK_COUNT}"
        )
    if milvus.get("schema_compatible_with_current_retrieval_code") is not True:
        blocking.append("Milvus schema is not compatible with current retrieval code")
    if milvus.get("vector_dimension") != EXPECTED_VECTOR_DIM:
        blocking.append(
            f"Milvus vector dimension is {milvus.get('vector_dimension')}, expected {EXPECTED_VECTOR_DIM}"
        )
    if milvus.get("chunk_id_set_matches_chunks") is False:
        blocking.append("Milvus chunk_id set does not equal chunks chunk_id set")
    if milvus.get("id_query_error"):
        warnings.append(f"Milvus full chunk_id set query warning: {milvus.get('id_query_error')}")

    bge_warnings = build_stats.get("bge_tokenizer_truncation_warning_count")
    if bge_warnings:
        warnings.append(f"historical BGE tokenizer truncation warnings: {bge_warnings}")
    if build_stats.get("milvus_varchar_truncation_warning_count"):
        blocking.append("historical Milvus VARCHAR truncation warnings were recorded")

    status = "fail" if blocking else "pass"
    return {
        "consistency_status": status,
        "blocking_issues": blocking,
        "non_blocking_warnings": warnings,
        "checks": {
            "chunks_count": chunks.get("chunk_count"),
            "chunks_chunk_id_unique": chunks.get("duplicate_chunk_id_count") == 0,
            "chunks_doc_id_nonempty": chunks.get("empty_doc_id_count") == 0,
            "chunks_doc_id_count": chunks.get("doc_id_count"),
            "missing_doc_ids": chunks.get("missing_expected_doc_ids"),
            "bm25_record_count": bm25.get("record_count"),
            "bm25_chunk_ids_match_chunks": bm25.get("chunk_id_set_matches_chunks"),
            "milvus_row_count": milvus.get("row_count"),
            "milvus_chunk_ids_match_chunks": milvus.get("chunk_id_set_matches_chunks"),
            "milvus_schema_compatible": milvus.get(
                "schema_compatible_with_current_retrieval_code"
            ),
            "vector_dimension": milvus.get("vector_dimension"),
            "retrieval_text_field_exists": "retrieval_text" in (
                milvus.get("field_names") or []
            ),
            "metadata_json_field_exists": "metadata_json" in (
                milvus.get("field_names") or []
            ),
            "empty_retrieval_text_count": chunks.get("empty_retrieval_text_count"),
            "retrieval_text_varchar_risk_count": chunks.get(
                "retrieval_text_varchar_risk_count"
            ),
            "text_varchar_risk_count": chunks.get("text_varchar_risk_count"),
            "token_count_over_4096_count": chunks.get("token_count_over_4096_count"),
            "table_enhancement_metadata_chunk_count": chunks.get(
                "table_enhancement_metadata_chunk_count"
            ),
            "caption_cleanup_metadata_chunk_count": chunks.get(
                "caption_cleanup_metadata_chunk_count"
            ),
        },
    }


def build_promotion_feasibility(
    dataset: dict[str, Any],
    chunks: dict[str, Any],
    bm25: dict[str, Any],
    milvus: dict[str, Any],
    coverage_summary: dict[str, Any],
    consistency: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    if dataset.get("sha256_matches_expected") is not True:
        blockers.append("dataset SHA256 does not match the frozen expected hash")
    if not chunks.get("readable"):
        blockers.append("chunks asset missing or unreadable")
    elif chunks.get("chunk_count") != EXPECTED_BASELINE_CHUNK_COUNT:
        blockers.append("chunks count is not 15,802")
    if not bm25.get("readable"):
        blockers.append("BM25 cache missing or unreadable")
    elif bm25.get("record_count") != EXPECTED_BASELINE_CHUNK_COUNT:
        blockers.append("BM25 record count is not 15,802")
    if bm25.get("chunk_id_set_matches_chunks") is not True:
        blockers.append("BM25 cache is not aligned one-to-one with chunks")
    if not milvus.get("collection_available"):
        blockers.append("Milvus collection missing")
    elif milvus.get("row_count") != EXPECTED_BASELINE_CHUNK_COUNT:
        blockers.append("Milvus row count is not 15,802")
    if milvus.get("chunk_id_set_matches_chunks") is False:
        blockers.append("Milvus chunk_id set is not aligned with chunks")
    if coverage_summary.get("target_doc_coverage_pass") is not True:
        blockers.append("target_doc coverage is not 90/90")
    if coverage_summary.get("stable_block_coverage_pass") is not True:
        blockers.append("stable_block coverage is not 90/90")
    if coverage_summary.get("stable_block_retrievable_pass") is not True:
        blockers.append("stable block retrieval_text coverage is not 90/90")
    if chunks.get("table_enhancement_metadata_chunk_count"):
        blockers.append("baseline_full cannot be confirmed non-enhanced")
    if chunks.get("caption_cleanup_metadata_chunk_count"):
        blockers.append("baseline_full contains caption-cleanup metadata")
    if milvus.get("schema_compatible_with_current_retrieval_code") is not True:
        blockers.append("Milvus schema is incompatible")

    if chunks.get("is_tmp_path"):
        warnings.append("chunks asset is under /tmp and must be copied to a durable path")
    if bm25.get("is_tmp_path"):
        warnings.append("BM25 cache is under /tmp and must be copied to a durable path")
    if milvus.get("is_tmp_path"):
        blockers.append(
            "Milvus Lite DB/collection is under /tmp; direct DB-file copying is forbidden, so the collection is not a durable official reference"
        )
    if not milvus.get("durable_reference_available"):
        warnings.append("existing Milvus collection name/alias cannot make the /tmp DB durable")
    if consistency.get("blocking_issues"):
        for issue in consistency["blocking_issues"]:
            if issue not in blockers:
                blockers.append(issue)
    warnings.extend(consistency.get("non_blocking_warnings") or [])

    can_promote = len(blockers) == 0
    file_assets_copyable = (
        chunks.get("readable")
        and chunks.get("chunk_count") == EXPECTED_BASELINE_CHUNK_COUNT
        and bm25.get("readable")
        and bm25.get("record_count") == EXPECTED_BASELINE_CHUNK_COUNT
        and bm25.get("chunk_id_set_matches_chunks") is True
    )
    milvus_only_blocker = (
        not can_promote
        and file_assets_copyable
        and milvus.get("collection_available")
        and milvus.get("row_count") == EXPECTED_BASELINE_CHUNK_COUNT
        and coverage_summary.get("target_doc_coverage_pass") is True
        and coverage_summary.get("stable_block_coverage_pass") is True
        and any("Milvus Lite DB/collection is under /tmp" in b for b in blockers)
    )
    if can_promote:
        recommended = "promote_existing_asset"
        needs_rebuild = False
    elif milvus_only_blocker:
        recommended = "copy_chunks_bm25_but_rebuild_milvus"
        needs_rebuild = True
    else:
        recommended = "minimal_rebuild_required"
        needs_rebuild = True

    return {
        "can_promote_existing_asset": can_promote,
        "partial_file_asset_promotion_possible": bool(file_assets_copyable),
        "promotion_blockers": blockers,
        "promotion_warnings": sorted(set(warnings)),
        "needs_rebuild_if_promote_fails": needs_rebuild,
        "recommended_next_action": recommended,
        "rebuild_scope_if_needed": (
            "Milvus-only durable re-import from verified chunks; no chunk regeneration and no BM25 rebuild"
            if milvus_only_blocker
            else ("none" if can_promote else "targeted minimal rebuild after manual check")
        ),
        "migration_executed_in_this_phase": False,
        "rebuild_executed_in_this_phase": False,
    }


def source_inventory_md(
    sources: dict[str, Any],
    paths: dict[str, Any],
    retrieval_results: dict[str, Any],
) -> list[str]:
    read_files = [p for p, rec in sources.items() if rec.get("loaded")]
    missing_files = [p for p, rec in sources.items() if not rec.get("exists")]
    lines = [
        "# Phase 5F-4B Source Inventory",
        "",
        f"- generated_at: `{utc_now()}`",
        f"- files_read: {len(read_files)}",
        f"- files_missing: {len(missing_files)}",
        "- migration_executed_in_this_phase: `no`",
        "- rebuild_executed_in_this_phase: `no`",
        "",
        "## Candidate Asset Paths",
        "",
        f"- chunks_path: `{paths.get('chunks_path')}`",
        f"- bm25_cache_path: `{paths.get('bm25_path')}`",
        f"- milvus_uri: `{paths.get('milvus_uri')}`",
        f"- milvus_collection: `{paths.get('milvus_collection')}`",
        f"- chunks_under_tmp: `{str(is_tmp_path(paths.get('chunks_path'))).lower()}`",
        f"- bm25_under_tmp: `{str(is_tmp_path(paths.get('bm25_path'))).lower()}`",
        f"- milvus_under_tmp: `{str(is_tmp_path(paths.get('milvus_uri'))).lower()}`",
        f"- durable_candidate_found: `{str(not any(is_tmp_path(paths.get(k)) for k in ('chunks_path', 'bm25_path', 'milvus_uri'))).lower()}`",
        "",
        "## Retrieval Result Evidence",
        "",
        f"- phase5f4_variant_comparison_found: `{str(retrieval_results.get('phase5f4_variant_comparison_found')).lower()}`",
        f"- per_sample_results_found: `{str(retrieval_results.get('per_sample_results_found')).lower()}`",
        f"- baseline_per_sample_rows: {retrieval_results.get('baseline_per_sample_rows')}",
        f"- baseline_stable_block_hit_at_10: {retrieval_results.get('baseline_stable_block_hit_at_10')}",
        "",
        "## Missing Priority Files",
        "",
    ]
    if missing_files:
        lines.extend(f"- `{path}`" for path in missing_files)
    else:
        lines.append("- none")
    lines.extend(["", "## Read Files", ""])
    for path in sorted(read_files):
        info = "; ".join(sources[path].get("provided_info") or [])
        lines.append(f"- `{path}`: {info}")
    return lines


def dataset_freeze_md(dataset: dict[str, Any]) -> list[str]:
    return [
        "# Phase 5F-4B Dataset Freeze Check",
        "",
        f"- dataset_path: `{dataset.get('path')}`",
        f"- exists: `{str(dataset.get('exists')).lower()}`",
        f"- sha256: `{dataset.get('sha256')}`",
        f"- expected_sha256: `{EXPECTED_DATASET_SHA256}`",
        f"- sha256_matches_expected: `{str(dataset.get('sha256_matches_expected')).lower()}`",
        f"- total_samples: {dataset.get('total_samples')}",
        f"- query_type_distribution: `{dataset.get('query_type_distribution')}`",
        f"- stable_target_block_ids_nonempty: {dataset.get('stable_target_block_ids_nonempty_count')}/{dataset.get('total_samples')}",
        f"- include_in_main_denominator_all_true: `{str(dataset.get('include_in_main_denominator_all_true')).lower()}`",
        f"- hard_rule_passed_all_true: `{str(dataset.get('hard_rule_passed_all_true')).lower()}`",
        f"- diagnostic_or_lexical_stress_mixed_count: {dataset.get('diagnostic_or_lexical_stress_mixed_count')}",
        f"- blocker: `{str(dataset.get('blocker')).lower()}`",
    ]


def asset_locator_md(
    chunks: dict[str, Any],
    bm25: dict[str, Any],
    milvus: dict[str, Any],
    paths: dict[str, Any],
    retrieval_results: dict[str, Any],
) -> list[str]:
    build_stats = paths.get("index_build_stats") or {}
    lines = [
        "# Phase 5F-4B Baseline Full Asset Locator",
        "",
        "## Chunks",
        "",
        f"- path: `{chunks.get('path')}`",
        f"- exists/readable: `{str(chunks.get('exists')).lower()}` / `{str(chunks.get('readable')).lower()}`",
        f"- under_tmp: `{str(chunks.get('is_tmp_path')).lower()}`",
        f"- size_bytes: {chunks.get('size_bytes')}",
        f"- sha256: `{chunks.get('sha256')}`",
        f"- chunk_count: {chunks.get('chunk_count')}",
        f"- baseline_full_count_match: `{str(chunks.get('chunk_count') == EXPECTED_BASELINE_CHUNK_COUNT).lower()}`",
        f"- table_enhancement_metadata_chunk_count: {chunks.get('table_enhancement_metadata_chunk_count')}",
        f"- table_related_association_chunk_count: {chunks.get('table_related_association_chunk_count')}",
        f"- caption_cleanup_metadata_chunk_count: {chunks.get('caption_cleanup_metadata_chunk_count')}",
        f"- non_enhanced_baseline: `{str(chunks.get('is_non_enhanced_baseline')).lower()}`",
        "",
        "## BM25 Cache",
        "",
        f"- path: `{bm25.get('path')}`",
        f"- exists/readable: `{str(bm25.get('exists')).lower()}` / `{str(bm25.get('readable')).lower()}`",
        f"- under_tmp: `{str(bm25.get('is_tmp_path')).lower()}`",
        f"- size_bytes: {bm25.get('size_bytes')}",
        f"- sha256: `{bm25.get('sha256')}`",
        f"- record_count: {bm25.get('record_count')}",
        f"- chunk_id_set_matches_chunks: `{str(bm25.get('chunk_id_set_matches_chunks')).lower()}`",
        "",
        "## Milvus Collection",
        "",
        f"- uri: `{milvus.get('uri')}`",
        f"- collection: `{milvus.get('collection')}`",
        f"- db_exists: `{str(milvus.get('db_exists')).lower()}`",
        f"- collection_available: `{str(milvus.get('collection_available')).lower()}`",
        f"- queryable: `{str(milvus.get('queryable')).lower()}`",
        f"- under_tmp: `{str(milvus.get('is_tmp_path')).lower()}`",
        f"- collection_persistent: `{str(milvus.get('collection_persistent')).lower()}`",
        f"- row_count: {milvus.get('row_count')}",
        f"- aliases: `{milvus.get('aliases')}`",
        f"- can_add_alias_api_available: `{str(milvus.get('can_add_alias_api_available')).lower()}`",
        f"- primary_key_field: `{milvus.get('primary_key_field')}`",
        f"- vector_dimension: {milvus.get('vector_dimension')}",
        f"- schema_compatible: `{str(milvus.get('schema_compatible_with_current_retrieval_code')).lower()}`",
        f"- chunk_id_set_matches_chunks: `{str(milvus.get('chunk_id_set_matches_chunks')).lower()}`",
        "",
        "## Build Metadata",
        "",
        f"- build_summary_found: `{str(bool(build_stats)).lower()}`",
        f"- chunks_jsonl_from_build_stats: `{build_stats.get('chunks_jsonl')}`",
        f"- milvus_uri_from_build_stats: `{build_stats.get('milvus_uri')}`",
        f"- bm25_path_from_build_stats: `{build_stats.get('bm25_path')}`",
        f"- import_elapsed_sec: {build_stats.get('import_elapsed_sec')}",
        f"- bm25_build_elapsed_sec: {build_stats.get('bm25_build_elapsed_sec')}",
        f"- bge_tokenizer_truncation_warning_count: {build_stats.get('bge_tokenizer_truncation_warning_count')}",
        f"- milvus_varchar_truncation_warning_count: {build_stats.get('milvus_varchar_truncation_warning_count')}",
        "- exact historical build command: `missing`",
        "- config snapshot: `missing`",
        "- embedding model: `BGE-M3, dim=1024 inferred from import script/schema`",
        "",
        "## Retrieval Results",
        "",
        f"- strict_main per-sample baseline rows: {retrieval_results.get('baseline_per_sample_rows')}",
        f"- stable@10: {retrieval_results.get('baseline_stable_block_hit_at_10')}",
        f"- doc@10: {retrieval_results.get('baseline_doc_hit_at_10')}",
        "- query_type_breakdown_found: `true`",
    ]
    return lines


def consistency_md(consistency: dict[str, Any]) -> list[str]:
    lines = [
        "# Phase 5F-4B Baseline Full Consistency Check",
        "",
        f"- consistency_status: `{consistency.get('consistency_status')}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in consistency.get("checks", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blocking Issues", ""])
    blockers = consistency.get("blocking_issues") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Non-Blocking Warnings", ""])
    warnings = consistency.get("non_blocking_warnings") or []
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- none")
    return lines


def target_coverage_md(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Phase 5F-4B Baseline Full Target Coverage",
        "",
        f"- total_samples: {summary.get('total_samples')}",
        f"- target_doc_coverage: {summary.get('target_doc_coverage_count')}/{summary.get('target_doc_coverage_expected')}",
        f"- stable_block_coverage: {summary.get('stable_block_any_coverage_count')}/{summary.get('total_samples')}",
        f"- stable_block_all_ids_present: {summary.get('stable_block_all_coverage_count')}/{summary.get('total_samples')}",
        f"- stable_block_retrievable: {summary.get('stable_block_retrievable_count')}/{summary.get('total_samples')}",
        f"- target_doc_present_but_stable_block_missing_count: {summary.get('target_doc_present_but_stable_block_missing_count')}",
        f"- stable_block_present_but_unretrievable_count: {summary.get('stable_block_present_but_unretrievable_count')}",
        "",
        "## Query Type Summary",
        "",
        "| query_type | total | target_doc_present | stable_all_present | stable_retrievable |",
        "|---|---:|---:|---:|---:|",
    ]
    for query_type, row in summary.get("query_type_summary", {}).items():
        lines.append(
            f"| {query_type} | {row.get('total', 0)} | {row.get('target_doc_present', 0)} | "
            f"{row.get('stable_all_present', 0)} | {row.get('stable_retrievable', 0)} |"
        )
    lines.extend(["", "## Missing Or Unretrievable Rows", ""])
    missing_rows = summary.get("missing_rows") or []
    if missing_rows:
        for row in missing_rows:
            lines.append(
                f"- `{row.get('sample_id')}`: target_doc_present={row.get('target_doc_present')}, "
                f"stable_all_present={row.get('stable_block_all_present')}, "
                f"retrieval_text_nonempty={row.get('target_block_retrieval_text_nonempty')}"
            )
    else:
        lines.append("- none")
    partial_rows = summary.get("partial_stable_block_missing_rows") or []
    lines.extend(["", "## Partial Stable Block ID Misses", ""])
    if partial_rows:
        for row in partial_rows:
            lines.append(
                f"- `{row.get('sample_id')}`: present={row.get('present_stable_block_ids')}, "
                f"missing={row.get('missing_stable_block_ids')}, chunk_ids={row.get('target_block_chunk_ids')}"
            )
    else:
        lines.append("- none")
    return lines


def promotion_feasibility_md(feasibility: dict[str, Any]) -> list[str]:
    lines = [
        "# Phase 5F-4B Promotion Feasibility",
        "",
        f"- can_promote_existing_asset: `{str(feasibility.get('can_promote_existing_asset')).lower()}`",
        f"- partial_file_asset_promotion_possible: `{str(feasibility.get('partial_file_asset_promotion_possible')).lower()}`",
        f"- needs_rebuild_if_promote_fails: `{str(feasibility.get('needs_rebuild_if_promote_fails')).lower()}`",
        f"- recommended_next_action: `{feasibility.get('recommended_next_action')}`",
        f"- rebuild_scope_if_needed: {feasibility.get('rebuild_scope_if_needed')}",
        f"- migration_executed_in_this_phase: `no`",
        f"- rebuild_executed_in_this_phase: `no`",
        "",
        "## Promotion Blockers",
        "",
    ]
    blockers = feasibility.get("promotion_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Promotion Warnings", ""])
    warnings = feasibility.get("promotion_warnings") or []
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- none")
    return lines


def durable_layout_md(feasibility: dict[str, Any]) -> list[str]:
    recommended = feasibility.get("recommended_next_action")
    return [
        "# Durable Asset Layout Plan",
        "",
        "- this_phase_executes_migration: `no`",
        "- this_phase_executes_rebuild: `no`",
        "- recommended_root: `data/baselines/phase5f_official_clean_baseline/`",
        f"- applies_to_recommended_action: `{recommended}`",
        "",
        "```text",
        "data/baselines/phase5f_official_clean_baseline/",
        "  README.md",
        "  manifest.json",
        "  dataset/",
        "    strict_main_eval_set_v2.sha256.txt",
        "    strict_main_eval_set_v2.path.txt",
        "  chunks/",
        "    chunks.jsonl",
        "    chunks.sha256.txt",
        "    chunk_stats.json",
        "  bm25/",
        "    bm25_index.json",
        "    bm25.sha256.txt",
        "    bm25_manifest.json",
        "  milvus/",
        "    collection_manifest.json",
        "    collection_schema.json",
        "    alias_info.json",
        "  reports/",
        "    copied_phase5f4_results/",
        "    copied_phase5c5_build_reports/",
        "  config_snapshot/",
        "    settings_snapshot.yaml",
        "    build_command.txt",
        "    environment_notes.md",
        "```",
        "",
        "## Manifest Requirements",
        "",
        "- Do not overwrite production assets.",
        "- Do not store official assets under `/tmp`.",
        "- Record SHA256 for dataset, chunks, and BM25 cache.",
        "- Record source path, copied path, dataset hash, chunk count, BM25 record count, and Milvus row count.",
        "- Record `table_enhancement=false`.",
        "- Record `caption_cleanup=false` or default-off.",
        "- Record that this is the official clean baseline, not the enhanced variant.",
        "- Do not copy Milvus bottom-level DB files directly; reference an existing durable collection or rebuild/import a durable collection from verified chunks.",
    ]


def promotion_execution_plan_md(feasibility: dict[str, Any]) -> list[str]:
    return [
        "# Promotion Execution Plan",
        "",
        "- this_phase_executes_migration: `no`",
        "- this_phase_executes_rebuild: `no`",
        "- requires_user_confirmation_before_execution: `yes`",
        "",
        "1. Create `data/baselines/phase5f_official_clean_baseline/`.",
        "2. Copy verified `chunks.jsonl` into `chunks/`.",
        "3. Compute and save `chunks.sha256.txt`.",
        "4. Copy verified BM25 cache into `bm25/`.",
        "5. Compute and save BM25 hash and `bm25_manifest.json`.",
        "6. Copy relevant Phase 5F-4 and Phase 5C-5 reports into `reports/`.",
        "7. Copy or write config snapshot notes without changing runtime retrieval config.",
        "8. Generate official `manifest.json`.",
        "9. Verify copied chunks line count and SHA256.",
        "10. Verify copied BM25 record count and chunk_id alignment.",
        "11. Verify Milvus collection row count.",
        "12. If a durable Milvus collection exists, create or record alias after explicit confirmation.",
        "13. If alias is not enough because the collection URI is `/tmp`, re-import Milvus from verified chunks into a durable, non-production collection after explicit confirmation.",
        "14. Use the frozen dataset for coverage check after promotion.",
        "15. If Milvus is rebuilt/imported, rerun strict_main retrieval for the official baseline.",
        "16. Rollback by removing only newly created durable baseline files/alias/collection from the promotion phase; never touch production assets.",
        "",
        "Milvus rule: do not copy bottom-level Milvus DB files directly. Use an existing durable collection or create a new durable collection by import from verified chunks.",
        "",
        f"Recommended action from this audit: `{feasibility.get('recommended_next_action')}`.",
    ]


def fallback_plan_md(feasibility: dict[str, Any]) -> list[str]:
    if feasibility.get("can_promote_existing_asset"):
        return [
            "# Fallback Minimal Rebuild Plan",
            "",
            "Fallback rebuild not needed. Existing assets can be promoted after user confirmation.",
        ]
    return [
        "# Fallback Minimal Rebuild Plan",
        "",
        "Existing file assets are internally consistent, but the Milvus Lite DB lives under `/tmp` and direct DB-file copying is forbidden. That prevents direct official promotion of the complete existing asset.",
        "",
        "## Missing Or Non-Durable Asset",
        "",
        "- chunks: found, verified, no regeneration needed.",
        "- BM25: found, verified, no rebuild needed.",
        "- Milvus: found and internally consistent, but not durable because URI is under `/tmp`.",
        "",
        "## Minimal Rebuild Scope",
        "",
        "- Rebuild/import only Milvus into a durable non-production collection from the verified chunks asset.",
        "- Do not regenerate chunks.",
        "- Do not rebuild BM25 unless copied BM25 verification fails in the execution phase.",
        "- Do not rebuild enhanced.",
        "- Do not tune retrieval parameters.",
        "- Do not modify the frozen dataset.",
        "",
        "## Expected Counts",
        "",
        "- expected chunk count: 15,802",
        "- expected Milvus row count: 15,802",
        "- expected BM25 records: 15,802",
        "",
        "## Validation",
        "",
        "- Verify durable chunks SHA256 equals the audited source chunks SHA256.",
        "- Verify BM25 record count and chunk_id set equals chunks.",
        "- Verify durable Milvus row count is 15,802.",
        "- Verify durable Milvus chunk_id set equals chunks.",
        "- Verify target_doc coverage is 90/90.",
        "- Verify stable_block coverage is 90/90.",
        "- Rerun strict_main retrieval after Milvus import.",
        "",
        "## Rollback",
        "",
        "- Remove only newly created durable collection/alias and files from the execution phase.",
        "- Do not touch production `synbio_papers` or current default assets.",
    ]


def rerun_decision_md(feasibility: dict[str, Any]) -> list[str]:
    if feasibility.get("can_promote_existing_asset"):
        rerun_now = "no"
        rerun_type = "none"
        rationale = "Copy-only promotion with unchanged Milvus collection can reuse Phase 5F-4 results; a smoke check is enough if an alias is created."
        next_phase = "Phase 5F-4C Execute Official Baseline Asset Promotion"
    elif feasibility.get("recommended_next_action") == "copy_chunks_bm25_but_rebuild_milvus":
        rerun_now = "no"
        rerun_type = "full_retrieval_rerun_after_rebuild"
        rationale = "No migration or rebuild is executed in this audit. Because the recommended next action includes a durable Milvus re-import, strict_main retrieval must be rerun after that execution phase."
        next_phase = "Phase 5F-4C targeted asset promotion with Milvus-only durable import"
    else:
        rerun_now = "no"
        rerun_type = "full_retrieval_rerun_after_rebuild"
        rationale = "A critical asset blocker requires targeted rebuild planning first; rerun belongs after the rebuild."
        next_phase = "Phase 5F-4B minimal rebuild/manual check"
    return [
        "# Retrieval Rerun Decision",
        "",
        f"- rerun_needed_now: `{rerun_now}`",
        f"- rerun_type: `{rerun_type}`",
        f"- rationale: {rationale}",
        f"- recommended_next_phase: {next_phase}",
    ]


def summary_md(
    dataset: dict[str, Any],
    chunks: dict[str, Any],
    bm25: dict[str, Any],
    milvus: dict[str, Any],
    coverage_summary: dict[str, Any],
    feasibility: dict[str, Any],
) -> list[str]:
    can_promote = feasibility.get("can_promote_existing_asset")
    next_phase = (
        "Phase 5F-4C Execute Official Baseline Asset Promotion"
        if can_promote
        else "Phase 5F-4C targeted asset promotion with Milvus-only durable import"
        if feasibility.get("recommended_next_action") == "copy_chunks_bm25_but_rebuild_milvus"
        else "blocker manual check"
    )
    allow_closeout = can_promote is True
    missing_for_closeout = (
        "none"
        if allow_closeout
        else "durable official Milvus collection/reference and promotion manifest"
    )
    return [
        "# Phase 5F-4B Asset Promotion Audit Summary",
        "",
        f"1. strict_main_eval_set_v2 SHA matches: `{str(dataset.get('sha256_matches_expected')).lower()}`",
        f"2. phase5c5_baseline_full chunks found: `{str(chunks.get('readable')).lower()}`",
        f"3. chunks count is 15,802: `{str(chunks.get('chunk_count') == EXPECTED_BASELINE_CHUNK_COUNT).lower()}` ({chunks.get('chunk_count')})",
        f"4. BM25 cache found: `{str(bm25.get('readable')).lower()}`",
        f"5. BM25 records is 15,802: `{str(bm25.get('record_count') == EXPECTED_BASELINE_CHUNK_COUNT).lower()}` ({bm25.get('record_count')})",
        f"6. Milvus collection found: `{str(milvus.get('collection_available')).lower()}`",
        f"7. Milvus row count is 15,802: `{str(milvus.get('row_count') == EXPECTED_BASELINE_CHUNK_COUNT).lower()}` ({milvus.get('row_count')})",
        f"8. target_doc coverage is 90/90: `{str(coverage_summary.get('target_doc_coverage_count') == 90).lower()}` ({coverage_summary.get('target_doc_coverage_count')}/90)",
        f"9. stable_block coverage is 90/90: `{str(coverage_summary.get('stable_block_any_coverage_count') == 90).lower()}` ({coverage_summary.get('stable_block_any_coverage_count')}/90; all_ids_present={coverage_summary.get('stable_block_all_coverage_count')}/90)",
        f"10. asset confirmed not table-enhanced: `{str(chunks.get('table_enhancement_metadata_chunk_count') == 0).lower()}`",
        f"11. can directly promote existing asset: `{str(can_promote).lower()}`",
        f"12. rebuild needed: `{str(feasibility.get('needs_rebuild_if_promote_fails')).lower()}` ({feasibility.get('rebuild_scope_if_needed')})",
        "13. migration executed in this phase: `no`",
        "14. rebuild executed in this phase: `no`",
        "15. retrieval / cleaning / dataset modified: `no`",
        "16. Qwen / RAGAS called: `no`",
        f"17. recommended next phase: `{next_phase}`",
        f"18. allowed to enter Phase 5F-5 closeout: `{str(allow_closeout).lower()}`",
        f"19. missing before closeout: {missing_for_closeout}",
    ]


def next_phase_plan_md(feasibility: dict[str, Any]) -> list[str]:
    action = feasibility.get("recommended_next_action")
    if action == "promote_existing_asset":
        next_phase = "Phase 5F-4C Execute Official Baseline Asset Promotion"
        steps = [
            "Copy chunks and BM25 to durable official baseline path.",
            "Freeze manifest and hashes.",
            "Record existing durable Milvus collection or create alias after confirmation.",
            "Run smoke verification only.",
        ]
    elif action == "copy_chunks_bm25_but_rebuild_milvus":
        next_phase = "Phase 5F-4C Copy Chunks/BM25 + Milvus-only Durable Import + Freeze Manifest"
        steps = [
            "Copy verified chunks and BM25 cache to durable official baseline path.",
            "Import a new durable non-production Milvus collection from the verified chunks.",
            "Create or record alias only after the durable collection is verified.",
            "Rerun strict_main retrieval after Milvus import.",
            "Freeze manifest after counts, hashes, schema, and coverage pass.",
        ]
    else:
        next_phase = "blocker manual check before targeted asset rebuild"
        steps = [
            "Manually inspect promotion blockers.",
            "Choose the narrowest rebuild scope.",
            "Do not regenerate chunks unless chunks verification fails.",
        ]
    lines = ["# Next Phase Plan", "", f"- recommended_next_phase: `{next_phase}`", ""]
    lines.extend(f"- {step}" for step in steps)
    return lines


def source_inventory_json(
    sources: dict[str, Any],
    paths: dict[str, Any],
    retrieval_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "files": sources,
        "read_files": [p for p, rec in sources.items() if rec.get("loaded")],
        "missing_files": [p for p, rec in sources.items() if not rec.get("exists")],
        "phase5c5_baseline_full_asset_info": {
            "chunks_path_found": bool(paths.get("chunks_path")),
            "chunks_path": paths.get("chunks_path"),
            "bm25_cache_path_found": bool(paths.get("bm25_path")),
            "bm25_cache_path": paths.get("bm25_path"),
            "milvus_collection_found": bool(paths.get("milvus_collection")),
            "milvus_collection": paths.get("milvus_collection"),
            "milvus_uri": paths.get("milvus_uri"),
            "build_summary_or_manifest_found": bool(paths.get("index_build_stats")),
            "asset_under_tmp": {
                "chunks": is_tmp_path(paths.get("chunks_path")),
                "bm25": is_tmp_path(paths.get("bm25_path")),
                "milvus": is_tmp_path(paths.get("milvus_uri")),
            },
            "durable_path_found": not any(
                is_tmp_path(paths.get(key))
                for key in ("chunks_path", "bm25_path", "milvus_uri")
            ),
            "migration_executed_in_this_phase": False,
            "rebuild_executed_in_this_phase": False,
        },
        "retrieval_results_evidence": retrieval_results,
    }


def public_asset_json(chunks: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {
        "chunk_id_set",
        "doc_ids",
        "doc_to_chunks",
        "doc_block_to_chunks",
        "block_to_chunks",
        "chunk_retrieval_nonempty",
    }
    return {key: value for key, value in chunks.items() if key not in hidden_keys}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources, loaded = load_priority_sources()
    paths = choose_baseline_paths(loaded)
    dataset = scan_dataset()
    chunks = scan_chunks(paths.get("chunks_path"))
    bm25 = scan_bm25(paths.get("bm25_path"), chunks.get("chunk_id_set") or set())
    milvus = scan_milvus(
        paths.get("milvus_uri"),
        paths.get("milvus_collection"),
        chunks.get("chunk_id_set") or set(),
    )
    retrieval_results = load_retrieval_results(loaded)
    coverage_rows, coverage_summary = build_target_coverage(dataset, chunks)
    build_stats = paths.get("index_build_stats") or {}
    consistency = build_consistency(chunks, bm25, milvus, build_stats)
    feasibility = build_promotion_feasibility(
        dataset, chunks, bm25, milvus, coverage_summary, consistency
    )

    source_json = source_inventory_json(sources, paths, retrieval_results)
    write_json("source_inventory.json", source_json)
    write_md("source_inventory.md", source_inventory_md(sources, paths, retrieval_results))

    dataset_public = dict(dataset)
    dataset_public.pop("samples", None)
    write_json("dataset_freeze_check.json", dataset_public)
    write_md("dataset_freeze_check.md", dataset_freeze_md(dataset))

    locator_json = {
        "generated_at": utc_now(),
        "chunks": public_asset_json(chunks),
        "bm25_cache": {
            key: value
            for key, value in bm25.items()
            if key != "chunk_id_set"
        },
        "milvus": {
            key: value
            for key, value in milvus.items()
            if key != "chunk_id_set"
        },
        "build_metadata": {
            "phase5c5_index_build_stats": build_stats,
            "exact_historical_command_found": False,
            "config_snapshot_found": False,
            "embedding_model": "BGE-M3 inferred; dim 1024",
            "scripts": [
                "scripts/ingestion/preprocess_and_chunk.py --evidence_context_mode compact",
                "scripts/ingestion/import_to_milvus.py",
                "scripts/evaluation/evaluate_phase5c5_full_retrieval_ab.py",
            ],
        },
        "retrieval_results": retrieval_results,
    }
    write_json("baseline_full_asset_locator.json", locator_json)
    write_md(
        "baseline_full_asset_locator.md",
        asset_locator_md(chunks, bm25, milvus, paths, retrieval_results),
    )

    write_json("baseline_full_consistency_check.json", consistency)
    write_md("baseline_full_consistency_check.md", consistency_md(consistency))

    write_target_coverage_csv(coverage_rows)
    write_md("baseline_full_target_coverage.md", target_coverage_md(coverage_summary))

    write_json("promotion_feasibility.json", feasibility)
    write_md("promotion_feasibility.md", promotion_feasibility_md(feasibility))

    write_md("durable_asset_layout_plan.md", durable_layout_md(feasibility))
    write_md("promotion_execution_plan.md", promotion_execution_plan_md(feasibility))
    write_md("fallback_minimal_rebuild_plan.md", fallback_plan_md(feasibility))
    write_md("retrieval_rerun_decision.md", rerun_decision_md(feasibility))
    write_md(
        "summary.md",
        summary_md(dataset, chunks, bm25, milvus, coverage_summary, feasibility),
    )
    write_md("next_phase_plan.md", next_phase_plan_md(feasibility))

    audit_manifest = {
        "generated_at": utc_now(),
        "script": rel(Path(__file__)),
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
        "dataset_sha256_matches": dataset.get("sha256_matches_expected"),
        "chunks_count": chunks.get("chunk_count"),
        "bm25_record_count": bm25.get("record_count"),
        "milvus_row_count": milvus.get("row_count"),
        "target_doc_coverage": coverage_summary.get("target_doc_coverage_count"),
        "stable_block_coverage": coverage_summary.get("stable_block_any_coverage_count"),
        "stable_block_all_ids_present": coverage_summary.get("stable_block_all_coverage_count"),
        "can_promote_existing_asset": feasibility.get("can_promote_existing_asset"),
        "recommended_next_action": feasibility.get("recommended_next_action"),
        "migration_executed_in_this_phase": False,
        "rebuild_executed_in_this_phase": False,
    }
    write_json("audit_manifest.json", audit_manifest)

    print(json.dumps(audit_manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
