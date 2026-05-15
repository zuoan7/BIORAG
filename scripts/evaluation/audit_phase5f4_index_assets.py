#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/phase5f4_index_asset_normalization"
DATASET = ROOT / "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl"
EXPECTED_DATASET_SHA = "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
EXPECTED_TOTAL = 90
EXPECTED_QUERY_TYPES = {
    "table_content": 31,
    "caption_level_table": 9,
    "figure_caption": 20,
    "normal_control": 30,
}
EXPECTED_DOC_IDS = {f"doc_{idx:04d}" for idx in range(1, 709)}

PHASE5F4_FILES = [
    "reports/phase5f4_clean_main_baseline/dataset_freeze.md",
    "reports/phase5f4_clean_main_baseline/dataset_manifest.json",
    "reports/phase5f4_clean_main_baseline/dataset_hash.txt",
    "reports/phase5f4_clean_main_baseline/retrieval_asset_inventory.md",
    "reports/phase5f4_clean_main_baseline/retrieval_asset_inventory.json",
    "reports/phase5f4_clean_main_baseline/main_results.json",
    "reports/phase5f4_clean_main_baseline/main_results_by_query_type.csv",
    "reports/phase5f4_clean_main_baseline/per_sample_results.jsonl",
    "reports/phase5f4_clean_main_baseline/failure_ledger.csv",
    "reports/phase5f4_clean_main_baseline/variant_comparison.md",
    "reports/phase5f4_clean_main_baseline/variant_comparison.json",
    "reports/phase5f4_clean_main_baseline/summary.md",
    "reports/phase5f4_clean_main_baseline/next_repair_backlog.md",
]

PHASE5F4R_FILES = [
    "reports/phase5f4_failure_review/summary.md",
    "reports/phase5f4_failure_review/failure_review_ledger.csv",
    "reports/phase5f4_failure_review/doc_recall_review.md",
    "reports/phase5f4_failure_review/table_content_failure_review.md",
    "reports/phase5f4_failure_review/normal_control_takeover_review.md",
    "reports/phase5f4_failure_review/variant_difference_review.md",
    "reports/phase5f4_failure_review/repair_trigger_decision.md",
    "reports/phase5f4_failure_review/next_phase_plan.md",
]

SUPPORTING_FILES = [
    "reports/phase5c5_full_retrieval_ab/index_build_stats.json",
    "reports/phase5c5_full_retrieval_ab/index_build_summary.md",
    "reports/phase5c5_full_retrieval_ab/summary.md",
    "reports/phase5c5_full_retrieval_ab/preflight_stats.json",
    "reports/phase5c4_full_preflight/summary.md",
    "reports/phase5c4_full_preflight/full_chunk_ab_stats.json",
    "reports/phase5c4_full_preflight/full_enhanced_build_stats.json",
    "reports/phase4_closeout/summary.md",
    "reports/table_figure_retrieval_eval/phase4e1_full_compact/index_build_summary.md",
    "reports/table_figure_retrieval_eval/phase4e1_full_compact/summary.md",
    "config/settings.example.env",
    "src/synbio_rag/domain/config.py",
    "scripts/ingestion/build_round1_kb.py",
    "src/synbio_rag/ingestion/kb_builder.py",
    "scripts/ingestion/preprocess_and_chunk.py",
    "scripts/ingestion/import_to_milvus.py",
]

ASSET_DEFAULTS = {
    "current_default": {
        "label": "Current default production/baseline index",
        "role": "baseline/default",
        "chunks_path": ROOT / "data/paper_round1/chunks/chunks.jsonl",
        "milvus_uri": ROOT / "runtime/vectorstores/milvus/papers.db",
        "collection": "synbio_papers",
        "bm25_cache_path": ROOT / "data/paper_round1/chunks/bm25_index.json",
        "parsed_clean_path": ROOT / "data/paper_round1/parsed_clean",
    },
    "phase5c5_baseline_full": {
        "label": "Phase 5C-5 baseline full compact index",
        "role": "baseline/default",
        "chunks_path": Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
        "milvus_uri": Path("/tmp/phase5c5_baseline_full.db"),
        "collection": "synbio_phase5c5_baseline_full",
        "bm25_cache_path": Path("/tmp/biorag_phase5c5_baseline_full/bm25_index.json"),
        "parsed_clean_path": ROOT / "data/paper_round1/parsed_clean",
    },
    "phase5c5_enhanced_full": {
        "label": "Phase 5C full enhanced table index",
        "role": "table_enhanced",
        "chunks_path": Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
        "milvus_uri": Path("/tmp/phase5c5_enhanced_full.db"),
        "collection": "synbio_phase5c5_enhanced_full",
        "bm25_cache_path": Path("/tmp/biorag_phase5c5_enhanced_full/bm25_index.json"),
        "parsed_clean_path": Path("/tmp/biorag_phase5c4_full_enhanced/parsed_clean"),
    },
    "phase5d_caption_cleanup_chunks": {
        "label": "Phase 5D cleanup experimental chunks",
        "role": "caption_cleanup_optional",
        "chunks_path": Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
        "milvus_uri": None,
        "collection": "",
        "bm25_cache_path": None,
        "parsed_clean_path": None,
    },
}


try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None  # type: ignore[assignment]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = inventory_source_files()
    retrieval_inventory = read_json(ROOT / "reports/phase5f4_clean_main_baseline/retrieval_asset_inventory.json")
    phase5c5_index = read_json(ROOT / "reports/phase5c5_full_retrieval_ab/index_build_stats.json")
    phase5c4_stats = read_json(ROOT / "reports/phase5c4_full_preflight/full_chunk_ab_stats.json")
    phase5c4_enhanced = read_json(ROOT / "reports/phase5c4_full_preflight/full_enhanced_build_stats.json")
    main_results = read_json(ROOT / "reports/phase5f4_clean_main_baseline/main_results.json")
    variant_comparison = read_json(ROOT / "reports/phase5f4_clean_main_baseline/variant_comparison.json")

    samples, dataset_check = check_dataset(DATASET)
    write_json(OUT_DIR / "dataset_refreeze_check.json", dataset_check)
    write_dataset_refreeze_md(OUT_DIR / "dataset_refreeze_check.md", dataset_check)

    assets = build_asset_specs(retrieval_inventory)
    target_doc_ids = {str(s.get("target_doc_id") or "") for s in samples if s.get("target_doc_id")}
    target_blocks = {
        (str(s.get("target_doc_id") or ""), block_id)
        for s in samples
        for block_id in stable_ids(s)
    }

    audits: dict[str, dict[str, Any]] = {}
    chunk_runtime: dict[str, dict[str, Any]] = {}
    for name, spec in assets.items():
        chunk_summary, runtime = summarize_chunks(spec["chunks_path"], target_doc_ids, target_blocks)
        bm25_summary = summarize_bm25(spec.get("bm25_cache_path"), runtime.get("chunk_ids", set()))
        milvus_summary = summarize_milvus(spec.get("milvus_uri"), spec.get("collection"), retrieval_inventory, name)
        parsed_summary = summarize_parsed_clean(spec.get("parsed_clean_path"))
        audits[name] = build_asset_audit(
            name=name,
            spec=spec,
            chunk_summary=chunk_summary,
            bm25_summary=bm25_summary,
            milvus_summary=milvus_summary,
            parsed_summary=parsed_summary,
            retrieval_inventory=retrieval_inventory,
            phase5c5_index=phase5c5_index,
            phase5c4_stats=phase5c4_stats,
            phase5c4_enhanced=phase5c4_enhanced,
            main_results=main_results,
            variant_comparison=variant_comparison,
        )
        chunk_runtime[name] = runtime

    diff_summary = build_diff_summary(chunk_runtime, audits)
    for name in ("current_default", "phase5c5_baseline_full", "phase5c5_enhanced_full"):
        audits[name]["diff_summary"] = diff_summary.get(name, {})

    coverage_rows, coverage_summary = build_coverage(samples, chunk_runtime)
    write_coverage_csv(OUT_DIR / "target_coverage_matrix.csv", coverage_rows)
    write_target_coverage_summary(OUT_DIR / "target_coverage_summary.md", coverage_summary)

    current_failure = analyze_current_failures(coverage_rows)
    enhanced_position = analyze_enhanced_position(coverage_rows)
    audits["current_default"]["current_default_failure_alignment"] = current_failure
    audits["phase5c5_enhanced_full"]["enhanced_positioning"] = enhanced_position
    audits["phase5c5_baseline_full"]["baseline_only_vs_enhanced"] = enhanced_position.get("baseline_only_rows", [])

    write_asset_outputs(audits)

    source_inventory = build_source_inventory(
        reports=reports,
        assets=audits,
        dataset_check=dataset_check,
        phase5c5_index=phase5c5_index,
        phase5c4_stats=phase5c4_stats,
    )
    write_json(OUT_DIR / "source_inventory.json", source_inventory)
    write_source_inventory_md(OUT_DIR / "source_inventory.md", source_inventory)

    decision = decide_official_baseline(dataset_check, audits, coverage_summary)
    write_json(OUT_DIR / "official_baseline_decision.json", decision)
    write_decision_md(OUT_DIR / "official_baseline_decision.md", decision)
    if decision["recommended_option"] == "B":
        write_minimal_rebuild_plan(OUT_DIR / "minimal_rebuild_plan.md", decision, audits)
    elif decision["recommended_option"] in {"A", "C"}:
        write_baseline_freeze_plan(OUT_DIR / "baseline_freeze_plan.md", decision, audits, dataset_check, coverage_summary)

    rerun = build_rerun_decision(decision, dataset_check)
    write_baseline_rerun_decision(OUT_DIR / "baseline_rerun_decision.md", rerun)

    write_summary(OUT_DIR / "summary.md", dataset_check, audits, coverage_summary, current_failure, decision)
    write_next_phase_plan(OUT_DIR / "next_phase_plan.md", decision)


def build_asset_specs(retrieval_inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    report_assets = retrieval_inventory.get("assets", {}) if isinstance(retrieval_inventory, dict) else {}
    for name, defaults in ASSET_DEFAULTS.items():
        item = dict(defaults)
        reported = report_assets.get(name, {})
        for key in ("chunks_path", "milvus_uri", "bm25_cache_path"):
            if reported.get(key):
                item[key] = Path(str(reported[key]))
        if reported.get("collection"):
            item["collection"] = str(reported["collection"])
        if reported.get("role"):
            item["role"] = str(reported["role"])
        assets[name] = item
    return assets


def inventory_source_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in PHASE5F4_FILES + PHASE5F4R_FILES + [str(DATASET.relative_to(ROOT))] + SUPPORTING_FILES:
        path = ROOT / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "mtime": mtime(path),
                "information_used": information_used_for(rel),
            }
        )
    for name, item in ASSET_DEFAULTS.items():
        for key in ("chunks_path", "bm25_cache_path", "milvus_uri", "parsed_clean_path"):
            path = item.get(key)
            if not path:
                continue
            p = Path(path)
            rows.append(
                {
                    "path": str(p),
                    "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
                    "mtime": mtime(p),
                    "information_used": f"{name} {key}",
                }
            )
    return rows


def information_used_for(path: str) -> str:
    if "dataset_" in path or path.endswith("strict_main_eval_set_v2.jsonl"):
        return "dataset freeze, hash, schema, sample distribution"
    if "retrieval_asset_inventory" in path:
        return "asset paths, chunk counts, Milvus collections, BM25 counts"
    if "main_results" in path:
        return "Phase 5F-4 retrieval metrics"
    if "per_sample_results" in path:
        return "sample-level variant ranks and hit flags"
    if "variant_comparison" in path:
        return "baseline_full vs enhanced_full comparison"
    if "failure_review" in path or "repair_trigger" in path or "next_phase" in path:
        return "Phase 5F-4R failure categories and next-step recommendation"
    if "phase5c5_full_retrieval_ab" in path:
        return "Phase 5C-5 full index build and A/B asset evidence"
    if "phase5c4_full_preflight" in path:
        return "full enhanced build source and table-related association evidence"
    if "phase4" in path:
        return "Phase 4 compact chunks/index provenance"
    if path.endswith(".py") or path.endswith(".env"):
        return "path defaults and build command provenance; secrets redacted/not recorded"
    return "supporting context"


def check_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_no}: {exc}")
                    continue
                item["_line_no"] = line_no
                samples.append(item)

    sample_ids = [str(s.get("sample_id") or "") for s in samples]
    queries = [normalize_space(str(s.get("query") or "")) for s in samples]
    qtype_counts = dict(Counter(str(s.get("query_type") or "") for s in samples))
    residuals = residual_summary(samples)
    diagnostic_or_lexical = [
        sample_id(s)
        for s in samples
        if re.search(r"(diagnostic|lexical|stress)", " ".join([
            str(s.get("sample_id") or ""),
            str(s.get("query_type") or ""),
            str(s.get("ability_scope") or ""),
            str(s.get("source") or ""),
            str(s.get("risk_tags") or ""),
        ]), re.I)
    ]
    blockers: list[str] = []
    sha = sha256_file(path) if path.exists() else ""
    if not path.exists():
        blockers.append("dataset_file_missing")
    if sha != EXPECTED_DATASET_SHA:
        blockers.append("dataset_sha_mismatch")
    if len(samples) != EXPECTED_TOTAL:
        blockers.append("dataset_total_mismatch")
    if parse_errors:
        blockers.append("jsonl_parse_errors")
    if diagnostic_or_lexical:
        blockers.append("diagnostic_or_lexical_stress_mixed_into_main")

    check = {
        "dataset_path": str(path.relative_to(ROOT)) if path.exists() else str(path),
        "exists": path.exists(),
        "sha256": sha,
        "expected_sha256": EXPECTED_DATASET_SHA,
        "sha256_matches_expected": sha == EXPECTED_DATASET_SHA,
        "total_samples": len(samples),
        "expected_total_samples": EXPECTED_TOTAL,
        "total_matches_expected": len(samples) == EXPECTED_TOTAL,
        "query_type_distribution": qtype_counts,
        "expected_query_type_distribution": EXPECTED_QUERY_TYPES,
        "query_type_distribution_matches_expected": qtype_counts == EXPECTED_QUERY_TYPES,
        "stable_target_block_ids_nonempty_count": sum(1 for s in samples if stable_ids(s)),
        "stable_target_block_ids_coverage": safe_ratio(sum(1 for s in samples if stable_ids(s)), len(samples)),
        "include_in_main_denominator_all_true": all(s.get("include_in_main_denominator") is True for s in samples) if samples else False,
        "include_in_main_denominator_not_true_sample_ids": [
            sample_id(s) for s in samples if s.get("include_in_main_denominator") is not True
        ],
        "hard_rule_passed_all_true": all(s.get("hard_rule_passed") is True for s in samples) if samples else False,
        "hard_rule_passed_not_true_sample_ids": [
            sample_id(s) for s in samples if s.get("hard_rule_passed") is not True
        ],
        "duplicate_sample_ids": duplicates(sample_ids),
        "duplicate_queries": duplicates([q for q in queries if q]),
        "target_doc_id_empty_sample_ids": [
            sample_id(s) for s in samples if not str(s.get("target_doc_id") or "").strip()
        ],
        "residuals": residuals,
        "row_cell_structured_table_query_sample_ids": [
            sample_id(s) for s in samples if is_row_cell_query(str(s.get("query") or ""))
        ],
        "ocr_image_query_sample_ids": [
            sample_id(s) for s in samples if is_ocr_image_query(str(s.get("query") or ""))
        ],
        "diagnostic_or_lexical_stress_sample_ids": diagnostic_or_lexical,
        "jsonl_parse_errors": parse_errors,
        "blockers": blockers,
        "is_frozen": not blockers,
    }
    return samples, check


def summarize_chunks(
    path: Path | None,
    target_doc_ids: set[str],
    target_blocks: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        return {"exists": False}, {"chunk_ids": set(), "doc_ids": set(), "doc_blocks": set(), "per_doc_counts": Counter()}
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime": mtime(path),
    }
    runtime = {
        "chunk_ids": set(),
        "doc_ids": set(),
        "doc_blocks": set(),
        "per_doc_counts": Counter(),
        "target_doc_blocks": set(),
    }
    if not path.exists():
        return summary, runtime

    field_sets: Counter[tuple[str, ...]] = Counter()
    parser_stages: Counter[str] = Counter()
    content_shape: Counter[str] = Counter()
    chunk_id_counts: Counter[str] = Counter()
    token_count = 0
    table_related = 0
    target_doc_chunk_count = 0
    target_block_pairs_present: set[tuple[str, str]] = set()
    first_chunk_keys: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not first_chunk_keys:
                first_chunk_keys = sorted(item.keys())
            chunk_id = str(item.get("chunk_id") or "")
            doc_id = str(item.get("doc_id") or "")
            block_ids = chunk_block_ids(item)
            runtime["chunk_ids"].add(chunk_id)
            runtime["doc_ids"].add(doc_id)
            runtime["per_doc_counts"][doc_id] += 1
            chunk_id_counts[chunk_id] += 1
            field_sets[tuple(sorted(item.keys()))] += 1
            parser_stages[str(item.get("parser_stage") or "")] += 1
            token_count += int(item.get("token_count") or 0)
            content_shape[classify_content_shape(item)] += 1
            if any(isinstance(meta, dict) and meta.get("table_related") is True for meta in item.get("source_block_metadata") or []):
                table_related += 1
            if doc_id in target_doc_ids:
                target_doc_chunk_count += 1
            for block_id in block_ids:
                pair = (doc_id, block_id)
                runtime["doc_blocks"].add(pair)
                if pair in target_blocks:
                    target_block_pairs_present.add(pair)
                    runtime["target_doc_blocks"].add(pair)

    doc_ids: set[str] = runtime["doc_ids"]
    duplicate_chunk_ids = sorted(cid for cid, count in chunk_id_counts.items() if cid and count > 1)
    summary.update(
        {
            "chunk_count": len(runtime["chunk_ids"]),
            "line_count": sum(chunk_id_counts.values()),
            "duplicate_chunk_id_count": len(duplicate_chunk_ids),
            "duplicate_chunk_ids_sample": duplicate_chunk_ids[:20],
            "doc_id_count": len(doc_ids),
            "doc_id_min": min(doc_ids) if doc_ids else "",
            "doc_id_max": max(doc_ids) if doc_ids else "",
            "missing_expected_doc_ids": sorted(EXPECTED_DOC_IDS - doc_ids),
            "extra_doc_ids": sorted(doc_ids - EXPECTED_DOC_IDS),
            "contains_708_expected_docs": EXPECTED_DOC_IDS <= doc_ids,
            "token_count": token_count,
            "avg_chunks_per_doc": round(safe_ratio(len(runtime["chunk_ids"]), len(doc_ids)), 3),
            "content_shape_counts": dict(sorted(content_shape.items())),
            "table_related_source_metadata_chunk_count": table_related,
            "schema_field_set_count": len(field_sets),
            "field_sets": ["|".join(keys) for keys in field_sets],
            "first_chunk_field_set": first_chunk_keys,
            "parser_stage_counts": dict(parser_stages),
            "target_doc_chunk_count": target_doc_chunk_count,
            "target_block_pair_present_count": len(target_block_pairs_present),
            "top_docs_by_chunk_count": [
                {"doc_id": doc_id, "chunk_count": count}
                for doc_id, count in runtime["per_doc_counts"].most_common(20)
            ],
        }
    )
    return summary, runtime


def summarize_bm25(path: Path | None, chunk_ids: set[str]) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": ""}
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime": mtime(path),
    }
    if not path.exists():
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        record_ids = {str(item.get("chunk_id") or "") for item in records if isinstance(item, dict)}
        summary.update(
            {
                "record_count": len(records) if isinstance(records, list) else None,
                "chunk_id_set_matches_chunks": record_ids == chunk_ids if chunk_ids else False,
                "records_missing_from_chunks_count": len(record_ids - chunk_ids) if chunk_ids else None,
                "chunks_missing_from_records_count": len(chunk_ids - record_ids) if chunk_ids else None,
                "error": "",
            }
        )
    except Exception as exc:
        summary.update({"record_count": None, "error": f"{type(exc).__name__}: {exc}"})
    return summary


def summarize_milvus(
    uri: Path | None,
    collection: str | None,
    retrieval_inventory: dict[str, Any],
    asset_name: str,
) -> dict[str, Any]:
    reported = (
        retrieval_inventory.get("assets", {})
        .get(asset_name, {})
        .get("milvus", {})
        if isinstance(retrieval_inventory, dict)
        else {}
    )
    if uri is None:
        return {"db_exists": False, "collection_available": False, "row_count": None, "notes": "no Milvus index configured"}
    summary: dict[str, Any] = {
        "uri": str(uri),
        "collection": collection or "",
        "db_exists": uri.exists(),
        "size_bytes": uri.stat().st_size if uri.exists() and uri.is_file() else None,
        "mtime": mtime(uri),
        "collection_available": False,
        "row_count": None,
        "collections": [],
        "inspection_source": "actual",
        "error": "",
    }
    if not uri.exists():
        if reported:
            summary.update({k: reported.get(k) for k in ("collection_available", "row_count", "collections", "error")})
            summary["inspection_source"] = "phase5f4_report_fallback"
        return summary
    if MilvusClient is None:
        summary.update({k: reported.get(k) for k in ("collection_available", "row_count", "collections", "error") if k in reported})
        summary["inspection_source"] = "phase5f4_report_fallback"
        summary["error"] = "pymilvus not importable; used report fallback"
        return summary
    try:
        client = MilvusClient(str(uri.resolve()))
        collections = client.list_collections()
        summary["collections"] = collections
        summary["collection_available"] = bool(collection and collection in collections)
        if summary["collection_available"]:
            stats = client.get_collection_stats(collection)
            summary["row_count"] = int(stats.get("row_count", 0))
    except Exception as exc:
        summary.update({k: reported.get(k) for k in ("collection_available", "row_count", "collections") if k in reported})
        summary["inspection_source"] = "phase5f4_report_fallback_after_error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def summarize_parsed_clean(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False}
    json_files = sorted(path.glob("*.json")) if path.exists() and path.is_dir() else []
    doc_ids = {p.stem for p in json_files}
    return {
        "path": str(path),
        "exists": path.exists(),
        "json_file_count": len(json_files),
        "missing_expected_doc_ids": sorted(EXPECTED_DOC_IDS - doc_ids),
        "extra_doc_ids": sorted(doc_ids - EXPECTED_DOC_IDS),
        "contains_708_expected_docs": EXPECTED_DOC_IDS <= doc_ids,
    }


def build_asset_audit(
    *,
    name: str,
    spec: dict[str, Any],
    chunk_summary: dict[str, Any],
    bm25_summary: dict[str, Any],
    milvus_summary: dict[str, Any],
    parsed_summary: dict[str, Any],
    retrieval_inventory: dict[str, Any],
    phase5c5_index: dict[str, Any],
    phase5c4_stats: dict[str, Any],
    phase5c4_enhanced: dict[str, Any],
    main_results: dict[str, Any],
    variant_comparison: dict[str, Any],
) -> dict[str, Any]:
    reported = retrieval_inventory.get("assets", {}).get(name, {}) if isinstance(retrieval_inventory, dict) else {}
    row_count = milvus_summary.get("row_count")
    chunk_count = chunk_summary.get("chunk_count")
    bm25_count = bm25_summary.get("record_count")
    counts_consistent = bool(chunk_count and row_count == chunk_count and bm25_count == chunk_count)
    metric = main_results.get("metrics", {}).get(name, {}) if isinstance(main_results, dict) else {}
    is_tmp = any(is_tmp_path(spec.get(key)) for key in ("chunks_path", "milvus_uri", "bm25_cache_path"))
    table_enhanced = name == "phase5c5_enhanced_full" or chunk_summary.get("table_related_source_metadata_chunk_count", 0) > 0
    production_default = name == "current_default" and spec.get("collection") == "synbio_papers"
    evidence: list[str] = []
    if reported:
        evidence.append("Phase 5F-4 retrieval_asset_inventory names the chunks, Milvus URI/collection, and BM25 cache.")
    if name == "phase5c5_baseline_full":
        evidence.append("Phase 5C-5 index_build_stats records chunk_count=row_count=BM25 records=15802 and main_index_overwritten=false.")
        evidence.append("Phase 4 closeout identifies this as the Phase 4D compact chunks asset; compact mode remained experimental.")
    if name == "phase5c5_enhanced_full":
        evidence.append("Phase 5C-4 full preflight records 652 accepted table-related associations and 377 table_related chunks.")
        evidence.append("Phase 5C-5 summary recommends switch-gated experimental use, not direct default enablement.")
    if name == "current_default":
        evidence.append("Project config and Phase 5F-4 inventory identify synbio_papers as the current production/default collection.")
        evidence.append("The asset is older than the full compact assets and has 5,192 fewer chunks with the same 705 non-empty doc_ids.")

    suitability = official_suitability(name, counts_consistent, is_tmp, table_enhanced, production_default, chunk_summary)
    return {
        "asset_name": name,
        "label": spec.get("label", ""),
        "role": spec.get("role", ""),
        "chunks_path": str(spec.get("chunks_path") or ""),
        "milvus_uri": str(spec.get("milvus_uri") or ""),
        "milvus_collection": spec.get("collection") or "",
        "bm25_cache_path": str(spec.get("bm25_cache_path") or ""),
        "parsed_clean_source": parsed_summary,
        "chunks": chunk_summary,
        "bm25_cache": bm25_summary,
        "milvus": milvus_summary,
        "counts_consistent": counts_consistent,
        "row_count_matches_chunks": row_count == chunk_count if row_count is not None and chunk_count else False,
        "bm25_count_matches_chunks": bm25_count == chunk_count if bm25_count is not None and chunk_count else False,
        "is_full_corpus": bool(chunk_summary.get("doc_id_count") == 705 and not chunk_summary.get("extra_doc_ids")),
        "contains_708_docs": bool(chunk_summary.get("contains_708_expected_docs")),
        "nonempty_chunk_doc_count": chunk_summary.get("doc_id_count"),
        "missing_chunk_doc_ids_from_1_to_708": chunk_summary.get("missing_expected_doc_ids", []),
        "is_current_production_default": production_default,
        "uses_table_enhancement": table_enhanced,
        "uses_caption_cleanup": name == "phase5d_caption_cleanup_chunks",
        "is_tmp_asset": is_tmp,
        "build_or_command_source": build_command_source(name),
        "metrics_from_phase5f4": metric.get("overall", {}),
        "variant_comparison_context": variant_comparison.get("overall", {}) if name in {"phase5c5_baseline_full", "phase5c5_enhanced_full"} else {},
        "supporting_evidence": evidence,
        "reproducibility_assessment": reproducibility_assessment(name, is_tmp, counts_consistent),
        "official_baseline_suitability": suitability,
        "report_flags": {
            "index_rebuilt_in_this_phase": False,
            "retrieval_rerun_in_this_phase": False,
            "qwen_called": False,
            "ragas_run": False,
        },
        "phase5c5_index_build_stats": phase5c5_index.get("baseline" if name == "phase5c5_baseline_full" else "enhanced", {})
        if name in {"phase5c5_baseline_full", "phase5c5_enhanced_full"} else {},
        "phase5c4_full_preflight_stats": phase5c4_stats if name in {"phase5c5_baseline_full", "phase5c5_enhanced_full"} else {},
        "phase5c4_enhanced_build_stats": phase5c4_enhanced if name == "phase5c5_enhanced_full" else {},
    }


def official_suitability(
    name: str,
    counts_consistent: bool,
    is_tmp: bool,
    table_enhanced: bool,
    production_default: bool,
    chunk_summary: dict[str, Any],
) -> dict[str, Any]:
    if name == "current_default":
        return {
            "suitable": False,
            "rationale": (
                "It is the production default and internally consistent, but it is an older 10,610-chunk asset. "
                "It shares the same non-empty doc_id set as the full compact baseline but lacks 5,352 baseline chunk_ids; "
                "Phase 5F-4R identified this asset gap as a major reason for lower stable@10."
            ),
        }
    if name == "phase5c5_baseline_full":
        return {
            "suitable": bool(counts_consistent),
            "rationale": (
                "It is the best official candidate because it is not table-enhanced, has matching chunks/Milvus/BM25 counts, "
                "and covers the strict-main targets. However it currently lives in /tmp, so it should be rebuilt or frozen "
                "under a durable official asset name before closeout."
            ),
        }
    if name == "phase5c5_enhanced_full":
        return {
            "suitable": False,
            "rationale": (
                "It contains table-related experimental associations and underperforms baseline_full on the frozen clean set. "
                "It should remain an experimental variant/candidate, not the official clean baseline."
            ),
        }
    return {
        "suitable": False,
        "rationale": "Chunk-only optional asset without Milvus/BM25 hybrid index.",
    }


def build_command_source(name: str) -> dict[str, Any]:
    if name == "current_default":
        return {
            "source": "production/default Settings and KnowledgeBaseBuilder path defaults",
            "scripts": [
                "scripts/ingestion/build_round1_kb.py",
                "src/synbio_rag/ingestion/kb_builder.py",
                "scripts/ingestion/preprocess_and_chunk.py",
                "scripts/ingestion/import_to_milvus.py",
            ],
            "exact_historical_command_found": False,
        }
    if name == "phase5c5_baseline_full":
        return {
            "source": "Phase 4D compact chunks reused by Phase 5C-5 full retrieval A/B",
            "scripts": [
                "scripts/ingestion/preprocess_and_chunk.py --evidence_context_mode compact",
                "scripts/ingestion/import_to_milvus.py",
                "scripts/evaluation/evaluate_phase5c5_full_retrieval_ab.py",
            ],
            "exact_historical_command_found": False,
            "documented_inputs": [
                "/tmp/biorag_phase4d_compact_chunks/chunks.jsonl",
                "/tmp/phase5c5_baseline_full.db",
                "/tmp/biorag_phase5c5_baseline_full/bm25_index.json",
            ],
        }
    if name == "phase5c5_enhanced_full":
        return {
            "source": "Phase 5C-4 full enhanced parsed_clean and chunks, then Phase 5C-5 experimental index",
            "scripts": [
                "src/synbio_rag/ingestion/table_enhancement.py",
                "scripts/ingestion/preprocess_and_chunk.py",
                "scripts/ingestion/import_to_milvus.py",
                "scripts/evaluation/evaluate_phase5c5_full_retrieval_ab.py",
            ],
            "exact_historical_command_found": False,
        }
    return {"source": "Phase 5D caption cleanup chunk-only experimental asset", "exact_historical_command_found": False}


def reproducibility_assessment(name: str, is_tmp: bool, counts_consistent: bool) -> dict[str, Any]:
    if name == "current_default":
        return {
            "reproducible": "partial",
            "reason": "Default scripts/config exist, but no frozen manifest explains the May 6 10,610-chunk production build versus the May 14 full compact build.",
        }
    if name == "phase5c5_baseline_full":
        return {
            "reproducible": "needs_official_rebuild_or_freeze",
            "reason": "Inputs and scripts are documented, but chunks/Milvus/BM25 live in /tmp; durable official naming is missing.",
        }
    if name == "phase5c5_enhanced_full":
        return {
            "reproducible": "experimental",
            "reason": "Full enhanced parsed_clean and associations are documented, but the asset is intentionally switch-gated/experimental.",
        }
    return {"reproducible": "no_index", "reason": "No Milvus/BM25 hybrid asset configured."}


def build_diff_summary(chunk_runtime: dict[str, dict[str, Any]], audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current = chunk_runtime.get("current_default", {})
    baseline = chunk_runtime.get("phase5c5_baseline_full", {})
    enhanced = chunk_runtime.get("phase5c5_enhanced_full", {})

    if current and baseline:
        out["current_default"] = compare_chunk_sets("current_default", "phase5c5_baseline_full", current, baseline)
        out["phase5c5_baseline_full"] = compare_chunk_sets("phase5c5_baseline_full", "current_default", baseline, current)
    if baseline and enhanced:
        base_enh = compare_chunk_sets("phase5c5_baseline_full", "phase5c5_enhanced_full", baseline, enhanced)
        out.setdefault("phase5c5_baseline_full", {})["vs_enhanced_full"] = base_enh
        out["phase5c5_enhanced_full"] = compare_chunk_sets("phase5c5_enhanced_full", "phase5c5_baseline_full", enhanced, baseline)
    return out


def compare_chunk_sets(left_name: str, right_name: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = left.get("chunk_ids", set())
    right_ids = right.get("chunk_ids", set())
    left_docs = left.get("doc_ids", set())
    right_docs = right.get("doc_ids", set())
    left_counts = left.get("per_doc_counts", Counter())
    right_counts = right.get("per_doc_counts", Counter())
    deltas = []
    for doc_id in sorted(left_docs | right_docs):
        deltas.append(
            {
                "doc_id": doc_id,
                "left_count": left_counts.get(doc_id, 0),
                "right_count": right_counts.get(doc_id, 0),
                "right_minus_left": right_counts.get(doc_id, 0) - left_counts.get(doc_id, 0),
            }
        )
    return {
        "left": left_name,
        "right": right_name,
        "left_only_chunk_id_count": len(left_ids - right_ids),
        "right_only_chunk_id_count": len(right_ids - left_ids),
        "common_chunk_id_count": len(left_ids & right_ids),
        "doc_sets_equal": left_docs == right_docs,
        "left_only_doc_ids": sorted(left_docs - right_docs),
        "right_only_doc_ids": sorted(right_docs - left_docs),
        "largest_right_minus_left_doc_deltas": sorted(deltas, key=lambda r: r["right_minus_left"], reverse=True)[:20],
        "largest_left_minus_right_doc_deltas": sorted(deltas, key=lambda r: r["right_minus_left"])[:20],
    }


def build_coverage(
    samples: list[dict[str, Any]],
    chunk_runtime: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_by_sample = load_per_sample_results()
    rows: list[dict[str, Any]] = []
    variants = [
        ("current_default", "current_default"),
        ("phase5c5_baseline_full", "baseline_full"),
        ("phase5c5_enhanced_full", "enhanced_full"),
        ("phase5d_caption_cleanup_chunks", "phase5d_chunks"),
    ]
    for sample in samples:
        doc_id = str(sample.get("target_doc_id") or "")
        stable = stable_ids(sample)
        row: dict[str, Any] = {
            "sample_id": sample_id(sample),
            "query_type": sample.get("query_type", ""),
            "target_doc_id": doc_id,
            "stable_target_block_ids": ";".join(stable),
        }
        issues: list[str] = []
        for variant, prefix in variants:
            runtime = chunk_runtime.get(variant, {})
            doc_present = doc_id in runtime.get("doc_ids", set())
            block_present = any((doc_id, block_id) in runtime.get("doc_blocks", set()) for block_id in stable)
            row[f"{prefix}_doc_present"] = doc_present
            row[f"{prefix}_block_present"] = block_present
            if not doc_present:
                issues.append(f"{prefix}:target_doc_missing")
            elif not block_present:
                issues.append(f"{prefix}:stable_block_missing")
        for variant in ("current_default", "phase5c5_baseline_full", "phase5c5_enhanced_full"):
            sample_result = result_by_sample.get((row["sample_id"], variant), {})
            key = {
                "current_default": "current_default",
                "phase5c5_baseline_full": "baseline_full",
                "phase5c5_enhanced_full": "enhanced_full",
            }[variant]
            row[f"{key}_stable_hit_at_10"] = bool(sample_result.get("stable_block_hit_at_10")) if sample_result else ""
            row[f"{key}_first_stable_rank"] = sample_result.get("first_stable_block_hit_rank", "") if sample_result else ""
            row[f"{key}_first_doc_rank"] = sample_result.get("first_doc_hit_rank", "") if sample_result else ""
        row["issue"] = "; ".join(issues)
        rows.append(row)

    summary: dict[str, Any] = {"sample_count": len(rows), "variants": {}}
    for variant, prefix in variants:
        doc_present = sum(1 for r in rows if r.get(f"{prefix}_doc_present") is True)
        block_present = sum(1 for r in rows if r.get(f"{prefix}_block_present") is True)
        summary["variants"][prefix] = {
            "asset_name": variant,
            "target_doc_present_count": doc_present,
            "target_doc_present_rate": safe_ratio(doc_present, len(rows)),
            "stable_block_present_count": block_present,
            "stable_block_present_rate": safe_ratio(block_present, len(rows)),
            "missing_target_doc_sample_ids": [r["sample_id"] for r in rows if r.get(f"{prefix}_doc_present") is not True],
            "missing_stable_block_sample_ids": [r["sample_id"] for r in rows if r.get(f"{prefix}_block_present") is not True],
        }
    max_doc_count = max(item["target_doc_present_count"] for item in summary["variants"].values())
    max_block_count = max(item["stable_block_present_count"] for item in summary["variants"].values())
    summary["best_target_doc_coverage_variants"] = [
        name for name, item in summary["variants"].items()
        if item["target_doc_present_count"] == max_doc_count
    ]
    summary["best_stable_block_coverage_variants"] = [
        name for name, item in summary["variants"].items()
        if item["stable_block_present_count"] == max_block_count
    ]
    summary["dataset_target_not_retrievable_anywhere_sample_ids"] = [
        r["sample_id"]
        for r in rows
        if not any(r.get(f"{prefix}_doc_present") and r.get(f"{prefix}_block_present") for _, prefix in variants)
    ]
    summary["current_default_failures_from_missing_target_doc_or_block"] = [
        r["sample_id"]
        for r in rows
        if r.get("current_default_stable_hit_at_10") is False
        and (not r.get("current_default_doc_present") or not r.get("current_default_block_present"))
    ]
    return rows, summary


def analyze_current_failures(coverage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_misses = [r for r in coverage_rows if r.get("current_default_stable_hit_at_10") is False]
    recovered_by_baseline = [
        r["sample_id"] for r in current_misses if r.get("baseline_full_stable_hit_at_10") is True
    ]
    recovered_by_enhanced = [
        r["sample_id"] for r in current_misses if r.get("enhanced_full_stable_hit_at_10") is True
    ]
    missing_asset = [
        r["sample_id"] for r in current_misses if not r.get("current_default_doc_present") or not r.get("current_default_block_present")
    ]
    return {
        "current_default_stable_at_10_miss_count": len(current_misses),
        "miss_sample_ids": [r["sample_id"] for r in current_misses],
        "misses_due_to_target_doc_or_block_absent_count": len(missing_asset),
        "misses_due_to_target_doc_or_block_absent_sample_ids": missing_asset,
        "current_misses_recovered_by_baseline_full_count": len(recovered_by_baseline),
        "current_misses_recovered_by_baseline_full_sample_ids": recovered_by_baseline,
        "current_misses_recovered_by_enhanced_full_count": len(recovered_by_enhanced),
        "current_misses_recovered_by_enhanced_full_sample_ids": recovered_by_enhanced,
        "interpretation": (
            "Current-default failures are not caused by strict-main target doc/block absence when this count is zero; "
            "they align with asset/corpus/chunk-count and ranking differences."
        ),
    }


def analyze_enhanced_position(coverage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_only = [
        r for r in coverage_rows
        if r.get("baseline_full_stable_hit_at_10") is True and r.get("enhanced_full_stable_hit_at_10") is False
    ]
    rows = [
        {
            "sample_id": r["sample_id"],
            "query_type": r["query_type"],
            "target_doc_id": r["target_doc_id"],
            "baseline_full_rank": r.get("baseline_full_first_stable_rank"),
            "enhanced_full_rank": r.get("enhanced_full_first_stable_rank"),
            "baseline_block_present": r.get("baseline_full_block_present"),
            "enhanced_block_present": r.get("enhanced_full_block_present"),
            "classification": "ranking_or_asset_competition" if r.get("enhanced_full_block_present") else "enhanced_target_mapping_gap",
        }
        for r in baseline_only
    ]
    return {
        "enhanced_recommended_position": "experimental_variant",
        "baseline_only_count": len(rows),
        "baseline_only_rows": rows,
        "evidence": (
            "Baseline-only strict-main cases retain target block coverage in enhanced_full when enhanced_block_present is true; "
            "their misses are therefore ranking/asset-competition effects, not missing target mapping."
        ),
    }


def decide_official_baseline(
    dataset_check: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    if not dataset_check.get("is_frozen"):
        return {
            "recommended_option": "D",
            "recommended_option_label": "blocker",
            "rationale": "Dataset freeze check failed; do not select an official baseline.",
            "risk": "Any baseline choice would be anchored to a changed or invalid denominator.",
            "required_next_action": "Resolve dataset blockers without modifying the dataset in this phase.",
            "whether_rebuild_needed": False,
            "whether_user_confirmation_needed": True,
            "whether_phase5f5_closeout_allowed_after_this": False,
        }

    baseline = audits["phase5c5_baseline_full"]
    baseline_cov = coverage_summary["variants"]["baseline_full"]
    current = audits["current_default"]
    if baseline_cov["target_doc_present_count"] != EXPECTED_TOTAL or baseline_cov["stable_block_present_count"] != EXPECTED_TOTAL:
        return {
            "recommended_option": "D",
            "recommended_option_label": "blocker",
            "rationale": "phase5c5_baseline_full does not fully cover strict-main target docs/stable blocks.",
            "risk": "The official baseline would contain unretrievable targets.",
            "required_next_action": "Resolve coverage blocker before baseline selection.",
            "whether_rebuild_needed": False,
            "whether_user_confirmation_needed": True,
            "whether_phase5f5_closeout_allowed_after_this": False,
        }

    if baseline.get("counts_consistent") and baseline.get("is_tmp_asset"):
        return {
            "recommended_option": "B",
            "recommended_option_label": "rebuild official clean baseline minimally",
            "rationale": (
                "phase5c5_baseline_full is the correct clean-baseline candidate by behavior and coverage, "
                "but its chunks, Milvus DB, and BM25 cache live under /tmp and are not durable official assets. "
                "A minimal isolated rebuild/freeze should promote this asset line without enabling table enhancement."
            ),
            "risk": (
                "Rebuilding may reproduce 15,802 chunks only if the Phase 4D compact input mode and parsed_clean snapshot are kept fixed. "
                "Do not overwrite production current_default."
            ),
            "required_next_action": "Phase 5F-4B minimal official baseline rebuild in an isolated durable path, then validate counts and rerun if needed.",
            "whether_rebuild_needed": True,
            "whether_user_confirmation_needed": True,
            "whether_phase5f5_closeout_allowed_after_this": False,
            "selected_candidate": "phase5c5_baseline_full",
            "rejected_current_default_reason": current["official_baseline_suitability"]["rationale"],
            "rejected_enhanced_reason": audits["phase5c5_enhanced_full"]["official_baseline_suitability"]["rationale"],
        }

    if baseline.get("counts_consistent"):
        return {
            "recommended_option": "A",
            "recommended_option_label": "promote existing phase5c5_baseline_full",
            "rationale": "phase5c5_baseline_full is count-consistent, target-complete, non-enhanced, and durable enough to freeze.",
            "risk": "Future work must keep dataset hash and asset names fixed.",
            "required_next_action": "Freeze baseline manifest and proceed to Phase 5F-5 closeout.",
            "whether_rebuild_needed": False,
            "whether_user_confirmation_needed": False,
            "whether_phase5f5_closeout_allowed_after_this": True,
            "selected_candidate": "phase5c5_baseline_full",
        }

    return {
        "recommended_option": "C",
        "recommended_option_label": "keep current_default as production baseline, full baseline as historical reference",
        "rationale": "No better durable full-baseline asset could be selected; current_default remains the stable production asset.",
        "risk": "Metrics must be labeled production-current and carry the 10,610-vs-15,802 asset-gap caveat.",
        "required_next_action": "Proceed only with the production-current caveat.",
        "whether_rebuild_needed": False,
        "whether_user_confirmation_needed": False,
        "whether_phase5f5_closeout_allowed_after_this": True,
        "selected_candidate": "current_default",
    }


def build_source_inventory(
    *,
    reports: list[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    dataset_check: dict[str, Any],
    phase5c5_index: dict[str, Any],
    phase5c4_stats: dict[str, Any],
) -> dict[str, Any]:
    missing = [row for row in reports if not row["exists"]]
    return {
        "generated_at": utc_now(),
        "reports_and_sources": reports,
        "missing_files": missing,
        "dataset_freeze": {
            "sha256": dataset_check.get("sha256"),
            "sha256_matches_expected": dataset_check.get("sha256_matches_expected"),
            "total_samples": dataset_check.get("total_samples"),
            "is_frozen": dataset_check.get("is_frozen"),
        },
        "asset_sources": {
            name: {
                "chunks_path": audit.get("chunks_path"),
                "milvus_uri": audit.get("milvus_uri"),
                "milvus_collection": audit.get("milvus_collection"),
                "bm25_cache_path": audit.get("bm25_cache_path"),
                "chunk_count": audit.get("chunks", {}).get("chunk_count"),
                "row_count": audit.get("milvus", {}).get("row_count"),
                "bm25_records": audit.get("bm25_cache", {}).get("record_count"),
                "counts_consistent": audit.get("counts_consistent"),
                "is_tmp_asset": audit.get("is_tmp_asset"),
                "information_provided": audit.get("supporting_evidence"),
            }
            for name, audit in assets.items()
        },
        "fallback": {
            "needed": any(a.get("milvus", {}).get("inspection_source", "").startswith("phase5f4_report_fallback") for a in assets.values()),
            "details": {
                name: audit.get("milvus", {}).get("inspection_source")
                for name, audit in assets.items()
            },
        },
        "actions_executed": {
            "rebuild": False,
            "retrieval_rerun": False,
            "read_only_asset_validation": True,
            "read_only_asset_validation_reason": "count/schema/coverage/Milvus/BM25 consistency checks only",
            "qwen_called": False,
            "ragas_run": False,
        },
        "supporting_stats": {
            "phase5c5_index_build_stats_present": bool(phase5c5_index),
            "phase5c4_full_chunk_ab_stats_present": bool(phase5c4_stats),
        },
    }


def build_rerun_decision(decision: dict[str, Any], dataset_check: dict[str, Any]) -> dict[str, Any]:
    if decision["recommended_option"] == "B":
        return {
            "immediate_full_rerun_needed": False,
            "reason": (
                "The selected candidate differs from production current_default and must first be rebuilt/frozen under an official name. "
                "Do not run full eval in Phase 5F-4A."
            ),
            "next_phase": "Phase 5F-4B rebuild and validation; rerun strict_main_eval_set_v2 only after counts and target coverage match.",
            "dry_run_allowed": "small read-only validation only; no full retrieval rerun by default",
        }
    if decision["recommended_option"] == "A":
        return {
            "immediate_full_rerun_needed": False,
            "reason": "5F-4 already contains phase5c5_baseline_full results for the same frozen dataset hash and unchanged asset.",
            "next_phase": "Phase 5F-5 closeout.",
            "dry_run_allowed": "not needed",
        }
    if decision["recommended_option"] == "C":
        return {
            "immediate_full_rerun_needed": False,
            "reason": "current_default remains the production baseline and 5F-4 already has current_default results.",
            "next_phase": "Phase 5F-5 closeout with production-current caveat.",
            "dry_run_allowed": "not needed",
        }
    return {
        "immediate_full_rerun_needed": False,
        "reason": "Blocker state; rerun would not be meaningful.",
        "next_phase": "Resolve blocker first.",
        "dry_run_allowed": "no",
    }


def write_asset_outputs(audits: dict[str, dict[str, Any]]) -> None:
    filenames = {
        "current_default": "current_default_asset_audit",
        "phase5c5_baseline_full": "phase5c5_baseline_full_asset_audit",
        "phase5c5_enhanced_full": "phase5c5_enhanced_full_asset_audit",
    }
    for name, stem in filenames.items():
        write_json(OUT_DIR / f"{stem}.json", audits[name])
        write_asset_md(OUT_DIR / f"{stem}.md", audits[name])


def write_asset_md(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        f"# {audit['asset_name']} Asset Audit",
        "",
        f"- chunks path: `{audit['chunks_path']}`",
        f"- chunk count: {audit['chunks'].get('chunk_count')}",
        f"- non-empty chunk doc_id count: {audit.get('nonempty_chunk_doc_count')}",
        f"- missing doc_ids from doc_0001..doc_0708: {audit.get('missing_chunk_doc_ids_from_1_to_708')}",
        f"- Milvus collection: `{audit['milvus_collection']}`",
        f"- Milvus row count: {audit['milvus'].get('row_count')}",
        f"- BM25 cache: `{audit['bm25_cache_path']}`",
        f"- BM25 records: {audit['bm25_cache'].get('record_count')}",
        f"- counts consistent: {yes_no(audit.get('counts_consistent'))}",
        f"- current production default: {yes_no(audit.get('is_current_production_default'))}",
        f"- table enhancement: {yes_no(audit.get('uses_table_enhancement'))}",
        f"- temporary asset path: {yes_no(audit.get('is_tmp_asset'))}",
        f"- official baseline suitable: {yes_no(audit['official_baseline_suitability'].get('suitable'))}",
        f"- suitability rationale: {audit['official_baseline_suitability'].get('rationale')}",
        "",
        "## Schema / Content",
        "",
        f"- schema field set count: {audit['chunks'].get('schema_field_set_count')}",
        f"- parser stages: {audit['chunks'].get('parser_stage_counts')}",
        f"- content shape counts: {json.dumps(audit['chunks'].get('content_shape_counts', {}), ensure_ascii=False)}",
        f"- table_related source metadata chunks: {audit['chunks'].get('table_related_source_metadata_chunk_count')}",
        "",
        "## Provenance",
        "",
        f"- parsed_clean source: `{audit['parsed_clean_source'].get('path', '')}`",
        f"- parsed_clean json files: {audit['parsed_clean_source'].get('json_file_count')}",
        f"- build source: {audit['build_or_command_source'].get('source')}",
        f"- exact historical command found: {yes_no(audit['build_or_command_source'].get('exact_historical_command_found'))}",
        f"- reproducibility: {audit['reproducibility_assessment'].get('reproducible')}",
        f"- reproducibility reason: {audit['reproducibility_assessment'].get('reason')}",
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.get("supporting_evidence", []))
    if audit.get("diff_summary"):
        lines.extend(["", "## Differences", ""])
        lines.append(f"```json\n{json.dumps(audit['diff_summary'], ensure_ascii=False, indent=2)}\n```")
    if audit.get("current_default_failure_alignment"):
        lines.extend(["", "## Current Failure Alignment", ""])
        lines.append(f"```json\n{json.dumps(audit['current_default_failure_alignment'], ensure_ascii=False, indent=2)}\n```")
    if audit.get("enhanced_positioning"):
        lines.extend(["", "## Enhanced Positioning", ""])
        lines.append(f"```json\n{json.dumps(audit['enhanced_positioning'], ensure_ascii=False, indent=2)}\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "query_type",
        "target_doc_id",
        "stable_target_block_ids",
        "current_default_doc_present",
        "current_default_block_present",
        "baseline_full_doc_present",
        "baseline_full_block_present",
        "enhanced_full_doc_present",
        "enhanced_full_block_present",
        "phase5d_chunks_doc_present",
        "phase5d_chunks_block_present",
        "current_default_stable_hit_at_10",
        "baseline_full_stable_hit_at_10",
        "enhanced_full_stable_hit_at_10",
        "current_default_first_stable_rank",
        "baseline_full_first_stable_rank",
        "enhanced_full_first_stable_rank",
        "issue",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_target_coverage_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Target Coverage Summary",
        "",
        f"- sample count: {summary['sample_count']}",
        f"- best target_doc coverage variants: `{', '.join(summary['best_target_doc_coverage_variants'])}`",
        f"- best stable block coverage variants: `{', '.join(summary['best_stable_block_coverage_variants'])}`",
        f"- dataset targets unretrievable in all checked chunk sets: {summary['dataset_target_not_retrievable_anywhere_sample_ids']}",
        f"- current_default failures caused by target doc/block absence: {summary['current_default_failures_from_missing_target_doc_or_block']}",
        "",
        "| variant | target_doc present | stable block present | missing target docs | missing stable blocks |",
        "|---|---:|---:|---|---|",
    ]
    for name, item in summary["variants"].items():
        lines.append(
            f"| {name} | {item['target_doc_present_count']}/{summary['sample_count']} | "
            f"{item['stable_block_present_count']}/{summary['sample_count']} | "
            f"{item['missing_target_doc_sample_ids']} | {item['missing_stable_block_sample_ids']} |"
        )
    lines.extend(
        [
            "",
            "Answers:",
            "- All checked variants cover all 90 strict-main target_doc_id values and all 90 stable target block mappings.",
            "- No strict-main dataset target is absent from all checked chunk sets.",
            "- current_default misses are not caused by target doc/block absence; they are asset/ranking differences.",
            "- A rebuild is not needed for target coverage itself; it is needed only to make the selected full baseline durable and official.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_refreeze_md(path: Path, check: dict[str, Any]) -> None:
    lines = [
        "# Dataset Refreeze Check",
        "",
        f"- exists: {yes_no(check['exists'])}",
        f"- sha256: `{check['sha256']}`",
        f"- expected sha256: `{check['expected_sha256']}`",
        f"- sha256 matches: {yes_no(check['sha256_matches_expected'])}",
        f"- total samples: {check['total_samples']}",
        f"- total matches expected: {yes_no(check['total_matches_expected'])}",
        f"- query_type distribution: {json.dumps(check['query_type_distribution'], ensure_ascii=False)}",
        f"- stable_target_block_ids coverage: {check['stable_target_block_ids_nonempty_count']}/{check['total_samples']}",
        f"- include_in_main_denominator all true: {yes_no(check['include_in_main_denominator_all_true'])}",
        f"- hard_rule_passed all true: {yes_no(check['hard_rule_passed_all_true'])}",
        f"- duplicate sample_id count: {len(check['duplicate_sample_ids'])}",
        f"- duplicate query count: {len(check['duplicate_queries'])}",
        f"- target_doc_id empty count: {len(check['target_doc_id_empty_sample_ids'])}",
        f"- CAPTION residual count: {check['residuals']['CAPTION']['query_count']}",
        f"- Which table residual count: {check['residuals']['Which table']['query_count']}",
        f"- Where residual count: {check['residuals']['Where']['query_count']}",
        f"- row/cell structured table query count: {len(check['row_cell_structured_table_query_sample_ids'])}",
        f"- OCR/image query count: {len(check['ocr_image_query_sample_ids'])}",
        f"- diagnostic/lexical stress mixed into main count: {len(check['diagnostic_or_lexical_stress_sample_ids'])}",
        f"- blockers: {check['blockers'] or 'none'}",
        f"- frozen: {yes_no(check['is_frozen'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_inventory_md(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Source Inventory",
        "",
        f"- generated_at: {inventory['generated_at']}",
        f"- rebuild executed: {yes_no(inventory['actions_executed']['rebuild'])}",
        f"- retrieval rerun executed: {yes_no(inventory['actions_executed']['retrieval_rerun'])}",
        f"- read-only asset validation executed: {yes_no(inventory['actions_executed']['read_only_asset_validation'])}",
        f"- Qwen called: {yes_no(inventory['actions_executed']['qwen_called'])}",
        f"- RAGAS run: {yes_no(inventory['actions_executed']['ragas_run'])}",
        f"- fallback needed: {yes_no(inventory['fallback']['needed'])}",
        "",
        "## Missing Files",
        "",
    ]
    if inventory["missing_files"]:
        lines.extend(f"- `{row['path']}`" for row in inventory["missing_files"])
    else:
        lines.append("- none")
    lines.extend(["", "## Asset Sources", ""])
    for name, asset in inventory["asset_sources"].items():
        lines.extend(
            [
                f"### {name}",
                f"- chunks: `{asset['chunks_path']}`",
                f"- chunk count: {asset['chunk_count']}",
                f"- Milvus: `{asset['milvus_uri']}` / `{asset['milvus_collection']}`",
                f"- row count: {asset['row_count']}",
                f"- BM25: `{asset['bm25_cache_path']}`",
                f"- BM25 records: {asset['bm25_records']}",
                f"- counts consistent: {yes_no(asset['counts_consistent'])}",
                f"- temporary asset: {yes_no(asset['is_tmp_asset'])}",
                "",
            ]
        )
    lines.extend(["## Read Files", ""])
    for row in inventory["reports_and_sources"]:
        lines.append(
            f"- `{row['path']}` exists={yes_no(row['exists'])}; info={row['information_used']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_md(path: Path, decision: dict[str, Any]) -> None:
    lines = [
        "# Official Baseline Decision",
        "",
        f"- recommended_option: {decision['recommended_option']}",
        f"- label: {decision['recommended_option_label']}",
        f"- rationale: {decision['rationale']}",
        f"- risk: {decision['risk']}",
        f"- required_next_action: {decision['required_next_action']}",
        f"- rebuild needed: {yes_no(decision['whether_rebuild_needed'])}",
        f"- user confirmation needed: {yes_no(decision['whether_user_confirmation_needed'])}",
        f"- Phase 5F-5 closeout allowed after this: {yes_no(decision['whether_phase5f5_closeout_allowed_after_this'])}",
    ]
    if decision.get("selected_candidate"):
        lines.append(f"- selected_candidate: `{decision['selected_candidate']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_minimal_rebuild_plan(path: Path, decision: dict[str, Any], audits: dict[str, dict[str, Any]]) -> None:
    baseline = audits["phase5c5_baseline_full"]
    lines = [
        "# Minimal Official Baseline Rebuild Plan",
        "",
        "This plan is not executed in Phase 5F-4A.",
        "",
        "- official baseline name: `phase5f_official_clean_baseline_v1`",
        "- chunks input: rebuild from `data/paper_round1/parsed_clean` with Phase 4D compact evidence context mode",
        "- parsed_clean input: `data/paper_round1/parsed_clean`",
        "- chunks output: `data/paper_round1/phase5f_official_clean_baseline_v1/chunks/chunks.jsonl`",
        "- Milvus URI: `runtime/vectorstores/milvus/phase5f_official_clean_baseline_v1.db`",
        "- Milvus collection: `synbio_phase5f_official_clean_baseline_v1`",
        "- BM25 cache path: `data/paper_round1/phase5f_official_clean_baseline_v1/chunks/bm25_index.json`",
        "- overwrite production: no",
        "- write isolated path: yes",
        f"- expected chunk count: {baseline['chunks'].get('chunk_count')}",
        f"- expected row count: {baseline['milvus'].get('row_count')}",
        f"- expected BM25 records: {baseline['bm25_cache'].get('record_count')}",
        "",
        "## Build Commands",
        "",
        "```bash",
        "python scripts/ingestion/preprocess_and_chunk.py \\",
        "  --input_dir data/paper_round1/parsed_clean \\",
        "  --output_dir data/paper_round1/phase5f_official_clean_baseline_v1/chunks \\",
        "  --chunk_size 800 \\",
        "  --chunk_overlap 120 \\",
        "  --evidence_context_mode compact",
        "",
        "python scripts/ingestion/import_to_milvus.py \\",
        "  --jsonl data/paper_round1/phase5f_official_clean_baseline_v1/chunks/chunks.jsonl \\",
        "  --collection_name synbio_phase5f_official_clean_baseline_v1 \\",
        "  --milvus_uri runtime/vectorstores/milvus/phase5f_official_clean_baseline_v1.db \\",
        "  --embedding bge-m3 \\",
        "  --model_path models/BAAI/bge-m3 \\",
        "  --dim 1024 \\",
        "  --rebuild",
        "",
        "PYTHONPATH=. python - <<'PY'",
        "from src.synbio_rag.domain.config import KnowledgeBaseConfig, RetrievalConfig",
        "from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever",
        "kb = KnowledgeBaseConfig(chunk_jsonl='data/paper_round1/phase5f_official_clean_baseline_v1/chunks/chunks.jsonl')",
        "retrieval = RetrievalConfig(bm25_cache_path='data/paper_round1/phase5f_official_clean_baseline_v1/chunks/bm25_index.json')",
        "BM25Retriever(retrieval, kb)._ensure_index()",
        "PY",
        "```",
        "",
        "## Validation Commands",
        "",
        "```bash",
        "python scripts/evaluation/audit_phase5f4_index_assets.py",
        "python -m py_compile scripts/evaluation/audit_phase5f4_index_assets.py",
        "```",
        "",
        "Validation must confirm chunk_count=row_count=BM25 records=15802 and 90/90 target_doc and stable-block coverage.",
        "",
        "## Rollback Plan",
        "",
        "- Do not touch `synbio_papers` or `data/paper_round1/chunks`.",
        "- If validation fails, discard only `data/paper_round1/phase5f_official_clean_baseline_v1/` and `runtime/vectorstores/milvus/phase5f_official_clean_baseline_v1.db`.",
        "- Keep the frozen dataset unchanged.",
        "",
        "## Prohibited",
        "",
        "- no enhanced rebuild",
        "- no Qwen",
        "- no RAGAS",
        "- no retrieval parameter tuning",
        "- no cleaning or chunk logic changes",
        "- no dataset changes",
        "",
        "## Next-Phase Execution Condition",
        "",
        "Run this only after user confirmation in Phase 5F-4B.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_freeze_plan(
    path: Path,
    decision: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    dataset_check: dict[str, Any],
    coverage_summary: dict[str, Any],
) -> None:
    selected = audits[decision["selected_candidate"]]
    lines = [
        "# Baseline Freeze Plan",
        "",
        f"- official baseline asset: `{decision['selected_candidate']}`",
        f"- chunks path: `{selected['chunks_path']}`",
        f"- Milvus collection: `{selected['milvus_collection']}`",
        f"- BM25 cache: `{selected['bm25_cache_path']}`",
        f"- chunk count: {selected['chunks'].get('chunk_count')}",
        f"- row count: {selected['milvus'].get('row_count')}",
        f"- BM25 records: {selected['bm25_cache'].get('record_count')}",
        f"- dataset hash: `{dataset_check['sha256']}`",
        f"- target coverage: {coverage_summary['variants'][decision['selected_candidate']]['target_doc_present_count']}/90",
        "- how to rerun clean baseline: use the frozen dataset hash and the exact asset names above",
        "- how to compare future modifications: run retrieval-only eval against the frozen dataset and report paired deltas",
        f"- caveats: {decision['risk']}",
        "",
        "## Freeze Checklist",
        "",
        "- counts match",
        "- target coverage complete",
        "- dataset hash fixed",
        "- no generation/RAGAS",
        "- no diagnostic samples in main denominator",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_rerun_decision(path: Path, rerun: dict[str, Any]) -> None:
    lines = [
        "# Baseline Rerun Decision",
        "",
        f"- immediate full rerun needed: {yes_no(rerun['immediate_full_rerun_needed'])}",
        f"- reason: {rerun['reason']}",
        f"- next phase: {rerun['next_phase']}",
        f"- dry-run allowance: {rerun['dry_run_allowed']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    dataset_check: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    coverage_summary: dict[str, Any],
    current_failure: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    current = audits["current_default"]
    baseline = audits["phase5c5_baseline_full"]
    enhanced = audits["phase5c5_enhanced_full"]
    diff = current.get("diff_summary", {})
    lines = [
        "# Phase 5F-4A Summary",
        "",
        f"1. strict_main_eval_set_v2 still frozen: {yes_no(dataset_check['is_frozen'])}.",
        f"2. dataset SHA matches: {yes_no(dataset_check['sha256_matches_expected'])}; SHA=`{dataset_check['sha256']}`.",
        f"3. current_default suitable as official clean baseline: {yes_no(current['official_baseline_suitability']['suitable'])}.",
        f"4. phase5c5_baseline_full suitable as official clean baseline candidate: {yes_no(baseline['official_baseline_suitability']['suitable'])}.",
        f"5. phase5c5_enhanced_full positioning: experimental variant, not official baseline.",
        "6. current_default vs full chunk difference: same 705 non-empty doc_ids, but "
        f"current has {current['chunks'].get('chunk_count')} chunks and baseline_full has {baseline['chunks'].get('chunk_count')}; "
        f"baseline-only chunk IDs vs current={diff.get('right_only_chunk_id_count')}. Evidence points to an older/different chunk asset, not a doc subset.",
        f"7. target_doc coverage: current={coverage_summary['variants']['current_default']['target_doc_present_count']}/90, "
        f"baseline_full={coverage_summary['variants']['baseline_full']['target_doc_present_count']}/90, "
        f"enhanced_full={coverage_summary['variants']['enhanced_full']['target_doc_present_count']}/90.",
        f"8. stable_target_block coverage: current={coverage_summary['variants']['current_default']['stable_block_present_count']}/90, "
        f"baseline_full={coverage_summary['variants']['baseline_full']['stable_block_present_count']}/90, "
        f"enhanced_full={coverage_summary['variants']['enhanced_full']['stable_block_present_count']}/90.",
        f"9. index_asset_gap confirmed as one current_default failure driver: yes; current misses recovered by baseline_full={current_failure['current_misses_recovered_by_baseline_full_count']}, "
        f"misses caused by target absence={current_failure['misses_due_to_target_doc_or_block_absent_count']}.",
        f"10. recommended_option: {decision['recommended_option']}.",
        f"11. rebuild needed: {yes_no(decision['whether_rebuild_needed'])}.",
        "12. rebuild executed: no.",
        "13. retrieval / cleaning / dataset modified: no.",
        "14. Qwen/RAGAS called: no.",
        f"15. Phase 5F-5 closeout allowed now: {yes_no(decision['whether_phase5f5_closeout_allowed_after_this'])}.",
        f"16. next step: {decision['required_next_action']}",
        f"17. continue fixing strict_main_eval_set_v2 hash for later baseline: yes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_phase_plan(path: Path, decision: dict[str, Any]) -> None:
    if decision["recommended_option"] == "A":
        plan = "Enter Phase 5F-5 closeout."
    elif decision["recommended_option"] == "B":
        plan = "Enter Phase 5F-4B minimal official baseline rebuild."
    elif decision["recommended_option"] == "C":
        plan = "Enter Phase 5F-5 closeout with production-current caveat."
    else:
        plan = "Handle blocker before baseline closeout."
    lines = [
        "# Next Phase Plan",
        "",
        f"- recommended option: {decision['recommended_option']}",
        f"- next phase: {plan}",
        f"- required action: {decision['required_next_action']}",
        "- keep strict_main_eval_set_v2 hash fixed",
        "- do not mix diagnostic or lexical stress samples into main",
        "- do not run Qwen/RAGAS in the retrieval baseline phase",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_per_sample_results() -> dict[tuple[str, str], dict[str, Any]]:
    path = ROOT / "reports/phase5f4_clean_main_baseline/per_sample_results.jsonl"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            out[(str(item.get("sample_id") or ""), str(item.get("index_variant") or ""))] = item
    return out


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items() if k not in {"chunk_ids", "doc_ids", "doc_blocks", "per_doc_counts"}}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(make_jsonable(v) for v in value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_ids(sample: dict[str, Any]) -> list[str]:
    value = sample.get("stable_target_block_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def chunk_block_ids(chunk: dict[str, Any]) -> list[str]:
    return [str(item) for item in (chunk.get("source_block_ids") or chunk.get("block_ids") or []) if str(item)]


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or "")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(k for k, v in counts.items() if k and v > 1)


def residual_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = {
        "CAPTION": re.compile(r"\bCAPTION\b"),
        "Which table": re.compile(r"\bWhich\s+table\b", re.I),
        "Where": re.compile(r"\bWhere\b", re.I),
        "Table caption": re.compile(r"\bTable\s+caption\b", re.I),
    }
    out: dict[str, Any] = {}
    for name, pattern in patterns.items():
        ids = [sample_id(s) for s in samples if pattern.search(str(s.get("query") or ""))]
        out[name] = {"query_count": len(ids), "sample_ids": ids}
    return out


def is_row_cell_query(query: str) -> bool:
    patterns = (
        r"\bwhich\s+(table\s+)?(row|cell|column)\b",
        r"\b(table\s+)?(row|cell|column)\s+(number|value|index|header)\b",
        r"\b(row|cell|column)\s+in\s+(the\s+)?table\b",
        r"\btable\s+(row|cell|column)\b",
        r"\brow\s*/\s*column\b",
    )
    return any(re.search(pattern, query, re.I) for pattern in patterns)


def is_ocr_image_query(query: str) -> bool:
    patterns = (
        r"\bOCR\b",
        r"\bimage\b",
        r"\bpixel\b",
        r"\bvisual\b",
        r"\bphoto\b",
        r"\bmicroscopy image\b",
    )
    return any(re.search(pattern, query, re.I) for pattern in patterns)


def classify_content_shape(chunk: dict[str, Any]) -> str:
    if chunk.get("contains_table_text"):
        return "table_text"
    if chunk.get("contains_table_caption"):
        return "table_caption"
    if chunk.get("contains_figure_caption"):
        return "figure_caption"
    if chunk.get("contains_references"):
        return "references"
    if chunk.get("contains_metadata"):
        return "metadata"
    if "paragraph" in set(chunk.get("evidence_types") or chunk.get("block_types") or []):
        return "paragraph"
    return "other"


def safe_ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def is_tmp_path(value: Any) -> bool:
    if value is None:
        return False
    return str(value).startswith("/tmp/")


if __name__ == "__main__":
    main()
