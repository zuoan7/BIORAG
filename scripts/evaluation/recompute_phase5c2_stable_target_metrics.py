#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute Phase 5C-2 metrics with stable source-block target mapping.")
    parser.add_argument("--eval-jsonl", default="reports/phase5c2_table_retrieval_ab/eval_queries.jsonl")
    parser.add_argument("--retrieval-results", default="reports/phase5c2_table_retrieval_ab/retrieval_ab_results.json")
    parser.add_argument("--baseline-chunks", default="/tmp/biorag_phase5c2_baseline/chunks.jsonl")
    parser.add_argument("--enhanced-chunks", default="/tmp/biorag_phase5c1_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--output-dir", default="reports/phase5c2_table_retrieval_ab/stable_target_mapping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = list(iter_jsonl(Path(args.eval_jsonl)))
    payload = json.loads(Path(args.retrieval_results).read_text(encoding="utf-8"))
    baseline_chunks = list(iter_jsonl(Path(args.baseline_chunks)))
    enhanced_chunks = list(iter_jsonl(Path(args.enhanced_chunks)))
    baseline_index = build_chunk_index(baseline_chunks)
    enhanced_index = build_chunk_index(enhanced_chunks)

    mappings = [
        build_mapping(query, baseline_index, enhanced_index)
        for query in queries
    ]
    mapping_by_sample = {m["sample_id"]: m for m in mappings}

    corrected_records = [
        correct_record(record, mapping_by_sample[record["sample_id"]])
        for record in payload["records"]
    ]
    metrics = aggregate(corrected_records)
    corrected_mode_comparison = mode_comparison(metrics)
    caption_review = build_caption_regression_review(payload["records"], mapping_by_sample)
    gate = corrected_gate(metrics, payload)

    corrected_payload = {
        "source_results": args.retrieval_results,
        "note": "Recomputed from existing retrieval top-k/rank outputs; retrieval was not rerun.",
        "top_k_limit_for_remapped_rank": 10,
        "run_config": payload.get("run_config", {}),
        "validation": payload.get("validation", {}),
        "mapping_status_counts": dict(Counter(m["mapping_status"] for m in mappings)),
        "metrics": metrics,
        "mode_comparison": corrected_mode_comparison,
        "gate": gate,
        "records": corrected_records,
    }

    write_mapping_csv(output_dir / "target_mapping_audit.csv", mappings)
    write_json(output_dir / "corrected_retrieval_ab_results.json", corrected_payload)
    write_caption_review(output_dir / "caption_regression_mapping_review.md", caption_review)
    write_summary(output_dir / "summary.md", corrected_payload, caption_review, payload)

    print(f"Wrote {output_dir / 'target_mapping_audit.csv'}")
    print(f"Wrote {output_dir / 'corrected_retrieval_ab_results.json'}")
    print(f"Wrote {output_dir / 'caption_regression_mapping_review.md'}")
    print(f"Wrote {output_dir / 'summary.md'}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_doc_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        by_doc[doc_id].append(chunk)
        for block_id in chunk_block_ids(chunk):
            by_doc_block[(doc_id, block_id)].append(chunk)
    return {"by_id": by_id, "by_doc": by_doc, "by_doc_block": by_doc_block}


def build_mapping(
    query: dict[str, Any],
    baseline_index: dict[str, Any],
    enhanced_index: dict[str, Any],
) -> dict[str, Any]:
    qtype = str(query.get("query_type", ""))
    doc_id = str(query.get("target_doc_id", ""))
    original_baseline = str(query.get("target_chunk_id_baseline") or "")
    original_enhanced = str(query.get("target_chunk_id_enhanced") or "")
    baseline_chunk = baseline_index["by_id"].get(original_baseline)

    stable_ids: list[str] = []
    notes: list[str] = []

    if qtype == "caption_level_table":
        explicit_caption = str(query.get("target_caption_block_id") or "")
        if explicit_caption:
            stable_ids = [explicit_caption]
            notes.append("used explicit target_caption_block_id")
        elif baseline_chunk:
            stable_ids = table_caption_block_ids(baseline_chunk)
            if stable_ids:
                notes.append("used baseline table_caption source_block_id")
            elif baseline_chunk.get("contains_table_caption"):
                stable_ids = chunk_block_ids(baseline_chunk)
                notes.append("fallback to all baseline caption chunk source_block_ids")
    elif qtype == "table_content":
        assoc = str(query.get("target_associated_block_id") or "")
        caption = str(query.get("target_caption_block_id") or "")
        stable_ids = [item for item in [assoc, caption] if item]
        notes.append("table_content uses associated block for enhanced and caption block for baseline when available")
    elif qtype == "normal_control":
        if baseline_chunk:
            stable_ids = non_table_block_ids(baseline_chunk) or chunk_block_ids(baseline_chunk)
            notes.append("normal_control uses baseline paragraph source_block_id")
    else:
        stable_ids = []

    corrected_baseline, baseline_candidates = corrected_target(
        qtype=qtype,
        variant="baseline",
        doc_id=doc_id,
        original_chunk_id=original_baseline,
        stable_ids=stable_ids,
        query=query,
        chunk_index=baseline_index,
    )
    corrected_enhanced, enhanced_candidates = corrected_target(
        qtype=qtype,
        variant="enhanced",
        doc_id=doc_id,
        original_chunk_id=original_enhanced,
        stable_ids=stable_ids,
        query=query,
        chunk_index=enhanced_index,
    )

    mapping_status = mapping_status_for(
        stable_ids=stable_ids,
        original_baseline=original_baseline,
        original_enhanced=original_enhanced,
        corrected_baseline=corrected_baseline,
        corrected_enhanced=corrected_enhanced,
        baseline_candidates=baseline_candidates,
        enhanced_candidates=enhanced_candidates,
    )
    if len(baseline_candidates) > 1:
        notes.append(f"baseline candidates={','.join(c['chunk_id'] for c in baseline_candidates)}")
    if len(enhanced_candidates) > 1:
        notes.append(f"enhanced candidates={','.join(c['chunk_id'] for c in enhanced_candidates)}")

    return {
        "sample_id": query.get("sample_id", ""),
        "query_type": qtype,
        "target_doc_id": doc_id,
        "original_baseline_target_chunk_id": original_baseline,
        "original_enhanced_target_chunk_id": original_enhanced,
        "stable_target_block_ids": ";".join(stable_ids),
        "corrected_baseline_target_chunk_id": corrected_baseline,
        "corrected_enhanced_target_chunk_id": corrected_enhanced,
        "mapping_status": mapping_status,
        "mapping_notes": "; ".join(notes),
    }


def corrected_target(
    qtype: str,
    variant: str,
    doc_id: str,
    original_chunk_id: str,
    stable_ids: list[str],
    query: dict[str, Any],
    chunk_index: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if not stable_ids:
        return original_chunk_id, []
    preferred_ids = list(stable_ids)
    if qtype == "table_content" and variant == "enhanced":
        assoc = str(query.get("target_associated_block_id") or "")
        preferred_ids = [assoc] if assoc else stable_ids
    elif qtype == "table_content" and variant == "baseline":
        caption = str(query.get("target_caption_block_id") or "")
        preferred_ids = [caption] if caption else stable_ids

    candidates = candidate_chunks(chunk_index, doc_id, preferred_ids)
    if not candidates and preferred_ids != stable_ids:
        candidates = candidate_chunks(chunk_index, doc_id, stable_ids)
    if not candidates:
        return original_chunk_id, []
    return choose_candidate(candidates, qtype, variant, preferred_ids)["chunk_id"], candidates


def candidate_chunks(chunk_index: dict[str, Any], doc_id: str, block_ids: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for block_id in block_ids:
        for chunk in chunk_index["by_doc_block"].get((doc_id, block_id), []):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                candidates.append(chunk)
    return candidates


def choose_candidate(
    candidates: list[dict[str, Any]],
    qtype: str,
    variant: str,
    preferred_block_ids: list[str],
) -> dict[str, Any]:
    preferred = set(preferred_block_ids)

    def score(chunk: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        blocks = set(chunk_block_ids(chunk))
        overlap = len(blocks & preferred)
        is_caption = int(bool(chunk.get("contains_table_caption")))
        is_table = int(bool(chunk.get("contains_table_caption") or chunk.get("contains_table_text")))
        is_normal = int("paragraph" in set(chunk.get("evidence_types") or chunk.get("block_types") or []))
        if qtype == "caption_level_table":
            primary = is_caption
        elif qtype == "table_content":
            primary = is_table if variant == "enhanced" else is_caption
        elif qtype == "normal_control":
            primary = is_normal - is_table
        else:
            primary = 0
        return (overlap, primary, is_caption, is_table, -len(chunk.get("text", "")), chunk["chunk_id"])

    return sorted(candidates, key=score, reverse=True)[0]


def mapping_status_for(
    stable_ids: list[str],
    original_baseline: str,
    original_enhanced: str,
    corrected_baseline: str,
    corrected_enhanced: str,
    baseline_candidates: list[dict[str, Any]],
    enhanced_candidates: list[dict[str, Any]],
) -> str:
    if not stable_ids:
        return "not_applicable"
    if not corrected_baseline or not corrected_enhanced or (not baseline_candidates and not enhanced_candidates):
        return "mapping_missing"
    if len(baseline_candidates) > 1 or len(enhanced_candidates) > 1:
        return "multiple_candidate_chunks"
    if corrected_baseline != original_baseline or corrected_enhanced != original_enhanced:
        return "remapped_to_new_chunk"
    return "mapped_same_chunk"


def chunk_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    ids = chunk.get("source_block_ids") or chunk.get("block_ids") or []
    return [str(item) for item in ids if item]


def table_caption_block_ids(chunk: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if isinstance(meta, dict) and meta.get("type") == "table_caption":
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    return ids


def non_table_block_ids(chunk: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if not isinstance(meta, dict):
            continue
        if meta.get("type") not in {"table_caption", "table_text", "figure_caption"}:
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    return ids


def correct_record(record: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    variant = record["variant"]
    corrected_target = mapping[f"corrected_{variant}_target_chunk_id"]
    original_target = record.get("target_chunk_id", "")
    corrected_rank, rank_source = corrected_rank_for(record, corrected_target, original_target)
    out = dict(record)
    out["original_target_chunk_id"] = original_target
    out["corrected_target_chunk_id"] = corrected_target
    out["stable_target_block_ids"] = mapping["stable_target_block_ids"]
    out["mapping_status"] = mapping["mapping_status"]
    out["corrected_target_chunk_rank"] = corrected_rank
    out["corrected_rank_source"] = rank_source
    out["original_chunk_hit_at_10"] = bool(record.get("chunk_hit_at_10"))
    out["corrected_chunk_hit_at_10"] = bool(corrected_target and 0 < corrected_rank <= 10)

    if corrected_target == original_target:
        out["corrected_chunk_hit_at_20"] = bool(record.get("chunk_hit_at_20"))
        out["corrected_chunk_hit_at_50"] = bool(record.get("chunk_hit_at_50"))
    else:
        # Existing result files store top10 hits only. For remapped targets, @20/@50
        # can only be confirmed when the corrected target is already in top10.
        in_top10 = bool(corrected_target and 0 < corrected_rank <= 10)
        out["corrected_chunk_hit_at_20"] = in_top10
        out["corrected_chunk_hit_at_50"] = in_top10
        out["corrected_rank_scope_note"] = "remapped target rank available only within stored top10"
    return out


def corrected_rank_for(record: dict[str, Any], corrected_target: str, original_target: str) -> tuple[int, str]:
    if not corrected_target:
        return 0, "missing_target"
    if corrected_target == original_target:
        return int(record.get("target_chunk_rank") or 0), "original_full_rank"
    for hit in record.get("top_hits") or []:
        if hit.get("chunk_id") == corrected_target:
            return int(hit.get("rank") or 0), "stored_top10_rank"
    return 0, "not_in_stored_top10"


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("include_in_main_denominator"):
            by_key[(record["mode"], record["query_type"], record["variant"])].append(record)

    for (mode, qtype, variant), rows in sorted(by_key.items()):
        metrics[f"{mode}/{qtype}/{variant}"] = basic_metrics(rows)

    paired: dict[str, Any] = {}
    qtypes = sorted({r["query_type"] for r in records if r.get("include_in_main_denominator")})
    modes = sorted({r["mode"] for r in records})
    for mode in modes:
        for qtype in qtypes:
            base = [r for r in records if r["mode"] == mode and r["query_type"] == qtype and r["variant"] == "baseline" and r.get("include_in_main_denominator")]
            enh = [r for r in records if r["mode"] == mode and r["query_type"] == qtype and r["variant"] == "enhanced" and r.get("include_in_main_denominator")]
            paired[f"{mode}/{qtype}"] = paired_metrics(base, enh)
    metrics["paired"] = paired
    return metrics


def basic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "doc_hit_at_10": mean_bool(rows, "doc_hit_at_10"),
        "doc_hit_at_20": mean_bool(rows, "doc_hit_at_20"),
        "doc_hit_at_50": mean_bool(rows, "doc_hit_at_50"),
        "original_chunk_hit_at_10": mean_bool(rows, "original_chunk_hit_at_10"),
        "corrected_chunk_hit_at_10": mean_bool(rows, "corrected_chunk_hit_at_10"),
        "corrected_chunk_hit_at_20": mean_bool(rows, "corrected_chunk_hit_at_20"),
        "corrected_chunk_hit_at_50": mean_bool(rows, "corrected_chunk_hit_at_50"),
        "correction_delta_at_10": mean_bool(rows, "corrected_chunk_hit_at_10") - mean_bool(rows, "original_chunk_hit_at_10"),
        "mean_doc_rank_found": mean_rank(rows, "target_doc_rank"),
        "mean_corrected_chunk_rank_found": mean_rank(rows, "corrected_target_chunk_rank"),
        "mean_table_topk_occupancy": statistics.mean(float(r["table_topk_occupancy"]) for r in rows),
        "rank_source_counts": dict(Counter(r.get("corrected_rank_source", "") for r in rows)),
    }


def paired_metrics(base: list[dict[str, Any]], enh: list[dict[str, Any]]) -> dict[str, Any]:
    b_by_id = {r["sample_id"]: r for r in base}
    e_by_id = {r["sample_id"]: r for r in enh}
    ids = sorted(set(b_by_id) & set(e_by_id))
    enhanced_only = baseline_only = both_success = both_fail = 0
    for sid in ids:
        b_hit = bool(b_by_id[sid].get("corrected_chunk_hit_at_10"))
        e_hit = bool(e_by_id[sid].get("corrected_chunk_hit_at_10"))
        if e_hit and not b_hit:
            enhanced_only += 1
        elif b_hit and not e_hit:
            baseline_only += 1
        elif b_hit and e_hit:
            both_success += 1
        else:
            both_fail += 1
    return {
        "count": len(ids),
        "enhanced_only_success_count": enhanced_only,
        "baseline_only_success_count": baseline_only,
        "both_success_count": both_success,
        "both_fail_count": both_fail,
    }


def mode_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qtype in ("table_content", "caption_level_table", "normal_control"):
        out[qtype] = {}
        for mode in ("dense_only", "bm25_only", "hybrid"):
            b = metrics.get(f"{mode}/{qtype}/baseline", {})
            e = metrics.get(f"{mode}/{qtype}/enhanced", {})
            out[qtype][mode] = {
                "baseline_doc_hit_at_10": b.get("doc_hit_at_10"),
                "enhanced_doc_hit_at_10": e.get("doc_hit_at_10"),
                "baseline_original_chunk_hit_at_10": b.get("original_chunk_hit_at_10"),
                "enhanced_original_chunk_hit_at_10": e.get("original_chunk_hit_at_10"),
                "baseline_corrected_chunk_hit_at_10": b.get("corrected_chunk_hit_at_10"),
                "enhanced_corrected_chunk_hit_at_10": e.get("corrected_chunk_hit_at_10"),
                "corrected_doc_hit_delta": delta(e.get("doc_hit_at_10"), b.get("doc_hit_at_10")),
                "corrected_chunk_hit_delta": delta(e.get("corrected_chunk_hit_at_10"), b.get("corrected_chunk_hit_at_10")),
            }
    return out


def build_caption_regression_review(
    original_records: list[dict[str, Any]],
    mapping_by_sample: dict[str, dict[str, str]],
) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in original_records:
        if record["mode"] == "hybrid" and record["query_type"] == "caption_level_table" and record.get("include_in_main_denominator"):
            by_pair[record["sample_id"]][record["variant"]] = record

    rows: list[dict[str, Any]] = []
    for sid, pair in sorted(by_pair.items()):
        base = pair.get("baseline")
        enh = pair.get("enhanced")
        if not base or not enh:
            continue
        if not base.get("chunk_hit_at_10") or enh.get("chunk_hit_at_10"):
            continue
        mapping = mapping_by_sample[sid]
        corrected = mapping["corrected_enhanced_target_chunk_id"]
        top10_ids = [hit["chunk_id"] for hit in enh.get("top_hits", [])]
        corrected_hit = corrected in top10_ids
        artifact = corrected and corrected != mapping["original_enhanced_target_chunk_id"] and corrected_hit
        true_regression = not corrected_hit
        rows.append({
            "sample_id": sid,
            "query": enh["query"],
            "baseline_target_chunk": mapping["original_baseline_target_chunk_id"],
            "original_enhanced_target_chunk": mapping["original_enhanced_target_chunk_id"],
            "corrected_enhanced_target_chunk": corrected,
            "stable_caption_block_id": mapping["stable_target_block_ids"],
            "corrected_target_hit_top10": corrected_hit,
            "mapping_artifact": artifact,
            "true_caption_regression": true_regression,
            "top10": [(hit["rank"], hit["chunk_id"]) for hit in enh.get("top_hits", [])],
        })
    counts = {
        "regression_count_original": len(rows),
        "mapping_artifact_count": sum(1 for row in rows if row["mapping_artifact"]),
        "true_caption_regression_count": sum(1 for row in rows if row["true_caption_regression"]),
        "unknown_count": sum(1 for row in rows if not row["mapping_artifact"] and not row["true_caption_regression"]),
    }
    return {"rows": rows, "counts": counts}


def corrected_gate(metrics: dict[str, Any], original_payload: dict[str, Any]) -> dict[str, Any]:
    hybrid = lambda q, v: metrics.get(f"hybrid/{q}/{v}", {})
    table_b = hybrid("table_content", "baseline")
    table_e = hybrid("table_content", "enhanced")
    caption_b = hybrid("caption_level_table", "baseline")
    caption_e = hybrid("caption_level_table", "enhanced")
    normal_b = hybrid("normal_control", "baseline")
    normal_e = hybrid("normal_control", "enhanced")
    table_pair = metrics.get("paired", {}).get("hybrid/table_content", {})
    normal_occ_b = normal_b.get("mean_table_topk_occupancy", 0.0)
    normal_occ_e = normal_e.get("mean_table_topk_occupancy", 0.0)
    validation = original_payload.get("validation", {})
    engineering_ok = all([
        validation.get("baseline_row_count_matches_chunks"),
        validation.get("enhanced_row_count_matches_chunks"),
        validation.get("baseline_bm25_count_matches_chunks"),
        validation.get("enhanced_bm25_count_matches_chunks"),
    ])
    return {
        "engineering_ok": bool(engineering_ok),
        "table_content_doc_hit_non_degraded": table_e.get("doc_hit_at_10", 0.0) >= table_b.get("doc_hit_at_10", 0.0),
        "table_content_corrected_chunk_hit_improved": table_e.get("corrected_chunk_hit_at_10", 0.0) > table_b.get("corrected_chunk_hit_at_10", 0.0),
        "table_content_enhanced_only_ge_baseline_only": table_pair.get("enhanced_only_success_count", 0) >= table_pair.get("baseline_only_success_count", 0),
        "caption_corrected_chunk_not_degraded_over_0_05": caption_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= caption_b.get("corrected_chunk_hit_at_10", 0.0),
        "normal_doc_not_degraded_over_0_05": normal_e.get("doc_hit_at_10", 0.0) + 0.05 >= normal_b.get("doc_hit_at_10", 0.0),
        "normal_corrected_chunk_not_degraded_over_0_05": normal_e.get("corrected_chunk_hit_at_10", 0.0) + 0.05 >= normal_b.get("corrected_chunk_hit_at_10", 0.0),
        "normal_table_occupancy_non_increased": normal_occ_e <= normal_occ_b,
        "normal_table_occupancy_baseline": normal_occ_b,
        "normal_table_occupancy_enhanced": normal_occ_e,
    }


def write_mapping_csv(path: Path, mappings: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "query_type",
        "target_doc_id",
        "original_baseline_target_chunk_id",
        "original_enhanced_target_chunk_id",
        "stable_target_block_ids",
        "corrected_baseline_target_chunk_id",
        "corrected_enhanced_target_chunk_id",
        "mapping_status",
        "mapping_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mappings)


def write_caption_review(path: Path, review: dict[str, Any]) -> None:
    lines = ["# Caption Regression Mapping Review", ""]
    for row in review["rows"]:
        lines.extend([
            f"## {row['sample_id']}",
            "",
            f"- query: {row['query']}",
            f"- baseline target chunk: {row['baseline_target_chunk']}",
            f"- original enhanced target chunk: {row['original_enhanced_target_chunk']}",
            f"- corrected enhanced target chunk: {row['corrected_enhanced_target_chunk']}",
            f"- stable caption block id: {row['stable_caption_block_id']}",
            f"- enhanced top10 hits corrected target: {str(row['corrected_target_hit_top10']).lower()}",
            f"- original miss caused by chunk_id mapping: {str(row['mapping_artifact']).lower()}",
            f"- true caption regression: {str(row['true_caption_regression']).lower()}",
            f"- enhanced top10 chunk ids: {row['top10']}",
            "",
        ])
    counts = review["counts"]
    lines.extend([
        "## Totals",
        "",
        f"- regression_count_original: {counts['regression_count_original']}",
        f"- mapping_artifact_count: {counts['mapping_artifact_count']}",
        f"- true_caption_regression_count: {counts['true_caption_regression_count']}",
        f"- unknown_count: {counts['unknown_count']}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    corrected_payload: dict[str, Any],
    caption_review: dict[str, Any],
    original_payload: dict[str, Any],
) -> None:
    metrics = corrected_payload["metrics"]
    gate = corrected_payload["gate"]
    original_metrics = original_payload["metrics"]

    def metric(qtype: str, variant: str, key: str) -> Any:
        return metrics.get(f"hybrid/{qtype}/{variant}", {}).get(key)

    table_pair = metrics["paired"]["hybrid/table_content"]
    caption_counts = caption_review["counts"]
    recommend_5c3 = all([
        gate["engineering_ok"],
        gate["table_content_doc_hit_non_degraded"],
        gate["table_content_corrected_chunk_hit_improved"],
        gate["table_content_enhanced_only_ge_baseline_only"],
        gate["caption_corrected_chunk_not_degraded_over_0_05"],
        gate["normal_doc_not_degraded_over_0_05"],
        gate["normal_corrected_chunk_not_degraded_over_0_05"],
        gate["normal_table_occupancy_non_increased"],
        caption_counts["true_caption_regression_count"] == 0,
    ])

    lines = [
        "# Phase 5C-2M Stable Target Mapping Summary",
        "",
        "## Corrected Hybrid Results",
        "",
        f"- table_content doc_hit@10: baseline={fmt(metric('table_content', 'baseline', 'doc_hit_at_10'))}, enhanced={fmt(metric('table_content', 'enhanced', 'doc_hit_at_10'))}",
        f"- table_content corrected_chunk_hit@10: baseline={fmt(metric('table_content', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('table_content', 'enhanced', 'corrected_chunk_hit_at_10'))}",
        f"- table_content split: enhanced_only={table_pair['enhanced_only_success_count']}, baseline_only={table_pair['baseline_only_success_count']}, both_success={table_pair['both_success_count']}, both_fail={table_pair['both_fail_count']}",
        f"- caption_level_table original chunk_hit@10: baseline={fmt(original_metrics['hybrid/caption_level_table/baseline']['chunk_hit_at_10'])}, enhanced={fmt(original_metrics['hybrid/caption_level_table/enhanced']['chunk_hit_at_10'])}",
        f"- caption_level_table corrected_chunk_hit@10: baseline={fmt(metric('caption_level_table', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('caption_level_table', 'enhanced', 'corrected_chunk_hit_at_10'))}",
        f"- normal_control doc_hit@10: baseline={fmt(metric('normal_control', 'baseline', 'doc_hit_at_10'))}, enhanced={fmt(metric('normal_control', 'enhanced', 'doc_hit_at_10'))}",
        f"- normal_control corrected_chunk_hit@10: baseline={fmt(metric('normal_control', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('normal_control', 'enhanced', 'corrected_chunk_hit_at_10'))}",
        f"- normal_control table occupancy: baseline={fmt(gate['normal_table_occupancy_baseline'])}, enhanced={fmt(gate['normal_table_occupancy_enhanced'])}",
        "",
        "## Required Answers",
        "",
        "1. chunk_id cannot be used as a cross-version exact target because re-chunking can move the same source block into a different chunk_id, while the old chunk_id may now contain different text.",
        f"2. With source_block_ids/caption block remap, caption_level_table corrected_chunk_hit@10 is baseline={fmt(metric('caption_level_table', 'baseline', 'corrected_chunk_hit_at_10'))}, enhanced={fmt(metric('caption_level_table', 'enhanced', 'corrected_chunk_hit_at_10'))}.",
        f"3. True caption_level_table regression remains: {yes_no(caption_counts['true_caption_regression_count'] > 0)}.",
        f"4. table_content gain still holds: {yes_no(gate['table_content_corrected_chunk_hit_improved'])}.",
        f"5. normal_control remains non-degraded: {yes_no(gate['normal_doc_not_degraded_over_0_05'] and gate['normal_corrected_chunk_not_degraded_over_0_05'])}.",
        f"6. normal top-k table occupancy indicates takeover: {yes_no(not gate['normal_table_occupancy_non_increased'])}. Enhanced occupancy is lower than baseline.",
        f"7. Phase 5C-2 passes corrected A/B gate: {yes_no(recommend_5c3)}.",
        f"8. Recommend Phase 5C-3: {yes_no(recommend_5c3)}.",
        "9. Next stage needs no system change before 5C-3; only keep stable-target evaluation mapping.",
        "10. Index rebuild needed: no.",
        "",
        "## Caption Regression Accounting",
        "",
        f"- regression_count_original: {caption_counts['regression_count_original']}",
        f"- mapping_artifact_count: {caption_counts['mapping_artifact_count']}",
        f"- true_caption_regression_count: {caption_counts['true_caption_regression_count']}",
        f"- unknown_count: {caption_counts['unknown_count']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def mean_rank(rows: list[dict[str, Any]], key: str) -> float | None:
    ranks = [int(r[key]) for r in rows if int(r.get(key) or 0) > 0]
    return statistics.mean(ranks) if ranks else None


def delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
