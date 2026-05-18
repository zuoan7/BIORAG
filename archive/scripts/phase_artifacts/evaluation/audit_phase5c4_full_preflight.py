#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TEXT_VARCHAR_MAX_LENGTH = 16384
RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH = 16384
BGE_MAX_TOKENS = 4096


@dataclass
class ChunkStats:
    chunks: int = 0
    tokens: int = 0
    table_focused: int = 0
    table_related: int = 0
    paragraph: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Phase 5C-4 full preflight reports.")
    parser.add_argument("--report_dir", default="reports/phase5c4_full_preflight")
    parser.add_argument("--baseline_chunks", default="/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
    parser.add_argument("--enhanced_chunks", default="/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--enhanced_parsed_clean", default="/tmp/biorag_phase5c4_full_enhanced/parsed_clean")
    parser.add_argument("--source_parsed_clean", default="data/paper_round1/parsed_clean")
    parser.add_argument("--model_path", default="models/BAAI/bge-m3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    build_stats = write_build_reports(
        report_dir=report_dir,
        enhanced_parsed_clean=Path(args.enhanced_parsed_clean),
        source_parsed_clean=Path(args.source_parsed_clean),
    )

    chunk_stats = None
    if Path(args.enhanced_chunks).exists() and Path(args.baseline_chunks).exists():
        chunk_stats = write_chunk_reports(
            report_dir=report_dir,
            baseline_chunks_path=Path(args.baseline_chunks),
            enhanced_chunks_path=Path(args.enhanced_chunks),
            enhanced_parsed_clean=Path(args.enhanced_parsed_clean),
        )
        truncation_stats = write_truncation_reports(
            report_dir=report_dir,
            baseline_chunks_path=Path(args.baseline_chunks),
            enhanced_chunks_path=Path(args.enhanced_chunks),
            model_path=Path(args.model_path),
        )
    else:
        truncation_stats = None

    write_summary(
        report_dir=report_dir,
        build_stats=build_stats,
        chunk_stats=chunk_stats,
        truncation_stats=truncation_stats,
    )
    if should_proceed(build_stats, chunk_stats, truncation_stats):
        write_phase5c5_plan(report_dir / "phase5c5_plan.md")


def write_build_reports(
    *,
    report_dir: Path,
    enhanced_parsed_clean: Path,
    source_parsed_clean: Path,
) -> dict[str, Any]:
    pilot_summary = read_json(report_dir / "pilot_enhancement_summary.json")
    audit_rows = read_csv(report_dir / "pilot_association_audit.csv")
    dryrun = read_json(report_dir / "dryrun_association_summary.json")
    accepted = [row for row in audit_rows if row.get("accepted_or_rejected") == "accepted"]
    conf = Counter(row.get("association_confidence", "") for row in accepted)
    per_doc = Counter(row.get("doc_id", "") for row in accepted)
    accepted_long_prose = sum(1 for row in accepted if is_long_prose(row.get("associated_text_preview", "")))
    uncertain_cases = sum(1 for row in audit_rows if row.get("accepted_or_rejected") == "uncertain")
    source_docs = len(list(source_parsed_clean.glob("*.json")))
    total_docs = int(pilot_summary.get("selected_docs") or source_docs)
    success_docs = int(pilot_summary.get("processed_docs") or 0)
    failed_docs = pilot_summary.get("failed_docs") or []
    failed_count = len(failed_docs)
    max_associations = max(per_doc.values(), default=0)
    top_high_risk_docs = dryrun.get("suspected_false_positive_docs") or dryrun.get("top_docs_by_association_count") or []
    success_rate = success_docs / total_docs if total_docs else 0.0
    dryrun_safe = bool(dryrun.get("dry_run_safe"))
    safe_to_chunk = bool(
        dryrun_safe
        and success_rate >= 0.99
        and accepted_long_prose <= max(10, len(accepted) * 0.03)
        and enhanced_parsed_clean.exists()
    )
    stats = {
        "total_docs": total_docs,
        "success_docs": success_docs,
        "failed_docs": failed_count,
        "failed_doc_details": failed_docs,
        "success_rate": success_rate,
        "total_associations": len(accepted),
        "high_confidence_count": conf.get("high", 0),
        "medium_confidence_count": conf.get("medium", 0),
        "low_confidence_count": conf.get("low", 0),
        "accepted_long_prose": accepted_long_prose,
        "uncertain_cases": uncertain_cases,
        "max_associations_per_doc": max_associations,
        "top_high_risk_docs": top_high_risk_docs[:10],
        "safe_to_chunk": safe_to_chunk,
        "dryrun_safe": dryrun_safe,
        "enhanced_parsed_clean": str(enhanced_parsed_clean),
    }
    (report_dir / "full_enhanced_build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 5C-4 Full Enhanced Build Summary",
        "",
        f"- total_docs: {stats['total_docs']}",
        f"- success_docs: {stats['success_docs']}",
        f"- failed_docs: {stats['failed_docs']}",
        f"- total_associations: {stats['total_associations']}",
        f"- confidence distribution: high={stats['high_confidence_count']}, "
        f"medium={stats['medium_confidence_count']}, low={stats['low_confidence_count']}",
        f"- accepted_long_prose: {stats['accepted_long_prose']}",
        f"- uncertain_cases: {stats['uncertain_cases']}",
        f"- max_associations_per_doc: {stats['max_associations_per_doc']}",
        f"- dryrun_safe: {str(stats['dryrun_safe']).lower()}",
        f"- safe_to_chunk: {str(stats['safe_to_chunk']).lower()}",
        "",
        "## Top high-risk docs",
        "",
    ]
    for row in top_high_risk_docs[:10]:
        lines.append(
            f"- {row.get('doc_id')}: accepted={row.get('accepted_associations')}, "
            f"risk={row.get('risk_score')}, reason={row.get('risk_reasons')}"
        )
    (report_dir / "full_enhanced_build_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return stats


def write_chunk_reports(
    *,
    report_dir: Path,
    baseline_chunks_path: Path,
    enhanced_chunks_path: Path,
    enhanced_parsed_clean: Path,
) -> dict[str, Any]:
    baseline = list(iter_jsonl(baseline_chunks_path))
    enhanced = list(iter_jsonl(enhanced_chunks_path))
    base_doc = chunk_stats_by_doc(baseline)
    enh_doc = chunk_stats_by_doc(enhanced)
    base_total = total_chunk_stats(baseline)
    enh_total = total_chunk_stats(enhanced)
    baseline_sets = schema_field_sets(baseline)
    enhanced_sets = schema_field_sets(enhanced)
    schema_same = len(baseline_sets) == 1 and len(enhanced_sets) == 1 and baseline_sets == enhanced_sets
    total_docs = len(list(enhanced_parsed_clean.glob("*.json")))
    failed, low_quality = parse_failed_log(enhanced_chunks_path.parent / "failed_docs.log")
    success_docs = total_docs - len(failed)
    doc_rows = per_doc_deltas(base_doc, enh_doc)
    chunk_delta = enh_total.chunks - base_total.chunks
    token_delta = enh_total.tokens - base_total.tokens
    chunk_delta_pct = pct(chunk_delta, base_total.chunks)
    token_delta_pct = pct(token_delta, base_total.tokens)
    suspicious = [
        row for row in doc_rows
        if (
            (row["chunk_count_delta"] > max(8, row["baseline_chunk_count"] * 0.2))
            or (row["token_count_delta"] > max(2000, row["baseline_token_count"] * 0.2))
        )
    ]
    suspicious_without_explanation = [
        row for row in suspicious
        if row.get("table_related_delta", 0) <= 0
    ]
    stats = {
        "total_docs": total_docs,
        "success_docs": success_docs,
        "failed_docs": len(failed),
        "failed_doc_details": failed,
        "low_quality_docs": len(low_quality),
        "low_quality_doc_details": low_quality,
        "baseline_chunk_count": base_total.chunks,
        "enhanced_chunk_count": enh_total.chunks,
        "chunk_count_delta": chunk_delta,
        "chunk_count_delta_pct": chunk_delta_pct,
        "baseline_token_count": base_total.tokens,
        "enhanced_token_count": enh_total.tokens,
        "token_count_delta": token_delta,
        "token_count_delta_pct": token_delta_pct,
        "baseline_table_focused_chunk_count": base_total.table_focused,
        "enhanced_table_focused_chunk_count": enh_total.table_focused,
        "enhanced_table_related_chunk_count": enh_total.table_related,
        "baseline_paragraph_chunk_count": base_total.paragraph,
        "enhanced_paragraph_chunk_count": enh_total.paragraph,
        "schema_same": schema_same,
        "field_set_count": len(enhanced_sets),
        "baseline_field_set_count": len(baseline_sets),
        "docs_with_largest_chunk_increase": top_delta(doc_rows, "chunk_count_delta"),
        "docs_with_largest_token_increase": top_delta(doc_rows, "token_count_delta"),
        "docs_with_largest_table_related_increase": top_delta(doc_rows, "table_related_delta"),
        "suspicious_growth_docs": top_delta(suspicious, "chunk_count_delta"),
        "suspicious_growth_docs_without_explanation": top_delta(
            suspicious_without_explanation, "chunk_count_delta"
        ),
        "safe_to_truncation_audit": bool(success_docs / total_docs >= 0.99 if total_docs else False),
        "gate_chunk_growth_ok": chunk_delta_pct <= 5.0,
        "gate_token_growth_ok": token_delta_pct <= 10.0,
    }
    (report_dir / "full_chunk_ab_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 5C-4 Full Chunk Build Summary",
        "",
        f"- success_docs: {stats['success_docs']} / {stats['total_docs']}",
        f"- failed_docs: {stats['failed_docs']}",
        f"- low_quality_docs: {stats['low_quality_docs']}",
        f"- baseline chunk_count: {stats['baseline_chunk_count']}",
        f"- enhanced chunk_count: {stats['enhanced_chunk_count']}",
        f"- chunk_count_delta: {stats['chunk_count_delta']} ({stats['chunk_count_delta_pct']:.2f}%)",
        f"- token_count_delta: {stats['token_count_delta']} ({stats['token_count_delta_pct']:.2f}%)",
        f"- baseline table-focused chunk count: {stats['baseline_table_focused_chunk_count']}",
        f"- enhanced table-focused chunk count: {stats['enhanced_table_focused_chunk_count']}",
        f"- enhanced table_related chunk count: {stats['enhanced_table_related_chunk_count']}",
        f"- baseline paragraph chunk count: {stats['baseline_paragraph_chunk_count']}",
        f"- enhanced paragraph chunk count: {stats['enhanced_paragraph_chunk_count']}",
        f"- schema same: {str(stats['schema_same']).lower()}",
        f"- field set count: {stats['field_set_count']}",
        f"- suspicious growth docs: {len(stats['suspicious_growth_docs'])}",
        f"- suspicious growth docs without explanation: "
        f"{len(stats['suspicious_growth_docs_without_explanation'])}",
    ]
    (report_dir / "full_chunk_build_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return stats


def write_truncation_reports(
    *,
    report_dir: Path,
    baseline_chunks_path: Path,
    enhanced_chunks_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(model_path)
    baseline = truncation_stats_for_chunks(baseline_chunks_path, tokenizer)
    enhanced = truncation_stats_for_chunks(enhanced_chunks_path, tokenizer)
    rows = baseline["risk_rows"] + enhanced["risk_rows"]
    csv_path = report_dir / "truncation_risk_chunks.csv"
    write_truncation_csv(csv_path, rows)
    delta = enhanced["bge_over_4096_count"] - baseline["bge_over_4096_count"]
    milvus_risk = (
        enhanced["retrieval_text_over_16384_count"]
        + enhanced["text_over_16384_count"]
    )
    stats = {
        "baseline_bge_tokenizer_gt_4096_token_chunks_count": baseline["bge_over_4096_count"],
        "enhanced_bge_tokenizer_gt_4096_token_chunks_count": enhanced["bge_over_4096_count"],
        "delta": delta,
        "baseline_retrieval_text_gt_16384_chars_count": baseline["retrieval_text_over_16384_count"],
        "enhanced_retrieval_text_gt_16384_chars_count": enhanced["retrieval_text_over_16384_count"],
        "baseline_text_gt_16384_chars_count": baseline["text_over_16384_count"],
        "enhanced_text_gt_16384_chars_count": enhanced["text_over_16384_count"],
        "chunks_at_milvus_varchar_risk": milvus_risk,
        "table_related_chunks_among_tokenizer_truncation_risks": enhanced["table_related_bge_over_4096_count"],
        "docs_with_many_truncation_risk_chunks": enhanced["docs_with_many_bge_risks"],
        "enhanced_significantly_increases_truncation_risk": significant_truncation_increase(
            baseline["bge_over_4096_count"],
            enhanced["bge_over_4096_count"],
        ),
        "tokenizer_backend": str(model_path),
    }
    lines = [
        "# Phase 5C-4 Truncation Risk Audit",
        "",
        f"- baseline BGE tokenizer >4096 token chunks count: "
        f"{stats['baseline_bge_tokenizer_gt_4096_token_chunks_count']}",
        f"- enhanced BGE tokenizer >4096 token chunks count: "
        f"{stats['enhanced_bge_tokenizer_gt_4096_token_chunks_count']}",
        f"- delta: {stats['delta']}",
        f"- baseline retrieval_text >16384 chars count: "
        f"{stats['baseline_retrieval_text_gt_16384_chars_count']}",
        f"- enhanced retrieval_text >16384 chars count: "
        f"{stats['enhanced_retrieval_text_gt_16384_chars_count']}",
        f"- baseline text >16384 chars count: {stats['baseline_text_gt_16384_chars_count']}",
        f"- enhanced text >16384 chars count: {stats['enhanced_text_gt_16384_chars_count']}",
        f"- chunks at Milvus VARCHAR risk: {stats['chunks_at_milvus_varchar_risk']}",
        f"- table_related chunks among tokenizer truncation risks: "
        f"{stats['table_related_chunks_among_tokenizer_truncation_risks']}",
        f"- enhanced significantly increases truncation risk: "
        f"{str(stats['enhanced_significantly_increases_truncation_risk']).lower()}",
        "",
        "## Docs with many truncation-risk chunks",
        "",
    ]
    for row in stats["docs_with_many_truncation_risk_chunks"]:
        lines.append(f"- {row['doc_id']}: {row['risk_count']}")
    (report_dir / "truncation_risk_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (report_dir / "truncation_risk_audit.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def write_summary(
    *,
    report_dir: Path,
    build_stats: dict[str, Any],
    chunk_stats: dict[str, Any] | None,
    truncation_stats: dict[str, Any] | None,
) -> None:
    dryrun = read_json(report_dir / "dryrun_association_summary.json")
    proceed = should_proceed(build_stats, chunk_stats, truncation_stats)
    need_tighten = not bool(dryrun.get("dry_run_safe")) or bool(
        build_stats.get("accepted_long_prose", 0) > max(10, build_stats.get("total_associations", 0) * 0.03)
    )
    lines = [
        "# Phase 5C-4 Full-corpus Preflight Summary",
        "",
        f"1. Full dry-run safe: {str(bool(dryrun.get('dry_run_safe'))).lower()}",
        f"2. Full enhanced parsed_clean generated: {str(build_stats.get('success_docs', 0) > 0).lower()}",
        f"3. Full enhanced chunks generated: {str(chunk_stats is not None).lower()}",
        f"4. table_related associations full scale: {build_stats.get('total_associations', 0)}",
        f"5. false positive risk controlled: {str(not need_tighten).lower()}",
        "6. chunk/token growth controlled: "
        + (
            f"{str(chunk_stats['gate_chunk_growth_ok'] and chunk_stats['gate_token_growth_ok']).lower()} "
            f"(chunk {chunk_stats['chunk_count_delta']} / {chunk_stats['chunk_count_delta_pct']:.2f}%, "
            f"token {chunk_stats['token_count_delta']} / {chunk_stats['token_count_delta_pct']:.2f}%)"
            if chunk_stats else "not_run"
        ),
        f"7. schema preserved: {str(chunk_stats.get('schema_same') if chunk_stats else False).lower()}",
        "8. BGE tokenizer truncation significantly increased: "
        + (
            str(truncation_stats["enhanced_significantly_increases_truncation_risk"]).lower()
            if truncation_stats else "not_run"
        ),
        "9. Milvus VARCHAR truncation risk is zero: "
        + (
            str(truncation_stats["chunks_at_milvus_varchar_risk"] == 0).lower()
            if truncation_stats else "not_run"
        ),
        f"10. Recommend Phase 5C-5 full enhanced experimental index + retrieval eval: {str(proceed).lower()}",
        f"11. Need tighten heuristic first: {str(need_tighten).lower()}",
        "12. Need OCR: false",
        "13. Need table_object/schema change: false",
        f"14. Later main-path integration can be considered: {str(proceed).lower()}",
        "15. Integration recommendation: keep switch-gated experimental path before any default main-path change",
        "",
        "## Gate",
        "",
        f"- proceed_to_phase5c5: {str(proceed).lower()}",
        f"- dryrun_no_major_blocker: {str(bool(dryrun.get('dry_run_safe'))).lower()}",
        f"- enhanced_build_success_rate: {build_stats.get('success_rate', 0.0):.4f}",
    ]
    if chunk_stats:
        lines.extend([
            f"- chunk_build_success_rate: "
            f"{(chunk_stats['success_docs'] / chunk_stats['total_docs']) if chunk_stats['total_docs'] else 0.0:.4f}",
            f"- schema_same: {str(chunk_stats['schema_same']).lower()}",
            f"- chunk_growth_pct: {chunk_stats['chunk_count_delta_pct']:.2f}",
            f"- token_growth_pct: {chunk_stats['token_count_delta_pct']:.2f}",
        ])
    if truncation_stats:
        lines.extend([
            f"- milvus_varchar_risk: {truncation_stats['chunks_at_milvus_varchar_risk']}",
            f"- bge_truncation_delta: {truncation_stats['delta']}",
        ])
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase5c5_plan(path: Path) -> None:
    lines = [
        "# Phase 5C-5 Plan",
        "",
        "- full enhanced experimental chunks: `/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl`",
        "- proposed Milvus URI: experimental local URI only; do not reuse production/main URI",
        "- proposed collection: `paper_round1_phase5c5_full_enhanced_exp`",
        "- BM25 cache path: `/tmp/biorag_phase5c5_full_enhanced/bm25_cache`",
        "- eval set: reuse Phase 5C-3 table_content, caption_level_table, normal_control queries and add full-corpus table-related cases from dry-run high/medium associations",
        "- stable target mapping: required before scoring; map expected doc_id and target block/chunk ids for baseline and enhanced",
        "- eval scope: retrieval-only hybrid A/B against Phase 4 compact full baseline",
        "- generation scope: no Qwen and no generation evaluation",
        "- rollback: delete experimental collection/cache and keep `/tmp/biorag_phase5c4_full_enhanced` disposable",
        "- baseline comparison: compare against `/tmp/biorag_phase4d_compact_chunks/chunks.jsonl` and existing Phase 4 compact full index",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_failed_log(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    failed: list[dict[str, str]] = []
    low_quality: list[dict[str, str]] = []
    if not path.exists():
        return failed, low_quality
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            status = parts[0]
            item = {"file": parts[1] if len(parts) > 1 else "", "reason": parts[2] if len(parts) > 2 else ""}
            if status == "FAILED":
                failed.append(item)
            elif status == "LOW_QUALITY":
                low_quality.append(item)
    return failed, low_quality


def is_long_prose(text: Any) -> bool:
    text = str(text or "")
    return len(text.split()) >= 45 and bool(__import__("re").search(r"[.!?]\s+[A-Z]", text))


def total_chunk_stats(chunks: list[dict[str, Any]]) -> ChunkStats:
    result = ChunkStats()
    for chunk in chunks:
        update_chunk_stats(result, chunk)
    return result


def chunk_stats_by_doc(chunks: list[dict[str, Any]]) -> dict[str, ChunkStats]:
    by_doc: dict[str, ChunkStats] = defaultdict(ChunkStats)
    for chunk in chunks:
        update_chunk_stats(by_doc[str(chunk.get("doc_id", ""))], chunk)
    return by_doc


def update_chunk_stats(stats: ChunkStats, chunk: dict[str, Any]) -> None:
    stats.chunks += 1
    stats.tokens += int(chunk.get("token_count") or 0)
    if chunk.get("contains_table_text") or chunk.get("contains_table_caption"):
        stats.table_focused += 1
    if is_table_related_chunk(chunk):
        stats.table_related += 1
    evidence_types = set(chunk.get("evidence_types") or chunk.get("block_types") or [])
    if "paragraph" in evidence_types:
        stats.paragraph += 1


def is_table_related_chunk(chunk: dict[str, Any]) -> bool:
    return any(
        isinstance(meta, dict) and meta.get("table_related") is True
        for meta in chunk.get("source_block_metadata", []) or []
    )


def schema_field_sets(chunks: list[dict[str, Any]]) -> set[frozenset[str]]:
    return {frozenset(chunk.keys()) for chunk in chunks}


def per_doc_deltas(
    baseline: dict[str, ChunkStats],
    enhanced: dict[str, ChunkStats],
) -> list[dict[str, Any]]:
    rows = []
    for doc_id in sorted(set(baseline) | set(enhanced)):
        base = baseline.get(doc_id, ChunkStats())
        enh = enhanced.get(doc_id, ChunkStats())
        rows.append({
            "doc_id": doc_id,
            "baseline_chunk_count": base.chunks,
            "enhanced_chunk_count": enh.chunks,
            "chunk_count_delta": enh.chunks - base.chunks,
            "baseline_token_count": base.tokens,
            "enhanced_token_count": enh.tokens,
            "token_count_delta": enh.tokens - base.tokens,
            "baseline_table_related_count": base.table_related,
            "enhanced_table_related_count": enh.table_related,
            "table_related_delta": enh.table_related - base.table_related,
        })
    return rows


def top_delta(rows: list[dict[str, Any]], key: str, limit: int = 20) -> list[dict[str, Any]]:
    return [
        row for row in sorted(rows, key=lambda item: (item.get(key, 0), item.get("doc_id", "")), reverse=True)[:limit]
        if row.get(key, 0) > 0
    ]


def pct(delta: int, base: int) -> float:
    return delta / base * 100.0 if base else 0.0


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def truncation_stats_for_chunks(path: Path, tokenizer) -> dict[str, Any]:
    risk_rows: list[dict[str, Any]] = []
    bge_over = 0
    retrieval_over = 0
    text_over = 0
    table_related_bge = 0
    doc_risks: Counter[str] = Counter()
    batch: list[dict[str, Any]] = []
    batch_size = 128

    def flush() -> None:
        nonlocal bge_over, retrieval_over, text_over, table_related_bge
        if not batch:
            return
        texts = [str(chunk.get("retrieval_text") or chunk.get("text") or "") for chunk in batch]
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        for chunk, token_len in zip(batch, lengths):
            retrieval_text = str(chunk.get("retrieval_text") or "")
            text = str(chunk.get("text") or "")
            retrieval_risk = len(retrieval_text) > RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH
            text_risk = len(text) > TEXT_VARCHAR_MAX_LENGTH
            bge_risk = token_len > BGE_MAX_TOKENS
            table_related = is_table_related_chunk(chunk)
            if bge_risk:
                bge_over += 1
                doc_risks[str(chunk.get("doc_id", ""))] += 1
                if table_related:
                    table_related_bge += 1
            if retrieval_risk:
                retrieval_over += 1
            if text_risk:
                text_over += 1
            if bge_risk or retrieval_risk or text_risk:
                risk_rows.append({
                    "dataset": path.parent.parent.name if path.parent.name == "chunks" else path.parent.name,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "bge_token_count": token_len,
                    "bge_gt_4096": bge_risk,
                    "retrieval_text_chars": len(retrieval_text),
                    "retrieval_text_gt_16384": retrieval_risk,
                    "text_chars": len(text),
                    "text_gt_16384": text_risk,
                    "table_related": table_related,
                    "contains_table_caption": bool(chunk.get("contains_table_caption")),
                    "contains_table_text": bool(chunk.get("contains_table_text")),
                })
        batch.clear()

    for chunk in iter_jsonl(path):
        batch.append(chunk)
        if len(batch) >= batch_size:
            flush()
    flush()

    return {
        "bge_over_4096_count": bge_over,
        "retrieval_text_over_16384_count": retrieval_over,
        "text_over_16384_count": text_over,
        "table_related_bge_over_4096_count": table_related_bge,
        "docs_with_many_bge_risks": [
            {"doc_id": doc_id, "risk_count": count}
            for doc_id, count in doc_risks.most_common(20)
            if count >= 2
        ],
        "risk_rows": risk_rows,
    }


def write_truncation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "chunk_id",
        "doc_id",
        "bge_token_count",
        "bge_gt_4096",
        "retrieval_text_chars",
        "retrieval_text_gt_16384",
        "text_chars",
        "text_gt_16384",
        "table_related",
        "contains_table_caption",
        "contains_table_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def significant_truncation_increase(baseline_count: int, enhanced_count: int) -> bool:
    delta = enhanced_count - baseline_count
    if delta <= 3:
        return False
    if baseline_count == 0:
        return enhanced_count > 3
    return delta / baseline_count > 0.5


def should_proceed(
    build_stats: dict[str, Any],
    chunk_stats: dict[str, Any] | None,
    truncation_stats: dict[str, Any] | None,
) -> bool:
    if not build_stats.get("safe_to_chunk"):
        return False
    if not chunk_stats or not truncation_stats:
        return False
    chunk_success = chunk_stats["success_docs"] / chunk_stats["total_docs"] if chunk_stats["total_docs"] else 0.0
    return bool(
        chunk_success >= 0.99
        and chunk_stats.get("schema_same")
        and chunk_stats.get("gate_chunk_growth_ok")
        and chunk_stats.get("gate_token_growth_ok")
        and truncation_stats.get("chunks_at_milvus_varchar_risk") == 0
        and not truncation_stats.get("enhanced_significantly_increases_truncation_risk")
        and not chunk_stats.get("suspicious_growth_docs_without_explanation")
    )


if __name__ == "__main__":
    main()
