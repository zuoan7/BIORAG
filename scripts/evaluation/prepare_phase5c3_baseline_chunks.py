#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 5C-3 baseline chunk subset and chunk A/B stats.")
    parser.add_argument("--selected-docs", default="reports/phase5c3_table_expansion/selected_docs.csv")
    parser.add_argument("--baseline-full-jsonl", default="/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
    parser.add_argument("--enhanced-jsonl", default="/tmp/biorag_phase5c3_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--association-audit", default="reports/phase5c3_table_expansion/association_audit/association_audit.csv")
    parser.add_argument("--baseline-output-jsonl", default="/tmp/biorag_phase5c3_baseline/chunks.jsonl")
    parser.add_argument("--output-dir", default="reports/phase5c3_table_expansion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_doc_ids = load_selected_doc_ids(Path(args.selected_docs))
    selected = set(selected_doc_ids)
    baseline_chunks = [
        chunk for chunk in iter_jsonl(Path(args.baseline_full_jsonl))
        if str(chunk.get("doc_id", "")) in selected
    ]
    enhanced_chunks = list(iter_jsonl(Path(args.enhanced_jsonl)))

    baseline_out = Path(args.baseline_output_jsonl)
    baseline_out.parent.mkdir(parents=True, exist_ok=True)
    with baseline_out.open("w", encoding="utf-8") as f:
        for chunk in baseline_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    accepted_associations = load_accepted_associations(Path(args.association_audit))
    baseline_doc_ids = {str(c.get("doc_id", "")) for c in baseline_chunks}
    enhanced_doc_ids = {str(c.get("doc_id", "")) for c in enhanced_chunks}
    baseline_tokens = token_count(baseline_chunks)
    enhanced_tokens = token_count(enhanced_chunks)
    baseline_chunk_count = len(baseline_chunks)
    enhanced_chunk_count = len(enhanced_chunks)
    stats = {
        "selected_doc_count": len(selected_doc_ids),
        "selected_doc_ids": selected_doc_ids,
        "baseline_output_jsonl": str(baseline_out),
        "baseline_chunk_count": baseline_chunk_count,
        "enhanced_chunk_count": enhanced_chunk_count,
        "chunk_count_delta": enhanced_chunk_count - baseline_chunk_count,
        "chunk_count_delta_pct": pct(enhanced_chunk_count - baseline_chunk_count, baseline_chunk_count),
        "baseline_token_count": baseline_tokens,
        "enhanced_token_count": enhanced_tokens,
        "token_count_delta": enhanced_tokens - baseline_tokens,
        "token_count_delta_pct": pct(enhanced_tokens - baseline_tokens, baseline_tokens),
        "baseline_table_focused_count": table_focused_count(baseline_chunks),
        "enhanced_table_focused_count": table_focused_count(enhanced_chunks),
        "baseline_caption_only_table_count": caption_only_table_count(baseline_chunks),
        "enhanced_caption_only_table_count": caption_only_table_count(enhanced_chunks),
        "baseline_paragraph_chunk_count": paragraph_count(baseline_chunks),
        "enhanced_paragraph_chunk_count": paragraph_count(enhanced_chunks),
        "enhanced_table_related_chunk_count": table_related_chunk_count(enhanced_chunks, accepted_associations),
        "baseline_doc_count": len(baseline_doc_ids),
        "enhanced_doc_count": len(enhanced_doc_ids),
        "baseline_doc_ids_match_selected": baseline_doc_ids == selected,
        "enhanced_doc_ids_match_selected": enhanced_doc_ids == selected,
        "baseline_doc_ids_match_enhanced": baseline_doc_ids == enhanced_doc_ids,
        "schema_same": schema_same(baseline_chunks, enhanced_chunks),
        "baseline_schema_field_sets": sorted(["|".join(sorted(s)) for s in schema_field_sets(baseline_chunks)]),
        "enhanced_schema_field_sets": sorted(["|".join(sorted(s)) for s in schema_field_sets(enhanced_chunks)]),
        "baseline_content_shape": content_shape_counts(baseline_chunks),
        "enhanced_content_shape": content_shape_counts(enhanced_chunks),
    }

    if baseline_doc_ids != selected:
        missing = sorted(selected - baseline_doc_ids)
        extra = sorted(baseline_doc_ids - selected)
        raise SystemExit(f"Baseline subset doc_id mismatch. missing={missing[:10]} extra={extra[:10]}")
    if enhanced_doc_ids != selected:
        missing = sorted(selected - enhanced_doc_ids)
        extra = sorted(enhanced_doc_ids - selected)
        raise SystemExit(f"Enhanced doc_id mismatch. missing={missing[:10]} extra={extra[:10]}")

    (output_dir / "chunk_ab_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "chunk_ab_summary.md", stats)
    print(f"Wrote {baseline_out}")
    print(f"Wrote {output_dir / 'chunk_ab_summary.md'}")


def load_selected_doc_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row["doc_id"] for row in csv.DictReader(f) if row.get("doc_id")]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_accepted_associations(path: Path) -> set[tuple[str, str]]:
    accepted: set[tuple[str, str]] = set()
    if not path.exists():
        return accepted
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("accepted_or_rejected") == "accepted":
                doc_id = row.get("doc_id", "")
                block_id = row.get("associated_block_id", "")
                if doc_id and block_id:
                    accepted.add((doc_id, block_id))
    return accepted


def token_count(chunks: list[dict[str, Any]]) -> int:
    return sum(int(c.get("token_count") or 0) for c in chunks)


def pct(delta: int, base: int) -> float:
    return delta / base * 100.0 if base else 0.0


def table_focused_count(chunks: list[dict[str, Any]]) -> int:
    return sum(1 for c in chunks if c.get("contains_table_text") or c.get("contains_table_caption"))


def caption_only_table_count(chunks: list[dict[str, Any]]) -> int:
    return sum(1 for c in chunks if c.get("contains_table_caption") and not c.get("contains_table_text"))


def paragraph_count(chunks: list[dict[str, Any]]) -> int:
    return sum(1 for c in chunks if "paragraph" in set(c.get("evidence_types") or c.get("block_types") or []))


def table_related_chunk_count(chunks: list[dict[str, Any]], accepted: set[tuple[str, str]]) -> int:
    count = 0
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        block_ids = set(str(x) for x in (chunk.get("source_block_ids") or [])) | set(str(x) for x in (chunk.get("block_ids") or []))
        source_meta_related = any(
            isinstance(meta, dict) and meta.get("table_related") is True
            for meta in chunk.get("source_block_metadata", []) or []
        )
        if source_meta_related or any((doc_id, block_id) in accepted for block_id in block_ids):
            count += 1
    return count


def schema_field_sets(chunks: list[dict[str, Any]]) -> set[frozenset[str]]:
    return {frozenset(chunk.keys()) for chunk in chunks}


def schema_same(baseline_chunks: list[dict[str, Any]], enhanced_chunks: list[dict[str, Any]]) -> bool:
    baseline_sets = schema_field_sets(baseline_chunks)
    enhanced_sets = schema_field_sets(enhanced_chunks)
    return len(baseline_sets) == 1 and len(enhanced_sets) == 1 and baseline_sets == enhanced_sets


def content_shape_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        if chunk.get("contains_table_text"):
            counter["table_text"] += 1
        elif chunk.get("contains_table_caption"):
            counter["table_caption_only"] += 1
        elif chunk.get("contains_figure_caption"):
            counter["figure_caption"] += 1
        elif chunk.get("contains_references"):
            counter["references"] += 1
        elif chunk.get("contains_metadata"):
            counter["metadata"] += 1
        elif "paragraph" in set(chunk.get("evidence_types") or chunk.get("block_types") or []):
            counter["paragraph"] += 1
        else:
            counter["other"] += 1
    return dict(sorted(counter.items()))


def write_summary(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# Phase 5C-3 Chunk A/B Summary",
        "",
        f"- selected doc count: {stats['selected_doc_count']}",
        f"- baseline chunk count: {stats['baseline_chunk_count']}",
        f"- enhanced chunk count: {stats['enhanced_chunk_count']}",
        f"- chunk_count_delta: {stats['chunk_count_delta']} ({stats['chunk_count_delta_pct']:.2f}%)",
        f"- token_count_delta: {stats['token_count_delta']} ({stats['token_count_delta_pct']:.2f}%)",
        f"- baseline table-focused count: {stats['baseline_table_focused_count']}",
        f"- enhanced table-focused count: {stats['enhanced_table_focused_count']}",
        f"- enhanced table_related chunk count: {stats['enhanced_table_related_chunk_count']}",
        f"- baseline caption-only table chunk count: {stats['baseline_caption_only_table_count']}",
        f"- enhanced caption-only table chunk count: {stats['enhanced_caption_only_table_count']}",
        f"- baseline paragraph chunk count: {stats['baseline_paragraph_chunk_count']}",
        f"- enhanced paragraph chunk count: {stats['enhanced_paragraph_chunk_count']}",
        f"- schema same: {str(stats['schema_same']).lower()}",
        f"- baseline doc ids match selected: {str(stats['baseline_doc_ids_match_selected']).lower()}",
        f"- enhanced doc ids match selected: {str(stats['enhanced_doc_ids_match_selected']).lower()}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
