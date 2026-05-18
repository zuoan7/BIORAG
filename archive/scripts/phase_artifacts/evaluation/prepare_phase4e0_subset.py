#!/usr/bin/env python3
"""Prepare a deterministic Phase 4E-0 retrieval sanity subset.

The script only reads an existing compact chunks.jsonl and writes a smaller
JSONL with complete documents. Chunk records are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FALSE_TABLE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[TABLE CAPTION\]\s*)?"
    r"table\s+s?\d+[.:]?\s+(?:the\s+)?[A-Z]\.?\s*$",
    re.I,
)
FALSE_FIGURE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[FIGURE CAPTION\]\s*)?(?:fig(?:ure)?\.?)\s+s?\d+[A-Z]?[.:]?\s*$",
    re.I,
)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                chunks.append(item)
    return chunks


def is_table_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption") or chunk.get("contains_table_text"))


def is_figure_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_figure_caption"))


def is_evidence_chunk(chunk: dict[str, Any]) -> bool:
    return is_table_focused(chunk) or is_figure_focused(chunk)


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def is_short_caption(chunk: dict[str, Any]) -> bool:
    return is_evidence_chunk(chunk) and len(clean_text(chunk.get("text", ""))) < 80


def is_likely_false_caption(chunk: dict[str, Any]) -> bool:
    body = clean_text(chunk.get("text", ""))
    if chunk.get("contains_table_caption") and FALSE_TABLE_CAPTION_PATTERN.match(body):
        return True
    if chunk.get("contains_figure_caption") and FALSE_FIGURE_CAPTION_PATTERN.match(body):
        return True
    return False


def schema_signature(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    field_sets = {tuple(sorted(chunk.keys())) for chunk in chunks}
    return {
        "field_set_count": len(field_sets),
        "fields": list(field_sets)[0] if len(field_sets) == 1 else None,
    }


def add_top_docs(
    selected: dict[str, set[str]],
    doc_stats: dict[str, Counter[str]],
    reason: str,
    metric: str,
    count: int,
    *,
    require_no_evidence: bool = False,
) -> None:
    candidates = []
    for doc_id, stats in doc_stats.items():
        if require_no_evidence and (stats["table_focused"] or stats["figure_focused"]):
            continue
        if stats[metric] <= 0:
            continue
        candidates.append((stats[metric], doc_id))
    for _value, doc_id in sorted(candidates, key=lambda item: (-item[0], item[1]))[:count]:
        selected[doc_id].add(reason)


def build_doc_stats(chunks: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        if not doc_id:
            continue
        stats[doc_id]["chunks"] += 1
        if is_table_focused(chunk):
            stats[doc_id]["table_focused"] += 1
        if is_figure_focused(chunk):
            stats[doc_id]["figure_focused"] += 1
        if not is_evidence_chunk(chunk):
            stats[doc_id]["paragraph"] += 1
        if is_evidence_chunk(chunk) and chunk.get("section") == "Title":
            stats[doc_id]["title_evidence"] += 1
        if is_short_caption(chunk):
            stats[doc_id]["short_caption"] += 1
        if is_likely_false_caption(chunk):
            stats[doc_id]["likely_false_caption"] += 1
    return stats


def select_docs(chunks: list[dict[str, Any]], max_docs: int) -> dict[str, set[str]]:
    doc_stats = build_doc_stats(chunks)
    selected: dict[str, set[str]] = defaultdict(set)

    if "doc_0367" in doc_stats:
        selected["doc_0367"].add("required_doc_0367")

    add_top_docs(selected, doc_stats, "table_focused_docs", "table_focused", 8)
    add_top_docs(selected, doc_stats, "figure_focused_docs", "figure_focused", 8)
    add_top_docs(selected, doc_stats, "section_title_evidence_docs", "title_evidence", 10)
    add_top_docs(selected, doc_stats, "short_caption_docs", "short_caption", 8)
    add_top_docs(selected, doc_stats, "likely_false_caption_docs", "likely_false_caption", 8)

    if len(selected) < max_docs:
        add_top_docs(
            selected,
            doc_stats,
            "paragraph_heavy_control_docs",
            "paragraph",
            max_docs - len(selected),
            require_no_evidence=True,
        )
    if len(selected) < max_docs:
        add_top_docs(
            selected,
            doc_stats,
            "paragraph_heavy_control_docs",
            "paragraph",
            max_docs - len(selected),
        )

    if len(selected) <= max_docs:
        return dict(selected)

    protected = {"doc_0367"}
    reason_priority = {
        "section_title_evidence_docs": 0,
        "likely_false_caption_docs": 1,
        "short_caption_docs": 2,
        "table_focused_docs": 3,
        "figure_focused_docs": 4,
        "paragraph_heavy_control_docs": 5,
    }

    def rank(doc_id: str) -> tuple[int, int, int, str]:
        reasons = selected[doc_id]
        best_reason = min(reason_priority.get(reason, 9) for reason in reasons)
        protected_rank = 0 if doc_id in protected else 1
        coverage = len(reasons)
        return (protected_rank, best_reason, -coverage, doc_id)

    kept = sorted(selected, key=rank)[:max_docs]
    return {doc_id: selected[doc_id] for doc_id in kept}


def summarize(chunks: list[dict[str, Any]], selected_docs: set[str]) -> dict[str, Any]:
    subset_chunks = [chunk for chunk in chunks if chunk.get("doc_id") in selected_docs]
    table_count = sum(1 for chunk in subset_chunks if is_table_focused(chunk))
    figure_count = sum(1 for chunk in subset_chunks if is_figure_focused(chunk))
    paragraph_count = sum(1 for chunk in subset_chunks if not is_evidence_chunk(chunk))
    title_evidence_count = sum(
        1
        for chunk in subset_chunks
        if is_evidence_chunk(chunk) and chunk.get("section") == "Title"
    )
    short_caption_count = sum(1 for chunk in subset_chunks if is_short_caption(chunk))
    false_caption_count = sum(1 for chunk in subset_chunks if is_likely_false_caption(chunk))
    return {
        "doc_count": len(selected_docs),
        "chunk_count": len(subset_chunks),
        "table_focused_chunk_count": table_count,
        "figure_focused_chunk_count": figure_count,
        "paragraph_chunk_count": paragraph_count,
        "contains_doc_0367": "doc_0367" in selected_docs,
        "section_title_evidence_chunk_count": title_evidence_count,
        "short_caption_chunk_count": short_caption_count,
        "likely_false_caption_chunk_count": false_caption_count,
        "contains_section_title_evidence": title_evidence_count > 0,
        "contains_suspicious_or_short_captions": short_caption_count > 0 or false_caption_count > 0,
        "schema": schema_signature(subset_chunks),
    }


def write_jsonl(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: dict[str, Any], manifest_path: Path, output_jsonl: Path) -> None:
    lines = [
        "# Phase 4E-0 Subset Summary",
        "",
        "## Inputs",
        "",
        f"- output_jsonl: `{output_jsonl}`",
        f"- manifest: `{manifest_path}`",
        "",
        "## Composition",
        "",
        f"- docs: {summary['doc_count']}",
        f"- chunks: {summary['chunk_count']}",
        f"- table_focused_chunks: {summary['table_focused_chunk_count']}",
        f"- figure_focused_chunks: {summary['figure_focused_chunk_count']}",
        f"- paragraph_chunks: {summary['paragraph_chunk_count']}",
        f"- contains_doc_0367: `{summary['contains_doc_0367']}`",
        f"- contains_section_title_evidence: `{summary['contains_section_title_evidence']}`",
        f"- section_title_evidence_chunks: {summary['section_title_evidence_chunk_count']}",
        f"- contains_suspicious_or_short_captions: `{summary['contains_suspicious_or_short_captions']}`",
        f"- short_caption_chunks: {summary['short_caption_chunk_count']}",
        f"- likely_false_caption_chunks: {summary['likely_false_caption_chunk_count']}",
        f"- schema_field_set_count: {summary['schema']['field_set_count']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_docs", type=int, default=50)
    args = parser.parse_args()

    chunks_jsonl = Path(args.chunks_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_jsonl)
    selected = select_docs(chunks, max_docs=args.max_docs)
    selected_docs = set(selected)
    subset_chunks = [chunk for chunk in chunks if chunk.get("doc_id") in selected_docs]
    summary = summarize(chunks, selected_docs)

    manifest = {
        "chunks_jsonl": str(chunks_jsonl),
        "output_jsonl": str(output_jsonl),
        "max_docs": args.max_docs,
        "summary": summary,
        "selected_docs": [
            {
                "doc_id": doc_id,
                "reasons": sorted(reasons),
            }
            for doc_id, reasons in sorted(selected.items())
        ],
    }

    write_jsonl(output_jsonl, subset_chunks)
    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(output_dir / "subset_summary.md", summary, manifest_path, output_jsonl)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
