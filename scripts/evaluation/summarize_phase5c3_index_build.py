#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 5C-3 small index build.")
    parser.add_argument("--baseline-chunks", default="/tmp/biorag_phase5c3_baseline/chunks.jsonl")
    parser.add_argument("--enhanced-chunks", default="/tmp/biorag_phase5c3_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--baseline-milvus-uri", default="/tmp/phase5c3_baseline.db")
    parser.add_argument("--baseline-collection", default="synbio_phase5c3_baseline")
    parser.add_argument("--baseline-bm25-path", default="/tmp/biorag_phase5c3_baseline/bm25_index.json")
    parser.add_argument("--enhanced-milvus-uri", default="/tmp/phase5c3_enhanced.db")
    parser.add_argument("--enhanced-collection", default="synbio_phase5c3_enhanced")
    parser.add_argument("--enhanced-bm25-path", default="/tmp/biorag_phase5c3_enhanced/bm25_index.json")
    parser.add_argument("--baseline-import-log", default="/tmp/phase5c3_baseline_import.log")
    parser.add_argument("--enhanced-import-log", default="/tmp/phase5c3_enhanced_import.log")
    parser.add_argument("--baseline-import-elapsed-sec", type=float, default=-1.0)
    parser.add_argument("--enhanced-import-elapsed-sec", type=float, default=-1.0)
    parser.add_argument("--output-dir", default="reports/phase5c3_table_expansion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_log = read_text(Path(args.baseline_import_log))
    enhanced_log = read_text(Path(args.enhanced_import_log))
    stats = {
        "baseline": variant_stats(
            chunks=Path(args.baseline_chunks),
            milvus_uri=args.baseline_milvus_uri,
            collection=args.baseline_collection,
            bm25_path=Path(args.baseline_bm25_path),
            import_elapsed_sec=args.baseline_import_elapsed_sec,
            import_log=baseline_log,
        ),
        "enhanced": variant_stats(
            chunks=Path(args.enhanced_chunks),
            milvus_uri=args.enhanced_milvus_uri,
            collection=args.enhanced_collection,
            bm25_path=Path(args.enhanced_bm25_path),
            import_elapsed_sec=args.enhanced_import_elapsed_sec,
            import_log=enhanced_log,
        ),
    }
    stats["all_counts_match"] = all(
        item["row_count_matches_chunk_count"] and item["bm25_record_count_matches_chunk_count"]
        for item in stats.values()
    )
    (out_dir / "index_build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(out_dir / "index_build_summary.md", stats)
    print(f"Wrote {out_dir / 'index_build_summary.md'}")


def variant_stats(
    *,
    chunks: Path,
    milvus_uri: str,
    collection: str,
    bm25_path: Path,
    import_elapsed_sec: float,
    import_log: str,
) -> dict[str, Any]:
    chunk_count = count_jsonl(chunks)
    row_count = milvus_row_count(milvus_uri, collection)
    bm25_records = bm25_record_count(bm25_path)
    tokenizer_warnings = warning_count(import_log, r"tokenizer tokens > max_length")
    varchar_truncation_warnings = warning_count(import_log, r"VARCHAR|字段超过")
    return {
        "chunks_jsonl": str(chunks),
        "milvus_uri": milvus_uri,
        "collection": collection,
        "bm25_path": str(bm25_path),
        "chunk_count": chunk_count,
        "row_count": row_count,
        "row_count_matches_chunk_count": row_count == chunk_count,
        "bm25_record_count": bm25_records,
        "bm25_record_count_matches_chunk_count": bm25_records == chunk_count if bm25_records is not None else False,
        "import_elapsed_sec": import_elapsed_sec,
        "bge_tokenizer_truncation_warning_count": tokenizer_warnings,
        "milvus_varchar_truncation_warning_count": varchar_truncation_warnings,
        "no_bge_tokenizer_truncation_warning": tokenizer_warnings == 0,
        "no_milvus_varchar_truncation": varchar_truncation_warnings == 0,
    }


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def milvus_row_count(uri: str, collection: str) -> int:
    client = MilvusClient(uri)
    stats = client.get_collection_stats(collection)
    return int(stats.get("row_count", 0))


def bm25_record_count(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get("records", []))


def warning_count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def write_summary(path: Path, stats: dict[str, Any]) -> None:
    lines = ["# Phase 5C-3 Index Build Summary", ""]
    for name in ("baseline", "enhanced"):
        item = stats[name]
        lines.extend([
            f"## {name.title()}",
            "",
            f"- chunks: {item['chunk_count']}",
            f"- row_count: {item['row_count']}",
            f"- row_count == chunk_count: {str(item['row_count_matches_chunk_count']).lower()}",
            f"- BM25 records: {item['bm25_record_count']}",
            f"- BM25 records == chunk_count: {str(item['bm25_record_count_matches_chunk_count']).lower()}",
            f"- import elapsed sec: {item['import_elapsed_sec']:.2f}",
            f"- BGE tokenizer truncation warnings: {item['bge_tokenizer_truncation_warning_count']}",
            f"- Milvus VARCHAR truncation warnings: {item['milvus_varchar_truncation_warning_count']}",
            "",
        ])
    lines.extend([
        "## Overall",
        "",
        f"- all counts match: {str(stats['all_counts_match']).lower()}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
