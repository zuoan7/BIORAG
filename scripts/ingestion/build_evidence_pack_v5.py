#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build v5 RAG evidence packs from parsed_clean JSON files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.document_cleaning_v5 import build_evidence_pack


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_build(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    stats: dict[str, Any] = {
        "input_doc_count": len(json_files),
        "success_doc_count": 0,
        "failed_doc_count": 0,
        "evidence_unit_count": 0,
        "excluded_block_counts": Counter(),
        "evidence_type_counts": Counter(),
        "failed_docs": [],
    }

    for path in json_files:
        print(f"  处理: {path.name} ...", end=" ")
        try:
            clean_data = load_json(path)
            evidence_pack = build_evidence_pack(clean_data)
            output_path = output_dir / f"{evidence_pack['doc_id']}.json"
            output_path.write_text(json.dumps(evidence_pack, ensure_ascii=False, indent=2), encoding="utf-8")

            stats["success_doc_count"] += 1
            stats["evidence_unit_count"] += len(evidence_pack.get("evidence_units", []) or [])
            stats["excluded_block_counts"].update(evidence_pack.get("excluded_block_counts", {}) or {})
            stats["evidence_type_counts"].update(
                evidence_pack.get("validation_summary", {}).get("evidence_type_counts", {}) or {}
            )
            print(f"OK ({len(evidence_pack.get('evidence_units', []) or [])} evidence units)")
        except Exception as exc:  # pragma: no cover - CLI safety net
            stats["failed_doc_count"] += 1
            stats["failed_docs"].append({"file": path.name, "error": str(exc)})
            print(f"FAILED ({exc})")

    stats["excluded_block_counts"] = dict(stats["excluded_block_counts"])
    stats["evidence_type_counts"] = dict(stats["evidence_type_counts"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="parsed_clean -> evidence_pack_v5")
    parser.add_argument("--input_dir", required=True, help="Directory containing parsed_clean JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory for evidence_pack_v5 JSON files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("构建 evidence_pack_v5")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()

    start = time.time()
    stats = batch_build(input_dir, output_dir)
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print("evidence_pack_v5 统计")
    print("=" * 60)
    print(f"输入文档数:     {stats['input_doc_count']}")
    print(f"成功文档数:     {stats['success_doc_count']}")
    print(f"失败文档数:     {stats['failed_doc_count']}")
    print(f"evidence units: {stats['evidence_unit_count']}")
    print(f"evidence types: {stats['evidence_type_counts']}")
    print(f"排除 block:     {stats['excluded_block_counts']}")
    print(f"耗时:           {elapsed:.2f}s")

    if stats["failed_docs"]:
        print()
        print("失败列表:")
        for item in stats["failed_docs"]:
            print(f"  - {item['file']}: {item['error']}")


if __name__ == "__main__":
    main()

