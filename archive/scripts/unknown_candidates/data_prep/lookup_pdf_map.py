#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主要功能:
- 根据当前文档编号 `doc_id`，快速反查该文档对应的真实文档名。
- 适合在测试 RAG 命中结果后，快速定位原始 PDF。

启动方法:
- 基本用法:
  `python scripts/data_prep/lookup_pdf_map.py doc_0009`
- 指定映射表:
  `python scripts/data_prep/lookup_pdf_map.py doc_0009 --map data/paper_round1_pdf_rename_map.csv`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_rows(mapping_path: Path) -> list[dict[str, str]]:
    suffix = mapping_path.suffix.lower()
    if suffix == ".csv":
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with mapping_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"不支持的映射文件格式: {mapping_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 doc_id 反查真实文档名"
    )
    parser.add_argument(
        "doc_id",
        help="文档编号，如 doc_0009",
    )
    parser.add_argument(
        "--map",
        default="data/paper_round1_pdf_rename_map.csv",
        help="映射表路径，支持 CSV/JSONL，默认: data/paper_round1_pdf_rename_map.csv",
    )
    args = parser.parse_args()

    mapping_path = Path(args.map).resolve()
    if not mapping_path.exists():
        print(f"[ERROR] 映射表不存在: {mapping_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(mapping_path)
    row = next((item for item in rows if item.get("doc_id") == args.doc_id), None)

    if row is None:
        print(f"[INFO] 未找到 {args.doc_id} 对应的文档。")
        sys.exit(1)

    real_name = row.get("old_name", "")
    current_name = row.get("new_name", "")

    print(f"doc_id: {args.doc_id}")
    print(f"real_name: {real_name}")
    print(f"current_name: {current_name}")


if __name__ == "__main__":
    main()
