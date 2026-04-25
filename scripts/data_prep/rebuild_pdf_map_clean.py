#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主要功能:
- 读取现有 PDF 重命名映射表中的旧记录。
- 使用多编码自动恢复逻辑，把映射表中的乱码 `old_name` 清洗成可读名称。
- 重新生成并覆盖 `paper_round1_pdf_rename_map.csv/jsonl` 两份映射表。
- 该脚本只覆盖映射表，不修改 PDF 文件本身。

启动方法:
- 先预览，不覆盖文件:
  `python scripts/data_prep/rebuild_pdf_map_clean.py --dry-run`
- 真正覆盖 CSV/JSONL 映射表:
  `python scripts/data_prep/rebuild_pdf_map_clean.py`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from restore_pdf_names_from_map import DEFAULT_ENCODINGS, decode_broken_name


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["doc_id", "old_name", "new_name", "old_path", "new_path"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_rows(
    rows: list[dict[str, str]],
    encodings: list[str],
    paper_dir: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    rebuilt: list[dict[str, str]] = []
    errors: list[str] = []

    for row in rows:
        doc_id = row.get("doc_id", "")
        old_name = row.get("old_name", "")
        new_name = row.get("new_name", "")

        if not doc_id or not old_name or not new_name:
            errors.append(f"{doc_id or '<missing>'}: 缺少必要字段")
            continue

        try:
            clean_old_name, _ = decode_broken_name(old_name, encodings)
        except Exception as exc:
            errors.append(f"{doc_id}: 文件名恢复失败: {old_name} ({exc})")
            continue

        rebuilt.append(
            {
                "doc_id": doc_id,
                "old_name": clean_old_name,
                "new_name": new_name,
                "old_path": str(paper_dir / clean_old_name),
                "new_path": str(paper_dir / new_name),
            }
        )

    return rebuilt, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重建并覆盖干净版 PDF 映射表"
    )
    parser.add_argument(
        "--csv",
        default="data/paper_round1_pdf_rename_map.csv",
        help="CSV 映射表路径，默认: data/paper_round1_pdf_rename_map.csv",
    )
    parser.add_argument(
        "--jsonl",
        default="data/paper_round1_pdf_rename_map.jsonl",
        help="JSONL 映射表路径，默认: data/paper_round1_pdf_rename_map.jsonl",
    )
    parser.add_argument(
        "--paper-dir",
        default="data/paper_round1/paper",
        help="PDF 目录，用于重建 old_path/new_path，默认: data/paper_round1/paper",
    )
    parser.add_argument(
        "--encodings",
        default=",".join(DEFAULT_ENCODINGS),
        help="按顺序尝试的编码列表，逗号分隔",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览，不实际覆盖文件",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    jsonl_path = Path(args.jsonl).resolve()
    paper_dir = Path(args.paper_dir).resolve()
    encodings = [item.strip() for item in args.encodings.split(",") if item.strip()]

    if not csv_path.exists():
        print(f"[ERROR] CSV 映射表不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not paper_dir.exists():
        print(f"[ERROR] PDF 目录不存在: {paper_dir}", file=sys.stderr)
        sys.exit(1)
    if not encodings:
        print("[ERROR] encodings 不能为空", file=sys.stderr)
        sys.exit(1)

    rows = load_csv_rows(csv_path)
    rebuilt, errors = rebuild_rows(rows, encodings, paper_dir)

    print(f"[INFO] CSV:       {csv_path}")
    print(f"[INFO] JSONL:     {jsonl_path}")
    print(f"[INFO] PDF 目录:  {paper_dir}")
    print(f"[INFO] 候选编码: {', '.join(encodings)}")
    print(f"[INFO] 可重建条数: {len(rebuilt)}")
    print(f"[INFO] 错误条数:   {len(errors)}")
    print()

    for row in rebuilt[:20]:
        print(f"{row['doc_id']} | {row['old_name']} -> {row['new_name']}")
    if len(rebuilt) > 20:
        print(f"... 共 {len(rebuilt)} 条，以上仅显示前 20 条")

    if errors:
        print("\n[WARN] 以下条目无法重建:")
        for item in errors[:20]:
            print(f"  - {item}")
        if len(errors) > 20:
            print(f"  ... 另有 {len(errors) - 20} 条")

    if args.dry_run:
        print("\n[INFO] dry-run 模式，不会覆盖映射表。")
        return

    if errors:
        print("\n[ERROR] 存在无法重建的条目，已停止覆盖。", file=sys.stderr)
        sys.exit(1)

    write_csv_rows(csv_path, rebuilt)
    write_jsonl_rows(jsonl_path, rebuilt)
    print("\n[OK] 已覆盖更新 CSV 和 JSONL 映射表。")


if __name__ == "__main__":
    main()
