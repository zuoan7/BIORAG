#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import json
import argparse
import sys


def collect_pdfs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def ensure_not_exists(path: Path):
    if path.exists():
        raise FileExistsError(f"输出文件已存在: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="批量重命名 PDF，并生成新旧文件名映射表"
    )
    parser.add_argument(
        "--dir",
        default="paper_round1/paper",
        help="PDF 所在目录，默认: paper_round1/paper"
    )
    parser.add_argument(
        "--prefix",
        default="doc",
        help="新文件名前缀，默认: doc"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="起始编号，默认: 1"
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="编号位数，默认: 4，对应 doc_0001.pdf"
    )
    parser.add_argument(
        "--csv",
        default="paper_round1_pdf_rename_map.csv",
        help="输出 CSV 映射表文件名"
    )
    parser.add_argument(
        "--jsonl",
        default="paper_round1_pdf_rename_map.jsonl",
        help="输出 JSONL 映射表文件名"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览，不真正改名"
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] 目录不存在或不是目录: {root}", file=sys.stderr)
        sys.exit(1)

    pdfs = collect_pdfs(root)
    if not pdfs:
        print(f"[ERROR] 目录中没有找到 PDF: {root}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv).resolve()
    jsonl_path = Path(args.jsonl).resolve()

    if not args.dry_run:
        ensure_not_exists(csv_path)
        ensure_not_exists(jsonl_path)

    # 先规划所有重命名，避免边改边冲突
    mappings = []
    used_new_names = set()

    for idx, old_path in enumerate(pdfs, start=args.start):
        doc_id = f"{args.prefix}_{idx:0{args.digits}d}"
        new_name = f"{doc_id}.pdf"

        if new_name in used_new_names:
            print(f"[ERROR] 新文件名重复: {new_name}", file=sys.stderr)
            sys.exit(1)

        used_new_names.add(new_name)
        new_path = root / new_name

        mappings.append({
            "doc_id": doc_id,
            "old_name": old_path.name,
            "new_name": new_name,
            "old_path": str(old_path),
            "new_path": str(new_path),
        })

    # 检查目标文件名是否会和“未参与本轮改名的文件”冲突
    existing_names = {p.name for p in root.iterdir() if p.is_file()}
    old_names = {m["old_name"] for m in mappings}

    for m in mappings:
        if m["new_name"] in existing_names and m["new_name"] not in old_names:
            print(
                f"[ERROR] 目标文件名已存在且不在本轮重命名集合中: {m['new_name']}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[INFO] 目录: {root}")
    print(f"[INFO] 找到 PDF 数量: {len(mappings)}")
    print(f"[INFO] dry-run: {args.dry_run}")
    print()

    for m in mappings[:20]:
        print(f"{m['old_name']}  ->  {m['new_name']}")
    if len(mappings) > 20:
        print(f"... 共 {len(mappings)} 个文件，以上仅显示前 20 个")
    print()

    if args.dry_run:
        print("[INFO] dry-run 模式，不会真正改名，也不会写映射表。")
        return

    # 两阶段改名，避免文件名互相覆盖
    temp_records = []
    for i, m in enumerate(mappings, start=1):
        old_path = Path(m["old_path"])
        temp_name = f".__tmp_rename__{i:08d}__.pdf"
        temp_path = root / temp_name

        while temp_path.exists():
            i += 1
            temp_name = f".__tmp_rename__{i:08d}__.pdf"
            temp_path = root / temp_name

        old_path.rename(temp_path)
        temp_records.append((temp_path, root / m["new_name"]))

    # 从临时名改到最终名
    for temp_path, final_path in temp_records:
        temp_path.rename(final_path)

    # 写 CSV
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doc_id", "old_name", "new_name", "old_path", "new_path"]
        )
        writer.writeheader()
        writer.writerows(mappings)

    # 写 JSONL
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in mappings:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] 已完成批量重命名")
    print(f"[OK] CSV 映射表: {csv_path}")
    print(f"[OK] JSONL 映射表: {jsonl_path}")


if __name__ == "__main__":
    main()
