#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主要功能:
- 根据 PDF 重命名映射表，把 `data/paper_round1/paper/` 目录下当前的
  `doc_0001.pdf` 这类编号文件恢复成原始文档名。
- 自动尝试多种“错误字符串 -> legacy 编码 -> utf-8 解码”的恢复方式，
  选择最像正常文件名的结果。
- 只修改原始 PDF 文件名，不修改 parsed/chunks/runtime/vectorstores 等已序列化数据。

启动方法:
- 先预览，不实际改名:
  `python scripts/data_prep/restore_pdf_names_from_map.py --dry-run`
- 真正执行改名:
  `python scripts/data_prep/restore_pdf_names_from_map.py`
- 指定目录和映射表:
  `python scripts/data_prep/restore_pdf_names_from_map.py --dir data/paper_round1/paper --map data/paper_round1_pdf_rename_map.csv`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path


def load_mappings(mapping_path: Path) -> list[dict[str, str]]:
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


DEFAULT_ENCODINGS = [
    "cp1251",
    "cp866",
    "cp437",
    "cp1252",
    "latin1",
]


def decode_broken_name(name: str, encodings: list[str]) -> tuple[str, str]:
    """
    自动尝试多种反向解码方式:
    错误字符串 -> legacy 编码 -> utf-8 解码
    返回 (最佳结果, 使用的编码)。
    """
    candidates: list[tuple[float, str, str]] = []

    for encoding in encodings:
        try:
            decoded = name.encode(encoding).decode("utf-8")
        except Exception:
            continue
        candidates.append((_score_filename(decoded), decoded, encoding))

    if not candidates:
        raise ValueError("所有候选编码都恢复失败")

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_name, best_encoding = candidates[0]
    return best_name, best_encoding


def ensure_safe_filename(name: str) -> str:
    if "/" in name or "\x00" in name:
        raise ValueError(f"文件名包含非法字符: {name!r}")
    if name in {".", ".."}:
        raise ValueError(f"非法文件名: {name!r}")
    return name


def _score_filename(name: str) -> float:
    """
    对恢复后的文件名打分，优先选择更像“正常论文标题/文件名”的结果。
    """
    score = 0.0

    for ch in name:
        codepoint = ord(ch)
        category = unicodedata.category(ch)

        if ch == "\ufffd":
            score -= 20
        elif 0x2500 <= codepoint <= 0x259f:
            score -= 12
        elif category.startswith("C"):
            score -= 10
        elif ch.isalnum():
            score += 2.5
        elif ch in " -_.,()[]{}&'+%":
            score += 1.2
        elif 0x4E00 <= codepoint <= 0x9FFF:
            score += 2.8
        elif 0x0400 <= codepoint <= 0x04FF:
            score += 0.4
        else:
            score += 0.2

    lowered = name.lower()
    if lowered.endswith(".pdf"):
        score += 10
    if "т" in lowered or "а" in lowered and "р" in lowered:
        score -= 4
    if "ablesci.com" in lowered:
        score -= 1

    return score


def plan_renames(
    pdf_dir: Path,
    mappings: list[dict[str, str]],
    encodings: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    plans: list[dict[str, str]] = []
    errors: list[str] = []

    for row in mappings:
        source_name = row.get("new_name") or ""
        broken_name = row.get("old_name") or ""
        doc_id = row.get("doc_id", source_name)

        if not source_name or not broken_name:
            errors.append(f"{doc_id}: 映射项缺少 new_name/old_name")
            continue

        source_path = pdf_dir / source_name
        if not source_path.exists():
            errors.append(f"{doc_id}: 源文件不存在: {source_path.name}")
            continue

        try:
            restored_name, chosen_encoding = decode_broken_name(broken_name, encodings)
            restored_name = ensure_safe_filename(restored_name)
        except Exception as exc:
            errors.append(f"{doc_id}: 恢复文件名失败: {broken_name} ({exc})")
            continue

        target_path = pdf_dir / restored_name
        plans.append(
            {
                "doc_id": doc_id,
                "source_name": source_name,
                "target_name": restored_name,
                "encoding": chosen_encoding,
                "source_path": str(source_path),
                "target_path": str(target_path),
            }
        )

    return plans, errors


def validate_plans(plans: list[dict[str, str]], pdf_dir: Path) -> list[str]:
    errors: list[str] = []
    seen_targets: set[str] = set()
    source_names = {item["source_name"] for item in plans}

    for item in plans:
        target_name = item["target_name"]
        target_path = pdf_dir / target_name

        if target_name in seen_targets:
            errors.append(f"{item['doc_id']}: 目标文件名重复: {target_name}")
            continue
        seen_targets.add(target_name)

        if target_path.exists() and target_name not in source_names:
            errors.append(f"{item['doc_id']}: 目标文件已存在: {target_name}")

    return errors


def apply_renames(plans: list[dict[str, str]], pdf_dir: Path) -> None:
    temp_moves: list[tuple[Path, Path]] = []

    for idx, item in enumerate(plans, start=1):
        source_path = Path(item["source_path"])
        temp_path = pdf_dir / f".__restore_tmp__{idx:08d}{source_path.suffix.lower()}"
        while temp_path.exists():
            idx += 1
            temp_path = pdf_dir / f".__restore_tmp__{idx:08d}{source_path.suffix.lower()}"
        source_path.rename(temp_path)
        temp_moves.append((temp_path, Path(item["target_path"])))

    for temp_path, target_path in temp_moves:
        temp_path.rename(target_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据映射表将 doc_XXXX.pdf 批量恢复为修复后的原始文件名"
    )
    parser.add_argument(
        "--dir",
        default="data/paper_round1/paper",
        help="当前 PDF 目录，默认: data/paper_round1/paper",
    )
    parser.add_argument(
        "--map",
        default="data/paper_round1_pdf_rename_map.csv",
        help="映射表路径，支持 CSV/JSONL，默认: data/paper_round1_pdf_rename_map.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览，不实际改名",
    )
    parser.add_argument(
        "--encodings",
        default=",".join(DEFAULT_ENCODINGS),
        help="按顺序尝试的编码列表，逗号分隔",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.dir).resolve()
    mapping_path = Path(args.map).resolve()
    encodings = [item.strip() for item in args.encodings.split(",") if item.strip()]

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"[ERROR] PDF 目录不存在: {pdf_dir}", file=sys.stderr)
        sys.exit(1)
    if not mapping_path.exists():
        print(f"[ERROR] 映射表不存在: {mapping_path}", file=sys.stderr)
        sys.exit(1)
    if not encodings:
        print("[ERROR] encodings 不能为空", file=sys.stderr)
        sys.exit(1)

    mappings = load_mappings(mapping_path)
    plans, decode_errors = plan_renames(pdf_dir, mappings, encodings)
    validation_errors = validate_plans(plans, pdf_dir)
    errors = decode_errors + validation_errors

    print(f"[INFO] PDF 目录: {pdf_dir}")
    print(f"[INFO] 映射表:   {mapping_path}")
    print(f"[INFO] 候选编码: {', '.join(encodings)}")
    print(f"[INFO] 可执行改名数: {len(plans)}")
    print(f"[INFO] 跳过/报错数: {len(errors)}")
    print()

    for item in plans[:20]:
        print(f"{item['source_name']} -> {item['target_name']} [{item['encoding']}]")
    if len(plans) > 20:
        print(f"... 共 {len(plans)} 个，以上仅显示前 20 个")

    if errors:
        print("\n[WARN] 以下条目被跳过:")
        for item in errors[:20]:
            print(f"  - {item}")
        if len(errors) > 20:
            print(f"  ... 另有 {len(errors) - 20} 条")

    if args.dry_run:
        print("\n[INFO] dry-run 模式，不会实际改名。")
        return

    if errors:
        print("\n[ERROR] 存在跳过/冲突条目，已停止执行。请先处理后重试。", file=sys.stderr)
        sys.exit(1)

    apply_renames(plans, pdf_dir)
    print("\n[OK] 批量改名完成。")


if __name__ == "__main__":
    main()
