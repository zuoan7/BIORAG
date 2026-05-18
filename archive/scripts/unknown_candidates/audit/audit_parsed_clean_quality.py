#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parsed_clean 质量审计脚本

遍历 parsed_clean/*.json，输出每个文档的结构质量指标。

用法：
    python scripts/audit/audit_parsed_clean_quality.py data/paper_round1/parsed_clean
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ============================================================
# 断词检测模式
# ============================================================

BROKEN_WORD_PATTERNS = [
    (re.compile(r"Bifido\s+bacterium", re.I), "Bifido bacterium"),
    (re.compile(r"Kleb\s+siella", re.I), "Kleb siella"),
    (re.compile(r"fermenta\s+tion", re.I), "fermenta tion"),
    (re.compile(r"struc\s+turally", re.I), "struc turally"),
    (re.compile(r"pri\s+mary", re.I), "pri mary"),
    (re.compile(r"sup\s+portive", re.I), "sup portive"),
    (re.compile(r"micro\s+biome", re.I), "micro biome"),
    (re.compile(r"re\s+view", re.I), "re view"),
]

# ============================================================
# 误判标题检测模式（只检测以 # 开头且紧跟误判模式的标题）
# ============================================================

FALSE_HEADING_PATTERNS = [
    re.compile(r"^#{1,3}\s+\d+S\s+rRNA", re.I),
    re.compile(r"^#{1,3}\s+\d+F\b"),
    re.compile(r"^#{1,3}\s+\d+R\b"),
    re.compile(r"^#{1,3}\s+13CH", re.I),
    re.compile(r"^#{1,3}\s+1F-β?-?fructofuranosyl", re.I),
    re.compile(r"^#{1,3}\s+\d+F-"),
    re.compile(r"^#{1,3}\s+\d+C-"),
]

# 标题残片检测：heading 以小写连接词开头，说明它可能是被截断的标题残片
HEADING_FRAGMENT_START_WORDS = frozenset({
    "from", "of", "in", "on", "by", "with", "for", "to",
    "and", "or", "the", "a", "an",
})

# 合法的 section 标题白名单
VALID_SECTION_HEADINGS = [
    "abstract", "introduction", "background",
    "materials and methods", "methods", "experimental procedures",
    "experimental methods", "experimental section",
    "results", "results and discussion", "discussion and results",
    "discussion", "conclusion", "conclusions",
    "references", "acknowledgements", "acknowledgments",
    "appendix", "supplementary data", "supplementary materials",
    "supplementary information",
]

VALID_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?("
    + "|".join(re.escape(h) for h in VALID_SECTION_HEADINGS)
    + r")\s*$",
    re.I,
)


def is_heading_fragment(text: str, btype: str) -> bool:
    """检测是否为标题残片（被 PDF 断行截断的标题）"""
    if btype not in ("section_heading", "subsection_heading"):
        return False

    # 去除 Markdown 标记
    stripped = text.lstrip("#").strip()
    if not stripped:
        return False

    # 合法 section 不算残片
    if VALID_SECTION_PATTERN.match(stripped):
        return False

    # 以小写连接词开头的标题很可能是残片
    first_word = stripped.split()[0].lower() if stripped.split() else ""
    if first_word in HEADING_FRAGMENT_START_WORDS:
        return True

    return False


# ============================================================
# 数据类
# ============================================================

@dataclass
class DocQualityReport:
    doc_id: str
    false_heading_count: int = 0
    false_heading_examples: list[str] = None
    split_heading_fragment_count: int = 0
    split_heading_fragment_examples: list[str] = None
    subsection_heading_count: int = 0
    figure_caption_count: int = 0
    table_caption_count: int = 0
    table_text_count: int = 0
    references_block_count: int = 0
    broken_word_count: int = 0
    broken_word_examples: list[str] = None
    empty_block_count: int = 0
    pages_count: int = 0
    block_count: int = 0
    title_count: int = 0
    section_heading_count: int = 0
    paragraph_count: int = 0
    noise_count: int = 0

    def __post_init__(self):
        if self.false_heading_examples is None:
            self.false_heading_examples = []
        if self.broken_word_examples is None:
            self.broken_word_examples = []
        if self.split_heading_fragment_examples is None:
            self.split_heading_fragment_examples = []


# ============================================================
# 审计逻辑
# ============================================================

def audit_document(data: dict) -> DocQualityReport:
    """审计单个 parsed_clean JSON 文档。"""
    doc_id = data.get("doc_id", "unknown")
    pages = data.get("pages", [])
    report = DocQualityReport(doc_id=doc_id)

    for page_data in pages:
        report.pages_count += 1
        blocks = page_data.get("blocks", [])

        for block in blocks:
            btype = block.get("type", "")
            btext = block.get("text", "")
            report.block_count += 1

            if not btext.strip():
                report.empty_block_count += 1
                continue

            # 统计 block 类型
            if btype == "subsection_heading":
                report.subsection_heading_count += 1
            elif btype == "figure_caption":
                report.figure_caption_count += 1
            elif btype == "table_caption":
                report.table_caption_count += 1
            elif btype == "table_text":
                report.table_text_count += 1
            elif btype == "references":
                report.references_block_count += 1
            elif btype == "title":
                report.title_count += 1
            elif btype == "section_heading":
                report.section_heading_count += 1
            elif btype == "paragraph":
                report.paragraph_count += 1
            elif btype == "noise":
                report.noise_count += 1

            # 检测断词
            for pat, name in BROKEN_WORD_PATTERNS:
                matches = pat.findall(btext)
                if matches:
                    report.broken_word_count += len(matches)
                    if len(report.broken_word_examples) < 5:
                        report.broken_word_examples.append(
                            f"[p{page_data.get('page', '?')}] {name}: {btext[:100]}"
                        )

            # 检测误判标题（仅匹配以误判模式开头的标题）
            if btype in ("section_heading", "subsection_heading"):
                for pat in FALSE_HEADING_PATTERNS:
                    if pat.match(btext):
                        report.false_heading_count += 1
                        if len(report.false_heading_examples) < 5:
                            # 提取误判原因
                            reason = "unknown"
                            if re.match(r'^#{1,3}\s+\d+S\s+rRNA', btext, re.I):
                                reason = "16S rRNA 开头"
                            elif re.search(r'\b27F\b', btext):
                                reason = "27F"
                            elif re.search(r'\b1492R\b', btext):
                                reason = "1492R"
                            elif re.search(r'13CH', btext, re.I):
                                reason = "13CH"
                            elif re.search(r'1F-β?-?fructofuranosyl', btext, re.I):
                                reason = "化合物名"
                            report.false_heading_examples.append(
                                f"[p{page_data.get('page', '?')}] {reason}: {btext[:100]}"
                            )
                        break  # 一个 block 只计一次

                # 检测标题残片
                if is_heading_fragment(btext, btype):
                    report.split_heading_fragment_count += 1
                    if len(report.split_heading_fragment_examples) < 5:
                        report.split_heading_fragment_examples.append(
                            f"[p{page_data.get('page', '?')}] {btext[:100]}"
                        )

    return report


# ============================================================
# 批量审计
# ============================================================

def batch_audit(input_dir: Path) -> list[DocQualityReport]:
    """批量审计 parsed_clean 目录。"""
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] 目录中没有找到 JSON 文件: {input_dir}")
        return []

    reports = []
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            report = audit_document(data)
            reports.append(report)
        except Exception as e:
            print(f"[ERROR] 处理 {json_path.name} 失败: {e}")

    return reports


# ============================================================
# 报告输出
# ============================================================

def print_summary(reports: list[DocQualityReport]) -> None:
    """打印汇总报告。"""
    if not reports:
        print("没有可审计的文档。")
        return

    print("=" * 100)
    print("parsed_clean 质量审计报告")
    print("=" * 100)
    print()

    # 表头
    header = (
        f"{'doc_id':<15} "
        f"{'false_hd':>9} "
        f"{'fragment':>8} "
        f"{'subsection':>10} "
        f"{'fig_cap':>7} "
        f"{'tbl_cap':>7} "
        f"{'tbl_txt':>7} "
        f"{'refs':>4} "
        f"{'broken_w':>9} "
        f"{'empty':>5} "
        f"{'pages':>5} "
        f"{'blocks':>6} "
        f"{'title':>5} "
        f"{'sec_hd':>6} "
        f"{'para':>5} "
        f"{'noise':>5}"
    )
    print(header)
    print("-" * 110)

    # 汇总统计
    total_false_headings = 0
    total_split_fragments = 0
    total_subsections = 0
    total_figure_captions = 0
    total_table_captions = 0
    total_table_texts = 0
    total_references = 0
    total_broken_words = 0
    total_empty_blocks = 0
    total_pages = 0
    total_blocks = 0

    # 按断词数排序
    sorted_reports = sorted(reports, key=lambda r: r.broken_word_count, reverse=True)

    for r in sorted_reports:
        row = (
            f"{r.doc_id:<15} "
            f"{r.false_heading_count:>9} "
            f"{r.split_heading_fragment_count:>8} "
            f"{r.subsection_heading_count:>10} "
            f"{r.figure_caption_count:>7} "
            f"{r.table_caption_count:>7} "
            f"{r.table_text_count:>7} "
            f"{r.references_block_count:>4} "
            f"{r.broken_word_count:>9} "
            f"{r.empty_block_count:>5} "
            f"{r.pages_count:>5} "
            f"{r.block_count:>6} "
            f"{r.title_count:>5} "
            f"{r.section_heading_count:>6} "
            f"{r.paragraph_count:>5} "
            f"{r.noise_count:>5}"
        )
        print(row)

        total_false_headings += r.false_heading_count
        total_split_fragments += r.split_heading_fragment_count
        total_subsections += r.subsection_heading_count
        total_figure_captions += r.figure_caption_count
        total_table_captions += r.table_caption_count
        total_table_texts += r.table_text_count
        total_references += r.references_block_count
        total_broken_words += r.broken_word_count
        total_empty_blocks += r.empty_block_count
        total_pages += r.pages_count
        total_blocks += r.block_count

    print("-" * 110)
    summary = (
        f"{'TOTAL':<15} "
        f"{total_false_headings:>9} "
        f"{total_split_fragments:>8} "
        f"{total_subsections:>10} "
        f"{total_figure_captions:>7} "
        f"{total_table_captions:>7} "
        f"{total_table_texts:>7} "
        f"{total_references:>4} "
        f"{total_broken_words:>9} "
        f"{total_empty_blocks:>5} "
        f"{total_pages:>5} "
        f"{total_blocks:>6}"
    )
    print(summary)
    print()

    # 输出异常文档
    docs_with_false_headings = [r for r in reports if r.false_heading_count > 0]
    docs_with_split_fragments = [r for r in reports if r.split_heading_fragment_count > 0]
    docs_with_broken_words = [r for r in reports if r.broken_word_count > 0]
    docs_with_no_subsections = [r for r in reports if r.subsection_heading_count == 0]
    docs_with_no_figure_captions = [r for r in reports if r.figure_caption_count == 0]
    docs_with_no_table_captions = [r for r in reports if r.table_caption_count == 0]

    print("=" * 60)
    print("诊断摘要")
    print("=" * 60)
    print(f"文档总数:                    {len(reports)}")
    print(f"含误判标题的文档:            {len(docs_with_false_headings)}")
    print(f"含标题残片的文档:            {len(docs_with_split_fragments)}")
    print(f"含断词的文档:                {len(docs_with_broken_words)}")
    print(f"无 subsection 的文档:        {len(docs_with_no_subsections)}")
    print(f"无 figure_caption 的文档:    {len(docs_with_no_figure_captions)}")
    print(f"无 table_caption 的文档:     {len(docs_with_no_table_captions)}")
    print()

    if docs_with_broken_words:
        print("含断词的文档（前 10）:")
        for r in docs_with_broken_words[:10]:
            print(f"  {r.doc_id}: {r.broken_word_count} 处")
            for ex in r.broken_word_examples[:3]:
                print(f"    - {ex}")
        print()

    if docs_with_split_fragments:
        print("含标题残片的文档（前 10）:")
        for r in docs_with_split_fragments[:10]:
            print(f"  {r.doc_id}: {r.split_heading_fragment_count} 处")
            for ex in r.split_heading_fragment_examples[:3]:
                print(f"    - {ex}")
        print()

    if docs_with_false_headings:
        print("含误判标题的文档（前 10）:")
        for r in docs_with_false_headings[:10]:
            print(f"  {r.doc_id}: {r.false_heading_count} 处")
            for ex in r.false_heading_examples[:3]:
                print(f"    - {ex}")
        print()


def save_reports(reports: list[DocQualityReport], output_dir: Path) -> None:
    """保存报告到文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 格式
    json_path = output_dir / "parsed_clean_quality.json"
    json_data = [asdict(r) for r in reports]
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON 报告已保存: {json_path}")

    # TXT 格式
    txt_path = output_dir / "parsed_clean_quality.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        import io
        old_stdout = sys.stdout
        sys.stdout = f
        print_summary(reports)
        sys.stdout = old_stdout
    print(f"TXT 报告已保存: {txt_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="parsed_clean 质量审计脚本"
    )
    parser.add_argument(
        "input_dir",
        help="parsed_clean 目录路径",
    )
    parser.add_argument(
        "--report_dir",
        default="reports",
        help="报告输出目录（默认: reports）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    report_dir = Path(args.report_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"审计目录: {input_dir}")
    print()

    reports = batch_audit(input_dir)
    print_summary(reports)
    save_reports(reports, report_dir)


if __name__ == "__main__":
    main()
