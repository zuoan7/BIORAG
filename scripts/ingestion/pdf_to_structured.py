#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 文档清洗与转换脚本
将 PDF 论文转换为结构化 JSON 格式，供 preprocess_and_chunk.py 后续处理。

核心能力:
  - 基于 pymupdf 提取文本，保留页码信息
  - 利用字体大小 + bold 标记识别标题层级
  - 去除页眉页脚、DOI、版权等噪声
  - 修复连字符断词
  - 输出与 preprocess_and_chunk.py 兼容的 JSON 格式

用法:
    # 单文件转换
    python pdf_to_structured.py --input doc_0001.pdf --output ./parsed_papers/

    # 批量转换目录下所有 PDF
    python pdf_to_structured.py --input_dir ./paper_round1/paper/ --output_dir ./parsed_papers/

    # 同时输出 txt 格式（方便人工检查）
    python pdf_to_structured.py --input_dir ./paper_round1/paper/ --output_dir ./parsed_papers/ --also_txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz


# ============================================================
# 常量
# ============================================================

TITLE_SIZE_THRESHOLD = 11.0
HEADING_SIZE_THRESHOLD = 9.5
BODY_SIZE_MIN = 6.0
BODY_SIZE_MAX = 10.5

NOISE_TEXT_PATTERNS = [
    re.compile(r"^A\s+R\s+T\s+I\s+C\s+L\s+E\s+I\s+N\s+F\s+O$", re.I),
    re.compile(r"^A\s+B\s+S\s+T\s+R\s+A\s+C\s+T$", re.I),
    re.compile(r"^H\s+I\s+G\s+H\s+L\s+I\s+G\s+H\s+T\s+S$", re.I),
    re.compile(r"^G\s+R\s+A\s+P\s+H\s+I\s+C\s+A\s+L\s+A\s+B\s+S\s+T\s+R\s+A\s+C\s+T$", re.I),
    re.compile(r"^K\s+E\s+Y\s+W\s+O\s+R\s+D\s+S?\s*:", re.I),
    re.compile(r"^Contents lists available at", re.I),
    re.compile(r"^journal homepage:", re.I),
    re.compile(r"^\d{4}-\d{4}/©\s*\d{4}", re.I),
    re.compile(r"^©\s*\d{4}\s+", re.I),
    re.compile(r"^All rights reserved", re.I),
    re.compile(r"^https?://doi\.org/", re.I),
    re.compile(r"^https?://www\.", re.I),
    re.compile(r"^Received\s+\d", re.I),
    re.compile(r"^Accepted\s+\d", re.I),
    re.compile(r"^Available online\s+", re.I),
    re.compile(r"^Revised\s+\d", re.I),
    re.compile(r"^\*\s*Corresponding\s+author", re.I),
    re.compile(r"^E-mail\s+addresses?", re.I),
    re.compile(r"^Supplementary\s+(?:material|data|information)", re.I),
    re.compile(r"^Please cite this article", re.I),
    re.compile(r"^Downloaded from", re.I),
]

JOURNAL_HEADER_PATTERN = re.compile(
    r"^\s*"
    r"(?:journal\s+of|proceedings\s+of|nature|science|cell|pnas|acs|"
    r"elsevier|springer|wiley|taylor\s+&\s+francis|"
    r"carbohydrate\s+polymers|advanced\s+research|"
    r"biotechnology\s+and\s+bioengineering|metabolic\s+engineering|"
    r"acs\s+synthetic\s+biology|nature\s+communications|"
    r"nature\s+biotechnology|frontiers\s+in|plos\s+one)"
    r"\b",
    re.I,
)

HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f]")
ZERO_WIDTH_CHARS_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")

PAGE_NUM_ONLY_RE = re.compile(r"^\d{1,4}$")

SECTION_HEADER_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?"
    r"(abstract|introduction|background|results?\s+and\s+discussion|"
    r"discussion\s+and\s+results?|results?|discussion|conclusions?|"
    r"materials?\s+and\s+methods|methods?\s+and\s+materials?|methods?|"
    r"experimental\s+(?:procedures?|section|methods?)|"
    r"references|bibliography|literature\s+cited|"
    r"acknowledgements?|funding|supplementary|"
    r"author\s+contributions?|competing\s+interests?|data\s+availability|"
    r"credite?d?\s+authorship|ethics\s+statement)"
    r"\s*$",
    re.I,
)


# ============================================================
# 数据类
# ============================================================

@dataclass
class TextLine:
    text: str
    size: float
    is_bold: bool
    is_italic: bool
    page_num: int
    y_pos: float


@dataclass
class ParsedPage:
    page_num: int
    text: str


# ============================================================
# PDF 文本提取
# ============================================================

def extract_lines_from_page(page: fitz.Page, page_num: int) -> list[TextLine]:
    """
    从单个 PDF 页面提取文本行，保留字体信息。
    """
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    lines: list[TextLine] = []

    for block in blocks:
        if block["type"] != 0:
            continue

        bbox = block.get("bbox", [0, 0, 0, 0])
        block_y = bbox[1]

        for line_dict in block["lines"]:
            line_text = ""
            line_size = 0.0
            line_flags = 0
            span_count = 0
            prev_span = None

            for span in line_dict["spans"]:
                text = span["text"]
                if not text.strip():
                    continue
                if prev_span is not None and _needs_space_between_spans(prev_span, span, line_text, text):
                    line_text += " "
                line_text += text
                line_size = span["size"]
                line_flags = span["flags"]
                span_count += 1
                prev_span = span

            line_text = line_text.strip()
            if not line_text:
                continue

            is_bold = bool(line_flags & (1 << 4))
            is_italic = bool(line_flags & (1 << 1))

            y_pos = line_dict["bbox"][1]

            lines.append(TextLine(
                text=line_text,
                size=line_size,
                is_bold=is_bold,
                is_italic=is_italic,
                page_num=page_num,
                y_pos=y_pos,
            ))

    return lines


def _needs_space_between_spans(
    prev_span: dict,
    curr_span: dict,
    current_text: str,
    next_text: str,
) -> bool:
    """根据 span 间水平距离和边界字符判断是否需要补空格。"""
    if not current_text or not next_text:
        return False

    prev_last = current_text[-1]
    next_first = next_text[0]

    if prev_last.isspace() or next_first.isspace():
        return False

    if prev_last in "([/{-" or next_first in ".,;:!?)]}/%-":
        return False

    prev_bbox = prev_span.get("bbox", [0, 0, 0, 0])
    curr_bbox = curr_span.get("bbox", [0, 0, 0, 0])
    gap = curr_bbox[0] - prev_bbox[2]
    font_size = max(float(prev_span.get("size", 0.0)), float(curr_span.get("size", 0.0)), 1.0)

    if gap > font_size * 0.15:
        return True

    if prev_last.isalnum() and next_first.isalnum() and gap > font_size * 0.02:
        return True

    return False


def is_noise_line(line: TextLine, page_header_footer_cache: dict) -> bool:
    """判断一行是否为噪声行。"""
    text = line.text

    if PAGE_NUM_ONLY_RE.match(text):
        return True

    for pat in NOISE_TEXT_PATTERNS:
        if pat.search(text):
            return True

    if JOURNAL_HEADER_PATTERN.search(text):
        if len(text.split()) <= 12:
            return True

    if line.size < BODY_SIZE_MIN and not line.is_bold:
        return True

    cache_key = (text, round(line.size, 1))
    if cache_key in page_header_footer_cache:
        if page_header_footer_cache[cache_key] >= 3:
            return True

    return False


def detect_header_footers(all_lines: list[TextLine]) -> dict:
    """
    检测重复出现的页眉页脚。
    在多页中重复出现且内容相同的短行视为页眉/页脚。
    """
    line_freq: dict[tuple, int] = {}
    for line in all_lines:
        if len(line.text.split()) > 15:
            continue
        key = (line.text.strip(), round(line.size, 1))
        line_freq[key] = line_freq.get(key, 0) + 1

    return {k: v for k, v in line_freq.items() if v >= 3}


def classify_line(line: TextLine) -> str:
    """
    分类文本行类型。
    返回: "title" | "heading" | "subheading" | "body" | "other"
    
    优先级:
      1. title: 大号字体（论文标题）
      2. heading: bold + 匹配 section 标题模式，或非 bold 但匹配 section 模式且字号偏大
      3. subheading: bold + 编号前缀（如 "2.1. Materials"）或短粗体行
      4. body: 正文字号
    """
    text = line.text.strip()
    word_count = len(text.split())

    if line.size >= TITLE_SIZE_THRESHOLD:
        if word_count >= 2:
            return "title"

    # Bold line: 区分 heading vs subheading
    if line.is_bold and line.size >= BODY_SIZE_MIN:
        if SECTION_HEADER_RE.match(text):
            return "heading"
        if re.match(r"^\d+\.?\s*[A-Z]", text) and word_count <= 15:
            if not re.match(r"^\d+\.\d+", text):
                return "heading"
        if re.match(r"^\d+\.\d+\.?\s*\w+", text) and word_count <= 15:
            return "subheading"
        if line.size >= HEADING_SIZE_THRESHOLD and 2 <= word_count <= 15:
            return "heading"
        if word_count <= 8 and word_count >= 1:
            return "subheading"

    # 非 bold 但匹配 section 标题模式 + 短行 → heading
    # 部分期刊（如 Journal of Advanced Research）的 section 标题不加粗但字号偏大
    if not line.is_bold and line.size >= BODY_SIZE_MIN:
        if SECTION_HEADER_RE.match(text) and word_count <= 10:
            return "heading"
        if re.match(r"^\d+\.?\s*[A-Z]", text) and word_count <= 10:
            if not re.match(r"^\d+\.\d+", text):
                return "heading"

    if line.is_italic and line.is_bold and line.size >= HEADING_SIZE_THRESHOLD:
        if word_count <= 15:
            return "heading"

    if BODY_SIZE_MIN <= line.size <= BODY_SIZE_MAX:
        return "body"

    return "other"


# ============================================================
# 文本清洗与结构化
# ============================================================

def clean_line_text(text: str) -> str:
    """清洗单行文本。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = ZERO_WIDTH_CHARS_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\ufffd", " ")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def merge_consecutive_titles(lines: list[TextLine]) -> list[TextLine]:
    """
    合并连续的 title 行（论文标题经常跨行）。
    同一页内，连续且字号相同的 title 行合并为一行。
    """
    if not lines:
        return lines

    merged: list[TextLine] = []
    for line in lines:
        if (
            merged
            and classify_line(merged[-1]) == "title"
            and classify_line(line) == "title"
            and abs(merged[-1].size - line.size) < 1.0
            and merged[-1].page_num == line.page_num
        ):
            merged[-1] = TextLine(
                text=merged[-1].text + " " + line.text,
                size=merged[-1].size,
                is_bold=merged[-1].is_bold or line.is_bold,
                is_italic=merged[-1].is_italic or line.is_italic,
                page_num=merged[-1].page_num,
                y_pos=merged[-1].y_pos,
            )
        else:
            merged.append(line)

    return merged


def build_structured_text(
    lines: list[TextLine],
    header_footer_cache: dict,
) -> str:
    """
    将提取的文本行构建为结构化文本。
    用 Markdown 风格标记标题层级：
      - ## Section Title（一级标题）
      - ### Subsection Title（二级标题）
    段落之间用空行分隔。
    """
    if not lines:
        return ""

    result_parts: list[str] = []
    prev_type = ""
    current_paragraph: list[str] = []

    for line in lines:
        if is_noise_line(line, header_footer_cache):
            continue

        text = clean_line_text(line.text)
        if not text:
            continue

        line_type = classify_line(line)

        if line_type in ("heading", "subheading"):
            if current_paragraph:
                result_parts.append(" ".join(current_paragraph))
                current_paragraph = []
                result_parts.append("")

            prefix = "##" if line_type == "heading" else "###"
            result_parts.append(f"{prefix} {text}")
            result_parts.append("")
            prev_type = line_type

        elif line_type == "title":
            if current_paragraph:
                result_parts.append(" ".join(current_paragraph))
                current_paragraph = []
                result_parts.append("")
            result_parts.append(f"# {text}")
            result_parts.append("")
            prev_type = "title"

        elif line_type == "body":
            if prev_type in ("heading", "subheading", "title"):
                current_paragraph = []

            if current_paragraph:
                last_text = current_paragraph[-1]
                # 判断上一行是否以句子结束标点结尾
                ends_sentence = bool(re.search(r'[.!?;:)\]"\']\s*$', last_text))
                # 判断当前行是否以大写字母开头（新段落指示）
                starts_capital = bool(re.match(r'^[A-Z]', text))

                if ends_sentence and starts_capital:
                    # 上一行结束句子且当前行以大写开头 → 新段落
                    result_parts.append(" ".join(current_paragraph))
                    result_parts.append("")
                    current_paragraph = [text]
                else:
                    # 继续同一段落
                    current_paragraph.append(text)
            else:
                current_paragraph.append(text)

            prev_type = "body"

        else:
            if current_paragraph:
                result_parts.append(" ".join(current_paragraph))
                current_paragraph = []
                result_parts.append("")
            prev_type = "other"

    if current_paragraph:
        result_parts.append(" ".join(current_paragraph))

    return "\n".join(result_parts).strip()


# ============================================================
# 单文档处理
# ============================================================

def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    also_txt: bool = False,
) -> dict:
    """
    处理单个 PDF 文件，输出结构化 JSON。
    返回处理结果统计。
    """
    doc_id = pdf_path.stem
    source_file = pdf_path.name

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count

    all_lines: list[TextLine] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_lines = extract_lines_from_page(page, page_num + 1)
        all_lines.extend(page_lines)

    header_footer_cache = detect_header_footers(all_lines)

    all_lines = merge_consecutive_titles(all_lines)

    pages_output: list[ParsedPage] = []
    for page_num in range(total_pages):
        page_lines = [l for l in all_lines if l.page_num == page_num + 1]
        page_text = build_structured_text(page_lines, header_footer_cache)
        pages_output.append(ParsedPage(
            page_num=page_num + 1,
            text=page_text,
        ))

    doc.close()

    output_data = {
        "doc_id": doc_id,
        "source_file": source_file,
        "total_pages": total_pages,
        "pages": [
            {"page": p.page_num, "text": p.text}
            for p in pages_output
        ],
    }

    json_path = output_dir / f"{doc_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if also_txt:
        txt_path = output_dir / f"{doc_id}.txt"
        full_text = "\n\n".join(p.text for p in pages_output)
        txt_path.write_text(full_text, encoding="utf-8")

    total_chars = sum(len(p.text) for p in pages_output)
    return {
        "doc_id": doc_id,
        "total_pages": total_pages,
        "total_chars": total_chars,
        "status": "ok",
    }


# ============================================================
# 批量处理
# ============================================================

def batch_process(
    input_dir: Path,
    output_dir: Path,
    also_txt: bool = False,
) -> None:
    """批量处理目录下所有 PDF 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] 目录中没有找到 PDF 文件: {input_dir}")
        return

    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    print()

    success = 0
    failed = 0
    failed_list: list[str] = []

    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"  [{i}/{len(pdf_files)}] {pdf_path.name} ...", end=" ")
        try:
            result = process_pdf(pdf_path, output_dir, also_txt=also_txt)
            print(f"OK ({result['total_pages']} 页, {result['total_chars']} 字符)")
            success += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1
            failed_list.append(f"{pdf_path.name}: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("PDF 转换统计")
    print("=" * 60)
    print(f"总文件数:   {len(pdf_files)}")
    print(f"成功:       {success}")
    print(f"失败:       {failed}")
    print(f"耗时:       {elapsed:.2f}s")
    print(f"输出目录:   {output_dir}")

    if failed_list:
        print()
        print("失败列表:")
        for item in failed_list:
            print(f"  - {item}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PDF 论文清洗与转换脚本"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        help="单个 PDF 文件路径",
    )
    group.add_argument(
        "--input_dir",
        help="PDF 文件所在目录（批量处理）",
    )

    parser.add_argument(
        "--output_dir",
        help="输出目录（默认: 与输入同目录下的 parsed_output）",
    )
    parser.add_argument(
        "--also_txt",
        action="store_true",
        help="同时输出 txt 格式（方便人工检查）",
    )

    args = parser.parse_args()

    if args.input:
        pdf_path = Path(args.input).resolve()
        if not pdf_path.exists():
            print(f"[ERROR] 文件不存在: {pdf_path}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir).resolve() if args.output_dir else pdf_path.parent / "parsed_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"输入: {pdf_path}")
        print(f"输出: {output_dir}")
        print()

        result = process_pdf(pdf_path, output_dir, also_txt=args.also_txt)
        print(f"完成: {result['total_pages']} 页, {result['total_chars']} 字符")
        print(f"输出: {output_dir / (result['doc_id'] + '.json')}")

    elif args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"[ERROR] 目录不存在: {input_dir}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "parsed_output"

        print("=" * 60)
        print("PDF 批量转换")
        print("=" * 60)
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print()

        batch_process(input_dir, output_dir, also_txt=args.also_txt)


if __name__ == "__main__":
    main()
