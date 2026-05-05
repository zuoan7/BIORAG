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
from typing import Any, Optional

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
UNICODE_SPACE_CHARS_RE = re.compile(r"[\u00a0\u202f\u2009\u2007]")
DASH_CHARS_RE = re.compile(r"[\u2013\u2014\u2212]")

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

JOURNAL_PREPROOF_NOISE_PATTERNS = [
    re.compile(r"^This is a PDF file of an article", re.I),
    re.compile(r"^PII:\s*", re.I),
    re.compile(r"^DOI:\s*", re.I),
    re.compile(r"^Reference:\s*", re.I),
    re.compile(r"^To appear in:\s*", re.I),
    re.compile(r"^Received Date:\s*", re.I),
    re.compile(r"^Revised Date:\s*", re.I),
    re.compile(r"^Accepted Date:\s*", re.I),
    re.compile(r"^Please cite this article as:", re.I),
    re.compile(r"^This manuscript has been accepted", re.I),
    re.compile(r"^The manuscript will undergo copyediting", re.I),
]

JOURNAL_PREPROOF_EXACT_RE = re.compile(r"^journal pre-proofs?$")

FRONT_MATTER_STOP_RE = re.compile(r"^\s*(?:#+\s*)?(Abstract|Introduction)\b", re.I)

FIGURE_CAPTION_CANDIDATE_RE = re.compile(
    r"^(?:Supplementary\s+)?(?:Fig\.?|Figure)\s+S?\d+[A-Za-z]?"
    r"(?:\s*[\.\:\-]|\s|$)",
    re.I,
)

TABLE_CAPTION_CANDIDATE_RE = re.compile(
    r"^(?:Supplementary\s+)?Table\s+S?\d+[A-Za-z]?"
    r"(?:\s*[\.\:\-]|\s|$)",
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
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    block_no: int = -1
    line_no: int = -1
    column: str = "UNK"
    block_type: str = "text"


@dataclass
class ParsedBlock:
    block_id: str
    type: str
    text: str
    bbox: list[float]
    column: str
    reading_order: int
    page: int
    size: Optional[float] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    block_no: Optional[int] = None
    line_no: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ParsedPage:
    page_num: int
    text: str
    blocks: list[ParsedBlock]


# ============================================================
# PDF 文本提取
# ============================================================

def _clean_bbox(bbox: Any) -> tuple[float, float, float, float]:
    """将 PyMuPDF bbox 统一为四元 float tuple。"""
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(round(float(v), 2) for v in bbox[:4])  # type: ignore[return-value]


def guess_column(x0: float, x1: float, page_width: float) -> str:
    """根据 bbox 横向位置保守判断 L/R/SPAN/UNK。"""
    if page_width <= 0 or x1 <= x0:
        return "UNK"

    mid = page_width / 2.0
    center = (x0 + x1) / 2.0
    width = x1 - x0
    gutter = max(18.0, min(34.0, page_width * 0.05))

    if abs(center - mid) <= gutter and width < page_width * 0.18:
        return "UNK"
    if width >= page_width * 0.62 or (x0 < mid - gutter and x1 > mid + gutter):
        return "SPAN"
    if x1 <= mid + gutter and center < mid:
        return "L"
    if x0 >= mid - gutter and center > mid:
        return "R"
    return "UNK"


def extract_lines_from_page(page: fitz.Page, page_num: int) -> list[TextLine]:
    """
    从单个 PDF 页面提取文本行，保留字体信息。
    """
    lines, _image_blocks = extract_layout_from_page(page, page_num)
    return lines


def extract_layout_from_page(page: fitz.Page, page_num: int) -> tuple[list[TextLine], list[ParsedBlock]]:
    """
    从单个 PDF 页面提取文本行和 image block 占位。
    image block 不做 OCR，只保留 bbox/页码/类型，供 parsed_raw_v4 审计使用。
    """
    page_width = float(page.rect.width)
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    lines, image_blocks = extract_layout_from_blocks(blocks, page_num, page_width)
    image_blocks.extend(
        extract_pymupdf_image_blocks(
            page,
            page_num,
            page_width,
            existing_image_blocks=image_blocks,
            start_index=len(image_blocks) + 1,
        )
    )
    return lines, image_blocks


def extract_layout_from_blocks(
    blocks: list[dict],
    page_num: int,
    page_width: float,
) -> tuple[list[TextLine], list[ParsedBlock]]:
    """从 PyMuPDF dict blocks 提取 v4 layout-aware 中间结构，便于单元测试 mock。"""
    lines: list[TextLine] = []
    image_blocks: list[ParsedBlock] = []
    image_count = 0

    for block_index, block in enumerate(blocks):
        block_type = block.get("type", 0)
        block_no = int(block.get("number", block_index))
        bbox = _clean_bbox(block.get("bbox", [0, 0, 0, 0]))

        if block_type != 0:
            image_count += 1
            image_blocks.append(ParsedBlock(
                block_id=f"p{page_num}_img{image_count:04d}",
                type="image",
                text="",
                bbox=list(bbox),
                column="SPAN",
                reading_order=0,
                page=page_num,
                block_no=block_no,
                metadata=_image_metadata_from_text_dict_block(block),
            ))
            continue

        for line_index, line_dict in enumerate(block.get("lines", [])):
            line_text = ""
            line_size = 0.0
            line_flags = 0
            prev_span = None

            for span in line_dict.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                if prev_span is not None and _needs_space_between_spans(prev_span, span, line_text, text):
                    line_text += " "
                line_text += text
                line_size = float(span.get("size", line_size))
                line_flags = int(span.get("flags", line_flags))
                prev_span = span

            line_text = clean_line_text(line_text)
            if not line_text:
                continue

            is_bold = bool(line_flags & (1 << 4))
            is_italic = bool(line_flags & (1 << 1))

            line_bbox = _clean_bbox(line_dict.get("bbox", bbox))
            column = guess_column(line_bbox[0], line_bbox[2], page_width)

            lines.append(TextLine(
                text=line_text,
                size=line_size,
                is_bold=is_bold,
                is_italic=is_italic,
                page_num=page_num,
                y_pos=line_bbox[1],
                bbox=line_bbox,
                block_no=block_no,
                line_no=line_index,
                column=column,
            ))

    return lines, image_blocks


def _image_metadata_from_text_dict_block(block: dict[str, Any]) -> dict[str, Any]:
    """Keep only compact scalar PyMuPDF image metadata; never serialize image bytes."""
    metadata: dict[str, Any] = {"image_source": "text_dict_block"}
    for key in (
        "width",
        "height",
        "ext",
        "colorspace",
        "xres",
        "yres",
        "bpc",
        "transform",
    ):
        value = block.get(key)
        if value is None:
            continue
        if key == "transform" and hasattr(value, "__iter__"):
            metadata[key] = [round(float(v), 4) for v in value]
        elif isinstance(value, (str, int, float, bool)):
            metadata[f"image_{key}" if key in {"width", "height"} else key] = value
    return metadata


def _bbox_area(bbox: list[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = _bbox_area([x0, y0, x1, y1])
    if intersection <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _same_image_bbox(a: list[float], b: list[float], tolerance: float = 2.0) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    if all(abs(a[i] - b[i]) <= tolerance for i in range(4)):
        return True
    return _bbox_iou(a, b) >= 0.92


def _image_tuple_metadata(image: tuple[Any, ...]) -> dict[str, Any]:
    """Map PyMuPDF get_images(full=True) tuples to compact metadata."""
    metadata: dict[str, Any] = {"image_source": "page_get_images"}
    names = [
        "xref",
        "smask",
        "image_width",
        "image_height",
        "bpc",
        "colorspace",
        "alt_colorspace",
        "image_name",
        "image_filter",
        "referencer",
    ]
    for idx, name in enumerate(names):
        if idx >= len(image):
            break
        value = image[idx]
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[name] = value
    return metadata


def extract_pymupdf_image_blocks(
    page: fitz.Page,
    page_num: int,
    page_width: float,
    *,
    existing_image_blocks: Optional[list[ParsedBlock]] = None,
    start_index: int = 1,
) -> list[ParsedBlock]:
    """
    Extract image placements with PyMuPDF get_images/get_image_rects.
    This is metadata-only: no OCR and no image bytes are serialized.
    """
    image_blocks: list[ParsedBlock] = []
    image_count = max(1, start_index)

    for image in page.get_images(full=True):
        metadata = _image_tuple_metadata(image)
        xref = metadata.get("xref")
        if not isinstance(xref, int):
            continue
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect_index, rect in enumerate(rects, start=1):
            bbox = list(_clean_bbox(rect))
            if _bbox_area(bbox) <= 0:
                continue
            duplicate_block = next(
                (
                    block for block in list(existing_image_blocks or []) + image_blocks
                    if _same_image_bbox(bbox, block.bbox)
                ),
                None,
            )
            if duplicate_block is not None:
                merged_metadata = dict(duplicate_block.metadata or {})
                merged_metadata.update({k: v for k, v in metadata.items() if k not in merged_metadata})
                sources = {str(merged_metadata.get("image_source", "")), "page_get_images"}
                sources.discard("")
                merged_metadata["image_source"] = "+".join(sorted(sources))
                merged_metadata["image_rect_index"] = rect_index
                duplicate_block.metadata = merged_metadata
                continue
            block_metadata = dict(metadata)
            block_metadata["image_rect_index"] = rect_index
            image_blocks.append(ParsedBlock(
                block_id=f"p{page_num}_img{image_count:04d}",
                type="image",
                text="",
                bbox=bbox,
                column=guess_column(bbox[0], bbox[2], page_width),
                reading_order=0,
                page=page_num,
                metadata=block_metadata,
            ))
            image_count += 1

    return image_blocks


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
    text = UNICODE_SPACE_CHARS_RE.sub(" ", text)
    text = DASH_CHARS_RE.sub("-", text)
    text = text.replace("\u00ad", "")
    text = ZERO_WIDTH_CHARS_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\ufffd", " ")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_pdf_noise_text(text: str) -> str:
    """用于噪声规则的保守归一化 key。"""
    text = unicodedata.normalize("NFKC", text)
    text = UNICODE_SPACE_CHARS_RE.sub(" ", text)
    text = DASH_CHARS_RE.sub("-", text)
    text = ZERO_WIDTH_CHARS_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lstrip("#").strip().lower()


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
                bbox=(
                    min(merged[-1].bbox[0], line.bbox[0]),
                    min(merged[-1].bbox[1], line.bbox[1]),
                    max(merged[-1].bbox[2], line.bbox[2]),
                    max(merged[-1].bbox[3], line.bbox[3]),
                ),
                block_no=merged[-1].block_no,
                line_no=merged[-1].line_no,
                column=merged[-1].column if merged[-1].column == line.column else "SPAN",
            )
        else:
            merged.append(line)

    return merged


def is_figure_caption_candidate(text: str) -> bool:
    """宽松识别 Figure/Fig caption 候选行，仅用于统计和审计。"""
    return bool(FIGURE_CAPTION_CANDIDATE_RE.match(clean_line_text(text)))


def is_table_caption_candidate(text: str) -> bool:
    """宽松识别 Table caption 候选行，仅用于统计和审计。"""
    return bool(TABLE_CAPTION_CANDIDATE_RE.match(clean_line_text(text)))


def collect_repeated_journal_preproof_keys(all_lines: list[TextLine]) -> set[str]:
    """
    收集跨页重复的 Journal Pre-proof key。
    只针对明确包含 journal pre-proof 的短文本，不泛化到 DOI/copyright/reference。
    """
    pages_by_key: dict[str, set[int]] = {}
    for line in all_lines:
        normalized = normalize_pdf_noise_text(line.text)
        if "journal pre-proof" not in normalized:
            continue
        if len(normalized) > 120:
            continue
        pages_by_key.setdefault(normalized, set()).add(line.page_num)
    return {
        key for key, pages in pages_by_key.items()
        if len(pages) >= 3
    }


def _is_header_like_journal_preproof(line: TextLine, normalized: str) -> bool:
    if "journal pre-proof" not in normalized:
        return False
    if len(normalized) > 80:
        return False
    y0 = line.bbox[1]
    return (
        y0 < 160
        or y0 > 620
        or line.size >= 20
        or line.column == "SPAN"
    )


def should_strip_journal_preproof_noise(
    line: TextLine,
    *,
    page_num: int,
    total_pages: int,
    front_matter_active: bool,
    repeated_journal_preproof_keys: set[str],
) -> tuple[bool, str]:
    """统一判断 Journal Pre-proof 噪声是否应从 text 和 blocks 中剥离。"""
    del total_pages  # 当前规则不依赖总页数，保留参数便于诊断调用一致。
    normalized = normalize_pdf_noise_text(line.text)

    if JOURNAL_PREPROOF_EXACT_RE.match(normalized):
        return True, "journal_preproof_exact_line"

    if "journal pre-proof" in normalized:
        if normalized in repeated_journal_preproof_keys:
            return True, "journal_preproof_repeated_header"
        if _is_header_like_journal_preproof(line, normalized):
            return True, "journal_preproof_running_header"
        return False, ""

    if front_matter_active:
        text = clean_line_text(line.text)
        if any(pat.search(text) for pat in JOURNAL_PREPROOF_NOISE_PATTERNS):
            if re.match(r"^(?:PII|DOI|Reference|To appear in|Received Date|Revised Date|Accepted Date):", text, re.I):
                return True, "journal_preproof_metadata_line"
            return True, "journal_preproof_front_matter_disclaimer"

    return False, ""


def strip_journal_preproof_noise(
    lines: list[TextLine],
    page_num: int,
    total_pages: int = 0,
    repeated_journal_preproof_keys: Optional[set[str]] = None,
) -> tuple[list[TextLine], dict]:
    """
    保守移除 Journal Pre-proof 噪声。
    遇到 Abstract/Introduction 后停止强过滤，避免误删正文。
    """
    repeated_journal_preproof_keys = repeated_journal_preproof_keys or set()
    diagnostics = {
        "page": page_num,
        "stripped_count": 0,
        "examples": [],
        "reason_counts": {},
    }

    filtered: list[TextLine] = []
    in_front_matter = page_num <= 2
    for line in lines:
        text = clean_line_text(line.text)
        if in_front_matter and FRONT_MATTER_STOP_RE.match(text):
            in_front_matter = False

        should_strip, reason = should_strip_journal_preproof_noise(
            line,
            page_num=page_num,
            total_pages=total_pages,
            front_matter_active=in_front_matter,
            repeated_journal_preproof_keys=repeated_journal_preproof_keys,
        )
        if should_strip:
            diagnostics["stripped_count"] += 1
            diagnostics["reason_counts"][reason] = diagnostics["reason_counts"].get(reason, 0) + 1
            if len(diagnostics["examples"]) < 8:
                diagnostics["examples"].append(text[:160])
            continue

        filtered.append(line)

    return filtered, diagnostics


def _line_counts_by_column(lines: list[TextLine]) -> dict[str, int]:
    counts = {"L": 0, "R": 0, "SPAN": 0, "UNK": 0}
    for line in lines:
        counts[line.column if line.column in counts else "UNK"] += 1
    return counts


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _body_like_lines(lines: list[TextLine]) -> list[TextLine]:
    return [
        line for line in lines
        if len(line.text.split()) >= 3
        and BODY_SIZE_MIN <= line.size <= BODY_SIZE_MAX
        and not PAGE_NUM_ONLY_RE.match(line.text)
    ]


def _is_references_page(lines: list[TextLine]) -> bool:
    lower_texts = [clean_line_text(line.text).lower() for line in lines]
    return any(text in {"references", "bibliography", "literature cited"} for text in lower_texts)


def _is_front_matter_like(lines: list[TextLine], page_num: int) -> bool:
    if page_num <= 1:
        return True
    if page_num > 2:
        return False

    body_like = _body_like_lines(lines)
    title_like_count = sum(1 for line in lines if line.size >= TITLE_SIZE_THRESHOLD and len(line.text.split()) >= 2)
    lower_text = "\n".join(clean_line_text(line.text).lower() for line in lines[:25])
    return title_like_count >= 2 or ("abstract" in lower_text and len(body_like) < 16)


def _caption_count(lines: list[TextLine]) -> int:
    return sum(
        1 for line in lines
        if is_figure_caption_candidate(line.text) or is_table_caption_candidate(line.text)
    )


def _is_figure_table_heavy_page(lines: list[TextLine], body_line_count: int) -> bool:
    caption_count = _caption_count(lines)
    if caption_count >= 3:
        return True
    return caption_count >= 2 and body_line_count < 24


def _line_width(line: TextLine) -> float:
    return max(0.0, float(line.bbox[2]) - float(line.bbox[0]))


def _is_region_anchor_line(line: TextLine, page_width: float, page_height: float) -> bool:
    """
    Identify conservative cross-column boundaries for mixed two-column pages.

    Anchors split the page into vertical regions; ordinary short SPAN fragments are
    intentionally ignored because PyMuPDF sometimes gives line fragments broad bboxes.
    """
    text = clean_line_text(line.text)
    if not text or PAGE_NUM_ONLY_RE.match(text):
        return False

    width = _line_width(line)
    word_count = len(text.split())
    if is_figure_caption_candidate(text) or is_table_caption_candidate(text):
        return True

    line_type = classify_line(line)
    if line_type in {"title", "heading", "subheading"} and word_count <= 22:
        return True

    if width >= page_width * 0.68 and 3 <= word_count <= 35:
        return bool(line.is_bold or line.size >= HEADING_SIZE_THRESHOLD)

    return False


def _region_anchor_lines(lines: list[TextLine], page_width: float, page_height: float) -> list[TextLine]:
    return [
        line for line in lines
        if line.column not in {"L", "R"}
        and _is_region_anchor_line(line, page_width, page_height)
    ]


def compute_page_layout_features(lines: list[TextLine], page_width: float, page_height: float) -> dict:
    """提取双栏判断和 audit 共用的页面 layout 特征。"""
    body_like = _body_like_lines(lines)
    counts = _line_counts_by_column(body_like)
    page_num = lines[0].page_num if lines else 0
    left_x1 = [line.bbox[2] for line in body_like if line.column == "L"]
    right_x0 = [line.bbox[0] for line in body_like if line.column == "R"]
    left_x0 = [line.bbox[0] for line in body_like if line.column == "L"]
    right_x1 = [line.bbox[2] for line in body_like if line.column == "R"]
    left_median_x1 = _median(left_x1)
    right_median_x0 = _median(right_x0)
    gap = None
    if left_median_x1 is not None and right_median_x0 is not None:
        gap = right_median_x0 - left_median_x1

    body_line_count = len(body_like)
    span_ratio = counts["SPAN"] / max(body_line_count, 1)

    return {
        "page": page_num,
        "body_line_count": body_line_count,
        "left_line_count": counts["L"],
        "right_line_count": counts["R"],
        "span_line_count": counts["SPAN"],
        "unk_line_count": counts["UNK"],
        "left_x_median": _median(left_x0),
        "right_x_median": _median(right_x1),
        "left_x1_median": left_median_x1,
        "right_x0_median": right_median_x0,
        "column_gap_estimate": gap,
        "span_line_ratio": round(span_ratio, 3),
        "page_width": page_width,
        "page_height": page_height,
        "is_references_page": _is_references_page(lines),
        "is_front_matter_like": _is_front_matter_like(lines, page_num),
        "is_figure_table_heavy_page": _is_figure_table_heavy_page(lines, body_line_count),
        "region_anchor_count": len(_region_anchor_lines(lines, page_width, page_height)),
    }


def _has_relaxed_two_column_signal(features: dict) -> bool:
    page_width = float(features.get("page_width") or 0.0)
    gap = features.get("column_gap_estimate")
    if page_width <= 0 or gap is None:
        return False
    if features["body_line_count"] < 10:
        return False
    if features["left_line_count"] < 4 or features["right_line_count"] < 4:
        return False
    if features["span_line_ratio"] > 0.68:
        return False
    if float(gap) < -page_width * 0.01:
        return False

    left_x = features.get("left_x_median")
    right_x = features.get("right_x_median")
    if left_x is None or right_x is None:
        return False
    if float(left_x) > page_width * 0.45 or float(right_x) < page_width * 0.55:
        return False
    return True


def _has_strict_two_column_signal(features: dict) -> tuple[bool, str]:
    if features["body_line_count"] < 16:
        return False, "not_enough_body_lines"
    if features["is_references_page"]:
        return False, "references_page"
    if features["is_figure_table_heavy_page"]:
        return False, "figure_table_heavy_page"
    if features["left_line_count"] < 8 or features["right_line_count"] < 8:
        return False, "insufficient_left_right_lines"
    if features["span_line_ratio"] > 0.35:
        return False, "too_many_span_lines"

    page_width = float(features["page_width"])
    gap = features.get("column_gap_estimate")
    if gap is None:
        return False, "missing_column_x_distribution"
    if float(gap) < page_width * 0.035:
        return False, "weak_column_gap"

    left_x1 = features.get("left_x1_median")
    right_x0 = features.get("right_x0_median")
    if left_x1 is None or right_x0 is None:
        return False, "missing_column_x_distribution"
    if float(left_x1) > page_width * 0.62 or float(right_x0) < page_width * 0.38:
        return False, "weak_left_right_distribution"

    return True, "left_right_distribution_clear"


def analyze_page_layout(
    lines: list[TextLine],
    page_width: float,
    page_height: float,
    *,
    document_two_column_prior: bool = False,
) -> dict:
    """生成双栏判断的诊断结果。"""
    features = compute_page_layout_features(lines, page_width, page_height)
    is_strict_two_column, strict_reason = _has_strict_two_column_signal(features)
    diagnostic = {
        **features,
        "is_two_column": False,
        "selected_order_strategy": "single_column_yx",
        "reason": strict_reason,
        "document_two_column_prior": document_two_column_prior,
        "likely_two_column_but_fallback": False,
        "fallback_reason": "",
    }

    if document_two_column_prior and features["is_front_matter_like"]:
        diagnostic["reason"] = "document_prior_but_exempt_front_matter"
        diagnostic["fallback_reason"] = strict_reason
        return diagnostic

    if is_strict_two_column:
        diagnostic["is_two_column"] = True
        diagnostic["selected_order_strategy"] = (
            "two_column_region_mixed" if features["region_anchor_count"] else "two_column_span_left_right"
        )
        return diagnostic

    relaxed_signal = _has_relaxed_two_column_signal(features)
    if document_two_column_prior:
        if features["is_references_page"]:
            diagnostic["reason"] = "document_prior_but_exempt_references"
            diagnostic["fallback_reason"] = strict_reason
            return diagnostic
        if features["is_figure_table_heavy_page"]:
            if relaxed_signal and features["region_anchor_count"] and features["body_line_count"] >= 12:
                diagnostic["is_two_column"] = True
                diagnostic["selected_order_strategy"] = "two_column_region_mixed"
                diagnostic["reason"] = "document_prior_region_mixed_figure_table"
                diagnostic["fallback_reason"] = strict_reason
                return diagnostic
            diagnostic["reason"] = "document_prior_but_exempt_figure_table"
            diagnostic["fallback_reason"] = strict_reason
            return diagnostic
        if features["body_line_count"] < 10:
            diagnostic["reason"] = "document_prior_but_not_enough_text"
            diagnostic["fallback_reason"] = strict_reason
            return diagnostic
        if relaxed_signal:
            diagnostic["is_two_column"] = True
            diagnostic["selected_order_strategy"] = (
                "two_column_region_mixed" if features["region_anchor_count"] else "two_column_span_left_right"
            )
            diagnostic["reason"] = "document_prior_relaxed_two_column"
            diagnostic["fallback_reason"] = strict_reason
            return diagnostic

    if relaxed_signal:
        diagnostic["likely_two_column_but_fallback"] = True
        diagnostic["fallback_reason"] = strict_reason
        diagnostic["reason"] = "likely_two_column_but_fallback"
    elif strict_reason == "":
        diagnostic["reason"] = "single_column_confirmed"

    return diagnostic


def is_likely_two_column_page(lines: list[TextLine], page_width: float, page_height: float) -> bool:
    """公开给测试使用的保守双栏判断。"""
    return bool(analyze_page_layout(lines, page_width, page_height)["is_two_column"])


def analyze_document_two_column_prior(page_layout_diagnostics: list[dict]) -> dict:
    """基于正文页横向簇稳定性生成文档级双栏先验。"""
    candidate_pages = [
        item for item in page_layout_diagnostics
        if item.get("page", 0) > 1
        and not item.get("is_front_matter_like")
        and not item.get("is_references_page")
        and not item.get("is_figure_table_heavy_page")
        and item.get("body_line_count", 0) >= 10
    ]
    strict_pages = [item for item in candidate_pages if item.get("is_two_column")]
    relaxed_pages = [item for item in candidate_pages if _has_relaxed_two_column_signal(item)]

    confidence = 0.0
    if candidate_pages:
        confidence = len(relaxed_pages) / len(candidate_pages)
        if len(strict_pages) >= 2:
            confidence = max(confidence, 0.75)

    prior = (
        len(strict_pages) >= 2
        or (len(relaxed_pages) >= 3 and confidence >= 0.25)
        or (len(candidate_pages) <= 4 and len(relaxed_pages) >= 2 and confidence >= 0.5)
    )

    left_medians = [float(item["left_x_median"]) for item in relaxed_pages if item.get("left_x_median") is not None]
    right_medians = [float(item["right_x_median"]) for item in relaxed_pages if item.get("right_x_median") is not None]
    page_widths = [float(item["page_width"]) for item in relaxed_pages if item.get("page_width")]
    stable_columns = False
    if len(left_medians) >= 2 and len(right_medians) >= 2 and page_widths:
        width = _median(page_widths) or page_widths[0]
        stable_columns = (
            max(left_medians) - min(left_medians) <= width * 0.16
            and max(right_medians) - min(right_medians) <= width * 0.16
        )
        if not stable_columns and len(strict_pages) < 2:
            prior = False

    return {
        "document_two_column_prior": prior,
        "document_two_column_confidence": round(confidence, 3),
        "candidate_body_pages": [item["page"] for item in candidate_pages],
        "strict_two_column_signal_pages": [item["page"] for item in strict_pages],
        "relaxed_two_column_signal_pages": [item["page"] for item in relaxed_pages],
        "stable_column_x_clusters": stable_columns,
    }


def sort_lines_reading_order(
    lines: list[TextLine],
    page_width: float,
    page_height: float,
    *,
    layout_diagnostic: Optional[dict] = None,
    document_two_column_prior: bool = False,
) -> list[TextLine]:
    """按页面布局返回文本行阅读顺序。"""
    if not lines:
        return []

    diagnostic = layout_diagnostic or analyze_page_layout(
        lines,
        page_width,
        page_height,
        document_two_column_prior=document_two_column_prior,
    )
    if not diagnostic["is_two_column"]:
        return sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))

    return sort_two_column_region_reading_order(lines, page_width, page_height)


def _line_yx_key(line: TextLine) -> tuple[float, float]:
    return (line.bbox[1], line.bbox[0])


def _sort_two_column_region_lines(
    lines: list[TextLine],
    page_width: float,
    page_height: float,
) -> list[TextLine]:
    if not lines:
        return []

    left = [line for line in lines if line.column == "L"]
    right = [line for line in lines if line.column == "R"]
    other = [line for line in lines if line.column not in {"L", "R"}]
    if len(left) < 2 or len(right) < 2:
        return sorted(lines, key=_line_yx_key)

    column_lines = [line for line in lines if line.column in {"L", "R"}]
    min_column_y = min((line.bbox[1] for line in column_lines), default=0.0)
    max_column_y = max((line.bbox[3] for line in column_lines), default=page_height)
    top_threshold = min_column_y + page_height * 0.04

    top_other = [line for line in other if line.bbox[1] <= top_threshold]
    bottom_other = [line for line in other if line not in top_other or line.bbox[1] >= max_column_y]

    return (
        sorted(top_other, key=_line_yx_key)
        + sorted(left, key=_line_yx_key)
        + sorted(right, key=_line_yx_key)
        + sorted(bottom_other, key=_line_yx_key)
    )


def _group_region_anchor_bands(anchors: list[TextLine]) -> list[list[TextLine]]:
    if not anchors:
        return []
    bands: list[list[TextLine]] = []
    current: list[TextLine] = []
    current_y1 = 0.0
    for anchor in sorted(anchors, key=_line_yx_key):
        if current and anchor.bbox[1] > current_y1 + 8.0:
            bands.append(current)
            current = []
        current.append(anchor)
        current_y1 = max(current_y1, anchor.bbox[3])
    if current:
        bands.append(current)
    return bands


def sort_two_column_region_reading_order(
    lines: list[TextLine],
    page_width: float,
    page_height: float,
) -> list[TextLine]:
    """
    Sort mixed two-column pages by vertical regions.

    Wide SPAN headings/captions become region anchors, so a middle caption/table
    heading stays between the text regions above and below it instead of being
    appended after the full left/right columns.
    """
    anchors = _region_anchor_lines(lines, page_width, page_height)
    if not anchors:
        return _sort_two_column_region_lines(lines, page_width, page_height)

    anchor_ids = {id(line) for line in anchors}
    remaining = sorted([line for line in lines if id(line) not in anchor_ids], key=_line_yx_key)
    output: list[TextLine] = []
    cursor_y = -1.0

    for band in _group_region_anchor_bands(anchors):
        band_y0 = min(line.bbox[1] for line in band)
        band_y1 = max(line.bbox[3] for line in band)
        before = [line for line in remaining if cursor_y <= line.bbox[1] < band_y0]
        if before:
            output.extend(_sort_two_column_region_lines(before, page_width, page_height))
        output.extend(sorted(band, key=_line_yx_key))
        overlap = [line for line in remaining if band_y0 <= line.bbox[1] <= band_y1 + 1.0]
        if overlap:
            output.extend(sorted(overlap, key=_line_yx_key))
        consumed_ids = {id(line) for line in before + overlap}
        remaining = [line for line in remaining if id(line) not in consumed_ids]
        cursor_y = band_y1 + 1.0

    if remaining:
        output.extend(_sort_two_column_region_lines(remaining, page_width, page_height))

    return output


def _make_text_block(line: TextLine, block_id: str, reading_order: int) -> ParsedBlock:
    return ParsedBlock(
        block_id=block_id,
        type="text",
        text=clean_line_text(line.text),
        bbox=list(line.bbox),
        column=line.column,
        reading_order=reading_order,
        page=line.page_num,
        size=round(float(line.size), 2),
        is_bold=line.is_bold,
        is_italic=line.is_italic,
        block_no=line.block_no,
        line_no=line.line_no,
    )


def build_page_blocks(
    sorted_lines: list[TextLine],
    image_blocks: list[ParsedBlock],
    page_width: float,
    page_height: float,
    *,
    layout_diagnostic: Optional[dict] = None,
) -> list[ParsedBlock]:
    """构建 pages[].blocks，文本行沿用阅读顺序，image block 保留占位。"""
    text_blocks = [
        _make_text_block(line, f"p{line.page_num}_b{i:04d}", 0)
        for i, line in enumerate(sorted_lines, start=1)
    ]

    diagnostic = layout_diagnostic or analyze_page_layout(sorted_lines, page_width, page_height)
    if diagnostic["is_two_column"]:
        blocks = _merge_images_into_text_block_order(text_blocks, image_blocks)
    else:
        blocks = sorted(text_blocks + image_blocks, key=lambda block: (block.bbox[1], block.bbox[0], block.type))

    for order, block in enumerate(blocks, start=1):
        block.reading_order = order
    return blocks


def _merge_images_into_text_block_order(
    text_blocks: list[ParsedBlock],
    image_blocks: list[ParsedBlock],
) -> list[ParsedBlock]:
    """Insert image placeholders by visual y position without reordering text blocks."""
    if not image_blocks:
        return list(text_blocks)
    blocks = list(text_blocks)
    for image in sorted(image_blocks, key=lambda block: (block.bbox[1], block.bbox[0])):
        insert_at = len(blocks)
        for index, block in enumerate(blocks):
            if block.type == "image":
                continue
            if block.bbox[1] > image.bbox[1]:
                insert_at = index
                break
        blocks.insert(insert_at, image)
    return blocks


def parsed_block_to_dict(block: ParsedBlock) -> dict:
    """输出紧凑 JSON，兼容后续人工审计。"""
    data = {
        "block_id": block.block_id,
        "type": block.type,
        "text": block.text,
        "bbox": block.bbox,
        "column": block.column,
        "reading_order": block.reading_order,
        "page": block.page,
    }
    if block.type == "text":
        data.update({
            "size": block.size,
            "is_bold": block.is_bold,
            "is_italic": block.is_italic,
            "block_no": block.block_no,
            "line_no": block.line_no,
        })
    elif block.block_no is not None:
        data["block_no"] = block.block_no
    if block.metadata:
        data["metadata"] = block.metadata
    return data


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

    page_records: list[dict] = []
    diagnostics = {
        "two_column_pages": [],
        "document_two_column_prior": False,
        "document_two_column_confidence": 0.0,
        "document_two_column_prior_diagnostics": {},
        "page_layout_diagnostics": [],
        "image_block_count": 0,
        "image_block_pages": [],
        "image_block_source_counts": {},
        "journal_preproof_pages": [],
        "stripped_noise_line_count": 0,
        "stripped_noise_examples": [],
        "stripped_noise_reason_counts": {},
        "figure_caption_candidate_count": 0,
        "table_caption_candidate_count": 0,
    }

    all_unfiltered_lines: list[TextLine] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_no = page_num + 1
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        page_lines, image_blocks = extract_layout_from_page(page, page_no)
        page_lines = sorted(page_lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        page_records.append({
            "page": page_no,
            "width": page_width,
            "height": page_height,
            "lines": page_lines,
            "image_blocks": image_blocks,
        })
        all_unfiltered_lines.extend(page_lines)

    repeated_journal_preproof_keys = collect_repeated_journal_preproof_keys(all_unfiltered_lines)

    initial_layout_diagnostics: list[dict] = []
    for record in page_records:
        initial_diag = analyze_page_layout(record["lines"], record["width"], record["height"])
        initial_diag["page"] = record["page"]
        initial_layout_diagnostics.append(initial_diag)

    document_prior_diag = analyze_document_two_column_prior(initial_layout_diagnostics)
    document_two_column_prior = bool(document_prior_diag["document_two_column_prior"])
    diagnostics["document_two_column_prior"] = document_two_column_prior
    diagnostics["document_two_column_confidence"] = document_prior_diag["document_two_column_confidence"]
    diagnostics["document_two_column_prior_diagnostics"] = document_prior_diag

    all_lines: list[TextLine] = []
    for record in page_records:
        page_no = record["page"]
        page_lines, preproof_diag = strip_journal_preproof_noise(
            record["lines"],
            page_no,
            total_pages=total_pages,
            repeated_journal_preproof_keys=repeated_journal_preproof_keys,
        )
        if preproof_diag["stripped_count"]:
            diagnostics["journal_preproof_pages"].append(page_no)
            diagnostics["stripped_noise_line_count"] += preproof_diag["stripped_count"]
            diagnostics["stripped_noise_examples"].extend(preproof_diag["examples"])
            for reason, count in preproof_diag["reason_counts"].items():
                diagnostics["stripped_noise_reason_counts"][reason] = (
                    diagnostics["stripped_noise_reason_counts"].get(reason, 0) + count
                )

        record["lines"] = page_lines
        layout_diag = analyze_page_layout(
            page_lines,
            record["width"],
            record["height"],
            document_two_column_prior=document_two_column_prior,
        )
        layout_diag["page"] = page_no
        record["layout_diagnostic"] = layout_diag
        if layout_diag["is_two_column"]:
            diagnostics["two_column_pages"].append(page_no)
        diagnostics["page_layout_diagnostics"].append(layout_diag)
        diagnostics["image_block_count"] += len(image_blocks)
        if image_blocks:
            diagnostics["image_block_pages"].append(page_no)
        for image_block in image_blocks:
            source = str((image_block.metadata or {}).get("image_source", "unknown"))
            diagnostics["image_block_source_counts"][source] = (
                diagnostics["image_block_source_counts"].get(source, 0) + 1
            )
        diagnostics["figure_caption_candidate_count"] += sum(
            1 for line in page_lines if is_figure_caption_candidate(line.text)
        )
        diagnostics["table_caption_candidate_count"] += sum(
            1 for line in page_lines if is_table_caption_candidate(line.text)
        )
        all_lines.extend(page_lines)

    header_footer_cache = detect_header_footers(all_lines)
    all_lines = merge_consecutive_titles(all_lines)

    pages_output: list[ParsedPage] = []
    for record in page_records:
        page_no = record["page"]
        page_lines = [l for l in all_lines if l.page_num == page_no]
        page_lines = sort_lines_reading_order(
            page_lines,
            record["width"],
            record["height"],
            layout_diagnostic=record.get("layout_diagnostic"),
            document_two_column_prior=document_two_column_prior,
        )
        page_text = build_structured_text(page_lines, header_footer_cache)
        page_blocks = build_page_blocks(
            page_lines,
            record["image_blocks"],
            record["width"],
            record["height"],
            layout_diagnostic=record.get("layout_diagnostic"),
        )
        pages_output.append(ParsedPage(
            page_num=page_no,
            text=page_text,
            blocks=page_blocks,
        ))

    doc.close()

    output_data = {
        "doc_id": doc_id,
        "source_file": source_file,
        "total_pages": total_pages,
        "parser_stage": "parsed_raw_v4",
        "pages": [
            {
                "page": p.page_num,
                "text": p.text,
                "blocks": [parsed_block_to_dict(block) for block in p.blocks],
            }
            for p in pages_output
        ],
        "diagnostics": diagnostics,
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
