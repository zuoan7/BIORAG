#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parsed_raw → parsed_clean + parsed_preview 清洗脚本

读取 parsed_raw/*.json，输出：
  - parsed_clean/*.json  （后续 chunking 的唯一输入）
  - parsed_preview/*.md  （人工审计，不参与 pipeline）

职责：
  1. 修复 PDF 断词（保守词典规则）
  2. 降级误判标题（16S rRNA、引物、化合物等）
  3. 拆分 inline subsection heading
  4. 分离 Figure / Table caption
  5. 标记 References 区段
  6. 输出结构化 parsed_clean JSON + Markdown preview

用法：
    python scripts/ingestion/clean_parsed_structure.py \
      --input_dir data/paper_round1/parsed_raw \
      --output_dir data/paper_round1/parsed_clean \
      --preview_dir data/paper_round1/parsed_preview
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ============================================================
# 断词修复词典（保守）
# ============================================================

BROKEN_WORD_FIXES: list[tuple[str, str]] = [
    # 细菌名
    (r"Bifido\s+bacterium", "Bifidobacterium"),
    (r"Bifidobacte\s+rium", "Bifidobacterium"),
    (r"Bifido\s+bacteria", "Bifidobacteria"),
    (r"Kleb\s+siella", "Klebsiella"),
    (r"pneu\s+moniae", "pneumoniae"),
    # 常见断词
    (r"fermenta\s+tion", "fermentation"),
    (r"fermenta\s+tions", "fermentations"),
    (r"struc\s+turally", "structurally"),
    (r"pri\s+mary", "primary"),
    (r"sup\s+portive", "supportive"),
    (r"micro\s+biome", "microbiome"),
    (r"re\s+view", "review"),
    (r"con\s+sumption", "consumption"),
    (r"fuco\s+syllactose", "fucosyllactose"),
    # 额外常见断词
    (r"estab\s+lish", "establish"),
    (r"estab\s+lishment", "establishment"),
    (r"sub\s+species", "subspecies"),
    (r"patho\s+gens", "pathogens"),
    (r"patho\s+gen", "pathogen"),
    (r"meta\s+bolism", "metabolism"),
    (r"meta\s+bolic", "metabolic"),
    (r"meta\s+bolite", "metabolite"),
    (r"fermenta\s+tive", "fermentative"),
    (r"hydro\s+lysis", "hydrolysis"),
    (r"gastro\s+intestinal", "gastrointestinal"),
    (r"pre\s+biotic", "prebiotic"),
    (r"pre\s+biotics", "prebiotics"),
    (r"se\s+quence", "sequence"),
    (r"se\s+quences", "sequences"),
    (r"se\s+quencing", "sequencing"),
    (r"asso\s+ciated", "associated"),
    (r"asso\s+ciation", "association"),
]

BROKEN_WORD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.I), replacement)
    for pat, replacement in BROKEN_WORD_FIXES
]

# 检测残存断词的模式
BROKEN_WORD_DETECT_PATTERNS = [
    re.compile(r"Bifido\s+bacterium", re.I),
    re.compile(r"Kleb\s+siella", re.I),
    re.compile(r"fermenta\s+tion", re.I),
    re.compile(r"struc\s+turally", re.I),
    re.compile(r"pri\s+mary", re.I),
    re.compile(r"sup\s+portive", re.I),
    re.compile(r"micro\s+biome", re.I),
    re.compile(r"re\s+view", re.I),
]


# ============================================================
# Section 标题白名单（真正的大 section）
# ============================================================

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


# ============================================================
# 误判标题检测
# ============================================================

FALSE_HEADING_PATTERNS = [
    re.compile(r"^#{1,3}\s+\d+S\s+rRNA", re.I),
    re.compile(r"^#{1,3}\s+\d+F\b"),
    re.compile(r"^#{1,3}\s+13CH", re.I),
    re.compile(r"^#{1,3}\s+1F-β?-?fructofuranosyl", re.I),
    re.compile(r"^#{1,3}\s+\d+F-"),
    re.compile(r"^#{1,3}\s+\d+R\b"),
    re.compile(r"^#{1,3}\s+\d+C-"),
]

FALSE_HEADING_DETECT_PATTERNS = [
    re.compile(r"16S\s+rRNA", re.I),
    re.compile(r"27F\b"),
    re.compile(r"1492R\b"),
    re.compile(r"13CH", re.I),
    re.compile(r"1F-β?-?fructofuranosyl", re.I),
]


# ============================================================
# Figure / Table caption 检测
# ============================================================

FIGURE_CAPTION_PATTERN = re.compile(
    r"^(?:Supplementary\s+)?(?:Fig\.?|Figure)\s+S?\d+",
    re.I,
)

TABLE_CAPTION_PATTERN = re.compile(
    r"^(?:Supplementary\s+)?Table\s+S?\d+",
    re.I,
)

FIGURE_CAPTION_INLINE_PATTERN = re.compile(
    r"(?:Supplementary\s+)?(?:Fig\.?|Figure)\s+S?\d+[\.\:]\s*",
    re.I,
)

TABLE_CAPTION_INLINE_PATTERN = re.compile(
    r"(?:Supplementary\s+)?Table\s+S?\d+[\.\:]\s*",
    re.I,
)


# ============================================================
# Subsection heading 检测
# ============================================================

SUBSECTION_HEADING_PATTERN = re.compile(
    r"^(\d+\.\d+\.?\s+[A-Z][^\n]{3,80}?)\s{2,}([A-Z][^\n]+)$"
)

SUBSECTION_HEADING_START_PATTERN = re.compile(
    r"^(\d+\.\d+\.?\s+[A-Z][^\n]{3,80}?)\s+([A-Z][a-z]+\s+[a-z])"
)

# 简单 subsection 编号检测
SUBSECTION_NUMBER_PATTERN = re.compile(
    r"^(\d+\.\d+\.?)\s+"
)


# ============================================================
# References 检测（增强版：防止表格列名误触发、支持退出 references）
# ============================================================

REFERENCES_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?"
    r"(references|bibliography|literature\s+cited|works\s+cited)\s*$",
    re.I,
)

# 正文 section 标题 — 出现时应退出 in_references
BODY_SECTION_HEADINGS = [
    "abstract", "introduction", "background",
    "results", "results and discussion", "discussion and results",
    "discussion", "conclusion", "conclusions",
    "methods", "materials and methods",
    "experimental procedures", "experimental methods", "experimental section",
]

BODY_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?("
    + "|".join(re.escape(h) for h in BODY_SECTION_HEADINGS)
    + r")\s*$",
    re.I,
)

# 尾部 section 标题 — 出现时也应退出 in_references（但标为 back_matter）
BACK_MATTER_HEADINGS = [
    "acknowledgments", "acknowledgements",
    "author contributions", "contributors",
    "additional information", "supplementary information", "supporting information",
    "supplementary data", "supplementary materials",
    "data availability", "data availability statement",
    "funding", "conflict of interest", "competing financial interests",
    "competing interests", "ethics statement", "ethics approval",
    "how to cite", "license", "open access",
    "author information",
]

BACK_MATTER_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?("
    + "|".join(re.escape(h) for h in BACK_MATTER_HEADINGS)
    + r")\s*$",
    re.I,
)

# 日期 / 期刊侧栏 metadata — 应降级为 paragraph，不应成为 section_heading
METADATA_HEADING_PATTERNS = [
    re.compile(r"^#{1,3}\s+(?:Received|Accepted|Published|Revised)$", re.I),
    re.compile(r"^#{1,3}\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}$", re.I),
    re.compile(r"^#{1,3}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}$", re.I),
    re.compile(r"^#{1,3}\s+\d{4}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.I),
    re.compile(r"^#{1,3}\s+Correspondence\s+and\s+requests", re.I),
    re.compile(r"^#{1,3}\s+Subject\s+areas", re.I),
    re.compile(r"^#{1,3}\s+www\.", re.I),
    re.compile(r"^#{1,3}\s+DOI", re.I),
    re.compile(r"^#{1,3}\s+SCIENTIFIC\s+REPORTS", re.I),
]

# 参考文献条目模式（编号 + 作者 + 年份/期刊）
REFERENCE_ENTRY_PATTERN = re.compile(
    r"^\s*\d{1,3}\.\s+[A-Z][a-z]+.*(?:\(\d{4}\)|et\s+al\.|doi|vol\.|p\.|pp\.)",
    re.I,
)

# 表格上下文中的 "references" — 不应触发 in_references
# 注意：仅匹配 "content references" 等多词形式，不匹配纯 "References"（那是合法标题）
TABLE_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"^(?:content\s+references?|selected\s+stud.*references?|column\s+references?)\s*$",
    re.I,
)


def is_references_heading(text: str, recent_block_types: list[str] | None = None) -> bool:
    """
    判断是否为真正的 References 章节标题（而非表格列名等）。

    条件：
    1. 匹配 REFERENCES_HEADING_PATTERN；
    2. 不是表格上下文中的 "references"；
    3. 文本不含 Table/Figure/caption/column 等表格/图注关键词；
    4. 不是 "content references" 等表格列名形式。
    """
    heading = text.lstrip("#").strip()
    if not REFERENCES_HEADING_PATTERN.match(heading):
        return False

    # 如果是 "content references" 等表格列名，不进入 references
    if TABLE_CONTEXT_REFERENCE_PATTERN.match(heading):
        return False

    # 如果 heading 包含表格/图注关键词，不进入 references
    table_keywords = ["table", "figure", "caption", "column", "selected stud"]
    heading_lower = heading.lower()
    for kw in table_keywords:
        if kw in heading_lower:
            return False

    # 如果最近几个 block 是 table_caption / table_text，大概率是表格列名
    if recent_block_types:
        recent = recent_block_types[-5:]  # 看最近 5 个 block
        table_nearby = sum(1 for bt in recent if bt in ("table_caption", "table_text"))
        if table_nearby >= 1:
            return False

    return True


def should_exit_references(heading_text: str) -> str | None:
    """
    判断是否应退出 in_references 状态。

    返回:
      "body" — 正文 section，应退出 references
      "back_matter" — 尾部 section，应退出 references
      None — 不退出
    """
    heading = heading_text.lstrip("#").strip()

    if BODY_SECTION_PATTERN.match(heading):
        return "body"
    if BACK_MATTER_PATTERN.match(heading):
        return "back_matter"
    return None


def is_false_heading_metadata(line: str) -> bool:
    """检测日期/期刊侧栏 metadata 误判标题"""
    for pat in METADATA_HEADING_PATTERNS:
        if pat.match(line):
            return True
    return False


def is_numbered_reference_entry(text: str) -> bool:
    """判断单行是否像编号参考文献条目"""
    stripped = text.strip()
    if not stripped:
        return False
    return bool(REFERENCE_ENTRY_PATTERN.match(stripped))


def is_table_context(recent_block_types: list[str] | None, window: int = 5) -> bool:
    """检查最近 block 是否处于表格上下文（table_caption / table_text）"""
    if not recent_block_types:
        return False
    recent = recent_block_types[-window:]
    return any(bt in ("table_caption", "table_text") for bt in recent)


def looks_like_numbered_ref_heading(text: str) -> bool:
    """
    检查 ## heading 以数字开头时是否实为参考文献条目。
    例如 "## 21. Non-Conventional Yeasts In Genetics... (Springer, 2003). doi:..."
    """
    heading = text.lstrip("#").strip()
    m = re.match(r'^(\d{1,3})\.\s+', heading)
    if not m:
        return False
    features = 0
    if re.search(r'et\s+al\.', heading, re.I):
        features += 1
    if re.search(r'\(\d{4}\)', heading):
        features += 1
    if re.search(r'(?:Springer|Wiley|Elsevier|Oxford|Cambridge|doi|vol\.|pp?\.)', heading, re.I):
        features += 1
    if re.search(r'\d+,\s*\d+[–\-]\d+', heading):
        features += 1
    if len(heading) > 80:
        features += 1
    return features >= 1


def _is_reference_like_paragraph(text: str, page_num: int, total_pages: int) -> bool:
    """
    保守判断段落是否像参考文献条目。仅在文档后部且信号强烈时返回 True。
    """
    stripped = text.strip()
    if not stripped:
        return False
    if total_pages > 0 and page_num / total_pages < 0.65:
        return False
    # 强信号：以编号参考文献开头
    if is_numbered_reference_entry(stripped):
        return True
    # 强信号：单段包含多条编号引用（如 "1. Author... 2. Author... 3. Author..."）
    ref_numbers = re.findall(r'(?:^|\.\s+)(\d{1,3})\.\s+[A-Z][a-z]', stripped)
    if len(ref_numbers) >= 2:
        return True
    # 中等信号：以期刊引用续行开头 + 新编号条目
    if re.match(r'^[A-Z][a-z]+\.?\s+\d+,\s*\d+[–\-]\d+\s*\(\d{4}\)', stripped):
        return True
    # 较弱信号：包含多个文献特征且文本较短
    features = 0
    if re.search(r'et\s+al\.', stripped, re.I):
        features += 1
    if re.search(r'\(\d{4}\)', stripped):
        features += 1
    if re.search(r'(?:Proc\.|Nat\.|Science|J\.|Acad\.|Biochem|Biotechnol|Microbiol|Yeast|Protein|Glycobiology|BMC|PLoS|Springer)', stripped, re.I):
        features += 1
    if re.search(r'\d+,\s*\d+[–\-]\d+', stripped):
        features += 1
    if re.search(r'doi', stripped, re.I):
        features += 1
    return features >= 3


# ============================================================
# 数据类
# ============================================================

@dataclass
class Block:
    block_id: str
    type: str  # title | abstract | section_heading | subsection_heading | paragraph | figure_caption | table_caption | table_text | references | noise
    text: str
    section_path: list[str] = field(default_factory=list)
    page: int = 1


@dataclass
class CleanPage:
    page: int
    text: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ProcessingCounters:
    processed_docs: int = 0
    total_blocks: int = 0
    fixed_broken_words: int = 0
    demoted_false_headings: int = 0
    detected_subsections: int = 0
    detected_figure_captions: int = 0
    detected_table_captions: int = 0
    detected_references_blocks: int = 0
    detected_table_text_blocks: int = 0
    merged_split_headings: int = 0


# ============================================================
# 文本清洗
# ============================================================

def fix_broken_words(text: str) -> tuple[str, int]:
    """修复断词，返回 (修复后文本, 修复次数)"""
    count = 0
    for pattern, replacement in BROKEN_WORD_PATTERNS:
        matches = pattern.findall(text)
        count += len(matches)
        text = pattern.sub(replacement, text)
    return text, count


def is_false_heading(line: str) -> bool:
    """检测误判标题"""
    for pat in FALSE_HEADING_PATTERNS:
        if pat.match(line):
            return True
    return False


def is_valid_section_heading(text: str) -> bool:
    """检测是否为合法的大 section 标题"""
    stripped = text.lstrip("#").strip()
    if VALID_SECTION_PATTERN.match(stripped):
        return True
    return False


# ============================================================
# 连续 heading 合并
# ============================================================

# 第二行以这些小写连接词开头时，可以合并到上一行
CONTINUATION_START_WORDS = frozenset({
    "from", "of", "in", "on", "by", "with", "for", "to",
    "and", "or", "the", "a", "an",
})

# 第一行以这些词结尾时，表示标题被截断，可以合并下一行
TRUNCATION_END_WORDS = frozenset({
    "to", "of", "in", "for", "with", "by", "and", "or",
    "real-time", "from",
})


def _get_heading_level(line: str) -> int:
    """获取 Markdown heading level（1-3），非 heading 返回 0"""
    stripped = line.strip()
    if stripped.startswith("### "):
        return 3
    if stripped.startswith("## "):
        return 2
    if stripped.startswith("# "):
        return 1
    return 0


def _heading_text(line: str) -> str:
    """提取 heading 文本（去除 # 前缀）"""
    stripped = line.strip()
    if stripped.startswith("### "):
        return stripped[4:].strip()
    if stripped.startswith("## "):
        return stripped[3:].strip()
    if stripped.startswith("# "):
        return stripped[2:].strip()
    return stripped


def should_merge_headings(prev_line: str, curr_line: str) -> bool:
    """
    判断两行连续 heading 是否应该合并。

    保守规则：
    1. 两行 heading level 相同或相近
    2. 第二行以小写连接词开头，或第一行以截断词结尾
    3. 第二行较短，且不像独立 section 标题
    4. 合并后不像两个独立标题
    """
    prev_level = _get_heading_level(prev_line)
    curr_level = _get_heading_level(curr_line)

    # 必须都是 heading
    if prev_level == 0 or curr_level == 0:
        return False

    # heading level 相同或相近（差异 ≤ 1）
    if abs(prev_level - curr_level) > 1:
        return False

    prev_text = _heading_text(prev_line)
    curr_text = _heading_text(curr_line)

    # 第二行太长则不像残片（>60 字符通常是独立标题）
    if len(curr_text) > 60:
        return False

    # 检查第二行是否以小写连接词开头
    curr_first_word = curr_text.split()[0].lower() if curr_text.split() else ""
    starts_with_continuation = curr_first_word in CONTINUATION_START_WORDS

    # 检查第一行是否以截断词结尾
    prev_words = prev_text.rstrip(".,;:").split()
    prev_last_word = prev_words[-1].lower() if prev_words else ""
    ends_with_truncation = prev_last_word in TRUNCATION_END_WORDS

    # 至少满足一个条件
    if not (starts_with_continuation or ends_with_truncation):
        return False

    # 第二行不能像独立 section 标题
    if is_valid_section_heading(curr_line):
        return False

    # 如果第二行以大写字母开头且不是连接词，则可能独立
    if curr_text[0:1].isupper() and not starts_with_continuation:
        return False

    return True


def merge_consecutive_headings(lines: list[str]) -> tuple[list[str], int]:
    """
    合并连续的 heading 行（中间允许空行）。
    返回 (合并后的行列表, 合并次数)
    """
    if not lines:
        return lines, 0

    merged_lines: list[str] = []
    merge_count = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        # 尝试与下一行合并（跳过中间空行）
        if _get_heading_level(line) > 0:
            # 查找下一个非空行
            j = i + 1
            skipped_blanks = 0
            while j < len(lines) and lines[j].strip() == "":
                j += 1
                skipped_blanks += 1

            if j < len(lines) and _get_heading_level(lines[j]) > 0:
                next_line = lines[j]
                if should_merge_headings(line, next_line):
                    # 合并：保留第一行的 level，拼接文本
                    level = _get_heading_level(line)
                    prefix = "#" * level
                    combined_text = f"{_heading_text(line)} {_heading_text(next_line)}"
                    merged_lines.append(f"{prefix} {combined_text}")
                    merge_count += 1
                    # 跳过空行和下一行
                    i = j + 1
                    continue

        merged_lines.append(line)
        i += 1

    return merged_lines, merge_count


def detect_figure_caption(text: str) -> bool:
    """检测是否以 figure caption 开头"""
    stripped = text.strip()
    return bool(FIGURE_CAPTION_PATTERN.match(stripped))


def detect_table_caption(text: str) -> bool:
    """检测是否以 table caption 开头"""
    stripped = text.strip()
    return bool(TABLE_CAPTION_PATTERN.match(stripped))


def extract_figure_caption_from_inline(text: str) -> Optional[tuple[str, str]]:
    """从正文中间提取 figure caption，返回 (caption, remainder) 或 None"""
    m = FIGURE_CAPTION_INLINE_PATTERN.search(text)
    if m:
        # 找到 Fig. / Figure 的位置
        start = m.start()
        prefix = text[:start].strip()
        # caption 延伸到句子结束
        caption_start = start
        rest = text[m.end():]
        # 找到下一个句号作为 caption 结束
        dot_pos = rest.find(".")
        if dot_pos >= 0 and dot_pos < 200:
            caption_text = text[caption_start:m.end() + dot_pos + 1].strip()
            remainder = text[m.end() + dot_pos + 1:].strip()
            if prefix:
                return (prefix, caption_text, remainder)
            else:
                return ("", caption_text, remainder)
    return None


def extract_table_caption_from_inline(text: str) -> Optional[tuple[str, str]]:
    """从正文中间提取 table caption，返回 (caption, remainder) 或 None"""
    m = TABLE_CAPTION_INLINE_PATTERN.search(text)
    if m:
        start = m.start()
        prefix = text[:start].strip()
        caption_start = start
        rest = text[m.end():]
        dot_pos = rest.find(".")
        if dot_pos >= 0 and dot_pos < 200:
            caption_text = text[caption_start:m.end() + dot_pos + 1].strip()
            remainder = text[m.end() + dot_pos + 1:].strip()
            if prefix:
                return (prefix, caption_text, remainder)
            else:
                return ("", caption_text, remainder)
    return None


# ============================================================
# 页面处理
# ============================================================

def classify_line_type(line: str, in_references: bool, recent_block_types: list[str] | None = None) -> str:
    """
    分类单行文本的类型。
    返回: title | section_heading | subsection_heading | paragraph | figure_caption | table_caption | references | noise
    """
    stripped = line.strip()

    if not stripped:
        return "noise"

    # Markdown 标题行
    if stripped.startswith("### "):
        heading_text = stripped[4:].strip()
        if is_false_heading(stripped):
            return "paragraph"
        # 日期/期刊 metadata 降级
        if is_false_heading_metadata(stripped):
            return "paragraph"
        # 编号参考文献条目伪装成 heading
        if looks_like_numbered_ref_heading(stripped):
            return "paragraph"
        # subsection heading 通常有数字编号
        if SUBSECTION_NUMBER_PATTERN.match(heading_text):
            return "subsection_heading"
        # 可能是误判，检查是否为合法 section
        if is_valid_section_heading(stripped):
            # 表格上下文中的 "references" 不应是 section_heading
            heading_lower = heading_text.lower()
            if heading_lower in ("references", "bibliography") and is_table_context(recent_block_types):
                return "paragraph"
            return "section_heading"
        return "subsection_heading"

    if stripped.startswith("## "):
        heading_text = stripped[3:].strip()
        if is_false_heading(stripped):
            return "paragraph"
        # 日期/期刊 metadata 降级
        if is_false_heading_metadata(stripped):
            return "paragraph"
        # 编号参考文献条目伪装成 heading（如 "## 21. Non-Conventional Yeasts..."）
        if looks_like_numbered_ref_heading(stripped):
            return "paragraph"
        if is_valid_section_heading(stripped):
            # 表格上下文中的 "references" 不应是 section_heading
            heading_lower = heading_text.lower()
            if heading_lower in ("references", "bibliography") and is_table_context(recent_block_types):
                return "paragraph"
            return "section_heading"
        # 有数字编号的可能是 section heading
        if re.match(r"^\d+\.?\s+[A-Z]", heading_text):
            # 编号参考文献条目伪装成 heading
            if looks_like_numbered_ref_heading(stripped):
                return "paragraph"
            # 检查是否是 subsection 级别（2.1, 3.2 等）
            if re.match(r"^\d+\.\d+", heading_text):
                return "subsection_heading"
            return "section_heading"
        # 纯 "references" 在表格上下文中降级为 paragraph
        heading_lower = heading_text.lower()
        if heading_lower in ("references", "bibliography") and is_table_context(recent_block_types):
            return "paragraph"
        return "section_heading"

    if stripped.startswith("# "):
        return "title"

    # Figure / Table caption
    if detect_figure_caption(stripped):
        return "figure_caption"
    if detect_table_caption(stripped):
        return "table_caption"

    # 在 references 区段中 — 但如果此行像正文 heading，应退出
    if in_references:
        # 检查是否应退出 references（正文 section / 尾部 section）
        exit_type = should_exit_references(stripped)
        if exit_type == "body":
            if stripped.startswith("## "):
                return "section_heading"
            elif stripped.startswith("### "):
                return "subsection_heading"
            else:
                return "section_heading"
        if exit_type == "back_matter":
            if stripped.startswith("## "):
                return "section_heading"
            elif stripped.startswith("### "):
                return "subsection_heading"
            else:
                return "section_heading"
        # 非标题行以尾部 section 关键词开头 → 退出 references
        check_text = stripped.lstrip("#").strip().lower()
        for bm in BACK_MATTER_HEADINGS:
            if check_text.startswith(bm):
                after = check_text[len(bm):]
                if not after or after[0] in (' ', '.', ',', ':', ';'):
                    return "paragraph"
        return "references"

    # 普通正文
    return "paragraph"


def split_inline_subsection(text: str) -> list[tuple[str, str]]:
    """
    拆分 inline subsection heading。
    例如: "2.3. In vitro fermentation with individual infant inocula In vitro fecal fermentation..."
    → [("subsection_heading", "2.3. In vitro fermentation with individual infant inocula"),
       ("paragraph", "In vitro fecal fermentation...")]

    返回 [(type, text), ...]
    """
    # 尝试匹配 "数字.数字 标题 大写开头的正文"
    m = SUBSECTION_HEADING_PATTERN.match(text)
    if m:
        heading = m.group(1).strip()
        body = m.group(2).strip()
        return [("subsection_heading", heading), ("paragraph", body)]

    # 更宽松的匹配
    m = SUBSECTION_HEADING_START_PATTERN.match(text)
    if m:
        heading = m.group(1).strip()
        body = text[len(m.group(1)):].strip()
        if body and len(heading.split()) >= 3:
            return [("subsection_heading", heading), ("paragraph", body)]

    return [("paragraph", text)]


def split_inline_section_heading(text: str) -> list[tuple[str, str]]:
    """
    拆分 inline section heading。
    例如: "2.1. Materials Six commercially available HMOs..."
    → [("subsection_heading", "2.1. Materials"), ("paragraph", "Six commercially available HMOs...")]
    """
    # 匹配 "数字.数字 标题词(1-5词) 正文"
    m = re.match(r"^(\d+\.\d+\.?\s+(?:[A-Z][a-z]+(?:\s+and\s+)?){1,5})\s+([A-Z][a-z])", text)
    if m:
        heading = m.group(1).strip()
        body = text[len(m.group(1)):].strip()
        if body and len(heading.split()) <= 10:
            return [("subsection_heading", heading), ("paragraph", body)]

    return [("paragraph", text)]


def process_page_text(
    page_text: str,
    page_num: int,
    block_index_start: int,
    counters: ProcessingCounters,
    section_path: Optional[list[str]] = None,
    in_references: bool = False,
    total_pages: int = 0,
    recent_block_types: Optional[list[str]] = None,
) -> tuple[list[Block], str, list[str], bool]:
    """
    处理单个页面的文本，返回 (blocks, 清洗后全文, section_path, in_references)。
    section_path 和 in_references 从上一页传入，实现跨页状态连续传递。
    """
    lines = page_text.split("\n")
    # 先合并连续 heading 行
    lines, merged_count = merge_consecutive_headings(lines)
    counters.merged_split_headings += merged_count

    blocks: list[Block] = []
    current_paragraph: list[str] = []
    if section_path is None:
        section_path = []
    # in_references 由参数传入，不再重新初始化
    block_idx = block_index_start
    # 追踪最近 block 类型，用于 table context 判断
    if recent_block_types is None:
        recent_block_types = []

    def _trim_recent():
        """保持 recent_block_types 在合理长度（使用 slice 赋值避免 reassignment）"""
        if len(recent_block_types) > 20:
            del recent_block_types[:len(recent_block_types)-20]

    def flush_paragraph():
        nonlocal block_idx
        if current_paragraph:
            text = " ".join(current_paragraph)
            current_paragraph.clear()
            if not text.strip():
                return

            # 修复断词
            text, fix_count = fix_broken_words(text)
            counters.fixed_broken_words += fix_count

            # 检查是否包含 inline subsection heading
            if SUBSECTION_NUMBER_PATTERN.match(text) and len(text) > 40:
                parts = split_inline_subsection(text)
                if len(parts) > 1:
                    for ptype, ptext in parts:
                        block_id = f"p{page_num}_b{block_idx:04d}"
                        ptext, fc = fix_broken_words(ptext)
                        counters.fixed_broken_words += fc
                        if ptype == "subsection_heading":
                            counters.detected_subsections += 1
                            section_path.append(ptext.strip())
                        blocks.append(Block(
                            block_id=block_id,
                            type=ptype,
                            text=ptext.strip(),
                            section_path=list(section_path),
                            page=page_num,
                        ))
                        block_idx += 1
                    counters.total_blocks += len(parts)
                    return

                parts = split_inline_section_heading(text)
                if len(parts) > 1:
                    for ptype, ptext in parts:
                        block_id = f"p{page_num}_b{block_idx:04d}"
                        ptext, fc = fix_broken_words(ptext)
                        counters.fixed_broken_words += fc
                        if ptype == "subsection_heading":
                            counters.detected_subsections += 1
                            section_path.append(ptext.strip())
                        blocks.append(Block(
                            block_id=block_id,
                            type=ptype,
                            text=ptext.strip(),
                            section_path=list(section_path),
                            page=page_num,
                        ))
                        block_idx += 1
                    counters.total_blocks += len(parts)
                    return

            # 检查是否包含 inline figure caption
            fig_result = extract_figure_caption_from_inline(text)
            if fig_result and len(fig_result) == 3:
                prefix, caption, remainder = fig_result
                if prefix:
                    block_id = f"p{page_num}_b{block_idx:04d}"
                    blocks.append(Block(
                        block_id=block_id, type="paragraph", text=prefix,
                        section_path=list(section_path), page=page_num,
                    ))
                    block_idx += 1
                block_id = f"p{page_num}_b{block_idx:04d}"
                blocks.append(Block(
                    block_id=block_id, type="figure_caption", text=caption,
                    section_path=list(section_path), page=page_num,
                ))
                counters.detected_figure_captions += 1
                block_idx += 1
                if remainder:
                    block_id = f"p{page_num}_b{block_idx:04d}"
                    blocks.append(Block(
                        block_id=block_id, type="paragraph", text=remainder,
                        section_path=list(section_path), page=page_num,
                    ))
                    block_idx += 1
                counters.total_blocks += 2 + (1 if prefix else 0) + (1 if remainder else 0)
                return

            # 检查是否包含 inline table caption
            tbl_result = extract_table_caption_from_inline(text)
            if tbl_result and len(tbl_result) == 3:
                prefix, caption, remainder = tbl_result
                if prefix:
                    block_id = f"p{page_num}_b{block_idx:04d}"
                    blocks.append(Block(
                        block_id=block_id, type="paragraph", text=prefix,
                        section_path=list(section_path), page=page_num,
                    ))
                    block_idx += 1
                block_id = f"p{page_num}_b{block_idx:04d}"
                blocks.append(Block(
                    block_id=block_id, type="table_caption", text=caption,
                    section_path=list(section_path), page=page_num,
                ))
                counters.detected_table_captions += 1
                block_idx += 1
                if remainder:
                    block_id = f"p{page_num}_b{block_idx:04d}"
                    blocks.append(Block(
                        block_id=block_id, type="paragraph", text=remainder,
                        section_path=list(section_path), page=page_num,
                    ))
                    block_idx += 1
                counters.total_blocks += 2 + (1 if prefix else 0) + (1 if remainder else 0)
                return

            block_id = f"p{page_num}_b{block_idx:04d}"
            btype = "references" if in_references else "paragraph"
            blocks.append(Block(
                block_id=block_id,
                type=btype,
                text=text.strip(),
                section_path=list(section_path),
                page=page_num,
            ))
            recent_block_types.append(btype)
            if len(recent_block_types) > 20:
                _trim_recent()
            counters.total_blocks += 1
            block_idx += 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 空行刷新段落
            flush_paragraph()
            continue

        line_type = classify_line_type(stripped, in_references, recent_block_types)

        # 先修复断词
        fixed_text, fix_count = fix_broken_words(stripped)
        counters.fixed_broken_words += fix_count

        # 如果 in_references 且 classify 返回 paragraph（尾部 section 退出），更新状态
        if in_references and line_type == "paragraph":
            check_text = stripped.lstrip("#").strip().lower()
            for bm in BACK_MATTER_HEADINGS:
                if check_text.startswith(bm):
                    after = check_text[len(bm):]
                    if not after or after[0] in (' ', '.', ',', ':', ';'):
                        in_references = False
                        break

        if line_type == "title":
            flush_paragraph()
            block_id = f"p{page_num}_b{block_idx:04d}"
            blocks.append(Block(
                block_id=block_id, type="title", text=fixed_text,
                section_path=["Title"], page=page_num,
            ))
            section_path = ["Title"]
            counters.total_blocks += 1
            block_idx += 1

        elif line_type == "section_heading":
            flush_paragraph()
            # 提取标题文本
            heading_text = fixed_text.lstrip("#").strip()
            # 更新 section_path（只保留当前大 section）
            section_path = [heading_text]
            # 检查是否应退出 references 状态
            exit_type = should_exit_references(heading_text)
            if exit_type is not None:
                in_references = False
            # 检查是否进入 References（使用更严格的判断）
            if is_references_heading(fixed_text, recent_block_types):
                in_references = True
            block_id = f"p{page_num}_b{block_idx:04d}"
            blocks.append(Block(
                block_id=block_id, type="section_heading", text=fixed_text,
                section_path=list(section_path), page=page_num,
            ))
            recent_block_types.append("section_heading")
            # 保持 recent_block_types 在合理长度
            if len(recent_block_types) > 20:
                _trim_recent()
            counters.total_blocks += 1
            block_idx += 1

        elif line_type == "subsection_heading":
            flush_paragraph()
            heading_text = fixed_text.lstrip("#").strip()
            # 更新 section_path（保留大 section + 当前 subsection）
            if len(section_path) > 0:
                # 替换最后一个 subsection
                if len(section_path) > 1 and SUBSECTION_NUMBER_PATTERN.match(section_path[-1]):
                    section_path[-1] = heading_text
                else:
                    section_path.append(heading_text)
            else:
                section_path = [heading_text]
            # 检查是否应退出 references（subsection_heading 也可能触发）
            exit_type = should_exit_references(heading_text)
            if exit_type is not None:
                in_references = False
            counters.detected_subsections += 1
            block_id = f"p{page_num}_b{block_idx:04d}"
            blocks.append(Block(
                block_id=block_id, type="subsection_heading", text=fixed_text,
                section_path=list(section_path), page=page_num,
            ))
            recent_block_types.append("subsection_heading")
            if len(recent_block_types) > 20:
                _trim_recent()
            counters.total_blocks += 1
            block_idx += 1

        elif line_type == "figure_caption":
            flush_paragraph()
            counters.detected_figure_captions += 1
            block_id = f"p{page_num}_b{block_idx:04d}"
            blocks.append(Block(
                block_id=block_id, type="figure_caption", text=fixed_text,
                section_path=list(section_path), page=page_num,
            ))
            recent_block_types.append("figure_caption")
            if len(recent_block_types) > 20:
                _trim_recent()
            counters.total_blocks += 1
            block_idx += 1

        elif line_type == "table_caption":
            flush_paragraph()
            counters.detected_table_captions += 1
            block_id = f"p{page_num}_b{block_idx:04d}"
            blocks.append(Block(
                block_id=block_id, type="table_caption", text=fixed_text,
                section_path=list(section_path), page=page_num,
            ))
            recent_block_types.append("table_caption")
            if len(recent_block_types) > 20:
                _trim_recent()
            counters.total_blocks += 1
            block_idx += 1

        elif line_type == "references":
            # 在 references 区段内的行，累积到当前段落
            counters.detected_references_blocks += 1
            current_paragraph.append(fixed_text)

        elif line_type == "paragraph":
            # 误判标题降级为正文
            if stripped.startswith("## ") and is_false_heading(stripped):
                counters.demoted_false_headings += 1
            if stripped.startswith("## ") and is_false_heading_metadata(stripped):
                counters.demoted_false_headings += 1
            current_paragraph.append(fixed_text)

        elif line_type == "noise":
            flush_paragraph()

    flush_paragraph()

    # 重建清洗后的页面文本
    cleaned_text = rebuild_page_text(blocks)

    return blocks, cleaned_text, section_path, in_references


def rebuild_page_text(blocks: list[Block]) -> str:
    """从 blocks 重建页面文本"""
    parts = []
    for block in blocks:
        if block.type == "title":
            parts.append(f"# {block.text}")
        elif block.type == "section_heading":
            parts.append(f"## {block.text.lstrip('#').strip()}")
        elif block.type == "subsection_heading":
            parts.append(f"### {block.text.lstrip('#').strip()}")
        elif block.type == "figure_caption":
            parts.append(f"[FIGURE CAPTION] {block.text}")
        elif block.type == "table_caption":
            parts.append(f"[TABLE CAPTION] {block.text}")
        elif block.type == "table_text":
            parts.append(f"[TABLE] {block.text}")
        elif block.type == "references":
            parts.append(block.text)
        else:
            parts.append(block.text)
    return "\n\n".join(parts)


def _post_process_numbered_references(
    blocks: list[Block], total_pages: int, counters: ProcessingCounters,
) -> list[Block]:
    """
    后处理：保守检测文档后部的无标题编号参考文献列表。
    触发条件（满足其一）：
    1. 连续出现 2+ 条文献样式段落；
    2. 单个段落包含 2+ 条编号引用（如 "1. Author... 2. Author..."）。
    确认后从该位置起标记为 references，直到遇到尾部 section 退出。
    """
    if not blocks or total_pages == 0:
        return blocks

    threshold_page = max(1, int(total_pages * 0.65))

    # 找到文档后部的文献样式段落索引
    ref_like_indices: list[int] = []
    for i, block in enumerate(blocks):
        if block.page >= threshold_page and block.type == "paragraph":
            if _is_reference_like_paragraph(block.text, block.page, total_pages):
                ref_like_indices.append(i)

    if not ref_like_indices:
        return blocks

    # 确定参考文献区域起点（取最早的位置）
    ref_start_idx = None

    # 条件 1：单块含多条编号引用 → 立即确认
    for idx in ref_like_indices:
        ref_numbers = re.findall(r'(?:^|\.\s+)(\d{1,3})\.\s+[A-Z][a-z]', blocks[idx].text)
        if len(ref_numbers) >= 2:
            ref_start_idx = idx
            break

    # 条件 2：连续 2+ 条文献样式段落（可能比条件1更早）
    if len(ref_like_indices) >= 2:
        groups: list[list[int]] = []
        current_group = [ref_like_indices[0]]
        for i in range(1, len(ref_like_indices)):
            if ref_like_indices[i] - ref_like_indices[i - 1] <= 3:
                current_group.append(ref_like_indices[i])
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [ref_like_indices[i]]
        if len(current_group) >= 2:
            groups.append(current_group)
        if groups:
            best_group = max(groups, key=len)
            group_start = best_group[0]
            # 取更早的位置
            if ref_start_idx is None or group_start < ref_start_idx:
                ref_start_idx = group_start

    if ref_start_idx is None:
        return blocks

    # 从起点开始，标记所有段落为 references，直到退出条件
    for idx in range(ref_start_idx, len(blocks)):
        block = blocks[idx]
        # 退出条件 1：遇到 body / back matter heading
        if block.type in ("section_heading", "subsection_heading"):
            heading = block.text.lstrip("#").strip()
            if should_exit_references(heading) is not None:
                break
            # 编号参考文献伪装成 heading → 降级为 references
            if looks_like_numbered_ref_heading(block.text):
                block.type = "references"
                counters.detected_references_blocks += 1
                continue
        # 退出条件 2：段落以尾部 section 关键词开头
        if block.type == "paragraph":
            text_lower = block.text.strip().lower()
            is_back_matter = False
            for bm in BACK_MATTER_HEADINGS:
                if text_lower.startswith(bm):
                    after = text_lower[len(bm):]
                    if not after or after[0] in (' ', '.', ',', ':', ';'):
                        is_back_matter = True
                        break
            if is_back_matter:
                break
        # 标记为 references
        if block.type == "paragraph":
            block.type = "references"
            counters.detected_references_blocks += 1

    return blocks


# ============================================================
# 单文档处理
# ============================================================

def process_document(
    input_path: Path,
    output_dir: Path,
    preview_dir: Path,
    counters: ProcessingCounters,
) -> dict:
    """
    处理单个 parsed_raw JSON，输出 parsed_clean JSON 和 parsed_preview MD。
    """
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    doc_id = raw_data.get("doc_id", input_path.stem)
    source_file = raw_data.get("source_file", input_path.name)
    total_pages = raw_data.get("total_pages", 0)
    raw_pages = raw_data.get("pages", [])

    # 检查是否已有 blocks（兼容旧格式）
    has_blocks = any(
        isinstance(p, dict) and "blocks" in p and p["blocks"]
        for p in raw_pages
    )

    clean_pages: list[dict] = []
    all_blocks: list[Block] = []
    global_block_idx = 0
    in_references = False
    section_path: list[str] = []
    recent_block_types: list[str] = []  # 追踪最近 block 类型用于 table context 判断

    for page_data in raw_pages:
        page_num = page_data.get("page", 0)
        page_text = page_data.get("text", "")

        if not page_text.strip():
            clean_pages.append({
                "page": page_num,
                "text": "",
                "blocks": [],
            })
            continue

        # 如果已有 blocks，直接使用（合并断词修复）
        if has_blocks and page_data.get("blocks"):
            blocks = []
            for raw_block in page_data["blocks"]:
                text = raw_block.get("text", "")
                text, fix_count = fix_broken_words(text)
                counters.fixed_broken_words += fix_count
                # 跨页状态传递：更新 section_path 和 in_references
                block_type = raw_block.get("type", "paragraph")
                raw_section_path = raw_block.get("section_path", [])
                if block_type == "section_heading":
                    heading_text = text.lstrip("#").strip()
                    # 表格上下文中的 "references" 降级为 paragraph
                    heading_lower = heading_text.lower()
                    if heading_lower in ("references", "bibliography") and is_table_context(recent_block_types):
                        block_type = "paragraph"
                        counters.demoted_false_headings += 1
                        # 不更新 section_path，不进入 references
                        block = Block(
                            block_id=raw_block.get("block_id", f"p{page_num}_b{global_block_idx:04d}"),
                            type=block_type,
                            text=text,
                            section_path=list(section_path),
                            page=page_num,
                        )
                        recent_block_types.append("paragraph")
                        if len(recent_block_types) > 20:
                            del recent_block_types[:len(recent_block_types)-20]
                        blocks.append(block)
                        global_block_idx += 1
                        continue
                    # 编号参考文献伪装成 heading → 降级
                    if looks_like_numbered_ref_heading(text):
                        block_type = "paragraph"
                        counters.demoted_false_headings += 1
                        block = Block(
                            block_id=raw_block.get("block_id", f"p{page_num}_b{global_block_idx:04d}"),
                            type=block_type,
                            text=text,
                            section_path=list(section_path),
                            page=page_num,
                        )
                        recent_block_types.append("paragraph")
                        if len(recent_block_types) > 20:
                            del recent_block_types[:len(recent_block_types)-20]
                        if in_references:
                            block.type = "references"
                            counters.detected_references_blocks += 1
                        blocks.append(block)
                        global_block_idx += 1
                        continue
                    section_path = [heading_text]
                    # 检查是否应退出 references
                    exit_type = should_exit_references(heading_text)
                    if exit_type is not None:
                        in_references = False
                    # 使用更严格的 references 判断
                    if is_references_heading(text, recent_block_types):
                        in_references = True
                    recent_block_types.append("section_heading")
                    if len(recent_block_types) > 20:
                        del recent_block_types[:len(recent_block_types)-20]
                elif block_type == "subsection_heading":
                    heading_text = text.lstrip("#").strip()
                    if len(section_path) > 0:
                        if len(section_path) > 1 and SUBSECTION_NUMBER_PATTERN.match(section_path[-1]):
                            section_path[-1] = heading_text
                        else:
                            section_path.append(heading_text)
                    else:
                        section_path = [heading_text]
                    recent_block_types.append("subsection_heading")
                    if len(recent_block_types) > 20:
                        del recent_block_types[:len(recent_block_types)-20]
                elif block_type == "title":
                    section_path = [text.lstrip("#").strip()]
                    recent_block_types.append("title")
                elif block_type in ("table_caption", "table_text"):
                    recent_block_types.append(block_type)
                    if len(recent_block_types) > 20:
                        del recent_block_types[:len(recent_block_types)-20]

                block = Block(
                    block_id=raw_block.get("block_id", f"p{page_num}_b{global_block_idx:04d}"),
                    type=block_type,
                    text=text,
                    section_path=list(section_path),
                    page=page_num,
                )
                # 覆盖 references 类型判断：如果文档已在 references 状态
                if in_references and block_type == "paragraph":
                    # 检查是否以尾部 section 关键词开头 → 退出 references
                    text_lower = text.strip().lower()
                    exit_refs = False
                    for bm in BACK_MATTER_HEADINGS:
                        if text_lower.startswith(bm):
                            after = text_lower[len(bm):]
                            if not after or after[0] in (' ', '.', ',', ':', ';'):
                                exit_refs = True
                                break
                    if exit_refs:
                        in_references = False
                    else:
                        block.type = "references"
                        counters.detected_references_blocks += 1
                blocks.append(block)
                global_block_idx += 1
            clean_text = rebuild_page_text(blocks)
            clean_pages.append({
                "page": page_num,
                "text": clean_text,
                "blocks": [asdict(b) for b in blocks],
            })
            all_blocks.extend(blocks)
            continue

        # 处理页面文本（无 blocks 路径）
        blocks, cleaned_text, section_path, in_references = process_page_text(
            page_text, page_num, global_block_idx, counters,
            section_path=section_path, in_references=in_references,
            total_pages=total_pages, recent_block_types=recent_block_types,
        )
        # 更新 recent_block_types
        for b in blocks:
            recent_block_types.append(b.type)
        if len(recent_block_types) > 20:
            del recent_block_types[:len(recent_block_types)-20]
        global_block_idx += len(blocks)

        clean_pages.append({
            "page": page_num,
            "text": cleaned_text,
            "blocks": [asdict(b) for b in blocks],
        })
        all_blocks.extend(blocks)

    # 后处理：保守检测文档后部的无标题编号参考文献列表
    _post_process_numbered_references(all_blocks, total_pages, counters)

    # 重建 clean_pages 文本（后处理可能改变了 block type）
    for page_data in clean_pages:
        page_blocks = [b for b in all_blocks if b.page == page_data.get("page", 0)]
        if page_blocks:
            page_data["text"] = rebuild_page_text(page_blocks)
            page_data["blocks"] = [asdict(b) for b in page_blocks]

    # 构建 parsed_clean JSON
    clean_data = {
        "doc_id": doc_id,
        "source_file": source_file,
        "total_pages": total_pages,
        "parser_stage": "parsed_clean_v1",
        "pages": clean_pages,
    }

    # 输出 parsed_clean JSON
    output_path = output_dir / f"{doc_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    # 输出 parsed_preview MD
    preview_path = preview_dir / f"{doc_id}.md"
    md_content = generate_preview_md(all_blocks)
    preview_path.write_text(md_content, encoding="utf-8")

    counters.processed_docs += 1

    return {
        "doc_id": doc_id,
        "total_pages": total_pages,
        "total_blocks": len(all_blocks),
        "status": "ok",
    }


def generate_preview_md(blocks: list[Block]) -> str:
    """从 blocks 生成 Markdown 预览"""
    parts = []

    for block in blocks:
        if block.type == "title":
            parts.append(f"# {block.text}\n")
        elif block.type == "section_heading":
            text = block.text.lstrip("#").strip()
            parts.append(f"\n## {text}\n")
        elif block.type == "subsection_heading":
            text = block.text.lstrip("#").strip()
            parts.append(f"\n### {text}\n")
        elif block.type == "figure_caption":
            parts.append(f"\n[FIGURE CAPTION] {block.text}\n")
        elif block.type == "table_caption":
            parts.append(f"\n[TABLE CAPTION] {block.text}\n")
        elif block.type == "table_text":
            parts.append(f"\n[TABLE] {block.text}\n")
        elif block.type == "references":
            if not parts or not parts[-1].startswith("[REFERENCES]"):
                parts.append(f"\n[REFERENCES]\n")
            parts.append(block.text)
        elif block.type == "noise":
            continue
        else:
            parts.append(block.text)

    return "\n".join(parts).strip()


# ============================================================
# 批量处理
# ============================================================

def batch_process(
    input_dir: Path,
    output_dir: Path,
    preview_dir: Path,
) -> ProcessingCounters:
    """批量处理 parsed_raw 目录下的所有 JSON 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] 目录中没有找到 JSON 文件: {input_dir}")
        return ProcessingCounters()

    print(f"找到 {len(json_files)} 个 JSON 文件")
    print()

    counters = ProcessingCounters()
    success = 0
    failed = 0
    failed_list: list[str] = []

    start_time = time.time()

    for i, json_path in enumerate(json_files, start=1):
        print(f"  [{i}/{len(json_files)}] {json_path.name} ...", end=" ")
        try:
            result = process_document(json_path, output_dir, preview_dir, counters)
            print(f"OK ({result['total_pages']} 页, {result['total_blocks']} blocks)")
            success += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1
            failed_list.append(f"{json_path.name}: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("parsed_clean 清洗统计")
    print("=" * 60)
    print(f"总文件数:           {len(json_files)}")
    print(f"成功:               {success}")
    print(f"失败:               {failed}")
    print(f"处理文档数:         {counters.processed_docs}")
    print(f"总 blocks:          {counters.total_blocks}")
    print(f"修复断词:           {counters.fixed_broken_words}")
    print(f"降级误判标题:       {counters.demoted_false_headings}")
    print(f"检测到 subsection:  {counters.detected_subsections}")
    print(f"检测到 figure cap:  {counters.detected_figure_captions}")
    print(f"检测到 table cap:   {counters.detected_table_captions}")
    print(f"检测到 table text:  {counters.detected_table_text_blocks}")
    print(f"检测到 references:  {counters.detected_references_blocks}")
    print(f"合并拆分标题:       {counters.merged_split_headings}")
    print(f"耗时:               {elapsed:.2f}s")
    print(f"输出目录:           {output_dir}")
    print(f"预览目录:           {preview_dir}")

    if failed_list:
        print()
        print("失败列表:")
        for item in failed_list:
            print(f"  - {item}")

    return counters


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="parsed_raw → parsed_clean + parsed_preview 清洗脚本"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="parsed_raw 目录（包含原始解析 JSON）",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="parsed_clean 输出目录",
    )
    parser.add_argument(
        "--preview_dir", required=True,
        help="parsed_preview 输出目录（Markdown 预览）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    preview_dir = Path(args.preview_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("parsed_raw → parsed_clean 清洗")
    print("=" * 60)
    print(f"输入目录:   {input_dir}")
    print(f"输出目录:   {output_dir}")
    print(f"预览目录:   {preview_dir}")
    print()

    batch_process(input_dir, output_dir, preview_dir)


if __name__ == "__main__":
    main()
