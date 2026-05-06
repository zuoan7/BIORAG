#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科研论文文本后处理与分块脚本
用于合成生物学论文 RAG 的数据预处理，输出适合 embedding 和导入 Milvus 的 JSONL 文件。

支持输入格式: txt, json, pdf

用法:
    # 从已解析的 txt/json 文件处理
    python preprocess_and_chunk.py --input_dir ./parsed_papers --output_dir ./chunks

    # 直接从 PDF 文件处理（需要 pymupdf）
    python preprocess_and_chunk.py --input_dir ./pdf_papers --output_dir ./chunks

    # 自定义分块参数
    python preprocess_and_chunk.py --input_dir ./parsed_papers --output_dir ./chunks --chunk_size 800 --chunk_overlap 120
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.document_cleaning_v5 import (
    evidence_pack_to_pages,
    is_contaminated_evidence_text,
    looks_like_false_table_text_body_sentence,
)


# ============================================================
# 常量与配置
# ============================================================

SECTION_PATTERNS = [
    (re.compile(r"^abstract\b", re.I), "Abstract"),
    (re.compile(r"^introduction\b", re.I), "Introduction"),
    (re.compile(r"^background\b", re.I), "Background"),
    (re.compile(r"^(?:results?\s+and\s+discussion|discussion\s+and\s+results?)\b", re.I), "Results and Discussion"),
    (re.compile(r"^results?\b", re.I), "Results"),
    (re.compile(r"^discussion\b", re.I), "Discussion"),
    (re.compile(r"^conclusions?\b", re.I), "Conclusion"),
    (re.compile(r"^(?:materials?\s+and\s+methods|methods?\s+and\s+materials?)\b", re.I), "Materials and Methods"),
    (re.compile(r"^methods?\b", re.I), "Methods"),
    (re.compile(r"^experimental\s+(?:procedures?|section|methods?)\b", re.I), "Experimental Procedures"),
]

# Subsection 识别模式：匹配 "数字.数字 标题" 格式
SUBSECTION_PATTERN = re.compile(
    r"^(\d+\.\d+\.?\s+\S+(?:\s+\S+){0,14})\s*$",
)

SECTION_NUMBERED_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)"
    r"(abstract|introduction|background|results?\s+and\s+discussion|"
    r"discussion\s+and\s+results?|results?|discussion|conclusions?|"
    r"materials?\s+and\s+methods|methods?\s+and\s+materials?|methods?|"
    r"experimental\s+(?:procedures?|section|methods?))\b",
    re.I,
)

REFERENCE_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?"
    r"(references|bibliography|literature\s+cited|works\s+cited)\b",
    re.I,
)

_METADATA_HEADINGS = {
    "acknowledgements", "acknowledgments", "funding", "data availability",
    "author contributions", "competing interests", "competing interest",
    "conflict of interest", "conflicts of interest",
    "corresponding author", "corresponding authors",
    "associated content", "author information", "supporting information",
    "references", "bibliography", "literature cited", "works cited",
    "supplementary material", "supplementary data", "supplementary information",
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*doi\s*:", re.I),
    re.compile(r"^\s*https?://", re.I),
    re.compile(r"^\s*www\.", re.I),
    re.compile(r"copyright\s", re.I),
    re.compile(r"all\s+rights\s+reserved", re.I),
    re.compile(r"^\s*received\s*:", re.I),
    re.compile(r"^\s*accepted\s*:", re.I),
    re.compile(r"^\s*published\s*:", re.I),
    re.compile(r"^\s*available\s+online\s", re.I),
    re.compile(r"^\s*downloaded\s+from", re.I),
    re.compile(r"^\s*this\s+article\s+is\s+protected\s+by\s+copyright", re.I),
    re.compile(r"^\s*see\s+last\s+page\s+for\s+", re.I),
    re.compile(r"^\s*vol\.\s*\d+", re.I),
    re.compile(r"^\s*pp?\.\s*\d+", re.I),
    re.compile(r"^\s*issn\s*", re.I),
    re.compile(r"^\s*e-?mail\s*:", re.I),
    re.compile(r"^\s*\*\s*corresponding\s+author", re.I),
    re.compile(r"^\s*supplementary\s+(?:material|data|information|figure|table)\b", re.I),
]

HEADER_FOOTER_PATTERN = re.compile(
    r"^\s*"
    r"(?:journal\s+of|proceedings\s+of|nature|science|cell|pnas|acs|elsevier|springer|wiley|taylor\s+&\s+francis)"
    r"\b",
    re.I,
)

HYPHEN_BREAK_PATTERN = re.compile(r"(\w)-\s+(\w)")
LEADING_MARKER_PATTERN = re.compile(r"^\s*(?:[#>*\-■▪►●→▸▹◆◇⬛]+)\s*")
TITLE_EXPLICIT_PATTERN = re.compile(r"^\s*(?:title|article)\s*:\s*(.+)$", re.I)
PAGE_MARKER_PATTERN = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of|/)\s*\d+\s*$", re.I)
DOWNLOADED_LINE_PATTERN = re.compile(r"^\s*downloaded\s+via\b", re.I)
ARTICLE_META_PATTERN = re.compile(
    r"(metrics\s*&\s*more|article\s+recommendations|author[’']?s\s+accepted\s+manuscript|"
    r"received:\s|accepted:\s|published:\s|open\s+access|full\s+terms\s*&\s*conditions)",
    re.I,
)
AUTHOR_NAME_PATTERN = re.compile(r"^[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){2,}")
JOURNAL_PREFIX_PATTERN = re.compile(
    r"^(?:biotechnology|journal|applied|front\.|nature|science|cell|nutrients|food\s+|enzyme\s+|"
    r"environmental|academic editor|published online|mol\.\s*sci\.|research article|review article)",
    re.I,
)

# Section fallback: body sections that should exist in a well-structured paper
_BODY_SECTIONS: set[str] = {
    "Introduction", "Background", "Methods", "Materials and Methods",
    "Experimental Section", "Experimental Procedures", "Results",
    "Results and Discussion", "Discussion", "Conclusion", "Conclusions",
    "Full Text",
}

# Fallback heading patterns: used when block-level section_heading metadata is broken.
# These are more conservative than SECTION_PATTERNS — they require the heading to
# be a standalone short line, not a substring match in running text.
_FALLBACK_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*(?:#+\s*)?(?:Introduction|INTROD\w+)\s*$"), "Introduction"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Background)\s*$", re.I), "Background"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Materials?\s+and\s+Methods|Methods?\s+and\s+Materials?)\s*$", re.I), "Materials and Methods"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Methods?)\s*$", re.I), "Methods"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Experimental\s+(?:Procedures?|Section|Methods?))\s*$", re.I), "Experimental Procedures"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Results?\s+and\s+Discussion|Discussion\s+and\s+Results?)\s*$", re.I), "Results and Discussion"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Results?)\s*$", re.I), "Results"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Discussion)\s*$", re.I), "Discussion"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Conclusions?)\s*$", re.I), "Conclusion"),
    (re.compile(r"^\s*(?:#+\s*)?(?:Abstract)\s*$", re.I), "Abstract"),
]

# Numbered heading: "1. Introduction", "2. Results", "3.1. Discussion" etc.
_FALLBACK_NUMBERED_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)(.+?)\s*$"
)

# Patterns that should NOT be treated as section headings even if they
# contain section-like keywords.
_FALSE_HEADING_FILTERS: list[re.Pattern] = [
    re.compile(r"^\s*(?:#+\s*)?(?:Received|Accepted|Published|Revised)\s*:", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:DOI|ORCID)\s*[:\d]", re.I),
    re.compile(r"^\s*https?://", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:Correspondence|Corresponding\s+Author)", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:Author\s+Contributions|Funding|Conflict\s+of\s+Interest|Competing\s+Interests)", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:Data\s+Availability|Ethics\s+Statement|Acknowledgments?|Acknowledgements)", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:Supplementary\s+(?:Material|Data|Information|Figure|Table))", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:References|Bibliography|Publisher'?s?\s+Note)", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:FIGURE|TABLE)\s*\d+", re.I),
    re.compile(r"^\s*(?:#+\s*)?(?:Supporting\s+Information)", re.I),
    # Reference entry lines: "7. Rabinowitz, M., and Lipmann, F. (1960)..."
    re.compile(r"^\s*\d+\.\s+[A-Z][a-z'-]+,\s+[A-Z]\.", re.I),
    # Address/zip code lines: "79104 Freiburg, Germany"
    re.compile(r"^\s*\d{4,6}\s+\w+,\s+\w+", re.I),
    # Instrument parameters: "1950 V; and mass range, 20-2000 m/z."
    re.compile(r"^\s*\d+\s*V\s*;", re.I),
    # Grant number lines: "2020B020226007), and the..."
    re.compile(r"^\s*\d{8,}[)）]", re.I),
    # Figure/panel references in body text: "5A,B). The pykA-knockout..."
    re.compile(r"^\s*\d+[A-F](?:,[A-F])*[)）]\.?\s+[a-z]", re.I),
    # Author name lines (3+ capitalized words)
    re.compile(r"^[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){2,}\s*$"),
]


# ============================================================
# 数据类
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    title: str
    section: str
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_index: int
    token_count: int
    text: str
    retrieval_text: str
    quality_score: float
    section_path: list[str] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    layout_columns: list[str] = field(default_factory=list)
    reading_order_span: dict[str, Optional[int]] = field(default_factory=dict)
    bbox_span: dict[str, Optional[float]] = field(default_factory=dict)
    source_block_metadata: list[dict[str, Any]] = field(default_factory=list)
    excluded_block_counts: dict[str, int] = field(default_factory=dict)
    contains_figure_caption: bool = False
    contains_table_caption: bool = False
    contains_table_text: bool = False
    contains_references: bool = False
    contains_metadata: bool = False
    contains_noise: bool = False
    contains_image: bool = False
    parser_stage: str = ""


@dataclass
class ProcessingStats:
    total_docs: int = 0
    success_docs: int = 0
    failed_docs: int = 0
    low_quality_docs: int = 0
    total_chunks: int = 0


# ============================================================
# Token 计数（可替换为真实 tokenizer）
# ============================================================

def count_tokens(text: str) -> int:
    """
    近似 token 计数：以英文单词数近似。
    后续可替换为 tiktoken / sentencepiece 等真实 tokenizer，
    只需保证函数签名不变 (str -> int)。
    """
    return len(text.split())


# ============================================================
# 文本清洗
# ============================================================

def clean_text(raw: str) -> str:
    """对原始文本执行基础清洗，返回清洗后的文本。"""
    lines = raw.split("\n")
    cleaned_lines = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        normalized = _normalize_heading_line(stripped)

        if _is_noise_line(normalized):
            continue

        # The first page often contains download banners, journal headers, and author blocks.
        if idx < 40 and _is_front_matter_noise(normalized):
            continue

        cleaned_lines.append(normalized)

    text = "\n".join(cleaned_lines)

    text = HYPHEN_BREAK_PATTERN.sub(r"\1\2", text)

    text = _merge_broken_lines(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _is_noise_line(line: str) -> bool:
    """判断一行是否为噪声行（页码、DOI、版权信息等）。"""
    if not line:
        return False

    for pat in NOISE_LINE_PATTERNS:
        if pat.search(line):
            return True

    if PAGE_MARKER_PATTERN.match(line):
        return True
    if DOWNLOADED_LINE_PATTERN.match(line):
        return True
    if ARTICLE_META_PATTERN.search(line):
        return True

    if HEADER_FOOTER_PATTERN.search(line):
        if len(line.split()) <= 10:
            return True

    return False


def _is_front_matter_noise(line: str) -> bool:
    if not line:
        return False
    if _looks_like_title(line):
        return False
    lowered = line.lower()
    if lowered in {"article", "research article", "review article", "accepted manuscript"}:
        return True
    if "correspondence" in lowered or "supplementary material" in lowered:
        return True
    if "@" in line:
        return True
    if AUTHOR_NAME_PATTERN.match(line):
        digit_ratio = sum(ch.isdigit() for ch in line) / max(len(line), 1)
        if digit_ratio < 0.12 and line.count(",") >= 1:
            return True
    return False


def _normalize_heading_line(line: str) -> str:
    if not line:
        return ""
    line = line.replace("\u00a0", " ").strip()
    line = LEADING_MARKER_PATTERN.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _merge_broken_lines(text: str) -> str:
    """
    合并 PDF 解析导致的断行。
    规则：如果一行以非句末标点结尾，且下一行以小写字母开头，则合并。
    """
    lines = text.split("\n")
    if not lines:
        return text

    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if (
            prev
            and not prev.endswith((".", "!", "?", ":", ";", ")", "]", "}", "—", "–"))
            and line
            and line[0].islower()
        ):
            merged[-1] = prev + " " + line
        else:
            merged.append(line)

    return "\n".join(merged)


# ============================================================
# 参考文献截断
# ============================================================

def truncate_at_references(text: str) -> str:
    """
    识别 References / Bibliography / Literature Cited 标题，
    截断其后的所有内容。
    """
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        if REFERENCE_SECTION_PATTERN.match(line.strip()):
            break
        result_lines.append(line)
    return "\n".join(result_lines)


# ============================================================
# 标题提取
# ============================================================

def extract_title(text: str, max_lines: int = 20) -> str:
    """
    从文档开头提取论文标题。
    策略：在前 max_lines 行中，找最像标题的一行（较长、不全大写、不含标点过多）。
    """
    lines = [_normalize_heading_line(l) for l in text.split("\n")[: max_lines * 2] if l.strip()]

    if not lines:
        return ""

    explicit_title = _extract_explicit_title(lines)
    if explicit_title:
        return explicit_title

    for idx, line in enumerate(lines[:12]):
        if _is_noise_line(line) or _is_front_matter_noise(line):
            continue
        candidate = line
        if idx + 1 < len(lines) and _looks_like_title_tail(lines[idx + 1]):
            candidate = f"{line} {lines[idx + 1].strip()}"
        if _is_primary_title_candidate(candidate):
            return candidate

    best = ""
    best_score = -1

    for idx, line in enumerate(lines[:max_lines]):
        if _is_noise_line(line) or _is_front_matter_noise(line):
            continue

        merged_line = line
        if idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if _looks_like_title(line) and _looks_like_title_tail(next_line):
                merged_line = f"{line} {next_line}"

        score = 0.0
        word_count = len(merged_line.split())
        digit_ratio = sum(ch.isdigit() for ch in merged_line) / max(len(merged_line), 1)
        punctuation_count = sum(ch in ",;:|/[]" for ch in merged_line)

        if 5 <= word_count <= 24:
            score += 12
        elif 3 <= word_count <= 32:
            score += 6

        if merged_line[0].isupper():
            score += 3

        if idx <= 2:
            score += 4

        if merged_line == merged_line.title():
            score += 2

        upper_ratio = sum(1 for c in merged_line if c.isupper()) / max(len(merged_line), 1)
        if upper_ratio > 0.6 and word_count > 3:
            score -= 5

        if digit_ratio > 0.12:
            score -= 4

        if punctuation_count > 3:
            score -= 3

        if merged_line.endswith((".", ",", ":", ";")):
            score -= 5

        lowered = merged_line.lower()
        if any(kw in lowered for kw in ("university", "department", "email", "corresponding", "received", "accepted")):
            score -= 10

        if any(kw in lowered for kw in ("abstract", "introduction", "keywords", "key words", "doi")):
            score -= 10

        if JOURNAL_PREFIX_PATTERN.match(merged_line):
            score -= 6

        if AUTHOR_NAME_PATTERN.match(merged_line) or merged_line.count(",") >= 2:
            score -= 8

        if _looks_like_title(merged_line):
            score += 5

        if score > best_score:
            best_score = score
            best = merged_line

    return best if best_score > 0 else ""


def _extract_explicit_title(lines: list[str]) -> str:
    for line in lines[:12]:
        match = TITLE_EXPLICIT_PATTERN.match(line)
        if match:
            candidate = match.group(1).strip(" .:-")
            if _looks_like_title(candidate):
                return candidate
    return ""


def _looks_like_title(line: str) -> bool:
    if not line:
        return False
    lowered = line.lower()
    words = line.split()
    if len(words) < 4:
        return False
    if JOURNAL_PREFIX_PATTERN.match(line):
        return False
    if ARTICLE_META_PATTERN.search(line):
        return False
    if "@" in line:
        return False
    if line.count(",") >= 3 and ":" not in line:
        return False
    digit_ratio = sum(ch.isdigit() for ch in line) / max(len(line), 1)
    if digit_ratio > 0.08:
        return False
    if any(token in lowered for token in ("vol.", "doi", "www.", "http://", "https://", "data availability statement")):
        return False
    if lowered.startswith(("college of", "school of", "department of", "people's republic", "funding:", "competing interests:")):
        return False
    alpha_words = [token for token in re.split(r"\s+", line) if re.search(r"[A-Za-z]", token)]
    if not alpha_words:
        return False
    if len(alpha_words) >= 6:
        return True
    return any(token in lowered for token in (" of ", " for ", " by ", " via ", " using ", " from ", " with ", ":"))


def _looks_like_title_tail(line: str) -> bool:
    if not line:
        return False
    if len(line.split()) > 4:
        return False
    if line.count(",") > 0:
        return False
    return all(token[:1].isupper() for token in line.split() if token)


def _is_primary_title_candidate(line: str) -> bool:
    if not _looks_like_title(line):
        return False
    words = line.split()
    if len(words) < 6:
        return False
    lowered = line.lower()
    if any(token in lowered for token in ("abstract", "introduction", "materials and methods", "keywords")):
        return False
    if line.count(",") >= 3:
        return False
    return True


# ============================================================
# Section 识别
# ============================================================

@dataclass
class SectionBlock:
    name: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def identify_sections(text: str, pages: Optional[list] = None) -> list[SectionBlock]:
    """
    识别论文中的 section，将文本按 section 分段。
    如果未识别到任何 section，返回一个兜底 block。
    """
    lines = text.split("\n")

    section_starts: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = _normalize_heading_line(line)
        if not stripped:
            continue

        matched_name = _match_section_title(stripped)
        if matched_name:
            section_starts.append((i, matched_name))

    if not section_starts:
        inferred = _infer_intro_abstract_blocks(lines, pages)
        if inferred:
            return inferred
        return [SectionBlock(name="Full Text", text=text)]

    if section_starts[0][0] > 0:
        preamble = "\n".join(lines[: section_starts[0][0]]).strip()
        if _looks_like_abstract_block(preamble):
            pg_start, pg_end = _estimate_page_range(lines, 0, section_starts[0][0] - 1, pages)
            section_starts.insert(0, (0, "Abstract"))

    blocks = []
    for idx, (start_line, sec_name) in enumerate(section_starts):
        if idx + 1 < len(section_starts):
            end_line = section_starts[idx + 1][0]
        else:
            end_line = len(lines)

        content_start = start_line + 1
        section_text = "\n".join(lines[content_start:end_line]).strip()

        pg_start, pg_end = _estimate_page_range(
            lines, start_line, end_line - 1, pages
        )

        blocks.append(SectionBlock(
            name=sec_name,
            text=section_text,
            page_start=pg_start,
            page_end=pg_end,
        ))

    return blocks


def _match_section_title(line: str) -> Optional[str]:
    """匹配行是否为 section 标题，返回标准化 section 名或 None。"""
    normalized = _normalize_heading_line(line)
    for pat, name in SECTION_PATTERNS:
        if pat.match(normalized):
            return name

    m = SECTION_NUMBERED_PATTERN.match(normalized)
    if m:
        raw = m.group(1).strip().lower()
        for pat, name in SECTION_PATTERNS:
            if pat.match(raw):
                return name

    # 识别 subsection 标题（如 "2.3. In vitro fermentation..."）
    subsec_match = SUBSECTION_PATTERN.match(normalized)
    if subsec_match:
        subsec_text = subsec_match.group(1).strip()
        # 尝试匹配 subsection 中的 section 关键词
        for pat, name in SECTION_PATTERNS:
            if pat.search(subsec_text):
                return name
        # 返回 subsection 原始标题作为 section 名
        return subsec_text

    return None


# ============================================================
# Section fallback: 修复 index metadata 错误的文档
# ============================================================


def _needs_section_fallback(section_groups: list[dict]) -> bool:
    """
    检查文档是否需要 section fallback。

    触发条件:
    1. body section 的 block 占比 < 10%
    2. Title/Abstract/Unknown 的 block 占比 > 80%

    满足两个条件才触发，避免影响结构正常的文档。
    """
    if not section_groups:
        return False

    total_blocks = sum(len(sg["blocks"]) for sg in section_groups)
    if total_blocks < 3:
        return False

    body_count = sum(
        len(sg["blocks"]) for sg in section_groups
        if sg["section"] in _BODY_SECTIONS
    )
    title_au_count = sum(
        len(sg["blocks"]) for sg in section_groups
        if sg["section"] in ("Title", "Abstract", "Unknown", "")
    )

    body_ratio = body_count / total_blocks
    tau_ratio = title_au_count / total_blocks

    return body_ratio < 0.10 and tau_ratio > 0.80


def _clean_heading_candidate(text: str) -> str:
    """清洗 heading 候选文本：去掉 markdown markers 和编号前缀。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?)\s*", "", cleaned)
    return cleaned.strip()


def _is_false_heading(text: str) -> bool:
    """检查文本是否为误标 heading（作者名、地址、元数据等）。"""
    for pat in _FALSE_HEADING_FILTERS:
        if pat.search(text):
            return True
    return False


def _detect_true_section_headings(
    all_blocks: list[dict],
) -> list[tuple[int, str]]:
    """
    在所有 block 的文本中检测真正的 section heading。

    返回: [(block_index, canonical_section_name), ...]
    按 block_index 排序。
    """
    detections: list[tuple[int, str]] = []

    for i, block in enumerate(all_blocks):
        text = block["text"].strip()
        if not text or len(text) > 250:  # heading 不应太长
            continue

        # 尝试 numbered heading 模式: "1. Introduction" 等
        nm = _FALLBACK_NUMBERED_HEADING.match(text)
        candidate = nm.group(1).strip() if nm else text

        # 清洗
        cleaned = _clean_heading_candidate(candidate)
        if not cleaned or len(cleaned) > 120:
            continue

        # 过滤误标
        if _is_false_heading(cleaned):
            continue

        # 匹配标准 section
        for pat, name in _FALLBACK_SECTION_PATTERNS:
            if pat.match(cleaned):
                # 额外安全检查：避免匹配到正文中嵌的 section 关键词
                # "the results showed that..." 不应被识别为 heading
                if nm or re.match(r"^\s*(?:#+\s*)?[A-Z]", text):
                    detections.append((i, name))
                break

    return detections


def _apply_fallback_section_grouping(
    all_blocks: list[dict],
    detected_headings: list[tuple[int, str]],
) -> list[dict]:
    """
    基于检测到的真实 section heading 重新分组 blocks。

    - 第一个 heading 之前的 blocks → 保持原始 section (Title/Abstract)
    - heading block 本身 → 作为 section marker
    - heading 后的 blocks → 归入该 section，直到下一个 heading
    - References 及之后的 → 标记为 Metadata（跳过）
    """
    if len(detected_headings) < 2:
        return []

    # 按 block_index 排序
    detected_headings.sort(key=lambda h: h[0])

    # 找到 "References" 类 heading 的边界
    reference_sections = {"References", "Bibliography", "Supplementary Material"}
    first_ref_idx: Optional[int] = None
    for idx, name in detected_headings:
        if name in reference_sections or _is_false_heading(name):
            first_ref_idx = idx
            break

    # 过滤 References 之后的 heading
    clean_headings = []
    for idx, name in detected_headings:
        if first_ref_idx is not None and idx >= first_ref_idx:
            break
        if name in reference_sections:
            break
        clean_headings.append((idx, name))

    if len(clean_headings) < 2:
        return []

    groups: list[dict] = []
    heading_indices = {h[0] for h in clean_headings}

    # 第一个 heading 之前的 blocks → 保持原始分类
    first_heading_idx = clean_headings[0][0]
    pre_heading_blocks = [b for i, b in enumerate(all_blocks) if i < first_heading_idx]
    if pre_heading_blocks:
        # 尝试识别 Abstract section
        abstract_idx = None
        for h_idx, h_name in clean_headings:
            if h_name == "Abstract":
                abstract_idx = h_idx
                break

        if abstract_idx is not None:
            # Abstract 之前 → Title; Abstract 到下一个 heading → Abstract
            abstract_heading_idx_in_headings = next(
                (j for j, (_, name) in enumerate(clean_headings) if name == "Abstract"), None
            )
            if abstract_heading_idx_in_headings is not None and abstract_heading_idx_in_headings > 0:
                # Abstract 不是第一个 heading → 前面的归 Title/Abstract
                mid_blocks = [b for i, b in enumerate(all_blocks) if i < abstract_idx]
                abstract_start = clean_headings[abstract_heading_idx_in_headings][0]
                abstract_blocks_list = [
                    b for i, b in enumerate(all_blocks)
                    if abstract_start <= i < (clean_headings[abstract_heading_idx_in_headings + 1][0]
                                               if abstract_heading_idx_in_headings + 1 < len(clean_headings)
                                               else len(all_blocks))
                ]
            else:
                # 只保留非 Abstract heading 之前的内容
                mid_blocks = pre_heading_blocks
                abstract_blocks_list = []
        else:
            mid_blocks = pre_heading_blocks
            abstract_blocks_list = []

        if mid_blocks:
            groups.append({
                "section": "Title",
                "section_path": ["Title"],
                "blocks": mid_blocks,
            })
        if abstract_blocks_list:
            groups.append({
                "section": "Abstract",
                "section_path": ["Abstract"],
                "blocks": abstract_blocks_list,
            })

    # 按 heading 分组
    for h_pos in range(len(clean_headings)):
        h_idx, h_name = clean_headings[h_pos]
        next_idx = clean_headings[h_pos + 1][0] if h_pos + 1 < len(clean_headings) else len(all_blocks)

        section_blocks = []
        for i in range(h_idx, next_idx):
            if i < len(all_blocks):
                section_blocks.append(all_blocks[i])

        if section_blocks:
            groups.append({
                "section": h_name,
                "section_path": [h_name],
                "blocks": section_blocks,
            })

    # 如果 fallback 分组后仍然没有 body section，返回空（让调用方使用原始分组）
    has_body = any(g["section"] in _BODY_SECTIONS for g in groups)
    if not has_body:
        return []

    return groups


def _apply_generic_fulltext_fallback(
    section_groups: list[dict],
    all_blocks: list[dict],
) -> list[dict]:
    """
    Path B fallback: 无法检测到具体 section heading 时使用。

    将误标为 Title/Abstract 的正文 blocks 重新归入 "Full Text" section，
    使它们能通过 BODY_ANY group 匹配。

    策略:
    - Title/Abstract section 中的前 2 个 blocks 保留原 section
    - Title/Abstract section 中剩余的 blocks → 归入 "Full Text"
    - 后续的非元数据 section → 也归入 "Full Text"
    """
    new_groups: list[dict] = []
    body_blocks: list[dict] = []
    seen_body = False

    for sg in section_groups:
        sec_name = sg["section"]
        blocks = sg["blocks"]

        if sec_name in ("Title", "Abstract") and not seen_body:
            # 前 2 个 blocks 保留为原 section，其余的进入 Full Text
            keep = blocks[:2]
            rest = blocks[2:]
            if keep:
                new_groups.append({
                    "section": sec_name,
                    "section_path": [sec_name],
                    "blocks": keep,
                })
            if rest:
                body_blocks.extend(rest)
                seen_body = True
        elif sec_name in ("References", "Bibliography"):
            continue
        elif _is_false_heading(sec_name):
            continue
        else:
            body_blocks.extend(blocks)

    if body_blocks:
        filtered = _filter_body_blocks(body_blocks)
        if filtered:
            new_groups.append({
                "section": "Full Text",
                "section_path": ["Full Text"],
                "blocks": filtered,
            })

    if len(new_groups) <= 1 and all(
        sg["section"] in ("Title", "Abstract") for sg in new_groups
    ):
        return []

    return new_groups


def _filter_body_blocks(blocks: list[dict]) -> list[dict]:
    """过滤 body blocks 中的元数据行和 noise。"""
    result = []
    for b in blocks:
        text = b.get("text", "").strip()
        if len(text) < 20:
            continue
        lower = text.lower()
        if any(kw in lower for kw in (
            "received:", "accepted:", "published:", "doi:", "http://", "https://",
            "correspondence", "corresponding author", "equal contribution",
            "supplementary material", "supporting information",
        )):
            continue
        result.append(b)
    return result


_INLINE_ABSTRACT_RE = re.compile(
    r"^(.*?)\b(ABSTRACT)\s*:\s*", re.I
)
# Standalone ABSTRACT without colon at paragraph start
_ABSTRACT_NO_COLON_RE = re.compile(
    r"^(ABSTRACT)\s+(.{30,})", re.I
)


def _split_inline_abstract(
    btext: str,
    page_num: int,
    block: dict,
) -> list[dict] | None:
    """检测并拆分 paragraph block 中的 inline ABSTRACT heading。

    处理两种形式:
    1. "prefix ABSTRACT: content..." (带冒号，可能与前置文本合并)
    2. "ABSTRACT content..." (无冒号，独立行首)
    """
    m = _INLINE_ABSTRACT_RE.match(btext)
    if m:
        prefix = m.group(1).strip()
        abstract_text = btext[m.end():].strip()
    else:
        m2 = _ABSTRACT_NO_COLON_RE.match(btext)
        if m2:
            prefix = ""
            abstract_text = m2.group(2).strip()
        else:
            return None

    # 安全检查: prefix 太短 → 跳过
    if prefix and len(prefix) < 3:
        return None
    # abstract_text 太短 → 不是真正的 abstract section
    if len(abstract_text) < 30:
        return None
    # prefix 看起来像普通句子 → 不宜拆分
    # (例如 "in this abstract we..." → 不应拆分)
    if prefix and re.search(r'(?i)\b(?:in|the|this|an)\s+abstract\b', prefix):
        return None

    result: list[dict] = []
    section_path = block.get("section_path", [])

    if prefix:
        # 过滤掉已知的 metadata 前缀（如 "Supporting Information"）
        prefix_lower = prefix.lower().strip().rstrip(".")
        if prefix_lower in _METADATA_HEADINGS:
            # metadata → 跳过，不保留为 paragraph
            pass
        else:
            result.append({
                "type": "paragraph",
                "text": prefix,
                "section_path": section_path,
                "page": page_num,
            })

    # ABSTRACT heading
    result.append({
        "type": "section_heading",
        "text": "ABSTRACT",
        "section_path": section_path,
        "page": page_num,
    })

    # Abstract 正文
    if abstract_text:
        result.append({
            "type": "paragraph",
            "text": abstract_text,
            "section_path": section_path,
            "page": page_num,
        })

    return result


def _infer_intro_abstract_blocks(lines: list[str], pages: Optional[list]) -> list[SectionBlock]:
    paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    blocks: list[SectionBlock] = []
    if len(paragraphs) >= 2 and _looks_like_abstract_block(paragraphs[0]):
        abstract_text = paragraphs[0]
        remainder = "\n\n".join(paragraphs[1:]).strip()
        if remainder:
            blocks.append(SectionBlock(name="Abstract", text=abstract_text))
            blocks.append(SectionBlock(name="Full Text", text=remainder))
            return blocks
    return []


def _looks_like_abstract_block(text: str) -> bool:
    words = text.split()
    if not 60 <= len(words) <= 320:
        return False
    lowered = text.lower()
    return any(
        token in lowered
        for token in ("this study", "we investigated", "we report", "results", "conclusion", "background")
    )


def _estimate_page_range(
    lines: list[str],
    start_line: int,
    end_line: int,
    pages: Optional[list],
) -> tuple[Optional[int], Optional[int]]:
    """
    估计 section 的页码范围。
    对于 json 输入（有 pages 信息），通过累积字符数估算；
    对于 txt 输入（无 pages 信息），返回 (None, None)。
    """
    if not pages:
        return None, None

    total_chars = sum(len(p.get("text", "")) for p in pages)
    if total_chars == 0:
        return None, None

    text_before_start = sum(len(l) + 1 for l in lines[:start_line])
    text_before_end = sum(len(l) + 1 for l in lines[:end_line + 1])

    cumulative = 0
    pg_start = None
    pg_end = None

    for p in pages:
        page_num = p.get("page", 0)
        page_len = len(p.get("text", ""))
        page_cum_end = cumulative + page_len

        if pg_start is None and page_cum_end >= text_before_start:
            pg_start = page_num
        if page_cum_end >= text_before_end:
            pg_end = page_num
            break

        cumulative = page_cum_end

    if pg_start is None:
        pg_start = pages[0].get("page", 1)
    if pg_end is None:
        pg_end = pages[-1].get("page", 1)

    return pg_start, pg_end


# ============================================================
# Block-based chunking（优先路径，基于 parsed_clean 的 blocks）
# ============================================================

def _chunk_source_block(block: dict, btype: str, text: str, page_num: int) -> dict:
    metadata = block.get("metadata", {}) or {}
    block_id = _normalize_block_id(block)
    source_block_id = _normalize_source_block_id(block) or block_id
    bbox = _normalize_bbox_value(block)
    column = _normalize_column_value(block)
    reading_order = _normalize_reading_order_value(block)
    page = _normalize_page_value(block, page_num)
    return {
        "type": btype,
        "text": text,
        "section_path": block.get("section_path", []),
        "page": page,
        "block_id": block_id,
        "source_block_id": source_block_id,
        "metadata": metadata,
        "bbox": bbox,
        "column": column,
        "reading_order": reading_order,
    }


def _block_text_for_chunk(block: dict) -> str:
    btype = block.get("type", "paragraph")
    btext = block.get("text", "")
    if btype == "section_heading":
        return f"## {btext.lstrip('#').strip()}"
    if btype == "subsection_heading":
        return f"### {btext.lstrip('#').strip()}"
    if btype == "figure_caption":
        return f"[FIGURE CAPTION] {btext}"
    if btype == "table_caption":
        return f"[TABLE CAPTION] {btext}"
    if btype == "table_text":
        return f"[TABLE TEXT] {btext}"
    return btext


def _unique_preserve_order(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _normalize_block_id(block: dict[str, Any]) -> Any:
    metadata = block.get("metadata", {}) or {}
    return _first_present(block, ("block_id", "id")) or _first_present(metadata, ("block_id", "id"))


def _normalize_source_block_id(block: dict[str, Any]) -> Any:
    metadata = block.get("metadata", {}) or {}
    return (
        _first_present(metadata, ("source_block_id", "block_id", "id"))
        or _first_present(block, ("source_block_id", "block_id", "id"))
    )


def _normalize_page_value(block: dict[str, Any], fallback_page: int | None = None) -> Any:
    metadata = block.get("metadata", {}) or {}
    return (
        _first_present(metadata, ("page", "page_number"))
        or _first_present(block, ("page", "page_number"))
        or fallback_page
    )


def _normalize_column_value(block: dict[str, Any]) -> Any:
    metadata = block.get("metadata", {}) or {}
    return (
        _first_present(metadata, ("column", "layout_column"))
        or _first_present(block, ("column", "layout_column"))
    )


def _normalize_reading_order_value(block: dict[str, Any]) -> Any:
    metadata = block.get("metadata", {}) or {}
    return (
        _first_present(metadata, ("reading_order", "order"))
        or _first_present(block, ("reading_order", "order"))
    )


def _normalize_bbox_value(block: dict[str, Any]) -> Any:
    metadata = block.get("metadata", {}) or {}
    return _first_present(metadata, ("bbox", "box")) or _first_present(block, ("bbox", "box"))


def _compact_text_preview(text: Any, limit: int = 120) -> str:
    preview = re.sub(r"\s+", " ", str(text or "")).strip()
    return preview[:limit]


def _chunk_metadata_from_blocks(group: list[dict]) -> dict[str, Any]:
    real_blocks = [b for b in group if b.get("type") != "overlap"]
    block_types = _unique_preserve_order([b.get("type") for b in real_blocks])
    pages = sorted({
        int(page)
        for b in real_blocks
        for page in [_normalize_page_value(b)]
        if page is not None and str(page).isdigit()
    })
    columns = _unique_preserve_order([
        column
        for b in real_blocks
        for column in [_normalize_column_value(b)]
        if column is not None
    ])
    source_block_ids = _unique_preserve_order([
        source_block_id
        for b in real_blocks
        for source_block_id in [_normalize_source_block_id(b)]
        if source_block_id is not None
    ])
    block_ids = _unique_preserve_order([
        block_id
        for b in real_blocks
        for block_id in [_normalize_block_id(b)]
        if block_id is not None
    ])
    reading_orders = [
        int(reading_order)
        for b in real_blocks
        for reading_order in [_normalize_reading_order_value(b)]
        if isinstance(reading_order, int)
        or (isinstance(reading_order, str) and str(reading_order).isdigit())
    ]

    bboxes = []
    for b in real_blocks:
        bbox = _normalize_bbox_value(b)
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            except (TypeError, ValueError):
                pass

    source_block_metadata = []
    for b in real_blocks[:80]:
        block_id = _normalize_block_id(b)
        source_block_id = _normalize_source_block_id(b) or block_id
        page = _normalize_page_value(b)
        meta = {
            "block_id": block_id,
            "source_block_id": source_block_id,
            "type": b.get("type"),
            "page": int(page) if page is not None and str(page).isdigit() else page,
            "section_path": b.get("section_path", []),
            "text_preview": _compact_text_preview(b.get("text", "")),
        }
        source_block_metadata.append({k: v for k, v in meta.items() if v is not None})

    return {
        "source_block_ids": source_block_ids,
        "block_ids": block_ids,
        "block_types": sorted(set(block_types)),
        "evidence_types": sorted(set(block_types)),
        "page_numbers": pages,
        "layout_columns": columns,
        "reading_order_span": {
            "start": min(reading_orders) if reading_orders else None,
            "end": max(reading_orders) if reading_orders else None,
        },
        "bbox_span": {
            "x0": min((b[0] for b in bboxes), default=None),
            "y0": min((b[1] for b in bboxes), default=None),
            "x1": max((b[2] for b in bboxes), default=None),
            "y1": max((b[3] for b in bboxes), default=None),
        },
        "source_block_metadata": source_block_metadata,
        "contains_figure_caption": "figure_caption" in block_types,
        "contains_table_caption": "table_caption" in block_types,
        "contains_table_text": "table_text" in block_types,
        "contains_references": "references" in block_types,
        "contains_metadata": "metadata" in block_types,
        "contains_noise": "noise" in block_types,
        "contains_image": "image" in block_types or "image_caption" in block_types,
    }


def chunk_by_blocks(
    pages: list[dict],
    doc_id: str,
    source_file: str,
    parser_stage: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    skip_references: bool = True,
    base_excluded_block_counts: Optional[dict[str, int]] = None,
) -> list[Chunk]:
    """
    基于 pages[].blocks 的结构化分块。
    按 section_path 聚合相邻 paragraph block，遇 section/subsection heading
    开启新 section 上下文。figure/table caption 进入 chunk 并在 metadata 中标记。
    页码由 block.page 最小/最大值计算，不再通过字符偏移估算。
    """
    # 按文档顺序收集所有有效 block
    all_blocks: list[dict] = []
    excluded_block_counts: dict[str, int] = {
        "metadata": 0,
        "noise": 0,
        "image": 0,
        "references": 0,
        "contamination": 0,
    }
    if base_excluded_block_counts:
        for key, value in base_excluded_block_counts.items():
            if isinstance(value, int):
                excluded_block_counts[key] = excluded_block_counts.get(key, 0) + value
    doc_title = ""
    last_table_caption_page: int | None = None
    for p in pages:
        page_num = p.get("page", 1)
        for block in p.get("blocks", []):
            btype = block.get("type", "paragraph")
            btext = block.get("text", "").strip()
            if btype in excluded_block_counts and (btype != "references" or skip_references):
                excluded_block_counts[btype] += 1
            if btype in ("noise", "metadata", "image"):
                continue
            if not btext:
                continue
            if btype == "title" and not doc_title:
                doc_title = btext.lstrip("#").strip()
            # references 处理：跳过或保留
            if btype == "references" and skip_references:
                continue
            # 跳过 References section heading 本身
            if btype == "section_heading" and skip_references:
                heading_text = btext.lstrip("#").strip()
                if REFERENCE_SECTION_PATTERN.match(heading_text):
                    continue
            contaminated, _contamination_reason = is_contaminated_evidence_text(btext)
            if contaminated:
                excluded_block_counts["contamination"] += 1
                continue
            strong_table_context = (
                btype == "table_text"
                and last_table_caption_page is not None
                and isinstance(page_num, int)
                and page_num <= last_table_caption_page + 1
            )
            if btype == "table_text" and looks_like_false_table_text_body_sentence(btext, strong_table_context):
                btype = "paragraph"
            if btype == "table_caption":
                last_table_caption_page = page_num if isinstance(page_num, int) else last_table_caption_page
            elif btype in {"figure_caption", "section_heading", "subsection_heading"}:
                last_table_caption_page = None

            # 检测 paragraph block 中的 inline ABSTRACT: heading
            # 例如 "Supporting Information ABSTRACT: Thanks to its ease..."
            # PDF parser 有时把 ABSTRACT: 和前置内容合并到同一个 paragraph block
            if btype == "paragraph":
                split_blocks = _split_inline_abstract(btext, page_num, block)
                if split_blocks is not None:
                    all_blocks.extend(split_blocks)
                    continue

            all_blocks.append(_chunk_source_block(block, btype, btext, page_num))

    if not all_blocks:
        return []

    # 按 section 聚合 blocks
    # 每个 section group 包含同一 section 下的连续 paragraph/caption blocks
    section_groups: list[dict] = []  # [{section, section_path, blocks: [{type, text, page}]}]

    current_section = ""
    current_section_path: list[str] = []
    current_blocks: list[dict] = []

    for block in all_blocks:
        btype = block["type"]
        btext = block["text"]
        bsection_path = block.get("section_path", [])
        bpage = block.get("page", 1)

        if btype in ("section_heading", "subsection_heading"):
            heading_text = btext.lstrip("#").strip()
            normalized = _normalize_heading_line(heading_text)
            standard_name = _match_section_title(normalized)

            is_metadata = normalized.lower().strip().rstrip(".") in _METADATA_HEADINGS

            if standard_name:
                # 保存当前 section group
                if current_blocks:
                    section_groups.append({
                        "section": current_section,
                        "section_path": list(current_section_path),
                        "blocks": current_blocks,
                    })
                heading_text = standard_name
                if btype == "section_heading":
                    current_section = heading_text
                    current_section_path = [heading_text]
                else:
                    if current_section_path:
                        if len(current_section_path) > 1:
                            current_section_path[-1] = heading_text
                        else:
                            current_section_path.append(heading_text)
                    else:
                        current_section_path = [heading_text]
                    current_section = heading_text
                current_blocks = [block]
                continue
            elif is_metadata:
                if skip_references:
                    continue
                current_blocks.append(block)
                continue
            elif current_section_path:
                current_blocks.append(block)
                continue
            else:
                if current_blocks:
                    section_groups.append({
                        "section": current_section,
                        "section_path": list(current_section_path),
                        "blocks": current_blocks,
                    })
                current_section = heading_text
                current_section_path = [heading_text]
                current_blocks = [block]
                continue

        if btype == "title":
            title_text = btext.lstrip("#").strip()
            title_normalized = _normalize_heading_line(title_text)
            title_as_section = _match_section_title(title_normalized)
            if title_as_section:
                if current_blocks:
                    section_groups.append({
                        "section": current_section,
                        "section_path": list(current_section_path),
                        "blocks": current_blocks,
                    })
                current_section = title_as_section
                current_section_path = [title_as_section]
                current_blocks = [block]
                continue
            elif current_section_path:
                current_blocks.append(block)
                continue
            else:
                if current_blocks:
                    section_groups.append({
                        "section": current_section,
                        "section_path": list(current_section_path),
                        "blocks": current_blocks,
                    })
                current_section = "Title"
                current_section_path = ["Title"]
                current_blocks = [block]
                continue

        if btype == "references":
            if skip_references:
                continue
            # 如果保留 references，归入当前 section

        # paragraph, figure_caption, table_caption 等
        current_blocks.append(block)

    # 最后一个 section group
    if current_blocks:
        section_groups.append({
            "section": current_section,
            "section_path": list(current_section_path),
            "blocks": current_blocks,
        })

    # Section fallback: 修复 index metadata 错误
    # 当 PDF parser 的 block-level section_heading 信息不可靠时，
    # 基于文本内容检测真实 section heading 并重新分组。
    if _needs_section_fallback(section_groups):
        detected = _detect_true_section_headings(all_blocks)
        old_sections = {sg["section"] for sg in section_groups}

        if len(detected) >= 2:
            # Path A: 检测到足够的标准 section heading → 精确重新分组
            fallback_groups = _apply_fallback_section_grouping(all_blocks, detected)
            if fallback_groups:
                new_sections = {sg["section"] for sg in fallback_groups}
                body_gained = [s for s in new_sections if s in _BODY_SECTIONS]
                if body_gained:
                    print(f"    [fallback:A] {doc_id}: sections {sorted(old_sections)} → "
                          f"{sorted(new_sections)} (gained: {body_gained})")
                    section_groups = fallback_groups
        else:
            # Path B: 无标准 section heading 可检测，但正文存在且被误标为 Title/Abstract
            # → 将非标题/非元数据的 blocks 重新标记为 "Full Text"
            fallback_groups = _apply_generic_fulltext_fallback(section_groups, all_blocks)
            if fallback_groups:
                new_sections = {sg["section"] for sg in fallback_groups}
                full_text_blocks = sum(
                    len(sg["blocks"]) for sg in fallback_groups
                    if sg["section"] == "Full Text"
                )
                if full_text_blocks > 0:
                    print(f"    [fallback:B] {doc_id}: sections {sorted(old_sections)} → "
                          f"{sorted(new_sections)} (Full Text blocks: {full_text_blocks})")
                    section_groups = fallback_groups

    # 对每个 section group 进行分块
    all_chunks: list[Chunk] = []
    chunk_idx = 0

    for sec_group in section_groups:
        sec_name = sec_group["section"] or "Unknown"
        sec_path = sec_group["section_path"]
        sec_blocks = sec_group["blocks"]

        # 将 blocks 按 chunk_size 聚合
        chunk_block_groups = _aggregate_blocks_into_chunks(
            sec_blocks, chunk_size, chunk_overlap
        )

        for group in chunk_block_groups:
            if not group:
                continue

            # 构造 chunk 文本
            chunk_text_parts = [_block_text_for_chunk(blk) for blk in group if blk.get("text")]
            meta = _chunk_metadata_from_blocks(group)
            evidence_hints = meta["evidence_types"]

            chunk_text = "\n\n".join(chunk_text_parts)
            tc = count_tokens(chunk_text)
            qs = compute_quality_score(chunk_text)

            page_start = min(meta["page_numbers"]) if meta["page_numbers"] else None
            page_end = max(meta["page_numbers"]) if meta["page_numbers"] else None

            chunk_idx += 1
            chunk_id = f"{doc_id}_sec{len(all_chunks):02d}_chunk{chunk_idx:02d}"

            section_display = sec_name
            if sec_path and len(sec_path) > 1:
                section_display = sec_path[0]

            retrieval_text = build_retrieval_text(
                title=doc_title,
                section=section_display,
                source_file=source_file,
                doc_id=doc_id,
                chunk_text=chunk_text,
                evidence_types=evidence_hints,
            )

            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_file=source_file,
                title=doc_title,
                section=section_display,
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_idx,
                token_count=tc,
                text=chunk_text,
                retrieval_text=retrieval_text,
                quality_score=qs,
                section_path=list(sec_path),
                block_types=meta["block_types"],
                source_block_ids=meta["source_block_ids"],
                block_ids=meta["block_ids"],
                evidence_types=meta["evidence_types"],
                page_numbers=meta["page_numbers"],
                layout_columns=meta["layout_columns"],
                reading_order_span=meta["reading_order_span"],
                bbox_span=meta["bbox_span"],
                source_block_metadata=meta["source_block_metadata"],
                excluded_block_counts=dict(excluded_block_counts),
                contains_figure_caption=meta["contains_figure_caption"],
                contains_table_caption=meta["contains_table_caption"],
                contains_table_text=meta["contains_table_text"],
                contains_references=meta["contains_references"],
                contains_metadata=meta["contains_metadata"],
                contains_noise=meta["contains_noise"],
                contains_image=meta["contains_image"],
                parser_stage=parser_stage,
            ))

    # 重新编号 chunk_index
    for i, chunk in enumerate(all_chunks, start=1):
        chunk.chunk_index = i

    return all_chunks


def _aggregate_blocks_into_chunks(
    blocks: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[dict]]:
    """
    将 blocks 合并为段落级 chunk groups。
    heading 附着到后续正文，不独立成 chunk；
    相邻 paragraph blocks 按 chunk_size 累加合并。
    """
    if not blocks:
        return []

    groups: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block["text"])
        joins_previous_table_caption = (
            block.get("type") == "table_text"
            and current
            and current[-1].get("type") == "table_caption"
        )

        if current_tokens + block_tokens > chunk_size and current_tokens > 0 and not joins_previous_table_caption:
            groups.append(current)
            overlap_text = _get_overlap_text_from_blocks(current, chunk_overlap)
            current = []
            current_tokens = 0
            if overlap_text:
                current.append({"type": "overlap", "text": overlap_text, "page": block["page"]})
                current_tokens = count_tokens(overlap_text)

        current.append(block)
        current_tokens += block_tokens

    if current:
        groups.append(current)

    return groups


def _get_overlap_text_from_blocks(blocks: list[dict], overlap_tokens: int) -> str:
    """从 blocks 列表末尾提取 overlap_tokens 个 token 的文本。"""
    if not blocks or overlap_tokens <= 0:
        return ""
    all_text = " ".join(b["text"] for b in blocks if b["type"] != "overlap")
    words = all_text.split()
    if len(words) <= overlap_tokens:
        return ""
    return " ".join(words[-overlap_tokens:])


# ============================================================
# 分块
# ============================================================

def chunk_section(
    section: SectionBlock,
    doc_id: str,
    source_file: str,
    title: str,
    section_idx: int,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """
    对单个 section 进行分块。
    优先按段落边界切，段落过长时硬切分。
    """
    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    token_groups = []
    current_group: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > chunk_size and not current_group:
            sub_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
            token_groups.extend(sub_chunks)
            continue

        if current_tokens + para_tokens > chunk_size and current_group:
            token_groups.append(current_group)

            overlap_text = _get_overlap_text(current_group, chunk_overlap)
            current_group = []
            current_tokens = 0

            if overlap_text:
                current_group.append(overlap_text)
                current_tokens = count_tokens(overlap_text)

        current_group.append(para)
        current_tokens += para_tokens

    if current_group:
        token_groups.append(current_group)

    chunks = []
    for chunk_idx, group in enumerate(token_groups, start=1):
        chunk_text = "\n\n".join(group)
        tc = count_tokens(chunk_text)
        qs = compute_quality_score(chunk_text)
        retrieval_text = build_retrieval_text(
            title=title,
            section=section.name,
            source_file=source_file,
            doc_id=doc_id,
            chunk_text=chunk_text,
        )

        chunk_id = f"{doc_id}_sec{section_idx:02d}_chunk{chunk_idx:02d}"

        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_file=source_file,
            title=title,
            section=section.name,
            page_start=section.page_start,
            page_end=section.page_end,
            chunk_index=chunk_idx,
            token_count=tc,
            text=chunk_text,
            retrieval_text=retrieval_text,
            quality_score=qs,
        ))

    return chunks


def _split_long_paragraph(
    para: str, chunk_size: int, chunk_overlap: int
) -> list[list[str]]:
    """对过长的段落按句子边界硬切分。"""
    sentences = re.split(r"(?<=[.!?])\s+", para)

    groups = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if current_tokens + sent_tokens > chunk_size and current:
            groups.append(["\n".join(current)])

            overlap_text = " ".join(current[-2:]) if len(current) >= 2 else current[-1] if current else ""
            overlap_words = overlap_text.split()
            if len(overlap_words) > chunk_overlap:
                overlap_text = " ".join(overlap_words[-chunk_overlap:])

            current = []
            current_tokens = 0

            if overlap_text:
                current.append(overlap_text)
                current_tokens = count_tokens(overlap_text)

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        groups.append(["\n".join(current)])

    return groups


def _get_overlap_text(group: list[str], overlap_tokens: int) -> str:
    """从当前 group 尾部提取 overlap 文本。"""
    tail = group[-1] if group else ""
    words = tail.split()
    if len(words) > overlap_tokens:
        return " ".join(words[-overlap_tokens:])
    return tail


# ============================================================
# 质量评分
# ============================================================

def compute_quality_score(text: str) -> float:
    """
    简单规则评分，0.0 ~ 1.0。
    依据：字符数、字母占比、是否过短、是否疑似噪声。
    """
    if not text:
        return 0.0

    score = 1.0

    char_count = len(text)
    word_count = len(text.split())

    if char_count < 50:
        score -= 0.4
    elif char_count < 100:
        score -= 0.2

    if word_count < 10:
        score -= 0.3
    elif word_count < 20:
        score -= 0.15

    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / max(char_count, 1)
    if alpha_ratio < 0.5:
        score -= 0.3
    elif alpha_ratio < 0.7:
        score -= 0.1

    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(char_count, 1)
    if digit_ratio > 0.3:
        score -= 0.2

    unique_words = len(set(text.lower().split()))
    if word_count > 0 and unique_words / word_count < 0.2:
        score -= 0.2

    return max(0.0, min(1.0, score))


def build_retrieval_text(
    title: str,
    section: str,
    source_file: str,
    doc_id: str,
    chunk_text: str,
    evidence_types: Optional[list[str]] = None,
) -> str:
    """
    构造用于 embedding / 稀疏检索的文本。
    将标题与 section 作为轻量前缀，提升开放检索和带过滤检索时的可命中性，
    但不污染最终给用户展示的正文 text。
    """
    header_parts = []
    if title:
        header_parts.append(f"title: {title}")
    if section:
        header_parts.append(f"section: {section}")
    if source_file:
        header_parts.append(f"source_file: {source_file}")
    if doc_id:
        header_parts.append(f"doc_id: {doc_id}")
    if evidence_types:
        header_parts.append(f"evidence_type: {', '.join(sorted(set(evidence_types)))}")

    if not header_parts:
        return chunk_text
    return "\n".join(header_parts) + "\n\n" + chunk_text


# ============================================================
# 输入文件读取
# ============================================================

def read_txt_file(filepath: Path) -> dict:
    """
    读取 txt 格式输入。
    doc_id 从文件名推导，pages 为 None（无页码信息）。
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    doc_id = filepath.stem
    source_file = filepath.name

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "pages": None,
        "raw_text": text,
        "has_blocks": False,
        "parser_stage": "",
    }


def read_json_file(filepath: Path) -> dict:
    """
    读取 json 格式输入。
    优先保留 pages[].blocks（parsed_clean 格式），供 block-based chunking 使用。
    同时生成 full_text 用于旧逻辑 fallback。
    """
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("doc_id", filepath.stem)
    source_file = data.get("source_file", filepath.name)
    pages = data.get("pages", [])
    parser_stage = data.get("parser_stage", "")

    if parser_stage == "evidence_pack_v5" or "evidence_units" in data:
        pages = evidence_pack_to_pages(data)
        full_text = "\n".join(p.get("text", "") for p in pages)
        return {
            "doc_id": doc_id,
            "source_file": source_file,
            "pages": pages,
            "raw_text": full_text,
            "has_blocks": bool(pages),
            "parser_stage": parser_stage or "evidence_pack_v5",
            "excluded_block_counts": data.get("excluded_block_counts", {}) or {},
        }

    # 检测是否有 blocks
    has_blocks = any(
        isinstance(p, dict) and p.get("blocks") for p in pages
    )

    # 构建 full_text（fallback 用）
    if has_blocks:
        page_texts = []
        for p in pages:
            blocks = p.get("blocks", [])
            if not blocks:
                page_texts.append(p.get("text", ""))
                continue
            block_parts = []
            for block in blocks:
                btype = block.get("type", "paragraph")
                btext = block.get("text", "")
                if not btext.strip():
                    continue
                if btype == "title":
                    block_parts.append(f"# {btext}")
                elif btype == "section_heading":
                    heading = btext.lstrip("#").strip()
                    block_parts.append(f"## {heading}")
                elif btype == "subsection_heading":
                    heading = btext.lstrip("#").strip()
                    block_parts.append(f"### {heading}")
                elif btype == "figure_caption":
                    block_parts.append(f"[FIGURE CAPTION] {btext}")
                elif btype == "table_caption":
                    block_parts.append(f"[TABLE CAPTION] {btext}")
                elif btype == "table_text":
                    block_parts.append(f"[TABLE TEXT] {btext}")
                elif btype == "references":
                    block_parts.append(f"## References\n{btext}")
                elif btype in ("noise", "metadata"):
                    continue
                else:
                    block_parts.append(btext)
            page_texts.append("\n\n".join(block_parts))
        full_text = "\n".join(page_texts)
    else:
        full_text = "\n".join(p.get("text", "") for p in pages)

    normalized_parser_stage = parser_stage or ("parsed_clean_v1" if has_blocks else "")

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "pages": pages,
        "raw_text": full_text,
        "has_blocks": has_blocks,
        "parser_stage": normalized_parser_stage,
        "excluded_block_counts": data.get("excluded_block_counts", {}) or {},
    }


def read_pdf_file(filepath: Path) -> dict:
    """
    直接读取 PDF 文件，使用 pymupdf 提取文本。
    输出格式与 read_json_file 一致，保留页码信息。
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "处理 PDF 文件需要 pymupdf 库，请安装: pip install pymupdf"
        )

    doc_id = filepath.stem
    source_file = filepath.name

    doc = fitz.open(str(filepath))
    pages = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append({"page": page_num + 1, "text": text})

    doc.close()

    full_text = "\n".join(p.get("text", "") for p in pages)

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "pages": pages,
        "raw_text": full_text,
        "has_blocks": False,
        "parser_stage": "",
    }


def read_input_file(filepath: Path) -> dict:
    """根据文件扩展名自动选择读取方式。支持 txt/json/pdf。"""
    suffix = filepath.suffix.lower()
    if suffix == ".json":
        return read_json_file(filepath)
    elif suffix == ".txt":
        return read_txt_file(filepath)
    elif suffix == ".pdf":
        return read_pdf_file(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


# ============================================================
# 单文档处理主流程
# ============================================================

def process_document(
    doc_data: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 50,
    min_chunk_words: int = 10,
    quality_threshold: float = 0.3,
) -> tuple[list[Chunk], bool]:
    """
    处理单篇论文，返回 (chunks, is_low_quality)。
    优先使用 block-based chunking（当 pages[].blocks 存在时），
    否则 fallback 到旧的 full_text + section 识别逻辑。
    """
    raw_text = doc_data["raw_text"]
    doc_id = doc_data["doc_id"]
    source_file = doc_data["source_file"]
    pages = doc_data.get("pages")
    has_blocks = doc_data.get("has_blocks", False)
    parser_stage = doc_data.get("parser_stage", "") or ("parsed_clean_v1" if has_blocks and pages else "")

    # 优先路径：block-based chunking
    if has_blocks and pages:
        all_chunks = chunk_by_blocks(
            pages=pages,
            doc_id=doc_id,
            source_file=source_file,
            parser_stage=parser_stage,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_excluded_block_counts=doc_data.get("excluded_block_counts", {}) or {},
        )
        # block-based 路径下，用 raw_text 判断是否有内容
        has_content = bool(raw_text and raw_text.strip())
    else:
        # Fallback 路径：旧的 full_text 逻辑
        cleaned = clean_text(raw_text)
        cleaned = truncate_at_references(cleaned)

        title = extract_title(cleaned)

        sections = identify_sections(cleaned, pages)

        all_chunks: list[Chunk] = []
        for sec_idx, section in enumerate(sections, start=1):
            if not section.text.strip():
                continue
            sec_chunks = chunk_section(
                section=section,
                doc_id=doc_id,
                source_file=source_file,
                title=title,
                section_idx=sec_idx,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(sec_chunks)

        has_content = bool(cleaned and cleaned.strip())

    filtered = []
    for chunk in all_chunks:
        has_table_text = "table_text" in getattr(chunk, "block_types", [])
        has_evidence = any(
            bt in getattr(chunk, "block_types", [])
            for bt in ("figure_caption", "table_caption", "table_text")
        )
        min_chars = 20 if has_evidence else min_chunk_chars
        min_words = 3 if has_evidence else min_chunk_words
        quality_floor = min(quality_threshold, 0.05) if has_evidence or has_table_text else quality_threshold

        if len(chunk.text) < min_chars:
            continue
        if len(chunk.text.split()) < min_words:
            continue
        if chunk.quality_score < quality_floor:
            continue
        filtered.append(chunk)

    for i, chunk in enumerate(filtered, start=1):
        chunk.chunk_index = i

    is_low_quality = len(filtered) == 0 and has_content

    return filtered, is_low_quality


# ============================================================
# 批量处理
# ============================================================

def batch_process(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 50,
    min_chunk_words: int = 10,
    quality_threshold: float = 0.3,
) -> ProcessingStats:
    """
    批量处理输入目录下的所有 txt/json 文件，输出 JSONL。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    txt_files = sorted(input_dir.glob("*.txt"))
    pdf_files = sorted(input_dir.glob("*.pdf"))

    json_doc_ids = {filepath.stem for filepath in json_files}
    preferred_txt_files = [
        filepath for filepath in txt_files
        if filepath.stem not in json_doc_ids
    ]

    input_files = sorted(json_files + preferred_txt_files + pdf_files)

    if not input_files:
        print(f"[WARN] 输入目录中没有找到 txt/json 文件: {input_dir}")
        return ProcessingStats()

    stats = ProcessingStats(total_docs=len(input_files))

    output_path = output_dir / "chunks.jsonl"
    failed_log_path = output_dir / "failed_docs.log"

    with output_path.open("w", encoding="utf-8") as out_f, \
         failed_log_path.open("w", encoding="utf-8") as log_f:

        for filepath in input_files:
            print(f"  处理: {filepath.name} ...", end=" ")
            try:
                doc_data = read_input_file(filepath)
                chunks, is_low_quality = process_document(
                    doc_data,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_chars=min_chunk_chars,
                    min_chunk_words=min_chunk_words,
                    quality_threshold=quality_threshold,
                )

                for chunk in chunks:
                    out_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

                stats.success_docs += 1
                stats.total_chunks += len(chunks)

                if is_low_quality:
                    stats.low_quality_docs += 1
                    log_f.write(f"LOW_QUALITY\t{filepath.name}\t0 chunks produced\n")

                print(f"OK ({len(chunks)} chunks)")

            except Exception as e:
                stats.failed_docs += 1
                log_f.write(f"FAILED\t{filepath.name}\t{e}\n")
                print(f"FAILED ({e})")

    return stats


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="科研论文文本后处理与分块脚本"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="输入目录，包含 txt/json/pdf 格式的论文文件",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="输出目录，存放 chunks.jsonl 和处理日志",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=800,
        help="每个 chunk 的目标 token 数（默认 800）",
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=120,
        help="chunk 之间的重叠 token 数（默认 120）",
    )
    parser.add_argument(
        "--min_chunk_chars", type=int, default=50,
        help="最短 chunk 字符数（默认 50）",
    )
    parser.add_argument(
        "--min_chunk_words", type=int, default=10,
        help="最短 chunk 单词数（默认 10）",
    )
    parser.add_argument(
        "--quality_threshold", type=float, default=0.3,
        help="chunk 最低质量分（默认 0.3）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("科研论文文本后处理与分块")
    print("=" * 60)
    print(f"输入目录:   {input_dir}")
    print(f"输出目录:   {output_dir}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"chunk_overlap: {args.chunk_overlap}")
    print()

    start_time = time.time()
    stats = batch_process(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        min_chunk_words=args.min_chunk_words,
        quality_threshold=args.quality_threshold,
    )
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("处理统计")
    print("=" * 60)
    print(f"总文档数:       {stats.total_docs}")
    print(f"成功处理数:     {stats.success_docs}")
    print(f"失败数:         {stats.failed_docs}")
    print(f"低质量文档数:   {stats.low_quality_docs}")
    print(f"总 chunk 数:    {stats.total_chunks}")
    print(f"耗时:           {elapsed:.2f}s")
    print()
    print(f"输出文件: {output_dir / 'chunks.jsonl'}")
    print(f"失败日志: {output_dir / 'failed_docs.log'}")


if __name__ == "__main__":
    main()
