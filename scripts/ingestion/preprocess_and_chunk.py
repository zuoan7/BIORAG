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
from typing import Optional


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

def chunk_by_blocks(
    pages: list[dict],
    doc_id: str,
    source_file: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    skip_references: bool = True,
) -> list[Chunk]:
    """
    基于 pages[].blocks 的结构化分块。
    按 section_path 聚合相邻 paragraph block，遇 section/subsection heading
    开启新 section 上下文。figure/table caption 进入 chunk 并在 metadata 中标记。
    页码由 block.page 最小/最大值计算，不再通过字符偏移估算。
    """
    # 按文档顺序收集所有有效 block
    all_blocks: list[dict] = []
    doc_title = ""
    for p in pages:
        page_num = p.get("page", 1)
        for block in p.get("blocks", []):
            btype = block.get("type", "paragraph")
            btext = block.get("text", "").strip()
            if not btext:
                continue
            if btype in ("noise", "metadata"):
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
            all_blocks.append({
                "type": btype,
                "text": btext,
                "section_path": block.get("section_path", []),
                "page": page_num,
            })

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
                current_blocks = [{"type": btype, "text": btext, "page": bpage}]
                continue
            elif is_metadata:
                if skip_references:
                    continue
                current_blocks.append({"type": btype, "text": btext, "page": bpage})
                continue
            elif current_section_path:
                current_blocks.append({"type": btype, "text": btext, "page": bpage})
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
                current_blocks = [{"type": btype, "text": btext, "page": bpage}]
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
                current_blocks = [{"type": btype, "text": btext, "page": bpage}]
                continue
            elif current_section_path:
                current_blocks.append({"type": btype, "text": btext, "page": bpage})
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
                current_blocks = [{"type": btype, "text": btext, "page": bpage}]
                continue

        if btype == "references":
            if skip_references:
                continue
            # 如果保留 references，归入当前 section

        # paragraph, figure_caption, table_caption 等
        current_blocks.append({"type": btype, "text": btext, "page": bpage})

    # 最后一个 section group
    if current_blocks:
        section_groups.append({
            "section": current_section,
            "section_path": list(current_section_path),
            "blocks": current_blocks,
        })

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
            chunk_text_parts = []
            block_types_set = set()
            pages_in_chunk = []

            for blk in group:
                btype = blk["type"]
                btext = blk["text"]
                block_types_set.add(btype)
                pages_in_chunk.append(blk["page"])

                if btype == "section_heading":
                    heading = btext.lstrip("#").strip()
                    chunk_text_parts.append(f"## {heading}")
                elif btype == "subsection_heading":
                    heading = btext.lstrip("#").strip()
                    chunk_text_parts.append(f"### {heading}")
                elif btype == "figure_caption":
                    chunk_text_parts.append(f"[FIGURE CAPTION] {btext}")
                elif btype == "table_caption":
                    chunk_text_parts.append(f"[TABLE CAPTION] {btext}")
                elif btype == "table_text":
                    chunk_text_parts.append(f"[TABLE TEXT] {btext}")
                elif btype == "title":
                    chunk_text_parts.append(btext)
                else:
                    chunk_text_parts.append(btext)

            chunk_text = "\n\n".join(chunk_text_parts)
            tc = count_tokens(chunk_text)
            qs = compute_quality_score(chunk_text)

            page_start = min(pages_in_chunk) if pages_in_chunk else None
            page_end = max(pages_in_chunk) if pages_in_chunk else None

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
                block_types=sorted(block_types_set),
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

        if current_tokens + block_tokens > chunk_size and current_tokens > 0:
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

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "pages": pages,
        "raw_text": full_text,
        "has_blocks": has_blocks,
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

    # 优先路径：block-based chunking
    if has_blocks and pages:
        all_chunks = chunk_by_blocks(
            pages=pages,
            doc_id=doc_id,
            source_file=source_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
        min_chars = 20 if has_table_text else min_chunk_chars
        min_words = 3 if has_table_text else min_chunk_words
        quality_floor = min(quality_threshold, 0.05) if has_table_text else quality_threshold

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
