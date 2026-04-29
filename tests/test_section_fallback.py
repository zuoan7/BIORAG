"""Test section fallback logic in preprocess_and_chunk.py."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingestion.preprocess_and_chunk import (
    _needs_section_fallback,
    _detect_true_section_headings,
    _is_false_heading,
    _apply_fallback_section_grouping,
    _apply_generic_fulltext_fallback,
    _BODY_SECTIONS,
)


class TestNeedsSectionFallback:
    def test_normal_doc_no_fallback(self):
        """结构正常的文档不应触发 fallback。"""
        groups = [
            {"section": "Title", "section_path": ["Title"], "blocks": [{"text": "x"}] * 5},
            {"section": "Introduction", "section_path": ["Introduction"], "blocks": [{"text": "x"}] * 10},
            {"section": "Results", "section_path": ["Results"], "blocks": [{"text": "x"}] * 20},
            {"section": "Discussion", "section_path": ["Discussion"], "blocks": [{"text": "x"}] * 15},
        ]
        assert _needs_section_fallback(groups) is False

    def test_all_title_triggers_fallback(self):
        """全部 Title 的文档应触发 fallback。"""
        groups = [
            {"section": "Title", "section_path": ["Title"], "blocks": [{"text": "x"}] * 50},
        ]
        assert _needs_section_fallback(groups) is True

    def test_title_abstract_dominant_triggers_fallback(self):
        """Title + Abstract 占比 > 80% 且 body < 10% 应触发。"""
        groups = [
            {"section": "Title", "section_path": ["Title"], "blocks": [{"text": "x"}] * 40},
            {"section": "Abstract", "section_path": ["Abstract"], "blocks": [{"text": "x"}] * 10},
        ]
        assert _needs_section_fallback(groups) is True

    def test_small_document_no_fallback(self):
        """过小的文档（< 3 blocks）不触发 fallback。"""
        groups = [{"section": "Title", "section_path": ["Title"], "blocks": [{"text": "x"}] * 2}]
        assert _needs_section_fallback(groups) is False

    def test_body_present_no_fallback(self):
        """即使有 Title/Abstract，只要 body > 10% 就不触发。"""
        groups = [
            {"section": "Title", "section_path": ["Title"], "blocks": [{"text": "x"}] * 80},
            {"section": "Results", "section_path": ["Results"], "blocks": [{"text": "x"}] * 20},
        ]
        assert _needs_section_fallback(groups) is False


class TestDetectSectionHeadings:
    def test_standard_headings(self):
        """能识别标准 section heading。"""
        blocks = [
            {"text": "## Introduction", "type": "paragraph", "page": 1},
            {"text": "Some body text here.", "type": "paragraph", "page": 1},
            {"text": "## Materials and Methods", "type": "paragraph", "page": 2},
            {"text": "Experimental details.", "type": "paragraph", "page": 2},
            {"text": "## Results", "type": "paragraph", "page": 3},
            {"text": "Data analysis.", "type": "paragraph", "page": 3},
            {"text": "## Discussion", "type": "paragraph", "page": 5},
            {"text": "Interpretation.", "type": "paragraph", "page": 5},
        ]
        detected = _detect_true_section_headings(blocks)
        names = [name for _, name in detected]
        assert "Introduction" in names
        assert "Materials and Methods" in names
        assert "Results" in names
        assert "Discussion" in names

    def test_numbered_headings(self):
        """能识别带编号的 section heading。"""
        blocks = [
            {"text": "1. Introduction", "type": "paragraph", "page": 1},
            {"text": "2. Results and Discussion", "type": "paragraph", "page": 3},
            {"text": "3. Conclusion", "type": "paragraph", "page": 8},
        ]
        detected = _detect_true_section_headings(blocks)
        names = [name for _, name in detected]
        assert "Introduction" in names
        assert "Results and Discussion" in names
        assert "Conclusion" in names

    def test_case_insensitive(self):
        """大小写不敏感（全大写 heading 可识别）。"""
        blocks = [
            {"text": "INTRODUCTION", "type": "paragraph", "page": 1},
            {"text": "RESULTS", "type": "paragraph", "page": 3},
            {"text": "DISCUSSION", "type": "paragraph", "page": 5},
        ]
        detected = _detect_true_section_headings(blocks)
        names = [name for _, name in detected]
        assert "Introduction" in names
        assert "Results" in names
        assert "Discussion" in names

    def test_does_not_match_body_text(self):
        """不应该把正文中的关键词误判为 heading。"""
        blocks = [
            {"text": "the results showed that the protein was active and the discussion highlighted the importance", "type": "paragraph", "page": 5},
            {"text": "in this discussion, we will explore the results of previous studies", "type": "paragraph", "page": 6},
        ]
        detected = _detect_true_section_headings(blocks)
        assert len(detected) == 0

    def test_long_text_not_heading(self):
        """长文本不应被识别为 heading。"""
        blocks = [{"text": "Results " + "x " * 200, "type": "paragraph", "page": 3}]
        detected = _detect_true_section_headings(blocks)
        assert len(detected) == 0


class TestFalseHeadingFilter:
    def test_funding_filtered(self):
        assert _is_false_heading("Funding") is True
        assert _is_false_heading("## Funding") is True

    def test_author_contributions_filtered(self):
        assert _is_false_heading("Author Contributions") is True

    def test_references_filtered(self):
        assert _is_false_heading("## References") is True
        assert _is_false_heading("REFERENCES") is True

    def test_conflict_of_interest_filtered(self):
        assert _is_false_heading("Conflict of Interest") is True

    def test_address_filtered(self):
        assert _is_false_heading("79104 Freiburg, Germany") is True

    def test_reference_entry_filtered(self):
        assert _is_false_heading("7. Rabinowitz, M., and Lipmann, F. (1960) Reversible phosphate") is True

    def test_figure_panel_ref_filtered(self):
        assert _is_false_heading("5A,B). The pykA-knockout strain produced 9.9 g/L") is True

    def test_instrument_params_filtered(self):
        assert _is_false_heading("1950 V; and mass range, 20-2000 m/z.") is True

    def test_real_section_not_filtered(self):
        assert _is_false_heading("Introduction") is False
        assert _is_false_heading("Results") is False
        assert _is_false_heading("Discussion") is False


class TestFallbackSectionGrouping:
    def test_basic_grouping(self):
        """基本 fallback grouping 测试。"""
        blocks = [
            {"text": "Paper Title", "type": "title", "page": 1},
            {"text": "Author list", "type": "paragraph", "page": 1},
            {"text": "## Introduction", "type": "paragraph", "page": 2},
            {"text": "Intro text paragraph 1.", "type": "paragraph", "page": 2},
            {"text": "Intro text paragraph 2.", "type": "paragraph", "page": 2},
            {"text": "## Results", "type": "paragraph", "page": 4},
            {"text": "Results text.", "type": "paragraph", "page": 4},
            {"text": "## Discussion", "type": "paragraph", "page": 6},
            {"text": "Discussion text.", "type": "paragraph", "page": 6},
        ]
        headings = _detect_true_section_headings(blocks)
        assert len(headings) >= 2

        groups = _apply_fallback_section_grouping(blocks, headings)
        sections = {g["section"] for g in groups}
        assert "Introduction" in sections
        assert "Results" in sections
        assert "Discussion" in sections


class TestGenericFulltextFallback:
    def test_title_blocks_split(self):
        """Title section 的 blocks 应被正确分割：前2个保留 Title，其余变 Full Text。"""
        groups = [
            {"section": "Title", "section_path": ["Title"],
             "blocks": [{"text": f"Block number {i} with enough length to pass filter"} for i in range(10)]},
        ]
        result = _apply_generic_fulltext_fallback(groups, groups[0]["blocks"])
        sections = {g["section"] for g in result}
        assert "Title" in sections
        assert "Full Text" in sections
        # Verify Title has 2 blocks, Full Text has 8
        for g in result:
            if g["section"] == "Title":
                assert len(g["blocks"]) == 2
            elif g["section"] == "Full Text":
                assert len(g["blocks"]) == 8


class TestBodySections:
    """确保 BODY_SECTIONS 包含了所有预期的 body section 名。"""
    def test_full_text_in_body(self):
        assert "Full Text" in _BODY_SECTIONS

    def test_common_body_sections(self):
        for s in ["Introduction", "Results", "Discussion", "Materials and Methods", "Methods", "Conclusion"]:
            assert s in _BODY_SECTIONS
