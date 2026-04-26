#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 修复的单元测试：
  P0-1: BGE embedding max_length 可配置
  P0-2: section_path / in_references 跨页状态传递
  P0-3: block-based chunking 优先路径
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ingestion.clean_parsed_structure import (
    Block,
    ProcessingCounters,
    process_page_text,
    process_document,
    batch_process,
    REFERENCES_HEADING_PATTERN,
)
from scripts.ingestion.preprocess_and_chunk import (
    Chunk,
    chunk_by_blocks,
    process_document as preprocess_process_document,
    read_json_file,
)


# ============================================================
# P0-2 测试：section_path / in_references 跨页状态传递
# ============================================================

class TestCrossPageState:
    """验证 section_path 和 in_references 在跨页时正确传递。"""

    def _make_raw_data(self, pages_text: list[str]) -> dict:
        """构造 parsed_raw 格式的 dict。"""
        return {
            "doc_id": "test_cross_page",
            "source_file": "test.pdf",
            "total_pages": len(pages_text),
            "pages": [
                {"page": i + 1, "text": text}
                for i, text in enumerate(pages_text)
            ],
        }

    def test_section_path_carried_across_pages(self):
        """第 1 页有 Results heading，第 2 页没有 heading → 第 2 页 block 继承 Results section_path。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "parsed_raw"
            output_dir = Path(tmpdir) / "parsed_clean"
            preview_dir = Path(tmpdir) / "parsed_preview"
            input_dir.mkdir()

            raw_data = self._make_raw_data([
                "## Results\nThe initial results show significant changes.",
                "Further analysis confirmed the findings in the results section.",
            ])
            json_path = input_dir / "test.json"
            json_path.write_text(json.dumps(raw_data, ensure_ascii=False))

            batch_process(input_dir, output_dir, preview_dir)

            clean_path = output_dir / "test_cross_page.json"
            clean_data = json.loads(clean_path.read_text())

            page1_blocks = clean_data["pages"][0]["blocks"]
            page2_blocks = clean_data["pages"][1]["blocks"]

            # 第 1 页 Results heading 的 section_path 应为 ["Results"]
            results_headings = [b for b in page1_blocks if b["type"] == "section_heading"]
            assert len(results_headings) >= 1
            assert "Results" in results_headings[0]["section_path"]

            # 第 2 页的 paragraph block 应继承 Results section_path
            p2_paragraphs = [b for b in page2_blocks if b["type"] == "paragraph"]
            assert len(p2_paragraphs) >= 1
            assert "Results" in p2_paragraphs[0]["section_path"]

    def test_references_carried_across_pages(self):
        """References heading 在第 3 页，第 4 页仍应为 references 类型。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "parsed_raw"
            output_dir = Path(tmpdir) / "parsed_clean"
            preview_dir = Path(tmpdir) / "parsed_preview"
            input_dir.mkdir()

            raw_data = self._make_raw_data([
                "## Results\nSome results here.",
                "More results discussion.",
                "## References\n1. Smith J, et al. Paper one. Nature. 2023.",
                "2. Wang L, et al. Paper two. Science. 2022.",
            ])
            json_path = input_dir / "test.json"
            json_path.write_text(json.dumps(raw_data, ensure_ascii=False))

            batch_process(input_dir, output_dir, preview_dir)

            clean_path = output_dir / "test_cross_page.json"
            clean_data = json.loads(clean_path.read_text())

            page4_blocks = clean_data["pages"][3]["blocks"]
            # 第 4 页的 block 应该是 references 类型（不是 paragraph）
            for b in page4_blocks:
                assert b["type"] == "references", (
                    f"Page 4 block should be references, got {b['type']}: {b['text'][:50]}"
                )

    def test_process_page_text_accepts_and_returns_state(self):
        """process_page_text 接收并返回 section_path / in_references 状态。"""
        counters = ProcessingCounters()

        # 第 1 页：有 Results heading
        blocks1, _, sp1, ref1 = process_page_text(
            "## Results\nSome results text.",
            page_num=1,
            block_index_start=0,
            counters=counters,
            section_path=[],
            in_references=False,
        )
        assert "Results" in sp1
        assert ref1 is False

        # 第 2 页：无 heading，传入上一页状态
        blocks2, _, sp2, ref2 = process_page_text(
            "Continuation of results analysis.",
            page_num=2,
            block_index_start=len(blocks1),
            counters=counters,
            section_path=sp1,
            in_references=ref1,
        )
        assert "Results" in sp2
        assert ref2 is False
        # 第 2 页的 paragraph 应继承 section_path
        p2_paragraphs = [b for b in blocks2 if b.type == "paragraph"]
        assert len(p2_paragraphs) >= 1
        assert "Results" in p2_paragraphs[0].section_path

    def test_process_page_text_references_state(self):
        """process_page_text 的 in_references 状态跨页传递。"""
        counters = ProcessingCounters()

        # 第 1 页：References heading
        blocks1, _, sp1, ref1 = process_page_text(
            "## References\n1. First reference here.",
            page_num=1,
            block_index_start=0,
            counters=counters,
            section_path=[],
            in_references=False,
        )
        assert ref1 is True

        # 第 2 页：无 heading，继续 references
        blocks2, _, sp2, ref2 = process_page_text(
            "2. Second reference continues on next page.",
            page_num=2,
            block_index_start=len(blocks1),
            counters=counters,
            section_path=sp1,
            in_references=ref1,
        )
        assert ref2 is True
        # 第 2 页 block 应为 references 类型
        for b in blocks2:
            if b.text.strip():
                assert b.type == "references", f"Expected references, got {b.type}"


# ============================================================
# P0-3 测试：block-based chunking
# ============================================================

class TestBlockBasedChunking:
    """验证 block-based chunking 优先路径。"""

    def _make_parsed_clean_data(self) -> dict:
        """构造包含 blocks 的 parsed_clean 格式数据。"""
        return {
            "doc_id": "test_block_chunk",
            "source_file": "test.pdf",
            "total_pages": 3,
            "pages": [
                {
                    "page": 1,
                    "text": "# Test Title\n\n## Introduction\nIntro text here.",
                    "blocks": [
                        {"block_id": "p1_b0000", "type": "title", "text": "Test Title", "section_path": ["Title"], "page": 1},
                        {"block_id": "p1_b0001", "type": "section_heading", "text": "## Introduction", "section_path": ["Introduction"], "page": 1},
                        {"block_id": "p1_b0002", "type": "paragraph", "text": "Intro text here.", "section_path": ["Introduction"], "page": 1},
                    ],
                },
                {
                    "page": 2,
                    "text": "More intro continued.\n\n## Methods\nWe used standard protocols.",
                    "blocks": [
                        {"block_id": "p2_b0000", "type": "paragraph", "text": "More intro continued.", "section_path": ["Introduction"], "page": 2},
                        {"block_id": "p2_b0001", "type": "section_heading", "text": "## Methods", "section_path": ["Methods"], "page": 2},
                        {"block_id": "p2_b0002", "type": "paragraph", "text": "We used standard protocols.", "section_path": ["Methods"], "page": 2},
                    ],
                },
                {
                    "page": 3,
                    "text": "Figure 1. Experimental setup.\n\nTable 1. Sample characteristics.",
                    "blocks": [
                        {"block_id": "p3_b0000", "type": "figure_caption", "text": "Figure 1. Experimental setup.", "section_path": ["Methods"], "page": 3},
                        {"block_id": "p3_b0001", "type": "table_caption", "text": "Table 1. Sample characteristics.", "section_path": ["Methods"], "page": 3},
                    ],
                },
            ],
        }

    def test_chunk_by_blocks_produces_section_path(self):
        """chunk_by_blocks 输出包含 section_path。"""
        data = self._make_parsed_clean_data()
        chunks = chunk_by_blocks(data["pages"], doc_id="test", source_file="test.pdf")

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk.section_path, list)
            # section_path 不应为空（至少有 section heading）
            if chunk.section:
                assert len(chunk.section_path) > 0, (
                    f"Chunk with section='{chunk.section}' should have non-empty section_path"
                )

    def test_chunk_by_blocks_produces_block_types(self):
        """chunk_by_blocks 输出包含 block_types。"""
        data = self._make_parsed_clean_data()
        chunks = chunk_by_blocks(data["pages"], doc_id="test", source_file="test.pdf")

        # 应该有 chunk 包含 figure_caption 或 table_caption
        all_block_types = set()
        for chunk in chunks:
            all_block_types.update(chunk.block_types)
            assert isinstance(chunk.block_types, list)

        assert "figure_caption" in all_block_types, "Should have figure_caption in block_types"
        assert "table_caption" in all_block_types, "Should have table_caption in block_types"

    def test_chunk_page_range_from_blocks(self):
        """chunk 的 page_start/page_end 由 block.page 计算。"""
        data = self._make_parsed_clean_data()
        chunks = chunk_by_blocks(data["pages"], doc_id="test", source_file="test.pdf")

        for chunk in chunks:
            assert chunk.page_start is not None
            assert chunk.page_end is not None
            assert chunk.page_start <= chunk.page_end

    def test_chunk_by_blocks_skips_references(self):
        """默认跳过 references blocks。"""
        data = {
            "doc_id": "test",
            "source_file": "test.pdf",
            "total_pages": 2,
            "pages": [
                {
                    "page": 1,
                    "text": "## Results\nSome results.",
                    "blocks": [
                        {"block_id": "p1_b0000", "type": "section_heading", "text": "## Results", "section_path": ["Results"], "page": 1},
                        {"block_id": "p1_b0001", "type": "paragraph", "text": "Some results.", "section_path": ["Results"], "page": 1},
                    ],
                },
                {
                    "page": 2,
                    "text": "## References\n1. Smith 2023.",
                    "blocks": [
                        {"block_id": "p2_b0000", "type": "section_heading", "text": "## References", "section_path": ["References"], "page": 2},
                        {"block_id": "p2_b0001", "type": "references", "text": "1. Smith 2023.", "section_path": ["References"], "page": 2},
                    ],
                },
            ],
        }
        chunks = chunk_by_blocks(data["pages"], doc_id="test", source_file="test.pdf")
        # References 不应出现在 chunks 中
        for chunk in chunks:
            assert chunk.section != "References"
            assert "references" not in chunk.block_types

    def test_fallback_without_blocks(self):
        """没有 blocks 时 fallback 到旧的 full_text 逻辑。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建没有 blocks 的 JSON（旧格式）
            data = {
                "doc_id": "test_no_blocks",
                "source_file": "test.pdf",
                "total_pages": 1,
                "pages": [
                    {
                        "page": 1,
                        "text": (
                            "# Effects of Probiotics on Gut Microbiome\n\n"
                            "## Introduction\n"
                            "This study investigates the effects of probiotic supplementation "
                            "on the human gut microbiome. We analyzed fecal samples from 50 "
                            "participants over a 12-week period. The results show significant "
                            "changes in microbial composition, particularly an increase in "
                            "Bifidobacterium populations. These findings suggest that regular "
                            "probiotic consumption can positively modulate gut microbiota diversity."
                        ),
                    },
                ],
            }
            json_path = Path(tmpdir) / "test.json"
            json_path.write_text(json.dumps(data, ensure_ascii=False))

            doc_data = read_json_file(json_path)
            assert doc_data["has_blocks"] is False

            chunks, is_low = preprocess_process_document(doc_data)
            assert len(chunks) > 0
            # fallback 路径下 section_path 和 block_types 为默认空列表
            for chunk in chunks:
                assert isinstance(chunk.section_path, list)
                assert isinstance(chunk.block_types, list)

    def test_caption_markers_in_chunk_text(self):
        """figure/table caption 在 chunk text 中有标记。"""
        data = self._make_parsed_clean_data()
        chunks = chunk_by_blocks(data["pages"], doc_id="test", source_file="test.pdf")

        all_text = " ".join(c.text for c in chunks)
        assert "[FIGURE CAPTION]" in all_text, "Should contain [FIGURE CAPTION] marker"
        assert "[TABLE CAPTION]" in all_text, "Should contain [TABLE CAPTION] marker"


# ============================================================
# P0-1 测试：BGE embedding max_length 可配置
# ============================================================

class TestEmbedMaxLength:
    """验证 BGE embedding max_length 不再硬编码为 512。"""

    def test_import_to_milvus_no_hardcoded_512(self):
        """import_to_milvus.py 中 embedding encode 不再有 max_length=512。"""
        script_path = ROOT / "scripts" / "ingestion" / "import_to_milvus.py"
        content = script_path.read_text()
        # 不应在 encode 调用中出现 max_length=512
        # 注意：Milvus schema 中 title VARCHAR max_length=512 仍可存在
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # 只检查 encode 调用中的 max_length
            if "model.encode" in line and "max_length=512" in line:
                pytest.fail(
                    f"Line {i+1}: Found hardcoded max_length=512 in model.encode call: {line.strip()}"
                )

    def test_embed_max_length_default_8192(self):
        """BGEEmbedder 默认 max_length 应为 8192。"""
        from scripts.ingestion.import_to_milvus import BGEEmbedder
        # 检查构造函数默认值
        import inspect
        sig = inspect.signature(BGEEmbedder.__init__)
        max_length_param = sig.parameters.get("max_length")
        assert max_length_param is not None
        assert max_length_param.default == 8192

    def test_env_var_override(self):
        """BGE_EMBED_MAX_LENGTH 环境变量应被读取。"""
        script_path = ROOT / "scripts" / "ingestion" / "import_to_milvus.py"
        content = script_path.read_text()
        assert "BGE_EMBED_MAX_LENGTH" in content

    def test_cli_arg_exists(self):
        """--embed-max-length CLI 参数应存在。"""
        script_path = ROOT / "scripts" / "ingestion" / "import_to_milvus.py"
        content = script_path.read_text()
        assert "--embed-max-length" in content
