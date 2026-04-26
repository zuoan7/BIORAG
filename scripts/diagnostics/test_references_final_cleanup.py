#!/usr/bin/env python3
"""
References 收尾修复诊断脚本
测试 3 个 fixture 场景 + 新增辅助函数
"""
import sys
sys.path.insert(0, "scripts/ingestion")

from clean_parsed_structure import (
    classify_line_type,
    is_table_context,
    looks_like_numbered_ref_heading,
    _is_reference_like_paragraph,
    _post_process_numbered_references,
    is_numbered_reference_entry,
    should_exit_references,
    is_references_heading,
    BACK_MATTER_HEADINGS,
    process_page_text,
    ProcessingCounters,
    Block,
)


def test_fixture_1_table_references_not_section_heading():
    """Fixture 1: 表格 references 列名不成为 section_heading"""
    print("=" * 60)
    print("Fixture 1: 表格 references 列名不成为 section_heading")
    print("=" * 60)

    # 表格上下文中的 "## references" 应降级为 paragraph
    recent = ["paragraph", "table_caption", "paragraph"]
    result = classify_line_type("## references", False, recent)
    print(f"  classify_line_type('## references', False, table_ctx): {result}")
    assert result == "paragraph", f"FAIL: should be paragraph, got {result}"

    # 表格上下文中的 "## References" 也应降级
    result = classify_line_type("## References", False, recent)
    print(f"  classify_line_type('## References', False, table_ctx): {result}")
    assert result == "paragraph", f"FAIL: should be paragraph, got {result}"

    # 非表格上下文中 "## References" 应为 section_heading
    result = classify_line_type("## References", False, ["paragraph", "section_heading"])
    print(f"  classify_line_type('## References', False, normal_ctx): {result}")
    assert result == "section_heading", f"FAIL: should be section_heading, got {result}"

    # "## Results" 在表格上下文中仍为 section_heading
    result = classify_line_type("## Results", False, recent)
    print(f"  classify_line_type('## Results', False, table_ctx): {result}")
    assert result == "section_heading", f"FAIL: Results should always be section_heading"

    # is_references_heading 一致性检查
    result = is_references_heading("## references", recent)
    print(f"  is_references_heading('## references', table_ctx): {result}")
    assert not result, "FAIL: table context references should not trigger"

    print("  ✅ Fixture 1 PASSED\n")


def test_fixture_2_numbered_reference_list():
    """Fixture 2: 无标题编号参考文献列表"""
    print("=" * 60)
    print("Fixture 2: 无标题编号参考文献列表")
    print("=" * 60)

    # 编号参考文献条目检测
    text1 = "1. Hasslacher, M. et al. High-level intracellular expression of hydroxynitrile lyase. Protein Expr. Purif. 11, 61–71 (1997)."
    result = _is_reference_like_paragraph(text1, 12, 13)
    print(f"  _is_reference_like_paragraph('1. Hasslacher...', page=12/13): {result}")
    assert result, "FAIL: should be reference-like"

    text2 = "2. Werten, M. W. et al. High-yield secretion of recombinant gelatins by Pichia pastoris. Yeast 15, 1087–1096 (1999)."
    result = _is_reference_like_paragraph(text2, 12, 13)
    print(f"  _is_reference_like_paragraph('2. Werten...', page=12/13): {result}")
    assert result, "FAIL: should be reference-like"

    # Acknowledgments 退出
    exit_type = should_exit_references("Acknowledgments")
    print(f"  should_exit_references('Acknowledgments'): {exit_type}")
    assert exit_type == "back_matter", "FAIL: Acknowledgments should exit (back_matter)"

    # 编号参考文献伪装成 heading
    heading = "## 21. Non-Conventional Yeasts In Genetics, Biochemistry And Biotechnology. (Springer"
    result = looks_like_numbered_ref_heading(heading)
    print(f"  looks_like_numbered_ref_heading('{heading[:50]}...'): {result}")
    assert result, "FAIL: should detect numbered ref heading"

    # 在 in_references 中，Acknowledgments 段落应退出
    result = classify_line_type("Acknowledgments The authors are grateful.", True, [])
    print(f"  classify_line_type('Acknowledgments ...', in_references=True): {result}")
    assert result == "paragraph", f"FAIL: should be paragraph (exit refs), got {result}"

    print("  ✅ Fixture 2 PASSED\n")


def test_fixture_3_numbered_steps_not_references():
    """Fixture 3: 正文编号步骤不误判 references"""
    print("=" * 60)
    print("Fixture 3: 正文编号步骤不误判 references")
    print("=" * 60)

    # 方法步骤不应被判为文献
    text1 = "1. Add methanol to the culture."
    result = _is_reference_like_paragraph(text1, 5, 13)
    print(f"  _is_reference_like_paragraph('1. Add methanol...', page=5/13): {result}")
    assert not result, "FAIL: method step should NOT be reference-like"

    text2 = "2. Incubate cells for 12 h."
    result = _is_reference_like_paragraph(text2, 5, 13)
    print(f"  _is_reference_like_paragraph('2. Incubate cells...', page=5/13): {result}")
    assert not result, "FAIL: method step should NOT be reference-like"

    text3 = "3. Measure OD600 and HRP activity."
    result = _is_reference_like_paragraph(text3, 5, 13)
    print(f"  _is_reference_like_paragraph('3. Measure OD600...', page=5/13): {result}")
    assert not result, "FAIL: method step should NOT be reference-like"

    # 非文档后部也不应触发
    text4 = "1. Hasslacher, M. et al. High-level expression. Protein Expr. Purif. 11, 61–71 (1997)."
    result = _is_reference_like_paragraph(text4, 3, 13)
    print(f"  _is_reference_like_paragraph(ref entry, page=3/13): {result}")
    assert not result, "FAIL: early page should NOT trigger"

    # is_numbered_reference_entry 对方法步骤应返回 False
    result = is_numbered_reference_entry("1. Add methanol to the culture.")
    print(f"  is_numbered_reference_entry('1. Add methanol...'): {result}")
    assert not result, "FAIL: method step should NOT match ref pattern"

    print("  ✅ Fixture 3 PASSED\n")


def test_full_process_page_text():
    """集成测试：完整页面文本处理"""
    print("=" * 60)
    print("集成测试：完整页面文本处理")
    print("=" * 60)

    # 场景 1：表格 references 不成为 section_heading
    page1 = """## Results
Knockout of OCH1 from Ppku70and PpMutS.
Table 1 | Humanization of N-glycans in P. pastoris.
## references
introduction of a-1,2-Mns, GnTI and an UDP-GlcNAc transporter.
## Discussion
Discussion paragraph here."""

    counters = ProcessingCounters()
    blocks, _, section_path, in_refs = process_page_text(
        page1, page_num=2, block_index_start=0, counters=counters,
        total_pages=13, recent_block_types=["paragraph", "table_caption"],
    )

    # 找 "references" 相关的 block
    refs_blocks = [b for b in blocks if "references" in b.text.lower() or b.type == "references"]
    heading_blocks = [b for b in blocks if b.type == "section_heading"]

    print(f"  场景1: heading blocks: {[(b.text[:50], b.section_path) for b in heading_blocks]}")
    print(f"  场景1: references-related blocks: {[(b.text[:50], b.type) for b in refs_blocks]}")

    # "## references" 不应是 section_heading
    refs_heading = [b for b in heading_blocks if "references" in b.text.lower()]
    assert len(refs_heading) == 0, f"FAIL: 'references' should not be section_heading, found {refs_heading}"

    # section_path 不应包含 "references"
    refs_path = [b for b in blocks if "references" in b.section_path]
    assert len(refs_path) == 0, f"FAIL: section_path should not contain 'references', found {refs_path}"

    # Results 和 Discussion 应正常
    results_headings = [b for b in heading_blocks if "Results" in b.text]
    assert len(results_headings) >= 1, "FAIL: Results should be section_heading"
    discussion_headings = [b for b in heading_blocks if "Discussion" in b.text]
    assert len(discussion_headings) >= 1, "FAIL: Discussion should be section_heading"

    print("  场景1: ✅ 表格 references 不成为 section_heading")

    # 场景 2：编号参考文献 + 尾部退出
    # 注意：参考文献条目之间可能没有空行（PDF 解析常见情况）
    page2 = """## Methods
Method paragraph.
1. Hasslacher, M. et al. High-level intracellular expression of hydroxynitrile lyase. Protein Expr. Purif. 11, 61-71 (1997).
2. Werten, M. W. et al. High-yield secretion of recombinant gelatins by Pichia pastoris. Yeast 15, 1087-1096 (1999).
3. Jahic, M. et al. Modeling of growth and energy metabolism. Bioprocess Biosyst. Eng. 24, 385-393 (2002).

Acknowledgments The authors are grateful.

## Author contributions
F.W.K. and O.S. conceived of and planned the study."""

    counters2 = ProcessingCounters()
    blocks2, _, sp2, in_refs2 = process_page_text(
        page2, page_num=12, block_index_start=0, counters=counters2,
        total_pages=13, recent_block_types=["paragraph"],
    )

    # 后处理：检测编号参考文献（process_document 中会调用，这里手动调用）
    _post_process_numbered_references(blocks2, 13, counters2)

    ref_blocks = [b for b in blocks2 if b.type == "references"]
    heading_blocks2 = [b for b in blocks2 if b.type == "section_heading"]
    para_blocks = [b for b in blocks2 if b.type == "paragraph"]

    print(f"  场景2: references blocks: {len(ref_blocks)}")
    print(f"  场景2: heading blocks: {[(b.text[:50], b.section_path) for b in heading_blocks2]}")

    # 应有 references blocks（至少 1 个，可能是合并的多个条目）
    assert len(ref_blocks) >= 1, f"FAIL: should have references blocks, got {len(ref_blocks)}"
    # 确认 references block 的文本包含编号引用
    total_ref_text = " ".join(b.text for b in ref_blocks)
    assert "1." in total_ref_text or "2." in total_ref_text, "FAIL: references should contain numbered entries"

    # Author contributions 不应是 references
    ac_blocks = [b for b in blocks2 if "Author contributions" in b.text]
    for b in ac_blocks:
        assert b.type != "references", f"FAIL: Author contributions should not be references"

    # Acknowledgments 不应是 references
    ack_blocks = [b for b in blocks2 if b.text.strip().lower().startswith("acknowledgments")]
    for b in ack_blocks:
        assert b.type != "references", f"FAIL: Acknowledgments should not be references, got {b.type}"

    print("  场景2: ✅ 编号参考文献识别 + 尾部退出")

    # 场景 3：方法步骤不误判
    page3 = """## Methods
The protocol was performed as follows.
1. Add methanol to the culture.
2. Incubate cells for 12 h.
3. Measure OD600 and HRP activity."""

    counters3 = ProcessingCounters()
    blocks3, _, sp3, in_refs3 = process_page_text(
        page3, page_num=5, block_index_start=0, counters=counters3,
        total_pages=13, recent_block_types=["paragraph"],
    )

    ref_blocks3 = [b for b in blocks3 if b.type == "references"]
    assert len(ref_blocks3) == 0, f"FAIL: method steps should not be references, got {len(ref_blocks3)}"

    print("  场景3: ✅ 方法步骤不误判 references")

    print("  ✅ 集成测试 PASSED\n")


def test_helper_functions():
    """测试新增辅助函数"""
    print("=" * 60)
    print("辅助函数测试")
    print("=" * 60)

    # is_table_context
    assert is_table_context(["paragraph", "table_caption"]) == True
    assert is_table_context(["paragraph", "section_heading"]) == False
    assert is_table_context([]) == False
    assert is_table_context(None) == False
    print("  is_table_context: ✅")

    # looks_like_numbered_ref_heading
    assert looks_like_numbered_ref_heading("## 21. Non-Conventional Yeasts In Genetics. (Springer, 2003). doi:10.1007")
    assert not looks_like_numbered_ref_heading("## 2.1. Materials and Methods")
    assert not looks_like_numbered_ref_heading("## Results")
    print("  looks_like_numbered_ref_heading: ✅")

    # _is_reference_like_paragraph
    assert _is_reference_like_paragraph(
        "Purif. 11, 61–71 (1997). 2. Werten, M. W. et al. High-yield secretion.",
        12, 13
    )
    assert not _is_reference_like_paragraph(
        "1. Add methanol to the culture.", 5, 13
    )
    assert not _is_reference_like_paragraph(
        "1. Hasslacher, M. et al. Protein Expr. Purif. 11, 61–71 (1997).",
        3, 13  # early page
    )
    print("  _is_reference_like_paragraph: ✅")

    print("  ✅ 辅助函数测试 PASSED\n")


if __name__ == "__main__":
    test_helper_functions()
    test_fixture_1_table_references_not_section_heading()
    test_fixture_2_numbered_reference_list()
    test_fixture_3_numbered_steps_not_references()
    test_full_process_page_text()
    print("=" * 60)
    print("ALL FIXTURES PASSED ✅")
    print("=" * 60)
