#!/usr/bin/env python3
"""
References 状态机修复诊断脚本
测试 4 个 fixture 场景
"""
import re
import sys
sys.path.insert(0, "scripts/ingestion")

from clean_parsed_structure import (
    is_references_heading,
    should_exit_references,
    is_false_heading_metadata,
    is_numbered_reference_entry,
    TABLE_CONTEXT_REFERENCE_PATTERN,
    REFERENCES_HEADING_PATTERN,
    BODY_SECTION_PATTERN,
    BACK_MATTER_PATTERN,
)


def test_fixture_1_table_reference_not_trigger():
    """Fixture 1: 表格 references 列名不触发 in_references"""
    print("=" * 60)
    print("Fixture 1: 表格 references 列名不触发 in_references")
    print("=" * 60)

    # "content references" 应不触发
    recent = ["paragraph", "table_caption", "paragraph"]
    result = is_references_heading("## content references", recent)
    print(f"  is_references_heading('## content references', table_ctx): {result}")
    assert not result, "FAIL: 'content references' 应不触发"

    # 纯 "References" 在表格上下文中应不触发
    result = is_references_heading("## references", recent)
    print(f"  is_references_heading('## references', table_ctx): {result}")
    assert not result, "FAIL: '## references' 在 table context 应不触发"

    # 纯 "References" 无表格上下文应触发
    result = is_references_heading("## References", ["paragraph", "section_heading"])
    print(f"  is_references_heading('## References', normal_ctx): {result}")
    assert result, "FAIL: '## References' 无 table context 应触发"

    # "Table 1 ... references" 不应触发
    result = is_references_heading("## Table 1 references", [])
    print(f"  is_references_heading('## Table 1 references', []): {result}")
    assert not result, "FAIL: 含 Table 关键词不应触发"

    print("  ✅ Fixture 1 PASSED\n")


def test_fixture_2_methods_after_discussion():
    """Fixture 2: Methods after Discussion 应退出 references"""
    print("=" * 60)
    print("Fixture 2: Methods after Discussion")
    print("=" * 60)

    # Methods 应退出 references
    exit_type = should_exit_references("Methods")
    print(f"  should_exit_references('Methods'): {exit_type}")
    assert exit_type == "body", "FAIL: Methods 应退出 references (body)"

    exit_type = should_exit_references("Materials and methods")
    print(f"  should_exit_references('Materials and methods'): {exit_type}")
    assert exit_type == "body", "FAIL: Materials and methods 应退出 (body)"

    exit_type = should_exit_references("Discussion")
    print(f"  should_exit_references('Discussion'): {exit_type}")
    assert exit_type == "body", "FAIL: Discussion 应退出 (body)"

    exit_type = should_exit_references("Results")
    print(f"  should_exit_references('Results'): {exit_type}")
    assert exit_type == "body", "FAIL: Results 应退出 (body)"

    # 普通文本不应退出
    exit_type = should_exit_references("Some random text")
    print(f"  should_exit_references('Some random text'): {exit_type}")
    assert exit_type is None, "FAIL: 普通文本不应退出"

    print("  ✅ Fixture 2 PASSED\n")


def test_fixture_3_numbered_references_and_back_matter():
    """Fixture 3: 无标题编号参考文献 + Acknowledgments 退出"""
    print("=" * 60)
    print("Fixture 3: 无标题编号参考文献 + 尾部 section 退出")
    print("=" * 60)

    # 编号参考文献条目检测
    result = is_numbered_reference_entry(
        "1. Hasslacher, M. et al. High-level intracellular expression ... Protein Expr. Purif. 11, 61–71 (1997)."
    )
    print(f"  is_numbered_reference_entry('1. Hasslacher...'): {result}")
    assert result, "FAIL: 编号参考文献条目应被识别"

    result = is_numbered_reference_entry(
        "2. Werten, M. W. et al. High-yield secretion ... Yeast 15, 1087–1096 (1999)."
    )
    print(f"  is_numbered_reference_entry('2. Werten...'): {result}")
    assert result, "FAIL: 编号参考文献条目应被识别"

    result = is_numbered_reference_entry("Some regular paragraph text.")
    print(f"  is_numbered_reference_entry('Some regular text'): {result}")
    assert not result, "FAIL: 普通文本不应被识别为参考文献"

    # Acknowledgments 退出
    exit_type = should_exit_references("Acknowledgments")
    print(f"  should_exit_references('Acknowledgments'): {exit_type}")
    assert exit_type == "back_matter", "FAIL: Acknowledgments 应退出 (back_matter)"

    exit_type = should_exit_references("Author contributions")
    print(f"  should_exit_references('Author contributions'): {exit_type}")
    assert exit_type == "back_matter", "FAIL: Author contributions 应退出 (back_matter)"

    exit_type = should_exit_references("Additional information")
    print(f"  should_exit_references('Additional information'): {exit_type}")
    assert exit_type == "back_matter", "FAIL: Additional information 应退出 (back_matter)"

    print("  ✅ Fixture 3 PASSED\n")


def test_fixture_4_date_metadata_false_heading():
    """Fixture 4: 日期/metadata 不误判 heading"""
    print("=" * 60)
    print("Fixture 4: 日期/metadata 不误判 heading")
    print("=" * 60)

    # 日期应降级
    result = is_false_heading_metadata("## 20 May 2013")
    print(f"  is_false_heading_metadata('## 20 May 2013'): {result}")
    assert result, "FAIL: 日期应降级"

    result = is_false_heading_metadata("## 4 November 2013")
    print(f"  is_false_heading_metadata('## 4 November 2013'): {result}")
    assert result, "FAIL: 日期应降级"

    result = is_false_heading_metadata("## Published")
    print(f"  is_false_heading_metadata('## Published'): {result}")
    assert result, "FAIL: Published 应降级"

    result = is_false_heading_metadata("## Received")
    print(f"  is_false_heading_metadata('## Received'): {result}")
    assert result, "FAIL: Received 应降级"

    result = is_false_heading_metadata("## Accepted")
    print(f"  is_false_heading_metadata('## Accepted'): {result}")
    assert result, "FAIL: Accepted 应降级"

    # 正常标题不应降级
    result = is_false_heading_metadata("## Results")
    print(f"  is_false_heading_metadata('## Results'): {result}")
    assert not result, "FAIL: Results 不应降级"

    result = is_false_heading_metadata("## Methods")
    print(f"  is_false_heading_metadata('## Methods'): {result}")
    assert not result, "FAIL: Methods 不应降级"

    print("  ✅ Fixture 4 PASSED\n")


if __name__ == "__main__":
    test_fixture_1_table_reference_not_trigger()
    test_fixture_2_methods_after_discussion()
    test_fixture_3_numbered_references_and_back_matter()
    test_fixture_4_date_metadata_false_heading()
    print("=" * 60)
    print("ALL FIXTURES PASSED ✅")
    print("=" * 60)
