from __future__ import annotations

from src.synbio_rag.application.generation_v2.branch_parser import parse_comparison_branches


def test_parse_ent084_style_two_strategy_question() -> None:
    result = parse_comparison_branches(
        "比较文库中 Pichia pastoris HAC1 过表达与 S. cerevisiae OCH1 缺失两种提升蛋白分泌的策略。"
    )

    assert result.parse_ok is True
    assert result.branches == ["Pichia pastoris HAC1 过表达", "S. cerevisiae OCH1 缺失"]


def test_parse_ent090_style_two_effect_question() -> None:
    result = parse_comparison_branches(
        "比较文库中 2′-FL 改善骨质疏松和 HMO 体外发酵调节肠道菌群两种健康效应研究的异同。"
    )

    assert result.parse_ok is True
    assert result.branches == ["2′-FL 改善骨质疏松", "HMO 体外发酵调节肠道菌群"]


def test_parse_a_yu_b_difference_form() -> None:
    result = parse_comparison_branches("比较 HAC1 过表达与 OCH1 缺失的异同。")

    assert result.parse_ok is True
    assert result.branches == ["HAC1 过表达", "OCH1 缺失"]


def test_parse_a_he_b_difference_form() -> None:
    result = parse_comparison_branches("比较 2′-FL 改善骨质疏松和 HMO 体外发酵调节肠道菌群的差异。")

    assert result.parse_ok is True
    assert result.branches == ["2′-FL 改善骨质疏松", "HMO 体外发酵调节肠道菌群"]


def test_existing_patterns_still_work() -> None:
    assert parse_comparison_branches("一类是优化甲醇诱导时机和能量利用，另一类是直接增加 AOX1 启动子调控表达盒拷贝数").parse_ok is True
    assert parse_comparison_branches("A vs B").branches == ["A", "B"]
    assert parse_comparison_branches("比较 A 和 B").branches == ["A", "B"]


def test_non_comparison_questions_stay_parse_failed() -> None:
    assert parse_comparison_branches("FAM20A 是否调控 FAM20C 定位？").parse_ok is False
    assert parse_comparison_branches("总结 Pichia pastoris 中提高 PPP 通量为何会提升重组蛋白产量。").parse_ok is False
    assert parse_comparison_branches("文库中是否有关于 mRNA 疫苗 III 期临床试验结果的数据？").parse_ok is False


def test_overlong_branch_is_rejected() -> None:
    long_left = "Pichia pastoris " + ("非常复杂的表达优化描述" * 20)
    result = parse_comparison_branches(f"比较 {long_left} 与 OCH1 缺失两种策略。")

    assert result.parse_ok is False
    assert result.reason == "branch_too_long"
