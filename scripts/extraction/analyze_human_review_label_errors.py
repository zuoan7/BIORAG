#!/usr/bin/env python3
"""Write Chinese reports for Phase7G-1 human review label freeze outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction.freeze_human_table_review_labels import load_csv, rel, resolve


DEFAULT_REVIEW_PACK_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_FREEZE_DIR = ROOT / "data/experiments/v7_phase7_human_review_label_freeze"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_human_review_label_freeze"
PHASE7G_REPORT_DIR = ROOT / "reports/v7_phase7_expanded_table_review_pack"
PHASE7F_REPORT_DIR = ROOT / "reports/v7_phase7_gold_seed_validation"


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def counter(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row.get(field, "") or "empty") for row in rows)


def md_counter(counts: Counter[str]) -> list[str]:
    if not counts:
        return ["- 无"]
    return [f"- `{key}`：{value}" for key, value in counts.most_common()]


def md_id_list(rows: list[dict[str, Any]], extra_field: str | None = None) -> list[str]:
    if not rows:
        return ["- 无"]
    lines = []
    for row in rows:
        suffix = f"；`{extra_field}`=`{row.get(extra_field, '')}`" if extra_field else ""
        lines.append(
            f"- `{row.get('candidate_id', '')}`：`{row.get('doc_id', '')}` / "
            f"`{row.get('table_id', '')}`{suffix}"
        )
    return lines


def read_required_context(
    phase7g_report_dir: Path,
    phase7f_report_dir: Path,
    auto_excluded_path: Path,
) -> dict[str, Any]:
    required_reports = [
        phase7g_report_dir / "phase7g_summary.md",
        phase7g_report_dir / "human_review_instructions.md",
        phase7g_report_dir / "candidate_pool_construction_report.md",
        phase7g_report_dir / "review_pack_generation_report.md",
    ]
    texts = {rel(path): path.read_text(encoding="utf-8") for path in required_reports}
    phase7f_path = phase7f_report_dir / "phase7f_summary.md"
    texts[rel(phase7f_path)] = phase7f_path.read_text(encoding="utf-8")
    auto_excluded_rows = load_csv(auto_excluded_path)
    return {"context_reports": texts, "auto_excluded_count": len(auto_excluded_rows)}


def summarize_generated_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def priority_quality_lines(complete_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for priority in ["P0_quick_review", "P1_review", "P2_optional_spotcheck"]:
        rows = [row for row in complete_rows if row.get("review_priority") == priority]
        decisions = counter(rows, "review_decision")
        confirmed = decisions.get("accept_confirmed_seed_candidate", 0)
        partial = decisions.get("accept_partial_seed_candidate", 0)
        rejected = decisions.get("reject_boundary", 0) + decisions.get("reject_grid", 0)
        lines.append(
            f"- `{priority}`：完整审计 {len(rows)}；confirmed draft 候选 {confirmed}；"
            f"partial 候选 {partial}；reject {rejected}。"
        )
    return lines


def suggested_deviation_lines(complete_rows: list[dict[str, Any]]) -> list[str]:
    cross = Counter(
        f"{row.get('suggested_decision') or 'empty'} -> {row.get('review_decision') or 'empty'}"
        for row in complete_rows
    )
    lines = md_counter(cross)
    deviations = [
        row
        for row in complete_rows
        if (
            row.get("suggested_decision") == "likely_confirmed_candidate"
            and row.get("review_decision") != "accept_confirmed_seed_candidate"
        )
        or (
            row.get("suggested_decision") == "likely_partial_candidate"
            and row.get("review_decision") == "accept_confirmed_seed_candidate"
        )
    ]
    lines.append(f"- 主要偏差候选数量：{len(deviations)}。")
    return lines


def write_guardrail(report_dir: Path) -> None:
    write_text(
        report_dir / "phase7g_1_guardrail.md",
        [
            "# Phase7G-1 护栏",
            "",
            "1. 本轮定位为 human review label freeze and seed draft routing。",
            "2. 本轮不是继续人工标注；不要求用户重新标注。",
            "3. 本轮不是正式 gold construction；不构造正式 confirmed_seed 或 partial_seed。",
            "4. 本轮不是 extractor validation；不运行 coverage evaluation 或 flat comparison。",
            "5. 本轮不扩大候选池；只读取 Phase7G 已生成的 review pack。",
            "6. 本轮不接 production；不修改 ingestion 主链路或 production pipeline。",
            "7. 本轮不访问 Milvus / BM25，不写入 Milvus，不读取或查询 BM25 index。",
            "8. 本轮不运行 retrieval / embedding / rerank / Qwen / RAGAS / OCR / VLM。",
            "9. 本轮不伪造 value-level bbox，也不把 cell bbox 写成 value-level bbox。",
            "10. `confirmed_seed_draft` 不等于 `confirmed_seed`，只能作为下一步草案输入。",
            "11. Route C 仍只是 backlog，本轮不进入 Route C implementation。",
        ],
    )


def write_human_label_audit_report(
    report_dir: Path,
    frozen_rows: list[dict[str, Any]],
    unreviewed_rows: list[dict[str, Any]],
) -> None:
    complete_rows = [row for row in frozen_rows if row.get("core_fields_complete") == "true"]
    whitespace_count = sum(1 for row in frozen_rows if row.get("whitespace_normalized_fields"))
    invalid_count = sum(1 for row in frozen_rows if row.get("label_status") == "invalid_label")
    contradiction_count = sum(1 for row in frozen_rows if row.get("contradiction_flags"))
    unit_counts = counter(frozen_rows, "unit_or_note_ok")
    reference_counts = counter(frozen_rows, "reference_ok")
    write_text(
        report_dir / "human_label_audit_report.md",
        [
            "# 人工标签审计报告",
            "",
            "## 审计结论",
            f"- 完整人工标签数量：{len(complete_rows)}。",
            f"- 缺失人工标签数量：{len(unreviewed_rows)}。",
            f"- 是否检测到第 30 条缺失：{'是' if len(complete_rows) == 29 and unreviewed_rows else '否'}；仅记录，不阻断本轮。",
            f"- whitespace / 枚举规范化影响行数：{whitespace_count}。",
            f"- 非法枚举数量：{invalid_count}。",
            f"- review_decision 与核心标签矛盾数量：{contradiction_count}。",
            "",
            "## review_decision 分布",
            *md_counter(counter(complete_rows, "review_decision")),
            "",
            "## review_priority 分布",
            *md_counter(counter(complete_rows, "review_priority")),
            "",
            "## P0 / P1 / P2 质量差异",
            *priority_quality_lines(complete_rows),
            "",
            "## suggested_decision 与人工 review_decision 偏差",
            *suggested_deviation_lines(complete_rows),
            "",
            "## unit_or_note_ok 分布",
            *md_counter(unit_counts),
            "",
            "## reference_ok 分布",
            *md_counter(reference_counts),
            "",
            "## 未审候选",
            *md_id_list(unreviewed_rows),
            "",
            "## 影响",
            "- `unit_or_note_ok` 和 `reference_ok` 当前仍为 `unchecked`，因此不得写成 unit/reference binding confirmed。",
            "- `accept_confirmed_seed_candidate` 只能生成 `confirmed_seed_draft`，不能直接生成正式 `confirmed_seed`。",
        ],
    )


def write_seed_draft_routing_report(
    report_dir: Path,
    confirmed_rows: list[dict[str, Any]],
    partial_rows: list[dict[str, Any]],
    reject_rows: list[dict[str, Any]],
) -> None:
    write_text(
        report_dir / "seed_draft_routing_report.md",
        [
            "# Seed Draft 分流报告",
            "",
            "## confirmed_seed_draft 清单",
            f"- 数量：{len(confirmed_rows)}。",
            *md_id_list(confirmed_rows, "seed_draft_status"),
            "",
            "## partial 分流清单",
            f"- 数量：{len(partial_rows)}。",
            *md_id_list(partial_rows, "routing_status"),
            "",
            "## reject_boundary 清单",
            f"- 数量：{len(reject_rows)}。",
            *md_id_list(reject_rows),
            "",
            "## 为什么本轮不能直接构造 confirmed_seed",
            "- 当前人工标签只确认 boundary/grid/key_values 三类核心可用性。",
            "- `unit_or_note_ok` 和 `reference_ok` 全部仍为 `unchecked`，unit、footnote、reference binding 不能被过度确认。",
            "- 本轮没有运行 extractor validation，也没有构造正式 gold schema。",
            "- 因此输出只允许停在 `confirmed_seed_draft`。",
            "",
            "## 下一步最小人工补充字段",
            "- 只需针对 `confirmed_seed_draft` 行补充 `unit_or_note_ok` 与 `reference_ok`。",
            "- 可选补充 `binding_notes`，不需要重新做大规模标注。",
            "",
            "## 对 candidate scoring 的反馈",
            "- `reject_boundary` 应作为后续 boundary gate 和 risk_tags 权重调整反馈。",
            "- partial 候选需要保持二次分流，不应与 confirmed draft 混合。",
            "- `grid_sparse_or_unreadable`、`long_table_or_cross_page`、`matrix_heavy` 等风险标签仍需保留在后续 scoring 诊断中。",
        ],
    )


def write_summary_report(
    report_dir: Path,
    context: dict[str, Any],
    frozen_rows: list[dict[str, Any]],
    unreviewed_rows: list[dict[str, Any]],
    confirmed_rows: list[dict[str, Any]],
    partial_rows: list[dict[str, Any]],
    reject_rows: list[dict[str, Any]],
) -> None:
    complete_rows = [row for row in frozen_rows if row.get("core_fields_complete") == "true"]
    partial_counts = counter(partial_rows, "routing_status")
    whitespace_count = sum(1 for row in frozen_rows if row.get("whitespace_normalized_fields"))
    invalid_count = sum(1 for row in frozen_rows if row.get("label_status") == "invalid_label")
    generated_files = [
        DEFAULT_FREEZE_DIR / "frozen_review_labels.csv",
        DEFAULT_FREEZE_DIR / "frozen_review_labels.jsonl",
        DEFAULT_FREEZE_DIR / "label_audit_summary.csv",
        DEFAULT_FREEZE_DIR / "confirmed_seed_draft_candidates.csv",
        DEFAULT_FREEZE_DIR / "confirmed_seed_draft_candidates.jsonl",
        DEFAULT_FREEZE_DIR / "partial_candidate_routing.csv",
        DEFAULT_FREEZE_DIR / "reject_boundary_feedback.csv",
        DEFAULT_FREEZE_DIR / "unreviewed_candidates.csv",
        DEFAULT_FREEZE_DIR / "unit_reference_followup_template.csv",
        report_dir / "phase7g_1_guardrail.md",
        report_dir / "human_label_audit_report.md",
        report_dir / "seed_draft_routing_report.md",
        report_dir / "partial_candidate_routing_report.md",
        report_dir / "phase7g_1_summary.md",
    ]
    write_text(
        report_dir / "phase7g_1_summary.md",
        [
            "# Phase7G-1 总结",
            "",
            "## 1. 本轮生成文件",
            *(f"- `{rel(path)}`" for path in generated_files),
            "",
            "## 2. 新增 / 修改脚本",
            "- 新增：`scripts/extraction/freeze_human_table_review_labels.py`",
            "- 新增：`scripts/extraction/build_seed_drafts_from_review_labels.py`",
            "- 新增：`scripts/extraction/analyze_human_review_label_errors.py`",
            "",
            "## 3. 新增测试",
            "- 新增：`tests/test_phase7_human_review_label_freeze.py`",
            "",
            "## 4. 是否读取 review_labels_template.csv",
            "- 是。只读取并冻结，不覆盖原始 CSV。",
            "",
            f"## 5. 完整人工审计结果数量：{len(complete_rows)}",
            f"## 6. 未审候选数量：{len(unreviewed_rows)}",
            "",
            "## 7. review_decision 统计",
            *md_counter(counter(complete_rows, "review_decision")),
            "",
            "## 8. review_priority 统计",
            *md_counter(counter(complete_rows, "review_priority")),
            "",
            f"## 9. confirmed_seed_draft 数量：{len(confirmed_rows)}",
            "",
            "## 10. confirmed_seed_draft 清单",
            *md_id_list(confirmed_rows, "seed_draft_status"),
            "",
            "## 11. partial candidate routing 统计",
            *md_counter(partial_counts),
            "",
            "## 12. reject_boundary 数量与清单",
            f"- 数量：{len(reject_rows)}。",
            *md_id_list(reject_rows),
            "",
            "## 13. whitespace / 枚举规范化问题",
            f"- 检测到 whitespace / 枚举规范化影响行数：{whitespace_count}。",
            f"- 非法枚举数量：{invalid_count}。",
            "",
            "## 14. unit_or_note_ok / reference_ok 状态",
            "- `unit_or_note_ok`：",
            *md_counter(counter(frozen_rows, "unit_or_note_ok")),
            "- `reference_ok`：",
            *md_counter(counter(frozen_rows, "reference_ok")),
            "- 结论：当前不能确认 unit / footnote / reference binding。",
            "",
            "## 15. 是否构造正式 confirmed_seed",
            "- 否。只生成 `confirmed_seed_draft`。",
            "",
            "## 16. 是否建议进入 light binding review",
            "- 是。建议只对 `confirmed_seed_draft` 执行最小 `unit_or_note_ok` / `reference_ok` 补充。",
            "",
            "## 17. 是否建议再做一轮大标注",
            "- 否。当前建议是最小 light binding review，不是新一轮大标注。",
            "",
            "## 18. 是否建议进入 production",
            "- 否。本轮不接 production，也不提供 production readiness 结论。",
            "",
            "## 19. baseline / guardrail 是否漂移",
            "- 未发现漂移。本轮未修改 official dataset、official baseline、configs、baseline registry、chunks、BM25 或 Milvus。",
            "- 已读取 Phase7G 报告数量：4；已读取 Phase7F summary：1；auto_excluded rows："
            f"{context['auto_excluded_count']}。",
            "",
            "## 20. Route C 是否仍只是 backlog",
            "- 是。Route C 仍只是 backlog，本轮未进入 implementation。",
            "",
            "## 21. 明确未执行事项",
            "- 未扩大候选池。",
            "- 未生成新 review pack。",
            "- 未构造 confirmed_seed。",
            "- 未运行 validation。",
            "- 未做 flat comparison。",
            "- 未引入 Camelot。",
            "- 未引入 PyMuPDF。",
            "- 未改 ingestion pipeline。",
            "- 未改 configs。",
            "- 未改 README。",
            "- 未改 baseline registry。",
            "- 未改 official dataset。",
            "- 未改 official baseline。",
            "- 未重建 chunks。",
            "- 未重建 BM25。",
            "- 未访问 Milvus。",
            "- 未写入 Milvus。",
            "- 未读取或查询 BM25 index。",
            "- 未跑 retrieval。",
            "- 未跑 embedding/rerank。",
            "- 未调用 Qwen/RAGAS/OCR/VLM。",
            "- 未接入 production。",
            "- 未进入 Route C。",
        ],
    )


def analyze_human_review_labels(
    freeze_dir: Path,
    report_dir: Path,
    review_pack_dir: Path,
    phase7g_report_dir: Path,
    phase7f_report_dir: Path,
) -> dict[str, Any]:
    freeze_dir = resolve(freeze_dir)
    report_dir = resolve(report_dir)
    review_pack_dir = resolve(review_pack_dir)
    phase7g_report_dir = resolve(phase7g_report_dir)
    phase7f_report_dir = resolve(phase7f_report_dir)

    context = read_required_context(
        phase7g_report_dir=phase7g_report_dir,
        phase7f_report_dir=phase7f_report_dir,
        auto_excluded_path=review_pack_dir / "auto_excluded_candidates.csv",
    )
    frozen_rows = load_csv(freeze_dir / "frozen_review_labels.csv")
    confirmed_rows = summarize_generated_csv(freeze_dir / "confirmed_seed_draft_candidates.csv")
    partial_rows = summarize_generated_csv(freeze_dir / "partial_candidate_routing.csv")
    reject_rows = summarize_generated_csv(freeze_dir / "reject_boundary_feedback.csv")
    unreviewed_rows = summarize_generated_csv(freeze_dir / "unreviewed_candidates.csv")

    write_guardrail(report_dir)
    write_human_label_audit_report(report_dir, frozen_rows, unreviewed_rows)
    write_seed_draft_routing_report(report_dir, confirmed_rows, partial_rows, reject_rows)
    write_summary_report(report_dir, context, frozen_rows, unreviewed_rows, confirmed_rows, partial_rows, reject_rows)
    return {
        "complete_review_count": sum(1 for row in frozen_rows if row.get("core_fields_complete") == "true"),
        "unreviewed_count": len(unreviewed_rows),
        "confirmed_seed_draft_count": len(confirmed_rows),
        "partial_routing_counts": counter(partial_rows, "routing_status"),
        "reject_boundary_count": len(reject_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--review-pack-dir", type=Path, default=DEFAULT_REVIEW_PACK_DIR)
    parser.add_argument("--phase7g-report-dir", type=Path, default=PHASE7G_REPORT_DIR)
    parser.add_argument("--phase7f-report-dir", type=Path, default=PHASE7F_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_human_review_labels(
        freeze_dir=args.freeze_dir,
        report_dir=args.report_dir,
        review_pack_dir=args.review_pack_dir,
        phase7g_report_dir=args.phase7g_report_dir,
        phase7f_report_dir=args.phase7f_report_dir,
    )
    print(
        "human_label_audit: "
        f"complete={result['complete_review_count']} "
        f"unreviewed={result['unreviewed_count']} "
        f"confirmed_seed_draft={result['confirmed_seed_draft_count']} "
        f"reject_boundary={result['reject_boundary_count']}"
    )


if __name__ == "__main__":
    main()
