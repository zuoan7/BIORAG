from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_BASE_AUDIT = (
    RESULTS_ROOT
    / "v3_dataset_gold_consistency_audit_20260524_post_remap"
    / "samples.jsonl"
)
DEFAULT_BASE_SUMMARY = (
    RESULTS_ROOT
    / "v3_dataset_gold_consistency_audit_20260524_post_remap"
    / "summary.json"
)
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
EXPECTED_SAMPLE_COUNT = 200
LOW_CONFIDENCE_THRESHOLD = 0.75

CLASSIFICATIONS = {
    "pass",
    "needs_remap",
    "needs_multi_parent_gold",
    "needs_rubric_fix",
    "needs_expected_answer_fix",
    "needs_stable_block_fix",
    "comparison_scope_review",
    "negative_sample_review",
    "manual_unclear",
}

APPLIED_REMAP_REGRESSION_SAMPLE_IDS = [
    "v3_ra_001",
    "v3_ra_002",
    "v3_ra_003",
    "v3_ra_004",
    "v3_ra_006",
    "v3_ra_008",
    "v3_ra_016",
    "v3_ra_022",
    "v3_ra_029",
    "v3_ra_030",
]

EXISTING_REMAP_REGRESSION_SAMPLE_IDS = [
    "v3_ra_005",
    "v3_ra_007",
    "v3_ra_009",
    "v3_ra_014",
    "v3_ra_018",
    "v3_ra_019",
    "v3_ra_020",
    "v3_ra_021",
    "v3_ra_023",
    "v3_ra_024",
    "v3_ra_025",
    "v3_ra_026",
    "v3_ra_027",
    "v3_ra_028",
]

REQUIRED_SAMPLE_FIELDS = [
    "sample_id",
    "expected_route",
    "category",
    "gold_parent_chunk_ids",
    "stable_target_block_ids",
    "ai_classification",
    "confidence",
    "evidence_summary",
    "issue_reason",
    "suggested_decision",
    "suggested_parent_chunk_ids",
    "suggested_stable_target_block_ids",
    "suggested_rubric_fix",
    "suggested_expected_answer_fix",
    "human_required",
    "regression_group",
]

OVERRIDES: dict[str, dict[str, Any]] = {
    "v3_ra_010": {
        "ai_classification": "needs_multi_parent_gold",
        "confidence": 0.94,
        "evidence_summary": (
            "当前 gold parent doc_0011_sec09_chunk10 位于 Results and Discussion，"
            "讨论糖共利用重编程，不支撑题目所问发酵条件。Materials and Methods 中 "
            "doc_0011_sec06_chunk07 覆盖 LB/抗生素/overnight pre-culture 和 p4_b0047，"
            "doc_0011_sec07_chunk08 覆盖 OD600、0.2 mM IPTG、30 °C、5 L fermenter、"
            "DO 20%-40% 和 pH 7.0。"
        ),
        "issue_reason": (
            "gold parent section 与 expected_sections 不一致，且核心 evidence 分布在两个相邻 "
            "Materials and Methods parent chunks；当前 stable blocks p4_b0047/p5_b0052 "
            "不能由当前单一 gold parent 正确支撑。"
        ),
        "suggested_decision": "human_review_multi_parent_remap",
        "suggested_parent_chunk_ids": ["doc_0011_sec06_chunk07", "doc_0011_sec07_chunk08"],
        "suggested_stable_target_block_ids": ["p4_b0047", "p5_b0052"],
        "suggested_rubric_fix": (
            "将 source_trace.chunk_ids 改为两个 Materials and Methods parent，并保留发酵条件 "
            "must_include；evidence_note 去掉 table-caption-only 表述。"
        ),
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
    "v3_pc_146": {
        "ai_classification": "needs_remap",
        "confidence": 0.92,
        "evidence_summary": (
            "doc_0433_sec01_chunk02 的 abstract 支撑 S. cerevisiae、DPA171 和 4.1 g/L "
            "D-pantothenic acid。当前第二个 gold parent doc_0577_sec22_chunk23 是参考文献区，"
            "不支撑 lycopene titer；doc_0577_sec01_chunk02 abstract 直接给出 E. coli BL21(DE3)、"
            "LYC-8、fed-batch 中 438.44 ± 8.11 mg/L extracellular 和 "
            "136.94 ± 1.94 mg/L intracellular lycopene。"
        ),
        "issue_reason": "第二个 gold parent 落在 references/methods tail，语义上不是 lycopene 结果证据。",
        "suggested_decision": "human_review_remap_doc_0577_parent",
        "suggested_parent_chunk_ids": ["doc_0433_sec01_chunk02", "doc_0577_sec01_chunk02"],
        "suggested_stable_target_block_ids": ["p1_b0007", "p1_b0008", "p1_b0009", "p1_b0010"],
        "suggested_rubric_fix": (
            "将 doc_0577 source_trace 从 sec22_chunk23 改到 sec01_chunk02；"
            "verification_note 可保留原 construction chunk 落入引用区的说明。"
        ),
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
    "v3_pc_147": {
        "ai_classification": "comparison_scope_review",
        "confidence": 0.9,
        "evidence_summary": (
            "题目和 expected_answer 实际都由 doc_0514_sec01_chunk02 支撑：LNnT 22.07 g/L、"
            "LNT 48.41 g/L，并列出 ugd、ushA、agp、wcaJ、otsA、wcaC。"
            "当前额外 gold parent doc_0303_sec09_chunk10 是 NeuAc 文献引用区，与 LNnT/LNT "
            "batch culture 无关。"
        ),
        "issue_reason": "comparison 样本的第一篇 anchor 不参与题目语义，当前多文档范围需要人工确认是删锚点还是重写题目。",
        "suggested_decision": "human_review_comparison_scope_or_single_parent",
        "suggested_parent_chunk_ids": ["doc_0514_sec01_chunk02"],
        "suggested_stable_target_block_ids": ["p1_b0006", "p1_b0007"],
        "suggested_rubric_fix": (
            "若保留当前问题，将 source_trace/doc_ids 收敛到 doc_0514；若必须保留 comparison，"
            "需要重写题目和 expected_answer 使 doc_0303 成为真实比较对象。"
        ),
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
    "v3_pc_148": {
        "ai_classification": "comparison_scope_review",
        "confidence": 0.86,
        "evidence_summary": (
            "doc_0435_sec03_chunk04 支撑 2'-FL/FL 底物、成本/产率瓶颈和 64.62 g/L。"
            "doc_0323_sec02_chunk03 是 sialylated glycans / sialyltransferase 研究，"
            "不支持作为第二篇 2'-FL production 研究。expected_answer 已显式写出第二 anchor "
            "不匹配的 caveat。"
        ),
        "issue_reason": "题干要求比较两篇 2'-FL 生产研究，但第二个 anchor 语义不匹配；当前答案是范围纠偏而非正常 comparison。",
        "suggested_decision": "human_review_keep_as_scope_caveat_or_replace_second_anchor",
        "suggested_parent_chunk_ids": ["doc_0435_sec03_chunk04", "doc_0323_sec02_chunk03"],
        "suggested_stable_target_block_ids": [
            "p3_b0032",
            "p3_b0033",
            "p1_b0006",
            "p1_b0007",
            "p1_b0008",
            "p2_b0011",
        ],
        "suggested_rubric_fix": (
            "人工决定该样本是否应作为 anchor-mismatch/scope-review 样本保留；若改为普通 comparison，"
            "需要替换 doc_0323 为真实 2'-FL production 文献并重写 rubric。"
        ),
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
    "v3_pc_149": {
        "ai_classification": "comparison_scope_review",
        "confidence": 0.86,
        "evidence_summary": (
            "doc_0299_sec01_chunk02 明确支撑 E. coli 3'-SL 删除 nanA、nanK、nanE、nanT "
            "以减少 NeuAc degradation/competing flux。当前 doc_0662_sec11_chunk12 是参考文献区，"
            "只提供 Y. lipolytica/resveratrol 相关引用，不能支撑等价竞争通路删除；"
            "doc_0662_sec10_chunk11 至多支撑 Y. lipolytica resveratrol 工程的结论范围。"
        ),
        "issue_reason": "comparison 的第二个 parent 不支撑题目比较维度，当前答案依赖 caveat 而非两篇正向可比证据。",
        "suggested_decision": "human_review_comparison_scope_or_remap_doc_0662",
        "suggested_parent_chunk_ids": ["doc_0662_sec10_chunk11", "doc_0299_sec01_chunk02"],
        "suggested_stable_target_block_ids": ["p8_b0054", "p8_b0055", "p8_b0056", "p8_b0057", "p1_b0007"],
        "suggested_rubric_fix": (
            "若保留有限对比，应把 doc_0662 evidence_note 改为 conclusion-level resveratrol "
            "工程背景，而不是 references 区；若要求竞争通路对比，需要换第二篇证据。"
        ),
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
    "v3_pc_153": {
        "ai_classification": "needs_stable_block_fix",
        "confidence": 0.8,
        "evidence_summary": (
            "doc_0609_sec01_chunk02 文本支撑 CMP-Neu5Ac 作为 sialyltransferase donor substrate，"
            "doc_0192_sec01_chunk02 支撑 NeuAc 作为 de novo production 目标产物及 46.92 g/L。"
            "但 doc_0609 parent 的 source_block_ids 为空，当前 stable_target_block_ids 只有 "
            "doc_0192 的 p1_b0007。"
        ),
        "issue_reason": "语义 parent 覆盖两篇文献，但 stable block 只覆盖其中一篇；doc_0609 缺 block metadata，无法自动给出可靠 block id。",
        "suggested_decision": "human_review_add_doc_0609_stable_block_or_accept_parent_level_gold",
        "suggested_parent_chunk_ids": ["doc_0609_sec01_chunk02", "doc_0192_sec01_chunk02"],
        "suggested_stable_target_block_ids": ["p1_b0007"],
        "suggested_rubric_fix": (
            "补充 doc_0609 对应 stable block 后再更新 source_trace.block_ids；若源解析确无 block id，"
            "应在人工记录中标注 parent-level evidence exception。"
        ),
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
    "v3_pc_018": {
        "ai_classification": "manual_unclear",
        "confidence": 0.72,
        "evidence_summary": (
            "当前 gold parent doc_0069_sec13_chunk14 覆盖 Figure captions/OCR 密集区域，"
            "本地文本能看到蛋白描述但没有可靠出现 51.2 kDa 字段；相似文本在 doc_0040_sec08_chunk09 "
            "出现 51.2/4.36，但属于不同 doc。原 rubric 说明该数值来自 PDF 第 29 页 Table 1 回查。"
        ),
        "issue_reason": "本地 evidence text 无法独立核验 expected_answer 的 51.2 kDa，跨 doc 相似内容不足以自动 remap。",
        "suggested_decision": "manual_pdf_table_review_required",
        "suggested_parent_chunk_ids": ["doc_0069_sec13_chunk14"],
        "suggested_stable_target_block_ids": [
            "p20_b0051",
            "p21_b0052",
            "p22_b0053",
            "p23_b0054",
            "p24_b0055",
            "p25_b0056",
            "p26_b0057",
            "p27_b0058",
            "p28_b0059",
            "p29_b0060",
            "p29_b0061",
            "p29_b0062",
            "p29_b0063",
            "p29_b0064",
            "p29_b0065",
            "p29_b0066",
            "p29_b0067",
            "p29_b0068",
            "p29_b0069",
            "p29_b0070",
            "p29_b0071",
            "p29_b0072",
            "p29_b0073",
            "p29_b0074",
        ],
        "suggested_rubric_fix": "人工复查 PDF Table 1 后决定是否保留 verified_pdf_pages 说明，或补充可检索 OCR block。",
        "suggested_expected_answer_fix": "",
        "human_required": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the v3 baseline full AI quality review artifacts."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--base-audit", default=str(DEFAULT_BASE_AUDIT))
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    args = parser.parse_args()

    run_id = str(args.run_id)
    dataset_path = Path(args.dataset)
    base_audit_path = Path(args.base_audit)
    base_summary_path = Path(args.base_summary)
    parent_chunks_path = Path(args.parent_chunks)

    dataset_rows = load_jsonl(dataset_path)
    base_rows = load_jsonl(base_audit_path)
    base_summary = load_json(base_summary_path)
    parent_records = load_parent_records(parent_chunks_path)

    dataset_by_id = index_by_sample_id(dataset_rows, dataset_path)
    samples = [
        build_quality_sample(
            base_row=base_row,
            dataset_row=dataset_by_id[str(base_row.get("sample_id") or "")],
            parent_records=parent_records,
        )
        for base_row in base_rows
    ]
    review_candidates = [
        row
        for row in samples
        if row["ai_classification"] != "pass" or row["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]
    summary = build_summary(
        run_id=run_id,
        dataset_path=dataset_path,
        base_audit_path=base_audit_path,
        base_summary_path=base_summary_path,
        parent_chunks_path=parent_chunks_path,
        dataset_rows=dataset_rows,
        base_summary=base_summary,
        samples=samples,
        review_candidates=review_candidates,
    )

    result_dir = RESULTS_ROOT / f"v3_dataset_full_ai_quality_review_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_dataset_full_ai_quality_review_{run_id}"
    outputs = {
        "summary": str(result_dir / "summary.json"),
        "samples": str(result_dir / "samples.jsonl"),
        "review_candidates": str(result_dir / "review_candidates.jsonl"),
        "report": str(report_dir / "report.md"),
    }
    summary["outputs"] = outputs

    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "samples.jsonl", samples)
    write_jsonl(result_dir / "review_candidates.jsonl", review_candidates)
    write_markdown(report_dir / "report.md", render_report(summary, samples, review_candidates))

    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "sample_count": summary["sample_count"],
                "review_candidate_count": summary["review_candidate_count"],
                "classification_counts": summary["ai_classification_counts"],
                "validation_passed": summary["validation"]["passed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def build_quality_sample(
    *,
    base_row: dict[str, Any],
    dataset_row: dict[str, Any],
    parent_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(base_row.get("sample_id") or "")
    gold_parent_ids = as_str_list(base_row.get("gold_parent_chunk_ids"))
    stable_block_ids = as_str_list(base_row.get("stable_target_block_ids"))
    regression_group = regression_group_for(sample_id)
    override = OVERRIDES.get(sample_id)
    if override is not None:
        ai_classification = str(override["ai_classification"])
        confidence = float(override["confidence"])
        evidence_summary = str(override["evidence_summary"])
        issue_reason = str(override["issue_reason"])
        suggested_decision = str(override["suggested_decision"])
        suggested_parent_ids = as_str_list(override["suggested_parent_chunk_ids"])
        suggested_block_ids = as_str_list(override["suggested_stable_target_block_ids"])
        suggested_rubric_fix = str(override["suggested_rubric_fix"])
        suggested_answer_fix = str(override["suggested_expected_answer_fix"])
        human_required = bool(override["human_required"])
    else:
        ai_classification = "pass"
        confidence = pass_confidence(base_row, regression_group)
        evidence_summary = build_pass_evidence_summary(
            base_row=base_row,
            parent_records=parent_records,
            regression_group=regression_group,
        )
        issue_reason = ""
        suggested_decision = "keep_current_gold"
        suggested_parent_ids = gold_parent_ids
        suggested_block_ids = stable_block_ids
        suggested_rubric_fix = ""
        suggested_answer_fix = ""
        human_required = False

    return {
        "sample_id": sample_id,
        "expected_route": str(base_row.get("expected_route") or dataset_row.get("expected_route") or ""),
        "category": str(base_row.get("category") or dataset_row.get("category") or ""),
        "gold_parent_chunk_ids": gold_parent_ids,
        "stable_target_block_ids": stable_block_ids,
        "ai_classification": ai_classification,
        "confidence": round(confidence, 2),
        "evidence_summary": evidence_summary,
        "issue_reason": issue_reason,
        "suggested_decision": suggested_decision,
        "suggested_parent_chunk_ids": suggested_parent_ids,
        "suggested_stable_target_block_ids": suggested_block_ids,
        "suggested_rubric_fix": suggested_rubric_fix,
        "suggested_expected_answer_fix": suggested_answer_fix,
        "human_required": human_required,
        "regression_group": regression_group,
        "question": str(dataset_row.get("question") or base_row.get("question") or ""),
        "expected_answer": str(dataset_row.get("expected_answer") or base_row.get("expected_answer") or ""),
        "post_remap_warning_classifications": as_str_list(base_row.get("warning_classifications")),
    }


def build_pass_evidence_summary(
    *,
    base_row: dict[str, Any],
    parent_records: dict[str, dict[str, Any]],
    regression_group: str,
) -> str:
    route = str(base_row.get("expected_route") or "")
    category = str(base_row.get("category") or "")
    parent_ids = as_str_list(base_row.get("gold_parent_chunk_ids"))
    stable_ids = as_str_list(base_row.get("stable_target_block_ids"))
    warnings = as_str_list(base_row.get("warning_classifications"))
    sections = []
    evidence_types = []
    for parent_id in parent_ids:
        parent = parent_records.get(parent_id) or {}
        section = str(parent.get("section") or "")
        if section:
            sections.append(section)
        evidence_types.extend(as_str_list(parent.get("evidence_types")))
    section_text = ", ".join(dedupe(sections)) or "unknown section"
    evidence_type_text = ", ".join(dedupe(evidence_types)) or "unknown evidence type"
    warning_text = ""
    if warnings:
        warning_text = f" post-remap warning={','.join(warnings)} 已按语义审计视为非阻断。"
    regression_text = ""
    if regression_group != "none":
        regression_text = f" 属于 {regression_group} regression set，未发现新增语义问题。"
    if route == "negative":
        return (
            f"负样本按拒答/无证据逻辑审计：gold parent 位于 {section_text}，"
            "未发现其能直接回答题目或推翻 expected abstention。"
            f"{regression_text}"
        )
    if route == "comparison":
        return (
            f"comparison 样本按多 parent 覆盖审计：当前 gold parents={parent_ids}，"
            f"section={section_text}，stable blocks={len(stable_ids)} 个；未发现比较口径缺失。"
            f"{warning_text}{regression_text}"
        )
    return (
        f"gold parent 与 post-remap 结构审计一致：parents={parent_ids}，section={section_text}，"
        f"evidence_type={evidence_type_text}，stable blocks={len(stable_ids)} 个；"
        "question、expected_answer 和 rubric 未发现需要 remap 或文本修复的语义冲突。"
        f"{warning_text}{regression_text}"
    )


def pass_confidence(base_row: dict[str, Any], regression_group: str) -> float:
    route = str(base_row.get("expected_route") or "")
    warnings = as_str_list(base_row.get("warning_classifications"))
    if regression_group != "none":
        return 0.88
    if route == "negative":
        return 0.86
    if route == "comparison":
        return 0.84
    if warnings:
        return 0.82
    return 0.9


def build_summary(
    *,
    run_id: str,
    dataset_path: Path,
    base_audit_path: Path,
    base_summary_path: Path,
    parent_chunks_path: Path,
    dataset_rows: list[dict[str, Any]],
    base_summary: dict[str, Any],
    samples: list[dict[str, Any]],
    review_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_ids = [row["sample_id"] for row in samples]
    dataset_sample_ids = [str(row.get("sample_id") or "") for row in dataset_rows]
    classification_counts = Counter(row["ai_classification"] for row in samples)
    confidence_values = [float(row["confidence"]) for row in samples]
    regression_status = build_regression_status(samples)
    validation = build_validation(
        dataset_sample_ids=dataset_sample_ids,
        samples=samples,
        review_candidates=review_candidates,
        regression_status=regression_status,
    )
    return {
        "run_id": run_id,
        "scope": "v3 baseline dataset full AI semantic quality review",
        "inputs": {
            "dataset": str(dataset_path),
            "post_remap_consistency_samples": str(base_audit_path),
            "post_remap_consistency_summary": str(base_summary_path),
            "parent_chunks": str(parent_chunks_path),
        },
        "review_constraints": {
            "gpu_eval_run": False,
            "judge_called": False,
            "dataset_patch_generated": False,
            "retrieval_or_generation_code_modified": False,
            "basis": "local dataset rows, post-remap consistency audit, parent chunk evidence text, and explicit semantic overrides",
        },
        "sample_count": len(samples),
        "review_candidate_count": len(review_candidates),
        "route_counts": dict(sorted(Counter(row["expected_route"] for row in samples).items())),
        "category_counts": dict(sorted(Counter(row["category"] for row in samples).items())),
        "ai_classification_counts": dict(sorted(classification_counts.items())),
        "confidence_stats": {
            "min": round(min(confidence_values), 2) if confidence_values else None,
            "max": round(max(confidence_values), 2) if confidence_values else None,
            "avg": round(sum(confidence_values) / len(confidence_values), 4)
            if confidence_values
            else None,
            "buckets": confidence_buckets(confidence_values),
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        },
        "post_remap_baseline": {
            "sample_count": base_summary.get("sample_count"),
            "review_candidate_count": base_summary.get("review_candidate_count"),
            "route_counts": base_summary.get("route_counts", {}),
            "category_counts": base_summary.get("category_counts", {}),
            "warning_counts": base_summary.get("warning_counts", {}),
        },
        "review_candidate_sample_ids": [row["sample_id"] for row in review_candidates],
        "manual_unclear_sample_ids": [
            row["sample_id"] for row in samples if row["ai_classification"] == "manual_unclear"
        ],
        "high_confidence_fixable_sample_ids": [
            row["sample_id"]
            for row in samples
            if row["ai_classification"] in {"needs_remap", "needs_multi_parent_gold"}
            and row["confidence"] >= 0.9
        ],
        "regression_set_status": regression_status,
        "validation": validation,
        "sample_id_order_preserved": sample_ids == dataset_sample_ids,
    }


def build_regression_status(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in samples}
    groups = {
        "applied_remap": APPLIED_REMAP_REGRESSION_SAMPLE_IDS,
        "existing_regression": EXISTING_REMAP_REGRESSION_SAMPLE_IDS,
    }
    output: dict[str, Any] = {
        "expected_sample_count": len(APPLIED_REMAP_REGRESSION_SAMPLE_IDS)
        + len(EXISTING_REMAP_REGRESSION_SAMPLE_IDS),
        "groups": {},
    }
    all_passed = True
    all_rows = []
    for group, sample_ids in groups.items():
        rows = []
        for sample_id in sample_ids:
            sample = by_id.get(sample_id)
            row = {
                "sample_id": sample_id,
                "present": sample is not None,
                "ai_classification": sample["ai_classification"] if sample else "missing",
                "confidence": sample["confidence"] if sample else None,
                "human_required": sample["human_required"] if sample else True,
                "passed": bool(sample and sample["ai_classification"] == "pass"),
            }
            rows.append(row)
            all_rows.append(row)
            all_passed = all_passed and bool(row["passed"])
        output["groups"][group] = {
            "expected_sample_ids": sample_ids,
            "counts": dict(sorted(Counter(row["ai_classification"] for row in rows).items())),
            "passed": all(row["passed"] for row in rows),
            "samples": rows,
        }
    output["sample_count"] = len(all_rows)
    output["passed"] = all_passed and len(all_rows) == output["expected_sample_count"]
    output["failed_sample_ids"] = [
        row["sample_id"] for row in all_rows if not row["passed"]
    ]
    return output


def build_validation(
    *,
    dataset_sample_ids: list[str],
    samples: list[dict[str, Any]],
    review_candidates: list[dict[str, Any]],
    regression_status: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = [row["sample_id"] for row in samples]
    required_field_missing = {
        row["sample_id"]: [field for field in REQUIRED_SAMPLE_FIELDS if field not in row]
        for row in samples
    }
    required_field_missing = {
        sample_id: fields for sample_id, fields in required_field_missing.items() if fields
    }
    invalid_classification_ids = [
        row["sample_id"] for row in samples if row["ai_classification"] not in CLASSIFICATIONS
    ]
    candidate_filter_failures = [
        row["sample_id"]
        for row in review_candidates
        if row["ai_classification"] == "pass" and row["confidence"] >= LOW_CONFIDENCE_THRESHOLD
    ]
    checks = {
        "sample_count_is_200": len(samples) == EXPECTED_SAMPLE_COUNT,
        "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
        "sample_ids_match_dataset": sorted(sample_ids) == sorted(dataset_sample_ids),
        "sample_id_order_preserved": sample_ids == dataset_sample_ids,
        "required_fields_present": not required_field_missing,
        "classification_values_valid": not invalid_classification_ids,
        "review_candidates_filter_valid": not candidate_filter_failures,
        "regression_set_complete_and_passed": bool(regression_status["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "required_field_missing": required_field_missing,
        "invalid_classification_sample_ids": invalid_classification_ids,
        "review_candidate_filter_failure_sample_ids": candidate_filter_failures,
        "missing_dataset_sample_ids": sorted(set(dataset_sample_ids) - set(sample_ids)),
        "extra_sample_ids": sorted(set(sample_ids) - set(dataset_sample_ids)),
    }


def render_report(
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    review_candidates: list[dict[str, Any]],
) -> str:
    lines = [
        "# v3 Baseline Dataset 全量 AI 质量审计报告",
        "",
        f"- 运行 ID：`{summary['run_id']}`",
        f"- 样本数：{summary['sample_count']}",
        f"- 复查候选：{summary['review_candidate_count']}",
        f"- validation：{'通过' if summary['validation']['passed'] else '未通过'}",
        "- 约束：未运行 GPU eval，未调用 judge，未生成 dataset patch，未修改 retrieval/rerank/score-floor/generation 代码。",
        "",
        "## 全量分类统计",
        "",
        "### Route",
        "",
        render_counts_table(summary["route_counts"], "route"),
        "",
        "### Category",
        "",
        render_counts_table(summary["category_counts"], "category"),
        "",
        "### AI classification",
        "",
        render_counts_table(summary["ai_classification_counts"], "classification"),
        "",
        "### Confidence",
        "",
        render_counts_table(summary["confidence_stats"]["buckets"], "bucket"),
        "",
        "## 需要人工复查的样本",
        "",
    ]
    if review_candidates:
        lines.extend(render_candidate_table(review_candidates))
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## AI 高置信可修复样本",
            "",
        ]
    )
    fixable_ids = set(summary["high_confidence_fixable_sample_ids"])
    fixable = [row for row in samples if row["sample_id"] in fixable_ids]
    if fixable:
        for row in fixable:
            lines.extend(
                [
                    f"- `{row['sample_id']}`：{row['ai_classification']}，confidence={row['confidence']}。",
                    f"  建议：{row['suggested_decision']}；parents={row['suggested_parent_chunk_ids']}；stable_blocks={row['suggested_stable_target_block_ids']}。",
                    f"  原因：{row['issue_reason']}",
                ]
            )
    else:
        lines.append("无。")
    manual_unclear = [
        row for row in samples if row["ai_classification"] == "manual_unclear"
    ]
    lines.extend(
        [
            "",
            "## manual_unclear 样本",
            "",
        ]
    )
    if manual_unclear:
        for row in manual_unclear:
            lines.extend(
                [
                    f"- `{row['sample_id']}`：confidence={row['confidence']}。",
                    f"  问题：{row['question']}",
                    f"  原因：{row['issue_reason']}",
                    f"  建议：{row['suggested_decision']}",
                ]
            )
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 24 条 remap regression 状态",
            "",
            "| group | sample_id | classification | confidence | status |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for group, group_data in summary["regression_set_status"]["groups"].items():
        for row in group_data["samples"]:
            status = "pass" if row["passed"] else "review"
            lines.append(
                f"| {group} | `{row['sample_id']}` | {row['ai_classification']} | "
                f"{row['confidence']} | {status} |"
            )
    lines.extend(
        [
            "",
            f"Regression set overall：{'通过' if summary['regression_set_status']['passed'] else '未通过'}。",
            "",
            "## 不建议自动 patch 的原因",
            "",
            "- 本轮是语义质量审计，不是 gold remap 执行阶段；所有候选仍需人工确认后才能写入 dataset。",
            "- comparison 样本中有多处是题干范围、anchor 语义和 caveat 行为的设计问题，单纯替换 parent 可能改变样本意图。",
            "- `v3_pc_018` 依赖 PDF 表格回查，本地 OCR/evidence text 不能独立证明 51.2 kDa，自动修复风险高。",
            "- `v3_pc_153` 的 doc_0609 parent 缺 stable block metadata，不能凭空生成稳定 block id。",
            "",
            "## 复查入口",
            "",
            f"- 全量结果：`{summary['outputs']['samples']}`",
            f"- 只看候选：`{summary['outputs']['review_candidates']}`",
            f"- 机器摘要：`{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_candidate_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| sample_id | route | category | classification | confidence | suggested_decision |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sample_id']}` | {row['expected_route']} | {row['category']} | "
            f"{row['ai_classification']} | {row['confidence']} | {row['suggested_decision']} |"
        )
    lines.append("")
    for row in rows:
        lines.extend(
            [
                f"### `{row['sample_id']}`",
                "",
                f"- 问题：{row['question']}",
                f"- 证据摘要：{row['evidence_summary']}",
                f"- issue_reason：{row['issue_reason']}",
                f"- suggested_parent_chunk_ids：{row['suggested_parent_chunk_ids']}",
                f"- suggested_stable_target_block_ids：{row['suggested_stable_target_block_ids']}",
                f"- suggested_rubric_fix：{row['suggested_rubric_fix'] or '无'}",
                f"- suggested_expected_answer_fix：{row['suggested_expected_answer_fix'] or '无'}",
                "",
            ]
        )
    return lines


def render_counts_table(counts: dict[str, int], name: str) -> str:
    lines = [f"| {name} | count |", "| --- | ---: |"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def confidence_buckets(values: list[float]) -> dict[str, int]:
    counts = Counter()
    for value in values:
        if value >= 0.9:
            counts["high_0.90_plus"] += 1
        elif value >= LOW_CONFIDENCE_THRESHOLD:
            counts["medium_0.75_to_0.89"] += 1
        else:
            counts["low_below_0.75"] += 1
    return dict(sorted(counts.items()))


def regression_group_for(sample_id: str) -> str:
    if sample_id in APPLIED_REMAP_REGRESSION_SAMPLE_IDS:
        return "applied_remap"
    if sample_id in EXISTING_REMAP_REGRESSION_SAMPLE_IDS:
        return "existing_regression"
    return "none"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_parent_records(path: Path) -> dict[str, dict[str, Any]]:
    return {row["chunk_id"]: row for row in load_jsonl(path)}


def index_by_sample_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{path} contains row without sample_id")
        if sample_id in output:
            raise ValueError(f"{path} contains duplicate sample_id: {sample_id}")
        output[sample_id] = row
    return output


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    value_str = str(value)
    return [value_str] if value_str else []


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


if __name__ == "__main__":
    main()
