from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_NAME = "v3_dataset_design_review_min_manual_20260524"
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_REVIEW_DECISIONS = (
    RESULTS_ROOT / "v3_dataset_full_ai_quality_review_20260524" / "review_decisions_7.jsonl"
)
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_CHILD_CHUNKS = Path("data/paper_round1/chunks/child_chunks.jsonl")
DEFAULT_PAPER_DIR = Path("data/paper_round1/paper")

EXPECTED_SAMPLE_IDS = [
    "v3_pc_018",
    "v3_pc_147",
    "v3_pc_148",
    "v3_pc_149",
    "v3_pc_153",
]
MANUAL_ALLOWED_SAMPLE_IDS = {"v3_pc_147", "v3_pc_148", "v3_pc_149"}

RECOMMENDED_ACTIONS: dict[str, dict[str, Any]] = {
    "v3_pc_018": {
        "recommended_action": "no_dataset_patch_pdf_supported_ocr_missing",
        "manual_required": False,
        "confidence": 0.95,
        "dataset_patch_allowed": False,
        "reason": (
            "PDF 第 29 页可提取到 lectin-like protein 行的 51.2 kDa；dataset 已记录 "
            "verified_pdf_pages=[29]，问题在 OCR/table extraction 可检索性，不是 dataset 语义。"
        ),
        "evidence_refs": [
            "data/paper_round1/paper/doc_0069.pdf#page=29",
            "doc_0069_sec13_chunk14",
        ],
        "next_step": "加入 OCR/table extraction backlog，不进入人工设计取舍。",
    },
    "v3_pc_147": {
        "recommended_action": "rewrite_comparison_patch_candidate",
        "manual_required": True,
        "confidence": 0.9,
        "dataset_patch_allowed": True,
        "reason": (
            "当前 doc_0303_sec09_chunk10 是引用/方法尾部，不参与 LNnT/LNT 题目语义；"
            "可把 doc_0303 anchor 改为 abstract 中的 NeuAc 中间体瓶颈优化，保留 comparison 覆盖。"
        ),
        "evidence_refs": [
            "doc_0303_sec02_chunk03",
            "doc_0514_sec01_chunk02",
        ],
        "next_step": "人工确认是否接受 rewrite patch candidate；确认后再单独 patch dataset。",
    },
    "v3_pc_148": {
        "recommended_action": "accept_as_scope_control_no_patch",
        "manual_required": True,
        "confidence": 0.86,
        "dataset_patch_allowed": False,
        "reason": (
            "expected_answer 和 rubric 已明确 doc_0323 不是 2′-FL production anchor，"
            "当前样本可作为 anchor-mismatch/scope-control；不应自动改成普通 comparison。"
        ),
        "evidence_refs": [
            "doc_0435_sec03_chunk04",
            "doc_0435_sec02_chunk03",
            "doc_0323_sec02_chunk03",
        ],
        "next_step": "人工确认保留 scope-control；若要普通 comparison，需另找真实第二篇 2′-FL 生产文献。",
    },
    "v3_pc_149": {
        "recommended_action": "accept_as_scope_control_with_parent_candidate",
        "manual_required": True,
        "confidence": 0.86,
        "dataset_patch_allowed": False,
        "reason": (
            "doc_0662_sec10_chunk11 比当前 references parent 更像结论区证据，但仍不能支撑"
            "“竞争通路删除”；默认保留 caveat 型有限对比，不自动 patch。"
        ),
        "evidence_refs": [
            "doc_0299_sec01_chunk02",
            "doc_0662_sec10_chunk11",
            "doc_0662_sec11_chunk12",
        ],
        "next_step": "人工确认保留 scope-control，或要求后续替换第二篇证据。",
    },
    "v3_pc_153": {
        "recommended_action": "no_dataset_patch_parser_metadata_issue",
        "manual_required": False,
        "confidence": 0.9,
        "dataset_patch_allowed": False,
        "reason": (
            "doc_0609_sec01_chunk02 语义 parent 正确，但 parent/child 的 block_ids 均为空，"
            "不能凭空生成 stable block id；问题属于 parser metadata。"
        ),
        "evidence_refs": [
            "doc_0609_sec01_chunk02",
            "doc_0609_sec01_chunk02::child001",
            "doc_0192_sec01_chunk02",
        ],
        "next_step": "加入 parser metadata backlog；修复解析元数据后再考虑 consistency audit。",
    },
}

PATCH_CANDIDATE_V3_PC_147: dict[str, Any] = {
    "sample_id": "v3_pc_147",
    "patch_type": "rewrite_comparison_keep_route",
    "new_expected_route": "comparison",
    "new_category": "comparison",
    "new_question": "NeuAc 中间体瓶颈优化和 LNnT/LNT 的 UDP-galactose 竞争通路删除分别解决了什么问题？",
    "new_expected_answer": (
        "NeuAc 的 B. subtilis 研究针对中间体积累造成的合成瓶颈：先重构并优化不产生 GlcNAc 的 "
        "UDP-GlcNAc epimerase pathway，使 NeuAc titer 提高 5.9 g/L 且消除 GlcNAc；"
        "再通过 NeuAc synthase 定向进化和增强 phosphoenolpyruvate 供应降低 ManNAc，"
        "ManNAc 下降 46.3%，NeuAc titer 提高 54.9% 至 7.9 g/L，最终 3 L 发酵达到 21.8 g/L。"
        "LNnT/LNT 的 E. coli 研究则通过删除 ugd、ushA、agp、wcaJ、otsA 和 wcaC 阻断 "
        "UDP-galactose 合成通路竞争，并在 supplemented batch culture 中分别得到 "
        "LNnT 22.07 g/L 和 LNT 48.41 g/L。"
    ),
    "new_expected_doc_ids": ["doc_0303", "doc_0514"],
    "new_expected_sections": ["Abstract"],
    "new_source_trace_chunk_ids": ["doc_0303_sec02_chunk03", "doc_0514_sec01_chunk02"],
    "new_source_trace_block_ids": [
        "p1_b0011",
        "p1_b0012",
        "p1_b0013",
        "p1_b0014",
        "p1_b0006",
        "p1_b0007",
    ],
    "new_stable_target_block_ids": [
        "p1_b0011",
        "p1_b0012",
        "p1_b0013",
        "p1_b0014",
        "p1_b0006",
        "p1_b0007",
    ],
    "new_evidence_note": (
        "doc_0303 abstract 支撑 NeuAc 中间体 GlcNAc/ManNAc 积累瓶颈及优化结果；"
        "doc_0514 abstract 支撑 LNnT/LNT 的 UDP-galactose 竞争通路删除基因和 titer。"
    ),
    "new_must_include": [
        "doc_0303: accumulated intermediates cause NeuAc bottlenecks",
        "GlcNAc elimination via optimized UDP-GlcNAc epimerase pathway",
        "ManNAc decreased by 46.3%",
        "NeuAc 7.9 g/L and final 21.8 g/L in 3-L fermenter",
        "doc_0514: delete ugd, ushA, agp, wcaJ, otsA, wcaC",
        "LNnT 22.07 g/L and LNT 48.41 g/L",
    ],
    "new_reject_if": [
        "把 doc_0303_sec09_chunk10 引用区继续作为证据",
        "把 doc_0303 写成 LNnT/LNT 文献",
        "漏掉 UDP-galactose 竞争通路删除基因列表",
        "把 comparison 降级为 factoid",
    ],
    "requires_human_approval": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a minimal manual design review pack for the remaining five v3 samples."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--review-decisions", default=str(DEFAULT_REVIEW_DECISIONS))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--paper-dir", default=str(DEFAULT_PAPER_DIR))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    review_decisions_path = Path(args.review_decisions)
    parent_chunks_path = Path(args.parent_chunks)
    child_chunks_path = Path(args.child_chunks)
    paper_dir = Path(args.paper_dir)
    run_name = str(args.run_name)

    dataset_rows = load_jsonl(dataset_path)
    review_decision_rows = load_jsonl(review_decisions_path)
    parent_rows = load_jsonl(parent_chunks_path)
    child_rows = load_jsonl(child_chunks_path)

    dataset_by_id = index_by_key(dataset_rows, "sample_id", dataset_path)
    review_decisions_by_id = index_by_key(review_decision_rows, "sample_id", review_decisions_path)
    parent_by_chunk_id = index_by_key(parent_rows, "chunk_id", parent_chunks_path)
    child_by_chunk_id = index_by_key(child_rows, "chunk_id", child_chunks_path)

    pdf_check = check_doc_0069_page_29(paper_dir / "doc_0069.pdf")
    actions = build_recommended_actions(pdf_check)
    patch_candidates = build_patch_candidates(parent_by_chunk_id)
    pack = [
        build_review_pack_row(
            sample_id=sample_id,
            dataset_row=dataset_by_id[sample_id],
            review_decision=review_decisions_by_id[sample_id],
            parent_by_chunk_id=parent_by_chunk_id,
            child_by_chunk_id=child_by_chunk_id,
            recommended_action=actions[sample_id],
            patch_candidate=next(
                (candidate for candidate in patch_candidates if candidate["sample_id"] == sample_id),
                None,
            ),
            pdf_check=pdf_check if sample_id == "v3_pc_018" else None,
        )
        for sample_id in EXPECTED_SAMPLE_IDS
    ]
    summary = build_summary(
        run_name=run_name,
        dataset_path=dataset_path,
        review_decisions_path=review_decisions_path,
        parent_chunks_path=parent_chunks_path,
        child_chunks_path=child_chunks_path,
        paper_dir=paper_dir,
        pack=pack,
        actions=list(actions.values()),
        patch_candidates=patch_candidates,
        parent_by_chunk_id=parent_by_chunk_id,
        child_by_chunk_id=child_by_chunk_id,
        pdf_check=pdf_check,
    )

    result_dir = RESULTS_ROOT / run_name
    report_dir = REPORTS_ROOT / run_name
    outputs = {
        "design_review_pack": str(result_dir / "design_review_pack.jsonl"),
        "recommended_actions": str(result_dir / "recommended_actions.jsonl"),
        "patch_candidates": str(result_dir / "patch_candidates.jsonl"),
        "summary": str(result_dir / "summary.json"),
        "report": str(report_dir / "report.md"),
    }
    summary["outputs"] = outputs

    write_jsonl(result_dir / "design_review_pack.jsonl", pack)
    write_jsonl(result_dir / "recommended_actions.jsonl", list(actions.values()))
    write_jsonl(result_dir / "patch_candidates.jsonl", patch_candidates)
    write_json(result_dir / "summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary, pack, list(actions.values()), patch_candidates))

    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report": str(report_dir / "report.md"),
                "sample_count": len(pack),
                "patch_candidate_count": len(patch_candidates),
                "validation_passed": summary["validation"]["passed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def build_recommended_actions(pdf_check: dict[str, Any]) -> dict[str, dict[str, Any]]:
    actions = {
        sample_id: {"sample_id": sample_id, **action}
        for sample_id, action in RECOMMENDED_ACTIONS.items()
    }
    if not pdf_check["contains_51_2"]:
        actions["v3_pc_018"] = {
            **actions["v3_pc_018"],
            "recommended_action": "manual_pdf_recheck_required",
            "manual_required": True,
            "confidence": 0.5,
            "reason": (
                "自动 PDF text extraction 未在 doc_0069 第 29 页找到 51.2；"
                "需要人工重新核验 PDF 表格后才能归档为 OCR/table extraction backlog。"
            ),
            "next_step": "人工复查 doc_0069.pdf 第 29 页 Table 1。",
        }
    return {sample_id: actions[sample_id] for sample_id in EXPECTED_SAMPLE_IDS}


def build_patch_candidates(parent_by_chunk_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = dict(PATCH_CANDIDATE_V3_PC_147)
    for chunk_id in candidate["new_source_trace_chunk_ids"]:
        parent = parent_by_chunk_id.get(chunk_id)
        if parent is None:
            raise ValueError(f"Patch candidate references missing parent chunk: {chunk_id}")
    return [candidate]


def build_review_pack_row(
    *,
    sample_id: str,
    dataset_row: dict[str, Any],
    review_decision: dict[str, Any],
    parent_by_chunk_id: dict[str, dict[str, Any]],
    child_by_chunk_id: dict[str, dict[str, Any]],
    recommended_action: dict[str, Any],
    patch_candidate: dict[str, Any] | None,
    pdf_check: dict[str, Any] | None,
) -> dict[str, Any]:
    source_trace = dataset_row.get("answer_rubric", {}).get("source_trace", {})
    current_chunk_ids = as_str_list(source_trace.get("chunk_ids"))
    selected_parent_ids = as_str_list(review_decision.get("selected_parent_chunk_ids"))
    evidence_ref_ids = as_str_list(recommended_action.get("evidence_refs"))
    all_chunk_ids = dedupe(
        [
            chunk_id
            for chunk_id in current_chunk_ids + selected_parent_ids + evidence_ref_ids
            if is_chunk_ref(chunk_id)
        ]
    )
    return {
        "sample_id": sample_id,
        "current_dataset": {
            "question": dataset_row.get("question", ""),
            "expected_route": dataset_row.get("expected_route", ""),
            "category": dataset_row.get("category", ""),
            "expected_answer": dataset_row.get("expected_answer", ""),
            "expected_doc_ids": as_str_list(dataset_row.get("expected_doc_ids")),
            "expected_sections": as_str_list(dataset_row.get("expected_sections")),
            "source_trace_chunk_ids": current_chunk_ids,
            "source_trace_block_ids": as_str_list(source_trace.get("block_ids")),
            "stable_target_block_ids": as_str_list(dataset_row.get("stable_target_block_ids")),
            "must_include": as_str_list(dataset_row.get("answer_rubric", {}).get("must_include")),
            "reject_if": as_str_list(dataset_row.get("answer_rubric", {}).get("reject_if")),
        },
        "prior_review_decision": {
            "review_decision": review_decision.get("review_decision", ""),
            "final_classification": review_decision.get("final_classification", ""),
            "selected_parent_chunk_ids": selected_parent_ids,
            "selected_stable_block_ids": as_str_list(review_decision.get("selected_stable_block_ids")),
            "notes": review_decision.get("notes", ""),
            "evidence_checked": as_str_list(review_decision.get("evidence_checked")),
        },
        "evidence_context": [
            summarize_chunk(chunk_id, parent_by_chunk_id, child_by_chunk_id)
            for chunk_id in all_chunk_ids
        ],
        "pdf_check": pdf_check or {},
        "recommended_action": recommended_action,
        "patch_candidate": patch_candidate or {},
    }


def build_summary(
    *,
    run_name: str,
    dataset_path: Path,
    review_decisions_path: Path,
    parent_chunks_path: Path,
    child_chunks_path: Path,
    paper_dir: Path,
    pack: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    patch_candidates: list[dict[str, Any]],
    parent_by_chunk_id: dict[str, dict[str, Any]],
    child_by_chunk_id: dict[str, dict[str, Any]],
    pdf_check: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_outputs(
        pack=pack,
        actions=actions,
        patch_candidates=patch_candidates,
        parent_by_chunk_id=parent_by_chunk_id,
        child_by_chunk_id=child_by_chunk_id,
        pdf_check=pdf_check,
    )
    return {
        "run_name": run_name,
        "scope": "only remaining unpatched five v3 design-review samples",
        "inputs": {
            "dataset": str(dataset_path),
            "review_decisions_7": str(review_decisions_path),
            "parent_chunks": str(parent_chunks_path),
            "child_chunks": str(child_chunks_path),
            "paper_dir": str(paper_dir),
        },
        "constraints": {
            "dataset_modified": False,
            "gpu_eval_run": False,
            "judge_called": False,
            "full_200_review_rerun": False,
            "patch_application_deferred": True,
        },
        "expected_sample_ids": EXPECTED_SAMPLE_IDS,
        "sample_count": len(pack),
        "recommended_action_counts": dict(
            sorted(Counter(row["recommended_action"] for row in actions).items())
        ),
        "manual_required_sample_ids": [
            row["sample_id"] for row in actions if bool(row["manual_required"])
        ],
        "automatic_backlog_sample_ids": [
            row["sample_id"] for row in actions if not bool(row["manual_required"])
        ],
        "patch_candidate_sample_ids": [row["sample_id"] for row in patch_candidates],
        "pdf_checks": {"v3_pc_018": pdf_check},
        "validation": validation,
    }


def validate_outputs(
    *,
    pack: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    patch_candidates: list[dict[str, Any]],
    parent_by_chunk_id: dict[str, dict[str, Any]],
    child_by_chunk_id: dict[str, dict[str, Any]],
    pdf_check: dict[str, Any],
) -> dict[str, Any]:
    pack_ids = [row["sample_id"] for row in pack]
    action_ids = [row["sample_id"] for row in actions]
    manual_ids = {row["sample_id"] for row in actions if bool(row["manual_required"])}
    doc_0609_parent = parent_by_chunk_id.get("doc_0609_sec01_chunk02") or {}
    doc_0609_child = child_by_chunk_id.get("doc_0609_sec01_chunk02::child001") or {}
    patch_candidate_chunk_ids = [
        chunk_id
        for candidate in patch_candidates
        for chunk_id in as_str_list(candidate.get("new_source_trace_chunk_ids"))
    ]
    patch_candidate_block_ids = [
        block_id
        for candidate in patch_candidates
        for block_id in as_str_list(candidate.get("new_stable_target_block_ids"))
    ]
    candidate_source_blocks = []
    for chunk_id in patch_candidate_chunk_ids:
        candidate_source_blocks.extend(as_str_list(parent_by_chunk_id.get(chunk_id, {}).get("block_ids")))
    checks = {
        "design_review_pack_has_5_rows": len(pack) == 5,
        "design_review_pack_sample_ids_exact": pack_ids == EXPECTED_SAMPLE_IDS,
        "design_review_pack_sample_ids_unique": len(pack_ids) == len(set(pack_ids)),
        "recommended_actions_has_5_rows": len(actions) == 5,
        "recommended_action_sample_ids_exact": action_ids == EXPECTED_SAMPLE_IDS,
        "manual_required_only_allowed_three": manual_ids <= MANUAL_ALLOWED_SAMPLE_IDS,
        "manual_required_expected_three": manual_ids == MANUAL_ALLOWED_SAMPLE_IDS,
        "patch_candidates_only_v3_pc_147": [row["sample_id"] for row in patch_candidates] == ["v3_pc_147"],
        "patch_candidate_parent_chunks_exist": all(
            chunk_id in parent_by_chunk_id for chunk_id in patch_candidate_chunk_ids
        ),
        "patch_candidate_blocks_exist_in_candidate_parents": all(
            block_id in candidate_source_blocks for block_id in patch_candidate_block_ids
        ),
        "v3_pc_018_pdf_page_29_extracts_51_2": bool(pdf_check["contains_51_2"]),
        "v3_pc_153_doc_0609_parent_block_ids_still_empty": not as_str_list(
            doc_0609_parent.get("block_ids")
        )
        and not as_str_list(doc_0609_parent.get("source_block_ids")),
        "v3_pc_153_doc_0609_child_block_ids_still_empty": not as_str_list(
            doc_0609_child.get("block_ids")
        )
        and not as_str_list(doc_0609_child.get("source_block_ids")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manual_required_sample_ids": sorted(manual_ids),
        "patch_candidate_chunk_ids": patch_candidate_chunk_ids,
        "patch_candidate_block_ids": patch_candidate_block_ids,
    }


def render_report(
    summary: dict[str, Any],
    pack: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    patch_candidates: list[dict[str, Any]],
) -> str:
    lines = [
        "# v3 剩余 5 条最小人工设计复查报告",
        "",
        f"- 运行：`{summary['run_name']}`",
        f"- 范围：`{', '.join(summary['expected_sample_ids'])}`",
        f"- validation：{'通过' if summary['validation']['passed'] else '未通过'}",
        "- 本轮不修改 dataset，不运行 GPU eval，不调用 judge，不重读 200 条全量样本。",
        "- 若后续接受 patch candidate，需要单独执行 dataset patch 并重跑 consistency audit。",
        "",
        "## 推荐动作",
        "",
        "| sample_id | recommended_action | manual_required | dataset_patch_allowed | next_step |",
        "| --- | --- | --- | --- | --- |",
    ]
    for action in actions:
        lines.append(
            f"| `{action['sample_id']}` | `{action['recommended_action']}` | "
            f"{str(action['manual_required']).lower()} | "
            f"{str(action['dataset_patch_allowed']).lower()} | {action['next_step']} |"
        )
    lines.extend(["", "## 人工只需确认 3 项", ""])
    lines.extend(
        [
            "- `v3_pc_147`：是否接受 rewrite patch candidate，保留 comparison 类型并把 `doc_0303_sec09_chunk10` 改为 `doc_0303_sec02_chunk03`。",
            "- `v3_pc_148`：是否接受为 scope-control 样本且不改；若要普通 comparison，需另找真实第二篇 2′-FL 生产文献。",
            "- `v3_pc_149`：是否接受为 caveat/scope-control 样本；`doc_0662_sec10_chunk11` 可替代引用区 parent，但仍不支撑“竞争通路删除”。",
            "",
            "## 自动归档",
            "",
        ]
    )
    lines.extend(
        [
            "- `v3_pc_018`：PDF 第 29 页 text extraction 已找到 `51.2`，dataset 不 patch；进入 OCR/table extraction backlog。",
            "- `v3_pc_153`：`doc_0609_sec01_chunk02` parent/child 的 `block_ids` 和 `source_block_ids` 仍为空，dataset 不 patch；进入 parser metadata backlog。",
            "",
        ]
    )
    if patch_candidates:
        candidate = patch_candidates[0]
        lines.extend(
            [
                "## Patch Candidate",
                "",
                f"- sample_id：`{candidate['sample_id']}`",
                f"- patch_type：`{candidate['patch_type']}`",
                f"- new_question：{candidate['new_question']}",
                f"- new_source_trace_chunk_ids：`{candidate['new_source_trace_chunk_ids']}`",
                f"- new_stable_target_block_ids：`{candidate['new_stable_target_block_ids']}`",
                f"- requires_human_approval：{str(candidate['requires_human_approval']).lower()}",
                "",
            ]
        )
    lines.extend(
        [
            "## 样本复查摘要",
            "",
        ]
    )
    for row in pack:
        action = row["recommended_action"]
        lines.extend(
            [
                f"### `{row['sample_id']}`",
                "",
                f"- 当前问题：{row['current_dataset']['question']}",
                f"- 当前 route/category：`{row['current_dataset']['expected_route']}` / `{row['current_dataset']['category']}`",
                f"- prior decision：`{row['prior_review_decision']['review_decision']}` / `{row['prior_review_decision']['final_classification']}`",
                f"- 推荐：`{action['recommended_action']}`；原因：{action['reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 输出文件",
            "",
            f"- design review pack：`{summary['outputs']['design_review_pack']}`",
            f"- recommended actions：`{summary['outputs']['recommended_actions']}`",
            f"- patch candidates：`{summary['outputs']['patch_candidates']}`",
            f"- summary：`{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_chunk(
    chunk_id: str,
    parent_by_chunk_id: dict[str, dict[str, Any]],
    child_by_chunk_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parent_id = normalize_to_parent_chunk_id(chunk_id)
    row = child_by_chunk_id.get(chunk_id) or parent_by_chunk_id.get(parent_id) or {}
    return {
        "chunk_id": chunk_id,
        "normalized_parent_chunk_id": parent_id,
        "exists_in_parent_index": parent_id in parent_by_chunk_id,
        "exists_in_child_index": chunk_id in child_by_chunk_id,
        "doc_id": row.get("doc_id", ""),
        "source_file": row.get("source_file", ""),
        "section": row.get("section", ""),
        "block_ids": as_str_list(row.get("block_ids")),
        "source_block_ids": as_str_list(row.get("source_block_ids")),
        "text_preview": one_line(str(row.get("text", "")))[:700],
    }


def check_doc_0069_page_29(pdf_path: Path) -> dict[str, Any]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return {
            "pdf_path": str(pdf_path),
            "page_number": 29,
            "extractor": "fitz",
            "available": False,
            "contains_51_2": False,
            "text_excerpt": "",
            "error": "fitz is not installed",
        }
    if not pdf_path.exists():
        return {
            "pdf_path": str(pdf_path),
            "page_number": 29,
            "extractor": "fitz",
            "available": False,
            "contains_51_2": False,
            "text_excerpt": "",
            "error": "PDF file is missing",
        }
    try:
        document = fitz.open(pdf_path)
        page_index = 28
        text = document.load_page(page_index).get_text()
        contains = "51.2" in text
        excerpt = extract_excerpt(text, "51.2")
        return {
            "pdf_path": str(pdf_path),
            "page_number": 29,
            "page_index": page_index,
            "extractor": "fitz",
            "available": True,
            "contains_51_2": contains,
            "text_excerpt": excerpt,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - defensive CLI guard.
        return {
            "pdf_path": str(pdf_path),
            "page_number": 29,
            "extractor": "fitz",
            "available": True,
            "contains_51_2": False,
            "text_excerpt": "",
            "error": str(exc),
        }


def extract_excerpt(text: str, marker: str, radius: int = 180) -> str:
    idx = text.find(marker)
    if idx < 0:
        return one_line(text[: radius * 2])
    return one_line(text[max(0, idx - radius) : idx + radius])


def is_chunk_ref(value: str) -> bool:
    return value.startswith("doc_") and "_chunk" in value


def normalize_to_parent_chunk_id(chunk_id: str) -> str:
    if "::" in chunk_id:
        return chunk_id.split("::", 1)[0]
    return chunk_id


def one_line(text: str) -> str:
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_by_key(rows: list[dict[str, Any]], key: str, path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{path} contains a row without {key}")
        if value in output:
            raise ValueError(f"{path} contains duplicate {key}: {value}")
        output[value] = row
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
