from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_DECISIONS = (
    RESULTS_ROOT
    / "v3_dataset_gold_consistency_audit_20260524"
    / "review_decisions.jsonl"
)
SUPPORTED_DECISIONS = {"remap", "remap_multi_parent"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply v3 dataset-level gold consistency remap decisions."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    run_id = str(args.run_id)
    dataset_path = Path(args.dataset)
    decisions_path = Path(args.decisions)
    parent_chunks_path = Path(args.parent_chunks)
    decisions = load_decisions(decisions_path)
    dataset_rows = load_jsonl(dataset_path)
    parent_records = load_parent_records(
        parent_chunks_path,
        {
            parent_id
            for decision in decisions
            for parent_id in decision["selected_parent_chunk_ids"]
        },
    )
    updated_rows, changes = apply_decisions_to_rows(
        rows=dataset_rows,
        decisions=decisions,
        parent_records=parent_records,
    )
    summary = build_summary(
        run_id=run_id,
        dataset_path=dataset_path,
        decisions_path=decisions_path,
        parent_chunks_path=parent_chunks_path,
        dry_run=bool(args.dry_run),
        changes=changes,
    )
    result_dir = RESULTS_ROOT / f"v3_dataset_gold_consistency_remap_applied_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_dataset_gold_consistency_remap_applied_{run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "changes.jsonl", changes)
    write_markdown(report_dir / "report.md", render_report(summary, changes))

    if not args.dry_run:
        write_jsonl_atomic(dataset_path, updated_rows)

    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "dry_run": bool(args.dry_run),
                "change_count": len(changes),
                "status_counts": summary["status_counts"],
                "validation_passed": summary["validation"]["passed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def load_decisions(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    decisions = []
    for line_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or "")
        decision = str(row.get("decision") or "")
        selected_parent_ids = as_str_list(row.get("selected_parent_chunk_ids"))
        selected_block_ids = as_str_list(row.get("selected_stable_block_ids"))
        if not sample_id:
            raise ValueError(f"{path}:{line_number} missing sample_id")
        if decision not in SUPPORTED_DECISIONS:
            raise ValueError(f"{path}:{line_number} unsupported decision: {decision!r}")
        if not selected_parent_ids:
            raise ValueError(f"{path}:{line_number} missing selected_parent_chunk_ids")
        if not selected_block_ids:
            raise ValueError(f"{path}:{line_number} missing selected_stable_block_ids")
        if decision == "remap" and len(selected_parent_ids) != 1:
            raise ValueError(f"{path}:{line_number} remap requires exactly one parent")
        if decision == "remap_multi_parent" and len(selected_parent_ids) < 2:
            raise ValueError(f"{path}:{line_number} remap_multi_parent requires 2+ parents")
        decisions.append(
            {
                "sample_id": sample_id,
                "decision": decision,
                "confidence": str(row.get("confidence") or ""),
                "human_required": bool(row.get("human_required")),
                "selected_parent_chunk_ids": selected_parent_ids,
                "selected_stable_block_ids": selected_block_ids,
                "rubric_change_needed": bool(row.get("rubric_change_needed")),
                "reason": str(row.get("reason") or ""),
                "notes": str(row.get("notes") or ""),
            }
        )
    return decisions


def apply_decisions_to_rows(
    *,
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = index_rows_by_sample_id(rows)
    changes = []
    for decision in decisions:
        sample_id = decision["sample_id"]
        if sample_id not in rows_by_id:
            raise ValueError(f"dataset missing sample_id: {sample_id}")
        row = rows_by_id[sample_id]
        selected_parent_ids = decision["selected_parent_chunk_ids"]
        selected_block_ids = decision["selected_stable_block_ids"]
        validate_selected_parent_blocks(
            sample_id=sample_id,
            selected_parent_ids=selected_parent_ids,
            selected_block_ids=selected_block_ids,
            parent_records=parent_records,
        )
        before = current_gold_state(row)
        already_current = (
            before["source_trace_chunk_ids"] == selected_parent_ids
            and before["source_trace_block_ids"] == selected_block_ids
            and before["stable_target_block_ids"] == selected_block_ids
            and before["target_chunk_id_candidate"] == selected_parent_ids[0]
        )
        if already_current:
            status = "already_applied"
        else:
            status = "applied"
            set_gold_state(row, selected_parent_ids, selected_block_ids)
        changes.append(
            {
                "sample_id": sample_id,
                "decision": decision["decision"],
                "status": status,
                "confidence": decision["confidence"],
                "human_required": decision["human_required"],
                "before": before,
                "after": current_gold_state(row),
                "reason": decision["reason"],
                "notes": decision["notes"],
            }
        )
    return rows, changes


def validate_selected_parent_blocks(
    *,
    sample_id: str,
    selected_parent_ids: list[str],
    selected_block_ids: list[str],
    parent_records: dict[str, dict[str, Any]],
) -> None:
    missing_parent_ids = [
        parent_id for parent_id in selected_parent_ids if parent_id not in parent_records
    ]
    if missing_parent_ids:
        raise ValueError(f"{sample_id} selected parents not found: {missing_parent_ids!r}")
    block_union = set()
    per_parent = {}
    for parent_id in selected_parent_ids:
        blocks = record_block_ids(parent_records[parent_id])
        per_parent[parent_id] = sorted(set(selected_block_ids) & blocks)
        block_union.update(blocks)
    missing_blocks = [block_id for block_id in selected_block_ids if block_id not in block_union]
    if missing_blocks:
        raise ValueError(
            f"{sample_id} selected parents {selected_parent_ids!r} miss stable blocks "
            f"{missing_blocks!r}"
        )
    empty_parent_ids = [
        parent_id for parent_id, hit_blocks in per_parent.items() if not hit_blocks
    ]
    if empty_parent_ids:
        raise ValueError(
            f"{sample_id} selected parents do not cover any selected stable block: "
            f"{empty_parent_ids!r}"
        )


def current_gold_state(row: dict[str, Any]) -> dict[str, Any]:
    source_trace = ((row.get("answer_rubric") or {}).get("source_trace") or {})
    return {
        "source_trace_chunk_ids": as_str_list(source_trace.get("chunk_ids")),
        "source_trace_block_ids": as_str_list(source_trace.get("block_ids")),
        "target_chunk_id_candidate": str(row.get("target_chunk_id_candidate") or ""),
        "stable_target_block_ids": as_str_list(row.get("stable_target_block_ids")),
    }


def set_gold_state(
    row: dict[str, Any],
    selected_parent_ids: list[str],
    selected_block_ids: list[str],
) -> None:
    answer_rubric = row.setdefault("answer_rubric", {})
    source_trace = answer_rubric.setdefault("source_trace", {})
    source_trace["chunk_ids"] = selected_parent_ids
    source_trace["block_ids"] = selected_block_ids
    row["target_chunk_id_candidate"] = selected_parent_ids[0]
    row["stable_target_block_ids"] = selected_block_ids


def build_summary(
    *,
    run_id: str,
    dataset_path: Path,
    decisions_path: Path,
    parent_chunks_path: Path,
    dry_run: bool,
    changes: list[dict[str, Any]],
) -> dict[str, Any]:
    result_dir = RESULTS_ROOT / f"v3_dataset_gold_consistency_remap_applied_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_dataset_gold_consistency_remap_applied_{run_id}"
    status_counts = Counter(change["status"] for change in changes)
    decision_counts = Counter(change["decision"] for change in changes)
    validation = {
        "passed": bool(changes)
        and all(change["status"] in {"applied", "already_applied"} for change in changes)
        and all(change["after"]["source_trace_chunk_ids"] for change in changes)
        and all(change["after"]["stable_target_block_ids"] for change in changes),
        "selected_parents_validated_against_parent_chunks": True,
        "selected_blocks_validated_against_parent_union": True,
        "multi_parent_sample_ids": [
            change["sample_id"]
            for change in changes
            if len(change["after"]["source_trace_chunk_ids"]) > 1
        ],
    }
    return {
        "run_id": run_id,
        "scope": "apply v3 dataset-level gold consistency remap decisions",
        "inputs": {
            "dataset": str(dataset_path),
            "decisions": str(decisions_path),
            "parent_chunks": str(parent_chunks_path),
        },
        "outputs": {
            "summary": str(result_dir / "summary.json"),
            "changes": str(result_dir / "changes.jsonl"),
            "report": str(report_dir / "report.md"),
        },
        "dry_run": dry_run,
        "change_count": len(changes),
        "status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "applied_sample_ids": [
            change["sample_id"] for change in changes if change["status"] == "applied"
        ],
        "already_applied_sample_ids": [
            change["sample_id"]
            for change in changes
            if change["status"] == "already_applied"
        ],
        "validation": validation,
    }


def render_report(summary: dict[str, Any], changes: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 dataset gold consistency remap 应用报告",
        "",
        "## 范围",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- dataset: `{summary['inputs']['dataset']}`",
        f"- decisions: `{summary['inputs']['decisions']}`",
        f"- dry_run: `{summary['dry_run']}`",
        f"- validation_passed: `{summary['validation']['passed']}`",
        (
            "- 口径：只修改 dataset gold 标注字段；不运行 eval/judge，"
            "不改 retrieval/rerank/score-floor/generation。"
        ),
        "",
        "## 变更",
        "",
        (
            "| sample_id | decision | status | old parent | new parent | "
            "old stable blocks | new stable blocks |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for change in changes:
        before = change["before"]
        after = change["after"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{change['sample_id']}`",
                    f"`{change['decision']}`",
                    f"`{change['status']}`",
                    format_code_list(
                        dedupe(
                            [
                                *before["source_trace_chunk_ids"],
                                before["target_chunk_id_candidate"],
                            ]
                        )
                    ),
                    format_code_list(after["source_trace_chunk_ids"]),
                    format_code_list(before["stable_target_block_ids"]),
                    format_code_list(after["stable_target_block_ids"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            (
                "- `target_chunk_id_candidate` 是单值字段；多 parent 样本使用"
                "第一个 selected parent，"
                "`answer_rubric.source_trace.chunk_ids` 保留完整 parent 列表。"
            ),
            "- 本次唯一多 parent 样本是 `v3_ra_006`。",
            "",
            "## 后续验证",
            "",
            "1. 重新运行 dataset-level consistency audit，确认这些样本不再被标为 mismatch。",
            "2. 如需评估指标变化，只做离线 no-judge 重算，不运行 GPU eval/judge。",
        ]
    )
    return "\n".join(lines) + "\n"


def index_rows_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_id = {}
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"dataset row {index} missing sample_id")
        if sample_id in rows_by_id:
            raise ValueError(f"dataset duplicate sample_id: {sample_id}")
        rows_by_id[sample_id] = row
    return rows_by_id


def load_parent_records(path: Path, parent_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id in parent_ids:
                records[chunk_id] = row
    return records


def record_block_ids(record: dict[str, Any]) -> set[str]:
    block_ids = set(as_str_list(record.get("source_block_ids") or record.get("block_ids")))
    metadata = record.get("source_block_metadata") or []
    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            block_ids.update(as_str_list(item.get("block_id")))
            block_ids.update(as_str_list(item.get("source_block_id")))
    return block_ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    temp_path.replace(path)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if str(value):
        return [str(value)]
    return []


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def format_code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def run_self_test() -> None:
    parent_records = {
        "doc_a_sec01_chunk01": {
            "chunk_id": "doc_a_sec01_chunk01",
            "source_block_ids": ["b1"],
        },
        "doc_a_sec02_chunk02": {
            "chunk_id": "doc_a_sec02_chunk02",
            "source_block_ids": ["b2"],
        },
    }
    rows = [
        {
            "sample_id": "case_one",
            "answer_rubric": {
                "source_trace": {
                    "chunk_ids": ["old_parent"],
                    "block_ids": ["old_block"],
                }
            },
            "target_chunk_id_candidate": "old_parent",
            "stable_target_block_ids": ["old_block"],
        }
    ]
    decisions = [
        {
            "sample_id": "case_one",
            "decision": "remap_multi_parent",
            "confidence": "high",
            "human_required": False,
            "selected_parent_chunk_ids": ["doc_a_sec01_chunk01", "doc_a_sec02_chunk02"],
            "selected_stable_block_ids": ["b1", "b2"],
            "reason": "test",
            "notes": "",
        }
    ]
    updated, changes = apply_decisions_to_rows(
        rows=rows,
        decisions=decisions,
        parent_records=parent_records,
    )
    state = current_gold_state(updated[0])
    assert changes[0]["status"] == "applied"
    assert state["source_trace_chunk_ids"] == ["doc_a_sec01_chunk01", "doc_a_sec02_chunk02"]
    assert state["source_trace_block_ids"] == ["b1", "b2"]
    assert state["target_chunk_id_candidate"] == "doc_a_sec01_chunk01"
    assert state["stable_target_block_ids"] == ["b1", "b2"]


if __name__ == "__main__":
    main()
