from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_LEDGER = (
    RESULTS_ROOT
    / "v3_p2_gold_stable_block_review_pack_20260524"
    / "decision_ledger_template.csv"
)
DECISION_OPTIONS = {"remap", "keep_gold_fix_blocks", "add_alternate_gold", "manual_unclear"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply reviewed P2 gold remap decisions to the v3 baseline dataset."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    dataset_path = Path(args.dataset)
    ledger_path = Path(args.ledger)
    parent_chunks_path = Path(args.parent_chunks)
    decisions = load_decisions(ledger_path)
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
        run_id=str(args.run_id),
        dataset_path=dataset_path,
        ledger_path=ledger_path,
        parent_chunks_path=parent_chunks_path,
        dry_run=bool(args.dry_run),
        changes=changes,
    )

    result_dir = RESULTS_ROOT / f"v3_p2_gold_remap_decisions_applied_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_p2_gold_remap_decisions_applied_{args.run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "changes.jsonl", changes)
    write_markdown(report_dir / "report.md", render_report(summary, changes))

    if not args.dry_run:
        write_jsonl(dataset_path, updated_rows)

    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "dry_run": bool(args.dry_run),
                "change_count": len(changes),
                "applied_change_count": sum(1 for item in changes if item["status"] == "applied"),
                "already_applied_count": sum(
                    1 for item in changes if item["status"] == "already_applied"
                ),
            },
            ensure_ascii=False,
        )
    )


def load_decisions(path: Path) -> list[dict[str, Any]]:
    decisions = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            sample_id = str(row.get("sample_id") or "").strip()
            decision = str(row.get("decision") or "").strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_number} missing sample_id")
            if decision not in DECISION_OPTIONS:
                raise ValueError(f"{path}:{line_number} invalid decision: {decision!r}")
            if decision != "remap":
                raise ValueError(
                    f"{path}:{line_number} only remap decisions are supported by this script"
                )
            selected_parent_ids = split_ids(row.get("selected_parent_chunk_ids"))
            selected_block_ids = split_ids(row.get("selected_stable_block_ids"))
            old_parent_ids = split_ids(row.get("old_parent_chunk_ids"))
            candidate_parent_ids = split_ids(row.get("candidate_parent_chunk_ids"))
            if len(selected_parent_ids) != 1:
                raise ValueError(
                    f"{path}:{line_number} remap requires exactly one selected parent chunk id"
                )
            if not selected_block_ids:
                raise ValueError(f"{path}:{line_number} remap requires selected stable block ids")
            decisions.append(
                {
                    "sample_id": sample_id,
                    "decision": decision,
                    "old_parent_chunk_ids": old_parent_ids,
                    "candidate_parent_chunk_ids": candidate_parent_ids,
                    "selected_parent_chunk_ids": selected_parent_ids,
                    "selected_stable_block_ids": selected_block_ids,
                    "reason": str(row.get("reason") or "").strip(),
                    "rubric_change_needed": str(row.get("rubric_change_needed") or "").strip(),
                    "notes": str(row.get("notes") or "").strip(),
                }
            )
    return decisions


def apply_decisions_to_rows(
    *,
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = {str(row.get("sample_id") or ""): row for row in rows}
    changes = []
    for decision in decisions:
        sample_id = decision["sample_id"]
        if sample_id not in rows_by_id:
            raise ValueError(f"dataset missing sample_id: {sample_id}")
        row = rows_by_id[sample_id]
        selected_parent_ids = decision["selected_parent_chunk_ids"]
        selected_block_ids = decision["selected_stable_block_ids"]
        selected_parent_id = selected_parent_ids[0]
        validate_parent_blocks(
            sample_id=sample_id,
            selected_parent_id=selected_parent_id,
            selected_block_ids=selected_block_ids,
            parent_records=parent_records,
        )
        before = current_gold_state(row)
        before_ids = dedupe(
            [*before["source_trace_chunk_ids"], before["target_chunk_id_candidate"]]
        )
        if before_ids != decision["old_parent_chunk_ids"] and before_ids != selected_parent_ids:
            raise ValueError(
                f"{sample_id} current gold ids {before_ids!r} do not match ledger old ids "
                f"{decision['old_parent_chunk_ids']!r}"
            )
        if selected_parent_id not in decision["candidate_parent_chunk_ids"]:
            raise ValueError(
                f"{sample_id} selected parent {selected_parent_id!r} is not in candidate ids"
            )
        already_current = (
            before_ids == selected_parent_ids
            and before["stable_target_block_ids"] == selected_block_ids
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
                "before": before,
                "after": current_gold_state(row),
                "reason": decision["reason"],
                "rubric_change_needed": decision["rubric_change_needed"],
                "notes": decision["notes"],
            }
        )
    return rows, changes


def validate_parent_blocks(
    *,
    sample_id: str,
    selected_parent_id: str,
    selected_block_ids: list[str],
    parent_records: dict[str, dict[str, Any]],
) -> None:
    record = parent_records.get(selected_parent_id)
    if not record:
        raise ValueError(f"{sample_id} selected parent not found: {selected_parent_id}")
    record_blocks = record_block_ids(record)
    missing_blocks = [block_id for block_id in selected_block_ids if block_id not in record_blocks]
    if missing_blocks:
        raise ValueError(
            f"{sample_id} selected parent {selected_parent_id} misses stable blocks "
            f"{missing_blocks!r}"
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
    row: dict[str, Any], selected_parent_ids: list[str], selected_block_ids: list[str]
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
    ledger_path: Path,
    parent_chunks_path: Path,
    dry_run: bool,
    changes: list[dict[str, Any]],
) -> dict[str, Any]:
    applied = [item["sample_id"] for item in changes if item["status"] == "applied"]
    already_applied = [
        item["sample_id"] for item in changes if item["status"] == "already_applied"
    ]
    result_dir = RESULTS_ROOT / f"v3_p2_gold_remap_decisions_applied_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_p2_gold_remap_decisions_applied_{run_id}"
    return {
        "run_id": run_id,
        "scope": "apply reviewed P2 gold remap decisions to v3 baseline dataset",
        "inputs": {
            "dataset": str(dataset_path),
            "ledger": str(ledger_path),
            "parent_chunks": str(parent_chunks_path),
        },
        "outputs": {
            "summary": str(result_dir / "summary.json"),
            "changes": str(result_dir / "changes.jsonl"),
            "report": str(report_dir / "report.md"),
        },
        "dry_run": dry_run,
        "change_count": len(changes),
        "applied_sample_ids": applied,
        "already_applied_sample_ids": already_applied,
        "validation": {
            "passed": len(changes) > 0 and all(
                item["status"] in {"applied", "already_applied"} for item in changes
            ),
            "all_decisions_are_remap": True,
            "selected_parents_validated_against_parent_chunks": True,
        },
    }


def render_report(summary: dict[str, Any], changes: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 P2 gold remap 决策应用报告",
        "",
        "## 范围",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- dataset: `{summary['inputs']['dataset']}`",
        f"- ledger: `{summary['inputs']['ledger']}`",
        f"- dry_run: `{summary['dry_run']}`",
        f"- validation_passed: `{summary['validation']['passed']}`",
        "",
        "## 变更",
        "",
        (
            "| sample_id | status | old parent | new parent | old stable blocks | "
            "new stable blocks |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for item in changes:
        before = item["before"]
        after = item["after"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['sample_id']}`",
                    f"`{item['status']}`",
                    format_code_list(
                        dedupe(
                            [
                                *before["source_trace_chunk_ids"],
                                before["target_chunk_id_candidate"],
                            ]
                        )
                    ),
                    format_code_list(
                        dedupe(
                            [
                                *after["source_trace_chunk_ids"],
                                after["target_chunk_id_candidate"],
                            ]
                        )
                    ),
                    format_code_list(before["stable_target_block_ids"]),
                    format_code_list(after["stable_target_block_ids"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 后续验证",
            "",
            "1. 重新运行 P0 offline rule metrics，确认全量 bucket 变化。",
            "2. 重新运行 P1/P2 offline audit，确认 raw_parent_absent 是否只剩真实检索缺口。",
            "3. 不运行 eval/judge，不改 retrieval/rerank/score floor/generation。",
        ]
    )
    return "\n".join(lines) + "\n"


def split_ids(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


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


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if str(value):
        return [str(value)]
    return []


def format_code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_self_test() -> None:
    assert split_ids("a; b,c") == ["a", "b", "c"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        parent_path = temp / "parents.jsonl"
        ledger_path = temp / "ledger.csv"
        parent = {
            "chunk_id": "doc_a_sec02_chunk03",
            "source_block_ids": ["b1", "b2"],
            "source_block_metadata": [{"block_id": "b3"}],
        }
        write_jsonl(parent_path, [parent])
        with ledger_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "old_parent_chunk_ids",
                    "candidate_parent_chunk_ids",
                    "decision",
                    "selected_parent_chunk_ids",
                    "selected_stable_block_ids",
                    "reason",
                    "rubric_change_needed",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "s1",
                    "old_parent_chunk_ids": "doc_a_sec01_chunk02",
                    "candidate_parent_chunk_ids": "doc_a_sec02_chunk03",
                    "decision": "remap",
                    "selected_parent_chunk_ids": "doc_a_sec02_chunk03",
                    "selected_stable_block_ids": "b1;b3",
                    "reason": "test",
                    "rubric_change_needed": "no",
                    "notes": "",
                }
            )
        rows = [
            {
                "sample_id": "s1",
                "answer_rubric": {
                    "source_trace": {
                        "chunk_ids": ["doc_a_sec01_chunk02"],
                        "block_ids": ["old_b"],
                    }
                },
                "target_chunk_id_candidate": "doc_a_sec01_chunk02",
                "stable_target_block_ids": ["old_b"],
            }
        ]
        decisions = load_decisions(ledger_path)
        parent_records = load_parent_records(parent_path, {"doc_a_sec02_chunk03"})
        updated_rows, changes = apply_decisions_to_rows(
            rows=rows,
            decisions=decisions,
            parent_records=parent_records,
        )
        state = current_gold_state(updated_rows[0])
        assert state["source_trace_chunk_ids"] == ["doc_a_sec02_chunk03"]
        assert state["source_trace_block_ids"] == ["b1", "b3"]
        assert state["target_chunk_id_candidate"] == "doc_a_sec02_chunk03"
        assert state["stable_target_block_ids"] == ["b1", "b3"]
        assert changes[0]["status"] == "applied"


if __name__ == "__main__":
    main()
