from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_P2_SAMPLES = (
    RESULTS_ROOT / "v3_p2_raw_retrieval_same_doc_wrong_parent_audit_20260524" / "samples.jsonl"
)
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
TARGET_CLASS = "gold_stable_block_mismatch_same_doc_candidate"
DECISION_OPTIONS = ("remap", "keep_gold_fix_blocks", "add_alternate_gold", "manual_unclear")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build review pack for P2 gold/stable-block mismatch samples."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--p2-samples", default=str(DEFAULT_P2_SAMPLES))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    p2_samples_path = Path(args.p2_samples)
    dataset_path = Path(args.dataset)
    parent_chunks_path = Path(args.parent_chunks)
    p2_rows = [
        row
        for row in load_jsonl(p2_samples_path)
        if row.get("primary_classification") == TARGET_CLASS
    ]
    dataset_rows = load_jsonl_by_id(dataset_path, "sample_id")
    parent_ids = collect_parent_ids(p2_rows)
    parent_records = load_parent_records(parent_chunks_path, parent_ids)
    review_rows = [
        build_review_row(
            p2_row=row,
            dataset_row=dataset_rows[str(row["sample_id"])],
            parent_records=parent_records,
        )
        for row in p2_rows
    ]
    summary = build_summary(
        args.run_id,
        p2_samples_path,
        dataset_path,
        parent_chunks_path,
        review_rows,
    )

    result_dir = RESULTS_ROOT / f"v3_p2_gold_stable_block_review_pack_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_p2_gold_stable_block_review_pack_{args.run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "review_samples.jsonl", review_rows)
    write_decision_template(result_dir / "decision_ledger_template.csv", review_rows)
    write_markdown(report_dir / "review_pack.md", render_review_pack(summary, review_rows))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "review_sample_count": len(review_rows),
                "decision_options": list(DECISION_OPTIONS),
            },
            ensure_ascii=False,
        )
    )


def build_review_row(
    *,
    p2_row: dict[str, Any],
    dataset_row: dict[str, Any],
    parent_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(p2_row.get("sample_id") or "")
    rubric = dataset_row.get("answer_rubric") or {}
    source_trace = rubric.get("source_trace") or {}
    stable_blocks = as_str_list(dataset_row.get("stable_target_block_ids"))
    current_gold_ids = as_str_list(p2_row.get("gold_parent_chunk_ids"))
    candidate_ids = [
        str(item.get("chunk_id") or "")
        for item in p2_row.get("stable_block_parent_candidates") or []
        if item.get("covers_all_stable_blocks")
    ]
    return {
        "sample_id": sample_id,
        "question": str(dataset_row.get("question") or p2_row.get("question") or ""),
        "expected_answer": str(dataset_row.get("expected_answer") or ""),
        "category": str(dataset_row.get("category") or p2_row.get("category") or ""),
        "expected_route": str(
            dataset_row.get("expected_route") or p2_row.get("expected_route") or ""
        ),
        "expected_doc_ids": as_str_list(dataset_row.get("expected_doc_ids")),
        "stable_target_block_ids": stable_blocks,
        "rubric": {
            "evidence_note": str(rubric.get("evidence_note") or ""),
            "must_include": as_str_list(rubric.get("must_include")),
            "acceptable_variants": as_str_list(rubric.get("acceptable_variants")),
            "reject_if": as_str_list(rubric.get("reject_if")),
            "source_trace_chunk_ids": as_str_list(source_trace.get("chunk_ids")),
            "source_trace_block_ids": as_str_list(source_trace.get("block_ids")),
        },
        "current_gold_parents": [
            build_parent_card(parent_records.get(parent_id) or {}, stable_blocks)
            for parent_id in current_gold_ids
        ],
        "full_cover_candidates": [
            build_parent_card(parent_records.get(parent_id) or {}, stable_blocks)
            for parent_id in candidate_ids
        ],
        "retrieval_what_if": p2_row.get("stable_block_full_cover_what_if") or {},
        "review_decision_template": {
            "decision": "",
            "selected_parent_chunk_ids": "",
            "selected_stable_block_ids": "",
            "reason": "",
            "rubric_change_needed": "",
            "notes": "",
        },
    }


def build_parent_card(record: dict[str, Any], stable_blocks: list[str]) -> dict[str, Any]:
    if not record:
        return {
            "chunk_id": "",
            "exists": False,
            "stable_block_hits": [],
            "stable_block_previews": [],
            "text_preview": "",
        }
    stable_set = set(stable_blocks)
    block_metadata = as_block_metadata(record.get("source_block_metadata"))
    stable_previews = [
        compact_block(item)
        for item in block_metadata
        if block_id(item) in stable_set or source_block_id(item) in stable_set
    ]
    record_block_ids = as_str_list(record.get("source_block_ids") or record.get("block_ids"))
    stable_hits = [item for item in record_block_ids if item in stable_set]
    return {
        "chunk_id": str(record.get("chunk_id") or ""),
        "exists": True,
        "doc_id": str(record.get("doc_id") or ""),
        "source_file": str(record.get("source_file") or ""),
        "section": str(record.get("section") or ""),
        "page_numbers": record.get("page_numbers") or [],
        "contains_table_caption": bool(record.get("contains_table_caption")),
        "contains_table_text": bool(record.get("contains_table_text")),
        "contains_figure_caption": bool(record.get("contains_figure_caption")),
        "block_count": len(record_block_ids),
        "stable_block_hits": stable_hits,
        "stable_block_previews": stable_previews,
        "text_preview": compact_text(
            record.get("retrieval_text") or record.get("text"),
            limit=2200,
        ),
    }


def render_review_pack(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 P2 gold/stable-block 不一致样本复核包",
        "",
        "## 复核步骤",
        "",
        (
            "1. 先读 `question`、`expected_answer`、`must_include` 和 `reject_if`，"
            "明确答案必须由哪些证据支撑。"
        ),
        "2. 对比 `stable_target_block_ids` 在当前 gold parent 与候选 parent 中的命中情况。",
        (
            "3. 优先查看 `stable block previews`：这些是标注指向的块级证据，"
            "比整段 chunk preview 更关键。"
        ),
        (
            "4. 判断候选 parent 是否直接支撑 expected answer；"
            "同时判断当前 gold parent 是否也能独立支撑答案。"
        ),
        (
            "5. 按决策项填写台账：`remap`、`keep_gold_fix_blocks`、"
            "`add_alternate_gold` 或 `manual_unclear`。"
        ),
        "",
        "## 决策含义",
        "",
        "- `remap`：候选 parent 覆盖 stable blocks 且内容支撑答案，当前 gold 不支撑或明显漂移。",
        "- `keep_gold_fix_blocks`：当前 gold 支撑答案，但 stable blocks 指错了，应修 block ids。",
        "- `add_alternate_gold`：当前 gold 和候选 parent 都能支撑答案，应保留/补充多 gold。",
        "- `manual_unclear`：候选和当前 gold 都不足，或 expected answer/rubric 本身需要人工重写。",
        "",
        "## 汇总",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- review_sample_count: {summary['review_sample_count']}",
        f"- decision ledger: `{summary['outputs']['decision_ledger_template']}`",
        "",
    ]
    for row in rows:
        lines.extend(render_sample_section(row))
    return "\n".join(lines) + "\n"


def render_sample_section(row: dict[str, Any]) -> list[str]:
    lines = [
        f"## `{row['sample_id']}`",
        "",
        f"- category/route: `{row['category']}` / `{row['expected_route']}`",
        f"- expected_doc_ids: {format_code_list(row['expected_doc_ids'])}",
        f"- stable_target_block_ids: {format_code_list(row['stable_target_block_ids'])}",
        f"- question: {row['question']}",
        f"- expected_answer: {row['expected_answer']}",
        f"- evidence_note: {row['rubric']['evidence_note']}",
        f"- must_include: {format_plain_list(row['rubric']['must_include'])}",
        f"- reject_if: {format_plain_list(row['rubric']['reject_if'])}",
        "",
        "### 当前 gold parent",
        "",
    ]
    for card in row["current_gold_parents"]:
        lines.extend(render_parent_card(card))
    lines.extend(["", "### stable-block full-cover 候选", ""])
    for card in row["full_cover_candidates"]:
        lines.extend(render_parent_card(card))
    what_if = row["retrieval_what_if"]
    lines.extend(
        [
            "",
            "### 命中状态",
            "",
            f"- candidate parent hit@10: `{what_if.get('parent_hit_at10')}`",
            f"- candidate support hit: `{what_if.get('support_parent_chunk_hit')}`",
            f"- candidate citation hit: `{what_if.get('citation_parent_chunk_hit')}`",
            f"- top10/support/citation rank: `{what_if.get('top10_rank')}` / "
            f"`{what_if.get('support_rank')}` / `{what_if.get('citation_rank')}`",
            "",
            "### 需要人工填写",
            "",
            f"- decision: one of {format_code_list(list(DECISION_OPTIONS))}",
            "- selected_parent_chunk_ids:",
            "- selected_stable_block_ids:",
            "- reason:",
            "- rubric_change_needed:",
            "- notes:",
            "",
        ]
    )
    return lines


def render_parent_card(card: dict[str, Any]) -> list[str]:
    if not card.get("exists"):
        return ["- missing parent record", ""]
    lines = [
        f"#### `{card['chunk_id']}`",
        "",
        f"- section: `{card['section']}`",
        f"- source_file/pages: `{card['source_file']}` / `{card['page_numbers']}`",
        (
            "- flags: "
            f"table_caption={card['contains_table_caption']}, "
            f"table_text={card['contains_table_text']}, "
            f"figure_caption={card['contains_figure_caption']}"
        ),
        f"- stable_block_hits: {format_code_list(card['stable_block_hits'])}",
        "",
        "stable block previews:",
    ]
    if card["stable_block_previews"]:
        for item in card["stable_block_previews"]:
            lines.append(
                f"- `{item['block_id']}` `{item['type']}` page={item['page']}: "
                f"{item['text_preview']}"
            )
    else:
        lines.append("- 无 stable block 命中。")
    lines.extend(["", "chunk preview:", "", card["text_preview"], ""])
    return lines


def build_summary(
    run_id: str,
    p2_samples_path: Path,
    dataset_path: Path,
    parent_chunks_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result_dir = RESULTS_ROOT / f"v3_p2_gold_stable_block_review_pack_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_p2_gold_stable_block_review_pack_{run_id}"
    return {
        "run_id": run_id,
        "scope": "manual review pack for P2 gold/stable-block mismatch samples",
        "inputs": {
            "p2_samples": str(p2_samples_path),
            "dataset": str(dataset_path),
            "parent_chunks": str(parent_chunks_path),
        },
        "outputs": {
            "summary": str(result_dir / "summary.json"),
            "review_samples": str(result_dir / "review_samples.jsonl"),
            "decision_ledger_template": str(result_dir / "decision_ledger_template.csv"),
            "review_pack": str(report_dir / "review_pack.md"),
        },
        "review_sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "decision_options": list(DECISION_OPTIONS),
    }


def collect_parent_ids(rows: list[dict[str, Any]]) -> set[str]:
    parent_ids = set()
    for row in rows:
        parent_ids.update(as_str_list(row.get("gold_parent_chunk_ids")))
        for candidate in row.get("stable_block_parent_candidates") or []:
            if candidate.get("covers_all_stable_blocks"):
                parent_ids.add(str(candidate.get("chunk_id") or ""))
    return parent_ids


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


def as_block_metadata(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def compact_block(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block_id(item) or source_block_id(item),
        "source_block_id": source_block_id(item),
        "type": str(item.get("type") or ""),
        "page": item.get("page"),
        "section_path": item.get("section_path") or [],
        "text_preview": compact_text(item.get("text_preview"), limit=520),
    }


def block_id(item: dict[str, Any]) -> str:
    return str(item.get("block_id") or "")


def source_block_id(item: dict[str, Any]) -> str:
    return str(item.get("source_block_id") or "")


def compact_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def format_code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def format_plain_list(values: list[str]) -> str:
    if not values:
        return "-"
    return "; ".join(values)


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if str(value):
        return [str(value)]
    return []


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for line_number, row in enumerate(load_jsonl(path), start=1):
        row_id = str(row.get(key) or "")
        if not row_id:
            raise ValueError(f"{path}:{line_number} missing key {key}")
        rows[row_id] = row
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_decision_template(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "old_parent_chunk_ids",
        "candidate_parent_chunk_ids",
        "decision",
        "selected_parent_chunk_ids",
        "selected_stable_block_ids",
        "reason",
        "rubric_change_needed",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "old_parent_chunk_ids": ";".join(
                        card["chunk_id"] for card in row["current_gold_parents"]
                    ),
                    "candidate_parent_chunk_ids": ";".join(
                        card["chunk_id"] for card in row["full_cover_candidates"]
                    ),
                    "decision": "",
                    "selected_parent_chunk_ids": "",
                    "selected_stable_block_ids": "",
                    "reason": "",
                    "rubric_change_needed": "",
                    "notes": "",
                }
            )


def run_self_test() -> None:
    record = {
        "chunk_id": "doc_a_sec01_chunk01",
        "doc_id": "doc_a",
        "section": "Methods",
        "source_file": "doc_a.pdf",
        "source_block_ids": ["b1", "b2"],
        "source_block_metadata": [
            {"block_id": "b1", "type": "table_caption", "page": 1, "text_preview": "Table 1"},
            {"block_id": "b2", "type": "paragraph", "page": 1, "text_preview": "Other"},
        ],
        "retrieval_text": "full text",
    }
    card = build_parent_card(record, ["b1"])
    assert card["stable_block_hits"] == ["b1"]
    assert card["stable_block_previews"][0]["block_id"] == "b1"
    assert "Table 1" in "\n".join(render_parent_card(card))
    print("self-test passed")


if __name__ == "__main__":
    main()
