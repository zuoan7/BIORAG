from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_MANIFEST = Path("data/eval/manifests/v3_baseline_dataset_manifest.json")
DEFAULT_PATCH_CANDIDATES = (
    RESULTS_ROOT
    / "v3_dataset_design_review_min_manual_20260524"
    / "patch_candidates.jsonl"
)
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
APPROVED_SAMPLE_IDS = {"v3_pc_147"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply human-approved v3 dataset design-review patch candidates."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--patch-candidates", default=str(DEFAULT_PATCH_CANDIDATES))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = str(args.run_id)
    dataset_path = Path(args.dataset)
    manifest_path = Path(args.manifest)
    patch_candidates_path = Path(args.patch_candidates)
    parent_chunks_path = Path(args.parent_chunks)

    rows = load_jsonl(dataset_path)
    manifest = load_json(manifest_path)
    parent_records = index_by_key(load_jsonl(parent_chunks_path), "chunk_id", parent_chunks_path)
    candidates = [
        row
        for row in load_jsonl(patch_candidates_path)
        if str(row.get("sample_id") or "") in APPROVED_SAMPLE_IDS
    ]
    updated_rows, changes = apply_candidates(rows, candidates, parent_records)
    updated_manifest = update_manifest(manifest, dataset_path, updated_rows, dry_run=bool(args.dry_run))
    summary = build_summary(
        run_id=run_id,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        patch_candidates_path=patch_candidates_path,
        parent_chunks_path=parent_chunks_path,
        dry_run=bool(args.dry_run),
        changes=changes,
        updated_manifest=updated_manifest,
    )

    result_dir = RESULTS_ROOT / f"v3_dataset_design_review_patch_applied_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_dataset_design_review_patch_applied_{run_id}"
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "changes.jsonl", changes)
    write_markdown(report_dir / "report.md", render_report(summary, changes))

    if not args.dry_run:
        write_jsonl_atomic(dataset_path, updated_rows)
        write_json(manifest_path, updated_manifest)

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


def apply_candidates(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = index_by_key(rows, "sample_id", Path("dataset rows"))
    changes = []
    if not candidates:
        raise ValueError("No approved patch candidates found")
    for candidate in candidates:
        sample_id = str(candidate.get("sample_id") or "")
        if sample_id not in APPROVED_SAMPLE_IDS:
            raise ValueError(f"Sample is not approved for patch: {sample_id}")
        if sample_id not in rows_by_id:
            raise ValueError(f"Dataset missing sample_id: {sample_id}")
        validate_candidate(candidate, parent_records)
        row = rows_by_id[sample_id]
        before = extract_patch_state(row)
        apply_candidate(row, candidate, parent_records)
        after = extract_patch_state(row)
        changes.append(
            {
                "sample_id": sample_id,
                "patch_type": str(candidate.get("patch_type") or ""),
                "status": "already_applied" if before == after else "applied",
                "before": before,
                "after": after,
            }
        )
    return rows, changes


def validate_candidate(candidate: dict[str, Any], parent_records: dict[str, dict[str, Any]]) -> None:
    sample_id = str(candidate.get("sample_id") or "")
    chunk_ids = as_str_list(candidate.get("new_source_trace_chunk_ids"))
    block_ids = as_str_list(candidate.get("new_stable_target_block_ids"))
    if not chunk_ids:
        raise ValueError(f"{sample_id} patch candidate has no chunk ids")
    if not block_ids:
        raise ValueError(f"{sample_id} patch candidate has no stable block ids")
    missing_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in parent_records]
    if missing_chunks:
        raise ValueError(f"{sample_id} patch candidate missing parent chunks: {missing_chunks}")
    available_blocks = []
    for chunk_id in chunk_ids:
        available_blocks.extend(as_str_list(parent_records[chunk_id].get("block_ids")))
        available_blocks.extend(as_str_list(parent_records[chunk_id].get("source_block_ids")))
    missing_blocks = [block_id for block_id in block_ids if block_id not in available_blocks]
    if missing_blocks:
        raise ValueError(f"{sample_id} patch candidate missing block ids: {missing_blocks}")


def apply_candidate(
    row: dict[str, Any],
    candidate: dict[str, Any],
    parent_records: dict[str, dict[str, Any]],
) -> None:
    chunk_ids = as_str_list(candidate["new_source_trace_chunk_ids"])
    doc_ids = as_str_list(candidate["new_expected_doc_ids"])
    block_ids = as_str_list(candidate["new_source_trace_block_ids"])
    stable_block_ids = as_str_list(candidate["new_stable_target_block_ids"])
    sections = as_str_list(candidate["new_expected_sections"])

    row["question"] = str(candidate["new_question"])
    row["expected_route"] = str(candidate["new_expected_route"])
    row["category"] = str(candidate["new_category"])
    row["expected_doc_ids"] = doc_ids
    row["expected_source_files"] = [f"{doc_id}.pdf" for doc_id in doc_ids]
    row["expected_sections"] = sections
    row["expected_answer"] = str(candidate["new_expected_answer"])
    row["stable_target_block_ids"] = stable_block_ids

    rubric = row.setdefault("answer_rubric", {})
    source_trace = rubric.setdefault("source_trace", {})
    source_trace["doc_ids"] = doc_ids
    source_trace["source_files"] = [f"{doc_id}.pdf" for doc_id in doc_ids]
    source_trace["chunk_ids"] = chunk_ids
    source_trace["block_ids"] = block_ids
    rubric["evidence_note"] = str(candidate["new_evidence_note"])
    rubric["must_include"] = as_str_list(candidate["new_must_include"])
    rubric["reject_if"] = as_str_list(candidate["new_reject_if"])

    metadata = row.setdefault("metadata", {})
    metadata["anchor_doc_ids"] = doc_ids
    metadata["anchor_source_files"] = [f"{doc_id}.pdf" for doc_id in doc_ids]
    metadata["anchor_sections"] = [
        str(parent_records[chunk_id].get("section") or "") for chunk_id in chunk_ids
    ]
    metadata["anchor_chunk_ids"] = chunk_ids
    metadata["anchor_block_ids"] = block_ids
    metadata["construction_source_chunk_ids"] = chunk_ids
    metadata["construction_source_doc_ids"] = doc_ids
    metadata["rationale"] = (
        "Human-approved design review rewrite: doc_0303 now supplies the NeuAc "
        "intermediate-accumulation bottleneck anchor, and doc_0514 supplies the "
        "LNnT/LNT UDP-galactose competition-deletion anchor."
    )
    metadata["answer_rubric_status"] = "drafted_from_source"
    metadata["design_review_patch"] = {
        "run_id": DEFAULT_RUN_ID,
        "patch_type": str(candidate.get("patch_type") or ""),
        "source_patch_candidate": str(DEFAULT_PATCH_CANDIDATES),
        "human_approved": True,
    }


def extract_patch_state(row: dict[str, Any]) -> dict[str, Any]:
    rubric = row.get("answer_rubric") if isinstance(row.get("answer_rubric"), dict) else {}
    source_trace = (
        rubric.get("source_trace") if isinstance(rubric.get("source_trace"), dict) else {}
    )
    return {
        "question": row.get("question", ""),
        "expected_route": row.get("expected_route", ""),
        "category": row.get("category", ""),
        "expected_doc_ids": as_str_list(row.get("expected_doc_ids")),
        "expected_sections": as_str_list(row.get("expected_sections")),
        "expected_answer": row.get("expected_answer", ""),
        "source_trace_chunk_ids": as_str_list(source_trace.get("chunk_ids")),
        "source_trace_block_ids": as_str_list(source_trace.get("block_ids")),
        "stable_target_block_ids": as_str_list(row.get("stable_target_block_ids")),
        "evidence_note": rubric.get("evidence_note", ""),
        "must_include": as_str_list(rubric.get("must_include")),
        "reject_if": as_str_list(rubric.get("reject_if")),
    }


def update_manifest(
    manifest: dict[str, Any],
    dataset_path: Path,
    updated_rows: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    updated = dict(manifest)
    dataset_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in updated_rows
    )
    route_counts = Counter(str(row.get("expected_route") or "") for row in updated_rows)
    category_counts = Counter(str(row.get("category") or "") for row in updated_rows)
    updated["sample_count"] = len(updated_rows)
    updated["unique_sample_id_count"] = len({str(row.get("sample_id") or "") for row in updated_rows})
    updated["route_distribution"] = dict(sorted(route_counts.items()))
    updated["category_distribution"] = dict(sorted(category_counts.items()))
    updated["canonical_hash"] = hashlib.sha256(dataset_text.encode("utf-8")).hexdigest()
    if not dry_run:
        updated["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    updated["notes"] = (
        "v3 新基线候选数据集：200 条均已按原文证据补写 expected_answer 和 answer_rubric；"
        "已按 2026-05-24 设计复查确认修正 v3_pc_147 comparison anchor。"
    )
    if updated.get("canonical_path") != str(dataset_path):
        raise ValueError(
            f"Manifest canonical_path={updated.get('canonical_path')!r} "
            f"does not match dataset path={str(dataset_path)!r}"
        )
    return updated


def build_summary(
    *,
    run_id: str,
    dataset_path: Path,
    manifest_path: Path,
    patch_candidates_path: Path,
    parent_chunks_path: Path,
    dry_run: bool,
    changes: list[dict[str, Any]],
    updated_manifest: dict[str, Any],
) -> dict[str, Any]:
    result_dir = RESULTS_ROOT / f"v3_dataset_design_review_patch_applied_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_dataset_design_review_patch_applied_{run_id}"
    status_counts = Counter(change["status"] for change in changes)
    validation = {
        "passed": bool(changes)
        and all(change["status"] in {"applied", "already_applied"} for change in changes)
        and [change["sample_id"] for change in changes] == ["v3_pc_147"]
        and updated_manifest.get("sample_count") == 200
        and updated_manifest.get("unique_sample_id_count") == 200,
        "dataset_sample_count_200": updated_manifest.get("sample_count") == 200,
        "dataset_unique_sample_id_count_200": updated_manifest.get("unique_sample_id_count") == 200,
        "only_v3_pc_147_changed": [change["sample_id"] for change in changes] == ["v3_pc_147"],
        "manifest_hash_recomputed": bool(updated_manifest.get("canonical_hash")),
    }
    return {
        "run_id": run_id,
        "scope": "apply human-approved v3_pc_147 design-review rewrite only",
        "inputs": {
            "dataset": str(dataset_path),
            "manifest": str(manifest_path),
            "patch_candidates": str(patch_candidates_path),
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
        "applied_sample_ids": [
            change["sample_id"] for change in changes if change["status"] == "applied"
        ],
        "already_applied_sample_ids": [
            change["sample_id"] for change in changes if change["status"] == "already_applied"
        ],
        "validation": validation,
    }


def render_report(summary: dict[str, Any], changes: list[dict[str, Any]]) -> str:
    lines = [
        "# v3 dataset design review patch 应用报告",
        "",
        f"- run_id：`{summary['run_id']}`",
        f"- dry_run：`{summary['dry_run']}`",
        f"- validation：`{summary['validation']['passed']}`",
        "- 范围：只应用人工确认的 `v3_pc_147` rewrite patch candidate。",
        "- 未修改 retrieval/rerank/generation 代码，未运行 GPU eval，未调用 judge。",
        "",
        "## 变更",
        "",
    ]
    for change in changes:
        lines.extend(
            [
                f"### `{change['sample_id']}`",
                "",
                f"- status：`{change['status']}`",
                f"- patch_type：`{change['patch_type']}`",
                f"- old_question：{change['before']['question']}",
                f"- new_question：{change['after']['question']}",
                f"- old_chunks：`{change['before']['source_trace_chunk_ids']}`",
                f"- new_chunks：`{change['after']['source_trace_chunk_ids']}`",
                f"- old_blocks：`{change['before']['stable_target_block_ids']}`",
                f"- new_blocks：`{change['after']['stable_target_block_ids']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 后续",
            "",
            "- 重新运行 dataset consistency audit，确认 parent/block 引用仍可解析。",
            "- 下次全量测试中不要忽略 `v3_pc_147`；它已按 comparison 样本修正。",
            "- `v3_pc_018` 和 `v3_pc_153` 应作为已知 OCR/parser metadata backlog 单独标注；`v3_pc_148`、`v3_pc_149` 按 scope-control/caveat 样本保留。",
        ]
    )
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    main()
