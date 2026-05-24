from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RUN_ID = "20260524"
DEFAULT_DATASET = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
DEFAULT_PARENT_CHUNKS = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_CHILD_CHUNKS = Path("data/paper_round1/chunks/child_chunks.jsonl")
DEFAULT_RESULTS = ""
EXPECTED_SAMPLE_COUNT = 200
REMAP_REGRESSION_SAMPLE_IDS = [
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
PRIMARY_CLASSES = {
    "pass_consistent_gold",
    "missing_parent_chunk",
    "missing_target_chunk_candidate",
    "missing_stable_block",
    "gold_parent_stable_block_mismatch_candidate",
    "stable_block_multi_parent_ambiguous",
    "expected_doc_mismatch",
    "comparison_multi_parent_scope_review",
    "negative_sample_skipped",
    "malformed_dataset_row",
}
REVIEW_CLASSES = {
    "missing_parent_chunk",
    "missing_target_chunk_candidate",
    "missing_stable_block",
    "gold_parent_stable_block_mismatch_candidate",
    "stable_block_multi_parent_ambiguous",
    "expected_doc_mismatch",
    "malformed_dataset_row",
}
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<section>\d+)_chunk(?P<chunk>\d+)$")
GENERIC_SECTIONS = {"", "full text", "abstract"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit v3 dataset gold parent / stable block consistency."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    run_id = str(args.run_id)
    dataset_path = Path(args.dataset)
    parent_chunks_path = Path(args.parent_chunks)
    child_chunks_path = Path(args.child_chunks)
    results_path = Path(args.results) if args.results else None

    dataset_rows = load_jsonl(dataset_path)
    chunk_index = build_chunk_index(
        parent_rows=load_jsonl(parent_chunks_path),
        child_rows=load_jsonl(child_chunks_path),
    )
    result_rows = load_optional_results(results_path)
    samples = [
        audit_dataset_row(row, chunk_index, result_rows.get(str(row.get("sample_id") or "")))
        for row in dataset_rows
    ]
    review_rows = build_review_rows(samples)
    summary = build_summary(
        run_id=run_id,
        input_paths={
            "dataset": str(dataset_path),
            "parent_chunks": str(parent_chunks_path),
            "child_chunks": str(child_chunks_path),
            "results": str(results_path) if results_path is not None else "",
        },
        samples=samples,
        review_rows=review_rows,
    )

    result_dir = RESULTS_ROOT / f"v3_dataset_gold_consistency_audit_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_dataset_gold_consistency_audit_{run_id}"
    outputs = {
        "summary": str(result_dir / "summary.json"),
        "samples": str(result_dir / "samples.jsonl"),
        "review_candidates": str(result_dir / "review_candidates.jsonl"),
        "report": str(report_dir / "report.md"),
    }
    summary["outputs"] = outputs
    write_json(result_dir / "summary.json", summary)
    write_jsonl(result_dir / "samples.jsonl", samples)
    write_jsonl(result_dir / "review_candidates.jsonl", review_rows)
    write_markdown(report_dir / "report.md", render_report(summary, samples, review_rows))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "report_dir": str(report_dir),
                "sample_count": summary["sample_count"],
                "review_candidate_count": summary["review_candidate_count"],
                "classification_counts": summary["classification_counts"],
                "validation_passed": summary["validation"]["passed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def audit_dataset_row(
    row: dict[str, Any],
    chunk_index: dict[str, Any],
    result_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    expected_route = str(row.get("expected_route") or "")
    category = str(row.get("category") or "")
    rubric = row.get("answer_rubric") if isinstance(row.get("answer_rubric"), dict) else {}
    source_trace = (
        rubric.get("source_trace") if isinstance(rubric.get("source_trace"), dict) else {}
    )
    expected_docs = as_str_list(row.get("expected_doc_ids"))
    expected_sections = as_str_list(row.get("expected_sections"))
    source_trace_docs = as_str_list(source_trace.get("doc_ids"))
    source_trace_chunks = as_str_list(source_trace.get("chunk_ids"))
    target_candidates = as_str_list(row.get("target_chunk_id_candidate"))
    source_trace_blocks = as_str_list(source_trace.get("block_ids"))
    stable_blocks = as_str_list(row.get("stable_target_block_ids"))
    malformed_reasons = collect_malformed_reasons(
        row=row,
        rubric=rubric,
        source_trace=source_trace,
        sample_id=sample_id,
    )
    source_chunk_status = [
        resolve_chunk_ref(value, chunk_index, role="source_trace_chunk")
        for value in source_trace_chunks
    ]
    target_status = [
        resolve_chunk_ref(value, chunk_index, role="target_chunk_id_candidate")
        for value in target_candidates
    ]
    gold_parent_ids = dedupe(
        [
            item["parent_chunk_id"]
            for item in [*source_chunk_status, *target_status]
            if item.get("parent_chunk_id")
        ]
    )
    gold_parent_records = [
        chunk_index["parent_records"].get(parent_id)
        for parent_id in gold_parent_ids
        if chunk_index["parent_records"].get(parent_id)
    ]
    gold_parent_doc_ids = dedupe([str(item.get("doc_id") or "") for item in gold_parent_records])
    doc_scope = dedupe([*expected_docs, *source_trace_docs, *gold_parent_doc_ids])
    stable_block_status = build_stable_block_status(
        stable_blocks=stable_blocks,
        doc_scope=doc_scope,
        chunk_index=chunk_index,
    )
    gold_coverage = build_gold_coverage(
        gold_parent_ids=gold_parent_ids,
        stable_blocks=stable_blocks,
        chunk_index=chunk_index,
    )
    stable_parent_candidates = find_stable_parent_candidates(
        stable_blocks=stable_blocks,
        doc_scope=doc_scope,
        gold_parent_ids=gold_parent_ids,
        chunk_index=chunk_index,
    )
    full_cover_candidates = [
        item for item in stable_parent_candidates if item["covers_all_stable_blocks"]
    ]
    same_doc_other_full_cover = [
        item for item in full_cover_candidates if not item["is_current_gold_parent"]
    ]
    doc_consistency = build_doc_consistency(
        expected_docs=expected_docs,
        source_trace_docs=source_trace_docs,
        gold_parent_doc_ids=gold_parent_doc_ids,
    )
    section_warning = build_section_warning(
        expected_sections=expected_sections,
        gold_parent_records=gold_parent_records,
    )
    evidence_warning = build_evidence_type_warning(
        category=category,
        evidence_note=str(rubric.get("evidence_note") or ""),
        must_include=as_str_list(rubric.get("must_include")),
        gold_parent_records=gold_parent_records,
        stable_blocks=stable_blocks,
    )
    warnings = []
    if section_warning["has_warning"]:
        warnings.append("section_mismatch_warning")
    if evidence_warning["has_warning"]:
        warnings.append("evidence_type_warning")
    primary = classify_sample(
        expected_route=expected_route,
        malformed_reasons=malformed_reasons,
        source_chunk_status=source_chunk_status,
        target_status=target_status,
        target_candidates=target_candidates,
        stable_blocks=stable_blocks,
        stable_block_status=stable_block_status,
        gold_coverage=gold_coverage,
        same_doc_other_full_cover=same_doc_other_full_cover,
        stable_parent_candidates=stable_parent_candidates,
        doc_consistency=doc_consistency,
    )
    return {
        "sample_id": sample_id,
        "question": str(row.get("question") or ""),
        "expected_answer": str(row.get("expected_answer") or ""),
        "category": category,
        "expected_route": expected_route,
        "expected_doc_ids": expected_docs,
        "expected_sections": expected_sections,
        "source_trace_doc_ids": source_trace_docs,
        "source_trace_chunk_ids": source_trace_chunks,
        "source_trace_block_ids": source_trace_blocks,
        "target_chunk_id_candidate": target_candidates[0] if target_candidates else "",
        "stable_target_block_ids": stable_blocks,
        "gold_parent_chunk_ids": gold_parent_ids,
        "gold_parent_doc_ids": gold_parent_doc_ids,
        "primary_classification": primary,
        "warning_classifications": warnings,
        "malformed_reasons": malformed_reasons,
        "source_trace_chunk_status": source_chunk_status,
        "target_chunk_status": target_status,
        "doc_consistency": doc_consistency,
        "stable_block_status": stable_block_status,
        "gold_stable_block_coverage": gold_coverage,
        "stable_block_parent_candidates": stable_parent_candidates,
        "same_doc_full_cover_candidate_ids": [
            item["chunk_id"] for item in same_doc_other_full_cover
        ],
        "stable_block_multi_parent_blocks": [
            item for item in stable_block_status["per_block"] if item["doc_scoped_parent_count"] > 1
        ],
        "section_warning": section_warning,
        "evidence_type_warning": evidence_warning,
        "retrieval_context": build_retrieval_context(result_row),
        "rubric": {
            "evidence_note": str(rubric.get("evidence_note") or ""),
            "must_include": as_str_list(rubric.get("must_include")),
            "acceptable_variants": as_str_list(rubric.get("acceptable_variants")),
            "reject_if": as_str_list(rubric.get("reject_if")),
        },
    }


def classify_sample(
    *,
    expected_route: str,
    malformed_reasons: list[str],
    source_chunk_status: list[dict[str, Any]],
    target_status: list[dict[str, Any]],
    target_candidates: list[str],
    stable_blocks: list[str],
    stable_block_status: dict[str, Any],
    gold_coverage: dict[str, Any],
    same_doc_other_full_cover: list[dict[str, Any]],
    stable_parent_candidates: list[dict[str, Any]],
    doc_consistency: dict[str, Any],
) -> str:
    route = expected_route.lower().strip()
    if malformed_reasons:
        return "malformed_dataset_row"
    if route == "negative":
        return "negative_sample_skipped"
    if any(chunk_reference_missing(item) for item in source_chunk_status):
        return "missing_parent_chunk"
    if any(chunk_reference_missing(item) for item in target_status):
        return "missing_target_chunk_candidate"
    if route not in {"comparison", "negative"} and not target_candidates:
        return "missing_target_chunk_candidate"
    if stable_blocks and stable_block_status["missing_doc_scoped_block_ids"]:
        return "missing_stable_block"
    if route not in {"comparison", "negative"} and not stable_blocks:
        return "missing_stable_block"
    if doc_consistency["has_mismatch"]:
        return "expected_doc_mismatch"
    if route == "comparison":
        return "comparison_multi_parent_scope_review"
    if (
        stable_blocks
        and not gold_coverage["gold_parent_covers_all_stable_blocks"]
        and same_doc_other_full_cover
    ):
        return "gold_parent_stable_block_mismatch_candidate"
    if has_multi_parent_ambiguity(
        stable_blocks=stable_blocks,
        gold_coverage=gold_coverage,
        stable_parent_candidates=stable_parent_candidates,
    ):
        return "stable_block_multi_parent_ambiguous"
    return "pass_consistent_gold"


def collect_malformed_reasons(
    *,
    row: dict[str, Any],
    rubric: dict[str, Any],
    source_trace: dict[str, Any],
    sample_id: str,
) -> list[str]:
    reasons = []
    if not sample_id:
        reasons.append("missing_sample_id")
    if not isinstance(row.get("answer_rubric"), dict):
        reasons.append("answer_rubric_not_object")
    if not isinstance(rubric.get("source_trace"), dict):
        reasons.append("source_trace_not_object")
    if not isinstance(row.get("expected_doc_ids", []), list):
        reasons.append("expected_doc_ids_not_list")
    if not isinstance(row.get("stable_target_block_ids", []), list):
        reasons.append("stable_target_block_ids_not_list")
    if not isinstance(source_trace.get("chunk_ids", []), list):
        reasons.append("source_trace_chunk_ids_not_list")
    if not isinstance(source_trace.get("doc_ids", []), list):
        reasons.append("source_trace_doc_ids_not_list")
    return reasons


def chunk_reference_missing(item: dict[str, Any]) -> bool:
    if not item["parent_exists"]:
        return True
    return item["reference_kind"] == "child" and item["child_exists"] is False


def build_chunk_index(
    *,
    parent_rows: list[dict[str, Any]],
    child_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_records = {
        str(row.get("chunk_id") or ""): compact_parent_record(row)
        for row in parent_rows
        if row.get("chunk_id")
    }
    child_records = {
        str(row.get("chunk_id") or ""): compact_child_record(row)
        for row in child_rows
        if row.get("chunk_id")
    }
    block_parent_index: dict[tuple[str, str], list[str]] = {}
    global_block_parent_index: dict[str, list[str]] = {}
    for parent_id, record in parent_records.items():
        for block in record["block_ids"]:
            block_parent_index.setdefault((record["doc_id"], block), []).append(parent_id)
            global_block_parent_index.setdefault(block, []).append(parent_id)
    for values in block_parent_index.values():
        values.sort()
    for values in global_block_parent_index.values():
        values.sort()
    return {
        "parent_records": parent_records,
        "child_records": child_records,
        "block_parent_index": block_parent_index,
        "global_block_parent_index": global_block_parent_index,
    }


def compact_parent_record(row: dict[str, Any]) -> dict[str, Any]:
    block_metadata = as_block_metadata(row.get("source_block_metadata"))
    block_ids = dedupe(
        [
            *as_str_list(row.get("source_block_ids") or row.get("block_ids")),
            *[
                block
                for item in block_metadata
                for block in [
                    str(item.get("block_id") or ""),
                    str(item.get("source_block_id") or ""),
                ]
                if block
            ],
        ]
    )
    evidence_types = dedupe(
        [
            *as_str_list(row.get("evidence_types") or row.get("block_types")),
            *[str(item.get("type") or "") for item in block_metadata if item.get("type")],
        ]
    )
    return {
        "chunk_id": str(row.get("chunk_id") or ""),
        "doc_id": str(row.get("doc_id") or ""),
        "source_file": str(row.get("source_file") or ""),
        "section": str(row.get("section") or ""),
        "page_numbers": as_str_list(row.get("page_numbers")),
        "block_ids": block_ids,
        "block_metadata": block_metadata,
        "evidence_types": evidence_types,
        "contains_table_caption": bool(row.get("contains_table_caption")),
        "contains_table_text": bool(row.get("contains_table_text")),
        "contains_figure_caption": bool(row.get("contains_figure_caption")),
        "text_preview": compact_text(row.get("retrieval_text") or row.get("text"), limit=700),
    }


def compact_child_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(row.get("chunk_id") or ""),
        "parent_chunk_id": str(row.get("parent_chunk_id") or parent_chunk_id(row.get("chunk_id"))),
        "doc_id": str(row.get("doc_id") or ""),
        "section": str(row.get("section") or ""),
        "block_ids": as_str_list(row.get("source_block_ids") or row.get("block_ids")),
    }


def resolve_chunk_ref(
    value: str,
    chunk_index: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    raw_id = str(value or "")
    is_child = "::child" in raw_id
    child_record = chunk_index["child_records"].get(raw_id) if is_child else None
    parent_id = (
        str(child_record.get("parent_chunk_id") or "")
        if child_record
        else parent_chunk_id(raw_id)
    )
    parent_record = chunk_index["parent_records"].get(parent_id)
    return {
        "role": role,
        "chunk_id": raw_id,
        "reference_kind": "child" if is_child else "parent",
        "child_exists": bool(child_record) if is_child else None,
        "parent_chunk_id": parent_id,
        "parent_exists": bool(parent_record),
        "doc_id": str(parent_record.get("doc_id") or "") if parent_record else doc_id(parent_id),
        "section": str(parent_record.get("section") or "") if parent_record else "",
    }


def build_stable_block_status(
    *,
    stable_blocks: list[str],
    doc_scope: list[str],
    chunk_index: dict[str, Any],
) -> dict[str, Any]:
    per_block = []
    missing_doc_scoped = []
    missing_global = []
    for block in stable_blocks:
        doc_scoped_parent_ids = dedupe(
            [
                parent_id
                for doc in doc_scope
                for parent_id in chunk_index["block_parent_index"].get((doc, block), [])
            ]
        )
        global_parent_ids = as_str_list(chunk_index["global_block_parent_index"].get(block))
        if not doc_scoped_parent_ids:
            missing_doc_scoped.append(block)
        if not global_parent_ids:
            missing_global.append(block)
        per_block.append(
            {
                "block_id": block,
                "doc_scoped_parent_ids": doc_scoped_parent_ids,
                "doc_scoped_parent_count": len(doc_scoped_parent_ids),
                "global_parent_ids": global_parent_ids[:20],
                "global_parent_count": len(global_parent_ids),
            }
        )
    return {
        "stable_block_count": len(stable_blocks),
        "doc_scope": doc_scope,
        "missing_doc_scoped_block_ids": missing_doc_scoped,
        "missing_global_block_ids": missing_global,
        "all_blocks_exist_in_doc_scope": bool(stable_blocks) and not missing_doc_scoped,
        "all_blocks_exist_globally": bool(stable_blocks) and not missing_global,
        "per_block": per_block,
    }


def build_gold_coverage(
    *,
    gold_parent_ids: list[str],
    stable_blocks: list[str],
    chunk_index: dict[str, Any],
) -> dict[str, Any]:
    stable_set = set(stable_blocks)
    covered = set()
    per_parent = []
    for parent_id in gold_parent_ids:
        record = chunk_index["parent_records"].get(parent_id) or {}
        block_ids = set(as_str_list(record.get("block_ids")))
        hits = sorted(block_ids & stable_set)
        covered.update(hits)
        per_parent.append(
            {
                "chunk_id": parent_id,
                "exists": bool(record),
                "doc_id": str(record.get("doc_id") or doc_id(parent_id)),
                "section": str(record.get("section") or ""),
                "hit_block_ids": hits,
                "hit_block_count": len(hits),
                "covers_all_stable_blocks": bool(stable_set) and set(hits) == stable_set,
            }
        )
    return {
        "gold_parent_stable_hit_block_ids": sorted(covered),
        "gold_parent_missing_stable_block_ids": sorted(stable_set - covered),
        "gold_parent_overlaps_stable_blocks": bool(covered) if stable_blocks else None,
        "gold_parent_covers_all_stable_blocks": bool(stable_set) and covered == stable_set,
        "per_gold_parent": per_parent,
    }


def find_stable_parent_candidates(
    *,
    stable_blocks: list[str],
    doc_scope: list[str],
    gold_parent_ids: list[str],
    chunk_index: dict[str, Any],
) -> list[dict[str, Any]]:
    if not stable_blocks:
        return []
    stable_set = set(stable_blocks)
    gold_set = set(gold_parent_ids)
    candidate_parent_ids = dedupe(
        [
            parent_id
            for block in stable_blocks
            for doc in doc_scope
            for parent_id in chunk_index["block_parent_index"].get((doc, block), [])
        ]
    )
    candidates = []
    for parent_id in candidate_parent_ids:
        record = chunk_index["parent_records"].get(parent_id)
        if not record:
            continue
        hits = sorted(set(record["block_ids"]) & stable_set)
        candidates.append(
            {
                **parent_card(record, stable_blocks, preview_limit=650),
                "hit_block_ids": hits,
                "hit_block_count": len(hits),
                "covers_all_stable_blocks": set(hits) == stable_set,
                "is_current_gold_parent": parent_id in gold_set,
            }
        )
    candidates.sort(
        key=lambda item: (
            not bool(item["covers_all_stable_blocks"]),
            not bool(item["is_current_gold_parent"]),
            -int(item["hit_block_count"]),
            str(item["chunk_id"]),
        )
    )
    return candidates


def build_doc_consistency(
    *,
    expected_docs: list[str],
    source_trace_docs: list[str],
    gold_parent_doc_ids: list[str],
) -> dict[str, Any]:
    expected = set(expected_docs)
    source = set(source_trace_docs)
    parents = set(gold_parent_doc_ids)
    mismatches = []
    if expected and source and source != expected:
        mismatches.append("source_trace_doc_ids_do_not_match_expected_doc_ids")
    if expected and parents and not parents <= expected:
        mismatches.append("gold_parent_doc_ids_not_subset_of_expected_doc_ids")
    if source and parents and not parents <= source:
        mismatches.append("gold_parent_doc_ids_not_subset_of_source_trace_doc_ids")
    return {
        "expected_doc_ids": expected_docs,
        "source_trace_doc_ids": source_trace_docs,
        "gold_parent_doc_ids": gold_parent_doc_ids,
        "has_mismatch": bool(mismatches),
        "mismatch_reasons": mismatches,
    }


def build_section_warning(
    *,
    expected_sections: list[str],
    gold_parent_records: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_expected = [
        normalize_section(section)
        for section in expected_sections
        if normalize_section(section) not in GENERIC_SECTIONS
    ]
    parent_sections = [str(record.get("section") or "") for record in gold_parent_records]
    normalized_parent = [normalize_section(section) for section in parent_sections]
    if not normalized_expected or not normalized_parent:
        return {
            "has_warning": False,
            "expected_sections": expected_sections,
            "gold_parent_sections": parent_sections,
            "reason": "",
        }
    has_overlap = any(
        expected in parent or parent in expected
        for expected in normalized_expected
        for parent in normalized_parent
        if expected and parent
    )
    return {
        "has_warning": not has_overlap,
        "expected_sections": expected_sections,
        "gold_parent_sections": parent_sections,
        "reason": "expected_sections_do_not_overlap_gold_parent_sections"
        if not has_overlap
        else "",
    }


def build_evidence_type_warning(
    *,
    category: str,
    evidence_note: str,
    must_include: list[str],
    gold_parent_records: list[dict[str, Any]],
    stable_blocks: list[str],
) -> dict[str, Any]:
    text = " ".join([category, evidence_note, *must_include]).lower()
    expects_table = category in {"table_content", "caption_level_table"} or "table" in text
    expects_figure = category == "figure_caption" or "figure" in text or "fig." in text
    current_flags = summarize_evidence_flags(gold_parent_records)
    reasons = []
    if stable_blocks and expects_table and not (
        current_flags["contains_table_caption"] or current_flags["contains_table_text"]
    ):
        reasons.append("table_evidence_expected_but_gold_parent_has_no_table_flag")
    if stable_blocks and expects_figure and not current_flags["contains_figure_caption"]:
        reasons.append("figure_evidence_expected_but_gold_parent_has_no_figure_flag")
    return {
        "has_warning": bool(reasons),
        "expected_table_evidence": expects_table,
        "expected_figure_evidence": expects_figure,
        "gold_parent_flags": current_flags,
        "reasons": reasons,
    }


def summarize_evidence_flags(records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_types = dedupe(
        [value for record in records for value in as_str_list(record.get("evidence_types"))]
    )
    return {
        "contains_table_caption": any(
            bool(record.get("contains_table_caption")) for record in records
        ),
        "contains_table_text": any(bool(record.get("contains_table_text")) for record in records),
        "contains_figure_caption": any(
            bool(record.get("contains_figure_caption")) for record in records
        ),
        "evidence_types": evidence_types,
    }


def has_multi_parent_ambiguity(
    *,
    stable_blocks: list[str],
    gold_coverage: dict[str, Any],
    stable_parent_candidates: list[dict[str, Any]],
) -> bool:
    if not stable_blocks:
        return False
    if not gold_coverage["gold_parent_covers_all_stable_blocks"]:
        return False
    full_cover_count = sum(
        1 for item in stable_parent_candidates if item["covers_all_stable_blocks"]
    )
    if full_cover_count > 1:
        return True
    return any(
        len(
            [
                item
                for item in stable_parent_candidates
                if block in set(as_str_list(item.get("hit_block_ids")))
            ]
        )
        > 1
        for block in stable_blocks
    )


def build_retrieval_context(result_row: dict[str, Any] | None) -> dict[str, Any]:
    if not result_row:
        return {}
    return {
        "raw_retrieved_parent_chunk_ids_preview": as_str_list(
            result_row.get("raw_retrieved_parent_chunk_ids")
        )[:15],
        "retrieved_parent_chunk_ids_top10": as_str_list(
            result_row.get("retrieved_parent_chunk_ids_top10")
        ),
        "support_parent_chunk_ids": as_str_list(result_row.get("support_parent_chunk_ids")),
        "citation_parent_chunk_ids": as_str_list(result_row.get("citation_parent_chunk_ids")),
    }


def build_review_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["primary_classification"] not in REVIEW_CLASSES:
            continue
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "primary_classification": sample["primary_classification"],
                "warning_classifications": sample["warning_classifications"],
                "question": sample["question"],
                "expected_answer": sample["expected_answer"],
                "category": sample["category"],
                "expected_route": sample["expected_route"],
                "expected_doc_ids": sample["expected_doc_ids"],
                "stable_target_block_ids": sample["stable_target_block_ids"],
                "gold_parent_chunk_ids": sample["gold_parent_chunk_ids"],
                "source_trace_chunk_ids": sample["source_trace_chunk_ids"],
                "target_chunk_id_candidate": sample["target_chunk_id_candidate"],
                "gold_stable_block_coverage": sample["gold_stable_block_coverage"],
                "doc_consistency": sample["doc_consistency"],
                "stable_block_status": sample["stable_block_status"],
                "rubric": sample["rubric"],
                "current_gold_parent_cards": [
                    item
                    for item in sample["stable_block_parent_candidates"]
                    if item["is_current_gold_parent"]
                ],
                "same_doc_full_cover_candidates": [
                    item
                    for item in sample["stable_block_parent_candidates"]
                    if item["covers_all_stable_blocks"]
                    and not item["is_current_gold_parent"]
                ],
                "all_stable_block_parent_candidates": sample["stable_block_parent_candidates"],
                "review_decision_template": {
                    "decision": "",
                    "selected_parent_chunk_ids": "",
                    "selected_stable_block_ids": "",
                    "reason": "",
                    "notes": "",
                },
            }
        )
    return rows


def build_summary(
    *,
    run_id: str,
    input_paths: dict[str, str],
    samples: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    classification_counts = Counter(sample["primary_classification"] for sample in samples)
    warning_counts = Counter(
        warning for sample in samples for warning in sample["warning_classifications"]
    )
    route_counts = Counter(sample["expected_route"] for sample in samples)
    category_counts = Counter(sample["category"] for sample in samples)
    remap_regression = build_remap_regression(samples)
    validation = build_validation(samples, classification_counts, remap_regression)
    return {
        "run_id": run_id,
        "scope": "v3 baseline dataset gold parent / stable block consistency audit",
        "inputs": input_paths,
        "sample_count": len(samples),
        "review_candidate_count": len(review_rows),
        "route_counts": dict(sorted(route_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "review_candidate_sample_ids": [row["sample_id"] for row in review_rows],
        "mismatch_candidate_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "gold_parent_stable_block_mismatch_candidate"
        ],
        "ambiguous_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "stable_block_multi_parent_ambiguous"
        ],
        "missing_parent_chunk_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "missing_parent_chunk"
        ],
        "missing_stable_block_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "missing_stable_block"
        ],
        "expected_doc_mismatch_sample_ids": [
            sample["sample_id"]
            for sample in samples
            if sample["primary_classification"] == "expected_doc_mismatch"
        ],
        "remap_regression": remap_regression,
        "validation": validation,
    }


def build_remap_regression(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    rows = []
    failed = []
    for sample_id in REMAP_REGRESSION_SAMPLE_IDS:
        sample = sample_by_id.get(sample_id)
        if not sample:
            rows.append(
                {
                    "sample_id": sample_id,
                    "present": False,
                    "primary_classification": "missing_from_dataset",
                    "passed": False,
                }
            )
            failed.append(sample_id)
            continue
        primary = sample["primary_classification"]
        passed = primary not in {
            "gold_parent_stable_block_mismatch_candidate",
            "stable_block_multi_parent_ambiguous",
            "missing_parent_chunk",
            "missing_target_chunk_candidate",
            "missing_stable_block",
            "expected_doc_mismatch",
            "malformed_dataset_row",
        }
        rows.append(
            {
                "sample_id": sample_id,
                "present": True,
                "primary_classification": primary,
                "warning_classifications": sample["warning_classifications"],
                "gold_parent_chunk_ids": sample["gold_parent_chunk_ids"],
                "stable_target_block_ids": sample["stable_target_block_ids"],
                "passed": passed,
            }
        )
        if not passed:
            failed.append(sample_id)
    return {
        "expected_sample_ids": REMAP_REGRESSION_SAMPLE_IDS,
        "passed": not failed,
        "failed_sample_ids": failed,
        "samples": rows,
    }


def build_validation(
    samples: list[dict[str, Any]],
    classification_counts: Counter[str],
    remap_regression: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = [sample["sample_id"] for sample in samples]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if sample_id and count > 1
    )
    criteria = [
        {
            "name": "sample_count_is_200",
            "passed": len(samples) == EXPECTED_SAMPLE_COUNT,
            "actual": len(samples),
            "expected": EXPECTED_SAMPLE_COUNT,
        },
        {
            "name": "sample_ids_are_present_and_unique",
            "passed": all(sample_ids) and not duplicate_ids,
            "duplicate_ids": duplicate_ids,
            "missing_count": sum(1 for sample_id in sample_ids if not sample_id),
        },
        {
            "name": "all_primary_classifications_are_known",
            "passed": all(key in PRIMARY_CLASSES for key in classification_counts),
            "unknown_classes": [
                key for key in sorted(classification_counts) if key not in PRIMARY_CLASSES
            ],
        },
        {
            "name": "remap_regression_samples_not_flagged_as_mismatch",
            "passed": bool(remap_regression["passed"]),
            "failed_sample_ids": remap_regression["failed_sample_ids"],
        },
    ]
    return {
        "passed": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "failed_criteria": [item for item in criteria if not item["passed"]],
    }


def render_report(
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# v3 baseline dataset gold consistency 审计报告",
        "",
        "## 范围",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- sample_count: {summary['sample_count']}",
        f"- review_candidate_count: {summary['review_candidate_count']}",
        "- 口径：只读 dataset、parent/child chunks；不运行 eval/judge，不修改 dataset。",
        f"- validation_passed: `{summary['validation']['passed']}`",
        "",
        "## 分类汇总",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["classification_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Warning 汇总",
            "",
            "| warning | count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(summary["warning_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    if not summary["warning_counts"]:
        lines.append("| - | 0 |")
    lines.extend(render_candidate_table("Mismatch candidates", review_rows))
    lines.extend(render_regression_section(summary["remap_regression"]))
    lines.extend(render_validation_section(summary["validation"]))
    lines.extend(
        [
            "",
            "## 后续",
            "",
            "- 对 `review_candidates.jsonl` 中的样本先做人工复核。",
            "- 若确认需要 remap，单独生成 decision ledger 和应用脚本；本审计不直接修改 dataset。",
            "- 当前重点仍是数据集质量门，不从 eval miss 反推 remap。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_candidate_table(title: str, review_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| sample_id | class | current gold | full-cover candidates | missing stable blocks |",
        "|---|---|---|---|---|",
    ]
    if not review_rows:
        lines.append("| - | - | - | - | - |")
        return lines
    for row in review_rows:
        coverage = row["gold_stable_block_coverage"]
        candidate_ids = [
            item["chunk_id"] for item in row["same_doc_full_cover_candidates"]
        ] or [
            item["chunk_id"]
            for item in row["all_stable_block_parent_candidates"]
            if item["covers_all_stable_blocks"]
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['sample_id']}`",
                    f"`{row['primary_classification']}`",
                    format_code_list(row["gold_parent_chunk_ids"]),
                    format_code_list(candidate_ids),
                    format_code_list(coverage["gold_parent_missing_stable_block_ids"]),
                ]
            )
            + " |"
        )
    return lines


def render_regression_section(regression: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## 14 条已 remap 样本回归检查",
        "",
        f"- passed: `{regression['passed']}`",
        f"- failed_sample_ids: {format_code_list(regression['failed_sample_ids'])}",
        "",
        "| sample_id | status | gold parent | stable blocks | warnings |",
        "|---|---|---|---|---|",
    ]
    for row in regression["samples"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['sample_id']}`",
                    f"`{row['primary_classification']}`",
                    format_code_list(row.get("gold_parent_chunk_ids") or []),
                    format_code_list(row.get("stable_target_block_ids") or []),
                    format_code_list(row.get("warning_classifications") or []),
                ]
            )
            + " |"
        )
    return lines


def render_validation_section(validation: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Validation",
        "",
        "| criterion | status | details |",
        "|---|---|---|",
    ]
    for item in validation["criteria"]:
        details = {key: value for key, value in item.items() if key not in {"name", "passed"}}
        lines.append(
            f"| `{item['name']}` | {'PASS' if item['passed'] else 'FAIL'} | "
            f"`{json.dumps(details, ensure_ascii=False, sort_keys=True)}` |"
        )
    return lines


def parent_card(
    record: dict[str, Any],
    stable_blocks: list[str],
    *,
    preview_limit: int,
) -> dict[str, Any]:
    stable_set = set(stable_blocks)
    block_metadata = as_block_metadata(record.get("block_metadata"))
    hit_previews = [
        compact_block(item)
        for item in block_metadata
        if block_id(item) in stable_set or source_block_id(item) in stable_set
    ]
    return {
        "chunk_id": str(record.get("chunk_id") or ""),
        "doc_id": str(record.get("doc_id") or ""),
        "source_file": str(record.get("source_file") or ""),
        "section": str(record.get("section") or ""),
        "page_numbers": as_str_list(record.get("page_numbers")),
        "evidence_types": as_str_list(record.get("evidence_types")),
        "contains_table_caption": bool(record.get("contains_table_caption")),
        "contains_table_text": bool(record.get("contains_table_text")),
        "contains_figure_caption": bool(record.get("contains_figure_caption")),
        "stable_block_previews": hit_previews,
        "text_preview": compact_text(record.get("text_preview"), limit=preview_limit),
    }


def as_block_metadata(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def compact_block(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block_id(item),
        "source_block_id": source_block_id(item),
        "type": str(item.get("type") or ""),
        "page": item.get("page"),
        "section_path": as_str_list(item.get("section_path")),
        "text_preview": compact_text(item.get("text_preview") or item.get("child_text_preview")),
    }


def block_id(item: dict[str, Any]) -> str:
    return str(item.get("block_id") or "")


def source_block_id(item: dict[str, Any]) -> str:
    return str(item.get("source_block_id") or "")


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def doc_id(parent_id_value: Any) -> str:
    value = str(parent_id_value or "")
    if "_sec" in value:
        return value.split("_sec", 1)[0]
    return ""


def parent_position(parent_id_value: Any) -> tuple[int, int] | None:
    match = PARENT_ID_RE.match(str(parent_id_value or ""))
    if not match:
        return None
    return int(match.group("section")), int(match.group("chunk"))


def normalize_section(value: str) -> str:
    normalized = str(value or "").lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def compact_text(value: Any, *, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_optional_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists() or path.is_dir():
        return {}
    rows = {}
    for line_number, row in enumerate(load_jsonl(path), start=1):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{path}:{line_number} missing sample_id")
        rows[sample_id] = row
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


def run_self_test() -> None:
    chunk_index = build_chunk_index(
        parent_rows=[
            parent_fixture("doc_a_sec01_chunk01", "doc_a", ["b1", "b2"]),
            parent_fixture("doc_a_sec02_chunk02", "doc_a", ["b3"], table_text=True),
            parent_fixture("doc_b_sec01_chunk01", "doc_b", ["b4"]),
            parent_fixture("doc_c_sec01_chunk01", "doc_c", ["c1"]),
            parent_fixture("doc_d_sec01_chunk01", "doc_d", ["d1"]),
            parent_fixture("doc_e_sec01_chunk01", "doc_e", ["e1"]),
        ],
        child_rows=[
            child_fixture("doc_a_sec01_chunk01::child001", "doc_a_sec01_chunk01", "doc_a"),
            child_fixture("doc_a_sec02_chunk02::child001", "doc_a_sec02_chunk02", "doc_a"),
        ],
    )
    consistent = audit_dataset_row(
        dataset_fixture(
            "case_consistent",
            expected_docs=["doc_a"],
            source_chunks=["doc_a_sec01_chunk01::child001"],
            target="doc_a_sec01_chunk01::child001",
            stable_blocks=["b1", "b2"],
        ),
        chunk_index,
    )
    assert consistent["primary_classification"] == "pass_consistent_gold"
    assert consistent["gold_parent_chunk_ids"] == ["doc_a_sec01_chunk01"]

    mismatch = audit_dataset_row(
        dataset_fixture(
            "case_mismatch",
            expected_docs=["doc_a"],
            source_chunks=["doc_a_sec01_chunk01"],
            target="doc_a_sec01_chunk01",
            stable_blocks=["b3"],
        ),
        chunk_index,
    )
    assert mismatch["primary_classification"] == "gold_parent_stable_block_mismatch_candidate"
    assert mismatch["same_doc_full_cover_candidate_ids"] == ["doc_a_sec02_chunk02"]

    missing_block = audit_dataset_row(
        dataset_fixture(
            "case_missing_block",
            expected_docs=["doc_a"],
            source_chunks=["doc_a_sec01_chunk01"],
            target="doc_a_sec01_chunk01",
            stable_blocks=["missing_b"],
        ),
        chunk_index,
    )
    assert missing_block["primary_classification"] == "missing_stable_block"

    comparison = audit_dataset_row(
        dataset_fixture(
            "case_comparison",
            expected_route="comparison",
            category="comparison",
            expected_docs=["doc_d", "doc_e"],
            source_chunks=["doc_d_sec01_chunk01", "doc_e_sec01_chunk01"],
            target="",
            stable_blocks=["d1", "e1"],
        ),
        chunk_index,
    )
    assert comparison["primary_classification"] == "comparison_multi_parent_scope_review"

    target_missing = audit_dataset_row(
        dataset_fixture(
            "case_target_missing",
            expected_docs=["doc_a"],
            source_chunks=["doc_a_sec01_chunk01"],
            target="doc_a_sec99_chunk99",
            stable_blocks=["b1"],
        ),
        chunk_index,
    )
    assert target_missing["primary_classification"] == "missing_target_chunk_candidate"

    regression = build_remap_regression(
        [
            {
                "sample_id": sample_id,
                "primary_classification": "pass_consistent_gold",
                "warning_classifications": [],
                "gold_parent_chunk_ids": ["doc_a_sec01_chunk01"],
                "stable_target_block_ids": ["b1"],
            }
            for sample_id in REMAP_REGRESSION_SAMPLE_IDS
        ]
    )
    assert regression["passed"] is True


def parent_fixture(
    chunk_id: str,
    doc: str,
    blocks: list[str],
    *,
    table_text: bool = False,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc,
        "source_file": f"{doc}.pdf",
        "section": "Results",
        "source_block_ids": blocks,
        "source_block_metadata": [
            {
                "block_id": block,
                "source_block_id": block,
                "type": "table_text" if table_text else "paragraph",
                "page": 1,
                "section_path": ["Results"],
                "text_preview": f"preview {block}",
            }
            for block in blocks
        ],
        "contains_table_text": table_text,
    }


def child_fixture(chunk_id: str, parent_id: str, doc: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "parent_chunk_id": parent_id,
        "doc_id": doc,
        "source_block_ids": ["b1"],
    }


def dataset_fixture(
    sample_id: str,
    *,
    expected_docs: list[str],
    source_chunks: list[str],
    target: str,
    stable_blocks: list[str],
    expected_route: str = "factoid",
    category: str = "normal_factoid",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "question": "question",
        "expected_answer": "answer",
        "expected_route": expected_route,
        "expected_doc_ids": expected_docs,
        "expected_sections": ["Results"],
        "category": category,
        "answer_rubric": {
            "source_trace": {
                "doc_ids": expected_docs,
                "chunk_ids": source_chunks,
                "block_ids": stable_blocks,
            },
            "evidence_note": "",
            "must_include": [],
        },
        "target_chunk_id_candidate": target,
        "stable_target_block_ids": stable_blocks,
    }


if __name__ == "__main__":
    main()
