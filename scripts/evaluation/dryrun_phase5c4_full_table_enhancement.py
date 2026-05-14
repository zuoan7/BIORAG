#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.enhance_table_like_paragraphs_pilot import (  # noqa: E402
    classify_candidate,
    clean_type,
    iter_blocks,
    nearby_short_run,
    normalize_text,
    page_value,
    preview,
    process_doc,
)


NON_CORE_HITS = {
    "after_caption_window",
    "before_caption_window",
    "caption_nearby_short_block_run",
}
METADATA_REJECT_REASONS = {
    "metadata_or_affiliation_signal",
    "reference_entry",
    "references_section",
    "figure_signal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5C-4 full-corpus dry-run audit for table-like paragraph association."
    )
    parser.add_argument("--input_dir", default="data/paper_round1/parsed_clean")
    parser.add_argument("--output_dir", default="reports/phase5c4_full_preflight")
    parser.add_argument("--window_after_caption", type=int, default=5)
    parser.add_argument("--window_before_caption", type=int, default=1)
    parser.add_argument("--max_associated_blocks_per_caption", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(input_dir.glob("*.json"))
    all_rows: list[dict[str, Any]] = []
    doc_stats: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in doc_paths:
        doc_id = path.stem
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
            blocks = iter_blocks(original)
            caption_count = sum(1 for block in blocks if clean_type(block) == "table_caption")
            table_like_candidate_count, likely_metadata_fp_count = preclassify_candidate_counts(
                original,
                window_before_caption=args.window_before_caption,
                window_after_caption=args.window_after_caption,
            )
            _enhanced, rows = process_doc(
                original,
                window_before_caption=args.window_before_caption,
                window_after_caption=args.window_after_caption,
                max_associated_blocks_per_caption=args.max_associated_blocks_per_caption,
            )
            enriched_rows = [enrich_row(row) for row in rows]
            all_rows.extend(enriched_rows)
            stats = summarize_doc(
                doc_id=doc_id,
                rows=enriched_rows,
                table_caption_count=caption_count,
                total_blocks=len(blocks),
                table_like_candidate_count=table_like_candidate_count,
                likely_metadata_fp_count=likely_metadata_fp_count,
            )
            doc_stats.append(stats)
        except Exception as exc:  # noqa: BLE001 - audit should preserve per-doc failures.
            failures.append({"doc_id": doc_id, "error": str(exc)})
            doc_stats.append({
                "doc_id": doc_id,
                "table_caption_count": 0,
                "candidate_rows": 0,
                "accepted_associations": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "uncertain_cases": 0,
                "rejected_nearby_blocks": 0,
                "accepted_long_prose": 0,
                "table_like_paragraph_candidates": 0,
                "likely_metadata_reference_affiliation_false_positive_count": 0,
                "risk_score": 1000,
                "risk_reasons": f"dry_run_failed: {exc}",
            })

    accepted = [row for row in all_rows if row["accepted_or_rejected"] == "accepted"]
    rejected = [row for row in all_rows if row["accepted_or_rejected"] == "rejected"]
    uncertain = [row for row in all_rows if row["accepted_or_rejected"] == "uncertain"]
    confidence_counts = Counter(row.get("association_confidence", "") for row in accepted)
    accepted_long_prose_count = sum(int(row["accepted_long_prose"]) for row in accepted)
    likely_metadata_fp_count = sum(
        int(row["likely_metadata_reference_affiliation_false_positive"])
        for row in all_rows
    )
    total_docs = len(doc_paths)
    association_counts = [int(row["accepted_associations"]) for row in doc_stats]
    high_threshold = unusual_threshold(association_counts)
    unusually_high_docs = [
        row for row in doc_stats
        if int(row["accepted_associations"]) > high_threshold and int(row["accepted_associations"]) > 0
    ]
    suspected_fp_docs = [
        row for row in doc_stats
        if int(row["accepted_long_prose"]) > 0
        or int(row["likely_metadata_reference_affiliation_false_positive_count"]) >= 3
        or (int(row["low_confidence"]) >= 5 and int(row["high_confidence"]) == 0)
    ]
    low_table_control_affected = sum(
        1
        for row in doc_stats
        if int(row["table_caption_count"]) <= 1 and int(row["accepted_associations"]) > 0
    )

    summary = {
        "total_docs": total_docs,
        "failed_docs": failures,
        "docs_with_table_caption": sum(1 for row in doc_stats if int(row["table_caption_count"]) > 0),
        "docs_with_table_related_candidates": sum(
            1
            for row in doc_stats
            if int(row["accepted_associations"]) > 0 or int(row["uncertain_cases"]) > 0
        ),
        "total_table_caption_count": sum(int(row["table_caption_count"]) for row in doc_stats),
        "estimated_table_related_associations": len(accepted),
        "associations_per_doc_avg": len(accepted) / total_docs if total_docs else 0.0,
        "high_confidence_count": confidence_counts.get("high", 0),
        "medium_confidence_count": confidence_counts.get("medium", 0),
        "low_confidence_count": confidence_counts.get("low", 0),
        "accepted_long_prose_count": accepted_long_prose_count,
        "uncertain_case_count": len(uncertain),
        "rejected_nearby_block_count": len(rejected),
        "top_docs_by_association_count": top_docs(doc_stats, "accepted_associations", 20),
        "docs_with_unusually_high_association_count": top_docs(unusually_high_docs, "accepted_associations", 20),
        "suspected_false_positive_docs": top_docs(suspected_fp_docs, "risk_score", 20, reverse=True),
        "low_table_control_docs_affected_count": low_table_control_affected,
        "table_like_paragraph_candidate_count": sum(
            int(row["table_like_paragraph_candidates"]) for row in doc_stats
        ),
        "likely_metadata_reference_affiliation_false_positive_count": likely_metadata_fp_count,
        "decision_counts": dict(Counter(row["accepted_or_rejected"] for row in all_rows)),
        "unusual_association_threshold": high_threshold,
        "dry_run_safe": dry_run_safe(
            failures=failures,
            accepted_count=len(accepted),
            accepted_long_prose_count=accepted_long_prose_count,
            suspected_fp_docs=suspected_fp_docs,
            unusually_high_docs=unusually_high_docs,
            total_docs=total_docs,
        ),
    }

    (output_dir / "dryrun_association_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_doc_stats(output_dir / "dryrun_doc_level_stats.csv", doc_stats)
    write_examples(
        output_dir / "dryrun_association_examples.md",
        accepted=accepted,
        rejected=rejected,
        uncertain=uncertain,
        high_risk_docs=top_docs(doc_stats, "risk_score", 10, reverse=True),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def preclassify_candidate_counts(
    data: dict[str, Any],
    *,
    window_before_caption: int,
    window_after_caption: int,
) -> tuple[int, int]:
    blocks = iter_blocks(data)
    table_like_candidate_count = 0
    likely_metadata_fp_count = 0
    captions = [
        (index, block)
        for index, block in enumerate(blocks)
        if clean_type(block) == "table_caption"
    ]
    for caption_index, caption in captions:
        positions = list(range(max(0, caption_index - window_before_caption), caption_index))
        positions += list(range(caption_index + 1, min(len(blocks), caption_index + window_after_caption + 1)))
        for block_index in positions:
            block = blocks[block_index]
            status, reject_reason, _confidence, hits = classify_candidate(
                block,
                caption,
                block_index - caption_index,
                nearby_short_run(blocks, caption_index, block_index),
            )
            signal_hits = [hit for hit in hits if hit not in NON_CORE_HITS]
            if status in {"accepted", "uncertain"} or signal_hits:
                table_like_candidate_count += 1
            if reject_reason in METADATA_REJECT_REASONS and signal_hits:
                likely_metadata_fp_count += 1
    return table_like_candidate_count, likely_metadata_fp_count


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("associated_text_preview", ""))
    words = text.split()
    sentence_boundary = bool(re.search(r"[.!?]\s+[A-Z]", text))
    long_prose = len(words) >= 45 and sentence_boundary
    metadata_fp = row.get("reject_reason") in METADATA_REJECT_REASONS
    result = dict(row)
    result["accepted_long_prose"] = (
        row.get("accepted_or_rejected") == "accepted" and long_prose
    )
    result["likely_metadata_reference_affiliation_false_positive"] = metadata_fp
    return result


def summarize_doc(
    *,
    doc_id: str,
    rows: list[dict[str, Any]],
    table_caption_count: int,
    total_blocks: int,
    table_like_candidate_count: int,
    likely_metadata_fp_count: int,
) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted_or_rejected"] == "accepted"]
    conf = Counter(row.get("association_confidence", "") for row in accepted)
    accepted_long_prose = sum(int(row["accepted_long_prose"]) for row in accepted)
    uncertain = sum(1 for row in rows if row["accepted_or_rejected"] == "uncertain")
    rejected = sum(1 for row in rows if row["accepted_or_rejected"] == "rejected")
    risk_reasons = []
    if accepted_long_prose:
        risk_reasons.append(f"accepted_long_prose={accepted_long_prose}")
    if likely_metadata_fp_count >= 3:
        risk_reasons.append(f"metadata_reference_affiliation_signals={likely_metadata_fp_count}")
    if conf.get("low", 0) >= 5 and conf.get("high", 0) == 0:
        risk_reasons.append(f"many_low_confidence={conf.get('low', 0)}")
    if len(accepted) >= 20:
        risk_reasons.append(f"many_associations={len(accepted)}")
    risk_score = (
        accepted_long_prose * 20
        + likely_metadata_fp_count * 3
        + conf.get("low", 0) * 2
        + len(accepted)
        + uncertain
    )
    return {
        "doc_id": doc_id,
        "table_caption_count": table_caption_count,
        "total_blocks": total_blocks,
        "candidate_rows": len(rows),
        "accepted_associations": len(accepted),
        "high_confidence": conf.get("high", 0),
        "medium_confidence": conf.get("medium", 0),
        "low_confidence": conf.get("low", 0),
        "uncertain_cases": uncertain,
        "rejected_nearby_blocks": rejected,
        "accepted_long_prose": accepted_long_prose,
        "table_like_paragraph_candidates": table_like_candidate_count,
        "likely_metadata_reference_affiliation_false_positive_count": likely_metadata_fp_count,
        "risk_score": risk_score,
        "risk_reasons": "; ".join(risk_reasons) if risk_reasons else "none",
    }


def unusual_threshold(values: list[int]) -> int:
    if not values:
        return 0
    avg = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return int(math.ceil(max(20, avg + 3 * std)))


def top_docs(
    rows: list[dict[str, Any]],
    key: str,
    limit: int,
    *,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(rows, key=lambda item: (int(item.get(key, 0)), str(item.get("doc_id", ""))), reverse=reverse)[:limit]
        if int(row.get(key, 0)) > 0
    ]


def dry_run_safe(
    *,
    failures: list[dict[str, str]],
    accepted_count: int,
    accepted_long_prose_count: int,
    suspected_fp_docs: list[dict[str, Any]],
    unusually_high_docs: list[dict[str, Any]],
    total_docs: int,
) -> bool:
    if failures:
        return False
    if accepted_count == 0:
        return False
    if accepted_long_prose_count > max(10, accepted_count * 0.03):
        return False
    if len(suspected_fp_docs) > max(10, total_docs * 0.05):
        return False
    if len(unusually_high_docs) > max(3, total_docs * 0.02):
        return False
    return True


def write_doc_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "table_caption_count",
        "total_blocks",
        "candidate_rows",
        "accepted_associations",
        "high_confidence",
        "medium_confidence",
        "low_confidence",
        "uncertain_cases",
        "rejected_nearby_blocks",
        "accepted_long_prose",
        "table_like_paragraph_candidates",
        "likely_metadata_reference_affiliation_false_positive_count",
        "risk_score",
        "risk_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_examples(
    path: Path,
    *,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    uncertain: list[dict[str, Any]],
    high_risk_docs: list[dict[str, Any]],
) -> None:
    lines = ["# Phase 5C-4 Dry-run Association Examples", ""]
    add_rows(lines, "Accepted Examples", accepted, 30)
    add_rows(lines, "Rejected Examples", rejected, 20)
    add_rows(lines, "Uncertain Examples", uncertain, 20)
    lines.extend(["## High-risk Docs", ""])
    for row in high_risk_docs[:10]:
        lines.extend([
            f"- `{row['doc_id']}`",
            f"  - reason: {row.get('risk_reasons', 'none')}",
            f"  - accepted: {row.get('accepted_associations', 0)}, "
            f"uncertain: {row.get('uncertain_cases', 0)}, "
            f"table captions: {row.get('table_caption_count', 0)}, "
            f"metadata/reference/affiliation signals: "
            f"{row.get('likely_metadata_reference_affiliation_false_positive_count', 0)}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_rows(
    lines: list[str],
    title: str,
    rows: list[dict[str, Any]],
    limit: int,
) -> None:
    lines.extend([f"## {title}", ""])
    for row in rows[:limit]:
        reason = row.get("reject_reason") or row.get("association_confidence") or "accepted"
        lines.extend([
            f"- `{row['doc_id']}` caption `{row['table_caption_block_id']}` -> "
            f"block `{row['associated_block_id']}` ({row['associated_block_type']})",
            f"  - decision: {row['accepted_or_rejected']} / {reason}",
            f"  - reason: hits={row.get('rule_hits', '')}; "
            f"distance={row.get('block_distance', '')}; page_distance={row.get('page_distance', '')}",
            f"  - caption: {preview(row.get('caption_text_preview', ''))}",
            f"  - block: {preview(row.get('associated_text_preview', ''))}",
        ])
    lines.append("")


if __name__ == "__main__":
    main()
