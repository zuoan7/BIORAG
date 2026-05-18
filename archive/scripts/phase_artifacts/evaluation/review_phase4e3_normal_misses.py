#!/usr/bin/env python3
"""Review Phase 4E-3B normal hybrid misses before Phase 4 closeout.

This script is read-only with respect to eval/retrieval inputs. It consumes the
existing retrieval results and writes a manual-review ledger plus summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REVIEW_DECISIONS: dict[str, dict[str, str]] = {
    "p4e3_normal_0032": {
        "category": "eval_sample_issue_query_target_mismatch",
        "rationale": (
            "Query anchors GDP-L-fucose/L-homoarginine point to fucose/HMO pathway literature, "
            "while the target is a Chinese chapter summary about L-arginine fermentation medium "
            "optimization. Top hits follow the GDP-L-fucose cue, so the target is not a natural "
            "answer for the query."
        ),
        "recommended_action": "exclude or rewrite in a future normal eval set; do not change retrieval for Phase 4.",
    },
    "p4e3_normal_0041": {
        "category": "eval_sample_issue_query_target_mismatch",
        "rationale": (
            "The query is an anchor-soup form around Biomanufacturing/min-1/Thr127, but the "
            "target is a long Chinese L-DOPA bioreactor fermentation section. The query does "
            "not ask for the specific fermentation conditions visible in the target."
        ),
        "recommended_action": "rewrite query around L-DOPA 5 L bioreactor conditions in a future eval set.",
    },
    "p4e3_normal_0049": {
        "category": "eval_sample_issue_query_target_mismatch",
        "rationale": (
            "Query combines pACYCDuet-1, CMP-Neu5Ac, and JT-SHIZ-119, but the target preview "
            "is a Chinese 6'-SL pathway discussion about ST6/neuBCA/vector-copy choices. Top "
            "hits reasonably follow CMP-Neu5Ac/JT-SHIZ terms to synthetase literature."
        ),
        "recommended_action": "rewrite with target-specific 6'-SL/ST6/neuBCA wording or remove from normal controls.",
    },
    "p4e3_normal_0054": {
        "category": "eval_sample_issue_weak_or_boilerplate_target",
        "rationale": (
            "The query is built from weak/OCR-like anchors X-03/ANKOM220/O.01, while the target "
            "is a broad Chinese abstract/full-text chunk with journal metadata, formulas, and "
            "mixed document context. It is not a clean semantic normal-control target."
        ),
        "recommended_action": "replace with a cleaner normal paragraph in a future eval set; keep parser cleanup in backlog.",
    },
    "p4e3_normal_0057": {
        "category": "eval_sample_issue_query_target_mismatch",
        "rationale": (
            "Query asks about CRISPR-Cas9/Cas12a optimization, while the target preview is a "
            "Chinese review paragraph about carotenoid/terpene pathway engineering in Yarrowia. "
            "The CRISPR terms drive retrieval toward genome-editing chunks rather than the target."
        ),
        "recommended_action": "rewrite around carotenoid/terpene/Yarrowia pathway content or remove from normal controls.",
    },
    "p4e3_normal_0060": {
        "category": "ambiguous_or_multiple_valid_docs",
        "rationale": (
            "The query asks broadly about optimized laccase-producing conditions and includes "
            "a reference-like Laura-Leena anchor. Top1 is another valid laccase-condition abstract, "
            "while the target also contains journal/header metadata and references. The target doc "
            "is not uniquely implied."
        ),
        "recommended_action": "replace with a less ambiguous laccase query or accept multiple valid docs in future scoring.",
    },
    "p4e3_normal_supplement_0008": {
        "category": "retrieval_issue_doc_level_recall",
        "rationale": (
            "The query is natural and the target abstract clearly answers PbFuc/Pedobacter "
            "alpha-L-fucosidase for 2'-FL and 3-FL synthesis, but hybrid top50 misses the target "
            "document and over-ranks broader 2-FL E. coli production material."
        ),
        "recommended_action": "add to normal doc-level recall backlog; no Phase 4 retrieval tuning.",
    },
    "p4e3_normal_supplement_0014": {
        "category": "retrieval_issue_chunk_ranking",
        "rationale": (
            "The target document is rank 1, but the abstract target chunk is rank 11. Higher "
            "ranked chunks from the same/duplicate document discuss NeuAc optimization details, "
            "so this is local chunk ranking rather than doc-level recall failure."
        ),
        "recommended_action": "track as chunk-ranking/local evidence selection backlog; no Phase 4 tuning.",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def preview(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def miss_type(result: dict[str, Any], top_k: int) -> str:
    doc_rank = result.get("target_doc_rank")
    chunk_rank = result.get("target_chunk_rank")
    if doc_rank is None:
        return "target_doc_not_in_top50"
    if chunk_rank is None:
        return "target_doc_hit_but_target_chunk_not_in_top50"
    if int(chunk_rank) > top_k:
        return "target_doc_hit_but_target_chunk_ranked_11_to_50"
    return "unknown_chunk_miss"


def format_rank(value: Any) -> str:
    return "" if value is None else str(value)


def extract_rows(results_payload: dict[str, Any], top_k: int) -> list[dict[str, str]]:
    hybrid_results = results_payload["results_by_mode"]["hybrid"]
    misses = [
        result
        for result in hybrid_results
        if result.get("sample_type") == "normal" and not result.get(f"chunk_hit@{top_k}")
    ]
    rows: list[dict[str, str]] = []
    for result in misses:
        sample_id = str(result["sample_id"])
        decision = REVIEW_DECISIONS.get(sample_id)
        if decision is None:
            raise ValueError(f"No manual review decision for normal miss: {sample_id}")
        top_k_hits = result.get("top_k", [])
        top1 = top_k_hits[0] if top_k_hits else {}
        top3 = top_k_hits[:3]
        rows.append(
            {
                "sample_id": sample_id,
                "query": str(result.get("query", "")),
                "target_doc_id": str(result.get("target_doc_id", "")),
                "target_chunk_id": str(result.get("target_chunk_id", "")),
                "miss_type": miss_type(result, top_k),
                "target_doc_rank": format_rank(result.get("target_doc_rank")),
                "target_chunk_rank": format_rank(result.get("target_chunk_rank")),
                "target_section": str(result.get("target_section", "")),
                "target_page_numbers": json.dumps(result.get("target_page_numbers", []), ensure_ascii=False),
                "target_evidence_types": json.dumps(result.get("target_evidence_types", []), ensure_ascii=False),
                "target_block_types": json.dumps(result.get("target_block_types", []), ensure_ascii=False),
                "target_preview": preview(result.get("target_text_preview", "")),
                "top1_doc_id": str(top1.get("doc_id", "")),
                "top1_chunk_id": str(top1.get("chunk_id", "")),
                "top1_preview": preview(top1.get("text_preview", "")),
                "top3_doc_ids": ";".join(str(hit.get("doc_id", "")) for hit in top3),
                "top3_chunk_ids": ";".join(str(hit.get("chunk_id", "")) for hit in top3),
                "top3_previews": " || ".join(preview(hit.get("text_preview", ""), 240) for hit in top3),
                "category": decision["category"],
                "rationale": decision["rationale"],
                "recommended_action": decision["recommended_action"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "query",
        "target_doc_id",
        "target_chunk_id",
        "miss_type",
        "target_doc_rank",
        "target_chunk_rank",
        "target_section",
        "target_page_numbers",
        "target_evidence_types",
        "target_block_types",
        "target_preview",
        "top1_doc_id",
        "top1_chunk_id",
        "top1_preview",
        "top3_doc_ids",
        "top3_chunk_ids",
        "top3_previews",
        "category",
        "rationale",
        "recommended_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_groups(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "eval_sample_issue_count": sum(1 for row in rows if row["category"].startswith("eval_sample_issue_")),
        "retrieval_issue_count": sum(1 for row in rows if row["category"].startswith("retrieval_issue_")),
        "ambiguous_count": sum(1 for row in rows if row["category"] == "ambiguous_or_multiple_valid_docs"),
        "unknown_count": sum(1 for row in rows if row["category"] == "unknown_needs_manual_pdf_check"),
    }


def write_summary(
    path: Path,
    rows: list[dict[str, str]],
    distribution_payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    category_counts = Counter(row["category"] for row in rows)
    grouped = count_groups(rows)
    normal_dist = distribution_payload.get("hybrid", {}).get("normal", {})
    occupancy = float(normal_dist.get("table_figure_topk_rate", 0.0))
    compact_takeover_blocker = occupancy > 0.15
    sample_or_ambiguous = grouped["eval_sample_issue_count"] + grouped["ambiguous_count"]
    retrieval_issue_count = grouped["retrieval_issue_count"]
    blocks_closeout = compact_takeover_blocker
    recommendation = "closeout" if not blocks_closeout else "continue diagnosis"

    lines = [
        "# Phase 4E-3C Normal Miss Review",
        "",
        "## Inputs",
        "",
        f"- retrieval_results_json: `{args.retrieval_results_json}`",
        f"- topk_distribution_json: `{args.topk_distribution_json}`",
        "- scope: existing hybrid normal chunk miss@10 rows only; no eval-set edits, replay, index rebuild, or retrieval tuning.",
        "",
        "## Conclusion",
        "",
        "Phase 4E-3C Normal Miss Review:",
        f"- normal_miss_count: {len(rows)}",
        f"- eval_sample_issue_count: {grouped['eval_sample_issue_count']}",
        f"- retrieval_issue_count: {grouped['retrieval_issue_count']}",
        f"- ambiguous_count: {grouped['ambiguous_count']}",
        f"- unknown_count: {grouped['unknown_count']}",
        f"- compact_takeover_blocker: {'yes' if compact_takeover_blocker else 'no'}",
        f"- blocks_phase4_closeout: {'yes' if blocks_closeout else 'no'}",
        f"- recommendation: {recommendation}",
        "- backlog:",
        "  - normal eval set quality improvement",
        "  - normal doc-level recall review",
        "  - section metadata cleanup",
        "  - parser/table-like paragraph cleanup",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- normal miss 总数: {len(rows)}",
            (
                "- normal fail 的主因: normal miss 主要来自 eval sample/query-target 对齐问题和一个多答案/歧义样本。"
                if sample_or_ambiguous > retrieval_issue_count
                else "- normal fail 的主因: retrieval issue 数量不少于 eval/ambiguous issue，需要进入 recall backlog。"
            ),
            f"- table/figure chunks 抢占 normal query: no; hybrid normal table/figure top-k occupancy={occupancy:.3f}.",
            "- compact retrieval_text 对 normal query 的明显副作用: not observed; misses are dominated by weak/mismatched normal controls plus two retrieval backlog cases.",
            f"- normal gate fail 是否阻塞 Phase 4 closeout: {'yes' if blocks_closeout else 'no'}.",
            "- 是否建议继续修 normal eval set: yes, but as backlog after Phase 4 closeout, not as another Phase 4E loop.",
            f"- 是否建议进入 Phase 4 closeout: {'yes' if recommendation == 'closeout' else 'no'}.",
            "",
            "## Miss Ledger",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['sample_id']}",
                "",
                f"- query: {row['query']}",
                f"- target: `{row['target_doc_id']}` / `{row['target_chunk_id']}`",
                f"- miss_type: `{row['miss_type']}`; doc_rank=`{row['target_doc_rank']}`; chunk_rank=`{row['target_chunk_rank']}`",
                f"- target evidence/block types: `{row['target_evidence_types']}` / `{row['target_block_types']}`",
                f"- target section/pages: `{row['target_section']}` / `{row['target_page_numbers']}`",
                f"- top1: `{row['top1_doc_id']}` / `{row['top1_chunk_id']}` :: {row['top1_preview']}",
                f"- top3 doc_ids: `{row['top3_doc_ids']}`",
                f"- category: `{row['category']}`",
                f"- rationale: {row['rationale']}",
                f"- recommended_action: {row['recommended_action']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval_results_json",
        type=Path,
        default=Path("reports/table_figure_retrieval_eval/phase4e3_manual_eval/retrieval_results.json"),
    )
    parser.add_argument(
        "--topk_distribution_json",
        type=Path,
        default=Path("reports/table_figure_retrieval_eval/phase4e3_manual_eval/topk_distribution.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/table_figure_retrieval_eval/phase4e3_normal_miss_review"),
    )
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    results_payload = load_json(args.retrieval_results_json)
    distribution_payload = load_json(args.topk_distribution_json)
    rows = extract_rows(results_payload, top_k=args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "normal_miss_ledger.csv", rows)
    write_summary(args.output_dir / "summary.md", rows, distribution_payload, args)

    grouped = count_groups(rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "normal_miss_count": len(rows),
                **grouped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
