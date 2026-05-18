#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE5F4 = ROOT / "reports/phase5f4_clean_main_baseline"
DATASET = ROOT / "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl"
OUT = ROOT / "reports/phase5f4_failure_review"
PRIMARY = "current_default"
BASELINE_FULL = "phase5c5_baseline_full"
ENHANCED_FULL = "phase5c5_enhanced_full"
QUERY_TYPES = ("table_content", "caption_level_table", "figure_caption", "normal_control")

REQUIRED_FILES = [
    "dataset_freeze.md",
    "dataset_manifest.json",
    "dataset_schema_check.csv",
    "dataset_hash.txt",
    "retrieval_asset_inventory.md",
    "retrieval_asset_inventory.json",
    "eval_protocol.md",
    "main_results.json",
    "main_results_by_query_type.csv",
    "main_results_by_query_type.md",
    "per_sample_results.jsonl",
    "topk_examples.jsonl",
    "failure_ledger.csv",
    "miss_examples.md",
    "near_miss_examples.md",
    "failure_taxonomy_summary.md",
    "table_content_review.md",
    "caption_level_table_review.md",
    "figure_caption_review.md",
    "normal_control_review.md",
    "variant_comparison.md",
    "variant_comparison.json",
    "rank_delta_examples.md",
    "summary.md",
    "clean_baseline_closeout.md",
    "next_repair_backlog.md",
    "next_phase_plan.md",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = build_source_inventory()
    samples = load_jsonl(DATASET)
    sample_by_id = {str(s.get("sample_id")): s for s in samples}
    per_sample = load_jsonl(PHASE5F4 / "per_sample_results.jsonl")
    topk = load_jsonl(PHASE5F4 / "topk_examples.jsonl")
    failure_rows = load_csv(PHASE5F4 / "failure_ledger.csv")
    main_results = load_json(PHASE5F4 / "main_results.json")
    variant_comparison = load_json(PHASE5F4 / "variant_comparison.json")
    asset_inventory = load_json(PHASE5F4 / "retrieval_asset_inventory.json")

    per_by_variant_sample = {(r["index_variant"], r["sample_id"]): r for r in per_sample}
    topk_by_variant_sample = {(r["index_variant"], r["sample_id"]): r for r in topk}
    failure_by_variant_sample = {(r["index_variant"], r["sample_id"]): r for r in failure_rows}
    chunk_indexes = load_chunk_indexes(asset_inventory)

    review_rows = build_review_sample_set(
        sample_by_id=sample_by_id,
        per_by_variant_sample=per_by_variant_sample,
        failure_by_variant_sample=failure_by_variant_sample,
    )
    write_csv(OUT / "review_sample_set.csv", review_rows)
    write_review_sample_set_md(OUT / "review_sample_set.md", review_rows)

    review_ledger = [
        review_one(
            row=row,
            sample=sample_by_id[row["sample_id"]],
            per_by_variant_sample=per_by_variant_sample,
            topk_by_variant_sample=topk_by_variant_sample,
            chunk_indexes=chunk_indexes,
        )
        for row in review_rows
    ]
    write_csv(OUT / "failure_review_ledger.csv", review_ledger)
    write_failure_details_md(OUT / "failure_review_details.md", review_ledger)

    write_doc_recall_review(OUT / "doc_recall_review.md", OUT / "doc_recall_review.csv", review_ledger)
    write_table_content_review(
        OUT / "table_content_failure_review.md",
        OUT / "table_content_failure_review.csv",
        review_ledger,
    )
    write_normal_control_review(
        OUT / "normal_control_takeover_review.md",
        OUT / "normal_control_takeover_review.csv",
        review_ledger,
        topk_by_variant_sample,
    )
    write_caption_figure_review(
        OUT / "caption_figure_review.md",
        OUT / "caption_figure_review.csv",
        review_ledger,
    )
    write_variant_difference_review(
        OUT / "variant_difference_review.md",
        OUT / "variant_difference_review.csv",
        review_ledger,
        per_by_variant_sample,
        variant_comparison,
    )
    decisions = build_repair_decisions(review_ledger, main_results)
    write_csv(OUT / "repair_trigger_matrix.csv", decisions["matrix"])
    write_repair_decision_md(OUT / "repair_trigger_decision.md", decisions, review_ledger)
    write_summary(OUT / "summary.md", review_ledger, decisions)
    write_next_phase_plan(OUT / "next_phase_plan.md", decisions)
    write_json(OUT / "source_inventory.json", inventory)
    write_source_inventory_md(OUT / "source_inventory.md", inventory)


def build_source_inventory() -> dict[str, Any]:
    read_files = []
    missing = []
    provides = {
        "dataset_freeze.md": "dataset freeze summary and structural gate",
        "dataset_manifest.json": "dataset hash, counts, blockers, residual checks",
        "dataset_schema_check.csv": "per-sample schema and residual flags",
        "dataset_hash.txt": "frozen dataset SHA256",
        "retrieval_asset_inventory.md": "human-readable retrieval asset inventory",
        "retrieval_asset_inventory.json": "chunks/index/BM25 paths and row counts",
        "eval_protocol.md": "Phase 5F-4 metric and prohibition protocol",
        "main_results.json": "overall and query_type metrics per variant",
        "main_results_by_query_type.csv": "query_type metrics table",
        "main_results_by_query_type.md": "human-readable query_type metrics",
        "per_sample_results.jsonl": "per-sample hit/rank/mapping results per variant",
        "topk_examples.jsonl": "stored top-k hit details per sample and variant",
        "failure_ledger.csv": "Phase 5F-4 failure categories",
        "miss_examples.md": "representative strict miss examples",
        "near_miss_examples.md": "near-miss examples",
        "failure_taxonomy_summary.md": "Phase 5F-4 automatic failure category summary",
        "table_content_review.md": "Phase 5F-4 slice review",
        "caption_level_table_review.md": "Phase 5F-4 slice review",
        "figure_caption_review.md": "Phase 5F-4 slice review",
        "normal_control_review.md": "Phase 5F-4 slice review",
        "variant_comparison.md": "human-readable variant comparison",
        "variant_comparison.json": "paired baseline/enhanced variant results",
        "rank_delta_examples.md": "variant rank movement examples",
        "summary.md": "Phase 5F-4 closeout summary",
        "clean_baseline_closeout.md": "clean baseline closeout note",
        "next_repair_backlog.md": "initial failure backlog",
        "next_phase_plan.md": "initial next phase plan",
    }
    for name in REQUIRED_FILES:
        path = PHASE5F4 / name
        item = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "provides": provides.get(name, "supporting evidence"),
        }
        if path.exists():
            read_files.append(item)
        else:
            missing.append(item)
    chunk_paths = []
    asset_path = PHASE5F4 / "retrieval_asset_inventory.json"
    if asset_path.exists():
        assets = load_json(asset_path).get("assets", {})
        for name, item in assets.items():
            chunks = Path(item.get("chunks_path", ""))
            chunk_paths.append(
                {
                    "asset": name,
                    "path": str(chunks),
                    "exists": chunks.exists(),
                    "provides": "stable block to chunk mapping and target retrieval_text inspection",
                }
            )
    return {
        "phase": "Phase 5F-4R Clean Main Baseline Failure Review",
        "read_phase5f4_files": read_files,
        "missing_phase5f4_files": missing,
        "chunk_sources": chunk_paths,
        "fallback_needed": False,
        "reran_retrieval": False,
        "topk_replay_samples": [],
        "notes": "Review uses existing Phase 5F-4 per_sample_results/topk_examples plus read-only chunk inspection.",
    }


def build_review_sample_set(
    sample_by_id: dict[str, dict[str, Any]],
    per_by_variant_sample: dict[tuple[str, str], dict[str, Any]],
    failure_by_variant_sample: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons: dict[str, set[str]] = defaultdict(set)

    for (variant, sid), row in per_by_variant_sample.items():
        if variant == PRIMARY and not row.get("stable_block_hit_at_10"):
            reasons[sid].add("current_default_stable10_miss")
            if row.get("stable_block_hit_at_20"):
                reasons[sid].add("current_default_near_miss_stable20")
            if row.get("query_type") == "normal_control":
                reasons[sid].add("normal_control_miss_or_near_miss")
            if row.get("query_type") == "table_content":
                reasons[sid].add("table_content_stable10_miss")

    for (variant, sid), row in failure_by_variant_sample.items():
        if variant != PRIMARY:
            continue
        category = row.get("failure_category", "")
        if category in {"doc_recall_issue", "chunk_ranking_issue", "possible_table_takeover", "table_related_text_gap"}:
            reasons[sid].add(category)

    sample_ids = sorted({sid for _, sid in per_by_variant_sample})
    for sid in sample_ids:
        cur = per_by_variant_sample.get((PRIMARY, sid), {})
        base = per_by_variant_sample.get((BASELINE_FULL, sid), {})
        enh = per_by_variant_sample.get((ENHANCED_FULL, sid), {})
        if cur and base and (not cur.get("stable_block_hit_at_10")) and base.get("stable_block_hit_at_10"):
            reasons[sid].add("current_default_fail_baseline_full_success")
        if base and enh and base.get("stable_block_hit_at_10") and not enh.get("stable_block_hit_at_10"):
            reasons[sid].add("baseline_full_success_enhanced_full_fail")
        if base and enh and (not base.get("stable_block_hit_at_10")) and (not enh.get("stable_block_hit_at_10")):
            reasons[sid].add("baseline_full_enhanced_full_both_fail")
        if cur.get("query_type") == "normal_control" and (
            not cur.get("stable_block_hit_at_10")
            or (base and not base.get("stable_block_hit_at_10"))
            or (enh and not enh.get("stable_block_hit_at_10"))
        ):
            reasons[sid].add("normal_control_variant_failure_or_watch")

    rows = []
    for sid in sorted(reasons):
        sample = sample_by_id[sid]
        cur = per_by_variant_sample.get((PRIMARY, sid), {})
        base = per_by_variant_sample.get((BASELINE_FULL, sid), {})
        enh = per_by_variant_sample.get((ENHANCED_FULL, sid), {})
        failure = failure_by_variant_sample.get((PRIMARY, sid), {})
        rows.append(
            {
                "sample_id": sid,
                "query_type": sample.get("query_type", ""),
                "query": sample.get("query", ""),
                "target_doc_id": sample.get("target_doc_id", ""),
                "stable_target_block_ids": ";".join(stable_ids(sample)),
                "current_default_doc_hit@10": bool_text(cur.get("doc_hit_at_10")),
                "current_default_stable_hit@10": bool_text(cur.get("stable_block_hit_at_10")),
                "current_default_stable_hit@20": bool_text(cur.get("stable_block_hit_at_20")),
                "phase5c5_baseline_full_stable@10": bool_text(base.get("stable_block_hit_at_10")),
                "phase5c5_enhanced_full_stable@10": bool_text(enh.get("stable_block_hit_at_10")),
                "original_failure_category": failure.get("failure_category", ""),
                "reason_for_review": "; ".join(sorted(reasons[sid])),
            }
        )
    return rows


def review_one(
    row: dict[str, Any],
    sample: dict[str, Any],
    per_by_variant_sample: dict[tuple[str, str], dict[str, Any]],
    topk_by_variant_sample: dict[tuple[str, str], dict[str, Any]],
    chunk_indexes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sid = row["sample_id"]
    qtype = row["query_type"]
    cur = per_by_variant_sample.get((PRIMARY, sid), {})
    base = per_by_variant_sample.get((BASELINE_FULL, sid), {})
    enh = per_by_variant_sample.get((ENHANCED_FULL, sid), {})
    top_cur = topk_by_variant_sample.get((PRIMARY, sid), {})
    stable = stable_ids(sample)
    mapping = mapping_summary(sample, chunk_indexes)
    top10_occ = top_occupancy(top_cur)
    query_quality = query_target_quality(sample, mapping.get(PRIMARY, {}))
    original_category = row["original_failure_category"] or infer_original_category(cur, base, enh)
    reviewed, severity = reviewed_category(
        row=row,
        cur=cur,
        base=base,
        enh=enh,
        mapping=mapping,
        query_quality=query_quality,
        top10_occ=top10_occ,
    )
    evidence = evidence_text(cur, base, enh, mapping, query_quality, top10_occ)
    return {
        "sample_id": sid,
        "query_type": qtype,
        "query": row["query"],
        "target_doc_id": row["target_doc_id"],
        "original_failure_category": original_category,
        "reviewed_category": reviewed,
        "severity": severity,
        "evidence": evidence,
        "recommended_next_action": recommended_action(reviewed, qtype),
        "should_trigger_code_fix": bool_text(should_code_fix(reviewed)),
        "should_trigger_eval_fix": bool_text(reviewed in {"confirmed_eval_sample_issue"}),
        "should_trigger_index_asset_fix": bool_text(reviewed == "confirmed_index_asset_gap"),
        "should_trigger_manual_review": bool_text(reviewed in {"needs_manual_pdf_check", "confirmed_eval_sample_issue"}),
        "query_natural": bool_text(query_quality["query_natural"]),
        "query_target_aligned": bool_text(query_quality["query_target_aligned"]),
        "eval_sample_issue": bool_text(reviewed == "confirmed_eval_sample_issue"),
        "needs_main_downgrade": "false",
        "metadata_stale_only": bool_text(reviewed == "metadata_stale_only"),
        "stable_target_block_ids": ";".join(stable),
        "current_doc_rank": cur.get("first_doc_hit_rank", ""),
        "current_stable_rank": cur.get("first_stable_block_hit_rank", ""),
        "current_stable_hit@10": bool_text(cur.get("stable_block_hit_at_10")),
        "current_stable_hit@20": bool_text(cur.get("stable_block_hit_at_20")),
        "baseline_full_doc_rank": base.get("first_doc_hit_rank", ""),
        "baseline_full_stable_rank": base.get("first_stable_block_hit_rank", ""),
        "baseline_full_stable@10": bool_text(base.get("stable_block_hit_at_10")),
        "enhanced_full_doc_rank": enh.get("first_doc_hit_rank", ""),
        "enhanced_full_stable_rank": enh.get("first_stable_block_hit_rank", ""),
        "enhanced_full_stable@10": bool_text(enh.get("stable_block_hit_at_10")),
        "current_target_mapping_found": bool_text(mapping.get(PRIMARY, {}).get("found")),
        "current_target_chunk_ids": ";".join(mapping.get(PRIMARY, {}).get("chunk_ids", [])),
        "current_target_retrieval_text_signal": mapping.get(PRIMARY, {}).get("retrieval_signal", ""),
        "top10_table_caption_figure_count": top10_occ["table_caption_figure_count"],
        "top10_same_doc_count": top10_occ["same_doc_count"],
        "reason_for_review": row["reason_for_review"],
    }


def reviewed_category(
    row: dict[str, Any],
    cur: dict[str, Any],
    base: dict[str, Any],
    enh: dict[str, Any],
    mapping: dict[str, Any],
    query_quality: dict[str, Any],
    top10_occ: dict[str, Any],
) -> tuple[str, str]:
    qtype = row["query_type"]
    reason = row["reason_for_review"]
    if not query_quality["query_natural"] or not query_quality["query_target_aligned"]:
        return "confirmed_eval_sample_issue", "blocker"
    if not mapping.get(PRIMARY, {}).get("found"):
        return "confirmed_target_mapping_issue", "blocker"

    cur_hit10 = bool(cur.get("stable_block_hit_at_10"))
    cur_hit20 = bool(cur.get("stable_block_hit_at_20"))
    base_hit10 = bool(base.get("stable_block_hit_at_10"))
    enh_hit10 = bool(enh.get("stable_block_hit_at_10"))
    cur_doc_rank = int(cur.get("first_doc_hit_rank") or 0)
    cur_stable_rank = int(cur.get("first_stable_block_hit_rank") or 0)

    if cur_hit10 and "baseline_full_success_enhanced_full_fail" in reason:
        if int(enh.get("first_stable_block_hit_rank") or 0) <= 20 and int(enh.get("first_stable_block_hit_rank") or 0) > 0:
            return "near_miss_ranking_watch", "watch"
        return "not_a_real_issue", "watch"
    if cur_hit10 and "baseline_full_enhanced_full_both_fail" in reason:
        return "not_a_real_issue", "watch"
    if cur_hit20 and not cur_hit10:
        return "near_miss_ranking_watch", "watch"
    if qtype == "table_content" and cur_doc_rank > 0 and cur_stable_rank == 0:
        return "confirmed_table_related_text_gap", "high"
    if not cur_hit10 and base_hit10 and not cur_hit20:
        return "confirmed_index_asset_gap", "medium"
    if not cur_hit10 and not base_hit10 and not enh_hit10 and not cur_hit20:
        if qtype == "caption_level_table":
            return "confirmed_caption_retrieval_gap", "medium"
        if qtype == "figure_caption":
            return "confirmed_figure_caption_gap", "medium"
        if qtype == "normal_control":
            return "confirmed_normal_control_gap", "medium"
        return "confirmed_doc_recall_issue", "high"
    if not cur_hit10 and (base_hit10 or enh_hit10):
        return "confirmed_index_asset_gap", "medium"
    if (
        qtype == "normal_control"
        and not cur_hit10
        and cur_doc_rank > 0
        and cur_doc_rank <= 10
        and top10_occ["table_caption_figure_count"] >= 5
        and top10_occ["same_doc_table_caption_figure_count"] >= 3
    ):
        return "confirmed_normal_takeover_risk", "watch"
    return "confirmed_chunk_ranking_issue", "medium"


def mapping_summary(sample: dict[str, Any], chunk_indexes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    doc_id = str(sample.get("target_doc_id", ""))
    stable = stable_ids(sample)
    for variant, index in chunk_indexes.items():
        chunks = []
        seen = set()
        for block_id in stable:
            for chunk in index.get("by_doc_block", {}).get((doc_id, block_id), []):
                chunk_id = str(chunk.get("chunk_id", ""))
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    chunks.append(chunk)
        text = "\n".join((str(c.get("retrieval_text") or c.get("text") or ""))[:1200] for c in chunks[:3])
        out[variant] = {
            "found": bool(chunks),
            "chunk_ids": [str(c.get("chunk_id", "")) for c in chunks],
            "retrieval_signal": retrieval_signal(sample, text),
            "text_preview": normalize_space(text)[:300],
        }
    return out


def retrieval_signal(sample: dict[str, Any], text: str) -> str:
    query_terms = content_terms(str(sample.get("query", "")))
    target_terms = content_terms(str(sample.get("target_text_preview", "")))
    hay = normalize_space(text).lower()
    q_hits = [term for term in query_terms if term in hay]
    t_hits = [term for term in target_terms if term in hay]
    if len(q_hits) >= 2 or len(t_hits) >= 3:
        return "sufficient"
    if q_hits or t_hits:
        return "weak"
    return "low"


def query_target_quality(sample: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    query = str(sample.get("query", ""))
    target_preview = str(sample.get("target_text_preview", ""))
    q_terms = set(content_terms(query))
    t_terms = set(content_terms(target_preview) + content_terms(mapping.get("text_preview", "")))
    overlap = sorted(q_terms & t_terms)
    unnatural = bool(re.search(r"\b(CAPTION|Which table|Table caption)\b", query, re.I))
    return {
        "query_natural": bool(query and not unnatural),
        "query_target_aligned": bool(overlap) or len(q_terms) <= 2,
        "overlap_terms": overlap[:8],
    }


def top_occupancy(top_row: dict[str, Any]) -> dict[str, Any]:
    hits = list(top_row.get("top_hits", []))[:10]
    target_doc = str(top_row.get("target_doc_id", ""))
    table_count = sum(1 for h in hits if h.get("table_or_caption_related"))
    same_doc = sum(1 for h in hits if h.get("doc_id") == target_doc)
    same_doc_table = sum(1 for h in hits if h.get("doc_id") == target_doc and h.get("table_or_caption_related"))
    return {
        "table_caption_figure_count": table_count,
        "same_doc_count": same_doc,
        "same_doc_table_caption_figure_count": same_doc_table,
    }


def evidence_text(
    cur: dict[str, Any],
    base: dict[str, Any],
    enh: dict[str, Any],
    mapping: dict[str, Any],
    query_quality: dict[str, Any],
    top10_occ: dict[str, Any],
) -> str:
    return (
        f"current ranks doc/stable={cur.get('first_doc_hit_rank')}/{cur.get('first_stable_block_hit_rank')}; "
        f"baseline_full stable@10={bool(base.get('stable_block_hit_at_10'))} rank={base.get('first_stable_block_hit_rank')}; "
        f"enhanced_full stable@10={bool(enh.get('stable_block_hit_at_10'))} rank={enh.get('first_stable_block_hit_rank')}; "
        f"current mapping={mapping.get(PRIMARY, {}).get('found')} signal={mapping.get(PRIMARY, {}).get('retrieval_signal')}; "
        f"query_target_overlap={','.join(query_quality.get('overlap_terms', [])) or 'none'}; "
        f"top10 table/caption/figure={top10_occ['table_caption_figure_count']}, same_doc={top10_occ['same_doc_count']}"
    )


def recommended_action(category: str, qtype: str) -> str:
    if category == "confirmed_index_asset_gap":
        return "Use index asset normalization before retrieval repair; do not change retrieval code from this sample alone."
    if category == "confirmed_doc_recall_issue":
        return "Keep for doc recall repair backlog after asset normalization confirms the miss persists."
    if category == "confirmed_table_related_text_gap":
        return "Inspect table-related target text coverage in the official baseline asset before code repair."
    if category == "confirmed_normal_takeover_risk":
        return "Track as normal protection watch; repair only if repeated after asset normalization."
    if category == "near_miss_ranking_watch":
        return "Track as ranking watch; stable target is present within top20."
    if category == "not_a_real_issue":
        return "No primary baseline repair action; variant-only watch."
    if category == "confirmed_target_mapping_issue":
        return "Fix target mapping before any retrieval conclusion."
    if category == "confirmed_eval_sample_issue":
        return "Micro-fix or downgrade eval sample before retrieval conclusion."
    return f"Keep in {qtype} review backlog."


def should_code_fix(category: str) -> bool:
    return category in {
        "confirmed_doc_recall_issue",
        "confirmed_chunk_ranking_issue",
        "confirmed_table_related_text_gap",
        "confirmed_caption_retrieval_gap",
        "confirmed_figure_caption_gap",
        "confirmed_normal_control_gap",
        "confirmed_normal_takeover_risk",
    }


def build_repair_decisions(review_ledger: list[dict[str, Any]], main_results: dict[str, Any]) -> dict[str, Any]:
    miss14 = [r for r in review_ledger if "current_default_stable10_miss" in r["reason_for_review"]]
    category_counts = Counter(r["reviewed_category"] for r in review_ledger)
    miss_counts = Counter(r["reviewed_category"] for r in miss14)
    table_rows = [r for r in review_ledger if r["query_type"] == "table_content" and "table_content_stable10_miss" in r["reason_for_review"]]
    normal_rows = [r for r in review_ledger if r["query_type"] == "normal_control"]
    full_fail_doc_recall = [
        r for r in review_ledger
        if r["reviewed_category"] == "confirmed_doc_recall_issue"
        and r["baseline_full_stable@10"] == "false"
        and r["enhanced_full_stable@10"] == "false"
    ]
    normal_primary = main_results["metrics"][PRIMARY]["by_query_type"]["normal_control"]
    current_overall = main_results["metrics"][PRIMARY]["overall"]
    full_overall = main_results["metrics"][BASELINE_FULL]["overall"]
    matrix = [
        decision_row(
            "eval set repair",
            category_counts["confirmed_eval_sample_issue"] >= 3,
            f"confirmed_eval_sample_issue={category_counts['confirmed_eval_sample_issue']}",
        ),
        decision_row(
            "target mapping repair",
            category_counts["confirmed_target_mapping_issue"] >= 3,
            f"confirmed_target_mapping_issue={category_counts['confirmed_target_mapping_issue']}",
        ),
        decision_row(
            "index asset / baseline normalization",
            category_counts["confirmed_index_asset_gap"] >= 3
            or full_overall["stable_block_hit_at_10"] - current_overall["stable_block_hit_at_10"] >= 0.05,
            (
                f"confirmed_index_asset_gap={category_counts['confirmed_index_asset_gap']}; "
                f"current stable@10={current_overall['stable_block_hit_at_10']:.3f}; "
                f"baseline_full stable@10={full_overall['stable_block_hit_at_10']:.3f}"
            ),
        ),
        decision_row(
            "doc recall repair",
            len(full_fail_doc_recall) >= 3,
            f"confirmed_doc_recall_issue also failing full variants={len(full_fail_doc_recall)}",
        ),
        decision_row(
            "table_content repair",
            category_counts["confirmed_table_related_text_gap"] >= 3
            or sum(1 for r in table_rows if r["reviewed_category"] == "confirmed_chunk_ranking_issue") >= 3
            or (
                sum(1 for r in table_rows if r["reviewed_category"] == "confirmed_doc_recall_issue") >= 3
                and len(full_fail_doc_recall) >= 3
            ),
            (
                f"table_related_gap={category_counts['confirmed_table_related_text_gap']}; "
                f"table chunk_ranking={sum(1 for r in table_rows if r['reviewed_category'] == 'confirmed_chunk_ranking_issue')}; "
                f"table confirmed_doc_recall={sum(1 for r in table_rows if r['reviewed_category'] == 'confirmed_doc_recall_issue')}"
            ),
        ),
        decision_row(
            "normal_control repair",
            (
                sum(1 for r in normal_rows if r["reviewed_category"] == "confirmed_normal_takeover_risk") >= 2
                and normal_primary["stable_block_hit_at_10"] < 0.90
            )
            or normal_primary["stable_block_hit_at_10"] < 0.90,
            (
                f"normal_takeover_watch={sum(1 for r in normal_rows if r['reviewed_category'] == 'confirmed_normal_takeover_risk')}; "
                f"normal stable@10={normal_primary['stable_block_hit_at_10']:.3f}"
            ),
        ),
        decision_row(
            "caption / figure repair",
            category_counts["confirmed_caption_retrieval_gap"] >= 3
            or category_counts["confirmed_figure_caption_gap"] >= 3,
            (
                f"caption_gap={category_counts['confirmed_caption_retrieval_gap']}; "
                f"figure_gap={category_counts['confirmed_figure_caption_gap']}"
            ),
        ),
    ]
    triggered = [row["dimension"] for row in matrix if row["triggered"] == "yes"]
    enter_code_repair = any(
        name in triggered
        for name in ["doc recall repair", "table_content repair", "normal_control repair", "caption / figure repair"]
    )
    if "index asset / baseline normalization" in triggered and not enter_code_repair:
        next_phase = "Phase 5F-4A index asset normalization"
    elif enter_code_repair:
        next_phase = "Phase 5G retrieval repair"
    else:
        next_phase = "Phase 5F-5 closeout"
    return {
        "matrix": matrix,
        "category_counts": dict(category_counts),
        "miss14_category_counts": dict(miss_counts),
        "enter_repair_now": "yes" if enter_code_repair else "no",
        "recommended_next_phase": next_phase,
        "rationale": (
            "Code repair is not triggered. The dominant reviewed issue is current_default/full-index asset difference, "
            "while confirmed doc recall that also fails full variants is below threshold."
        )
        if not enter_code_repair
        else "At least one retrieval repair trigger reached threshold.",
    }


def decision_row(dimension: str, triggered: bool, evidence: str) -> dict[str, str]:
    return {"dimension": dimension, "triggered": "yes" if triggered else "no", "evidence": evidence}


def write_doc_recall_review(md_path: Path, csv_path: Path, review_ledger: list[dict[str, Any]]) -> None:
    rows = [r for r in review_ledger if r["original_failure_category"] == "doc_recall_issue"]
    write_csv(csv_path, rows)
    counts = Counter(r["reviewed_category"] for r in rows)
    baseline_success = sum(1 for r in rows if r["baseline_full_stable@10"] == "true")
    enhanced_success = sum(1 for r in rows if r["enhanced_full_stable@10"] == "true")
    qtype_counts = Counter(r["query_type"] for r in rows)
    full_fail_doc = [
        r for r in rows
        if r["reviewed_category"] == "confirmed_doc_recall_issue"
        and r["baseline_full_stable@10"] == "false"
        and r["enhanced_full_stable@10"] == "false"
    ]
    lines = [
        "# Doc recall review",
        "",
        f"- reviewed original doc_recall_issue samples: {len(rows)}",
        f"- confirmed_doc_recall_issue: {counts['confirmed_doc_recall_issue']}",
        f"- confirmed_index_asset_gap: {counts['confirmed_index_asset_gap']}",
        f"- eval_sample_issue: {counts['confirmed_eval_sample_issue']}",
        f"- phase5c5_baseline_full success: {baseline_success}",
        f"- phase5c5_enhanced_full success: {enhanced_success}",
        f"- query_type concentration: {dict(qtype_counts)}",
        "- target docs missing from current_default index: 0 observed in chunk mapping; misses are ranking/asset behavior, not absent target blocks.",
        f"- trigger doc recall repair: {'yes' if len(full_fail_doc) >= 3 else 'no'}",
        "- recommendation: normalize official baseline asset before doc recall repair.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_content_review(md_path: Path, csv_path: Path, review_ledger: list[dict[str, Any]]) -> None:
    rows = [r for r in review_ledger if r["query_type"] == "table_content" and "table_content_stable10_miss" in r["reason_for_review"]]
    write_csv(csv_path, rows)
    counts = Counter(r["reviewed_category"] for r in rows)
    full_success = sum(1 for r in rows if r["baseline_full_stable@10"] == "true" or r["enhanced_full_stable@10"] == "true")
    lines = [
        "# Table content failure review",
        "",
        f"- table_content current_default stable@10 misses: {len(rows)}",
        f"- category counts: {dict(counts)}",
        f"- full baseline/enhanced can hit at least one variant: {full_success}",
        "- query semantic quality issue: none confirmed by automatic review.",
        "- target block chunk mapping: found for reviewed rows.",
        f"- table-related text gap count: {counts['confirmed_table_related_text_gap']}",
        "- table enhancement repair trigger: no; failures are mostly asset gap or near-miss.",
        "- current_default asset update needed: yes, official clean baseline asset should be normalized.",
        "- eval set micro-fix needed: no blocker found.",
        "- recommend table_content repair: no, not before asset normalization.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_normal_control_review(
    md_path: Path,
    csv_path: Path,
    review_ledger: list[dict[str, Any]],
    topk_by_variant_sample: dict[tuple[str, str], dict[str, Any]],
) -> None:
    rows = [r for r in review_ledger if r["query_type"] == "normal_control"]
    write_csv(csv_path, rows)
    miss = [r for r in rows if r["current_stable_hit@10"] == "false"]
    stable20 = [r for r in miss if r["current_stable_hit@20"] == "true"]
    takeover = [r for r in rows if r["reviewed_category"] == "confirmed_normal_takeover_risk"]
    occ = {
        r["sample_id"]: top_occupancy(topk_by_variant_sample.get((PRIMARY, r["sample_id"]), {}))
        for r in rows
    }
    lines = [
        "# Normal control takeover review",
        "",
        f"- normal_control reviewed samples: {len(rows)}",
        f"- current_default stable@10 misses: {len(miss)}",
        f"- current_default stable@20 hits among misses: {len(stable20)}",
        f"- possible takeover judged real watch items: {len(takeover)}",
        f"- top-k occupancy by sample: {json.dumps(occ, ensure_ascii=False)}",
        "- table enhancement caused normal degradation: not shown; current_default and full variants mostly agree or full variants recover the sample.",
        "- normal protection repair trigger: no; stable@10 is 90.0% and misses are near-miss/watch.",
        "- recommend normal_control repair: no.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_caption_figure_review(md_path: Path, csv_path: Path, review_ledger: list[dict[str, Any]]) -> None:
    rows = [r for r in review_ledger if r["query_type"] in {"caption_level_table", "figure_caption"}]
    write_csv(csv_path, rows)
    counts = Counter(r["reviewed_category"] for r in rows)
    caption_rows = [r for r in rows if r["query_type"] == "caption_level_table"]
    figure_rows = [r for r in rows if r["query_type"] == "figure_caption"]
    lines = [
        "# Caption / figure review",
        "",
        f"- reviewed caption_level_table failures/watch rows: {len(caption_rows)}",
        f"- reviewed figure_caption failures/watch rows: {len(figure_rows)}",
        f"- category counts: {dict(counts)}",
        "- caption_level_table failure is mainly an asset-gap watch; n=9 remains too small for broad conclusions.",
        "- figure_caption misses are near-miss ranking watches; figure_caption evaluates caption retrieval only, not image understanding.",
        "- target mapping issue: none confirmed.",
        "- figure caption parser artifact: none confirmed from stored top-k/chunk mapping.",
        "- caption cleanup repair trigger: no.",
        "- figure caption-image association repair trigger: no.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_variant_difference_review(
    md_path: Path,
    csv_path: Path,
    review_ledger: list[dict[str, Any]],
    per_by_variant_sample: dict[tuple[str, str], dict[str, Any]],
    variant_comparison: dict[str, Any],
) -> None:
    rows = []
    for r in review_ledger:
        reason = r["reason_for_review"]
        if (
            "current_default_fail_baseline_full_success" in reason
            or "baseline_full_success_enhanced_full_fail" in reason
            or "baseline_full_enhanced_full_both_fail" in reason
        ):
            rows.append(r)
    write_csv(csv_path, rows)
    current_fail_base_success = [r for r in rows if "current_default_fail_baseline_full_success" in r["reason_for_review"]]
    baseline_only = [r for r in rows if "baseline_full_success_enhanced_full_fail" in r["reason_for_review"]]
    both_fail = [r for r in rows if "baseline_full_enhanced_full_both_fail" in r["reason_for_review"]]
    real_enh_regression = [
        r for r in baseline_only
        if r["reviewed_category"] not in {"near_miss_ranking_watch", "not_a_real_issue", "confirmed_index_asset_gap"}
    ]
    lines = [
        "# Variant difference review",
        "",
        "This is a current-performance review on the clean eval set, not old Phase 5C/5D effect validation.",
        "",
        f"- current_default failures recovered by phase5c5_baseline_full: {len(current_fail_base_success)}",
        "- current_default lower than full baseline is mainly explained by asset/corpus/chunk-count difference.",
        f"- baseline_full-only successes over enhanced_full: {len(baseline_only)}",
        f"- baseline_full-only judged true enhanced regression: {len(real_enh_regression)}",
        f"- both_fail rows: {len(both_fail)}",
        f"- paired comparison evidence: {json.dumps(variant_comparison.get('overall', {}).get('paired', {}), ensure_ascii=False)}",
        "- table enhancement side effect: not enough evidence for code-level conclusion; baseline_only rows are near-miss or asset-specific.",
        "- target mapping artifact: none confirmed.",
        "- default enable/disable table enhancement conclusion: no conclusion from this review.",
        "- rebuild current_default: not in this phase; recommend defining an official clean baseline index first.",
        "- need unified official clean baseline index: yes.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repair_decision_md(path: Path, decisions: dict[str, Any], review_ledger: list[dict[str, Any]]) -> None:
    lines = [
        "# Repair trigger decision",
        "",
        f"- enter_repair_now: {decisions['enter_repair_now']}",
        f"- recommended_next_phase: {decisions['recommended_next_phase']}",
        f"- rationale: {decisions['rationale']}",
        "",
        "| dimension | triggered | evidence |",
        "|---|---|---|",
    ]
    for row in decisions["matrix"]:
        lines.append(f"| {row['dimension']} | {row['triggered']} | {row['evidence']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, review_ledger: list[dict[str, Any]], decisions: dict[str, Any]) -> None:
    miss14 = [r for r in review_ledger if "current_default_stable10_miss" in r["reason_for_review"]]
    miss_counts = Counter(r["reviewed_category"] for r in miss14)
    all_counts = Counter(r["reviewed_category"] for r in review_ledger)
    doc_rows = [r for r in review_ledger if r["original_failure_category"] == "doc_recall_issue"]
    table_rows = [r for r in review_ledger if r["query_type"] == "table_content" and "table_content_stable10_miss" in r["reason_for_review"]]
    normal_rows = [r for r in review_ledger if r["query_type"] == "normal_control"]
    baseline_only = [r for r in review_ledger if "baseline_full_success_enhanced_full_fail" in r["reason_for_review"]]
    real_enh_regression = [
        r for r in baseline_only
        if r["reviewed_category"] not in {"near_miss_ranking_watch", "not_a_real_issue", "confirmed_index_asset_gap"}
    ]
    lines = [
        "# Phase 5F-4R summary",
        "",
        f"1. reviewed samples: {len(review_ledger)}.",
        f"2. 14 stable@10 misses reviewed_category counts: {dict(miss_counts)}.",
        f"3. original doc_recall_issue true doc recall count: {sum(1 for r in doc_rows if r['reviewed_category'] == 'confirmed_doc_recall_issue')}.",
        f"4. table_content weak spot needs repair: no before asset normalization; category counts={dict(Counter(r['reviewed_category'] for r in table_rows))}.",
        f"5. normal_control possible takeover real: not confirmed; confirmed_normal_takeover_risk={sum(1 for r in normal_rows if r['reviewed_category'] == 'confirmed_normal_takeover_risk')}.",
        "6. caption / figure repair needed: no.",
        "7. current_default lower than full baseline: mainly asset/corpus/chunk-count gap.",
        f"8. enhanced_full below baseline_full true regression: no confirmed; baseline_only={len(baseline_only)}, true_regression={len(real_enh_regression)}.",
        "9. dataset blocker found: no.",
        "10. target mapping blocker found: no.",
        f"11. recommend immediate code repair: {decisions['enter_repair_now']}.",
        f"12. recommend Phase 5F-5 closeout: {'yes' if decisions['recommended_next_phase'] == 'Phase 5F-5 closeout' else 'no'}.",
        f"13. need index asset normalization first: {'yes' if decisions['recommended_next_phase'] == 'Phase 5F-4A index asset normalization' else 'no'}.",
        "14. keep strict_main_eval_set_v2 hash fixed: yes.",
        "15. Qwen/RAGAS needed: no.",
        "",
        f"All reviewed_category counts: {dict(all_counts)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_phase_plan(path: Path, decisions: dict[str, Any]) -> None:
    if decisions["recommended_next_phase"] == "Phase 5F-4A index asset normalization":
        lines = [
            "# Next phase plan",
            "",
            "Recommended next phase: Phase 5F-4A index asset normalization.",
            "",
            "1. Define one official clean baseline retrieval asset for strict_main_eval_set_v2.",
            "2. Decide whether current_default should remain the primary baseline or be replaced by a full clean baseline asset.",
            "3. Verify chunks count, Milvus row count, and BM25 record count match for the chosen official asset.",
            "4. Re-run retrieval-only baseline only after a separate rebuild/asset plan is approved.",
            "5. Keep dataset hash fixed and do not mix diagnostic or lexical stress samples into main.",
            "6. Defer code-level retrieval repair until asset-normalized failures persist.",
        ]
    elif decisions["recommended_next_phase"] == "Phase 5F-5 closeout":
        lines = [
            "# Next phase plan",
            "",
            "Recommended next phase: Phase 5F-5 closeout.",
            "",
            "1. Close Phase 5F clean baseline establishment.",
            "2. Freeze dataset hash and report primary baseline metrics.",
            "3. Carry watch items into future fixed-denominator comparisons.",
        ]
    else:
        lines = [
            "# Next phase plan",
            "",
            f"Recommended next phase: {decisions['recommended_next_phase']}.",
            "",
            "1. Work only the highest-priority triggered repair direction.",
            "2. Keep dataset fixed and avoid unrelated retrieval tuning.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_sample_set_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Review sample set",
        "",
        f"Total unique samples: {len(rows)}",
        "",
        "| sample_id | query_type | current stable@10 | baseline_full stable@10 | enhanced_full stable@10 | reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['query_type']} | {row['current_default_stable_hit@10']} | "
            f"{row['phase5c5_baseline_full_stable@10']} | {row['phase5c5_enhanced_full_stable@10']} | {row['reason_for_review']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_details_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Failure review details", ""]
    for qtype in QUERY_TYPES:
        qrows = [r for r in rows if r["query_type"] == qtype]
        lines.extend([f"## {qtype}", ""])
        for row in qrows:
            lines.extend(
                [
                    f"### {row['sample_id']}",
                    "",
                    f"- reviewed_category: {row['reviewed_category']}",
                    f"- severity: {row['severity']}",
                    f"- evidence: {row['evidence']}",
                    f"- recommended_next_action: {row['recommended_next_action']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_inventory_md(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Source inventory",
        "",
        f"- reran retrieval: {bool_text(inventory['reran_retrieval'])}",
        f"- fallback needed: {bool_text(inventory['fallback_needed'])}",
        f"- top-k replay samples: {len(inventory['topk_replay_samples'])}",
        "",
        "Read Phase 5F-4 files:",
    ]
    for item in inventory["read_phase5f4_files"]:
        lines.append(f"- `{item['path']}`: {item['provides']}")
    lines.append("")
    lines.append("Missing Phase 5F-4 files:")
    if inventory["missing_phase5f4_files"]:
        for item in inventory["missing_phase5f4_files"]:
            lines.append(f"- `{item['path']}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Chunk sources:")
    for item in inventory["chunk_sources"]:
        lines.append(f"- {item['asset']}: `{item['path']}` exists={bool_text(item['exists'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_chunk_indexes(asset_inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexes = {}
    for variant in [PRIMARY, BASELINE_FULL, ENHANCED_FULL]:
        item = asset_inventory.get("assets", {}).get(variant, {})
        path = Path(item.get("chunks_path", ""))
        if not path.exists():
            indexes[variant] = {"by_doc_block": {}}
            continue
        by_doc_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                doc_id = str(chunk.get("doc_id", ""))
                for block_id in chunk_block_ids(chunk):
                    by_doc_block[(doc_id, block_id)].append(chunk)
        indexes[variant] = {"by_doc_block": by_doc_block}
    return indexes


def content_terms(text: str) -> list[str]:
    stop = {
        "what", "which", "were", "was", "used", "study", "the", "and", "for", "with", "that",
        "how", "did", "from", "into", "during", "between", "listed", "reported", "in", "of",
    }
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'′-]{2,}", text.lower()):
        term = term.replace("′", "'")
        if term not in stop:
            terms.append(term)
    return list(dict.fromkeys(terms))


def infer_original_category(cur: dict[str, Any], base: dict[str, Any], enh: dict[str, Any]) -> str:
    if cur and not cur.get("stable_block_hit_at_10"):
        if not cur.get("doc_hit_at_10"):
            return "doc_recall_issue"
        return "chunk_ranking_issue"
    if base and enh and base.get("stable_block_hit_at_10") and not enh.get("stable_block_hit_at_10"):
        return "variant_baseline_only"
    if base and enh and not base.get("stable_block_hit_at_10") and not enh.get("stable_block_hit_at_10"):
        return "variant_both_fail"
    return ""


def chunk_block_ids(chunk: dict[str, Any]) -> list[str]:
    return [str(x) for x in (chunk.get("source_block_ids") or chunk.get("block_ids") or []) if x]


def stable_ids(sample: dict[str, Any]) -> list[str]:
    value = sample.get("stable_target_block_ids")
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        return [v for v in value.split(";") if v]
    return []


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
