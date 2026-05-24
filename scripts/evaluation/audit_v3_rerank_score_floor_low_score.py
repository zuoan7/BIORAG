from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.audit_v3_gold_child_source_to_retrieval import as_list, safe_int  # noqa: E402
from scripts.evaluation.audit_v3_retrieval_drift import (  # noqa: E402
    load_jsonl,
    load_jsonl_by_id,
    parent_chunk_id,
    round_number,
    write_json,
    write_jsonl,
    write_markdown,
)
from src.synbio_rag.application.rerank_common import _rerank_text  # noqa: E402
from src.synbio_rag.application.rerank_features import (  # noqa: E402
    _evidence_aware_bonus,
    _route_bonus,
    _strategy_bonus,
    _structure_marker_bonus,
)
from src.synbio_rag.domain.config import Settings  # noqa: E402
from src.synbio_rag.domain.schemas import RetrievedChunk  # noqa: E402
from src.synbio_rag.infrastructure.vectorstores.bm25 import _tokenize, tokenize_query  # noqa: E402


RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_SOURCE_AUDIT = (
    RESULTS_ROOT
    / "v3_gold_child_source_to_retrieval_audit_20260524_gold_child_source_to_retrieval_audit"
    / "gold_child_source_to_retrieval_samples.jsonl"
)
DEFAULT_REWRITE_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260523_support_selector_retention"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit why rerank_score_floor samples receive low rerank scores."
    )
    parser.add_argument("--source-audit", default=str(DEFAULT_SOURCE_AUDIT))
    parser.add_argument("--rewrite-results", default=str(DEFAULT_REWRITE_RESULTS))
    parser.add_argument("--run-id", default="20260524_rerank_score_floor_low_score_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    settings = Settings.from_env()
    source_rows = [
        row
        for row in load_jsonl(Path(args.source_audit))
        if row.get("loss_stage") == "rerank_score_floor"
    ]
    rewrite_rows = load_jsonl_by_id(Path(args.rewrite_results), "sample_id")
    child_records = load_jsonl_by_id(Path(settings.kb.child_chunk_jsonl), "chunk_id")
    parent_records = load_jsonl_by_id(Path(settings.kb.parent_chunk_jsonl), "chunk_id")

    samples = [
        audit_sample(
            row=row,
            rewrite_row=rewrite_rows.get(str(row.get("sample_id") or "")) or {},
            child_records=child_records,
            parent_records=parent_records,
            settings=settings,
        )
        for row in source_rows
    ]
    summary = build_summary(args.run_id, samples, args.source_audit, args.rewrite_results)
    result_dir = RESULTS_ROOT / f"v3_rerank_score_floor_low_score_audit_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_rerank_score_floor_low_score_audit_{args.run_id}"
    write_json(result_dir / "rerank_score_floor_low_score_summary.json", summary)
    write_jsonl(result_dir / "rerank_score_floor_low_score_samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(json.dumps({"result_dir": str(result_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def audit_sample(
    *,
    row: dict[str, Any],
    rewrite_row: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    gold_children = as_list(row.get("gold_child_chunk_ids"))
    gold_parents = as_list(row.get("gold_parent_chunk_ids"))
    expected_docs = set(as_list(row.get("expected_doc_ids")))
    question = str(row.get("question") or rewrite_row.get("question") or "")
    debug = rewrite_row.get("debug_digest") or {}
    rerank_hits = debug.get("rerank_hits") or {}
    selection = rerank_hits.get("selection") or {}
    ranking_trace = rerank_hits.get("ranking_trace") or []
    retrieval_output = debug.get("retrieval_output") or {}
    matched_by_parent = retrieval_output.get("matched_child_chunk_ids_by_chunk_id") or {}

    target = best_target_trace(ranking_trace, set(gold_parents))
    top = best_top_trace(ranking_trace)
    target_parent = str(target.get("parent_chunk_id") or target.get("chunk_id") or "")
    top_parent = str(top.get("parent_chunk_id") or top.get("chunk_id") or "")
    target_chunk = build_chunk(
        trace_item=target,
        parent_records=parent_records,
        child_records=child_records,
        matched_child_ids=as_list(matched_by_parent.get(target_parent)),
    )
    top_chunk = build_chunk(
        trace_item=top,
        parent_records=parent_records,
        child_records=child_records,
        matched_child_ids=as_list(matched_by_parent.get(top_parent)),
    )
    target_text = _rerank_text(target_chunk) if target_chunk else ""
    top_text = _rerank_text(top_chunk) if top_chunk else ""
    target_features = score_features(question, target, target_chunk, target_text, settings)
    top_features = score_features(question, top, top_chunk, top_text, settings)
    floor_debug = selection.get("score_floor") or {}
    floor = safe_float(floor_debug.get("floor"))
    target_score = safe_float(target.get("score"))
    top_score = safe_float(floor_debug.get("top_score")) or safe_float(top.get("score"))
    target_matched = as_list(matched_by_parent.get(target_parent))
    reason = classify_reason(
        question=question,
        target=target,
        top=top,
        target_features=target_features,
        top_features=top_features,
        target_matched_child_ids=target_matched,
        gold_children=gold_children,
        expected_docs=expected_docs,
        gold_parents=set(gold_parents),
    )
    return {
        "sample_id": sample_id,
        "category": row.get("category"),
        "expected_route": row.get("expected_route"),
        "question": question,
        "rewritten_query": row.get("rewritten_query"),
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "source_status": (row.get("source_status") or {}).get("status"),
        "index_status": (row.get("index_status") or {}).get("status"),
        "score_floor": {
            "ratio": floor_debug.get("ratio"),
            "top_score": round_number(top_score),
            "floor": round_number(floor),
            "target_score": round_number(target_score),
            "target_minus_floor": round_number(target_score - floor) if floor is not None and target_score is not None else None,
            "top_minus_target": round_number(top_score - target_score) if top_score is not None and target_score is not None else None,
        },
        "target": {
            "trace": compact_trace(target),
            "matched_child_chunk_ids": target_matched,
            "matched_gold_child": bool(set(target_matched) & set(gold_children)) if gold_children else None,
            "features": target_features,
            "text_preview": preview(target_text, 520),
        },
        "top_competitor": {
            "trace": compact_trace(top),
            "is_expected_doc_wrong_parent": (
                str(top.get("doc_id") or "") in expected_docs
                and str(top.get("parent_chunk_id") or parent_chunk_id(top.get("chunk_id"))) not in set(gold_parents)
            ),
            "features": top_features,
            "text_preview": preview(top_text, 520),
        },
        "low_score_reason": reason["reason"],
        "reason_tags": reason["tags"],
        "interpretation": reason["interpretation"],
    }


def best_target_trace(trace: list[dict[str, Any]], gold_parents: set[str]) -> dict[str, Any]:
    targets = [
        item
        for item in trace
        if str(item.get("parent_chunk_id") or parent_chunk_id(item.get("chunk_id"))) in gold_parents
    ]
    targets.sort(key=lambda item: safe_int(item.get("pre_floor_rerank_rank")) or 999999)
    return targets[0] if targets else {}


def best_top_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trace, key=lambda item: safe_int(item.get("pre_floor_rerank_rank")) or 999999)
    return ordered[0] if ordered else {}


def build_chunk(
    *,
    trace_item: dict[str, Any],
    parent_records: dict[str, dict[str, Any]],
    child_records: dict[str, dict[str, Any]],
    matched_child_ids: list[str],
) -> RetrievedChunk | None:
    chunk_id = str(trace_item.get("chunk_id") or "")
    parent_id = str(trace_item.get("parent_chunk_id") or parent_chunk_id(chunk_id))
    record = parent_records.get(parent_id) or child_records.get(chunk_id) or {}
    if not record and not trace_item:
        return None
    metadata = {
        "chunk_index": record.get("chunk_index"),
        "retrieval_text": record.get("retrieval_text", ""),
        "matched_child_chunk_ids": matched_child_ids,
        "matched_child_snippets": [
            child_snippet(child_records[child_id])
            for child_id in matched_child_ids
            if child_id in child_records
        ],
    }
    return RetrievedChunk(
        chunk_id=parent_id or chunk_id,
        doc_id=str(record.get("doc_id") or trace_item.get("doc_id") or ""),
        source_file=str(record.get("source_file") or trace_item.get("source_file") or ""),
        title=str(record.get("title") or ""),
        section=str(record.get("section") or trace_item.get("section") or ""),
        text=str(record.get("text") or ""),
        page_start=safe_int(record.get("page_start")),
        page_end=safe_int(record.get("page_end")),
        vector_score=safe_float(trace_item.get("vector_score")) or 0.0,
        bm25_score=safe_float(trace_item.get("bm25_score")) or 0.0,
        rerank_score=safe_float(trace_item.get("score")) or 0.0,
        fusion_score=safe_float(trace_item.get("fusion_score")) or 0.0,
        metadata=metadata,
    )


def child_snippet(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": record.get("chunk_id"),
        "text": record.get("text", ""),
        "child_index": record.get("child_index"),
        "child_start_token": record.get("child_start_token"),
        "child_end_token": record.get("child_end_token"),
        "block_types": record.get("block_types") or [],
        "evidence_types": record.get("evidence_types") or [],
        "contains_table_caption": bool(record.get("contains_table_caption")),
        "contains_table_text": bool(record.get("contains_table_text")),
        "contains_figure_caption": bool(record.get("contains_figure_caption")),
        "contains_image": bool(record.get("contains_image")),
    }


def score_features(
    question: str,
    trace_item: dict[str, Any],
    chunk: RetrievedChunk | None,
    rerank_text: str,
    settings: Settings,
) -> dict[str, Any]:
    query_scores = [safe_float(value) for value in trace_item.get("query_scores") or []]
    query_scores = [value for value in query_scores if value is not None]
    max_query = max(query_scores) if query_scores else None
    mean_query = sum(query_scores) / len(query_scores) if query_scores else None
    bonuses = {}
    if chunk is not None:
        bonuses = {
            "strategy": _strategy_bonus(question, chunk, settings.retrieval),
            "route": _route_bonus(question, chunk, settings.retrieval),
            "evidence": _evidence_aware_bonus(chunk, settings.retrieval),
            "structure": _structure_marker_bonus(question, chunk, settings.retrieval),
        }
    total_bonus = sum(float(value or 0.0) for value in bonuses.values())
    lexical = lexical_overlap(question, rerank_text)
    query_score_component = (
        max_query + settings.retrieval.rerank_subquery_aggregate_alpha * mean_query
        if max_query is not None and mean_query is not None
        else None
    )
    return {
        "query_scores": [round_number(value) for value in query_scores],
        "max_query_score": round_number(max_query),
        "mean_query_score": round_number(mean_query),
        "query_score_component": round_number(query_score_component),
        "bonuses": {key: round_number(value) for key, value in bonuses.items()},
        "total_bonus": round_number(total_bonus),
        "lexical_overlap": lexical,
        "text_chars": len(rerank_text),
        "contains_table_marker": "[table" in rerank_text.lower(),
        "contains_figure_marker": "[figure" in rerank_text.lower(),
    }


def lexical_overlap(question: str, text: str) -> dict[str, Any]:
    query_terms = tokenize_query(question)
    if not query_terms:
        query_terms = _tokenize(question)
    text_terms = set(_tokenize(text))
    overlap = [term for term in query_terms if term in text_terms]
    query_symbols = extract_symbols(question)
    text_lower = text.lower()
    symbol_hits = [symbol for symbol in query_symbols if symbol.lower() in text_lower]
    return {
        "query_term_count": len(query_terms),
        "overlap_count": len(overlap),
        "overlap_ratio": round_number(len(overlap) / len(query_terms)) if query_terms else 0.0,
        "overlap_terms": overlap[:30],
        "query_symbol_count": len(query_symbols),
        "symbol_hit_count": len(symbol_hits),
        "symbol_hits": symbol_hits[:20],
    }


def extract_symbols(text: str) -> list[str]:
    pattern = re.compile(r"[A-Za-z]*\d+[A-Za-z0-9'′α-ωΑ-Ω._-]*|[A-Za-z]{2,}\d*[A-Za-z0-9'′._-]*")
    seen = set()
    result = []
    for match in pattern.finditer(text):
        value = match.group(0).strip(".,;:()[]{}").lower()
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def classify_reason(
    *,
    question: str,
    target: dict[str, Any],
    top: dict[str, Any],
    target_features: dict[str, Any],
    top_features: dict[str, Any],
    target_matched_child_ids: list[str],
    gold_children: list[str],
    expected_docs: set[str],
    gold_parents: set[str],
) -> dict[str, Any]:
    tags = []
    target_raw = safe_float(target_features.get("max_query_score"))
    top_raw = safe_float(top_features.get("max_query_score"))
    target_bonus = safe_float(target_features.get("total_bonus")) or 0.0
    top_bonus = safe_float(top_features.get("total_bonus")) or 0.0
    target_overlap = (target_features.get("lexical_overlap") or {}).get("overlap_count") or 0
    top_overlap = (top_features.get("lexical_overlap") or {}).get("overlap_count") or 0
    target_symbols = (target_features.get("lexical_overlap") or {}).get("symbol_hit_count") or 0
    top_symbols = (top_features.get("lexical_overlap") or {}).get("symbol_hit_count") or 0
    top_parent = str(top.get("parent_chunk_id") or parent_chunk_id(top.get("chunk_id")))

    if gold_children and not (set(target_matched_child_ids) & set(gold_children)):
        tags.append("target_materialized_without_gold_child_snippet")
    if not gold_children:
        tags.append("parent_only_gold_label_no_child_level_text")
    if top_raw is not None and target_raw is not None and top_raw - target_raw >= 1.0:
        tags.append("reranker_raw_score_gap_vs_top")
    if top_bonus - target_bonus >= 0.4:
        tags.append("top_competitor_bonus_advantage")
    if top_overlap > target_overlap or top_symbols > target_symbols:
        tags.append("top_competitor_stronger_query_overlap")
    if str(top.get("doc_id") or "") in expected_docs and top_parent not in gold_parents:
        tags.append("same_expected_doc_wrong_parent_is_top")
    if contains_cjk(question):
        tags.append("cjk_rerank_query_against_english_evidence")

    if "target_materialized_without_gold_child_snippet" in tags:
        reason = "target_rerank_text_missing_gold_child"
        interpretation = "目标父块进了 rerank，但实际用于 rerank 的 focused text 没有包含 gold child snippet。"
    elif "reranker_raw_score_gap_vs_top" in tags and "same_expected_doc_wrong_parent_is_top" in tags:
        reason = "reranker_prefers_same_doc_wrong_parent"
        interpretation = "同一 expected doc 的错误父块获得明显更高 raw reranker score，score floor 只是把这个偏好硬化。"
    elif "reranker_raw_score_gap_vs_top" in tags:
        reason = "reranker_raw_semantic_score_low"
        interpretation = "target 的模型原始相关性分数显著低于 top candidate，不主要是加分项缺失。"
    elif "top_competitor_bonus_advantage" in tags:
        reason = "bonus_gap_amplified_floor"
        interpretation = "top candidate 的 evidence/section/structure bonus 明显更高，抬高 top_score 和 floor。"
    elif "top_competitor_stronger_query_overlap" in tags:
        reason = "query_overlap_weaker_than_top"
        interpretation = "target 的查询词/符号覆盖弱于 top candidate，reranker 给低分有词面证据。"
    else:
        reason = "relative_floor_too_aggressive_for_moderate_target"
        interpretation = "target 分数不是链路缺失，但相对 top_score 的 0.4 floor 过于严格。"
    return {"reason": reason, "tags": tags, "interpretation": interpretation}


def compact_trace(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item.get("chunk_id"),
        "parent_chunk_id": item.get("parent_chunk_id"),
        "doc_id": item.get("doc_id"),
        "section": item.get("section"),
        "raw_retrieval_rank": item.get("raw_retrieval_rank"),
        "pre_floor_rerank_rank": item.get("pre_floor_rerank_rank"),
        "post_floor_rank": item.get("post_floor_rank"),
        "final_top10_rank": item.get("final_top10_rank"),
        "score": item.get("score"),
        "query_scores": item.get("query_scores") or [],
        "vector_score": item.get("vector_score"),
        "bm25_score": item.get("bm25_score"),
        "fusion_score": item.get("fusion_score"),
        "final_drop_reason": item.get("final_drop_reason"),
    }


def build_summary(
    run_id: str,
    samples: list[dict[str, Any]],
    source_audit: str,
    rewrite_results: str,
) -> dict[str, Any]:
    reason_counts = Counter(sample["low_score_reason"] for sample in samples)
    tag_counts = Counter(tag for sample in samples for tag in sample.get("reason_tags") or [])
    category_counts = Counter(str(sample.get("category") or "unknown") for sample in samples)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "samples with loss_stage == rerank_score_floor in gold-child source-to-retrieval audit",
        "inputs": {
            "source_audit": source_audit,
            "rewrite_results": rewrite_results,
        },
        "target_sample_count": len(samples),
        "reason_counts": dict(reason_counts),
        "tag_counts": dict(tag_counts),
        "category_counts": dict(category_counts),
        "dominant_reason": reason_counts.most_common(1)[0][0] if reason_counts else "",
        "score_floor_gap_stats": score_gap_stats(samples),
        "floor_only_rescue_upper_bound": floor_only_rescue_upper_bound(samples),
    }


def score_gap_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    target_minus_floor = [
        float((sample.get("score_floor") or {}).get("target_minus_floor"))
        for sample in samples
        if (sample.get("score_floor") or {}).get("target_minus_floor") is not None
    ]
    top_minus_target = [
        float((sample.get("score_floor") or {}).get("top_minus_target"))
        for sample in samples
        if (sample.get("score_floor") or {}).get("top_minus_target") is not None
    ]
    return {
        "target_minus_floor_min": round_number(min(target_minus_floor)) if target_minus_floor else None,
        "target_minus_floor_max": round_number(max(target_minus_floor)) if target_minus_floor else None,
        "top_minus_target_min": round_number(min(top_minus_target)) if top_minus_target else None,
        "top_minus_target_max": round_number(max(top_minus_target)) if top_minus_target else None,
    }


def floor_only_rescue_upper_bound(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [
        str(sample.get("sample_id") or "")
        for sample in samples
        if safe_int(((sample.get("target") or {}).get("trace") or {}).get("pre_floor_rerank_rank")) is not None
        and int(((sample.get("target") or {}).get("trace") or {}).get("pre_floor_rerank_rank")) <= 10
    ]
    near_floor_ids = [
        str(sample.get("sample_id") or "")
        for sample in samples
        if ((sample.get("score_floor") or {}).get("target_minus_floor") is not None)
        and float((sample.get("score_floor") or {}).get("target_minus_floor")) >= -0.1
    ]
    return {
        "pre_floor_rank_at_or_above_10_count": len(sample_ids),
        "pre_floor_rank_at_or_above_10_sample_ids": sample_ids,
        "near_floor_within_0_1_count": len(near_floor_ids),
        "near_floor_within_0_1_sample_ids": near_floor_ids,
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Rerank Score-Floor Low-Score Audit",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- target samples: **{summary['target_sample_count']}**",
        f"- dominant low-score reason: **{summary['dominant_reason']}**",
        "",
        "## Counts",
        "",
        "| group | counts |",
        "|---|---|",
        f"| `reason_counts` | `{json.dumps(summary['reason_counts'], ensure_ascii=False, sort_keys=True)}` |",
        f"| `tag_counts` | `{json.dumps(summary['tag_counts'], ensure_ascii=False, sort_keys=True)}` |",
        f"| `category_counts` | `{json.dumps(summary['category_counts'], ensure_ascii=False, sort_keys=True)}` |",
        "",
        "## Score Gap",
        "",
        f"- target_minus_floor range: `{summary['score_floor_gap_stats']['target_minus_floor_min']}` to `{summary['score_floor_gap_stats']['target_minus_floor_max']}`",
        f"- top_minus_target range: `{summary['score_floor_gap_stats']['top_minus_target_min']}` to `{summary['score_floor_gap_stats']['top_minus_target_max']}`",
        f"- floor-only final top10 rescue upper bound: `{summary['floor_only_rescue_upper_bound']['pre_floor_rank_at_or_above_10_count']}` samples",
        f"- near-floor cases within 0.1 score: `{summary['floor_only_rescue_upper_bound']['near_floor_within_0_1_count']}` samples",
        "",
        "## Sample Detail",
        "",
        "| sample | category | target score/floor/top | target raw | top raw | target bonus | top bonus | overlap target/top | reason |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for sample in samples:
        floor = sample["score_floor"]
        target_features = sample["target"]["features"]
        top_features = sample["top_competitor"]["features"]
        target_overlap = target_features["lexical_overlap"]
        top_overlap = top_features["lexical_overlap"]
        lines.append(
            f"| `{sample['sample_id']}` | `{sample.get('category')}` | "
            f"{floor['target_score']}/{floor['floor']}/{floor['top_score']} | "
            f"{target_features['max_query_score']} | {top_features['max_query_score']} | "
            f"{target_features['total_bonus']} | {top_features['total_bonus']} | "
            f"{target_overlap['overlap_count']}+{target_overlap['symbol_hit_count']}/"
            f"{top_overlap['overlap_count']}+{top_overlap['symbol_hit_count']} | "
            f"`{sample['low_score_reason']}` |"
        )
    lines.extend(["", "## Files", ""])
    lines.append("- JSON summary: `rerank_score_floor_low_score_summary.json`")
    lines.append("- JSONL samples: `rerank_score_floor_low_score_samples.jsonl`")
    return "\n".join(lines)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def preview(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def run_self_test() -> None:
    assert extract_symbols("Sec61 EMDB EMD-1234")[:2] == ["sec61", "emdb"]
    assert contains_cjk("中文") is True
    assert contains_cjk("English") is False
    print("self-test ok")


if __name__ == "__main__":
    main()
