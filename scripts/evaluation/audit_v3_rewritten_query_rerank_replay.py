from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.audit_v3_gold_child_source_to_retrieval import (  # noqa: E402
    B0_RETRIEVAL_OVERRIDES,
    as_list,
    safe_int,
)
from scripts.evaluation.audit_v3_rerank_score_floor_low_score import (  # noqa: E402
    build_chunk,
    safe_float,
)
from scripts.evaluation.audit_v3_retrieval_drift import (  # noqa: E402
    BASELINE_RETRIEVAL_OVERRIDES,
    apply_overrides,
    load_jsonl,
    load_jsonl_by_id,
    parent_chunk_id,
    round_number,
    write_json,
    write_jsonl,
    write_markdown,
)
from src.synbio_rag.application.rerank_service import LocalBGERerankerService  # noqa: E402
from src.synbio_rag.domain.config import Settings  # noqa: E402
from src.synbio_rag.domain.router import QueryRouter  # noqa: E402


RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_LOW_SCORE_AUDIT = (
    RESULTS_ROOT
    / "v3_rerank_score_floor_low_score_audit_20260524_rerank_score_floor_low_score_audit"
    / "rerank_score_floor_low_score_samples.jsonl"
)
DEFAULT_REWRITE_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260523_support_selector_retention"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the same rerank candidates with rewritten query for score-floor misses."
    )
    parser.add_argument("--low-score-audit", default=str(DEFAULT_LOW_SCORE_AUDIT))
    parser.add_argument("--rewrite-results", default=str(DEFAULT_REWRITE_RESULTS))
    parser.add_argument("--run-id", default="20260524_rewritten_query_rerank_replay")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    settings = Settings.from_env()
    overrides = dict(BASELINE_RETRIEVAL_OVERRIDES)
    overrides.update(B0_RETRIEVAL_OVERRIDES)
    apply_overrides(settings, overrides)
    patch_transformers_prepare_for_model()

    rows = load_jsonl(Path(args.low_score_audit))
    rewrite_rows = load_jsonl_by_id(Path(args.rewrite_results), "sample_id")
    child_records = load_jsonl_by_id(Path(settings.kb.child_chunk_jsonl), "chunk_id")
    parent_records = load_jsonl_by_id(Path(settings.kb.parent_chunk_jsonl), "chunk_id")
    router = QueryRouter(settings.retrieval)
    reranker = LocalBGERerankerService(
        model_path=settings.reranker.model_path,
        batch_size=settings.reranker.batch_size,
        use_fp16=settings.reranker.use_fp16,
        retrieval_config=settings.retrieval,
    )

    samples = [
        audit_sample(
            row=row,
            rewrite_row=rewrite_rows.get(str(row.get("sample_id") or "")) or {},
            child_records=child_records,
            parent_records=parent_records,
            router=router,
            reranker=reranker,
        )
        for row in rows
    ]
    summary = build_summary(args.run_id, samples, args.low_score_audit, args.rewrite_results)
    result_dir = RESULTS_ROOT / f"v3_rewritten_query_rerank_replay_{args.run_id}"
    report_dir = REPORTS_ROOT / f"v3_rewritten_query_rerank_replay_{args.run_id}"
    write_json(result_dir / "rewritten_query_rerank_replay_summary.json", summary)
    write_jsonl(result_dir / "rewritten_query_rerank_replay_samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(json.dumps({"result_dir": str(result_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def audit_sample(
    *,
    row: dict[str, Any],
    rewrite_row: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    router: QueryRouter,
    reranker: LocalBGERerankerService,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    original_query = str(row.get("question") or rewrite_row.get("question") or "")
    rewritten_query = str(row.get("rewritten_query") or original_query)
    gold_parents = as_list(row.get("gold_parent_chunk_ids"))
    debug = rewrite_row.get("debug_digest") or {}
    rerank_hits = debug.get("rerank_hits") or {}
    ranking_trace = rerank_hits.get("ranking_trace") or []
    retrieval_output = debug.get("retrieval_output") or {}
    matched_by_parent = retrieval_output.get("matched_child_chunk_ids_by_chunk_id") or {}
    candidates = rebuild_candidates(
        ranking_trace=ranking_trace,
        matched_by_parent=matched_by_parent,
        child_records=child_records,
        parent_records=parent_records,
    )
    analysis = router.analyze(original_query)
    rewritten_final = reranker.rerank(
        rewritten_query,
        candidates,
        top_k=analysis.rerank_top_k,
        analysis=analysis,
        mode="plain",
    )
    replay_trace = list((reranker.last_debug or {}).get("ranking_trace") or [])
    original_target = best_target_trace(ranking_trace, set(gold_parents))
    rewritten_target = best_target_trace(replay_trace, set(gold_parents))
    comparison = compare_target(original_target, rewritten_target)
    return {
        "sample_id": sample_id,
        "category": row.get("category"),
        "expected_route": row.get("expected_route"),
        "question": original_query,
        "rewritten_query": rewritten_query,
        "gold_parent_chunk_ids": gold_parents,
        "candidate_count": len(candidates),
        "analysis_intent": analysis.intent.value,
        "original": {
            "target": compact_target(original_target),
            "query_variants": (rerank_hits.get("query_variants") or []),
        },
        "rewritten_replay": {
            "target": compact_target(rewritten_target),
            "query_variants": (reranker.last_debug or {}).get("query_variants") or [],
            "final_chunk_ids": [chunk.chunk_id for chunk in rewritten_final],
        },
        "comparison": comparison,
        "outcome": classify_outcome(comparison, rewritten_target),
    }


def rebuild_candidates(
    *,
    ranking_trace: list[dict[str, Any]],
    matched_by_parent: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
) -> list[Any]:
    ordered = sorted(ranking_trace, key=lambda item: safe_int(item.get("raw_retrieval_rank")) or 999999)
    chunks = []
    seen: set[str] = set()
    for item in ordered:
        parent_id = str(item.get("parent_chunk_id") or parent_chunk_id(item.get("chunk_id")))
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        chunk = build_chunk(
            trace_item=item,
            parent_records=parent_records,
            child_records=child_records,
            matched_child_ids=as_list(matched_by_parent.get(parent_id)),
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def best_target_trace(trace: list[dict[str, Any]], gold_parents: set[str]) -> dict[str, Any]:
    targets = [
        item
        for item in trace
        if str(item.get("parent_chunk_id") or parent_chunk_id(item.get("chunk_id"))) in gold_parents
    ]
    targets.sort(key=lambda item: safe_int(item.get("pre_floor_rerank_rank")) or 999999)
    return targets[0] if targets else {}


def compact_target(item: dict[str, Any]) -> dict[str, Any]:
    query_scores = [safe_float(value) for value in item.get("query_scores") or []]
    query_scores = [value for value in query_scores if value is not None]
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
        "max_query_score": round_number(max(query_scores)) if query_scores else None,
        "query_scores": [round_number(value) for value in query_scores],
        "final_drop_reason": item.get("final_drop_reason"),
    }


def compare_target(original: dict[str, Any], rewritten: dict[str, Any]) -> dict[str, Any]:
    original_raw = max_score(original)
    rewritten_raw = max_score(rewritten)
    original_score = safe_float(original.get("score"))
    rewritten_score = safe_float(rewritten.get("score"))
    original_rank = safe_int(original.get("pre_floor_rerank_rank"))
    rewritten_rank = safe_int(rewritten.get("pre_floor_rerank_rank"))
    return {
        "raw_query_score_delta": round_number(rewritten_raw - original_raw)
        if original_raw is not None and rewritten_raw is not None
        else None,
        "rerank_score_delta": round_number(rewritten_score - original_score)
        if original_score is not None and rewritten_score is not None
        else None,
        "pre_floor_rank_delta": original_rank - rewritten_rank
        if original_rank is not None and rewritten_rank is not None
        else None,
        "post_floor_changed_to_hit": original.get("post_floor_rank") is None
        and rewritten.get("post_floor_rank") is not None,
        "final_top10_changed_to_hit": original.get("final_top10_rank") is None
        and rewritten.get("final_top10_rank") is not None,
    }


def max_score(item: dict[str, Any]) -> float | None:
    scores = [safe_float(value) for value in item.get("query_scores") or []]
    scores = [value for value in scores if value is not None]
    return max(scores) if scores else None


def classify_outcome(comparison: dict[str, Any], rewritten_target: dict[str, Any]) -> str:
    if comparison.get("final_top10_changed_to_hit"):
        return "rewritten_query_rescues_final_top10"
    if comparison.get("post_floor_changed_to_hit"):
        return "rewritten_query_rescues_post_floor_only"
    rank_delta = comparison.get("pre_floor_rank_delta")
    score_delta = comparison.get("raw_query_score_delta")
    if isinstance(rank_delta, int) and rank_delta > 0 and isinstance(score_delta, float) and score_delta > 0:
        return "rewritten_query_improves_score_and_rank"
    if isinstance(score_delta, float) and score_delta > 0:
        return "rewritten_query_improves_score_only"
    if isinstance(score_delta, float) and score_delta < 0:
        return "rewritten_query_worsens_score"
    if not rewritten_target:
        return "target_missing_in_replay"
    return "no_material_improvement"


def build_summary(
    run_id: str,
    samples: list[dict[str, Any]],
    low_score_audit: str,
    rewrite_results: str,
) -> dict[str, Any]:
    outcome_counts = Counter(sample["outcome"] for sample in samples)
    improved_score = [
        sample["sample_id"]
        for sample in samples
        if (sample.get("comparison") or {}).get("raw_query_score_delta") is not None
        and float((sample.get("comparison") or {}).get("raw_query_score_delta")) > 0
    ]
    improved_rank = [
        sample["sample_id"]
        for sample in samples
        if (sample.get("comparison") or {}).get("pre_floor_rank_delta") is not None
        and int((sample.get("comparison") or {}).get("pre_floor_rank_delta")) > 0
    ]
    post_floor_rescued = [
        sample["sample_id"]
        for sample in samples
        if (sample.get("comparison") or {}).get("post_floor_changed_to_hit")
    ]
    final_rescued = [
        sample["sample_id"]
        for sample in samples
        if (sample.get("comparison") or {}).get("final_top10_changed_to_hit")
    ]
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "Replay same rerank candidates for score-floor misses with rewritten query.",
        "inputs": {
            "low_score_audit": low_score_audit,
            "rewrite_results": rewrite_results,
        },
        "target_sample_count": len(samples),
        "outcome_counts": dict(outcome_counts),
        "raw_score_improved_count": len(improved_score),
        "raw_score_improved_sample_ids": improved_score,
        "pre_floor_rank_improved_count": len(improved_rank),
        "pre_floor_rank_improved_sample_ids": improved_rank,
        "post_floor_rescued_count": len(post_floor_rescued),
        "post_floor_rescued_sample_ids": post_floor_rescued,
        "final_top10_rescued_count": len(final_rescued),
        "final_top10_rescued_sample_ids": final_rescued,
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Rewritten Query Rerank Replay",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- target samples: **{summary['target_sample_count']}**",
        f"- raw score improved: **{summary['raw_score_improved_count']}**",
        f"- pre-floor rank improved: **{summary['pre_floor_rank_improved_count']}**",
        f"- post-floor rescued: **{summary['post_floor_rescued_count']}**",
        f"- final top10 rescued: **{summary['final_top10_rescued_count']}**",
        "",
        "## Outcomes",
        "",
        f"`{json.dumps(summary['outcome_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Sample Detail",
        "",
        "| sample | category | original raw/rank/post/final | rewritten raw/rank/post/final | raw delta | rank delta | outcome |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for sample in samples:
        original = sample["original"]["target"]
        rewritten = sample["rewritten_replay"]["target"]
        comparison = sample["comparison"]
        lines.append(
            f"| `{sample['sample_id']}` | `{sample.get('category')}` | "
            f"{fmt(original.get('max_query_score'))}/{fmt(original.get('pre_floor_rerank_rank'))}/"
            f"{fmt(original.get('post_floor_rank'))}/{fmt(original.get('final_top10_rank'))} | "
            f"{fmt(rewritten.get('max_query_score'))}/{fmt(rewritten.get('pre_floor_rerank_rank'))}/"
            f"{fmt(rewritten.get('post_floor_rank'))}/{fmt(rewritten.get('final_top10_rank'))} | "
            f"{fmt(comparison.get('raw_query_score_delta'))} | "
            f"{fmt(comparison.get('pre_floor_rank_delta'))} | `{sample['outcome']}` |"
        )
    lines.extend(["", "## Files", ""])
    lines.append("- JSON summary: `rewritten_query_rerank_replay_summary.json`")
    lines.append("- JSONL samples: `rewritten_query_rerank_replay_samples.jsonl`")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    return "N/A" if value is None else str(value)


def patch_transformers_prepare_for_model() -> None:
    try:
        from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
    except Exception:
        return
    if hasattr(XLMRobertaTokenizer, "prepare_for_model"):
        return

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        truncation=False,
        max_length=None,
        padding=False,
        **_kwargs,
    ):
        first = list(ids or [])
        second = list(pair_ids or []) if pair_ids is not None else None
        if max_length is not None and second is not None:
            special_count = self.num_special_tokens_to_add(pair=True)
            allowed_second = max(0, int(max_length) - len(first) - special_count)
            if truncation in {"only_second", True} and len(second) > allowed_second:
                second = second[:allowed_second]
        if second is None:
            input_ids = [self.cls_token_id] + first + [self.sep_token_id]
        else:
            input_ids = [self.cls_token_id] + first + [self.sep_token_id, self.sep_token_id] + second + [self.sep_token_id]
        token_type_ids = [self.pad_token_type_id] * len(input_ids)
        return {"input_ids": input_ids, "token_type_ids": token_type_ids}

    XLMRobertaTokenizer.prepare_for_model = prepare_for_model


def run_self_test() -> None:
    assert classify_outcome(
        {"final_top10_changed_to_hit": True},
        {"chunk_id": "x"},
    ) == "rewritten_query_rescues_final_top10"
    assert compare_target(
        {"query_scores": [1.0], "score": 1.0, "pre_floor_rerank_rank": 5},
        {"query_scores": [2.0], "score": 2.0, "pre_floor_rerank_rank": 3},
    )["pre_floor_rank_delta"] == 2
    print("self-test ok")


if __name__ == "__main__":
    main()
