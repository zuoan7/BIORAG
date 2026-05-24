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

from scripts.evaluation.audit_v3_retrieval_drift import (  # noqa: E402
    BASELINE_RETRIEVAL_OVERRIDES,
    RetrievalProber,
    apply_overrides,
    chunk_parent_id,
    dedupe,
    infer_dense_id_kind,
    load_jsonl,
    load_jsonl_by_id,
    load_parent_index,
    nearest_parent_distance,
    parent_chunk_id,
    rank_in_chunks,
    reciprocal_rank_fusion_multi,
    round_number,
    write_json,
    write_jsonl,
    write_markdown,
)
from src.synbio_rag.application.retrieval_postprocessor import RetrievalPostProcessor  # noqa: E402
from src.synbio_rag.domain.config import Settings  # noqa: E402
from src.synbio_rag.infrastructure.index.parent_store import ParentStore  # noqa: E402


RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_FOCUS_SAMPLES = (
    RESULTS_ROOT
    / "v3_b0_rewrite_remaining_miss_focus_20260523_support_selector_retention"
    / "focus_samples.jsonl"
)
DEFAULT_TRANSITION_SAMPLES = (
    RESULTS_ROOT
    / "v3_b0_rewrite_transition_20260523_support_selector_retention"
    / "transition_samples.jsonl"
)
DEFAULT_REWRITE_RESULTS = (
    RESULTS_ROOT
    / "v3_b0_rewrite_enabled_20260523_support_selector_retention"
    / "b0_rewrite_enabled"
    / "results.jsonl"
)
DEFAULT_PREVIOUS_AUDIT = (
    RESULTS_ROOT
    / "v3_doc_parent_miss_audit_20260523_doc_parent_miss_audit"
    / "doc_parent_miss_samples.jsonl"
)
TOP_N = 200
B0_RETRIEVAL_OVERRIDES: dict[str, Any] = {
    "retrieval.hybrid_enabled": True,
    "retrieval.bm25_enabled": True,
    "retrieval.source_floor_enabled": True,
    "retrieval.parent_expansion_enabled": False,
    "retrieval.rerank_mode": "plain",
    "retrieval.rerank_top_k": 10,
    "retrieval.final_top_k": 8,
    "retrieval.table_preview_enabled": True,
    "retrieval.table_preview_merge_enabled": False,
    "retrieval.table_preview_allow_formal_citation": False,
    "retrieval.original_cn_fallback_enabled": False,
    "query_rewrite.mode": "enabled",
    "table_enhancement.enabled": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-to-retrieval audit for v3 B0+rewrite doc_hit_parent_miss gold children."
    )
    parser.add_argument("--focus-samples", default=str(DEFAULT_FOCUS_SAMPLES))
    parser.add_argument("--transition-samples", default=str(DEFAULT_TRANSITION_SAMPLES))
    parser.add_argument("--rewrite-results", default=str(DEFAULT_REWRITE_RESULTS))
    parser.add_argument("--previous-audit", default=str(DEFAULT_PREVIOUS_AUDIT))
    parser.add_argument("--run-id", default="20260524_gold_child_source_to_retrieval_audit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    run_id = str(args.run_id)
    result_dir = RESULTS_ROOT / f"v3_gold_child_source_to_retrieval_audit_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_gold_child_source_to_retrieval_audit_{run_id}"

    focus_rows = [
        row
        for row in load_jsonl(Path(args.focus_samples))
        if row.get("focus_bucket") == "doc_hit_parent_miss"
    ]
    if args.limit > 0:
        focus_rows = focus_rows[: args.limit]
    transition_rows = load_jsonl_by_id(Path(args.transition_samples), "sample_id")
    rewrite_rows = load_jsonl_by_id(Path(args.rewrite_results), "sample_id")
    previous_rows = (
        load_jsonl_by_id(Path(args.previous_audit), "sample_id")
        if Path(args.previous_audit).exists()
        else {}
    )

    settings = Settings.from_env()
    overrides = dict(BASELINE_RETRIEVAL_OVERRIDES)
    overrides.update(B0_RETRIEVAL_OVERRIDES)
    apply_overrides(settings, overrides)

    child_records = load_jsonl_by_id(Path(settings.kb.child_chunk_jsonl), "chunk_id")
    parent_records = load_jsonl_by_id(Path(settings.kb.parent_chunk_jsonl), "chunk_id")
    parent_index = load_parent_index(Path(settings.retrieval.parent_index_path))
    parent_store = ParentStore.from_jsonl(
        settings.retrieval.parent_index_path,
        chunk_jsonl_path=settings.kb.child_chunk_jsonl,
        parent_chunk_jsonl_path=settings.kb.parent_chunk_jsonl,
    )

    prober = RetrievalProber(settings)
    postprocessor = RetrievalPostProcessor()
    bm25_index_ids = load_bm25_index_ids(prober)
    dense_index = audit_dense_index(prober, focus_rows)

    probe_cache: dict[tuple[str, str], dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    for row in focus_rows:
        sample_id = str(row.get("sample_id") or "")
        transition = transition_rows.get(sample_id) or {}
        rewrite = rewrite_rows.get(sample_id) or {}
        previous = previous_rows.get(sample_id) or {}
        merged = merge_sample(row, transition, rewrite, previous)
        original_query = str(merged.get("question") or "")
        rewritten_query = str(merged.get("rewritten_query") or original_query)

        original_probe = get_probe(
            cache=probe_cache,
            prober=prober,
            retrieval_query=original_query,
            original_question=original_query,
        )
        rewritten_probe = get_probe(
            cache=probe_cache,
            prober=prober,
            retrieval_query=rewritten_query,
            original_question=original_query,
        )
        samples.append(
            audit_sample(
                sample=merged,
                rewrite_row=rewrite,
                child_records=child_records,
                parent_records=parent_records,
                parent_index=parent_index,
                parent_store=parent_store,
                bm25_index_ids=bm25_index_ids,
                dense_index=dense_index,
                original_probe=original_probe,
                rewritten_probe=rewritten_probe,
                postprocessor=postprocessor,
                retrieval_config=settings.retrieval,
            )
        )

    summary = build_summary(
        run_id=run_id,
        samples=samples,
        settings=settings,
        dense_index=dense_index,
        inputs={
            "focus_samples": str(args.focus_samples),
            "transition_samples": str(args.transition_samples),
            "rewrite_results": str(args.rewrite_results),
            "previous_audit": str(args.previous_audit),
        },
    )
    write_json(result_dir / "gold_child_source_to_retrieval_summary.json", summary)
    write_jsonl(result_dir / "gold_child_source_to_retrieval_samples.jsonl", samples)
    write_markdown(report_dir / "report.md", render_report(summary, samples))
    print(json.dumps({"result_dir": str(result_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def merge_sample(
    focus: dict[str, Any],
    transition: dict[str, Any],
    rewrite: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(focus.get("sample_id") or transition.get("sample_id") or rewrite.get("sample_id") or "")
    gold_children = as_list(
        focus.get("gold_child_chunk_ids")
        or transition.get("gold_chunk_ids")
        or rewrite.get("gold_chunk_ids")
    )
    gold_children = [item for item in gold_children if "::child" in item]
    gold_parents = as_list(
        focus.get("gold_parent_chunk_ids")
        or transition.get("gold_parent_chunk_ids")
        or rewrite.get("gold_parent_chunk_ids")
        or previous.get("gold_parent_chunk_ids")
        or [parent_chunk_id(item) for item in gold_children]
    )
    return {
        "sample_id": sample_id,
        "focus_bucket": focus.get("focus_bucket"),
        "category": focus.get("category") or transition.get("category") or rewrite.get("category"),
        "expected_route": focus.get("expected_route") or transition.get("expected_route") or rewrite.get("expected_route"),
        "difficulty": focus.get("difficulty") or transition.get("difficulty") or rewrite.get("difficulty"),
        "question": focus.get("question") or transition.get("question") or rewrite.get("question"),
        "rewritten_query": focus.get("rewritten_query") or transition.get("rewritten_query"),
        "expected_doc_ids": as_list(
            focus.get("expected_doc_ids")
            or transition.get("expected_doc_ids")
            or rewrite.get("expected_doc_ids")
        ),
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "previous_secondary": previous.get("secondary_break_classification"),
        "previous_root_cause": previous.get("root_cause_classification"),
        "previous_recommended_next_bucket": previous.get("recommended_next_bucket"),
        "focus_raw_child_context": focus.get("raw_child_context") or {},
        "focus_rerank_target_trace": focus.get("rerank_target_trace") or {},
        "focus_same_doc_context": focus.get("same_doc_context") or {},
    }


def audit_sample(
    *,
    sample: dict[str, Any],
    rewrite_row: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    parent_index: dict[str, dict[str, Any]],
    parent_store: ParentStore,
    bm25_index_ids: set[str],
    dense_index: dict[str, Any],
    original_probe: dict[str, Any],
    rewritten_probe: dict[str, Any],
    postprocessor: RetrievalPostProcessor,
    retrieval_config: Any,
) -> dict[str, Any]:
    gold_children = as_list(sample.get("gold_child_chunk_ids"))
    gold_parents = as_list(sample.get("gold_parent_chunk_ids"))
    expected_docs = as_list(sample.get("expected_doc_ids"))
    source_status = audit_source_status(
        gold_children=gold_children,
        gold_parents=gold_parents,
        child_records=child_records,
        parent_records=parent_records,
        parent_index=parent_index,
        parent_store=parent_store,
    )
    index_status = audit_index_status(
        gold_children=gold_children,
        gold_parents=gold_parents,
        bm25_index_ids=bm25_index_ids,
        dense_index=dense_index,
    )
    original_status = summarize_probe(
        probe=original_probe,
        gold_children=gold_children,
        gold_parents=gold_parents,
        expected_docs=expected_docs,
        postprocessor=postprocessor,
        retrieval_config=retrieval_config,
        parent_store=parent_store,
    )
    rewritten_status = summarize_probe(
        probe=rewritten_probe,
        gold_children=gold_children,
        gold_parents=gold_parents,
        expected_docs=expected_docs,
        postprocessor=postprocessor,
        retrieval_config=retrieval_config,
        parent_store=parent_store,
    )
    actual_pipeline = summarize_actual_pipeline(
        rewrite_row=rewrite_row,
        gold_children=gold_children,
        gold_parents=gold_parents,
        expected_docs=expected_docs,
    )
    loss = classify_loss_stage(
        source_status=source_status,
        index_status=index_status,
        original_status=original_status,
        rewritten_status=rewritten_status,
        actual_pipeline=actual_pipeline,
        previous_secondary=str(sample.get("previous_secondary") or ""),
    )
    return {
        "sample_id": sample.get("sample_id"),
        "category": sample.get("category"),
        "expected_route": sample.get("expected_route"),
        "difficulty": sample.get("difficulty"),
        "question": sample.get("question"),
        "rewritten_query": sample.get("rewritten_query"),
        "expected_doc_ids": expected_docs,
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "previous_secondary": sample.get("previous_secondary"),
        "previous_root_cause": sample.get("previous_root_cause"),
        "previous_recommended_next_bucket": sample.get("previous_recommended_next_bucket"),
        "source_status": source_status,
        "index_status": index_status,
        "retrieval_status": {
            "original_query": original_status,
            "rewritten_query": rewritten_status,
        },
        "actual_pipeline": actual_pipeline,
        "loss_stage": loss["loss_stage"],
        "root_bucket": loss["root_bucket"],
        "loss_reason": loss["reason"],
        "recommended_fix_direction": loss["recommended_fix_direction"],
    }


def audit_source_status(
    *,
    gold_children: list[str],
    gold_parents: list[str],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    parent_index: dict[str, dict[str, Any]],
    parent_store: ParentStore,
) -> dict[str, Any]:
    gold_parent_set = set(gold_parents)
    child_checks = []
    for child_id in gold_children:
        child = child_records.get(child_id)
        child_parent = str((child or {}).get("parent_chunk_id") or parent_chunk_id(child_id))
        child_parent_id = str((child or {}).get("parent_id") or "")
        index_record = parent_index["by_child_id"].get(child_id)
        parent_store_matches = [
            parent
            for parent in parent_store.get_parents_for_chunk(child_id)
            if parent.parent_type == "retrieval_parent"
        ]
        store_parent_chunk_ids = [str(parent.parent_chunk_id or "") for parent in parent_store_matches]
        child_checks.append(
            {
                "gold_child_chunk_id": child_id,
                "exists_in_child_chunks": child is not None,
                "child_metadata_parent_chunk_id": child_parent,
                "child_metadata_parent_matches_gold": child_parent in gold_parent_set,
                "child_metadata_parent_id": child_parent_id,
                "parent_index_record_found": index_record is not None,
                "parent_index_parent_chunk_id": str((index_record or {}).get("parent_chunk_id") or ""),
                "parent_index_contains_child": child_id in set((index_record or {}).get("child_chunk_ids") or []),
                "parent_store_retrieval_parent_chunk_ids": store_parent_chunk_ids,
                "parent_store_maps_to_gold_parent": any(item in gold_parent_set for item in store_parent_chunk_ids),
            }
        )
    parent_checks = [
        {
            "gold_parent_chunk_id": parent_id,
            "exists_in_parent_chunks": parent_id in parent_records,
        }
        for parent_id in gold_parents
    ]
    status = "ok"
    if not gold_children:
        if not parent_checks or any(not item["exists_in_parent_chunks"] for item in parent_checks):
            status = "gold_parent_missing_from_parent_chunks"
        else:
            status = "no_gold_child_id_parent_exists"
    elif any(not item["exists_in_child_chunks"] for item in child_checks):
        status = "gold_child_missing_from_child_chunks"
    elif any(not item["child_metadata_parent_matches_gold"] for item in child_checks):
        status = "child_metadata_parent_mismatch"
    elif any(not item["exists_in_parent_chunks"] for item in parent_checks):
        status = "gold_parent_missing_from_parent_chunks"
    elif any(not item["parent_store_maps_to_gold_parent"] for item in child_checks):
        status = "parent_store_retrieval_parent_mismatch"
    return {
        "status": status,
        "gold_child_all_exist": all(item["exists_in_child_chunks"] for item in child_checks) if child_checks else None,
        "child_metadata_parent_all_match_gold": all(
            item["child_metadata_parent_matches_gold"] for item in child_checks
        )
        if child_checks
        else None,
        "gold_parent_all_exist": all(item["exists_in_parent_chunks"] for item in parent_checks) if parent_checks else False,
        "parent_store_child_to_retrieval_parent_all_ok": all(
            item["parent_store_maps_to_gold_parent"] for item in child_checks
        )
        if child_checks
        else None,
        "child_checks": child_checks,
        "parent_checks": parent_checks,
    }


def audit_index_status(
    *,
    gold_children: list[str],
    gold_parents: list[str],
    bm25_index_ids: set[str],
    dense_index: dict[str, Any],
) -> dict[str, Any]:
    dense_child_ids = set(dense_index.get("exact_child_chunk_ids_found") or [])
    dense_parent_ids = set(dense_index.get("parent_chunk_ids_found") or [])
    bm25_missing = [child_id for child_id in gold_children if child_id not in bm25_index_ids]
    dense_children_missing = [child_id for child_id in gold_children if child_id not in dense_child_ids]
    dense_parents_missing = [parent_id for parent_id in gold_parents if parent_id not in dense_parent_ids]
    dense_verified = dense_index.get("query_status") == "ok"
    if not gold_children:
        if not dense_verified:
            status = "no_gold_child_id_dense_unverified"
        elif dense_parents_missing:
            status = "no_gold_child_id_dense_parent_missing"
        else:
            status = "no_gold_child_id_dense_parent_present"
    elif bm25_missing:
        status = "bm25_gold_child_missing"
    elif dense_verified and dense_children_missing and dense_index.get("dense_id_kind") == "child_chunk_id":
        status = "dense_gold_child_missing"
    elif dense_verified and dense_children_missing and not dense_parents_missing:
        status = "bm25_child_ok_dense_parent_index_only"
    elif not dense_verified:
        status = "bm25_child_ok_dense_unverified"
    else:
        status = "ok"
    return {
        "status": status,
        "bm25": {
            "record_count": len(bm25_index_ids),
            "gold_child_present_all": (not bm25_missing and bool(gold_children)) if gold_children else None,
            "missing_gold_child_chunk_ids": bm25_missing,
        },
        "dense_milvus": {
            "query_status": dense_index.get("query_status"),
            "query_error": dense_index.get("query_error"),
            "dense_id_kind": dense_index.get("dense_id_kind"),
            "gold_child_exact_present_all": (
                dense_verified and not dense_children_missing and bool(gold_children)
            )
            if gold_children
            else None,
            "gold_parent_present_all": dense_verified and not dense_parents_missing and bool(gold_parents),
            "missing_gold_child_exact_ids": dense_children_missing if dense_verified else [],
            "missing_gold_parent_ids": dense_parents_missing if dense_verified else [],
            "note": dense_index.get("note", ""),
        },
    }


def load_bm25_index_ids(prober: RetrievalProber) -> set[str]:
    prober.bm25_retriever._ensure_index()
    return {str(chunk.chunk_id or "") for chunk in prober.bm25_retriever._records}


def audit_dense_index(prober: RetrievalProber, focus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    child_ids = sorted(
        {
            str(child_id)
            for row in focus_rows
            for child_id in row.get("gold_child_chunk_ids") or []
            if str(child_id)
        }
    )
    parent_ids = sorted(
        {
            str(parent_id)
            for row in focus_rows
            for parent_id in row.get("gold_parent_chunk_ids") or []
            if str(parent_id)
        }
    )
    try:
        dense_id_kind = infer_dense_id_kind(prober.dense_retriever.client, prober.settings.retrieval.collection_name)
        child_found = query_milvus_chunk_ids(
            prober.dense_retriever.client,
            prober.settings.retrieval.collection_name,
            child_ids,
        )
        parent_found = query_milvus_chunk_ids(
            prober.dense_retriever.client,
            prober.settings.retrieval.collection_name,
            parent_ids,
        )
        note = ""
        if dense_id_kind == "parent_or_base_chunk_id":
            note = "Milvus chunk_id appears to store parent/base chunk ids; exact ::child ids are not expected."
        return {
            "query_status": "ok",
            "query_error": "",
            "dense_id_kind": dense_id_kind,
            "exact_child_chunk_ids_found": sorted(child_found),
            "parent_chunk_ids_found": sorted(parent_found),
            "note": note,
        }
    except Exception as exc:  # pragma: no cover - depends on local Milvus state.
        return {
            "query_status": "unverified",
            "query_error": f"{type(exc).__name__}: {exc}",
            "dense_id_kind": "unverified",
            "exact_child_chunk_ids_found": [],
            "parent_chunk_ids_found": [],
            "note": "Milvus exact membership could not be proven in this environment.",
        }


def query_milvus_chunk_ids(client: Any, collection_name: str, chunk_ids: list[str]) -> set[str]:
    if not chunk_ids:
        return set()
    found: set[str] = set()
    batch_size = 200
    for idx in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[idx : idx + batch_size]
        values = ",".join(json.dumps(item) for item in batch)
        rows = client.query(
            collection_name=collection_name,
            filter=f"chunk_id in [{values}]",
            limit=len(batch),
            output_fields=["chunk_id"],
        )
        found.update(str(row.get("chunk_id") or "") for row in rows if row.get("chunk_id"))
    return found


def get_probe(
    *,
    cache: dict[tuple[str, str], dict[str, Any]],
    prober: RetrievalProber,
    retrieval_query: str,
    original_question: str,
) -> dict[str, Any]:
    key = (retrieval_query, original_question)
    if key not in cache:
        cache[key] = probe_query(prober, retrieval_query=retrieval_query, original_question=original_question)
    return cache[key]


def probe_query(prober: RetrievalProber, *, retrieval_query: str, original_question: str) -> dict[str, Any]:
    analysis = prober.router.analyze(original_question)
    query_plan = prober.query_planner.build(
        retrieval_query,
        analysis,
        prober.config,
        original_question,
    )
    dense_weight = prober.config.dense_rrf_weight
    bm25_weight = prober.config.bm25_rrf_weight
    if prober.query_planner.contains_cjk(retrieval_query):
        bm25_weight *= prober.config.cjk_query_bm25_weight

    dense_runs = []
    sparse_runs = []
    variants = []
    for idx, variant in enumerate(query_plan):
        dense_results = prober.dense_retriever.search(str(variant["query"]), limit=TOP_N)
        sparse_query = str(variant["query"])
        if getattr(prober.config, "alias_expansion_enabled", False):
            sparse_query = prober.alias_policy.expand(sparse_query, prober.config)
        sparse_results = prober.bm25_retriever.search(sparse_query, limit=TOP_N)
        dense_runs.append((dense_results, dense_weight * float(variant["weight"])))
        sparse_runs.append((sparse_results, bm25_weight * float(variant["weight"])))
        variants.append(
            {
                "variant_index": idx,
                "kind": variant["kind"],
                "query": variant["query"],
                "sparse_query": sparse_query,
                "weight": variant["weight"],
                "dense_count": len(dense_results),
                "bm25_count": len(sparse_results),
            }
        )
    fused_top200 = reciprocal_rank_fusion_multi(
        dense_runs=dense_runs,
        sparse_runs=sparse_runs,
        limit=TOP_N,
        rrf_k=prober.config.rrf_k,
    )
    return {
        "retrieval_query": retrieval_query,
        "original_question": original_question,
        "intent": analysis.intent.value,
        "search_limit": analysis.search_limit,
        "actual_dense_limit": max(analysis.search_limit, prober.config.dense_limit),
        "actual_bm25_limit": max(analysis.search_limit, prober.config.bm25_limit),
        "actual_fusion_pool_limit": max(analysis.search_limit * 3, analysis.search_limit + 12),
        "query_variants": variants,
        "dense_runs": dense_runs,
        "sparse_runs": sparse_runs,
        "fused_top200": fused_top200,
        "analysis_object": analysis,
    }


def summarize_probe(
    *,
    probe: dict[str, Any],
    gold_children: list[str],
    gold_parents: list[str],
    expected_docs: list[str],
    postprocessor: RetrievalPostProcessor,
    retrieval_config: Any,
    parent_store: ParentStore,
) -> dict[str, Any]:
    dense_runs = probe["dense_runs"]
    sparse_runs = probe["sparse_runs"]
    dense_best = summarize_weighted_runs(dense_runs, gold_children, gold_parents, expected_docs)
    bm25_best = summarize_weighted_runs(sparse_runs, gold_children, gold_parents, expected_docs)
    fused_top200 = summarize_stage(probe["fused_top200"], gold_children, gold_parents, expected_docs)
    stages = simulate_retrieval_stages(
        probe=probe,
        postprocessor=postprocessor,
        retrieval_config=retrieval_config,
        parent_store=parent_store,
        gold_children=gold_children,
        gold_parents=gold_parents,
        expected_docs=expected_docs,
    )
    return {
        "retrieval_query": probe["retrieval_query"],
        "intent": probe["intent"],
        "search_limit": probe["search_limit"],
        "actual_dense_limit": probe["actual_dense_limit"],
        "actual_bm25_limit": probe["actual_bm25_limit"],
        "actual_fusion_pool_limit": probe["actual_fusion_pool_limit"],
        "query_variants": probe["query_variants"],
        "dense_top200": dense_best,
        "bm25_top200": bm25_best,
        "rrf_top200": fused_top200,
        "pipeline_stages": stages,
    }


def summarize_weighted_runs(
    weighted_runs: list[tuple[list[Any], float]],
    gold_children: list[str],
    gold_parents: list[str],
    expected_docs: list[str],
) -> dict[str, Any]:
    best_summary: dict[str, Any] | None = None
    for variant_index, (chunks, weight) in enumerate(weighted_runs):
        summary = summarize_stage(chunks, gold_children, gold_parents, expected_docs)
        summary["query_variant_index"] = variant_index
        summary["weight"] = round_number(weight)
        if best_summary is None or rank_value(summary["gold_parent_child"]) < rank_value(best_summary["gold_parent_child"]):
            best_summary = summary
    return best_summary or empty_stage_summary()


def simulate_retrieval_stages(
    *,
    probe: dict[str, Any],
    postprocessor: RetrievalPostProcessor,
    retrieval_config: Any,
    parent_store: ParentStore,
    gold_children: list[str],
    gold_parents: list[str],
    expected_docs: list[str],
) -> dict[str, Any]:
    actual_dense_limit = int(probe["actual_dense_limit"])
    actual_bm25_limit = int(probe["actual_bm25_limit"])
    actual_fusion_pool = int(probe["actual_fusion_pool_limit"])
    dense_runs = [(list(chunks)[:actual_dense_limit], weight) for chunks, weight in probe["dense_runs"]]
    sparse_runs = [(list(chunks)[:actual_bm25_limit], weight) for chunks, weight in probe["sparse_runs"]]
    fused = reciprocal_rank_fusion_multi(
        dense_runs=dense_runs,
        sparse_runs=sparse_runs,
        limit=actual_fusion_pool,
        rrf_k=retrieval_config.rrf_k,
    )
    boosted = postprocessor.apply_title_keyword_boost(fused, probe["retrieval_query"], retrieval_config)
    boosted = postprocessor.apply_structure_marker_boost(boosted, probe["retrieval_query"], retrieval_config)
    diversified = postprocessor.apply_comparison_diversity(
        boosted,
        int(probe["search_limit"]),
        probe.get("analysis_object"),
        retrieval_config,
    )
    first_dense = dense_runs[0][0] if dense_runs else []
    first_sparse = sparse_runs[0][0] if sparse_runs else []
    expanded = postprocessor.apply_same_doc_body_expansion(
        diversified=diversified,
        dense_results=first_dense,
        bm25_results=first_sparse,
        config=retrieval_config,
        question=probe["retrieval_query"],
        bm25_retriever=None,
    )
    source_floor = postprocessor.apply_source_floor(
        expanded=expanded,
        dense_results=first_dense,
        sparse_results=first_sparse,
        config=retrieval_config,
    )
    materialized = parent_store.materialize_parent_hits(source_floor)
    return {
        "rrf_actual_pool": summarize_stage(fused, gold_children, gold_parents, expected_docs),
        "boosted": summarize_stage(boosted, gold_children, gold_parents, expected_docs),
        "diversified_limit": summarize_stage(diversified, gold_children, gold_parents, expected_docs),
        "source_floor": summarize_stage(source_floor, gold_children, gold_parents, expected_docs),
        "parent_materialized": summarize_stage(materialized, gold_children, gold_parents, expected_docs),
    }


def summarize_stage(
    chunks: list[Any],
    gold_children: list[str],
    gold_parents: list[str],
    expected_docs: list[str],
) -> dict[str, Any]:
    gold_child_set = set(gold_children)
    gold_parent_set = set(gold_parents)
    expected_doc_set = set(expected_docs)
    exact_child = rank_in_chunks(chunks, gold_child_set, by_parent=False)
    parent_child = rank_in_chunks(chunks, gold_parent_set, by_parent=True)
    matched_child = rank_by_matched_child(chunks, gold_child_set)
    wrong = first_expected_doc_wrong_parent(chunks, expected_doc_set, gold_parent_set)
    return {
        "count": len(chunks),
        "gold_child_exact": compact_rank(exact_child),
        "gold_child_matched_after_materialize": compact_rank(matched_child),
        "gold_parent_child": compact_rank(parent_child),
        "expected_doc_wrong_parent_child": wrong,
    }


def empty_stage_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "gold_child_exact": compact_rank(None),
        "gold_child_matched_after_materialize": compact_rank(None),
        "gold_parent_child": compact_rank(None),
        "expected_doc_wrong_parent_child": {},
    }


def rank_by_matched_child(chunks: list[Any], gold_children: set[str]) -> dict[str, Any] | None:
    if not gold_children:
        return None
    for rank, chunk in enumerate(chunks, start=1):
        metadata = getattr(chunk, "metadata", {}) or {}
        matched = [str(item) for item in metadata.get("matched_child_chunk_ids") or []]
        single = str(metadata.get("matched_child_chunk_id") or "")
        if single:
            matched.append(single)
        hit = next((item for item in matched if item in gold_children), "")
        if hit:
            return {
                "rank": rank,
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "matched_child_chunk_id": hit,
                "parent_chunk_id": chunk_parent_id(chunk),
                "doc_id": str(getattr(chunk, "doc_id", "") or ""),
                "section": str(getattr(chunk, "section", "") or ""),
                "vector_score": round_number(getattr(chunk, "vector_score", 0.0)),
                "bm25_score": round_number(getattr(chunk, "bm25_score", 0.0)),
                "fusion_score": round_number(getattr(chunk, "fusion_score", 0.0)),
            }
    return None


def first_expected_doc_wrong_parent(
    chunks: list[Any],
    expected_docs: set[str],
    gold_parents: set[str],
) -> dict[str, Any]:
    if not expected_docs:
        return {}
    for rank, chunk in enumerate(chunks, start=1):
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        parent_id = chunk_parent_id(chunk)
        if doc_id in expected_docs and parent_id not in gold_parents:
            return {
                "rank": rank,
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "parent_chunk_id": parent_id,
                "doc_id": doc_id,
                "section": str(getattr(chunk, "section", "") or ""),
                "parent_distance_to_gold": nearest_parent_distance([parent_id], list(gold_parents)),
                "vector_score": round_number(getattr(chunk, "vector_score", 0.0)),
                "bm25_score": round_number(getattr(chunk, "bm25_score", 0.0)),
                "fusion_score": round_number(getattr(chunk, "fusion_score", 0.0)),
            }
    return {}


def compact_rank(hit: dict[str, Any] | None) -> dict[str, Any]:
    if not hit:
        return {"hit_top200": False, "rank": None}
    allowed = {
        "rank",
        "chunk_id",
        "matched_child_chunk_id",
        "parent_chunk_id",
        "doc_id",
        "section",
        "vector_score",
        "bm25_score",
        "fusion_score",
        "query_variant_index",
    }
    compact = {
        key: value
        for key, value in hit.items()
        if key in allowed
    }
    compact["hit_top200"] = int(hit.get("rank") or 999999) <= TOP_N
    return compact


def summarize_actual_pipeline(
    *,
    rewrite_row: dict[str, Any],
    gold_children: list[str],
    gold_parents: list[str],
    expected_docs: list[str],
) -> dict[str, Any]:
    debug_digest = rewrite_row.get("debug_digest") or {}
    raw_trace = (debug_digest.get("raw_child_trace") or {}).get("raw_child_trace") or []
    retrieval_output = debug_digest.get("retrieval_output") or {}
    rerank_hits = debug_digest.get("rerank_hits") or {}
    ranking_trace = rerank_hits.get("ranking_trace") or []
    gold_parent_set = set(gold_parents)
    gold_child_set = set(gold_children)
    raw_child_summary = summarize_raw_trace(raw_trace, gold_child_set, gold_parent_set, set(expected_docs))
    retrieval_summary = summarize_debug_stage(retrieval_output, gold_child_set, gold_parent_set, set(expected_docs))
    rerank_summary = summarize_rerank_trace(ranking_trace, gold_parent_set)
    return {
        "raw_child_trace": raw_child_summary,
        "retrieval_output_after_materialize_table_preview": retrieval_summary,
        "rerank": rerank_summary,
        "stored_raw_retrieved_parent_chunk_ids_count": len(rewrite_row.get("raw_retrieved_parent_chunk_ids") or []),
        "stored_retrieved_parent_chunk_ids_top10": rewrite_row.get("retrieved_parent_chunk_ids_top10") or [],
        "stored_final_parent_chunk_ids": rewrite_row.get("final_parent_chunk_ids") or [],
    }


def summarize_raw_trace(
    raw_trace: list[dict[str, Any]],
    gold_children: set[str],
    gold_parents: set[str],
    expected_docs: set[str],
) -> dict[str, Any]:
    gold_child_hits = []
    gold_parent_hits = []
    wrong = {}
    for item in raw_trace:
        rank = safe_int(item.get("rank"))
        child_id = str(item.get("child_chunk_id") or "")
        parent_id = str(item.get("parent_chunk_id") or parent_chunk_id(child_id))
        if child_id in gold_children:
            gold_child_hits.append(rank)
        if parent_id in gold_parents:
            gold_parent_hits.append(rank)
        if not wrong and str(item.get("doc_id") or "") in expected_docs and parent_id not in gold_parents:
            wrong = {
                "rank": rank,
                "child_chunk_id": child_id,
                "parent_chunk_id": parent_id,
                "doc_id": str(item.get("doc_id") or ""),
                "section": str(item.get("section") or ""),
                "parent_distance_to_gold": nearest_parent_distance([parent_id], list(gold_parents)),
            }
    return {
        "count": len(raw_trace),
        "gold_child_exact_ranks": sorted(rank for rank in gold_child_hits if rank is not None),
        "gold_parent_child_ranks": sorted(rank for rank in gold_parent_hits if rank is not None),
        "gold_child_exact_hit": bool(gold_child_hits),
        "gold_parent_child_hit": bool(gold_parent_hits),
        "expected_doc_wrong_parent_child": wrong,
    }


def summarize_debug_stage(
    stage: dict[str, Any],
    gold_children: set[str],
    gold_parents: set[str],
    expected_docs: set[str],
) -> dict[str, Any]:
    chunk_ids = [str(item) for item in stage.get("chunk_ids") or []]
    parent_ids = [parent_chunk_id(item) for item in (stage.get("parent_chunk_ids") or chunk_ids)]
    doc_ids = [str(item) for item in stage.get("doc_ids") or []]
    matched_by_chunk = stage.get("matched_child_chunk_ids_by_chunk_id") or {}
    gold_child_ranks = []
    gold_parent_ranks = []
    matched_child_ranks = []
    wrong = {}
    for idx, chunk_id in enumerate(chunk_ids, start=1):
        parent_id = parent_chunk_id(parent_ids[idx - 1] if idx - 1 < len(parent_ids) else chunk_id)
        doc_id = doc_ids[idx - 1] if idx - 1 < len(doc_ids) else ""
        matched = [str(item) for item in matched_by_chunk.get(chunk_id) or []]
        if chunk_id in gold_children:
            gold_child_ranks.append(idx)
        if any(item in gold_children for item in matched):
            matched_child_ranks.append(idx)
        if parent_id in gold_parents:
            gold_parent_ranks.append(idx)
        if not wrong and doc_id in expected_docs and parent_id not in gold_parents:
            wrong = {
                "rank": idx,
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_id,
                "doc_id": doc_id,
                "parent_distance_to_gold": nearest_parent_distance([parent_id], list(gold_parents)),
            }
    return {
        "count": len(chunk_ids),
        "gold_child_exact_ranks": gold_child_ranks,
        "gold_child_matched_ranks": matched_child_ranks,
        "gold_parent_child_ranks": gold_parent_ranks,
        "gold_parent_child_hit": bool(gold_parent_ranks),
        "expected_doc_wrong_parent_child": wrong,
    }


def summarize_rerank_trace(
    ranking_trace: list[dict[str, Any]],
    gold_parents: set[str],
) -> dict[str, Any]:
    target_items = [
        item
        for item in ranking_trace
        if str(item.get("parent_chunk_id") or parent_chunk_id(item.get("chunk_id"))) in gold_parents
    ]
    if not target_items:
        return {
            "trace_present": bool(ranking_trace),
            "target_trace_count": 0,
            "best_target": {},
            "target_raw_retrieval_rank": None,
            "target_pre_floor_rank": None,
            "target_post_floor_rank": None,
            "target_final_top10_rank": None,
            "target_drop_reasons": [],
        }
    target_items.sort(
        key=lambda item: (
            safe_int(item.get("final_top10_rank")) or 999999,
            safe_int(item.get("post_floor_rank")) or 999999,
            safe_int(item.get("pre_floor_rerank_rank")) or 999999,
            safe_int(item.get("raw_retrieval_rank")) or 999999,
        )
    )
    best = target_items[0]
    return {
        "trace_present": bool(ranking_trace),
        "target_trace_count": len(target_items),
        "best_target": best,
        "target_raw_retrieval_rank": best.get("raw_retrieval_rank"),
        "target_pre_floor_rank": best.get("pre_floor_rerank_rank"),
        "target_post_floor_rank": best.get("post_floor_rank"),
        "target_final_top10_rank": best.get("final_top10_rank"),
        "target_drop_reasons": dedupe(
            [str(item.get("final_drop_reason") or "") for item in target_items if item.get("final_drop_reason")]
        ),
    }


def classify_loss_stage(
    *,
    source_status: dict[str, Any],
    index_status: dict[str, Any],
    original_status: dict[str, Any],
    rewritten_status: dict[str, Any],
    actual_pipeline: dict[str, Any],
    previous_secondary: str,
) -> dict[str, str]:
    non_blocking_source_statuses = {"ok", "no_gold_child_id_parent_exists"}
    if source_status["status"] not in non_blocking_source_statuses:
        return loss("source_integrity", "source", source_status["status"], "fix_source_or_parent_index")
    if index_status["status"] == "bm25_gold_child_missing":
        return loss("bm25_index_missing_gold_child", "index", index_status["status"], "rebuild_bm25_child_index")
    if index_status["status"] == "dense_gold_child_missing":
        return loss("dense_index_missing_gold_child", "index", index_status["status"], "rebuild_dense_child_index")

    rewritten_source_hit = route_hit(rewritten_status["dense_top200"]) or route_hit(rewritten_status["bm25_top200"])
    original_source_hit = route_hit(original_status["dense_top200"]) or route_hit(original_status["bm25_top200"])
    if not rewritten_source_hit:
        if original_source_hit:
            return loss(
                "query_rewrite_single_route_regression_top200",
                "query",
                "original query has a top200 gold-parent signal but rewritten query does not",
                "query_union_or_rewrite_guard",
            )
        return loss(
            "query_single_route_top200_miss",
            "query",
            "neither dense nor BM25 finds the gold parent/child within top200 for the rewritten query",
            "query_reformulation_or_index_signal",
        )

    stages = rewritten_status["pipeline_stages"]
    if not stage_hit(stages["rrf_actual_pool"]):
        return loss(
            "rrf_or_source_limit_drop",
            "RRF",
            "gold signal exists in top200 source retrieval but not in the actual RRF pool",
            "increase_source_or_fusion_pool_for_targeted_cases",
        )
    if stage_hit(stages["rrf_actual_pool"]) and not stage_hit(stages["boosted"]):
        return loss("boost_drop", "postprocess", "gold signal disappeared during boost ordering", "audit_boost_sorting")
    if stage_hit(stages["boosted"]) and not stage_hit(stages["diversified_limit"]):
        return loss(
            "postprocess_limit_drop",
            "postprocess",
            "gold signal exists after boost but is outside the diversified retrieval limit",
            "same_doc_parent_expansion_or_query_union",
        )
    if stage_hit(stages["source_floor"]) and not stage_hit(stages["parent_materialized"]):
        return loss(
            "parent_materialize_drop",
            "materialize",
            "gold child/parent signal exists before materialization but not after ParentStore materialization",
            "fix_parent_store_materialization",
        )

    actual_raw = actual_pipeline["raw_child_trace"]
    rerank = actual_pipeline["rerank"]
    if not actual_raw.get("gold_parent_child_hit") and previous_secondary == "gold_parent_not_in_raw_retrieval":
        return loss(
            "postprocess_limit_or_actual_raw_miss",
            "postprocess",
            "simulated top200 may contain signal, but stored B0+rewrite raw retrieval did not include the gold parent",
            "same_doc_parent_expansion_or_query_union",
        )
    if rerank.get("target_pre_floor_rank") is None:
        return loss(
            "rerank_input_missing",
            "rerank",
            "gold parent is absent from stored rerank ranking_trace",
            "same_doc_parent_expansion_or_query_union",
        )
    if rerank.get("target_post_floor_rank") is None:
        return loss(
            "rerank_score_floor",
            "rerank",
            "gold parent entered rerank pre-floor but was filtered before post-floor",
            "rerank_score_floor_policy",
        )
    if rerank.get("target_final_top10_rank") is None:
        return loss(
            "rerank_final_selection",
            "rerank",
            "gold parent survived score floor but was not selected in final top10",
            "rerank_final_selection_policy",
        )
    return loss(
        "after_rerank_or_metric_scope",
        "rerank",
        "gold parent appears to survive rerank; remaining miss is outside source-to-rerank audit scope",
        "metric_scope_or_support_chain_audit",
    )


def loss(stage: str, bucket: str, reason: str, direction: str) -> dict[str, str]:
    return {
        "loss_stage": stage,
        "root_bucket": bucket,
        "reason": reason,
        "recommended_fix_direction": direction,
    }


def route_hit(stage: dict[str, Any]) -> bool:
    return bool((stage.get("gold_parent_child") or {}).get("rank"))


def stage_hit(stage: dict[str, Any]) -> bool:
    return bool(
        (stage.get("gold_parent_child") or {}).get("rank")
        or (stage.get("gold_child_matched_after_materialize") or {}).get("rank")
    )


def build_summary(
    *,
    run_id: str,
    samples: list[dict[str, Any]],
    settings: Settings,
    dense_index: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    root_counts = Counter(str(sample.get("root_bucket") or "unknown") for sample in samples)
    loss_counts = Counter(str(sample.get("loss_stage") or "unknown") for sample in samples)
    direction_counts = Counter(str(sample.get("recommended_fix_direction") or "unknown") for sample in samples)
    source_counts = Counter(str((sample.get("source_status") or {}).get("status") or "unknown") for sample in samples)
    index_counts = Counter(str((sample.get("index_status") or {}).get("status") or "unknown") for sample in samples)
    dominant_direction, dominant_count = direction_counts.most_common(1)[0] if direction_counts else ("", 0)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "focus_samples where focus_bucket == doc_hit_parent_miss after support-selector retention run",
        "inputs": inputs,
        "target_sample_count": len(samples),
        "acceptance": {
            "all_samples_have_source_status": all(bool(sample.get("source_status")) for sample in samples),
            "all_samples_have_index_status": all(bool(sample.get("index_status")) for sample in samples),
            "all_samples_have_retrieval_status": all(bool(sample.get("retrieval_status")) for sample in samples),
            "all_samples_have_loss_stage": all(bool(sample.get("loss_stage")) for sample in samples),
        },
        "paths": {
            "child_chunk_jsonl": settings.kb.child_chunk_jsonl,
            "parent_chunk_jsonl": settings.kb.parent_chunk_jsonl,
            "parent_index_path": settings.retrieval.parent_index_path,
            "bm25_cache_path": settings.retrieval.bm25_cache_path,
            "milvus_uri": settings.retrieval.milvus_uri,
            "milvus_collection": settings.retrieval.collection_name,
        },
        "dense_index": dense_index,
        "source_status_counts": dict(source_counts),
        "index_status_counts": dict(index_counts),
        "root_bucket_counts": dict(root_counts),
        "loss_stage_counts": dict(loss_counts),
        "recommended_fix_direction_counts": dict(direction_counts),
        "dominant_root_bucket": root_counts.most_common(1)[0][0] if root_counts else "",
        "next_step_recommendation": {
            "direction": dominant_direction,
            "estimated_affected_samples": dominant_count,
            "sample_ids": [
                str(sample.get("sample_id") or "")
                for sample in samples
                if sample.get("recommended_fix_direction") == dominant_direction
            ],
        },
    }


def render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Gold Child Source-to-Retrieval Audit",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- target samples: **{summary['target_sample_count']}**",
        f"- dominant root bucket: **{summary['dominant_root_bucket']}**",
        f"- recommended next direction: **{summary['next_step_recommendation']['direction']}** "
        f"({summary['next_step_recommendation']['estimated_affected_samples']} samples)",
        "",
        "## Acceptance",
        "",
    ]
    for key, value in summary["acceptance"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Status Counts", ""])
    lines.append("| group | counts |")
    lines.append("|---|---|")
    for key in (
        "source_status_counts",
        "index_status_counts",
        "root_bucket_counts",
        "loss_stage_counts",
        "recommended_fix_direction_counts",
    ):
        lines.append(f"| `{key}` | `{json.dumps(summary[key], ensure_ascii=False, sort_keys=True)}` |")
    lines.extend(["", "## Index Notes", ""])
    dense = summary["dense_index"]
    lines.append(f"- dense query status: `{dense.get('query_status')}`")
    lines.append(f"- dense id kind: `{dense.get('dense_id_kind')}`")
    if dense.get("note"):
        lines.append(f"- dense note: {dense.get('note')}")
    if dense.get("query_error"):
        lines.append(f"- dense query error: `{dense.get('query_error')}`")

    lines.extend(["", "## Sample Detail", ""])
    lines.append(
        "| sample | category | source | index | rewritten dense gp rank | rewritten BM25 gp rank | "
        "RRF actual gp rank | materialized gp rank | rerank pre/post/final | loss_stage |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|")
    for sample in samples:
        rewritten = (sample.get("retrieval_status") or {}).get("rewritten_query") or {}
        dense_rank = rank_or_na(((rewritten.get("dense_top200") or {}).get("gold_parent_child") or {}))
        bm25_rank = rank_or_na(((rewritten.get("bm25_top200") or {}).get("gold_parent_child") or {}))
        stages = rewritten.get("pipeline_stages") or {}
        rrf_rank = rank_or_na(((stages.get("rrf_actual_pool") or {}).get("gold_parent_child") or {}))
        materialized_rank = rank_or_na(((stages.get("parent_materialized") or {}).get("gold_parent_child") or {}))
        rerank = ((sample.get("actual_pipeline") or {}).get("rerank") or {})
        lines.append(
            f"| `{sample['sample_id']}` | `{sample.get('category')}` | "
            f"`{(sample.get('source_status') or {}).get('status')}` | "
            f"`{(sample.get('index_status') or {}).get('status')}` | "
            f"{dense_rank} | {bm25_rank} | {rrf_rank} | {materialized_rank} | "
            f"{fmt_rank(rerank.get('target_pre_floor_rank'))}/"
            f"{fmt_rank(rerank.get('target_post_floor_rank'))}/"
            f"{fmt_rank(rerank.get('target_final_top10_rank'))} | "
            f"`{sample.get('loss_stage')}` |"
        )
    lines.extend(["", "## Files", ""])
    lines.append("- JSON summary: `gold_child_source_to_retrieval_summary.json`")
    lines.append("- JSONL samples: `gold_child_source_to_retrieval_samples.jsonl`")
    return "\n".join(lines)


def rank_value(stage_rank: dict[str, Any]) -> int:
    rank = stage_rank.get("rank") if isinstance(stage_rank, dict) else None
    return int(rank) if isinstance(rank, int) else 999999


def rank_or_na(stage_rank: dict[str, Any]) -> str:
    rank = stage_rank.get("rank") if isinstance(stage_rank, dict) else None
    return str(rank) if rank is not None else "N/A"


def fmt_rank(value: Any) -> str:
    return str(value) if value is not None else "N/A"


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def run_self_test() -> None:
    assert as_list(None) == []
    assert parent_chunk_id("doc_0001_sec01_chunk02::child003") == "doc_0001_sec01_chunk02"
    assert compact_rank({"rank": 3, "chunk_id": "c"})["hit_top200"] is True
    assert route_hit({"gold_parent_child": {"rank": 2}}) is True
    assert stage_hit({"gold_parent_child": {"rank": None}, "gold_child_matched_after_materialize": {"rank": 1}}) is True
    print("self-test ok")


if __name__ == "__main__":
    main()
