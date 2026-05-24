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

from src.synbio_rag.application.alias_expansion_policy import AliasExpansionPolicy
from src.synbio_rag.application.retrieval_query_planner import RetrievalQueryPlanner
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever, _tokenize, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.fusion import reciprocal_rank_fusion_multi
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_RAW_CHILD_TRACE_DIR = RESULTS_ROOT / "v3_raw_child_trace_20260523_raw_child_trace_debug"
TOP_N = 200
PARENT_ID_RE = re.compile(r"^(?P<doc>.+?)_sec(?P<sec>\d+)_chunk(?P<chunk>\d+)$")

BASELINE_RETRIEVAL_OVERRIDES: dict[str, Any] = {
    "retrieval.hybrid_enabled": True,
    "retrieval.bm25_enabled": True,
    "retrieval.source_floor_enabled": True,
    "retrieval.rerank_mode": "plain",
    "retrieval.rerank_top_k": 10,
    "retrieval.final_top_k": 8,
    "retrieval.table_preview_enabled": True,
    "retrieval.table_preview_merge_enabled": False,
    "retrieval.table_preview_allow_formal_citation": False,
    "retrieval.original_cn_fallback_enabled": False,
    "query_rewrite.mode": "off",
    "table_enhancement.enabled": False,
}

CLASSIFICATION_LABELS = {
    "gold_child_not_indexed": "gold child 不存在于 child_chunks.jsonl",
    "gold_child_parent_mapping_missing": "gold child 缺少 parent_index 映射",
    "dense_miss_bm25_hit": "BM25 召回 gold child，但 dense parent proxy 未召回",
    "bm25_miss_dense_hit": "dense parent proxy 召回 gold parent，但 BM25 未召回 gold child",
    "both_dense_bm25_miss": "dense parent proxy 与 BM25 gold child 均未召回",
    "fusion_dropped_gold_child": "source 召回后，RRF/fusion top200 未保留 gold signal",
    "source_floor_or_postprocess_dropped": "fusion 有 gold signal，但默认 raw 输出未保留",
    "query_text_mismatch": "query 与 gold child 词面重叠弱于 wrong candidate",
    "chunk_boundary_mismatch": "同文档相邻 parent 命中，疑似 chunk boundary 漂移",
    "metadata_or_id_mismatch": "child metadata、gold id 或 parent id 不一致",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v3 same-doc retrieval drift cases.")
    parser.add_argument("--raw-child-trace-dir", default=str(DEFAULT_RAW_CHILD_TRACE_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    raw_dir = Path(args.raw_child_trace_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw child trace dir not found: {raw_dir}")

    run_id = args.run_id or derive_run_id(raw_dir)
    output_dir = RESULTS_ROOT / f"v3_retrieval_drift_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_retrieval_drift_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    apply_overrides(settings, BASELINE_RETRIEVAL_OVERRIDES)

    child_records = load_jsonl_by_id(Path(settings.kb.child_chunk_jsonl), "chunk_id")
    parent_records = load_jsonl_by_id(Path(settings.kb.parent_chunk_jsonl), "chunk_id")
    parent_index = load_parent_index(Path(settings.retrieval.parent_index_path))
    prober = RetrievalProber(settings)

    probe_cache: dict[str, dict[str, Any]] = {}
    samples_by_variant: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = load_jsonl(raw_dir / f"{variant}_samples.jsonl")
        targets = [
            row
            for row in rows
            if row.get("classification") == "same_doc_child_wrong_parent"
            and row.get("gold_child_chunk_ids")
        ]
        samples: list[dict[str, Any]] = []
        for row in targets:
            question = str(row.get("question") or "")
            if question not in probe_cache:
                probe_cache[question] = prober.probe(question, top_n=TOP_N)
            samples.append(
                audit_sample(
                    variant=variant,
                    row=row,
                    probe=probe_cache[question],
                    child_records=child_records,
                    parent_records=parent_records,
                    parent_index=parent_index,
                    dense_id_kind=prober.dense_id_kind,
                )
            )
        samples_by_variant[variant] = samples
        summaries[variant] = summarize_samples(samples)
        write_jsonl(output_dir / f"{variant}_samples.jsonl", samples)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_child_trace_dir": str(raw_dir),
        "child_chunk_jsonl": settings.kb.child_chunk_jsonl,
        "parent_chunk_jsonl": settings.kb.parent_chunk_jsonl,
        "parent_index_path": settings.retrieval.parent_index_path,
        "dense_index_id_kind": prober.dense_id_kind,
        "scope": "classification == same_doc_child_wrong_parent and gold_child_chunk_ids present",
        "top_n": TOP_N,
        "variants": summaries,
        "comparison": compare_variants(samples_by_variant),
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary, samples_by_variant))
    print(json.dumps({"output_dir": str(output_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


class RetrievalProber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config = settings.retrieval
        self.router = QueryRouter(settings.retrieval)
        self.query_planner = RetrievalQueryPlanner()
        self.alias_policy = AliasExpansionPolicy()
        self.embedder = BGEM3Embedder(
            model_path=settings.kb.embedding_model_path,
            dim=settings.kb.embedding_dim,
            max_length=settings.kb.embedding_max_length,
        )
        self.dense_retriever = MilvusRetriever(settings.retrieval, self.embedder)
        self.bm25_retriever = BM25Retriever(
            retrieval_config=settings.retrieval,
            kb_config=settings.kb,
            milvus_client=self.dense_retriever.client,
        )
        self.dense_id_kind = infer_dense_id_kind(
            self.dense_retriever.client,
            settings.retrieval.collection_name,
        )

    def probe(self, question: str, *, top_n: int) -> dict[str, Any]:
        analysis = self.router.analyze(question)
        query_plan = self.query_planner.build(question, analysis, self.config, question)
        dense_runs = []
        sparse_runs = []
        variant_debug: list[dict[str, Any]] = []

        dense_weight = self.config.dense_rrf_weight
        bm25_weight = self.config.bm25_rrf_weight
        if self.query_planner.contains_cjk(question):
            bm25_weight *= self.config.cjk_query_bm25_weight

        for idx, variant in enumerate(query_plan):
            dense_results = self.dense_retriever.search(str(variant["query"]), limit=top_n)
            sparse_query = str(variant["query"])
            if getattr(self.config, "alias_expansion_enabled", False):
                sparse_query = self.alias_policy.expand(sparse_query, self.config)
            sparse_results = self.bm25_retriever.search(sparse_query, limit=top_n)
            dense_runs.append((dense_results, dense_weight * float(variant["weight"])))
            sparse_runs.append((sparse_results, bm25_weight * float(variant["weight"])))
            variant_debug.append(
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

        fused = reciprocal_rank_fusion_multi(
            dense_runs=dense_runs,
            sparse_runs=sparse_runs,
            limit=top_n,
            rrf_k=self.config.rrf_k,
        )
        return {
            "question": question,
            "intent": analysis.intent.value,
            "search_limit": analysis.search_limit,
            "actual_dense_limit": max(analysis.search_limit, self.config.dense_limit),
            "actual_bm25_limit": max(analysis.search_limit, self.config.bm25_limit),
            "actual_fusion_pool_limit": max(analysis.search_limit * 3, analysis.search_limit + 12),
            "query_variants": variant_debug,
            "dense_runs": dense_runs,
            "sparse_runs": sparse_runs,
            "fused": fused,
        }


def audit_sample(
    *,
    variant: str,
    row: dict[str, Any],
    probe: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    parent_index: dict[str, Any],
    dense_id_kind: str,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    question = str(row.get("question") or "")
    gold_children = [str(item) for item in row.get("gold_child_chunk_ids") or []]
    gold_parents = [str(item) for item in row.get("gold_parent_chunk_ids") or []]
    expected_docs = [str(item) for item in row.get("expected_doc_ids") or []]
    raw_preview = [item for item in row.get("raw_child_trace_preview") or [] if isinstance(item, dict)]

    child_checks = [
        audit_gold_child_mapping(gold_child, gold_parents, child_records, parent_index)
        for gold_child in gold_children
    ]
    indexed = all(item["exists_in_child_chunks"] for item in child_checks) if child_checks else False
    mapping_ok = all(item["maps_to_gold_parent"] for item in child_checks) if child_checks else False
    metadata_mismatch = any(item["metadata_parent_mismatch"] for item in child_checks)

    dense_exact = best_rank_across_runs(probe["dense_runs"], set(gold_children), by_parent=False)
    dense_parent = best_rank_across_runs(probe["dense_runs"], set(gold_parents), by_parent=True)
    bm25_exact = best_rank_across_runs(probe["sparse_runs"], set(gold_children), by_parent=False)
    bm25_parent = best_rank_across_runs(probe["sparse_runs"], set(gold_parents), by_parent=True)
    fused_exact = rank_in_chunks(probe["fused"], set(gold_children), by_parent=False)
    fused_parent = rank_in_chunks(probe["fused"], set(gold_parents), by_parent=True)
    raw_gold_child_rank = min([int(rank) for rank in row.get("gold_child_raw_ranks") or []], default=None)
    raw_gold_parent_hit = bool(row.get("raw_parent_ids_contains_gold_parent"))

    wrong_candidate = choose_wrong_candidate(raw_preview, expected_docs, gold_parents)
    gold_record = first_record(gold_children, child_records)
    wrong_record = record_for_candidate(wrong_candidate, child_records, parent_records)
    comparison = compare_gold_to_wrong(
        question=question,
        gold_record=gold_record,
        wrong_record=wrong_record,
        wrong_candidate=wrong_candidate,
        gold_parents=gold_parents,
    )

    diagnostic_tags = build_diagnostic_tags(
        indexed=indexed,
        mapping_ok=mapping_ok,
        metadata_mismatch=metadata_mismatch,
        comparison=comparison,
    )
    classification = classify_sample(
        indexed=indexed,
        mapping_ok=mapping_ok,
        metadata_mismatch=metadata_mismatch,
        dense_signal=dense_parent or dense_exact,
        bm25_signal=bm25_exact,
        fused_signal=fused_parent or fused_exact,
        raw_gold_child_rank=raw_gold_child_rank,
        raw_gold_parent_hit=raw_gold_parent_hit,
        diagnostic_tags=diagnostic_tags,
    )
    drop_stage = classify_drop_stage(
        indexed=indexed,
        mapping_ok=mapping_ok,
        dense_signal=dense_parent or dense_exact,
        bm25_signal=bm25_exact,
        fused_signal=fused_parent or fused_exact,
        raw_gold_child_rank=raw_gold_child_rank,
        raw_gold_parent_hit=raw_gold_parent_hit,
        probe=probe,
    )

    return {
        "variant_key": variant,
        "sample_id": sample_id,
        "category": row.get("category"),
        "question": question,
        "expected_doc_ids": expected_docs,
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "dense_index_id_kind": dense_id_kind,
        "gold_child_checks": child_checks,
        "gold_child_exists_in_child_chunks": indexed,
        "gold_child_maps_to_gold_parent": mapping_ok,
        "metadata_parent_mismatch": metadata_mismatch,
        "retrieval_probe": {
            "intent": probe["intent"],
            "search_limit": probe["search_limit"],
            "actual_dense_limit": probe["actual_dense_limit"],
            "actual_bm25_limit": probe["actual_bm25_limit"],
            "actual_fusion_pool_limit": probe["actual_fusion_pool_limit"],
            "query_variants": probe["query_variants"],
        },
        "dense_gold_child_exact": topn_presence(dense_exact),
        "dense_gold_parent_proxy": topn_presence(dense_parent),
        "bm25_gold_child_exact": topn_presence(bm25_exact),
        "bm25_gold_parent_proxy": topn_presence(bm25_parent),
        "fused_gold_child_exact": topn_presence(fused_exact),
        "fused_gold_parent_proxy": topn_presence(fused_parent),
        "raw_gold_child_rank": raw_gold_child_rank,
        "raw_gold_parent_hit": raw_gold_parent_hit,
        "raw_expected_doc_first_child_rank": row.get("raw_expected_doc_first_child_rank"),
        "raw_expected_doc_parent_ids": row.get("raw_expected_doc_parent_ids") or [],
        "wrong_candidate_comparison": comparison,
        "drop_stage": drop_stage,
        "classification": classification,
        "diagnostic_tags": diagnostic_tags,
    }


def audit_gold_child_mapping(
    gold_child: str,
    gold_parents: list[str],
    child_records: dict[str, dict[str, Any]],
    parent_index: dict[str, Any],
) -> dict[str, Any]:
    child = child_records.get(gold_child)
    if child is None:
        return {
            "gold_child_chunk_id": gold_child,
            "exists_in_child_chunks": False,
            "child_parent_chunk_id": "",
            "child_parent_id": "",
            "parent_index_record_found": False,
            "parent_index_parent_chunk_id": "",
            "parent_index_contains_child": False,
            "maps_to_gold_parent": False,
            "metadata_parent_mismatch": False,
        }

    child_parent_chunk_id = str(child.get("parent_chunk_id") or parent_chunk_id(gold_child))
    child_parent_id = str(child.get("parent_id") or "")
    parent_record = parent_index["by_parent_id"].get(child_parent_id)
    if parent_record is None:
        parent_record = parent_index["by_child_id"].get(gold_child)
    parent_index_parent_chunk_id = str((parent_record or {}).get("parent_chunk_id") or "")
    parent_index_contains_child = gold_child in set((parent_record or {}).get("child_chunk_ids") or [])
    maps_to_gold_parent = (
        child_parent_chunk_id in set(gold_parents)
        and parent_record is not None
        and parent_index_parent_chunk_id in set(gold_parents)
        and parent_index_contains_child
    )
    return {
        "gold_child_chunk_id": gold_child,
        "exists_in_child_chunks": True,
        "child_parent_chunk_id": child_parent_chunk_id,
        "child_parent_id": child_parent_id,
        "parent_index_record_found": parent_record is not None,
        "parent_index_parent_chunk_id": parent_index_parent_chunk_id,
        "parent_index_contains_child": parent_index_contains_child,
        "maps_to_gold_parent": maps_to_gold_parent,
        "metadata_parent_mismatch": child_parent_chunk_id not in set(gold_parents),
    }


def best_rank_across_runs(
    weighted_runs: list[tuple[list[Any], float]],
    target_ids: set[str],
    *,
    by_parent: bool,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for variant_index, (chunks, _weight) in enumerate(weighted_runs):
        hit = rank_in_chunks(chunks, target_ids, by_parent=by_parent)
        if hit is None:
            continue
        hit["query_variant_index"] = variant_index
        if best is None or int(hit["rank"]) < int(best["rank"]):
            best = hit
    return best


def rank_in_chunks(chunks: list[Any], target_ids: set[str], *, by_parent: bool) -> dict[str, Any] | None:
    if not target_ids:
        return None
    for rank, chunk in enumerate(chunks, start=1):
        chunk_id = str(getattr(chunk, "chunk_id", "") or "")
        candidate_id = chunk_parent_id(chunk) if by_parent else chunk_id
        if candidate_id in target_ids:
            return {
                "rank": rank,
                "chunk_id": chunk_id,
                "parent_chunk_id": chunk_parent_id(chunk),
                "doc_id": str(getattr(chunk, "doc_id", "") or ""),
                "section": str(getattr(chunk, "section", "") or ""),
                "vector_score": round_number(getattr(chunk, "vector_score", 0.0)),
                "bm25_score": round_number(getattr(chunk, "bm25_score", 0.0)),
                "fusion_score": round_number(getattr(chunk, "fusion_score", 0.0)),
            }
    return None


def topn_presence(hit: dict[str, Any] | None) -> dict[str, Any]:
    rank = int(hit["rank"]) if hit else None
    result = {
        "hit_top100": bool(rank and rank <= 100),
        "hit_top200": bool(rank and rank <= 200),
        "rank": rank,
    }
    if hit:
        result.update(hit)
    return result


def choose_wrong_candidate(
    raw_preview: list[dict[str, Any]],
    expected_docs: list[str],
    gold_parents: list[str],
) -> dict[str, Any]:
    expected_doc_set = set(expected_docs)
    gold_parent_set = set(gold_parents)
    candidates = [
        item
        for item in raw_preview
        if str(item.get("doc_id") or "") in expected_doc_set
        and str(item.get("parent_chunk_id") or "") not in gold_parent_set
    ]
    candidates.sort(key=lambda item: int(item.get("rank") or 999999))
    return candidates[0] if candidates else {}


def record_for_candidate(
    candidate: dict[str, Any],
    child_records: dict[str, dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidate:
        return None
    child_id = str(candidate.get("child_chunk_id") or "")
    parent_id_value = str(candidate.get("parent_chunk_id") or "")
    return child_records.get(child_id) or parent_records.get(child_id) or parent_records.get(parent_id_value)


def first_record(ids: list[str], records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for item in ids:
        if item in records:
            return records[item]
    return None


def compare_gold_to_wrong(
    *,
    question: str,
    gold_record: dict[str, Any] | None,
    wrong_record: dict[str, Any] | None,
    wrong_candidate: dict[str, Any],
    gold_parents: list[str],
) -> dict[str, Any]:
    gold_overlap = query_overlap(question, gold_record)
    wrong_overlap = query_overlap(question, wrong_record)
    wrong_parent = str(wrong_candidate.get("parent_chunk_id") or "")
    gold_parent = str((gold_record or {}).get("parent_chunk_id") or "")
    return {
        "gold": compact_record(gold_record, gold_overlap),
        "wrong": {
            **compact_record(wrong_record, wrong_overlap),
            "raw_rank": wrong_candidate.get("rank"),
            "raw_child_chunk_id": wrong_candidate.get("child_chunk_id"),
            "raw_parent_chunk_id": wrong_parent,
            "raw_source": wrong_candidate.get("source"),
            "raw_vector_score": wrong_candidate.get("vector_score"),
            "raw_bm25_score": wrong_candidate.get("bm25_score"),
            "raw_fusion_score": wrong_candidate.get("fusion_score"),
        },
        "parent_distance": nearest_parent_distance([wrong_parent], gold_parents),
        "same_section": bool(gold_record and wrong_record and gold_record.get("section") == wrong_record.get("section")),
        "gold_parent_chunk_id": gold_parent,
        "wrong_parent_chunk_id": wrong_parent,
    }


def compact_record(record: dict[str, Any] | None, overlap: dict[str, Any]) -> dict[str, Any]:
    if record is None:
        return {
            "chunk_id": "",
            "parent_chunk_id": "",
            "section": "",
            "chunk_index": None,
            "child_index": None,
            "overlap": overlap,
            "text_preview": "",
        }
    return {
        "chunk_id": record.get("chunk_id"),
        "parent_chunk_id": record.get("parent_chunk_id") or parent_chunk_id(record.get("chunk_id")),
        "section": record.get("section"),
        "chunk_index": record.get("chunk_index"),
        "child_index": record.get("child_index"),
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "block_types": record.get("block_types") or [],
        "evidence_types": record.get("evidence_types") or [],
        "overlap": overlap,
        "text_preview": preview(record.get("text") or record.get("retrieval_text") or "", 360),
    }


def query_overlap(question: str, record: dict[str, Any] | None) -> dict[str, Any]:
    query_terms = tokenize_query(question)
    if not query_terms:
        query_terms = _tokenize(question)
    query_set = set(query_terms)
    if record is None or not query_set:
        return {"query_term_count": len(query_set), "overlap_count": 0, "overlap_ratio": 0.0, "terms": []}
    text = str(record.get("retrieval_text") or record.get("text") or "")
    record_terms = set(_tokenize(text))
    terms = sorted(query_set & record_terms)
    return {
        "query_term_count": len(query_set),
        "overlap_count": len(terms),
        "overlap_ratio": round_number(len(terms) / len(query_set)),
        "terms": terms[:30],
    }


def build_diagnostic_tags(
    *,
    indexed: bool,
    mapping_ok: bool,
    metadata_mismatch: bool,
    comparison: dict[str, Any],
) -> list[str]:
    tags: list[str] = []
    if indexed and not mapping_ok:
        tags.append("metadata_or_id_mismatch" if metadata_mismatch else "gold_child_parent_mapping_missing")
    gold_overlap = (comparison.get("gold") or {}).get("overlap") or {}
    wrong_overlap = (comparison.get("wrong") or {}).get("overlap") or {}
    gold_ratio = float(gold_overlap.get("overlap_ratio") or 0.0)
    wrong_ratio = float(wrong_overlap.get("overlap_ratio") or 0.0)
    if wrong_ratio > gold_ratio and (gold_ratio == 0.0 or wrong_ratio >= gold_ratio * 2):
        tags.append("query_text_mismatch")
    distance = comparison.get("parent_distance")
    if isinstance(distance, int) and distance <= 1:
        tags.append("chunk_boundary_mismatch")
    return dedupe(tags)


def classify_sample(
    *,
    indexed: bool,
    mapping_ok: bool,
    metadata_mismatch: bool,
    dense_signal: dict[str, Any] | None,
    bm25_signal: dict[str, Any] | None,
    fused_signal: dict[str, Any] | None,
    raw_gold_child_rank: int | None,
    raw_gold_parent_hit: bool,
    diagnostic_tags: list[str],
) -> str:
    if not indexed:
        return "gold_child_not_indexed"
    if not mapping_ok:
        return "metadata_or_id_mismatch" if metadata_mismatch else "gold_child_parent_mapping_missing"
    if dense_signal is None and bm25_signal is None:
        if "query_text_mismatch" in diagnostic_tags:
            return "query_text_mismatch"
        if "chunk_boundary_mismatch" in diagnostic_tags:
            return "chunk_boundary_mismatch"
        return "both_dense_bm25_miss"
    if dense_signal is None and bm25_signal is not None:
        return "dense_miss_bm25_hit"
    if bm25_signal is None and dense_signal is not None:
        return "bm25_miss_dense_hit"
    if fused_signal is None:
        return "fusion_dropped_gold_child"
    if raw_gold_child_rank is None and not raw_gold_parent_hit:
        return "source_floor_or_postprocess_dropped"
    return "source_floor_or_postprocess_dropped"


def classify_drop_stage(
    *,
    indexed: bool,
    mapping_ok: bool,
    dense_signal: dict[str, Any] | None,
    bm25_signal: dict[str, Any] | None,
    fused_signal: dict[str, Any] | None,
    raw_gold_child_rank: int | None,
    raw_gold_parent_hit: bool,
    probe: dict[str, Any],
) -> str:
    if not indexed:
        return "index_missing"
    if not mapping_ok:
        return "parent_mapping"
    if dense_signal is None and bm25_signal is None:
        return "dense_bm25_candidate_generation_top200"
    if not source_signal_within_actual_limits(dense_signal, bm25_signal, probe):
        return "source_rank_beyond_actual_dense_bm25_limit"
    if fused_signal is None:
        return "rrf_fusion_top200"
    if int(fused_signal.get("rank") or 999999) > int(probe.get("actual_fusion_pool_limit") or 0):
        return "fused_rank_beyond_actual_pool"
    if int(fused_signal.get("rank") or 999999) > int(probe.get("search_limit") or 0):
        return "fused_rank_beyond_default_final_limit"
    if raw_gold_child_rank is None and not raw_gold_parent_hit:
        return "source_floor_or_postprocess"
    return "not_dropped_in_raw"


def source_signal_within_actual_limits(
    dense_signal: dict[str, Any] | None,
    bm25_signal: dict[str, Any] | None,
    probe: dict[str, Any],
) -> bool:
    if dense_signal is not None and int(dense_signal.get("rank") or 999999) <= int(probe["actual_dense_limit"]):
        return True
    if bm25_signal is not None and int(bm25_signal.get("rank") or 999999) <= int(probe["actual_bm25_limit"]):
        return True
    return False


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(sample.get("classification") or "unknown") for sample in samples)
    tag_counts = Counter(tag for sample in samples for tag in sample.get("diagnostic_tags") or [])
    return {
        "target_case_count": len(samples),
        "classification_counts": dict(counts),
        "diagnostic_tag_counts": dict(tag_counts),
        "gold_child_indexed_count": sum(1 for sample in samples if sample.get("gold_child_exists_in_child_chunks")),
        "gold_child_parent_mapping_ok_count": sum(1 for sample in samples if sample.get("gold_child_maps_to_gold_parent")),
        "dense_gold_child_exact_top100_count": count_top(samples, "dense_gold_child_exact", "hit_top100"),
        "dense_gold_child_exact_top200_count": count_top(samples, "dense_gold_child_exact", "hit_top200"),
        "dense_gold_parent_proxy_top100_count": count_top(samples, "dense_gold_parent_proxy", "hit_top100"),
        "dense_gold_parent_proxy_top200_count": count_top(samples, "dense_gold_parent_proxy", "hit_top200"),
        "bm25_gold_child_exact_top100_count": count_top(samples, "bm25_gold_child_exact", "hit_top100"),
        "bm25_gold_child_exact_top200_count": count_top(samples, "bm25_gold_child_exact", "hit_top200"),
        "fused_gold_child_exact_top100_count": count_top(samples, "fused_gold_child_exact", "hit_top100"),
        "fused_gold_child_exact_top200_count": count_top(samples, "fused_gold_child_exact", "hit_top200"),
        "fused_gold_parent_proxy_top100_count": count_top(samples, "fused_gold_parent_proxy", "hit_top100"),
        "fused_gold_parent_proxy_top200_count": count_top(samples, "fused_gold_parent_proxy", "hit_top200"),
    }


def count_top(samples: list[dict[str, Any]], field: str, key: str) -> int:
    return sum(1 for sample in samples if (sample.get(field) or {}).get(key))


def compare_variants(samples_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    b0 = {str(sample.get("sample_id") or ""): sample for sample in samples_by_variant.get("b0_stable", [])}
    b1 = {str(sample.get("sample_id") or ""): sample for sample in samples_by_variant.get("b1_parent_expansion", [])}
    shared = sorted(set(b0) & set(b1))
    changed = [
        sample_id
        for sample_id in shared
        if b0[sample_id].get("classification") != b1[sample_id].get("classification")
        or b0[sample_id].get("drop_stage") != b1[sample_id].get("drop_stage")
    ]
    return {
        "same_sample_ids": set(b0) == set(b1),
        "shared_sample_count": len(shared),
        "b0_only": sorted(set(b0) - set(b1)),
        "b1_only": sorted(set(b1) - set(b0)),
        "classification_or_stage_changed_count": len(changed),
        "changed_sample_ids": changed,
    }


def render_report(summary: dict[str, Any], samples_by_variant: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Retrieval Drift 审计报告",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- raw_child_trace_dir: `{summary['raw_child_trace_dir']}`",
        f"- scope: `{summary['scope']}`",
        f"- dense_index_id_kind: `{summary['dense_index_id_kind']}`",
        "",
        "## 关键说明",
        "",
        "- 当前 dense/Milvus 返回的是 parent/chunk id，不是 `::childNNN` id；因此报告同时给出 `dense_gold_child_exact` 和 `dense_gold_parent_proxy`。",
        "- BM25 缓存来自 `child_chunks.jsonl`，`bm25_gold_child_exact` 才是 child 级精确召回信号。",
        "- 本脚本只做离线 debug/audit；未调用 judge，未修改检索、rerank、support/citation 或生成行为。",
        "",
        "## 总览",
        "",
        "| 变体 | 目标样本 | child indexed | parent mapping ok | dense parent@200 | BM25 child@200 | fused parent@200 | fused child@200 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, item in summary["variants"].items():
        lines.append(
            f"| {variant} | {item['target_case_count']} | {item['gold_child_indexed_count']} | "
            f"{item['gold_child_parent_mapping_ok_count']} | {item['dense_gold_parent_proxy_top200_count']} | "
            f"{item['bm25_gold_child_exact_top200_count']} | {item['fused_gold_parent_proxy_top200_count']} | "
            f"{item['fused_gold_child_exact_top200_count']} |"
        )

    lines.extend(["", "## 分类", ""])
    lines.append("| 分类 | 含义 | b0_stable | b1_parent_expansion |")
    lines.append("|---|---|---:|---:|")
    classes = sorted(
        set(summary["variants"].get("b0_stable", {}).get("classification_counts", {}))
        | set(summary["variants"].get("b1_parent_expansion", {}).get("classification_counts", {}))
    )
    for name in classes:
        b0 = summary["variants"].get("b0_stable", {}).get("classification_counts", {}).get(name, 0)
        b1 = summary["variants"].get("b1_parent_expansion", {}).get("classification_counts", {}).get(name, 0)
        lines.append(f"| `{name}` | {CLASSIFICATION_LABELS.get(name, name)} | {b0} | {b1} |")

    lines.extend(["", "## 样本明细", ""])
    lines.append("| 变体 | sample_id | 分类 | drop_stage | dense parent rank | BM25 child rank | fused parent rank | wrong distance | overlap gold/wrong |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for variant, samples in samples_by_variant.items():
        for sample in samples:
            comparison = sample.get("wrong_candidate_comparison") or {}
            gold_overlap = ((comparison.get("gold") or {}).get("overlap") or {}).get("overlap_ratio")
            wrong_overlap = ((comparison.get("wrong") or {}).get("overlap") or {}).get("overlap_ratio")
            lines.append(
                f"| {variant} | `{sample['sample_id']}` | `{sample['classification']}` | "
                f"`{sample['drop_stage']}` | {rank(sample.get('dense_gold_parent_proxy'))} | "
                f"{rank(sample.get('bm25_gold_child_exact'))} | {rank(sample.get('fused_gold_parent_proxy'))} | "
                f"{fmt(comparison.get('parent_distance'))} | {fmt(gold_overlap)}/{fmt(wrong_overlap)} |"
            )
    return "\n".join(lines)


def derive_run_id(raw_dir: Path) -> str:
    prefix = "v3_raw_child_trace_"
    name = raw_dir.name
    if name.startswith(prefix):
        return name[len(prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def apply_overrides(settings: Settings, overrides: dict[str, Any]) -> None:
    for dotted, value in overrides.items():
        target: Any = settings
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)


def infer_dense_id_kind(client: Any, collection_name: str) -> str:
    rows = client.query(
        collection_name=collection_name,
        filter="",
        limit=20,
        output_fields=["chunk_id"],
    )
    ids = [str(row.get("chunk_id") or "") for row in rows]
    if ids and all("::child" in item for item in ids):
        return "child_chunk_id"
    if ids and all("::child" not in item for item in ids):
        return "parent_or_base_chunk_id"
    return "mixed_or_unknown"


def load_parent_index(path: Path) -> dict[str, dict[str, Any]]:
    by_parent_id: dict[str, dict[str, Any]] = {}
    by_child_id: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(path):
        parent_id_value = str(item.get("parent_id") or "")
        if parent_id_value:
            by_parent_id[parent_id_value] = item
        if item.get("parent_type") == "retrieval_parent":
            for child_id in item.get("child_chunk_ids") or []:
                by_child_id[str(child_id)] = item
    return {"by_parent_id": by_parent_id, "by_child_id": by_child_id}


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get(key) or "")
            if item_id:
                records[item_id] = item
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def chunk_parent_id(chunk: Any) -> str:
    metadata = getattr(chunk, "metadata", {}) or {}
    parent_id_value = str(metadata.get("parent_chunk_id") or "")
    if parent_id_value:
        return parent_id_value
    return parent_chunk_id(getattr(chunk, "chunk_id", ""))


def parent_chunk_id(value: Any) -> str:
    text = str(value or "")
    if "::child" in text:
        return text.split("::child", 1)[0]
    return text


def nearest_parent_distance(parent_ids: list[str], gold_parents: list[str]) -> int | None:
    distances: list[int] = []
    for parent_id_value in parent_ids:
        parsed_parent = parse_parent_id(parent_id_value)
        if parsed_parent is None:
            continue
        for gold_parent in gold_parents:
            parsed_gold = parse_parent_id(gold_parent)
            if parsed_gold is None or parsed_parent[0] != parsed_gold[0]:
                continue
            distances.append(abs(parsed_parent[2] - parsed_gold[2]))
    return min(distances) if distances else None


def parse_parent_id(value: str) -> tuple[str, int, int] | None:
    match = PARENT_ID_RE.match(str(value or ""))
    if not match:
        return None
    return match.group("doc"), int(match.group("sec")), int(match.group("chunk"))


def preview(text: Any, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def round_number(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def rank(item: dict[str, Any] | None) -> str:
    value = (item or {}).get("rank")
    return str(value) if value is not None else "N/A"


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def run_self_test() -> None:
    assert parent_chunk_id("doc_a_sec01_chunk02::child003") == "doc_a_sec01_chunk02"
    assert parent_chunk_id("doc_a_sec01_chunk02") == "doc_a_sec01_chunk02"
    assert nearest_parent_distance(["doc_a_sec02_chunk03"], ["doc_a_sec01_chunk02"]) == 1
    child_records = {
        "doc_a_sec01_chunk02::child001": {
            "chunk_id": "doc_a_sec01_chunk02::child001",
            "parent_id": "p1",
            "parent_chunk_id": "doc_a_sec01_chunk02",
            "text": "alpha beta",
        }
    }
    parent_index = {
        "by_parent_id": {
            "p1": {
                "parent_id": "p1",
                "parent_chunk_id": "doc_a_sec01_chunk02",
                "child_chunk_ids": ["doc_a_sec01_chunk02::child001"],
            }
        },
        "by_child_id": {},
    }
    check = audit_gold_child_mapping(
        "doc_a_sec01_chunk02::child001",
        ["doc_a_sec01_chunk02"],
        child_records,
        parent_index,
    )
    assert check["maps_to_gold_parent"], check
    classification = classify_sample(
        indexed=True,
        mapping_ok=True,
        metadata_mismatch=False,
        dense_signal=None,
        bm25_signal=None,
        fused_signal=None,
        raw_gold_child_rank=None,
        raw_gold_parent_hit=False,
        diagnostic_tags=[],
    )
    assert classification == "both_dense_bm25_miss", classification
    print("self-test passed")


if __name__ == "__main__":
    main()
