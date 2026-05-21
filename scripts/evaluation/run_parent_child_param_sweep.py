#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.application.pipeline_stages import (
    ContextStage,
    GenerationStage,
    RerankStage,
    ResponseStage,
    RetrievalStage,
)
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters, RAGPipelineResponse, RetrievedChunk


DEFAULT_DATASET = (
    ROOT / "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl"
)
DEFAULT_MILVUS_URI = ROOT / "runtime/vectorstores/milvus/parent_child_round1_160.db"
DEFAULT_COLLECTION = "synbio_parent_child_round1_pc160"
DEFAULT_CHILD_CHUNKS = ROOT / "data/paper_round1/chunks/child_chunks.jsonl"
DEFAULT_PARENT_CHUNKS = ROOT / "data/paper_round1/chunks/parent_chunks.jsonl"
DEFAULT_PARENT_INDEX = ROOT / "data/paper_round1/chunks/parent_index.jsonl"
DEFAULT_BM25 = ROOT / "data/paper_round1/chunks/bm25_index.json"
DEFAULT_RESULTS_DIR = ROOT / "results/parent_child_param_sweep"
DEFAULT_REPORTS_DIR = ROOT / "reports/parent_child_param_sweep"


@dataclass(frozen=True)
class SweepProfile:
    name: str
    retrieval: dict[str, Any] = field(default_factory=dict)
    parent: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    def merged_with(self, other: "SweepProfile", name: str | None = None) -> "SweepProfile":
        return SweepProfile(
            name=name or other.name,
            retrieval={**self.retrieval, **other.retrieval},
            parent={**self.parent, **other.parent},
            generation={**self.generation, **other.generation},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "retrieval": dict(self.retrieval),
            "parent": dict(self.parent),
            "generation": dict(self.generation),
        }


@dataclass(frozen=True)
class Phase5FSample:
    sample_id: str
    query: str
    query_type: str
    target_doc_id: str
    stable_target_block_ids: tuple[str, ...]
    target_chunk_id_candidate: str = ""
    include_in_main_denominator: bool = True
    target_doc_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        target_doc_ids = self.target_doc_ids or ((self.target_doc_id,) if self.target_doc_id else ())
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "query_type": self.query_type,
            "target_doc_id": self.target_doc_id,
            "target_doc_ids": list(target_doc_ids),
            "stable_target_block_ids": list(self.stable_target_block_ids),
            "target_chunk_id_candidate": self.target_chunk_id_candidate,
            "include_in_main_denominator": self.include_in_main_denominator,
        }


@dataclass
class PipelineTrace:
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    seed_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    parent_expansion_debug: dict[str, Any]
    response: RAGPipelineResponse


RECALL_PROFILES: tuple[SweepProfile, ...] = (
    SweepProfile(
        name="baseline",
        retrieval={
            "dense_limit": 40,
            "bm25_limit": 40,
            "search_limit": 40,
            "rerank_top_k": 10,
            "final_top_k": 8,
        },
    ),
    SweepProfile(
        name="recall_80_rerank10",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 10,
            "final_top_k": 8,
        },
    ),
    SweepProfile(
        name="recall_80_rerank20",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 8,
        },
    ),
    SweepProfile(
        name="recall_80_rerank20_final12",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 12,
        },
    ),
    SweepProfile(
        name="recall_100_rerank20_final12",
        retrieval={
            "dense_limit": 100,
            "bm25_limit": 100,
            "search_limit": 100,
            "rerank_top_k": 20,
            "final_top_k": 12,
        },
    ),
)

PARENT_PROFILES: tuple[SweepProfile, ...] = (
    SweepProfile(
        name="parent_disabled",
        parent={
            "parent_expansion_enabled": False,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="parent_default",
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="parent_tighter",
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 8,
            "parent_expansion_per_seed_limit": 1,
        },
    ),
    SweepProfile(
        name="parent_wider",
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 16,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="parent_wider_per_seed",
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 18,
            "parent_expansion_per_seed_limit": 3,
        },
    ),
)

GENERATION_PROFILES: tuple[SweepProfile, ...] = (
    SweepProfile(
        name="gen_default",
        generation={
            "v2_max_support_factoid": 3,
            "v2_max_support_summary": 5,
            "v2_max_support_comparison": 6,
            "v2_max_extractive_evidence_lines": 6,
        },
    ),
    SweepProfile(
        name="gen_factoid_tight",
        generation={
            "v2_max_support_factoid": 2,
            "v2_max_support_summary": 5,
            "v2_max_support_comparison": 6,
            "v2_max_extractive_evidence_lines": 4,
        },
    ),
    SweepProfile(
        name="gen_summary_wide",
        generation={
            "v2_max_support_factoid": 3,
            "v2_max_support_summary": 6,
            "v2_max_support_comparison": 8,
            "v2_max_extractive_evidence_lines": 6,
        },
    ),
)

AB500_PROFILES: tuple[SweepProfile, ...] = (
    SweepProfile(
        name="baseline",
        retrieval={
            "dense_limit": 40,
            "bm25_limit": 40,
            "search_limit": 40,
            "rerank_top_k": 10,
            "final_top_k": 8,
        },
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="recall80_f8_parent_default",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 8,
        },
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="recall80_f12_parent_disabled",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 12,
        },
        parent={
            "parent_expansion_enabled": False,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="recall80_f12_parent_default",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 12,
        },
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
    ),
    SweepProfile(
        name="recall80_f12_parent_preserve_seed_extra2",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 12,
        },
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
            "parent_expansion_preserve_seed_chunks": True,
            "parent_expansion_max_added": 2,
        },
    ),
    SweepProfile(
        name="recall80_f12_parent_preserve_seed_extra4",
        retrieval={
            "dense_limit": 80,
            "bm25_limit": 80,
            "search_limit": 80,
            "rerank_top_k": 20,
            "final_top_k": 12,
        },
        parent={
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
            "parent_expansion_preserve_seed_chunks": True,
            "parent_expansion_max_added": 4,
        },
    ),
)

PROFILE_GROUPS = {
    "ab500": AB500_PROFILES,
    "recall": RECALL_PROFILES,
    "parent": PARENT_PROFILES,
    "generation": GENERATION_PROFILES,
}

BOOL_METRICS = (
    "retrieved_top5_doc",
    "retrieved_top5_stable_parent",
    "retrieved_top10_doc",
    "retrieved_top10_stable_parent",
    "rerank_top5_doc",
    "rerank_top5_stable_parent",
    "final_topK_doc",
    "final_topK_stable_parent",
    "materialized_top5_has_matched_child",
    "support_child_view_used",
    "cited_child_view_used",
    "citation_quote_has_child_marker",
    "all_citations_parent_ids",
    "legal_final_citation",
    "citation_target_doc_hit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase5F parent-child retrieval parameter sweeps without rebuilding indexes."
        )
    )
    parser.add_argument("--round", choices=sorted(PROFILE_GROUPS), default="recall")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--base-profile", default="baseline")
    parser.add_argument("--parent-profile", default="parent_default")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--milvus-uri", default=str(DEFAULT_MILVUS_URI))
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--child-chunks-jsonl", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--parent-chunks-jsonl", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--parent-index", default=str(DEFAULT_PARENT_INDEX))
    parser.add_argument("--bm25-cache", default=str(DEFAULT_BM25))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--main-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_phase5f_samples(
    path: str | Path,
    *,
    main_only: bool = True,
    limit: int = 0,
) -> list[Phase5FSample]:
    samples: list[Phase5FSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            include = bool(item.get("include_in_main_denominator", True))
            if main_only and not include:
                continue
            stable_ids = _stable_block_ids(item)
            target_doc_ids = _target_doc_ids(item)
            target_doc_id = str(item.get("target_doc_id") or "")
            if not target_doc_id and target_doc_ids:
                target_doc_id = target_doc_ids[0]
            samples.append(
                Phase5FSample(
                    sample_id=str(item.get("sample_id") or item.get("id") or ""),
                    query=str(item.get("query") or item["question"]),
                    query_type=str(
                        item.get("query_type")
                        or item.get("category")
                        or item.get("expected_route")
                        or "unknown"
                    ),
                    target_doc_id=target_doc_id,
                    stable_target_block_ids=tuple(stable_ids),
                    target_chunk_id_candidate=str(item.get("target_chunk_id_candidate") or ""),
                    include_in_main_denominator=include,
                    target_doc_ids=tuple(target_doc_ids),
                )
            )
            if limit and len(samples) >= limit:
                break
    return samples


def _stable_block_ids(item: dict[str, Any]) -> list[str]:
    values = list(item.get("stable_target_block_ids") or [])
    for key in ("target_associated_block_id", "target_caption_block_id"):
        value = str(item.get(key) or "").strip()
        if value:
            values.append(value)
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _target_doc_ids(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    target_doc_id = str(item.get("target_doc_id") or "").strip()
    if target_doc_id:
        values.append(target_doc_id)
    expected_doc_ids = item.get("expected_doc_ids") or []
    if isinstance(expected_doc_ids, list):
        values.extend(str(value) for value in expected_doc_ids if str(value or "").strip())
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        construction_doc_ids = metadata.get("construction_source_doc_ids") or []
        if isinstance(construction_doc_ids, list):
            values.extend(
                str(value) for value in construction_doc_ids if str(value or "").strip()
            )
    return list(dict.fromkeys(values))


def select_profiles(
    round_name: str,
    requested: list[str] | None,
    *,
    base_profile_name: str,
    parent_profile_name: str,
) -> list[SweepProfile]:
    available = {profile.name: profile for profile in PROFILE_GROUPS[round_name]}
    names = requested or list(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown {round_name} profile(s): {unknown}")

    if round_name == "recall":
        return [available[name] for name in names]

    recall_base = _profile_by_name(RECALL_PROFILES, base_profile_name)
    if round_name == "parent":
        return [
            recall_base.merged_with(available[name], name=name)
            for name in names
        ]

    parent_base = _profile_by_name(PARENT_PROFILES, parent_profile_name)
    base = recall_base.merged_with(parent_base, name=f"{base_profile_name}+{parent_profile_name}")
    return [base.merged_with(available[name], name=name) for name in names]


def _profile_by_name(profiles: tuple[SweepProfile, ...], name: str) -> SweepProfile:
    for profile in profiles:
        if profile.name == name:
            return profile
    raise SystemExit(f"unknown profile {name!r}")


def build_settings(args: argparse.Namespace, profile: SweepProfile) -> Settings:
    settings = Settings.from_env()
    settings.retrieval.milvus_uri = str(Path(args.milvus_uri).resolve())
    settings.retrieval.collection_name = args.collection_name
    settings.retrieval.bm25_cache_path = str(Path(args.bm25_cache).resolve())
    settings.retrieval.parent_index_path = str(Path(args.parent_index).resolve())
    settings.kb.child_chunk_jsonl = str(Path(args.child_chunks_jsonl).resolve())
    settings.kb.parent_chunk_jsonl = str(Path(args.parent_chunks_jsonl).resolve())
    settings.kb.chunk_jsonl = str(Path(args.child_chunks_jsonl).resolve())
    settings.query_rewrite.mode = "off"
    settings.generation.v2_use_qwen_synthesis = False

    for key, value in profile.retrieval.items():
        setattr(settings.retrieval, key, value)
    for key, value in profile.parent.items():
        setattr(settings.retrieval, key, value)
    for key, value in profile.generation.items():
        setattr(settings.generation, key, value)
    return settings


def run_pipeline_trace(pipeline: SynBioRAGPipeline, question: str) -> PipelineTrace:
    start = time.perf_counter()
    filters = QueryFilters(tenant_id="default")
    retrieval = RetrievalStage(
        settings=pipeline.settings,
        router=pipeline.router,
        rewrite_service=pipeline._rewrite_svc,
        search_with_filter_fallback=pipeline._search_with_filter_fallback,
        table_preview_provider=getattr(pipeline, "table_preview_provider", None),
        fallback_pipeline=pipeline,
    ).run(question=question, filters=filters)

    rerank = RerankStage(settings=pipeline.settings, reranker=pipeline.reranker).run(
        question=question,
        retrieved=retrieval.retrieved,
        analysis=retrieval.analysis,
    )
    context = ContextStage(
        settings=pipeline.settings,
        retriever=pipeline.retriever,
        parent_expander=getattr(pipeline, "parent_expander", None),
    ).run(
        question=question,
        analysis=retrieval.analysis,
        seed_chunks=rerank.seed_chunks,
        reranked=rerank.reranked,
    )
    generation = GenerationStage(
        settings=pipeline.settings,
        confidence_scorer=pipeline.confidence_scorer,
        generator_v2=pipeline.generator_v2,
    ).run(
        question=question,
        analysis=retrieval.analysis,
        seed_chunks=context.seed_chunks,
        final_chunks=context.final_chunks,
        table_preview_debug=retrieval.table_preview_debug,
        history=None,
    )
    response = ResponseStage(
        settings=pipeline.settings,
        retriever=pipeline.retriever,
        reranker=pipeline.reranker,
    ).run(
        retrieval=retrieval,
        rerank=rerank,
        context=context,
        generation=generation,
        session_id=None,
        filters=filters,
        start_time=start,
    )
    return PipelineTrace(
        retrieved=retrieval.retrieved,
        reranked=rerank.reranked,
        seed_chunks=rerank.seed_chunks,
        final_chunks=context.final_chunks,
        parent_expansion_debug=context.parent_expansion_debug,
        response=response,
    )


def evaluate_profile(
    samples: list[Phase5FSample],
    profile: SweepProfile,
    settings: Settings,
) -> dict[str, Any]:
    pipeline = SynBioRAGPipeline(settings)
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{profile.name}] {index}/{len(samples)} {sample.sample_id}", flush=True)
        try:
            trace = run_pipeline_trace(pipeline, sample.query)
            records.append(build_record(sample, profile, settings, trace, error=""))
        except Exception as exc:
            records.append(error_record(sample, profile, settings, exc))

    return {
        "profile": profile.to_dict(),
        "run_config": run_config(settings),
        "metrics": aggregate_records(records),
        "records": records,
    }


def build_record(
    sample: Phase5FSample,
    profile: SweepProfile,
    settings: Settings,
    trace: PipelineTrace,
    *,
    error: str,
) -> dict[str, Any]:
    response = trace.response
    gv2 = (response.debug or {}).get("generation_v2", {}) or {}
    support_pack = gv2.get("support_pack") or []
    citation_binding = ((gv2.get("support_selection_debug") or {}).get("citation_binding") or {})
    ordered_evidence_ids = list(citation_binding.get("ordered_evidence_ids") or [])
    child_view_by_eid = (
        citation_binding.get("parent_child_generation_view_used_by_evidence_id") or {}
    )
    cited_child_view_used = any(bool(child_view_by_eid.get(eid)) for eid in ordered_evidence_ids)
    citations = list(response.citations or [])
    target_doc_ids = sample_target_doc_ids(sample)

    bool_values = {
        "retrieved_top5_doc": has_doc_hit(trace.retrieved, sample, limit=5),
        "retrieved_top5_stable_parent": has_stable_parent_hit(trace.retrieved, sample, limit=5),
        "retrieved_top10_doc": has_doc_hit(trace.retrieved, sample, limit=10),
        "retrieved_top10_stable_parent": has_stable_parent_hit(trace.retrieved, sample, limit=10),
        "rerank_top5_doc": has_doc_hit(trace.reranked, sample, limit=5),
        "rerank_top5_stable_parent": has_stable_parent_hit(trace.reranked, sample, limit=5),
        "final_topK_doc": has_doc_hit(trace.final_chunks, sample, limit=0),
        "final_topK_stable_parent": has_stable_parent_hit(trace.final_chunks, sample, limit=0),
        "materialized_top5_has_matched_child": any(
            chunk_has_matched_child(chunk) for chunk in trace.retrieved[:5]
        ),
        "support_child_view_used": any(
            bool(item.get("parent_child_generation_view_used")) for item in support_pack
        ),
        "cited_child_view_used": cited_child_view_used,
        "citation_quote_has_child_marker": any(
            "matched_child_evidence:" in (citation.quote or "")
            or "matched child:" in (citation.quote or "")
            for citation in citations
        ),
        "all_citations_parent_ids": bool(citations)
        and all(not is_child_chunk_id(citation.chunk_id) for citation in citations),
        "legal_final_citation": bool(citations)
        and all(
            citation.doc_id and not is_child_chunk_id(citation.chunk_id)
            for citation in citations
        ),
        "citation_target_doc_hit": any(
            citation.doc_id in target_doc_ids for citation in citations
        ),
    }
    latency_ms = float((response.debug or {}).get("latency_ms") or 0.0)
    parent_expansion_audit = build_parent_expansion_audit(settings, trace)
    return {
        "sample": sample.to_dict(),
        "profile_name": profile.name,
        "query_type": sample.query_type,
        "error": error,
        **bool_values,
        "retrieved_count": len(trace.retrieved),
        "rerank_candidate_count": len(trace.retrieved),
        "reranked_count": len(trace.reranked),
        "seed_context_count": len(trace.seed_chunks),
        "final_context_count": len(trace.final_chunks),
        "parent_expansion": parent_expansion_audit,
        "support_pack_size": int(gv2.get("support_pack_count") or len(support_pack)),
        "citation_count": len(citations),
        "answer_mode": str(gv2.get("answer_mode") or ""),
        "latency_ms": latency_ms,
        "top_hits": {
            "retrieved": serialize_chunks(trace.retrieved[:10], sample),
            "reranked": serialize_chunks(trace.reranked[:10], sample),
            "final": serialize_chunks(trace.final_chunks[:20], sample),
        },
        "citations": [
            {
                "chunk_id": citation.chunk_id,
                "doc_id": citation.doc_id,
                "source_file": citation.source_file,
                "section": citation.section,
                "page_start": citation.page_start,
                "page_end": citation.page_end,
                "quote_has_child_marker": (
                    "matched_child_evidence:" in (citation.quote or "")
                    or "matched child:" in (citation.quote or "")
                ),
            }
            for citation in citations
        ],
    }


def error_record(
    sample: Phase5FSample,
    profile: SweepProfile,
    settings: Settings,
    exc: Exception,
) -> dict[str, Any]:
    del settings
    record = {
        "sample": sample.to_dict(),
        "profile_name": profile.name,
        "query_type": sample.query_type,
        "error": f"{type(exc).__name__}: {exc}",
        "retrieved_count": 0,
        "rerank_candidate_count": 0,
        "reranked_count": 0,
        "seed_context_count": 0,
        "final_context_count": 0,
        "parent_expansion": {
            "enabled_config": False,
            "enabled_debug": False,
            "reason": "error",
            "input_count": 0,
            "output_count": 0,
            "added_count": 0,
            "added_chunk_ids": [],
            "added_parent_ids": [],
            "added_parent_types": [],
            "effective_max_total": 0,
            "effective_per_seed_limit": 0,
            "over_budget": False,
            "disabled_but_added": False,
            "enabled_but_disabled_reason": False,
            "debug_consistent": False,
        },
        "support_pack_size": 0,
        "citation_count": 0,
        "answer_mode": "error",
        "latency_ms": 0.0,
        "top_hits": {"retrieved": [], "reranked": [], "final": []},
        "citations": [],
    }
    record.update({metric: False for metric in BOOL_METRICS})
    return record


def build_parent_expansion_audit(settings: Settings, trace: PipelineTrace) -> dict[str, Any]:
    debug = trace.parent_expansion_debug or {}
    added_chunk_ids = list(debug.get("added_chunk_ids") or [])
    added_parent_ids = list(debug.get("added_parent_ids") or [])
    added_parent_types = list(debug.get("added_parent_types") or [])
    enabled_config = bool(settings.retrieval.parent_expansion_enabled)
    enabled_debug = bool(debug.get("enabled"))
    input_count = int(debug.get("input_count") or len(trace.seed_chunks))
    output_count = int(debug.get("output_count") or len(trace.final_chunks))
    effective_max_total = int(
        debug.get("effective_max_total")
        or settings.retrieval.parent_expansion_max_total
        or 0
    )
    effective_final_context_cap = int(debug.get("effective_final_context_cap") or effective_max_total)
    reason = str(debug.get("reason") or "")
    return {
        "enabled_config": enabled_config,
        "enabled_debug": enabled_debug,
        "reason": reason,
        "input_count": input_count,
        "output_count": output_count,
        "added_count": len(added_chunk_ids),
        "added_chunk_ids": added_chunk_ids,
        "added_parent_ids": added_parent_ids,
        "added_parent_types": added_parent_types,
        "effective_intent": str(debug.get("effective_intent") or ""),
        "effective_max_total": effective_max_total,
        "effective_per_seed_limit": int(debug.get("effective_per_seed_limit") or 0),
        "seed_preservation_enabled": bool(debug.get("seed_preservation_enabled")),
        "effective_max_added": int(debug.get("effective_max_added") or 0),
        "effective_final_context_cap": effective_final_context_cap,
        "limit_reason": str(debug.get("limit_reason") or ""),
        "caption_mode_trigger_source": str(debug.get("caption_mode_trigger_source") or ""),
        "selected_parent_types": list(debug.get("selected_parent_types") or []),
        "per_seed_added": dict(debug.get("per_seed_added") or {}),
        "per_doc_added": dict(debug.get("per_doc_added") or {}),
        "over_budget": effective_final_context_cap > 0 and output_count > effective_final_context_cap,
        "disabled_but_added": not enabled_config and bool(added_chunk_ids),
        "enabled_but_disabled_reason": enabled_config and reason == "disabled",
        "debug_consistent": enabled_config == enabled_debug,
    }


def has_doc_hit(chunks: list[RetrievedChunk], sample: Phase5FSample, *, limit: int) -> bool:
    selected = chunks[:limit] if limit else chunks
    target_doc_ids = sample_target_doc_ids(sample)
    return bool(target_doc_ids) and any(chunk.doc_id in target_doc_ids for chunk in selected)


def has_stable_parent_hit(
    chunks: list[RetrievedChunk],
    sample: Phase5FSample,
    *,
    limit: int,
) -> bool:
    selected = chunks[:limit] if limit else chunks
    return any(chunk_matches_stable_parent(chunk, sample) for chunk in selected)


def chunk_matches_stable_parent(chunk: RetrievedChunk, sample: Phase5FSample) -> bool:
    target_doc_ids = sample_target_doc_ids(sample)
    if not target_doc_ids or chunk.doc_id not in target_doc_ids:
        return False
    if sample.target_chunk_id_candidate and (
        chunk.chunk_id == sample.target_chunk_id_candidate
        or sample.target_chunk_id_candidate in matched_child_chunk_ids(chunk)
    ):
        return True
    stable_ids = set(sample.stable_target_block_ids)
    if not stable_ids:
        return True
    return bool(stable_ids.intersection(chunk_block_ids(chunk)))


def sample_target_doc_ids(sample: Phase5FSample) -> set[str]:
    values = sample.target_doc_ids or ((sample.target_doc_id,) if sample.target_doc_id else ())
    return {str(value) for value in values if str(value or "").strip()}


def chunk_block_ids(chunk: RetrievedChunk) -> set[str]:
    metadata = chunk.metadata or {}
    values: list[str] = []
    for key in ("source_block_ids", "block_ids"):
        raw = metadata.get(key) or []
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if str(value or "").strip())
    for row in metadata.get("source_block_metadata") or []:
        if not isinstance(row, dict):
            continue
        for key in ("source_block_id", "block_id"):
            value = str(row.get(key) or "").strip()
            if value:
                values.append(value)
    return set(values)


def matched_child_chunk_ids(chunk: RetrievedChunk) -> list[str]:
    metadata = chunk.metadata or {}
    value = metadata.get("matched_child_chunk_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    value = str(metadata.get("matched_child_chunk_id") or "").strip()
    return [value] if value else []


def chunk_has_matched_child(chunk: RetrievedChunk) -> bool:
    metadata = chunk.metadata or {}
    return bool(
        metadata.get("parent_child_materialized")
        or matched_child_chunk_ids(chunk)
        or metadata.get("matched_child_snippets")
    )


def is_child_chunk_id(chunk_id: str) -> bool:
    return "::child" in str(chunk_id or "")


def serialize_chunks(chunks: list[RetrievedChunk], sample: Phase5FSample) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_doc_ids = sample_target_doc_ids(sample)
    for rank, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "score": {
                    "fusion": round(float(chunk.fusion_score or 0.0), 6),
                    "rerank": round(float(chunk.rerank_score or 0.0), 6),
                    "vector": round(float(chunk.vector_score or 0.0), 6),
                    "bm25": round(float(chunk.bm25_score or 0.0), 6),
                },
                "doc_hit": chunk.doc_id in target_doc_ids,
                "stable_parent_hit": chunk_matches_stable_parent(chunk, sample),
                "has_matched_child": chunk_has_matched_child(chunk),
                "matched_child_chunk_ids": matched_child_chunk_ids(chunk),
                "block_ids": sorted(chunk_block_ids(chunk))[:20],
            }
        )
    return rows


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": aggregate_bucket(records),
        "by_query_type": {
            query_type: aggregate_bucket(items)
            for query_type, items in sorted(group_by_query_type(records).items())
        },
        "gate_checks": {},
    }


def aggregate_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    metric_values = {
        name: metric_count(records, name)
        for name in BOOL_METRICS
    }
    latency_values = [
        float(record.get("latency_ms") or 0.0)
        for record in records
        if not record.get("error")
    ]
    citation_counts = [int(record.get("citation_count") or 0) for record in records]
    support_sizes = [int(record.get("support_pack_size") or 0) for record in records]
    rerank_candidates = [int(record.get("rerank_candidate_count") or 0) for record in records]
    return {
        "sample_count": total,
        **metric_values,
        "citation_count": {
            "average": round(_mean(citation_counts), 4),
            "zero_count": sum(1 for value in citation_counts if value == 0),
            "distribution": dict(Counter(citation_counts)),
        },
        "answer_mode_distribution": dict(
            Counter(record.get("answer_mode") or "" for record in records)
        ),
        "cost": {
            "latency_ms_median": round(float(statistics.median(latency_values)), 2)
            if latency_values
            else 0.0,
            "latency_ms_p95": round(percentile(latency_values, 95), 2),
            "average_rerank_candidate_count": round(_mean(rerank_candidates), 4),
            "average_support_pack_size": round(_mean(support_sizes), 4),
        },
        "parent_expansion_audit": aggregate_parent_expansion(records),
        "error_count": sum(1 for record in records if record.get("error")),
        "error_sample_ids": [
            record["sample"]["sample_id"] for record in records if record.get("error")
        ],
    }


def aggregate_parent_expansion(records: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [
        record.get("parent_expansion") or {}
        for record in records
        if not record.get("error")
    ]
    added_counts = [int(audit.get("added_count") or 0) for audit in audits]
    output_counts = [int(audit.get("output_count") or 0) for audit in audits]
    parent_types: Counter[str] = Counter()
    for audit in audits:
        parent_types.update(str(value) for value in audit.get("added_parent_types") or [])
    disabled_but_added_ids = [
        record["sample"]["sample_id"]
        for record in records
        if (record.get("parent_expansion") or {}).get("disabled_but_added")
    ]
    enabled_but_disabled_reason_ids = [
        record["sample"]["sample_id"]
        for record in records
        if (record.get("parent_expansion") or {}).get("enabled_but_disabled_reason")
    ]
    over_budget_ids = [
        record["sample"]["sample_id"]
        for record in records
        if (record.get("parent_expansion") or {}).get("over_budget")
    ]
    inconsistent_debug_ids = [
        record["sample"]["sample_id"]
        for record in records
        if not (record.get("parent_expansion") or {}).get("debug_consistent", True)
    ]
    return {
        "enabled_config_distribution": dict(
            Counter(str(audit.get("enabled_config")) for audit in audits)
        ),
        "enabled_debug_distribution": dict(
            Counter(str(audit.get("enabled_debug")) for audit in audits)
        ),
        "reason_distribution": dict(
            Counter(str(audit.get("reason") or "") for audit in audits)
        ),
        "expanded_sample_count": sum(1 for value in added_counts if value > 0),
        "added_count": {
            "average": round(_mean_float(added_counts), 4),
            "p95": round(percentile_float(added_counts, 95), 4),
            "max": max(added_counts) if added_counts else 0,
            "distribution": dict(Counter(added_counts)),
        },
        "output_count": {
            "average": round(_mean_float(output_counts), 4),
            "p95": round(percentile_float(output_counts, 95), 4),
            "max": max(output_counts) if output_counts else 0,
        },
        "added_parent_type_distribution": dict(parent_types),
        "disabled_but_added_count": len(disabled_but_added_ids),
        "disabled_but_added_sample_ids": disabled_but_added_ids[:20],
        "enabled_but_disabled_reason_count": len(enabled_but_disabled_reason_ids),
        "enabled_but_disabled_reason_sample_ids": enabled_but_disabled_reason_ids[:20],
        "over_budget_count": len(over_budget_ids),
        "over_budget_sample_ids": over_budget_ids[:20],
        "inconsistent_debug_count": len(inconsistent_debug_ids),
        "inconsistent_debug_sample_ids": inconsistent_debug_ids[:20],
    }


def metric_count(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    total = len(records)
    hit = sum(1 for record in records if bool(record.get(name)))
    return {"hit": hit, "total": total, "rate": round(hit / total, 6) if total else 0.0}


def group_by_query_type(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("query_type") or "unknown")].append(record)
    return grouped


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round((percentile_value / 100) * len(ordered)))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def percentile_float(values: list[int], percentile_value: int) -> float:
    return percentile([float(value) for value in values], percentile_value)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_float(values: list[int]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def add_gate_checks(profile_results: list[dict[str, Any]], baseline_name: str = "baseline") -> None:
    baseline = next(
        (result for result in profile_results if result["profile"]["name"] == baseline_name),
        None,
    )
    if baseline is None:
        return
    base_overall = baseline["metrics"]["overall"]
    base_categories = baseline["metrics"]["by_query_type"]
    base_p95 = float(base_overall["cost"]["latency_ms_p95"] or 0.0)
    for result in profile_results:
        if result["profile"]["name"] == baseline_name:
            result["metrics"]["gate_checks"] = {
                "baseline_reference": True,
                "candidate_gate_passed": None,
            }
            continue
        overall = result["metrics"]["overall"]
        categories = result["metrics"]["by_query_type"]
        figure_base = base_categories.get("figure_caption", {})
        figure_current = categories.get("figure_caption", {})
        table_base = base_categories.get("table_content", {})
        table_current = categories.get("table_content", {})
        caption_base = base_categories.get("caption_level_table", {})
        caption_current = categories.get("caption_level_table", {})
        current_p95 = float(overall["cost"]["latency_ms_p95"] or 0.0)
        parent_audit = overall.get("parent_expansion_audit") or {}
        parent_enabled = bool((result.get("run_config") or {}).get("parent_expansion_enabled", True))
        accuracy_gain = (
            overall["final_topK_stable_parent"]["hit"]
            - base_overall["final_topK_stable_parent"]["hit"]
        )
        checks = {
            "figure_caption_no_regress": metric_hit(figure_current, "final_topK_stable_parent")
            >= metric_hit(figure_base, "final_topK_stable_parent"),
            "phase5f_final_doc_match_or_improve": overall["final_topK_doc"]["hit"]
            >= base_overall["final_topK_doc"]["hit"],
            "phase5f_final_stable_match_or_improve": overall["final_topK_stable_parent"]["hit"]
            >= base_overall["final_topK_stable_parent"]["hit"],
            "table_or_caption_stable_improves": (
                metric_hit(table_current, "final_topK_stable_parent")
                > metric_hit(table_base, "final_topK_stable_parent")
                or metric_hit(caption_current, "final_topK_stable_parent")
                > metric_hit(caption_base, "final_topK_stable_parent")
            ),
            "support_child_view_full_rate": overall["support_child_view_used"]["hit"]
            == overall["support_child_view_used"]["total"],
            "cited_child_view_full_rate": overall["cited_child_view_used"]["hit"]
            == overall["cited_child_view_used"]["total"],
            "parent_citation_id_full_rate": overall["all_citations_parent_ids"]["hit"]
            == overall["all_citations_parent_ids"]["total"],
            "p95_latency_under_2x_or_accuracy_gain": (
                base_p95 <= 0.0 or current_p95 <= base_p95 * 2.0 or accuracy_gain > 0
            ),
            "parent_expansion_debug_consistent": int(parent_audit.get("inconsistent_debug_count") or 0) == 0,
            "parent_expansion_no_budget_violation": int(parent_audit.get("over_budget_count") or 0) == 0,
            "parent_disabled_no_additions": (
                parent_enabled
                or int(parent_audit.get("disabled_but_added_count") or 0) == 0
            ),
        }
        checks["candidate_gate_passed"] = all(checks.values())
        checks["p95_latency_ratio_vs_baseline"] = (
            round(current_p95 / base_p95, 4) if base_p95 > 0 else None
        )
        result["metrics"]["gate_checks"] = checks


def metric_hit(bucket: dict[str, Any], metric_name: str) -> int:
    metric = bucket.get(metric_name) or {}
    return int(metric.get("hit") or 0)


def run_config(settings: Settings) -> dict[str, Any]:
    return {
        "milvus_uri": settings.retrieval.milvus_uri,
        "collection_name": settings.retrieval.collection_name,
        "query_rewrite_mode": settings.query_rewrite.mode,
        "qwen_synthesis_enabled": settings.generation.v2_use_qwen_synthesis,
        "dense_limit": settings.retrieval.dense_limit,
        "bm25_limit": settings.retrieval.bm25_limit,
        "search_limit": settings.retrieval.search_limit,
        "rerank_top_k": settings.retrieval.rerank_top_k,
        "final_top_k": settings.retrieval.final_top_k,
        "parent_expansion_enabled": settings.retrieval.parent_expansion_enabled,
        "parent_expansion_max_total": settings.retrieval.parent_expansion_max_total,
        "parent_expansion_per_seed_limit": settings.retrieval.parent_expansion_per_seed_limit,
        "parent_expansion_preserve_seed_chunks": settings.retrieval.parent_expansion_preserve_seed_chunks,
        "parent_expansion_max_added": settings.retrieval.parent_expansion_max_added,
        "max_support_factoid": settings.generation.v2_max_support_factoid,
        "max_support_summary": settings.generation.v2_max_support_summary,
        "max_support_comparison": settings.generation.v2_max_support_comparison,
        "max_extractive_evidence_lines": settings.generation.v2_max_extractive_evidence_lines,
    }


def write_profile_outputs(
    result: dict[str, Any],
    *,
    results_dir: Path,
    reports_dir: Path,
) -> None:
    name = result["profile"]["name"]
    write_json(results_dir / f"{name}.json", result)
    (reports_dir / f"{name}.md").write_text(profile_markdown(result), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def profile_markdown(result: dict[str, Any]) -> str:
    profile = result["profile"]
    metrics = result["metrics"]["overall"]
    categories = result["metrics"]["by_query_type"]
    lines = [
        f"# Parent-Child Param Sweep: {profile['name']}",
        "",
        "## Profile",
        f"- retrieval: `{profile['retrieval']}`",
        f"- parent: `{profile['parent']}`",
        f"- generation: `{profile['generation']}`",
        "",
        "## Overall Metrics",
    ]
    for key in (
        "retrieved_top5_doc",
        "retrieved_top5_stable_parent",
        "retrieved_top10_doc",
        "retrieved_top10_stable_parent",
        "rerank_top5_doc",
        "rerank_top5_stable_parent",
        "final_topK_doc",
        "final_topK_stable_parent",
        "materialized_top5_has_matched_child",
        "support_child_view_used",
        "cited_child_view_used",
        "citation_quote_has_child_marker",
        "all_citations_parent_ids",
        "legal_final_citation",
        "citation_target_doc_hit",
    ):
        lines.append(f"- {key}: `{format_metric(metrics[key])}`")
    lines.extend(
        [
            f"- citation_count: `{metrics['citation_count']}`",
            f"- answer_mode_distribution: `{metrics['answer_mode_distribution']}`",
            f"- cost: `{metrics['cost']}`",
            f"- errors: `{metrics['error_count']}`",
            "",
            "## Parent Expansion Audit",
            f"- enabled_config_distribution: `{metrics['parent_expansion_audit']['enabled_config_distribution']}`",
            f"- reason_distribution: `{metrics['parent_expansion_audit']['reason_distribution']}`",
            f"- expanded_sample_count: `{metrics['parent_expansion_audit']['expanded_sample_count']}`",
            f"- added_count: `{metrics['parent_expansion_audit']['added_count']}`",
            f"- added_parent_type_distribution: `{metrics['parent_expansion_audit']['added_parent_type_distribution']}`",
            f"- disabled_but_added_count: `{metrics['parent_expansion_audit']['disabled_but_added_count']}`",
            f"- over_budget_count: `{metrics['parent_expansion_audit']['over_budget_count']}`",
            "",
            "## Category Final Stable Hits",
        ]
    )
    for query_type, bucket in categories.items():
        lines.append(
            f"- {query_type}: final_topK_doc=`{format_metric(bucket['final_topK_doc'])}` "
            f"final_topK_stable_parent=`{format_metric(bucket['final_topK_stable_parent'])}`"
        )
    if result["metrics"].get("gate_checks"):
        lines.extend(["", "## Gate Checks", f"`{result['metrics']['gate_checks']}`"])
    lines.append("")
    return "\n".join(lines)


def summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Parent-Child Param Sweep Summary",
        "",
        "## Run",
        f"- round: `{payload['round']}`",
        f"- dataset: `{payload['dataset']}`",
        f"- sample_count: `{payload['sample_count']}`",
        f"- query_rewrite_mode: `off`",
        f"- qwen_synthesis_enabled: `False`",
        "",
        "## Profiles",
    ]
    for result in payload["profiles"]:
        name = result["profile"]["name"]
        overall = result["metrics"]["overall"]
        gates = result["metrics"].get("gate_checks", {})
        parent_audit = overall["parent_expansion_audit"]
        lines.append(
            f"- {name}: final_topK_doc=`{format_metric(overall['final_topK_doc'])}` "
            f"final_topK_stable_parent=`{format_metric(overall['final_topK_stable_parent'])}` "
            f"support_child_view=`{format_metric(overall['support_child_view_used'])}` "
            f"cited_child_view=`{format_metric(overall['cited_child_view_used'])}` "
            f"parent_citation_ids=`{format_metric(overall['all_citations_parent_ids'])}` "
            f"parent_added_avg=`{parent_audit['added_count']['average']}` "
            f"parent_expanded_samples=`{parent_audit['expanded_sample_count']}` "
            f"parent_disabled_but_added=`{parent_audit['disabled_but_added_count']}` "
            f"p95_ms=`{overall['cost']['latency_ms_p95']}` "
            f"gate_passed=`{gates.get('candidate_gate_passed')}`"
        )
    lines.extend(
        [
            "",
            "## Regression Commands",
            "- `pytest tests/test_generation_v2.py tests/test_generation_v2_qwen_synthesis.py "
            "tests/test_generation_v2_summary_support.py tests/test_parent_child_index.py "
            "tests/test_parent_expansion.py tests/test_pipeline_stages.py -q`",
            "- `python scripts/evaluation/run_parent_child_param_sweep.py --round recall`",
            "- `python scripts/evaluation/run_parent_child_param_sweep.py --round parent "
            "--base-profile <recall_winner>`",
            "- `python scripts/evaluation/run_parent_child_param_sweep.py --round generation "
            "--base-profile <recall_winner> --parent-profile <parent_winner>`",
            "- `python scripts/evaluation/run_phase21a9_smoke200_rebaseline.py` after "
            "Phase5F passes and query rewrite is intentionally enabled.",
            "",
        ]
    )
    return "\n".join(lines)


def format_metric(metric: dict[str, Any]) -> str:
    return f"{metric['hit']}/{metric['total']} ({metric['rate']:.3f})"


def build_payload(
    *,
    args: argparse.Namespace,
    profiles: list[SweepProfile],
    samples: list[Phase5FSample],
    profile_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "round": args.round,
        "dataset": str(Path(args.dataset).resolve()),
        "sample_count": len(samples),
        "profiles_requested": [profile.name for profile in profiles],
        "base_profile": args.base_profile,
        "parent_profile": args.parent_profile,
        "fixed_index": {
            "milvus_uri": str(Path(args.milvus_uri).resolve()),
            "collection_name": args.collection_name,
            "child_chunks_jsonl": str(Path(args.child_chunks_jsonl).resolve()),
            "parent_chunks_jsonl": str(Path(args.parent_chunks_jsonl).resolve()),
            "parent_index": str(Path(args.parent_index).resolve()),
        },
        "profiles": profile_results,
    }


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    samples = load_phase5f_samples(
        args.dataset,
        main_only=args.main_only,
        limit=max(0, args.limit),
    )
    profiles = select_profiles(
        args.round,
        args.profiles,
        base_profile_name=args.base_profile,
        parent_profile_name=args.parent_profile,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "round": args.round,
                    "sample_count": len(samples),
                    "profiles": [profile.to_dict() for profile in profiles],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    run_id = args.run_id or f"{args.round}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(args.results_dir) / run_id
    reports_dir = Path(args.reports_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    profile_results: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for profile in profiles:
        settings = build_settings(args, profile)
        result = evaluate_profile(samples, profile, settings)
        profile_results.append(result)
        all_records.extend(result["records"])

    add_gate_checks(profile_results)
    for result in profile_results:
        write_profile_outputs(result, results_dir=results_dir, reports_dir=reports_dir)

    payload = build_payload(
        args=args,
        profiles=profiles,
        samples=samples,
        profile_results=profile_results,
    )
    write_json(results_dir / "summary.json", payload)
    write_jsonl(results_dir / "records.jsonl", all_records)
    (reports_dir / "summary.md").write_text(summary_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "results": str(results_dir),
                "reports": str(reports_dir),
                "profiles": [result["profile"]["name"] for result in profile_results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
